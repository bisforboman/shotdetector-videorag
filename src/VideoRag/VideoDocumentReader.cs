using ManagedCodecs.Thumbnails;
using Microsoft.Extensions.DataIngestion;
using ShotDetector;

namespace VideoRag;

/// <summary>How <see cref="VideoDocumentReader"/> reads a video.</summary>
public sealed record VideoDocumentReaderOptions
{
    /// <summary>The detector and its settings; null: <see cref="ShotDetection"/>'s defaults (adaptive).</summary>
    public DetectionOptions? Detection { get; init; }

    /// <summary>
    /// Folder of FFmpeg 8's shared libraries (an FFmpeg 8 "shared" build, or FrameReader.Native's): with them detection
    /// decodes in-process. Null: the usual places, else detection runs through the ffmpeg executable on PATH. The stills
    /// don't need them.
    /// </summary>
    public string? FfmpegLibraries { get; init; }

    /// <summary>The stills' largest width, in pixels; narrower videos keep their own.</summary>
    public int StillMaxWidth { get; init; } = 640;

    /// <summary>The stills' JPEG quality, 1 to 100 (libjpeg's scale).</summary>
    public int StillQuality { get; init; } = 85;

    /// <summary>How many stills are decoded at once; null: one per processor.</summary>
    public int? StillThreads { get; init; }

    /// <summary>
    /// Speech to text for the soundtrack (e.g. <see cref="WhisperTranscriber"/>): each shot gets a paragraph of what's
    /// said in it. The soundtrack is decoded by ManagedCodecs (AAC in MP4 or ADTS); null, or other audio: no transcripts.
    /// </summary>
    public ITranscriber? Transcriber { get; init; }

    /// <summary>
    /// Whether to keep a transcript's sound annotations: <c>[MUSIC PLAYING]</c>, <c>(laughs)</c>, ♪. False (the default)
    /// removes them from every transcriber's segments, and a segment left without a letter or digit, so a shot with only
    /// music says nothing and shots never look alike for sharing "[MUSIC]".
    /// </summary>
    public bool KeepSoundTags { get; init; }

    /// <summary>
    /// A shot longer than this gets a still per stretch of this length (the middle of each), not only its middle: a fade
    /// or a dissolve the detector doesn't cut at leaves one long "shot" over several scenes. Null: the middle only.
    /// </summary>
    public TimeSpan? StillInterval { get; init; } = TimeSpan.FromSeconds(8);

    /// <summary>
    /// Whether to look for black and still pictures on the detection pass (ShotDetector's blackdetect and freezedetect,
    /// ffmpeg's defaults: spans of 2 s or more; true by default). A shot at least 90% black says "Black screen." and gets
    /// no stills; a still picture in a shot says "Still picture from A to B." and gets one still, however long it is.
    /// </summary>
    public bool Qc { get; init; } = true;
}

/// <summary>
/// A video as an <see cref="IngestionDocument"/>: one section per shot, headed by the shot's times
/// (<c>## 00:00:02.000 – 00:00:04.000</c>), which <c>SectionChunker</c> makes every chunk's context and
/// <c>VectorStoreWriter</c> stores, so a search result says where in the video it is. Each section also holds a
/// paragraph naming the shot, with a <see cref="VideoDocumentReaderOptions.Transcriber"/> a paragraph of what's said
/// in the shot, and, with FFmpeg's libraries, its middle frame as a JPEG with no alternative text (a long shot: one per
/// <see cref="VideoDocumentReaderOptions.StillInterval"/>), for <see cref="ImageAlternativeTextEnricher"/> to caption.
/// </summary>
public sealed class VideoDocumentReader(VideoDocumentReaderOptions? options = null) : IngestionDocumentReader
{
    /// <summary>Element metadata: the shot's number, and its start and end in seconds (the end is the next shot's start).</summary>
    public const string ShotKey = "shot";

    public const string StartKey = "start";

    public const string EndKey = "end";

    /// <summary>The start of a shot's transcript paragraph.</summary>
    public const string SaidPrefix = "Said: ";

    readonly VideoDocumentReaderOptions _options = options ?? new();

    /// <summary>Whether detection decodes in-process (FFmpeg's libraries load), not through the ffmpeg executable.</summary>
    public bool DetectsInProcess => global::FrameReader.FFmpegLibraries.CanLoad(_options.FfmpegLibraries);

    /// <summary>A video file: detection and stills seek in it.</summary>
    public override Task<IngestionDocument> ReadAsync(FileInfo source, string identifier, string? mediaType = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        return ReadPathAsync(source.FullName, identifier, cancellationToken);
    }

    /// <summary>A video as a stream: a file's stream is read from its file, any other is copied to a temporary file first (the stills seek).</summary>
    public override async Task<IngestionDocument> ReadAsync(Stream source, string identifier, string mediaType, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source is FileStream { CanSeek: true } file)
            return await ReadPathAsync(file.Name, identifier, cancellationToken).ConfigureAwait(false);
        string temp = Path.GetTempFileName();
        try
        {
            await using (var copy = File.Create(temp))
                await source.CopyToAsync(copy, cancellationToken).ConfigureAwait(false);
            return await ReadPathAsync(temp, identifier, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            File.Delete(temp);
        }
    }

    async Task<IngestionDocument> ReadPathAsync(string path, string identifier, CancellationToken cancellationToken)
    {
        _ = DetectsInProcess; // loads the libraries first, so detection decodes in-process when it can
        var detection = _options.Qc
            ? (_options.Detection ?? new DetectionOptions()) with { BlackDetect = new BlackDetectOptions(), FreezeDetect = new FreezeDetectOptions() }
            : _options.Detection;
        var result = await ShotDetection.DetectAsync(path, detection, cancellationToken).ConfigureAwait(false);
        var document = new IngestionDocument(identifier);
        var black = (result.BlackSpans ?? []).Select(b => (b.Start, b.End)).ToList();
        var frozen = (result.FreezeSpans ?? []).Select(f => (f.Start, End: f.End ?? double.PositiveInfinity)).ToList();
        var looks = new List<(bool Black, IReadOnlyList<(double Start, double End)> Freezes)>();
        foreach (var shot in result.Shots)
        {
            var section = Section(shot, result.Shots.Count);
            bool isBlack = IsBlack(shot.Start.Seconds, shot.End.Seconds, black);
            var freezes = isBlack ? [] : FreezesIn(shot.Start.Seconds, shot.End.Seconds, frozen);
            if (isBlack)
                section.Elements.Add(Tagged(new IngestionDocumentParagraph(BlackScreen), shot));
            foreach (var (from, to) in freezes)
                section.Elements.Add(Tagged(new IngestionDocumentParagraph($"Still picture from {Timecode(TimeSpan.FromSeconds(from))} to {Timecode(TimeSpan.FromSeconds(to))}."), shot));
            document.Sections.Add(section);
            looks.Add((isBlack, freezes));
        }
        if (_options.Transcriber is { } transcriber)
            await AddTranscriptsAsync(path, transcriber, _options.KeepSoundTags, result.Shots, document, cancellationToken).ConfigureAwait(false);
        await Task.Run(() => AddStills(path, result.Shots, looks, document, cancellationToken), cancellationToken).ConfigureAwait(false);
        return document;
    }

    /// <summary>The paragraph of a shot that is (almost) all black.</summary>
    public const string BlackScreen = "Black screen.";

    /// <summary>Whether at least 90% of the shot from <paramref name="start"/> to <paramref name="end"/> is inside black spans.</summary>
    internal static bool IsBlack(double start, double end, IReadOnlyList<(double Start, double End)> black)
    {
        double covered = black.Sum(b => Math.Max(0, Math.Min(end, b.End) - Math.Max(start, b.Start)));
        return end > start && covered >= 0.9 * (end - start);
    }

    /// <summary>The still pictures in a shot, clipped to it, each at least 1 s of it (a freeze that only grazes a shot isn't its).</summary>
    internal static IReadOnlyList<(double Start, double End)> FreezesIn(double start, double end, IReadOnlyList<(double Start, double End)> frozen) =>
        [.. frozen.Select(f => (Start: Math.Max(start, f.Start), End: Math.Min(end, f.End))).Where(f => f.End - f.Start >= 1)];

    /// <summary>
    /// The still times of a shot (<see cref="StillTimes"/>) with those inside one freeze collapsed to the first, and none
    /// for a black shot.
    /// </summary>
    internal static IReadOnlyList<double> PlannedStills(double start, double duration, TimeSpan? interval, bool black, IReadOnlyList<(double Start, double End)> freezes)
    {
        if (black)
            return [];
        var taken = new HashSet<int>();
        var times = new List<double>();
        foreach (double t in StillTimes(start, duration, interval))
        {
            int freeze = -1;
            for (int f = 0; f < freezes.Count; f++)
            {
                if (t >= freezes[f].Start && t < freezes[f].End)
                    freeze = f;
            }

            if (freeze < 0 || taken.Add(freeze))
                times.Add(t);
        }

        return times;
    }

    /// <summary>The header every chunk of a shot gets as its context.</summary>
    public static string Header(Shot shot) => $"{shot.Start.Timecode()} – {shot.End.Timecode()}";

    static IngestionDocumentSection Section(Shot shot, int count)
    {
        var section = new IngestionDocumentSection($"shot-{shot.Number}");
        section.Elements.Add(Tagged(new IngestionDocumentHeader($"## {Header(shot)}") { Level = 2 }, shot));
        section.Elements.Add(Tagged(new IngestionDocumentParagraph(
            $"Shot {shot.Number} of {count}, from {shot.Start.Timecode()} to {shot.End.Timecode()} ({shot.Duration.SecondsText()} s)."),
            shot));
        return section;
    }

    static async Task AddTranscriptsAsync(string path, ITranscriber transcriber, bool keepSoundTags, IReadOnlyList<Shot> shots, IngestionDocument document, CancellationToken cancellationToken)
    {
        if (await Task.Run(() => Soundtrack.Read(path, cancellationToken), cancellationToken).ConfigureAwait(false) is not var (samples, start))
            return;
        var said = new List<string>?[shots.Count];
        await foreach (var segment in transcriber.TranscribeAsync(samples, cancellationToken).ConfigureAwait(false))
        {
            string text = keepSoundTags ? segment.Text.Trim() : WithoutSoundTags(segment.Text);
            if (text.Length > 0 && ShotOf(shots, (start + segment.Start).TotalSeconds, (start + segment.End).TotalSeconds) is int i)
                (said[i] ??= []).Add(text);
        }

        for (int i = 0; i < shots.Count; i++)
        {
            if (said[i] is { } lines)
                document.Sections[i].Elements.Add(Tagged(new IngestionDocumentParagraph($"{SaidPrefix}{string.Join(' ', lines)}"), shots[i]));
        }
    }

    /// <summary>
    /// The text without sound annotations (<c>[...]</c>, <c>(...)</c>, ♪ and ♫) and with single spaces, or "" when no
    /// letter or digit is left.
    /// </summary>
    internal static string WithoutSoundTags(string text)
    {
        string spoken = System.Text.RegularExpressions.Regex.Replace(text, @"\[[^\]]*\]|\([^)]*\)|[♪♫]", " ");
        spoken = System.Text.RegularExpressions.Regex.Replace(spoken, @"\s+", " ").Trim();
        spoken = System.Text.RegularExpressions.Regex.Replace(spoken, @" (?=[.,!?;:])", "");
        return spoken.Any(char.IsLetterOrDigit) ? spoken : "";
    }

    /// <summary>
    /// When a shot's stills are taken: the middle of each of the equal stretches, at most <paramref name="interval"/>
    /// long, that the shot divides into; the shot's middle alone without an interval.
    /// </summary>
    internal static IReadOnlyList<double> StillTimes(double start, double duration, TimeSpan? interval)
    {
        int count = interval is { } every && every > TimeSpan.Zero ? Math.Max(1, (int)Math.Ceiling(duration / every.TotalSeconds)) : 1;
        return [.. Enumerable.Range(0, count).Select(k => start + (duration * ((2 * k) + 1) / (2 * count)))];
    }

    /// <summary>The index of the shot a stretch overlaps most (the first on a tie), null if it overlaps none.</summary>
    internal static int? ShotOf(IReadOnlyList<Shot> shots, double start, double end)
    {
        int? best = null;
        double most = 0;
        for (int i = 0; i < shots.Count; i++)
        {
            double overlap = Math.Min(end, shots[i].End.Seconds) - Math.Max(start, shots[i].Start.Seconds);
            if (overlap > most || (best is null && overlap == 0 && start >= shots[i].Start.Seconds && start < shots[i].End.Seconds))
                (best, most) = (i, overlap);
        }

        return best;
    }

    /// <summary>
    /// Each shot's stills, decoded and encoded in managed code by ManagedCodecs.Thumbnails: the frame shown at each
    /// time (as ffmpeg's -ss takes it), upright, at most <see cref="VideoDocumentReaderOptions.StillMaxWidth"/> wide, as
    /// a JPEG. MP4 and AVI with H.264 or MJPEG; any other file gets no stills.
    /// </summary>
    void AddStills(string path, IReadOnlyList<Shot> shots, IReadOnlyList<(bool Black, IReadOnlyList<(double Start, double End)> Freezes)> looks, IngestionDocument document, CancellationToken cancellationToken)
    {
        // Every still, in document order: (section, shot, time, file name).
        var stills = new List<(int Section, Shot Shot, TimeSpan At, string Name)>();
        for (int i = 0; i < shots.Count; i++)
        {
            var times = PlannedStills(shots[i].Start.Seconds, shots[i].Duration.Seconds, _options.StillInterval, looks[i].Black, looks[i].Freezes);
            for (int k = 0; k < times.Count; k++)
            {
                string name = times.Count == 1 ? $"shot-{shots[i].Number}.jpg" : $"shot-{shots[i].Number}-{k + 1}.jpg";
                stills.Add((i, shots[i], TimeSpan.FromSeconds(times[k]), name));
            }
        }

        // Each still decodes from the keyframe before it (most of its time: GOPs are often seconds long) and doesn't
        // depend on the others, so they run in parallel, each worker with its own stream.
        var jpegs = new byte[stills.Count][];
        bool unreadable = false;
        var parallel = new ParallelOptions { CancellationToken = cancellationToken, MaxDegreeOfParallelism = _options.StillThreads ?? Environment.ProcessorCount };
        Parallel.For(0, stills.Count, parallel, () => File.OpenRead(path), (j, loop, file) =>
        {
            try
            {
                file.Position = 0;
                var options = new ThumbnailOptions { At = stills[j].At, MaxWidth = _options.StillMaxWidth, MaxHeight = int.MaxValue };
                jpegs[j] = VideoThumbnail.Create(file, options, cancellationToken).ToJpeg(_options.StillQuality);
            }
            catch (Exception e) when (e is NotSupportedException or InvalidDataException)
            {
                unreadable = true; // a container or codec ManagedCodecs doesn't read (MKV, HEVC, MPEG-4 Part 2, ...)
                loop.Stop();
            }

            return file;
        }, file => file.Dispose());
        if (unreadable)
            return; // no stills rather than some

        for (int j = 0; j < stills.Count; j++)
        {
            var (section, shot, at, name) = stills[j];
            document.Sections[section].Elements.Add(Tagged(new IngestionDocumentImage($"![Shot {shot.Number} at {Timecode(at)}]({name})")
            {
                Content = jpegs[j],
                MediaType = "image/jpeg",
            },
                shot));
        }
    }

    static string Timecode(TimeSpan time) => time.ToString(@"hh\:mm\:ss\.fff", System.Globalization.CultureInfo.InvariantCulture);

    static T Tagged<T>(T element, Shot shot)
        where T : IngestionDocumentElement
    {
        element.Metadata[ShotKey] = shot.Number;
        element.Metadata[StartKey] = shot.Start.Seconds;
        element.Metadata[EndKey] = shot.End.Seconds;
        return element;
    }
}
