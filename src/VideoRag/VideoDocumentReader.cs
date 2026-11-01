using FrameReader;
using Microsoft.Extensions.DataIngestion;
using ShotDetector;

namespace VideoRag;

/// <summary>How <see cref="VideoDocumentReader"/> reads a video.</summary>
public sealed record VideoDocumentReaderOptions
{
    /// <summary>The detector and its settings; null: <see cref="ShotDetection"/>'s defaults (adaptive).</summary>
    public DetectionOptions? Detection { get; init; }

    /// <summary>
    /// Folder of FFmpeg 8's shared libraries (an FFmpeg 8 "shared" build, or FrameReader.Native's): with them each shot
    /// gets its middle frame as a JPEG, and detection decodes in-process. Null: the usual places, else no stills (and
    /// detection through the ffmpeg executable on PATH).
    /// </summary>
    public string? FfmpegLibraries { get; init; }

    /// <summary>The stills' largest width, in pixels; narrower videos keep their own.</summary>
    public int StillMaxWidth { get; init; } = 640;

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

    /// <summary>Whether this reader's sections get stills (FFmpeg's libraries load).</summary>
    public bool HasStills => global::FrameReader.FFmpegLibraries.CanLoad(_options.FfmpegLibraries);

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
        bool stills = HasStills; // loads the libraries first, so detection decodes in-process too
        var result = await ShotDetection.DetectAsync(path, _options.Detection, cancellationToken).ConfigureAwait(false);
        var document = new IngestionDocument(identifier);
        foreach (var shot in result.Shots)
            document.Sections.Add(Section(shot, result.Shots.Count));
        if (_options.Transcriber is { } transcriber)
            await AddTranscriptsAsync(path, transcriber, _options.KeepSoundTags, result.Shots, document, cancellationToken).ConfigureAwait(false);
        if (stills)
            await Task.Run(() => AddStills(path, result.Shots, document, cancellationToken), cancellationToken).ConfigureAwait(false);
        return document;
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

    void AddStills(string path, IReadOnlyList<Shot> shots, IngestionDocument document, CancellationToken cancellationToken)
    {
        using var reader = new VideoFrameReader(path);
        using var encoder = new ImageEncoder(new ImageEncoderOptions { Format = ImageFormat.Jpg });
        var size = new ImageFilterOptions { MaxWidth = _options.StillMaxWidth };
        for (int i = 0; i < shots.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var shot = shots[i];
            var times = StillTimes(shot.Start.Seconds, shot.Duration.Seconds, _options.StillInterval);
            for (int k = 0; k < times.Count; k++)
            {
                var at = TimeSpan.FromSeconds(times[k]);
                if (!reader.TryTakeAt(at, out var frame))
                    continue;
                using (frame)
                {
                    string file = times.Count == 1 ? $"shot-{shot.Number}.jpg" : $"shot-{shot.Number}-{k + 1}.jpg";
                    document.Sections[i].Elements.Add(Tagged(new IngestionDocumentImage($"![Shot {shot.Number} at {Timecode(at)}]({file})")
                    {
                        Content = encoder.Encode(frame, size),
                        MediaType = "image/jpeg",
                    },
                        shot));
                }
            }
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
