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
}

/// <summary>
/// A video as an <see cref="IngestionDocument"/>: one section per shot, headed by the shot's times
/// (<c>## 00:00:02.000 – 00:00:04.000</c>), which <c>SectionChunker</c> makes every chunk's context and
/// <c>VectorStoreWriter</c> stores, so a search result says where in the video it is. Each section also holds a
/// paragraph naming the shot and, with FFmpeg's libraries, its middle frame as a JPEG with no alternative text, for
/// <see cref="ImageAlternativeTextEnricher"/> to caption.
/// </summary>
public sealed class VideoDocumentReader(VideoDocumentReaderOptions? options = null) : IngestionDocumentReader
{
    /// <summary>Element metadata: the shot's number, and its start and end in seconds (the end is the next shot's start).</summary>
    public const string ShotKey = "shot";

    public const string StartKey = "start";

    public const string EndKey = "end";

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

    void AddStills(string path, IReadOnlyList<Shot> shots, IngestionDocument document, CancellationToken cancellationToken)
    {
        using var reader = new VideoFrameReader(path);
        using var encoder = new ImageEncoder(new ImageEncoderOptions { Format = ImageFormat.Jpg });
        var size = new ImageFilterOptions { MaxWidth = _options.StillMaxWidth };
        for (int i = 0; i < shots.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var shot = shots[i];
            var middle = TimeSpan.FromSeconds(shot.Start.Seconds + (shot.Duration.Seconds / 2));
            if (!reader.TryTakeAt(middle, out var frame))
                continue;
            using (frame)
            {
                document.Sections[i].Elements.Add(Tagged(new IngestionDocumentImage($"![Shot {shot.Number} at {Timecode(middle)}](shot-{shot.Number}.jpg)")
                {
                    Content = encoder.Encode(frame, size),
                    MediaType = "image/jpeg",
                },
                    shot));
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
