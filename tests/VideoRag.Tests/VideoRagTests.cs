using System.Diagnostics;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataIngestion;
using ShotDetector;

namespace VideoRag.Tests;

/// <summary>
/// Three shots, cuts at frames 50 and 100 (150 frames at 25 fps), as ShotDetector.Tests' Clips.ThreeShots; with a
/// soundtrack, a 440 Hz tone in stereo AAC at 48 kHz; H.264, so ManagedCodecs decodes the stills. Also a copy with no
/// audio, and one in MPEG-4 Part 2, a codec ManagedCodecs doesn't decode.
/// </summary>
public sealed class ThreeShots : IDisposable
{
    readonly string _dir = Directory.CreateTempSubdirectory("videorag-tests-").FullName;

    public ThreeShots()
    {
        Path = System.IO.Path.Combine(_dir, "three.mp4");
        Silent = System.IO.Path.Combine(_dir, "silent.mp4");
        Mpeg4Part2 = System.IO.Path.Combine(_dir, "part2.mp4");
        Ffmpeg("-f lavfi -i testsrc2=s=320x240:r=25:d=2 -f lavfi -i smptebars=s=320x240:r=25:d=2 -f lavfi -i mandelbrot=s=320x240:r=25 "
            + "-f lavfi -i sine=frequency=440:sample_rate=48000:duration=6 "
            + "-filter_complex [2]trim=duration=2,setpts=PTS-STARTPTS[m];[0][1][m]concat=n=3:v=1:a=0,format=yuv420p[v] "
            + $"-map [v] -map 3:a -r 25 -c:v libx264 -crf 18 -g 25 -c:a aac -ac 2 -b:a 128k {Path}");
        Ffmpeg($"-i {Path} -map 0:v -c copy {Silent}");
        Ffmpeg($"-i {Path} -map 0:v -c:v mpeg4 -q:v 3 {Mpeg4Part2}");
    }

    public string Path { get; }

    public string Silent { get; }

    public string Mpeg4Part2 { get; }

    static void Ffmpeg(string arguments)
    {
        var psi = new ProcessStartInfo("ffmpeg", "-v error -y " + arguments) { RedirectStandardError = true };
        using var p = Process.Start(psi)!;
        string err = p.StandardError.ReadToEnd();
        p.WaitForExit();
        Assert.True(p.ExitCode == 0, $"ffmpeg failed: {err}");
    }

    public void Dispose() => Directory.Delete(_dir, true);
}

public sealed class VideoRagTests(ThreeShots clip) : IClassFixture<ThreeShots>
{
    internal static readonly string[] Headers = ["00:00:00.000 – 00:00:02.000", "00:00:02.000 – 00:00:04.000", "00:00:04.000 – 00:00:06.000"];

    /// <summary>FFmpeg 8's libraries, as the other tests take them: set on CI, so the stills are checked there.</summary>
    static readonly string? Libraries = Environment.GetEnvironmentVariable("SHOTDETECTOR_FFMPEG_LIBS") is { Length: > 0 } dir ? dir : null;

    internal static VideoDocumentReaderOptions Options => new() { FfmpegLibraries = Libraries };

    [Fact]
    public async Task ReaderMakesASectionPerShotHeadedByItsTimes()
    {
        var reader = new VideoDocumentReader(Options);
        var document = await reader.ReadAsync(new FileInfo(clip.Path), "three.mp4");

        Assert.Equal(3, document.Sections.Count);
        for (int i = 0; i < 3; i++)
        {
            var elements = document.Sections[i].Elements;
            var header = Assert.IsType<IngestionDocumentHeader>(elements[0]);
            Assert.Equal($"## {Headers[i]}", header.GetMarkdown());
            Assert.Equal(i + 1, header.Metadata[VideoDocumentReader.ShotKey]);
            Assert.Equal(i * 2.0, header.Metadata[VideoDocumentReader.StartKey]);
            Assert.Equal((i + 1) * 2.0, header.Metadata[VideoDocumentReader.EndKey]);
            Assert.Contains($"Shot {i + 1} of 3", Assert.IsType<IngestionDocumentParagraph>(elements[1]).GetMarkdown());
        }

        if (Libraries is not null)
            Assert.True(reader.DetectsInProcess, $"FFmpeg's libraries didn't load from {Libraries}");
        foreach (var section in document.Sections)
        {
            var image = Assert.IsType<IngestionDocumentImage>(Assert.Single(section.Elements, e => e is IngestionDocumentImage));
            Assert.Equal("image/jpeg", image.MediaType);
            byte[] jpeg = image.Content!.Value.ToArray();
            Assert.True(jpeg.Length > 1000 && jpeg[0] == 0xFF && jpeg[1] == 0xD8, "not a JPEG");
            Assert.Null(image.AlternativeText);
        }
    }

    [Fact]
    public async Task StreamIsReadLikeTheFile()
    {
        var reader = new VideoDocumentReader(Options);
        await using var pipe = new NonSeekable(File.OpenRead(clip.Path));
        var document = await reader.ReadAsync(pipe, "three.mp4", "video/mp4");

        Assert.Equal(Headers.Select(h => $"## {h}"), document.Sections.Select(s => s.Elements[0].GetMarkdown()));
    }

    [Fact]
    public async Task PipelineStoresAChunkPerShotWithItsTimesAsContext()
    {
        var chat = new CaptionClient();
        using var index = new VideoIndex(chat, new WordEmbeddings(), WordEmbeddings.Dimensions, Options);

        var result = Assert.Single(await index.AddAsync([new FileInfo(clip.Path)]));
        Assert.True(result.Succeeded, result.Exception?.ToString());

        var hits = await index.SearchAsync("shot", top: 10);
        Assert.Equal(Headers.Select(h => $"## {h}"), hits.Select(h => h.Context).Order(StringComparer.Ordinal));
        // The enricher sent the three stills in one request and its captions became the chunks' text.
        Assert.Equal(1, chat.Requests);
        Assert.Equal(3, chat.Images);
        var bars = Assert.Single(await index.SearchAsync("colour bars", top: 1));
        Assert.Equal($"## {Headers[1]}", bars.Context);
        Assert.Contains(CaptionClient.Captions[1], bars.Content);
    }

    [Fact]
    public void SoundtrackIsTheToneAt16kHzMono()
    {
        var (samples, start) = Soundtrack.Read(clip.Path) ?? throw new Xunit.Sdk.XunitException("no soundtrack");

        Assert.InRange(start.TotalSeconds, -0.05, 0.05);
        Assert.InRange(samples.Length, 6 * 15800, 6 * 16200);
        var middle = samples.AsSpan(16000, 4 * 16000); // away from the AAC encoder's ramp at either end
        Assert.InRange(Frequency(middle, Soundtrack.SampleRate), 435, 445);
        Assert.InRange(Rms(middle), 0.058, 0.066); // lavfi's sine, 1/8 full scale (RMS 0.088), upmixed at -3 dB: 0.0624, as ffmpeg's own decode and downmix
    }

    [Fact]
    public void SoundtrackOfAVideoWithoutAudioIsNull() => Assert.Null(Soundtrack.Read(clip.Silent));

    [Fact]
    public void ResampleKeepsTonesBelowTheNewNyquistAndRemovesThoseAbove()
    {
        float[] low = Soundtrack.Resample(Tone(1000, 44100, 44100), 44100, 16000);
        Assert.Equal(16000, low.Length);
        Assert.InRange(Frequency(low.AsSpan(1000, 14000), 16000), 995, 1005);
        Assert.InRange(Rms(low.AsSpan(1000, 14000)), 0.70, 0.72); // a full-scale sine: 1/sqrt(2)

        float[] high = Soundtrack.Resample(Tone(10000, 48000, 48000), 48000, 16000); // would alias to 6 kHz
        Assert.True(Rms(high.AsSpan(1000, 14000)) < 0.01, $"10 kHz leaked through: RMS {Rms(high.AsSpan(1000, 14000))}");

        float[] up = Soundtrack.Resample(Tone(440, 8000, 8000), 8000, 16000);
        Assert.Equal(16000, up.Length);
        Assert.InRange(Frequency(up.AsSpan(1000, 14000), 16000), 437, 443);
    }

    [Fact]
    public async Task EachShotGetsWhatIsSaidInIt()
    {
        var transcriber = new ScriptTranscriber();
        var reader = new VideoDocumentReader(Options with { Transcriber = transcriber });
        var document = await reader.ReadAsync(new FileInfo(clip.Path), "three.mp4");

        Assert.InRange(transcriber.SampleCount, 6 * 15800, 6 * 16200);
        string[] said = [.. document.Sections.Select(s => string.Concat(s.Elements.OfType<IngestionDocumentParagraph>().Skip(1).Select(p => p.GetMarkdown())))];
        Assert.Equal(["Said: Welcome to the test card.", "Said: These are colour bars, and now", "Said: a fractal."], said);
        var paragraph = document.Sections[1].Elements.OfType<IngestionDocumentParagraph>().Last();
        Assert.Equal(2, paragraph.Metadata[VideoDocumentReader.ShotKey]);
    }

    [Fact]
    public async Task VideoWithoutAudioGetsNoTranscript()
    {
        var transcriber = new ScriptTranscriber();
        var document = await new VideoDocumentReader(Options with { Transcriber = transcriber }).ReadAsync(new FileInfo(clip.Silent), "silent.mp4");

        Assert.Equal(0, transcriber.Calls);
        Assert.All(document.Sections, s => Assert.Single(s.Elements, e => e is IngestionDocumentParagraph));
    }

    [Fact]
    public async Task TranscriptsAreSearchable()
    {
        using var index = new VideoIndex(new CaptionClient(), new WordEmbeddings(), WordEmbeddings.Dimensions, Options with { Transcriber = new ScriptTranscriber() });
        Assert.True(Assert.Single(await index.AddAsync([new FileInfo(clip.Path)])).Succeeded);

        var hit = Assert.Single(await index.SearchAsync("welcome test card", top: 1));
        Assert.Equal($"## {Headers[0]}", hit.Context);
        Assert.Contains("Said: Welcome to the test card.", hit.Content);
    }

    [Theory]
    [InlineData("[MUSIC PLAYING]", "")]
    [InlineData(" [MUSIC PLAYING] [GUNSHOTS] ", "")]
    [InlineData("(laughs)", "")]
    [InlineData("♪ ♪", "")]
    [InlineData("- [door slams] -", "")]
    [InlineData("I'm searching (sighs) for them. [MUSIC PLAYING]", "I'm searching for them.")]
    [InlineData("♪ Over the hills ♪", "Over the hills")]
    [InlineData("Wait [gunshot] , what?", "Wait, what?")]
    [InlineData("Plain speech.", "Plain speech.")]
    public void SoundTagsAreRemoved(string text, string spoken) => Assert.Equal(spoken, VideoDocumentReader.WithoutSoundTags(text));

    [Fact]
    public async Task ShotsWithOnlySoundsSayNothingUnlessTagsAreKept()
    {
        var script = new ScriptTranscriber(
            new(TimeSpan.FromSeconds(0.2), TimeSpan.FromSeconds(1.8), "[MUSIC PLAYING]"),
            new(TimeSpan.FromSeconds(2.2), TimeSpan.FromSeconds(3.8), "I'm searching (sighs) for them. [MUSIC PLAYING]"));

        var removed = await new VideoDocumentReader(Options with { Transcriber = script }).ReadAsync(new FileInfo(clip.Path), "three.mp4");
        var kept = await new VideoDocumentReader(Options with { Transcriber = script, KeepSoundTags = true }).ReadAsync(new FileInfo(clip.Path), "three.mp4");

        Assert.Equal(["", "Said: I'm searching for them.", ""], Said(removed));
        Assert.Equal(["Said: [MUSIC PLAYING]", "Said: I'm searching (sighs) for them. [MUSIC PLAYING]", ""], Said(kept));

        static string[] Said(IngestionDocument document) =>
            [.. document.Sections.Select(s => string.Concat(s.Elements.OfType<IngestionDocumentParagraph>().Select(p => p.GetMarkdown()).Where(t => t.StartsWith(VideoDocumentReader.SaidPrefix, StringComparison.Ordinal))))];
    }

    [Theory]
    [InlineData(0, 2, 8, new[] { 1.0 })]
    [InlineData(0, 8, 8, new[] { 4.0 })]
    [InlineData(0, 24, 8, new[] { 4.0, 12, 20 })]
    [InlineData(10, 10, 8, new[] { 12.5, 17.5 })]
    [InlineData(28, 24.2, 0, new[] { 40.1 })] // no interval: the middle
    public void LongShotsGetAStillPerInterval(double start, double duration, double every, double[] expected) =>
        Assert.Equal(expected, VideoDocumentReader.StillTimes(start, duration, every > 0 ? TimeSpan.FromSeconds(every) : null), (a, b) => Math.Abs(a - b) < 1e-9);

    [Fact]
    public async Task EachIntervalOfAShotGetsItsStill()
    {
        var reader = new VideoDocumentReader(Options with { StillInterval = TimeSpan.FromSeconds(0.8) });
        var document = await reader.ReadAsync(new FileInfo(clip.Path), "three.mp4");

        // Each 2-second shot: three stretches of 2/3 s, stills at 1/3, 1 and 5/3 s into it (timecodes truncate to ms).
        foreach (var (section, i) in document.Sections.Select((s, i) => (s, i)))
        {
            var images = section.Elements.OfType<IngestionDocumentImage>().ToList();
            Assert.Equal(3, images.Count);
            Assert.Equal(
                [$"![Shot {i + 1} at 00:00:0{(i * 2) + 0}.333](shot-{i + 1}-1.jpg)", $"![Shot {i + 1} at 00:00:0{(i * 2) + 1}.000](shot-{i + 1}-2.jpg)", $"![Shot {i + 1} at 00:00:0{(i * 2) + 1}.666](shot-{i + 1}-3.jpg)"],
                images.Select(m => m.GetMarkdown()));
        }
    }

    [Fact]
    public async Task StillsAreTheFramesAtTheirTimes()
    {
        // Shot 2 is SMPTE colour bars: its still's top-left area is the first bar, light grey (75% white).
        var document = await new VideoDocumentReader(Options).ReadAsync(new FileInfo(clip.Path), "three.mp4");
        var image = document.Sections[1].Elements.OfType<IngestionDocumentImage>().Single();
        using var frame = new ManagedCodecs.Mjpeg.MjpegDecoder().Decode(image.Content!.Value.Span);
        Assert.Equal((320, 240), (frame.Width, frame.Height));
        Assert.InRange(frame.GetRow(0, 20)[10], 170, 200); // luma of 75% white, full range: about 191
    }

    [Fact]
    public async Task AVideoManagedCodecsDoesNotDecodeHasNoStills()
    {
        var document = await new VideoDocumentReader(Options).ReadAsync(new FileInfo(clip.Mpeg4Part2), "part2.mp4");

        Assert.Equal(Headers.Select(h => $"## {h}"), document.Sections.Select(s => s.Elements[0].GetMarkdown()));
        Assert.All(document.Sections, s => Assert.DoesNotContain(s.Elements, e => e is IngestionDocumentImage));
    }

    [Fact]
    public async Task DetectorsCanBeChosenAndCombined()
    {
        foreach (var detection in new[]
        {
            new DetectionOptions { Detector = DetectorKind.Content },
            new DetectionOptions { Detectors = [new(DetectorKind.Content), new(DetectorKind.Threshold)] },
        })
        {
            var document = await new VideoDocumentReader(Options with { Detection = detection }).ReadAsync(new FileInfo(clip.Path), "three.mp4");
            Assert.Equal(Headers.Select(h => $"## {h}"), document.Sections.Select(s => s.Elements[0].GetMarkdown()));
        }
    }

    static float[] Tone(double frequency, int rate, int count) =>
        [.. Enumerable.Range(0, count).Select(i => (float)Math.Sin(2 * Math.PI * frequency * i / rate))];

    /// <summary>A tone's frequency from its upward zero crossings.</summary>
    static double Frequency(ReadOnlySpan<float> samples, int rate)
    {
        int first = -1, last = -1, crossings = 0;
        for (int i = 1; i < samples.Length; i++)
        {
            if (samples[i - 1] < 0 && samples[i] >= 0)
            {
                if (first < 0)
                    first = i;
                else
                    crossings++;
                last = i;
            }
        }

        return crossings * (double)rate / (last - first);
    }

    static double Rms(ReadOnlySpan<float> samples)
    {
        double sum = 0;
        foreach (float s in samples)
            sum += s * s;
        return Math.Sqrt(sum / samples.Length);
    }

    /// <summary>The speech model: a fixed script, whatever the audio; the third line straddles the cut at 4 s, mostly before it.</summary>
    internal sealed class ScriptTranscriber(params TranscriptSegment[] script) : ITranscriber
    {
        static readonly TranscriptSegment[] Default =
        [
            new(TimeSpan.FromSeconds(0.2), TimeSpan.FromSeconds(1.5), " Welcome to the test card."),
            new(TimeSpan.FromSeconds(2.5), TimeSpan.FromSeconds(3.5), "These are colour bars,"),
            new(TimeSpan.FromSeconds(3.6), TimeSpan.FromSeconds(4.3), "and now"),
            new(TimeSpan.FromSeconds(4.5), TimeSpan.FromSeconds(5.5), "a fractal."),
        ];

        public int Calls { get; private set; }

        public int SampleCount { get; private set; }

        public async IAsyncEnumerable<TranscriptSegment> TranscribeAsync(ReadOnlyMemory<float> samples, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            Calls++;
            SampleCount = samples.Length;
            await Task.Yield();
            foreach (var segment in script.Length > 0 ? script : Default)
                yield return segment;
        }
    }

    /// <summary>The vision model: a fixed caption per image, as the structured array the enricher asks for.</summary>
    internal sealed class CaptionClient : IChatClient
    {
        internal static readonly string[] Captions = ["A test card with a moving gradient", "Vertical colour bars", "A fractal zoom"];

        public int Requests { get; private set; }

        public int Images { get; private set; }

        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            int count = messages.Last().Contents.OfType<DataContent>().Count(c => c.MediaType == "image/jpeg");
            Requests++;
            Images += count;
            string json = System.Text.Json.JsonSerializer.Serialize(new { data = Captions.Take(count) });
            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, json)));
        }

        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public object? GetService(Type serviceType, object? serviceKey = null) => null;

        public void Dispose()
        {
        }
    }

    /// <summary>The embedding model: a bag of words, each word hashed to one of a few dimensions.</summary>
    internal sealed class WordEmbeddings : IEmbeddingGenerator<string, Embedding<float>>
    {
        internal const int Dimensions = 64;

        public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default) =>
            Task.FromResult(new GeneratedEmbeddings<Embedding<float>>(values.Select(Embed)));

        static Embedding<float> Embed(string text)
        {
            float[] vector = new float[Dimensions];
            vector[0] = 1e-3f; // never all zeros
            foreach (string word in text.ToLowerInvariant().Split([' ', ',', '.', '(', ')'], StringSplitOptions.RemoveEmptyEntries))
            {
                uint hash = 2166136261;
                foreach (char c in word)
                    hash = (hash ^ c) * 16777619;
                vector[1 + (hash % (Dimensions - 1))] += 1;
            }

            return new Embedding<float>(vector);
        }

        public object? GetService(Type serviceType, object? serviceKey = null) => null;

        public void Dispose()
        {
        }
    }

    /// <summary>A stream that can't seek, as a download or a pipe.</summary>
    sealed class NonSeekable(Stream inner) : Stream
    {
        public override bool CanRead => true;

        public override bool CanSeek => false;

        public override bool CanWrite => false;

        public override long Length => throw new NotSupportedException();

        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, count);

        public override void Flush()
        {
        }

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        protected override void Dispose(bool disposing)
        {
            if (disposing)
                inner.Dispose();
            base.Dispose(disposing);
        }
    }
}
