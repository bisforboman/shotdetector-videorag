using System.Diagnostics;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataIngestion;
using ShotDetector;

namespace VideoRag.Tests;

/// <summary>Three shots, cuts at frames 50 and 100 (150 frames at 25 fps), as ShotDetector.Tests' Clips.ThreeShots.</summary>
public sealed class ThreeShots : IDisposable
{
    readonly string _dir = Directory.CreateTempSubdirectory("videorag-tests-").FullName;

    public ThreeShots()
    {
        Path = System.IO.Path.Combine(_dir, "three.mp4");
        var psi = new ProcessStartInfo("ffmpeg",
            "-v error -y -f lavfi -i testsrc2=s=320x240:r=25:d=2 -f lavfi -i smptebars=s=320x240:r=25:d=2 -f lavfi -i mandelbrot=s=320x240:r=25 "
            + "-filter_complex [2]trim=duration=2,setpts=PTS-STARTPTS[m];[0][1][m]concat=n=3:v=1:a=0,format=yuv420p "
            + $"-r 25 -c:v mpeg4 -q:v 3 {Path}")
        { RedirectStandardError = true };
        using var p = Process.Start(psi)!;
        string err = p.StandardError.ReadToEnd();
        p.WaitForExit();
        Assert.True(p.ExitCode == 0, $"ffmpeg failed: {err}");
    }

    public string Path { get; }

    public void Dispose() => Directory.Delete(_dir, true);
}

public sealed class VideoRagTests(ThreeShots clip) : IClassFixture<ThreeShots>
{
    static readonly string[] Headers = ["00:00:00.000 – 00:00:02.000", "00:00:02.000 – 00:00:04.000", "00:00:04.000 – 00:00:06.000"];

    /// <summary>FFmpeg 8's libraries, as the other tests take them: set on CI, so the stills are checked there.</summary>
    static readonly string? Libraries = Environment.GetEnvironmentVariable("SHOTDETECTOR_FFMPEG_LIBS") is { Length: > 0 } dir ? dir : null;

    static VideoDocumentReaderOptions Options => new() { FfmpegLibraries = Libraries };

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
            Assert.True(reader.HasStills, $"FFmpeg's libraries didn't load from {Libraries}");
        if (!reader.HasStills)
            return;
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
        if (!new VideoDocumentReader(Options).HasStills)
            return;

        // The enricher sent the three stills in one request and its captions became the chunks' text.
        Assert.Equal(1, chat.Requests);
        Assert.Equal(3, chat.Images);
        var bars = Assert.Single(await index.SearchAsync("colour bars", top: 1));
        Assert.Equal($"## {Headers[1]}", bars.Context);
        Assert.Contains(CaptionClient.Captions[1], bars.Content);
    }

    /// <summary>The vision model: a fixed caption per image, as the structured array the enricher asks for.</summary>
    sealed class CaptionClient : IChatClient
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
    sealed class WordEmbeddings : IEmbeddingGenerator<string, Embedding<float>>
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
