using Microsoft.Extensions.DataIngestion;
using Microsoft.Extensions.DataIngestion.Chunkers;
using Microsoft.ML.Tokenizers;
using static VideoRag.Tests.VideoRagTests;

namespace VideoRag.Tests;

public sealed class SceneTests(ThreeShots clip) : IClassFixture<ThreeShots>
{
    [Fact]
    public async Task SimilarNeighboursBecomeOneSceneWithEachShotUnderItsOwnHeader()
    {
        var document = Document(
            (2, "A red car drives along the coast road", null),
            (2, "Vertical colour bars on a test card", "These are the colour bars"),
            (2, "Vertical colour bars on a test card", "colour bars"));

        await new SceneGrouper(new WordEmbeddings()).ProcessAsync(document);

        Assert.Equal(["## 00:00:00.000 – 00:00:02.000", "## 00:00:02.000 – 00:00:06.000"], Heads(document));
        var scene = document.Sections[1].Elements;
        Assert.Equal(
            ["## 00:00:02.000 – 00:00:06.000", "### Shot 2: 00:00:02.000 – 00:00:04.000", "Shot 2 of 3.", "Said: These are the colour bars", "image",
             "### Shot 3: 00:00:04.000 – 00:00:06.000", "Shot 3 of 3.", "Said: colour bars", "image"],
            scene.Select(e => e is IngestionDocumentImage ? "image" : e.GetMarkdown()));
        Assert.Equal(2, scene[0].Metadata[SceneGrouper.SceneKey]);
        Assert.Equal(2.0, scene[0].Metadata[VideoDocumentReader.StartKey]);
        Assert.Equal(6.0, scene[0].Metadata[VideoDocumentReader.EndKey]);
        Assert.Equal(3, scene[5].Metadata[VideoDocumentReader.ShotKey]);
        Assert.Equal(4.0, scene[5].Metadata[VideoDocumentReader.StartKey]);
    }

    [Fact]
    public async Task ShotsWithNeitherCaptionNorTranscriptStayApart()
    {
        var document = Document((2, null, null), (2, null, null), (2, null, null));

        await new SceneGrouper(new WordEmbeddings()).ProcessAsync(document);

        Assert.Equal(3, document.Sections.Count);
        Assert.All(document.Sections, s => Assert.DoesNotContain(s.Elements, e => e is IngestionDocumentHeader { Level: 3 }));
    }

    [Fact]
    public async Task ASceneStopsAtItsLongestDuration()
    {
        var document = Document((30, "colour bars", null), (30, "colour bars", null), (30, "colour bars", null));

        await new SceneGrouper(new WordEmbeddings(), new() { MaxDuration = TimeSpan.FromSeconds(60) }).ProcessAsync(document);

        Assert.Equal(["## 00:00:00.000 – 00:01:00.000", "## 00:01:00.000 – 00:01:30.000"], Heads(document));
    }

    [Fact]
    public async Task ThresholdDecides()
    {
        var document = Document((2, "colour bars on a test card", null), (2, "colour bars and a tone", null));

        await new SceneGrouper(new WordEmbeddings(), new() { Threshold = 0.99 }).ProcessAsync(document);

        Assert.Equal(2, document.Sections.Count);
    }

    [Fact]
    public async Task TheChunkerMakesOneChunkPerSceneWithTheScenesTimesAsContext()
    {
        var document = Document((2, "A red car", null), (2, "colour bars", null), (2, "colour bars", null));
        await new SceneGrouper(new WordEmbeddings()).ProcessAsync(document);

        var chunks = new List<IngestionChunk<string>>();
        await foreach (var chunk in new SectionChunker(new IngestionChunkerOptions(TiktokenTokenizer.CreateForModel("gpt-4o"))).ProcessAsync(document))
            chunks.Add(chunk);

        Assert.Equal(["## 00:00:00.000 – 00:00:02.000", "## 00:00:02.000 – 00:00:06.000"], chunks.Select(c => c.Context));
        Assert.Contains("### Shot 3: 00:00:04.000 – 00:00:06.000", chunks[1].Content);
    }

    [Fact]
    public async Task PipelineStoresAChunkPerScene()
    {
        // Shots 2 and 3 talk about the same thing; shot 1 doesn't. With stills, the stub captions join in (shot 3's is
        // "A fractal zoom"), and shots 2 and 3 still share "the colour bars".
        var transcriber = new ScriptTranscriber(
            new(TimeSpan.FromSeconds(0.5), TimeSpan.FromSeconds(1.5), "Welcome everyone."),
            new(TimeSpan.FromSeconds(2.5), TimeSpan.FromSeconds(3.5), "Look at the colour bars."),
            new(TimeSpan.FromSeconds(4.5), TimeSpan.FromSeconds(5.5), "Still the colour bars."));
        using var index = new VideoIndex(new CaptionClient(), new WordEmbeddings(), WordEmbeddings.Dimensions,
            Options with { Transcriber = transcriber }, new SceneGrouperOptions { Threshold = 0.3 });

        var result = Assert.Single(await index.AddAsync([new FileInfo(clip.Path)]));
        Assert.True(result.Succeeded, result.Exception?.ToString());
        Assert.Equal([$"## {Headers[0]}", "## 00:00:02.000 – 00:00:06.000"], Heads(result.Document!));
        var hit = Assert.Single(await index.SearchAsync("colour bars", top: 1));
        Assert.Equal($"## 00:00:02.000 – 00:00:06.000", hit.Context);
        Assert.Contains("### Shot 2", hit.Content);
        Assert.Contains("### Shot 3", hit.Content);
    }

    /// <summary>Sections as <see cref="VideoDocumentReader"/> makes them, back to back: each shot's length, caption and transcript.</summary>
    static IngestionDocument Document(params (double Seconds, string? Caption, string? Said)[] shots)
    {
        var document = new IngestionDocument("video.mp4");
        double start = 0;
        for (int i = 0; i < shots.Length; i++)
        {
            var (seconds, caption, said) = shots[i];
            double end = start + seconds;
            var section = new IngestionDocumentSection($"shot-{i + 1}");
            section.Elements.Add(Tag(new IngestionDocumentHeader($"## {Code(start)} – {Code(end)}") { Level = 2 }));
            section.Elements.Add(Tag(new IngestionDocumentParagraph($"Shot {i + 1} of {shots.Length}.")));
            if (said is not null)
                section.Elements.Add(Tag(new IngestionDocumentParagraph(VideoDocumentReader.SaidPrefix + said)));
            if (caption is not null)
                section.Elements.Add(Tag(new IngestionDocumentImage($"![Shot {i + 1}](shot-{i + 1}.jpg)") { AlternativeText = caption, MediaType = "image/jpeg" }));
            document.Sections.Add(section);
            start = end;

            T Tag<T>(T element)
                where T : IngestionDocumentElement
            {
                element.Metadata[VideoDocumentReader.ShotKey] = i + 1;
                element.Metadata[VideoDocumentReader.StartKey] = end - seconds;
                element.Metadata[VideoDocumentReader.EndKey] = end;
                return element;
            }
        }

        return document;
    }

    static IEnumerable<string> Heads(IngestionDocument document) => document.Sections.Select(s => s.Elements[0].GetMarkdown());

    static string Code(double seconds) => TimeSpan.FromSeconds(seconds).ToString(@"hh\:mm\:ss\.fff", System.Globalization.CultureInfo.InvariantCulture);
}
