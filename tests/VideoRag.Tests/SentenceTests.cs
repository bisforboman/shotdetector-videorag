using Microsoft.Extensions.DataIngestion;
using Microsoft.ML.Tokenizers;
using StanzaSharp;
using StanzaSharp.DataIngestion;
using static VideoRag.Tests.VideoRagTests;

namespace VideoRag.Tests;

/// <summary>Stanza's English tokenizer from <c>STANZASHARP_MODELS</c> (a language folder, as StanzaSharp's tests take it), loaded once.</summary>
public sealed class StanzaTokenizer : IDisposable
{
    public StanzaTokenizer()
    {
        if (Environment.GetEnvironmentVariable("STANZASHARP_MODELS") is not { Length: > 0 } dir)
        {
            // CI downloads the models (ci.yml), so there they must load; elsewhere these tests have nothing to run on.
            Assert.True(Environment.GetEnvironmentVariable("CI") is null, "CI sets STANZASHARP_MODELS for the sentence tests");
            return;
        }

        Pipeline = StanzaSharp.Pipeline.Load(dir, new PipelineOptions { Processors = "tokenize" });
    }

    public Pipeline? Pipeline { get; }

    public void Dispose() => Pipeline?.Dispose();
}

/// <summary>StanzaSharp's <see cref="SentenceChunker"/> in place of MEDI's <c>SectionChunker</c> (#4).</summary>
public sealed class SentenceTests(ThreeShots clip, StanzaTokenizer stanza) : IClassFixture<ThreeShots>, IClassFixture<StanzaTokenizer>
{
    static readonly Tokenizer Tokens = TiktokenTokenizer.CreateForModel("gpt-4o");

    [Fact]
    public async Task ChunksKeepTheShotTimesAsContext()
    {
        if (stanza.Pipeline is not { } nlp)
            return;
        using var index = new VideoIndex(new CaptionClient(), new WordEmbeddings(), WordEmbeddings.Dimensions,
            Options with { Transcriber = new ScriptTranscriber() }, sentences: nlp);

        var result = Assert.Single(await index.AddAsync([new FileInfo(clip.Path)]));
        Assert.True(result.Succeeded, result.Exception?.ToString());

        var hits = await index.SearchAsync("shot", top: 10);
        Assert.Equal(Headers.Select(h => $"## {h}"), hits.Select(h => h.Context).Order(StringComparer.Ordinal));
        var welcome = Assert.Single(await index.SearchAsync("welcome test card", top: 1));
        Assert.Equal($"## {Headers[0]}", welcome.Context);
        Assert.Contains("Said: Welcome to the test card.", welcome.Content);
    }

    [Fact]
    public async Task ASceneStaysOneChunk()
    {
        if (stanza.Pipeline is not { } nlp)
            return;
        var document = SceneTests.Document((2, "A red car", null), (2, "colour bars", "Look at the bars."), (2, "colour bars", "Still the bars."));
        await new SceneGrouper(new WordEmbeddings()).ProcessAsync(document);

        var chunks = await Chunk(new SentenceChunker(nlp, new IngestionChunkerOptions(Tokens)), document);

        Assert.Equal(["## 00:00:00.000 – 00:00:02.000", "## 00:00:02.000 – 00:00:06.000"], chunks.Select(c => c.Context));
        Assert.Contains("Shot 2: 00:00:02.000 – 00:00:04.000.", chunks[1].Content);
        Assert.Contains("Said: Still the bars.", chunks[1].Content);
    }

    [Fact]
    public async Task ALongShotIsCutBetweenSentences()
    {
        if (stanza.Pipeline is not { } nlp)
            return;
        string[] sentences = [.. Enumerable.Range(1, 12).Select(i => $"Line {i} of the interview says something about the colour bars, Dr. Smith noted.")];
        var document = SceneTests.Document((20, null, string.Join(' ', sentences)));

        var chunks = await Chunk(new SentenceChunker(nlp, new IngestionChunkerOptions(Tokens) { MaxTokensPerChunk = 60, OverlapTokens = 0 }), document);

        Assert.True(chunks.Count > 2, $"{chunks.Count} chunks");
        Assert.All(chunks, c => Assert.Equal("## 00:00:00.000 – 00:00:20.000", c.Context));
        Assert.All(chunks, c => Assert.True(Tokens.CountTokens(c.Content) <= 60, c.Content));
        // Every sentence whole in one chunk ("Dr." doesn't end one: Stanza's splitter, not a full stop, decides).
        Assert.All(sentences, s => Assert.Single(chunks, c => c.Content.Contains(s, StringComparison.Ordinal)));
    }

    static async Task<List<IngestionChunk<string>>> Chunk(IngestionChunker<string> chunker, IngestionDocument document)
    {
        var chunks = new List<IngestionChunk<string>>();
        await foreach (var chunk in chunker.ProcessAsync(document))
            chunks.Add(chunk);
        return chunks;
    }
}
