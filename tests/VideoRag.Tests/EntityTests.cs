using StanzaSharp;
using static VideoRag.Tests.VideoRagTests;

namespace VideoRag.Tests;

/// <summary>Stanza's English default_fast ner from <c>STANZASHARP_MODELS</c>, loaded once (see <see cref="StanzaTokenizer"/>).</summary>
public sealed class StanzaNer : IDisposable
{
    public StanzaNer()
    {
        if (Environment.GetEnvironmentVariable("STANZASHARP_MODELS") is not { Length: > 0 } dir)
        {
            Assert.True(Environment.GetEnvironmentVariable("CI") is null, "CI sets STANZASHARP_MODELS for the entity tests");
            return;
        }

        Pipeline = StanzaSharp.Pipeline.Load(dir, new PipelineOptions { Package = "default_fast", Processors = "tokenize,ner" });
    }

    public Pipeline? Pipeline { get; }

    public void Dispose() => Pipeline?.Dispose();
}

/// <summary>StanzaSharp's EntityEnricher: each chunk's named entities stored, returned and filtered by.</summary>
public sealed class EntityTests(ThreeShots clip, StanzaNer stanza) : IClassFixture<ThreeShots>, IClassFixture<StanzaNer>
{
    // Shot 2 names a person and a city, and a date, a type the index leaves out.
    static ScriptTranscriber Script() => new(
        new(TimeSpan.FromSeconds(0.5), TimeSpan.FromSeconds(1.5), "Welcome everyone."),
        new(TimeSpan.FromSeconds(2.2), TimeSpan.FromSeconds(3.8), "Barack Obama spoke in Paris on Monday."),
        new(TimeSpan.FromSeconds(4.5), TimeSpan.FromSeconds(5.5), "The colour bars again."));

    [Fact]
    public async Task HitsCarryTheirChunksEntities()
    {
        if (stanza.Pipeline is not { } nlp)
            return;
        using var index = new VideoIndex(new CaptionClient(), new WordEmbeddings(), WordEmbeddings.Dimensions,
            Options with { Transcriber = Script() }, entities: nlp);
        Assert.True(Assert.Single(await index.AddAsync([new FileInfo(clip.Path)])).Succeeded);

        var hits = await index.SearchAsync("shot", top: 10);
        var obama = Assert.Single(hits, h => h.Content.Contains("Barack Obama", StringComparison.Ordinal));
        Assert.Equal($"## {Headers[1]}", obama.Context);
        Assert.Contains("Barack Obama", obama.Entities);
        Assert.Contains("Paris", obama.Entities);
        Assert.DoesNotContain("Monday", obama.Entities); // DATE: not one of VideoIndex.EntityTypes
        Assert.All(hits.Where(h => h != obama), h => Assert.DoesNotContain("Barack Obama", h.Entities));
    }

    [Fact]
    public async Task SearchCanBeLimitedToChunksNamingAnEntity()
    {
        if (stanza.Pipeline is not { } nlp)
            return;
        using var index = new VideoIndex(new CaptionClient(), new WordEmbeddings(), WordEmbeddings.Dimensions,
            Options with { Transcriber = Script() }, entities: nlp);
        Assert.True(Assert.Single(await index.AddAsync([new FileInfo(clip.Path)])).Succeeded);

        // "colour bars" alone finds shot 3 first; naming Paris leaves only shot 2.
        var hit = Assert.Single(await index.SearchAsync("colour bars", top: 10, entity: "Paris"));
        Assert.Equal($"## {Headers[1]}", hit.Context);
        Assert.Empty(await index.SearchAsync("colour bars", top: 10, entity: "London"));
    }

    [Fact]
    public async Task WithoutAnEntitiesPipelineThereAreNone()
    {
        using var index = new VideoIndex(new CaptionClient(), new WordEmbeddings(), WordEmbeddings.Dimensions, Options);
        Assert.True(Assert.Single(await index.AddAsync([new FileInfo(clip.Path)])).Succeeded);

        Assert.All(await index.SearchAsync("shot", top: 10), h => Assert.Empty(h.Entities));
        await Assert.ThrowsAsync<InvalidOperationException>(() => index.SearchAsync("shot", entity: "Paris"));
    }
}
