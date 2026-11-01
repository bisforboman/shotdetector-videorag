using System.Numerics.Tensors;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataIngestion;

namespace VideoRag;

/// <summary>How <see cref="SceneGrouper"/> groups shots.</summary>
public sealed record SceneGrouperOptions
{
    /// <summary>
    /// The cosine similarity between neighbouring shots' embeddings at or above which they are one scene. It depends
    /// on the embedding model: 0.6 suits OpenAI's text-embedding-3-small; local models may need another.
    /// </summary>
    public double Threshold { get; init; } = 0.6;

    /// <summary>A scene this long takes no more shots, so a talking head doesn't become one chunk; null: no cap.</summary>
    public TimeSpan? MaxDuration { get; init; } = TimeSpan.FromSeconds(60);
}

/// <summary>
/// Groups <see cref="VideoDocumentReader"/>'s shots into scenes, after the image enricher has captioned the stills:
/// each shot's captions and transcript are embedded, and a shot joins the scene before it when it is similar enough to
/// the shot before it (neighbour to neighbour). A shot with neither captions nor transcript stays on its own.
/// A scene of several shots becomes one section headed <c>## start – end</c> (the chunk's context), each shot under
/// its own <c>### Shot n: start – end</c>; a section must stay flat, since <c>SectionChunker</c> makes a chunk per nested section.
/// A scene of one shot keeps the shot's section as it was.
/// </summary>
public sealed class SceneGrouper(IEmbeddingGenerator<string, Embedding<float>> embeddings, SceneGrouperOptions? options = null) : IngestionDocumentProcessor
{
    /// <summary>The scene header's metadata: the scene's number, and its start and end in seconds (as the shots' <see cref="VideoDocumentReader.StartKey"/>, <see cref="VideoDocumentReader.EndKey"/>).</summary>
    public const string SceneKey = "scene";

    readonly SceneGrouperOptions _options = options ?? new();

    public override async Task<IngestionDocument> ProcessAsync(IngestionDocument document, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(document);
        var shots = document.Sections.Select(Shot.Of).ToList();
        if (shots.Count < 2 || shots.Any(s => s is null))
            return document; // not (only) the reader's shots: leave it alone

        var texts = shots.Select(s => s!.Text).ToList();
        var withText = Enumerable.Range(0, texts.Count).Where(i => texts[i].Length > 0).ToList();
        var vectors = new float[]?[texts.Count];
        if (withText.Count > 1)
        {
            var generated = await embeddings.GenerateAsync(withText.Select(i => texts[i]), cancellationToken: cancellationToken).ConfigureAwait(false);
            for (int k = 0; k < withText.Count; k++)
                vectors[withText[k]] = generated[k].Vector.ToArray();
        }

        var scenes = new List<List<int>> { new() { 0 } };
        for (int i = 1; i < shots.Count; i++)
        {
            var scene = scenes[^1];
            bool similar = vectors[i - 1] is { } a && vectors[i] is { } b && TensorPrimitives.CosineSimilarity(a, b) >= _options.Threshold;
            bool full = _options.MaxDuration is { } max && shots[i]!.End - shots[scene[0]]!.Start > max.TotalSeconds;
            if (similar && !full)
                scene.Add(i);
            else
                scenes.Add([i]);
        }

        var sections = document.Sections.ToList();
        document.Sections.Clear();
        for (int n = 0; n < scenes.Count; n++)
            document.Sections.Add(scenes[n].Count == 1 ? sections[scenes[n][0]] : Merge(n + 1, scenes[n].Select(i => (sections[i], shots[i]!)).ToList()));
        return document;
    }

    static IngestionDocumentSection Merge(int number, List<(IngestionDocumentSection Section, Shot Shot)> shots)
    {
        var first = shots[0].Shot;
        var last = shots[^1].Shot;
        var scene = new IngestionDocumentSection($"scene-{number}");
        var header = new IngestionDocumentHeader($"## {first.From} – {last.To}") { Level = 2 };
        header.Metadata[SceneKey] = number;
        header.Metadata[VideoDocumentReader.StartKey] = first.Start;
        header.Metadata[VideoDocumentReader.EndKey] = last.End;
        scene.Elements.Add(header);
        foreach (var (section, shot) in shots)
        {
            var shotHeader = new IngestionDocumentHeader($"### Shot {shot.Number}: {shot.From} – {shot.To}") { Level = 3 };
            foreach (var (key, value) in shot.Header.Metadata)
                shotHeader.Metadata[key] = value;
            scene.Elements.Add(shotHeader);
            foreach (var element in section.Elements.Skip(1))
                scene.Elements.Add(element);
        }

        return scene;
    }

    /// <summary>
    /// A reader's shot section: its header (<c>## from – to</c>, with the shot's metadata), the header's timecodes, and
    /// the text to compare.
    /// </summary>
    sealed record Shot(IngestionDocumentHeader Header, int Number, double Start, double End, string From, string To, string Text)
    {
        static object? Value(IngestionDocumentElement element, string key) => element.Metadata.TryGetValue(key, out var value) ? value : null;

        public static Shot? Of(IngestionDocumentSection section)
        {
            if (section.Elements is not [IngestionDocumentHeader { Level: 2 } header, ..]
                || header.GetMarkdown().Split(" – ") is not [['#', '#', ' ', .. var from], var to]
                || Value(header, VideoDocumentReader.ShotKey) is not int number
                || Value(header, VideoDocumentReader.StartKey) is not double start
                || Value(header, VideoDocumentReader.EndKey) is not double end)
                return null;
            var parts = section.Elements.Select(e => e switch
            {
                IngestionDocumentImage image => image.AlternativeText,
                IngestionDocumentParagraph paragraph when paragraph.GetMarkdown() is var said && said.StartsWith(VideoDocumentReader.SaidPrefix, StringComparison.Ordinal) => said[VideoDocumentReader.SaidPrefix.Length..],
                _ => null,
            });
            return new(header, number, start, end, from, to, string.Join(' ', parts.Where(p => !string.IsNullOrWhiteSpace(p))));
        }
    }
}
