using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataIngestion;
using Microsoft.Extensions.DataIngestion.Chunkers;
using Microsoft.Extensions.VectorData;
using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel.Connectors.InMemory;
using StanzaSharp;
using StanzaSharp.DataIngestion;

namespace VideoRag;

/// <summary>A search hit: the stored chunk, the shot times its context carries, and its named entities (empty without them).</summary>
public sealed record Hit(string Context, string Content, double Score, IReadOnlyList<string> Entities);

/// <summary>
/// Videos in an in-memory vector store, through DataIngestion's own pieces: <see cref="VideoDocumentReader"/>,
/// <see cref="ImageAlternativeTextEnricher"/> (captions the stills with <paramref name="chat"/>), <see cref="SectionChunker"/>
/// (a chunk per shot, its header the context) and <see cref="VectorStoreWriter{T}"/> (embeds with <paramref name="embeddings"/>).
/// With <paramref name="scenes"/>, <see cref="SceneGrouper"/> groups the captioned shots into scenes first: a chunk per scene.
/// With <paramref name="sentences"/> (a StanzaSharp pipeline with at least its tokenizer), StanzaSharp's
/// <see cref="SentenceChunker"/> chunks instead: the same header context, but a section too long for one chunk is cut
/// between sentences, never inside one. With <paramref name="entities"/> (a StanzaSharp pipeline with <c>ner</c>),
/// StanzaSharp's <see cref="EntityEnricher"/> stores each chunk's named entities of <see cref="EntityTypes"/> (from its
/// captions and transcript), which <see cref="SearchAsync"/> returns and can filter by. The index disposes neither pipeline.
/// </summary>
public sealed class VideoIndex(IChatClient chat, IEmbeddingGenerator<string, Embedding<float>> embeddings, int dimensions, VideoDocumentReaderOptions? options = null, SceneGrouperOptions? scenes = null, Pipeline? sentences = null, Pipeline? entities = null)
    : IDisposable
{
    readonly VectorStoreWriter<string> _writer = new(
        new InMemoryVectorStore(new InMemoryVectorStoreOptions { EmbeddingGenerator = embeddings }),
        dimensions,
        new VectorStoreWriterOptions { CollectionName = "shots" });

    /// <summary>
    /// The entity types kept (OntoNotes'): people, organizations, places, groups, events, works and products. The numeric
    /// types (DATE, TIME, CARDINAL, MONEY, ...) are left out: timecodes and captions would fill them with noise.
    /// </summary>
    public static IReadOnlySet<string> EntityTypes { get; } =
        new HashSet<string> { "PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT", "WORK_OF_ART", "PRODUCT" };

    /// <summary>Reads, captions, chunks and stores the videos.</summary>
    public async Task<IReadOnlyList<IngestionResult>> AddAsync(IEnumerable<FileInfo> videos, CancellationToken cancellationToken = default)
    {
        var chunking = new IngestionChunkerOptions(TiktokenTokenizer.CreateForModel("gpt-4o"));
        IngestionChunker<string> chunker = sentences is null ? new SectionChunker(chunking) : new SentenceChunker(sentences, chunking);
        using var pipeline = new IngestionPipeline<string>(new VideoDocumentReader(options), chunker, _writer);
        pipeline.DocumentProcessors.Add(new ImageAlternativeTextEnricher(new EnricherOptions(chat)));
        if (scenes is not null)
            pipeline.DocumentProcessors.Add(new SceneGrouper(embeddings, scenes));
        if (entities is not null)
            pipeline.ChunkProcessors.Add(new EntityEnricher(entities, EntityTypes));
        var results = new List<IngestionResult>();
        await foreach (var result in pipeline.ProcessAsync(videos, cancellationToken).ConfigureAwait(false))
            results.Add(result);
        return results;
    }

    /// <summary>The <paramref name="top"/> chunks nearest the question.</summary>
    /// <param name="entity">Only chunks naming this entity, as the enricher wrote it (e.g. "Barack Obama"); needs <c>entities</c>.</param>
    public async Task<IReadOnlyList<Hit>> SearchAsync(string question, int top = 3, string? entity = null, CancellationToken cancellationToken = default)
    {
        var hits = new List<Hit>();
        if (entity is not null && entities is null)
            throw new InvalidOperationException("Filtering by entity needs the index made with an entities pipeline.");
        var search = new VectorSearchOptions<Dictionary<string, object?>>
        {
            Filter = entity is null ? null : r => ((string[])r[EntityEnricher.MetadataKey]!).Contains(entity),
        };
        await foreach (var hit in _writer.VectorStoreCollection.SearchAsync(question, top, search, cancellationToken).ConfigureAwait(false))
        {
            string[] named = entities is not null && hit.Record.TryGetValue(EntityEnricher.MetadataKey, out var value) && value is string[] list ? list : [];
            hits.Add(new((string)hit.Record["context"]!, (string)hit.Record["content"]!, hit.Score ?? 0, named));
        }
        return hits;
    }

    public void Dispose() => _writer.Dispose();
}
