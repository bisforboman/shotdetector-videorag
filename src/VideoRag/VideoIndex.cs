using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataIngestion;
using Microsoft.Extensions.DataIngestion.Chunkers;
using Microsoft.Extensions.VectorData;
using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel.Connectors.InMemory;

namespace VideoRag;

/// <summary>A search hit: the stored chunk and the shot times its context carries.</summary>
public sealed record Hit(string Context, string Content, double Score);

/// <summary>
/// Videos in an in-memory vector store, through DataIngestion's own pieces: <see cref="VideoDocumentReader"/>,
/// <see cref="ImageAlternativeTextEnricher"/> (captions the stills with <paramref name="chat"/>), <see cref="SectionChunker"/>
/// (a chunk per shot, its header the context) and <see cref="VectorStoreWriter{T}"/> (embeds with <paramref name="embeddings"/>).
/// </summary>
public sealed class VideoIndex(IChatClient chat, IEmbeddingGenerator<string, Embedding<float>> embeddings, int dimensions, VideoDocumentReaderOptions? options = null)
    : IDisposable
{
    readonly VectorStoreWriter<string> _writer = new(
        new InMemoryVectorStore(new InMemoryVectorStoreOptions { EmbeddingGenerator = embeddings }),
        dimensions,
        new VectorStoreWriterOptions { CollectionName = "shots" });

    /// <summary>Reads, captions, chunks and stores the videos.</summary>
    public async Task<IReadOnlyList<IngestionResult>> AddAsync(IEnumerable<FileInfo> videos, CancellationToken cancellationToken = default)
    {
        using var pipeline = new IngestionPipeline<string>(
            new VideoDocumentReader(options),
            new SectionChunker(new IngestionChunkerOptions(TiktokenTokenizer.CreateForModel("gpt-4o"))),
            _writer);
        pipeline.DocumentProcessors.Add(new ImageAlternativeTextEnricher(new EnricherOptions(chat)));
        var results = new List<IngestionResult>();
        await foreach (var result in pipeline.ProcessAsync(videos, cancellationToken).ConfigureAwait(false))
            results.Add(result);
        return results;
    }

    /// <summary>The <paramref name="top"/> chunks nearest the question.</summary>
    public async Task<IReadOnlyList<Hit>> SearchAsync(string question, int top = 3, CancellationToken cancellationToken = default)
    {
        var hits = new List<Hit>();
        await foreach (var hit in _writer.VectorStoreCollection.SearchAsync(question, top, cancellationToken: cancellationToken).ConfigureAwait(false))
            hits.Add(new((string)hit.Record["context"]!, (string)hit.Record["content"]!, hit.Score ?? 0));
        return hits;
    }

    public void Dispose() => _writer.Dispose();
}
