// Video RAG with Microsoft.Extensions.DataIngestion (README.md): videos in, one chunk per shot in an
// in-memory vector store, answers with the shot's times.
//
//   dotnet run --project src/VideoRag -- [--ffmpeg-libs <dir>] [--whisper-model <ggml file>] [--scenes [--scene-threshold 0.6] [--scene-max 60]] [--sentences <stanza model dir> [--sentences-lang en]] [--ask "question"]... <video>...
//
// --sentences chunks with StanzaSharp's SentenceChunker (cuts only between sentences) on Stanza's tokenizer from
// <stanza model dir>, a language's folder: `dotnet tool install -g StanzaSharp.Tool`, then
// `stanzasharp download models/stanza/en --processors tokenize` (about 1.2 MB).
//
// --scenes groups neighbouring shots whose captions and transcript embed alike into one chunk (a scene), up to
// --scene-max seconds; --scene-threshold is the cosine similarity that counts as alike (it depends on the embedding model).
//
// With a Whisper model (--whisper-model or VIDEORAG_WHISPER_MODEL, a ggml file such as ggml-base.en.bin) each shot
// also gets what's said in it; the soundtrack is decoded in managed code (AAC in MP4 or ADTS).
// The models: any OpenAI-compatible endpoint (OPENAI_BASE_URL, e.g. http://localhost:11434/v1 for Ollama;
// OPENAI_API_KEY), VIDEORAG_CHAT_MODEL (a vision model, captions the stills; default gpt-4o-mini),
// VIDEORAG_EMBEDDING_MODEL (default text-embedding-3-small) and VIDEORAG_DIMENSIONS (its vector size; default 1536).
using System.ClientModel;
using System.Globalization;
using Microsoft.Extensions.AI;
using OpenAI;
using VideoRag;

string? libs = Environment.GetEnvironmentVariable("SHOTDETECTOR_FFMPEG_LIBS");
string? whisperModel = Environment.GetEnvironmentVariable("VIDEORAG_WHISPER_MODEL");
SceneGrouperOptions? scenes = null;
string? sentencesDir = null;
string sentencesLang = "en";
var questions = new List<string>();
var videos = new List<FileInfo>();
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--ffmpeg-libs" when i + 1 < args.Length:
            libs = args[++i];
            break;
        case "--whisper-model" when i + 1 < args.Length:
            whisperModel = args[++i];
            break;
        case "--scenes":
            scenes ??= new();
            break;
        case "--scene-threshold" when i + 1 < args.Length:
            scenes = (scenes ?? new()) with { Threshold = double.Parse(args[++i], CultureInfo.InvariantCulture) };
            break;
        case "--scene-max" when i + 1 < args.Length:
            scenes = (scenes ?? new()) with { MaxDuration = TimeSpan.FromSeconds(double.Parse(args[++i], CultureInfo.InvariantCulture)) };
            break;
        case "--sentences" when i + 1 < args.Length:
            sentencesDir = args[++i];
            break;
        case "--sentences-lang" when i + 1 < args.Length:
            sentencesLang = args[++i];
            break;
        case "--ask" when i + 1 < args.Length:
            questions.Add(args[++i]);
            break;
        default:
            videos.Add(new FileInfo(args[i]));
            break;
    }
}

if (videos.Count == 0)
{
    Console.Error.WriteLine("Usage: VideoRag [--ffmpeg-libs <dir>] [--whisper-model <ggml file>] [--scenes [--scene-threshold 0.6] [--scene-max 60]] [--sentences <stanza model dir> [--sentences-lang en]] [--ask \"question\"]... <video>...");
    return 2;
}

string? baseUrl = Environment.GetEnvironmentVariable("OPENAI_BASE_URL");
string key = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? (baseUrl is null ? "" : "unused");
if (key.Length == 0)
{
    Console.Error.WriteLine("Set OPENAI_API_KEY, or OPENAI_BASE_URL for a local OpenAI-compatible server.");
    return 2;
}

var client = new OpenAIClient(new ApiKeyCredential(key), baseUrl is null ? new() : new() { Endpoint = new Uri(baseUrl) });
var chat = client.GetChatClient(Environment.GetEnvironmentVariable("VIDEORAG_CHAT_MODEL") ?? "gpt-4o-mini").AsIChatClient();
var embeddings = client.GetEmbeddingClient(Environment.GetEnvironmentVariable("VIDEORAG_EMBEDDING_MODEL") ?? "text-embedding-3-small").AsIEmbeddingGenerator();
int dimensions = int.Parse(Environment.GetEnvironmentVariable("VIDEORAG_DIMENSIONS") ?? "1536", CultureInfo.InvariantCulture);

if (whisperModel is { Length: > 0 } && !File.Exists(whisperModel))
{
    Console.Error.WriteLine($"No Whisper model at {whisperModel}.");
    return 2;
}

using var transcriber = whisperModel is { Length: > 0 } ? new WhisperTranscriber(whisperModel) : null;
var options = new VideoDocumentReaderOptions { FfmpegLibraries = libs, Transcriber = transcriber };
if (!new VideoDocumentReader(options).HasStills)
    Console.Error.WriteLine("FFmpeg 8's shared libraries didn't load (--ffmpeg-libs): shots without stills, detection through ffmpeg on PATH.");

StanzaSharp.Pipeline? sentences = null;
if (sentencesDir is not null)
{
    try
    {
        sentences = StanzaSharp.Pipeline.Load(sentencesDir, new StanzaSharp.PipelineOptions { Language = sentencesLang, Processors = "tokenize" });
    }
    catch (Exception e) when (e is IOException or ArgumentException or InvalidDataException)
    {
        Console.Error.WriteLine($"Stanza's tokenizer didn't load from {sentencesDir}: {e.Message}");
        Console.Error.WriteLine($"Download it: dotnet tool install -g StanzaSharp.Tool; stanzasharp download {sentencesDir} --lang {sentencesLang} --processors tokenize");
        return 2;
    }
}

using var sentencesPipeline = sentences;
using var index = new VideoIndex(chat, embeddings, dimensions, options, scenes, sentences);
foreach (var result in await index.AddAsync(videos))
{
    Console.WriteLine(result.Succeeded
        ? $"{result.DocumentId}: {result.Document!.Sections.Count} {(scenes is null ? "shots" : "scenes")}"
        : $"{result.DocumentId}: failed, {result.Exception!.Message}");
}

if (questions.Count == 0)
{
    Console.Write("Question (empty to quit): ");
    while (Console.ReadLine() is { Length: > 0 } line)
    {
        await Answer(line);
        Console.Write("Question (empty to quit): ");
    }
}

foreach (string question in questions)
    await Answer(question);
return 0;

async Task Answer(string question)
{
    Console.WriteLine(question);
    foreach (var hit in await index.SearchAsync(question))
        Console.WriteLine($"  {hit.Score:F3}  {hit.Content.ReplaceLineEndings(" ")}"); // the content starts with its header, the context
}
