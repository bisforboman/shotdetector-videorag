// Video RAG with Microsoft.Extensions.DataIngestion (README.md): videos in, one chunk per shot in an
// in-memory vector store, answers with the shot's times.
//
//   dotnet run --project src/VideoRag -- [--ffmpeg-libs <dir>] [--detector adaptive|content|threshold|histogram|hash[,...]] [--still-every 8] [--no-qc] [--whisper-model <ggml file> [--keep-sound-tags]] [--scenes [--scene-threshold 0.6] [--scene-max 60]] [--sentences <stanza model dir>] [--entities <stanza model dir> [--entity NAME]] [--stanza-lang en] [--ask "question"]... <video>...
//
// --entities stores each chunk's named entities (people, organizations, places, ...: StanzaSharp's EntityEnricher on
// Stanza's default_fast ner, `stanzasharp download models/stanza/en --package default_fast --processors tokenize,ner`
// after the tokenize download below, about 135 MB) and prints them with every hit; --entity NAME searches only chunks
// naming NAME. --stanza-lang (or --sentences-lang) is both Stanza pipelines' language.
//
// --sentences chunks with StanzaSharp's SentenceChunker (cuts only between sentences) on Stanza's tokenizer from
// <stanza model dir>, a language's folder: `dotnet tool install -g StanzaSharp.Tool`, then
// `stanzasharp download models/stanza/en --processors tokenize` (about 1.2 MB).
//
// --scenes groups neighbouring shots whose captions and transcript embed alike into one chunk (a scene), up to
// --scene-max seconds; --scene-threshold is the cosine similarity that counts as alike (it depends on the embedding model).
//
// --detector picks ShotDetector's detectors (default adaptive; several, comma-separated, run together: content,threshold
// also cuts at fades to black). --still-every N takes a still per N seconds of a long shot (default 8; 0: the middle
// only), since a dissolve the detector doesn't cut at leaves one shot over several scenes. --keep-sound-tags keeps the
// transcript's [MUSIC], (laughs) and ♪, which are removed by default. Black and still pictures are found on the
// detection pass: a black shot says so and gets no stills, a still picture gets one; --no-qc turns that off.
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
ShotDetector.DetectionOptions? detection = null;
TimeSpan? stillInterval = TimeSpan.FromSeconds(8);
bool keepSoundTags = false;
bool qc = true;
string? sentencesDir = null;
string sentencesLang = "en";
string? entitiesDir = null;
string? entity = null;
var questions = new List<string>();
var videos = new List<FileInfo>();
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--ffmpeg-libs" when i + 1 < args.Length:
            libs = args[++i];
            break;
        case "--detector" when i + 1 < args.Length:
            var kinds = args[++i].Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
                .Select(k => Enum.Parse<ShotDetector.DetectorKind>(k, ignoreCase: true)).ToList();
            detection = kinds.Count == 1
                ? new ShotDetector.DetectionOptions { Detector = kinds[0] }
                : new ShotDetector.DetectionOptions { Detectors = [.. kinds.Select(k => new ShotDetector.DetectorSettings(k))] };
            break;
        case "--still-every" when i + 1 < args.Length:
            double every = double.Parse(args[++i], CultureInfo.InvariantCulture);
            stillInterval = every > 0 ? TimeSpan.FromSeconds(every) : null;
            break;
        case "--no-qc":
            qc = false;
            break;
        case "--keep-sound-tags":
            keepSoundTags = true;
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
        case "--entities" when i + 1 < args.Length:
            entitiesDir = args[++i];
            break;
        case "--entity" when i + 1 < args.Length:
            entity = args[++i];
            break;
        case "--stanza-lang" or "--sentences-lang" when i + 1 < args.Length:
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
    Console.Error.WriteLine("Usage: VideoRag [--ffmpeg-libs <dir>] [--detector adaptive|content|threshold|histogram|hash[,...]] [--still-every 8] [--no-qc] [--whisper-model <ggml file> [--keep-sound-tags]] [--scenes [--scene-threshold 0.6] [--scene-max 60]] [--sentences <stanza model dir>] [--entities <stanza model dir> [--entity NAME]] [--stanza-lang en] [--ask \"question\"]... <video>...");
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
var options = new VideoDocumentReaderOptions
{
    FfmpegLibraries = libs,
    Detection = detection,
    StillInterval = stillInterval,
    Transcriber = transcriber,
    KeepSoundTags = keepSoundTags,
    Qc = qc,
};
if (!new VideoDocumentReader(options).DetectsInProcess)
    Console.Error.WriteLine("FFmpeg 8's shared libraries didn't load (--ffmpeg-libs): detection through ffmpeg on PATH.");

if (entity is not null && entitiesDir is null)
{
    Console.Error.WriteLine("--entity needs --entities.");
    return 2;
}

using var sentences = sentencesDir is null ? null : LoadStanza(sentencesDir, "default", "tokenize");
using var entities = entitiesDir is null ? null : LoadStanza(entitiesDir, "default_fast", "tokenize,ner");
if ((sentencesDir is not null && sentences is null) || (entitiesDir is not null && entities is null))
    return 2;

using var index = new VideoIndex(chat, embeddings, dimensions, options, scenes, sentences, entities);
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

StanzaSharp.Pipeline? LoadStanza(string dir, string package, string processors)
{
    try
    {
        return StanzaSharp.Pipeline.Load(dir, new StanzaSharp.PipelineOptions { Language = sentencesLang, Package = package, Processors = processors });
    }
    catch (Exception e) when (e is IOException or ArgumentException or InvalidDataException)
    {
        Console.Error.WriteLine($"Stanza's {processors} didn't load from {dir}: {e.Message}");
        Console.Error.WriteLine($"Download it: dotnet tool install -g StanzaSharp.Tool; stanzasharp download {dir} --lang {sentencesLang} --processors tokenize"
            + (package == "default" ? "" : $"; stanzasharp download {dir} --lang {sentencesLang} --package {package} --processors {processors}"));
        return null;
    }
}

async Task Answer(string question)
{
    Console.WriteLine(question);
    foreach (var hit in await index.SearchAsync(question, entity: entity))
    {
        Console.WriteLine($"  {hit.Score:F3}  {hit.Content.ReplaceLineEndings(" ")}"); // the content starts with its header, the context
        if (hit.Entities.Count > 0)
            Console.WriteLine($"         entities: {string.Join(", ", hit.Entities)}");
    }
}
