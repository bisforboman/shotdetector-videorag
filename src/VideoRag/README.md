# VideoRag: a video reader for Microsoft.Extensions.DataIngestion

A sample, not a package ([the scoping note](https://github.com/bisforboman/shotdetector/blob/main/docs/ideas/dataingestion-video-reader.md)). It turns a video
into searchable chunks with [Microsoft.Extensions.DataIngestion](https://learn.microsoft.com/dotnet/ai/conceptual/data-ingestion)
(MEDI) and answers a question with the shot it found and that shot's times.

```
video ──VideoDocumentReader──► one section per shot ──ImageAlternativeTextEnricher──► captioned stills
      ──SceneGrouper (--scenes)──► alike neighbours as one section
      ──SectionChunker──► a chunk per shot (or scene), its "## start – end" header as context ──VectorStoreWriter──► in-memory store
```

- **`VideoDocumentReader`** (an `IngestionDocumentReader`) detects the shots with `ShotDetection.DetectAsync` and makes
  one `IngestionDocumentSection` per shot: a header `## 00:00:02.000 – 00:00:04.000`, a paragraph naming the shot, and,
  when FFmpeg 8's shared libraries load, the shot's middle frame as a JPEG (FrameReader's `ImageEncoder`, at most
  640 px wide) with no alternative text yet. Each element's metadata has the shot's number and its start and end in
  seconds. A `Stream` that isn't a file's is copied to a temporary file first, since the stills seek.
- **Transcripts**: with an `ITranscriber` (`WhisperTranscriber`: whisper.cpp through Whisper.net, a ggml model file)
  each shot also gets a paragraph `Said: …` of the speech in it; a segment goes to the shot it overlaps most. The
  soundtrack is decoded in managed code by [ManagedCodecs](https://github.com/bisforboman/ManagedCodecs) (AAC LC and
  HE-AAC in MP4 or ADTS), downmixed and resampled to Whisper's 16 kHz mono (`Soundtrack`, a windowed-sinc resampler).
  No FFmpeg is involved; other containers (MKV, WebM, AVI's audio) and codecs (Opus, MP3) get no transcript.
- **Why the header**: MEDI's chunkers don't carry element metadata into the chunks, but `SectionChunker` makes a
  section's first header the context of every chunk in it, and `VectorStoreWriter` stores that context. So each stored
  record says where in the video it is, with no custom chunker.
- **`ImageAlternativeTextEnricher`** (MEDI's own) captions every still with a vision model; the chunker then embeds the
  caption with the transcript. That's the visual and spoken search.
- **`SceneGrouper`** (an `IngestionDocumentProcessor`, opt-in with `--scenes`) runs after the enricher: it embeds each
  shot's captions and transcript and merges a shot into the scene before it when its cosine similarity to the previous
  shot is at least `Threshold` (0.6, `--scene-threshold`; it depends on the embedding model), until the scene reaches
  `MaxDuration` (60 s, `--scene-max`). A shot with neither caption nor transcript stays on its own. A scene is one
  flat section: `## start – end` (the chunk's context), then each shot under `### Shot n: start – end`. Flat, because
  `SectionChunker` makes a chunk per nested section.
- **`VideoIndex`** puts the pipeline together on Semantic Kernel's in-memory vector store and searches it.

## Run it

```
dotnet run --project src/VideoRag -- --ffmpeg-libs <dir> --whisper-model ggml-base.en.bin --ask "where are the colour bars?" video.mp4
```

Without `--ask` it asks for questions on the console. Each answer lists the nearest chunks with their score, the shot's
times and its text.

- **FFmpeg**: `--ffmpeg-libs` (or `SHOTDETECTOR_FFMPEG_LIBS`) names a folder with FFmpeg 8's shared libraries: an
  FFmpeg 8 "shared" build, or the ones in the [FrameReader.Native](https://github.com/bisforboman/shotdetector/blob/main/src/FrameReader.Native/README.md) packages
  (LGPL-2.1-or-later; the sample doesn't reference them). Without them the shots still become sections, with no stills,
  and detection runs through the `ffmpeg` executable on PATH.
- **Whisper**: `--whisper-model` (or `VIDEORAG_WHISPER_MODEL`) names a ggml model file, e.g. `ggml-base.en.bin` from
  [whisper.cpp's models](https://huggingface.co/ggerganov/whisper.cpp). Nothing is downloaded; without one, no
  transcripts. The language is detected.
- **Models**: any OpenAI-compatible endpoint, through Microsoft.Extensions.AI.OpenAI.

  | Variable | Default | |
  |---|---|---|
  | `OPENAI_API_KEY` | (needed for OpenAI) | Any value for a local server. |
  | `OPENAI_BASE_URL` | OpenAI | e.g. `http://localhost:11434/v1` for Ollama. |
  | `VIDEORAG_CHAT_MODEL` | `gpt-4o-mini` | A vision model: it captions the stills. |
  | `VIDEORAG_EMBEDDING_MODEL` | `text-embedding-3-small` | |
  | `VIDEORAG_DIMENSIONS` | `1536` | The embedding model's vector size. |

## Caveats

- MEDI is preview; the sample pins `10.11.0-preview.1.26513.7` (Directory.Packages.props) and may need changes on the
  next one.
- A shot is the unit unless `--scenes`: a long shot is one chunk (the chunker splits it only past its token limit).
  Scenes compare neighbours only, so a slow drift (A like B, B like C) can chain into one scene; the cap bounds it.
- A Whisper segment is a sentence or so; one that spans a cut lands in one shot, the one it overlaps most.
- AOT and trimming are untested here.

The test, tests/VideoRag.Tests, runs the reader and the whole pipeline offline on a 3-shot clip with a 440 Hz AAC
soundtrack, with a stub chat client, a scripted transcriber and a bag-of-words embedding generator; no Whisper model.
