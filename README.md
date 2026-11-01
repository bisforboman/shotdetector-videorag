# shotdetector-videorag

Video as searchable, timestamped chunks for retrieval-augmented generation in .NET. A
[Microsoft.Extensions.DataIngestion](https://learn.microsoft.com/dotnet/ai/conceptual/data-ingestion) reader makes one
section per shot, using [ShotDetector](https://github.com/bisforboman/shotdetector) from NuGet, so an answer can cite
"00:12:34 in the video". MIT.

It's a sample, not a package: DataIngestion is still a monthly preview. How it works, how to run it and which models it
uses: [src/VideoRag/README.md](src/VideoRag/README.md).

## Layout

- `src/VideoRag`: `VideoDocumentReader` (the reader), `VideoIndex` (the pipeline into Semantic Kernel's in-memory
  vector store) and a console that answers questions.
- `tests/VideoRag.Tests`: the reader and pipeline on a 3-shot clip made with ffmpeg, offline (stub chat and embedding
  clients).

## Checks

`dotnet test ShotDetector.VideoRag.slnx` needs `ffmpeg` on PATH. With `SHOTDETECTOR_FFMPEG_LIBS` set to a folder of
FFmpeg 8's shared libraries, it also checks the stills; CI sets it.

## History

Started as `examples/VideoRag` in the shotdetector repository (bisforboman/shotdetector#641, issue #635), moved here on
2026-11-01 (shotdetector's docs/decisions/2026-11-01-videorag-repo.md).
