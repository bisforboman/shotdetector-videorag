# shotdetector-videorag

A Microsoft.Extensions.DataIngestion (MEDI) video reader: one section per shot (ShotDetector from NuGet, MIT), the shot's
middle frame (one per 8 s of a long shot) as a JPEG from ManagedCodecs.Thumbnails for MEDI's image enricher, into Semantic Kernel's in-memory vector store. A sample, not a package.
It moved here from shotdetector's `examples/VideoRag` on 2026-11-01 (the owner's choice; shotdetector's
docs/decisions/2026-11-01-videorag-repo.md and 2026-10-31-videorag-sample.md, scoping in docs/ideas/dataingestion-video-reader.md).

## Layout

- `src/VideoRag`: `VideoDocumentReader`, `SceneGrouper`, `VideoIndex`, `Transcripts.cs` (`Soundtrack` via
  ManagedCodecs, `WhisperTranscriber`), `Program` (the console); README.md there explains the design.
- `tests/VideoRag.Tests`: xunit, offline; makes a 3-shot clip with `ffmpeg`.
- `Directory.Packages.props`: central versions. MEDI is preview and pinned; Dependabot moves it.

## Working here

- Work on a branch and open a PR, don't push to main; squash-merge it once CI is green and nothing is left to push
  (auto-merge is off in this repository).
- Before pushing: `dotnet test ShotDetector.VideoRag.slnx` (ffmpeg with libx264 on PATH; `SHOTDETECTOR_FFMPEG_LIBS` for in-process detection;
  `STANZASHARP_MODELS`, a folder with Stanza's English tokenizer and default_fast ner, for the sentence and entity tests:
  `stanzasharp download models/stanza/en --processors tokenize`, then the same with `--package default_fast
  --processors tokenize,ner`).
- Pinned actions: by commit SHA with the tag in a comment. `.github/actions/btbn-ffmpeg` is copied from shotdetector.
- Slices are tracked in this repository's issues. Transcripts (#2), scenes (#3), StanzaSharp's sentence chunker (#4)
  and entity enricher, and stills via ManagedCodecs 0.7.0 without FFmpeg (#6) are in.
- Always prompt the owner for input (Claude's AskUserQuestion) before choosing what to do next or between options;
  don't settle open choices in prose.
- Commit messages: what and why. Small commits.
