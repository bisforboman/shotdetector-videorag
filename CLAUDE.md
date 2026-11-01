# shotdetector-videorag

A Microsoft.Extensions.DataIngestion (MEDI) video reader: one section per shot (ShotDetector from NuGet, MIT), the shot's
middle frame as a JPEG for MEDI's image enricher, into Semantic Kernel's in-memory vector store. A sample, not a package.
It moved here from shotdetector's `examples/VideoRag` on 2026-11-01 (the owner's choice; shotdetector's
docs/decisions/2026-11-01-videorag-repo.md and 2026-10-31-videorag-sample.md, scoping in docs/ideas/dataingestion-video-reader.md).

## Layout

- `src/VideoRag`: `VideoDocumentReader`, `VideoIndex`, `Program` (the console); README.md there explains the design.
- `tests/VideoRag.Tests`: xunit, offline; makes a 3-shot clip with `ffmpeg`.
- `Directory.Packages.props`: central versions. MEDI is preview and pinned; Dependabot moves it.

## Working here

- Work on a branch and open a PR, don't push to main; turn auto-merge on once nothing is left to push.
- Before pushing: `dotnet test ShotDetector.VideoRag.slnx` (ffmpeg on PATH; `SHOTDETECTOR_FFMPEG_LIBS` for the stills).
- Pinned actions: by commit SHA with the tag in a comment. `.github/actions/btbn-ffmpeg` is copied from shotdetector.
- The next slices are tracked in this repository's issues: transcripts with Whisper.net (#2), shots grouped into
  scenes (#3), the StanzaSharp chunker once bisforboman/StanzaSharp#233 exists (#4).
- Always prompt the owner for input (Claude's AskUserQuestion) before choosing what to do next or between options;
  don't settle open choices in prose.
- Commit messages: what and why. Small commits.
