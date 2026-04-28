# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-04-28

### Added
- `--codec {libx264,mpeg4}` CLI flag and `codec` kwarg on `encode`, `make_video`, `make_videos`. Default remains `libx264`. Use `mpeg4` on systems where ffmpeg lacks libx264 (e.g. Sherlock HPC modules).

### Fixed
- Encoder now fails with a clear `ValueError` on unsupported codec strings instead of relying on ffmpeg to error.

## [0.1.0] - 2026-04-27

### Added
- Initial release.
- `Renderer` and `FrameSource` Protocols.
- In-tree renderers: `mosaic` (default), `triplet`.
- In-tree sources: `FmriprepFrameSource`, `ManifestFrameSource`.
- CLI: `brm bids|list|render` with named exit codes.
- On-disk mean cache with mtime invalidation.
- MP4 encoding via system ffmpeg.
