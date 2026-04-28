# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-27

### Added
- Initial release.
- `Renderer` and `FrameSource` Protocols.
- In-tree renderers: `mosaic` (default), `triplet`.
- In-tree sources: `FmriprepFrameSource`, `ManifestFrameSource`.
- CLI: `brm bids|list|render` with named exit codes.
- On-disk mean cache with mtime invalidation.
- MP4 encoding via system ffmpeg.
