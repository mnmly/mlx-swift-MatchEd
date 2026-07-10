# mlx-swift-MatchED

MLX (Apple Silicon) port of MatchED / PiDiNet crisp edge detection. The
consumer-facing library is `MatchEDKit`; `matched` is the CLI; the SwiftUI demo
lives under `Examples/MatchEDDemo`.

## Build & test

- Build/test with **xcodebuild**, not `swift test` (the latter can't load MLX's
  `default.metallib`):
  - `xcodebuild -scheme mlx-swift-MatchED-Package -destination 'platform=macOS' test`
- The model runs the *converted* (vanilla-conv) form; PDC→conv folding happens
  in `Scripts/*.py` at weight-conversion time. See `README.md`.

## Documentation

`MatchEDKit` ships DocC-generated reference docs (see
`Sources/MatchEDKit/Documentation.docc/` and `Scripts/build_docs.sh`).
**`///` doc comments on public/`open` symbols are published** to the static site
at https://mnmly.github.io/mlx-swift-MatchED/ (once Pages is enabled) and, if
`EMIT_LLMS_TXT=1` is used, into `docs/llms.txt`.

When you add or modify a `public` or `open` declaration:

- Write a `///` doc comment. One-sentence summary, then a paragraph if the *why*
  is non-obvious. Skip restating what the signature already says.
- Document each parameter with `- Parameter name:` (use the **internal** name
  when there's an external label — DocC warns otherwise).
- Cross-reference related symbols with double-backtick links, e.g.
  `` ``PiDiNet/loadWeights(url:dtype:)`` ``. DocC link syntax is
  signature-sensitive: `foo(_:)` and `foo(_:_:)` are different.
- When you add a new top-level symbol that belongs in the curated sidebar, add
  it under the appropriate `## Topics` group in
  `Sources/MatchEDKit/Documentation.docc/MatchEDKit.md`. Topics are organized by
  *user task*, not alphabetic order.

Verify before declaring documentation work done:

```bash
Scripts/build_docs.sh
```

Expect exit 0 and no new "doesn't exist at" or "external name used to document
parameter" warnings attributable to your changes.
