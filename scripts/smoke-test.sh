#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TMP_DIR="$(mktemp -d)"
NPM_CACHE_DIR="$TMP_DIR/npm-cache"
TARBALL=""
INSTALL_DIR="$TMP_DIR/install"

cleanup() {
  rm -rf "$TMP_DIR"
  if [ -n "$TARBALL" ]; then
    rm -f "$PROJECT_DIR/$TARBALL"
  fi
}
trap cleanup EXIT

echo "═══════════════════════════════════════════════════"
echo "  voxtral-transcribe-ts pre-publish smoke test"
echo "═══════════════════════════════════════════════════"
echo ""

echo "Step 1: Build..."
cd "$PROJECT_DIR"
export npm_config_cache="$NPM_CACHE_DIR"
npm run build
echo "  ✓ Build succeeded"

echo "Step 2: Pack..."
TARBALL="$(npm pack 2>/dev/null | tail -1)"
echo "  ✓ Packed: $TARBALL"

echo "Step 3: Install tarball in a pristine temp project..."
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"
npm init -y --silent >/dev/null
npm install "$PROJECT_DIR/$TARBALL"
echo "  ✓ Installed tarball and resolved dependencies"

PKG_DIR="$INSTALL_DIR/node_modules/voxtral-transcribe-ts"

echo "Step 4: Verify packaged files..."
[ -f "$PKG_DIR/dist/index.node.js" ] || { echo "  ✗ dist/index.node.js missing"; exit 1; }
[ -f "$PKG_DIR/dist/index.browser.js" ] || { echo "  ✗ dist/index.browser.js missing"; exit 1; }
[ -f "$PKG_DIR/dist/index.node.d.ts" ] || { echo "  ✗ dist/index.node.d.ts missing"; exit 1; }
[ -f "$PKG_DIR/dist/index.browser.d.ts" ] || { echo "  ✗ dist/index.browser.d.ts missing"; exit 1; }
echo "  ✓ Dist files are bundled"

echo "Step 5: Verify runtime dependencies were installed in the temp project..."
[ -d "$INSTALL_DIR/node_modules/onnxruntime-node" ] || { echo "  ✗ onnxruntime-node missing from clean install"; exit 1; }
[ ! -L "$INSTALL_DIR/node_modules/onnxruntime-node" ] || { echo "  ✗ onnxruntime-node should be installed, not symlinked"; exit 1; }
[ -d "$INSTALL_DIR/node_modules/@huggingface/transformers" ] || { echo "  ✗ @huggingface/transformers missing from clean install"; exit 1; }
echo "  ✓ Runtime dependencies were installed from npm"

echo "Step 6: Verify published exports..."
cd "$INSTALL_DIR"
node --input-type=module <<'EOF'
const expected = [
  "DEFAULT_MISTRAL_TRANSCRIPTION_MODEL",
  "MISTRAL_REALTIME_TRANSCRIPTION_MODEL",
  "MistralVoxtralApiClient",
  "MistralVoxtralApiTranscriber",
  "VoxtralTranscriber",
  "createTranscriber",
  "transcribeAudio",
  "transcribeFile",
  "transcribeFileWithMistral",
  "createDefaultAudioDecoderBackend",
  "createDefaultInferenceBackend",
];

const modules = [
  ["root", await import("voxtral-transcribe-ts")],
  ["node", await import("voxtral-transcribe-ts/node")],
  ["browser", await import("voxtral-transcribe-ts/browser")],
];

for (const [label, mod] of modules) {
  for (const key of expected) {
    if (!(key in mod)) {
      throw new Error(`${label} export missing: ${key}`);
    }
  }
}

if (typeof modules[1][1].FfmpegDecoder !== "function") {
  throw new Error("node export missing: FfmpegDecoder");
}

if (typeof modules[2][1].BrowserNativeAudioDecoder !== "function") {
  throw new Error("browser export missing: BrowserNativeAudioDecoder");
}

console.log("  ✓ Root, node, and browser exports verified");
EOF

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ Smoke test passed"
echo "═══════════════════════════════════════════════════"
