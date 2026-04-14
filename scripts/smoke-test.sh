#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TMP_DIR="$(mktemp -d)"
NPM_CACHE_DIR="$TMP_DIR/npm-cache"
TARBALL=""

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

echo "Step 3: Extract tarball in a clean directory..."
mkdir -p "$TMP_DIR/node_modules"
tar -xzf "$PROJECT_DIR/$TARBALL" -C "$TMP_DIR"
mv "$TMP_DIR/package" "$TMP_DIR/node_modules/voxtral-transcribe-ts"
mkdir -p "$TMP_DIR/node_modules/@huggingface"
ln -s "$PROJECT_DIR/node_modules/@huggingface/transformers" "$TMP_DIR/node_modules/@huggingface/transformers"
ln -s "$PROJECT_DIR/node_modules/onnxruntime-common" "$TMP_DIR/node_modules/onnxruntime-common"
ln -s "$PROJECT_DIR/node_modules/onnxruntime-node" "$TMP_DIR/node_modules/onnxruntime-node"
echo "  ✓ Extracted tarball"

PKG_DIR="$TMP_DIR/node_modules/voxtral-transcribe-ts"

echo "Step 4: Verify packaged files..."
[ -f "$PKG_DIR/dist/index.node.js" ] || { echo "  ✗ dist/index.node.js missing"; exit 1; }
[ -f "$PKG_DIR/dist/index.browser.js" ] || { echo "  ✗ dist/index.browser.js missing"; exit 1; }
[ -f "$PKG_DIR/dist/index.node.d.ts" ] || { echo "  ✗ dist/index.node.d.ts missing"; exit 1; }
[ -f "$PKG_DIR/dist/index.browser.d.ts" ] || { echo "  ✗ dist/index.browser.d.ts missing"; exit 1; }
echo "  ✓ Dist files are bundled"

echo "Step 5: Verify published exports..."
cd "$TMP_DIR"
node --input-type=module <<'EOF'
const expected = [
  "VoxtralTranscriber",
  "createTranscriber",
  "transcribeAudio",
  "transcribeFile",
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
