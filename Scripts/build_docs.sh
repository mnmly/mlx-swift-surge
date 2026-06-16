#!/usr/bin/env bash
# Build a static DocC site for MLXSurGe into ./docs (GitHub Pages-ready).
#
# This package depends on mlx-swift, which compiles Metal sources. The SwiftPM
# CLI (`swift package generate-documentation`, used by swift-docc-plugin) can't
# locate the Metal toolchain on this setup, so we drive DocC through
# `xcodebuild docbuild` (which handles Metal) and then transform the resulting
# `.doccarchive` for static hosting with Xcode's `docc`.
#
# Usage:
#   Scripts/build_docs.sh                # build the site into ./docs
#   Scripts/build_docs.sh preview        # build, then open index.html
#   EMIT_LLMS_TXT=1 Scripts/build_docs.sh  # also write docs/llms.txt
#
# Env:
#   SCHEME             Xcode scheme to docbuild. Default: MLXSurGe.
#   TARGETS            DocC module name(s). Default: MLXSurGe.
#   HOSTING_BASE_PATH  GitHub Pages repo path. Default: mlx-swift-SurGe.
#   OUTPUT_DIR         Default: docs.
#   DERIVED_DATA       Default: .xcdd.
#   EMIT_LLMS_TXT=1    Derive docs/llms.txt from the render index.
set -euo pipefail

cd "$(dirname "$0")/.."

SCHEME="${SCHEME:-MLXSurGe}"
TARGETS="${TARGETS:-MLXSurGe}"
HOSTING_BASE_PATH="${HOSTING_BASE_PATH:-mlx-swift-SurGe}"
OUTPUT_DIR="${OUTPUT_DIR:-docs}"
DERIVED_DATA="${DERIVED_DATA:-.xcdd}"

MODE="build"
for arg in "$@"; do
    case "$arg" in
        preview) MODE="preview" ;;
    esac
done

echo ">> docbuild scheme=$SCHEME"
xcodebuild docbuild \
    -scheme "$SCHEME" \
    -destination 'platform=macOS' \
    -derivedDataPath "$DERIVED_DATA" \
    >/dev/null

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

DOCC_BIN="$(xcrun --find docc)"

for TARGET in $TARGETS; do
    archive="$(find "$DERIVED_DATA/Build/Products" -name "${TARGET}.doccarchive" -type d | head -1)"
    if [[ -z "$archive" ]]; then
        echo "error: ${TARGET}.doccarchive not found under $DERIVED_DATA" >&2
        exit 1
    fi
    echo ">> transform-for-static-hosting $TARGET → $OUTPUT_DIR/$TARGET"
    "$DOCC_BIN" process-archive transform-for-static-hosting "$archive" \
        --output-path "$OUTPUT_DIR/$TARGET" \
        --hosting-base-path "${HOSTING_BASE_PATH}/${TARGET}"
done

# Top-level redirect so the Pages root URL doesn't 404.
first_target="${TARGETS%% *}"
first_slug="$(echo "$first_target" | tr '[:upper:]' '[:lower:]')"
redirect_url="/${HOSTING_BASE_PATH}/${first_target}/documentation/${first_slug}/"
cat > "$OUTPUT_DIR/index.html" <<HTML
<!doctype html>
<meta charset="utf-8">
<title>${HOSTING_BASE_PATH}</title>
<meta http-equiv="refresh" content="0; url=${redirect_url}">
<link rel="canonical" href="${redirect_url}">
<p>Redirecting to <a href="${redirect_url}">${redirect_url}</a>.</p>
HTML

if [[ "${EMIT_LLMS_TXT:-0}" == "1" ]]; then
    python3 Scripts/generate_llms_txt.py "$OUTPUT_DIR" $TARGETS
fi

echo
echo "Docs written to $OUTPUT_DIR/. Open $OUTPUT_DIR/$first_target/index.html"

if [[ "$MODE" == "preview" ]]; then
    open "$OUTPUT_DIR/$first_target/index.html" 2>/dev/null || true
fi
