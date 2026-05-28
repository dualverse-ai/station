#!/usr/bin/env bash
set -euo pipefail

# Setup Lean Mathlib and a Station.Theory package so Theory Room submissions can import them.

CLEAR_SETUP=false
for arg in "$@"; do
  case "$arg" in
    --clear) CLEAR_SETUP=true ;;
    *)
      echo "Usage: $0 [--clear]" >&2
      exit 1
      ;;
  esac
done

PREFERRED_TOOLCHAIN="${THEORY_LEAN_TOOLCHAIN:-leanprover/lean4:v4.27.0-rc1}"
TOOLCHAIN_DIR="$HOME/.elan/toolchains/${PREFERRED_TOOLCHAIN//\//--}"
TOOLCHAIN_DIR="${TOOLCHAIN_DIR//:/---}"
PREFERRED_LEAN="$TOOLCHAIN_DIR/bin/lean"
PREFERRED_LAKE="$TOOLCHAIN_DIR/bin/lake"
MATHLIB_REV="${THEORY_MATHLIB_REV:-}"
# By default, pin mathlib to the tag matching the Lean toolchain (avoids Lake auto-bumping toolchains
# when mathlib HEAD moves ahead of our preferred Lean version).
if [ -z "${MATHLIB_REV}" ]; then
  case "$PREFERRED_TOOLCHAIN" in
    leanprover/lean4:v*) MATHLIB_REV="${PREFERRED_TOOLCHAIN#*:}" ;;
  esac
fi
MATHLIB_REQ_SUFFIX=""
if [ -n "${MATHLIB_REV}" ]; then
  # Keep quotes literal in the generated lakefile.
  MATHLIB_REQ_SUFFIX=" @ \"${MATHLIB_REV}\""
fi

if [ -x "$HOME/.elan/bin/elan" ]; then
  export PATH="$HOME/.elan/bin:$PATH"
elif ! command -v lean >/dev/null 2>&1 && [ -x "$HOME/.elan/bin/lean" ]; then
  export PATH="$HOME/.elan/bin:$PATH"
fi

if [ ! -x "$PREFERRED_LEAN" ] || [ ! -x "$PREFERRED_LAKE" ]; then
  echo "Installing required toolchain: $PREFERRED_TOOLCHAIN"
  elan toolchain install "$PREFERRED_TOOLCHAIN"
fi

export PATH="$(dirname "$PREFERRED_LEAN"):$PATH"
LEAN_BIN="$PREFERRED_LEAN"
LAKE_BIN="$PREFERRED_LAKE"
echo "Using lean at: $LEAN_BIN"
echo "Lean version: $($LEAN_BIN --version | head -n1)"

ROOT_DIR="$(pwd)"
REPO_HASH="$(printf "%s" "$ROOT_DIR" | md5sum | cut -c1-8)"
DEFAULT_SETUP_DIR="${THEORY_SETUP_DIR:-$HOME/.cache/station_theory_$REPO_HASH}"
REPO_SETUP_LINK="$ROOT_DIR/.theory_setup"

echo "Installing Mathlib oleans for current toolchain..."
MATHLIB_DIR="$DEFAULT_SETUP_DIR/.lake/packages/mathlib"
NEED_MATHLIB=true
if [ -d "$MATHLIB_DIR" ]; then
  NEED_MATHLIB=false
fi
if [ "${REBUILD_ONLY:-false}" = "true" ]; then
  if [ "$NEED_MATHLIB" = true ]; then
    echo "REBUILD_ONLY=true but mathlib cache missing at $MATHLIB_DIR."
    echo "Populate the cache first (run scripts/setup_theory.sh once online) then retry."
    exit 1
  fi
  echo "REBUILD_ONLY set: using cached mathlib at $MATHLIB_DIR; skipping mathlib download."
else
  "$LAKE_BIN" env -- mathlib get
fi

if [ -d "$REPO_SETUP_LINK" ] && [ ! -L "$REPO_SETUP_LINK" ]; then
  echo "Removing existing .theory_setup directory to relocate under $DEFAULT_SETUP_DIR for speed..."
  rm -rf "$REPO_SETUP_LINK"
fi

SETUP_DIR="$DEFAULT_SETUP_DIR"
mkdir -p "$(dirname "$SETUP_DIR")"

if [ "$CLEAR_SETUP" = true ] && [ -d "$SETUP_DIR" ]; then
  case "$SETUP_DIR" in
    "$HOME/.cache/station_theory_"*)
      echo "Clearing existing Station.Theory setup at $SETUP_DIR"
      rm -rf "$SETUP_DIR"
      ;;
    *)
      echo "Refusing to clear unexpected directory: $SETUP_DIR" >&2
      exit 1
      ;;
  esac
fi

if [ -d "$SETUP_DIR" ]; then
  echo "Existing Station.Theory setup found at $SETUP_DIR; reusing."
else
  echo "Creating Station.Theory lean project in $SETUP_DIR"
  mkdir -p "$SETUP_DIR"
  cd "$SETUP_DIR"
  "$LAKE_BIN" init StationTheory
fi

# Ensure repo-local symlink points to the setup dir
cd "$SETUP_DIR"
if [ ! -L "$REPO_SETUP_LINK" ]; then
  ln -s "$SETUP_DIR" "$REPO_SETUP_LINK" 2>/dev/null || true
fi

LAKEFILE="lakefile.lean"
if [ ! -f "$LAKEFILE" ] || ! grep -q "mathlib" "$LAKEFILE"; then
  echo "Configuring mathlib dependency..."
  cat > "$LAKEFILE" <<EOF
import Lake
open Lake DSL

package stationTheory

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"${MATHLIB_REQ_SUFFIX}

lean_lib StationTheory where
  roots := #[\`Station.Theory, \`Station.TheoryEnv]
EOF
fi

# If the project already exists, ensure mathlib is pinned to our requested revision/toolchain.
# Without this, `lake update` may pull a newer mathlib that bumps `lean-toolchain` forward.
python3 - "$LAKEFILE" "$MATHLIB_REV" <<'PY'
import sys
from pathlib import Path

lakefile = Path(sys.argv[1])
rev = sys.argv[2].strip()
if not lakefile.exists() or not rev:
    sys.exit(0)

lines = lakefile.read_text(encoding="utf-8").splitlines(True)
if not any("require mathlib from git" in l for l in lines):
    sys.exit(0)

# Ensure the mathlib URL line has an inline `@ "rev"` (Lake rejects `@` on its own line).
url = '"https://github.com/leanprover-community/mathlib4.git"'
out = []
i = 0
patched = False
while i < len(lines):
    line = lines[i]
    if (not patched) and (url in line):
        indent = line[: len(line) - len(line.lstrip(" "))]
        out.append(f'{indent}{url} @ "{rev}"\n')
        patched = True
        # Drop a following standalone `@ "..."` line if present (older script versions wrote this form).
        if i + 1 < len(lines) and lines[i + 1].lstrip().startswith("@ "):
            i += 2
            continue
        i += 1
        continue
    out.append(line)
    i += 1

if patched:
    lakefile.write_text("".join(out), encoding="utf-8")
PY

# Ensure Station.TheoryEnv is listed as a root so it gets built.
python3 - "$LAKEFILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.exists():
    sys.exit(0)

text = path.read_text()
if "Station.TheoryEnv" in text:
    sys.exit(0)

old = "roots := #[`Station.Theory]"
new = "roots := #[`Station.Theory, `Station.TheoryEnv]"
if old in text:
    path.write_text(text.replace(old, new))
    sys.exit(0)

lines = text.splitlines()
out = []
inserted = False
for line in lines:
    out.append(line)
    if not inserted and line.strip().startswith("lean_lib StationTheory"):
        out.append("  roots := #[`Station.Theory, `Station.TheoryEnv]")
        inserted = True

if inserted:
    out_text = "\n".join(out)
    if text.endswith("\n"):
        out_text += "\n"
    path.write_text(out_text)
PY

# Pin the toolchain so Lake/mathlib stay aligned with the Lean binary we use
echo "$PREFERRED_TOOLCHAIN" > "$SETUP_DIR/lean-toolchain"

mkdir -p Station
# Ensure we can write the Theory files; bail with guidance if permissions are wrong
if [ -f Station/Theory.lean ] && [ ! -w Station/Theory.lean ]; then
  echo "ERROR: Cannot write $SETUP_DIR/Station/Theory.lean (permission denied)."
  echo "Please ensure you own $SETUP_DIR (e.g., chown -R $(whoami) $SETUP_DIR) or remove it and rerun this script."
  exit 1
fi
if [ -f Station/TheoryEnv.lean ] && [ ! -w Station/TheoryEnv.lean ]; then
  echo "ERROR: Cannot write $SETUP_DIR/Station/TheoryEnv.lean (permission denied)."
  echo "Please ensure you own $SETUP_DIR (e.g., chown -R $(whoami) $SETUP_DIR) or remove it and rerun this script."
  exit 1
fi
cat > Station/Theory.lean <<'EOF'
import Mathlib
import Station.TheoryEnv

/-!
Station.Theory aggregates verified lemmas/theories via Station.TheoryEnv.
-/

EOF

# Populate Station.TheoryEnv from station_data item content if available.
INDEX_FILE="$ROOT_DIR/station_data/rooms/theory/index.json"
LEMMA_FILE="$ROOT_DIR/station_data/rooms/theory/lemmas.yamll"
THEORY_FILE="$ROOT_DIR/station_data/rooms/theory/theories.yamll"
python3 - "$INDEX_FILE" "$LEMMA_FILE" "$THEORY_FILE" > Station/TheoryEnv.lean <<'PY'
import json
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

def load_legacy(path: Path):
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            if yaml:
                items.append(yaml.safe_load(line))
            else:
                items.append(json.loads(line))
        except Exception:
            continue
    return items

try:
    index_path = Path(sys.argv[1])
    lemma_path = Path(sys.argv[2])
    theory_path = Path(sys.argv[3])

    items = []
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        for kind in ("lemma", "theory"):
            for key, item in (data.get(kind) or {}).items():
                if isinstance(item, dict):
                    item = dict(item)
                    item["_kind"] = kind
                    if "id" not in item:
                        try:
                            item["id"] = int(key)
                        except Exception:
                            item["id"] = 0
                    items.append(item)

    if not items:
        for item in load_legacy(lemma_path):
            item = dict(item)
            item["_kind"] = "lemma"
            items.append(item)
        for item in load_legacy(theory_path):
            item = dict(item)
            item["_kind"] = "theory"
            items.append(item)

    def sort_key(it):
        return (it.get("submitted_tick") or 0, it.get("_kind") or "", it.get("id") or 0)

    items.sort(key=sort_key)
    contents = [it.get("content", "") for it in items if it.get("content")]

    seen = set()
    imports = []
    for content in contents:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped == "import Station.Theory":
                continue
            if stripped.startswith("import "):
                if stripped not in seen:
                    seen.add(stripped)
                    imports.append(stripped)
    if "import Mathlib" not in seen:
        imports.insert(0, "import Mathlib")

    def strip_imports(code: str) -> str:
        lines = []
        for line in code.splitlines():
            if line.strip().startswith("import "):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    body_parts = []
    for content in contents:
        cleaned = strip_imports(content)
        if cleaned:
            body_parts.append(cleaned)
    body = "\n\n".join(body_parts)

    output = "\n".join(imports)
    if body:
        output = f"{output}\n\n{body}\n"
    sys.stdout.write(output)
except Exception:
    sys.exit(0)
PY

# If shared storage exists, snapshot imports into station_data for backups.
SHARED_STORAGE_PATH="$(python3 - "$ROOT_DIR" <<'PY'
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
base_path = repo_root / "station_data"
research_storage = base_path / "rooms" / "research" / "storage"
if research_storage.is_symlink():
    research_storage = Path(os.path.realpath(research_storage))
shared_dir = research_storage / "shared"
if shared_dir.exists():
    print(shared_dir)
PY
)"

SNAPSHOT_ROOT="$ROOT_DIR/station_data/rooms/theory/shared_snapshots"
if [ -n "$SHARED_STORAGE_PATH" ]; then
  mkdir -p "$SNAPSHOT_ROOT/Storage/Shared"
  python3 - "$SETUP_DIR/Station/TheoryEnv.lean" "$SHARED_STORAGE_PATH" "$SNAPSHOT_ROOT/Storage/Shared" <<'PY'
import os
import sys

env_path = sys.argv[1]
shared_root = sys.argv[2]
dest_root = sys.argv[3]

imports = set()
with open(env_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.split("--", 1)[0].strip()
        if not line.startswith("import "):
            continue
        tail = line[len("import ") :].strip()
        for token in tail.split():
            if token.startswith("Storage.Shared"):
                imports.add(token)

missing = []
for module in sorted(imports):
    if module == "Storage.Shared":
        rel_path = "Shared.lean"
    else:
        rel = module[len("Storage.Shared.") :]
        rel_path = rel.replace(".", os.sep) + ".lean"
    dest_path = os.path.join(dest_root, rel_path)
    if os.path.exists(dest_path):
        continue
    source_path = os.path.join(shared_root, rel_path)
    if not os.path.exists(source_path):
        missing.append(module)
        continue
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(source_path, "rb") as src, open(dest_path, "wb") as dst:
        dst.write(src.read())

if missing:
    sys.stderr.write("ERROR: Missing shared modules for TheoryEnv build: " + ", ".join(missing) + "\n")
    sys.exit(1)
PY
  if [ $? -ne 0 ]; then
    exit 1
  fi

  # Make Storage.* imports resolvable during `lake build` by linking the snapshot into a Lake search-path root.
  # (Lake does not reliably preserve externally-provided LEAN_PATH additions during builds.)
  LAKE_BUILD_LIB="$SETUP_DIR/.lake/build/lib/lean"
  mkdir -p "$LAKE_BUILD_LIB" 2>/dev/null || true
  if [ ! -e "$LAKE_BUILD_LIB/Storage" ] && [ ! -L "$LAKE_BUILD_LIB/Storage" ]; then
    ln -s "$SNAPSHOT_ROOT/Storage" "$LAKE_BUILD_LIB/Storage" 2>/dev/null || true
  fi
fi

echo "Updating dependencies and building..."
if [ "${REBUILD_ONLY:-false}" = "true" ] && [ "$NEED_MATHLIB" = false ]; then
  echo "REBUILD_ONLY set: skipping lake update; running lake build StationTheory"
  "$LAKE_BIN" build StationTheory
else
  "$LAKE_BIN" update
  "$LAKE_BIN" build StationTheory
fi
ACTUAL_TC="$(tr -d '[:space:]' < "$SETUP_DIR/lean-toolchain" 2>/dev/null || true)"
if [ -n "$ACTUAL_TC" ] && [ "$ACTUAL_TC" != "$PREFERRED_TOOLCHAIN" ]; then
  echo "ERROR: Project toolchain became '$ACTUAL_TC' but requested '$PREFERRED_TOOLCHAIN'."
  echo "Pin THEORY_MATHLIB_REV to a mathlib commit compatible with $PREFERRED_TOOLCHAIN or use the default toolchain."
  exit 1
fi

echo "Computing LEAN_PATH..."
RAW_LEAN_PATH_STR="$(cd "$SETUP_DIR" && "$LAKE_BIN" env printenv LEAN_PATH | tail -n 1)"
# Deduplicate components while preserving order so we don't re-append identical entries
LEAN_PATH_STR="$(python3 - "$RAW_LEAN_PATH_STR" <<'PY'
import os, sys
parts, seen = [], set()
for seg in sys.argv[1].split(os.pathsep):
    seg = seg.strip()
    if seg and seg not in seen:
        seen.add(seg)
        parts.append(seg)
print(os.pathsep.join(parts))
PY
)"

echo "Warming up Lean import of Mathlib + Station.Theory (first run may take a few seconds)..."
# Mimic TheoryRoom runtime: run lean directly with the computed LEAN_PATH (no env scrubbing).
WARMUP_FILE="$(mktemp)"
cat > "$WARMUP_FILE" <<'EOF'
import Mathlib
import Station.Theory

example : 1 + 1 = 2 := by decide
#eval (1+1)
EOF
SNAPSHOT_ROOT="$ROOT_DIR/station_data/rooms/theory/shared_snapshots"
LEAN_PATH_WITH_SNAP="$LEAN_PATH_STR"
if [ -d "$SNAPSHOT_ROOT" ]; then
  LEAN_PATH_WITH_SNAP="$SNAPSHOT_ROOT:$LEAN_PATH_STR"
fi
if LEAN_PATH="$LEAN_PATH_WITH_SNAP" "$LEAN_BIN" "$WARMUP_FILE"; then
  echo "Warm-up succeeded."
else
  echo "ERROR: Warm-up failed. Please check LEAN_PATH or rerun after fixing the above error."
  rm -f "$WARMUP_FILE"
  exit 1
fi
rm -f "$WARMUP_FILE"

# Persist LEAN_PATH/LEAN_BIN into .env for convenience (overwrite old entries to avoid mixed toolchains)
ENV_FILE="$ROOT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  sed -i '/^[[:space:]]*\(export[[:space:]]\+\)\?LEAN_PATH=/d' "$ENV_FILE"
  sed -i '/^[[:space:]]*\(export[[:space:]]\+\)\?LEAN_BIN=/d' "$ENV_FILE"
fi

{
  if [ -d "$SNAPSHOT_ROOT" ]; then
    echo "LEAN_PATH=$SNAPSHOT_ROOT:$LEAN_PATH_STR"
  else
    echo "LEAN_PATH=$LEAN_PATH_STR"
  fi
  echo "LEAN_BIN=$LEAN_BIN"
} >> "$ENV_FILE"

echo "Station.Theory setup complete. Ensure LEAN_PATH includes:"
echo "  $SETUP_DIR/.lake/packages/*/.lake/build/lib/lean"
echo "  $SETUP_DIR/.lake/build/lib/lean"
echo "  $SETUP_DIR"
echo "You can set:"
echo "  export LEAN_PATH=$LEAN_PATH_STR"
echo "  (also written to .env)"
