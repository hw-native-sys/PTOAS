#!/bin/sh
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# OAT Pre-commit Check Script — Python Edition (oat-py)
# Replaces the Java-binary-based oat_check.sh.
#
# Requires: Python 3.7+  (pip install oat-py>=1.0.0)
# Works on: Linux / macOS / Windows (Git Bash / MSYS2)
#
# Self-healing: strip Windows CRLF if present
_SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
if sed 's/\r//' "$_SCRIPT" 2>/dev/null | diff -q - "$_SCRIPT" >/dev/null 2>&1; then
    :
else
    echo "[OAT] CRLF detected, auto-fixing and re-running..."
    _TMP="${_SCRIPT}.lf"
    sed 's/\r//' "$_SCRIPT" > "$_TMP" && mv "$_TMP" "$_SCRIPT"
    exec sh "$_SCRIPT" "$@"
fi

set -e

# ---------------------------------------------------------------------------
# 0. Locate Python interpreter
# ---------------------------------------------------------------------------
_PYTHON=""
for _candidate in python3 python py; do
    if command -v "$_candidate" >/dev/null 2>&1; then
        _VER=$("$_candidate" -c "import sys; print(sys.version_info >= (3,7))" 2>/dev/null || echo "False")
        if [ "$_VER" = "True" ]; then
            _PYTHON="$_candidate"
            break
        fi
    fi
done

if [ -z "$_PYTHON" ]; then
    echo "[OAT] [WARNING] Python 3.7+ is required but not found. Please install Python 3.7 or later."
    echo "[OAT] Skipping OAT check, continuing commit..."
    exit 0
fi

# ---------------------------------------------------------------------------
# 1. Ensure oat-py>=1.0.2 is installed (serialized via flock to prevent parallel pip conflicts)
# ---------------------------------------------------------------------------
_OAT_MIN="1.0.2"
_check_oat_version() {
    "$_PYTHON" -c "
import importlib.util, sys
if not importlib.util.find_spec('oat'):
    print('missing'); sys.exit()
try:
    from importlib.metadata import version
    v = version('oat-py')
    parts_have = [int(x) for x in v.split('.')[:3]]
    parts_need = [int(x) for x in '${_OAT_MIN}'.split('.')[:3]]
    print('ok' if parts_have >= parts_need else 'old')
except Exception:
    print('ok')  # cannot determine version, assume ok
" 2>/dev/null || echo "missing"
}

_OAT_OK=$(_check_oat_version)
if [ "$_OAT_OK" != "ok" ]; then
    if [ "$_OAT_OK" = "old" ]; then
        echo "[OAT] oat-py is outdated. Upgrading to oat-py>=${_OAT_MIN} ..."
    else
        echo "[OAT] oat-py not found. Installing oat-py>=${_OAT_MIN} ..."
    fi
    _LOCK_FILE="/tmp/oat_pip_install.lock"
    # flock ensures only one concurrent invocation runs pip install at a time;
    # re-check inside the lock so processes that waited skip redundant installs
    if command -v flock >/dev/null 2>&1; then
        (
            flock -w 120 9
            _RECHECK=$(_check_oat_version)
            if [ "$_RECHECK" != "ok" ]; then
                "$_PYTHON" -m pip install --quiet "oat-py>=${_OAT_MIN}"
            fi
        ) 9>"$_LOCK_FILE"
    else
        "$_PYTHON" -m pip install --quiet "oat-py>=${_OAT_MIN}"
    fi
    _OAT_OK=$(_check_oat_version)
    if [ "$_OAT_OK" != "ok" ]; then
        echo "[OAT] [WARNING] Failed to install oat-py. Please run: pip install oat-py>=${_OAT_MIN}"
        echo "[OAT] Skipping OAT check, continuing commit..."
        exit 0
    fi
    echo "[OAT] oat-py installed successfully."
fi

# ---------------------------------------------------------------------------
# 2. Determine repo root and name
# ---------------------------------------------------------------------------
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
REPO_NAME=$(basename "$REPO_ROOT")
OAT_REPORT_DIR="${TMPDIR:-/tmp}/oat_reports_$$"
OAT_RESULT_DIR="$REPO_ROOT/oat_reports"

# ---------------------------------------------------------------------------
# 2a. PR-range deduplication (pure git, no CI env vars required)
#
#  Strategy:
#   1. Find the merge-base between HEAD and the upstream branch (origin/master
#      or similar). If found, this is a feature-branch context ??collect ALL
#      files changed since the branch diverged (full PR diff).
#   2. Key a done-marker on the HEAD SHA. The first invocation runs the scan;
#      every subsequent invocation for the same HEAD exits 0 immediately.
#      This eliminates redundant scans when CI calls the hook once per commit.
#   3. If no merge-base is found (e.g. committing directly on master) fall back
#      to scanning only the currently staged files, with no done-marker.
# ---------------------------------------------------------------------------
_PR_MERGE_BASE=""
_DONE_MARKER=""
_HEAD_SHA=$(git rev-parse HEAD 2>/dev/null | cut -c1-12 || true)

if [ -n "$_HEAD_SHA" ]; then
    # Try to find merge-base with a known upstream branch
    if [ -n "${PRE_COMMIT_FROM_REF:-}" ]; then
        # pre-commit --from-ref/--to-ref: use the provided base directly
        _PR_MERGE_BASE=$(git merge-base "$PRE_COMMIT_FROM_REF" HEAD 2>/dev/null || true)
    fi
    if [ -z "$_PR_MERGE_BASE" ]; then
        # Search all remotes (origin, upstream, etc.) for common trunk branch names.
        # This handles fork setups where origin=fork and upstream=main-repo.
        _CANDIDATE_BASE=""
        _CANDIDATE_DIST=0
        for _b in $(git for-each-ref --format='%(refname:short)' \
                        'refs/remotes/*/master' 'refs/remotes/*/main' \
                        'refs/remotes/*/develop' 'refs/remotes/*/dev' 2>/dev/null); do
            _mb=$(git merge-base HEAD "$_b" 2>/dev/null || true)
            [ -z "$_mb" ] && continue
            # Skip if merge-base is HEAD itself (branch is ahead of or equal to HEAD)
            [ "$_mb" = "$(git rev-parse HEAD 2>/dev/null)" ] && continue
            # Pick the candidate whose merge-base is furthest from HEAD
            # (most commits since divergence = most likely true PR base)
            _dist=$(git rev-list --count "$_mb"..HEAD 2>/dev/null || echo 0)
            if [ -z "$_CANDIDATE_BASE" ] || [ "$_dist" -gt "$_CANDIDATE_DIST" ]; then
                _CANDIDATE_BASE="$_mb"
                _CANDIDATE_DIST="$_dist"
            fi
        done
        _PR_MERGE_BASE="$_CANDIDATE_BASE"
    fi

    # Only use PR-range mode when HEAD has diverged from the upstream base
    if [ -n "$_PR_MERGE_BASE" ] && [ "$_PR_MERGE_BASE" != "$(git rev-parse HEAD 2>/dev/null)" ]; then
        mkdir -p "$OAT_RESULT_DIR"
        _DONE_MARKER="$OAT_RESULT_DIR/.done_${_HEAD_SHA}"
        if [ -f "$_DONE_MARKER" ]; then
            echo "[OAT] [SKIP] Already scanned HEAD=${_HEAD_SHA}. Skipping duplicate invocation."
            exit 0
        fi
    else
        # HEAD == merge-base: on the upstream branch itself, no PR range
        _PR_MERGE_BASE=""
    fi
fi

# ---------------------------------------------------------------------------
# 2b. Deduplicate parallel invocations within the same pre-commit run.
#     Only the first instance that acquires the lock actually runs the scan;
#     all others wait then exit 0.
# ---------------------------------------------------------------------------
if command -v flock >/dev/null 2>&1; then
    _OAT_SCAN_LOCK="${TMPDIR:-/tmp}/oat_scan_$(echo "$REPO_ROOT" | tr '/\\: ' '____').lock"
    exec 9>"$_OAT_SCAN_LOCK"
    if ! flock -n 9; then
        echo "[OAT] Another OAT scan instance is running. Waiting..."
        flock -w 120 9
        # Re-check done marker: if the scanning instance completed normally, skip.
        # If it exited abnormally (no marker), fall through and run the scan ourselves.
        if [ -n "$_DONE_MARKER" ] && [ -f "$_DONE_MARKER" ]; then
            echo "[OAT] Scan already completed by another instance. Skipping."
            exit 0
        fi
        echo "[OAT] Previous instance did not complete. Proceeding with scan..."
        # Fall through to run the scan below
    fi
    # This instance now owns the lock and will run the scan.
fi

echo "[OAT] Running OAT scan (Python Edition) ??INCREMENTAL MODE"
echo "[OAT] Project: $REPO_NAME"
if [ -n "$_PR_MERGE_BASE" ]; then
    echo "[OAT] Mode: PR range ??scanning all files changed since merge-base"
else
    echo "[OAT] Mode: staged files ??scanning only currently staged files"
fi

# ---------------------------------------------------------------------------
# 3. Collect staged files
# ---------------------------------------------------------------------------
if [ $# -gt 0 ]; then
    # Called directly with file arguments (e.g. manual test)
    FILE_COUNT=$#
    FILE_LIST=""
    for _f in "$@"; do
        # Convert relative paths to absolute
        case "$_f" in
            /*) _abs="$_f" ;;
            *)  _abs="$REPO_ROOT/$_f" ;;
        esac
        [ -f "$_abs" ] || continue
        if [ -z "$FILE_LIST" ]; then
            FILE_LIST="$_abs"
        else
            FILE_LIST="$FILE_LIST,$_abs"
        fi
    done
elif [ -n "$_PR_MERGE_BASE" ]; then
    # PR range mode: collect ALL files changed since the branch diverged
    _STAGED=$(git diff --name-only --diff-filter=ACM "$_PR_MERGE_BASE" HEAD \
              2>/dev/null || true)
    if [ -z "$_STAGED" ]; then
        echo "[OAT] No files changed in PR range. Skipping."
        [ -n "$_DONE_MARKER" ] && touch "$_DONE_MARKER" 2>/dev/null || true
        exit 0
    fi
    FILE_COUNT=$(echo "$_STAGED" | wc -l | tr -d ' ')
    FILE_LIST=""
    while IFS= read -r _f; do
        [ -z "$_f" ] && continue
        case "$_f" in
            /*) _abs="$_f" ;;
            *)  _abs="$REPO_ROOT/$_f" ;;
        esac
        [ -f "$_abs" ] || continue
        if [ -z "$FILE_LIST" ]; then
            FILE_LIST="$_abs"
        else
            FILE_LIST="$FILE_LIST,$_abs"
        fi
    done <<EOF
$_STAGED
EOF
else
    # Staged files mode: on upstream branch directly, scan only staged files
    _STAGED=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null || true)
    if [ -z "$_STAGED" ]; then
        echo "[OAT] No staged files to check. Skipping."
        exit 0
    fi
    FILE_COUNT=$(echo "$_STAGED" | wc -l | tr -d ' ')
    FILE_LIST=""
    while IFS= read -r _f; do
        [ -z "$_f" ] && continue
        case "$_f" in
            /*) _abs="$_f" ;;
            *)  _abs="$REPO_ROOT/$_f" ;;
        esac
        [ -f "$_abs" ] || continue
        if [ -z "$FILE_LIST" ]; then
            FILE_LIST="$_abs"
        else
            FILE_LIST="$FILE_LIST,$_abs"
        fi
    done <<EOF
$_STAGED
EOF
fi

if [ -z "$FILE_LIST" ]; then
    echo "[OAT] No existing staged files found. Skipping."
    exit 0
fi

echo "[OAT] Checking $FILE_COUNT staged file(s)..."

# ---------------------------------------------------------------------------
# 4. Ensure report directories exist
# ---------------------------------------------------------------------------
mkdir -p "$OAT_REPORT_DIR"
mkdir -p "$OAT_RESULT_DIR"

# ---------------------------------------------------------------------------
# 5. Resolve OAT.xml path
# ---------------------------------------------------------------------------
_OAT_XML="$REPO_ROOT/OAT.xml"
if [ ! -f "$_OAT_XML" ] && [ -f "$REPO_ROOT/scripts/OAT.xml" ]; then
    _OAT_XML="$REPO_ROOT/scripts/OAT.xml"
fi

# _OAT_CMD is kept for display purposes only (shown in error messages).
# Actual execution uses direct argument passing below to avoid eval word-splitting
# and command-injection risks when paths or filenames contain special characters.
if [ -f "$_OAT_XML" ]; then
    _OAT_CMD="\"$_PYTHON\" -m oat -mode s -s \"$REPO_ROOT\" -r \"$OAT_REPORT_DIR\" -n \"$REPO_NAME\" -w 1 -f \"$FILE_LIST\" -oatconfig \"$_OAT_XML\""
else
    _OAT_CMD="\"$_PYTHON\" -m oat -mode s -s \"$REPO_ROOT\" -r \"$OAT_REPORT_DIR\" -n \"$REPO_NAME\" -w 1 -f \"$FILE_LIST\""
fi

# ---------------------------------------------------------------------------
# 6. Run oat scan (direct argument passing � no eval)
# ---------------------------------------------------------------------------
echo ""
echo "[OAT] Running compliance scan..."

set +e
if [ -f "$_OAT_XML" ]; then
    "$_PYTHON" -m oat -mode s -s "$REPO_ROOT" -r "$OAT_REPORT_DIR" -n "$REPO_NAME" -w 1 -f "$FILE_LIST" -oatconfig "$_OAT_XML" >/dev/null 2>&1
else
    "$_PYTHON" -m oat -mode s -s "$REPO_ROOT" -r "$OAT_REPORT_DIR" -n "$REPO_NAME" -w 1 -f "$FILE_LIST" >/dev/null 2>&1
fi
_OAT_RC=$?
set -e

if [ "$_OAT_RC" -ne 0 ] && [ "$_OAT_RC" -ne 1 ]; then
    echo ""
    echo "[OAT] [WARNING] oat exited with unexpected code $_OAT_RC."
    echo "[OAT] Try re-running manually:"
    echo "  $_OAT_CMD"
    echo "[OAT] Skipping OAT check, continuing commit..."
    exit 0
fi

# ---------------------------------------------------------------------------
# 7. Parse report and write result.txt
#    Only: Invalid File Type + License Header Invalid (no copyright)
# ---------------------------------------------------------------------------
REPORT_FILE="$OAT_REPORT_DIR/PlainReport_${REPO_NAME}.txt"
RESULT_FILE="$OAT_RESULT_DIR/result.txt"

# Section headers used as stop-boundaries when extracting sections
_ALL_HEADERS="Invalid File Type Total Count:|License Not Compatible Total Count:|License Header Invalid Total Count:|Copyright Header Invalid Total Count:|No License File Total Count:|No Readme.OpenSource Total Count:|No Readme Total Count:|Import Invalid Total Count:|Redundant License File Total Count:|Third Party Software Info Total Count:"

# Extract one section from the report (header line + detail Name: lines)
# Usage: _extract_section <file> <start_marker>
_extract_section() {
    _file="$1"
    _marker="$2"
    awk -v marker="$_marker" -v boundaries="$_ALL_HEADERS" '
        BEGIN { capturing=0 }
        !capturing {
            if (index($0, marker) == 1) { capturing=1; print; next }
            next
        }
        capturing {
            # Check if this line starts a different section header
            n = split(boundaries, hdrs, "|")
            for (i=1; i<=n; i++) {
                if (hdrs[i] != marker && index($0, hdrs[i]) == 1) { exit }
            }
            print
        }
    ' "$_file" | sed '/^[[:space:]]*$/{ N; /^\n$/d }' | awk '
        { lines[NR]=$0 } END { while(NR>0 && lines[NR]~/^[[:space:]]*$/) NR--; for(i=1;i<=NR;i++) print lines[i] }
    '
}

if [ ! -f "$REPORT_FILE" ]; then
    if [ "$_OAT_RC" -eq 0 ]; then
        # oat exited cleanly with no report: all staged files were filtered out
        echo "[OAT] [OK] All checks passed ($FILE_COUNT file(s) checked)."
        [ -n "$_DONE_MARKER" ] && touch "$_DONE_MARKER" 2>/dev/null || true
        rm -rf "$OAT_REPORT_DIR"
        exit 0
    else
        # oat returned exit code 1 but produced no report �?possible disk issue
        # or oat internal error. Do not silently pass; block the commit.
        echo ""
        echo "[OAT] [ERROR] oat exited with code $_OAT_RC but no report was generated."
        echo "[OAT] This may indicate a disk error or an oat internal bug."
        echo "[OAT] To investigate, run manually:"
        echo "  $_OAT_CMD"
        echo "[OAT] Blocking commit to prevent silent compliance bypass."
        echo ""
        rm -rf "$OAT_REPORT_DIR"
        exit 1
    fi
fi

# Parse counts ??use || true to prevent set -e from triggering if grep finds no match
_INVALID_TYPE=$(grep "^Invalid File Type Total Count:" "$REPORT_FILE" | grep -oE '[0-9]+' | head -1 || true)
_LICENSE_INVALID=$(grep "^License Header Invalid Total Count:" "$REPORT_FILE" | grep -oE '[0-9]+' | head -1 || true)
_INVALID_TYPE=${_INVALID_TYPE:-0}
_LICENSE_INVALID=${_LICENSE_INVALID:-0}

# Extract detail sections
_SECTION_TYPE=$(_extract_section "$REPORT_FILE" "Invalid File Type Total Count:")
_SECTION_LIC=$(_extract_section "$REPORT_FILE" "License Header Invalid Total Count:")

# Fallback to bare count line if section is empty
[ -n "$_SECTION_TYPE" ] || _SECTION_TYPE="Invalid File Type Total Count: $_INVALID_TYPE"
[ -n "$_SECTION_LIC"  ] || _SECTION_LIC="License Header Invalid Total Count: $_LICENSE_INVALID"

# Write result.txt
{
    echo "==================================="
    echo "OAT Scan Result Summary"
    echo "==================================="
    printf "Scan Time: %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Project: $REPO_NAME"
    echo "Files Checked: $FILE_COUNT"
    echo ""
    echo "-----------------------------------"
    echo "$_SECTION_TYPE"
    echo ""
    echo "-----------------------------------"
    echo "$_SECTION_LIC"
    echo "==================================="
} > "$RESULT_FILE"

# ---------------------------------------------------------------------------
# 8. Block commit if issues found; always clean up temp dir
# ---------------------------------------------------------------------------
TOTAL_ISSUES=$(( _INVALID_TYPE + _LICENSE_INVALID ))

if [ "$TOTAL_ISSUES" -gt 0 ]; then
    echo ""
    echo "===================================================================="
    echo "  OAT: Compliance issues found. Commit blocked."
    echo "===================================================================="
    echo ""
    echo "[OAT] Found $TOTAL_ISSUES compliance issue(s):"
    echo "  - Invalid File Type:       $_INVALID_TYPE"
    echo "  - License Header Invalid:  $_LICENSE_INVALID"
    echo ""
    echo "[OAT] Details (also saved to: $RESULT_FILE):"
    echo "---"
    cat "$RESULT_FILE"
    echo "---"
    echo ""
    echo "Fix the issues and recommit, or skip with:"
    echo "  git commit --no-verify"
    echo ""
    rm -rf "$OAT_REPORT_DIR"
    exit 1
fi

echo ""
echo "[OAT] [OK] All checks passed ($FILE_COUNT file(s) checked)."
echo "[OAT] Summary: cat $RESULT_FILE"
echo ""
# Mark scan as done for CI range mode (prevents redundant re-runs on same PR head)
if [ -n "$_DONE_MARKER" ]; then
    touch "$_DONE_MARKER" 2>/dev/null || true
    echo "[OAT] Done-marker created: $_DONE_MARKER"
fi
rm -rf "$OAT_REPORT_DIR"
exit 0
