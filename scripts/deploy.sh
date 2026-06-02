#!/usr/bin/env bash
# =============================================================================
# deploy.sh  —  Set up one GPU lab server and launch training
#
# Open 3 terminal windows and run one command in each:
#
#   bash scripts/deploy.sh 1   →  gpulab06  M1: 3D CNN Baseline
#   bash scripts/deploy.sh 2   →  gpulab07  M2: ResNet-18 Hierarchical
#   bash scripts/deploy.sh 3   →  gpulab08  M3: Swin Transformer
#
# Each run:
#   1. Pushes branch 'final' to GitHub  (idempotent — safe to run from all 3)
#   2. SSHes into the server            (one password prompt)
#   3. Clones or updates the repo       (skips clone if already present)
#   4. Installs dependencies            (skips PyTorch if already installed)
#   5. Syncs data/                      (rsync: only transfers new/changed files)
#   6. Launches training in tmux        (skips if session already running)
# =============================================================================
set -euo pipefail

# ── Validate argument ─────────────────────────────────────────────────────────
usage() { echo "Usage: bash scripts/deploy.sh [1|2|3]"; exit 1; }
[[ $# -eq 1 ]] || usage
[[ "$1" =~ ^[123]$ ]] || usage

IDX=$(( $1 - 1 ))

# ── Config ────────────────────────────────────────────────────────────────────
REMOTE_USER="ketsia"
REMOTE_BASE="/data/ketsia"          # lab servers require work under /data/ketsia
RDIR="$REMOTE_BASE/rna-motif-classification"
BRANCH="final"
REPO_URL="https://github.com/ketsiambaku/rna-motif-classification.git"
LOCAL_DATA="$(pwd)/data"

SERVERS=("gpulab06.uwb.edu" "gpulab07.uwb.edu" "gpulab08.uwb.edu")
MODELS=("cnn_baseline"      "resnet3d"           "swin3d")
LABELS=("M1: 3D CNN Baseline" "M2: ResNet-18 Hierarchical" "M3: Swin Transformer")
SESSIONS=("train_m1"        "train_m2"           "train_m3")

SERVER="${SERVERS[$IDX]}"
MODEL="${MODELS[$IDX]}"
LABEL="${LABELS[$IDX]}"
SESSION="${SESSIONS[$IDX]}"
HOST="$REMOTE_USER@$SERVER"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
info() { echo -e "  ${BLUE}→${NC} $*"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $*"; }
die()  { echo -e "  \033[0;31m✗ ERROR: $*${NC}" >&2; exit 1; }

# ── Sanity checks ─────────────────────────────────────────────────────────────
for cmd in git ssh rsync; do
    command -v "$cmd" &>/dev/null || die "$cmd is required"
done
[[ -f "data/manifest.csv" ]] || die "Run from the project root (data/manifest.csv not found)"

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}  Deploy $1/3  →  $SERVER${NC}"
echo -e "${BOLD}${BLUE}  $LABEL${NC}"
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════${NC}\n"

# ── Step 1: Git push ──────────────────────────────────────────────────────────
info "Pushing branch '$BRANCH' to GitHub..."

CURRENT=$(git rev-parse --abbrev-ref HEAD)
[[ "$CURRENT" == "$BRANCH" ]] || warn "You are on '$CURRENT' not '$BRANCH'"

# Always stage everything first — 'git diff' misses untracked files
git add .
if ! git diff --cached --quiet; then
    git commit -m "deploy: $(date '+%Y-%m-%d %H:%M')"
fi
git push origin "$BRANCH"
ok "Branch '$BRANCH' pushed"

# ── Step 2: SSH into server (one password prompt — socket reused after) ───────
CTRL="/tmp/ssh_ctrl_${SERVER}"
# Remove stale socket from a previous run so ControlMaster starts clean
ssh -O exit -o "ControlPath=$CTRL" "$HOST" 2>/dev/null || true
info "Connecting to $HOST — enter your password once:"
ssh \
    -o ControlMaster=yes \
    -o "ControlPath=$CTRL" \
    -o ControlPersist=600 \
    -o StrictHostKeyChecking=accept-new \
    "$HOST" echo "  Connected to $SERVER" || die "SSH failed"
ok "SSH connection established (reused for all remaining steps)"

# All subsequent ssh/rsync calls reuse the socket — no more password prompts
_ssh() { ssh -o ControlMaster=no -o "ControlPath=$CTRL" "$HOST" "$@"; }

# ── Step 3: Clone or update repo ─────────────────────────────────────────────
info "Syncing repo (branch: $BRANCH)..."
_ssh "bash -s" <<REMOTE
set -e
RDIR="$RDIR"
if [ -d "\$RDIR/.git" ]; then
    echo "  Repo exists — pulling latest..."
    cd "\$RDIR"
    git fetch --quiet origin
    git checkout --quiet $BRANCH 2>/dev/null || git checkout --quiet -b $BRANCH origin/$BRANCH
    git pull --quiet origin $BRANCH
else
    echo "  Fresh clone into $RDIR..."
    mkdir -p "$REMOTE_BASE"
    git clone --quiet --branch $BRANCH "$REPO_URL" "\$RDIR"
fi
echo "  Branch: \$(cd \$RDIR && git rev-parse --abbrev-ref HEAD)  commit: \$(cd \$RDIR && git rev-parse --short HEAD)"
REMOTE
ok "Repo ready on branch $BRANCH"

# ── Step 4: Install dependencies ──────────────────────────────────────────────
info "Checking dependencies..."
_ssh "bash -s" <<'REMOTE'
set -e
PIP="pip3"; command -v pip3 &>/dev/null || PIP="pip"

# PyTorch — skip if already installed
if python3 -c "import torch; print('  PyTorch', torch.__version__, 'already installed')" 2>/dev/null; then
    : # already good
else
    # Detect CUDA and install matching wheel
    CUDA_VER=""
    command -v nvcc        &>/dev/null && CUDA_VER=$(nvcc --version      | grep -oP 'release \K[0-9]+\.[0-9]+')
    [[ -z "$CUDA_VER" ]] && \
    command -v nvidia-smi  &>/dev/null && CUDA_VER=$(nvidia-smi          | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')

    if [[ -z "$CUDA_VER" ]]; then
        echo "  No CUDA found — installing CPU-only PyTorch"
        $PIP install --user -q "torch>=2.0"
    else
        MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        if   [[ $MAJOR -eq 11 ]];                        then WHEEL="cu118"
        elif [[ $MAJOR -eq 12 && $MINOR -le 1 ]];        then WHEEL="cu121"
        else                                                   WHEEL="cu124"; fi
        echo "  CUDA $CUDA_VER → installing PyTorch ($WHEEL)"
        $PIP install --user -q torch --index-url "https://download.pytorch.org/whl/$WHEEL"
    fi
fi

# Remaining packages (pip skips anything already up to date)
cd /data/ketsia/rna-motif-classification
$PIP install --user -q -r requirements.txt
echo "  All packages ready"
REMOTE
ok "Dependencies ready"

# ── Step 5: Sync data/ ────────────────────────────────────────────────────────
info "Syncing data/ (~12 GB first run, fast on re-runs)..."
rsync -az --progress \
    -e "ssh -o ControlMaster=no -o ControlPath=$CTRL" \
    "$LOCAL_DATA/" \
    "$HOST:$RDIR/data/"
ok "Data synced"

# ── Step 6: Run training in the foreground ───────────────────────────────────
# Output streams directly to your terminal. Ctrl+C to stop.
# Log is also saved to checkpoints/$MODEL/train.log on the server.
info "Starting training — output appears below (Ctrl+C to stop):"
echo "────────────────────────────────────────────"
_ssh "bash -s" <<REMOTE
set -e
cd $RDIR
mkdir -p checkpoints/$MODEL
PYTHON=\$(command -v python3 || command -v python)
\$PYTHON scripts/run_training.py --model $MODEL 2>&1 | tee checkpoints/$MODEL/train.log
REMOTE
echo "────────────────────────────────────────────"
echo -e "${BOLD}${GREEN}  ✓ Training finished on $SERVER${NC}"
echo -e "  Retrieve checkpoint:"
echo -e "    scp $HOST:$RDIR/checkpoints/$MODEL/best.pt ./checkpoints/"
