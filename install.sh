#!/usr/bin/env bash
# Install linker-lab into ./.venv from uv.lock, then check the installation.
#
# Needs: Linux x86_64 or aarch64 with glibc >= 2.34 (Ubuntu 22.04 or newer), git (uv fetches LW-v0's
# robot files from linker-sim-assets with it), an NVIDIA GPU with driver >= 580 (CUDA 13),
# internet access and 10 GB of free disk. Everything else (uv, Python 3.12, Isaac Lab, the
# physics engine) is installed by this script. Run it from anywhere: ./install.sh
set -euo pipefail

UV_VERSION="0.11.7"   # installed only if uv is missing
MIN_UV="0.11.0"
MIN_DRIVER=580
MIN_GLIBC="2.34"
MIN_DISK_GB=10

fail() { echo "install.sh: $*" >&2; exit 1; }
older() { [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" = "$1" ]; }

cd "$(dirname "$0")"

[ "$(uname -s)" = Linux ] && case "$(uname -m)" in x86_64|aarch64) true ;; *) false ;; esac \
    || fail "linker-lab supports Linux x86_64 and aarch64 only."

glibc="$(getconf GNU_LIBC_VERSION 2>/dev/null | cut -d' ' -f2)"
[ -n "$glibc" ] || fail "glibc not found; linker-lab needs a glibc-based Linux (glibc >= ${MIN_GLIBC})."
older "$glibc" "$MIN_GLIBC" && fail "glibc ${glibc} is too old; need >= ${MIN_GLIBC} (Ubuntu 22.04 or newer): some of Isaac Lab's packages do not load on older systems."

free_gb="$(( $(df -Pk . | awk 'NR == 2 {print $4}') / 1024 / 1024 ))"
[ "$free_gb" -ge "$MIN_DISK_GB" ] || fail "${free_gb} GB free here; need ${MIN_DISK_GB} GB (the environment is about 8 GB)."

command -v git >/dev/null || fail "git not found. Install it first (e.g. 'sudo apt install git'); uv fetches LW-v0's robot files with it."

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found. Install the NVIDIA driver (>= ${MIN_DRIVER}) first."
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
[ "${driver%%.*}" -ge "$MIN_DRIVER" ] || fail "NVIDIA driver ${driver} is too old; need >= ${MIN_DRIVER}."
echo "NVIDIA driver ${driver}"

if ! command -v uv >/dev/null; then
    echo "Installing uv ${UV_VERSION} (https://docs.astral.sh/uv/) into ~/.local/bin"
    url="https://astral.sh/uv/${UV_VERSION}/install.sh"
    if command -v curl >/dev/null; then curl -LsSf "$url" | sh
    elif command -v wget >/dev/null; then wget -qO- "$url" | sh
    else fail "need curl or wget to download uv."
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi
uv_version="$(uv --version | cut -d' ' -f2)"
older "$uv_version" "$MIN_UV" && fail "uv ${uv_version} is too old; run 'uv self update' (need >= ${MIN_UV})."

echo "Creating .venv from uv.lock (first run downloads about 4 GB)"
uv sync --locked

uv run linker-lab doctor

echo
echo "linker-lab is installed in $(pwd)/.venv."
echo "Run commands with 'uv run linker-lab <command>', or 'source .venv/bin/activate' first."
