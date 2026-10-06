#!/bin/bash

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt_retry() {
	local attempts=0
	until sudo apt-get "$@"; do
		attempts=$((attempts + 1))
		if (( attempts >= 24 )); then
			echo "apt-get failed after ${attempts} attempts: $*" >&2
			return 1
		fi
		echo "apt-get is unavailable; retrying in 5 seconds (${attempts}/24)..." >&2
		sleep 5
	done
}

apt_retry update
apt_retry install -y \
	build-essential \
	cmake \
	make \
	libopenblas-dev \
	libyaml-dev \
	ffmpeg \
	wget \
	ca-certificates \
	git-lfs

# Optional: reduce image size a bit
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

git lfs install

uv sync
