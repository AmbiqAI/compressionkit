#!/bin/bash

export DEBIAN_FRONTEND=noninteractive

sudo apt update
sudo apt install -y \
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
sudo apt clean
sudo rm -rf /var/lib/apt/lists/*

git lfs install

uv sync
