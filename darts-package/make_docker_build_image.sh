#!/bin/bash
CUDA_VERSION=11.2.2

docker build --rm -f "Dockerfile_build" -t cuda$CUDA_VERSION-devel-py --build-arg VERSION=$CUDA_VERSION "."
