# Multi-mode Dockerfile for reproducing the project environment.
# Build ARGs:
#   BASE=python:3.10-slim
#   MODE=portable  (or lockfile)

ARG BASE=python:3.10-slim
ARG MODE=portable
FROM ${BASE} AS base

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ffmpeg wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN python -m pip install --upgrade pip wheel setuptools

RUN if [ "${MODE}" = "portable" ]; then \
      pip install -r requirements-repro.txt; \
    else \
      pip install -r requirements-lock.txt; \
    fi

RUN pip install -e .

CMD ["bash"]

# Usage examples:
# Portable image:
# docker build --build-arg MODE=portable -t cotracker:portable .
# Exact environment snapshot:
# docker build --build-arg MODE=lockfile -t cotracker:lockfile .
