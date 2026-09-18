#!/usr/bin/env bash
# Build the C++ acquisition executable; forward arguments to CMake configuration.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${project_dir}/build"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: build.sh [CMake configuration options]

Builds build/measure_lin. Requires CMake, a C++17 compiler and MVS SDK headers.
Example: ./build.sh -DMVS_INCLUDE_DIR=/opt/MVS/include
Set CMAKE_BUILD_PARALLEL_LEVEL to control build parallelism.
HELP
    exit 0
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "Error: cmake is required for building real_data_auto." >&2
    exit 1
fi

cmake -S "${project_dir}" -B "${build_dir}" "$@"
cmake --build "${build_dir}" --target measure_lin --parallel
printf 'Built: %s/measure_lin\n' "${build_dir}"
