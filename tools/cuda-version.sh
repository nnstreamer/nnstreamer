#!/bin/bash
set -e -o pipefail
nvcc --version | grep release | awk '{print $5}' | tr ',' ' '
