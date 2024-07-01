#!/bin/bash
set -e
nvcc --version | grep release | awk '{print $5}' | tr ',' ' '
