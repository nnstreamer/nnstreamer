#!/bin/bash
nvcc --version | grep release | awk '{print $5}' | tr ',' ' '
