#!/bin/bash
set -e
python3 -c "import nvidia.dali as dali; print(dali.sysconfig.get_include_dir()); print(dali.sysconfig.get_lib_dir()); print(\" \".join(dali.sysconfig.get_compile_flags()))"
