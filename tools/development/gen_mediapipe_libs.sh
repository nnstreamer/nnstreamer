#!/usr/bin/env bash

CC=$(which gcc)
NM=$(which nm)

# check the MEDIAPIPE_HOME
if [[ -z "${MEDIAPIPE_HOME}" ]]; then
  echo "Define MEDIAPIPE_HOME First!"
  exit 1
fi

if [ -z "$1" ] || [ ! -d "$1" ]; then
  echo "[$1] is not existed!"
  exit 1
fi

# `for the external objects
for ext_obj in $(find ${MEDIAPIPE_HOME}/bazel-bin/external/ -name '*.o' ); do
  ext_obj_arr+=($ext_obj)
done
$($CC ${ext_obj_arr[@]} -shared -o libmediapipe_external.so)
if [ $? -ne 0 ]; then
  exit 1
fi
$(mv libmediapipe_external.so $1)

# for the internal objects
for int_obj in $(find ${MEDIAPIPE_HOME}/bazel-bin/mediapipe/ -name '*.o' ); do
  # remove object files which contain main functions
  if [[ $($NM -Ca $int_obj) != *"T main"* ]]; then
    int_obj_arr+=($int_obj)
  fi
done
out=$($CC ${int_obj_arr[@]} -shared -o libmediapipe_internal.so)
if [ $? -ne 0 ]; then
  exit 1
fi
$(mv libmediapipe_internal.so $1)

exit 0
