#!/usr/bin/env bash

# check the MEDIAPIPE_HOME
if [[ -z "${MEDIAPIPE_HOME}" ]]; then
  echo 'ERROR: Define MEDIAPIPE_HOME First!'
else
  # for the external objects
  for ext_obj in $(find ${MEDIAPIPE_HOME}/bazel-bin/external/ -name '*.o' ); do
    ext_obj_arr+=($ext_obj)
  done
  $(gcc ${ext_obj_arr[@]} -shared -o libmediapipe_external.so -fPIC)
  $(mv libmediapipe_external.so $1)

  # for the internal objects
  for int_obj in $(find ${MEDIAPIPE_HOME}/bazel-bin/mediapipe/ -name '*.o' ); do
    # remove object files which contain main functions
    if [[ $(nm -Ca $int_obj) != *"T main"* ]]; then
      int_obj_arr+=($int_obj)
    fi
  done
  $(gcc ${int_obj_arr[@]} -shared -o libmediapipe_internal.so -fPIC)
  $(mv libmediapipe_internal.so $1)
fi
