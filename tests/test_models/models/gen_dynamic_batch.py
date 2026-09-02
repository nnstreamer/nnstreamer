##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2026 Samsung Electronics
#
# @file    gen_dynamic_batch.py
# @brief   Generate dynamic_batch_add_one.tflite, a model with a dynamic batch
# @author  MyungJoo Ham <myungjoo.ham@samsung.com>
#
# tests/test_models held no model with an undefined dimension, so nothing could
# exercise a runtime reshape. This builds the smallest thing that can: one
# float32 input of shape [-1, 4] and an output of the same shape holding
# input + 1.
#
# The batch dimension is left undefined on purpose. LiteRT's strict resize
# (LiteRtCompiledModelResizeInputTensor) refuses a tensor whose signature
# declares no dynamic dimension, so a model with a fixed shape cannot stand in
# for this one.
#
# The arithmetic is elementwise and exact in float32, which lets a test assert
# the output bit-for-bit at any batch size instead of comparing against a
# tolerance.
#
# Usage: python3 gen_dynamic_batch.py   (in this directory)
# Requires: tensorflow

"""Generate dynamic_batch_add_one.tflite (input [-1, 4], output = input + 1)."""

import tensorflow as tf

OUT = "dynamic_batch_add_one.tflite"


class AddOne(tf.Module):
    """A module adding one to a [-1, 4] float32 input."""

    @tf.function(input_signature=[tf.TensorSpec([None, 4], tf.float32, name="x")])
    def __call__(self, x):
        """Return x + 1, preserving the caller's batch size."""
        return tf.add(x, 1.0, name="y")


def main():
    """Convert the module and write the flatbuffer."""
    model = AddOne()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [model.__call__.get_concrete_function()], model
    )
    tflite = converter.convert()

    with open(OUT, "wb") as f:
        f.write(tflite)

    interp = tf.lite.Interpreter(model_content=tflite)
    shape = interp.get_input_details()[0]["shape_signature"]
    print(f"wrote {OUT} ({len(tflite)} bytes), input shape signature {shape}")
    if shape[0] != -1:
        raise SystemExit("the batch dimension is not dynamic; the model is unusable here")


if __name__ == "__main__":
    main()
