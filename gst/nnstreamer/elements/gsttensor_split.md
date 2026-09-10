---
title: tensor_split
...

# NNStreamer::tensor\_split

## Supported features

tensor\_split cuts a single tensor into segments and sends each segment out as a separate single-tensor stream. It is the opposite of `tensor_merge`.

The segments are consecutive byte ranges of the incoming tensor: the first segment starts at the beginning of the buffer and each following segment starts where the previous one ends. Every segment keeps the element type of the input tensor.

## Sink Pads

One "Always" sink pad exists. It accepts a single tensor (`other/tensor`, or `other/tensors` with `num_tensors=1`).

## Source Pads

"Sometimes" source pads named `src_0`, `src_1`, ... are created when the first buffer arrives. Each pad carries one segment, and its caps describe that segment: `dimensions` is the segment's rule in `tensorseg`, and `types` and `framerate` are those of the input.

## Properties

- tensorseg: The dimensions of each segment, separated by commas, in the order they appear in the input.

  For example, `tensorseg=1:100:100,2:100:100` cuts a `3:100:100` uint8 tensor into a `1:100:100` segment (the first 10000 bytes) and a `2:100:100` segment (the next 20000 bytes).

- tensorpick: Optional. The indices of the `tensorseg` segments to output, separated by commas. The other segments are dropped.

  The pad numbering does not follow the indices. The picked segments go out on `src_0`, `src_1`, ... in ascending index order, and each pad carries the dimensions of the segment it outputs, not those of the segment with the same number.

  | tensorseg | tensorpick | src\_0 | src\_1 |
  | --- | --- | --- | --- |
  | `1:4:4,2:4:4` | (unset) | segment 0, `1:4:4` | segment 1, `2:4:4` |
  | `1:4:4,2:4:4` | `1` | segment 1, `2:4:4` | - |
  | `1:4:4,2:4:4,3:4:4` | `0,2` | segment 0, `1:4:4` | segment 2, `3:4:4` |
  | `1:4:4,2:4:4,3:4:4` | `2,0` | segment 0, `1:4:4` | segment 2, `3:4:4` |

- silent: Do not produce verbose output.

## Usage Examples

Split an RGB image into its first third and the remaining two thirds:

```
$ gst-launch-1.0 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! \
    video/x-raw,format=RGB,width=100,height=100,framerate=0/1 ! tensor_converter ! \
    tensor_split name=split tensorseg=1:100:100,2:100:100 \
    split.src_0 ! queue ! filesink location=src0.log \
    split.src_1 ! queue ! filesink location=src1.log
```

Output only the second segment. It comes out on `src_0`, and its caps say `dimensions=2:100:100`:

```
$ gst-launch-1.0 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! \
    video/x-raw,format=RGB,width=100,height=100,framerate=0/1 ! tensor_converter ! \
    tensor_split name=split tensorseg=1:100:100,2:100:100 tensorpick=1 \
    split.src_0 ! queue ! filesink location=src0.log
```
