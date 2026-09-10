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

"Sometimes" source pads named `src_0`, `src_1`, ... are created when the first buffer arrives. Each pad carries one segment. Its caps are set when the pad is created: `dimensions` is the segment's rule in `tensorseg`, and `types` and `framerate` are those of the input at that moment. See [Limitations](#limitations) for what happens when these change later.

## Properties

- tensorseg: The dimensions of each segment, separated by commas, in the order they appear in the input.

  For example, `tensorseg=1:100:100,2:100:100` cuts a `3:100:100` uint8 tensor into a `1:100:100` segment (the first 10000 bytes) and a `2:100:100` segment (the next 20000 bytes).

- tensorpick: Optional. The indices of the `tensorseg` segments to output, separated by commas. The other segments are dropped.

  The pad numbering does not follow the indices. Pads are numbered in the order they are created, so when `tensorseg` and `tensorpick` are set before the stream starts, the picked segments go out on `src_0`, `src_1`, ... in ascending index order. Each pad carries the dimensions of the segment it outputs, not those of the segment with the same number.

  | tensorseg | tensorpick | src\_0 | src\_1 |
  | --- | --- | --- | --- |
  | `1:4:4,2:4:4` | (unset) | segment 0, `1:4:4` | segment 1, `2:4:4` |
  | `1:4:4,2:4:4` | `1` | segment 1, `2:4:4` | - |
  | `1:4:4,2:4:4,3:4:4` | `0,2` | segment 0, `1:4:4` | segment 2, `3:4:4` |
  | `1:4:4,2:4:4,3:4:4` | `2,0` | segment 0, `1:4:4` | segment 2, `3:4:4` |

- silent: Do not produce verbose output.

## Limitations

A source pad takes its caps once, when it is created, and keeps them while the element is running. Nothing that changes afterwards updates the pads that already exist, and in the cases below no error or warning is given:

- Setting `tensorseg` again while the stream runs changes the bytes each existing pad carries, but not its caps. For example, after `tensorseg=1:4:4,2:4:4` becomes `2:4:4,1:4:4`, `src_0` still says `dimensions=1:4:4` (16 bytes) while it pushes 32 bytes.
- A new input type or framerate from upstream does not reach the existing pads' caps, although the new type is used to cut the segments. For example, if the input changes from `uint8` to `float32`, the pads keep saying `types=uint8` while they push four times as many bytes. A new input shape alone leaves the caps right, because a pad's dimensions come from `tensorseg`; an input too small for the segments is refused with an error.
- Setting `tensorpick` again while the stream runs creates pads for the newly picked segments after the existing ones, so pad numbers no longer follow segment order. For example, after `tensorpick=1` becomes `0,1`, segment 1 stays on `src_0` and segment 0 goes out on `src_1`.

Elements downstream read the buffers by those caps, so the first two cases give them tensors of the wrong shape, type or framerate.

To change any of these, stop the pipeline to `READY` or `NULL`, set `tensorseg` and `tensorpick`, and play it again. The source pads are removed on the way down and created anew, with the new caps and numbering, when the next stream starts, so link the new pads again (for example, from the `pad-added` signal).

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
