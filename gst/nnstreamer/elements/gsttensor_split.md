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

"Sometimes" source pads named `src_0`, `src_1`, ... are created when the first buffer arrives. Each pad carries one segment. Its caps are set when the pad is created: `dimensions` is the segment's rule in `tensorseg`, and `types` and `framerate` are those of the input. When the input is renegotiated later, the pads follow it; see [Changing what the element does](#changing-what-the-element-does).

## Properties

- tensorseg: The dimensions of each segment, separated by commas, in the order they appear in the input.

  For example, `tensorseg=1:100:100,2:100:100` cuts a `3:100:100` uint8 tensor into a `1:100:100` segment (the first 10000 bytes) and a `2:100:100` segment (the next 20000 bytes).

  Set it before the stream starts. The first buffer fixes it, and a later set to another rule is refused with a warning; see [Changing what the element does](#changing-what-the-element-does).

- tensorpick: Optional. The indices of the `tensorseg` segments to output, separated by commas. The other segments are dropped. Like `tensorseg`, it is fixed by the first buffer.

  The pad numbering does not follow the indices. Pads are numbered in the order they are created, so when `tensorseg` and `tensorpick` are set before the stream starts, the picked segments go out on `src_0`, `src_1`, ... in ascending index order. Each pad carries the dimensions of the segment it outputs, not those of the segment with the same number.

  | tensorseg | tensorpick | src\_0 | src\_1 |
  | --- | --- | --- | --- |
  | `1:4:4,2:4:4` | (unset) | segment 0, `1:4:4` | segment 1, `2:4:4` |
  | `1:4:4,2:4:4` | `1` | segment 1, `2:4:4` | - |
  | `1:4:4,2:4:4,3:4:4` | `0,2` | segment 0, `1:4:4` | segment 2, `3:4:4` |
  | `1:4:4,2:4:4,3:4:4` | `2,0` | segment 0, `1:4:4` | segment 2, `3:4:4` |

- silent: Do not produce verbose output.

## Changing what the element does

A source pad takes its caps when it is created, and its dimensions come from `tensorseg`. The two ways the pads could end up announcing something they no longer carry are handled differently.

### The rules are fixed by the first buffer

The first buffer to be split fixes `tensorseg` and `tensorpick`, whether or not any segment of it was output. Setting either of them to something else afterwards is refused: the element keeps the value it had and posts a warning on the bus, since `g_object_set()` cannot report the refusal itself. Setting one to the rule it already has changes nothing and is not reported. The pads therefore never carry a segment of another size than they announce, and no pad is ever added after `no-more-pads`.

To change them, stop the pipeline to `READY` or `NULL`, set the properties, and play it again. The source pads are removed on the way down and created anew, with the new caps and numbering, when the next stream starts, so link the new pads again (for example, from the `pad-added` signal).

### The input may be renegotiated

Upstream may change the input caps while the stream runs. The segments stay as `tensorseg` describes them, so a new input shape alone leaves the pads as they are, and a new type or framerate is passed on to every source pad before the next buffer. An input the rule no longer fits is refused per buffer with an error.

A pad keeps the media type it was created with, `other/tensors` with `num_tensors=1`. If a pad cannot take its new caps, because what is linked to it is pinned to the old ones, the element fails the negotiation instead of pushing buffers that disagree with the caps. Every peer is asked before any pad changes, so a refusal leaves all the pads announcing what they did before, and no other peer is reconfigured for a stream that goes no further.

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
