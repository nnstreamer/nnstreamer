# NNStreamer::tensor\_aggregator

## Supported features

GstTensorAggregator is a plugin to aggregate the tensor using GstAdapter.

This plugin handles the buffer with the unit **frame**.
Each incoming or outgoing buffer is supposed a single tensor, which may contain one or multi frames.

GstTensorAggregator gets the size of one frame with ```frames-in```, aggregates the frames, and pushes a buffer with ```frames-out``` frames.
After pushing an outgoing buffer, GstTensorAggregator flushes the ```frames-flush``` frames.

For example, GstTensorAggregator with the properties ```frames-in=3```, ```frames-out=4```, ```frames-flush=2```

```
Incoming buffer
--------------------------------------------------------------------
|  1st buffer  |  2nd buffer  |  3rd buffer  |  4th buffer  |
--------------------------------------------------------------------
| 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 
--------------------------------------------------------------------
Outgoing buffer
--------------------------------------------------------------------
|  1st out-buffer   |
--------------------------------------------------------------------
 flushed  |  2nd out-buffer   |
--------------------------------------------------------------------
            flushed |  3rd out-buffer   |
--------------------------------------------------------------------
                      flushed |  4th out-buffer   |
--------------------------------------------------------------------
                                flushed |  5th out-buffer   |
--------------------------------------------------------------------
```

Please be informed that, to ensure the tensor configuration, you have to change the dimension if input and output frames are different. (See the property ```frames-dim```.)

## Sink Pads

One "Always" sink pad exists. The capability of sink pad is ```other/tensor```.

## Source Pads

One "Always" source pad exists. The capability of source pad is ```other/tensor```.
It does not support ```other/tensors``` because each frame (or a set of frames consisting a buffer) is supposed to be represented by a **single** tensor instance.

## Properties

- frames-in: The number of frames in incoming buffer. (Default 1)

  GstTensorAggregator itself cannot get the number of frames in buffer.
  This plugin calculates the size of one frame with this property.

- frames-out: The number of frames in outgoing buffer. (Default 1)

  GstTensorAggregator calculates the size of outgoing frames and pushes a buffer to source pad.

- frames-flush: The number of frames to flush. (Default 0)

  GstTensorAggregator flushes the bytes (```frames-flush``` frames) in GstAdapter after pushing a buffer.
  If set 0 (default value), all outgoing frames will be flushed.

- frames-dim: The dimension index of frames in tensor. (Default value is (NNS_TENSOR_RANK_LIMIT - 1))

  If frames-in and frames-out are different, GstTensorAggregator has to change the dimension of tensor.
  With this property, GstTensorAggregator changes the out-caps.
  
  If set this value in 0 ~ (NNS_TENSOR_RANK_LIMIT - 2), GstTensorAggregator will concatenate the output buffer.

### Properties for debugging

- silent: Enable/disable debugging messages.

## Usage Examples

```
$ gst-launch videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! tensor_converter ! tensor_aggregator frames-out=10 frames-flush=5 frames-dim=3 ! tensor_sink
```

GstTensorAggregator receives a buffer with 1 frame (dimension 3:640:480:1), pushes a buffer with 10 frames (dimension 3:640:480:10), and flushes 5 frames after pushing a buffer.
