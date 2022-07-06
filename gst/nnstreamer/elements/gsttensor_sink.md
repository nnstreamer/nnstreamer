---
title: tensor_sink
...

# NNStreamer::tensor\_sink

## Supported features

GstTensorSink is a sink plugin for making an application to get a buffer of tensor (or tensors).

GstTensorSink emits a signal when receiving a buffer from up-stream element.
An application can connect a signal ```new-data```, then will get the buffer of tensor.

## Sink Pads

One "Always" sink pad exists. The capability of sink pad is ```other/tensor``` and ```other/tensors```.

## Signals

- new-data: Signal to get the buffer from GstTensorSink.

- stream-start: Optional. An application can use this signal to detect the start of a new stream, instead of the message ```GST_MESSAGE_STREAM_START``` from pipeline.

- eos: Optional. An application can use this signal to detect the EOS (end-of-stream), instead of the message ```GST_MESSAGE_EOS``` from pipeline.

## Properties

- signal-rate: New data signals per second (Default 0 for unlimited, MAX 500)

  If ```signal-rate``` is larger than 0, GstTensorSink calculates the time to emit a signal with this property.

  If set 0 (default value), all the received buffers will be passed to the application.

  Please note that this property does not guarantee the periodic signals.
  This means if GstTensorSink cannot get the buffers in time, it will pass all the buffers. (working like default 0)

- emit-signal: Flag to emit the signals for new data, stream start, and eos. (Default true)

### Properties for debugging

- silent: Enable/disable debugging messages.

## Usage Examples

```
$ gst-launch-1.0 videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! tensor_converter ! tensor_sink
```

For more details, see the [examples](https://github.com/nnstreamer/nnstreamer/tree/master/nnstreamer_example/example_sink) to handle the buffer from GstTensorSink.
