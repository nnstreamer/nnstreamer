---
title: tensor_converter
...

# NNStreamer::tensor\_converter

## Supported features

- Video: direct conversion of video/x-raw / non-interlace(progressive) to [height][width][#Colorspace] tensor. (#Colorspace:width:height:frames-per-tensor)
  - Supported colorspaces: RGB (3), BGRx (4), Gray8 (1)
  - You may express ```frames-per-tensor``` to have multiple image frames in a tensor like audio and text as well.
  - If ```frames-per-tensor``` is not configured, the default value is 1.
  - Golden tests for such input
- Audio: direct conversion of audio/x-raw with arbitrary numbers of channels and frames per tensor to [frames-per-tensor][channels] tensor. (channels:frames-per-tensor)
  - The number of frames per tensor is supposed to be configured manually by stream pipeline developer with the property of ```frames-per-tensor```.
  - If ```frames-per-tensor``` is not configured, the default value is 1.
- Text: direct conversion of text/x-raw with UTF-8 to [frames-per-tensor][input-dim] tensor. (input-dim:frames-per-tensor)
  - The number of frames per tensor is supposed to be configured manually by stream pipeline developer with the property of ```frames-per-tensor```.
  - If ```frames-per-tensor``` is not configured, the default value is 1.
  - The size of a text frame should be configured by developer with the property ```input-dim```. Because the dimension of tensor is the key metadata of a tensor stream pipeline, we need to fix the value before actually looking at the actual stream data.
- Octet stream: direct conversion of application/octet-stream.
  - Octet stream to static tensor: You should set ```input-type``` and ```input-dim``` to describe tensor(s) information of outgoing buffer.
    If setting multiple tensors, converter will divide incoming buffer and set multiple memory chunks in outgoing buffer.

    e.g, converting 10 bytes of octet stream to 2 static tensors:

    ```... ! application/octet-stream ! tensor_converter input-dim=2:1:1:1,2:1:1:1 input-type=int32,int8 ! ...```
  - Octet stream to flexible tensor: With a caps filter (```other/tensors,format=flexible```), tensor-converter generates flexible tensor.
    In this case, you don't need to denote ```input-type``` and ```input-dim``` in pipeline description.
    Converter sets the dimension with buffer size (size:1:1:1) and type uint8, and appends this information (tensor-meta) in the memory of outgoing buffer.

    e.g, converting octet stream to flexible tensor:

    ```... ! application/octet-stream ! tensor_converter ! other/tensors,format=flexible ! ...```
  - Media to octet stream: If you need to convert media to octet stream, use [capssetter](https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gst-plugins-good/html/gst-plugins-good-plugins-capssetter.html).
    This element can update the caps of incoming buffer using it's properties. After updating the mimetye of media stream, it can be converted to tensor stream with tensor_converter.

    e.g., converting jpeg to flexible tensor:

    ```... ! jpegenc ! capssetter caps="application/octet-stream" replace=true join=false ! tensor_converter ! other/tensors,format=flexible ! ...```
  - Only single frame. ```frames-per-tensor``` should be 1 (default value).
- Flexible tensor: conversion to static tensor stream.
  - You can convert mime type (flexible to static) if incoming tensor has fixed data format and size.
  - With ```input-type``` and ```input-dim```, converter will set the output capability on src pad.
- Serialized data: conversion to static tensor stream.
  - Supported serialization format: Protocol Buffers, Flatbuffers and Flexbuffers.
  - The converter gets input capability from the peer pad of the sink pad, or you can specify the capability.
  - You don't need to specify the option because the sub-plugin is registered using the capability.

## Planned features

From higher priority
- Support other color spaces (IUV, BGGR, ...)

## Sink Pads

One "Always" sink pad exists. The capability of sink pad is ```video/x-raw```, ```audio/x-raw```, ```text/x-raw```, ```application/octet-stream```, and ```other/tensors-flexible```.

If you require another pad caps to convert media stream to tensor(s), you can implement new sub-plugin or register custom converter.

## Source Pads

One "Always" source pad exists. The capability of source pad is ```other/tensor```, ```other/tensors```, and ```other/tensors-flexible```.

Note that, only octet-stream in the default capabilities of sink pad supports configuring multiple tensors in outgoing buffer.
When incoming media type is video, audio, or text, each frame (or a set of frames consisting a buffer) is supposed to be represented by a **single** tensor instance and it will have ```other/tensor``` capability.

## Performance Characteristics

- Video
  - Unless it is RGB with ```width % 4 > 0``` or Gray8 with ```width % 4 > 0```, there are no memcpy or data modification processes. It only converts meta data in such cases.
  - Otherwise, there will be one memcpy for each frame.
- Audio
  - TBD.
- Text
  - TBD.

## Properties

- frames-per-tensor: The number of incoming media frames that will be contained in a single instance of tensors. With the value > 1, you can put multiple frames in a single tensor.

### Properties for debugging

- silent: Enable/disable debugging messages.

## Usage Examples

```
$ gst-launch videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! tensor_converter ! tensor_sink
```

### flatbuffers to tensors stream
Convert to flatbuffers using tensor decoder and then convert back to tensors stream.
```
$ gst-launch videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! tensor_converter ! tensor_decoder mode=flatbuf ! tensor_converter ! tensor_sink
```

### protocol buffers to tensors stream
Convert to protocol buffers using tensor decoder and then convert back to tensors stream.
```
$ gst-launch videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! tensor_converter ! tensor_decoder mode=protobuf ! tensor_converter ! tensor_sink
```

### flexbuffers to tensors stream
Convert to flexbuffers using tensor decoder and then convert back to tensors stream.
```
$ gst-launch videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! tensor_converter ! tensor_decoder mode=flexbuf ! tensor_converter ! tensor_sink
```

## Custom converter
If you want to convert any media type to tensors, you can use custom mode of the tensor converter.

### Code mode
This is an example of a callback type custom mode.
```
// Define custom callback function
GstBuffer * tensor_converter_custom_cb (GstBuffer *in_buf, void *data, GstTensorsConfig *config) {
  // Write a code to convert any media type to tensors.
}
...
// Register custom callback function
nnstreamer_converter_custom_register ("tconv", tensor_converter_custom_cb, NULL);
...
// Use the custom tensor converter in a pipeline.
// E.g., Pipeline of " ... (any media stream) ! tensor_converter mode=custom-code:tconv ! (tensors)... "
...
// After everything is done.
nnstreamer_converter_custom_unregister ("tconv");
```

### Script mode
* Note: Currently only Python is supported.
  - If you want to use FlatBuffers Python in Tizen, install package `flatbuffers-python`. It also includes a Flexbuffers Python.
  - If you want to use Flatbuffers Python in Ubuntu, install package using pip `pip install flatbuffers`. It also includes a Flexbuffers Python.

This is an example of a python script.
```
# @file custom_converter_example.py
import numpy as np
import nnstreamer_python as nns
## @brief  User-defined custom converter
class CustomConverter(object):
  def convert (self, input_array):
    ## Write a code to convert any media type to tensors.
    return (tensors_info, out_array, rate_n, rate_d)
```
Example pipeline
```
... (any media stream) ! tensor_converter mode=custom-script:custom_converter_example.py ! (tensors) ...
```
