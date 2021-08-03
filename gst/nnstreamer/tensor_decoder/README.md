---
title: tensor_decoder
...

# NNStreamer::tensor\_decoder

## Supported features

With given properties from users or pipeline developers, support the following conversions. The list is not exclusive and we may need to implement more features later.

| Mode | Main property (input tensor semantics) | Additional & mandatory property | Output |
| -| - | - | - |
| directvideo | other/tensors | N/A | video/x-raw |
| bounding_boxes | Bounding boxes (other/tensor) | File path to labels, decoding schems, out dim, in dim | video/x-raw |
| image_labeling | Image label (other/tensor) | File path to labels | text/x-raw |
| image_segment | segmentaion info | expected model | video/x-raw |
| pose_estimation | pose info | out dim, in dim,  File path to labels, mode | video/x-raw |
| flatbuf | other/tensors | N/A | flatbuffers |
| protobuf | other/tensors | N/A | protocol buffers |
| flexbuf | other/tensors | N/A | flexbuffers |
| | ... more features coming ... | | |


## Sink Pads

One always sink pad.

- other/tensors (current)

## Source Pads

One always source pad.

- video/x-raw
- text/x-raw
- flatbuffers
- protocol buffers
- flexbuffers

## Performance Characteristics

TBD.

## Properties

- output-type: Integer denoting VIDEO, AUDIO, or TEXT.
  - Experimental: we need to update this. Current "output-type" is not satisfactory.

The following properties are suggested and planned.
- (update) output-type: String denoting "bounding-boxes", "image-label", "bounding-boxes-with-label", ...
- additional-file-1: String denoting the file path to the **first** data file for decoding; e.g., label list text file for image labeling.
- additional-file-2: **second** data file if the corresponding output-type requires two or more.
- additional-file-N: ... **N'th** data file if the corresponding output-type requires N or more.


## Properties for debugging

- silent: disable or enable debugging messages

## Usage Examples

```
$ gst-launch somevideosrc_with_xraw ! tee name=t ! queue ! tensor_converter ! tensor_filter SOME_OPTION ! tensor_decoder output-type=image-label additional-file-1=/tmp/labels.txt ! txt. t. ! queue ! textoverlay name=txt ! autovideosink
```
Note: not tested. not sure if the syntax is correct with ```txt. t. !```. Regard the above as pseudo code.

### tensor stream to flatbuffers
```
$ gst-launch videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! tensor_converter ! tensor_decoder mode=flatbuf ! fakesink
```

### tensor stream to protocol buffers
```
$ gst-launch videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! tensor_converter ! tensor_decoder mode=protobuf ! fakesink
```

### tensor stream to flexbuffers
```
$ gst-launch videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! tensor_converter ! tensor_decoder mode=flexbuf ! fakesink
```

## Custom decoder
If you want to convert tensors to any media type, You can use custom mode of the tensor decoder.
### code mode
This is an example of a callback type custom mode.
```
// Define custom callback function
int tensor_decoder_custom_cb (const GstTensorMemory *input,
const GstTensorsConfig *config, void *data, GstBuffer *out_buf) {
  // Write a code to convert tensors to any media type.
}

...
// Register custom callback function
nnstreamer_decoder_custom_register ("tdec", tensor_decoder_custom_cb, NULL);
...
// Use the custom tensor decoder in a pipeline.
// E.g., Pipeline of " ... (tensors) ! tensor_decoder mode=custom-code option1=tdec ! (any media stream)... "
...
// After everything is done.
nnstreamer_decoder_custom_unregister ("tdec");
```

### script mode
* Note: Currently only Python is supported.
  - If you want to use FlatBuffers Python in Tizen, install package `flatbuffers-python`. It also includes a Flexbuffers Python.
  - If you want to use Flatbuffers Python in Ubuntu, install package using pip `pip install flatbuffers`. It also includes a Flexbuffers Python.

This is an example of a python script.
```
# @file custom_decoder_example.py
## @brief  User-defined custom decoder
class CustomDecoder(object):
## @breif  Python callback: getOutCaps
  def getOutCaps (self):
    # Write capability of the media type.
    return bytes('@CAPS_STRING@', 'UTF-8')

## @breif  Python callback: decode
  def decode (self, raw_data, in_info, rate_n, rate_d):
    # return decoded raw data as `bytes` type.
    return data

```
Example pipeline
```
... (tensors) ! tensor_decoder mode=python3 option1=custom_decoder_example.py ! (any media stream) ...
```
