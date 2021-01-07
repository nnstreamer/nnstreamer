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

## Planned features

From higher priority
- Support other color spaces (IUV, BGGR, ...)

## Sink Pads

One "Always" sink pad exists. The capability of sink pad is ```video/x-raw```, ```audio/x-raw```, ```text/x-raw``` and ```application/octet-stream```.

## Source Pads

One "Always" source pad exists. The capability of source pad is ```other/tensor```. It does not support ```other/tensors``` because each frame (or a set of frames consisting a buffer) is supposed to be represented by a **single** tensor instance.

For each outgoing frame (on the source pad), there always is a **single** instance of ```other/tensor```. Not less and not more.

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
