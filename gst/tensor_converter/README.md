# NNStreamer::tensor\_converter

## Supported features

- Video: direct conversion of video/x-raw / non-interace(progressive) to [height][width][#Colorspace] tensor. (#Colorspace:width:height:1)
  - Supported colorspaces: RGB (3), BGRx (4), Gray8 (1)
  - You may express ```frames-per-tensor``` to have multiple image frames in a tensor like audio and text as well.
  - If ```frames-per-tensor``` is not configured, the default value is 1.
  - Golden tests for such input
- Audio: direct conversion of audio/x-raw with arbitrary numbers of channels and frames per tensor to [frames-per-tensor][channels] tensor. (channels:frames-per-tensor:1:1)
  - The number of frames per tensor is supposed to be configured manually by stream pipeline developer with the property of ```frames-per-tensor```.
  - If ```frames-per-tensor``` is not configured, the default value is 1.
- Text: direct conversion of text/x-raw with UTF-8 to [frames-per-tensor][1024] tensor. (1024:frames-per-tensor:1:1)
  - The number of frames per tensor is supposed to be configured manually by stream pipeline developer with the property of ```frames-per-tensor```.
  - If ```frames-per-tensor``` is not configured, the default value is 1.
  - The size of a text frame, 1024, is assumed to be large enough for any single frame of strings. Because the dimension of tensor is the key metadata of a tensor stream pipeline, we need to fix the value before actually looking at the actual stream data.
  - TODO (Schedule TBD): Allow to accept longer text frames without having larger default text frame size.

## Planned features

From higher priority
- Support other color spaces (IUV, BGGR, ...)

## Sink Pads

One "Always" sink pad exists. The capability of sink pad is ```video/x-raw```, ```audio/x-raw```, and ```text/x-raw```.

## Source Pads

One "Always" source pad exists. The capability of source pad is ```other/tensor```. It does not support ```other/tensors``` because each frame (or a set of frames consisting a buffer) is supposed to be represented by a **single** tensor instance.

For each outgoing frame (on the source pad), there always is a **single** instance of ```other/tensor```. Not less and not more.

## Performance Characteristics

- Video
  - Unless it is RGB with ```width % 4 > 0``` or Gray8 with ```width % 4 > 0```, there is no memcpy or data modification processes. It only converts meta data in such cases.
  - Otherwise, there will be one memcpy for each frame.
- Audio
  - TBD.
- Text
  - TBD.

## Properties

- frames-per-buffer: The number of incoming media frames that will be contained in a single instance of tensors. With the value > 1, you can put multiple frames in a single tensor.

### Properties for debugging

- silent: Enable/diable debugging messages.

## Usage Examples

```
$ gst-launch videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! tensor_convert ! tensor_sink
```
