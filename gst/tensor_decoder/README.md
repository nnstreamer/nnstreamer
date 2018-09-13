# NNStreamer::tensor\_decoder

## Supported features

- Direct conversion of other/tensor with video/x-raw semantics back to video/x-raw stream.

## Planned features

- With given properties from users or pipeline developers, support the following conversions. The list is not exclusive and we may need to implement more features later.


| Main property (input tensor semantics) | Additional & mandatory property | Output |
| -------------------------------------- | ------------------- | ------ |
| Bounding boxes (other/tensor)          | N/A                 | video/x-raw |
| Image label (other/tensor)             | File path to labels | text/x-raw |
| Bounding boxes + labels (other/tensors)| File path to labels | video/x-raw (textoverlay processed?) |
| ... more features coming ... | | |


## Sink Pads

One always sink pad.

- other/tensor (current)
- Either other/tensor or other/tensors (future, planned)

## Source Pads

One always source pad.

- video/x-raw (current)
- Either video/x-raw or text/x-raw (future, planned)

TBD: we may need to support multiple source pads (bounding boxes + labels?)

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
