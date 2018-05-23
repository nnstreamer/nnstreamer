# NNStreamer::convert2tensor

TODO Version 0.0.1
Direct conversion of video/x-raw 1view / non-interace(progressive) stream to [height][width-stride-4][RGB] tensor.
The main objective of this version is to draft standard tensor format for Gstreamer.

TODO Version 0.0.2
Support basic dimension reform (order of dimsisions, color space changes)

TODO Version 0.0.3
Support dimension reshape (width/height)

TODO Version 0.0.4
Support color space conversions

TODO Version 0.1.0
Direct conversion of general audio stream to FORMAT-TO-BE-DETERMINED


# Gstreamer standard tensor media type draft

- Proposed Name: other/tensor
- Properties
  - rank: int (0: scalar, 1: vector, 2: matrix, 3: 3-tensor, ...)
  - dim1: int (depth / color-RGB)
  - dim2: int (width)
  - dim3: int (height) / With version 0.0.1, this is with rstride-4.
  - dim4: int (batch. 1 for image stream)
  - type: string: int32, uint32, float32, float64, int16, uint16, int8, uint8
  - framerate; fraction (TODO: to redefine the range)

  - data: (binary data, can be treated as an C array of [dim4][dim3][dim2][dim1])
