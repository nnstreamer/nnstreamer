# NNStreamer::convert2tensor

TODO Version 0.0.1
Direct conversion of bitmap stream to [RGB][height][width] tensor.
The main objective of this version is to draft standard tensor format for Gstreamer.

TODO Version 0.0.2
Direct conversion of general video stream to [RGB][height][width] tensor.

TODO Version 0.0.3
Support basic dimension reform (order of dimsisions, color space changes)

TODO Version 0.0.4
Support dimension reshape (width/height)

TODO Version 0.0.5
Support color space conversions

TODO Version 0.1.0
Direct conversion of general audio stream to FORMAT-TO-BE-DETERMINED


# Gstreamer standard tensor media type draft

- Proposed Name: other/tensor
- Properties
  - rank: int (0: scalar, 1: vector, 2: matrix, 3: 3-tensor, ...)
  - dimension: int[] (1 .. rank)
  - type: int32., float32, int8, uint8, ... (C types only?)
  - data: (binary data, can be treated as an C array of [dimension[0]][dimension[1]]...)

