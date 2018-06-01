# NNStreamer::tensor\_converter

## Supported features

- Direct conversion of video/x-raw / non-interace(progressive) / RGB or BGRx stream to [height][width][RGB/BGRx] tensor.
- Golden test for such inputs

## Planned features

From higher priority
- Support other color spaces (IUV, BGGR, ...)
- Support audio stream
- Support text stream

# Gstreamer standard tensor media type draft

- Proposed Name: other/tensor
- Properties
  - rank: int (1: vector, 2: matrix, 3: 3-tensor, 4: 4-tensor
  - dim1: int (depth / color-RGB)
  - dim2: int (width)
  - dim3: int (height) / With version 0.0.1, this is with rstride-4.
  - dim4: int (batch. 1 for image stream)
  - type: string: int32, uint32, float32, float64, int16, uint16, int8, uint8
  - framerate; fraction

  - data: (binary data, can be treated as an C array of [dim4][dim3][dim2][dim1])
