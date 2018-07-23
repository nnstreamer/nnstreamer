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

# other/tensors File Data Format, Version 1.0 Draft

```
Header-0: The first 20 bytes, global header
=============================================================================================================
|    8 bytes               |   4 bytes         |    4 bytes                  | 4 bytes                      |
| The type header          | Protocol Version  | Number of tensors per frame | Frame size in bytes          |
|           "TENSORST"     |    (uint32) 1     |  (uint32) 1~16 (N)          | (uint32) 1 ~ MAXUINT (S)     |
|                          |                   | (v.1 supports up to 16)     | Counting data only (no meta) |
=============================================================================================================
|    20 bytes. RESERVED in v1.                                                                              |
=============================================================================================================
| Header-1: Following Header-1, Description of Tensor-1 of each frame (24 bytes)                   |
| 4 bytes                           | 4 bytes              | 4 bytes | 4 bytes | 4 bytes | 4 bytes |
| Element Type (enum)               | RANK (uint32)        | Dim-1   | Dime-2  | Dim-3   | Dim-4   |
| "tensor_type" in tensor_typedef.h | v.1 supports 1 to 4. |         |         |         |         |
====================================================================================================
| ...                                                                                              |
====================================================================================================
| Header-N                                                                                         |
====================================================================================================
Data of frame-1, tensor-1 starts at the offset of (40 + 24 x N).
Data of frame-1, tensor-i starts at the offset of (40 + 24 x N + Sum(x=1..i-1)(tensor_element_size[tensor-type of Tx] x dim1-of-Tx x dim2-of-Tx x dim3-of-Tx x dim4-of-Tx)).
...
Data of frame-F, tensor-1 starts at the offset of (40 + 24 x N + S x (F - 1))
...
Assert (S = Sum(x=1..N)(tensor_element_size[tensor-type of Tx] x dim1-of-Tx x dim2-of-Tx x dim3-of-Tx x dim4-of-Tx))

Add a custom footer
```

Note that once the stream is loaded in GStreamer, tensor\_\* elements uses the data parts only without the headers.
The header exists only when the tensor stream is stored as a file.
