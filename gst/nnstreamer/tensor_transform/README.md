---
title: tensor_transform
...

# NNStreamer::tensor_transform

## Supported Features

- Transformation the shape, data values (arithmetics or normalization), or data type of ```other/tensor``` stream.
- If possible, the tensor_transform element exploits [ORC: Optimized inner Loop Runtime Compiler](https://gitlab.freedesktop.org/gstreamer/orc) to accelerate the supported operations.
- Aggregate multiple operators into a single transform instance for performance optimization.
  - E.g., ```tensor_transform mode=typecast option=uint8 ! tensor_transform mode=arithmetic option=mul:4 ! tensor_transform mode=arithmetic option=add:25 can be optimized by tensor_transform mode=arithmetic option=typecast:uint8,mul:8,add:25```

## Planned Features

- TBD

## Pads

- SINK

   - One always sink pad named 'sink'
   - other/tensor

- SRC

   - One always source pad named 'src'
   - other/tensor

## Properties

- mode (readable, writable): Mode used for transforming tensor
  - Enum "gtt_mode_type" Default: -1, "unknown"
    - (0): dimchg
      - A mode for changing tensor dimensions
      - An option should be provided as option=FROM_DIM:TO_DIM (with a regex, ^([0-3]):([0-3])$, where NNS_TENSOR_RANK_LIMIT is 4).
      - Example: Move 1st dim to 2nd dim (i.e., [a][H][W][C] ==> [a][C][H][W])

        ```bash
        ... ! tensor_converter ! tensor_transform mode=dimchg option=0:2 ! ...
        ```

    - (1): typecast
      - A mode for casting data type of tensor
      - An option should be provided as option=TARGET_TYPE (with a regex, ^[u]?int(8|16|32|64)$|^float(32|64)$)
      - Example: Cast the data type of upstream tensor to uint8

        ```bash
        ... ! tensor_converter ! tensor_transform mode=typecast option=uint8 ! ...
        ```

    - (2): arithmetic
      - A mode for arithmetic operations with tensor
      - An option should be provided as option=[typecast:TYPE,]add|mul|div:NUMBER..., ...
      - Example 1: Element-wise add 25 and multiply 4

        ```bash
        ... ! tensor_converter ! tensor_transform mode=arithmetic option=add:25,mul:4 ! ...
        ```

      - Example 2: Cast the data type of upstream tensor to float32 and element-wise subtract 25

        ```bash
        ... ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-25 ! ...
        ```

    - (3): transpose
      - A mode for transposing shape of tensor
      - An option should be provided as D1':D2':D3':D4 (fixed to 3)
      - Example: 640:480:3:1 ==> 3:480:640:1

        ```bash
        ... ! tensor_converter input-dim=640:480:3:1 ! tensor_transform mode=transpose option=2:1:0:3 ! ...
        ```

    - (4): stand
      - A Mode for statistical standardization of tensor, option=default

- acceleration (readable, writable): A flat indicating whether to enable ```orc``` acceleration

## Properties for debugging

- silent: disable or enable debugging messages
