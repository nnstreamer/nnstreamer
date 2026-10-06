---
title: tensor_transform
...

# NNStreamer::tensor_transform

## Supported Features

- Transformation the shape, data values (arithmetics or normalization), or data type of ```other/tensor``` stream.
- If possible, the tensor_transform element exploits [ORC: Optimized inner Loop Runtime Compiler](https://gitlab.freedesktop.org/gstreamer/orc) to accelerate the supported operations.
- Aggregate multiple operators into a single transform instance for performance optimization.
  - E.g., ```tensor_transform mode=typecast option=uint8 ! tensor_transform mode=arithmetic option=mul:4 ! tensor_transform mode=arithmetic option=add:25 can be optimized by tensor_transform mode=arithmetic option=typecast:uint8,mul:4,add:25```

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
      - An option should be provided as option=FROM_DIM:TO_DIM (with a regex, ^([0-9]|1[0-5]):([0-9]|1[0-5])$, where NNS_TENSOR_RANK_LIMIT is 16).
      - FROM_DIM should be less than TO_DIM, and the incoming tensor should have a TO_DIM-th dimension.
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
      - An option should be provided as option=[typecast:TYPE,][per-channel:(false|true@DIM),]add|mul|div:NUMBER[@CH_IDX]..., ...
      - Example 1: Element-wise add 25 and multiply 4

        ```bash
        ... ! tensor_converter ! tensor_transform mode=arithmetic option=add:25,mul:4 ! ...
        ```

      - Example 2: Cast the data type of upstream tensor to float32 and element-wise subtract 25

        ```bash
        ... ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-25 ! ...
        ```

      - For "per-channel", DIM means the dimension which should be viewed as channel and CH_IDX means the idx of channel the given operation should be applied to. When CH_IDX is not given, the operation is applied to all channels.
      - DIM should be less than NNS_TENSOR_RANK_LIMIT (16), and the incoming tensor should have that dimension. CH_IDX should be one of the channels of that dimension. An option outside those ranges is refused, and so is a stream the option does not fit.
      - Example 3: Add 255 only for 1-th channel when 0-th dim is channel (for RGB image, add 255 for G channel)

        ```bash
        ... ! video/x-raw,format=RGB ! tensor_converter ! tensor_transform mode=arithmetic option=per-channel:true@0,add:255@1 ! ...
        ```

    - (3): transpose
      - A mode for transposing shape of tensor
      - An option should be provided as D1':D2':D3':D4 (fixed to 3)
      - In a tensor of rank 5 or higher, the dimensions after D4 keep their place as D4 does
      - Example: 3:640:480:1 (NHWC) ==> 640:480:3:1 (NCHW)

        ```bash
        ... ! tensor_converter input-dim=3:640:480:1 ! tensor_transform mode=transpose option=1:2:0:3 ! ...
        ```

    - (4): stand
      - A mode for statistical standardization or normalization of tensor
      - An option should be provided as option=(default|dc-average)[:TYPE] where `default` for statistical standardization and `dc-average` to remove DC offset (average value). `TYPE` denotes output data type.
      - Example: Remove DC offset, output type to float32

        ```bash
        ... ! tensor_converter ! tensor_transform mode=stand option=dc-average:float32 ! ...
        ```

    - (6): padding
      - A mode for zero-padding the first three dimensions of tensor
      - An option should be provided as left|right|top|bottom|front|back:NUMBER[,layout:(NCHW|NHWC)], where left/right pad the 1st dimension, top/bottom the 2nd and front/back the 3rd (left/right and front/back swap with `layout:NHWC`).
      - A tensor of rank 1 or 2 is padded as if its missing dimensions were 1: `4:3` with `front:1` becomes `4:3:2`, and `4:3` with `left:1` stays rank 2 as `5:3`.
      - `NUMBER` is a decimal number of any length. The option is refused if a pad or the sum of the two pads of a dimension exceeds 4294967295, and the stream is refused if a padded dimension does. The output is allocated at the padded size, so a large pad needs that much memory.
      - Example: 100:50:3 ==> 102:54:3

        ```bash
        ... ! tensor_converter input-dim=100:50:3 ! tensor_transform mode=padding option=left:1,right:1,top:2,bottom:2 ! ...
        ```

- acceleration (readable, writable): A flat indicating whether to enable ```orc``` acceleration

- apply (readable, writable): The indices of the tensors to transform, separated with ',' (e.g., `apply=0,2`). The other tensors pass through unchanged, and an empty value transforms every tensor. Each index is written in decimal digits only, without a sign; a value with any other token is rejected and the previous value is kept.

## Changing properties while streaming

- `mode`, `option` and `apply` can be changed while the pipeline is in PLAYING state. Each change takes effect from the next buffer.
- If a change alters the type or dimension of the output tensors, the element renegotiates its output caps before the next buffer. If the downstream elements cannot accept the new caps, the stream stops with a not-negotiated error.
- Setting a property while a buffer is being transformed waits until that buffer is done; that buffer uses the previous values.
- A change that lands after the element has checked for a pending renegotiation, but before it transforms the buffer, drops that one buffer instead of sending it with caps that do not describe it.
- A change that lands while a previous change is being renegotiated makes that negotiation fail. The element then negotiates again with the latest values, up to 8 attempts per buffer. Each failed attempt posts a "not negotiated" warning on the bus. If the properties keep changing through all the attempts, the stream stops with a not-negotiated error.
- If downstream also accepts flexible tensors, a change that lands during such a negotiation may send one buffer as a flexible tensor. Its header describes the data, and the following buffer is negotiated with the latest values.
- `mode` and `option` are separate properties. When changing both, a buffer arriving between the two changes sees the new mode with the old option; if that option does not fit the new mode, the stream stops with an error.

## Properties for debugging

- silent: disable or enable debugging messages
