---
title: tensor_if
...

# NNStreamer::tensor\_if

## Supported features

The if (tensor_if) element allows creating conditional branches based on tensor values.  
For example, you may skip frames if there is no object detected with high confidence.  
The input and output stream data type is either `other/tensor` or `other/tensors`.  

## Properties
- compared-value: Specifies the compared value and is represented as operand 1 from input tensors.
  * A_VALUE: Decided based on a single scalar value.
  * TENSOR_AVERAGE_VALUE: Decided based on an average value of a specific tensor.
  * CUSTOM: Decided based on a user-defined callback.

- compared-value-option: Specifies an element of the nth tensor or you can pick one from the tensors.
  * [C][W][H][B],n: used for A_VALUE of the compared-value, for example 0:1:2:3,0 means [0][1][2][3] value of first tensor.
  * nth tensor: used for TENSOR_AVERAGE_VALUE of the compared-value, and specifies which tensor is used.

- supplied-value: Specifies the supplied value (SV) from the user.
  * SV
  * SV1, SV2 (used for RANGE operators)

- operator: Comparison Operator
  * EQ: Check if CV == SV
  * NEQ: Check if CV != SV
  * GT: Check if CV > SV
  * GE: Check if CV >= SV
  * LT: Check if CV < SV
  * LE: Check if CV <= SV
  * RANGE_INCLUSIVE: Check if SV1 <= CV and CV <= SV2
  * RANGE_EXCLUSIVE: Check if SV1 < CV and CV < SV2
  * NOT_IN_RANGE_INCLUSIVE: Check if CV < SV1 or SV2 < CV
  * NOT_IN_RANGE_EXCLUSIVE: Check if CV <= SV1 or SV2 <= CV

- then: Action if it is TRUE
  * PASSTHROUGH: Does not let you make changes to the buffers. Buffers are pushed straight through.
  * SKIP: Does not let you generate the output frame (frame skip).
  * TENSORPICK: Lets you choose the nth tensor among the input tensors.
```
   [ tensor 0 ]
   [ tensor 1 ]  ->   tensor if    ->    [ tensor 0 ]
   [ tensor 2 ]    (tensorpick 0,2)      [ tensor 2 ]
  input tensors                         output tensors
```

- then-option: Option for TRUE Action
  * nth tensor: used for TENSORPICK option, for example, `then-option`=0,2 means tensor 0 and tensor 2 are selected as output tensors among the input tensors.

- else: Action if it is FALSE
  * PASSTHROUGH: Does not let you make changes to the buffers. Buffers are pushed straight through.
  * SKIP: Does not let you generate the output frame (frame skip).
  * TENSORPICK: Lets you choose the nth tensor among the input tensors.

- else-option: Option for FALSE Action
  * nth tensor: used for TENSORPICK option, for example, `else-option`=0,2 means tensor 0 and tensor 2 are selected as output tensors among the input tensors.

## Usage Examples

 The format of statement with tensor-if is:
 If (Compared_Value OPERATOR Supplied_Value(s))) then THEN else ELSE
   - Compared_Value and Supplied_Value are the operands.
   - Compared_Value is a value from input tensor(s).
   - Supplied_Value is a value from tensor-if properties.

If the given if-condition is simple enough (e.g., if a specific element is between a given range in a tensor frame), it can be expressed as:
 #### Example launch line with simple if condition

```
gst-launch ... (some tensor stream) !
      tensor_if name=tif \
                compared-value=A_VALUE compared-value_option=3:4:2:5,0 \
                operator=RANGE_INCLUSIVE \
                supplied-value=10,100 \
                then=PASSTHROUGH \
                else=TENSORPICK \
                else-option=0 \
    ! tif.src_0 ! (tensor(s) stream for TRUE action) ...
    ! tif.src_1 ! (tensor(s) stream for FALSE action) ...
```


However, if the if-condition is complex and cannot be expressed with tensor-if expressions, you may create a corresponding custom filter with tensor-filter, whose output is other/tensors with an additional tensor that is "1:1:1:1, uint8", which is 1 (true) or 0 (false) as the first tensor of other/tensors and the input tensor/tensors.

Then, you can create a pipeline as follows:

#### Example launch line with complex if condition

```
gst-launch ... (some tensor stream)
       ! tensor_filter framework=custom name=your_code.so
       ! tensor_if name=tif \
           compared-value=A_VALUE \
           compared-value-option=0:0:0:0,0 \# 1st tensor's [0][0][0][0].
           operator=EQ \
           supplied-value=1 \
           then=PASSTHROUGH \# or whatsoever you want
           else=SKIP \# or whatsoever you want
       ! tif.src_0 ! tensor_demux name=d \ #for TRUE action
          d.src_0 ! queue ! fakesink # throw away the 1/0 value.
          d.src_1 ! queue ! do whatever you want here...
       ! tif.src_1 ! (tensor(s) stream for FALSE action) ...
```

#### Example launch line with custom operation

```
gst-launch ... (some tensor stream) !
       ! tensor_if name=custom_if
       ! (tensor_stream)
       ...
register call back function for custom_if
int TensorIfCallback(const GstBuffer * in_buf, GstBuffer * out_buf) {
  /* Define condition and handle buffer */
}
```

#### Test pipeline
```
gst-launch-1.0 v4l2src ! videoconvert ! videoscale ! video/x-raw, format=RGB, width=640, height=480, framerate=30/1 ! tensor_converter ! mux.sink_0 \
               videotestsrc ! videoconvert ! videoscale ! video/x-raw, format=RGB, width=640, height=480, framerate=30/1 ! tensor_converter ! mux.sink_1 \
               tensor_mux name=mux ! tensor_if name=tif \
                                               compared_value=A_VALUE \
                                               compared-value-option=0:0:0:0,0 \
                                               supplied-value=70,150 \
                                               operator=RANGE_INCLUSIVE \
                                               then = PASSTHROUGH\
                                               else = TENSORPICK \
                                               else-option= 1 \
                tif.src_0 ! tensor_demux name=demux \
                    demux.src_0 ! tensor_decoder mode=direct_video ! videoconvert ! ximagesink async=false sync=false \
                    demux.src_1 ! tensor_decoder mode=direct_video ! videoconvert ! ximagesink async=false sync=false \
                tif.src_1 ! tensor_decoder mode=direct_video ! videoconvert ! ximagesink async=false sync=false
```
