# NNStreamer::tensor\_if

## Supported features

GstTensorIF controls its src-pad based on the values (other/tensor(s)) of its sink-pad.

For example, you may skip frames if there is no object detected with high confidence.

The input/output stream data type is either ```other/tensor``` or ```other/tensors```.

## Properties

- compared_value: Compared Value (CV), operand 1 (from input tensor(s))
  * A_VALUE:  Decide based on a single scalar value
  * TENSOR_TOTAL_VALUE: Decide based on a total (sum) value of a specific tensor
  * ALL_TENSORS_TOTAL_VALUE:  Decide based on a total (sum) value of tensors or a specific tensor
  * TENSOR_AVERAGE_VALUE: Decide based on a average value of a specific tensor
  * ALL_TENSORS_AVERAGE_VALUE: Decide based on a average value of tensors or a specific tensor

- compared_value_option: An element of the nth tensor
  * [C][W][H][B],n (e.g., [0][0][0][0],0 means [0][0][0][0] value of first tensor)
  * nth tensor,nth tensor,...

- supplied_values: Supplied Value (SV), operand 2 (from the properties)
  * SV
  * SV1,SV2 (for RANGE operators)

- supplied_values_option: Supplied Value Option
  * type of the supplied value (e.g., type:uint8)

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
  * PASSTHROUGH: The element will not make changes to the buffers, buffers are pushed straight through.
  * SKIP: Do not generate output frame (frame skip)
  * FILL_ZERO: Fill output frame with zeros
  * FILL_VALUES: Fill output frame with a user given value
  * FILL_WITH_FILE: Fill output frame with a user given file (a raw data of tensor/tensors). If the filesize is smaller, the reset is filled with 0
  * FILL_WITH_FILE_RPT: Fill output frame with a user given file (a raw data of tensor/tensors). If the filesize is smally, the file is repeatedly used
  * REPEAT_PREVIOUS_FRAME: Resend the previous output frame. If this is the first, send ZERO values.
  * TENSORPICK: Choose nth tensor (or tensors) among tensors

```
   [ tensor 0 ]
   [ tensor 1 ]  ->   tensor if    ->    [ tensor 0 ]
   [ tensor 2 ]    (tensorpick 0,2)      [ tensor 2 ]
  input tensors                         output tensors
```

- then_option: Option for TRUE Action
  * nth tensor, nth tensor, ... (for TENSORPICK option)
  * ${path_to_file}

- else: Action if it is FALSE
  * PASSTHROUGH: The element will not make changes to the buffers, buffers are pushed straight through.
  * SKIP: Do not generate output frame (frame skip)
  * FILL_ZERO: Fill output frame with zeros
  * FILL_VALUES: Fill output frame with a user given value
  * FILL_WITH_FILE: Fill output frame with a user given file (a raw data of tensor/tensors). If the filesize is smaller, the reset is filled with 0
  * FILL_WITH_FILE_RPT: Fill output frame with a user given file (a raw data of tensor/tensors). If the filesize is smally, the file is repeatedly used
  * REPEAT_PREVIOUS_FRAME: Resend the previous output frame. If this is the first, send ZERO values.
  * TENSORPICK: Choose nth tensor (or tensors) among tensors

- else_option: Option for FALSE Action
  * nth tensor,nth tensor,... (for TENSORPICK option)
  * ${path_to_file}

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
                supplied-values=10,100 \
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
           compared_value=A_VALUE \
           compared_value_option=0:0:0:0,0 \# 1st tensor's [0][0][0][0].
           operator=EQ \
           supplied_values=1 \
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
