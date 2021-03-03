---
title: Data type and flow control
...

[Rank counting with other/tensor types](rank-counting-with-other-tensor.md)

# GStreamer data types (pad capabilities)

All NNStreamer's GStreamer data types as pad capabilities (```other/tensor*```) have the following common rules

1. In each buffer, there is only ONE frame. That is for each buffer, with the type of ```other/tensor```, there is only one instance of tensor for each buffer at any time. There cannot be multiple tensors in each buffer.
2. The data types do not hold data semantics. Filters should NOT try to determine data semantics (e.g., is it a video?) dynamically based solely on the dimensions, framerates, or element types of the data types. However, if a filter has additional information available including property values from pipeline developers or users, a filter may determine data semantics. For example, ```tensor_decoder``` transforms ```other/tensor``` stream into ```video/x-raw``` or ```text/x-raw``` depending on the property values.

## other/tensor

The GStreamer pad capability has the following structure:
```
other/tensor
    framerate: (fraction) [ 0/1, 2147483647/1 ]
        # We are still unsure how to handle framerate w/ filters.
    dimension: (string with int:int:int:int) [1, 65535]:[1, 65535]:[1, 65535]:[1, 65535]
        # We support up to 4th dimensions only. Supporting higher arbitrary dimension is TBD item.
    type: (string) { uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float64 }
```

The buffer with offset 0 looks like the following with ```dim1=2, dim2=2, dim3=2, dim4=2```:

|      |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| ---- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| dim4 | 0 |||||||| 1 |
| dim3 | 0 |||| 1 |||| 0 |||| 1
| dim2 | 0 || 1 || 0 || 1 || 0 || 1 || 0 || 1 | 
| dim1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| offset/type=uint8 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| offset/type=uint16 | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 |

Therefore, array in C corresponding to the buffer of a ```other/tensor``` becomes:
```
type buffer[dim4][dim3][dim2][dim1];
```

Note that in some context (such as properties of ```tensor_*``` plugins), dimensions are described in a colon-separated string that allows omitting unused dimensions:
```
dim1:dim2:dim3:dim4
```
If rank = 2 (dim3 = 1 and dim4 = 1), then, it can be expressed as well as:
```
dim1:dim2
```

Be careful! Colon-separated tensor dimension expression has the opposite order to the C-array type expression.

## other/tensors

```other/tensors``` is defined to have multiple instances of ```other/tensor``` in a buffer of a single stream path. Compared to having multiple streams (thus multiple pads) with ```other/tensor``` that goes to or comes from a single element, having a single stream with ```other/tensors``` has the following advantages:

- ```tensor_filter```, which is the main engine to communicate deep neural network frameworks and models, becomes simple and robust. With neural network models requiring multiple input tensors, if the input tensor streams are not fully synchronized, we need to somehow synchronize them and provide all input tensors at the same time to the model. Hereby, being fully synchronized means that the streams should provide new data simultaneously, which requires to have the exactly same framerate and timing, which is mostly impossible. With ```other/tensors``` and single input and output streams for ```tensor_filter```, we can delegate the responsibilities of synchronization to GStreamer and its basic plugins, who are extremely good at such tasks.
- During transmissions on a stream pipeline, passing through various stream filters, we can guarantee that the same set of input tensors are being processed without the worries of synchronizations after the point of merging or muxing.

The GStreamer pad capability of ```other/tensors``` is as follows:
```
other/tensors
    num_tensors = (int) [1, 16]  # GST_MAX_MEMCHUNK_PER_BUFFER
    framerate = (fraction) [0/1, 2147483647/1]
    types = (string) Typestrings
    dimensions = (string) Dimensions

Typestrings = (string) Typestring
            | (string) TypeString, TypeStrings
Typestring = (string) { float32, float64, int64, uint64, int32, uint32, int16, uint16, int8, uint8 }
Dimensions = (string) Dimension
           | (string) Dimension, Dimensions
Dimension = (string) [1-65535]:[1-65535]:[1-65535]:[1-65535]
```

The buffer of ```other/tensors``` streams have multiple memory chunks. Each memory chunk represents one single tensor in the buffer format of ```other/tensor```. With default configurations of Gstreamer 1.0, the maximum allowed number of memory chunks in a buffer is **16**. Thus, with such configurations of Gstreamer 1.0, ```other/tensors``` may include up to **16** ```other/tensor```.


## other/tensorsave (TBU)

```other/tensorsave```, along with its ```typefind``` definition, is defined to enable to save ```other/tensors``` streams as files and load such files are ```other/tensors``` stream. With the definitions of headers defined with ```other/tensorsave```, GStreamer can decode the given file and determine that the file belongs to ```other/tensorsave```.

The detailed description of the file format is at [Design External Save Format for other/tensor and other/tensors Stream for TypeFind](https://github.com/nnstreamer/nnstreamer/wiki/Design-External-Save-Format-for-other-tensor-and-other-tensors-Stream-for-TypeFind)



# Flow control

## Timestamps
In general, tensor_* chooses the most recent timestamp when there are multiple candidates. For example, if we are merging/muxing/aggregating two frames from sinkpads, ```T``` and ```T+a```, where ```a > 0```, the source pads are supposed to have ```T+a```.  
We have the following principles for timestamp policies. Timestamping policies of ```tensor_*``` filters should follow the given principles.  
- Timestamp from the input source (sensors) should be preserved to sink elements.
- When there are multiple flows merging into one (an element with multiple sink pads), a timestamp of the most recent should be preserved.
    - For the current frame buffer in a sink pad, ```i``` in ```1 ... n```, ```FB(i)```, and ```T(FB(i))``` is the timestamp of the current frame buffer, the timestamp of the corresponding frame buffer at source pads generated by ```FB(1)``` ... ```FB(n)``` is ```max(i = 1 .. n, T(FB(i)))```, where larger timestamp value means the more recent event.
    - Example: when multiple frames of the same stream are combined by ```tensor_mux```, according to the principle, the timestamp of the last frame of the combined frames is used for output timestamp.
    - Note that this principle might cause confusion when we apply ```tensor_demux```, where we may extract some "old" frames from incoming combined frames. However, as a single frame in a GStreamer stream has a single timestamp, we are going to ignore it for now.

## Synchronization of frames in sink pads with Mux and Merge

Besides timestamping, we have additional synchronization issues when there are merging streams. We need to determine which frames are going to be merged (or muxed) when we have multiple available and unused frames in an incoming sink pad. In general, we might say that the synchronization of frames determines which frames to be used for mux/merge and timestamping rule determins which timestamp to be used among the chosen frames for mux/merge.  
In principle and by default,
- If there are mutliple unused and available frames in a sink pad, unlike most media filters, we take a buffer that arrived most recently.
- For more about the synchronization policies, see [Synchronization policies at Mux and Merge](synchronization-policies-at-mux-merge.md)

### Leaky Queue

In some usage cases, we may need to drop frames from a queue. With the timestamp values of frames, a queue may drop frames with different policies according to GStreamer applications. Such policies include:

* Leaky on upstream: Drop more recent frames, keep older frames
* Leaky on downstream: Drop older frames, keep newer frames

Note that in the case of many multi-modal neural networks, mux/merge elements are supposed to drop any older frames with incoming frames in the incoming (sink pad) queue.

## Synchronization of frames in source pads with Demux and Split

This is an obvious case. The timestamp is copied to all source pads from the sink pads. We do not preserve original timestamps in Merge or Mux; thus, the processed timestamp after Mux or Merge will only be applied.

## Synchronization with Aggregator
Unlike mux and merge, aggregator merges tensors chronologically, not spatially.  
Moreover, unlike mux and merge, which merges entries into one entry, aggregator, depending on the properties, may divide or even simultaneously merge and divide entries. Thus, timestamping and synchronization may become much more complicated.  
The timestamp of the outging buffer is timestamp of the oldest frame from the aggregated frames.
