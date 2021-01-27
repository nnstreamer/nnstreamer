---
title: Synchronization policies
...

# Synchronization policies at Mux and Merge
There are multiple synchronization policies for tensor_mux and tensor_merge.
They are based on PTS ( presentation timestamp in nanoseconds (as a GstClockTime) ) and assume that every tensor buffer has PTS and can be accessed by GST_BUFFER_PTS(buf).  
However, there is stream which does not have PTS such as application/octet-stream. In such cases, tensor_converter generates timestamp and set PTS. If framerate is given, tensor_converter generates proper PTS according to framerate and for the case without framerate, PTS is decided with running time which is calculated as absolute-time - base-time.  
Currently, three synchronization policies are implemented.

# No synchronization

With "nosync" mode, it merges or mux tensors in an order of updated.  
Just merge/mux buffers from pads without concern of synchronization. First comes, first merge/mux.  
For the case of merge/mux element which has three pads with different framerate (30/1, 20/1,10/1), it push merged or muxed buffer as below.

```
    *srcpad0        *srcpad1        *srcpad2
       0               0               0     
    33333333        50000000      1000000000  // output buffers timestamp with 1000000000
    66666666       100000000      2000000000  // output buffers timestamp with 2000000000
    99999999       150000000      3000000000  // ...
   133333332       200000000      4000000000 
   166666666       250000000      5000000000 
   199999999       300000000      6000000000
   233333331       350000000      7000000000 
   266666666       400000000      8000000000 
   299999997       450000000      9000000000 
```

# Slowest

"Slowest" policy (sync_mode=slowest) synchronize tensors based on slowest timestamp among pads.  
Finding slowest timestamp among pads and used as base timestamp. It drops buffers which is earlier than this base timestamp. However if the difference with timestamp of previous buffer is smaller than with current buffer, then previous buffer is used.  
For the case of three pad with different framerates, It merged/muxed

```
    srcpad0         srcpad1         *srcpad2
       0               0               0     
    99999999       100000000      1000000000  // output buffers timestamp with 1000000000
   199999999       200000000      2000000000  // output buffers timestamp with 2000000000
   299999997       300000000      3000000000  // output buffers timestamp with 3000000000
```

As you can see, second and third buffers of srcpad0 are dropped. Because it is smaller than slowest pts (100000000). For the case of 99999999 PTS of srcpad0, it is taken even if it is also smaller than slowest pts. That is because difference with slowest pts ( 100000000 - 99999999 ) is smaller than of next buffer (133333333 - 100000000).

# BasePad

With "Base Pad", Base timestamp is decided with designated pad which is given by user with sync_option.  
Sync_option consists of two variables and first denotes the base pad number and second is duration in nanoseconds ( as a GstClockTime ). In this policy, every buffer which has pts is within duration from base time stamp is merged/muxed. For the case of buffer with greater pts than base timestamp plus duration, previous buffer is going to used instead.  
Test case with "sync_mode=basepad sync_option=0:33333333" is below,

```
    *srcpad0        srcpad1         srcpad2
       0               0               0     
    33333333        50000000           0     
    66666666        50000000           0    
    99999999       100000000      1000000000  // output buffers! timestamp: 99999999
   133333332       150000000      1000000000
   166666666       150000000      1000000000
   199999999       200000000      2000000000  // output buffers! timestamp: 199999999
   233333331       250000000      2000000000
   266666666       250000000      2000000000 
   299999997       300000000      3000000000  // output buffers! timestamp: 299999997
```

The base timestamp is 0, so that every buffer of srcpad0 is pushed to downstream.

# Refresh

The other is "Refresh" policy. The Base timestamp is decided with the pad which receives a new buffer.  
The above 3 policies require all pads are collected state. It means all of the sinkpads of `tensor_mux` have to be filled. However, with "Refresh", `tensor_mux` pushes the buffers to srcpad when each sinkpad receives a new buffer. For the sinkpads which not received the new buffer will use again the previous one.  

Test case with "sync_mode=refresh" is below,

```
    sinkpad0         sinkpad1         sinkpad2
       0                0                0        <- At the first time, all of the sinkpads have to be filled.
       0                1                0        <- sinkpad1 receives new data `1`
       0                1                2        <- sinkpad2 receives new data `2`
       0                1                3        <- sinkpad2 receives new data `3`
       4                1                3        <- sinkpad0 receives new data `4`, output buffers! timestamp of the buffer which is arrived on sinkpad0
       4                1                5        <- sinkpad2 receives new data `5`, output buffers! timestamp of the buffer which is arrived on sinkpad2
```
