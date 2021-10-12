---
title: tensor_query
...

# NNStreamer::tensor\_query
Tensor query allows devices which have weak AI computational power to to use resources from higher-performance devices.  
Suppose you have a device at home with sufficient computing power (server) and a network of lightweight devices connected to it (clients).  
The client asks the server to handle heavy tasks and receive results from the server.  
Therefore, there is no need for cloud server by running AI on a local network.

## Elements
### tensor_query_client
- Used for lightweight device.
- The capability of source and sink pad is ```other/tensors```.

### tensor_query_serversrc
- Used for heavyweight device.
- Receive requests and data from clients.
- The capability of source and sink pad is ```other/tensors```.

### tensor_query_serversink
- Used for heavyweight device.
- Send the results processed by the server to the clients.
- The capability of source and sink pad is ```other/tensors```.

## Usage Example
### echo server
As the simplest example, the server sends the data received from the client back to the client.
 * If you didn't install nnstreamer, see [here](/Documentation/how-to-run-examples.md).
#### server
```bash
$ gst-launch-1.0 tensor_query_serversrc ! other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8,framerate=30/1 ! tensor_query_serversink
```
#### client
```bash
$ gst-launch-1.0 v4l2src ! videoconvert ! videoscale !  video/x-raw,width=300,height=300,format=RGB,framerate=30/1 ! tensor_converter ! tensor_query_client ! tensor_decoder mode=direct_video ! videoconvert ! ximagesink
```
#### client 2 (Optional, To test multiple clients)
```bash
$ gst-launch-1.0 videotestsrc ! videoconvert ! videoscale !  video/x-raw,width=300,height=300,format=RGB,framerate=30/1 ! tensor_converter ! tensor_query_client ! tensor_decoder mode=direct_video ! videoconvert ! ximagesink
```

### Object-detection
The client sends the video to the server, the server performs object detection(which requires high-performance work) and send the results to the client.
#### server
```bash
$ gst-launch-1.0 \
    tensor_query_serversrc ! tensor_filter framework=tensorflow-lite model=tflite_model/ssd_mobilenet_v2_coco.tflite ! \
    tensor_decoder mode=bounding_boxes option1=mobilenet-ssd option2=tflite_model/coco_labels_list.txt option3=tflite_model/box_priors.txt option4=640:480 option5=300:300 ! \
    tensor_converter ! other/tensors,num_tensors=1,dimensions=4:640:480:1,types=uint8 ! tensor_query_serversink
```
#### client
```bash
$ gst-launch-1.0 \
    compositor name=mix sink_0::zorder=2 sink_1::zorder=1 ! videoconvert ! ximagesink \
        v4l2src ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=RGB,framerate=10/1 ! tee name=t \
            t. ! queue ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! queue leaky=2 max-size-buffers=2 ! \
                tensor_query_client ! tensor_decoder mode=direct_video ! videoconvert ! video/x-raw,width=640,height=480,format=RGBA ! mix.sink_0 \
            t. ! queue ! mix.sink_1
```

* How to get object detection model
```
$ git clone https://github.com/nnstreamer/nnstreamer-example.git
$ cd nnstreamer-example/bash_script/example_models
$ ./get-model.sh ./get-model.sh object-detection-tflite
```

## tensor query test
### To check the results without running the test: [Daily build result](http://nnstreamer.mooo.com/nnstreamer/ci/daily-build/build_result/latest/log/).
 - GTest results
 ```
 [  173s] ~/rpmbuild/BUILD/nnstreamer-1.7.2
[  173s] ++ pwd
[  173s] + bash /home/abuild/rpmbuild/BUILD/nnstreamer-1.7.2/packaging/run_unittests_binaries.sh ./tests
[  173s] ~/rpmbuild/BUILD/nnstreamer-1.7.2/build ~/rpmbuild/BUILD/nnstreamer-1.7.2
[  173s] [==========] Running 8 tests from 2 test suites.
[  173s] [----------] Global test environment set-up.
[  173s] [----------] 5 tests from tensorQuery
[  173s] [ RUN      ] tensorQuery.serverProperties0
...
```
 - SSAT results
 ```
 [  407s] [Starting] nnstreamer_query
[  408s] ==================================================
[  408s]     Test Group nnstreamer_query Starts.
[  408s] [PASSED] 1-2:gst-launch of case 1-2
...
[  408s] ==================================================
[  408s] [PASSED] Test Group nnstreamer_query Passed
[  408s]
```

### Run test on Tizen
```bash
$ git clone https://github.com/nnstreamer/nnstreamer.git
$ cd nnstreamer
$ gbs build --define "unit_test 1"
```
 * About gbs build, refer [here](/Documentation/getting-started-tizen.md)

### Run test on Ubuntu
For gtest based test cases
```bash
$ cd nnstreamer
$ meson build
$ ninja -C build test
# or if you want to run tensor_query only
$ cd nnstreamer
$ meson build
$ ninja -C build install
$ cd build
$ ./tests/unittest_query
```
For SSAT based test cases
```bash
$ cd nnstreamer/tests/nnstreamer_query
$ ssat # or $ bash runTest.sh
```
 * For more detailed installation methods, see [here](/Documentation/how-to-run-examples.md).
