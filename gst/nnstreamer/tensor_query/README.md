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
- The capability of source and sink pad is ```ANY```.
- The capability of the tensor_client sink must match the capability of the tensor_query_serversrc.
- The capability of the tensor_client source must match the capability of the tensor_query_serversink.

### tensor_query_serversrc
- Used for heavyweight device.
- Receive requests and data from clients.
- The capability of tensor_query_serversrc is ```ANY```.

### tensor_query_serversink
- Used for heavyweight device.
- Send the results processed by the server to the clients.
- The capability of tensor_query_serversink is ```ANY```.

## Usage Example
### echo server
As the simplest example, the server sends the data received from the client back to the client.
 * If you didn't install nnstreamer, see [here](/Documentation/how-to-run-examples.md).
#### server
```bash
$ gst-launch-1.0 tensor_query_serversrc ! video/x-raw,width=300,height=300,format=RGB,framerate=30/1 ! tensor_query_serversink
```
#### client
```bash
$ gst-launch-1.0 v4l2src ! videoconvert ! videoscale !  video/x-raw,width=300,height=300,format=RGB,framerate=30/1 ! tensor_query_client ! videoconvert ! ximagesink
```
#### client 2 (Optional, To test multiple clients)
```bash
$ gst-launch-1.0 videotestsrc ! videoconvert ! videoscale !  video/x-raw,width=300,height=300,format=RGB,framerate=30/1 ! tensor_query_client ! videoconvert ! ximagesink
```

### Object-detection
The client sends the video to the server, the server performs object detection(which requires high-performance work) and send the results to the client.
#### server
```bash
$ gst-launch-1.0 \
    tensor_query_serversrc ! video/x-raw,width=640,height=480,format=RGB,framerate=0/1 ! \
        videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! \
        tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! \
        tensor_filter framework=tensorflow-lite model=tflite_model/ssd_mobilenet_v2_coco.tflite ! \
        tensor_decoder mode=bounding_boxes option1=mobilenet-ssd option2=tflite_model/coco_labels_list.txt option3=tflite_model/box_priors.txt option4=640:480 option5=300:300 ! \
        videoconvert ! tensor_query_serversink
```
#### client
```bash
$ gst-launch-1.0 \
    compositor name=mix sink_0::zorder=2 sink_1::zorder=1 ! videoconvert ! ximagesink \
        v4l2src ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=RGB,framerate=10/1 ! tee name=t \
            t. ! queue ! tensor_query_client ! videoconvert ! mix.sink_0 \
            t. ! queue ! mix.sink_1
```

* How to get object detection model
```
$ git clone https://github.com/nnstreamer/nnstreamer-example.git
$ cd nnstreamer-example/bash_script/example_models
$ ./get-model.sh ./get-model.sh object-detection-tflite
```

### MQTT-hybrid
Above two examples, `echo sever` and `Object-detection`, use TCP direct connection. Tensor query provides two methods for connection: TCP direct connection and MQTT-hybrid.
  1) TCP direct connection:  
    The connection between the client and the server uses the IP and port given by the user or default values. Therefore, when the server stops working, the client cannot find another alternative server and stops. With TCP direct connections, flexibility and robustness cannot be provided.
  2) MQTT-hybrid:  
    MQTT-hybrid exchanges the connection information using MQTT. The server publishes connection information to the MQTT broker, the client subscribes them from the MQTT broker and creates TCP connections for data transmission using the information gotten by MQTT. So, MQTT broker transmits only small data such as connection information, and high-bandwidth data is transmitted through TCP direct connection. Because MQTT broker is not suitable for large amounts of data such as high resolution video. The reason for using MQTT is that it can manage connection information through MQTT, so if the connected server stops, the client can find an alternative server, create a new connection and start streaming again. Therefore, MQTT-hybrid has the advantage of flexibility and robustness of the connection by using MQTT and a high-bandwidth data transmission capability through TCP direct connection.

#### server 1
If server 1 is stopped, the client will connect to server 2. Run server 2 and stop server 1 during operation.
```bash
$ gst-launch-1.0 \
    tensor_query_serversrc operation=passthrough port=3001 ! video/x-raw,width=640,height=480,format=RGB,framerate=0/1 ! \
        videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! \
        tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! \
        tensor_filter framework=tensorflow-lite model=tflite_model/ssd_mobilenet_v2_coco.tflite ! \
        tensor_decoder mode=bounding_boxes option1=mobilenet-ssd option2=tflite_model/coco_labels_list.txt option3=tflite_model/box_priors.txt option4=640:480 option5=300:300 ! \
        videoconvert ! tensor_query_serversink port=3000
```

#### server 2
```bash
$ gst-launch-1.0 \
    tensor_query_serversrc operation=passthrough port=3003 ! video/x-raw,width=640,height=480,format=RGB,framerate=0/1 ! \
        videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! \
        tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! \
        tensor_filter framework=tensorflow-lite model=tflite_model/ssd_mobilenet_v2_coco.tflite ! \
        tensor_decoder mode=bounding_boxes option1=mobilenet-ssd option2=tflite_model/coco_labels_list.txt option3=tflite_model/box_priors.txt option4=640:480 option5=300:300 ! \
        videoconvert ! tensor_query_serversink port=3002
```

#### client
```bash
$ gst-launch-1.0 \
    compositor name=mix sink_0::zorder=2 sink_1::zorder=1 ! videoconvert ! ximagesink \
        v4l2src ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=RGB,framerate=10/1 ! tee name=t \
            t. ! queue ! tensor_query_client operation=passthrough ! videoconvert ! mix.sink_0 \
            t. ! queue ! mix.sink_1
```

#### Prerequisite
 - NNStreamer: [link](https://github.com/nnstreamer/nnstreamer/wiki/usage-examples-screenshots)
 - NNStreamer-edge (nnsquery): [link](https://github.com/nnstreamer/nnstreamer-edge/tree/master/src/libsensor)
 - Install mosquitto broker: `$ sudo apt install mosquitto mosquitto-clients`

## tensor query test
### To check the results without running the test: [Daily build result](http://ci.nnstreamer.ai/nnstreamer/ci/daily-build/build_result/latest/log/).
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


## Appendix
### Available elements on query server.
Multiple `tensor_query_client` can connect to the query server. The `query_serversrc` add a unique client ID (given by the query server) to the meta of the GstBuffer to distinguish clients. If there is an element that does not copy meta information, the `tensor_query_serversink` cannot send it to the client because it does not know which client receive the buffer.  
Please check list [here](https://github.com/nnstreamer/nnstreamer/wiki/Available-elements-on-query-server)
