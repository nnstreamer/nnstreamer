---
title: gst-launch script examples
...

### Script of Producer/Consumer

####  GStreamer: producer
```bash
$ ffmpeg -f x11grab -r 15 -s 1280x720 -i :0.0+0,0 -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 /dev/video0

$ gst-launch-1.0 videotestsrc ! v4l2sink device=/dev/video0

$ wget http://file-examples.com/wp-content/uploads/2017/04/file_example_MP4_640_3MG.mp4
$ gst-launch-1.0  filesrc location=./file_example_MP4_640_3MG.mp4 ! decodebin ! videoconvert ! v4l2sink device=/dev/video0
```


#### GStreamer: consumer

```bash
$ gst-launch-1.0 v4l2src device=/dev/video0 ! xvimagesink
(Tip: In case of remote connection such as VNC, run "gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! ximagesink")

$ ./nnstreamer_example_filter
```

### Script of nnstreamer_example_filter using tensorflow lite model (e.g., Mobilenet)
* ltrace -f -tt  -e gst_element_get_type -e memcpy  -o mytracing1.log  ...
* $ ./get-model.sh image-classification-tflite
```bash
gst-launch-1.0 -v -m --gst-debug=3 \
v4l2src ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=RGB ! tee name=t_raw \
textoverlay name=overlay font-desc="Sans, 24" ! videoconvert ! ximagesink \
t_raw. ! queue ! overlay.video_sink \
t_raw. ! queue ! videoscale ! video/x-raw,width=224,height=224 ! tensor_converter ! \
tensor_filter framework=tensorflow-lite model=./tflite_model_img/mobilenet_v1_1.0_224_quant.tflite ! \
tensor_decoder mode=image_labeling option1=./tflite_model_img/labels.txt ! overlay.text_sink
```

### Script of nnstreamer_example_filter using tensorflow model (e.g., Mobilenet)
* ltrace -f -tt  -e gst_element_get_type -e memcpy  -o mytracing1.log  ...
* $ ./get-model.sh object-detection-tf

#### Object detection using tee
 * The video is the same as the original camera output and the labels and bounding boxes are updated after processing in the tensor filter.  
   (The video output rate is the same as the original video frame rate.)
```bash
gst-launch-1.0 -v -m --gst-debug=3 \
v4l2src name=cam_src ! videoscale ! videoconvert ! video/x-raw,width=640,height=480,format=RGB,framerate=30/1 ! tee name=t \
  t. ! queue leaky=2 max-size-buffers=2 ! videoscale ! tensor_converter ! \
    tensor_filter framework=tensorflow model=tf_model/ssdlite_mobilenet_v2.pb \
      input=3:640:480:1 inputname=image_tensor inputtype=uint8 \
      output=1:1:1:1,100:1:1:1,100:1:1:1,4:100:1:1 \
      outputname=num_detections,detection_classes,detection_scores,detection_boxes \
      outputtype=float32,float32,float32,float32 ! \
    tensor_decoder mode=bounding_boxes option1=tf-ssd option2=tf_model/coco_labels_list.txt option4=640:480 option5=640:480 ! \
    compositor name=mix sink_0::zorder=2 sink_1::zorder=1 ! videoconvert ! ximagesink \
  t. ! queue leaky=2 max-size-buffers=10 ! mix.
```
#### Object detection using output-combination option of the tensor filter
 * Video, labels, and bounding boxes are updated after processing in the tensor filter.  
   (The video output rate is the same as the processing rate of the tensor filter.)
```bash
gst-launch-1.0 -v -m --gst-debug=3 \
v4l2src name=cam_src ! videoscale ! videoconvert ! video/x-raw,width=640,height=480,format=RGB,framerate=30/1 ! \
  tensor_converter ! tensor_filter framework=tensorflow model=tf_model/ssdlite_mobilenet_v2.pb \
      input=3:640:480:1 inputname=image_tensor inputtype=uint8 \
      output=1:1:1:1,100:1:1:1,100:1:1:1,4:100:1:1 \
      outputname=num_detections,detection_classes,detection_scores,detection_boxes \
      outputtype=float32,float32,float32,float32 output-combination=i0,o0,o1,o2,o3 ! \
  tensor_demux name=demux tensorpick=0,1:2:3:4 demux.src_1 ! queue leaky=2 max-size-buffers=2 ! \
    tensor_decoder mode=bounding_boxes option1=tf-ssd option2=tf_model/coco_labels_list.txt option4=640:480 option5=640:480 ! \
    compositor name=mix sink_0::zorder=2 sink_1::zorder=1 ! videoconvert ! ximagesink \
  demux.src_0 ! queue leaky=2 max-size-buffers=2 ! tensor_decoder mode=direct_video ! videoconvert ! mix.
```

## Others

```
gst-launch-1.0 -v \
videotestsrc pattern=1 ! video/x-raw,width=200,height=200,format=RGB \
    ! tee name=t \
videomixer name=mix \
      sink_0::xpos=0   sink_0::ypos=0    sink_0::zorder=0\
      sink_1::xpos=100 sink_1::ypos=0    sink_1::zorder=1\
      sink_2::xpos=200 sink_2::ypos=200  sink_2::zorder=2\
      sink_3::xpos=0   sink_3::ypos=200  sink_3::zorder=3\
    ! videoconvert ! autovideosink \
videotestsrc pattern="black" ! video/x-raw,width=200,height=200,format=RGB \
    ! mix.sink_0 \
t. ! queue ! mix.sink_1 \
t. ! queue ! mix.sink_2 \
t. ! queue ! mix.sink_3
```


```
gst-launch-1.0 \
v4l2src name=cam_src ! videoconvert ! videoscale ! \
video/x-raw,width=640,height=480,format=RGB,framerate=30/1 ! tee name=t_raw \
videomixer name=mix \
sink_0::xpos=0 sink_0::ypos=0 sink_0::zorder=0 sink_0::alpha=0.7 \
sink_1::xpos=50 sink_1::ypos=50 sink_1::zorder=1 sink_1::alpha=0.5 ! \
videoconvert ! ximagesink \
t_raw. ! queue ! tensor_converter ! tensor_decoder mode=direct_video ! videoconvert ! tee name=t_tensor \
t_raw. ! queue ! mix.sink_0 \
t_tensor. ! queue ! mix.sink_1 \
t_tensor. ! queue ! videoconvert ! ximagesink
```


```
gst-launch-1.0 \
v4l2src name=cam_src ! videoconvert ! videoscale ! \
video/x-raw,width=640,height=480,format=RGB,framerate=30/1 ! tee name=t_raw \
videomixer name=mix \
sink_0::xpos=0 sink_0::ypos=0 sink_0::zorder=0 sink_0::alpha=0.7 \
sink_1::xpos=50 sink_1::ypos=50 sink_1::zorder=1 sink_1::alpha=0.5 ! \
videoconvert ! ximagesink \
t_raw. ! queue ! tensor_converter ! tensor_decoder mode=direct_video ! videoconvert ! ximagesink \
t_raw. ! queue ! mix.sink_0 \
t_raw. ! queue ! tensor_converter ! tensor_decoder mode=direct_video ! videoconvert ! mix.sink_1
```



## with CAM plugin


```
gst-launch-1.0 \
  v4l2src name=cam_src ! videoconvert ! videoscale ! \
  video/x-raw,width=640,height=480,format=RGB,framerate=\(fraction\)15/1 ! \
  videomixer name=mix sink_0::xpos=0 sink_0::ypos=0 sink_1::xpos=0 ! \
  videoconvert ! xvimagesink \
  videotestsrc ! \
  video/x-raw,width=320,height=240,format=RGB,framerate=\(fraction\)15/1 ! mix.
```


```
gst-launch-1.0 \
  v4l2src name=cam_src ! videoconvert ! videoscale ! \
  videoconvert ! video/x-raw,width=640,height=480,format=RGB,framerate=\(fraction\)30/1 ! \
  videobox border-alpha=0 top=0 left=0 ! \
  videomixer name=mix sink_0::alpha=0.7 sink_1::alpha=0.5 ! \
  videoconvert ! xvimagesink \
  videotestsrc ! \
  video/x-raw,format=RGB,framerate=\(fraction\)5/1,width=320,height=240 ! mix.
```

```
gst-launch-1.0 \
v4l2src name=cam_src ! \
videoconvert ! video/x-raw,width=640,height=480,format=RGB,framerate=\(fraction\)30/1 ! tee name=t ! \
queue ! videoconvert ! videomixer name=mix ! ximagesink \
t. ! queue ! tensor_converter ! tensor_decoder mode=direct_video ! videoconvert ! mix.
```

```
 gst-launch-1.0 -e videomixer name=mix ! videoconvert ! ximagesink \
   videotestsrc pattern=1 ! video/x-raw, framerate=5/1, width=320, height=180, format=RGB ! videobox border-alpha=0 top=0 left=0 ! mix. \
   videotestsrc pattern=15 ! video/x-raw, framerate=5/1, width=320, height=180, format=RGB ! videobox border-alpha=0 top=0 left=-320 ! mix. \
   videotestsrc pattern=13 ! video/x-raw, framerate=5/1, width=320, height=180, format=RGB ! videobox border-alpha=0 top=-180 left=0 ! mix. \
   videotestsrc pattern=0 ! video/x-raw, framerate=5/1, width=320, height=180, format=RGB ! videobox border-alpha=0 top=-180 left=-320 ! mix. \
   videotestsrc pattern=3 ! video/x-raw, framerate=5/1, width=640, height=360, format=RGB ! mix.
```

```
gst-launch-1.0 -v -m --gst-debug=3 \
    videomixer name=mix sink_0::xpos=0 sink_0::ypos=0 sink_1::xpos=0 sink_1::ypos=0 !  videoconvert ! glvideosink sync=false \
    filesrc location=bbb_sunflower_720p_24fps_equal.mp4 ! qtdemux name=demux \
    demux.video_0 ! h264parse ! omxh264dec ! tee name=t \
    t.src_0 ! queue ! mix.sink_0 \
    t.src_1 ! queue ! mix.sink_1
```
