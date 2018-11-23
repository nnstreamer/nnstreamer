#!/usr/bin/env bash
gst-launch-1.0 \
	v4l2src name=cam_src ! videoscale ! videoconvert ! video/x-raw,width=640,height=480,format=RGB,framerate=30/1 ! tee name=t \
	t. ! queue leaky=2 max-size-buffers=2 ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! \
		tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! \
		tensor_filter framework=tensorflow-lite model=ssd_mobilenet_v2_coco.tflite ! \
		tensor_decoder mode=bounding_boxes option1=ssd option2=coco_labels_list.txt option3=box_priors.txt option4=640:480 option5=300:300 ! \
		compositor name=mix sink_0::zorder=2 sink_1::zorder=1 ! videoconvert ! ximagesink \
	t. ! queue leaky=2 max-size-buffers=10 ! mix.
