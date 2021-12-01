---
title: T5. GStreamer API
...

# Tutorial 5. Build application using GStreamer API
The `gst-launch` tool is convenient, but it is recommended to use it simply to test pipeline description.  
In this tutorial, let's learn how to create an application. (Full example source code can be found [here](https://github.com/nnstreamer/nnstreamer-example/blob/main/native/example_object_detection_tensorflow_lite/nnstreamer_example_object_detection_tflite.cc)).  
Creating a pipeline application is simple. You can use `gst_parse_launch ()` which is GStreamer API instead of `gst-launch` tool.  
Refer to [here](../how-to-run-examples.md) for building nnstreamer-example.

This is a part of the code explanation. The code lines below are essential for operating a pipeline application. The omitted codes are error handling and additional.  

Part of **nnstreamer_example_object_detection_tflite.cc**
```
int
main (int argc, char ** argv)
{
  ...

  /* Init gstreamer */
  gst_init (&argc, &argv);

  /* Same pipeline description as tutorial 2 */
  str_pipeline =
      g_strdup_printf
      ("v4l2src name=src ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! tee name=t "
          "t. ! queue leaky=2 max-size-buffers=2 ! videoscale ! video/x-raw,format=RGB,width=300,height=300 ! tensor_converter ! "
            "tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! "
            "tensor_filter framework=tensorflow-lite model=%s ! "
            "tensor_decoder mode=bounding_boxes option1=mobilenet-ssd option2=%s option3=%s option4=640:480 option5=300:300 !"
            "compositor name=mix sink_0::zorder=2 sink_1::zorder=1 ! videoconvert ! ximagesink "
       "t. ! queue leaky=2 max-size-buffers=10 ! mix. ",
      g_app.tflite_info.model_path, g_app.tflite_info.label_path, g_app.tflite_info.box_prior_path);

  ...

  /* Create a new pipeline */
  g_app.pipeline = gst_parse_launch (str_pipeline, NULL);

  ...

  /* Start pipeline. Setting pipeline to the PLAYING state. */
  gst_element_set_state (g_app.pipeline, GST_STATE_PLAYING);

  ...
  /* Pipeline is running */
  ...

  /* Stop and release the resources */
  gst_element_set_state (g_app.pipeline, GST_STATE_NULL);
  gst_object_unref (g_app.pipeline);

  ...
}
```

Then let's run the application.
```
# If you built tutorial 5 yourself, go to the path you specified, and if you installed it using apt, go to /usr/lib/nnstreamer/bin.
$ cd /usr/lib/nnstreamer/bin
$ ./nnstreamer_example_object_detection_tflite
```
