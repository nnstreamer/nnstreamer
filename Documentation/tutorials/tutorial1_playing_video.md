---
title: T1. Playing Video
...

# Tutorial 1. Playing video
It's very simple. One line is enough to play the video.  
We will simply run the pipeline using the `gst-launch` tool.

## Prerequisites
```
# Install GStreamer packages for Tutorial 1 and later use.
$ sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
```

## Playing video!
```
$ gst-launch-1.0 v4l2src ! videoconvert ! videoscale ! video/x-raw, width=640, height=480, framerate=30/1 ! autovideosink
# If you don't have a camera device, use `videotestsrc`.
$ gst-launch-1.0 videotestsrc ! videoconvert ! videoscale ! video/x-raw, width=640, height=480, framerate=30/1 ! autovideosink
```

That's all!  
Change the width, height, and frame rate of the video.  
Now that we're close to GStreamer, let's move on to [nnstreamer](tutorial2_object_detection.md) tutorials.

## Additional description for used elements.
Use the `gst-inspect-1.0` tool for more information of the element.
```
$ gst-inspect-1.0 videoconvert
Factory Details:
  Rank                     none (0)
  Long-name                Colorspace converter
  Klass                    Filter/Converter/Video
  Description              Converts video from one colorspace to another
...
```
 - v4l2src: Reads frames from a linux video device.
 - videotestsrc: Creates a test video stream.
 - videoconvert: Converts video format.
 - videoscale: Resizes video.
 - autovideosink: Automatically detects video sink.

