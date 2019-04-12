# NNStreamer::tensor_source

## Motivation

Allow non Gstreamer standard input sources to provide other/tensor or other/tensors stream.

Such sources include, but not limited to:

- Output of general external applications including instances of nerual network frameworks and models not using NNStreamer suite (TBD).
- Output of non gstreamer compatible sensors, such as Linux IIO devices: [Industrial I/O, 01.org](https://01.org/linuxgraphics/gfx-docs/drm/driver-api/iio/index.html). We may need to get streams from thermostats, light detectors, IMUs, or signals from GPIO pins.
- Output of LIDAR and RADAR in case we do not have V4L2 interfaces for them (TBD).


## Output Format (src_pad)

other/tensor or other/tensors
