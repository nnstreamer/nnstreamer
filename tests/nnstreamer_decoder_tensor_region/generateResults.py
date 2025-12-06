##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2023 Harsh Jain <hjain24in@gmail.com>
#
# @file generateResults.py
# @brief To generate Golden Results
# @author Harsh Jain <hjain24in@gmail.com>

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import sys


def resize_and_crop_image(input_image, output_image, x, y, width, height, new_width, new_height):
    Gst.init(None)

    pipeline_str = (
        f"filesrc location={input_image} ! decodebin ! "
        f"videoconvert ! videoscale ! video/x-raw,width={new_width},height={new_height},format=RGB ! "
        f"videoconvert  ! video/x-raw,format=RGBx ! appsink name=sinkx"
    )

    pipeline = Gst.parse_launch(pipeline_str)

    # Create a GstAppSink to receive the raw RGB data
    appsink = pipeline.get_by_name("sinkx")
    appsink.set_property("sync", False)
    appsink.set_property("max-buffers", 1)

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # Wait until the first buffer is received
    sample = appsink.emit("pull-sample")

    # Extract the raw RGB data from the sample
    buffer = sample.get_buffer()
    result, map_info = buffer.map(Gst.MapFlags.READ)
    raw_data = np.frombuffer(map_info.data, dtype=np.uint8)

    # Unmap the buffer
    buffer.unmap(map_info)

    # Stop the pipeline
    pipeline.set_state(Gst.State.NULL)

    # Reshape the raw data into the image dimensions
    image_data = raw_data.reshape((new_height, new_width, 4))

    # Crop the image
    cropped_image = image_data[y:y+height, x:x+width, :]

    # Save the cropped image as output_image
    with open(output_image, "wb") as file:
        file.write(cropped_image.tobytes())


# Example usage
input_image = sys.argv[1]
output_image = sys.argv[2]
x = int(sys.argv[3])
y = int(sys.argv[4])
width = int(sys.argv[5])
height = int(sys.argv[6])
new_width = 300  # Resized width
new_height = 300  # Resized height

resize_and_crop_image(input_image, output_image, x, y, width, height, new_width, new_height)
