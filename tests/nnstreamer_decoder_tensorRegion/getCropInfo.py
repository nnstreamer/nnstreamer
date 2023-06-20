##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2023 Harsh Jain <hjain24in@gmail.com>
#
# @file getCropInfo.py
# @brief generate human readable output from out buffers
# @author Harsh Jain <hjain24in@gmail.com>

import sys
import gi
import struct

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# Check if the image path argument is provided
if len(sys.argv) < 2:
    print("Please provide the path to the image file as an argument.")
    sys.exit(1)

image_path = sys.argv[1]

# Create the GStreamer pipeline
pipeline_str = """
    filesrc location={} ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=RGB ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter framework=tensorflow2-lite model=ssd_mobilenet_v2_coco.tflite ! tensor_decoder mode=tensor_region option1=1 option2=../nnstreamer_decoder_boundingbox/coco_labels_list.txt option3=../nnstreamer_decoder_boundingbox/box_priors.txt ! appsink name=output
""".format(image_path)

pipeline = Gst.parse_launch(pipeline_str)

# Define callback function to process the tensor data

def process_tensor_data(appsink):
    sample = appsink.emit("pull-sample")
    buffer = sample.get_buffer()

    # Extract tensor data
    header_buffer = buffer.extract_dup(0, 128)
    tensor_data_buffer = buffer.extract_dup(128, 16)  # Extract the remaining data as the tensor

    # Process tensor data
    tensor_size = len(tensor_data_buffer) // 4  # Assuming each value is a uint32 (4 bytes)
    tensor_data = struct.unpack(f"{tensor_size}I", tensor_data_buffer)

    # Print the tensor data
    print("Tensor data:", tensor_data)

    # Stop the main loop
    loop.quit()

    return Gst.FlowReturn.OK




# Set up the appsink element to capture tensor data
output = pipeline.get_by_name("output")
output.set_property("emit-signals", True)
output.set_property("max-buffers", 1)
output.connect("new-sample", process_tensor_data)

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)

# Run the main loop
loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    pass

# Stop the pipeline and clean up
pipeline.set_state(Gst.State.NULL)
