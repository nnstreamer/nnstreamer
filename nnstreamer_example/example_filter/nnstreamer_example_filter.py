# !/usr/bin/env python3

"""
@file		nnstreamer_example_filter.py
@date		18 July 2018
@brief		Tensor stream example with filter
@see		https://github.sec.samsung.net/STAR/nnstreamer
@author		Jaeyun Jung <jy1210.jung@samsung.com>
@bug		No known bugs.

NNStreamer example for image recognition.

Pipeline :
v4l2src -- tee -- textoverlay -- videoconvert -- xvimagesink
            |
            --- tensor_converter -- tensor_filter -- tensor_sink

This app displays video sink (xvimagesink).
'tensor_filter' for image recognition.
'tensor_sink' updates recognition result to display in textoverlay.

Run example :
Before running this example, GST_PLUGIN_PATH should be updated for nnstreamer plugin.
$ export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:<nnstreamer plugin path>
$ python nnstreamer_example_filter.py

See https://lazka.github.io/pgi-docs/#Gst-1.0 for Gst API details.
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject


class NNStreamerExample:
    """NNStreamer example for image recognition."""

    def __init__(self):
        self.loop = None
        self.pipeline = None
        self.running = False
        self.received = 0

        GObject.threads_init()
        Gst.init(None)

    def run_example(self):
        """Init pipeline and run example.

        :return: None
        """
        # main loop
        self.loop = GObject.MainLoop()

        # init pipeline
        # TODO: add tensor filter
        self.pipeline = Gst.parse_launch(
            "v4l2src name=cam_src ! "
            "video/x-raw,width=640,height=480,format=RGB,framerate=30/1 ! tee name=t_raw "
            "t_raw. ! queue ! textoverlay name=tensor_res font-desc=\"Sans, 24\" ! "
            "videoconvert ! xvimagesink name=img_tensor "
            "t_raw. ! queue ! tensor_converter ! tensor_sink name=tensor_sink"
        )

        # bus and message callback
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

        # tensor sink signal : new data callback
        tensor_sink = self.pipeline.get_by_name("tensor_sink")
        tensor_sink.connect("new-data", self.on_new_data)

        # timer to update result
        GObject.timeout_add(500, self.on_timer_update_result)

        # start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True

        # set window title
        self.set_window_title("img_tensor", "NNStreamer Example")

        # run main loop
        self.loop.run()
        # quit when received eos or error message

        self.running = False
        self.pipeline.set_state(Gst.State.NULL)

        bus.remove_signal_watch()

    def on_bus_message(self, bus, message):
        """Callback for message.

        :param bus: pipeline bus
        :param message: message from pipeline
        :return: None
        """
        if message.type == Gst.MessageType.EOS:
            print("received eos message")
            self.loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            print(f"error {error} {debug}")
            self.loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            error, debug = message.parse_warning()
            print(f"warning {error} {debug}")
        elif message.type == Gst.MessageType.STREAM_START:
            print("received start message")

    def on_new_data(self, sink, buffer):
        """Callback for tensor sink signal.

        :param sink: tensor sink element
        :param buffer: buffer from element
        :return: None
        """
        # print progress
        self.received += 1
        if (self.received % 150) == 0:
            print(f"receiving new data [{self.received}]")

        if self.running:
            # TODO: update textoverlay
            for idx in range(buffer.n_memory()):
                mem = buffer.peek_memory(idx)
                result, mapinfo = mem.map(Gst.MapFlags.READ)
                if result:
                    # print(f"received {mapinfo.size}")
                    mem.unmap(mapinfo)

    def on_timer_update_result(self):
        """Timer callback for textoverlay.

        :return: True to ensure the timer continues
        """
        if self.running:
            # TODO: update textoverlay
            tensor_res = f"total received {self.received}"

            textoverlay = self.pipeline.get_by_name("tensor_res")
            textoverlay.set_property("text", tensor_res)
        return True

    def set_window_title(self, name, title):
        """Set window title.

        :param name: GstXImageSink element name
        :param title: window title
        :return: None
        """
        element = self.pipeline.get_by_name(name)
        if element is not None:
            pad = element.get_static_pad("sink")
            if pad is not None:
                tags = Gst.TagList.new_empty()
                tags.add_value(Gst.TagMergeMode.APPEND, "title", title)
                pad.send_event(Gst.Event.new_tag(tags))


if __name__ == "__main__":
    example = NNStreamerExample()
    example.run_example()
