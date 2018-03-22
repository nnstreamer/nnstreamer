#include <gst/gst.h>

typedef struct _CustomData {
  GstElement *pipeline;
  GstElement *source;
  GstElement *convert;
  GstElement *sink;

  GstElement *vconvert;
  GstElement *vsink;
} CustomData;

static void pad_added_handler (GstElement *src, GstPad *pad, CustomData *data);

int main(int argc, char *argv[]) {
  CustomData data;
  GstBus *bus;
  GstMessage *msg;
  GstStateChangeReturn ret;
  gboolean terminate = FALSE;

  gst_init(&argc, &argv);

  data.source = gst_element_factory_make("uridecodebin", "source");
  data.convert = gst_element_factory_make("audioconvert", "convert");
  data.sink = gst_element_factory_make("autoaudiosink", "sink");
  data.vconvert = gst_element_factory_make("videoconvert", "vconvert");
  data.vsink = gst_element_factory_make("autovideosink", "vsink");

  data.pipeline = gst_pipeline_new("test-pipeline");

  if (!data.pipeline || !data.source || !data.convert || !data.sink || !data.vconvert || !data.vsink) {
    g_printerr("There is an element not created.\n");
    return -1;
  }

  gst_bin_add_many(GST_BIN(data.pipeline), data.source, data.convert, data.sink, data.vconvert, data.vsink, NULL);
  if (!gst_element_link(data.convert, data.sink)) {
    g_printerr("Element cannot be linked for audio.\n");
    gst_object_unref(data.pipeline);
    return -1;
  }
  if (!gst_element_link(data.vconvert, data.vsink)) {
    g_printerr("Element cannot be linked for video.\n");
    gst_object_unref(data.pipeline);
    return -1;
  }

  g_object_set(data.source, "uri", "https://www.freedesktop.org/software/gstreamer-sdk/data/media/sintel_trailer-480p.webm", NULL);

  g_signal_connect(data.source, "pad-added", G_CALLBACK(pad_added_handler), &data);

  ret = gst_element_set_state(data.pipeline, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    g_printerr("Unsable to set th epipeline to the playing state.\n");
    gst_object_unref(data.pipeline);
    return -1;
  }

  bus = gst_element_get_bus(data.pipeline);
  do {
    msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, GST_MESSAGE_STATE_CHANGED | GST_MESSAGE_ERROR | GST_MESSAGE_EOS);

    if (msg != NULL) {
      GError *err;
      gchar *debug_info;

      switch(GST_MESSAGE_TYPE (msg)) {
      case GST_MESSAGE_ERROR:
        gst_message_parse_error(msg, &err, &debug_info);
	g_printerr("Error from %s: %s\n", GST_OBJECT_NAME(msg->src), err->message);
	g_printerr("Debug info: %s\n", debug_info ? debug_info : "none");
	g_clear_error(&err);
	g_free(debug_info);
	terminate = TRUE;
	break;
      case GST_MESSAGE_EOS:
        g_print("EOS.\n");
	terminate = TRUE;
	break;
      case GST_MESSAGE_STATE_CHANGED:
        if (GST_MESSAGE_SRC(msg) == GST_OBJECT(data.pipeline)) {
	  GstState old_state, new_state, pending_state;
	  gst_message_parse_state_changed(msg, &old_state, &new_state, &pending_state);
	  g_print("Pipeline state %s --> %s\n", gst_element_state_get_name(old_state), gst_element_state_get_name(new_state));
	}
	break;
      default:
        g_printerr("Unexpected message.\n");
	break;
      }
      gst_message_unref(msg);
    }
  } while(!terminate);

  gst_object_unref(bus);
  gst_element_set_state(data.pipeline, GST_STATE_NULL);
  gst_object_unref(data.pipeline);
  return 0;
}

static void pad_added_handler(GstElement *src, GstPad *new_pad, CustomData *data) {
  GstPad *sink_pad = gst_element_get_static_pad(data->convert, "sink");
  GstPad *vsink_pad = gst_element_get_static_pad(data->vconvert, "sink");
  GstPadLinkReturn ret;
  GstCaps *new_pad_caps = NULL;
  GstStructure *new_pad_struct = NULL;
  const gchar *new_pad_type = NULL;

  g_print("Received new pad '%s' from '%s':\n", GST_PAD_NAME(new_pad), GST_ELEMENT_NAME(src));

  if (gst_pad_is_linked(sink_pad)) {
    g_print("  We are already linked. Ignoring\n");
    goto exit;
  }

  new_pad_caps = gst_pad_query_caps(new_pad, NULL);
  new_pad_struct = gst_caps_get_structure(new_pad_caps, 0);
  new_pad_type = gst_structure_get_name(new_pad_struct);
  if (g_str_has_prefix(new_pad_type, "audio/x-raw")) {
    ret = gst_pad_link(new_pad, sink_pad);
    if (GST_PAD_LINK_FAILED(ret)) {
      g_print("  Type is '%s' but link failed\n", new_pad_type);
    } else {
      g_print("  Link succeeded (type '%s')\n", new_pad_type);
    }
  } else if (g_str_has_prefix(new_pad_type, "video/x-raw")) {
    ret = gst_pad_link(new_pad, vsink_pad);
    if (GST_PAD_LINK_FAILED(ret)) {
      g_print("  Type is '%s' but link failed for video\n", new_pad_type);
    } else {
      g_print("  Link succeeded for video (type '%s')\n", new_pad_type);
    }
  } else {
    g_print("  It has type '%s' which is not raw audio. Ignoring.\n", new_pad_type);
    goto exit;
  }

exit:
  if (new_pad_caps != NULL)
    gst_caps_unref(new_pad_caps);
  gst_object_unref(sink_pad);
}
