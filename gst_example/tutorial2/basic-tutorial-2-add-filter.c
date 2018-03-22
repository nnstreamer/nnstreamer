#include <gst/gst.h>

int main(int argc, char *argv[]) {
  GstElement *pipeline, *source, *sink, *filter;
  GstBus *bus;
  GstMessage *msg;
  GstStateChangeReturn ret;

  gst_init(&argc, &argv);

  source = gst_element_factory_make("videotestsrc", "source");
  sink = gst_element_factory_make("autovideosink", "sink");
  filter = gst_element_factory_make("vertigotv", "filter1");

  pipeline = gst_pipeline_new("test-pipeline");

  if (!pipeline || !source || !sink || !filter) {
    g_printerr("Cannot create elements.\n");
    return -1;
  }

  gst_bin_add_many(GST_BIN(pipeline), source, sink, filter, NULL);
  if (gst_element_link(source, filter) != TRUE) {
    g_printerr("Cannot linke source and filter\n");
    gst_object_unref(pipeline);
    return -1;
  }
  if (gst_element_link(filter, sink) != TRUE) {
    g_printerr("Cannot linke filter and sink\n");
    gst_object_unref(pipeline);
    return -1;
  }

  g_object_set(source, "pattern", 0, NULL);

  ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    g_printerr("Cannot setup pipeline as playing");
    gst_object_unref(pipeline);
    return -1;
  }

  bus = gst_element_get_bus(pipeline);
  msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, GST_MESSAGE_ERROR | GST_MESSAGE_EOS);


  if (msg != NULL) {
    GError *err;
    gchar *debug_info;

    switch(GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_ERROR:
      gst_message_parse_error(msg, &err, &debug_info);
      g_printerr("Error received from element %s: %s\n", GST_OBJECT_NAME(msg->src), err->message);
      g_printerr("Debugging information: %s\n", debug_info ? debug_info : "none");
      g_clear_error(&err);
      g_free(debug_info);
      break;
    case GST_MESSAGE_EOS:
      g_print("End Of Stream.\n");
      break;
    default:
      g_printerr("Undefined gst message.\n");
    }
  }

  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  return 0;
}
