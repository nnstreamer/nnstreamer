/**
 * @file    tensor_filter_reload_test.c
 * @data    19 Dec 2019
 * @brief   test case to test a filter's model reload
 * @see     https://github.com/nnsuite/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except NYI.
 */

#include <string.h>
#include <stdlib.h>
#include <gst/gst.h>
#include <tensor_common.h>

#define print_log(...) if (!silent) g_message (__VA_ARGS__)
#define make_gst_element(element) do{\
  element = gst_element_factory_make(#element, #element);\
  if (!element) {\
    g_printerr ("element %s could not be created.\n", #element);\
    return_val = -1;\
    goto out_unref;\
  }\
} while (0);

#define IMAGE_FPS (25)
#define EVENT_INTERVAL (1000)
#define EVENT_TIMEOUT (10000)

static gboolean silent = TRUE;
static gchar *input_img_path = NULL;
static gchar *first_model_path = NULL;
static gchar *second_model_path = NULL;

static GMainLoop *loop = NULL;
static gint return_val = 0;

/**
 * @brief Bus callback function
 */
static gboolean
bus_callback (GstBus * bus, GstMessage * message, gpointer data)
{
  print_log ("Got %s message\n", GST_MESSAGE_TYPE_NAME (message));

  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_ERROR:{
      GError *err;
      gchar *debug;

      gst_message_parse_error (message, &err, &debug);
      print_log ("Error: %s\n", err->message);
      g_error_free (err);
      g_free (debug);

      g_main_loop_quit (loop);
      break;
    }
    case GST_MESSAGE_EOS:
      g_main_loop_quit (loop);
      break;
    default:
      break;
  }

  return TRUE;
}

/**
 * @brief Find the index with a maximum value
 */
static gint
get_maximum_index (guint8 *data, gsize size)
{
  gint idx, max_idx = 0;
  guint8 maximum = 0;

  for (idx = 0; idx < size; ++idx) {
    if (data[idx] > maximum) {
      maximum = data[idx];
      max_idx = idx;
    }
  }

  return max_idx;
}

/**
 * @brief Signal to handle new output data
 */
static GstFlowReturn
check_output (GstElement *sink, void *data __attribute__((unused)))
{
  static gint prev_index = -1;

  GstSample *sample;
  GstBuffer *buffer;
  GstMapInfo info;
  gint index;

  g_signal_emit_by_name (sink, "pull-sample", &sample);
  if (!sample)
    return GST_FLOW_ERROR;

  buffer = gst_sample_get_buffer (sample);
  if (!buffer)
    return GST_FLOW_ERROR;

  if (!gst_buffer_map (buffer, &info, GST_MAP_READ))
    return GST_FLOW_ERROR;

  /**
   * find the maximum entry; this value should be the same with
   * the previous one even if a model is switched to the other one
   */
  index = get_maximum_index (info.data, info.size);
  if (prev_index != -1 && prev_index != index) {
    g_critical ("Output is different! %d vs %d\n", prev_index, index);
    return_val = -1;
  }

  prev_index = index;

  gst_buffer_unmap (buffer, &info);
  gst_sample_unref (sample);

  return GST_FLOW_OK;
}

/**
 * @brief Reload a tensor filter's model (v1 <-> v2)
 */
static gboolean
reload_model (GstElement *tensor_filter)
{
  static gboolean is_first = TRUE;
  const gchar *model_path;

  if (!tensor_filter)
    return FALSE;

  model_path = is_first ? second_model_path : first_model_path;

  g_object_set (G_OBJECT (tensor_filter), "model", model_path, NULL);

  print_log ("Model %s is just reloaded\n", model_path);

  is_first = !is_first;

  /* repeat if it's playing*/
  return (GST_STATE (GST_ELEMENT (tensor_filter)) == GST_STATE_PLAYING);
}

/**
 * @brief Stop the main loop callback
 */
static gboolean
stop_loop (GMainLoop *loop)
{
  if (!loop)
    return FALSE;

  g_main_loop_quit (loop);

  print_log ("Now stop the loop\n");

  /* stop */
  return FALSE;
}

/**
 * @brief Main function to evalute tensor_filter's model reload functionality
 * @note feed the same input image to the tensor filter; So, even if a detection model
 * is updated (mobilenet v1 <-> v2), the output should be the same for all frames.
 */
int
main (int argc, char *argv[])
{
  GMainLoop *loop;
  GstElement *pipeline;
  GstElement *filesrc, *pngdec, *videoscale, *imagefreeze, *videoconvert;
  GstElement *capsfilter, *tensor_converter, *tensor_filter, *appsink;
  GstCaps *caps;
  GstBus *bus;

  GError *error = NULL;
  GOptionContext *opt_context;
  const GOptionEntry opt_entries[] = {
    {"input_img", 'i', G_OPTION_FLAG_NONE, G_OPTION_ARG_STRING, &input_img_path,
      "The path of input image file",
      "e.g., data/orange.png"},
    {"first_model", 0, G_OPTION_FLAG_NONE, G_OPTION_ARG_STRING, &first_model_path,
      "The path of first model file",
      "e.g., models/mobilenet_v1_1.0_224_quant.tflite"},
    {"second_model", 0, G_OPTION_FLAG_NONE, G_OPTION_ARG_STRING, &second_model_path,
      "The path of second model file",
      "e.g., models/mobilenet_v2_1.0_224_quant.tflite"},
    {"silent", 's', G_OPTION_FLAG_NONE, G_OPTION_FLAG_NONE, &silent,
      "Hide debug message", "TRUE (default)"},
    {NULL}
  };

  /* parse options */
  opt_context = g_option_context_new (NULL);
  g_option_context_add_main_entries (opt_context, opt_entries, NULL);

  if (!g_option_context_parse (opt_context, &argc, &argv, &error)) {
    g_printerr ("Option parsing failed: %s\n", error->message);
    g_error_free (error);
    return -1;
  }

  g_option_context_free (opt_context);

  if (!(input_img_path && first_model_path && second_model_path)) {
    g_printerr ("No valid arguments provided\n");
    return -1;
  }

  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* make pipeline & elements */
  pipeline = gst_pipeline_new ("Pipeline with a model-updatable tensor filter");
  make_gst_element (filesrc);
  make_gst_element (pngdec);
  make_gst_element (videoscale);
  make_gst_element (imagefreeze); /* feed the same input image */
  make_gst_element (videoconvert);
  make_gst_element (capsfilter);
  make_gst_element (tensor_converter);
  make_gst_element (tensor_filter);
  make_gst_element (appsink); /* output is verified in appsink callback */

  /* set arguments of each element */
  g_object_set (G_OBJECT (filesrc), "location", input_img_path, NULL);

  caps = gst_caps_new_simple (
      "video/x-raw",
      "format", G_TYPE_STRING, "RGB",
      "framerate", GST_TYPE_FRACTION, IMAGE_FPS, 1,
      NULL);
  g_object_set (G_OBJECT (capsfilter), "caps", caps, NULL);
  gst_caps_unref (caps);

  g_object_set (G_OBJECT (tensor_filter), "framework", "tensorflow-lite", NULL);
  g_object_set (G_OBJECT (tensor_filter), "model", first_model_path, NULL);
  g_object_set (G_OBJECT (tensor_filter), "is-updatable", TRUE, NULL);

  g_object_set (G_OBJECT (appsink), "emit-signals", TRUE, NULL);
  g_signal_connect (appsink, "new-sample", G_CALLBACK (check_output), NULL);

  /* link elements to the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), filesrc,
      pngdec, videoscale, imagefreeze, videoconvert, capsfilter,
      tensor_converter, tensor_filter, appsink, NULL);
  gst_element_link_many (filesrc,
      pngdec, videoscale, imagefreeze, videoconvert, capsfilter,
      tensor_converter, tensor_filter, appsink, NULL);

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  gst_bus_add_watch (bus, bus_callback, NULL);
  gst_object_unref (bus);

  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* add timeout events */
  g_timeout_add (EVENT_INTERVAL, (GSourceFunc) reload_model, tensor_filter);
  g_timeout_add (EVENT_TIMEOUT, (GSourceFunc) stop_loop, loop);

  g_main_loop_run (loop);

  gst_element_set_state (pipeline, GST_STATE_NULL);

out_unref:
  gst_object_unref (GST_OBJECT (pipeline));
  g_main_loop_unref (loop);

  return return_val;
}
