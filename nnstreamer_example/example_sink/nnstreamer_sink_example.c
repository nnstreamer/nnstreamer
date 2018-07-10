/**
 * @file	nnstreamer_sink_example.c
 * @date	3 July 2018
 * @brief	Sample code for tensor sink plugin
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs.
 *
 * Simple example to init tensor sink element and get data.
 *
 * Run example :
 * ./nnstreamer_sink_example --gst-plugin-path=<nnstreamer plugin path>
 */

#include <gst/gst.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG TRUE
#endif

/**
 * @brief Macro for debug message.
 */
#define _print_log(...) if (DBG) g_message (__VA_ARGS__)

/**
 * @brief Macro to check error case.
 */
#define _check_cond_err(cond) \
  if (!(cond)) { \
    _print_log ("app failed! [line : %d]", __LINE__); \
    goto error; \
  }

/**
 * @brief Data structure for app.
 */
typedef struct
{
  GMainLoop *loop; /**< main event loop */
  GstElement *pipeline; /**< gst pipeline for test */
  GstBus *bus; /**< gst bus for test */
  GstElement *sink; /**< tensor sink element */

  guint received; /**< received buffer count */
} AppData;

/**
 * @brief Data for pipeline and result.
 */
static AppData g_app_data;

/**
 * @brief Free resources in app data.
 */
static void
_free_app_data (void)
{
  if (g_app_data.loop) {
    g_main_loop_unref (g_app_data.loop);
    g_app_data.loop = NULL;
  }

  if (g_app_data.bus) {
    gst_bus_remove_signal_watch (g_app_data.bus);
    gst_object_unref (g_app_data.bus);
    g_app_data.bus = NULL;
  }

  if (g_app_data.sink) {
    gst_object_unref (g_app_data.sink);
    g_app_data.sink = NULL;
  }

  if (g_app_data.pipeline) {
    gst_object_unref (g_app_data.pipeline);
    g_app_data.pipeline = NULL;
  }
}

/**
 * @brief Function to print error message.
 */
static void
_parse_err_message (GstMessage * message)
{
  gchar *debug;
  GError *error;

  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_ERROR:
      gst_message_parse_error (message, &error, &debug);
      break;

    case GST_MESSAGE_WARNING:
      gst_message_parse_warning (message, &error, &debug);
      break;

    default:
      return;
  }

  gst_object_default_error (GST_MESSAGE_SRC (message), error, debug);
  g_error_free (error);
  g_free (debug);
}

/**
 * @brief Callback for message.
 */
static void
_message_cb (GstBus * bus, GstMessage * message, gpointer user_data)
{
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_EOS:
      _print_log ("received eos message");
      g_main_loop_quit (g_app_data.loop);
      break;

    case GST_MESSAGE_ERROR:
      _print_log ("received error message");
      _parse_err_message (message);
      g_main_loop_quit (g_app_data.loop);
      break;

    case GST_MESSAGE_WARNING:
      _print_log ("received warning message");
      _parse_err_message (message);
      break;

    case GST_MESSAGE_STREAM_START:
      _print_log ("received start message");
      break;

    default:
      break;
  }
}

/**
 * @brief Callback for signal new-data.
 */
static void
_new_data_cb (GstElement * element, GstBuffer * buffer, gpointer user_data)
{
  g_app_data.received++;
  _print_log ("new data callback [%d]", g_app_data.received);

  /* example to get data */
  if (DBG) {
    GstMemory *mem;
    GstMapInfo info;
    guint i;
    guint num_mems;

    num_mems = gst_buffer_n_memory (buffer);
    for (i = 0; i < num_mems; i++) {
      mem = gst_buffer_peek_memory (buffer, i);

      if (gst_memory_map (mem, &info, GST_MAP_READ)) {
        /* check data (info.data, info.size) */
        _print_log ("received %zd", info.size);

        gst_memory_unmap (mem, &info);
      }
    }
  }

  /* example to get negotiated caps */
  if (DBG) {
    GstPad *sink_pad;
    GstCaps *caps;

    sink_pad = gst_element_get_static_pad (g_app_data.sink, "sink");

    if (sink_pad) {
      caps = gst_pad_get_current_caps (sink_pad);

      if (caps) {
        guint caps_size, i;

        caps_size = gst_caps_get_size (caps);
        _print_log ("caps size is %d", caps_size);

        for (i = 0; i < caps_size; i++) {
          GstStructure *structure = gst_caps_get_structure (caps, i);
          gchar *str = gst_structure_to_string (structure);

          _print_log ("[%d] %s", i, str);
          g_free (str);
        }

        gst_caps_unref (caps);
      }

      gst_object_unref (sink_pad);
    }
  }
}

/**
 * @brief Callback for signal stream-start.
 */
static void
_stream_start_cb (GstElement * element, GstBuffer * buffer, gpointer user_data)
{
  _print_log ("stream start callback");
}

/**
 * @brief Callback for signal eos.
 */
static void
_eos_cb (GstElement * element, GstBuffer * buffer, gpointer user_data)
{
  _print_log ("eos callback");
}

/**
 * @brief Main function.
 */
int
main (int argc, char **argv)
{
  const guint num_buffers = 100;
  const guint width = 640;
  const guint height = 480;

  gchar *str_pipeline;
  gulong handle_id;
  GstStateChangeReturn state_ret;

  /* init gstreamer */
  gst_init (&argc, &argv);

  /* init app variable */
  g_app_data.received = 0;

  /* main loop and pipeline */
  g_app_data.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_app_data.loop != NULL);

  /* 640x480 30fps for test */
  str_pipeline =
      g_strdup_printf
      ("videotestsrc num-buffers=%d ! video/x-raw,format=RGB,width=%d,height=%d ! "
      "tensor_converter ! tensor_sink name=tensor_sink",
      num_buffers, width, height);
  g_app_data.pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  _check_cond_err (g_app_data.pipeline != NULL);

  /* message callback */
  g_app_data.bus = gst_element_get_bus (g_app_data.pipeline);
  _check_cond_err (g_app_data.bus != NULL);

  gst_bus_add_signal_watch (g_app_data.bus);
  handle_id = g_signal_connect (g_app_data.bus, "message",
      (GCallback) _message_cb, NULL);
  _check_cond_err (handle_id > 0);

  /* get tensor sink element using name */
  g_app_data.sink =
      gst_bin_get_by_name (GST_BIN (g_app_data.pipeline), "tensor_sink");
  _check_cond_err (g_app_data.sink != NULL);

  if (DBG) {
    /* print logs, default TRUE */
    g_object_set (g_app_data.sink, "silent", (gboolean) FALSE, NULL);
  }

  /* enable emit-signal, default TRUE */
  g_object_set (g_app_data.sink, "emit-signal", (gboolean) TRUE, NULL);

  /* tensor sink signal : new data callback */
  handle_id = g_signal_connect (g_app_data.sink, "new-data",
      (GCallback) _new_data_cb, NULL);
  _check_cond_err (handle_id > 0);

  /* tensor sink signal : stream-start callback, optional */
  handle_id = g_signal_connect (g_app_data.sink, "stream-start",
      (GCallback) _stream_start_cb, NULL);
  _check_cond_err (handle_id > 0);

  /* tensor sink signal : eos callback, optional */
  handle_id = g_signal_connect (g_app_data.sink, "eos",
      (GCallback) _eos_cb, NULL);
  _check_cond_err (handle_id > 0);

  /* start pipeline */
  state_ret = gst_element_set_state (g_app_data.pipeline, GST_STATE_PLAYING);
  _check_cond_err (state_ret != GST_STATE_CHANGE_FAILURE);

  g_main_loop_run (g_app_data.loop);

  /* quit when received eos message */
  state_ret = gst_element_set_state (g_app_data.pipeline, GST_STATE_NULL);
  _check_cond_err (state_ret != GST_STATE_CHANGE_FAILURE);

  _print_log ("total received %d", g_app_data.received);

error:
  _free_app_data ();
  return 0;
}
