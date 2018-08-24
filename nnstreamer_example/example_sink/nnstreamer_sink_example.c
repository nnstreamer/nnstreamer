/**
 * @file	nnstreamer_sink_example.c
 * @date	3 July 2018
 * @brief	Sample code for tensor sink plugin
 * @see		https://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs.
 *
 * Simple example to init tensor sink element and get data.
 *
 * Run example :
 * Before running this example, GST_PLUGIN_PATH should be updated for nnstreamer plug-in.
 * $ export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:<nnstreamer plugin path>
 * $ ./nnstreamer_sink_example
 */

#include <stdlib.h>
#include <string.h>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

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
 * @brief Test media type.
 */
typedef enum
{
  TEST_TYPE_VIDEO,
  TEST_TYPE_AUDIO,
  TEST_TYPE_TEXT
} test_media_type;

/**
 * @brief Data structure for app.
 */
typedef struct
{
  GMainLoop *loop; /**< main event loop */
  GstElement *pipeline; /**< gst pipeline for test */
  GstBus *bus; /**< gst bus for test */

  guint received; /**< received buffer count */
  test_media_type media_type; /**< test media type */
} AppData;

/**
 * @brief Data for pipeline and result.
 */
static AppData g_app;

/**
 * @brief Free resources in app data.
 */
static void
_free_app_data (void)
{
  if (g_app.loop) {
    if (g_main_loop_is_running (g_app.loop)) {
      g_main_loop_quit (g_app.loop);
    }

    g_main_loop_unref (g_app.loop);
    g_app.loop = NULL;
  }

  if (g_app.bus) {
    gst_bus_remove_signal_watch (g_app.bus);
    gst_object_unref (g_app.bus);
    g_app.bus = NULL;
  }

  if (g_app.pipeline) {
    gst_object_unref (g_app.pipeline);
    g_app.pipeline = NULL;
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

  g_return_if_fail (message != NULL);

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
 * @brief Function to print caps.
 */
static void
_parse_caps (GstCaps * caps)
{
  guint caps_size, i;

  g_return_if_fail (caps != NULL);

  caps_size = gst_caps_get_size (caps);

  for (i = 0; i < caps_size; i++) {
    GstStructure *structure = gst_caps_get_structure (caps, i);
    gchar *str = gst_structure_to_string (structure);

    _print_log ("[%d] %s", i, str);
    g_free (str);
  }
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
      g_main_loop_quit (g_app.loop);
      break;

    case GST_MESSAGE_ERROR:
      _print_log ("received error message");
      _parse_err_message (message);
      g_main_loop_quit (g_app.loop);
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
  g_app.received++;
  if (g_app.received % 150 == 0) {
    _print_log ("receiving new data [%d]", g_app.received);
  }

  /** example to get data */
  {
    GstMemory *mem;
    GstMapInfo info;
    guint i;
    guint num_mems;

    num_mems = gst_buffer_n_memory (buffer);
    for (i = 0; i < num_mems; i++) {
      mem = gst_buffer_peek_memory (buffer, i);

      if (gst_memory_map (mem, &info, GST_MAP_READ)) {
        /** check data (info.data, info.size) */
        if (g_app.media_type == TEST_TYPE_TEXT) {
          _print_log ("received %zd [%s]", info.size, (gchar *) info.data);
        } else {
          _print_log ("received %zd", info.size);
        }

        gst_memory_unmap (mem, &info);
      }
    }
  }

  /** example to get caps */
  {
    GstPad *sink_pad;
    GstCaps *caps;

    sink_pad = gst_element_get_static_pad (element, "sink");

    if (sink_pad) {
      /** negotiated */
      caps = gst_pad_get_current_caps (sink_pad);

      if (caps) {
        _parse_caps (caps);
        gst_caps_unref (caps);
      }

      /** template */
      caps = gst_pad_get_pad_template_caps (sink_pad);

      if (caps) {
        _parse_caps (caps);
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
 * @brief Timer callback to push buffer.
 * @return True to ensure the timer continues
 */
static gboolean
_test_src_timer_cb (gpointer user_data)
{
  GstElement *appsrc;
  GstBuffer *buf;
  GstMapInfo info;
  guint buffer_index;

  buffer_index = g_app.received + 1;
  appsrc = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "appsrc");

  switch (g_app.media_type) {
    case TEST_TYPE_TEXT:
    {
      gchar *text_data;

      /** send 20 text buffers */
      if (buffer_index > 20) {
        if (gst_app_src_end_of_stream (GST_APP_SRC (appsrc)) != GST_FLOW_OK) {
          _print_log ("failed to indicate eos");
        }
        return FALSE;
      }

      text_data = g_strdup_printf ("example for text [%d/20]", buffer_index);

      buf = gst_buffer_new_allocate (NULL, strlen (text_data) + 1, NULL);
      gst_buffer_map (buf, &info, GST_MAP_WRITE);

      strcpy ((gchar *) info.data, text_data);

      gst_buffer_unmap (buf, &info);

      GST_BUFFER_PTS (buf) = 20 * GST_MSECOND * buffer_index;
      GST_BUFFER_DTS (buf) = GST_BUFFER_PTS (buf);

      if (gst_app_src_push_buffer (GST_APP_SRC (appsrc), buf) != GST_FLOW_OK) {
        _print_log ("failed to push buffer [%d]", buffer_index);
      }

      g_free (text_data);
      break;
    }
    default:
      /** nothing to do */
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Test pipeline for given type.
 * @param type test media type
 */
static gchar *
_test_pipeline (test_media_type type)
{
  gchar *str_pipeline;

  switch (type) {
    case TEST_TYPE_VIDEO:
      /** video 640x480 30fps 100 buffers */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=100 ! video/x-raw,format=RGB,width=640,height=480 ! "
          "tensor_converter ! tensor_sink name=tensor_sink");
      break;

    case TEST_TYPE_AUDIO:
      /** audio sample rate 16000 (16 bits, signed, little endian) 30 buffers */
      str_pipeline =
          g_strdup_printf
          ("audiotestsrc num-buffers=30 ! audio/x-raw,format=S16LE,rate=16000 ! "
          "tensor_converter ! tensor_sink name=tensor_sink");
      break;

    case TEST_TYPE_TEXT:
      /** text 20 buffers */
      str_pipeline =
          g_strdup_printf
          ("appsrc name=appsrc caps=text/x-raw,format=utf8 ! "
          "tensor_converter ! tensor_sink name=tensor_sink");
      break;

    default:
      return NULL;
  }

  return str_pipeline;
}

/**
 * @brief Main function.
 */
int
main (int argc, char **argv)
{
  test_media_type test_type = TEST_TYPE_VIDEO;
  gchar *str_pipeline;
  gulong handle_id;
  GstStateChangeReturn state_ret;
  GstElement *element;

  if (argc > 1) {
    test_type = atoi (argv[1]);
  }

  /** init gstreamer */
  gst_init (&argc, &argv);

  /** init app variable */
  g_app.received = 0;
  g_app.media_type = test_type;

  /** main loop and pipeline */
  g_app.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_app.loop != NULL);

  str_pipeline = _test_pipeline (test_type);
  g_app.pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  _check_cond_err (g_app.pipeline != NULL);

  /** message callback */
  g_app.bus = gst_element_get_bus (g_app.pipeline);
  _check_cond_err (g_app.bus != NULL);

  gst_bus_add_signal_watch (g_app.bus);
  handle_id = g_signal_connect (g_app.bus, "message",
      (GCallback) _message_cb, NULL);
  _check_cond_err (handle_id > 0);

  /** get tensor sink element using name */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "tensor_sink");
  _check_cond_err (element != NULL);

  if (DBG) {
    /** print logs, default TRUE */
    g_object_set (element, "silent", (gboolean) FALSE, NULL);
  }

  /** enable emit-signal, default TRUE */
  g_object_set (element, "emit-signal", (gboolean) TRUE, NULL);

  /** tensor sink signal : new data callback */
  handle_id = g_signal_connect (element, "new-data",
      (GCallback) _new_data_cb, NULL);
  _check_cond_err (handle_id > 0);

  /** tensor sink signal : stream-start callback, optional */
  handle_id = g_signal_connect (element, "stream-start",
      (GCallback) _stream_start_cb, NULL);
  _check_cond_err (handle_id > 0);

  /** tensor sink signal : eos callback, optional */
  handle_id = g_signal_connect (element, "eos", (GCallback) _eos_cb, NULL);
  _check_cond_err (handle_id > 0);

  gst_object_unref (element);

  /** start pipeline */
  state_ret = gst_element_set_state (g_app.pipeline, GST_STATE_PLAYING);
  _check_cond_err (state_ret != GST_STATE_CHANGE_FAILURE);

  _check_cond_err (g_timeout_add (20, _test_src_timer_cb, NULL) > 0);

  /** run main loop */
  g_main_loop_run (g_app.loop);
  /** quit when received eos message */

  state_ret = gst_element_set_state (g_app.pipeline, GST_STATE_NULL);
  _check_cond_err (state_ret != GST_STATE_CHANGE_FAILURE);

  _print_log ("total received %d", g_app.received);

error:
  _free_app_data ();
  return 0;
}
