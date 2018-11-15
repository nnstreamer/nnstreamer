/**
 * @file	nnstreamer_example_cam.c
 * @date	10 July 2018
 * @brief	Tensor stream example
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs.
 *
 * Sample code for tensor stream, this app shows video frame.
 *
 * Run example :
 * Before running this example, GST_PLUGIN_PATH should be updated for nnstreamer plug-in.
 * $ export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:<nnstreamer plugin path>
 * $ ./nnstreamer_example_cam
 */

#include <fcntl.h>
#include <unistd.h>

#include <gst/gst.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
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
  GstElement *pipeline; /**< gst pipeline for data stream */
  GstBus *bus; /**< gst bus for data pipeline */
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
 * @brief Set window title.
 * @param name GstXImageSink element name
 * @param title window title
 */
static void
_set_window_title (const gchar * name, const gchar * title)
{
  GstTagList *tags;
  GstPad *sink_pad;
  GstElement *element;

  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), name);

  g_return_if_fail (element != NULL);

  sink_pad = gst_element_get_static_pad (element, "sink");

  if (sink_pad) {
    tags = gst_tag_list_new (GST_TAG_TITLE, title, NULL);
    gst_pad_send_event (sink_pad, gst_event_new_tag (tags));
    gst_object_unref (sink_pad);
  }

  gst_object_unref (element);
}

/**
 * @brief Main function.
 */
int
main (int argc, char **argv)
{
  const guint width = 640;
  const guint height = 480;

  gchar *str_pipeline;
  gulong handle_id;
  GstElement *element;

  _print_log ("start app..");

  /** init gstreamer */
  gst_init (&argc, &argv);

  /** main loop */
  g_app.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_app.loop != NULL);

  /** init pipeline */
  str_pipeline =
      g_strdup_printf
      ("v4l2src name=cam_src ! videoscale ! "
      "video/x-raw,width=%d,height=%d,format=RGB ! tee name=t_raw "
      "videomixer name=mix "
      "sink_0::xpos=0 sink_0::ypos=0 sink_0::zorder=0 "
      "sink_1::xpos=0 sink_1::ypos=0 sink_1::zorder=1 sink_1::alpha=0.7 ! "
      "videoconvert ! ximagesink name=img_mixed "
      "t_raw. ! queue ! mix.sink_0 "
      "t_raw. ! queue ! tensor_converter ! tensor_decoder mode=direct_video ! videoscale ! video/x-raw,width=%d,height=%d ! mix.sink_1 "
      "t_raw. ! queue ! videoconvert ! ximagesink name=img_origin",
      width, height, width / 2, height / 2);
  g_app.pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  _check_cond_err (g_app.pipeline != NULL);

  /** bus and message callback */
  g_app.bus = gst_element_get_bus (g_app.pipeline);
  _check_cond_err (g_app.bus != NULL);

  gst_bus_add_signal_watch (g_app.bus);
  handle_id = g_signal_connect (g_app.bus, "message",
      (GCallback) _message_cb, NULL);
  _check_cond_err (handle_id > 0);

  /** start pipeline */
  gst_element_set_state (g_app.pipeline, GST_STATE_PLAYING);

  /** set window title */
  _set_window_title ("img_mixed", "Mixed");
  _set_window_title ("img_origin", "Original");

  /** run main loop */
  g_main_loop_run (g_app.loop);
  /** quit when received eos or error message */

  /** cam source element */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "cam_src");

  gst_element_set_state (element, GST_STATE_READY);
  gst_element_set_state (g_app.pipeline, GST_STATE_READY);

  g_usleep (200 * 1000);

  gst_element_set_state (element, GST_STATE_NULL);
  gst_element_set_state (g_app.pipeline, GST_STATE_NULL);

  g_usleep (200 * 1000);
  gst_object_unref (element);

error:
  _print_log ("close app..");
  _free_app_data ();
  return 0;
}
