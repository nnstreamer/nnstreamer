/**
 * @file	nnstreamer_example_cam.c
 * @date	10 July 2018
 * @brief	Tensor stream example
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs.
 *
 * Sample code for tensor stream, this app shows video frame.
 *
 * Run example :
 * ./nnstreamer_example_cam --gst-plugin-path=<nnstreamer plugin path>
 */

#include <fcntl.h>
#include <unistd.h>

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
 * @brief Check cam device connected.
 */
static gboolean
_check_cam_device (const gchar * cam_dev)
{
  int fd;

  if ((fd = open (cam_dev, O_RDONLY)) < 0) {
    _print_log ("cannot detect cam, check your device and start again.");
    return FALSE;
  }

  close (fd);
  return TRUE;
}

/**
 * @brief Set window title.
 * @param element pointer to GstXImageSink
 * @param title window title
 */
static void
_set_window_title (GstElement * element, const gchar * title)
{
  GstTagList *tags;
  GstPad *sink_pad;

  sink_pad = gst_element_get_static_pad (element, "sink");

  if (sink_pad) {
    tags = gst_tag_list_new (GST_TAG_TITLE, title, NULL);
    gst_pad_send_event (sink_pad, gst_event_new_tag (tags));
    gst_object_unref (sink_pad);
  }
}

/**
 * @brief Main function.
 */
int
main (int argc, char **argv)
{
  const gchar cam_dev[] = "/dev/video0";
  const guint width = 640;
  const guint height = 480;

  gchar *str_pipeline;
  gulong handle_id;
  GstElement *element;

  _print_log ("start app..");

  /* check cam */
  _check_cond_err (_check_cam_device (cam_dev));

  /* init gstreamer */
  gst_init (&argc, &argv);

  /* main loop */
  g_app.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_app.loop != NULL);

  /* init data pipeline */
  str_pipeline =
      g_strdup_printf
      ("v4l2src name=cam_src ! "
      "video/x-raw,width=%d,height=%d,format=RGB,framerate=30/1 ! tee name=t_raw "
      "videomixer name=mix "
      "sink_0::xpos=0 sink_0::ypos=0 sink_0::zorder=0 "
      "sink_1::xpos=0 sink_1::ypos=0 sink_1::zorder=1 sink_1::alpha=0.7 ! "
      "videoconvert ! xvimagesink name=img_mixed "
      "t_raw. ! queue ! mix.sink_0 "
      "t_raw. ! queue ! tensor_converter ! tensordec ! videoscale ! video/x-raw,width=%d,height=%d ! mix.sink_1 "
      "t_raw. ! queue ! videoconvert ! xvimagesink name=img_origin",
      width, height, width / 2, height / 2);
  g_app.pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  _check_cond_err (g_app.pipeline != NULL);

  /* data message callback */
  g_app.bus = gst_element_get_bus (g_app.pipeline);
  _check_cond_err (g_app.bus != NULL);

  gst_bus_add_signal_watch (g_app.bus);
  handle_id = g_signal_connect (g_app.bus, "message",
      (GCallback) _message_cb, NULL);
  _check_cond_err (handle_id > 0);

  /* start pipeline */
  gst_element_set_state (g_app.pipeline, GST_STATE_PLAYING);

  /* set window title */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "img_mixed");
  _set_window_title (element, "Mixed");
  gst_object_unref (element);

  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "img_origin");
  _set_window_title (element, "Original");
  gst_object_unref (element);

  /* run main loop */
  g_main_loop_run (g_app.loop);

  /* cam source element */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "cam_src");

  /* quit when received eos or error message */
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
