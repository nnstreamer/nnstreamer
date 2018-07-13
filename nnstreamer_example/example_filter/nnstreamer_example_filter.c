/**
 * @file	nnstreamer_example_filter.c
 * @date	13 July 2018
 * @brief	Tensor stream example with filter
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs.
 *
 * Application for tensor stream.
 *
 * Run example :
 * ./nnstreamer_example_filter --gst-plugin-path=<nnstreamer plugin path>
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

  gboolean running; /**< true when app is running */
  guint received; /**< received buffer count */
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
 * @brief Callback for tensor sink signal.
 */
static void
_new_data_cb (GstElement * element, GstBuffer * buffer, gpointer user_data)
{
  g_app.received++;
  if (g_app.received % 150 == 0) {
    _print_log ("receiving new data [%d]", g_app.received);
  }

  /** @todo prepare demo, update textoverlay. */
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

  g_return_if_fail (element != NULL);

  sink_pad = gst_element_get_static_pad (element, "sink");

  if (sink_pad) {
    tags = gst_tag_list_new (GST_TAG_TITLE, title, NULL);
    gst_pad_send_event (sink_pad, gst_event_new_tag (tags));
    gst_object_unref (sink_pad);
  }
}

/**
 * @brief Callback for textoverlay.
 */
static gboolean
_textoverlay_tensor_res_cb (gpointer user_data)
{
  if (g_app.running) {
    GstElement *textoverlay;
    gchar *tensor_res;
    GstClockTime now = gst_util_get_timestamp ();

    textoverlay = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "tensor_res");

    /** @todo prepare demo, update textoverlay. */
    tensor_res = g_strdup_printf ("%u:%02u:%02u",
        (guint) (now / (GST_SECOND * 60 * 60)),
        (guint) ((now / (GST_SECOND * 60)) % 60),
        (guint) ((now / GST_SECOND) % 60));

    g_object_set (textoverlay, "text", tensor_res, NULL);

    g_free (tensor_res);
    gst_object_unref (textoverlay);
  }

  return TRUE;
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
  guint timer_id = 0;
  GstElement *element;

  _print_log ("start app..");

  /** init app variable */
  g_app.running = FALSE;
  g_app.received = 0;

  /** check cam */
  _check_cond_err (_check_cam_device (cam_dev));

  /** init gstreamer */
  gst_init (&argc, &argv);

  /** main loop */
  g_app.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_app.loop != NULL);

  /** init data pipeline */
  /** @todo prepare demo, add tensor filter. */
  str_pipeline =
      g_strdup_printf
      ("v4l2src name=cam_src ! "
      "video/x-raw,width=%d,height=%d,format=RGB,framerate=30/1 ! tee name=t_raw "
      "t_raw. ! queue ! textoverlay name=tensor_res font-desc=\"Sans, 24\" ! videoconvert ! xvimagesink name=img_tensor "
      "t_raw. ! queue ! tensor_converter ! tensor_sink name=tensor_sink "
      "t_raw. ! queue ! videoconvert ! xvimagesink name=img_origin",
      width, height);
  g_app.pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  _check_cond_err (g_app.pipeline != NULL);

  /** data message callback */
  g_app.bus = gst_element_get_bus (g_app.pipeline);
  _check_cond_err (g_app.bus != NULL);

  gst_bus_add_signal_watch (g_app.bus);
  handle_id = g_signal_connect (g_app.bus, "message",
      (GCallback) _message_cb, NULL);
  _check_cond_err (handle_id > 0);

  /** tensor sink signal : new data callback */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "tensor_sink");
  handle_id = g_signal_connect (element, "new-data",
      (GCallback) _new_data_cb, NULL);
  gst_object_unref (element);
  _check_cond_err (handle_id > 0);

  /** timer to update result */
  timer_id = g_timeout_add (500, _textoverlay_tensor_res_cb, NULL);
  _check_cond_err (timer_id > 0);

  /** start pipeline */
  gst_element_set_state (g_app.pipeline, GST_STATE_PLAYING);

  g_app.running = TRUE;

  /** set window title */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "img_tensor");
  _set_window_title (element, "Tensor");
  gst_object_unref (element);

  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "img_origin");
  _set_window_title (element, "Original");
  gst_object_unref (element);

  /** run main loop */
  g_main_loop_run (g_app.loop);
  /** quit when received eos or error message */
  g_app.running = FALSE;

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

  if (timer_id > 0) {
    g_source_remove (timer_id);
  }

  _free_app_data ();
  return 0;
}
