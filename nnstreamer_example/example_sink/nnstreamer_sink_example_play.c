/**
 * @file	nnstreamer_sink_example_play.c
 * @date	5 July 2018
 * @brief	Sample code for tensor sink plugin
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs.
 *
 * This sample app shows video frame using two pipelines.
 *
 * [1st pipeline : videotestsrc-tensor_converter-tensor_sink]
 * push buffer to appsrc
 * [2nd pipeline : appsrc-tensordec-videoconvert-xvimagesink]
 *
 * Run example :
 * ./nnstreamer_sink_example_play --gst-plugin-path=<nnstreamer plugin path>
 */

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
 * @brief Data structure for app.
 */
typedef struct
{
  GMainLoop *loop; /**< main event loop */
  GstElement *data_pipeline; /**< gst pipeline for data stream */
  GstBus *data_bus; /**< gst bus for data pipeline */
  GstElement *player_pipeline; /**< gst pipeline for player */
  GstBus *player_bus; /**< gst bus for player pipeline */

  gboolean set_caps; /**< caps passed to player pipeline */
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

  if (g_app.data_bus) {
    gst_bus_remove_signal_watch (g_app.data_bus);
    gst_object_unref (g_app.data_bus);
    g_app.data_bus = NULL;
  }

  if (g_app.player_bus) {
    gst_bus_remove_signal_watch (g_app.player_bus);
    gst_object_unref (g_app.player_bus);
    g_app.player_bus = NULL;
  }

  if (g_app.data_pipeline) {
    gst_object_unref (g_app.data_pipeline);
    g_app.data_pipeline = NULL;
  }

  if (g_app.player_pipeline) {
    gst_object_unref (g_app.player_pipeline);
    g_app.player_pipeline = NULL;
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
_data_message_cb (GstBus * bus, GstMessage * message, gpointer user_data)
{
  GstElement *player_src;

  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_EOS:
      _print_log ("[data] received eos message");
      player_src = gst_bin_get_by_name (GST_BIN (g_app.player_pipeline),
          "player_src");
      gst_app_src_end_of_stream (GST_APP_SRC (player_src));
      gst_object_unref (player_src);
      break;

    case GST_MESSAGE_ERROR:
      _print_log ("[data] received error message");
      _parse_err_message (message);
      g_main_loop_quit (g_app.loop);
      break;

    case GST_MESSAGE_WARNING:
      _print_log ("[data] received warning message");
      _parse_err_message (message);
      break;

    case GST_MESSAGE_STREAM_START:
      _print_log ("[data] received start message");
      break;

    default:
      break;
  }
}

/**
 * @brief Callback for message.
 */
static void
_player_message_cb (GstBus * bus, GstMessage * message, gpointer user_data)
{
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_EOS:
      _print_log ("[player] received eos message");
      g_main_loop_quit (g_app.loop);
      break;

    case GST_MESSAGE_ERROR:
      _print_log ("[player] received error message");
      _parse_err_message (message);
      g_main_loop_quit (g_app.loop);
      break;

    case GST_MESSAGE_WARNING:
      _print_log ("[player] received warning message");
      _parse_err_message (message);
      break;

    case GST_MESSAGE_STREAM_START:
      _print_log ("[player] received start message");
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
  GstElement *player_src;

  g_app.received++;
  if (g_app.received % 150 == 0) {
    _print_log ("receiving new data [%d]", g_app.received);
  }

  player_src =
      gst_bin_get_by_name (GST_BIN (g_app.player_pipeline), "player_src");

  if (!g_app.set_caps) {
    GstPad *sink_pad;
    GstCaps *caps;

    sink_pad = gst_element_get_static_pad (element, "sink");

    if (sink_pad) {
      caps = gst_pad_get_current_caps (sink_pad);

      if (caps) {
        gst_app_src_set_caps (GST_APP_SRC (player_src), caps);

        gst_caps_unref (caps);
        g_app.set_caps = TRUE;
      }

      gst_object_unref (sink_pad);
    }
  }

  gst_app_src_push_buffer (GST_APP_SRC (player_src), gst_buffer_copy (buffer));
  gst_object_unref (player_src);
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

  /** init gstreamer */
  gst_init (&argc, &argv);

  /** init app variable */
  g_app.set_caps = FALSE;
  g_app.received = 0;

  /** main loop and pipeline */
  g_app.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_app.loop != NULL);

  str_pipeline =
      g_strdup_printf
      ("videotestsrc is-live=TRUE ! video/x-raw,format=RGB,width=%d,height=%d ! "
      "tensor_converter ! tensor_sink name=tensor_sink", width, height);
  g_app.data_pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  _check_cond_err (g_app.data_pipeline != NULL);

  /** data message callback */
  g_app.data_bus = gst_element_get_bus (g_app.data_pipeline);
  _check_cond_err (g_app.data_bus != NULL);

  gst_bus_add_signal_watch (g_app.data_bus);
  handle_id = g_signal_connect (g_app.data_bus, "message",
      (GCallback) _data_message_cb, NULL);
  _check_cond_err (handle_id > 0);

  /** get tensor sink element using name */
  element = gst_bin_get_by_name (GST_BIN (g_app.data_pipeline), "tensor_sink");
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

  gst_object_unref (element);

  /** init player pipeline */
  str_pipeline =
      g_strdup_printf
      ("appsrc name=player_src ! tensordec ! videoconvert ! xvimagesink");
  g_app.player_pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  _check_cond_err (g_app.player_pipeline != NULL);

  /** player message callback */
  g_app.player_bus = gst_element_get_bus (g_app.player_pipeline);
  _check_cond_err (g_app.player_bus != NULL);

  gst_bus_add_signal_watch (g_app.player_bus);
  handle_id = g_signal_connect (g_app.player_bus, "message",
      (GCallback) _player_message_cb, NULL);
  _check_cond_err (handle_id > 0);

  /** start pipeline */
  gst_element_set_state (g_app.data_pipeline, GST_STATE_PLAYING);
  gst_element_set_state (g_app.player_pipeline, GST_STATE_PLAYING);

  /** run main loop */
  g_main_loop_run (g_app.loop);
  /** quit when received eos message */

  gst_element_set_state (g_app.data_pipeline, GST_STATE_NULL);
  gst_element_set_state (g_app.player_pipeline, GST_STATE_NULL);

error:
  _free_app_data ();
  return 0;
}
