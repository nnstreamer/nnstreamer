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
  GstElement *tensor_sink; /**< tensor sink element */
  GstElement *player_src; /**< player source element */

  guint received; /**< received buffer count */
  gboolean set_caps; /**< caps passed to player pipeline */
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

  if (g_app_data.data_bus) {
    gst_bus_remove_signal_watch (g_app_data.data_bus);
    gst_object_unref (g_app_data.data_bus);
    g_app_data.data_bus = NULL;
  }

  if (g_app_data.player_bus) {
    gst_bus_remove_signal_watch (g_app_data.player_bus);
    gst_object_unref (g_app_data.player_bus);
    g_app_data.player_bus = NULL;
  }

  if (g_app_data.tensor_sink) {
    gst_object_unref (g_app_data.tensor_sink);
    g_app_data.tensor_sink = NULL;
  }

  if (g_app_data.player_src) {
    gst_object_unref (g_app_data.player_src);
    g_app_data.player_src = NULL;
  }

  if (g_app_data.data_pipeline) {
    gst_object_unref (g_app_data.data_pipeline);
    g_app_data.data_pipeline = NULL;
  }

  if (g_app_data.player_pipeline) {
    gst_object_unref (g_app_data.player_pipeline);
    g_app_data.player_pipeline = NULL;
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
_data_message_cb (GstBus * bus, GstMessage * message, gpointer user_data)
{
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_EOS:
      _print_log ("[data] received eos message");
      gst_app_src_end_of_stream (GST_APP_SRC (g_app_data.player_src));
      break;

    case GST_MESSAGE_ERROR:
      _print_log ("[data] received error message");
      _parse_err_message (message);
      g_main_loop_quit (g_app_data.loop);
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
      g_main_loop_quit (g_app_data.loop);
      break;

    case GST_MESSAGE_ERROR:
      _print_log ("[player] received error message");
      _parse_err_message (message);
      g_main_loop_quit (g_app_data.loop);
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
  g_app_data.received++;
  _print_log ("new data callback [%d]", g_app_data.received);

  if (!g_app_data.set_caps) {
    GstPad *sink_pad;
    GstCaps *caps;

    sink_pad = gst_element_get_static_pad (g_app_data.tensor_sink, "sink");

    if (sink_pad) {
      caps = gst_pad_get_current_caps (sink_pad);

      if (caps) {
        gst_app_src_set_caps (GST_APP_SRC (g_app_data.player_src), caps);

        gst_caps_unref (caps);
        g_app_data.set_caps = TRUE;
      }

      gst_object_unref (sink_pad);
    }
  }

  gst_app_src_push_buffer (GST_APP_SRC (g_app_data.player_src),
      gst_buffer_copy (buffer));
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

  /* init gstreamer */
  gst_init (&argc, &argv);

  /* init app variable */
  g_app_data.received = 0;
  g_app_data.set_caps = FALSE;

  /* main loop and pipeline */
  g_app_data.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_app_data.loop != NULL);

  str_pipeline =
      g_strdup_printf
      ("videotestsrc is-live=TRUE ! video/x-raw,format=RGB,width=%d,height=%d ! "
      "tensor_converter ! tensor_sink name=tensor_sink", width, height);
  g_app_data.data_pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  _check_cond_err (g_app_data.data_pipeline != NULL);

  /* data message callback */
  g_app_data.data_bus = gst_element_get_bus (g_app_data.data_pipeline);
  _check_cond_err (g_app_data.data_bus != NULL);

  gst_bus_add_signal_watch (g_app_data.data_bus);
  handle_id = g_signal_connect (g_app_data.data_bus, "message",
      (GCallback) _data_message_cb, NULL);
  _check_cond_err (handle_id > 0);

  /* get tensor sink element using name */
  g_app_data.tensor_sink =
      gst_bin_get_by_name (GST_BIN (g_app_data.data_pipeline), "tensor_sink");
  _check_cond_err (g_app_data.tensor_sink != NULL);

  if (DBG) {
    /* print logs, default TRUE */
    g_object_set (g_app_data.tensor_sink, "silent", (gboolean) FALSE, NULL);
  }

  /* enable emit-signal, default TRUE */
  g_object_set (g_app_data.tensor_sink, "emit-signal", (gboolean) TRUE, NULL);

  /* tensor sink signal : new data callback */
  handle_id = g_signal_connect (g_app_data.tensor_sink, "new-data",
      (GCallback) _new_data_cb, NULL);
  _check_cond_err (handle_id > 0);

  /* init player pipeline */
  str_pipeline =
      g_strdup_printf
      ("appsrc name=player_src ! tensordec ! videoconvert ! xvimagesink");
  g_app_data.player_pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  _check_cond_err (g_app_data.player_pipeline != NULL);

  /* player message callback */
  g_app_data.player_bus = gst_element_get_bus (g_app_data.player_pipeline);
  _check_cond_err (g_app_data.player_bus != NULL);

  gst_bus_add_signal_watch (g_app_data.player_bus);
  handle_id = g_signal_connect (g_app_data.player_bus, "message",
      (GCallback) _player_message_cb, NULL);
  _check_cond_err (handle_id > 0);

  /* player source element */
  g_app_data.player_src =
      gst_bin_get_by_name (GST_BIN (g_app_data.player_pipeline), "player_src");
  _check_cond_err (g_app_data.player_src != NULL);

  /* start pipeline */
  gst_element_set_state (g_app_data.data_pipeline, GST_STATE_PLAYING);
  gst_element_set_state (g_app_data.player_pipeline, GST_STATE_PLAYING);

  g_main_loop_run (g_app_data.loop);

  /* quit when received eos message */
  gst_element_set_state (g_app_data.data_pipeline, GST_STATE_NULL);
  gst_element_set_state (g_app_data.player_pipeline, GST_STATE_NULL);

error:
  _free_app_data ();
  return 0;
}
