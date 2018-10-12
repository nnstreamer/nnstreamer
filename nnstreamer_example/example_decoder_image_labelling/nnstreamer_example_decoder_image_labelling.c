/**
 * @file	nnstreamer_example_decoder.c
 * @date	4 Oct 2018
 * @brief	Tensor stream example with tensor decoder
 * @see	https://github.com/nnsuite/nnstreamer	
 * @author	Jinhyuck Park <jinhyuck83.park@samsung.com>
 * @bug		No known bugs.
 *
 * NNStreamer example for image recognition with only gstreamer plug-in's include decoder.
 *
 * Pipeline :
 * v4l2src -- tee --  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- textoverlay -- videoconvert -- ximagesink
 *                  |                                                                    |
 *                  --- videoscale -- tensor_converter -- tensor_filter -- tensor_decoder 
 *
 *
 * 'tensor_filter' for image recognition.
 * Download tflite moel 'Mobilenet_1.0_224_quant' from below link,
 * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md#image-classification-quantized-models
 *
 * 'tensor decoder' updates recognition result links to text_sink of textoverlay.
 *
 * Run example :
 * Before running this example, GST_PLUGIN_PATH should be updated for nnstreamer plug-in.
 * $ export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:<nnstreamer plugin path>
 * $ ./nnstreamer_example_decoder
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <glib.h>
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
 * @brief Data structure for tflite model info.
 */
typedef struct
{
  gchar *model_path; /**< tflite model file path */
} tflite_info_s;

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
  tflite_info_s tflite_info; /**< tflite model info */
} AppData;

/**
 * @brief Data for pipeline and result.
 */
static AppData g_app;

/**
 * @brief Free data in tflite info structure.
 */
static void
_tflite_free_info (tflite_info_s * tflite_info)
{
  g_return_if_fail (tflite_info != NULL);

  if (tflite_info->model_path) {
    g_free (tflite_info->model_path);
    tflite_info->model_path = NULL;
  }

}

/**
 * @brief Check tflite model and load labels.
 *
 * This example uses 'Mobilenet_1.0_224_quant' for image classification.
 */
static gboolean
_tflite_init_info (tflite_info_s * tflite_info, const gchar * path)
{
  const gchar tflite_model[] = "mobilenet_v1_1.0_224_quant.tflite";

  g_return_val_if_fail (tflite_info != NULL, FALSE);

  tflite_info->model_path = NULL;

  /** check model file exists */
  tflite_info->model_path = g_strdup_printf ("%s/%s", path, tflite_model);

  if (access (tflite_info->model_path, F_OK) != 0) {
    _print_log ("cannot find tflite model [%s]", tflite_info->model_path);
    return FALSE;
  }

  return TRUE;
}


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

  _tflite_free_info (&g_app.tflite_info);
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
 * @brief Function to print qos message.
 */
static void
_parse_qos_message (GstMessage * message)
{
  GstFormat format;
  guint64 processed;
  guint64 dropped;

  gst_message_parse_qos_stats (message, &format, &processed, &dropped);
  _print_log ("format[%d] processed[%" G_GUINT64_FORMAT "] dropped[%"
      G_GUINT64_FORMAT "]", format, processed, dropped);
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

    case GST_MESSAGE_QOS:
      _parse_qos_message (message);
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
  const gchar tflite_model_path[] = "./tflite_model";
  const gchar tflite_label[] = "./tflite_model/labels.txt";
  /** 224x224 for tflite model */
  const guint width = 224;
  const guint height = 224;

  gchar *str_pipeline;
  gulong handle_id;
  GstElement *element;

  _print_log ("start app..");

  /** init app variable */
  g_app.running = FALSE;
  g_app.received = 0;

  _check_cond_err (_tflite_init_info (&g_app.tflite_info, tflite_model_path));

  /** init gstreamer */
  gst_init (&argc, &argv);

  /** main loop */
  g_app.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_app.loop != NULL);

  /** init pipeline */
  str_pipeline =
      g_strdup_printf
      ("textoverlay name=overlay font-desc=\"Sans, 24\" ! videoconvert ! ximagesink name=img_test "
      "v4l2src name=cam_src ! videoscale ! video/x-raw,width=640,height=480,format=RGB ! tee name=t_raw "
      "t_raw. ! queue ! overlay.video_sink "
      "t_raw. ! queue ! videoscale ! video/x-raw,width=%d,height=%d !tensor_converter !"
      "tensor_filter framework=tensorflow-lite model=%s ! "
      "tensor_decoder output-type=2 mode=image_labeling mode-option-1=%s ! overlay.text_sink ",
      width, height, g_app.tflite_info.model_path, tflite_label);
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

  g_app.running = TRUE;

  /** set window title */
  _set_window_title ("img_test", "NNStreamer Example");

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

  _free_app_data ();
  return 0;
}
