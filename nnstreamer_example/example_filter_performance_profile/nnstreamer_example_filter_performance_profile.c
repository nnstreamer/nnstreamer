/**
 * @file	nnstreamer_example_filter_performance_profile.c
 * @date	27 August 2018
 * @brief	A NNStreamer Example of tensor_filter using TensorFlow Lite:
 * 		Perfromance Profiling (i.e., FPS)
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Wook Song <wook16.song@samsung.com>
 * @bug		No known bugs.
 *
 * A NNStreamer example application (with tensor_filter using TensorFlow Lite)
 * for performance profiling.
 *
 * Pipeline :
 * v4l2src -- videoconvert -- tee (optional) -- queue -- textoverlay -- fpsdisplaysink
 *                             |
 *                              -- queue -- videoscale -- videoconvert -- tensor_converter -- tensor_filter -- tensor_sink
 *
 * This example application currently only supports MOBINET for Tensorflow Lite via 'tensor_filter'.
 * Download tflite moel 'Mobilenet_1.0_224_quant' from below link,
 * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md#image-classification-quantized-models
 * By using the 'new-data' signal of tensor_sink, Frames per Second (FPS) is measured
 * as well as the clasification result is fed to 'textoverlay'.
 *
 * How to run this application: Before running this example,
 * GST_PLUGIN_PATH should be updated for the path where the nnstreamer plug-ins are placed.
 * $ export GST_PLUGIN_PATH=/where/NNSTreamer/plugins/located:$GST_PLUGIN_PATH
 * The model file and its related miscellaneous files (e.g. *.tflite, labels.txt, and etc.)
 * are also required to be placed in the ./model directory.
 *
 * $ ./nnstreamer_example_filter_performance_profile --help
 * Usage:
 * nnstreamer_example_filter_performance_profile [OPTION...]

 * Help Options:
 * -h, --help                                                             Show help options

 * Application Options:
 * -c, --capture=/dev/videoX                                              A device node of the video capture device you wish to use
 * -f, --file=/where/your/video/file/located                              A video file location to play
 * --width= (Defaults: 1920)                                              Width of input source
 * --height= (Defaults: 1080)                                             Height of input source
 * --framerates= (Defaults: 5/1)                                          Frame rates of input source
 * --tensor-filter-desc=mobinet-tflite|... (Defaults: mobinet-tflite)     NN model and framework description for tensor_filter
 * --nnline-only                                                          Do not play audio/video input source
 *
 * For example, in order to run the Mobinet Tensorflow Lite model using the NNStreamer pipeline for the input source,
 * from the video capture device (/dev/video0), of which the resolution is 1920x1080 and the frame rates is 5,
 *
 * $ ./nnstreamer_example_filter_performance_profile -c /dev/video0 --tensor-filter-desc=mobinet-tflite
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gst/gst.h>

/**
 * @brief A data type definition for the command line option, -c/--capture
 */
typedef enum _input_src_t
{
  CAM_SRC = 0,
  FILE_SRC,
} input_src_t;

/**
 * @brief A data type definition for the command line option, --tensor-filter-desc
 */
typedef enum _nn_tensor_filter_desc_t
{
  DESC_NOT_SUPPORTED = -1,
  /* default */
  TF_LITE_MOBINET = 0,
} nn_tensor_filter_desc_t;

/**
 * @brief Constant values
 */
enum constant_values
{
  MAX_NUM_OF_SIGNALS = 128,
  DEFAULT_WIDTH_INPUT_SRC = 1920,
  DEFAULT_HEIGHT_INPUT_SRC = 1080,
  DEFAULT_WIDTH_TFLITE_MOBINET = 224,
  DEFAULT_HEIGHT_TFLITE_MOBINET = 224,
};
static const char DEFAULT_FRAME_RATES_INPUT_SRC[] = "5/1";
static const char DEFAULT_FORMAT_TENSOR_CONVERTER[] = "RGB";
static const char DEFAULT_PATH_MODEL_TENSOR_FILTER[] = "./model/";
static const char NAME_APP_PIPELINE[] = "NNStreamer Pipeline";
static const char NAME_PROP_DEVICE_V4L2SRC[] = "device";
static const char NAME_V4L2_PIPELINE_INPUT_SRC[] = "usbcam";
static const char NAME_V4L2_PIPELINE_INPUT_VIDEOCONVERT[] =
    "Colorspace converter for input source";
static const char NAME_V4L2_PIPELINE_INPUT_CAPSFILTER[] =
    "CAPS filter for input source";
static const char NAME_V4L2_PIPELINE_TEE[] = "TEE";
static const char NAME_V4L2_PIPELINE_OUTPUT_QUEUE[] = "Queue for image sink";
static const char NAME_V4L2_PIPELINE_OUTPUT_SINK[] = "Xv-based image sink";
static const char NAME_V4L2_PIPELINE_OUTPUT_TEXTOVERLAY[] =
    "Textoverlay to display the inference result";
static const char NAME_NN_TFLITE_PIPELINE_QUEUE[] = "Queue for NN-TFlite";
static const char NAME_NN_TFLITE_PIPELINE_VIDEOSCALE[] =
    "Video scaler for NN-TFlite";
static const char NAME_NN_TFLITE_PIPELINE_VIDEOCONVERT[] =
    "Colorspace converter  for NN-TFlite";
static const char NAME_NN_TFLITE_PIPELINE_INPUT_CAPSFILTER[] =
    "CAPS filter for input source of NN-TFlite";
static const char NAME_NN_TFLITE_PIPELINE_TENSOR_CONVERTER[] =
    "Tensor converter for NN-TFlite";
static const char NAME_NN_TFLITE_PIPELINE_TENSOR_FILTER[] =
    "Tensor filter for NN-TFlite";
static const char NAME_NN_TFLITE_PIPELINE_TENSOR_SINK[] =
    "Tensor sink for NN-TFlite";

static const char *DESC_LIST_TENSOR_FILTER[] = {
  [TF_LITE_MOBINET] = "mobinet-tflite",
  /* sentinel */
  NULL,
};

static const char *FRAMEWORK_LIST_TENSOR_FILTER[] = {
  [TF_LITE_MOBINET] = "tensorflow-lite",
  /* sentinel */
  NULL,
};

static const char *NAME_LIST_OF_MODEL_FILE_TENSOR_FILTER[] = {
  [TF_LITE_MOBINET] = "mobilenet_v1_1.0_224_quant.tflite",
  /* sentinel */
  NULL,
};

/**
 * TODO: Currently, only one misc file per each model is supported.
 */
static const char *NAME_LIST_OF_MISC_FILE_TENSOR_FILTER[] = {
  [TF_LITE_MOBINET] = "labels.txt",
  /* sentinel */
  NULL,
};

/**
 * @brief A data type definition for the information needed to set up the GstElements corresponding to the input source: v4l2src
 */
typedef struct _v4l2src_property_info_t
{
  gchar *device;
} v4l2src_property_info_t;

/**
 * @brief A data type definition for the information needed to set up the GstElements corresponding to the input source
 */
typedef union _src_property_info_t
{
  v4l2src_property_info_t v4l2src_property_info;
} src_property_info_t;

/**
 * @brief A data type definition for the input and output pipeline
 *
 * GstElements required to construct the input and output pipeline are here.
 */
typedef struct _v4l2src_pipeline_container_t
{
  GstElement *input_source;
  GstElement *input_videoconvert;
  GstElement *input_capsfilter;
  GstElement *tee;
  GstElement *output_queue;
  GstElement *output_textoverlay;
  GstElement *output_sink; /**< fpsdisplaysink */
} v4l2src_pipeline_container_t;

/**
 * @brief A data type definition for the NNStreamer pipeline
 *
 * GstElements required to construct the NNStreamer pipeline are here.
 */
typedef struct _nn_tflite_pipeline_container_t
{
  GstElement *nn_tflite_queue;
  GstElement *nn_tflite_videoscale;
  GstElement *nn_tflite_videoconvert;
  GstElement *nn_tflite_capsfilter;
  GstElement *nn_tflite_tensor_converter;
  GstElement *nn_tflite_tensor_filter;
  GstElement *nn_tflite_tensor_sink;
} nn_tflite_pipeline_container_t;

/**
 * @brief A data type definition for the pipelines
 */
typedef struct _pipeline_container_t
{
  v4l2src_pipeline_container_t v4l2src_pipeline_container;
  nn_tflite_pipeline_container_t nn_tflite_pipeline_container;
} pipeline_container_t;

/**
 * @brief A data type definition for the model specific information: the Mobinet Tensorflow-lite model
 */
typedef struct _tflite_mobinet_info_t
{
  GList *labels;
} tflite_mobinet_info_t;

/**
 * @brief A data type definition for the NNStreamer application context data
 */
typedef struct _nnstrmr_app_context_t
{
  /* Variables for the command line options */
  input_src_t input_src; /**< -c/--capture */
  gint input_src_width; /**< --width */
  gint input_src_height; /**< --height */
  gchar *input_src_framerates; /**< --framerates */
  nn_tensor_filter_desc_t nn_tensorfilter_desc; /**< --tensor-filter-desc */
  gboolean flag_nnline_only; /**< --nnline-only */
  /* Variables for the information need to initialize this application */
  gchar *nn_tensor_filter_model_path; /**< the path where the NN model files located */
  tflite_mobinet_info_t tflite_mobinet_info; /**< model specific information for mobinet+tf-lite */
  src_property_info_t src_property_info; /**< information required to set up the input source */
  GMainLoop *mainloop;
  GstElement *pipeline;
  pipeline_container_t pipeline_container; /**< pipeline container, which indirectly includes the GstElements */
  GstPad *tee_output_line_pad; /**< a static src pad of tee for the output pipeline */
  GstPad *tee_nn_line_pad; /**< a static src pad of tee for the nnstreamer pipeline */
  /* Variables for the GSignal maintenance */
  GMutex signals_mutex;
  guint signals_connected[MAX_NUM_OF_SIGNALS];
  gint signal_idx;
  /* Variables for the performance profiling */
  GstClockTime time_pipeline_start;
  GstClockTime time_last_profile;
} nnstrmr_app_context_t;

/**
 * @brief callback function for watching bus of the pipeline
 */
static gboolean
_cb_bus_watch (GstBus * bus, GstMessage * msg, gpointer app_ctx_gptr)
{
  nnstrmr_app_context_t *app_ctx = (nnstrmr_app_context_t *) app_ctx_gptr;
  gchar *debug;
  GError *error;

  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_STREAM_STATUS:
    {
      GstStreamStatusType streamstatus;
      gst_message_parse_stream_status (msg, &streamstatus, NULL);
      g_print ("gstreamer stream status %d ==> %s\n",
          streamstatus, GST_OBJECT_NAME (msg->src));
      break;
    }
    case GST_MESSAGE_STATE_CHANGED:
    {
      if (GST_MESSAGE_SRC (msg) == GST_OBJECT (app_ctx->pipeline)) {
        GstState old_state, new_state, pending_state;
        gst_message_parse_state_changed (msg, &old_state, &new_state,
            &pending_state);
        if ((old_state == GST_STATE_PAUSED) && (new_state == GST_STATE_PLAYING)) {
          GstClock *clock;
          clock = gst_element_get_clock (app_ctx->pipeline);
          app_ctx->time_pipeline_start = gst_clock_get_time (clock);
          app_ctx->time_last_profile = gst_clock_get_time (clock);
        }
      }
      break;
    }
    case GST_MESSAGE_EOS:
    {
      g_print ("INFO: End of stream!\n");
      g_main_loop_quit (app_ctx->mainloop);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
      gst_message_parse_error (msg, &error, &debug);
      g_free (debug);

      g_printerr ("ERR: %s\n", error->message);
      g_error_free (error);

      g_main_loop_quit (app_ctx->mainloop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

/**
 * @brief Set up the v4l2src GstElement
 * @param v4l2src a pointer of the v4l2src GstElement
 * @param ctx the application context data
 * @return TRUE, if it is succeeded
 */
static gboolean
_set_properties_of_v4l2src (GstElement * v4l2src,
    const nnstrmr_app_context_t ctx)
{
  gchar *dev = ctx.src_property_info.v4l2src_property_info.device;
  gboolean ret = TRUE;

  if (!g_file_test (dev, G_FILE_TEST_EXISTS)) {
    g_printerr ("ERR: the device node %s does not exist\n", dev);
    ret = FALSE;
  }

  g_object_set (G_OBJECT (v4l2src), NAME_PROP_DEVICE_V4L2SRC, dev, NULL);

  g_free (dev);

  return ret;
}

/**
 * @brief Parse the command line option arguments
 * @param argc
 * @param argv
 * @param ctx a pointer of the application context data
 * @return none
 */
static void
_set_and_parse_option_info (int argc, char *argv[], nnstrmr_app_context_t * ctx)
{
  gchar *cap_dev_node = NULL;
  gchar *file_path = NULL;
  gchar *framerates = NULL;
  gchar *tensorfilter_desc = NULL;
  gchar *width_arg_desc =
      g_strdup_printf (" (Defaults: %d)", DEFAULT_WIDTH_INPUT_SRC);
  gchar *height_arg_desc =
      g_strdup_printf (" (Defaults: %d)", DEFAULT_HEIGHT_INPUT_SRC);
  gchar *framerates_arg_desc =
      g_strdup_printf (" (Defaults: %s)", DEFAULT_FRAME_RATES_INPUT_SRC);
  gchar *tf_desc_arg_desc =
      g_strdup_printf ("%s|... (Defaults: %s)", DESC_LIST_TENSOR_FILTER[0],
      DESC_LIST_TENSOR_FILTER[0]);
  gint width = -1;
  gint height = -1;
  gint ret = 0;
  gboolean flag_nnline_only = FALSE;
  GError *error = NULL;
  GOptionContext *optionctx;

  const GOptionEntry main_entries[] = {
    {"capture", 'c', G_OPTION_FLAG_NONE, G_OPTION_ARG_STRING, &cap_dev_node,
          "A device node of the video capture device you wish to use",
        "/dev/videoX"},
    {"file", 'f', G_OPTION_FLAG_NONE, G_OPTION_ARG_STRING, &file_path,
        "A video file location to play", "/where/your/video/file/located"},
    {"width", 0, G_OPTION_FLAG_NONE, G_OPTION_ARG_INT, &width,
        "Width of input source", width_arg_desc},
    {"height", 0, G_OPTION_FLAG_NONE, G_OPTION_ARG_INT, &height,
        "Height of input source", height_arg_desc},
    {"framerates", 0, G_OPTION_FLAG_NONE, G_OPTION_ARG_STRING, &framerates,
        "Frame rates of input source", framerates_arg_desc},
    {"tensor-filter-desc", 0, G_OPTION_FLAG_NONE, G_OPTION_ARG_STRING,
          &tensorfilter_desc,
          "NN model and framework description for tensor_filter",
        tf_desc_arg_desc},
    {"nnline-only", 0, G_OPTION_FLAG_NONE, G_OPTION_ARG_NONE,
          &flag_nnline_only, "Do not play audio/video input source",
        NULL},
    {NULL}
  };

  optionctx = g_option_context_new (NULL);
  g_option_context_add_main_entries (optionctx, main_entries, NULL);

  if (!g_option_context_parse (optionctx, &argc, &argv, &error)) {
    g_print ("option parsing failed: %s\n", error->message);
    ret = -1;
    goto common_cleanup;
  }

  if ((cap_dev_node != NULL) && (file_path != NULL)) {
    g_printerr ("ERR: \'capture\' and \'file\' options "
        "cannot be used simultaneously\n");
    g_free (cap_dev_node);
    g_free (file_path);
    ret = -1;
    goto common_cleanup;
  } else if ((cap_dev_node == NULL) && (file_path == NULL)) {
    g_printerr ("ERR: one of the application options should be provided; "
        "-c, --capture=/dev/videoX or "
        "-f, --file=/where/your/video/file/located\n");
    ret = -1;
    goto common_cleanup;
  }

  if (cap_dev_node != NULL) {
    ctx->input_src = CAM_SRC;
    ctx->src_property_info.v4l2src_property_info.device = cap_dev_node;
  } else {
    /* TODO */
    ctx->input_src = FILE_SRC;
  }

  if (width == -1) {
    width = DEFAULT_WIDTH_INPUT_SRC;
  }
  ctx->input_src_width = width;

  if (height == -1) {
    height = DEFAULT_HEIGHT_INPUT_SRC;
  }
  ctx->input_src_height = height;

  if (framerates == NULL) {
    framerates = g_strndup (DEFAULT_FRAME_RATES_INPUT_SRC,
        strlen (DEFAULT_FRAME_RATES_INPUT_SRC));
  }
  ctx->input_src_framerates = framerates;

  if (tensorfilter_desc == NULL) {
    ctx->nn_tensorfilter_desc = TF_LITE_MOBINET;
  } else {
    int i = 0;
    const char *desc;

    ctx->nn_tensorfilter_desc = DESC_NOT_SUPPORTED;
    while ((desc = DESC_LIST_TENSOR_FILTER[i]) != NULL) {
      if (!strncmp (desc, tensorfilter_desc, strlen (tensorfilter_desc))) {
        ctx->nn_tensorfilter_desc = i;
      }
      i++;
    }

    if (ctx->nn_tensorfilter_desc == DESC_NOT_SUPPORTED) {
      g_printerr
          ("ERR: tensor_filter does not support the pair of the framework and model: %s\n",
          tensorfilter_desc);
      g_free (tensorfilter_desc);
      ret = -1;
      goto common_cleanup;
    }
  }

  ctx->nn_tensor_filter_model_path =
      g_strconcat (DEFAULT_PATH_MODEL_TENSOR_FILTER,
      NAME_LIST_OF_MODEL_FILE_TENSOR_FILTER[ctx->nn_tensorfilter_desc], NULL);
  if (!g_file_test (ctx->nn_tensor_filter_model_path, G_FILE_TEST_EXISTS
          || G_FILE_TEST_IS_REGULAR)) {
    g_printerr ("ERR: the model file %s corresponding to %s does not exist\n",
        ctx->nn_tensor_filter_model_path,
        DESC_LIST_TENSOR_FILTER[ctx->nn_tensorfilter_desc]);
    g_free (ctx->nn_tensor_filter_model_path);
    ret = -1;
    goto common_cleanup;
  }

  ctx->flag_nnline_only = flag_nnline_only;

common_cleanup:
  g_free (width_arg_desc);
  g_free (height_arg_desc);
  g_free (framerates_arg_desc);
  g_free (tf_desc_arg_desc);
  g_option_context_free (optionctx);
  if (ret != 0) {
    g_main_loop_unref (ctx->mainloop);
    exit (ret);
  }
}

/**
 * @brief Construct the v4l2src input and output pipeline
 *
 * In this function, the GstElements included in the pipelines are made, added,
 * and linked. Setting the properties for those GstElements are also done.
 *
 * @param ctx a pointer of the application context data
 * @return TRUE, if it is succeeded
 */
static gboolean
_construct_v4l2src_pipeline (nnstrmr_app_context_t * ctx)
{
  GstElement *pipeline = ctx->pipeline;
  v4l2src_pipeline_container_t *pipeline_cntnr =
      &((ctx->pipeline_container).v4l2src_pipeline_container);
  gboolean ret;
  GstCaps *caps;
  gchar *str_caps;

  pipeline_cntnr->input_source =
      gst_element_factory_make ("v4l2src", NAME_V4L2_PIPELINE_INPUT_SRC);
  pipeline_cntnr->input_videoconvert =
      gst_element_factory_make ("videoconvert",
      NAME_V4L2_PIPELINE_INPUT_VIDEOCONVERT);
  pipeline_cntnr->input_capsfilter =
      gst_element_factory_make ("capsfilter",
      NAME_V4L2_PIPELINE_INPUT_CAPSFILTER);
  pipeline_cntnr->tee =
      gst_element_factory_make ("tee", NAME_V4L2_PIPELINE_TEE);
  pipeline_cntnr->output_queue =
      gst_element_factory_make ("queue", NAME_V4L2_PIPELINE_OUTPUT_QUEUE);
  pipeline_cntnr->output_textoverlay =
      gst_element_factory_make ("textoverlay",
      NAME_V4L2_PIPELINE_OUTPUT_TEXTOVERLAY);
  pipeline_cntnr->output_sink =
      gst_element_factory_make ("fpsdisplaysink",
      NAME_V4L2_PIPELINE_OUTPUT_SINK);

  if (!pipeline_cntnr->input_source || !pipeline_cntnr->input_videoconvert
      || !pipeline_cntnr->input_capsfilter || !pipeline_cntnr->tee
      || !pipeline_cntnr->output_queue || !pipeline_cntnr->output_textoverlay
      || !pipeline_cntnr->output_sink) {
    g_printerr ("ERR: cannot create one (or more) of the elements "
        "which the application pipeline consists of\n");
    g_free (ctx->input_src_framerates);
    return FALSE;
  }

  str_caps =
      g_strdup_printf ("video/x-raw,width=%d,hegith=%d,framerate=%s",
      ctx->input_src_width, ctx->input_src_height, ctx->input_src_framerates);
  caps = gst_caps_from_string (str_caps);
  g_object_set (G_OBJECT (pipeline_cntnr->input_capsfilter),
      "caps", caps, NULL);
  g_free (ctx->input_src_framerates);
  g_free (str_caps);
  gst_caps_unref (caps);

  g_object_set (G_OBJECT (pipeline_cntnr->output_textoverlay), "valignment",
      /** top */ 2, NULL);
  g_object_set (G_OBJECT (pipeline_cntnr->output_textoverlay), "font-desc",
      "Sans, 24", NULL);

  ret = _set_properties_of_v4l2src (pipeline_cntnr->input_source, *ctx);
  if (ret == FALSE) {
    return ret;
  }

  gst_bin_add_many (GST_BIN (pipeline), pipeline_cntnr->input_source,
      pipeline_cntnr->input_videoconvert, pipeline_cntnr->input_capsfilter,
      pipeline_cntnr->tee, pipeline_cntnr->output_queue,
      pipeline_cntnr->output_textoverlay, pipeline_cntnr->output_sink, NULL);

  ret = gst_element_link_many (pipeline_cntnr->input_source,
      pipeline_cntnr->input_videoconvert, pipeline_cntnr->input_capsfilter,
      pipeline_cntnr->tee, pipeline_cntnr->output_queue,
      pipeline_cntnr->output_textoverlay, pipeline_cntnr->output_sink, NULL);
  if (ret == FALSE) {
    g_printerr ("ERR: cannot link one (or more) of the elements "
        "which the application pipeline consists of\n");
    return FALSE;
  }

  ctx->tee_output_line_pad =
      gst_element_get_static_pad (pipeline_cntnr->tee, "src_0");
  ctx->tee_nn_line_pad =
      gst_element_get_request_pad (pipeline_cntnr->tee, "src_%u");


  return TRUE;
}

/**
 * @brief A signal handler for 'new-data' emitted by 'tensor-sink'
 *
 * The suffix _nn means that this handler is registered at the NNStreamer pipeline.
 * The performance profiling is done by this fuction.
 *
 * @param object a pointer of the 'tensor-sink' GstElement
 * @buffer buffer a pointer of the buffer in the 'tensor-sink' GstElement
 * @buffer user_data a pointer of the application context data
 * @return none
 */
static void
_handle_tensor_sink_new_data_nn (GstElement * object, GstBuffer * buffer,
    gpointer user_data)
{
  /* Performance profiling */
  nnstrmr_app_context_t *ctx = (nnstrmr_app_context_t *) user_data;
  GstClock *clock;
  GstClockTime now;
  static guint total_passed = 0;
  gint64 msecs_elapsed;
  gint64 msecs_interval;

  if (!GST_CLOCK_TIME_IS_VALID (ctx->time_pipeline_start)
      || !GST_CLOCK_TIME_IS_VALID (ctx->time_last_profile)) {
    return;
  }

  total_passed++;

  clock = gst_element_get_clock (ctx->pipeline);
  now = gst_clock_get_time (clock);
  msecs_elapsed =
      GST_TIME_AS_MSECONDS (GST_CLOCK_DIFF (ctx->time_pipeline_start, now));
  msecs_interval =
      GST_TIME_AS_MSECONDS (GST_CLOCK_DIFF (ctx->time_last_profile, now));
  ctx->time_last_profile = now;

  g_print ("Avg. FPS = %lf (processed: %u, elapsed time (ms): %" G_GINT64_FORMAT "), ",
      (gdouble) total_passed * G_GINT64_CONSTANT (1000) / msecs_elapsed,
      total_passed, msecs_elapsed);
  g_print ("Cur. FPS = %lf\n",
      (gdouble) 1 * G_GINT64_CONSTANT (1000) / msecs_interval);
}

/**
 * @brief Construct the filesec input and output pipeline (TODO)
 *
 * @param ctx a pointer of the application context data
 * @return TRUE, if it is succeeded
 */
static gboolean
_construct_filesrc_pipeline (nnstrmr_app_context_t * ctx)
{
  return TRUE;
}

/**
 * @brief Construct the NNStreamer pipeline
 *
 * In this function, the GstElements included in the pipelines are made, added,
 * and linked. The sink pad of the queue is also linked to the src pad of tee in
 * the input source pipeline. Setting the properties, 'framework' and 'model',
 * for the 'tensor_filter' GstElement are done here.
 *
 * @param ctx a pointer of the application context data
 * @return TRUE, if it is succeeded
 */
static gboolean
_construct_nn_tflite_pipeline (nnstrmr_app_context_t * ctx)
{
  GstElement *pipeline = ctx->pipeline;
  nn_tflite_pipeline_container_t *pipeline_cntnr =
      &((ctx->pipeline_container).nn_tflite_pipeline_container);
  GstCaps *caps;
  gchar *str_caps;
  gboolean ret = TRUE;
  GstPad *pad;

  pipeline_cntnr->nn_tflite_queue =
      gst_element_factory_make ("queue", NAME_NN_TFLITE_PIPELINE_QUEUE);
  pipeline_cntnr->nn_tflite_videoscale =
      gst_element_factory_make ("videoscale",
      NAME_NN_TFLITE_PIPELINE_VIDEOSCALE);
  pipeline_cntnr->nn_tflite_videoconvert =
      gst_element_factory_make ("videoconvert",
      NAME_NN_TFLITE_PIPELINE_VIDEOCONVERT);
  pipeline_cntnr->nn_tflite_capsfilter =
      gst_element_factory_make ("capsfilter",
      NAME_NN_TFLITE_PIPELINE_INPUT_CAPSFILTER);
  pipeline_cntnr->nn_tflite_tensor_converter =
      gst_element_factory_make ("tensor_converter",
      NAME_NN_TFLITE_PIPELINE_TENSOR_CONVERTER);
  pipeline_cntnr->nn_tflite_tensor_filter =
      gst_element_factory_make ("tensor_filter",
      NAME_NN_TFLITE_PIPELINE_TENSOR_FILTER);
  pipeline_cntnr->nn_tflite_tensor_sink =
      gst_element_factory_make ("tensor_sink",
      NAME_NN_TFLITE_PIPELINE_TENSOR_SINK);

  g_object_set (G_OBJECT (pipeline_cntnr->nn_tflite_tensor_sink),
      "max-lateness", (gint64) - 1, NULL);
  g_object_set (G_OBJECT (pipeline_cntnr->nn_tflite_tensor_filter), "framework",
      FRAMEWORK_LIST_TENSOR_FILTER[ctx->nn_tensorfilter_desc], NULL);
  g_object_set (G_OBJECT (pipeline_cntnr->nn_tflite_tensor_filter), "model",
      ctx->nn_tensor_filter_model_path, NULL);
  g_free (ctx->nn_tensor_filter_model_path);

  str_caps =
      g_strdup_printf ("video/x-raw,width=%d,height=%d,format=%s",
      DEFAULT_WIDTH_TFLITE_MOBINET, DEFAULT_HEIGHT_TFLITE_MOBINET,
      DEFAULT_FORMAT_TENSOR_CONVERTER);
  caps = gst_caps_from_string (str_caps);
  g_object_set (G_OBJECT (pipeline_cntnr->nn_tflite_capsfilter), "caps", caps,
      NULL);
  g_free (str_caps);
  gst_caps_unref (caps);

  gst_bin_add_many (GST_BIN (pipeline), pipeline_cntnr->nn_tflite_queue,
      pipeline_cntnr->nn_tflite_videoscale,
      pipeline_cntnr->nn_tflite_videoconvert,
      pipeline_cntnr->nn_tflite_capsfilter,
      pipeline_cntnr->nn_tflite_tensor_converter,
      pipeline_cntnr->nn_tflite_tensor_filter,
      pipeline_cntnr->nn_tflite_tensor_sink, NULL);

  ret = gst_element_link_many (pipeline_cntnr->nn_tflite_queue,
      pipeline_cntnr->nn_tflite_videoscale,
      pipeline_cntnr->nn_tflite_videoconvert,
      pipeline_cntnr->nn_tflite_capsfilter,
      pipeline_cntnr->nn_tflite_tensor_converter,
      pipeline_cntnr->nn_tflite_tensor_filter,
      pipeline_cntnr->nn_tflite_tensor_sink, NULL);
  if (ret == FALSE) {
    g_printerr ("ERR: cannot link one (or more) of the elements "
        "which the application pipeline consists of\n");
    return ret;
  }

  pad = gst_element_get_static_pad (pipeline_cntnr->nn_tflite_queue, "sink");
  if (gst_pad_link (ctx->tee_nn_line_pad, pad) != GST_PAD_LINK_OK) {
    g_printerr ("ERR: cannot link the pad %s of %s to the pad %s of %s\n: ",
        gst_pad_get_name (ctx->tee_nn_line_pad), NAME_V4L2_PIPELINE_TEE,
        gst_pad_get_name (pad), NAME_NN_TFLITE_PIPELINE_QUEUE);
  }
  gst_object_unref (pad);

  return TRUE;
}

/**
 * @brief Load model-specific files (or information) and initialize the application context using it
 *
 * @param ctx a pointer of the application context data
 * @return none
 */
static void
_load_model_specific (nnstrmr_app_context_t * ctx)
{
  switch (ctx->nn_tensorfilter_desc) {
    case TF_LITE_MOBINET:
    {
      FILE *fp;
      char *eachline;
      size_t readcnt, len;
      gchar *path_label = g_strconcat (DEFAULT_PATH_MODEL_TENSOR_FILTER,
          NAME_LIST_OF_MISC_FILE_TENSOR_FILTER[ctx->nn_tensorfilter_desc],
          NULL);

      ctx->tflite_mobinet_info.labels = NULL;

      fp = fopen (path_label, "r");
      g_free (path_label);
      len = 0;
      if (fp != NULL) {
        while ((readcnt = getline (&eachline, &len, fp)) != -1) {
          ctx->tflite_mobinet_info.labels =
              g_list_append (ctx->tflite_mobinet_info.labels,
              g_strndup (eachline, readcnt));
        }
        fclose (fp);
      } else {
        g_printerr
            ("ERR: failed to load the model specific files for MOBINET with Tensowflow-lite: %s\n",
            NAME_LIST_OF_MISC_FILE_TENSOR_FILTER[TF_LITE_MOBINET]);
        return;
      }
      break;
    }
    default:
    {
      g_printerr ("ERR: undefined tensor_filter model description\n");
    }
  }
}

/**
 * @brief Finalize the model specific information in the application context data
 *
 * @param ctx a pointer of the application context data
 * @return none
 */
static void
_cleanup_model_specific (nnstrmr_app_context_t * ctx)
{
  switch (ctx->nn_tensorfilter_desc) {
    case TF_LITE_MOBINET:
    {
      if (ctx->tflite_mobinet_info.labels != NULL) {
        g_list_free_full (ctx->tflite_mobinet_info.labels, free);
      }
      break;
    }
    default:
    {
      g_printerr ("ERR: undefined tensor_filter model description\n");
    }
  }

}

/**
 * @brief A callback function to block the src pad of tee in the input and output pipeline
 *
 * In order to handle the command line option, --nnline-only, this probe function for the src pad
 * of tee blocks the input and output pipeline and dynamically unlinks the output pipeline from the whole pipeline.
 *
 * @param ctx a pointer of the application context data
 * @return GST_PAD_PROBE_REMOVE
 */
static GstPadProbeReturn
_cb_probe_tee_output_line_pad (GstPad * pad, GstPadProbeInfo * info,
    gpointer user_data)
{
  nnstrmr_app_context_t *ctx = (nnstrmr_app_context_t *) user_data;

  gst_element_set_state (ctx->pipeline, GST_STATE_PAUSED);

  switch (ctx->input_src) {
    case CAM_SRC:
    {
      GstElement *output_queue =
          ctx->pipeline_container.v4l2src_pipeline_container.output_queue;
      GstElement *output_textoverlay =
          ctx->pipeline_container.v4l2src_pipeline_container.output_textoverlay;
      GstElement *output_sink =
          ctx->pipeline_container.v4l2src_pipeline_container.output_sink;
      GstPad *sinkpad_output_queue =
          gst_element_get_static_pad (output_queue, "sink");

      /* Unlink sinkpad of queue on output line from tee on input source line */
      gst_pad_unlink (ctx->tee_output_line_pad, sinkpad_output_queue);
      gst_object_unref (sinkpad_output_queue);

      gst_element_set_state (output_queue, GST_STATE_NULL);
      gst_element_set_state (output_textoverlay, GST_STATE_NULL);
      gst_element_set_state (output_sink, GST_STATE_NULL);
      gst_element_unlink_many (output_queue, output_textoverlay, output_sink,
          NULL);
      gst_bin_remove_many (GST_BIN (ctx->pipeline), output_queue,
          output_textoverlay, output_sink, NULL);
      break;
    }
    case FILE_SRC:
    {
      /* TODO */
      break;
    }
    default:
    {
      g_printerr ("ERR: undefined input source\n");
    }
  }

  gst_element_set_state (ctx->pipeline, GST_STATE_PLAYING);

  return GST_PAD_PROBE_REMOVE;
}

/**
 * @brief A signal handler for 'new-data' emitted by 'tensor-sink'
 *
 * The suffix _output means that this handler is registered at the output pipeline.
 * Extracting the clasification result made by the mobinet tensorflow-lite model and
 * feeding it to the 'textoverlay' GstElement are done here.
 *
 * @param object a pointer of the 'tensor-sink' GstElement
 * @buffer buffer a pointer of the buffer in the 'tensor-sink' GstElement
 * @buffer user_data a pointer of the application context data
 * @return none
 */
static void
_handle_tensor_sink_new_data_output (GstElement * object, GstBuffer * buffer,
    gpointer user_data)
{
  nnstrmr_app_context_t *ctx = (nnstrmr_app_context_t *) user_data;
  switch (ctx->input_src) {
    case CAM_SRC:
    {
      v4l2src_pipeline_container_t *v4l2src_pipeline_cntnr =
          &((ctx->pipeline_container).v4l2src_pipeline_container);
      GstMemory *mem = gst_buffer_get_all_memory (buffer);
      GstMapInfo map_info;

      if (gst_memory_map (mem, &map_info, GST_MAP_READ)) {
        int max_score_idx = -1;
        guint8 max_score = 0;
        int i;
        gchar *class_result;

        for (i = 0; i < map_info.size; i++) {
          if ((guint8) map_info.data[i] > max_score) {
            max_score = (guint8) map_info.data[i];
            max_score_idx = i;
          }
        }
        class_result = "UNKNOWN";
        if (max_score_idx != -1) {
          class_result =
              (gchar *) g_list_nth_data (ctx->tflite_mobinet_info.labels,
              max_score_idx);
        }
        g_object_set (G_OBJECT (v4l2src_pipeline_cntnr->output_textoverlay),
            "text", class_result, NULL);
        gst_memory_unmap (mem, &map_info);
      }

    }
    default:
    {
      /* Do nothing */
    }
  }
}

/**
 * @brief A helper function for registering signal handlers at the output pipeline side
 *
 * @param ctx a pointer of the application context data
 * @return none
 */
static void
_register_signals_output (nnstrmr_app_context_t * ctx)
{
  guint signal_id;
  nn_tflite_pipeline_container_t *nn_pipeline_cntnr =
      &((ctx->pipeline_container).nn_tflite_pipeline_container);

  signal_id =
      g_signal_connect (G_OBJECT (nn_pipeline_cntnr->nn_tflite_tensor_sink),
      "new-data", G_CALLBACK (_handle_tensor_sink_new_data_output), ctx);
  g_mutex_lock (&ctx->signals_mutex);
  ctx->signals_connected[ctx->signal_idx++] = signal_id;
  g_mutex_unlock (&ctx->signals_mutex);
}

/**
 * @brief A helper function for registering signal handlers at the nnstreamer pipeline side
 *
 * @param ctx a pointer of the application context data
 * @return none
 */
static void
_register_signals_nn (nnstrmr_app_context_t * ctx)
{
  guint signal_id;
  nn_tflite_pipeline_container_t *nn_pipeline_cntnr =
      &((ctx->pipeline_container).nn_tflite_pipeline_container);

  signal_id =
      g_signal_connect (G_OBJECT (nn_pipeline_cntnr->nn_tflite_tensor_sink),
      "new-data", G_CALLBACK (_handle_tensor_sink_new_data_nn), ctx);
  g_mutex_lock (&ctx->signals_mutex);
  ctx->signals_connected[ctx->signal_idx++] = signal_id;
  g_mutex_unlock (&ctx->signals_mutex);
}

/**
 * @brief A helper function for unregistering all signal handlers and finalizing them
 *
 * @param ctx a pointer of the application context data
 * @return none
 */
static void
_unregister_signals (nnstrmr_app_context_t * ctx)
{
  int i;
  nn_tflite_pipeline_container_t *nn_pipeline_cntnr =
      &((ctx->pipeline_container).nn_tflite_pipeline_container);

  for (i = 0; i < ctx->signal_idx; i++) {
    g_signal_handler_disconnect (G_OBJECT
        (nn_pipeline_cntnr->nn_tflite_tensor_sink), ctx->signals_connected[i]);
  }
}

/**
 * @brief Main function.
 */
int
main (int argc, char *argv[])
{
  nnstrmr_app_context_t app_ctx;
  gboolean ret;
  GstBus *bus;
  guint bus_watch_id;

  /* Initailization */
  gst_init (&argc, &argv);
  app_ctx.mainloop = g_main_loop_new (NULL, FALSE);
  _set_and_parse_option_info (argc, argv, &app_ctx);

  app_ctx.signal_idx = 0;
  /* This is not mandatory porcedure */
  g_mutex_init (&app_ctx.signals_mutex);

  app_ctx.time_last_profile = GST_CLOCK_TIME_NONE;
  app_ctx.time_pipeline_start = GST_CLOCK_TIME_NONE;

  /* Create gstreamer elements */
  app_ctx.pipeline = gst_pipeline_new (NAME_APP_PIPELINE);

  if (!app_ctx.pipeline) {
    g_printerr ("ERR: cannot create the application pipeline, %s\n",
        NAME_APP_PIPELINE);
    g_main_loop_unref (app_ctx.mainloop);
    return -1;
  }

  _load_model_specific (&app_ctx);

  switch (app_ctx.input_src) {
    case CAM_SRC:
    {
      /* Set up the pipeline */
      ret = _construct_v4l2src_pipeline (&app_ctx);
      if (ret == FALSE) {
        goto common_cleanup;
      }
      break;
    }
    case FILE_SRC:
    {
      /* TODO */
      _construct_filesrc_pipeline (&app_ctx);
      break;
    }
    default:
    {
      g_printerr ("ERR: undefined input source\n");
    }
  }

  _construct_nn_tflite_pipeline (&app_ctx);

  /**
   * When the --nnline-only command line option is provided, the output pipeline
   * is dynamically unlinked from the whole pipeline.
   */
  if (app_ctx.flag_nnline_only) {
    gst_pad_add_probe (app_ctx.tee_output_line_pad, GST_PAD_PROBE_TYPE_BLOCK,
        _cb_probe_tee_output_line_pad, &app_ctx, NULL);
  } else {
    _register_signals_output (&app_ctx);
  }
  _register_signals_nn (&app_ctx);

  /* Add a bus watcher */
  bus = gst_pipeline_get_bus (GST_PIPELINE (app_ctx.pipeline));
  bus_watch_id = gst_bus_add_watch (bus, _cb_bus_watch, &app_ctx);
  gst_object_unref (bus);

  /* Set the pipeline to "playing" state */
  gst_element_set_state (app_ctx.pipeline, GST_STATE_PLAYING);

  /* Run the main loop */
  g_main_loop_run (app_ctx.mainloop);

  /* Out of the main loop, clean up */
  g_source_remove (bus_watch_id);
  gst_object_unref (app_ctx.tee_output_line_pad);
  gst_object_unref (app_ctx.tee_nn_line_pad);

common_cleanup:
  _unregister_signals (&app_ctx);
  _cleanup_model_specific (&app_ctx);
  gst_element_set_state (app_ctx.pipeline, GST_STATE_NULL);
  gst_object_unref (GST_OBJECT (app_ctx.pipeline));
  g_main_loop_unref (app_ctx.mainloop);

  return 0;
}
