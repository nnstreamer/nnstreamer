/**
 * @file	unittest_sink.cpp
 * @date	29 June 2018
 * @brief	Unit test for tensor sink plugin
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs.
 */

#include <string.h>
#include <stdlib.h>
#include <gtest/gtest.h>
#include <glib/gstdio.h>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <tensor_common.h>

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
    _print_log ("test failed!! [line : %d]", __LINE__); \
    goto error; \
  }

gchar *custom_dir;

/**
 * @brief Current status.
 */
typedef enum
{
  TEST_START, /**< start to setup pipeline */
  TEST_INIT, /**< init done */
  TEST_ERR_MESSAGE, /**< received error message */
  TEST_STREAM, /**< stream started */
  TEST_EOS /**< end of stream */
} TestStatus;

/**
 * @brief Test type.
 */
typedef enum
{
  TEST_TYPE_VIDEO_RGB, /**< pipeline for video (RGB) */
  TEST_TYPE_VIDEO_BGR, /**< pipeline for video (BGR) */
  TEST_TYPE_VIDEO_RGB_PADDING, /**< pipeline for video (RGB), remove padding */
  TEST_TYPE_VIDEO_BGR_PADDING, /**< pipeline for video (BGR), remove padding */
  TEST_TYPE_VIDEO_RGB_3F, /**< pipeline for video (RGB) 3 frames */
  TEST_TYPE_VIDEO_RGBA, /**< pipeline for video (RGBA) */
  TEST_TYPE_VIDEO_BGRA, /**< pipeline for video (BGRA) */
  TEST_TYPE_VIDEO_ARGB, /**< pipeline for video (ARGB) */
  TEST_TYPE_VIDEO_ABGR, /**< pipeline for video (ABGR) */
  TEST_TYPE_VIDEO_RGBx, /**< pipeline for video (RGBx) */
  TEST_TYPE_VIDEO_xRGB, /**< pipeline for video (xRGB) */
  TEST_TYPE_VIDEO_xBGR, /**< pipeline for video (xBGR) */
  TEST_TYPE_VIDEO_BGRx, /**< pipeline for video (BGRx) */
  TEST_TYPE_VIDEO_BGRx_2F, /**< pipeline for video (BGRx) 2 frames */
  TEST_TYPE_VIDEO_GRAY8, /**< pipeline for video (GRAY8) */
  TEST_TYPE_VIDEO_GRAY8_PADDING, /**< pipeline for video (GRAY8), remove padding */
  TEST_TYPE_VIDEO_GRAY8_3F_PADDING, /**< pipeline for video (GRAY8) 3 frames, remove padding */
  TEST_TYPE_AUDIO_S8, /**< pipeline for audio (S8) */
  TEST_TYPE_AUDIO_U8_100F, /**< pipeline for audio (U8) 100 frames */
  TEST_TYPE_AUDIO_S16, /**< pipeline for audio (S16) */
  TEST_TYPE_AUDIO_U16_1000F, /**< pipeline for audio (U16) 1000 frames */
  TEST_TYPE_AUDIO_S32, /**< pipeline for audio (S32) */
  TEST_TYPE_AUDIO_U32, /**< pipeline for audio (U32) */
  TEST_TYPE_AUDIO_F32, /**< pipeline for audio (F32) */
  TEST_TYPE_AUDIO_F64, /**< pipeline for audio (F64) */
  TEST_TYPE_TEXT, /**< pipeline for text */
  TEST_TYPE_TEXT_3F, /**< pipeline for text 3 frames */
  TEST_TYPE_OCTET_CUR_TS, /**< pipeline for octet stream, timestamp current time */
  TEST_TYPE_OCTET_RATE_TS, /**< pipeline for octet stream, timestamp framerate */
  TEST_TYPE_OCTET_VALID_TS, /**< pipeline for octet stream, valid timestamp */
  TEST_TYPE_OCTET_INVALID_TS, /**< pipeline for octet stream, invalid timestamp */
  TEST_TYPE_OCTET_2F, /**< pipeline for octet stream, 2 frames */
  TEST_TYPE_TENSORS, /**< pipeline for tensors with tensor_mux */
  TEST_TYPE_TENSORS_MIX, /**< pipeline for tensors with tensor_mux, tensor_demux */
  TEST_TYPE_CUSTOM_TENSOR, /**< pipeline for single tensor with passthrough custom filter */
  TEST_TYPE_CUSTOM_TENSORS, /**< pipeline for tensors with passthrough custom filter */
  TEST_TYPE_CUSTOM_BUF_DROP, /**< pipeline to test buffer-drop in tensor_filter using custom filter */
  TEST_TYPE_NEGO_FAILED, /**< pipeline to test caps negotiation */
  TEST_TYPE_VIDEO_RGB_SPLIT, /**< pipeline to test tensor_split */
  TEST_TYPE_VIDEO_RGB_AGGR_1, /**< pipeline to test tensor_aggregator (change dimension index 3 : 1 > 10)*/
  TEST_TYPE_VIDEO_RGB_AGGR_2, /**< pipeline to test tensor_aggregator (change dimension index 1 : 160 > 1600) */
  TEST_TYPE_VIDEO_RGB_AGGR_3, /**< pipeline to test tensor_aggregator (test to get frames with the property concat) */
  TEST_TYPE_AUDIO_S16_AGGR, /**< pipeline to test tensor_aggregator */
  TEST_TYPE_AUDIO_U16_AGGR, /**< pipeline to test tensor_aggregator */
  TEST_TYPE_TYPECAST, /**< pipeline for typecast with tensor_transform */
  TEST_TYPE_ISSUE739_MUX_PARALLEL_1, /**< pipeline to test Mux/Parallel case in #739 */
  TEST_TYPE_ISSUE739_MUX_PARALLEL_2, /**< pipeline to test Mux/Parallel case in #739 */
  TEST_TYPE_ISSUE739_MUX_PARALLEL_3, /**< pipeline to test Mux/Parallel case in #739 */
  TEST_TYPE_ISSUE739_MUX_PARALLEL_4, /**< pipeline to test Mux/Parallel case in #739 */
  TEST_TYPE_ISSUE739_MERGE_PARALLEL_1, /**< pipeline to test Merge/Parallel case in #739 */
  TEST_TYPE_ISSUE739_MERGE_PARALLEL_2, /**< pipeline to test Merge/Parallel case in #739 */
  TEST_TYPE_ISSUE739_MERGE_PARALLEL_3, /**< pipeline to test Merge/Parallel case in #739 */
  TEST_TYPE_ISSUE739_MERGE_PARALLEL_4, /**< pipeline to test Merge/Parallel case in #739 */
  TEST_TYPE_DECODER_PROPERTY, /**< pipeline to test get/set_property of decoder */
  TEST_TYPE_UNKNOWN /**< unknonwn */
} TestType;

/**
 * @brief Test options.
 */
typedef struct
{
  guint num_buffers; /**< count of buffers */
  TestType test_type; /**< test pipeline */
  tensor_type t_type; /**< tensor type */
  char *tmpfile; /**< tmpfile to write */
} TestOption;

/**
 * @brief Data structure for test.
 */
typedef struct
{
  GMainLoop *loop; /**< main event loop */
  GstElement *pipeline; /**< gst pipeline for test */
  GstBus *bus; /**< gst bus for test */
  GstElement *sink; /**< tensor sink element */
  TestStatus status; /**< current status */
  TestType tc_type; /**< pipeline for testcase type */
  tensor_type t_type; /**< tensor type */
  guint received; /**< received buffer count */
  guint mem_blocks; /**< memory blocks in received buffer */
  gsize received_size; /**< received buffer size */
  gboolean invalid_timestamp; /**< flag to check timestamp */
  gboolean test_failed; /**< flag to indicate error */
  gboolean start; /**< stream started (for tensor_sink signal) */
  gboolean end; /**< eos reached (for tensor_sink signal) */
  GstCaps *current_caps; /**< negotiated caps */
  gchar *caps_name; /**< negotiated caps name */
  GstTensorConfig tensor_config; /**< tensor config from negotiated caps */
  GstTensorsConfig tensors_config; /**< tensors config from negotiated caps */
} TestData;

/**
 * @brief Data for pipeline and test result.
 */
static TestData g_test_data;

/**
 * @brief Free resources in test data.
 */
static void
_free_test_data (void)
{
  if (g_test_data.loop) {
    g_main_loop_unref (g_test_data.loop);
    g_test_data.loop = NULL;
  }

  if (g_test_data.bus) {
    gst_bus_remove_signal_watch (g_test_data.bus);
    gst_object_unref (g_test_data.bus);
    g_test_data.bus = NULL;
  }

  if (g_test_data.sink) {
    gst_object_unref (g_test_data.sink);
    g_test_data.sink = NULL;
  }

  if (g_test_data.pipeline) {
    gst_object_unref (g_test_data.pipeline);
    g_test_data.pipeline = NULL;
  }

  if (g_test_data.current_caps) {
    gst_caps_unref (g_test_data.current_caps);
    g_test_data.current_caps = NULL;
  }

  g_free (g_test_data.caps_name);
}

/**
 * @brief Callback for message.
 */
static void
_message_cb (GstBus * bus, GstMessage * message, gpointer user_data)
{
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_ERROR:
    case GST_MESSAGE_WARNING:
      _print_log ("received error message");
      g_test_data.status = TEST_ERR_MESSAGE;
      g_test_data.test_failed = TRUE;
      g_main_loop_quit (g_test_data.loop);
      break;

    case GST_MESSAGE_EOS:
      _print_log ("received eos message");
      g_test_data.status = TEST_EOS;
      g_main_loop_quit (g_test_data.loop);
      break;

    case GST_MESSAGE_STREAM_START:
      _print_log ("received start message");
      g_test_data.status = TEST_STREAM;
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
  gsize buf_size;
  guint mem_blocks;

  if (!GST_IS_BUFFER (buffer)) {
    _print_log ("received invalid buffer");
    g_test_data.test_failed = TRUE;
    return;
  }

  buf_size = gst_buffer_get_size (buffer);
  mem_blocks = gst_buffer_n_memory (buffer);

  if (g_test_data.received > 0) {
    if (g_test_data.mem_blocks != mem_blocks) {
      _print_log ("invalid memory, old[%d] new[%d]", g_test_data.mem_blocks,
          mem_blocks);
      g_test_data.test_failed = TRUE;
    }

    if (g_test_data.received_size != buf_size) {
      _print_log ("invalid size, old[%zd] new[%zd]", g_test_data.received_size,
          buf_size);
      g_test_data.test_failed = TRUE;
    }
  }

  if (DBG) {
    _print_log ("pts %" GST_TIME_FORMAT,
        GST_TIME_ARGS (GST_BUFFER_PTS (buffer)));
    _print_log ("dts %" GST_TIME_FORMAT,
        GST_TIME_ARGS (GST_BUFFER_DTS (buffer)));
    _print_log ("duration %" GST_TIME_FORMAT,
        GST_TIME_ARGS (GST_BUFFER_DURATION (buffer)));
  }

  /** check timestamp */
  if (!GST_CLOCK_TIME_IS_VALID (GST_BUFFER_DTS_OR_PTS (buffer))) {
    g_test_data.invalid_timestamp = TRUE;
  }

  g_test_data.received++;
  g_test_data.received_size = buf_size;
  g_test_data.mem_blocks = mem_blocks;

  _print_log ("new data callback [%d] size [%zd]",
      g_test_data.received, g_test_data.received_size);

  if (g_test_data.caps_name == NULL) {
    GstPad *sink_pad;
    GstCaps *caps;
    GstStructure *structure;

    /** get negotiated caps */
    sink_pad = gst_element_get_static_pad (element, "sink");
    caps = gst_pad_get_current_caps (sink_pad);
    structure = gst_caps_get_structure (caps, 0);

    g_test_data.caps_name = g_strdup (gst_structure_get_name (structure));
    _print_log ("caps name [%s]", g_test_data.caps_name);

    if (g_str_equal (g_test_data.caps_name, "other/tensor")) {
      if (!gst_tensor_config_from_structure (&g_test_data.tensor_config,
              structure)) {
        _print_log ("failed to get tensor config from caps");
        g_test_data.test_failed = TRUE;
      }
    } else if (g_str_equal (g_test_data.caps_name, "other/tensors")) {
      if (!gst_tensors_config_from_structure (&g_test_data.tensors_config,
              structure)) {
        _print_log ("failed to get tensors config from caps");
        g_test_data.test_failed = TRUE;
      }
    }

    /** copy current caps */
    g_test_data.current_caps = gst_caps_copy (caps);
    gst_caps_unref (caps);
  }
}

/**
 * @brief Callback for signal stream-start.
 */
static void
_stream_start_cb (GstElement * element, gpointer user_data)
{
  g_test_data.start = TRUE;
  _print_log ("stream start callback");
}

/**
 * @brief Callback for signal eos.
 */
static void
_eos_cb (GstElement * element, gpointer user_data)
{
  g_test_data.end = TRUE;
  _print_log ("eos callback");
}

/**
 * @brief Push text data to appsrc for text utf8 type.
 */
static gboolean
_push_text_data (const guint num_buffers, const gboolean timestamps = TRUE)
{
  GstElement *appsrc;
  gboolean failed = FALSE;
  guint i;

  appsrc = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "appsrc");

  for (i = 0; i < num_buffers; i++) {
    GstBuffer *buf = gst_buffer_new_allocate (NULL, 10, NULL);
    GstMapInfo info;

    gst_buffer_map (buf, &info, GST_MAP_WRITE);
    snprintf ((char *) info.data, 10, "%d", i);
    gst_buffer_unmap (buf, &info);

    if (timestamps) {
      GST_BUFFER_PTS (buf) = (i + 1) * 10 * GST_MSECOND;
      GST_BUFFER_DURATION (buf) = 10 * GST_MSECOND;
    }

    if (gst_app_src_push_buffer (GST_APP_SRC (appsrc), buf) != GST_FLOW_OK) {
      _print_log ("failed to push buffer [%d]", i);
      g_test_data.test_failed = failed = TRUE;
      goto error;
    }
  }

  if (gst_app_src_end_of_stream (GST_APP_SRC (appsrc)) != GST_FLOW_OK) {
    _print_log ("failed to set eos");
    g_test_data.test_failed = failed = TRUE;
    goto error;
  }

error:
  gst_object_unref (appsrc);
  return !failed;
}

/**
 * @brief Prepare test pipeline.
 */
static gboolean
_setup_pipeline (TestOption & option)
{
  gchar *str_pipeline;
  gulong handle_id;

  g_test_data.status = TEST_START;
  g_test_data.received = 0;
  g_test_data.mem_blocks = 0;
  g_test_data.received_size = 0;
  g_test_data.invalid_timestamp = FALSE;
  g_test_data.test_failed = FALSE;
  g_test_data.start = FALSE;
  g_test_data.end = FALSE;
  g_test_data.current_caps = NULL;
  g_test_data.caps_name = NULL;
  g_test_data.tc_type = option.test_type;
  g_test_data.t_type = option.t_type;
  gst_tensor_config_init (&g_test_data.tensor_config);
  gst_tensors_config_init (&g_test_data.tensors_config);

  _print_log ("option num_buffers[%d] test_type[%d]",
      option.num_buffers, option.test_type);

  g_test_data.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_test_data.loop != NULL);

  switch (option.test_type) {
    case TEST_TYPE_VIDEO_RGB:
      /** video 160x120 RGB */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_BGR:
      /** video 160x120 BGR */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=BGR,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_RGB_PADDING:
      /** video 162x120 RGB, remove padding */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=RGB,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_BGR_PADDING:
      /** video 162x120 BGR, remove padding */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=BGR,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_RGB_3F:
      /** video 160x120 RGB, 3 frames */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! "
          "tensor_converter frames-per-tensor=3 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_RGBA:
      /** video 162x120 RGBA */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=RGBA,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_BGRA:
      /** video 162x120 BGRA */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=BGRA,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_ARGB:
      /** video 162x120 ARGB */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=ARGB,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_ABGR:
      /** video 162x120 ABGR */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=ABGR,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_RGBx:
      /** video 162x120 RGBx */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=RGBx,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_xRGB:
      /** video 162x120 xRGB */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=xRGB,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_xBGR:
      /** video 162x120 xBGR */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=xBGR,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_BGRx:
      /** video 162x120 BGRx */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=BGRx,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_BGRx_2F:
      /** video 160x120 BGRx, 2 frames */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=BGRx,framerate=(fraction)30/1 ! "
          "tensor_converter frames-per-tensor=2 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_GRAY8:
      /** video 160x120 GRAY8 */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=GRAY8,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_GRAY8_PADDING:
      /** video 162x120 GRAY8, remove padding */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=GRAY8,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_GRAY8_3F_PADDING:
      /** video 162x120 GRAY8, 3 frames, remove padding */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=GRAY8,framerate=(fraction)30/1 ! "
          "tensor_converter frames-per-tensor=3 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_AUDIO_S8:
      /** audio sample rate 16000 (8 bits, signed, little endian) */
      str_pipeline =
          g_strdup_printf
          ("audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=S8,rate=16000 ! "
          "tensor_converter frames-per-tensor=500 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_AUDIO_U8_100F:
      /** audio sample rate 16000 (8 bits, unsigned, little endian), 100 frames */
      str_pipeline =
          g_strdup_printf
          ("audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=U8,rate=16000 ! "
          "tensor_converter frames-per-tensor=100 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_AUDIO_S16:
      /** audio sample rate 16000 (16 bits, signed, little endian) */
      str_pipeline =
          g_strdup_printf
          ("audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=S16LE,rate=16000 ! "
          "tensor_converter frames-per-tensor=500 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_AUDIO_U16_1000F:
      /** audio sample rate 16000 (16 bits, unsigned, little endian), 1000 frames */
      str_pipeline =
          g_strdup_printf
          ("audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=U16LE,rate=16000 ! "
          "tensor_converter frames-per-tensor=1000 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_AUDIO_S32:
      /** audio sample rate 44100 (32 bits, signed, little endian) */
      str_pipeline =
          g_strdup_printf
          ("audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=S32LE,rate=44100 ! "
          "tensor_converter frames-per-tensor=500 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_AUDIO_U32:
      /** audio sample rate 44100 (32 bits, unsigned, little endian) */
      str_pipeline =
          g_strdup_printf
          ("audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=U32LE,rate=44100 ! "
          "tensor_converter frames-per-tensor=500 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_AUDIO_F32:
      /** audio sample rate 44100 (32 bits, floating point, little endian) */
      str_pipeline =
          g_strdup_printf
          ("audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=F32LE,rate=44100 ! "
          "tensor_converter frames-per-tensor=500 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_AUDIO_F64:
      /** audio sample rate 44100 (64 bits, floating point, little endian) */
      str_pipeline =
          g_strdup_printf
          ("audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=F64LE,rate=44100 ! "
          "tensor_converter frames-per-tensor=500 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_TEXT:
      /** text stream */
      str_pipeline =
          g_strdup_printf
          ("appsrc name=appsrc caps=text/x-raw,format=utf8 ! "
          "tensor_converter input-dim=20 ! tensor_sink name=test_sink");
      break;
    case TEST_TYPE_TEXT_3F:
      /** text stream 3 frames */
      str_pipeline =
          g_strdup_printf
          ("appsrc name=appsrc caps=text/x-raw,format=utf8,framerate=(fraction)100/1 ! "
          "tensor_converter name=convert input-dim=30 frames-per-tensor=3 ! tensor_sink name=test_sink");
      break;
    case TEST_TYPE_OCTET_CUR_TS:
      /** byte stream, timestamp current time */
      str_pipeline =
          g_strdup_printf
          ("appsrc name=appsrc caps=application/octet-stream ! "
          "tensor_converter input-dim=1:10 input-type=uint8 ! tensor_sink name=test_sink");
      break;
    case TEST_TYPE_OCTET_RATE_TS:
      /** byte stream, timestamp framerate */
      str_pipeline =
          g_strdup_printf
          ("appsrc name=appsrc caps=application/octet-stream,framerate=(fraction)50/1 ! "
          "tensor_converter input-dim=1:10 input-type=uint8 ! tensor_sink name=test_sink");
      break;
    case TEST_TYPE_OCTET_VALID_TS:
      /** byte stream, send buffer with valid timestamp */
      str_pipeline =
          g_strdup_printf
          ("appsrc name=appsrc caps=application/octet-stream ! "
          "tensor_converter name=convert input-dim=1:10 input-type=uint8 set-timestamp=false ! tensor_sink name=test_sink");
      break;
    case TEST_TYPE_OCTET_INVALID_TS:
      /** byte stream, send buffer with invalid timestamp */
      str_pipeline =
          g_strdup_printf
          ("appsrc name=appsrc caps=application/octet-stream ! "
          "tensor_converter name=convert input-dim=1:10 input-type=uint8 set-timestamp=false ! tensor_sink name=test_sink");
      break;
    case TEST_TYPE_OCTET_2F:
      /** byte stream, 2 frames */
      str_pipeline =
          g_strdup_printf
          ("appsrc name=appsrc caps=application/octet-stream,framerate=(fraction)100/1 ! "
          "tensor_converter name=convert input-dim=1:5 input-type=int8 ! tensor_sink name=test_sink");
      break;
    case TEST_TYPE_TENSORS:
      /** other/tensors with tensor_mux */
      str_pipeline =
          g_strdup_printf
          ("tensor_mux name=mux ! tensor_sink name=test_sink "
          "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
          "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_1",
          option.num_buffers, option.num_buffers);
      break;
    case TEST_TYPE_TENSORS_MIX:
      /** other/tensors with tensor_mux, tensor_demux */
      str_pipeline =
          g_strdup_printf
          ("tensor_mux name=mux synch=false ! tensor_demux name=demux "
          "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
          "audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! tensor_converter frames-per-tensor=500 ! mux.sink_1 "
          "videotestsrc num-buffers=%d ! video/x-raw,width=64,height=48,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_2 "
          "demux.src_0 ! queue ! tensor_sink "
          "demux.src_1 ! queue ! tensor_sink name=test_sink "
          "demux.src_2 ! queue ! tensor_sink",
          option.num_buffers, option.num_buffers * 3, option.num_buffers + 3);
      break;
    case TEST_TYPE_CUSTOM_TENSOR:
      /** video 160x120 RGB, passthrough custom filter */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_filter framework=custom model=%s/libnnstreamer_customfilter_passthrough_variable.so ! tensor_sink name=test_sink",
	   option.num_buffers, custom_dir? custom_dir : "./nnstreamer_example/custom_example_passthrough");
      break;
    case TEST_TYPE_CUSTOM_TENSORS:
      /** other/tensors with tensormux, passthrough custom filter */
      str_pipeline =
          g_strdup_printf
          ("tensor_mux name=mux ! tensor_filter framework=custom model=%s/libnnstreamer_customfilter_passthrough_variable.so ! tensor_sink name=test_sink "
          "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
          "videotestsrc num-buffers=%d ! video/x-raw,width=120,height=80,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_1 "
          "videotestsrc num-buffers=%d ! video/x-raw,width=64,height=48,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_2",
	   custom_dir? custom_dir : "./nnstreamer_example/custom_example_passthrough" , option.num_buffers, option.num_buffers, option.num_buffers);
      break;
    case TEST_TYPE_CUSTOM_BUF_DROP:
      /* audio stream to test buffer-drop using custom filter */
      str_pipeline =
          g_strdup_printf
          ("audiotestsrc num-buffers=%d samplesperbuffer=200 ! audioconvert ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! "
          "tensor_converter frames-per-tensor=200 ! tensor_filter framework=custom model=%s/libnnscustom_drop_buffer.so ! tensor_sink name=test_sink",
	   option.num_buffers, custom_dir? custom_dir :"./tests");
      break;
    case TEST_TYPE_NEGO_FAILED:
      /** caps negotiation failed */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! "
          "videoconvert ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_RGB_SPLIT:
      /** video stream with tensor_split */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_split silent=TRUE name=split tensorseg=1:160:120,1:160:120,1:160:120 tensorpick=0,1,2 "
          "split.src_0 ! queue ! tensor_sink "
          "split.src_1 ! queue ! tensor_sink name=test_sink "
          "split.src_2 ! queue ! tensor_sink", option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_RGB_AGGR_1:
      /** video stream with tensor_aggregator */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_aggregator frames-out=10 frames-flush=5 frames-dim=3 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_RGB_AGGR_2:
      /** video stream with tensor_aggregator */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_aggregator frames-out=10 frames-flush=5 frames-dim=1 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_VIDEO_RGB_AGGR_3:
      /** video stream with tensor_aggregator */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=64,height=48,format=RGB,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_aggregator frames-out=10 frames-dim=1 concat=false ! "
          "tensor_aggregator frames-in=10 frames-out=8 frames-flush=10 frames-dim=1 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_AUDIO_S16_AGGR:
      /** audio stream with tensor_aggregator, 4 buffers with 2000 frames */
      str_pipeline =
          g_strdup_printf
          ("audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! "
          "tensor_converter frames-per-tensor=500 ! tensor_aggregator frames-in=500 frames-out=2000 frames-dim=1 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_AUDIO_U16_AGGR:
      /** audio stream with tensor_aggregator, divided into 5 buffers with 100 frames */
      str_pipeline =
          g_strdup_printf
          ("audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=U16LE,rate=16000,channels=1 ! "
          "tensor_converter frames-per-tensor=500 ! tensor_aggregator frames-in=500 frames-out=100 frames-dim=1 ! tensor_sink name=test_sink",
          option.num_buffers);
      break;
    case TEST_TYPE_TYPECAST:
      /** text stream to test typecast */
      str_pipeline =
          g_strdup_printf
          ("appsrc name=appsrc caps=text/x-raw,format=utf8 ! "
          "tensor_converter input-dim=10 ! tensor_transform mode=typecast option=%s ! tensor_sink name=test_sink",
          tensor_element_typename[option.t_type]);
      break;
    case TEST_TYPE_ISSUE739_MUX_PARALLEL_1:
      /** 4x4 tensor stream, different FPS, tensor_mux them @ slowest */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! mux.sink_0 "
          "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=25/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! mux.sink_1 "
          "tensor_mux sync_mode=slowest name=mux ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! tee name=t ! queue ! tensor_sink sync=true name=test_sink t. ! queue ! filesink location=%s",
	   option.num_buffers * 10, custom_dir? custom_dir : "./tests", option.num_buffers * 25, custom_dir? custom_dir : "./tests", custom_dir? custom_dir : "./tests",  option.tmpfile);
      break;
    case TEST_TYPE_ISSUE739_MUX_PARALLEL_2:
      /** 4x4 tensor stream, different FPS, tensor_mux them @ basepad*/
      str_pipeline =
          g_strdup_printf
          ("videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! mux.sink_0 "
          "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=25/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! mux.sink_1 "
          "tensor_mux sync_mode=basepad sync_option=0:0 name=mux ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! tee name=t ! queue ! tensor_sink sync=true name=test_sink t. ! queue ! filesink location=%s",
	   option.num_buffers * 10, custom_dir? custom_dir : "./tests", option.num_buffers * 25, custom_dir? custom_dir : "./tests", custom_dir? custom_dir : "./tests", option.tmpfile);
      break;
    case TEST_TYPE_ISSUE739_MUX_PARALLEL_3:
      /** 4x4 tensor stream, different FPS, tensor_mux them @ basepad*/
      str_pipeline =
          g_strdup_printf
          ("videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! mux.sink_0 "
          "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=25/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! mux.sink_1 "
          "tensor_mux sync_mode=basepad sync_option=1:0 name=mux ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! tee name=t ! queue ! tensor_sink sync=true name=test_sink t. ! queue ! filesink location=%s",
	   option.num_buffers * 10, custom_dir? custom_dir : "./tests", option.num_buffers * 25, custom_dir? custom_dir : "./tests", custom_dir? custom_dir : "./tests",  option.tmpfile);
      break;
    case TEST_TYPE_ISSUE739_MUX_PARALLEL_4:
      /** 4x4 tensor stream, different FPS, tensor_mux them @ basepad*/
      /** @todo Because of the bug mentioned in #739, this is not registered as gtest case, yet */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! mux.sink_0 "
          "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=25/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! mux.sink_1 "
          "tensor_mux sync_mode=basepad sync_option=1:1000000000 name=mux ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! tee name=t ! queue ! tensor_sink sync=true name=test_sink t. ! queue ! filesink location=%s",
	   option.num_buffers * 10, custom_dir? custom_dir : "./tests", option.num_buffers * 25, custom_dir? custom_dir : "./tests", custom_dir? custom_dir : "./tests",  option.tmpfile);
      break;
    case TEST_TYPE_ISSUE739_MERGE_PARALLEL_1:
      /** 4x4 tensor stream, different FPS, tensor_mux them @ slowest */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! mux.sink_0 "
          "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=25/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! mux.sink_1 "
          "tensor_merge mode=linear option=3 sync_mode=slowest name=mux ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! tee name=t ! queue ! tensor_sink sync=true name=test_sink t. ! queue ! filesink location=%s",
	   option.num_buffers * 10, custom_dir? custom_dir : "./tests", option.num_buffers * 25, custom_dir? custom_dir : "./tests", custom_dir? custom_dir : "./tests", option.tmpfile);
      break;
    case TEST_TYPE_ISSUE739_MERGE_PARALLEL_2:
      /** 4x4 tensor stream, different FPS, tensor_merge them @ basepad*/
      str_pipeline =
          g_strdup_printf
          ("videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! mux.sink_0 "
          "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=25/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! mux.sink_1 "
          "tensor_merge mode=linear option=3 sync_mode=basepad sync_option=0:0 name=mux ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! tee name=t ! queue ! tensor_sink sync=true name=test_sink t. ! queue ! filesink location=%s",
	   option.num_buffers * 10, custom_dir? custom_dir : "./tests", option.num_buffers * 25, custom_dir? custom_dir : "./tests", custom_dir? custom_dir : "./tests", option.tmpfile);
      break;
    case TEST_TYPE_ISSUE739_MERGE_PARALLEL_3:
      /** 4x4 tensor stream, different FPS, tensor_merge them @ basepad*/
      str_pipeline =
          g_strdup_printf
          ("videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! mux.sink_0 "
          "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=25/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! mux.sink_1 "
          "tensor_merge mode=linear option=3 sync_mode=basepad sync_option=1:0 name=mux ! tensor_filter framework=custom model=%s/libnnscustom_framecounter.so ! tee name=t ! queue ! tensor_sink sync=true name=test_sink t. ! queue ! filesink location=%s",
	   option.num_buffers * 10, custom_dir? custom_dir : "./tests", option.num_buffers * 25, custom_dir? custom_dir : "./tests", custom_dir? custom_dir : "./tests", option.tmpfile);
      break;
    /** @todo Add tensor_mux policy = more policies! */
    case TEST_TYPE_DECODER_PROPERTY:
      str_pipeline =
          g_strdup_printf
          ("videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_decoder mode=direct_video name=decoder option1=whatthehell option2=isgoingon option3=nothing option4=iswrong option5=keepcalm option6=\"and have a break\" option7=\"iwill=not\" option8=\"break=your\" option9=\"system=1234\" ! video/x-raw,format=BGRx ! tensor_converter ! tensor_sink name=test_sink ",
          option.num_buffers);
      break;
    default:
      goto error;
  }

  g_test_data.pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  _check_cond_err (g_test_data.pipeline != NULL);

  g_test_data.bus = gst_element_get_bus (g_test_data.pipeline);
  _check_cond_err (g_test_data.bus != NULL);

  gst_bus_add_signal_watch (g_test_data.bus);
  handle_id = g_signal_connect (g_test_data.bus, "message",
      (GCallback) _message_cb, NULL);
  _check_cond_err (handle_id > 0);

  g_test_data.sink =
      gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "test_sink");
  _check_cond_err (g_test_data.sink != NULL);

  if (DBG) {
    /** print logs */
    g_object_set (g_test_data.sink, "silent", (gboolean) FALSE, NULL);
  }

  /** signal for new data */
  handle_id = g_signal_connect (g_test_data.sink, "new-data",
      (GCallback) _new_data_cb, NULL);
  _check_cond_err (handle_id > 0);

  g_test_data.status = TEST_INIT;
  return TRUE;

error:
  g_test_data.test_failed = TRUE;
  _free_test_data ();
  return FALSE;
}

/**
 * @brief Get temp file name.
 * @return file name (should free string with g_free)
 */
static gchar *
_get_temp_filename (void)
{
  const gchar *tmp_dir;
  gchar *tmp_fn;
  gint fd;

  if ((tmp_dir = g_get_tmp_dir ()) == NULL) {
    _print_log ("failed to get tmp dir");
    return NULL;
  }

  tmp_fn = g_build_filename (tmp_dir, "nnstreamer_unittest_temp_XXXXXX", NULL);
  fd = g_mkstemp (tmp_fn);

  if (fd < 0) {
    _print_log ("failed to create temp file %s", tmp_fn);
    g_free (tmp_fn);
    return NULL;
  }

  g_close (fd, NULL);
  if (g_remove (tmp_fn) != 0) {
    _print_log ("failed to remove temp file %s", tmp_fn);
  }

  return tmp_fn;
}

/**
 * @brief Test for tensor sink properties.
 */
TEST (tensor_sink_test, properties)
{
  guint rate, res_rate;
  gint64 lateness, res_lateness;
  gboolean silent, res_silent;
  gboolean emit, res_emit;
  gboolean sync, res_sync;
  gboolean qos, res_qos;
  TestOption option = { 1, TEST_TYPE_VIDEO_RGB };

  ASSERT_TRUE (_setup_pipeline (option));

  /** default signal-rate is 0 */
  g_object_get (g_test_data.sink, "signal-rate", &rate, NULL);
  EXPECT_EQ (rate, 0);

  rate += 10;
  g_object_set (g_test_data.sink, "signal-rate", rate, NULL);
  g_object_get (g_test_data.sink, "signal-rate", &res_rate, NULL);
  EXPECT_EQ (res_rate, rate);

  /** default emit-signal is TRUE */
  g_object_get (g_test_data.sink, "emit-signal", &emit, NULL);
  EXPECT_EQ (emit, TRUE);

  g_object_set (g_test_data.sink, "emit-signal", !emit, NULL);
  g_object_get (g_test_data.sink, "emit-signal", &res_emit, NULL);
  EXPECT_EQ (res_emit, !emit);

  /** default silent is TRUE */
  g_object_get (g_test_data.sink, "silent", &silent, NULL);
  EXPECT_EQ (silent, (DBG) ? FALSE : TRUE);

  g_object_set (g_test_data.sink, "silent", !silent, NULL);
  g_object_get (g_test_data.sink, "silent", &res_silent, NULL);
  EXPECT_EQ (res_silent, !silent);

  /** GstBaseSink:sync TRUE */
  g_object_get (g_test_data.sink, "sync", &sync, NULL);
  EXPECT_EQ (sync, TRUE);

  g_object_set (g_test_data.sink, "sync", !sync, NULL);
  g_object_get (g_test_data.sink, "sync", &res_sync, NULL);
  EXPECT_EQ (res_sync, !sync);

  /** GstBaseSink:max-lateness -1 (unlimited time) */
  g_object_get (g_test_data.sink, "max-lateness", &lateness, NULL);
  EXPECT_EQ (lateness, -1);

  lateness = 30 * GST_MSECOND;
  g_object_set (g_test_data.sink, "max-lateness", lateness, NULL);
  g_object_get (g_test_data.sink, "max-lateness", &res_lateness, NULL);
  EXPECT_EQ (res_lateness, lateness);

  /** GstBaseSink:qos TRUE */
  g_object_get (g_test_data.sink, "qos", &qos, NULL);
  EXPECT_EQ (qos, TRUE);

  g_object_set (g_test_data.sink, "qos", !qos, NULL);
  g_object_get (g_test_data.sink, "qos", &res_qos, NULL);
  EXPECT_EQ (res_qos, !qos);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for tensor sink signals.
 */
TEST (tensor_sink_test, signals)
{
  const guint num_buffers = 5;
  gulong handle_id;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB };

  ASSERT_TRUE (_setup_pipeline (option));

  /** tensor sink signals */
  handle_id = g_signal_connect (g_test_data.sink, "stream-start",
      (GCallback) _stream_start_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  handle_id = g_signal_connect (g_test_data.sink, "eos",
      (GCallback) _eos_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.start, TRUE);
  EXPECT_EQ (g_test_data.end, TRUE);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check caps and config for tensor */
  {
    GstCaps *caps;
    GstStructure *structure;
    GstTensorConfig config;

    caps = gst_tensor_caps_from_config (&g_test_data.tensor_config);
    structure = gst_caps_get_structure (caps, 0);

    EXPECT_TRUE (gst_tensor_config_from_structure (&config, structure));
    EXPECT_TRUE (gst_tensor_config_is_equal (&config,
            &g_test_data.tensor_config));
    EXPECT_TRUE (gst_caps_is_equal (g_test_data.current_caps, caps));

    gst_caps_unref (caps);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for tensor sink emit-signal (case for no signal).
 */
TEST (tensor_sink_test, emit_signal)
{
  const guint num_buffers = 5;
  gulong handle_id;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB };

  ASSERT_TRUE (_setup_pipeline (option));

  /** set emit-signal FALSE (no signal) */
  g_object_set (g_test_data.sink, "emit-signal", (gboolean) FALSE, NULL);

  /** tensor sink signals */
  handle_id = g_signal_connect (g_test_data.sink, "stream-start",
      (GCallback) _stream_start_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  handle_id = g_signal_connect (g_test_data.sink, "eos",
      (GCallback) _eos_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, 0);
  EXPECT_EQ (g_test_data.start, FALSE);
  EXPECT_EQ (g_test_data.end, FALSE);

  /** check caps name is null (no signal) */
  EXPECT_TRUE (g_test_data.caps_name == NULL);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for tensor sink signal-rate.
 */
TEST (tensor_sink_test, signal_rate)
{
  const guint num_buffers = 6;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB };

  ASSERT_TRUE (_setup_pipeline (option));

  /** set signal-rate */
  g_object_set (g_test_data.sink, "signal-rate", (guint) 15, NULL);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_TRUE (g_test_data.received < num_buffers);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check caps and config for tensor */
  {
    GstCaps *caps;
    GstStructure *structure;
    GstTensorConfig config;

    caps = gst_tensor_caps_from_config (&g_test_data.tensor_config);
    structure = gst_caps_get_structure (caps, 0);

    EXPECT_TRUE (gst_tensor_config_from_structure (&config, structure));
    EXPECT_TRUE (gst_tensor_config_is_equal (&config,
            &g_test_data.tensor_config));
    EXPECT_TRUE (gst_caps_is_equal (g_test_data.current_caps, caps));

    gst_caps_unref (caps);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for caps negotiation failed.
 */
TEST (tensor_sink_test, caps_error)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_NEGO_FAILED };

  /** failed : cannot link videoconvert and tensor_sink */
  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check error message */
  EXPECT_EQ (g_test_data.status, TEST_ERR_MESSAGE);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, 0);

  /** check caps and config for tensor */
  {
    GstCaps *caps;
    GstCaps *raw_caps;

    /** tensor config is invalid */
    EXPECT_FALSE (gst_tensor_config_validate (&g_test_data.tensor_config));

    caps = gst_tensor_caps_from_config (&g_test_data.tensor_config);
    raw_caps = gst_caps_from_string (GST_TENSOR_CAP_DEFAULT);

    /** compare with default caps */
    EXPECT_TRUE (gst_caps_is_equal (caps, raw_caps));

    gst_caps_unref (caps);
    gst_caps_unref (raw_caps);
  }

  EXPECT_TRUE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for other/tensors caps negotiation.
 */
TEST (tensor_sink_test, caps_tensors)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSORS };
  guint i;

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 2);
  EXPECT_EQ (g_test_data.received_size, 3 * 160 * 120 * 2);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensors"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensors config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.num_tensors, 2);

  for (i = 0; i < g_test_data.tensors_config.info.num_tensors; i++) {
    EXPECT_EQ (g_test_data.tensors_config.info.info[i].type, _NNS_UINT8);
    EXPECT_EQ (g_test_data.tensors_config.info.info[i].dimension[0], 3);
    EXPECT_EQ (g_test_data.tensors_config.info.info[i].dimension[1], 160);
    EXPECT_EQ (g_test_data.tensors_config.info.info[i].dimension[2], 120);
    EXPECT_EQ (g_test_data.tensors_config.info.info[i].dimension[3], 1);
  }

  EXPECT_EQ (g_test_data.tensors_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  /** check caps and config for tensors */
  {
    GstCaps *caps;
    GstStructure *structure;
    GstTensorsConfig config;

    caps = gst_tensors_caps_from_config (&g_test_data.tensors_config);
    structure = gst_caps_get_structure (caps, 0);

    EXPECT_TRUE (gst_tensors_config_from_structure (&config, structure));
    EXPECT_TRUE (gst_tensors_config_is_equal (&config,
            &g_test_data.tensors_config));
    EXPECT_TRUE (gst_caps_is_equal (g_test_data.current_caps, caps));

    gst_caps_unref (caps);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format RGB.
 */
TEST (tensor_stream_test, video_rgb)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 3 * 160 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 3);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 160);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format BGR.
 */
TEST (tensor_stream_test, video_bgr)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_BGR };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 3 * 160 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 3);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 160);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format RGB, remove padding.
 */
TEST (tensor_stream_test, video_rgb_padding)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB_PADDING };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 3 * 162 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 3);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 162);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format BGR, remove padding.
 */
TEST (tensor_stream_test, video_bgr_padding)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_BGR_PADDING };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 3 * 162 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 3);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 162);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format RGB, 3 frames from tensor_converter.
 */
TEST (tensor_stream_test, video_rgb_3f)
{
  const guint num_buffers = 7;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB_3F };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers / 3);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 3 * 160 * 120 * 3);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 3);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 160);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 3);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format RGBA.
 */
TEST (tensor_stream_test, video_rgba)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGBA };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4 * 162 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 4);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 162);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format BGRA.
 */
TEST (tensor_stream_test, video_bgra)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_BGRA };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4 * 162 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 4);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 162);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format ARGB.
 */
TEST (tensor_stream_test, video_argb)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_ARGB };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4 * 162 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 4);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 162);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format ABGR.
 */
TEST (tensor_stream_test, video_abgr)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_ABGR };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4 * 162 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 4);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 162);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format RGBx.
 */
TEST (tensor_stream_test, video_rgbx)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGBx };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4 * 162 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 4);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 162);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format xRGB.
 */
TEST (tensor_stream_test, video_xrgb)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_xRGB };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4 * 162 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 4);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 162);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format xBGR.
 */
TEST (tensor_stream_test, video_xbgr)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_xBGR };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4 * 162 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 4);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 162);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format BGRx.
 */
TEST (tensor_stream_test, video_bgrx)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_BGRx };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4 * 162 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 4);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 162);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format BGRx, 2 frames from tensor_converter.
 */
TEST (tensor_stream_test, video_bgrx_2f)
{
  const guint num_buffers = 6;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_BGRx_2F };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers / 2);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4 * 160 * 120 * 2);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 4);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 160);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 2);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format GRAY8.
 */
TEST (tensor_stream_test, video_gray8)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_GRAY8 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 160 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 160);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format GRAY8, remove padding.
 */
TEST (tensor_stream_test, video_gray8_padding)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_GRAY8_PADDING };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 162 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 162);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video format GRAY8, 3 frames from tensor_converter, remove padding.
 */
TEST (tensor_stream_test, video_gray8_3f_padding)
{
  const guint num_buffers = 6;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_GRAY8_3F_PADDING };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers / 3);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 162 * 120 * 3);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 162);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 3);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for audio format S8.
 */
TEST (tensor_stream_test, audio_s8)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_S8 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 500);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_INT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 500);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 16000);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for audio format U8, 100 frames from tensor_converter.
 */
TEST (tensor_stream_test, audio_u8_100f)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_U8_100F };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers * 5);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 100);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 100);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 16000);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for audio format S16.
 */
TEST (tensor_stream_test, audio_s16)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_S16 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 500 * 2);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_INT16);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 500);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 16000);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for audio format U16, 1000 frames from tensor_converter.
 */
TEST (tensor_stream_test, audio_u16_1000f)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_U16_1000F };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers / 2);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 500 * 2 * 2);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT16);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1000);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 16000);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for audio format S32.
 */
TEST (tensor_stream_test, audio_s32)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_S32 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 500 * 4);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_INT32);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 500);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 44100);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for audio format U32.
 */
TEST (tensor_stream_test, audio_u32)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_U32 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 500 * 4);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT32);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 500);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 44100);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for audio format F32.
 */
TEST (tensor_stream_test, audio_f32)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_F32 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 500 * 4);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_FLOAT32);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 500);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 44100);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for audio format F64.
 */
TEST (tensor_stream_test, audio_f64)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_F64 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 500 * 8);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_FLOAT64);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 500);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 44100);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for text format utf8.
 */
TEST (tensor_stream_test, text_utf8)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_TEXT };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 20);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 20);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for text format utf8, 3 frames from tensor_converter.
 */
TEST (tensor_stream_test, text_utf8_3f)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_TEXT_3F };
  GstElement *convert;
  gchar *prop_str;
  gboolean prop_bool;
  guint prop_uint;

  ASSERT_TRUE (_setup_pipeline (option));

  /* tensor_converter properties */
  convert = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "convert");
  ASSERT_TRUE (convert != NULL);

  g_object_get (convert, "input-dim", &prop_str, NULL);
  EXPECT_STREQ (prop_str, "30:1:1:1");
  g_free (prop_str);

  g_object_get (convert, "input-type", &prop_str, NULL);
  EXPECT_STREQ (prop_str, "");
  g_free (prop_str);

  g_object_get (convert, "set-timestamp", &prop_bool, NULL);
  EXPECT_EQ (prop_bool, TRUE);

  g_object_get (convert, "silent", &prop_bool, NULL);
  EXPECT_EQ (prop_bool, TRUE);

  g_object_get (convert, "frames-per-tensor", &prop_uint, NULL);
  EXPECT_EQ (prop_uint, 3);

  gst_object_unref (convert);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers / 3);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 30 * 3);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 30);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 3);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 100);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for octet stream.
 */
TEST (tensor_stream_test, octet_current_ts)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_OCTET_CUR_TS };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers, FALSE);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 10);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 10);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for octet stream.
 */
TEST (tensor_stream_test, octet_framerate_ts)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_OCTET_RATE_TS };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers, FALSE);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 10);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 10);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 50);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for octet stream.
 */
TEST (tensor_stream_test, octet_valid_ts)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_OCTET_VALID_TS };
  GstElement *convert;
  gchar *prop_str;
  gboolean prop_bool;
  guint prop_uint;

  ASSERT_TRUE (_setup_pipeline (option));

  /* tensor_converter properties */
  convert = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "convert");
  ASSERT_TRUE (convert != NULL);

  g_object_get (convert, "input-dim", &prop_str, NULL);
  EXPECT_STREQ (prop_str, "1:10:1:1");
  g_free (prop_str);

  g_object_get (convert, "input-type", &prop_str, NULL);
  EXPECT_STREQ (prop_str, "uint8");
  g_free (prop_str);

  g_object_get (convert, "set-timestamp", &prop_bool, NULL);
  EXPECT_EQ (prop_bool, FALSE);

  g_object_get (convert, "silent", &prop_bool, NULL);
  EXPECT_EQ (prop_bool, TRUE);

  g_object_get (convert, "frames-per-tensor", &prop_uint, NULL);
  EXPECT_EQ (prop_uint, 1);

  gst_object_unref (convert);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers, TRUE);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 10);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 10);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for octet stream.
 */
TEST (tensor_stream_test, octet_invalid_ts)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_OCTET_INVALID_TS };
  GstElement *convert;
  gchar *prop_str;
  gboolean prop_bool;
  guint prop_uint;

  ASSERT_TRUE (_setup_pipeline (option));

  /* tensor_converter properties */
  convert = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "convert");
  ASSERT_TRUE (convert != NULL);

  g_object_get (convert, "input-dim", &prop_str, NULL);
  EXPECT_STREQ (prop_str, "1:10:1:1");
  g_free (prop_str);

  g_object_get (convert, "input-type", &prop_str, NULL);
  EXPECT_STREQ (prop_str, "uint8");
  g_free (prop_str);

  g_object_get (convert, "set-timestamp", &prop_bool, NULL);
  EXPECT_EQ (prop_bool, FALSE);

  g_object_get (convert, "silent", &prop_bool, NULL);
  EXPECT_EQ (prop_bool, TRUE);

  g_object_get (convert, "frames-per-tensor", &prop_uint, NULL);
  EXPECT_EQ (prop_uint, 1);

  gst_object_unref (convert);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers, FALSE);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 10);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check invalid timestamp */
  EXPECT_TRUE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 10);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for octet stream.
 */
TEST (tensor_stream_test, octet_2f)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_OCTET_2F };
  GstElement *convert;
  gchar *prop_str;
  gboolean prop_bool;
  guint prop_uint;

  ASSERT_TRUE (_setup_pipeline (option));

  /* tensor_converter properties */
  convert = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "convert");
  ASSERT_TRUE (convert != NULL);

  g_object_get (convert, "input-dim", &prop_str, NULL);
  EXPECT_STREQ (prop_str, "1:5:1:1");
  g_free (prop_str);

  g_object_get (convert, "input-type", &prop_str, NULL);
  EXPECT_STREQ (prop_str, "int8");
  g_free (prop_str);

  g_object_get (convert, "set-timestamp", &prop_bool, NULL);
  EXPECT_EQ (prop_bool, TRUE);

  g_object_get (convert, "silent", &prop_bool, NULL);
  EXPECT_EQ (prop_bool, TRUE);

  g_object_get (convert, "frames-per-tensor", &prop_uint, NULL);
  EXPECT_EQ (prop_uint, 1);

  gst_object_unref (convert);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers, FALSE);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers * 2);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 5);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_INT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 5);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 100);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for other/tensor, passthrough custom filter.
 */
TEST (tensor_stream_test, custom_filter_tensor)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_CUSTOM_TENSOR };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 3 * 160 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 3);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 160);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  /** check caps and config for tensor */
  {
    GstCaps *caps;
    GstStructure *structure;
    GstTensorConfig config;

    caps = gst_tensor_caps_from_config (&g_test_data.tensor_config);
    structure = gst_caps_get_structure (caps, 0);

    EXPECT_TRUE (gst_tensor_config_from_structure (&config, structure));
    EXPECT_TRUE (gst_tensor_config_is_equal (&config,
            &g_test_data.tensor_config));
    EXPECT_TRUE (gst_caps_is_equal (g_test_data.current_caps, caps));

    gst_caps_unref (caps);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for other/tensors, passthrough custom filter.
 */
TEST (tensor_stream_test, custom_filter_tensors)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_CUSTOM_TENSORS };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 3);
  EXPECT_EQ (g_test_data.received_size, 95616); /** 160 * 120 * 3 + 120 * 80 * 3 + 64 * 48 * 3 */

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensors"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensors config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.num_tensors, 3);

  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1);

  EXPECT_EQ (g_test_data.tensors_config.info.info[1].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[0], 3);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[1], 120);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[2], 80);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[3], 1);

  EXPECT_EQ (g_test_data.tensors_config.info.info[2].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[0], 3);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[1], 64);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[2], 48);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[3], 1);

  EXPECT_EQ (g_test_data.tensors_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  /** check caps and config for tensors */
  {
    GstCaps *caps;
    GstStructure *structure;
    GstTensorsConfig config;

    caps = gst_tensors_caps_from_config (&g_test_data.tensors_config);
    structure = gst_caps_get_structure (caps, 0);

    EXPECT_TRUE (gst_tensors_config_from_structure (&config, structure));
    EXPECT_TRUE (gst_tensors_config_is_equal (&config,
            &g_test_data.tensors_config));
    EXPECT_TRUE (gst_caps_is_equal (g_test_data.current_caps, caps));

    gst_caps_unref (caps);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test to drop incoming buffer in tensor_filter using custom filter.
 */
TEST (tensor_stream_test, custom_filter_drop_buffer)
{
  const guint num_buffers = 22;
  TestOption option = { num_buffers, TEST_TYPE_CUSTOM_BUF_DROP };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, 2);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 200 * 2);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_INT16);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 200);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 16000);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for tensors (mixed, video and audio).
 */
TEST (tensor_stream_test, tensors_mix)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSORS_MIX };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 500 * 2);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_INT16);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 500);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30); /** 30 fps from video stream */
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  /** check caps and config for tensor */
  {
    GstCaps *caps;
    GstStructure *structure;
    GstTensorConfig config;

    caps = gst_tensor_caps_from_config (&g_test_data.tensor_config);
    structure = gst_caps_get_structure (caps, 0);

    EXPECT_TRUE (gst_tensor_config_from_structure (&config, structure));
    EXPECT_TRUE (gst_tensor_config_is_equal (&config,
            &g_test_data.tensor_config));
    EXPECT_TRUE (gst_caps_is_equal (g_test_data.current_caps, caps));

    gst_caps_unref (caps);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for typecast to int32 using tensor_transform.
 */
TEST (tensor_stream_test, typecast_int32)
{
  const guint num_buffers = 2;
  const tensor_type t_type = _NNS_INT32;
  TestOption option = { num_buffers, TEST_TYPE_TYPECAST, t_type };
  unsigned int t_size = tensor_element_size[t_type];

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 10 * t_size);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, t_type);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 10);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for typecast to uint32 using tensor_transform.
 */
TEST (tensor_stream_test, typecast_uint32)
{
  const guint num_buffers = 2;
  const tensor_type t_type = _NNS_UINT32;
  TestOption option = { num_buffers, TEST_TYPE_TYPECAST, t_type };
  unsigned int t_size = tensor_element_size[t_type];

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 10 * t_size);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, t_type);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 10);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for typecast to int16 using tensor_transform.
 */
TEST (tensor_stream_test, typecast_int16)
{
  const guint num_buffers = 2;
  const tensor_type t_type = _NNS_INT16;
  TestOption option = { num_buffers, TEST_TYPE_TYPECAST, t_type };
  unsigned int t_size = tensor_element_size[t_type];

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 10 * t_size);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, t_type);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 10);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for typecast to uint16 using tensor_transform.
 */
TEST (tensor_stream_test, typecast_uint16)
{
  const guint num_buffers = 2;
  const tensor_type t_type = _NNS_UINT16;
  TestOption option = { num_buffers, TEST_TYPE_TYPECAST, t_type };
  unsigned int t_size = tensor_element_size[t_type];

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 10 * t_size);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, t_type);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 10);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for typecast to float64 using tensor_transform.
 */
TEST (tensor_stream_test, typecast_float64)
{
  const guint num_buffers = 2;
  const tensor_type t_type = _NNS_FLOAT64;
  TestOption option = { num_buffers, TEST_TYPE_TYPECAST, t_type };
  unsigned int t_size = tensor_element_size[t_type];

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 10 * t_size);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, t_type);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 10);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for typecast to float32 using tensor_transform.
 */
TEST (tensor_stream_test, typecast_float32)
{
  const guint num_buffers = 2;
  const tensor_type t_type = _NNS_FLOAT32;
  TestOption option = { num_buffers, TEST_TYPE_TYPECAST, t_type };
  unsigned int t_size = tensor_element_size[t_type];

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 10 * t_size);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, t_type);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 10);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for typecast to int64 using tensor_transform.
 */
TEST (tensor_stream_test, typecast_int64)
{
  const guint num_buffers = 2;
  const tensor_type t_type = _NNS_INT64;
  TestOption option = { num_buffers, TEST_TYPE_TYPECAST, t_type };
  unsigned int t_size = tensor_element_size[t_type];

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 10 * t_size);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, t_type);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 10);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for typecast to uint64 using tensor_transform.
 */
TEST (tensor_stream_test, typecast_uint64)
{
  const guint num_buffers = 2;
  const tensor_type t_type = _NNS_UINT64;
  TestOption option = { num_buffers, TEST_TYPE_TYPECAST, t_type };
  unsigned int t_size = tensor_element_size[t_type];

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  _push_text_data (num_buffers);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 10 * t_size);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, t_type);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 10);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video stream with tensor_split.
 */
TEST (tensor_stream_test, video_split)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB_SPLIT };
  GstElement *split;
  gchar *str;
  gboolean silent;

  ASSERT_TRUE (_setup_pipeline (option));

  /** Check properties of tensor_split */
  split = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "split");
  ASSERT_TRUE (split != NULL);

  g_object_get (split, "tensorpick", &str, NULL);
  EXPECT_STREQ (str, "0,1,2");
  g_free (str);

  g_object_get (split, "tensorseg", &str, NULL);
  EXPECT_STREQ (str, "1:160:120:1,1:160:120:1,1:160:120:1");
  g_free (str);

  g_object_get (split, "silent", &silent, NULL);
  EXPECT_EQ (silent, TRUE);

  gst_object_unref (split);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 160 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 160);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video stream with tensor_aggregator.
 */
TEST (tensor_stream_test, video_aggregate_1)
{
  const guint num_buffers = 35;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB_AGGR_1 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, (num_buffers - 10) / 5 + 1);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 3 * 160 * 120 * 10);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 3);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 160);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 10);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video stream with tensor_aggregator.
 */
TEST (tensor_stream_test, video_aggregate_2)
{
  const guint num_buffers = 35;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB_AGGR_2 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, (num_buffers - 10) / 5 + 1);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 3 * 1600 * 120);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 3);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1600);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 120);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for video stream with tensor_aggregator.
 */
TEST (tensor_stream_test, video_aggregate_3)
{
  const guint num_buffers = 40;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB_AGGR_3 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, (num_buffers / 10));
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 3 * 64 * 48 * 8);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 3);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 64 * 8);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 48);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for audio stream with tensor_aggregator.
 */
TEST (tensor_stream_test, audio_aggregate_s16)
{
  const guint num_buffers = 21;
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_S16_AGGR };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers / 4);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 500 * 2 * 4);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_INT16);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 2000);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 16000);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test for audio stream with tensor_aggregator.
 */
TEST (tensor_stream_test, audio_aggregate_u16)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_U16_AGGR };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers * 5);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 500 * 2 / 5);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT16);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 100);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);
  EXPECT_EQ (g_test_data.tensor_config.rate_n, 16000);
  EXPECT_EQ (g_test_data.tensor_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test multi-stream sync & frame-dropping of Issue #739, 1st subissue
 */
TEST (tensor_stream_test, issue739_mux_parallel_1)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MUX_PARALLEL_1 };

  option.tmpfile = _get_temp_filename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers * 10);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4);     /* uint32_t, 1:1:1:1 */

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT32);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);

  /** @todo Check contents in the sink */
  if (option.tmpfile) {
    gchar *data;
    gsize read, i;

    if (g_file_get_contents (option.tmpfile, &data, &read, NULL)) {
      read /= 4;
      EXPECT_EQ (read, num_buffers * 10);
      for (i = 0; i < num_buffers * 2U; i++)
        EXPECT_EQ (((uint32_t *) data)[i], i);

      g_free (data);
    }

    /* remove temp file */
    if (g_remove (option.tmpfile) != 0) {
      _print_log ("failed to remove temp file %s", option.tmpfile);
    }
    g_free (option.tmpfile);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test multi-stream sync & frame-dropping of Issue #739, 1st subissue
 */
TEST (tensor_stream_test, issue739_mux_parallel_2)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MUX_PARALLEL_2 };

  option.tmpfile = _get_temp_filename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers * 10);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4);     /* uint32_t, 1:1:1:1 */

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT32);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);

  /** @todo Check contents in the sink */
  if (option.tmpfile) {
    gchar *data;
    gsize read, i;

    if (g_file_get_contents (option.tmpfile, &data, &read, NULL)) {
      read /= 4;
      EXPECT_EQ (read, num_buffers * 10);
      for (i = 0; i < num_buffers * 2U; i++)
        EXPECT_EQ (((uint32_t *) data)[i], i);

      g_free (data);
    }

    /* remove temp file */
    if (g_remove (option.tmpfile) != 0) {
      _print_log ("failed to remove temp file %s", option.tmpfile);
    }
    g_free (option.tmpfile);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test multi-stream sync & frame-dropping of Issue #739, 1st subissue
 */
TEST (tensor_stream_test, issue739_mux_parallel_3)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MUX_PARALLEL_3 };

  option.tmpfile = _get_temp_filename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_GE (g_test_data.received, num_buffers * 25 - 1);
  EXPECT_LE (g_test_data.received, num_buffers * 25);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4);     /* uint32_t, 1:1:1:1 */

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT32);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);

  /** @todo Check contents in the sink */
  if (option.tmpfile) {
    gchar *data;
    gsize read, i;
    uint32_t lastval;

    if (g_file_get_contents (option.tmpfile, &data, &read, NULL)) {
      read /= 4;
      EXPECT_TRUE (read >= (num_buffers * 25 - 1));
      EXPECT_TRUE (read <= (num_buffers * 25));

      lastval = 0;
      for (i = 0; i < read; i++) {
        EXPECT_TRUE (((uint32_t *) data)[i] >= lastval);
        EXPECT_TRUE (((uint32_t *) data)[i] <= lastval + 1);
        lastval = ((uint32_t *) data)[i];
      }
      EXPECT_TRUE (lastval <= (num_buffers * 10));
      EXPECT_TRUE (lastval >= (num_buffers * 10 - 1));

      g_free (data);
    }

    /* remove temp file */
    if (g_remove (option.tmpfile) != 0) {
      _print_log ("failed to remove temp file %s", option.tmpfile);
    }
    g_free (option.tmpfile);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test multi-stream sync & frame-dropping of Issue #739, 1st subissue
 */
TEST (tensor_stream_test, issue739_mux_parallel_4)
{
  /** @todo Write this after the tensor-mux/merge sync-option "basepad" is updated */
  EXPECT_EQ (1, 1);
}

/**
 * @brief Test multi-stream sync & frame-dropping of Issue #739, 1st subissue
 */
TEST (tensor_stream_test, issue739_merge_parallel_1)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MERGE_PARALLEL_1 };

  option.tmpfile = _get_temp_filename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers * 10);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4);     /* uint32_t, 1:1:1:1 */

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT32);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);

  /** @todo Check contents in the sink */
  if (option.tmpfile) {
    gchar *data;
    gsize read, i;

    if (g_file_get_contents (option.tmpfile, &data, &read, NULL)) {
      read /= 4;
      EXPECT_EQ (read, num_buffers * 10);
      for (i = 0; i < num_buffers * 2U; i++)
        EXPECT_EQ (((uint32_t *) data)[i], i);

      g_free (data);
    }

    /* remove temp file */
    if (g_remove (option.tmpfile) != 0) {
      _print_log ("failed to remove temp file %s", option.tmpfile);
    }
    g_free (option.tmpfile);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test multi-stream sync & frame-dropping of Issue #739, 1st subissue
 */
TEST (tensor_stream_test, issue739_merge_parallel_2)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MERGE_PARALLEL_2 };

  option.tmpfile = _get_temp_filename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers * 10);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4);     /* uint32_t, 1:1:1:1 */

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT32);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);

  /** @todo Check contents in the sink */
  if (option.tmpfile) {
    gchar *data;
    gsize read, i;

    if (g_file_get_contents (option.tmpfile, &data, &read, NULL)) {
      read /= 4;
      EXPECT_EQ (read, num_buffers * 10);
      for (i = 0; i < num_buffers * 2U; i++)
        EXPECT_EQ (((uint32_t *) data)[i], i);

      g_free (data);
    }

    /* remove temp file */
    if (g_remove (option.tmpfile) != 0) {
      _print_log ("failed to remove temp file %s", option.tmpfile);
    }
    g_free (option.tmpfile);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test multi-stream sync & frame-dropping of Issue #739, 1st subissue
 */
TEST (tensor_stream_test, issue739_merge_parallel_3)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MERGE_PARALLEL_3 };

  option.tmpfile = _get_temp_filename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_GE (g_test_data.received, num_buffers * 25 - 1);
  EXPECT_LE (g_test_data.received, num_buffers * 25);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 4);     /* uint32_t, 1:1:1:1 */

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config */
  EXPECT_TRUE (gst_tensor_config_validate (&g_test_data.tensor_config));
  EXPECT_EQ (g_test_data.tensor_config.info.type, _NNS_UINT32);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[0], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[1], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[2], 1);
  EXPECT_EQ (g_test_data.tensor_config.info.dimension[3], 1);

  /** @todo Check contents in the sink */
  if (option.tmpfile) {
    gchar *data;
    gsize read, i;
    uint32_t lastval;

    if (g_file_get_contents (option.tmpfile, &data, &read, NULL)) {
      read /= 4;
      EXPECT_TRUE (read >= (num_buffers * 25 - 1));
      EXPECT_TRUE (read <= (num_buffers * 25));

      lastval = 0;
      for (i = 0; i < read; i++) {
        EXPECT_TRUE (((uint32_t *) data)[i] >= lastval);
        EXPECT_TRUE (((uint32_t *) data)[i] <= lastval + 1);
        lastval = ((uint32_t *) data)[i];
      }
      EXPECT_GE (lastval, (num_buffers - 1) * 25);
      EXPECT_LE (lastval, num_buffers * 25);

      g_free (data);
    }

    /* remove temp file */
    if (g_remove (option.tmpfile) != 0) {
      _print_log ("failed to remove temp file %s", option.tmpfile);
    }
    g_free (option.tmpfile);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Test get/set property of tensor_decoder
 */
TEST (tensor_stream_test, tensor_decoder_property)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_DECODER_PROPERTY };
  GstElement *dec;
  gchar *str;
  gboolean silent;

  ASSERT_TRUE (_setup_pipeline (option));

  /** Check properties of tensor_decoder */
  dec = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "decoder");
  ASSERT_TRUE (dec != NULL);

  g_object_get (dec, "mode", &str, NULL);
  EXPECT_STREQ (str, "direct_video");
  g_free (str);

  g_object_get (dec, "silent", &silent, NULL);
  EXPECT_EQ (silent, TRUE);

  g_object_get (dec, "option1", &str, NULL);
  EXPECT_STREQ (str, "whatthehell");
  g_free (str);
  g_object_get (dec, "option2", &str, NULL);
  EXPECT_STREQ (str, "isgoingon");
  g_free (str);
  g_object_get (dec, "option3", &str, NULL);
  EXPECT_STREQ (str, "nothing");
  g_free (str);
  g_object_get (dec, "option4", &str, NULL);
  EXPECT_STREQ (str, "iswrong");
  g_free (str);
  g_object_get (dec, "option5", &str, NULL);
  EXPECT_STREQ (str, "keepcalm");
  g_free (str);
  g_object_get (dec, "option6", &str, NULL);
  EXPECT_STREQ (str, "and have a break");
  g_free (str);
  g_object_get (dec, "option7", &str, NULL);
  EXPECT_STREQ (str, "iwill=not");
  g_free (str);
  g_object_get (dec, "option8", &str, NULL);
  EXPECT_STREQ (str, "break=your");
  g_free (str);
  g_object_get (dec, "option9", &str, NULL);
  EXPECT_STREQ (str, "system=1234");
  g_free (str);

  gst_object_unref (dec);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, 5);
  EXPECT_EQ (g_test_data.mem_blocks, 1);
  EXPECT_EQ (g_test_data.received_size, 64);     /* uint8_t, 4:4:4:1 */

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data ();
}

/**
 * @brief Main function for unit test.
 */
int
main (int argc, char **argv)
{
  testing::InitGoogleTest (&argc, argv);
  int optind;
  for(optind = 1; optind < argc && argv[optind][0] == '-'; optind++){
    switch(argv[optind][1]){
    case 'd': custom_dir=g_strdup(argv[optind+1]); break;
    default:
      break;
    }
  }

  gst_init (&argc, &argv);

  return RUN_ALL_TESTS ();
}
