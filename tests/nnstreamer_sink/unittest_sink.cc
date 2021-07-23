/**
 * @file	unittest_sink.cc
 * @date	29 June 2018
 * @brief	Unit test for tensor sink plugin
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs.
 */

#include <gtest/gtest.h>
#include <glib/gstdio.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <stdlib.h>
#include <string.h>

#include <nnstreamer_conf.h>
#include <unittest_util.h>
#include "nnstreamer_plugin_api_filter.h"
#include "tensor_common.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief Macro to check error case.
 */
#define _check_cond_err(cond)                           \
  if (!(cond)) {                                        \
    _print_log ("test failed!! [line : %d]", __LINE__); \
    goto error;                                         \
  }

static const guint TEST_TIME_OUT_MSEC = 2000;
static const guint DEFAULT_TIME_INTERVAL = 10000;
static const gulong MSEC_PER_USEC = 1000;
static const gulong DEFAULT_JITTER = 0UL;
static const gulong DEFAULT_FPS = 30UL;
static gchar *custom_dir = NULL;
static gulong jitter = DEFAULT_JITTER;
static gulong fps = DEFAULT_FPS;

/**
 * @brief Current status.
 */
typedef enum {
  TEST_START, /**< start to setup pipeline */
  TEST_INIT, /**< init done */
  TEST_ERR_MESSAGE, /**< received error message */
  TEST_STREAM, /**< stream started */
  TEST_EOS /**< end of stream */
} TestStatus;

/**
 * @brief Test type.
 */
typedef enum {
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
  TEST_TYPE_OCTET_MULTI_TENSORS, /**< pipeline for octet stream, byte array to multi tensors */
  TEST_TYPE_FLEX_TENSOR_1, /**< pipeline for flexible tensor (sink) */
  TEST_TYPE_FLEX_TENSOR_2, /**< pipeline for flexible tensor (converter, static multi-tensors to flex) */
  TEST_TYPE_FLEX_TENSOR_3, /**< pipeline for flexible tensor (converter, flex to static) */
  TEST_TYPE_TENSORS_MUX_1, /**< pipeline for tensors with tensor_mux (static tensor stream) */
  TEST_TYPE_TENSORS_MUX_2, /**< pipeline for tensors with tensor_mux (static and flex tensor stream combined) */
  TEST_TYPE_TENSORS_MUX_3, /**< pipeline for tensors with tensor_mux, tensor_demux (static and flex tensor stream combined) */
  TEST_TYPE_TENSORS_FLEX_NEGO_FAILED_1, /**< pipeline for nego failure case (mux, cannot link flex and static pad) */
  TEST_TYPE_TENSORS_FLEX_NEGO_FAILED_2, /**< pipeline for nego failure case (demux, cannot link flex and static pad) */
  TEST_TYPE_TENSORS_MIX_1, /**< pipeline for tensors with tensor_mux, tensor_demux */
  TEST_TYPE_TENSORS_MIX_2, /**< pipeline for tensors with tensor_mux, tensor_demux pick 0,2 */
  TEST_TYPE_TENSORS_MIX_3, /**< pipeline for tensors with tensor_mux, tensor_demux pick 1,2 */
  TEST_TYPE_CUSTOM_TENSOR, /**< pipeline for single tensor with passthrough custom filter */
  TEST_TYPE_CUSTOM_TENSORS_1, /**< pipeline for tensors with passthrough custom filter */
  TEST_TYPE_CUSTOM_TENSORS_2, /**< pipeline for tensors with passthrough custom filter properties specified */
  TEST_TYPE_TENSOR_CAP_1, /**< pipeline for tensor out test (tensor caps are specified) */
  TEST_TYPE_TENSOR_CAP_2, /**< pipeline for tensor out test (tensor caps are not specified) */
  TEST_TYPE_TENSORS_CAP_1, /**< pipeline for tensors out test (tensors caps are specified, num_tensors is 1) */
  TEST_TYPE_TENSORS_CAP_2, /**< pipeline for tensors out test (tensors caps are specified, num_tensors is 3) */
  TEST_TYPE_CUSTOM_MULTI, /**< pipeline with multiple custom filters */
  TEST_TYPE_CUSTOM_BUF_DROP, /**< pipeline to test buffer-drop in tensor_filter using custom filter */
  TEST_TYPE_CUSTOM_PASSTHROUGH, /**< pipeline to test custom passthrough without so file */
  TEST_TYPE_NEGO_FAILED, /**< pipeline to test caps negotiation */
  TEST_TYPE_VIDEO_RGB_SPLIT, /**< pipeline to test tensor_split */
  TEST_TYPE_VIDEO_RGB_AGGR_1, /**< pipeline to test tensor_aggregator (change dimension index 3 : 1 > 10)*/
  TEST_TYPE_VIDEO_RGB_AGGR_2, /**< pipeline to test tensor_aggregator (change dimension index 1 : 160 > 1600) */
  TEST_TYPE_VIDEO_RGB_AGGR_3, /**< pipeline to test tensor_aggregator (test to get frames with the property concat) */
  TEST_TYPE_AUDIO_S16_AGGR, /**< pipeline to test tensor_aggregator */
  TEST_TYPE_AUDIO_U16_AGGR, /**< pipeline to test tensor_aggregator */
  TEST_TYPE_TRANSFORM_CAPS_NEGO_1, /**< pipeline for caps negotiation in tensor_transform (typecast mode) */
  TEST_TYPE_TRANSFORM_CAPS_NEGO_2, /**< pipeline for caps negotiation in tensor_transform (arithmetic mode) */
  TEST_TYPE_TRANSFORM_TENSORS, /**< pipeline for tensors with tensor_transform (typecast mode) */
  TEST_TYPE_TRANSFORM_APPLY, /**< pipeline for apply option with tensor_transform (typecast mode) */
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
  TEST_CUSTOM_EASY_ICF_01, /**< pipeline to test easy-custom in code func */
  TEST_TYPE_UNKNOWN /**< unknonwn */
} TestType;

/**
 * @brief Test options.
 */
typedef struct {
  guint num_buffers; /**< count of buffers */
  TestType test_type; /**< test pipeline */
  tensor_type t_type; /**< tensor type */
  char *tmpfile; /**< tmpfile to write */
  gboolean need_sync; /**< sync on the clock */
} TestOption;

/**
 * @brief Data structure for test.
 */
typedef struct {
  GMainLoop *loop; /**< main event loop */
  GstElement *pipeline; /**< gst pipeline for test */
  GstBus *bus; /**< gst bus for test */
  GstElement *sink; /**< tensor sink element */
  TestStatus status; /**< current status */
  guint received; /**< received buffer count */
  guint mem_blocks; /**< memory blocks in received buffer */
  gsize received_size; /**< received buffer size */
  gboolean invalid_timestamp; /**< flag to check timestamp */
  gboolean test_failed; /**< flag to indicate error */
  gboolean start; /**< stream started (for tensor_sink signal) */
  gboolean end; /**< eos reached (for tensor_sink signal) */
  GstCaps *current_caps; /**< negotiated caps */
  gchar *caps_name; /**< negotiated caps name */
  GstTensorsConfig tensors_config; /**< tensors config from negotiated caps */
  GstTensorMetaInfo meta[NNS_TENSOR_SIZE_LIMIT]; /**< tensor meta (flexible tensor) */
  TestOption option; /**< test option */
  guint buffer_index; /**< index of buffers sent by appsrc */
} TestData;

/**
 * @brief Data for pipeline and test result.
 */
static TestData g_test_data;

/**
 * @brief Free resources in test data.
 */
static void
_free_test_data (TestOption &option)
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

  /** remove temp file */
  if (option.tmpfile) {
    if (g_remove (option.tmpfile) != 0) {
      _print_log ("failed to remove temp file %s", option.tmpfile);
    }
    g_free (option.tmpfile);
  }

  gst_tensors_config_free (&g_test_data.tensors_config);
}

/**
 * @brief Callback for message.
 */
static void
_message_cb (GstBus *bus, GstMessage *message, gpointer user_data)
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
_new_data_cb (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  gsize buf_size;
  guint i, mem_blocks;

  if (!GST_IS_BUFFER (buffer)) {
    _print_log ("received invalid buffer");
    g_test_data.test_failed = TRUE;
    return;
  }

  buf_size = gst_buffer_get_size (buffer);
  mem_blocks = gst_buffer_n_memory (buffer);

  if (g_test_data.received > 0) {
    if (g_test_data.mem_blocks != mem_blocks) {
      _print_log ("invalid memory, old[%d] new[%d]", g_test_data.mem_blocks, mem_blocks);
      g_test_data.test_failed = TRUE;
    }

    if (g_test_data.received_size != buf_size) {
      _print_log ("invalid size, old[%zd] new[%zd]", g_test_data.received_size, buf_size);
      g_test_data.test_failed = TRUE;
    }
  }

  if (DBG) {
    _print_log ("pts %" GST_TIME_FORMAT, GST_TIME_ARGS (GST_BUFFER_PTS (buffer)));
    _print_log ("dts %" GST_TIME_FORMAT, GST_TIME_ARGS (GST_BUFFER_DTS (buffer)));
    _print_log ("duration %" GST_TIME_FORMAT, GST_TIME_ARGS (GST_BUFFER_DURATION (buffer)));
  }

  /** check timestamp */
  if (!GST_CLOCK_TIME_IS_VALID (GST_BUFFER_DTS_OR_PTS (buffer))) {
    g_test_data.invalid_timestamp = TRUE;
  }

  g_test_data.received++;
  g_test_data.received_size = buf_size;
  g_test_data.mem_blocks = mem_blocks;

  _print_log ("new data callback [%d] size [%zd]", g_test_data.received,
      g_test_data.received_size);

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

    if (!gst_tensors_config_from_structure (&g_test_data.tensors_config, structure)) {
      _print_log ("failed to get tensors config from caps");
      g_test_data.test_failed = TRUE;
    }

    if (gst_tensors_info_is_flexible (&g_test_data.tensors_config.info)) {
      /**
       * Cannot get data type and shape from caps.
       * For the test, set type uint8 and dim with buffer size.
       */
      g_test_data.tensors_config.info.num_tensors = mem_blocks;
      for (i = 0; i < mem_blocks; i++) {
        GstMemory *mem = gst_buffer_peek_memory (buffer, i);
        guint mem_size = gst_memory_get_sizes (mem, NULL, NULL);

        g_test_data.tensors_config.info.info[i].type = _NNS_UINT8;
        g_test_data.tensors_config.info.info[i].dimension[0] = mem_size;
        g_test_data.tensors_config.info.info[i].dimension[1] = 1U;
        g_test_data.tensors_config.info.info[i].dimension[2] = 1U;
        g_test_data.tensors_config.info.info[i].dimension[3] = 1U;

        gst_tensor_meta_info_parse_memory (&g_test_data.meta[i], mem);
      }
    }

    /** copy current caps */
    g_test_data.current_caps = gst_caps_copy (caps);
    gst_caps_unref (caps);
    gst_object_unref (sink_pad);
  }
}

/**
 * @brief Callback for signal stream-start.
 */
static void
_stream_start_cb (GstElement *element, gpointer user_data)
{
  g_test_data.start = TRUE;
  _print_log ("stream start callback");
}

/**
 * @brief Callback for signal eos.
 */
static void
_eos_cb (GstElement *element, gpointer user_data)
{
  g_test_data.end = TRUE;
  _print_log ("eos callback");
}

/**
 * @brief Calculate buffer size from test data if test is flexible tensor stream.
 */
static gsize
_calc_expected_buffer_size (gint index)
{
  guint i;
  gsize bsize = 0;

  if (index < 0) {
    for (i = 0; i < g_test_data.mem_blocks; i++) {
      bsize += gst_tensor_meta_info_get_header_size (&g_test_data.meta[i]);
      bsize += gst_tensor_meta_info_get_data_size (&g_test_data.meta[i]);
    }
  } else {
    bsize += gst_tensor_meta_info_get_header_size (&g_test_data.meta[index]);
    bsize += gst_tensor_meta_info_get_data_size (&g_test_data.meta[index]);
  }

  return bsize;
}

/**
 * @brief Timer callback to push buffer.
 * @return True to ensure the timer continues
 */
static gboolean
_test_src_push_timer_cb (gpointer user_data)
{
  GstElement *appsrc;
  GstBuffer *buf;
  GstMapInfo map;
  gboolean timestamps = GPOINTER_TO_INT (user_data);
  gboolean continue_timer = TRUE;
  gboolean is_flexible;
  GstTensorMetaInfo meta;
  GstTensorInfo info;
  GstPad *pad;
  gsize dsize, hsize;

  appsrc = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "appsrc");

  /* push 10 bytes into appsrc */
  dsize = 10;
  hsize = 0;

  gst_tensor_info_init (&info);
  info.type = _NNS_UINT8;
  gst_tensor_parse_dimension ("10:1:1:1", info.dimension);

  pad = gst_element_get_static_pad (appsrc, "src");
  is_flexible = gst_tensor_pad_caps_is_flexible (pad);
  gst_object_unref (pad);

  if (is_flexible) {
    info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

    /* add header when pushing flexible tensor */
    gst_tensor_info_convert_to_meta (&info, &meta);
    hsize = gst_tensor_meta_info_get_header_size (&meta);
    dsize += hsize;
  }

  buf = gst_buffer_new_allocate (NULL, dsize, NULL);

  if (!gst_buffer_map (buf, &map, GST_MAP_WRITE)) {
    g_critical ("failed to get mem map");
    g_test_data.test_failed = TRUE;
    gst_buffer_unref (buf);
    goto error;
  }

  if (is_flexible)
    gst_tensor_meta_info_update_header (&meta, map.data);

  snprintf ((char *) (map.data + hsize), 10, "%d", g_test_data.buffer_index);
  gst_buffer_unmap (buf, &map);

  if (timestamps) {
    GST_BUFFER_PTS (buf) = g_test_data.buffer_index * 100 * GST_MSECOND;
    GST_BUFFER_DURATION (buf) = 100 * GST_MSECOND;
  }

  if (gst_app_src_push_buffer (GST_APP_SRC (appsrc), buf) != GST_FLOW_OK) {
    g_critical ("failed to push buffer [%d]", g_test_data.buffer_index);
    g_test_data.test_failed = TRUE;
    goto error;
  }

error:
  g_test_data.buffer_index++;
  if (g_test_data.buffer_index >= g_test_data.option.num_buffers || g_test_data.test_failed) {
    /* eos */
    if (gst_app_src_end_of_stream (GST_APP_SRC (appsrc)) != GST_FLOW_OK) {
      g_critical ("failed to set eos");
      g_test_data.test_failed = TRUE;
    }

    continue_timer = FALSE;
  }

  gst_object_unref (appsrc);
  return (continue_timer && !g_test_data.test_failed);
}

/**
 * @brief Timer callback to check eos event.
 * @return False to stop timer
 */
static gboolean
_test_src_eos_timer_cb (gpointer user_data)
{
  GMainLoop *loop;

  loop = (GMainLoop *)user_data;

  if (g_main_loop_is_running (loop)) {
    g_critical ("Supposed eos event is not reached, stop main loop.");
    g_main_loop_quit (loop);
  }

  return FALSE;
}

/**
 * @brief Wait until the pipeline processing the buffers
 * @return TRUE on success, FALSE when a time-out occurs
 */
static gboolean
_wait_pipeline_process_buffers (guint expected_num_buffers)
{
  guint timer_count = 0;
  /* Waiting for expected buffers to arrive */
  while (g_test_data.received < expected_num_buffers) {
    timer_count++;
    g_usleep (DEFAULT_TIME_INTERVAL);
    if (timer_count > (TEST_TIME_OUT_MSEC / 10)) {
      return FALSE;
    }
  }
  return TRUE;
}

/**
 * @brief Prepare test pipeline.
 */
static gboolean
_setup_pipeline (TestOption &option)
{
  gchar *str_pipeline;
  gulong handle_id;
  guint i;

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
  g_test_data.option = option;
  g_test_data.buffer_index = 0;
  gst_tensors_config_init (&g_test_data.tensors_config);
  for (i = 0; i < NNS_TENSOR_SIZE_LIMIT; i++)
    gst_tensor_meta_info_init (&g_test_data.meta[i]);

  _print_log ("option num_buffers[%d] test_type[%d]", option.num_buffers, option.test_type);

  g_test_data.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_test_data.loop != NULL);

  switch (option.test_type) {
  case TEST_TYPE_VIDEO_RGB:
    /** video 160x120 RGB */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)%lu/1 ! "
                                    "tensor_converter ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_BGR:
    /** video 160x120 BGR */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=BGR,framerate=(fraction)%lu/1 ! "
                                    "tensor_converter ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_RGB_PADDING:
    /** video 162x120 RGB, remove padding */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=RGB,framerate=(fraction)%lu/1 ! "
                                    "tensor_converter ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_BGR_PADDING:
    /** video 162x120 BGR, remove padding */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=BGR,framerate=(fraction)%lu/1 ! "
                                    "tensor_converter ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_RGB_3F:
    /** video 160x120 RGB, 3 frames */
    str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)%lu/1 ! "
        "tensor_converter frames-per-tensor=3 ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_RGBA:
    /** video 162x120 RGBA */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=RGBA,framerate=(fraction)%lu/1 ! "
                                    "tensor_converter ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_BGRA:
    /** video 162x120 BGRA */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=BGRA,framerate=(fraction)%lu/1 ! "
                                    "tensor_converter ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_ARGB:
    /** video 162x120 ARGB */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=ARGB,framerate=(fraction)%lu/1 ! "
                                    "tensor_converter ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_ABGR:
    /** video 162x120 ABGR */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=ABGR,framerate=(fraction)%lu/1 ! "
                                    "tensor_converter ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_RGBx:
    /** video 162x120 RGBx */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=RGBx,framerate=(fraction)%lu/1 ! "
                                    "tensor_converter ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_xRGB:
    /** video 162x120 xRGB */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=xRGB,framerate=(fraction)%lu/1 ! "
                                    "tensor_converter ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_xBGR:
    /** video 162x120 xBGR */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=xBGR,framerate=(fraction)%lu/1 ! "
                                    "tensor_converter ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_BGRx:
    /** video 162x120 BGRx */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=BGRx,framerate=(fraction)%lu/1 ! "
                                    "tensor_converter ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_BGRx_2F:
    /** video 160x120 BGRx, 2 frames */
    str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=BGRx,framerate=(fraction)%lu/1 ! "
        "tensor_converter frames-per-tensor=2 ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_GRAY8:
    /** video 160x120 GRAY8 */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=GRAY8,framerate=(fraction)%lu/1 ! "
                                    "tensor_converter ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_GRAY8_PADDING:
    /** video 162x120 GRAY8, remove padding */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=GRAY8,framerate=(fraction)%lu/1 ! "
                                    "tensor_converter ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_GRAY8_3F_PADDING:
    /** video 162x120 GRAY8, 3 frames, remove padding */
    str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=162,height=120,format=GRAY8,framerate=(fraction)%lu/1 ! "
        "tensor_converter frames-per-tensor=3 ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_AUDIO_S8:
    /** audio sample rate 16000 (8 bits, signed, little endian) */
    str_pipeline = g_strdup_printf (
        "audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=S8,rate=16000 ! "
        "tensor_converter frames-per-tensor=500 ! tensor_sink name=test_sink",
        option.num_buffers);
    break;
  case TEST_TYPE_AUDIO_U8_100F:
    /** audio sample rate 16000 (8 bits, unsigned, little endian), 100 frames */
    str_pipeline = g_strdup_printf (
        "audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=U8,rate=16000 ! "
        "tensor_converter frames-per-tensor=100 ! tensor_sink name=test_sink",
        option.num_buffers);
    break;
  case TEST_TYPE_AUDIO_S16:
    /** audio sample rate 16000 (16 bits, signed, little endian) */
    str_pipeline = g_strdup_printf (
        "audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=S16LE,rate=16000 ! "
        "tensor_converter frames-per-tensor=500 ! tensor_sink name=test_sink",
        option.num_buffers);
    break;
  case TEST_TYPE_AUDIO_U16_1000F:
    /** audio sample rate 16000 (16 bits, unsigned, little endian), 1000 frames
     */
    str_pipeline = g_strdup_printf (
        "audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=U16LE,rate=16000 ! "
        "tensor_converter frames-per-tensor=1000 ! tensor_sink name=test_sink",
        option.num_buffers);
    break;
  case TEST_TYPE_AUDIO_S32:
    /** audio sample rate 44100 (32 bits, signed, little endian) */
    str_pipeline = g_strdup_printf (
        "audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=S32LE,rate=44100 ! "
        "tensor_converter frames-per-tensor=500 ! tensor_sink name=test_sink",
        option.num_buffers);
    break;
  case TEST_TYPE_AUDIO_U32:
    /** audio sample rate 44100 (32 bits, unsigned, little endian) */
    str_pipeline = g_strdup_printf (
        "audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=U32LE,rate=44100 ! "
        "tensor_converter frames-per-tensor=500 ! tensor_sink name=test_sink",
        option.num_buffers);
    break;
  case TEST_TYPE_AUDIO_F32:
    /** audio sample rate 44100 (32 bits, floating point, little endian) */
    str_pipeline = g_strdup_printf (
        "audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=F32LE,rate=44100 ! "
        "tensor_converter frames-per-tensor=500 ! tensor_sink name=test_sink",
        option.num_buffers);
    break;
  case TEST_TYPE_AUDIO_F64:
    /** audio sample rate 44100 (64 bits, floating point, little endian) */
    str_pipeline = g_strdup_printf (
        "audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=F64LE,rate=44100 ! "
        "tensor_converter frames-per-tensor=500 ! tensor_sink name=test_sink",
        option.num_buffers);
    break;
  case TEST_TYPE_TEXT:
    /** text stream */
    str_pipeline = g_strdup_printf (
        "appsrc name=appsrc caps=text/x-raw,format=utf8 ! "
        "tensor_converter input-dim=20 ! tensor_sink name=test_sink");
    break;
  case TEST_TYPE_TEXT_3F:
    /** text stream 3 frames */
    str_pipeline = g_strdup_printf (
        "appsrc name=appsrc caps=text/x-raw,format=utf8,framerate=(fraction)10/1 ! "
        "tensor_converter name=convert input-dim=30 frames-per-tensor=3 ! tensor_sink name=test_sink");
    break;
  case TEST_TYPE_OCTET_CUR_TS:
    /** byte stream, timestamp current time */
    str_pipeline = g_strdup_printf (
        "appsrc name=appsrc caps=application/octet-stream ! "
        "tensor_converter input-dim=1:10 input-type=uint8 ! tensor_sink name=test_sink");
    break;
  case TEST_TYPE_OCTET_RATE_TS:
    /** byte stream, timestamp framerate */
    str_pipeline = g_strdup_printf (
        "appsrc name=appsrc caps=application/octet-stream,framerate=(fraction)10/1 ! "
        "tensor_converter input-dim=1:10 input-type=uint8 ! tensor_sink name=test_sink");
    break;
  case TEST_TYPE_OCTET_VALID_TS:
    /** byte stream, send buffer with valid timestamp */
    str_pipeline = g_strdup_printf (
        "appsrc name=appsrc caps=application/octet-stream ! "
        "tensor_converter name=convert input-dim=1:10 input-type=uint8 set-timestamp=false ! tensor_sink name=test_sink");
    break;
  case TEST_TYPE_OCTET_INVALID_TS:
    /** byte stream, send buffer with invalid timestamp */
    str_pipeline = g_strdup_printf (
        "appsrc name=appsrc caps=application/octet-stream ! "
        "tensor_converter name=convert input-dim=1:10 input-type=uint8 set-timestamp=false ! tensor_sink name=test_sink");
    break;
  case TEST_TYPE_OCTET_2F:
    /** byte stream, 2 frames */
    str_pipeline = g_strdup_printf (
        "appsrc name=appsrc caps=application/octet-stream,framerate=(fraction)10/1 ! "
        "tensor_converter name=convert input-dim=1:5 input-type=int8 ! tensor_sink name=test_sink");
    break;
  case TEST_TYPE_OCTET_MULTI_TENSORS:
    /** byte stream, byte array to multi tensors */
    str_pipeline = g_strdup_printf (
        "appsrc name=appsrc caps=application/octet-stream,framerate=(fraction)10/1 ! "
        "tensor_converter name=convert input-dim=2,2 input-type=int32,int8 ! tensor_sink name=test_sink");
    break;
  case TEST_TYPE_FLEX_TENSOR_1:
    str_pipeline = g_strdup_printf (
        "appsrc name=appsrc caps=other/tensors,format=flexible,framerate=(fraction)10/1 ! tensor_sink name=test_sink");
    break;
  case TEST_TYPE_FLEX_TENSOR_2:
    str_pipeline = g_strdup_printf (
        "appsrc name=appsrc caps=application/octet-stream,framerate=(fraction)10/1 ! "
        "tensor_converter name=convert input-dim=2,2 input-type=int32,int8 ! "
        "other/tensors,format=flexible ! tensor_sink name=test_sink");
    break;
  case TEST_TYPE_FLEX_TENSOR_3:
    str_pipeline = g_strdup_printf (
        "appsrc name=appsrc caps=other/tensors,format=flexible,framerate=(fraction)10/1 ! "
        "tensor_converter name=convert input-dim=10 input-type=int8 ! tensor_sink name=test_sink");
    break;
  case TEST_TYPE_TENSORS_MUX_1:
    /** other/tensors with tensor_mux */
    str_pipeline = g_strdup_printf (
        "tensor_mux name=mux ! tensor_sink name=test_sink "
        "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_1",
        option.num_buffers, option.num_buffers);
    break;
  case TEST_TYPE_TENSORS_MUX_2:
    /** other/tensors with tensor_mux (flex-tensor) */
    str_pipeline = g_strdup_printf (
        "tensor_mux name=mux ! tensor_sink name=test_sink "
        "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! other/tensors,format=flexible ! mux.sink_1",
        option.num_buffers, option.num_buffers);
    break;
  case TEST_TYPE_TENSORS_MUX_3:
    /** other/tensors with tensor_mux, tensor_demux (flex-tensor) */
    str_pipeline = g_strdup_printf (
        "tensor_mux name=mux ! tensor_demux name=demux "
        "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=320,height=240,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! other/tensors,format=flexible ! mux.sink_1 "
        "demux.src_0 ! queue ! tensor_sink "
        "demux.src_1 ! queue ! tensor_sink name=test_sink",
        option.num_buffers, option.num_buffers);
    break;
  case TEST_TYPE_TENSORS_FLEX_NEGO_FAILED_1:
    /** tensor_mux nego failure case */
    str_pipeline = g_strdup_printf (
        "tensor_mux name=mux ! other/tensors,format=static ! tensor_sink name=test_sink "
        "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! other/tensors,format=flexible ! mux.sink_1",
        option.num_buffers, option.num_buffers);
    break;
  case TEST_TYPE_TENSORS_FLEX_NEGO_FAILED_2:
    /** tensor_demux nego failure case */
    str_pipeline = g_strdup_printf (
        "tensor_mux name=mux ! other/tensors,format=flexible ! tensor_demux name=demux "
        "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=320,height=240,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_1 "
        "demux.src_0 ! queue ! tensor_sink "
        "demux.src_1 ! queue ! other/tensors,format=static ! tensor_sink name=test_sink",
        option.num_buffers, option.num_buffers);
    break;
  case TEST_TYPE_TENSORS_MIX_1:
    /** other/tensors with tensor_mux, tensor_demux */
    str_pipeline = g_strdup_printf (
        "tensor_mux name=mux ! tensor_demux name=demux "
        "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
        "audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! tensor_converter frames-per-tensor=500 ! mux.sink_1 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=64,height=48,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_2 "
        "demux.src_0 ! queue ! tensor_sink "
        "demux.src_1 ! queue ! tensor_sink name=test_sink "
        "demux.src_2 ! queue ! tensor_sink",
        option.num_buffers, option.num_buffers * 3, option.num_buffers + 3);
    break;
  case TEST_TYPE_TENSORS_MIX_2:
    /** other/tensors with tensor_mux, tensor_demux pick 0,2 */
    str_pipeline = g_strdup_printf (
        "tensor_mux name=mux ! tensor_demux name=demux tensorpick=0,2 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
        "audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! tensor_converter frames-per-tensor=500 ! mux.sink_1 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=64,height=48,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_2 "
        "demux. ! queue ! tensor_sink "
        "demux. ! queue ! tensor_sink name=test_sink",
        option.num_buffers, option.num_buffers * 3, option.num_buffers + 3);
    break;
  case TEST_TYPE_TENSORS_MIX_3:
    /** other/tensors with tensor_mux, tensor_demux pick 1,2 */
    str_pipeline = g_strdup_printf (
        "tensor_mux name=mux ! tensor_demux name=demux tensorpick=1,2 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
        "audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! tensor_converter frames-per-tensor=500 ! mux.sink_1 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=64,height=48,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_2 "
        "demux. ! queue ! tensor_sink name=test_sink "
        "demux. ! queue ! tensor_sink",
        option.num_buffers, option.num_buffers * 3, option.num_buffers + 3);
    break;
  case TEST_TYPE_CUSTOM_TENSOR:
    /** video 160x120 RGB, passthrough custom filter */
    str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)%lu/1 ! "
        "tensor_converter ! tensor_filter name=test_filter framework=custom model=%s/libnnstreamer_customfilter_passthrough_variable%s ! tensor_sink name=test_sink",
        option.num_buffers, fps, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION);
    break;
  case TEST_TYPE_CUSTOM_TENSORS_1:
    /** other/tensors with tensormux, passthrough custom filter */
    str_pipeline = g_strdup_printf (
        "tensor_mux name=mux ! tensor_filter name=test_filter framework=custom model=%s/libnnstreamer_customfilter_passthrough_variable%s ! tensor_sink name=test_sink "
        "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=120,height=80,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_1 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=64,height=48,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_2",
        custom_dir ? custom_dir : "./nnstreamer_example", NNSTREAMER_SO_FILE_EXTENSION,
        option.num_buffers, option.num_buffers, option.num_buffers);
    break;
  case TEST_TYPE_CUSTOM_TENSORS_2:
    /** other/tensors with tensormux, passthrough custom filter */
    str_pipeline = g_strdup_printf (
        "tensor_mux name=mux ! tensor_filter name=test_filter framework=custom model=%s/libnnstreamer_customfilter_passthrough_variable%s "
        "input=3:160:120:1,3:120:80:1,3:64:48:1 output=3:160:120:1,3:120:80:1,3:64:48:1 inputtype=uint8,uint8,uint8 outputtype=uint8,uint8,uint8 "
        "inputlayout=NCHW,NHWC,NONE outputlayout=ANY,NHCW,NCHW ! tensor_sink name=test_sink "
        "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=120,height=80,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_1 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=64,height=48,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_2",
        custom_dir ? custom_dir : "./nnstreamer_example", NNSTREAMER_SO_FILE_EXTENSION,
        option.num_buffers, option.num_buffers, option.num_buffers);
    break;
  case TEST_TYPE_TENSOR_CAP_1:
    /** other/tensor out, caps are specifed*/
    str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)%lu/1 ! "
        "tensor_converter ! other/tensor,format=static ! tensor_sink name=test_sink async=false",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_TENSOR_CAP_2:
    /** other/tensor out, caps are not specifed (other/tensor or other/tensors) */
    str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)%lu/1 ! "
        "tensor_converter ! tensor_sink name=test_sink async=false",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_TENSORS_CAP_1:
    /** other/tensors, caps are specifed (num_tensors is 1) */
    str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)%lu/1 ! "
        "tensor_converter ! other/tensors,format=static ! tensor_sink name=test_sink async=false",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_TENSORS_CAP_2:
    /** other/tensors, caps are not specifed (num_tensors is 3) */
    str_pipeline = g_strdup_printf (
        "tensor_mux name=mux ! tensor_sink name=test_sink "
        "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=120,height=80,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_1 "
        "videotestsrc num-buffers=%d ! video/x-raw,width=64,height=48,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_2",
        option.num_buffers, option.num_buffers, option.num_buffers);
    break;
  case TEST_TYPE_CUSTOM_MULTI:
    /* multiple custom filters */
    str_pipeline = g_strdup_printf (
        "tensor_mux name=mux ! tensor_sink name=test_sink "
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! "
        "tensor_filter framework=custom model=%s/libnnstreamer_customfilter_passthrough_variable%s ! mux.sink_0 "
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=280,height=40,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! "
        "tensor_filter framework=custom model=%s/libnnstreamer_customfilter_passthrough%s ! mux.sink_1 "
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=320,height=240,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! "
        "tensor_filter framework=custom model=%s/libnnstreamer_customfilter_scaler%s custom=640x480 ! mux.sink_2",
        option.num_buffers, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.num_buffers,
        custom_dir ? custom_dir : "./nnstreamer_example", NNSTREAMER_SO_FILE_EXTENSION,
        option.num_buffers, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION);
    break;
  case TEST_TYPE_CUSTOM_BUF_DROP:
    /* audio stream to test buffer-drop using custom filter */
    str_pipeline = g_strdup_printf (
        "audiotestsrc num-buffers=%d samplesperbuffer=200 ! audioconvert ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! "
        "tensor_converter frames-per-tensor=200 ! tensor_filter framework=custom model=%s/libnnscustom_drop_buffer%s ! tensor_sink name=test_sink",
        option.num_buffers, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION);
    break;
  case TEST_TYPE_CUSTOM_PASSTHROUGH:
    /* video 160x120 RGB, passthrough custom filter without so file */
    str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)%lu/1 ! "
        "tensor_converter ! tensor_filter framework=custom-passthrough ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_NEGO_FAILED:
    /** caps negotiation failed */
    str_pipeline = g_strdup_printf ("videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)%lu/1 ! "
                                    "videoconvert ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_RGB_SPLIT:
    /** video stream with tensor_split */
    str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)%lu/1 ! "
        "tensor_converter ! tensor_split silent=TRUE name=split tensorseg=1:160:120:1,1:160:120:1,1:160:120:1 tensorpick=0,1,2 "
        "split.src_0 ! queue ! tensor_sink "
        "split.src_1 ! queue ! tensor_sink name=test_sink "
        "split.src_2 ! queue ! tensor_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_RGB_AGGR_1:
    /** video stream with tensor_aggregator */
    str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)%lu/1 ! "
        "tensor_converter ! tensor_aggregator frames-out=10 frames-flush=5 frames-dim=3 ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_RGB_AGGR_2:
    /** video stream with tensor_aggregator */
    str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)%lu/1 ! "
        "tensor_converter ! tensor_aggregator frames-out=10 frames-flush=5 frames-dim=1 ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_VIDEO_RGB_AGGR_3:
    /** video stream with tensor_aggregator */
    str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=64,height=48,format=RGB,framerate=(fraction)%lu/1 ! "
        "tensor_converter ! tensor_aggregator frames-out=10 frames-dim=1 concat=false ! "
        "tensor_aggregator frames-in=10 frames-out=8 frames-flush=10 frames-dim=1 ! tensor_sink name=test_sink",
        option.num_buffers, fps);
    break;
  case TEST_TYPE_AUDIO_S16_AGGR:
    /** audio stream with tensor_aggregator, 4 buffers with 2000 frames */
    str_pipeline = g_strdup_printf (
        "audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! "
        "tensor_converter frames-per-tensor=500 ! tensor_aggregator frames-in=500 frames-out=2000 frames-dim=1 ! tensor_sink name=test_sink",
        option.num_buffers);
    break;
  case TEST_TYPE_AUDIO_U16_AGGR:
    /** audio stream with tensor_aggregator, divided into 5 buffers with 100
     * frames */
    str_pipeline = g_strdup_printf (
        "audiotestsrc num-buffers=%d samplesperbuffer=500 ! audioconvert ! audio/x-raw,format=U16LE,rate=16000,channels=1 ! "
        "tensor_converter frames-per-tensor=500 ! tensor_aggregator frames-in=500 frames-out=100 frames-dim=1 ! tensor_sink name=test_sink",
        option.num_buffers);
    break;
  case TEST_TYPE_TRANSFORM_CAPS_NEGO_1:
    /** test for caps negotiation in tensor_transform, push data to
     * tensor_transform directly. */
    str_pipeline = g_strdup_printf (
        "appsrc name=appsrc ! other/tensor,type=(string)uint8,dimension=(string)10:1:1:1,framerate=(fraction)0/1 ! "
        "tensor_transform mode=typecast option=%s ! tensor_sink name=test_sink",
        gst_tensor_get_type_string (option.t_type));
    break;
  case TEST_TYPE_TRANSFORM_CAPS_NEGO_2:
    /** test for caps negotiation in tensor_transform, push data to
     * tensor_transform directly. */
    str_pipeline = g_strdup_printf (
        "appsrc name=appsrc ! other/tensor,type=(string)uint8,dimension=(string)10:1:1:1,framerate=(fraction)0/1 ! "
        "tensor_transform mode=arithmetic option=typecast:%s,add:1 ! tensor_sink name=test_sink",
        gst_tensor_get_type_string (option.t_type));
    break;
  case TEST_TYPE_TRANSFORM_TENSORS:
    /* tensors stream with tensor_transform, typecast to float32 */
    str_pipeline = g_strdup_printf (
        "tensor_mux name=mux ! tensor_transform mode=typecast option=float32 ! tensor_sink name=test_sink "
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=280,height=40,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_1 "
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=320,height=240,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_2",
        option.num_buffers, option.num_buffers, option.num_buffers);
    break;
  case TEST_TYPE_TRANSFORM_APPLY:
    /* tensors stream with apply option, typecast to float32 */
    str_pipeline = g_strdup_printf (
        "tensor_mux name=mux ! tensor_transform mode=typecast option=float32 apply=2,0 ! tensor_sink name=test_sink "
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=280,height=40,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_1 "
        "videotestsrc num-buffers=%d ! videoconvert ! video/x-raw,width=320,height=240,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_2",
        option.num_buffers, option.num_buffers, option.num_buffers);
    break;
  case TEST_TYPE_TYPECAST:
    /** text stream to test typecast */
    str_pipeline = g_strdup_printf (
        "appsrc name=appsrc caps=text/x-raw,format=utf8 ! "
        "tensor_converter input-dim=10 ! tensor_transform mode=typecast option=%s ! tensor_sink name=test_sink",
        gst_tensor_get_type_string (option.t_type));
    break;
  case TEST_TYPE_ISSUE739_MUX_PARALLEL_1:
    /** 4x4 tensor stream, different FPS, tensor_mux them @ slowest */
    str_pipeline = g_strdup_printf (
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! mux.sink_0 "
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=25/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! mux.sink_1 "
        "tensor_mux sync-mode=slowest name=mux ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! tee name=t ! queue ! tensor_sink name=test_sink t. ! queue ! filesink location=%s",
        option.num_buffers * 10, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.num_buffers * 25,
        custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.tmpfile);
    break;
  case TEST_TYPE_ISSUE739_MUX_PARALLEL_2:
    /** 4x4 tensor stream, different FPS, tensor_mux them @ basepad*/
    str_pipeline = g_strdup_printf (
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! mux.sink_0 "
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=25/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! mux.sink_1 "
        "tensor_mux sync-mode=basepad sync-option=0:0 name=mux ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! tee name=t ! queue ! tensor_sink name=test_sink t. ! queue ! filesink location=%s",
        option.num_buffers * 10, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.num_buffers * 25,
        custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.tmpfile);
    break;
  case TEST_TYPE_ISSUE739_MUX_PARALLEL_3:
    /** 4x4 tensor stream, different FPS, tensor_mux them @ basepad*/
    str_pipeline = g_strdup_printf (
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! mux.sink_0 "
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=25/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! mux.sink_1 "
        "tensor_mux sync-mode=basepad sync-option=1:0 name=mux ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! tee name=t ! queue ! tensor_sink name=test_sink t. ! queue ! filesink location=%s",
        option.num_buffers * 10, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.num_buffers * 25,
        custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.tmpfile);
    break;
  case TEST_TYPE_ISSUE739_MUX_PARALLEL_4:
    /** 4x4 tensor stream, different FPS, tensor_mux them @ basepad*/
    /** @todo Because of the bug mentioned in #739, this is not registered as gtest case, yet */
    str_pipeline = g_strdup_printf (
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! mux.sink_0 "
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=25/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! mux.sink_1 "
        "tensor_mux sync-mode=basepad sync-option=1:1000000000 name=mux ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! tee name=t ! queue ! tensor_sink name=test_sink t. ! queue ! filesink location=%s",
        option.num_buffers * 10, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.num_buffers * 25,
        custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.tmpfile);
    break;
  case TEST_TYPE_ISSUE739_MERGE_PARALLEL_1:
    /** 4x4 tensor stream, different FPS, tensor_merge them @ slowest */
    str_pipeline = g_strdup_printf (
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! mux.sink_0 "
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=25/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! mux.sink_1 "
        "tensor_merge mode=linear option=3 sync-mode=slowest name=mux ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! tee name=t ! queue ! tensor_sink name=test_sink t. ! queue ! filesink location=%s",
        option.num_buffers * 10, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.num_buffers * 25,
        custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.tmpfile);
    break;
  case TEST_TYPE_ISSUE739_MERGE_PARALLEL_2:
    /** 4x4 tensor stream, different FPS, tensor_merge them @ basepad*/
    str_pipeline = g_strdup_printf (
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! mux.sink_0 "
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=25/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! mux.sink_1 "
        "tensor_merge mode=linear option=3 sync-mode=basepad sync-option=0:0 name=mux ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! tee name=t ! queue ! tensor_sink name=test_sink t. ! queue ! filesink location=%s",
        option.num_buffers * 10, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.num_buffers * 25,
        custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.tmpfile);
    break;
  case TEST_TYPE_ISSUE739_MERGE_PARALLEL_3:
    /** 4x4 tensor stream, different FPS, tensor_merge them @ basepad*/
    str_pipeline = g_strdup_printf (
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! mux.sink_0 "
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=25/1 ! tensor_converter ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! mux.sink_1 "
        "tensor_merge mode=linear option=3 sync-mode=basepad sync-option=1:0 name=mux ! tensor_filter framework=custom model=%s/libnnscustom_framecounter%s ! tee name=t ! queue ! tensor_sink name=test_sink t. ! queue ! filesink location=%s",
        option.num_buffers * 10, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.num_buffers * 25,
        custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, custom_dir ? custom_dir : "./nnstreamer_example",
        NNSTREAMER_SO_FILE_EXTENSION, option.tmpfile);
    break;
  /** @todo Add tensor_mux policy = more policies! */
  case TEST_TYPE_DECODER_PROPERTY:
    str_pipeline = g_strdup_printf (
        "videotestsrc pattern=snow num-buffers=%d ! video/x-raw,format=BGRx,height=4,width=4,framerate=10/1 ! tensor_converter ! tensor_decoder mode=direct_video name=decoder option1=whatthehell option2=isgoingon option3=nothing option4=iswrong option5=keepcalm option6=\"and have a break\" option7=\"iwill=not\" option8=\"break=your\" option9=\"system=1234\" ! video/x-raw,format=BGRx ! tensor_converter ! tensor_sink name=test_sink ",
        option.num_buffers);
    break;
  case TEST_CUSTOM_EASY_ICF_01:
    str_pipeline = g_strdup_printf ("appsrc name=appsrc caps=application/octet-stream ! "
                                    "tensor_converter input-dim=1:10 input-type=uint8 ! "
                                    "tensor_filter framework=custom-easy model=safe_memcpy_10x10 ! "
                                    "tensor_filter framework=custom-easy model=safe_memcpy_10x10 ! "
                                    "tensor_filter framework=custom-easy model=safe_memcpy_10x10 ! "
                                    "tensor_sink name=test_sink");
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
  handle_id = g_signal_connect (g_test_data.bus, "message", (GCallback)_message_cb, NULL);
  _check_cond_err (handle_id > 0);

  g_test_data.sink = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "test_sink");
  _check_cond_err (g_test_data.sink != NULL);

  if (DBG) {
    /** print logs */
    g_object_set (g_test_data.sink, "silent", (gboolean)FALSE, NULL);
  }
  g_object_set (g_test_data.sink, "sync", (gboolean)option.need_sync, NULL);

  /** signal for new data */
  handle_id = g_signal_connect (g_test_data.sink, "new-data", (GCallback)_new_data_cb, NULL);
  _check_cond_err (handle_id > 0);

  g_test_data.status = TEST_INIT;
  return TRUE;

error:
  g_test_data.test_failed = TRUE;
  _free_test_data (option);
  return FALSE;
}

/**
 * @brief Test for tensor sink properties.
 */
TEST (tensorSinkTest, properties)
{
  guint rate, res_rate;
  gint64 lateness, res_lateness;
  gboolean silent, res_silent;
  gboolean emit, res_emit;
  gboolean sync, res_sync;
  gboolean qos, res_qos;
  TestOption option = { 1, TEST_TYPE_VIDEO_RGB };

  option.need_sync = TRUE;
  ASSERT_TRUE (_setup_pipeline (option));

  /** default signal-rate is 0 */
  g_object_get (g_test_data.sink, "signal-rate", &rate, NULL);
  EXPECT_EQ (rate, 0U);

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
  _free_test_data (option);
}

/**
 * @brief Test for tensor sink signals.
 */
TEST (tensorSinkTest, signals)
{
  const guint num_buffers = 5;
  gulong handle_id;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB };

  ASSERT_TRUE (_setup_pipeline (option));

  /** tensor sink signals */
  handle_id = g_signal_connect (
      g_test_data.sink, "stream-start", (GCallback)_stream_start_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  handle_id = g_signal_connect (g_test_data.sink, "eos", (GCallback)_eos_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.start, TRUE);
  EXPECT_EQ (g_test_data.end, TRUE);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check caps and config for tensor */
  {
    GstCaps *caps;
    GstStructure *structure;
    GstTensorsConfig config;

    caps = gst_tensors_caps_from_config (&g_test_data.tensors_config);
    structure = gst_caps_get_structure (caps, 0);

    EXPECT_TRUE (gst_tensors_config_from_structure (&config, structure));
    EXPECT_TRUE (gst_tensors_config_is_equal (&config, &g_test_data.tensors_config));

    gst_caps_unref (caps);
    gst_tensors_config_free (&config);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for tensor sink emit-signal (case for no signal).
 */
TEST (tensorSinkTest, emitSignal)
{
  const guint num_buffers = 5;
  gulong handle_id;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB };

  ASSERT_TRUE (_setup_pipeline (option));

  /** set emit-signal FALSE (no signal) */
  g_object_set (g_test_data.sink, "emit-signal", (gboolean)FALSE, NULL);

  /** tensor sink signals */
  handle_id = g_signal_connect (
      g_test_data.sink, "stream-start", (GCallback)_stream_start_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  handle_id = g_signal_connect (g_test_data.sink, "eos", (GCallback)_eos_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  g_usleep (jitter);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, 0U);
  EXPECT_EQ (g_test_data.start, FALSE);
  EXPECT_EQ (g_test_data.end, FALSE);

  /** check caps name is null (no signal) */
  EXPECT_TRUE (g_test_data.caps_name == NULL);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for tensor sink signal-rate.
 */
TEST (tensorSinkTest, signalRate)
{
  const guint num_buffers = 6;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB };

  ASSERT_TRUE (_setup_pipeline (option));

  /** set signal-rate */
  g_object_set (g_test_data.sink, "signal-rate", (guint) (fps / 2), NULL);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  g_usleep (jitter);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_TRUE (g_test_data.received < num_buffers);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check caps and config for tensor */
  {
    GstCaps *caps;
    GstStructure *structure;
    GstTensorsConfig config;

    caps = gst_tensors_caps_from_config (&g_test_data.tensors_config);
    structure = gst_caps_get_structure (caps, 0);

    EXPECT_TRUE (gst_tensors_config_from_structure (&config, structure));
    EXPECT_TRUE (gst_tensors_config_is_equal (&config, &g_test_data.tensors_config));

    gst_caps_unref (caps);
    gst_tensors_config_free (&config);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for caps negotiation failed.
 */
TEST (tensorSinkTest, capsError_n)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_NEGO_FAILED };

  /** failed : cannot link videoconvert and tensor_sink */
  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  g_usleep (jitter);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check error message */
  EXPECT_EQ (g_test_data.status, TEST_ERR_MESSAGE);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, 0U);

  EXPECT_TRUE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for other/tensors with tensor_mux.
 */
TEST (tensorStreamTest, muxStaticTensors)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSORS_MUX_1 };
  guint i;

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 2U);
  EXPECT_EQ (g_test_data.received_size, 3U * 160 * 120 * 2);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensors config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.num_tensors, 2U);

  for (i = 0; i < g_test_data.tensors_config.info.num_tensors; i++) {
    EXPECT_EQ (g_test_data.tensors_config.info.info[i].type, _NNS_UINT8);
    EXPECT_EQ (g_test_data.tensors_config.info.info[i].dimension[0], 3U);
    EXPECT_EQ (g_test_data.tensors_config.info.info[i].dimension[1], 160U);
    EXPECT_EQ (g_test_data.tensors_config.info.info[i].dimension[2], 120U);
    EXPECT_EQ (g_test_data.tensors_config.info.info[i].dimension[3], 1U);
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
    EXPECT_TRUE (gst_tensors_config_is_equal (&config, &g_test_data.tensors_config));

    gst_caps_unref (caps);
    gst_tensors_config_free (&config);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for flexible tensors with tensor_mux.
 */
TEST (tensorStreamTest, muxFlexTensors)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSORS_MUX_2 };
  guint i;

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 2U);
  EXPECT_EQ (g_test_data.received_size, _calc_expected_buffer_size (-1));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensors config for flex tensor */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_TRUE (gst_tensors_info_is_flexible (&g_test_data.tensors_config.info));
  EXPECT_EQ (g_test_data.tensors_config.info.num_tensors, 2U);

  for (i = 0; i < g_test_data.tensors_config.info.num_tensors; i++) {
    EXPECT_EQ (g_test_data.tensors_config.info.info[i].dimension[0], _calc_expected_buffer_size (i));
  }

  EXPECT_EQ (g_test_data.tensors_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  /* check meta info */
  EXPECT_EQ (g_test_data.meta[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.meta[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.meta[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.meta[0].dimension[2], 120U);
  EXPECT_EQ ((media_type) g_test_data.meta[0].media_type, _NNS_TENSOR);

  EXPECT_EQ (g_test_data.meta[1].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.meta[1].dimension[0], 3U);
  EXPECT_EQ (g_test_data.meta[1].dimension[1], 160U);
  EXPECT_EQ (g_test_data.meta[1].dimension[2], 120U);
  EXPECT_EQ ((media_type) g_test_data.meta[1].media_type, _NNS_VIDEO);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for flexible tensors with tensor_mux and tensor_demux.
 */
TEST (tensorStreamTest, demuxFlexTensors)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSORS_MUX_3 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, _calc_expected_buffer_size (-1));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for flex tensor */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_TRUE (gst_tensors_info_is_flexible (&g_test_data.tensors_config.info));
  EXPECT_EQ (g_test_data.tensors_config.info.num_tensors, 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], _calc_expected_buffer_size (0));
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  /* check meta info */
  EXPECT_EQ (g_test_data.meta[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.meta[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.meta[0].dimension[1], 320U);
  EXPECT_EQ (g_test_data.meta[0].dimension[2], 240U);
  EXPECT_EQ ((media_type) g_test_data.meta[0].media_type, _NNS_VIDEO);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for flexible tensors with tensor_mux (nego failure).
 */
TEST (tensorStreamTest, muxFlexNegoFailed_n)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSORS_FLEX_NEGO_FAILED_1 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  g_usleep (jitter);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /* failed : cannot link mux (flex) and tensor_sink (static) */
  EXPECT_EQ (g_test_data.status, TEST_ERR_MESSAGE);
  EXPECT_EQ (g_test_data.received, 0U);
}

/**
 * @brief Test for flexible tensors with tensor_demux (nego failure).
 */
TEST (tensorStreamTest, demuxFlexNegoFailed_n)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSORS_FLEX_NEGO_FAILED_2 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  g_usleep (jitter);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /* failed : cannot link demux (flex) and tensor_sink (static) */
  EXPECT_EQ (g_test_data.status, TEST_ERR_MESSAGE);
  EXPECT_EQ (g_test_data.received, 0U);
}

/**
 * @brief Test for video format RGB.
 */
TEST (tensorStreamTest, videoRgb)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 3U * 160 * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format BGR.
 */
TEST (tensorStreamTest, videoBgr)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_BGR };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 3U * 160 * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format RGB, remove padding.
 */
TEST (tensorStreamTest, videoRgbPadding)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB_PADDING };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 3U * 162 * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 162U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format BGR, remove padding.
 */
TEST (tensorStreamTest, videoBgrPadding)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_BGR_PADDING };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 3U * 162 * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 162U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format RGB, 3 frames from tensor_converter.
 */
TEST (tensorStreamTest, videoRgb3f)
{
  const guint num_buffers = 7;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB_3F };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers / 3));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers / 3);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 3U * 160 * 120 * 3);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 3U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format RGBA.
 */
TEST (tensorStreamTest, videoRgba)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGBA };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U * 162 * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 4U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 162U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format BGRA.
 */
TEST (tensorStreamTest, videoBgra)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_BGRA };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U * 162 * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 4U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 162U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format ARGB.
 */
TEST (tensorStreamTest, videoArgb)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_ARGB };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U * 162 * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 4U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 162U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format ABGR.
 */
TEST (tensorStreamTest, videoAbgr)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_ABGR };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U * 162 * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 4U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 162U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format RGBx.
 */
TEST (tensorStreamTest, videoRgbx)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGBx };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U * 162 * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 4U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 162U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format xRGB.
 */
TEST (tensorStreamTest, videoXrgb)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_xRGB };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U * 162 * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 4U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 162U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format xBGR.
 */
TEST (tensorStreamTest, videoXbgr)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_xBGR };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U * 162 * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 4U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 162U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format BGRx.
 */
TEST (tensorStreamTest, videoBgrx)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_BGRx };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U * 162 * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 4U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 162U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format BGRx, 2 frames from tensor_converter.
 */
TEST (tensorStreamTest, videoBgrx2f)
{
  const guint num_buffers = 6;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_BGRx_2F };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers / 2));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers / 2);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U * 160 * 120 * 2);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 4U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 2U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format GRAY8.
 */
TEST (tensorStreamTest, videoGray8)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_GRAY8 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 160U * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format GRAY8, remove padding.
 */
TEST (tensorStreamTest, videoGray8Padding)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_GRAY8_PADDING };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 162U * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 162U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video format GRAY8, 3 frames from tensor_converter, remove padding.
 */
TEST (tensorStreamTest, videoGray83fPadding)
{
  const guint num_buffers = 6;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_GRAY8_3F_PADDING };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers / 3));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers / 3);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 162U * 120 * 3);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 162U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 3U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for audio format S8.
 */
TEST (tensorStreamTest, audioS8)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_S8 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 500U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_INT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 500U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 16000);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for audio format U8, 100 frames from tensor_converter.
 */
TEST (tensorStreamTest, audioU8100f)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_U8_100F };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers * 5));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers * 5);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 100U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 100U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 16000);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for audio format S16.
 */
TEST (tensorStreamTest, audioS16)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_S16 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 500U * 2);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_INT16);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 500U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 16000);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for audio format U16, 1000 frames from tensor_converter.
 */
TEST (tensorStreamTest, audioU161000f)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_U16_1000F };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers / 2));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers / 2);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 500U * 2 * 2);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT16);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 1000U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 16000);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for audio format S32.
 */
TEST (tensorStreamTest, audioS32)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_S32 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 500U * 4);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_INT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 500U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 44100);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for audio format U32.
 */
TEST (tensorStreamTest, audioU32)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_U32 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 500U * 4);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 500U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 44100);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for audio format F32.
 */
TEST (tensorStreamTest, audioF32)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_F32 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 500U * 4);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_FLOAT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 500U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 44100);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for audio format F64.
 */
TEST (tensorStreamTest, audioF64)
{
  const guint num_buffers = 5; /** 5 * 500 frames */
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_F64 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 500U * 8);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_FLOAT64);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 500U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 44100);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for text format utf8.
 */
TEST (tensorStreamTest, textUtf8)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_TEXT };
  guint timeout_id;

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  g_timeout_add (100, _test_src_push_timer_cb, GINT_TO_POINTER (TRUE));

  timeout_id = g_timeout_add (5000, _test_src_eos_timer_cb, g_test_data.loop);
  g_main_loop_run (g_test_data.loop);
  g_source_remove (timeout_id);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 20U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 20U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for text format utf8, 3 frames from tensor_converter.
 */
TEST (tensorStreamTest, textUtf83f)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_TEXT_3F };
  GstElement *convert;
  gchar *prop_str;
  gboolean prop_bool;
  guint prop_uint;
  guint timeout_id;

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
  EXPECT_EQ (prop_uint, 3U);

  gst_object_unref (convert);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  g_timeout_add (100, _test_src_push_timer_cb, GINT_TO_POINTER (TRUE));

  timeout_id = g_timeout_add (5000, _test_src_eos_timer_cb, g_test_data.loop);
  g_main_loop_run (g_test_data.loop);
  g_source_remove (timeout_id);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers / 3));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers / 3);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 30U * 3);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 30U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 10);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for octet stream.
 */
TEST (tensorStreamTest, octetCurrentTs)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_OCTET_CUR_TS };
  guint timeout_id;

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  g_timeout_add (100, _test_src_push_timer_cb, GINT_TO_POINTER (FALSE));

  timeout_id = g_timeout_add (5000, _test_src_eos_timer_cb, g_test_data.loop);
  g_main_loop_run (g_test_data.loop);
  g_source_remove (timeout_id);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 10U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 10U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for octet stream.
 */
TEST (tensorStreamTest, octetFramerateTs)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_OCTET_RATE_TS };
  guint timeout_id;

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  g_timeout_add (100, _test_src_push_timer_cb, GINT_TO_POINTER (FALSE));

  timeout_id = g_timeout_add (5000, _test_src_eos_timer_cb, g_test_data.loop);
  g_main_loop_run (g_test_data.loop);
  g_source_remove (timeout_id);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 10U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 10U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 10);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for octet stream.
 */
TEST (tensorStreamTest, octetValidTs)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_OCTET_VALID_TS };
  GstElement *convert;
  gchar *prop_str;
  gboolean prop_bool;
  guint prop_uint;
  guint timeout_id;

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
  EXPECT_EQ (prop_uint, 1U);

  gst_object_unref (convert);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  g_timeout_add (100, _test_src_push_timer_cb, GINT_TO_POINTER (TRUE));

  timeout_id = g_timeout_add (5000, _test_src_eos_timer_cb, g_test_data.loop);
  g_main_loop_run (g_test_data.loop);
  g_source_remove (timeout_id);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 10U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 10U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for octet stream.
 */
TEST (tensorStreamTest, octetInvalidTs)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_OCTET_INVALID_TS };
  GstElement *convert;
  gchar *prop_str;
  gboolean prop_bool;
  guint prop_uint;
  guint timeout_id;

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
  EXPECT_EQ (prop_uint, 1U);

  gst_object_unref (convert);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  g_timeout_add (100, _test_src_push_timer_cb, GINT_TO_POINTER (FALSE));

  timeout_id = g_timeout_add (5000, _test_src_eos_timer_cb, g_test_data.loop);
  g_main_loop_run (g_test_data.loop);
  g_source_remove (timeout_id);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 10U);

  /** check invalid timestamp */
  EXPECT_TRUE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 10U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for octet stream.
 */
TEST (tensorStreamTest, octet2f)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_OCTET_2F };
  GstElement *convert;
  gchar *prop_str;
  gboolean prop_bool;
  guint prop_uint;
  guint timeout_id;

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
  EXPECT_EQ (prop_uint, 1U);

  gst_object_unref (convert);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  g_timeout_add (100, _test_src_push_timer_cb, GINT_TO_POINTER (FALSE));

  timeout_id = g_timeout_add (5000, _test_src_eos_timer_cb, g_test_data.loop);
  g_main_loop_run (g_test_data.loop);
  g_source_remove (timeout_id);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers * 2));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers * 2);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 5U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_INT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 5U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 10);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for octet stream.
 */
TEST (tensorStreamTest, octetMultiTensors)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_OCTET_MULTI_TENSORS };
  GstElement *convert;
  gchar *prop_str;
  guint prop_uint;
  guint timeout_id;

  ASSERT_TRUE (_setup_pipeline (option));

  /* tensor_converter properties */
  convert = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "convert");
  ASSERT_TRUE (convert != NULL);

  g_object_get (convert, "input-dim", &prop_str, NULL);
  EXPECT_STREQ (prop_str, "2:1:1:1,2:1:1:1");
  g_free (prop_str);

  g_object_get (convert, "input-type", &prop_str, NULL);
  EXPECT_STREQ (prop_str, "int32,int8");
  g_free (prop_str);

  g_object_get (convert, "frames-per-tensor", &prop_uint, NULL);
  EXPECT_EQ (prop_uint, 1U);

  gst_object_unref (convert);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  g_timeout_add (100, _test_src_push_timer_cb, GINT_TO_POINTER (TRUE));

  timeout_id = g_timeout_add (5000, _test_src_eos_timer_cb, g_test_data.loop);
  g_main_loop_run (g_test_data.loop);
  g_source_remove (timeout_id);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 2U);
  EXPECT_EQ (g_test_data.received_size, 10U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for multi tensors */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.num_tensors, 2U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_INT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 2U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[1].type, _NNS_INT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[0], 2U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[1], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.rate_n, 10);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for flexible tensor stream.
 */
TEST (tensorStreamTest, flexOnSink)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_FLEX_TENSOR_1 };
  guint timeout_id;

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  g_timeout_add (100, _test_src_push_timer_cb, GINT_TO_POINTER (TRUE));

  timeout_id = g_timeout_add (5000, _test_src_eos_timer_cb, g_test_data.loop);
  g_main_loop_run (g_test_data.loop);
  g_source_remove (timeout_id);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, _calc_expected_buffer_size (-1));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check buffer size from tensors config */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_TRUE (gst_tensors_info_is_flexible (&g_test_data.tensors_config.info));
  EXPECT_EQ (g_test_data.tensors_config.info.num_tensors, 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], _calc_expected_buffer_size (0));
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 10);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  /* check meta info */
  EXPECT_EQ (g_test_data.meta[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.meta[0].dimension[0], 10U);
  EXPECT_EQ (g_test_data.meta[0].format, _NNS_TENSOR_FORMAT_FLEXIBLE);
  EXPECT_EQ ((media_type) g_test_data.meta[0].media_type, _NNS_TENSOR);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for flexible tensor stream.
 */
TEST (tensorStreamTest, staticToFlex)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_FLEX_TENSOR_2 };
  guint timeout_id;

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  g_timeout_add (100, _test_src_push_timer_cb, GINT_TO_POINTER (TRUE));

  timeout_id = g_timeout_add (5000, _test_src_eos_timer_cb, g_test_data.loop);
  g_main_loop_run (g_test_data.loop);
  g_source_remove (timeout_id);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 2U);
  EXPECT_EQ (g_test_data.received_size, _calc_expected_buffer_size (-1));

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check buffer size from tensors config */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_TRUE (gst_tensors_info_is_flexible (&g_test_data.tensors_config.info));
  EXPECT_EQ (g_test_data.tensors_config.info.num_tensors, 2U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], _calc_expected_buffer_size (0));
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[0], _calc_expected_buffer_size (1));
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 10);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  /* check meta info */
  EXPECT_EQ (g_test_data.meta[0].type, _NNS_INT32);
  EXPECT_EQ (g_test_data.meta[0].dimension[0], 2U);
  EXPECT_EQ ((media_type) g_test_data.meta[0].media_type, _NNS_OCTET);

  EXPECT_EQ (g_test_data.meta[1].type, _NNS_INT8);
  EXPECT_EQ (g_test_data.meta[1].dimension[0], 2U);
  EXPECT_EQ ((media_type) g_test_data.meta[1].media_type, _NNS_OCTET);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for flexible tensor stream.
 */
TEST (tensorStreamTest, flexToStatic)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_FLEX_TENSOR_3 };
  guint timeout_id;

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  g_timeout_add (100, _test_src_push_timer_cb, GINT_TO_POINTER (TRUE));

  timeout_id = g_timeout_add (5000, _test_src_eos_timer_cb, g_test_data.loop);
  g_main_loop_run (g_test_data.loop);
  g_source_remove (timeout_id);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 10U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_INT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 10U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 10);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for other/tensor, passthrough custom filter.
 */
TEST (tensorStreamTest, customFilterTensor)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_CUSTOM_TENSOR };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 3U * 160 * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  /** check caps and config for tensor */
  {
    GstCaps *caps;
    GstStructure *structure;
    GstTensorsConfig config;

    caps = gst_tensors_caps_from_config (&g_test_data.tensors_config);
    structure = gst_caps_get_structure (caps, 0);

    EXPECT_TRUE (gst_tensors_config_from_structure (&config, structure));
    EXPECT_TRUE (gst_tensors_config_is_equal (&config, &g_test_data.tensors_config));

    gst_caps_unref (caps);
    gst_tensors_config_free (&config);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for other/tensors, passthrough custom filter.
 */
TEST (tensorStreamTest, customFilterTensors)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_CUSTOM_TENSORS_1 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 3U);
  /* expected size: 160 * 120 * 3 + 120 * 80 * 3 + 64 * 48 * 3 */
  EXPECT_EQ (g_test_data.received_size, 95616U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensors config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.num_tensors, 3U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[1].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[1], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[2], 80U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[2].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[1], 64U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[2], 48U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[3], 1U);

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
    EXPECT_TRUE (gst_tensors_config_is_equal (&config, &g_test_data.tensors_config));

    gst_caps_unref (caps);
    gst_tensors_config_free (&config);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for multiple custom filters.
 */
TEST (tensorStreamTest, customFilterMulti)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_CUSTOM_MULTI };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 3U);
  /* expected size: 160 * 120 * 3 + 280 * 40 * 3 + 640 * 480 * 3 */
  EXPECT_EQ (g_test_data.received_size, 1012800U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensors config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.num_tensors, 3U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[1].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[1], 280U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[2], 40U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[2].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[1], 640U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[2], 480U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for tensor filter properties.
 */
TEST (tensorStreamTest, filterProperties1)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_CUSTOM_TENSOR };

  GstElement *filter;
  gboolean silent, res_silent;
  gchar *str = NULL;
  gchar *model = NULL;

  ASSERT_TRUE (_setup_pipeline (option));

  filter = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "test_filter");

  /* default silent is TRUE */
  g_object_get (filter, "silent", &silent, NULL);
  EXPECT_TRUE (silent);

  /* framework */
  g_object_get (filter, "framework", &str, NULL);
  EXPECT_STREQ (str, "custom");
  g_free (str);

  /* model */
  g_object_get (filter, "model", &str, NULL);
  model = g_strdup_printf ("libnnstreamer_customfilter_passthrough_variable%s",
      NNSTREAMER_SO_FILE_EXTENSION);
  EXPECT_TRUE (g_str_has_suffix (str, model));
  g_free (str);
  g_free (model);

  /* input */
  g_object_get (filter, "input", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  /* inputtype */
  g_object_get (filter, "inputtype", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  /* output */
  g_object_get (filter, "output", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  /* outputtype */
  g_object_get (filter, "outputtype", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  /* inputlayout */
  g_object_get (filter, "inputlayout", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  /* outputlayout */
  g_object_get (filter, "outputlayout", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  /* silent */
  g_object_set (filter, "silent", !silent, NULL);
  g_object_get (filter, "silent", &res_silent, NULL);
  EXPECT_EQ (res_silent, !silent);

  /* input */
  g_object_get (filter, "input", &str, NULL);
  EXPECT_STREQ (str, "3:160:120:1");
  g_free (str);

  /* inputtype */
  g_object_get (filter, "inputtype", &str, NULL);
  EXPECT_STREQ (str, "uint8");
  g_free (str);

  /* inputname */
  g_object_get (filter, "inputname", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  /* output */
  g_object_get (filter, "output", &str, NULL);
  EXPECT_STREQ (str, "3:160:120:1");
  g_free (str);

  /* outputtype */
  g_object_get (filter, "outputtype", &str, NULL);
  EXPECT_STREQ (str, "uint8");
  g_free (str);

  /* outputname */
  g_object_get (filter, "outputname", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  /* inputlayout */
  g_object_get (filter, "inputlayout", &str, NULL);
  EXPECT_STREQ (str, "ANY");
  g_free (str);

  /* outputlayout */
  g_object_get (filter, "outputlayout", &str, NULL);
  EXPECT_STREQ (str, "ANY");
  g_free (str);

  /* sub-plugins */
  g_object_get (filter, "sub-plugins", &str, NULL);
  /* custom / custom-easy are always available */
  ASSERT_TRUE (str != NULL);
  EXPECT_TRUE (strstr (str, "custom") != NULL);
  EXPECT_TRUE (strstr (str, "custom-easy") != NULL);
#ifdef ENABLE_TENSORFLOW_LITE
  EXPECT_TRUE ((strstr (str, "tensorflow-lite") != NULL)
               || (strstr (str, "tensorflow1-lite") != NULL)
               || (strstr (str, "tensorflow2-lite") != NULL));
#endif
  g_free (str);

  /* custom-properties */
  g_object_get (filter, "custom", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  /* accelerator */
  g_object_get (filter, "accelerator", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  gst_object_unref (filter);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for tensor filter properties.
 */
TEST (tensorStreamTest, filterProperties2)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_CUSTOM_TENSORS_1 };

  GstElement *filter;
  gboolean silent, res_silent;
  gchar *str = NULL;
  gchar *model = NULL;

  ASSERT_TRUE (_setup_pipeline (option));

  filter = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "test_filter");

  /* default silent is TRUE */
  g_object_get (filter, "silent", &silent, NULL);
  EXPECT_TRUE (silent);

  /* framework */
  g_object_get (filter, "framework", &str, NULL);
  EXPECT_STREQ (str, "custom");
  g_free (str);

  /* model */
  g_object_get (filter, "model", &str, NULL);
  model = g_strdup_printf ("libnnstreamer_customfilter_passthrough_variable%s",
      NNSTREAMER_SO_FILE_EXTENSION);
  EXPECT_TRUE (g_str_has_suffix (str, model));
  g_free (str);
  g_free (model);

  /* input */
  g_object_get (filter, "input", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  /* inputtype */
  g_object_get (filter, "inputtype", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  /* output */
  g_object_get (filter, "output", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  /* outputtype */
  g_object_get (filter, "outputtype", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  /* silent */
  g_object_set (filter, "silent", !silent, NULL);
  g_object_get (filter, "silent", &res_silent, NULL);
  EXPECT_EQ (res_silent, !silent);

  /* input */
  g_object_get (filter, "input", &str, NULL);
  EXPECT_STREQ (str, "3:160:120:1,3:120:80:1,3:64:48:1");
  g_free (str);

  /* inputtype */
  g_object_get (filter, "inputtype", &str, NULL);
  EXPECT_STREQ (str, "uint8,uint8,uint8");
  g_free (str);

  /* inputname */
  g_object_get (filter, "inputname", &str, NULL);
  EXPECT_STREQ (str, ",,");
  g_free (str);

  /* output */
  g_object_get (filter, "output", &str, NULL);
  EXPECT_STREQ (str, "3:160:120:1,3:120:80:1,3:64:48:1");
  g_free (str);

  /* outputtype */
  g_object_get (filter, "outputtype", &str, NULL);
  EXPECT_STREQ (str, "uint8,uint8,uint8");
  g_free (str);

  /* outputname */
  g_object_get (filter, "outputname", &str, NULL);
  EXPECT_STREQ (str, ",,");
  g_free (str);

  /* inputlayout */
  g_object_get (filter, "inputlayout", &str, NULL);
  EXPECT_STREQ (str, "ANY,ANY,ANY");
  g_free (str);

  /* outputlayout */
  g_object_get (filter, "outputlayout", &str, NULL);
  EXPECT_STREQ (str, "ANY,ANY,ANY");
  g_free (str);

  gst_object_unref (filter);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for tensor filter properties.
 */
TEST (tensorStreamTest, filterProperties3)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_CUSTOM_TENSORS_2 };

  GstElement *filter;
  gboolean silent, res_silent;
  gchar *str = NULL;
  gchar *model = NULL;

  ASSERT_TRUE (_setup_pipeline (option));

  filter = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "test_filter");

  /* default silent is TRUE */
  g_object_get (filter, "silent", &silent, NULL);
  EXPECT_TRUE (silent);

  /* framework */
  g_object_get (filter, "framework", &str, NULL);
  EXPECT_STREQ (str, "custom");
  g_free (str);

  /* model */
  g_object_get (filter, "model", &str, NULL);
  model = g_strdup_printf ("libnnstreamer_customfilter_passthrough_variable%s",
      NNSTREAMER_SO_FILE_EXTENSION);
  EXPECT_TRUE (g_str_has_suffix (str, model));
  g_free (str);
  g_free (model);

  /* silent */
  g_object_set (filter, "silent", !silent, NULL);
  g_object_get (filter, "silent", &res_silent, NULL);
  EXPECT_EQ (res_silent, !silent);

  /* input */
  g_object_get (filter, "input", &str, NULL);
  EXPECT_STREQ (str, "3:160:120:1,3:120:80:1,3:64:48:1");
  g_free (str);

  /* inputtype */
  g_object_get (filter, "inputtype", &str, NULL);
  EXPECT_STREQ (str, "uint8,uint8,uint8");
  g_free (str);

  /* inputname */
  g_object_get (filter, "inputname", &str, NULL);
  EXPECT_STREQ (str, ",,");
  g_free (str);

  /* output */
  g_object_get (filter, "output", &str, NULL);
  EXPECT_STREQ (str, "3:160:120:1,3:120:80:1,3:64:48:1");
  g_free (str);

  /* outputtype */
  g_object_get (filter, "outputtype", &str, NULL);
  EXPECT_STREQ (str, "uint8,uint8,uint8");
  g_free (str);

  /* outputname */
  g_object_get (filter, "outputname", &str, NULL);
  EXPECT_STREQ (str, ",,");
  g_free (str);

  /* inputlayout */
  g_object_get (filter, "inputlayout", &str, NULL);
  EXPECT_STREQ (str, "NCHW,NHWC,NONE");
  g_free (str);

  /* outputlayout */
  g_object_get (filter, "outputlayout", &str, NULL);
  EXPECT_STREQ (str, "ANY,NONE,NCHW");
  g_free (str);

  /* verify properties are maintained after state set to playing */
  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  /* silent */
  g_object_set (filter, "silent", !silent, NULL);
  g_object_get (filter, "silent", &res_silent, NULL);
  EXPECT_EQ (res_silent, !silent);

  /* input */
  g_object_get (filter, "input", &str, NULL);
  EXPECT_STREQ (str, "3:160:120:1,3:120:80:1,3:64:48:1");
  g_free (str);

  /* inputtype */
  g_object_get (filter, "inputtype", &str, NULL);
  EXPECT_STREQ (str, "uint8,uint8,uint8");
  g_free (str);

  /* inputname */
  g_object_get (filter, "inputname", &str, NULL);
  EXPECT_STREQ (str, ",,");
  g_free (str);

  /* output */
  g_object_get (filter, "output", &str, NULL);
  EXPECT_STREQ (str, "3:160:120:1,3:120:80:1,3:64:48:1");
  g_free (str);

  /* outputtype */
  g_object_get (filter, "outputtype", &str, NULL);
  EXPECT_STREQ (str, "uint8,uint8,uint8");
  g_free (str);

  /* outputname */
  g_object_get (filter, "outputname", &str, NULL);
  EXPECT_STREQ (str, ",,");
  g_free (str);

  /* inputlayout */
  g_object_get (filter, "inputlayout", &str, NULL);
  EXPECT_STREQ (str, "NCHW,NHWC,NONE");
  g_free (str);

  /* outputlayout */
  g_object_get (filter, "outputlayout", &str, NULL);
  EXPECT_STREQ (str, "ANY,NONE,NCHW");
  g_free (str);

  gst_object_unref (filter);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for tensor filter properties.
 */
TEST (tensorStreamTest, filterProperties4_n)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_CUSTOM_TENSOR };
  GstElement *filter;
  gchar *str = NULL;

  ASSERT_TRUE (_setup_pipeline (option));

  filter = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "test_filter");

  /* try to set/get unknown property */
  g_object_set (filter, "invalid_prop", "invalid_value", NULL);
  g_object_get (filter, "invalid_prop", &str, NULL);
  /* getting unknown property, str should be null */
  EXPECT_TRUE (str == NULL);

  gst_object_unref (filter);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test to drop incoming buffer in tensor_filter using custom filter.
 */
TEST (tensorStreamTest, customFilterDropBuffer)
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
  EXPECT_EQ (g_test_data.received, 2U);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 200U * 2);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_INT16);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 200U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 16000);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief custom fw name to run without model file.
 */
static const char test_fw_custom_name[] = "custom-passthrough";

/**
 * @brief The mandatory callback for GstTensorFilterFramework.
 */
static int
test_custom_v0_invoke (const GstTensorFilterProperties *prop,
    void **private_data, const GstTensorMemory *input, GstTensorMemory *output)
{
  guint i, num;

  num = prop->input_meta.num_tensors;

  for (i = 0; i < num; i++) {
    g_assert (input[i].size == output[i].size);
    memcpy (output[i].data, input[i].data, input[i].size);
  }

  return 0;
}

/**
 * @brief The optional callback for GstTensorFilterFramework.
 */
static int
test_custom_v0_setdim (const GstTensorFilterProperties *prop,
    void **private_data, const GstTensorsInfo *in_info, GstTensorsInfo *out_info)
{
  gst_tensors_info_copy (out_info, in_info);
  return 0;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework (v1).
 */
static int
test_custom_v1_invoke (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  return test_custom_v0_invoke (prop, &private_data, input, output);
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework (v1).
 */
static int
test_custom_v1_getFWInfo (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data,
    GstTensorFilterFrameworkInfo *fw_info)
{
  fw_info->name = test_fw_custom_name;
  fw_info->allow_in_place = 0;
  fw_info->allocate_in_invoke = 0;
  fw_info->run_without_model = 1;
  fw_info->verify_model_path = 0;
  fw_info->hw_list = NULL;
  fw_info->num_hw = 0;
  return 0;
}

/**
 * @brief Invalid callback for GstTensorFilterFramework (v1).
 */
static int
test_custom_v1_getFWInfo_f1 (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data,
    GstTensorFilterFrameworkInfo *fw_info)
{
  memset (fw_info, 0, sizeof (GstTensorFilterFrameworkInfo));
  fw_info->name = NULL;
  return 0;
}

/**
 * @brief Invalid callback for GstTensorFilterFramework (v1).
 */
static int
test_custom_v1_getFWInfo_f2 (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data,
    GstTensorFilterFrameworkInfo *fw_info)
{
  memset (fw_info, 0, sizeof (GstTensorFilterFrameworkInfo));
  return -1;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework (v1).
 */
static int
test_custom_v1_getModelInfo (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data,
    model_info_ops ops, GstTensorsInfo *in_info, GstTensorsInfo *out_info)
{
  if (ops == SET_INPUT_INFO) {
    return test_custom_v0_setdim (prop, &private_data, in_info, out_info);
  }

  return -ENOENT;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework (v1).
 */
static int
test_custom_v1_eventHandler (const GstTensorFilterFramework *self,
    const GstTensorFilterProperties *prop, void *private_data, event_ops ops,
    GstTensorFilterFrameworkEventData *data)
{
  return -ENOENT;
}

/**
 * @brief Test for passthrough custom filter without model.
 */
static void
test_custom_run_pipeline (void)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_CUSTOM_PASSTHROUGH };

  /* construct pipeline for test */
  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /* check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /* check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 3U * 160 * 120);

  /* check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /* check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for plugin registration with invalid param.
 */
TEST (tensorStreamTest, subpluginInvalidVer_n)
{
  GstTensorFilterFramework *fw = g_new0 (GstTensorFilterFramework, 1);

  ASSERT_TRUE (fw != NULL);
  fw->version = 0; /* invalid ver */
  fw->invoke = test_custom_v1_invoke;
  fw->getFrameworkInfo = test_custom_v1_getFWInfo;
  fw->getModelInfo = test_custom_v1_getModelInfo;
  fw->eventHandler = test_custom_v1_eventHandler;

  EXPECT_FALSE (nnstreamer_filter_probe (fw));

  g_free (fw);
}

/**
 * @brief Test for passthrough custom filter without model.
 */
TEST (tensorStreamTest, subpluginV0Run)
{
  GstTensorFilterFramework *fw = g_new0 (GstTensorFilterFramework, 1);

  ASSERT_TRUE (fw != NULL);
  fw->version = GST_TENSOR_FILTER_FRAMEWORK_V0;
  fw->name = (char *)test_fw_custom_name;
  fw->run_without_model = TRUE;
  fw->invoke_NN = test_custom_v0_invoke;
  fw->setInputDimension = test_custom_v0_setdim;

  /* register custom filter */
  EXPECT_TRUE (nnstreamer_filter_probe (fw));

  test_custom_run_pipeline ();

  /* unregister custom filter */
  nnstreamer_filter_exit (test_fw_custom_name);
  g_free (fw);
}

/**
 * @brief Test for preserved sub-plugin name.
 */
TEST (tensorStreamTest, subpluginV0PreservedName_n)
{
  GstTensorFilterFramework *fw = g_new0 (GstTensorFilterFramework, 1);

  ASSERT_TRUE (fw != NULL);
  fw->version = GST_TENSOR_FILTER_FRAMEWORK_V0;
  fw->run_without_model = TRUE;
  fw->invoke_NN = test_custom_v0_invoke;
  fw->setInputDimension = test_custom_v0_setdim;

  fw->name = g_strdup ("any"); /* preserved name 'any' */
  EXPECT_FALSE (nnstreamer_filter_probe (fw));
  g_free (fw->name);

  fw->name = g_strdup ("auto"); /* preserved name 'auto' */
  EXPECT_FALSE (nnstreamer_filter_probe (fw));
  g_free (fw->name);

  g_free (fw);
}

/**
 * @brief Test for plugin registration with invalid param.
 */
TEST (tensorStreamTest, subpluginV0NullName_n)
{
  GstTensorFilterFramework *fw = g_new0 (GstTensorFilterFramework, 1);

  ASSERT_TRUE (fw != NULL);
  fw->version = GST_TENSOR_FILTER_FRAMEWORK_V0;
  fw->name = NULL; /* failed to register with null */
  fw->run_without_model = TRUE;
  fw->invoke_NN = test_custom_v0_invoke;
  fw->setInputDimension = test_custom_v0_setdim;

  EXPECT_FALSE (nnstreamer_filter_probe (fw));

  g_free (fw);
}

/**
 * @brief Test for plugin registration with invalid param.
 */
TEST (tensorStreamTest, subpluginV0NullCb_n)
{
  GstTensorFilterFramework *fw = g_new0 (GstTensorFilterFramework, 1);

  ASSERT_TRUE (fw != NULL);
  fw->version = GST_TENSOR_FILTER_FRAMEWORK_V0;
  fw->name = g_strdup ("custom-invalid");
  fw->run_without_model = TRUE;

  fw->invoke_NN = NULL;
  fw->setInputDimension = test_custom_v0_setdim;
  EXPECT_FALSE (nnstreamer_filter_probe (fw));

  fw->invoke_NN = test_custom_v0_invoke;
  fw->setInputDimension = NULL;
  EXPECT_FALSE (nnstreamer_filter_probe (fw));

  g_free (fw->name);
  g_free (fw);
}

/**
 * @brief Test for passthrough custom filter without model (v1).
 */
TEST (tensorStreamTest, subpluginV1Run)
{
  GstTensorFilterFramework *fw = g_new0 (GstTensorFilterFramework, 1);

  ASSERT_TRUE (fw != NULL);
  fw->version = GST_TENSOR_FILTER_FRAMEWORK_V1;
  fw->invoke = test_custom_v1_invoke;
  fw->getFrameworkInfo = test_custom_v1_getFWInfo;
  fw->getModelInfo = test_custom_v1_getModelInfo;
  fw->eventHandler = test_custom_v1_eventHandler;

  /* register custom filter */
  EXPECT_TRUE (nnstreamer_filter_probe (fw));

  test_custom_run_pipeline ();

  /* unregister custom filter */
  nnstreamer_filter_exit (test_fw_custom_name);
  g_free (fw);
}

/**
 * @brief Test for plugin registration with invalid param (v1).
 */
TEST (tensorStreamTest, subpluginV1NullName_n)
{
  GstTensorFilterFramework *fw = g_new0 (GstTensorFilterFramework, 1);

  ASSERT_TRUE (fw != NULL);
  fw->version = GST_TENSOR_FILTER_FRAMEWORK_V1;
  fw->invoke = test_custom_v1_invoke;
  fw->getFrameworkInfo = test_custom_v1_getFWInfo_f1;
  fw->getModelInfo = test_custom_v1_getModelInfo;
  fw->eventHandler = test_custom_v1_eventHandler;

  EXPECT_FALSE (nnstreamer_filter_probe (fw));

  g_free (fw);
}

/**
 * @brief Test for plugin registration with invalid param (v1).
 */
TEST (tensorStreamTest, subpluginV1InvalidCb_n)
{
  GstTensorFilterFramework *fw = g_new0 (GstTensorFilterFramework, 1);

  ASSERT_TRUE (fw != NULL);
  fw->version = GST_TENSOR_FILTER_FRAMEWORK_V1;
  fw->invoke = test_custom_v1_invoke;
  fw->getFrameworkInfo = test_custom_v1_getFWInfo_f2;
  fw->getModelInfo = test_custom_v1_getModelInfo;
  fw->eventHandler = test_custom_v1_eventHandler;

  EXPECT_FALSE (nnstreamer_filter_probe (fw));

  g_free (fw);
}

/**
 * @brief Test for plugin registration with invalid param (v1).
 */
TEST (tensorStreamTest, subpluginV1NullCb_n)
{
  GstTensorFilterFramework *fw = g_new0 (GstTensorFilterFramework, 1);

  ASSERT_TRUE (fw != NULL);
  fw->version = GST_TENSOR_FILTER_FRAMEWORK_V1;
  fw->invoke = test_custom_v1_invoke;
  fw->getFrameworkInfo = test_custom_v1_getFWInfo;

  fw->getModelInfo = NULL;
  fw->eventHandler = test_custom_v1_eventHandler;
  EXPECT_FALSE (nnstreamer_filter_probe (fw));

  fw->getModelInfo = test_custom_v1_getModelInfo;
  fw->eventHandler = NULL;
  EXPECT_FALSE (nnstreamer_filter_probe (fw));

  g_free (fw);
}

/**
 * @brief Test for tensors (mixed, video and audio).
 */
TEST (tensorStreamTest, tensorsMix)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSORS_MIX_1 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 500U * 2);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_INT16);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 500U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 30); /** 30 fps from video stream */
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  /** check caps and config for tensor */
  {
    GstCaps *caps;
    GstStructure *structure;
    GstTensorsConfig config;

    caps = gst_tensors_caps_from_config (&g_test_data.tensors_config);
    structure = gst_caps_get_structure (caps, 0);

    EXPECT_TRUE (gst_tensors_config_from_structure (&config, structure));
    EXPECT_TRUE (gst_tensors_config_is_equal (&config, &g_test_data.tensors_config));

    gst_caps_unref (caps);
    gst_tensors_config_free (&config);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for tensor demux properties.
 */
TEST (tensorStreamTest, demuxProperties1)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSORS_MIX_2 };

  GstElement *demux;
  gboolean silent, res_silent;
  gchar *pick = NULL;

  ASSERT_TRUE (_setup_pipeline (option));

  demux = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "demux");

  /* default silent is TRUE */
  g_object_get (demux, "silent", &silent, NULL);
  EXPECT_TRUE (silent);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  /* silent */
  g_object_set (demux, "silent", !silent, NULL);
  g_object_get (demux, "silent", &res_silent, NULL);
  EXPECT_EQ (res_silent, !silent);

  /* tensorpick */
  g_object_get (demux, "tensorpick", &pick, NULL);
  EXPECT_STREQ (pick, "0,2");
  g_free (pick);

  gst_object_unref (demux);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /* check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /* check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 3U * 64 * 48);

  /* check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /* check tensor config */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 64U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 48U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for tensor demux properties.
 */
TEST (tensorStreamTest, demuxProperties2)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSORS_MIX_3 };

  GstElement *demux;
  gboolean silent, res_silent;
  gchar *pick = NULL;

  ASSERT_TRUE (_setup_pipeline (option));

  demux = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "demux");

  /* default silent is TRUE */
  g_object_get (demux, "silent", &silent, NULL);
  EXPECT_TRUE (silent);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  /* silent */
  g_object_set (demux, "silent", !silent, NULL);
  g_object_get (demux, "silent", &res_silent, NULL);
  EXPECT_EQ (res_silent, !silent);

  /* tensorpick */
  g_object_get (demux, "tensorpick", &pick, NULL);
  EXPECT_STREQ (pick, "1,2");
  g_free (pick);

  gst_object_unref (demux);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /* check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /* check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 500U * 2);

  /* check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /* check tensor config */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_INT16);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 500U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for tensor demux properties.
 */
TEST (tensorStreamTest, demuxProperties3_n)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSORS_MIX_3 };
  GstElement *demux;
  gchar *str = NULL;

  ASSERT_TRUE (_setup_pipeline (option));
  demux = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "demux");

  /* try to set/get unknown property */
  g_object_set (demux, "invalid_prop", "invalid_value", NULL);
  g_object_get (demux, "invalid_prop", &str, NULL);
  /* getting unknown property, str should be null */
  EXPECT_TRUE (str == NULL);

  gst_object_unref (demux);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Internal function to test typecast in tensor_transform.
 */
static void
_test_transform_typecast (TestType test, tensor_type type, guint buffers)
{
  TestOption option = { buffers, test, type };
  gsize t_size = gst_tensor_get_element_size (type);
  guint timeout_id;

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  g_timeout_add (100, _test_src_push_timer_cb, GINT_TO_POINTER (TRUE));

  timeout_id = g_timeout_add (5000, _test_src_eos_timer_cb, g_test_data.loop);
  g_main_loop_run (g_test_data.loop);
  g_source_remove (timeout_id);

  EXPECT_TRUE (_wait_pipeline_process_buffers (buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 10U * t_size);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, type);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 10U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for typecast to int32 using tensor_transform.
 */
TEST (tensorStreamTest, typecastInt32)
{
  _test_transform_typecast (TEST_TYPE_TYPECAST, _NNS_INT32, 2);
}

/**
 * @brief Test for typecast to uint32 using tensor_transform.
 */
TEST (tensorStreamTest, typecastUint32)
{
  _test_transform_typecast (TEST_TYPE_TYPECAST, _NNS_UINT32, 2);
}

/**
 * @brief Test for typecast to int16 using tensor_transform.
 */
TEST (tensorStreamTest, typecastInt16)
{
  _test_transform_typecast (TEST_TYPE_TYPECAST, _NNS_INT16, 2);
}

/**
 * @brief Test for typecast to uint16 using tensor_transform.
 */
TEST (tensorStreamTest, typecastUint16)
{
  _test_transform_typecast (TEST_TYPE_TYPECAST, _NNS_UINT16, 2);
}

/**
 * @brief Test for typecast to float64 using tensor_transform.
 */
TEST (tensorStreamTest, typecastFloat64)
{
  _test_transform_typecast (TEST_TYPE_TYPECAST, _NNS_FLOAT64, 2);
}

/**
 * @brief Test for typecast to float32 using tensor_transform.
 */
TEST (tensorStreamTest, typecastFloat32)
{
  _test_transform_typecast (TEST_TYPE_TYPECAST, _NNS_FLOAT32, 2);
}

/**
 * @brief Test for typecast to int64 using tensor_transform.
 */
TEST (tensorStreamTest, typecastInt64)
{
  _test_transform_typecast (TEST_TYPE_TYPECAST, _NNS_INT64, 2);
}

/**
 * @brief Test for typecast to uint64 using tensor_transform.
 */
TEST (tensorStreamTest, typecastUint64)
{
  _test_transform_typecast (TEST_TYPE_TYPECAST, _NNS_UINT64, 2);
}

/**
 * @brief Test for caps negotiation in tensor_transform.
 */
TEST (tensorStreamTest, transformCapsNegoTypecastFloat32)
{
  _test_transform_typecast (TEST_TYPE_TRANSFORM_CAPS_NEGO_1, _NNS_FLOAT32, 2);
}

/**
 * @brief Test for caps negotiation in tensor_transform.
 */
TEST (tensorStreamTest, transformCapsNegoArithmeticFloat32)
{
  _test_transform_typecast (TEST_TYPE_TRANSFORM_CAPS_NEGO_2, _NNS_FLOAT32, 2);
}

/**
 * @brief Test for caps negotiation in tensor_transform.
 */
TEST (tensorStreamTest, transformCapsNegoTypecastUint32)
{
  _test_transform_typecast (TEST_TYPE_TRANSFORM_CAPS_NEGO_1, _NNS_UINT32, 2);
}

/**
 * @brief Test for caps negotiation in tensor_transform.
 */
TEST (tensorStreamTest, transformCapsNegoArithmeticUint32)
{
  _test_transform_typecast (TEST_TYPE_TRANSFORM_CAPS_NEGO_2, _NNS_UINT32, 2);
}

/**
 * @brief Test for tensors stream of the tensor transform
 */
TEST (tensorStreamTest, transformTensors)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TRANSFORM_TENSORS };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 3U);
  EXPECT_EQ (g_test_data.received_size, 1286400U);
  /** (160 * 120 * 3 + 280 * 40 * 3 + 320 * 240 * 3) * 4 */

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensors config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.num_tensors, 3U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_FLOAT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[1].type, _NNS_FLOAT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[1], 280U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[2], 40U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[2].type, _NNS_FLOAT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[1], 320U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[2], 240U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for apply option of the tensor transform
 */
TEST (tensorStreamTest, transformApplyOpt)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TRANSFORM_APPLY };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 3U);
  EXPECT_EQ (g_test_data.received_size, 1185600U);
  /** (160 * 120 * 3 + 320 * 240 * 3) * 4 +  (280 * 40 * 3) */

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensors config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.num_tensors, 3U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_FLOAT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[1].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[1], 280U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[2], 40U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[2].type, _NNS_FLOAT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[1], 320U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[2], 240U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.rate_n, 30);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video stream with tensor_split.
 */
TEST (tensorStreamTest, videoSplit)
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

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 160U * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for tensor_split properties.
 */
TEST (tensorStreamTest, splitProperties1_n)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB_SPLIT };
  GstElement *split;
  gchar *str = NULL;

  ASSERT_TRUE (_setup_pipeline (option));

  /** Check properties of tensor_split */
  split = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "split");
  ASSERT_TRUE (split != NULL);

  /* try to set/get unknown property */
  g_object_set (split, "invalid_prop", "invalid_value", NULL);
  g_object_get (split, "invalid_prop", &str, NULL);
  /* getting unknown property, str should be null */
  EXPECT_TRUE (str == NULL);

  gst_object_unref (split);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video stream with tensor_aggregator.
 */
TEST (tensorStreamTest, videoAggregate1)
{
  const guint num_buffers = 35;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB_AGGR_1 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers ((num_buffers - 10) / 5 + 1));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, (num_buffers - 10) / 5 + 1);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 3U * 160 * 120 * 10);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 10U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video stream with tensor_aggregator.
 */
TEST (tensorStreamTest, videoAggregate2)
{
  const guint num_buffers = 35;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB_AGGR_2 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers ((num_buffers - 10) / 5 + 1));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, (num_buffers - 10) / 5 + 1);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 3U * 1600 * 120);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 1600U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for video stream with tensor_aggregator.
 */
TEST (tensorStreamTest, videoAggregate3)
{
  const guint num_buffers = 40;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO_RGB_AGGR_3 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers ((num_buffers / 10)));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, (num_buffers / 10));
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 3U * 64 * 48 * 8);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 64U * 8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 48U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, (int)fps);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for audio stream with tensor_aggregator.
 */
TEST (tensorStreamTest, audioAggregateS16)
{
  const guint num_buffers = 21;
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_S16_AGGR };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers / 4));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers / 4);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 500U * 2 * 4);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_INT16);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 2000U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 16000);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for audio stream with tensor_aggregator.
 */
TEST (tensorStreamTest, audioAggregateU16)
{
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_TYPE_AUDIO_U16_AGGR };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers * 5));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers * 5);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 500U * 2 / 5);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for audio */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT16);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 100U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 16000);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test multi-stream sync & frame-dropping of Issue #739, 1st subissue
 */
TEST (tensorStreamTest, issue739MuxParallel1)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MUX_PARALLEL_1 };

  option.need_sync = TRUE;
  option.tmpfile = getTempFilename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers * 10));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers * 10);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U); /* uint32_t, 1:1:1:1 */

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

  /** @todo Check contents in the sink */
  if (option.tmpfile) {
    gchar *data = NULL;
    gsize read, i;

    if (g_file_get_contents (option.tmpfile, &data, &read, NULL)) {
      read /= 4;
      EXPECT_EQ (read, num_buffers * 10);
      for (i = 0; i < num_buffers * 2U; i++)
        EXPECT_EQ (((uint32_t *)data)[i], i);

      g_free (data);
    }
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test multi-stream sync & frame-dropping of Issue #739, 1st subissue
 */
TEST (tensorStreamTest, issue739MuxParallel2)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MUX_PARALLEL_2 };

  option.need_sync = TRUE;
  option.tmpfile = getTempFilename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers * 10));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers * 10);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U); /* uint32_t, 1:1:1:1 */

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

  /** @todo Check contents in the sink */
  if (option.tmpfile) {
    gchar *data = NULL;
    gsize read, i;

    if (g_file_get_contents (option.tmpfile, &data, &read, NULL)) {
      read /= 4;
      EXPECT_EQ (read, num_buffers * 10);
      for (i = 0; i < num_buffers * 2U; i++)
        EXPECT_EQ (((uint32_t *)data)[i], i);

      g_free (data);
    }
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test multi-stream sync & frame-dropping of Issue #739, 1st subissue
 */
TEST (tensorStreamTest, issue739MuxParallel3)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MUX_PARALLEL_3 };

  option.need_sync = TRUE;
  option.tmpfile = getTempFilename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers * 25 - 1));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_GE (g_test_data.received, num_buffers * 25 - 1);
  EXPECT_LE (g_test_data.received, num_buffers * 25);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U); /* uint32_t, 1:1:1:1 */

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

  /** @todo Check contents in the sink */
  if (option.tmpfile) {
    gchar *data = NULL;
    gsize read, i;
    uint32_t lastval;

    if (g_file_get_contents (option.tmpfile, &data, &read, NULL)) {
      read /= 4;
      EXPECT_TRUE (read >= (num_buffers * 25 - 1));
      EXPECT_TRUE (read <= (num_buffers * 25));

      lastval = 0;
      for (i = 0; i < read; i++) {
        EXPECT_TRUE (((uint32_t *)data)[i] >= lastval);
        EXPECT_TRUE (((uint32_t *)data)[i] <= lastval + 1);
        lastval = ((uint32_t *)data)[i];
      }
      EXPECT_TRUE (lastval <= (num_buffers * 10));
      EXPECT_TRUE (lastval >= (num_buffers * 10 - 1));

      g_free (data);
    }
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test multi-stream sync & frame-dropping of Issue #739, 1st subissue
 */
TEST (tensorStreamTest, issue739MuxParallel4)
{
  /** @todo Write this after the tensor-mux/merge sync-option "basepad" is updated */
  EXPECT_EQ (1, 1);
}

/**
 * @brief Test multi-stream sync & frame-dropping of Issue #739, 1st subissue
 */
TEST (tensorStreamTest, issue739MergeParallel1)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MERGE_PARALLEL_1 };

  option.need_sync = TRUE;
  option.tmpfile = getTempFilename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers * 10));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers * 10);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U); /* uint32_t, 1:1:1:1 */

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

  /** @todo Check contents in the sink */
  if (option.tmpfile) {
    gchar *data = NULL;
    gsize read, i;

    if (g_file_get_contents (option.tmpfile, &data, &read, NULL)) {
      read /= 4;
      EXPECT_EQ (read, num_buffers * 10);
      for (i = 0; i < num_buffers * 2U; i++)
        EXPECT_EQ (((uint32_t *)data)[i], i);

      g_free (data);
    }
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test multi-stream sync & frame-dropping of Issue #739, 1st subissue
 */
TEST (tensorStreamTest, issue739MergeParallel2)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MERGE_PARALLEL_2 };

  option.need_sync = TRUE;
  option.tmpfile = getTempFilename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers * 10));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers * 10);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U); /* uint32_t, 1:1:1:1 */

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

  /** @todo Check contents in the sink */
  if (option.tmpfile) {
    gchar *data = NULL;
    gsize read, i;

    if (g_file_get_contents (option.tmpfile, &data, &read, NULL)) {
      read /= 4;
      EXPECT_EQ (read, num_buffers * 10);
      for (i = 0; i < num_buffers * 2U; i++)
        EXPECT_EQ (((uint32_t *)data)[i], i);

      g_free (data);
    }
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test multi-stream sync & frame-dropping of Issue #739, 1st subissue
 */
TEST (tensorStreamTest, issue739MergeParallel3)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MERGE_PARALLEL_3 };

  option.need_sync = TRUE;
  option.tmpfile = getTempFilename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers * 25 - 1));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_GE (g_test_data.received, num_buffers * 25 - 1);
  EXPECT_LE (g_test_data.received, num_buffers * 25);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 4U); /* uint32_t, 1:1:1:1 */

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT32);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

  /** @todo Check contents in the sink */
  if (option.tmpfile) {
    gchar *data = NULL;
    gsize read, i;
    uint32_t lastval;

    if (g_file_get_contents (option.tmpfile, &data, &read, NULL)) {
      read /= 4;
      EXPECT_TRUE (read >= (num_buffers * 25 - 1));
      EXPECT_TRUE (read <= (num_buffers * 25));

      lastval = 0;
      for (i = 0; i < read; i++) {
        EXPECT_TRUE (((uint32_t *)data)[i] >= lastval);
        EXPECT_TRUE (((uint32_t *)data)[i] <= lastval + 1);
        lastval = ((uint32_t *)data)[i];
      }
      EXPECT_GE (lastval, (num_buffers - 1) * 25);
      EXPECT_LE (lastval, num_buffers * 25);

      g_free (data);
    }
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for tensor_mux properties.
 */
TEST (tensorStreamTest, muxProperties1)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MUX_PARALLEL_2 };
  GstElement *mux;
  gboolean silent, res_silent;
  gchar *str = NULL;

  option.need_sync = TRUE;
  option.tmpfile = getTempFilename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  mux = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "mux");
  ASSERT_TRUE (mux != NULL);

  /* default silent is TRUE */
  g_object_get (mux, "silent", &silent, NULL);
  EXPECT_TRUE (silent);
  g_object_set (mux, "silent", !silent, NULL);
  g_object_get (mux, "silent", &res_silent, NULL);
  EXPECT_EQ (res_silent, !silent);

  /* sync-mode */
  g_object_get (mux, "sync-mode", &str, NULL);
  EXPECT_STREQ (str, "basepad");
  g_free (str);

  /* sync-option */
  g_object_get (mux, "sync-option", &str, NULL);
  EXPECT_STREQ (str, "0:0");
  g_free (str);

  gst_object_unref (mux);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for tensor_mux properties.
 */
TEST (tensorStreamTest, muxProperties2_n)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MUX_PARALLEL_2 };
  GstElement *mux;
  gchar *str = NULL;

  option.need_sync = TRUE;
  option.tmpfile = getTempFilename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  mux = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "mux");
  ASSERT_TRUE (mux != NULL);

  /* try to set/get unknown property */
  g_object_set (mux, "invalid_prop", "invalid_value", NULL);
  g_object_get (mux, "invalid_prop", &str, NULL);
  /* getting unknown property, str should be null */
  EXPECT_TRUE (str == NULL);

  gst_object_unref (mux);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for tensor_merge properties.
 */
TEST (tensorStreamTest, mergeProperties1)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MERGE_PARALLEL_2 };
  GstElement *merge;
  gboolean silent, res_silent;
  gchar *str = NULL;

  option.need_sync = TRUE;
  option.tmpfile = getTempFilename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  merge = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "mux");
  ASSERT_TRUE (merge != NULL);

  /* default silent is TRUE */
  g_object_get (merge, "silent", &silent, NULL);
  EXPECT_TRUE (silent);
  g_object_set (merge, "silent", !silent, NULL);
  g_object_get (merge, "silent", &res_silent, NULL);
  EXPECT_EQ (res_silent, !silent);

  /* mode */
  g_object_get (merge, "mode", &str, NULL);
  EXPECT_STREQ (str, "linear");
  g_free (str);

  /* option */
  g_object_get (merge, "option", &str, NULL);
  EXPECT_STREQ (str, "3");
  g_free (str);

  /* sync-mode */
  g_object_get (merge, "sync-mode", &str, NULL);
  EXPECT_STREQ (str, "basepad");
  g_free (str);

  /* sync-option */
  g_object_get (merge, "sync-option", &str, NULL);
  EXPECT_STREQ (str, "0:0");
  g_free (str);

  gst_object_unref (merge);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for tensor_merge properties.
 */
TEST (tensorStreamTest, mergeProperties2_n)
{
  const guint num_buffers = 2;
  TestOption option = { num_buffers, TEST_TYPE_ISSUE739_MERGE_PARALLEL_2 };
  GstElement *merge;
  gchar *str = NULL;

  option.need_sync = TRUE;
  option.tmpfile = getTempFilename ();
  EXPECT_TRUE (option.tmpfile != NULL);

  ASSERT_TRUE (_setup_pipeline (option));

  merge = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "mux");
  ASSERT_TRUE (merge != NULL);

  /* try to set/get unknown property */
  g_object_set (merge, "invalid_prop", "invalid_value", NULL);
  g_object_get (merge, "invalid_prop", &str, NULL);
  /* getting unknown property, str should be null */
  EXPECT_TRUE (str == NULL);

  gst_object_unref (merge);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test get/set property of tensor_decoder
 */
TEST (tensorStreamTest, tensorDecoderProperty1)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_DECODER_PROPERTY };
  GstElement *dec;
  gchar *str;
  gboolean silent, res_silent;

  ASSERT_TRUE (_setup_pipeline (option));

  /** Check properties of tensor_decoder */
  dec = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "decoder");
  ASSERT_TRUE (dec != NULL);

  g_object_get (dec, "mode", &str, NULL);
  EXPECT_STREQ (str, "direct_video");
  g_free (str);

  g_object_get (dec, "silent", &silent, NULL);
  EXPECT_EQ (silent, TRUE);
  g_object_set (dec, "silent", !silent, NULL);
  g_object_get (dec, "silent", &res_silent, NULL);
  EXPECT_EQ (res_silent, !silent);

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

  /* sub-plugins */
  g_object_get (dec, "sub-plugins", &str, NULL);
  ASSERT_TRUE (str != NULL);
  EXPECT_TRUE (strstr (str, "direct_video") != NULL);
  EXPECT_TRUE (strstr (str, "bounding_boxes") != NULL);
  g_free (str);

  gst_object_unref (dec);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, 5U);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 64U); /* uint8_t, 4:4:4:1 */

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test get/set property of tensor_decoder
 */
TEST (tensorStreamTest, tensorDecoderProperty2_n)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_DECODER_PROPERTY };
  GstElement *dec;
  gchar *str = NULL;

  ASSERT_TRUE (_setup_pipeline (option));

  /** Check properties of tensor_decoder */
  dec = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "decoder");
  ASSERT_TRUE (dec != NULL);

  /* try to set/get unknown property */
  g_object_set (dec, "invalid_prop", "invalid_value", NULL);
  g_object_get (dec, "invalid_prop", &str, NULL);
  /* getting unknown property, str should be null */
  EXPECT_TRUE (str == NULL);

  gst_object_unref (dec);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test tensor out
 */
TEST (tensorStreamTest, tensorCap0)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSOR_CAP_1 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  /** 160 * 120 * 3 + 120 * 80 * 3 + 64 * 48 * 3 */
  EXPECT_EQ (g_test_data.received_size, 57600U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensors config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));

  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

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
    EXPECT_TRUE (gst_tensors_config_is_equal (&config, &g_test_data.tensors_config));

    gst_caps_unref (caps);
    gst_tensors_config_free (&config);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test tensor out
 */
TEST (tensorStreamTest, tensorCap1)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSOR_CAP_2 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  /** 160 * 120 * 3 + 120 * 80 * 3 + 64 * 48 * 3 */
  EXPECT_EQ (g_test_data.received_size, 57600U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensors config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));

  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

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
    EXPECT_TRUE (gst_tensors_config_is_equal (&config, &g_test_data.tensors_config));

    gst_caps_unref (caps);
    gst_tensors_config_free (&config);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test tensors out when the number of the tensors is 1.
 */
TEST (tensorStreamTest, tensorsCap0)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSORS_CAP_1 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  /** 160 * 120 * 3 + 120 * 80 * 3 + 64 * 48 * 3 */
  EXPECT_EQ (g_test_data.received_size, 57600U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensors config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.num_tensors, 1U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

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
    EXPECT_TRUE (gst_tensors_config_is_equal (&config, &g_test_data.tensors_config));

    gst_caps_unref (caps);
    gst_tensors_config_free (&config);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

/**
 * @brief Test for other/tensors, passthrough custom filter.
 */
TEST (tensorStreamTest, tensorsCap1)
{
  const guint num_buffers = 5;
  TestOption option = { num_buffers, TEST_TYPE_TENSORS_CAP_2 };

  ASSERT_TRUE (_setup_pipeline (option));

  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);

  EXPECT_TRUE (_wait_pipeline_process_buffers (num_buffers));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 3U);
  /** 160 * 120 * 3 + 120 * 80 * 3 + 64 * 48 * 3 */
  EXPECT_EQ (g_test_data.received_size, 95616U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensors config for video */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.num_tensors, 3U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 160U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[1].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[1], 120U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[2], 80U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[1].dimension[3], 1U);

  EXPECT_EQ (g_test_data.tensors_config.info.info[2].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[0], 3U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[1], 64U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[2], 48U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[2].dimension[3], 1U);

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
    EXPECT_TRUE (gst_tensors_config_is_equal (&config, &g_test_data.tensors_config));

    gst_caps_unref (caps);
    gst_tensors_config_free (&config);
  }

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);
}

#include <tensor_filter_custom_easy.h>

/**
 * @brief In-Code Test Function for custom-easy filter
 */
static int
cef_func_safe_memcpy (void *data, const GstTensorFilterProperties *prop,
    const GstTensorMemory *in, GstTensorMemory *out)
{
  unsigned int t;
  for (t = 0; t < prop->output_meta.num_tensors; t++) {
    if (prop->input_meta.num_tensors <= t)
      memset (out[t].data, 0, out[t].size);
    else
      memcpy (out[t].data, in[t].data, MIN (in[t].size, out[t].size));
  }
  return 0;
}

/**
 * @brief Test custom-easy filter with an in-code function.
 */
TEST (tensorFilterCustomEasy, inCodeFunc01)
{
  int ret;
  const guint num_buffers = 10;
  TestOption option = { num_buffers, TEST_CUSTOM_EASY_ICF_01 };
  guint timeout_id;
  GstTensorsInfo info_in;
  GstTensorsInfo info_out;

  gst_tensors_info_init (&info_in);
  gst_tensors_info_init (&info_out);
  info_in.num_tensors = 1U;
  info_in.info[0].name = NULL;
  info_in.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("1:10:1:1", info_in.info[0].dimension);

  info_out.num_tensors = 1U;
  info_out.info[0].name = NULL;
  info_out.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("1:10:1:1", info_out.info[0].dimension);

  ret = NNS_custom_easy_register (
      "safe_memcpy_10x10", cef_func_safe_memcpy, NULL, &info_in, &info_out);
  ASSERT_EQ (ret, 0);

  ASSERT_TRUE (_setup_pipeline (option));
  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_timeout_add (100, _test_src_push_timer_cb, GINT_TO_POINTER (FALSE));

  timeout_id = g_timeout_add (5000, _test_src_eos_timer_cb, g_test_data.loop);
  g_main_loop_run (g_test_data.loop);
  g_source_remove (timeout_id);

  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.mem_blocks, 1U);
  EXPECT_EQ (g_test_data.received_size, 10U);

  /** check timestamp */
  EXPECT_FALSE (g_test_data.invalid_timestamp);

  /** check tensor config for text */
  EXPECT_TRUE (gst_tensors_config_validate (&g_test_data.tensors_config));
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[0], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[1], 10U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (g_test_data.tensors_config.info.info[0].dimension[3], 1U);
  EXPECT_EQ (g_test_data.tensors_config.rate_n, 0);
  EXPECT_EQ (g_test_data.tensors_config.rate_d, 1);

  EXPECT_FALSE (g_test_data.test_failed);
  _free_test_data (option);

  /** cleanup registered custom_easy filter */
  ret = NNS_custom_easy_unregister ("safe_memcpy_10x10");
  ASSERT_EQ (ret, 0);

  /** @todo: Check the data at sink */
}

/**
 * @brief Test unregister custom_easy filter
 */
TEST (tensorFilterCustomEasy, unregister1_p)
{
  int ret;
  GstTensorsInfo info_in;
  GstTensorsInfo info_out;

  gst_tensors_info_init (&info_in);
  gst_tensors_info_init (&info_out);
  info_in.num_tensors = 1U;
  info_in.info[0].name = NULL;
  info_in.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("1:10:1:1", info_in.info[0].dimension);

  info_out.num_tensors = 1U;
  info_out.info[0].name = NULL;
  info_out.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("1:10:1:1", info_out.info[0].dimension);

  ret = NNS_custom_easy_register (
      "safe_memcpy_10x10", cef_func_safe_memcpy, NULL, &info_in, &info_out);
  ASSERT_EQ (ret, 0);

  /** check unregister custom_easy filter */
  ret = NNS_custom_easy_unregister ("safe_memcpy_10x10");
  ASSERT_EQ (ret, 0);
}

/**
 * @brief Test non-registered custom_easy filter
 */
TEST (tensorFilterCustomEasy, unregister1_n)
{
  int ret;

  /** check non-registered custom_easy filter */
  ret = NNS_custom_easy_unregister ("not_registered");
  ASSERT_NE (ret, 0);
}

/**
 * @brief Main function for unit test.
 */
int
main (int argc, char **argv)
{
  int ret = -1;
  gchar *jitter_cmd_arg = NULL;
  gchar *fps_cmd_arg = NULL;
  const GOptionEntry main_entries[]
      = { { "customdir", 'd', G_OPTION_FLAG_NONE, G_OPTION_ARG_STRING, &custom_dir,
              "A directory containing custom sub-plugins to use this test",
              "build/nnstreamer_example" },
          { "jitter", 'j', G_OPTION_FLAG_NONE, G_OPTION_ARG_STRING, &jitter_cmd_arg,
              "Jitter in ms between starting and stopping test pipelines", "0 (default)" },
          { "fps", 'f', G_OPTION_FLAG_NONE, G_OPTION_ARG_STRING, &fps_cmd_arg,
              "Frames per second to run tests made of videotestsrc-based single pipeline",
              "30 (default)" },
          { NULL } };
  GError *error = NULL;
  GOptionContext *optionctx;
  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  optionctx = g_option_context_new (NULL);
  g_option_context_add_main_entries (optionctx, main_entries, NULL);

  if (!g_option_context_parse (optionctx, &argc, &argv, &error)) {
    g_print ("option parsing failed: %s\n", error->message);
    g_clear_error (&error);
  }

  if (jitter_cmd_arg != NULL) {
    jitter = (gulong)g_ascii_strtoull (jitter_cmd_arg, NULL, 10) * MSEC_PER_USEC;
  }

  if (fps_cmd_arg != NULL) {
    fps = (gulong)g_ascii_strtoull (fps_cmd_arg, NULL, 10);
    if (fps == 0) {
      fps = DEFAULT_FPS;
    }
  }


  try {
    ret = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }
  return ret;
}
