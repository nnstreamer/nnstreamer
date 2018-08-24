/**
 * @file	unittest_sink.cpp
 * @date	29 June 2018
 * @brief	Unit test for tensor sink plugin
 * @see		https://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs.
 */

#include <string.h>
#include <gtest/gtest.h>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

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
  TEST_TYPE_VIDEO, /**< pipeline for video */
  TEST_TYPE_AUDIO, /**< pipeline for audio */
  TEST_TYPE_TEXT, /**< pipeline for text */
  TEST_TYPE_TENSORS, /**< pipeline for tensors with tensormux */
  TEST_TYPE_NEGO_FAILED, /**< pipeline to test caps negotiation */
} TestType;

/**
 * @brief Test options.
 */
typedef struct
{
  guint num_buffers; /**< count of buffers */
  TestType test_type; /**< test pipeline */
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
  guint received; /**< received buffer count */
  gboolean start; /**< stream started */
  gboolean end; /**< eos reached */
  gchar *caps_name; /**< negotiated caps name */
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
  g_test_data.received++;
  _print_log ("new data callback [%d]", g_test_data.received);

  if (g_test_data.caps_name == NULL) {
    GstPad *sink_pad;
    GstCaps *caps;
    GstStructure *structure;

    /** get negotiated caps */
    sink_pad = gst_element_get_static_pad (element, "sink");
    caps = gst_pad_get_current_caps (sink_pad);
    structure = gst_caps_get_structure (caps, 0);

    g_test_data.caps_name = (gchar *) gst_structure_get_name (structure);
    _print_log ("caps name [%s]", g_test_data.caps_name);

    gst_caps_unref (caps);
  }
}

/**
 * @brief Callback for signal stream-start.
 */
static void
_stream_start_cb (GstElement * element, GstBuffer * buffer, gpointer user_data)
{
  g_test_data.start = TRUE;
  _print_log ("stream start callback");
}

/**
 * @brief Callback for signal eos.
 */
static void
_eos_cb (GstElement * element, GstBuffer * buffer, gpointer user_data)
{
  g_test_data.end = TRUE;
  _print_log ("eos callback");
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
  g_test_data.start = FALSE;
  g_test_data.end = FALSE;
  g_test_data.caps_name = NULL;

  _print_log ("option num_buffers[%d] test_type[%d]",
      option.num_buffers, option.test_type);

  g_test_data.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_test_data.loop != NULL);

  switch (option.test_type) {
    case TEST_TYPE_VIDEO:
      /** video 160x120 */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_AUDIO:
      /** audio sample rate 16000 (16 bits, signed, little endian) */
      str_pipeline =
          g_strdup_printf
          ("audiotestsrc num-buffers=%d ! audio/x-raw,format=S16LE,rate=16000 ! "
          "tensor_converter ! tensor_sink name=test_sink", option.num_buffers);
      break;
    case TEST_TYPE_TEXT:
      str_pipeline =
          g_strdup_printf
          ("appsrc name=appsrc caps=text/x-raw,format=utf8 ! "
          "tensor_converter ! tensor_sink name=test_sink");
      break;
    case TEST_TYPE_TENSORS:
      /** other/tensors with tensormux */
      str_pipeline =
          g_strdup_printf
          ("tensormux name=mux ! tensor_sink name=test_sink "
          "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_0 "
          "videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! tensor_converter ! mux.sink_1 ",
          option.num_buffers, option.num_buffers);
      break;
    case TEST_TYPE_NEGO_FAILED:
      /** caps negotiation failed */
      str_pipeline =
          g_strdup_printf
          ("videotestsrc num-buffers=%d ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! "
          "videoconvert ! tensor_sink name=test_sink", option.num_buffers);
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

  g_test_data.status = TEST_INIT;
  return TRUE;

error:
  _free_test_data ();
  return FALSE;
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
  TestOption option = { 1, TEST_TYPE_VIDEO };

  ASSERT_TRUE (_setup_pipeline (option));

  /** default signal-rate is 0 */
  g_object_get (g_test_data.sink, "signal-rate", &rate, NULL);
  EXPECT_EQ (rate, 0);

  g_object_set (g_test_data.sink, "signal-rate", (rate + 10), NULL);
  g_object_get (g_test_data.sink, "signal-rate", &res_rate, NULL);
  EXPECT_EQ (res_rate, (rate + 10));

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

  _free_test_data ();
}

/**
 * @brief Test for tensor sink signals.
 */
TEST (tensor_sink_test, signals)
{
  const guint num_buffers = 10;
  gulong handle_id;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO };

  ASSERT_TRUE (_setup_pipeline (option));

  /** tensor sink signals */
  handle_id = g_signal_connect (g_test_data.sink, "new-data",
      (GCallback) _new_data_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  handle_id = g_signal_connect (g_test_data.sink, "stream-start",
      (GCallback) _stream_start_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  handle_id = g_signal_connect (g_test_data.sink, "eos",
      (GCallback) _eos_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  _print_log ("start pipeline for signals test");
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

  _free_test_data ();
}

/**
 * @brief Test for tensor sink signal-rate.
 */
TEST (tensor_sink_test, signal_rate)
{
  const guint num_buffers = 10;
  gulong handle_id;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO };

  ASSERT_TRUE (_setup_pipeline (option));

  /** set signal-rate */
  g_object_set (g_test_data.sink, "signal-rate", (guint) 15, NULL);

  /** signal for new data */
  handle_id = g_signal_connect (g_test_data.sink, "new-data",
      (GCallback) _new_data_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  _print_log ("start pipeline for signal-rate test");
  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_TRUE (g_test_data.received < num_buffers);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  _free_test_data ();
}

/**
 * @brief Test for unknown property and signal.
 */
TEST (tensor_sink_test, unknown_case)
{
  const guint num_buffers = 5;
  gulong handle_id;
  gint unknown = -1;
  TestOption option = { num_buffers, TEST_TYPE_VIDEO };

  ASSERT_TRUE (_setup_pipeline (option));

  /** try to set/get unknown property */
  g_object_set (g_test_data.sink, "unknown-prop", 1, NULL);
  g_object_get (g_test_data.sink, "unknown-prop", &unknown, NULL);
  EXPECT_EQ (unknown, -1);

  /** unknown signal */
  handle_id = g_signal_connect (g_test_data.sink, "unknown-sig",
      (GCallback) _new_data_cb, NULL);
  EXPECT_EQ (handle_id, 0);

  _print_log ("start pipeline for unknown case test");
  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, 0);

  _free_test_data ();
}

/**
 * @brief Test for caps negotiation failed.
 */
TEST (tensor_sink_test, caps_error)
{
  const guint num_buffers = 5;
  gulong handle_id;
  TestOption option = { num_buffers, TEST_TYPE_NEGO_FAILED };

  /** failed : cannot link videoconvert and tensor_sink */
  ASSERT_TRUE (_setup_pipeline (option));

  /** signal for new data */
  handle_id = g_signal_connect (g_test_data.sink, "new-data",
      (GCallback) _new_data_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  _print_log ("start pipeline for caps error test");
  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check error message */
  EXPECT_EQ (g_test_data.status, TEST_ERR_MESSAGE);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, 0);

  _free_test_data ();
}

/**
 * @brief Test for other/tensors caps negotiation.
 */
TEST (tensor_sink_test, caps_tensors)
{
  const guint num_buffers = 5;
  gulong handle_id;
  TestOption option = { num_buffers, TEST_TYPE_TENSORS };

  ASSERT_TRUE (_setup_pipeline (option));

  /** signal for new data */
  handle_id = g_signal_connect (g_test_data.sink, "new-data",
      (GCallback) _new_data_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  _print_log ("start pipeline to test caps other/tensors");
  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensors"));

  _free_test_data ();
}

/**
 * @brief Test for audio stream.
 */
TEST (tensor_sink_test, audio_stream)
{
  const guint num_buffers = 10;
  gulong handle_id;
  TestOption option = { num_buffers, TEST_TYPE_AUDIO };

  ASSERT_TRUE (_setup_pipeline (option));

  /** signal for new data */
  handle_id = g_signal_connect (g_test_data.sink, "new-data",
      (GCallback) _new_data_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  _print_log ("start pipeline to test audio stream");
  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  _free_test_data ();
}

/**
 * @brief Test for text stream.
 */
TEST (tensor_sink_test, text_stream)
{
  const guint num_buffers = 10;
  gulong handle_id;
  guint i;
  GstElement *appsrc;
  TestOption option = { num_buffers, TEST_TYPE_TEXT };

  ASSERT_TRUE (_setup_pipeline (option));

  appsrc = gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "appsrc");

  /** signal for new data */
  handle_id = g_signal_connect (g_test_data.sink, "new-data",
      (GCallback) _new_data_cb, NULL);
  EXPECT_TRUE (handle_id > 0);

  _print_log ("start pipeline to test text stream");
  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);

  for (i = 0; i < num_buffers; i++) {
    GstBuffer *buf = gst_buffer_new_allocate (NULL, 5, NULL);
    GstMapInfo info;

    gst_buffer_map (buf, &info, GST_MAP_WRITE);
    strcpy ((gchar *) info.data, "test");
    gst_buffer_unmap (buf, &info);

    GST_BUFFER_PTS (buf) = (i + 1) * 20 * GST_MSECOND;
    GST_BUFFER_DTS (buf) = GST_BUFFER_PTS (buf);

    EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc), buf),
        GST_FLOW_OK);
  }

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc)), GST_FLOW_OK);

  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /** check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /** check received buffers */
  EXPECT_EQ (g_test_data.received, num_buffers);

  /** check caps name */
  EXPECT_TRUE (g_str_equal (g_test_data.caps_name, "other/tensor"));

  _free_test_data ();
}

/**
 * @brief Main function for unit test.
 */
int
main (int argc, char **argv)
{
  testing::InitGoogleTest (&argc, argv);

  gst_init (&argc, &argv);

  return RUN_ALL_TESTS ();
}
