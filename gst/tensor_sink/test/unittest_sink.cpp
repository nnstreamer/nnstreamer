/**
 * @file	unittest_sink.cpp
 * @date	29 June 2018
 * @brief	Unit test for tensor sink plugin
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 */

#include <gtest/gtest.h>
#include <gst/gst.h>

/**
 * @brief Macro for debug mode.
 */
#define DBG FALSE

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

  if (DBG) {
    GstMemory *mem;
    GstMapInfo info;
    guint i;
    guint num_mems;

    num_mems = gst_buffer_n_memory (buffer);
    for (i = 0; i < num_mems; i++) {
      mem = gst_buffer_peek_memory (buffer, i);

      if (gst_memory_map (mem, &info, GST_MAP_READ)) {
        /* check data (info.data, info.size) */
        _print_log ("received %ld", info.size);

        gst_memory_unmap (mem, &info);
      }
    }
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
_setup_pipeline (const guint num_buffers)
{
  gchar *str_pipeline;

  g_test_data.status = TEST_START;
  g_test_data.received = 0;
  g_test_data.start = FALSE;
  g_test_data.end = FALSE;

  g_test_data.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_test_data.loop != NULL);

  str_pipeline =
      g_strdup_printf
      ("videotestsrc num-buffers=%d ! video/x-raw,width=640,height=480,framerate=(fraction)30/1 ! videoconvert ! video/x-raw,format=RGB ! tensor_converter ! tensor_sink name=test_sink",
      num_buffers);
  g_test_data.pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  _check_cond_err (g_test_data.pipeline != NULL);

  g_test_data.bus = gst_element_get_bus (g_test_data.pipeline);
  _check_cond_err (g_test_data.bus != NULL);

  gst_bus_add_signal_watch (g_test_data.bus);
  g_signal_connect (g_test_data.bus, "message", (GCallback) _message_cb, NULL);

  g_test_data.sink =
      gst_bin_get_by_name (GST_BIN (g_test_data.pipeline), "test_sink");
  _check_cond_err (g_test_data.sink != NULL);

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
  guint64 rate, res_rate;
  gboolean silent, res_silent;
  gboolean emit, res_emit;

  ASSERT_TRUE (_setup_pipeline (1));

  /* default render-rate is 0 */
  g_object_get (g_test_data.sink, "render-rate", &rate, NULL);
  EXPECT_EQ (rate, 0);

  g_object_set (g_test_data.sink, "render-rate", (rate + 10), NULL);
  g_object_get (g_test_data.sink, "render-rate", &res_rate, NULL);
  EXPECT_EQ (res_rate, (rate + 10));

  /* default emit-signal is TRUE */
  g_object_get (g_test_data.sink, "emit-signal", &emit, NULL);
  EXPECT_EQ (emit, TRUE);

  g_object_set (g_test_data.sink, "emit-signal", !emit, NULL);
  g_object_get (g_test_data.sink, "emit-signal", &res_emit, NULL);
  EXPECT_EQ (res_emit, !emit);

  /* default silent is TRUE */
  g_object_get (g_test_data.sink, "silent", &silent, NULL);
  EXPECT_EQ (silent, TRUE);

  g_object_set (g_test_data.sink, "silent", !silent, NULL);
  g_object_get (g_test_data.sink, "silent", &res_silent, NULL);
  EXPECT_EQ (res_silent, !silent);

  _free_test_data ();
}

/**
 * @brief Test for tensor sink signals.
 */
TEST (tensor_sink_test, signals)
{
  const guint num_buffers = 10;

  ASSERT_TRUE (_setup_pipeline (num_buffers));

  /* enable emit-signal */
  g_object_set (g_test_data.sink, "emit-signal", TRUE, NULL);

  /* tensor sink signals */
  g_signal_connect (g_test_data.sink, "new-data",
      (GCallback) _new_data_cb, NULL);
  g_signal_connect (g_test_data.sink, "stream-start",
      (GCallback) _stream_start_cb, NULL);
  g_signal_connect (g_test_data.sink, "eos", (GCallback) _eos_cb, NULL);

  _print_log ("start pipeline for signals test");
  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /* check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /* check received buffers and signals */
  EXPECT_EQ (g_test_data.received, num_buffers);
  EXPECT_EQ (g_test_data.start, TRUE);
  EXPECT_EQ (g_test_data.end, TRUE);

  _free_test_data ();
}

/**
 * @brief Test for tensor sink render-rate.
 */
TEST (tensor_sink_test, render_rate)
{
  const guint num_buffers = 10;

  ASSERT_TRUE (_setup_pipeline (num_buffers));

  /* enable emit-signal */
  g_object_set (g_test_data.sink, "emit-signal", TRUE, NULL);

  /* set render-rate */
  g_object_set (g_test_data.sink, "render-rate", 15, NULL);

  /* signal for new data */
  g_signal_connect (g_test_data.sink, "new-data",
      (GCallback) _new_data_cb, NULL);

  _print_log ("start pipeline for render-rate test");
  gst_element_set_state (g_test_data.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (g_test_data.loop);
  gst_element_set_state (g_test_data.pipeline, GST_STATE_NULL);

  /* check eos message */
  EXPECT_EQ (g_test_data.status, TEST_EOS);

  /* check received buffers */
  EXPECT_TRUE (g_test_data.received < num_buffers);

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
