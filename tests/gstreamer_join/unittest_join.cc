/**
 * @file        unittest_join.cc
 * @date        10 Nov 2020
 * @brief       Unit test for gstreamer join element
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <nnstreamer_util.h>
#include <unittest_util.h>

static int data_received;

/**
 * @brief Test data for join (2 frames with dimension 3:4:2:2)
 */
const gint test_frames[2][48]
    = { { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112,
            1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124,
            1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212,
            1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224 },
        { 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113,
            2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2201,
            2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213,
            2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224 } };

/**
 * @brief Callback for tensor sink signal.
 */
static void
new_data_cb (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  GstMemory *mem_res;
  GstMapInfo info_res;
  gboolean mapped;
  gint *output, i;
  gint index = *(gint *) user_data;
  (void) element;

  data_received++;
  /* Index 100 means a callback that is not allowed. */
  EXPECT_NE (100, index);
  mem_res = gst_buffer_get_memory (buffer, 0);
  mapped = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
  ASSERT_TRUE (mapped);
  output = (gint *) info_res.data;

  for (i = 0; i < 48; i++) {
    EXPECT_EQ (test_frames[index][i], output[i]);
  }
  gst_memory_unmap (mem_res, &info_res);
  gst_memory_unref (mem_res);
}

/**
 * @brief Test join element with appsrc
 */
TEST (join, normal0)
{
  gint idx, n_pads;
  GstBuffer *buf_0, *buf_1, *buf_3, *buf_4;
  GstElement *appsrc_handle_0, *appsrc_handle_1, *sink_handle, *join_handle;
  GstPad *active_pad;
  gchar *active_name;

  gchar *str_pipeline = g_strdup (
      "appsrc name=appsrc_0 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_0 "
      "appsrc name=appsrc_1 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_1 "
      "join name=join ! other/tensor,dimension=(string)3:4:2:2, type=(string)int32, framerate=(fraction)0/1 ! "
      "tensor_sink name=sinkx async=false");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);

  appsrc_handle_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  EXPECT_NE (appsrc_handle_0, nullptr);

  appsrc_handle_1 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_1");
  EXPECT_NE (appsrc_handle_1, nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, (gpointer) &idx);

  buf_0 = gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192);
  buf_3 = gst_buffer_copy (buf_0);

  buf_1 = gst_buffer_new_wrapped (_g_memdup (test_frames[1], 192), 192);
  buf_4 = gst_buffer_copy (buf_1);

  data_received = 0;
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  idx = 0;
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle_0), buf_0), GST_FLOW_OK);
  g_usleep (100000);

  idx = 1;
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle_1), buf_1), GST_FLOW_OK);
  g_usleep (100000);

  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle_1), buf_4), GST_FLOW_OK);
  g_usleep (100000);

  g_object_get (join_handle, "active-pad", &active_pad, NULL);
  EXPECT_NE (nullptr, active_pad);
  active_name = gst_pad_get_name (active_pad);
  EXPECT_STREQ ("sink_1", active_name);
  gst_object_unref (active_pad);
  g_free (active_name);

  idx = 0;
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle_0), buf_3), GST_FLOW_OK);
  g_usleep (100000);

  g_object_get (join_handle, "active-pad", &active_pad, NULL);
  EXPECT_NE (nullptr, active_pad);
  active_name = gst_pad_get_name (active_pad);
  EXPECT_STREQ ("sink_0", active_name);
  gst_object_unref (active_pad);
  g_free (active_name);

  g_object_get (join_handle, "n-pads", &n_pads, NULL);
  EXPECT_EQ (2, n_pads);

  gst_object_unref (sink_handle);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);
  EXPECT_EQ (4, data_received);

  gst_object_unref (appsrc_handle_0);
  gst_object_unref (appsrc_handle_1);
  gst_object_unref (join_handle);
  gst_object_unref (pipeline);
}

/**
 * @brief Callback counting the buffers that reached the sink.
 */
static void
count_data_cb (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  guint *received = (guint *) user_data;
  (void) element;
  (void) buffer;

  (*received)++;
}

/**
 * @brief Wait for the EOS (or an error) message of the given pipeline.
 * @return TRUE if EOS was received before the time-out.
 */
static gboolean
wait_pipeline_eos (GstElement *pipeline, guint timeout_ms)
{
  GstBus *bus = gst_element_get_bus (pipeline);
  GstMessage *msg;
  gboolean got_eos = FALSE;

  if (bus == NULL)
    return FALSE;

  msg = gst_bus_timed_pop_filtered (bus, timeout_ms * GST_MSECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  if (msg != NULL) {
    got_eos = (GST_MESSAGE_TYPE (msg) == GST_MESSAGE_EOS);
    gst_message_unref (msg);
  }

  gst_object_unref (bus);
  return got_eos;
}

/**
 * @brief Test that EOS of one sink pad does not cut off the other streams.
 * @detail Join is an N-to-1 element, so the buffers of every input stream
 *         should be forwarded until all of them have ended. Before the EOS
 *         aggregation was added, the EOS of the sink pad that happened to be
 *         active was forwarded right away and the downstream elements dropped
 *         everything the remaining streams pushed afterwards.
 */
TEST (join, eosAggregation)
{
  guint received = 0;
  guint i;
  GstElement *pipeline, *appsrc_0, *appsrc_1, *sink_handle;
  GstBuffer *buf;

  gchar *str_pipeline = g_strdup (
      "appsrc name=appsrc_0 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_0 "
      "appsrc name=appsrc_1 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_1 "
      "join name=join ! other/tensor,dimension=(string)3:4:2:2, type=(string)int32, framerate=(fraction)0/1 ! "
      "tensor_sink name=sinkx async=false sync=false");

  pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  ASSERT_NE (appsrc_0, nullptr);
  appsrc_1 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_1");
  ASSERT_NE (appsrc_1, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) count_data_cb, (gpointer) &received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  /* sink_0 pushes first, so it becomes the active pad. */
  buf = gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192);
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_0), buf), GST_FLOW_OK);
  EXPECT_TRUE (wait_pipeline_process_buffers (&received, 1, TEST_TIMEOUT_LIMIT_MS));

  /* The active pad ends first; the other stream must not be cut off. */
  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_0)), GST_FLOW_OK);
  EXPECT_FALSE (wait_pipeline_eos (pipeline, 200));

  for (i = 0; i < 3; i++) {
    buf = gst_buffer_new_wrapped (_g_memdup (test_frames[1], 192), 192);
    EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_1), buf), GST_FLOW_OK);
  }
  EXPECT_TRUE (wait_pipeline_process_buffers (&received, 4, TEST_TIMEOUT_LIMIT_MS));
  EXPECT_EQ (4U, received);

  /* Once every sink pad is EOS, EOS is forwarded downstream. */
  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_1)), GST_FLOW_OK);
  EXPECT_TRUE (wait_pipeline_eos (pipeline, TEST_TIMEOUT_LIMIT_MS));

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (4U, received);

  gst_object_unref (sink_handle);
  gst_object_unref (appsrc_0);
  gst_object_unref (appsrc_1);
  gst_object_unref (pipeline);
}

/**
 * @brief Test that a join of N finite sources delivers every buffer downstream.
 * @detail This is the pipeline shape used by the datarepo test fixtures. The
 *         three branches run in their own streaming threads, so which sink pad
 *         is active when the first branch ends is a race; no buffer may be lost
 *         because of it.
 */
TEST (join, forwardAllBuffers)
{
  const guint num_srcs = 3;
  const guint num_buffers = 10;
  guint received = 0;
  guint i;
  GstElement *pipeline, *sink_handle;
  GString *desc = g_string_new (NULL);

  for (i = 0; i < num_srcs; i++) {
    g_string_append_printf (desc,
        "videotestsrc num-buffers=%u ! "
        "video/x-raw,format=RGB,width=64,height=48,framerate=30/1 ! "
        "tensor_converter ! queue ! join0.sink_%u ",
        num_buffers, i);
  }
  /* qos=false: a late-buffer QoS event would make upstream drop frames. */
  g_string_append (desc,
      "join name=join0 ! tensor_sink name=sinkx sync=false qos=false async=false");

  pipeline = gst_parse_launch (desc->str, NULL);
  g_string_free (desc, TRUE);
  ASSERT_NE (pipeline, nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) count_data_cb, (gpointer) &received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_pipeline_eos (pipeline, TEST_TIMEOUT_LIMIT_MS));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_EQ (num_srcs * num_buffers, received);

  gst_object_unref (sink_handle);
  gst_object_unref (pipeline);
}

/**
 * @brief Test that releasing a sink pad completes the EOS aggregation.
 * @detail A sink pad that is gone can no longer deliver EOS, so releasing the
 *         last pad the join is still waiting for has to end the output stream.
 */
TEST (join, releasePadAfterEos)
{
  guint received = 0;
  guint n_pads = 0;
  GstElement *pipeline, *appsrc_0, *join_handle, *sink_handle;
  GstBuffer *buf;
  GstPad *sinkpad_1, *peer_1;

  gchar *str_pipeline = g_strdup (
      "appsrc name=appsrc_0 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_0 "
      "appsrc name=appsrc_1 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_1 "
      "join name=join ! other/tensor,dimension=(string)3:4:2:2, type=(string)int32, framerate=(fraction)0/1 ! "
      "tensor_sink name=sinkx async=false sync=false");

  pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  ASSERT_NE (appsrc_0, nullptr);
  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) count_data_cb, (gpointer) &received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  buf = gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192);
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_0), buf), GST_FLOW_OK);
  EXPECT_TRUE (wait_pipeline_process_buffers (&received, 1, TEST_TIMEOUT_LIMIT_MS));

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_0)), GST_FLOW_OK);
  EXPECT_FALSE (wait_pipeline_eos (pipeline, 200));

  /* sink_1 never gets EOS; dropping the pad has to complete the aggregation. */
  sinkpad_1 = gst_element_get_static_pad (join_handle, "sink_1");
  ASSERT_NE (sinkpad_1, nullptr);
  peer_1 = gst_pad_get_peer (sinkpad_1);
  if (peer_1 != NULL) {
    gst_pad_unlink (peer_1, sinkpad_1);
    gst_object_unref (peer_1);
  }
  gst_element_release_request_pad (join_handle, sinkpad_1);
  gst_object_unref (sinkpad_1);

  EXPECT_TRUE (wait_pipeline_eos (pipeline, TEST_TIMEOUT_LIMIT_MS));

  g_object_get (join_handle, "n-pads", &n_pads, NULL);
  EXPECT_EQ (1U, n_pads);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (1U, received);

  gst_object_unref (sink_handle);
  gst_object_unref (join_handle);
  gst_object_unref (appsrc_0);
  gst_object_unref (pipeline);
}

/**
 * @brief Test get property with invalid parameter
 */
TEST (join, prop0_n)
{
  GstElement *join_handle;
  gchar *str_val = NULL;

  gchar *str_pipeline = g_strdup (
      "appsrc name=appsrc_0 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_0 "
      "appsrc name=appsrc_1 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_1 "
      "join name=join ! other/tensor,dimension=(string)3:4:2:2, type=(string)int32, framerate=(fraction)0/1 ! "
      "tensor_sink name=sinkx async=false");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);

  g_object_get (G_OBJECT (join_handle), "invalid_prop", &str_val, NULL);
  EXPECT_TRUE (str_val == NULL);

  gst_object_unref (join_handle);
  gst_object_unref (pipeline);
}

/**
 * @brief Test get property with NULL parameter
 */
TEST (join, prop1_n)
{
  GstElement *join_handle;
  gchar *str_val = NULL;

  gchar *str_pipeline = g_strdup (
      "appsrc name=appsrc_0 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_0 "
      "appsrc name=appsrc_1 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_1 "
      "join name=join ! other/tensor,dimension=(string)3:4:2:2, type=(string)int32, framerate=(fraction)0/1 ! "
      "tensor_sink name=sinkx async=false");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);

  g_object_get (G_OBJECT (join_handle), NULL, &str_val, NULL);
  EXPECT_TRUE (str_val == NULL);

  gst_object_unref (join_handle);
  gst_object_unref (pipeline);
}

/**
 * @brief Main GTest
 */
int
main (int argc, char **argv)
{
  int result = -1;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
