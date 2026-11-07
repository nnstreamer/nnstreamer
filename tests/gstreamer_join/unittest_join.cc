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
 * @brief Pipeline description shared by the join test cases.
 */
static const gchar *join_pipeline_desc
    = "appsrc name=appsrc_0 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_0 "
      "appsrc name=appsrc_1 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_1 "
      "join name=join ! other/tensor,dimension=(string)3:4:2:2, type=(string)int32, framerate=(fraction)0/1 ! "
      "tensor_sink name=sinkx async=false";

/**
 * @brief Wait for EOS or ERROR on the pipeline bus.
 * @return the message type received, or GST_MESSAGE_UNKNOWN on timeout.
 */
static GstMessageType
pop_eos_or_error (GstElement *pipeline, guint timeout_ms)
{
  GstBus *bus = gst_element_get_bus (pipeline);
  GstMessage *msg;
  GstMessageType type = GST_MESSAGE_UNKNOWN;

  msg = gst_bus_timed_pop_filtered (bus, timeout_ms * GST_MSECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  if (msg) {
    type = GST_MESSAGE_TYPE (msg);
    gst_message_unref (msg);
  }
  gst_object_unref (bus);

  return type;
}

/**
 * @brief Feed one buffer to each source and end both streams, checking that
 *        EOS is posted only after the last stream ends.
 */
static void
run_eos_cycle (GstElement *pipeline, GstElement *appsrc_0, GstElement *appsrc_1, gint *idx)
{
  ASSERT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  *idx = 0;
  ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_0),
                 gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192)),
      GST_FLOW_OK);
  g_usleep (100000);

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_0)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, 300), GST_MESSAGE_UNKNOWN);

  *idx = 1;
  ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_1),
                 gst_buffer_new_wrapped (_g_memdup (test_frames[1], 192), 192)),
      GST_FLOW_OK);
  g_usleep (100000);

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_1)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, UNITTEST_STATECHANGE_TIMEOUT), GST_MESSAGE_EOS);
}

/**
 * @brief Test that join forwards EOS only after every sink pad received it.
 */
TEST (join, eosAfterAllPads)
{
  GstElement *appsrc_0, *appsrc_1, *sink_handle;
  gint idx = 0;

  GstElement *pipeline = gst_parse_launch (join_pipeline_desc, NULL);
  ASSERT_NE (pipeline, nullptr);

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  ASSERT_NE (appsrc_0, nullptr);
  appsrc_1 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_1");
  ASSERT_NE (appsrc_1, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, (gpointer) &idx);

  data_received = 0;
  run_eos_cycle (pipeline, appsrc_0, appsrc_1, &idx);
  EXPECT_EQ (2, data_received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (appsrc_1);
  gst_object_unref (appsrc_0);
  gst_object_unref (pipeline);
}

/**
 * @brief Test that join tracks EOS again after the pipeline is restarted.
 */
TEST (join, eosAfterRestart)
{
  GstElement *appsrc_0, *appsrc_1, *sink_handle;
  gint idx = 0;

  GstElement *pipeline = gst_parse_launch (join_pipeline_desc, NULL);
  ASSERT_NE (pipeline, nullptr);

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  ASSERT_NE (appsrc_0, nullptr);
  appsrc_1 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_1");
  ASSERT_NE (appsrc_1, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, (gpointer) &idx);

  data_received = 0;
  run_eos_cycle (pipeline, appsrc_0, appsrc_1, &idx);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  run_eos_cycle (pipeline, appsrc_0, appsrc_1, &idx);
  EXPECT_EQ (4, data_received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (appsrc_1);
  gst_object_unref (appsrc_0);
  gst_object_unref (pipeline);
}

/**
 * @brief Test that a branch that streamed before a restart and that no stream
 *        can reach afterwards does not hold EOS.
 * @detail sink_1 starts a stream in the first run. Its source is then removed,
 *         so in the second run sink_1 stays linked to a capsfilter that nothing
 *         feeds. The stream state of the first run must not carry over.
 */
TEST (join, eosAfterRestartWithIdleBranch)
{
  GstElement *appsrc_0, *appsrc_1, *sink_handle;
  GstPad *srcpad_1, *peer_1;
  gint idx = 0;

  GstElement *pipeline = gst_parse_launch (join_pipeline_desc, NULL);
  ASSERT_NE (pipeline, nullptr);

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  ASSERT_NE (appsrc_0, nullptr);
  appsrc_1 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_1");
  ASSERT_NE (appsrc_1, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, (gpointer) &idx);

  data_received = 0;
  run_eos_cycle (pipeline, appsrc_0, appsrc_1, &idx);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  srcpad_1 = gst_element_get_static_pad (appsrc_1, "src");
  ASSERT_NE (srcpad_1, nullptr);
  peer_1 = gst_pad_get_peer (srcpad_1);
  ASSERT_NE (peer_1, nullptr);
  EXPECT_TRUE (gst_pad_unlink (srcpad_1, peer_1));
  gst_object_unref (peer_1);
  gst_object_unref (srcpad_1);
  EXPECT_TRUE (gst_bin_remove (GST_BIN (pipeline), appsrc_1));

  ASSERT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  idx = 0;
  ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_0),
                 gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192)),
      GST_FLOW_OK);
  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_0)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, UNITTEST_STATECHANGE_TIMEOUT), GST_MESSAGE_EOS);
  EXPECT_EQ (3, data_received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (appsrc_1);
  gst_object_unref (appsrc_0);
  gst_object_unref (pipeline);
}

/**
 * @brief Test that a join of N finite sources delivers every buffer downstream.
 * @detail This is the pipeline shape used by the datarepo fixtures. The
 *         branches run in their own streaming threads, so which sink pad is
 *         active when the first branch ends is a race; no buffer may be lost
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
  g_signal_connect (sink_handle, "new-data", (GCallback) count_output, (gpointer) &received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (pop_eos_or_error (pipeline, UNITTEST_STATECHANGE_TIMEOUT), GST_MESSAGE_EOS);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_EQ (num_srcs * num_buffers, received);

  gst_object_unref (sink_handle);
  gst_object_unref (pipeline);
}

/**
 * @brief Test that releasing a sink pad completes the EOS aggregation.
 * @detail A released pad can no longer end its stream, so releasing the pad the
 *         join is still waiting for has to end the output stream. The pad
 *         released here has streamed a buffer and is the active one, so the
 *         release also has to stop referencing it as active.
 */
TEST (join, eosAfterPadRelease)
{
  GstElement *appsrc_0, *appsrc_1, *join_handle, *sink_handle;
  GstPad *sinkpad_0, *peer_0, *active_pad = NULL;
  guint n_pads = 0;
  gint idx = 1;

  GstElement *pipeline = gst_parse_launch (join_pipeline_desc, NULL);
  ASSERT_NE (pipeline, nullptr);

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  ASSERT_NE (appsrc_0, nullptr);
  appsrc_1 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_1");
  ASSERT_NE (appsrc_1, nullptr);
  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, (gpointer) &idx);

  data_received = 0;
  ASSERT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  idx = 1;
  ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_1),
                 gst_buffer_new_wrapped (_g_memdup (test_frames[1], 192), 192)),
      GST_FLOW_OK);
  g_usleep (100000);

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_1)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, 300), GST_MESSAGE_UNKNOWN);

  /* sink_0 streams last, so it is the active pad when it is released. */
  idx = 0;
  ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_0),
                 gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192)),
      GST_FLOW_OK);
  g_usleep (100000);

  sinkpad_0 = gst_element_get_static_pad (join_handle, "sink_0");
  ASSERT_NE (sinkpad_0, nullptr);
  g_object_get (join_handle, "active-pad", &active_pad, NULL);
  EXPECT_EQ (sinkpad_0, active_pad);
  g_clear_object (&active_pad);

  peer_0 = gst_pad_get_peer (sinkpad_0);
  if (peer_0) {
    gst_pad_unlink (peer_0, sinkpad_0);
    gst_object_unref (peer_0);
  }
  gst_element_release_request_pad (join_handle, sinkpad_0);

  EXPECT_EQ (pop_eos_or_error (pipeline, UNITTEST_STATECHANGE_TIMEOUT), GST_MESSAGE_EOS);

  g_object_get (join_handle, "active-pad", &active_pad, NULL);
  EXPECT_NE (sinkpad_0, active_pad);
  g_clear_object (&active_pad);
  gst_object_unref (sinkpad_0);

  g_object_get (join_handle, "n-pads", &n_pads, NULL);
  EXPECT_EQ (1U, n_pads);
  EXPECT_EQ (2, data_received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (join_handle);
  gst_object_unref (appsrc_1);
  gst_object_unref (appsrc_0);
  gst_object_unref (pipeline);
}

/**
 * @brief Shared state of the buffer feeding thread.
 */
typedef struct {
  GstElement *appsrc;
  gint stop;
  guint *received;
  guint pushed;
  guint max_lead;
} JoinFeeder;

/**
 * @brief Buffers the feeder may push before the sink has taken them.
 * @detail appsrc queues whatever it is given, so an unheld feeder runs ahead
 *         by as much as the test thread is slow - milliseconds natively, and
 *         minutes under a memory checker, where the queue it leaves behind is
 *         then torn down one buffer at a time. A lead keeps the pad streaming
 *         without letting that cost grow with the environment.
 */
#define JOIN_FEEDER_LEAD (8U)

/**
 * @brief Push buffers into an appsrc until the caller asks it to stop.
 */
static gpointer
join_feeder_thread (gpointer data)
{
  JoinFeeder *feeder = (JoinFeeder *) data;

  while (!g_atomic_int_get (&feeder->stop)) {
    GstBuffer *buf;
    guint taken = g_atomic_int_get (feeder->received);
    guint lead = feeder->pushed > taken ? feeder->pushed - taken : 0;

    if (lead > feeder->max_lead)
      feeder->max_lead = lead;

    if (lead >= JOIN_FEEDER_LEAD) {
      g_thread_yield ();
      continue;
    }

    buf = gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192);

    if (gst_app_src_push_buffer (GST_APP_SRC (feeder->appsrc), buf) != GST_FLOW_OK)
      break;

    feeder->pushed++;
  }

  return NULL;
}

/**
 * @brief Attempts made at the pad release race.
 * @detail Twenty detect a reintroduced ordering defect 20 out of 20 times on
 *         an idle machine, which is_parallel:false gives this test.
 */
#define PAD_RELEASE_RACE_ATTEMPTS (20U)

/**
 * @brief Wall clock the pad release race attempts may spend altogether.
 * @detail They cost about 250 ms together, and far more where a single attempt
 *         is expensive - under a memory checker one attempt alone has taken
 *         93 s. This stops such an environment from paying for all of them.
 */
#define PAD_RELEASE_RACE_BUDGET_US (10 * G_USEC_PER_SEC)

/**
 * @brief Release the sink pad a feeder thread is streaming into.
 * @return TRUE if the released pad is no longer referenced as the active one.
 */
static gboolean
run_pad_release_while_streaming (void)
{
  GstElement *appsrc_0, *join_handle, *sink_handle;
  GstPad *sinkpad_0, *active_pad = NULL;
  JoinFeeder feeder = {};
  GThread *thread;
  guint received = 0, n_pads = 0;
  gboolean released = FALSE, started = TRUE;

  GstElement *pipeline = gst_parse_launch (join_pipeline_desc, NULL);
  if (pipeline == NULL)
    return FALSE;

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  if (appsrc_0 == NULL || join_handle == NULL || sink_handle == NULL) {
    g_clear_object (&appsrc_0);
    g_clear_object (&join_handle);
    g_clear_object (&sink_handle);
    gst_object_unref (pipeline);
    return FALSE;
  }
  g_signal_connect (sink_handle, "new-data", (GCallback) count_output, (gpointer) &received);

  if (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT) != 0)
    started = FALSE;

  feeder.appsrc = appsrc_0;
  feeder.received = &received;
  thread = g_thread_new ("join-feeder", join_feeder_thread, &feeder);
  if (!wait_pipeline_process_buffers (&received, 1, TEST_TIMEOUT_LIMIT_MS))
    started = FALSE;

  /* Released while still linked, so the feeder keeps sink_0 streaming. */
  sinkpad_0 = gst_element_get_static_pad (join_handle, "sink_0");
  gst_element_release_request_pad (join_handle, sinkpad_0);

  g_atomic_int_set (&feeder.stop, 1);
  g_thread_join (thread);
  EXPECT_LE (feeder.max_lead, JOIN_FEEDER_LEAD);
  EXPECT_GT (feeder.max_lead, 0U);

  g_object_get (join_handle, "active-pad", &active_pad, NULL);
  g_object_get (join_handle, "n-pads", &n_pads, NULL);
  released = started && (active_pad != sinkpad_0) && (n_pads == 1U);
  g_clear_object (&active_pad);
  gst_object_unref (sinkpad_0);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  gst_object_unref (sink_handle);
  gst_object_unref (join_handle);
  gst_object_unref (appsrc_0);
  gst_object_unref (pipeline);

  return released;
}

/**
 * @brief Test that releasing a sink pad that is still streaming is safe.
 * @detail The streaming thread of the pad makes it the active pad on every
 *         buffer, so the release has to stop that thread before it drops the
 *         element's reference. Otherwise active-pad is left pointing at a pad
 *         that is no longer part of the element, and the events of the
 *         surviving pads are silently dropped from then on. The window is a
 *         few instructions wide, so the run is repeated; see
 *         PAD_RELEASE_RACE_ATTEMPTS and PAD_RELEASE_RACE_BUDGET_US for how
 *         many times and for how long.
 */
TEST (join, padReleaseWhileStreaming)
{
  gint64 deadline = g_get_monotonic_time () + PAD_RELEASE_RACE_BUDGET_US;
  guint i;

  for (i = 0; i < PAD_RELEASE_RACE_ATTEMPTS; i++) {
    ASSERT_TRUE (run_pad_release_while_streaming ()) << "iteration " << i;
    /* Far slower under a memory checker, which is not testing this race. */
    if (g_get_monotonic_time () > deadline)
      break;
  }
}

/**
 * @brief Test that a flush withdraws the pad from the collected EOS state.
 */
TEST (join, eosAfterFlush)
{
  GstElement *appsrc_0, *appsrc_1, *join_handle, *sink_handle;
  GstPad *sinkpad_0;
  gint idx = 0;

  GstElement *pipeline = gst_parse_launch (join_pipeline_desc, NULL);
  ASSERT_NE (pipeline, nullptr);

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  ASSERT_NE (appsrc_0, nullptr);
  appsrc_1 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_1");
  ASSERT_NE (appsrc_1, nullptr);
  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, (gpointer) &idx);

  data_received = 0;
  ASSERT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_0)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, 300), GST_MESSAGE_UNKNOWN);

  sinkpad_0 = gst_element_get_static_pad (join_handle, "sink_0");
  ASSERT_NE (sinkpad_0, nullptr);
  EXPECT_TRUE (gst_pad_send_event (sinkpad_0, gst_event_new_flush_start ()));
  EXPECT_TRUE (gst_pad_send_event (sinkpad_0, gst_event_new_flush_stop (TRUE)));
  gst_object_unref (sinkpad_0);

  /* sink_0 is no longer EOS, so ending sink_1 must not end the output stream. */
  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_1)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, 300), GST_MESSAGE_UNKNOWN);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (join_handle);
  gst_object_unref (appsrc_1);
  gst_object_unref (appsrc_0);
  gst_object_unref (pipeline);
}

/**
 * @brief Test that an unlinked sink pad does not hold the output stream open.
 */
TEST (join, eosAfterUnlink)
{
  GstElement *appsrc_0, *appsrc_1, *join_handle, *sink_handle;
  GstPad *sinkpad_0, *srcpad_0;
  gint idx = 0;

  GstElement *pipeline = gst_parse_launch (join_pipeline_desc, NULL);
  ASSERT_NE (pipeline, nullptr);

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  ASSERT_NE (appsrc_0, nullptr);
  appsrc_1 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_1");
  ASSERT_NE (appsrc_1, nullptr);
  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, (gpointer) &idx);

  data_received = 0;
  ASSERT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  sinkpad_0 = gst_element_get_static_pad (join_handle, "sink_0");
  ASSERT_NE (sinkpad_0, nullptr);
  srcpad_0 = gst_pad_get_peer (sinkpad_0);
  ASSERT_NE (srcpad_0, nullptr);
  EXPECT_TRUE (gst_pad_unlink (srcpad_0, sinkpad_0));
  gst_object_unref (srcpad_0);
  gst_object_unref (sinkpad_0);

  /* No stream can end on the unlinked pad, so it must not be waited for. */
  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_1)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, UNITTEST_STATECHANGE_TIMEOUT), GST_MESSAGE_EOS);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (join_handle);
  gst_object_unref (appsrc_1);
  gst_object_unref (appsrc_0);
  gst_object_unref (pipeline);
}

/**
 * @brief Buffer count, and the frame each buffer must carry in arrival order.
 */
typedef struct {
  const gint *expected[3];
  guint received;
} JoinCounter;

/**
 * @brief Count a buffer once its payload matches the frame that was pushed.
 * @note Every expected frame is in place before the pipeline plays, so the
 *       streaming thread never reads a field the test thread writes. That is
 *       what the index shared with new_data_cb did, and it is also why this
 *       needs no ordering between the count and the frame it selects. A buffer
 *       arriving past the frames a test named fails on the bound or on the
 *       empty slot, rather than being compared against whatever is there.
 */
static void
count_checked_data_cb (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  JoinCounter *counter = (JoinCounter *) user_data;
  const gint *expected;
  GstMemory *mem_res;
  GstMapInfo info_res;
  gboolean mapped;
  gint *output, i;
  (void) element;

  ASSERT_LT (counter->received, G_N_ELEMENTS (counter->expected));
  expected = counter->expected[counter->received];
  ASSERT_NE (expected, nullptr);

  mem_res = gst_buffer_get_memory (buffer, 0);
  mapped = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
  ASSERT_TRUE (mapped);
  output = (gint *) info_res.data;

  for (i = 0; i < 48; i++) {
    EXPECT_EQ (expected[i], output[i]);
  }
  gst_memory_unmap (mem_res, &info_res);
  gst_memory_unref (mem_res);

  g_atomic_int_inc (&counter->received);
}

/**
 * @brief Test that unlinking the sink pad the join is waiting for ends the
 *        output stream.
 * @detail The reverse order of join.eosAfterUnlink. sink_0 is still linked and
 *         carrying a stream when the other branch ends, so that EOS has to
 *         wait; nothing further arrives on any pad, which leaves the unlink as
 *         the only thing that can complete the set.
 */
TEST (join, eosAfterUnlinkLast)
{
  GstElement *appsrc_0, *appsrc_1, *join_handle, *sink_handle;
  GstPad *sinkpad_0, *srcpad_0;
  JoinCounter counter = { { test_frames[0], NULL, NULL }, 0 };

  GstElement *pipeline = gst_parse_launch (join_pipeline_desc, NULL);
  ASSERT_NE (pipeline, nullptr);

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  ASSERT_NE (appsrc_0, nullptr);
  appsrc_1 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_1");
  ASSERT_NE (appsrc_1, nullptr);
  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) count_checked_data_cb,
      (gpointer) &counter);

  ASSERT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_0),
                 gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192)),
      GST_FLOW_OK);
  ASSERT_TRUE (wait_pipeline_process_buffers (&counter.received, 1, TEST_TIMEOUT_LIMIT_MS));

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_1)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, 300), GST_MESSAGE_UNKNOWN);

  sinkpad_0 = gst_element_get_static_pad (join_handle, "sink_0");
  ASSERT_NE (sinkpad_0, nullptr);
  srcpad_0 = gst_pad_get_peer (sinkpad_0);
  ASSERT_NE (srcpad_0, nullptr);
  EXPECT_TRUE (gst_pad_unlink (srcpad_0, sinkpad_0));
  gst_object_unref (srcpad_0);
  gst_object_unref (sinkpad_0);

  EXPECT_EQ (pop_eos_or_error (pipeline, UNITTEST_STATECHANGE_TIMEOUT), GST_MESSAGE_EOS);
  EXPECT_EQ (1U, counter.received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (join_handle);
  gst_object_unref (appsrc_1);
  gst_object_unref (appsrc_0);
  gst_object_unref (pipeline);
}

/**
 * @brief Test that unlinking a sink pad does not end a stream that is running.
 * @detail Unlinking retires a branch; it does not end the output on its own.
 *         With no stream ended anywhere there is nothing to forward, and the
 *         surviving branch has to keep streaming afterwards.
 */
TEST (join, noEosOnUnlinkWhileRunning)
{
  GstElement *appsrc_0, *appsrc_1, *join_handle, *sink_handle;
  GstPad *sinkpad_0, *srcpad_0;
  /* Frame 1 is branch 1's, so the buffer after the unlink must be its own. */
  JoinCounter counter = { { test_frames[0], test_frames[1], NULL }, 0 };

  GstElement *pipeline = gst_parse_launch (join_pipeline_desc, NULL);
  ASSERT_NE (pipeline, nullptr);

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  ASSERT_NE (appsrc_0, nullptr);
  appsrc_1 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_1");
  ASSERT_NE (appsrc_1, nullptr);
  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) count_checked_data_cb,
      (gpointer) &counter);

  ASSERT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_0),
                 gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192)),
      GST_FLOW_OK);
  ASSERT_TRUE (wait_pipeline_process_buffers (&counter.received, 1, TEST_TIMEOUT_LIMIT_MS));

  sinkpad_0 = gst_element_get_static_pad (join_handle, "sink_0");
  ASSERT_NE (sinkpad_0, nullptr);
  srcpad_0 = gst_pad_get_peer (sinkpad_0);
  ASSERT_NE (srcpad_0, nullptr);
  EXPECT_TRUE (gst_pad_unlink (srcpad_0, sinkpad_0));
  gst_object_unref (srcpad_0);
  gst_object_unref (sinkpad_0);

  EXPECT_EQ (pop_eos_or_error (pipeline, 300), GST_MESSAGE_UNKNOWN);

  ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_1),
                 gst_buffer_new_wrapped (_g_memdup (test_frames[1], 192), 192)),
      GST_FLOW_OK);
  ASSERT_TRUE (wait_pipeline_process_buffers (&counter.received, 2, TEST_TIMEOUT_LIMIT_MS));

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_1)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, UNITTEST_STATECHANGE_TIMEOUT), GST_MESSAGE_EOS);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (join_handle);
  gst_object_unref (appsrc_1);
  gst_object_unref (appsrc_0);
  gst_object_unref (pipeline);
}

/**
 * @brief Test that unlinking one of three sink pads does not end the output.
 * @detail Three branches into one join is the shape the datarepo and
 *         custom-easy filter fixtures use, where an output cut short would
 *         silently truncate a recording. Retiring one branch has to leave the
 *         join waiting for the one that is still linked and still running.
 */
TEST (join, noEosOnUnlinkWhileOthersRun)
{
  const gchar *desc
      = "appsrc name=appsrc_0 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_0 "
        "appsrc name=appsrc_1 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_1 "
        "appsrc name=appsrc_2 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_2 "
        "join name=join ! other/tensor,dimension=(string)3:4:2:2, type=(string)int32, framerate=(fraction)0/1 ! "
        "tensor_sink name=sinkx async=false";
  GstElement *appsrc[3], *join_handle, *sink_handle;
  GstPad *sinkpad_0, *srcpad_0;
  JoinCounter counter = { { test_frames[0], test_frames[0], test_frames[0] }, 0 };
  guint i;

  GstElement *pipeline = gst_parse_launch (desc, NULL);
  ASSERT_NE (pipeline, nullptr);

  for (i = 0; i < 3; i++) {
    gchar *name = g_strdup_printf ("appsrc_%u", i);
    appsrc[i] = gst_bin_get_by_name (GST_BIN (pipeline), name);
    g_free (name);
    ASSERT_NE (appsrc[i], nullptr);
  }
  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) count_checked_data_cb,
      (gpointer) &counter);

  ASSERT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  for (i = 0; i < 3; i++) {
    ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc[i]),
                   gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192)),
        GST_FLOW_OK);
  }
  ASSERT_TRUE (wait_pipeline_process_buffers (&counter.received, 3, TEST_TIMEOUT_LIMIT_MS));

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc[1])), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, 300), GST_MESSAGE_UNKNOWN);

  sinkpad_0 = gst_element_get_static_pad (join_handle, "sink_0");
  ASSERT_NE (sinkpad_0, nullptr);
  srcpad_0 = gst_pad_get_peer (sinkpad_0);
  ASSERT_NE (srcpad_0, nullptr);
  EXPECT_TRUE (gst_pad_unlink (srcpad_0, sinkpad_0));
  gst_object_unref (srcpad_0);
  gst_object_unref (sinkpad_0);

  /* sink_2 is still linked and has not ended, so nothing may be forwarded. */
  EXPECT_EQ (pop_eos_or_error (pipeline, 300), GST_MESSAGE_UNKNOWN);

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc[2])), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, UNITTEST_STATECHANGE_TIMEOUT), GST_MESSAGE_EOS);
  EXPECT_EQ (3U, counter.received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (join_handle);
  for (i = 0; i < 3; i++)
    gst_object_unref (appsrc[i]);
  gst_object_unref (pipeline);
}

/**
 * @brief Run a pipeline whose join.sink_1 is linked to a branch no stream can
 *        reach, and check that the 5 buffers of the other branch and EOS arrive.
 * @param desc The pipeline; the element named "split" never creates src_1.
 */
static void
run_idle_linked_branch (const gchar *desc)
{
  GstElement *pipeline, *split_handle, *sink_handle;
  GstPad *split_src_1;
  guint received = 0;

  pipeline = gst_parse_launch (desc, NULL);
  ASSERT_NE (pipeline, nullptr);

  split_handle = gst_bin_get_by_name (GST_BIN (pipeline), "split");
  ASSERT_NE (split_handle, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) count_output, (gpointer) &received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (pop_eos_or_error (pipeline, TEST_TIMEOUT_LIMIT_MS), GST_MESSAGE_EOS);
  EXPECT_EQ (5U, (guint) g_atomic_int_get (&received));

  /* src_1 must not have been created, or this is not the idle case. */
  split_src_1 = gst_element_get_static_pad (split_handle, "src_1");
  EXPECT_EQ (split_src_1, nullptr);
  if (split_src_1)
    gst_object_unref (split_src_1);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (split_handle);
  gst_object_unref (pipeline);
}

/**
 * @brief Test that a linked branch that no stream can reach does not hold EOS.
 * @detail tensor_if creates its src pads on demand, so with a condition that is
 *         always true src_1 never exists and the queue behind it is never
 *         linked upstream, while join.sink_1 stays linked to that queue. The
 *         EOS of the branch that did stream has to end the output.
 */
TEST (join, eosWithIdleLinkedBranch)
{
  run_idle_linked_branch (
      "videotestsrc num-buffers=5 ! video/x-raw,format=RGB,width=64,height=48,framerate=30/1 ! "
      "tensor_converter ! tensor_if name=split compared-value=A_VALUE compared-value-option=0:0:0:0,0 "
      "supplied-value=0 operator=GE then=PASSTHROUGH else=PASSTHROUGH "
      "split.src_0 ! queue ! join.sink_0 split.src_1 ! queue ! join.sink_1 "
      "join name=join ! tensor_sink name=sinkx sync=false async=false");
}

/**
 * @brief Test that an idle branch behind tensor_demux and a chain of elements
 *        does not hold EOS.
 * @detail tensor_demux creates one src pad per tensor it receives, so with a
 *         single tensor src_1 never exists. The idle branch passes a queue and
 *         a bin around another one, so the unlinked pad is more than one
 *         element upstream of join and the walk crosses ghost pads.
 */
TEST (join, eosWithIdleDemuxBranch)
{
  run_idle_linked_branch (
      "videotestsrc num-buffers=5 ! video/x-raw,format=RGB,width=64,height=48,framerate=30/1 ! "
      "tensor_converter ! tensor_demux name=split "
      "split.src_0 ! queue ! join.sink_0 split.src_1 ! queue ! ( queue ) ! join.sink_1 "
      "join name=join ! tensor_sink name=sinkx sync=false async=false");
}

/**
 * @brief Pad probe that holds every downstream event while it is installed.
 */
static GstPadProbeReturn
hold_events_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  (void) pad;
  (void) info;
  (void) user_data;

  return GST_PAD_PROBE_OK;
}

/**
 * @brief Test that a linked branch whose stream has not started yet holds EOS.
 * @detail A branch whose stream-start has not reached join yet, but whose
 *         source is linked, is a stream that is still on its way, as when its
 *         streaming thread is scheduled late. Ending the other branch must not
 *         end the output and cut that stream off.
 */
TEST (join, noEosWhileUnstartedBranchLinked_n)
{
  GstElement *appsrc_0, *appsrc_1, *join_handle, *sink_handle;
  GstPad *srcpad_1, *sinkpad_1;
  GstEvent *stream_start;
  gulong probe_id;
  JoinCounter counter = { { test_frames[0], NULL, NULL }, 0 };

  GstElement *pipeline = gst_parse_launch (join_pipeline_desc, NULL);
  ASSERT_NE (pipeline, nullptr);

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  ASSERT_NE (appsrc_0, nullptr);
  appsrc_1 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_1");
  ASSERT_NE (appsrc_1, nullptr);
  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) count_checked_data_cb,
      (gpointer) &counter);

  srcpad_1 = gst_element_get_static_pad (appsrc_1, "src");
  ASSERT_NE (srcpad_1, nullptr);
  probe_id = gst_pad_add_probe (srcpad_1,
      (GstPadProbeType) (GST_PAD_PROBE_TYPE_BLOCK | GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM),
      hold_events_probe, NULL, NULL);
  ASSERT_NE (probe_id, 0UL);

  ASSERT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_0),
                 gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192)),
      GST_FLOW_OK);
  ASSERT_TRUE (wait_pipeline_process_buffers (&counter.received, 1, TEST_TIMEOUT_LIMIT_MS));

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_0)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, 300), GST_MESSAGE_UNKNOWN);

  sinkpad_1 = gst_element_get_static_pad (join_handle, "sink_1");
  ASSERT_NE (sinkpad_1, nullptr);
  stream_start = gst_pad_get_sticky_event (sinkpad_1, GST_EVENT_STREAM_START, 0);
  EXPECT_EQ (stream_start, nullptr);
  if (stream_start)
    gst_event_unref (stream_start);
  gst_object_unref (sinkpad_1);

  gst_pad_remove_probe (srcpad_1, probe_id);
  gst_object_unref (srcpad_1);

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_1)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, UNITTEST_STATECHANGE_TIMEOUT), GST_MESSAGE_EOS);
  EXPECT_EQ (1U, counter.received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (join_handle);
  gst_object_unref (appsrc_1);
  gst_object_unref (appsrc_0);
  gst_object_unref (pipeline);
}

/**
 * @brief Test that a source behind an inactive pad of an upstream join holds EOS.
 * @detail The upstream join reports only its active sink pad as an internal
 *         link, and that pad is the first one requested, which stays unlinked.
 *         The late source is linked to its other sink pad, and its stream-start
 *         goes no further until a buffer makes that pad active. Once that
 *         stream-start has made the first pad active, ending the other branch
 *         of the downstream join must not cut that source off.
 */
TEST (join, noEosWhileNestedJoinSourceLinked_n)
{
  GstElement *appsrc_0, *appsrc_1, *up_handle, *late_caps, *sink_handle;
  GstPad *idle_pad, *late_pad, *late_src, *active = NULL;
  GstPadTemplate *templ;
  guint i;
  JoinCounter counter = { { test_frames[0], test_frames[1], NULL }, 0 };

  GstElement *pipeline = gst_parse_launch (
      "appsrc name=appsrc_0 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_0 "
      "appsrc name=appsrc_1 ! capsfilter name=late_caps caps=other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 "
      "join name=up ! join.sink_1 "
      "join name=join ! other/tensor,dimension=(string)3:4:2:2, type=(string)int32, framerate=(fraction)0/1 ! "
      "tensor_sink name=sinkx async=false",
      NULL);
  ASSERT_NE (pipeline, nullptr);

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  ASSERT_NE (appsrc_0, nullptr);
  appsrc_1 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_1");
  ASSERT_NE (appsrc_1, nullptr);
  up_handle = gst_bin_get_by_name (GST_BIN (pipeline), "up");
  ASSERT_NE (up_handle, nullptr);
  late_caps = gst_bin_get_by_name (GST_BIN (pipeline), "late_caps");
  ASSERT_NE (late_caps, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) count_checked_data_cb,
      (gpointer) &counter);

  templ = gst_element_get_pad_template (up_handle, "sink_%u");
  ASSERT_NE (templ, nullptr);
  idle_pad = gst_element_request_pad (up_handle, templ, NULL, NULL);
  ASSERT_NE (idle_pad, nullptr);
  late_pad = gst_element_request_pad (up_handle, templ, NULL, NULL);
  ASSERT_NE (late_pad, nullptr);
  late_src = gst_element_get_static_pad (late_caps, "src");
  ASSERT_NE (late_src, nullptr);
  ASSERT_EQ (gst_pad_link (late_src, late_pad), GST_PAD_LINK_OK);
  gst_object_unref (late_src);

  ASSERT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_0),
                 gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192)),
      GST_FLOW_OK);
  ASSERT_TRUE (wait_pipeline_process_buffers (&counter.received, 1, TEST_TIMEOUT_LIMIT_MS));

  for (i = 0; i < TEST_TIMEOUT_LIMIT_MS / 10; i++) {
    g_object_get (up_handle, "active-pad", &active, NULL);
    if (active == idle_pad)
      break;
    if (active)
      gst_object_unref (active);
    active = NULL;
    g_usleep (10000);
  }
  ASSERT_EQ (active, idle_pad);
  gst_object_unref (active);

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_0)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, 300), GST_MESSAGE_UNKNOWN);

  ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_1),
                 gst_buffer_new_wrapped (_g_memdup (test_frames[1], 192), 192)),
      GST_FLOW_OK);
  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_1)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, UNITTEST_STATECHANGE_TIMEOUT), GST_MESSAGE_EOS);
  EXPECT_EQ (2U, counter.received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_element_release_request_pad (up_handle, late_pad);
  gst_object_unref (late_pad);
  gst_element_release_request_pad (up_handle, idle_pad);
  gst_object_unref (idle_pad);
  gst_object_unref (sink_handle);
  gst_object_unref (late_caps);
  gst_object_unref (up_handle);
  gst_object_unref (appsrc_1);
  gst_object_unref (appsrc_0);
  gst_object_unref (pipeline);
}

/**
 * @brief Test that EOS is held when the walk upstream of an idle branch gives up.
 * @detail join.sink_1 sits behind a chain of tee-to-funnel diamonds whose first
 *         tee is unlinked, so no stream can reach it, but the paths upstream of
 *         it double with every diamond. The walk has to stop at its budget and
 *         keep waiting, rather than visit every path while holding the join.
 */
TEST (join, noEosWhenUpstreamWalkGivesUp_n)
{
  const guint n_diamonds = 8;
  GstElement *appsrc_0, *join_handle, *sink_handle;
  GstPad *sinkpad_1, *peer_1;
  GString *desc;
  guint i;
  JoinCounter counter = { { test_frames[0], NULL, NULL }, 0 };

  desc = g_string_new ("appsrc name=appsrc_0 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_0 "
                       "join name=join ! other/tensor,dimension=(string)3:4:2:2, type=(string)int32, framerate=(fraction)0/1 ! "
                       "tensor_sink name=sinkx async=false ");
  for (i = 0; i < n_diamonds; i++) {
    if (i > 0)
      g_string_append_printf (desc, "f%u. ! ", i - 1);
    g_string_append_printf (desc,
        "tee name=t%u t%u. ! f%u.sink_0 t%u. ! f%u.sink_1 funnel name=f%u ", i,
        i, i, i, i, i);
  }
  g_string_append_printf (desc, "f%u. ! join.sink_1", n_diamonds - 1);

  GstElement *pipeline = gst_parse_launch (desc->str, NULL);
  g_string_free (desc, TRUE);
  ASSERT_NE (pipeline, nullptr);

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  ASSERT_NE (appsrc_0, nullptr);
  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) count_checked_data_cb,
      (gpointer) &counter);

  ASSERT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_0),
                 gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192)),
      GST_FLOW_OK);
  ASSERT_TRUE (wait_pipeline_process_buffers (&counter.received, 1, TEST_TIMEOUT_LIMIT_MS));

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_0)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, 300), GST_MESSAGE_UNKNOWN);

  sinkpad_1 = gst_element_get_static_pad (join_handle, "sink_1");
  ASSERT_NE (sinkpad_1, nullptr);
  peer_1 = gst_pad_get_peer (sinkpad_1);
  ASSERT_NE (peer_1, nullptr);
  EXPECT_TRUE (gst_pad_unlink (peer_1, sinkpad_1));
  gst_object_unref (peer_1);
  gst_object_unref (sinkpad_1);

  EXPECT_EQ (pop_eos_or_error (pipeline, UNITTEST_STATECHANGE_TIMEOUT), GST_MESSAGE_EOS);
  EXPECT_EQ (1U, counter.received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (join_handle);
  gst_object_unref (appsrc_0);
  gst_object_unref (pipeline);
}

/**
 * @brief Test that a sink pad linked to a pad with no element holds EOS.
 * @detail Nothing is known upstream of a pad without a parent, so it is taken
 *         as the origin of a stream that has not started yet.
 */
TEST (join, noEosWhileParentlessPeerLinked_n)
{
  GstElement *appsrc_0, *appsrc_1, *join_handle, *sink_handle;
  GstPad *orphan, *sinkpad_2;
  GstPadTemplate *templ;
  JoinCounter counter = { { test_frames[0], NULL, NULL }, 0 };

  GstElement *pipeline = gst_parse_launch (join_pipeline_desc, NULL);
  ASSERT_NE (pipeline, nullptr);

  appsrc_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  ASSERT_NE (appsrc_0, nullptr);
  appsrc_1 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_1");
  ASSERT_NE (appsrc_1, nullptr);
  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) count_checked_data_cb,
      (gpointer) &counter);

  templ = gst_element_get_pad_template (join_handle, "sink_%u");
  ASSERT_NE (templ, nullptr);
  sinkpad_2 = gst_element_request_pad (join_handle, templ, NULL, NULL);
  ASSERT_NE (sinkpad_2, nullptr);
  orphan = gst_pad_new ("orphan", GST_PAD_SRC);
  ASSERT_NE (orphan, nullptr);
  EXPECT_TRUE (gst_pad_set_active (orphan, TRUE));
  ASSERT_EQ (gst_pad_link (orphan, sinkpad_2), GST_PAD_LINK_OK);

  ASSERT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_0),
                 gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192)),
      GST_FLOW_OK);
  ASSERT_TRUE (wait_pipeline_process_buffers (&counter.received, 1, TEST_TIMEOUT_LIMIT_MS));

  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_0)), GST_FLOW_OK);
  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (appsrc_1)), GST_FLOW_OK);
  EXPECT_EQ (pop_eos_or_error (pipeline, 300), GST_MESSAGE_UNKNOWN);

  EXPECT_TRUE (gst_pad_unlink (orphan, sinkpad_2));
  EXPECT_EQ (pop_eos_or_error (pipeline, UNITTEST_STATECHANGE_TIMEOUT), GST_MESSAGE_EOS);
  EXPECT_EQ (1U, counter.received);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_pad_set_active (orphan, FALSE);
  gst_object_unref (orphan);
  gst_element_release_request_pad (join_handle, sinkpad_2);
  gst_object_unref (sinkpad_2);
  gst_object_unref (sink_handle);
  gst_object_unref (join_handle);
  gst_object_unref (appsrc_1);
  gst_object_unref (appsrc_0);
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
