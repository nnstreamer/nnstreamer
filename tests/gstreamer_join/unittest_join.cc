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
  g_signal_connect (sink_handle, "new-data", (GCallback) count_data_cb, (gpointer) &received);

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
} JoinFeeder;

/**
 * @brief Push buffers into an appsrc until the caller asks it to stop.
 */
static gpointer
join_feeder_thread (gpointer data)
{
  JoinFeeder *feeder = (JoinFeeder *) data;

  while (!g_atomic_int_get (&feeder->stop)) {
    GstBuffer *buf = gst_buffer_new_wrapped (_g_memdup (test_frames[0], 192), 192);

    if (gst_app_src_push_buffer (GST_APP_SRC (feeder->appsrc), buf) != GST_FLOW_OK)
      break;
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
  JoinFeeder feeder = { NULL, 0 };
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
  g_signal_connect (sink_handle, "new-data", (GCallback) count_data_cb, (gpointer) &received);

  if (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT) != 0)
    started = FALSE;

  feeder.appsrc = appsrc_0;
  thread = g_thread_new ("join-feeder", join_feeder_thread, &feeder);
  if (!wait_pipeline_process_buffers (&received, 1, TEST_TIMEOUT_LIMIT_MS))
    started = FALSE;

  /* Released while still linked, so the feeder keeps sink_0 streaming. */
  sinkpad_0 = gst_element_get_static_pad (join_handle, "sink_0");
  gst_element_release_request_pad (join_handle, sinkpad_0);

  g_atomic_int_set (&feeder.stop, 1);
  g_thread_join (thread);

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

  counter->received++;
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
