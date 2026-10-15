/**
 * @file        unittest_merge.cc
 * @date        18 Sep 2026
 * @brief       Unit test for tensor_merge
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>
#include <tuple>
#include <unittest_util.h>
#include <vector>

/**
 * @brief Number of sink pads that make the merged stream reach
 *        GstTensorsInfo::extra, which holds the tensors above
 *        NNS_TENSOR_MEMORY_MAX.
 */
#define EXTRA_NUM_SINKS ((guint) (NNS_TENSOR_MEMORY_MAX + 1))

/**
 * @brief Size of the tensor one sink pad of the tests carries, 4x4 uint8.
 */
#define MERGE_TENSOR_SIZE (16U)

/**
 * @brief Deadline for the merged stream, which the memcheck runs of these
 *        pipelines of 35 elements need to be well above the default.
 */
#define MERGE_TIMEOUT_MS (60000U)

/**
 * @brief Size of the buffer the merged stream carries.
 */
static gsize received_size = 0;

/**
 * @brief Number of the buffers the merged stream carried.
 */
static guint received_count = 0;

/**
 * @brief Record the size of the buffer tensor_sink received.
 */
static void
_record_size_cb (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  UNUSED (element);
  UNUSED (user_data);

  received_size = gst_buffer_get_size (buffer);
}

/**
 * @brief Wait for the pipeline to report that it cannot negotiate.
 * @return TRUE if the pipeline posted an error before the deadline
 */
static gboolean
_wait_negotiation_error (GstElement *pipeline, guint timeout_ms)
{
  GstBus *bus = gst_element_get_bus (pipeline);
  GstMessage *msg;
  gboolean got_error;

  msg = gst_bus_timed_pop_filtered (bus, timeout_ms * GST_MSECOND,
      (GstMessageType) (GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
  got_error = (msg != NULL && GST_MESSAGE_TYPE (msg) == GST_MESSAGE_ERROR);

  if (msg)
    gst_message_unref (msg);
  gst_object_unref (bus);

  return got_error;
}

/**
 * @brief Build a pipeline merging the given number of single-tensor streams.
 * @param num_sinks the number of sink pads the merge element gets
 * @return the pipeline description, which the caller should free
 */
static gchar *
_merge_pipeline (guint num_sinks)
{
  GString *desc = g_string_new (NULL);
  guint i;

  g_string_append (desc, "tensor_merge name=merge mode=linear option=3 "
                         "sync-mode=nosync ! tensor_sink name=sinkx");

  for (i = 0; i < num_sinks; i++) {
    g_string_append_printf (desc,
        " videotestsrc num-buffers=1 pattern=2 ! "
        "video/x-raw,format=GRAY8,width=4,height=4,framerate=30/1 ! "
        "tensor_converter ! merge.sink_%u",
        i);
  }

  return g_string_free (desc, FALSE);
}

/**
 * @brief Merge more streams than NNS_TENSOR_MEMORY_MAX, which fills the
 *        configuration of tensor_merge with extra tensors.
 */
TEST (tensorMergeExtraTensors, mergeExtraSinks)
{
  gchar *desc = _merge_pipeline (EXTRA_NUM_SINKS);
  GstElement *pipeline, *sink;

  received_size = 0;
  received_count = 0;

  pipeline = gst_parse_launch (desc, NULL);
  g_free (desc);
  ASSERT_NE (pipeline, nullptr);

  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink, nullptr);
  g_signal_connect (sink, "new-data", G_CALLBACK (_record_size_cb), NULL);
  g_signal_connect (sink, "new-data", G_CALLBACK (count_output), &received_count);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_TRUE (wait_pipeline_process_buffers (&received_count, 1U, MERGE_TIMEOUT_MS));
  EXPECT_EQ (received_size, MERGE_TENSOR_SIZE * EXTRA_NUM_SINKS);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (sink);
  gst_object_unref (pipeline);
}

/**
 * @brief Merge streams whose merged dimension does not line up.
 */
TEST (tensorMergeExtraTensors, mergeExtraSinks_n)
{
  GString *desc = g_string_new (NULL);
  GstElement *pipeline, *sink;
  guint i;

  received_count = 0;

  g_string_append (desc, "tensor_merge name=merge mode=linear option=0 "
                         "sync-mode=nosync ! tensor_sink name=sinkx");

  for (i = 0; i < EXTRA_NUM_SINKS; i++) {
    g_string_append_printf (desc,
        " videotestsrc num-buffers=1 pattern=2 ! "
        "video/x-raw,format=GRAY8,width=4,height=%u,framerate=30/1 ! "
        "tensor_converter ! merge.sink_%u",
        i + 1, i);
  }

  pipeline = gst_parse_launch (desc->str, NULL);
  g_string_free (desc, TRUE);
  ASSERT_NE (pipeline, nullptr);

  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink, nullptr);
  g_signal_connect (sink, "new-data", G_CALLBACK (count_output), &received_count);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);

  EXPECT_TRUE (_wait_negotiation_error (pipeline, MERGE_TIMEOUT_MS));
  EXPECT_EQ (received_count, 0U);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (sink);
  gst_object_unref (pipeline);
}

/**
 * @brief Caps of a static stream of one 4-byte uint8 tensor.
 */
#define MERGE_ONE_TENSOR_CAPS                              \
  "other/tensors,format=static,num_tensors=1,types=uint8," \
  "dimensions=4:1:1:1,framerate=0/1"

/**
 * @brief Create a harness on the first sink pad of tensor_merge.
 * @return the harness, which the caller should free with _merge_harness_teardown()
 */
static GstHarness *
_merge_harness_new (void)
{
  GstHarness *h = gst_harness_new_with_padnames ("tensor_merge", "sink_0", "src");
  GstBus *bus = gst_bus_new ();

  /* GST_ELEMENT_ERROR needs a bus to reach the application. */
  gst_element_set_bus (h->element, bus);
  gst_object_unref (bus);

  g_object_set (h->element, "mode", "linear", "option", "0", "sync-mode", "nosync", NULL);
  gst_harness_set_src_caps_str (h, MERGE_ONE_TENSOR_CAPS);
  return h;
}

/**
 * @brief Free a harness made by _merge_harness_new().
 * @details The messages queued on the bus hold the element, so the bus goes
 *          first or the element is never freed.
 */
static void
_merge_harness_teardown (GstHarness *h)
{
  gst_element_set_bus (h->element, NULL);
  gst_harness_teardown (h);
}

/**
 * @brief Tell whether the element posted an error on its bus.
 */
static gboolean
_element_posted_error (GstElement *element)
{
  GstBus *bus = gst_element_get_bus (element);
  GstMessage *msg = gst_bus_pop_filtered (bus, GST_MESSAGE_ERROR);
  gboolean posted = (msg != NULL);

  if (msg)
    gst_message_unref (msg);
  gst_object_unref (bus);
  return posted;
}

/**
 * @brief A static pad of one tensor takes the tensor split over two memories
 *        and joins them by the config of the pad.
 */
TEST (tensorMergeCollect, staticTwoMemories)
{
  const guint8 data[4] = { 1, 2, 3, 4 };
  GstHarness *h = _merge_harness_new ();
  GstBuffer *in = gst_buffer_new ();
  GstBuffer *out;

  gst_buffer_append_memory (in, gst_allocator_alloc (NULL, 2, NULL));
  gst_buffer_append_memory (in, gst_allocator_alloc (NULL, 2, NULL));
  EXPECT_EQ (gst_buffer_fill (in, 0, data, sizeof (data)), sizeof (data));
  GST_BUFFER_PTS (in) = 0;

  EXPECT_EQ (gst_harness_push (h, in), GST_FLOW_OK);
  EXPECT_FALSE (_element_posted_error (h->element));

  out = gst_harness_try_pull (h);
  ASSERT_NE (out, nullptr);
  EXPECT_EQ (gst_buffer_n_memory (out), 1U);
  EXPECT_EQ (gst_buffer_get_size (out), 4U);
  EXPECT_EQ (gst_buffer_memcmp (out, 0, data, sizeof (data)), 0);
  gst_buffer_unref (out);

  EXPECT_TRUE (gst_harness_push_event (h, gst_event_new_eos ()));
  _merge_harness_teardown (h);
}

/**
 * @brief A static pad of one tensor takes a buffer too short for it:
 *        tensor_merge refuses it instead of aborting.
 */
TEST (tensorMergeCollect, staticShortBuffer_n)
{
  GstHarness *h = _merge_harness_new ();
  GstBuffer *in = gst_buffer_new ();

  gst_buffer_append_memory (in, gst_allocator_alloc (NULL, 1, NULL));
  gst_buffer_append_memory (in, gst_allocator_alloc (NULL, 1, NULL));
  gst_buffer_memset (in, 0, 0x2A, 2);
  GST_BUFFER_PTS (in) = 0;

  EXPECT_EQ (gst_harness_push (h, in), GST_FLOW_ERROR);
  EXPECT_TRUE (_element_posted_error (h->element));
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  _merge_harness_teardown (h);
}

/**
 * @brief Exercise abbreviated dimensions along each supported merge axis.
 */
class TensorMergeDimensions : public testing::TestWithParam<std::tuple<guint, guint>>
{
  protected:
  GstElement *pipeline = nullptr;
  tensor_dim dims[2] = {};
  tensor_dim merged = {};
  std::vector<guint8> input[2];
  std::vector<guint8> expected;

  /**
   * @brief Build input shapes and a coordinate-based concatenation oracle.
   */
  void SetUp () override
  {
    const guint rank = std::get<0> (GetParam ());
    const guint axis = std::get<1> (GetParam ());
    const guint shape[4] = { 2, 3, 2, 1 };

    for (guint i = 0; i < 4; i++) {
      dims[0][i] = dims[1][i] = i < rank ? shape[i] : 1;
      merged[i] = dims[0][i];
    }
    if (axis < rank)
      dims[1][axis]++;
    merged[axis] += dims[1][axis];

    for (guint i = 0; i < 2; i++) {
      input[i].resize (gst_tensor_get_element_count (dims[i]));
      for (gsize j = 0; j < input[i].size (); j++)
        input[i][j] = 1 + j + i * 100;
    }

    /* Map output coordinates to their input, independently of the copy loops. */
    for (gsize i = 0; i < gst_tensor_get_element_count (merged); i++) {
      gsize index = i;
      guint coord[4];
      for (guint j = 0; j < 4; j++) {
        coord[j] = index % merged[j];
        index /= merged[j];
      }
      const guint source = coord[axis] >= dims[0][axis] ? 1 : 0;
      if (source)
        coord[axis] -= dims[0][axis];
      index = coord[3];
      for (gint j = 2; j >= 0; j--)
        index = index * dims[source][j] + coord[j];
      expected.push_back (input[source][index]);
    }
  }

  /**
   * @brief Stop the pipeline even after a failed assertion.
   */
  void TearDown () override
  {
    if (pipeline) {
      gst_element_set_state (pipeline, GST_STATE_NULL);
      gst_object_unref (pipeline);
      pipeline = nullptr;
    }
  }

  /**
   * @brief Select full, abbreviated, or mixed dimension notation on the inputs.
   * @param notation bit i abbreviates input i to the parameterized rank
   */
  void configure (guint notation)
  {
    TearDown ();
    pipeline = gst_parse_launch ("tensor_merge name=merge mode=linear sync-mode=nosync ! "
                                 "appsink name=out sync=false async=false wait-on-eos=false "
                                 "appsrc name=in0 format=time ! merge.sink_0 "
                                 "appsrc name=in1 format=time ! merge.sink_1",
        NULL);

    const guint rank = std::get<0> (GetParam ());
    g_autofree gchar *option = g_strdup_printf ("%u", std::get<1> (GetParam ()));
    g_autoptr (GstElement) merge = gst_bin_get_by_name (GST_BIN (pipeline), "merge");
    g_object_set (merge, "option", option, NULL);
    for (guint i = 0; i < 2; i++) {
      g_autofree gchar *name = g_strdup_printf ("in%u", i);
      g_autoptr (GstElement) src = gst_bin_get_by_name (GST_BIN (pipeline), name);
      g_autofree gchar *dimensions = gst_tensor_get_rank_dimension_string (
          dims[i], notation & (1U << i) ? rank : 4, FALSE);
      g_autoptr (GstCaps) caps = gst_caps_new_simple ("other/tensors", "format",
          G_TYPE_STRING, "static", "num_tensors", G_TYPE_INT, 1, "types",
          G_TYPE_STRING, "uint8", "dimensions", G_TYPE_STRING, dimensions,
          "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
      gst_app_src_set_caps (GST_APP_SRC (src), caps);
    }
  }

  /**
   * @brief Push both tensors with an optional size error.
   * @param size_error number of bytes to add to or remove from the buffer
   * @return the pipeline's EOS or error message type, or UNKNOWN on timeout
   */
  GstMessageType run (gint size_error = 0)
  {
    gst_element_set_state (pipeline, GST_STATE_PLAYING);
    for (guint i = 0; i < 2; i++) {
      std::vector<guint8> bytes = input[i];
      if (i == 0)
        bytes.resize (bytes.size () + size_error);
      GstBuffer *in = gst_buffer_new_allocate (NULL, bytes.size (), NULL);
      gst_buffer_fill (in, 0, bytes.data (), bytes.size ());
      GST_BUFFER_PTS (in) = 0;
      g_autofree gchar *name = g_strdup_printf ("in%u", i);
      g_autoptr (GstElement) src = gst_bin_get_by_name (GST_BIN (pipeline), name);
      gst_app_src_push_buffer (GST_APP_SRC (src), in);
      gst_app_src_end_of_stream (GST_APP_SRC (src));
    }
    g_autoptr (GstBus) bus = gst_element_get_bus (pipeline);
    g_autoptr (GstMessage) msg = gst_bus_timed_pop_filtered (bus,
        10 * GST_SECOND, (GstMessageType) (GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    return msg ? GST_MESSAGE_TYPE (msg) : GST_MESSAGE_UNKNOWN;
  }
};

/**
 * @brief Full and abbreviated shapes produce the same bytes and output caps.
 */
TEST_P (TensorMergeDimensions, equivalentNotation)
{
  for (guint notation = 0; notation < 4; notation++) {
    SCOPED_TRACE (notation);
    configure (notation);
    EXPECT_EQ (run (), GST_MESSAGE_EOS);

    g_autoptr (GstElement) sink = gst_bin_get_by_name (GST_BIN (pipeline), "out");
    g_autoptr (GstSample) sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), 0);
    ASSERT_NE (sample, nullptr);
    GstBuffer *out = gst_sample_get_buffer (sample);
    EXPECT_EQ (gst_buffer_get_size (out), expected.size ());
    EXPECT_EQ (gst_buffer_memcmp (out, 0, expected.data (), expected.size ()), 0);

    GstCaps *caps = gst_sample_get_caps (sample);
    ASSERT_NE (caps, nullptr);
    GstTensorsConfig config;
    gst_tensors_config_init (&config);
    EXPECT_TRUE (gst_tensors_config_from_caps (&config, caps, TRUE));
    EXPECT_EQ (config.info.num_tensors, 1U);
    EXPECT_EQ (config.info.info[0].type, _NNS_UINT8);
    for (guint i = 0; i < 4; i++)
      EXPECT_EQ (config.info.info[0].dimension[i], merged[i]);
    gst_tensors_config_free (&config);
  }
}

/**
 * @brief Abbreviated dimensions must not admit short or oversized input buffers.
 */
TEST_P (TensorMergeDimensions, wrongBufferSize_n)
{
  for (guint notation = 0; notation < 4; notation++) {
    for (gint size_error : { -1, 1 }) {
      SCOPED_TRACE (notation);
      SCOPED_TRACE (size_error);
      configure (notation);
      EXPECT_EQ (run (size_error), GST_MESSAGE_ERROR);
      g_autoptr (GstElement) sink = gst_bin_get_by_name (GST_BIN (pipeline), "out");
      g_autoptr (GstSample) sample
          = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), 0);
      EXPECT_EQ (sample, nullptr);
    }
  }
}

INSTANTIATE_TEST_SUITE_P (RanksAndAxes, TensorMergeDimensions,
    testing::Combine (testing::Values (1U, 2U, 3U), testing::Values (0U, 1U, 2U, 3U)));

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
