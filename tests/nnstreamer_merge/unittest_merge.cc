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
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>
#include <unittest_util.h>

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
