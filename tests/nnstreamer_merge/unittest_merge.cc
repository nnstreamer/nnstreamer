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
#include <gst/gst.h>
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
