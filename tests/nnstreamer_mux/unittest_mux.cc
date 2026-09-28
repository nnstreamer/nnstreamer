/**
 * @file        unittest_mux.cc
 * @date        28 Sep 2026
 * @brief       Unit test for tensor_mux
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api.h>
#include <tensor_common.h>
#include <unittest_util.h>

/**
 * @brief Caps of a static stream of two 4-byte uint8 tensors.
 */
#define MUX_TWO_TENSORS_CAPS                                     \
  "other/tensors,format=static,num_tensors=2,types=uint8.uint8," \
  "dimensions=4:1:1:1.4:1:1:1,framerate=0/1"

/**
 * @brief Caps of a flexible stream.
 */
#define MUX_FLEXIBLE_CAPS "other/tensors,format=flexible,framerate=0/1"

/**
 * @brief Deadline for the pipeline to post an error.
 */
#define MUX_TIMEOUT_MS (10000U)

/**
 * @brief Create a harness on the first sink pad of tensor_mux.
 * @param caps the caps of the sink pad
 * @return the harness, which the caller should free with _mux_harness_teardown()
 */
static GstHarness *
_mux_harness_new (const gchar *caps)
{
  GstHarness *h = gst_harness_new_with_padnames ("tensor_mux", "sink_0", "src");
  GstBus *bus = gst_bus_new ();

  /* GST_ELEMENT_ERROR needs a bus to reach the application. */
  gst_element_set_bus (h->element, bus);
  gst_object_unref (bus);

  g_object_set (h->element, "sync-mode", "nosync", NULL);
  gst_harness_set_src_caps_str (h, caps);
  return h;
}

/**
 * @brief Free a harness made by _mux_harness_new().
 * @details The messages queued on the bus hold the element, so the bus goes
 *          first or the element is never freed.
 */
static void
_mux_harness_teardown (GstHarness *h)
{
  gst_element_set_bus (h->element, NULL);
  gst_harness_teardown (h);
}

/**
 * @brief Create a buffer holding one memory of the given size, whose byte i
 *        is i + 1.
 * @return the buffer, which the caller should unref
 */
static GstBuffer *
_single_memory_buffer (gsize size)
{
  GstBuffer *buf = gst_buffer_new_allocate (NULL, size, NULL);
  GstMapInfo map;
  gsize i;

  if (gst_buffer_map (buf, &map, GST_MAP_WRITE)) {
    for (i = 0; i < map.size; i++)
      map.data[i] = (guint8) (i + 1);
    gst_buffer_unmap (buf, &map);
  }

  GST_BUFFER_PTS (buf) = 0;
  return buf;
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
 * @brief A static pad of two tensors takes a single-memory buffer holding
 *        both tensors and splits it by the config of the pad.
 */
TEST (tensorMuxCollect, staticSingleMemory)
{
  const guint8 expected[2][4] = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } };
  GstHarness *h = _mux_harness_new (MUX_TWO_TENSORS_CAPS);
  GstBuffer *out;
  guint i;

  EXPECT_EQ (gst_harness_push (h, _single_memory_buffer (8)), GST_FLOW_OK);
  EXPECT_FALSE (_element_posted_error (h->element));

  out = gst_harness_try_pull (h);
  ASSERT_NE (out, nullptr);
  EXPECT_EQ (gst_tensor_buffer_get_count (out), 2U);
  EXPECT_EQ (gst_buffer_get_size (out), 8U);

  for (i = 0; i < 2; i++) {
    GstMemory *mem = gst_tensor_buffer_get_nth_memory (out, i);
    GstMapInfo map;

    ASSERT_NE (mem, nullptr);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
    EXPECT_EQ (map.size, 4U);
    EXPECT_EQ (memcmp (map.data, expected[i], 4), 0);
    gst_memory_unmap (mem, &map);
    gst_memory_unref (mem);
  }
  gst_buffer_unref (out);

  EXPECT_TRUE (gst_harness_push_event (h, gst_event_new_eos ()));
  _mux_harness_teardown (h);
}

/**
 * @brief A static pad of two tensors takes a single-memory buffer too short
 *        for them: tensor_mux refuses it instead of aborting.
 */
TEST (tensorMuxCollect, staticShortSingleMemory_n)
{
  GstHarness *h = _mux_harness_new (MUX_TWO_TENSORS_CAPS);

  EXPECT_EQ (gst_harness_push (h, _single_memory_buffer (4)), GST_FLOW_ERROR);
  EXPECT_TRUE (_element_posted_error (h->element));
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  _mux_harness_teardown (h);
}

/**
 * @brief Create a buffer of the given number of 1-byte tensors.
 * @return the buffer, which the caller should unref
 */
static GstBuffer *
_tensors_buffer (guint num_tensors)
{
  GstBuffer *buf = gst_buffer_new ();
  GstTensorInfo info;
  guint i;

  gst_tensor_info_init (&info);
  info.type = _NNS_UINT8;
  info.dimension[0] = 1;

  for (i = 0; i < num_tensors; i++) {
    GstMemory *mem = gst_allocator_alloc (NULL, 1, NULL);

    if (!gst_tensor_buffer_append_memory (buf, mem, &info))
      break;
  }

  GST_BUFFER_PTS (buf) = 0;
  return buf;
}

/**
 * @brief A static pad of NNS_TENSOR_MEMORY_MAX tensors takes a buffer of as
 *        many memories whose last one carries an extra tensor: the buffer
 *        holds more tensors than the pad, and tensor_mux refuses it.
 */
TEST (tensorMuxCollect, staticExtraTensor_n)
{
  GString *caps = g_string_new ("other/tensors,format=static,framerate=0/1");
  GstHarness *h;
  GstBuffer *buf;
  guint i;

  g_string_append_printf (caps, ",num_tensors=%d,types=uint8", NNS_TENSOR_MEMORY_MAX);
  for (i = 1; i < NNS_TENSOR_MEMORY_MAX; i++)
    g_string_append (caps, ".uint8");
  g_string_append (caps, ",dimensions=1");
  for (i = 1; i < NNS_TENSOR_MEMORY_MAX; i++)
    g_string_append (caps, ".1");
  h = _mux_harness_new (caps->str);
  g_string_free (caps, TRUE);

  buf = _tensors_buffer (NNS_TENSOR_MEMORY_MAX + 1);
  EXPECT_EQ (gst_buffer_n_memory (buf), (guint) NNS_TENSOR_MEMORY_MAX);
  EXPECT_EQ (gst_tensor_buffer_get_count (buf), (guint) NNS_TENSOR_MEMORY_MAX + 1);

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);
  EXPECT_TRUE (_element_posted_error (h->element));
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  _mux_harness_teardown (h);
}

/**
 * @brief Two flexible pads carrying the maximum number of tensors each
 *        exceed what one buffer can hold: tensor_mux posts an error instead
 *        of aborting.
 */
TEST (tensorMuxCollect, flexibleTooManyTensors_n)
{
  GstElement *pipeline, *src0, *src1;
  GstBuffer *buf;
  GstBus *bus;
  GstMessage *msg;
  GstFlowReturn ret;

  pipeline = gst_parse_launch ("appsrc name=src0 caps=" MUX_FLEXIBLE_CAPS " ! mux.sink_0 "
                               "appsrc name=src1 caps=" MUX_FLEXIBLE_CAPS " ! mux.sink_1 "
                               "tensor_mux name=mux sync-mode=nosync ! fakesink",
      NULL);
  ASSERT_NE (pipeline, nullptr);

  src0 = gst_bin_get_by_name (GST_BIN (pipeline), "src0");
  src1 = gst_bin_get_by_name (GST_BIN (pipeline), "src1");
  ASSERT_NE (src0, nullptr);
  ASSERT_NE (src1, nullptr);

  /* The sink prerolls only once tensor_mux pushes, so do not wait here. */
  EXPECT_NE (gst_element_set_state (pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);

  buf = _tensors_buffer (NNS_TENSOR_SIZE_LIMIT);
  EXPECT_EQ (gst_tensor_buffer_get_count (buf), (guint) NNS_TENSOR_SIZE_LIMIT);
  g_signal_emit_by_name (src0, "push-buffer", buf, &ret);
  EXPECT_EQ (ret, GST_FLOW_OK);
  gst_buffer_unref (buf);

  buf = _tensors_buffer (NNS_TENSOR_SIZE_LIMIT);
  EXPECT_EQ (gst_tensor_buffer_get_count (buf), (guint) NNS_TENSOR_SIZE_LIMIT);
  g_signal_emit_by_name (src1, "push-buffer", buf, &ret);
  EXPECT_EQ (ret, GST_FLOW_OK);
  gst_buffer_unref (buf);

  bus = gst_element_get_bus (pipeline);
  msg = gst_bus_timed_pop_filtered (bus, MUX_TIMEOUT_MS * GST_MSECOND,
      (GstMessageType) (GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
  ASSERT_NE (msg, nullptr);
  EXPECT_EQ (GST_MESSAGE_TYPE (msg), GST_MESSAGE_ERROR);
  gst_message_unref (msg);
  gst_object_unref (bus);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (src0);
  gst_object_unref (src1);
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
