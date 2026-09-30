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
 * @brief Data size of each flexible tensor the tests below make.
 */
#define MUX_FLEX_DATA_SIZE (4U)

/**
 * @brief Create a flexible tensor: a meta header and 4 uint8 values, the
 *        j-th of which is (id * 4 + j + 1).
 * @return the memory, which the caller should unref
 */
static GstMemory *
_flexible_tensor (guint id)
{
  GstTensorInfo info;
  GstTensorMetaInfo meta;
  GstMemory *data, *mem;
  GstMapInfo map;
  guint j;

  gst_tensor_info_init (&info);
  info.type = _NNS_UINT8;
  info.dimension[0] = MUX_FLEX_DATA_SIZE;
  gst_tensor_info_convert_to_meta (&info, &meta);
  meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  data = gst_allocator_alloc (NULL, MUX_FLEX_DATA_SIZE, NULL);
  if (gst_memory_map (data, &map, GST_MAP_WRITE)) {
    for (j = 0; j < MUX_FLEX_DATA_SIZE; j++)
      map.data[j] = (guint8) (id * MUX_FLEX_DATA_SIZE + j + 1);
    gst_memory_unmap (data, &map);
  }

  mem = gst_tensor_meta_info_append_header (&meta, data);
  gst_memory_unref (data);
  return mem;
}

/**
 * @brief Create a buffer of one memory holding flexible tensors 0 to
 *        num_tensors - 1 back to back, followed by tail bytes of 0xff.
 * @return the buffer, which the caller should unref
 */
static GstBuffer *
_flexible_single_memory (guint num_tensors, gsize tail)
{
  GstMemory *first = _flexible_tensor (0);
  gsize tsize = gst_memory_get_sizes (first, NULL, NULL);
  GstBuffer *buf;
  GstMapInfo map;
  gsize offset = 0;
  guint i;

  gst_memory_unref (first);
  buf = gst_buffer_new_allocate (NULL, tsize * num_tensors + tail, NULL);

  if (gst_buffer_map (buf, &map, GST_MAP_WRITE)) {
    for (i = 0; i < num_tensors; i++) {
      GstMemory *mem = _flexible_tensor (i);
      GstMapInfo tmap;

      if (gst_memory_map (mem, &tmap, GST_MAP_READ)) {
        memcpy (map.data + offset, tmap.data, tmap.size);
        gst_memory_unmap (mem, &tmap);
      }
      gst_memory_unref (mem);
      offset += tsize;
    }
    memset (map.data + offset, 0xff, tail);
    gst_buffer_unmap (buf, &map);
  }

  GST_BUFFER_PTS (buf) = 0;
  return buf;
}

/**
 * @brief Check that the n-th tensor of a buffer is the whole flexible tensor
 *        made by _flexible_tensor (id).
 */
static void
_expect_flexible_tensor (GstBuffer *buf, guint nth, guint id)
{
  GstMemory *mem = gst_tensor_buffer_get_nth_memory (buf, nth);
  GstMemory *expected = _flexible_tensor (id);
  GstMapInfo map, emap;

  ASSERT_NE (mem, nullptr);
  ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
  ASSERT_TRUE (gst_memory_map (expected, &emap, GST_MAP_READ));
  EXPECT_EQ (map.size, emap.size) << "tensor " << nth;
  if (map.size == emap.size) {
    EXPECT_EQ (memcmp (map.data, emap.data, map.size), 0) << "tensor " << nth;
  }
  gst_memory_unmap (expected, &emap);
  gst_memory_unmap (mem, &map);
  gst_memory_unref (expected);
  gst_memory_unref (mem);
}

/**
 * @brief Push a buffer to a flexible tensor_mux pad and check that the
 *        output carries flexible tensors 0 to num_tensors - 1 in order.
 */
static void
_expect_flexible_collect (GstBuffer *in, guint num_tensors)
{
  GstHarness *h = _mux_harness_new (MUX_FLEXIBLE_CAPS);
  GstBuffer *out;
  guint i;

  EXPECT_EQ (gst_harness_push (h, in), GST_FLOW_OK);
  EXPECT_FALSE (_element_posted_error (h->element));

  out = gst_harness_try_pull (h);
  EXPECT_NE (out, nullptr);
  if (out) {
    EXPECT_EQ (gst_tensor_buffer_get_count (out), num_tensors);
    for (i = 0; i < num_tensors && i < gst_tensor_buffer_get_count (out); i++)
      _expect_flexible_tensor (out, i, i);
    gst_buffer_unref (out);
  }

  EXPECT_TRUE (gst_harness_push_event (h, gst_event_new_eos ()));
  _mux_harness_teardown (h);
}

/**
 * @brief A flexible pad takes one memory holding two flexible tensors and
 *        collects both of them, in order.
 */
TEST (tensorMuxCollect, flexibleSingleMemory)
{
  _expect_flexible_collect (_flexible_single_memory (2, 0), 2);
}

/**
 * @brief A flexible pad takes one memory holding NNS_TENSOR_MEMORY_MAX
 *        flexible tensors, the most that gst_tensor_buffer_from_config() can
 *        split, and collects all of them.
 */
TEST (tensorMuxCollect, flexibleSingleMemoryMax)
{
  _expect_flexible_collect (
      _flexible_single_memory (NNS_TENSOR_MEMORY_MAX, 0), NNS_TENSOR_MEMORY_MAX);
}

/**
 * @brief A flexible pad takes one memory made from a buffer of more than
 *        NNS_TENSOR_MEMORY_MAX flexible tensors, whose last memory holds the
 *        extra tensors, and collects all of them.
 */
TEST (tensorMuxCollect, flexibleSingleMemoryExtraTensors)
{
  const guint num_tensors = NNS_TENSOR_MEMORY_MAX + 2;
  GstBuffer *tensors = gst_buffer_new ();
  GstBuffer *in = gst_buffer_new ();
  GstTensorInfo info;
  guint i;

  gst_tensor_info_init (&info);
  info.type = _NNS_UINT8;
  info.dimension[0] = MUX_FLEX_DATA_SIZE;
  for (i = 0; i < num_tensors; i++)
    ASSERT_TRUE (gst_tensor_buffer_append_memory (tensors, _flexible_tensor (i), &info));
  ASSERT_EQ (gst_tensor_buffer_get_count (tensors), num_tensors);

  gst_buffer_append_memory (in, gst_buffer_get_all_memory (tensors));
  gst_buffer_unref (tensors);
  GST_BUFFER_PTS (in) = 0;

  _expect_flexible_collect (in, num_tensors);
}

/**
 * @brief The bytes after the last flexible tensor of a single memory are not
 *        a tensor, and tensor_mux does not collect them.
 */
TEST (tensorMuxCollect, flexibleSingleMemoryTail)
{
  /* shorter than a meta header, and as long as one but not a valid header */
  _expect_flexible_collect (_flexible_single_memory (2, 10), 2);
  _expect_flexible_collect (_flexible_single_memory (2, 200), 2);
}

/**
 * @brief A single flexible tensor followed by bytes that are not a tensor
 *        still gives that tensor alone.
 */
TEST (tensorMuxCollect, flexibleSingleTensorTail)
{
  _expect_flexible_collect (_flexible_single_memory (1, 10), 1);
  _expect_flexible_collect (_flexible_single_memory (1, 200), 1);
}

/**
 * @brief Flexible inputs that were never split keep their tensor count: a
 *        single tensor in one memory, and one tensor per memory.
 */
TEST (tensorMuxCollect, flexibleUnsplit)
{
  GstBuffer *in = gst_buffer_new ();
  guint i;

  _expect_flexible_collect (_flexible_single_memory (1, 0), 1);

  for (i = 0; i < 3; i++)
    gst_buffer_append_memory (in, _flexible_tensor (i));
  GST_BUFFER_PTS (in) = 0;
  _expect_flexible_collect (in, 3);
}

/**
 * @brief Flexible tensors from two pads, one of which sends them in a single
 *        memory, are collected in pad order.
 */
TEST (tensorMuxCollect, flexibleSingleMemoryTwoPads)
{
  GstElement *pipeline, *src0, *src1, *sink;
  GstBuffer *buf;
  GstSample *sample = NULL;
  GstFlowReturn ret;
  guint i;

  pipeline = gst_parse_launch (
      "appsrc name=src0 caps=" MUX_FLEXIBLE_CAPS " ! mux.sink_0 "
      "appsrc name=src1 caps=" MUX_FLEXIBLE_CAPS " ! mux.sink_1 "
      "tensor_mux name=mux sync-mode=nosync ! appsink name=sink sync=false",
      NULL);
  ASSERT_NE (pipeline, nullptr);

  src0 = gst_bin_get_by_name (GST_BIN (pipeline), "src0");
  src1 = gst_bin_get_by_name (GST_BIN (pipeline), "src1");
  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");
  ASSERT_NE (src0, nullptr);
  ASSERT_NE (src1, nullptr);
  ASSERT_NE (sink, nullptr);

  EXPECT_NE (gst_element_set_state (pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);

  buf = _flexible_single_memory (2, 0);
  g_signal_emit_by_name (src0, "push-buffer", buf, &ret);
  EXPECT_EQ (ret, GST_FLOW_OK);
  gst_buffer_unref (buf);

  buf = gst_buffer_new ();
  gst_buffer_append_memory (buf, _flexible_tensor (2));
  GST_BUFFER_PTS (buf) = 0;
  g_signal_emit_by_name (src1, "push-buffer", buf, &ret);
  EXPECT_EQ (ret, GST_FLOW_OK);
  gst_buffer_unref (buf);

  g_signal_emit_by_name (sink, "try-pull-sample", MUX_TIMEOUT_MS * GST_MSECOND, &sample);
  ASSERT_NE (sample, nullptr);
  buf = gst_sample_get_buffer (sample);
  ASSERT_NE (buf, nullptr);
  EXPECT_EQ (gst_tensor_buffer_get_count (buf), 3U);
  for (i = 0; i < 3 && i < gst_tensor_buffer_get_count (buf); i++)
    _expect_flexible_tensor (buf, i, i);
  gst_sample_unref (sample);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (src0);
  gst_object_unref (src1);
  gst_object_unref (sink);
  gst_object_unref (pipeline);
}

/**
 * @brief One memory holding more flexible tensors than
 *        gst_tensor_buffer_from_config() can split, with no extra-tensors
 *        header: tensor_mux refuses it instead of passing on the first
 *        tensors only.
 */
TEST (tensorMuxCollect, flexibleSingleMemoryTooManyTensors_n)
{
  GstHarness *h = _mux_harness_new (MUX_FLEXIBLE_CAPS);

  EXPECT_EQ (gst_harness_push (h, _flexible_single_memory (NNS_TENSOR_MEMORY_MAX + 1, 0)),
      GST_FLOW_ERROR);
  EXPECT_TRUE (_element_posted_error (h->element));
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  _mux_harness_teardown (h);
}

/**
 * @brief One memory holding a flexible tensor whose header claims more data
 *        than is left: tensor_mux refuses it.
 */
TEST (tensorMuxCollect, flexibleSingleMemoryTruncated_n)
{
  GstHarness *h = _mux_harness_new (MUX_FLEXIBLE_CAPS);
  GstBuffer *in = _flexible_single_memory (2, 0);

  gst_buffer_resize (in, 0, gst_buffer_get_size (in) - 1);
  EXPECT_EQ (gst_harness_push (h, in), GST_FLOW_ERROR);
  EXPECT_TRUE (_element_posted_error (h->element));
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  _mux_harness_teardown (h);
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
