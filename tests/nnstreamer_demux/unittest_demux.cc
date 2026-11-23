/**
 * @file        unittest_demux.cc
 * @date        18 Sep 2026
 * @brief       Unit test for tensor_demux
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <tensor_common.h>
#include <unittest_util.h>
#include "../gst/nnstreamer/elements/gsttensor_demux.h"

/**
 * @brief Number of tensors a stream carries to reach GstTensorsInfo::extra.
 */
#define EXTRA_NUM_TENSORS ((guint) (NNS_TENSOR_MEMORY_MAX + 4))

/**
 * @brief Start a standalone tensor_demux and hand out its sink pad.
 * @param demux the element to start, filled in by this function
 * @return the sink pad of the element, which the caller should unref
 */
static GstPad *
_start_tensor_demux (GstElement **demux)
{
  GstElement *element;
  GstPad *sinkpad;

  element = gst_element_factory_make ("tensor_demux", NULL);
  if (element == NULL)
    return NULL;

  gst_element_set_state (element, GST_STATE_PLAYING);
  sinkpad = gst_element_get_static_pad (element, "sink");
  gst_pad_send_event (sinkpad, gst_event_new_stream_start ("tensordemux-extra"));

  *demux = element;
  return sinkpad;
}

/**
 * @brief Renegotiate a stream of more tensors than NNS_TENSOR_MEMORY_MAX, which
 *        reparses the configuration of tensor_demux.
 */
TEST (tensorDemuxExtraTensors, capsRenegotiation)
{
  GstElement *demux = NULL;
  GstPad *sinkpad;
  GstCaps *caps;

  sinkpad = _start_tensor_demux (&demux);
  ASSERT_NE (sinkpad, nullptr);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_caps (caps)));
  gst_caps_unref (caps);

  EXPECT_EQ (GST_TENSOR_DEMUX (demux)->tensors_config.info.num_tensors, EXTRA_NUM_TENSORS);
  EXPECT_NE (GST_TENSOR_DEMUX (demux)->tensors_config.info.extra, nullptr);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS + 1, EXTRA_NUM_TENSORS + 1);
  EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_caps (caps)));
  gst_caps_unref (caps);

  EXPECT_EQ (GST_TENSOR_DEMUX (demux)->tensors_config.info.num_tensors,
      EXTRA_NUM_TENSORS + 1);

  gst_element_set_state (demux, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (demux);
}

/**
 * @brief Negotiate a stream whose tensors are not all described.
 */
TEST (tensorDemuxExtraTensors, capsRenegotiation_n)
{
  GstElement *demux = NULL;
  GstPad *sinkpad;
  GstCaps *caps;

  sinkpad = _start_tensor_demux (&demux);
  ASSERT_NE (sinkpad, nullptr);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_caps (caps)));
  gst_caps_unref (caps);

  /* the element does not propagate a parse failure, so check its config */
  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS - 1);
  gst_pad_send_event (sinkpad, gst_event_new_caps (caps));
  gst_caps_unref (caps);

  EXPECT_EQ (GST_TENSOR_DEMUX (demux)->tensors_config.info.num_tensors, 0U);

  gst_element_set_state (demux, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (demux);
}

/**
 * @brief Chain a buffer into a standalone tensor_demux negotiated with the
 *        given caps, and tell whether the element posted an error message.
 * @param caps the caps to negotiate, or NULL to chain before any caps
 * @param buffer the buffer to chain, which this function takes
 * @param posted set to TRUE if the element posted an error message
 * @return the flow return of the chain
 */
static GstFlowReturn
_chain_tensor_demux (GstCaps *caps, GstBuffer *buffer, gboolean *posted)
{
  GstElement *demux = NULL;
  GstPad *sinkpad;
  GstBus *bus;
  GstMessage *msg;
  GstSegment segment;
  GstFlowReturn ret;

  sinkpad = _start_tensor_demux (&demux);
  if (sinkpad == NULL) {
    gst_buffer_unref (buffer);
    return GST_FLOW_CUSTOM_ERROR;
  }

  bus = gst_bus_new ();
  gst_element_set_bus (demux, bus);

  if (caps) {
    EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_caps (caps)));
  }

  gst_segment_init (&segment, GST_FORMAT_TIME);
  EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_segment (&segment)));

  ret = gst_pad_chain (sinkpad, buffer);

  msg = gst_bus_pop_filtered (bus, GST_MESSAGE_ERROR);
  *posted = (msg != NULL);
  if (msg)
    gst_message_unref (msg);

  gst_element_set_state (demux, GST_STATE_NULL);
  gst_element_set_bus (demux, NULL);
  gst_object_unref (bus);
  gst_object_unref (sinkpad);
  gst_object_unref (demux);
  return ret;
}

/**
 * @brief Build a buffer of the given number of zeroed memories.
 * @param num_mems the number of memories
 * @param size the size of each memory
 * @return the buffer, which the caller should unref
 */
static GstBuffer *
_buffer_with_memories (guint num_mems, gsize size)
{
  GstBuffer *buffer = gst_buffer_new ();
  guint i;

  for (i = 0; i < num_mems; i++)
    gst_buffer_append_memory (buffer, gst_allocator_alloc (NULL, size, NULL));

  gst_buffer_memset (buffer, 0, 0, gst_buffer_get_size (buffer));
  return buffer;
}

/**
 * @brief Build a buffer of the tensors the caps declare, more tensors
 *        than NNS_TENSOR_MEMORY_MAX gathered in its last memory.
 * @param caps the caps declaring the tensors
 * @param distinct fill each tensor with its index instead of zero
 * @return the buffer, which the caller should unref
 */
static GstBuffer *
_buffer_with_tensors (GstCaps *caps, gboolean distinct = FALSE)
{
  GstTensorsConfig config;
  GstBuffer *buffer = gst_buffer_new ();
  guint i;

  EXPECT_TRUE (gst_tensors_config_from_caps (&config, caps, TRUE));

  for (i = 0; i < config.info.num_tensors; i++) {
    GstTensorInfo *_info = gst_tensors_info_get_nth_info (&config.info, i);
    GstMemory *mem = gst_allocator_alloc (NULL, gst_tensor_info_get_size (_info), NULL);
    GstMapInfo map;

    EXPECT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));
    memset (map.data, distinct ? i : 0, map.size);
    gst_memory_unmap (mem, &map);

    EXPECT_TRUE (gst_tensor_buffer_append_memory (buffer, mem, _info));
  }

  gst_tensors_config_free (&config);
  return buffer;
}

/**
 * @brief Chain a buffer holding as many tensors as the caps declare.
 */
TEST (tensorDemuxTensorCount, matchingBuffer)
{
  GstCaps *caps = caps_with_tensors (2, 2);
  gboolean posted = TRUE;

  /* the src pad is created by the chain and has no peer to push to */
  EXPECT_EQ (_chain_tensor_demux (caps, _buffer_with_memories (2, 4), &posted),
      GST_FLOW_NOT_LINKED);
  EXPECT_FALSE (posted);

  gst_caps_unref (caps);
}

/**
 * @brief Chain a buffer carrying the two tensors of the caps in one memory,
 *        which the element splits by the caps.
 */
TEST (tensorDemuxTensorCount, splitBuffer)
{
  GstCaps *caps = caps_with_tensors (2, 2);
  gboolean posted = TRUE;

  EXPECT_EQ (_chain_tensor_demux (caps, _buffer_with_memories (1, 8), &posted),
      GST_FLOW_NOT_LINKED);
  EXPECT_FALSE (posted);

  gst_caps_unref (caps);
}

/**
 * @brief Chain a buffer holding more tensors than NNS_TENSOR_MEMORY_MAX, as
 *        many as the caps declare.
 */
TEST (tensorDemuxTensorCount, matchingExtraBuffer)
{
  GstCaps *caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  GstBuffer *buffer = _buffer_with_tensors (caps);
  gboolean posted = TRUE;

  ASSERT_EQ (gst_tensor_buffer_get_count (buffer), EXTRA_NUM_TENSORS);
  EXPECT_EQ (_chain_tensor_demux (caps, buffer, &posted), GST_FLOW_NOT_LINKED);
  EXPECT_FALSE (posted);

  gst_caps_unref (caps);
}

/**
 * @brief Pick a packed extra tensor and group two tensors in reverse index order.
 */
TEST (tensorDemuxExtraTensors, selectsPackedTensorData)
{
  GstElement *pipeline, *src, *single, *group;
  GstCaps *caps;
  GstBuffer *buffer;
  GstSample *sample;
  gchar *description;
  const guint extra = NNS_TENSOR_MEMORY_MAX + 1;
  const guint last = EXTRA_NUM_TENSORS - 1;
  const guint indices[2][2] = { { extra, 0 }, { last, 0 } };

  description = g_strdup_printf (
      "appsrc name=input format=time ! tensor_demux name=demux tensorpick=%u,%u:0 "
      "demux.src_0 ! appsink name=single async=false sync=false "
      "demux.src_1 ! appsink name=group async=false sync=false",
      extra, last);
  pipeline = gst_parse_launch_full (description, NULL, GST_PARSE_FLAG_FATAL_ERRORS, NULL);
  g_free (description);
  ASSERT_NE (pipeline, nullptr);

  src = gst_bin_get_by_name (GST_BIN (pipeline), "input");
  single = gst_bin_get_by_name (GST_BIN (pipeline), "single");
  group = gst_bin_get_by_name (GST_BIN (pipeline), "group");
  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  gst_app_src_set_caps (GST_APP_SRC (src), caps);
  buffer = _buffer_with_tensors (caps, TRUE);
  gst_caps_unref (caps);

  EXPECT_EQ (gst_buffer_n_memory (buffer), (guint) NNS_TENSOR_MEMORY_MAX);
  EXPECT_NE (gst_element_set_state (pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (src), buffer), GST_FLOW_OK);
  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (src)), GST_FLOW_OK);

  GstElement *sinks[] = { single, group };
  for (guint i = 0; i < G_N_ELEMENTS (sinks); i++) {
    sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sinks[i]), GST_SECOND);
    EXPECT_NE (sample, nullptr);
    if (!sample)
      continue;

    const GstStructure *structure
        = gst_caps_get_structure (gst_sample_get_caps (sample), 0);
    gint num_tensors = 0;
    EXPECT_STREQ (gst_structure_get_name (structure), "other/tensors");
    EXPECT_TRUE (gst_structure_get_int (structure, "num_tensors", &num_tensors));
    EXPECT_EQ (num_tensors, (gint) i + 1);
    buffer = gst_sample_get_buffer (sample);
    EXPECT_EQ (gst_buffer_n_memory (buffer), i + 1);
    for (guint j = 0; j <= i && j < gst_buffer_n_memory (buffer); j++) {
      GstMemory *mem = gst_buffer_peek_memory (buffer, j);
      GstMapInfo map;
      const gboolean mapped = gst_memory_map (mem, &map, GST_MAP_READ);
      EXPECT_TRUE (mapped);
      if (mapped) {
        EXPECT_EQ (map.size, 4U);
        for (gsize k = 0; k < map.size; k++)
          EXPECT_EQ (map.data[k], indices[i][j]);
        gst_memory_unmap (mem, &map);
      }
    }
    gst_sample_unref (sample);
  }

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (group);
  gst_object_unref (single);
  gst_object_unref (src);
  gst_object_unref (pipeline);
}

/**
 * @brief Reject a packed buffer with fewer tensors than the negotiated caps.
 */
TEST (tensorDemuxTensorCount, shortPackedBuffer_n)
{
  GstCaps *buffer_caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  GstCaps *input_caps = caps_with_tensors (EXTRA_NUM_TENSORS + 1, EXTRA_NUM_TENSORS + 1);
  GstBuffer *buffer = _buffer_with_tensors (buffer_caps);
  gboolean posted = FALSE;

  EXPECT_EQ (gst_tensor_buffer_get_count (buffer), EXTRA_NUM_TENSORS);
  EXPECT_EQ (_chain_tensor_demux (input_caps, buffer, &posted), GST_FLOW_ERROR);
  EXPECT_TRUE (posted);

  gst_caps_unref (input_caps);
  gst_caps_unref (buffer_caps);
}

/**
 * @brief Chain a buffer too short for the tensors of the caps, which the
 *        element cannot split.
 */
TEST (tensorDemuxTensorCount, shortBuffer_n)
{
  GstCaps *caps = caps_with_tensors (2, 2);
  gboolean posted = FALSE;

  EXPECT_EQ (_chain_tensor_demux (caps, _buffer_with_memories (1, 4), &posted), GST_FLOW_ERROR);
  EXPECT_TRUE (posted);

  gst_caps_unref (caps);
}

/**
 * @brief Chain a buffer before the caps are negotiated.
 */
TEST (tensorDemuxTensorCount, bufferBeforeCaps_n)
{
  gboolean posted = FALSE;

  EXPECT_EQ (_chain_tensor_demux (NULL, _buffer_with_memories (1, 4), &posted), GST_FLOW_ERROR);
  EXPECT_TRUE (posted);
}

/**
 * @brief Chain a buffer whose last memory lacks the header of the extra
 *        tensors the caps declare, so it holds fewer tensors than the caps.
 */
TEST (tensorDemuxTensorCount, extraHeaderMissing_n)
{
  GstCaps *caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  GstBuffer *buffer = _buffer_with_tensors (caps);
  GstMemory *last;
  GstMapInfo map;
  gboolean posted = FALSE;

  ASSERT_EQ (gst_buffer_n_memory (buffer), (guint) NNS_TENSOR_MEMORY_MAX);
  last = gst_buffer_peek_memory (buffer, NNS_TENSOR_MEMORY_MAX - 1);
  ASSERT_TRUE (gst_memory_map (last, &map, GST_MAP_WRITE));
  memset (map.data, 0, map.size);
  gst_memory_unmap (last, &map);
  ASSERT_EQ (gst_tensor_buffer_get_count (buffer), (guint) NNS_TENSOR_MEMORY_MAX);

  EXPECT_EQ (_chain_tensor_demux (caps, buffer, &posted), GST_FLOW_ERROR);
  EXPECT_TRUE (posted);

  gst_caps_unref (caps);
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
