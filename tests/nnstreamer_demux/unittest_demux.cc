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
#include <vector>
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
 * @brief Build a buffer of the zeroed tensors the caps declare, more tensors
 *        than NNS_TENSOR_MEMORY_MAX gathered in its last memory.
 * @param caps the caps declaring the tensors
 * @return the buffer, which the caller should unref
 */
static GstBuffer *
_buffer_with_tensors (GstCaps *caps)
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
    memset (map.data, 0, map.size);
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
 * @brief Private extra-header layout, used only to damage a mux output.
 */
typedef struct {
  uint32_t magic;
  uint32_t version;
  uint32_t num_extra_tensors;
  uint64_t reserved;
  GstTensorInfo infos[NNS_TENSOR_SIZE_EXTRA_LIMIT];
} DemuxTestExtraInfo;

/**
 * @brief Exercise flexible tensor bytes across linked mux and demux elements.
 */
class TensorMuxDemuxFlexible : public ::testing::Test
{
  protected:
  GstElement *pipeline = nullptr;
  GstElement *mux = nullptr;
  GstBus *bus = nullptr;
  std::vector<GstElement *> inputs;
  std::vector<GstElement *> outputs;
  std::vector<std::vector<guint8>> expected;
  gint damaged = 0;
  guint invalid_index = 0;

  /** @brief Stop the streaming thread before releasing its sinks and bus. */
  void TearDown () override
  {
    if (pipeline) {
      gst_element_set_state (pipeline, GST_STATE_NULL);
      gst_object_unref (pipeline);
    }
    if (bus)
      gst_object_unref (bus);
  }

  /** @brief Link each dynamic demux output to its own collecting sink. */
  static void link_output (GstElement *, GstPad *pad, gpointer user_data)
  {
    auto *self = static_cast<TensorMuxDemuxFlexible *> (user_data);
    const gchar *name = GST_PAD_NAME (pad);
    EXPECT_TRUE (g_str_has_prefix (name, "src_"));
    if (!g_str_has_prefix (name, "src_"))
      return;
    guint64 index = g_ascii_strtoull (name + 4, nullptr, 10);
    EXPECT_LT (index, self->outputs.size ());
    if (index >= self->outputs.size ())
      return;
    GstPad *sinkpad = gst_element_get_static_pad (self->outputs[index], "sink");
    EXPECT_EQ (gst_pad_link (pad, sinkpad), GST_PAD_LINK_OK);
    gst_object_unref (sinkpad);
  }

  /** @brief Corrupt the first extra entry after mux has produced valid bytes. */
  static GstPadProbeReturn damage_extra (GstPad *, GstPadProbeInfo *probe, gpointer user_data)
  {
    auto *self = static_cast<TensorMuxDemuxFlexible *> (user_data);
    GstBuffer *buffer = gst_buffer_make_writable (GST_PAD_PROBE_INFO_BUFFER (probe));
    GST_PAD_PROBE_INFO_DATA (probe) = buffer;
    EXPECT_EQ (gst_buffer_n_memory (buffer), (guint) NNS_TENSOR_MEMORY_MAX);
    EXPECT_EQ (gst_tensor_buffer_get_count (buffer), EXTRA_NUM_TENSORS);
    GstMemory *memory = gst_buffer_peek_memory (buffer, NNS_TENSOR_MEMORY_MAX - 1);
    GstMapInfo map;
    if (!gst_memory_map (memory, &map, GST_MAP_WRITE)) {
      ADD_FAILURE () << "Cannot damage the mux extra header";
      return GST_PAD_PROBE_OK;
    }
    if (map.size >= sizeof (DemuxTestExtraInfo)) {
      GstTensorInfo entry;
      gsize offset = G_STRUCT_OFFSET (DemuxTestExtraInfo, infos)
                     + self->invalid_index * sizeof (GstTensorInfo);
      memcpy (&entry, map.data + offset, sizeof (entry));
      EXPECT_EQ (entry.type, _NNS_UINT8);
      EXPECT_EQ (entry.dimension[0],
          self->expected[NNS_TENSOR_MEMORY_MAX + self->invalid_index].size ());
      entry.dimension[0] = G_MAXUINT32;
      memcpy (map.data + offset, &entry, sizeof (entry));
      g_atomic_int_set (&self->damaged, 1);
    } else {
      ADD_FAILURE () << "Mux output lacks the extra header";
    }
    gst_memory_unmap (memory, &map);
    return GST_PAD_PROBE_OK;
  }

  /** @brief Build distinct complete flexible tensors and a real pipeline. */
  void prepare (guint count, gboolean corrupt)
  {
    pipeline = gst_pipeline_new (nullptr);
    mux = gst_element_factory_make ("tensor_mux", nullptr);
    GstElement *demux = gst_element_factory_make ("tensor_demux", nullptr);
    ASSERT_NE (mux, nullptr);
    ASSERT_NE (demux, nullptr);
    gst_bin_add_many (GST_BIN (pipeline), mux, demux, nullptr);
    g_object_set (mux, "sync-mode", "nosync", nullptr);
    ASSERT_TRUE (gst_element_link (mux, demux));
    GstCaps *caps = gst_caps_from_string ("other/tensors,format=flexible,framerate=0/1");
    for (guint i = 0; i < 2; i++) {
      GstElement *input = gst_element_factory_make ("appsrc", nullptr);
      ASSERT_NE (input, nullptr);
      gst_bin_add (GST_BIN (pipeline), input);
      ASSERT_TRUE (gst_element_link (input, mux));
      g_object_set (input, "caps", caps, "format", GST_FORMAT_TIME, nullptr);
      inputs.push_back (input);
    }
    gst_caps_unref (caps);

    std::vector<guint8> bytes[2];
    for (guint i = 0; i < count; i++) {
      /* Header alignment is covered separately; vary aligned payload sizes. */
      const guint payload_size = (i + 1) * 8;
      GstTensorInfo info;
      GstTensorMetaInfo meta;
      gst_tensor_info_init (&info);
      info.type = _NNS_UINT8;
      info.dimension[0] = payload_size;
      gst_tensor_info_convert_to_meta (&info, &meta);
      meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
      GstMemory *payload = gst_allocator_alloc (nullptr, payload_size, nullptr);
      GstMapInfo map;
      ASSERT_TRUE (gst_memory_map (payload, &map, GST_MAP_WRITE));
      memset (map.data, 0x40 + i, map.size);
      gst_memory_unmap (payload, &map);
      GstMemory *tensor = gst_tensor_meta_info_append_header (&meta, payload);
      gst_memory_unref (payload);
      ASSERT_NE (tensor, nullptr);
      ASSERT_TRUE (gst_memory_map (tensor, &map, GST_MAP_READ));
      expected.emplace_back (map.data, map.data + map.size);
      auto &part = bytes[i / (count / 2)];
      part.insert (part.end (), map.data, map.data + map.size);
      gst_memory_unmap (tensor, &map);
      gst_memory_unref (tensor);

      GstElement *sink = gst_element_factory_make ("appsink", nullptr);
      ASSERT_NE (sink, nullptr);
      g_object_set (sink, "sync", FALSE, "async", FALSE, "wait-on-eos", FALSE, nullptr);
      gst_bin_add (GST_BIN (pipeline), sink);
      outputs.push_back (sink);
    }
    g_signal_connect (demux, "pad-added", G_CALLBACK (link_output), this);
    if (corrupt) {
      GstPad *srcpad = gst_element_get_static_pad (mux, "src");
      gst_pad_add_probe (srcpad, GST_PAD_PROBE_TYPE_BUFFER, damage_extra, this, nullptr);
      gst_object_unref (srcpad);
    }
    bus = gst_element_get_bus (pipeline);
    ASSERT_NE (gst_element_set_state (pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
    for (guint i = 0; i < inputs.size (); i++) {
      GstBuffer *buffer = gst_buffer_new_allocate (nullptr, bytes[i].size (), nullptr);
      ASSERT_EQ (gst_buffer_fill (buffer, 0, bytes[i].data (), bytes[i].size ()),
          bytes[i].size ());
      GST_BUFFER_PTS (buffer) = 123 * GST_MSECOND;
      ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (inputs[i]), buffer), GST_FLOW_OK);
    }
    for (auto *input : inputs)
      ASSERT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (input)), GST_FLOW_OK);
  }

  /** @brief Require one byte-identical header and payload on this output. */
  void check_output (guint index)
  {
    GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (outputs[index]), 0);
    ASSERT_NE (sample, nullptr) << index;
    GstBuffer *buffer = gst_sample_get_buffer (sample);
    GstMapInfo map;
    EXPECT_EQ (GST_BUFFER_PTS (buffer), 123 * GST_MSECOND);
    EXPECT_EQ (gst_buffer_n_memory (buffer), 1U);
    EXPECT_STREQ (gst_structure_get_string (
                      gst_caps_get_structure (gst_sample_get_caps (sample), 0), "format"),
        "flexible");
    if (gst_buffer_map (buffer, &map, GST_MAP_READ)) {
      EXPECT_EQ (map.size, expected[index].size ()) << index;
      if (map.size == expected[index].size ()) {
        EXPECT_EQ (memcmp (map.data, expected[index].data (), map.size), 0) << index;
      }
      gst_buffer_unmap (buffer, &map);
    } else {
      ADD_FAILURE () << "Cannot read output " << index;
    }
    gst_sample_unref (sample);
    sample = gst_app_sink_try_pull_sample (GST_APP_SINK (outputs[index]), 0);
    EXPECT_EQ (sample, nullptr) << index;
    if (sample)
      gst_sample_unref (sample);
  }

  /** @brief Wait a bounded time for the pipeline's final EOS or error. */
  GstMessage *finish ()
  {
    return gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
        (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  }

  /** @brief Preserve earlier outputs, but refuse this and later extra tensors. */
  void check_invalid (guint index)
  {
    invalid_index = index;
    ASSERT_NO_FATAL_FAILURE (prepare (EXTRA_NUM_TENSORS, TRUE));
    GstMessage *message = finish ();
    ASSERT_NE (message, nullptr);
    EXPECT_EQ (GST_MESSAGE_TYPE (message), GST_MESSAGE_ERROR);
    gst_message_unref (message);
    EXPECT_EQ (g_atomic_int_get (&damaged), 1);
    for (guint i = 0; i < NNS_TENSOR_MEMORY_MAX + index; i++)
      check_output (i);
    for (guint i = NNS_TENSOR_MEMORY_MAX + index; i < outputs.size (); i++) {
      GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (outputs[i]), 0);
      EXPECT_EQ (sample, nullptr) << i;
      if (sample)
        gst_sample_unref (sample);
    }
  }
};

/** @brief Pin the same full-byte contract below the extra-header boundary. */
TEST_F (TensorMuxDemuxFlexible, regularMemories)
{
  ASSERT_NO_FATAL_FAILURE (prepare (NNS_TENSOR_MEMORY_MAX, FALSE));
  GstMessage *message = finish ();
  ASSERT_NE (message, nullptr);
  EXPECT_EQ (GST_MESSAGE_TYPE (message), GST_MESSAGE_EOS);
  gst_message_unref (message);
  for (guint i = 0; i < outputs.size (); i++)
    check_output (i);
}

/** @brief Keep all twenty flexible headers and payloads across mux and demux. */
TEST_F (TensorMuxDemuxFlexible, extraMemories)
{
  ASSERT_NO_FATAL_FAILURE (prepare (EXTRA_NUM_TENSORS, FALSE));
  GstMessage *message = finish ();
  ASSERT_NE (message, nullptr);
  EXPECT_EQ (GST_MESSAGE_TYPE (message), GST_MESSAGE_EOS);
  gst_message_unref (message);
  for (guint i = 0; i < outputs.size (); i++)
    check_output (i);
}

/** @brief Refuse a forged seventeenth size instead of delivering extra tensors. */
TEST_F (TensorMuxDemuxFlexible, forgedExtraSize_n)
{
  check_invalid (0);
}

/** @brief Also reject a forged final entry after preceding extras were sent. */
TEST_F (TensorMuxDemuxFlexible, forgedLastExtraSize_n)
{
  check_invalid (EXTRA_NUM_TENSORS - NNS_TENSOR_MEMORY_MAX - 1);
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
