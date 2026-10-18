/**
 * @file	unittest_sparse.cc
 * @date	17 September 2026
 * @brief	Unit test for the size of the tensors tensor_sparse decodes.
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs.
 */

#include <gtest/gtest.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api.h>
#include <string.h>
#include <tensor_common.h>
#include <utility>

#include "../gst/nnstreamer/elements/gsttensor_sparsedec.h"
#include "../gst/nnstreamer/elements/gsttensor_sparseenc.h"
#include "../gst/nnstreamer/elements/gsttensor_sparseutil.h"
#include "../unittest_util.h"

/**
 * @brief Dimension of a tensor no process can allocate: 4 * 10^18 int32 bytes.
 */
#define SPARSE_HUGE_DIMENSION "1000000000:1000000000"

/**
 * @brief Caps of the dense tensor stream the decoder negotiates in these tests.
 */
#define SPARSE_DENSE_CAPS_STR                  \
  "other/tensors,format=static,num_tensors=1," \
  "dimensions=(string)40:1:1:1,types=(string)int32,framerate=0/1"

/**
 * @brief Build the memory of a sparse int32 tensor.
 * @param dimension the dimension the header declares
 * @param nnz the number of non-zero elements
 * @param values the non-zero values, nnz of them, or NULL when nnz is 0
 * @param indices the indices of the values, nnz of them, or NULL when nnz is 0
 * @return the memory holding the header, the values and the indices
 */
static GstMemory *
_sparse_new_memory (const gchar *dimension, guint nnz, const gint32 *values, const guint *indices)
{
  GstTensorMetaInfo meta;
  guint8 *data;
  gsize header_size, size;

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_INT32;
  gst_tensor_parse_dimension (dimension, meta.dimension);
  meta.format = _NNS_TENSOR_FORMAT_SPARSE;
  meta.media_type = _NNS_TENSOR;
  meta.sparse_info.nnz = nnz;

  header_size = gst_tensor_meta_info_get_header_size (&meta);
  size = header_size + (gsize) nnz * (sizeof (gint32) + sizeof (guint));
  data = (guint8 *) g_malloc0 (size);

  if (!gst_tensor_meta_info_update_header (&meta, data)) {
    g_free (data);
    return NULL;
  }

  if (nnz > 0) {
    memcpy (data + header_size, values, nnz * sizeof (gint32));
    memcpy (data + header_size + nnz * sizeof (gint32), indices, nnz * sizeof (guint));
  }

  return gst_memory_new_wrapped ((GstMemoryFlags) 0, data, size, 0, size, data, g_free);
}

/**
 * @brief Wrap the given memory in a new buffer.
 * @param mem the memory, whose reference the buffer takes
 * @return the buffer
 */
static GstBuffer *
_sparse_new_buffer (GstMemory *mem)
{
  GstBuffer *buf = gst_buffer_new ();

  gst_buffer_append_memory (buf, mem);
  return buf;
}

/**
 * @brief Test for tensor_sparse util, a header declaring a tensor too large to
 *        allocate.
 * @details The 128-byte memory declares 10^18 int32 elements and no non-zero
 * one. Every field is consistent, so the only thing that can refuse it is the
 * allocation. With g_malloc0() the process aborts inside GLib.
 */
TEST (testTensorSparseSize, utilToDenseHugeDimension_n)
{
  GstTensorMetaInfo meta;
  GstMemory *in, *out;

  in = _sparse_new_memory (SPARSE_HUGE_DIMENSION, 0U, NULL, NULL);
  ASSERT_TRUE (in != NULL);

  out = gst_tensor_sparse_to_dense (&meta, in);
  EXPECT_TRUE (out == NULL);
  if (out)
    gst_memory_unref (out);

  gst_memory_unref (in);
}

/**
 * @brief Test for tensor_sparse util, a large tensor that can be allocated is
 *        still decoded, zeroed except for its non-zero elements.
 */
TEST (testTensorSparseSize, utilToDenseLargeDimension)
{
  GstTensorMetaInfo meta;
  GstMemory *in, *out;
  GstMapInfo map;
  gint32 *dense;
  const gint32 values[] = { 7, -3 };
  const guint indices[] = { 0U, 999999U };

  in = _sparse_new_memory ("1000:1000", 2U, values, indices);
  ASSERT_TRUE (in != NULL);

  out = gst_tensor_sparse_to_dense (&meta, in);
  ASSERT_TRUE (out != NULL);

  EXPECT_EQ (meta.format, _NNS_TENSOR_FORMAT_STATIC);
  EXPECT_EQ (meta.dimension[0], 1000U);
  EXPECT_EQ (meta.dimension[1], 1000U);

  ASSERT_TRUE (gst_memory_map (out, &map, GST_MAP_READ));
  ASSERT_EQ (map.size, 1000000U * sizeof (gint32));
  dense = (gint32 *) map.data;
  EXPECT_EQ (dense[0], 7);
  EXPECT_EQ (dense[1], 0);
  EXPECT_EQ (dense[500000], 0);
  EXPECT_EQ (dense[999998], 0);
  EXPECT_EQ (dense[999999], -3);
  gst_memory_unmap (out, &map);

  gst_memory_unref (out);
  gst_memory_unref (in);
}

/**
 * @brief Test for tensor_sparse_dec, a buffer declaring a tensor too large to
 *        allocate is refused and the stream goes on.
 * @details The refusal is a flow error of the element, not an abort of the
 * process, so a valid buffer pushed afterwards is decoded as usual.
 */
TEST (testTensorSparseSize, decHugeDimension_n)
{
  GstHarness *h;
  GstBuffer *out;
  GstMapInfo map;
  const gint32 values[] = { 5 };
  const guint indices[] = { 39U };
  GstMemory *mem;

  h = gst_harness_new ("tensor_sparse_dec");
  ASSERT_TRUE (h != NULL);

  gst_harness_set_sink_caps_str (h, SPARSE_DENSE_CAPS_STR);
  gst_harness_set_src_caps_str (h, "other/tensors,format=sparse,framerate=0/1");

  mem = _sparse_new_memory (SPARSE_HUGE_DIMENSION, 0U, NULL, NULL);
  ASSERT_TRUE (mem != NULL);
  EXPECT_EQ (gst_harness_push (h, _sparse_new_buffer (mem)), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  mem = _sparse_new_memory ("40", 1U, values, indices);
  ASSERT_TRUE (mem != NULL);
  EXPECT_EQ (gst_harness_push (h, _sparse_new_buffer (mem)), GST_FLOW_OK);
  ASSERT_EQ (gst_harness_buffers_received (h), 1U);

  out = gst_harness_pull (h);
  ASSERT_TRUE (out != NULL);
  ASSERT_EQ (gst_buffer_n_memory (out), 1U);
  ASSERT_TRUE (gst_buffer_map (out, &map, GST_MAP_READ));
  ASSERT_EQ (map.size, 40U * sizeof (gint32));
  EXPECT_EQ (((gint32 *) map.data)[0], 0);
  EXPECT_EQ (((gint32 *) map.data)[39], 5);
  gst_buffer_unmap (out, &map);

  gst_buffer_unref (out);
  gst_harness_teardown (h);
}

/**
 * @brief Number of tensors a stream carries to reach GstTensorsInfo::extra.
 */
#define EXTRA_NUM_TENSORS ((guint) (NNS_TENSOR_MEMORY_MAX + 4))

/**
 * @brief Renegotiate a stream of more tensors than NNS_TENSOR_MEMORY_MAX, which
 *        reparses the input configuration of tensor_sparse_enc.
 */
TEST (testTensorSparseExtra, encCapsRenegotiation)
{
  GstHarness *h = gst_harness_new ("tensor_sparse_enc");
  GstTensorSparseEnc *self;
  GstCaps *caps;

  ASSERT_NE (h, nullptr);
  self = GST_TENSOR_SPARSE_ENC (h->element);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  gst_harness_set_src_caps (h, gst_caps_copy (caps));
  EXPECT_EQ (self->in_config.info.num_tensors, EXTRA_NUM_TENSORS);
  EXPECT_NE (self->in_config.info.extra, nullptr);

  gst_harness_set_src_caps (h, caps);
  EXPECT_EQ (self->in_config.info.num_tensors, EXTRA_NUM_TENSORS);

  gst_harness_teardown (h);
}

/**
 * @brief Renegotiate the sparse input of tensor_sparse_dec while its peer
 *        carries more tensors than NNS_TENSOR_MEMORY_MAX.
 */
TEST (testTensorSparseExtra, decCapsRenegotiation)
{
  GstHarness *h = gst_harness_new ("tensor_sparse_dec");
  GstTensorSparseDec *self;
  GstCaps *caps;

  ASSERT_NE (h, nullptr);
  self = GST_TENSOR_SPARSE_DEC (h->element);

  gst_harness_set_sink_caps (h, caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS));

  caps = gst_caps_from_string ("other/tensors,format=sparse,framerate=0/1");
  gst_harness_set_src_caps (h, gst_caps_copy (caps));
  EXPECT_EQ (self->out_config.info.num_tensors, EXTRA_NUM_TENSORS);
  EXPECT_NE (self->out_config.info.extra, nullptr);

  gst_harness_set_src_caps (h, caps);
  EXPECT_EQ (self->out_config.info.num_tensors, EXTRA_NUM_TENSORS);

  gst_harness_teardown (h);
}

/**
 * @brief Negotiate a stream whose tensors are not all described.
 */
TEST (testTensorSparseExtra, encCapsRenegotiation_n)
{
  GstHarness *h = gst_harness_new ("tensor_sparse_enc");
  GstTensorSparseEnc *self;
  GstCaps *caps;
  GstPad *sinkpad;

  ASSERT_NE (h, nullptr);
  self = GST_TENSOR_SPARSE_ENC (h->element);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS - 1);

  sinkpad = gst_element_get_static_pad (h->element, "sink");
  ASSERT_NE (sinkpad, nullptr);
  EXPECT_FALSE (gst_pad_send_event (sinkpad, gst_event_new_caps (caps)));
  gst_caps_unref (caps);
  gst_object_unref (sinkpad);

  EXPECT_EQ (self->in_config.info.num_tensors, 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Describe count dense int32 tensors with the given number of elements.
 */
static GstCaps *
_sparse_caps_with_size (guint count, guint elements)
{
  GstTensorsConfig config;
  gst_tensors_config_init (&config);
  config.info.num_tensors = count;
  config.rate_n = 0;
  config.rate_d = 1;
  gchar *dimension = g_strdup_printf ("%u:1:1:1", elements);
  for (guint i = 0; i < count; i++) {
    GstTensorInfo *info = gst_tensors_info_get_nth_info (&config.info, i);
    info->type = _NNS_INT32;
    gst_tensor_parse_dimension (dimension, info->dimension);
  }
  GstCaps *caps = gst_tensors_caps_from_config (&config);
  g_free (dimension);
  gst_tensors_config_free (&config);
  return caps;
}

/**
 * @brief Build sparse inputs, optionally enlarging only the last dense tensor.
 */
static GstBuffer *
_sparse_buffer_with_size (guint count, guint elements, gboolean oversized_last)
{
  GstBuffer *buffer = gst_buffer_new ();
  for (guint i = 0; i < count; i++) {
    const guint size = elements + (oversized_last && i + 1 == count);
    const gint32 value = 7;
    const guint index = size - 1;
    gchar *dimension = g_strdup_printf ("%u", size);
    GstMemory *memory = _sparse_new_memory (dimension, 1, &value, &index);
    g_free (dimension);
    EXPECT_NE (memory, nullptr);
    EXPECT_TRUE (gst_tensor_buffer_append_memory (buffer, memory, NULL));
  }
  return buffer;
}

/**
 * @brief Check every decoded tensor, including tensors in the extra block.
 */
static void
_sparse_check_output_size (GstHarness *h, guint count, guint elements)
{
  EXPECT_EQ (gst_harness_buffers_in_queue (h), 1U);
  if (gst_harness_buffers_in_queue (h) == 0)
    return;
  GstBuffer *output = gst_harness_pull (h);
  EXPECT_EQ (gst_tensor_buffer_get_count (output), count);
  for (guint i = 0; i < count; i++) {
    GstMemory *memory = gst_tensor_buffer_get_nth_memory (output, i);
    ASSERT_NE (memory, nullptr);
    GstMapInfo map;
    if (gst_memory_map (memory, &map, GST_MAP_READ)) {
      EXPECT_EQ (map.size, elements * sizeof (gint32));
      if (map.size == elements * sizeof (gint32)) {
        EXPECT_EQ (((gint32 *) map.data)[0], 0);
        EXPECT_EQ (((gint32 *) map.data)[elements - 1], 7);
      }
      gst_memory_unmap (memory, &map);
    } else {
      ADD_FAILURE () << "Cannot map decoded tensor " << i;
    }
    gst_memory_unref (memory);
  }
  gst_buffer_unref (output);
}

/**
 * @brief Cover single, multiple and extra-tensor negotiated outputs.
 */
class TensorSparseLimitTest : public ::testing::TestWithParam<guint>
{
};

/**
 * @brief Refuse an oversized last tensor and recover on the same decoder.
 */
TEST_P (TensorSparseLimitTest, oversizedTensor_n)
{
  const guint count = GetParam ();
  GstHarness *h = gst_harness_new ("tensor_sparse_dec");
  ASSERT_NE (h, nullptr);
  gst_harness_set_sink_caps (h, _sparse_caps_with_size (count, 40));
  gst_harness_set_src_caps_str (h, "other/tensors,format=sparse,framerate=0/1");
  EXPECT_EQ (gst_harness_push (h, _sparse_buffer_with_size (count, 40, TRUE)), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  EXPECT_EQ (gst_harness_push (h, _sparse_buffer_with_size (count, 40, FALSE)), GST_FLOW_OK);
  _sparse_check_output_size (h, count, 40);
  gst_harness_teardown (h);
}

/**
 * @brief Valid output follows renegotiation rather than retaining a stale limit.
 */
TEST_P (TensorSparseLimitTest, renegotiatedSize)
{
  const guint count = GetParam ();
  GstHarness *h = gst_harness_new ("tensor_sparse_dec");
  ASSERT_NE (h, nullptr);
  for (const guint elements : { 40U, 80U }) {
    gst_harness_set_sink_caps (h, _sparse_caps_with_size (count, elements));
    gst_harness_set_src_caps_str (h, "other/tensors,format=sparse,framerate=0/1");
    EXPECT_EQ (gst_harness_push (h, _sparse_buffer_with_size (count, elements, FALSE)),
        GST_FLOW_OK);
    _sparse_check_output_size (h, count, elements);
  }
  gst_harness_teardown (h);
}

INSTANTIATE_TEST_SUITE_P (Negotiated, TensorSparseLimitTest,
    ::testing::Values (1U, 2U, EXTRA_NUM_TENSORS));

/**
 * @brief Exercise both missing and extra tensors against the negotiated count.
 */
class TensorSparseCountTest : public ::testing::TestWithParam<std::pair<guint, guint>>
{
};

/**
 * @brief A count mismatch is refused without decoding any of its tensors.
 */
TEST_P (TensorSparseCountTest, mismatchedCount_n)
{
  const guint expected = GetParam ().first;
  const guint actual = GetParam ().second;
  GstHarness *h = gst_harness_new ("tensor_sparse_dec");
  ASSERT_NE (h, nullptr);
  gst_harness_set_sink_caps (h, _sparse_caps_with_size (expected, 40));
  gst_harness_set_src_caps_str (h, "other/tensors,format=sparse,framerate=0/1");
  EXPECT_EQ (gst_harness_push (h, _sparse_buffer_with_size (actual, 40, FALSE)), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

INSTANTIATE_TEST_SUITE_P (Negotiated, TensorSparseCountTest,
    ::testing::Values (std::make_pair (1U, 2U), std::make_pair (2U, 1U),
        std::make_pair (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS - 1),
        std::make_pair (EXTRA_NUM_TENSORS - 1, EXTRA_NUM_TENSORS)));

/**
 * @brief Preserve the existing drop behavior for shapes within the size limit.
 */
class TensorSparseShapeTest : public ::testing::TestWithParam<const char *>
{
};

/**
 * @brief A smaller or equally sized incompatible shape is still dropped.
 */
TEST_P (TensorSparseShapeTest, incompatibleShape_n)
{
  GstHarness *h = gst_harness_new ("tensor_sparse_dec");
  ASSERT_NE (h, nullptr);
  gst_harness_set_sink_caps_str (h, SPARSE_DENSE_CAPS_STR);
  gst_harness_set_src_caps_str (h, "other/tensors,format=sparse,framerate=0/1");
  const gint32 value = 7;
  const guint index = 0;
  GstMemory *memory = _sparse_new_memory (GetParam (), 1, &value, &index);
  ASSERT_NE (memory, nullptr);
  EXPECT_EQ (gst_harness_push (h, _sparse_new_buffer (memory)), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

INSTANTIATE_TEST_SUITE_P (
    Negotiated, TensorSparseShapeTest, ::testing::Values ("20", "20:2"));

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
