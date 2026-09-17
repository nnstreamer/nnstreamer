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
#include <string.h>
#include <tensor_common.h>

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
