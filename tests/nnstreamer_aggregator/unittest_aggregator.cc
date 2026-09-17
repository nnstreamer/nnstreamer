/**
 * @file        unittest_aggregator.cc
 * @date        17 Sep 2026
 * @brief       Unit test for tensor_aggregator buffer ownership
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <nnstreamer_util.h>
#include <string.h>
#include <tensor_common.h>

/** @brief Number of int32 elements in a 3:4:2:2 tensor */
#define AGGR_NUM_ELEMENTS (48U)

/**
 * @brief Test input: two frames of 3:4:1:2 along frames-dim 2, one 3:4:2:2 tensor.
 */
static const gint aggr_input[AGGR_NUM_ELEMENTS]
    = { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112,
        1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124,
        2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112,
        2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124 };

/**
 * @brief The input above concatenated along frames-dim 2.
 */
static const gint aggr_concat[AGGR_NUM_ELEMENTS]
    = { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112,
        2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112,
        1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124,
        2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124 };

/**
 * @brief Create a harness for tensor_aggregator with a 3:4:2:2 int32 input.
 */
static GstHarness *
_aggr_harness_new (guint frames_in, guint frames_out, gboolean concat)
{
  GstHarness *h;
  GstTensorsConfig config;

  h = gst_harness_new ("tensor_aggregator");
  g_object_set (h->element, "frames-in", frames_in, "frames-out", frames_out,
      "frames-dim", 2, "concat", concat, NULL);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  gst_tensors_config_free (&config);

  return h;
}

/**
 * @brief Create a buffer holding the first @a num elements of the test input.
 */
static GstBuffer *
_aggr_buffer_new (GstHarness *h, guint num)
{
  GstBuffer *buf;

  buf = gst_harness_create_buffer (h, sizeof (gint) * num);
  gst_buffer_fill (buf, 0, aggr_input, sizeof (gint) * num);

  return buf;
}

/**
 * @brief Weak-reference callback marking a buffer or a memory as released.
 */
static void
_aggr_released (gpointer data, GstMiniObject *)
{
  *((gboolean *) data) = TRUE;
}

/**
 * @brief Check that a buffer holds exactly the expected int32 elements.
 */
static void
_aggr_check_data (GstBuffer *buf, const gint *expected)
{
  GstMapInfo map;

  ASSERT_TRUE (gst_buffer_map (buf, &map, GST_MAP_READ));
  EXPECT_EQ (map.size, sizeof (gint) * AGGR_NUM_ELEMENTS);
  if (map.size == sizeof (gint) * AGGR_NUM_ELEMENTS) {
    EXPECT_EQ (memcmp (map.data, expected, map.size), 0);
  }
  gst_buffer_unmap (buf, &map);
}

/**
 * @brief Concatenating a buffer the element owns alone happens in place.
 */
TEST (testTensorAggregatorOwnership, concatWritableInput)
{
  GstHarness *h;
  GstBuffer *in, *out;
  gboolean released = FALSE;

  h = _aggr_harness_new (2, 2, TRUE);

  in = _aggr_buffer_new (h, AGGR_NUM_ELEMENTS);
  gst_mini_object_weak_ref (GST_MINI_OBJECT_CAST (in), _aggr_released, &released);

  EXPECT_EQ (gst_harness_push (h, in), GST_FLOW_OK);
  ASSERT_EQ (gst_harness_buffers_received (h), 1U);

  out = gst_harness_pull (h);
  /* The element owns the buffer alone, so it must concatenate it without a copy. */
  EXPECT_FALSE (released);
  EXPECT_EQ (out, in);
  _aggr_check_data (out, aggr_concat);
  gst_buffer_unref (out);

  gst_harness_teardown (h);
}

/**
 * @brief Concatenating a shared buffer pushes a new buffer and leaves the shared one untouched.
 */
TEST (testTensorAggregatorOwnership, concatSharedInput)
{
  GstHarness *h;
  GstBuffer *in, *out;

  h = _aggr_harness_new (2, 2, TRUE);

  in = _aggr_buffer_new (h, AGGR_NUM_ELEMENTS);
  gst_buffer_ref (in);

  EXPECT_EQ (gst_harness_push (h, in), GST_FLOW_OK);
  ASSERT_EQ (gst_harness_buffers_received (h), 1U);

  out = gst_harness_pull (h);
  /* If the element pushed the input, it gave away the reference held here. */
  ASSERT_NE (out, in);
  _aggr_check_data (out, aggr_concat);
  gst_buffer_unref (out);

  EXPECT_EQ (GST_MINI_OBJECT_REFCOUNT_VALUE (in), 1);
  _aggr_check_data (in, aggr_input);
  gst_buffer_unref (in);

  gst_harness_teardown (h);
}

/**
 * @brief Without concatenation a shared buffer passes through as it is.
 */
TEST (testTensorAggregatorOwnership, noConcatSharedInput)
{
  GstHarness *h;
  GstBuffer *in, *out;

  h = _aggr_harness_new (2, 2, FALSE);

  in = _aggr_buffer_new (h, AGGR_NUM_ELEMENTS);
  gst_buffer_ref (in);

  EXPECT_EQ (gst_harness_push (h, in), GST_FLOW_OK);
  ASSERT_EQ (gst_harness_buffers_received (h), 1U);

  out = gst_harness_pull (h);
  EXPECT_EQ (out, in);
  EXPECT_EQ (GST_MINI_OBJECT_REFCOUNT_VALUE (in), 2);
  _aggr_check_data (out, aggr_input);
  gst_buffer_unref (out);
  gst_buffer_unref (in);

  gst_harness_teardown (h);
}

#ifdef __TIZEN__
/**
 * @brief A buffer of a wrong size is released when the element refuses it.
 * @note ml_logf aborts on Linux distros, so the refusal returns only on Tizen.
 */
TEST (testTensorAggregatorOwnership, invalidFrameSize_n)
{
  GstHarness *h;
  GstBuffer *in;

  h = _aggr_harness_new (2, 2, TRUE);

  in = _aggr_buffer_new (h, AGGR_NUM_ELEMENTS - 2);
  gst_buffer_ref (in);

  EXPECT_EQ (gst_harness_push (h, in), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  EXPECT_EQ (GST_MINI_OBJECT_REFCOUNT_VALUE (in), 1);
  gst_buffer_unref (in);

  gst_harness_teardown (h);
}

/**
 * @brief A frame taken from the adapter is released when the element refuses it.
 * @note ml_logf aborts on Linux distros, so the refusal returns only on Tizen.
 */
TEST (testTensorAggregatorOwnership, invalidFrameSizeAdapter_n)
{
  GstHarness *h;
  GstBuffer *in;
  gboolean released = FALSE;

  h = _aggr_harness_new (2, 1, TRUE);

  in = _aggr_buffer_new (h, AGGR_NUM_ELEMENTS - 2);
  gst_mini_object_weak_ref (GST_MINI_OBJECT_CAST (gst_buffer_peek_memory (in, 0)),
      _aggr_released, &released);

  EXPECT_EQ (gst_harness_push (h, in), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  /* The adapter drops what is left on teardown; the refused frame must not outlive it. */
  gst_harness_teardown (h);
  EXPECT_TRUE (released);
}

/**
 * @brief Memory of the allocator below.
 */
typedef struct {
  GstMemory mem;
  gpointer data;
} AggrReadOnlyMemory;

/**
 * @brief Allocator whose memory maps for reading only (or not at all), to fail concatenation.
 */
typedef struct {
  GstAllocator parent;
  gboolean refuse_read; /**< refuse read maps too */
} AggrReadOnlyAllocator;

/**
 * @brief Class of AggrReadOnlyAllocator.
 */
typedef struct {
  GstAllocatorClass parent_class;
} AggrReadOnlyAllocatorClass;

G_DEFINE_TYPE (AggrReadOnlyAllocator, aggr_read_only_allocator, GST_TYPE_ALLOCATOR);

/**
 * @brief Create a memory of AggrReadOnlyAllocator.
 */
static GstMemory *
_aggr_ro_memory_new (GstAllocator *allocator, GstMemory *parent, gpointer data,
    gsize maxsize, gsize offset, gsize size)
{
  AggrReadOnlyMemory *mem = g_new0 (AggrReadOnlyMemory, 1);

  gst_memory_init (GST_MEMORY_CAST (mem), (GstMemoryFlags) 0, allocator, parent,
      maxsize, 0, offset, size);
  mem->data = data;

  return GST_MEMORY_CAST (mem);
}

/**
 * @brief Map a memory of AggrReadOnlyAllocator; writing (and reading, if set) is refused.
 */
static gpointer
_aggr_ro_memory_map (GstMemory *mem, gsize, GstMapFlags flags)
{
  if ((flags & GST_MAP_WRITE) || ((AggrReadOnlyAllocator *) mem->allocator)->refuse_read)
    return NULL;

  return ((AggrReadOnlyMemory *) mem)->data;
}

/**
 * @brief Unmap a memory of AggrReadOnlyAllocator.
 */
static void
_aggr_ro_memory_unmap (GstMemory *)
{
}

/**
 * @brief Copy a memory of AggrReadOnlyAllocator into the same allocator.
 */
static GstMemory *
_aggr_ro_memory_copy (GstMemory *mem, gssize offset, gssize size)
{
  gsize len = (size == -1) ? mem->size - offset : (gsize) size;
  gpointer data = _g_memdup (
      (guint8 *) ((AggrReadOnlyMemory *) mem)->data + mem->offset + offset, len);

  return _aggr_ro_memory_new (mem->allocator, NULL, data, len, 0, len);
}

/**
 * @brief Share a memory of AggrReadOnlyAllocator.
 */
static GstMemory *
_aggr_ro_memory_share (GstMemory *mem, gssize offset, gssize size)
{
  GstMemory *parent = mem->parent ? mem->parent : mem;
  gsize len = (size == -1) ? mem->size - offset : (gsize) size;

  return _aggr_ro_memory_new (mem->allocator, parent,
      ((AggrReadOnlyMemory *) mem)->data, mem->maxsize, mem->offset + offset, len);
}

/**
 * @brief Free a memory of AggrReadOnlyAllocator.
 */
static void
aggr_read_only_allocator_free (GstAllocator *, GstMemory *mem)
{
  if (mem->parent == NULL)
    g_free (((AggrReadOnlyMemory *) mem)->data);
  g_free (mem);
}

/**
 * @brief Initialize the class of AggrReadOnlyAllocator.
 */
static void
aggr_read_only_allocator_class_init (AggrReadOnlyAllocatorClass *klass)
{
  GST_ALLOCATOR_CLASS (klass)->free = aggr_read_only_allocator_free;
}

/**
 * @brief Initialize an AggrReadOnlyAllocator.
 */
static void
aggr_read_only_allocator_init (AggrReadOnlyAllocator *self)
{
  GstAllocator *allocator = GST_ALLOCATOR_CAST (self);

  allocator->mem_type = "AggrReadOnly";
  allocator->mem_map = _aggr_ro_memory_map;
  allocator->mem_unmap = _aggr_ro_memory_unmap;
  allocator->mem_copy = _aggr_ro_memory_copy;
  allocator->mem_share = _aggr_ro_memory_share;
  GST_OBJECT_FLAG_SET (allocator, GST_ALLOCATOR_FLAG_CUSTOM_ALLOC);
}

/**
 * @brief Push a buffer that cannot be concatenated and check it is released.
 */
static void
_aggr_test_concat_map_failure (gboolean refuse_read)
{
  GstHarness *h;
  GstAllocator *allocator;
  GstMemory *mem;
  GstBuffer *in;
  gsize size = sizeof (gint) * AGGR_NUM_ELEMENTS;
  gboolean released = FALSE;

  h = _aggr_harness_new (2, 2, TRUE);

  allocator = GST_ALLOCATOR_CAST (
      g_object_new (aggr_read_only_allocator_get_type (), NULL));
  ((AggrReadOnlyAllocator *) allocator)->refuse_read = refuse_read;
  mem = _aggr_ro_memory_new (allocator, NULL, _g_memdup (aggr_input, size), size, 0, size);
  gst_mini_object_weak_ref (GST_MINI_OBJECT_CAST (mem), _aggr_released, &released);
  in = gst_buffer_new ();
  gst_buffer_append_memory (in, mem);

  EXPECT_EQ (gst_harness_push (h, in), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  /* Nothing else holds the buffer, so its memory must be gone once the push returns. */
  EXPECT_TRUE (released);

  gst_harness_teardown (h);
  gst_object_unref (allocator);
}

/**
 * @brief A buffer that cannot be mapped for writing is released when the element refuses it.
 * @note ml_logf aborts on Linux distros, so the refusal returns only on Tizen.
 */
TEST (testTensorAggregatorOwnership, concatMapFailure_n)
{
  _aggr_test_concat_map_failure (FALSE);
}

/**
 * @brief A buffer that cannot be mapped at all is released when the element refuses it.
 * @note ml_logf aborts on Linux distros, so the refusal returns only on Tizen.
 */
TEST (testTensorAggregatorOwnership, concatSourceMapFailure_n)
{
  _aggr_test_concat_map_failure (TRUE);
}
#endif /* __TIZEN__ */

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
