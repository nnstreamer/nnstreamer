/**
 * @file        unittest_converter_config.cc
 * @date        17 Sep 2026
 * @brief       Unit test for the property and tensors-config ownership of tensor_converter
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 * @note        A leak on these paths does not change what the element does, so
 *              the cases here fail on it only under the memory checker CI runs
 *              every unit test binary with.
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_converter.h>
#include <nnstreamer_util.h>
#include <tensor_converter_custom.h>
#include <unittest_util.h>

/** Beyond NNS_TENSOR_MEMORY_MAX, so the last tensor lives in the extra memory */
#define EXTRA_TENSORS (NNS_TENSOR_MEMORY_MAX + 1U)
#define TENSOR_SIZE (4U)
#define FLEX_CAPS "other/tensors,format=flexible,framerate=(fraction)0/1"
#define OCTET_CAPS "application/octet-stream"

/**
 * @brief Join @a n copies of @a item with commas.
 */
static gchar *
repeat_str (const gchar *item, guint n)
{
  GString *str = g_string_new (NULL);
  guint i;

  for (i = 0; i < n; i++) {
    if (i > 0)
      g_string_append_c (str, ',');
    g_string_append (str, item);
  }

  return g_string_free (str, FALSE);
}

/**
 * @brief Get static caps of @a n uint8 tensors of the dimension @a dim.
 */
static gchar *
static_caps_str (guint n, const gchar *dim, const gchar *framerate)
{
  gchar *dims = repeat_str (dim, n);
  gchar *types = repeat_str ("uint8", n);
  gchar *caps = g_strdup_printf ("other/tensors,format=static,num_tensors=%u,"
                                 "dimensions=(string)\"%s\",types=(string)\"%s\",framerate=%s",
      n, dims, types, framerate);

  g_free (dims);
  g_free (types);
  return caps;
}

/**
 * @brief Set properties input-dim and input-type for @a n uint8 tensors of @a dim.
 */
static void
set_input_info (GstElement *element, guint n, const gchar *dim)
{
  gchar *dims = repeat_str (dim, n);
  gchar *types = repeat_str ("uint8", n);

  g_object_set (element, "input-dim", dims, "input-type", types, NULL);
  g_free (dims);
  g_free (types);
}

/**
 * @brief Get a flexible buffer of @a num uint8 tensors, tensor i filled with i.
 * @param last_size bytes the last tensor holds
 * @param last_declared bytes the meta header of the last tensor declares
 */
static GstBuffer *
flex_buffer_new (guint num, gsize last_size, gsize last_declared)
{
  GstBuffer *buf = gst_buffer_new ();
  GstTensorMetaInfo meta;
  GstTensorInfo info;
  GstMemory *mem;
  guint8 *data;
  gsize hsize, data_size;
  guint i;

  for (i = 0; i < num; i++) {
    gst_tensor_info_init (&info);
    info.type = _NNS_UINT8;
    info.dimension[0] = (i == num - 1) ? last_declared : TENSOR_SIZE;
    data_size = (i == num - 1) ? last_size : TENSOR_SIZE;

    gst_tensor_info_convert_to_meta (&info, &meta);
    hsize = gst_tensor_meta_info_get_header_size (&meta);

    data = (guint8 *) g_malloc0 (hsize + data_size);
    gst_tensor_meta_info_update_header (&meta, data);
    memset (data + hsize, i, data_size);

    mem = gst_memory_new_wrapped ((GstMemoryFlags) 0, data, hsize + data_size,
        0, hsize + data_size, data, g_free);

    /* an extra tensor is sized by its whole memory, meta header included */
    info.dimension[0] = hsize + data_size;
    gst_tensor_buffer_append_memory (buf, mem, &info);
  }

  return buf;
}

/**
 * @brief Check @a buf holds @a num tensors, tensor i is filled with i.
 */
static void
check_output (GstBuffer *buf, guint num, gsize last_size)
{
  GstMemory *mem;
  GstMapInfo map;
  gsize j;
  guint i;

  ASSERT_TRUE (buf != NULL);
  ASSERT_EQ (gst_tensor_buffer_get_count (buf), num);

  for (i = 0; i < num; i++) {
    mem = gst_tensor_buffer_get_nth_memory (buf, i);
    ASSERT_TRUE (mem != NULL);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));

    EXPECT_EQ (map.size, (i == num - 1) ? last_size : TENSOR_SIZE);
    for (j = 0; j < map.size; j++)
      EXPECT_EQ (map.data[j], i);

    gst_memory_unmap (mem, &map);
    gst_memory_unref (mem);
  }
}

/**
 * @brief Get the string field @a name of the current caps of the harness sink pad.
 */
static gchar *
get_output_caps_field (GstHarness *h, const gchar *name)
{
  GstCaps *caps = gst_pad_get_current_caps (h->sinkpad);
  gchar *value = NULL;

  if (caps) {
    GstStructure *st = gst_caps_get_structure (caps, 0);

    if (gst_structure_has_field (st, name))
      value = gst_value_serialize (gst_structure_get_value (st, name));
    gst_caps_unref (caps);
  }

  return value;
}

/**
 * @brief Custom converter splitting the input into single-byte uint8 tensors.
 */
static GstBuffer *
split_bytes_cb (GstBuffer *in_buf, void *data, GstTensorsConfig *config)
{
  guint *called = (guint *) data;
  gsize i, size = gst_buffer_get_size (in_buf);
  GstBuffer *out = gst_buffer_new ();
  GstMemory *all = gst_buffer_get_all_memory (in_buf);
  GstTensorInfo *info;

  config->rate_n = 0;
  config->rate_d = 1;
  config->info.num_tensors = size;

  for (i = 0; i < size; i++) {
    info = gst_tensors_info_get_nth_info (&config->info, i);
    info->type = _NNS_UINT8;
    info->dimension[0] = 1;
    gst_tensor_buffer_append_memory (out, gst_memory_share (all, i, 1), info);
  }

  gst_memory_unref (all);
  gst_buffer_copy_into (out, in_buf, GST_BUFFER_COPY_METADATA, 0, -1);
  (*called)++;

  return out;
}

/**
 * @brief External converter callback, splits the input into single-byte tensors.
 */
static GstBuffer *
ext_convert (GstBuffer *in_buf, GstTensorsConfig *config, void *priv_data)
{
  guint called = 0;

  UNUSED (priv_data);
  return split_bytes_cb (in_buf, &called, config);
}

/**
 * @brief External converter callback, initializes the config as in-tree converters do.
 */
static gboolean
ext_get_out_config (const GstCaps *in_caps, GstTensorsConfig *config)
{
  UNUSED (in_caps);
  gst_tensors_config_init (config);
  config->rate_n = 0;
  config->rate_d = 1;
  config->info.num_tensors = 1;
  config->info.info[0].type = _NNS_UINT8;
  config->info.info[0].dimension[0] = 1;
  return TRUE;
}

/**
 * @brief External converter callback, claims the media type custom-script mode looks up.
 */
static GstCaps *
ext_query_caps (const GstTensorsConfig *config)
{
  UNUSED (config);
  return gst_caps_new_empty_simple ("python3");
}

/**
 * @brief Push an octet buffer of @a size bytes, byte i filled with i.
 */
static GstFlowReturn
push_octet (GstHarness *h, gsize size)
{
  GstBuffer *buf = gst_buffer_new_allocate (NULL, size, NULL);
  GstMapInfo map;
  gsize i;

  if (!gst_buffer_map (buf, &map, GST_MAP_WRITE)) {
    gst_buffer_unref (buf);
    return GST_FLOW_ERROR;
  }

  for (i = 0; i < size; i++)
    map.data[i] = i;
  gst_buffer_unmap (buf, &map);

  return gst_harness_push (h, buf);
}

/**
 * @brief Setting mode again replaces the option it kept.
 */
TEST (tensorConverterConfig, modeSetTwice)
{
  GstElement *converter = gst_element_factory_make ("tensor_converter", NULL);
  gchar *mode = NULL;

  ASSERT_TRUE (converter != NULL);

  g_object_set (converter, "mode", "custom-script:first.py", NULL);
  g_object_set (converter, "mode", "custom-script:second.py", NULL);
  g_object_get (converter, "mode", &mode, NULL);
  EXPECT_STREQ (mode, "custom-script:second.py");
  g_free (mode);

  gst_object_unref (converter);
}

/**
 * @brief An unregistered custom-code name is kept and reported.
 */
TEST (tensorConverterConfig, modeUnregisteredCustomCode_n)
{
  GstElement *converter = gst_element_factory_make ("tensor_converter", NULL);
  gchar *mode = NULL;

  ASSERT_TRUE (converter != NULL);

  g_object_set (converter, "mode", "custom-code:c8c9_not_registered", NULL);
  g_object_set (converter, "mode", "custom-code:c8c9_not_registered", NULL);
  g_object_get (converter, "mode", &mode, NULL);
  EXPECT_STREQ (mode, "custom-code:c8c9_not_registered");
  g_free (mode);

  gst_object_unref (converter);
}

/**
 * @brief A mode without an option is refused and keeps the previous one.
 */
TEST (tensorConverterConfig, modeWithoutOption_n)
{
  GstElement *converter = gst_element_factory_make ("tensor_converter", NULL);
  gchar *mode = NULL;

  ASSERT_TRUE (converter != NULL);

  g_object_set (converter, "mode", "custom-script:first.py", NULL);
  g_object_set (converter, "mode", "custom-script", NULL);
  g_object_get (converter, "mode", &mode, NULL);
  EXPECT_STREQ (mode, "custom-script:first.py");
  g_free (mode);

  gst_object_unref (converter);
}

/**
 * @brief A registered custom-code set after an unregistered one converts.
 */
TEST (tensorConverterConfig, modeRegisteredAfterUnregistered)
{
  GstHarness *h;
  GstBuffer *out;
  guint called = 0;

  ASSERT_EQ (0, nnstreamer_converter_custom_register (
                    "c8c9_split_mode", split_bytes_cb, &called));

  h = gst_harness_new ("tensor_converter");
  g_object_set (h->element, "mode", "custom-code:c8c9_not_registered", NULL);
  g_object_set (h->element, "mode", "custom-code:c8c9_split_mode", NULL);
  gst_harness_set_src_caps_str (h, OCTET_CAPS);

  EXPECT_EQ (push_octet (h, 3U), GST_FLOW_OK);
  EXPECT_EQ (called, 1U);

  out = gst_harness_try_pull (h);
  EXPECT_EQ (gst_tensor_buffer_get_count (out), 3U);
  gst_buffer_unref (out);

  gst_harness_teardown (h);
  EXPECT_EQ (0, nnstreamer_converter_custom_unregister ("c8c9_split_mode"));
}

/**
 * @brief Flexible to static conversion of more tensors than a buffer has memories.
 */
TEST (tensorConverterConfig, flexToStaticExtraTensors)
{
  GstHarness *h = gst_harness_new ("tensor_converter");
  GstBuffer *out;
  gchar *num;
  guint i;

  gst_harness_set_src_caps_str (h, FLEX_CAPS);

  for (i = 0; i < 3U; i++) {
    EXPECT_EQ (gst_harness_push (h, flex_buffer_new (EXTRA_TENSORS, TENSOR_SIZE, TENSOR_SIZE)),
        GST_FLOW_OK);

    out = gst_harness_try_pull (h);
    check_output (out, EXTRA_TENSORS, TENSOR_SIZE);
    gst_buffer_unref (out);
  }

  num = get_output_caps_field (h, "num_tensors");
  EXPECT_STREQ (num, "17");
  g_free (num);

  gst_harness_teardown (h);
}

/**
 * @brief A changed extra tensor reconfigures the static output.
 */
TEST (tensorConverterConfig, flexToStaticExtraTensorsReconfigure)
{
  GstHarness *h = gst_harness_new ("tensor_converter");
  GstBuffer *out;
  gchar *dims, *expected;

  gst_harness_set_src_caps_str (h, FLEX_CAPS);

  EXPECT_EQ (gst_harness_push (h, flex_buffer_new (EXTRA_TENSORS, TENSOR_SIZE, TENSOR_SIZE)),
      GST_FLOW_OK);
  out = gst_harness_try_pull (h);
  check_output (out, EXTRA_TENSORS, TENSOR_SIZE);
  gst_buffer_unref (out);

  EXPECT_EQ (gst_harness_push (h, flex_buffer_new (EXTRA_TENSORS, 8U, 8U)), GST_FLOW_OK);
  out = gst_harness_try_pull (h);
  check_output (out, EXTRA_TENSORS, 8U);
  gst_buffer_unref (out);

  dims = get_output_caps_field (h, "dimensions");
  ASSERT_TRUE (dims != NULL);
  expected = g_strrstr (dims, ",");
  ASSERT_TRUE (expected != NULL);
  EXPECT_TRUE (g_str_has_prefix (expected, ",8"));
  g_free (dims);

  gst_harness_teardown (h);
}

/**
 * @brief An extra tensor holding other than what its header declares is refused.
 */
TEST (tensorConverterConfig, flexToStaticExtraTensorsSize_n)
{
  GstHarness *h = gst_harness_new ("tensor_converter");

  gst_harness_set_src_caps_str (h, FLEX_CAPS);

  EXPECT_NE (gst_harness_push (h, flex_buffer_new (EXTRA_TENSORS, 8U, TENSOR_SIZE)), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Extra tensors other than the given properties are refused.
 */
TEST (tensorConverterConfig, flexToStaticExtraTensorsProperty_n)
{
  GstHarness *h = gst_harness_new ("tensor_converter");
  gchar *dim = g_strdup_printf ("%u", TENSOR_SIZE);

  set_input_info (h->element, EXTRA_TENSORS, dim);
  gst_harness_set_src_caps_str (h, FLEX_CAPS);

  EXPECT_NE (gst_harness_push (h, flex_buffer_new (EXTRA_TENSORS, 8U, 8U)), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  g_free (dim);
  gst_harness_teardown (h);
}

/**
 * @brief Octet stream configured by downstream caps of extra tensors.
 */
TEST (tensorConverterConfig, octetPeerExtraTensors)
{
  GstHarness *h = gst_harness_new ("tensor_converter");
  gchar *caps = static_caps_str (EXTRA_TENSORS, "1", "(fraction)0/1");
  GstBuffer *out;
  GstMemory *mem;
  GstMapInfo map;
  guint i;

  gst_harness_set_sink_caps_str (h, caps);
  gst_harness_set_src_caps_str (h, OCTET_CAPS);

  EXPECT_EQ (push_octet (h, EXTRA_TENSORS), GST_FLOW_OK);
  out = gst_harness_try_pull (h);
  ASSERT_TRUE (out != NULL);
  ASSERT_EQ (gst_tensor_buffer_get_count (out), EXTRA_TENSORS);

  for (i = 0; i < EXTRA_TENSORS; i++) {
    mem = gst_tensor_buffer_get_nth_memory (out, i);
    ASSERT_TRUE (mem != NULL);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
    EXPECT_EQ (map.size, 1U);
    EXPECT_EQ (map.data[0], i);
    gst_memory_unmap (mem, &map);
    gst_memory_unref (mem);
  }

  gst_buffer_unref (out);
  g_free (caps);
  gst_harness_teardown (h);
}

/**
 * @brief Downstream caps of extra tensors are refused with multiple frames.
 */
TEST (tensorConverterConfig, octetPeerExtraTensorsFrames_n)
{
  GstHarness *h = gst_harness_new ("tensor_converter");
  gchar *caps = static_caps_str (EXTRA_TENSORS, "1", "(fraction)0/1");

  g_object_set (h->element, "frames-per-tensor", 2U, NULL);
  gst_harness_set_sink_caps_str (h, caps);
  gst_harness_set_src_caps_str (h, OCTET_CAPS);

  EXPECT_NE (push_octet (h, EXTRA_TENSORS * 2U), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  g_free (caps);
  gst_harness_teardown (h);
}

/**
 * @brief Media caps proposed from downstream caps of extra tensors.
 */
TEST (tensorConverterConfig, mediaCapsPeerExtraTensors)
{
  GstHarness *h = gst_harness_new ("tensor_converter");
  gchar *caps_str = static_caps_str (EXTRA_TENSORS, "1:4:2", "(fraction)0/1");
  GstCaps *caps;
  GstStructure *st;
  gboolean found;
  gint width, height;
  guint i, j;

  gst_harness_set_sink_caps_str (h, caps_str);

  for (i = 0; i < 3U; i++) {
    caps = gst_pad_query_caps (GST_PAD_PEER (h->srcpad), NULL);
    ASSERT_TRUE (caps != NULL);

    found = FALSE;
    for (j = 0; j < gst_caps_get_size (caps); j++) {
      st = gst_caps_get_structure (caps, j);
      if (gst_structure_has_name (st, "video/x-raw")
          && gst_structure_get_int (st, "width", &width)
          && gst_structure_get_int (st, "height", &height)) {
        EXPECT_EQ (width, 4);
        EXPECT_EQ (height, 2);
        found = TRUE;
      }
    }
    EXPECT_TRUE (found);
    gst_caps_unref (caps);
  }

  g_free (caps_str);
  gst_harness_teardown (h);
}

/**
 * @brief Custom code with unfixed downstream caps of extra tensors.
 */
TEST (tensorConverterConfig, customCodeUnfixedPeerExtraTensors)
{
  GstHarness *h;
  gchar *caps = static_caps_str (EXTRA_TENSORS, "1", "(fraction)[0/1,100/1]");
  GstBuffer *out;
  guint called = 0;

  ASSERT_EQ (0, nnstreamer_converter_custom_register (
                    "c8c9_split_unfixed", split_bytes_cb, &called));

  h = gst_harness_new ("tensor_converter");
  g_object_set (h->element, "mode", "custom-code:c8c9_split_unfixed", NULL);
  gst_harness_set_sink_caps_str (h, caps);
  gst_harness_set_src_caps_str (h, OCTET_CAPS);

  EXPECT_EQ (push_octet (h, EXTRA_TENSORS), GST_FLOW_OK);
  EXPECT_EQ (called, 1U);

  out = gst_harness_try_pull (h);
  ASSERT_TRUE (out != NULL);
  EXPECT_EQ (gst_tensor_buffer_get_count (out), EXTRA_TENSORS);
  gst_buffer_unref (out);

  g_free (caps);
  gst_harness_teardown (h);
  EXPECT_EQ (0, nnstreamer_converter_custom_unregister ("c8c9_split_unfixed"));
}

/**
 * @brief External converter with unfixed downstream caps of extra tensors.
 */
TEST (tensorConverterConfig, externalUnfixedPeerExtraTensors)
{
  static NNStreamerExternalConverter ext = {};
  GstHarness *h;
  gchar *caps = static_caps_str (EXTRA_TENSORS, "1", "(fraction)[0/1,100/1]");
  GstBuffer *out;

  ext.name = "c8c9_ext";
  ext.convert = ext_convert;
  ext.get_out_config = ext_get_out_config;
  ext.query_caps = ext_query_caps;
  ASSERT_TRUE (registerExternalConverter (&ext));

  h = gst_harness_new ("tensor_converter");
  g_object_set (h->element, "mode", "custom-script:c8c9.py", NULL);
  gst_harness_set_sink_caps_str (h, caps);
  gst_harness_set_src_caps_str (h, OCTET_CAPS);

  EXPECT_EQ (push_octet (h, EXTRA_TENSORS), GST_FLOW_OK);
  out = gst_harness_try_pull (h);
  ASSERT_TRUE (out != NULL);
  EXPECT_EQ (gst_tensor_buffer_get_count (out), EXTRA_TENSORS);
  gst_buffer_unref (out);

  g_free (caps);
  gst_harness_teardown (h);
  unregisterExternalConverter ("c8c9_ext");
}

/**
 * @brief Downstream caps of extra tensors other than the given properties are refused.
 */
TEST (tensorConverterConfig, customCodeFixedPeerMismatch_n)
{
  GstHarness *h;
  gchar *caps = static_caps_str (EXTRA_TENSORS, "1", "(fraction)0/1");
  guint called = 0;

  ASSERT_EQ (0, nnstreamer_converter_custom_register (
                    "c8c9_split_mismatch", split_bytes_cb, &called));

  h = gst_harness_new ("tensor_converter");
  g_object_set (h->element, "mode", "custom-code:c8c9_split_mismatch", NULL);
  set_input_info (h->element, 1U, "2");
  gst_harness_set_sink_caps_str (h, caps);
  gst_harness_set_src_caps_str (h, OCTET_CAPS);

  EXPECT_NE (push_octet (h, 2U), GST_FLOW_OK);
  EXPECT_EQ (called, 0U);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  g_free (caps);
  gst_harness_teardown (h);
  EXPECT_EQ (0, nnstreamer_converter_custom_unregister ("c8c9_split_mismatch"));
}

/**
 * @brief New input caps replace the configuration of extra tensors.
 */
TEST (tensorConverterConfig, renegotiateExtraTensors)
{
  GstHarness *h = gst_harness_new ("tensor_converter");
  GstBuffer *out;
  gchar *rate;

  set_input_info (h->element, EXTRA_TENSORS, "1");

  gst_harness_set_src_caps_str (h, OCTET_CAPS ",framerate=(fraction)10/1");
  EXPECT_EQ (push_octet (h, EXTRA_TENSORS), GST_FLOW_OK);
  out = gst_harness_try_pull (h);
  ASSERT_TRUE (out != NULL);
  EXPECT_EQ (gst_tensor_buffer_get_count (out), EXTRA_TENSORS);
  gst_buffer_unref (out);

  gst_harness_set_src_caps_str (h, OCTET_CAPS ",framerate=(fraction)20/1");
  EXPECT_EQ (push_octet (h, EXTRA_TENSORS), GST_FLOW_OK);
  out = gst_harness_try_pull (h);
  ASSERT_TRUE (out != NULL);
  EXPECT_EQ (gst_tensor_buffer_get_count (out), EXTRA_TENSORS);
  gst_buffer_unref (out);

  rate = get_output_caps_field (h, "framerate");
  EXPECT_STREQ (rate, "20/1");
  g_free (rate);

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
