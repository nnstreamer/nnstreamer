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
#include <tensor_converter_custom.h>
#include <unittest_util.h>

#define OCTET_CAPS "application/octet-stream"

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
