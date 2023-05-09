/**
 * Copyright (C) 2021 Junhwan Kim <jejudo.kim@samsung.com>
 *
 * @file    unittest_filter_shared_model.cc
 * @date    19 Aug 2021
 * @brief   Unit test for nnstreamer filter shared model features.
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>
#include <unittest_util.h>

static const gchar model_name1[] = "mobilenet_v1_1.0_224_quant.tflite";
static const gchar model_name2[] = "mobilenet_v2_1.0_224_quant.tflite";
static const gchar data_name[] = "orange.png";
static const gchar shared_key[] = "mobilenet";
static guint res[2];

/**
 * @brief callback for tensor sink to get arg max
 */
static void
_new_data_cb (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  GstMemory *mem;
  GstMapInfo info;
  gsize i, max_i = 0;
  guint8 max_val = 0;
  gint idx = *(gint *) user_data;
  UNUSED (element);

  mem = gst_buffer_get_memory (buffer, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

  for (i = 0; i < info.size; ++i) {
    if (info.data[i] > max_val) {
      max_val = info.data[i];
      max_i = i;
    }
  }
  res[idx] = (guint) max_i;
  gst_memory_unmap (mem, &info);
  gst_memory_unref (mem);
}

/**
 * @brief helper to get base pipeline string
 */
static void
_get_pipeline_str (gchar **str, const gchar *model1, const gchar *model2)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
  gchar *model_path1
      = g_build_filename (root_path, "tests", "test_models", "models", model1, NULL);
  gchar *model_path2
      = g_build_filename (root_path, "tests", "test_models", "models", model2, NULL);
  gchar *image_path
      = g_build_filename (root_path, "tests", "test_models", "data", data_name, NULL);

  ASSERT_TRUE (g_file_test (model_path1, G_FILE_TEST_EXISTS));
  ASSERT_TRUE (g_file_test (model_path2, G_FILE_TEST_EXISTS));
  ASSERT_TRUE (g_file_test (image_path, G_FILE_TEST_EXISTS));

  *str = g_strdup_printf (
      "filesrc location=%s ! pngdec ! videoscale ! imagefreeze ! videoconvert ! "
      "video/x-raw,format=RGB,framerate=10/1 ! tensor_converter ! tee name=t t. ! "
      "queue ! tensor_filter name=filter1 framework=tensorflow-lite model=%s is-updatable=TRUE "
      "shared-tensor-filter-key=%s ! tensor_sink name=sink1 t. ! "
      "queue ! tensor_filter name=filter2 framework=tensorflow-lite model=%s is-updatable=TRUE "
      "shared-tensor-filter-key=%s ! tensor_sink name=sink2",
      image_path, model_path1, shared_key, model_path2, shared_key);
  g_free (root_path);
  g_free (model_path1);
  g_free (model_path2);
  g_free (image_path);
}

/**
 * @brief Test filters share key but have different model paths.
 */
TEST (nnstreamerFilterSharedModel, tfliteSharedModelNotEqual_n)
{
  gchar *pipeline_str;
  GstElement *pipeline;
  _get_pipeline_str (&pipeline_str, model_name1, model_name2);
  pipeline = gst_parse_launch (pipeline_str, NULL);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (TEST_DEFAULT_SLEEP_TIME);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_free (pipeline_str);
  gst_object_unref (pipeline);
}

/**
 * @brief Test filter has invalid shape for shared model
 */
TEST (nnstreamerFilterSharedModel, tfliteInvalidShape_n)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
  gchar *model_path1 = g_build_filename (
      root_path, "tests", "test_models", "models", model_name1, NULL);
  gchar *image_path
      = g_build_filename (root_path, "tests", "test_models", "data", data_name, NULL);
  gchar *pipeline_str;
  GstElement *pipeline;
  ASSERT_TRUE (g_file_test (model_path1, G_FILE_TEST_EXISTS));
  ASSERT_TRUE (g_file_test (image_path, G_FILE_TEST_EXISTS));

  pipeline_str = g_strdup_printf (
      "filesrc location=%s ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tee name=t t. ! "
      "queue ! videoscale ! videoconvert ! video/x-raw ! "
      "tensor_converter ! tensor_filter name=filter1 framework=tensorflow-lite model=%s is-updatable=TRUE "
      "shared-tensor-filter-key=%s ! tensor_sink name=sink1 t. ! "
      "queue ! videoscale ! videoconvert ! video/x-raw,width=30,height=30 ! "
      "tensor_converter ! tensor_filter name=filter2 framework=tensorflow-lite model=%s is-updatable=TRUE "
      "shared-tensor-filter-key=%s ! tensor_sink name=sink2",
      image_path, model_path1, shared_key, model_path1, shared_key);
  g_free (root_path);
  g_free (model_path1);
  g_free (image_path);

  pipeline = gst_parse_launch (pipeline_str, NULL);

  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (TEST_DEFAULT_SLEEP_TIME);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_free (pipeline_str);
  gst_object_unref (pipeline);
}

/**
 * @brief Test filters to reload new model
 */
TEST (nnstreamerFilterSharedModel, tfliteSharedReload)
{
  gchar *pipeline_str;
  GstElement *pipeline, *filter1, *filter2, *sink1, *sink2;
  gint idx0 = 0, idx1 = 1;
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
  gchar *new_model_path = g_build_filename (
      root_path, "tests", "test_models", "models", model_name2, NULL);
  gchar *path;
  g_free (root_path);

  _get_pipeline_str (&pipeline_str, model_name1, model_name1);
  pipeline = gst_parse_launch (pipeline_str, NULL);
  g_free (pipeline_str);
  memset (res, 0, sizeof (res));

  filter1 = gst_bin_get_by_name (GST_BIN (pipeline), "filter1");
  ASSERT_TRUE (filter1 != NULL);
  filter2 = gst_bin_get_by_name (GST_BIN (pipeline), "filter2");
  ASSERT_TRUE (filter2 != NULL);

  sink1 = gst_bin_get_by_name (GST_BIN (pipeline), "sink1");
  EXPECT_NE (sink1, nullptr);
  g_signal_connect (sink1, "new-data", (GCallback) _new_data_cb, (gpointer) &idx0);
  sink2 = gst_bin_get_by_name (GST_BIN (pipeline), "sink2");
  EXPECT_NE (sink2, nullptr);
  g_signal_connect (sink2, "new-data", (GCallback) _new_data_cb, (gpointer) &idx1);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (TEST_DEFAULT_SLEEP_TIME);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PAUSED, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (TEST_DEFAULT_SLEEP_TIME);

  /* check two filters have same output */
  EXPECT_NE (res[0], 0U);
  EXPECT_EQ (res[0], res[1]);
  memset (res, 0, sizeof (res));

  /* reload filter */
  g_object_set (filter1, "model", new_model_path, NULL);
  g_object_get (filter1, "model", &path, NULL);
  EXPECT_STREQ (new_model_path, path);
  g_free (new_model_path);
  g_free (path);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (TEST_DEFAULT_SLEEP_TIME);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PAUSED, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (TEST_DEFAULT_SLEEP_TIME);

  /* same output with new model */
  EXPECT_NE (res[0], 0U);
  EXPECT_EQ (res[0], res[1]);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (filter1);
  gst_object_unref (filter2);
  gst_object_unref (sink1);
  gst_object_unref (sink2);
  gst_object_unref (pipeline);
}

/**
 * @brief Main gtest
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
