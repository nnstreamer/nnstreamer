/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_python3.cc
 * @date    26 Mar 2024
 * @brief   Unit test for Python3 tensor filter sub-plugin
 * @author  Yelin Jeong <yelini.jeong@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <unittest_util.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>

/**
 * @brief Set tensor filter properties
 */
static void
_SetFilterProp (GstTensorFilterProperties *prop, const gchar *name, const gchar **models)
{
  memset (prop, 0, sizeof (GstTensorFilterProperties));
  prop->fwname = name;
  prop->fw_opened = 0;
  prop->model_files = models;
  prop->num_models = g_strv_length ((gchar **) models);
}

/**
 * @brief Test subplugin existence.
 */
TEST (nnstreamerFilterPython3, checkExistence)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);
}

/**
 * @brief Negative test case with invalid model file path
 */
TEST (nnstreamerFilterPython3, openClose00_n)
{
  int ret;
  void *data = NULL;
  const gchar *model_files[] = {
    "some/invalid/model/path.py",
    NULL,
  };
  GstTensorFilterProperties prop;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);

  _SetFilterProp (&prop, "python3", model_files);
  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Positive case with open/close
 */
TEST (nnstreamerFilterPython3, openClose01)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.py", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "python3", model_files);

  /* close before open */
  sp->close (&prop, &data);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);

  /* double close */
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Positive case with successful getModelInfo
 */
TEST (nnstreamerFilterPython3, getModelInfo00)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.py", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "python3", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  GstTensorsInfo in_info, out_info;

  ret = sp->getModelInfo (NULL, NULL, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  constexpr uint32_t CHANNEL = 3;
  constexpr uint32_t WIDTH = 280;
  constexpr uint32_t HEIGHT = 40;

  EXPECT_EQ (in_info.num_tensors, 1U);
  EXPECT_EQ (in_info.info[0].dimension[0], CHANNEL);
  EXPECT_EQ (in_info.info[0].dimension[1], WIDTH);
  EXPECT_EQ (in_info.info[0].dimension[2], HEIGHT);
  EXPECT_EQ (in_info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (out_info.num_tensors, 1U);
  EXPECT_EQ (out_info.info[0].dimension[0], CHANNEL);
  EXPECT_EQ (out_info.info[0].dimension[1], WIDTH);
  EXPECT_EQ (out_info.info[0].dimension[2], HEIGHT);
  EXPECT_EQ (out_info.info[0].type, _NNS_UINT8);

  sp->close (&prop, &data);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  g_free (model_file);
}

/**
 * @brief Negative case calling getModelInfo before open
 */
TEST (nnstreamerFilterPython3, getModelInfo01_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.py", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);

  GstTensorsInfo in_info, out_info;

  ret = sp->getModelInfo (NULL, NULL, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);
  _SetFilterProp (&prop, "python3", model_files);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Negative case with invalid argument
 */
TEST (nnstreamerFilterPython3, getModelInfo02_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.py", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "python3", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);

  GstTensorsInfo in_info, out_info;

  /* not supported */
  ret = sp->getModelInfo (NULL, NULL, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Negative test case with invoke before open
 */
TEST (nnstreamerFilterPython3, invoke00_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  gchar *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.py", NULL);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "python3", model_files);

  ret = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_NE (ret, 0);

  g_free (model_file);
  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Negative test case with invoke before open
 */
TEST (nnstreamerFilterPython3, invoke01)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  gchar *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.py", NULL);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  output.size = input.size = sizeof (float) * 3 * 280 * 40; // channel * width * height

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  memset (input.data, 0, input.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "python3", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, (void *) NULL);

  ret = sp->invoke (NULL, NULL, data, &input, &output);

  EXPECT_EQ (ret, 0);
  EXPECT_EQ (output.size, input.size);

  g_free (model_file);
  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
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
