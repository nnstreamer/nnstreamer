/**
 * @file        unittest_filter_armnn.cpp
 * @date        13 Dec 2019
 * @brief       Unit test for armnn tensor filter plugin.
 * @see         https://github.com/nnsuite/nnstreamer
 * @author      Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>        /* GStatBuf */
#include <nnstreamer_plugin_api_filter.h>

/**
 * @brief Test armnn subplugin existence.
 */
TEST (nnstreamer_filter_armnn, check_existence)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);
}

/**
 * @brief Test armnn subplugin with failing open/close (no model file)
 */
TEST (nnstreamer_filter_armnn, open_close_00_n)
{
  int ret;
  const gchar *model_files[] = { "temp.armnn", NULL, };
  GstTensorFilterProperties prop = {
    .fwname = "armnn",
    .fw_opened = 0,
    .model_files = model_files,
    .num_models = 1,
  };
  void *data = NULL;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Test armnn subplugin with successful open/close
 */
TEST (nnstreamer_filter_armnn, open_close_01_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");

  ASSERT_NE (root_path, nullptr);

  model_file = g_build_filename (root_path, "tests", "test_models", "models",
      "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = { model_file, NULL, };
  GstTensorFilterProperties prop = {
    .fwname = "armnn",
    .fw_opened = 0,
    .model_files = model_files,
    .num_models = 1,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  /** close without open, should not crash */
  sp->close (&prop, &data);

  /** open and close successfully */
  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);

  /** close twice, should not crash */
  sp->close (&prop, &data);
}

/**
 * @brief Get input/output dimensions with armnn subplugin
 */
TEST (nnstreamer_filter_armnn, get_dimension)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  GstTensorsInfo info, res;

  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (root_path, "tests", "test_models", "models",
      "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = { model_file, NULL, };
  GstTensorFilterProperties prop = {
    .fwname = "armnn",
    .fw_opened = 0,
    .model_files = model_files,
    .num_models = 1,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  /** get input/output dimension without open */
  ret = sp->getInputDimension (&prop, &data, &res);
  EXPECT_NE (ret, 0);
  ret = sp->getOutputDimension (&prop, &data, &res);
  EXPECT_NE (ret, 0);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  info.num_tensors = 1;
  info.info[0].type = _NNS_FLOAT32;
  info.info[0].dimension[0] = 1;
  info.info[0].dimension[1] = 1;
  info.info[0].dimension[2] = 1;
  info.info[0].dimension[3] = 1;

  /** get input/output dimension successfully */
  ret = sp->getInputDimension (&prop, &data, NULL);
  EXPECT_NE (ret, 0);
  ret = sp->getInputDimension (&prop, &data, &res);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (res.num_tensors, info.num_tensors);
  EXPECT_EQ (res.info[0].type, info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], info.info[0].dimension[3]);

  ret = sp->getOutputDimension (&prop, &data, NULL);
  EXPECT_NE (ret, 0);
  ret = sp->getOutputDimension (&prop, &data, &res);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (res.num_tensors, info.num_tensors);
  EXPECT_EQ (res.info[0].type, info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], info.info[0].dimension[3]);

  sp->close (&prop, &data);
}

/**
 * @brief Test armnn subplugin with successful invoke
 */
TEST (nnstreamer_filter_armnn, invoke_basic)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  GstTensorMemory input, output;

  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (root_path, "tests", "test_models", "models",
      "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = { model_file, NULL, };
  GstTensorFilterProperties prop = {
    .fwname = "armnn",
    .fw_opened = 0,
    .model_files = model_files,
    .num_models = 1,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  output.type = input.type = _NNS_FLOAT32;
  output.size = input.size = sizeof(float) * 1;

  input.data = g_malloc(input.size);
  output.data = g_malloc(output.size);

  /** invoke without open */
  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_NE (ret, 0);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke successful */
  ret = sp->invoke_NN (&prop, &data, NULL, NULL);
  EXPECT_NE (ret, 0);
  ret = sp->invoke_NN (&prop, &data, &input, NULL);
  EXPECT_NE (ret, 0);
  ret = sp->invoke_NN (&prop, &data, NULL, &output);
  EXPECT_NE (ret, 0);

  *((float *)input.data) = 10.0;
  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (*((float *)output.data), 12.0);

  *((float *)input.data) = 1.0;
  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (*((float *)output.data), 3.0);

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
  int result;

  testing::InitGoogleTest (&argc, argv);

  result = RUN_ALL_TESTS ();

  return result;
}

