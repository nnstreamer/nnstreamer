/**
 * @file        unittest_tizen_capi.cpp
 * @date        13 Mar 2019
 * @brief       Unit test for Tizen CAPI of NNStreamer. Basis of TCT in the future.
 * @see         https://github.com/nnsuite/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>        /* GStatBuf */
#include <stdlib.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api.h>

/**
 * @brief Test nnfw subplugin existence.
 */
TEST (nnstreamer_nnfw_runtime_raw_functions, check_existence)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  EXPECT_NE (sp, (void *) NULL);
}

/**
 * @brief Test nnfw subplugin with failing open/close (no model file)
 */
TEST (nnstreamer_nnfw_runtime_raw_functions, open_close_00_n)
{
  int ret;
  const gchar *model_files[] = {
    "null.nnfw", NULL,
  };
  GstTensorFilterProperties prop = {
    .fwname = "nnfw",
    .fw_opened = 0,
    .model_files = model_files,
    .num_models = 1,
  };
  void *data = NULL;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  EXPECT_NE (sp, (void *) NULL);

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Get model file after validation checks
 * @returns model file path, NULL on error
 * @note caller has to be free the returned model file path
 */
static gchar *
get_model_file ()
{
  gchar *model_file;
  gchar *meta_file;
  gchar *model_path;
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");

  g_return_val_if_fail (root_path != nullptr, FALSE);

  /** nnfw needs a directory with model file and metadata in that directory */
  model_path = g_build_filename (root_path, "tests", "test_models", "models",
      NULL);

  meta_file = g_build_filename (model_path, "metadata", "MANIFEST", NULL);
  if (!g_file_test (meta_file, G_FILE_TEST_EXISTS)) {
    g_free (model_path);
    g_free (meta_file);
    return NULL;
  }

  model_file = g_build_filename (model_path, "add.tflite", NULL);
  g_free (meta_file);
  g_free (model_path);

  if (!g_file_test (model_file, G_FILE_TEST_EXISTS)) {
    g_free (model_file);
    return NULL;
  }

  return model_file;
}

/**
 * @brief Test nnfw subplugin with successful open/close
 */
TEST (nnstreamer_nnfw_runtime_raw_functions, open_close_01_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;

  model_file = get_model_file ();
  ASSERT_TRUE (model_file != nullptr);
  const gchar *model_files[] = {
    model_file, NULL,
  };
  GstTensorFilterProperties prop = {
    .fwname = "nnfw",
    .fw_opened = 0,
    .model_files = model_files,
    .num_models = 1,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  EXPECT_NE (sp, (void *) NULL);

  /** close without open, should not crash */
  sp->close (&prop, &data);

  /** open and close successfully */
  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);

  /** close twice, should not crash */
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Get input/output dimensions with nnfw subplugin
 */
TEST (nnstreamer_nnfw_runtime_raw_functions, get_dimension)
{
  int ret;
  void *data = NULL;
  GstTensorsInfo info, res;
  gchar *model_file;

  model_file = get_model_file ();
  ASSERT_TRUE (model_file != nullptr);
  const gchar *model_files[] = {
    model_file, NULL,
  };
  GstTensorFilterProperties prop = {
    .fwname = "nnfw",
    .fw_opened = 0,
    .model_files = model_files,
    .num_models = 1,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
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
  g_free (model_file);
}

/**
 * @brief Test nnfw subplugin with successful invoke
 */
TEST (nnstreamer_nnfw_runtime_raw_functions, invoke)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  gchar *model_file;

  model_file = get_model_file ();
  ASSERT_TRUE (model_file != nullptr);
  const gchar *model_files[] = {
    model_file, NULL,
  };
  GstTensorFilterProperties prop = {
    .fwname = "nnfw",
    .fw_opened = 0,
    .model_files = model_files,
    .num_models = 1,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  EXPECT_NE (sp, (void *) NULL);

  output.type = input.type = _NNS_FLOAT32;
  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

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

  *((float *) input.data) = 10.0;
  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (*((float *) output.data), 12.0);

  *((float *) input.data) = 1.0;
  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (*((float *) output.data), 3.0);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief get argmax from the array
 */
size_t
get_argmax (guint8 * array, size_t size)
{
  size_t idx, max_idx = 0;
  guint8 max_value = 0;
  for (idx = 0; idx < size; idx++) {
    if (max_value < array[idx]) {
      max_idx = idx;
      max_value = array[idx];
    }
  }

  return max_idx;
}

/**
 * @brief Test armnn subplugin with successful invoke for tflite advanced model
 */
TEST (nnstreamer_filter_armnn, invoke_advanced)
{
  int ret;
  void *data = NULL;
  gchar *model_file, *manifest_file, *data_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  const gchar *orig_model = "add.tflite";
  const gchar *new_model = "mobilenet_v1_1.0_224_quant.tflite";
  GstTensorMemory input, output;
  GstTensorsInfo info, res;
  char *replace_command;
  gsize data_read;
  size_t max_idx;
  gboolean status;

  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  status = g_file_test (model_file, G_FILE_TEST_EXISTS);
  if (status == FALSE) {
    g_free (model_file);
    ASSERT_EQ (status, TRUE);
  }

  manifest_file = g_build_filename (root_path, "tests", "test_models", "models",
      "metadata", "MANIFEST", NULL);
  status = g_file_test (manifest_file, G_FILE_TEST_EXISTS);
  if (status == FALSE) {
    g_free (model_file);
    g_free (manifest_file);
    ASSERT_EQ (status, TRUE);
  }

  replace_command =
      g_strdup_printf ("sed -i '/%s/c\\\"models\" : [ \"%s\" ],' %s",
      orig_model, new_model, manifest_file);
  ret = system (replace_command);
  g_free (replace_command);

  if (ret != 0) {
    g_free (model_file);
    g_free (manifest_file);
    ASSERT_EQ (ret, 0);
  }

  const gchar *model_files[] = { model_file, NULL, };
  GstTensorFilterProperties prop = {
    .fwname = "nnfw",
    .fw_opened = 0,
    .model_files = model_files,
    .num_models = 1,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  EXPECT_NE (sp, (void *) NULL);

  info.num_tensors = 1;
  info.info[0].type = _NNS_UINT8;
  info.info[0].dimension[0] = 3;
  info.info[0].dimension[1] = 224;
  info.info[0].dimension[2] = 224;
  info.info[0].dimension[3] = 1;

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** get input/output dimension successfully */
  ret = sp->getInputDimension (&prop, &data, &res);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (res.num_tensors, info.num_tensors);
  EXPECT_EQ (res.info[0].type, info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], info.info[0].dimension[3]);

  input.type = res.info[0].type;
  input.size = gst_tensor_info_get_size (&res.info[0]);

  ret = sp->getOutputDimension (&prop, &data, &res);
  EXPECT_EQ (ret, 0);

  info.num_tensors = 1;
  info.info[0].type = _NNS_UINT8;
  info.info[0].dimension[0] = 1001;
  info.info[0].dimension[1] = 1;
  info.info[0].dimension[2] = 1;
  info.info[0].dimension[3] = 1;

  EXPECT_EQ (res.num_tensors, info.num_tensors);
  EXPECT_EQ (res.info[0].type, info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], info.info[0].dimension[3]);

  output.type = res.info[0].type;
  output.size = gst_tensor_info_get_size (&res.info[0]);

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  data_file = g_build_filename (root_path, "tests", "test_models", "data",
      "orange.raw", NULL);
  status = g_file_get_contents (data_file, (gchar **) &input.data, &data_read,
      NULL);
  EXPECT_EQ (data_read, input.size);

  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_EQ (ret, 0);

  /** entry 952 (idx 951) is orange as per tests/test_models/labels/labels.txt */
  max_idx = get_argmax ((guint8 *) output.data, output.size);
  EXPECT_EQ (max_idx, 951);

  g_free (data_file);
  g_free (output.data);
  g_free (input.data);
  g_free (model_file);
  sp->close (&prop, &data);

  replace_command =
      g_strdup_printf ("sed -i '/%s/c\\\"models\" : [ \"%s\" ],' %s", new_model,
      orig_model, manifest_file);
  ret = system (replace_command);
  g_free (replace_command);
  g_free (manifest_file);
  ASSERT_EQ (ret, 0);
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
