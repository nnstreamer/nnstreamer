/**
 * @file        unittest_tizen_nnfw_runtime_raw.cc
 * @date        07 Oct 2019
 * @brief       Unit test for NNFW (ONE) tensor filter plugin, using tensor-filter properties.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>

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
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  /** nnfw needs a directory with model file and metadata in that directory */
  model_path = g_build_filename (root_path, "tests", "test_models", "models", NULL);

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
    .fwname = "nnfw", .fw_opened = 0, .model_files = model_files, .num_models = 1,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  EXPECT_NE (sp, (void *)NULL);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  info.num_tensors = 1;
  info.info[0].type = _NNS_FLOAT32;
  info.info[0].dimension[0] = 1;
  info.info[0].dimension[1] = 1;
  info.info[0].dimension[2] = 1;
  info.info[0].dimension[3] = 1;

  /** get input/output dimension successfully */
  ret = sp->getInputDimension (&prop, &data, &res);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (res.num_tensors, info.num_tensors);
  EXPECT_EQ (res.info[0].type, info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], info.info[0].dimension[3]);

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
 * @brief Set input dimensions with nnfw subplugin
 */
TEST (nnstreamer_nnfw_runtime_raw_functions, set_dimension)
{
  int ret;
  void *data = NULL;
  GstTensorsInfo in_info, out_info, res;
  GstTensorMemory input, output;
  gchar *model_file;
  int tensor_size;

  model_file = get_model_file ();
  ASSERT_TRUE (model_file != nullptr);
  const gchar *model_files[] = {
    model_file, NULL,
  };
  GstTensorFilterProperties prop = {
    .fwname = "nnfw", .fw_opened = 0, .model_files = model_files, .num_models = 1,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  EXPECT_NE (sp, (void *)NULL);

  /** set input dimension without open */
  ret = sp->setInputDimension (&prop, &data, &in_info, &out_info);
  EXPECT_NE (ret, 0);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  tensor_size = 5;

  res.num_tensors = 1;
  res.info[0].type = _NNS_FLOAT32;
  res.info[0].dimension[0] = tensor_size;
  res.info[0].dimension[1] = 1;
  res.info[0].dimension[2] = 1;
  res.info[0].dimension[3] = 1;

  /** get input/output dimension successfully */
  ret = sp->getInputDimension (&prop, &data, &in_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (res.num_tensors, in_info.num_tensors);
  EXPECT_EQ (res.info[0].type, in_info.info[0].type);
  EXPECT_NE (res.info[0].dimension[0], in_info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], in_info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], in_info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], in_info.info[0].dimension[3]);

  ret = sp->setInputDimension (&prop, &data, &res, &out_info);
  EXPECT_EQ (ret, 0);

  /** get input/output dimension successfully */
  ret = sp->getInputDimension (&prop, &data, &in_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (res.num_tensors, in_info.num_tensors);
  EXPECT_EQ (res.info[0].type, in_info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], in_info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], in_info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], in_info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], in_info.info[0].dimension[3]);

  ret = sp->getOutputDimension (&prop, &data, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (res.num_tensors, out_info.num_tensors);
  EXPECT_EQ (res.info[0].type, out_info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], out_info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], out_info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], out_info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], out_info.info[0].dimension[3]);

  input.size = gst_tensor_info_get_size (&in_info.info[0]);
  output.size = gst_tensor_info_get_size (&out_info.info[0]);

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  /* generate dummy data */
  for (int idx = 0; idx < tensor_size; idx++)
    ((float *)input.data)[idx] = (float)idx;

  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_EQ (ret, 0);

  for (int idx = 0; idx < tensor_size; idx++)
    EXPECT_FLOAT_EQ (((float *)output.data)[idx], (float)(idx + 2));

  g_free (input.data);
  g_free (output.data);

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
    .fwname = "nnfw", .fw_opened = 0, .model_files = model_files, .num_models = 1,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  EXPECT_NE (sp, (void *)NULL);

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke successful */
  *((float *)input.data) = 10.0;
  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_FLOAT_EQ (*((float *)output.data), 12.0);

  *((float *)input.data) = 1.0;
  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_FLOAT_EQ (*((float *)output.data), 3.0);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief get argmax from the array
 */
static size_t
get_argmax (guint8 *array, size_t size)
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
 * @brief Test nnfw subplugin with successful invoke for tflite advanced model
 */
TEST (nnstreamer_nnfw_runtime_raw_functions, invoke_advanced)
{
  int ret;
  void *data = NULL;
  gchar *model_file, *manifest_file, *data_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar *orig_model = "add.tflite";
  const gchar *new_model = "mobilenet_v1_1.0_224_quant.tflite";
  GstTensorMemory input, output;
  GstTensorsInfo info, res;
  char *replace_command;
  gsize data_read;
  size_t max_idx;
  gboolean status;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  /** nnfw needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  status = g_file_test (model_file, G_FILE_TEST_EXISTS);
  if (status == FALSE) {
    g_free (model_file);
    ASSERT_EQ (status, TRUE);
  }

  manifest_file = g_build_filename (
      root_path, "tests", "test_models", "models", "metadata", "MANIFEST", NULL);
  status = g_file_test (manifest_file, G_FILE_TEST_EXISTS);
  if (status == FALSE) {
    g_free (model_file);
    g_free (manifest_file);
    ASSERT_EQ (status, TRUE);
  }

  const gchar *model_files[] = {
    model_file, NULL,
  };
  GstTensorFilterProperties prop = {
    .fwname = "nnfw", .fw_opened = 0, .model_files = model_files, .num_models = 1,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  EXPECT_NE (sp, (void *)NULL);

  replace_command = g_strdup_printf ("sed -i '/%s/c\\\"models\" : [ \"%s\" ],' %s",
      orig_model, new_model, manifest_file);
  ret = system (replace_command);
  g_free (replace_command);

  if (ret != 0) {
    g_free (model_file);
    g_free (manifest_file);
    ASSERT_EQ (ret, 0);
  }

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

  output.size = gst_tensor_info_get_size (&res.info[0]);

  input.data = NULL;
  output.data = g_malloc (output.size);

  data_file = g_build_filename (
      root_path, "tests", "test_models", "data", "orange.raw", NULL);
  status = g_file_get_contents (data_file, (gchar **)&input.data, &data_read, NULL);
  EXPECT_EQ (data_read, input.size);

  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_EQ (ret, 0);

  /**
   * entry 952 (idx 951) is orange as per tests/test_models/labels/labels.txt
   */
  max_idx = get_argmax ((guint8 *)output.data, output.size);
  EXPECT_EQ (max_idx, 951U);

  g_free (data_file);
  g_free (output.data);
  g_free (input.data);
  g_free (model_file);
  sp->close (&prop, &data);

  replace_command = g_strdup_printf ("sed -i '/%s/c\\\"models\" : [ \"%s\" ],' %s",
      new_model, orig_model, manifest_file);
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
  int result = -1;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
