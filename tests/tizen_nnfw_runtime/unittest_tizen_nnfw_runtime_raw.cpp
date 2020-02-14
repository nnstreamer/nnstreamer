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
#include <nnstreamer.h>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer-capi-private.h>
#include <nnstreamer-single.h>

/**
 * @brief Test nnfw subplugin existence.
 */
TEST (nnstreamer_nnfw_runtime_raw_functions, check_existence)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  EXPECT_NE (sp, (void *) NULL);
}

/**
 * @brief Test nnfw subplugin with failing open/flose (no model file)
 */
TEST (nnstreamer_nnfw_runtime_raw_functions, open_close_00_n)
{
  int ret;
  GstTensorFilterProperties prop = {
    .fwname = "nnfw",
    .fw_opened = 0,
    .model_file = "null.nnfw",
  };
  void *data;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  EXPECT_NE (sp, (void *) NULL);

  data = NULL;
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
  GstTensorFilterProperties prop = {
    .fwname = "nnfw",
    .fw_opened = 0,
    .model_file = model_file,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  EXPECT_NE (sp, (void *) NULL);

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
  GstTensorFilterProperties prop = {
    .fwname = "nnfw",
    .fw_opened = 0,
    .model_file = model_file,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  EXPECT_NE (sp, (void *) NULL);

  output.type = input.type = _NNS_FLOAT32;
  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke successful */
  *((float *)input.data) = 10.0;
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
 * @brief Test nnfw subplugin with successful invoke (single ML-API)
 */
TEST (nnstreamer_nnfw_mlapi, invoke_single_00)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_info_h in_res, out_res;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim, res_dim;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;
  unsigned int count = 0;
  int status;
  float *data;
  size_t data_size;

  gchar *test_model;

  test_model = get_model_file ();
  ASSERT_TRUE (test_model != nullptr);

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);
  ml_tensors_info_create (&in_res);
  ml_tensors_info_create (&out_res);

  in_dim[0] = in_dim[1] = in_dim[2] = in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = out_dim[1] = out_dim[2] = out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_NNFW, ML_NNFW_HW_AUTO);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* input tensor in filter */
  status = ml_single_get_input_info (single, &in_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (in_res, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_type (in_res, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_FLOAT32);

  ml_tensors_info_get_tensor_dimension (in_res, 0, res_dim);
  EXPECT_TRUE (in_dim[0] == res_dim[0]);
  EXPECT_TRUE (in_dim[1] == res_dim[1]);
  EXPECT_TRUE (in_dim[2] == res_dim[2]);
  EXPECT_TRUE (in_dim[3] == res_dim[3]);

  /* output tensor in filter */
  status = ml_single_get_output_info (single, &out_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (out_res, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_type (out_res, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_FLOAT32);

  ml_tensors_info_get_tensor_dimension (out_res, 0, res_dim);
  EXPECT_TRUE (out_dim[0] == res_dim[0]);
  EXPECT_TRUE (out_dim[1] == res_dim[1]);
  EXPECT_TRUE (out_dim[2] == res_dim[2]);
  EXPECT_TRUE (out_dim[3] == res_dim[3]);

  input = output = NULL;

  /* generate data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status =
      ml_tensors_data_get_tensor_data (input, 0, (void **) &data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, sizeof (float));
  *data = 10.0;

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  status =
      ml_tensors_data_get_tensor_data (output, 0, (void **) &data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, sizeof (float));
  EXPECT_EQ (*data, 12.0);

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
  ml_tensors_info_destroy (in_res);
  ml_tensors_info_destroy (out_res);
}

/**
 * @brief Test nnfw subplugin with unsuccessful invoke (single ML-API)
 * @detail Model is not found
 */
TEST (nnstreamer_nnfw_mlapi, invoke_single_01_n)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  /* Model does not exist. */
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "invalid_model.tflite", NULL);
  EXPECT_FALSE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  in_dim[0] = in_dim[1] = in_dim[2] = in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = out_dim[1] = out_dim[2] = out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_NNFW, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  input = output = NULL;

  /* generate data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test nnfw subplugin with unsuccessful invoke (single ML-API)
 * @detail Dimension of model is not matched.
 */
TEST (nnstreamer_nnfw_mlapi, invoke_single_02_n)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_info_h in_res;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim, res_dim;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;
  unsigned int count = 0;
  int status;
  float *data;
  size_t data_size;
  gchar *test_model;

  test_model = get_model_file ();
  ASSERT_TRUE (test_model != nullptr);

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);
  ml_tensors_info_create (&in_res);

  in_dim[0] = in_dim[1] = in_dim[2] = in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = out_dim[1] = out_dim[2] = out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  /* Open model with proper dimension */
  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_NNFW, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* input tensor in filter */
  status = ml_single_get_input_info (single, &in_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (in_res, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_type (in_res, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_FLOAT32);

  ml_tensors_info_get_tensor_dimension (in_res, 0, res_dim);
  EXPECT_TRUE (in_dim[0] == res_dim[0]);
  EXPECT_TRUE (in_dim[1] == res_dim[1]);
  EXPECT_TRUE (in_dim[2] == res_dim[2]);
  EXPECT_TRUE (in_dim[3] == res_dim[3]);

  input = output = NULL;

  /* Change and update dimension for mismatch */
  in_dim[0] = in_dim[1] = in_dim[2] = in_dim[3] = 2;
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status =
      ml_tensors_data_get_tensor_data (input, 0, (void **) &data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, sizeof (float) * 16);
  data[0] = 10.0;

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
  ml_tensors_info_destroy (in_res);
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
  /* ignore tizen feature status while running the testcases */
  set_feature_state (1);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }
  set_feature_state (-1);

  return result;
}
