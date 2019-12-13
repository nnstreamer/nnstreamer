/**
 * @file        unittest_tizen_nnfw_runtime_raw.cpp
 * @date        13 Mar 2019
 * @brief       Unit test for Tizen nnfw tensor filter plugin.
 * @see         https://github.com/nnsuite/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>        /* GStatBuf */
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer.h>
#include <nnstreamer-single.h>
#include <nnstreamer-capi-private.h>

#define MODEL_WIDTH   224
#define MODEL_HEIGHT  224

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
TEST (nnstreamer_nnfw_runtime_raw_functions, get_dimension_00)
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
 * @brief Get input/output dimensions with nnfw subplugin
 * @detail Check NNFW supports multidimensional input/output
 */
TEST (nnstreamer_nnfw_runtime_raw_functions, get_dimension_01)
{
  int ret;
  void *data = NULL;
  GstTensorsInfo info, res;
  gchar *model_file;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  model_file = g_build_filename (root_path, "tests", "test_models", "models",
      "text_classification.tflite", NULL);
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
  info.info[0].dimension[0] = 256;
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

  info.info[0].dimension[0] = 2;
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
 * @brief Callback for tensor sink signal.
 */
static void
new_data_cb (const ml_tensors_data_h data, const ml_tensors_info_h info,
    void *user_data)
{
  int status;
  float *data_ptr;
  size_t data_size;

  status =
      ml_tensors_data_get_tensor_data (data, 0, (void **) &data_ptr,
      &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (*data_ptr, 12.0);
}

/**
 * @brief Test nnfw subplugin with successful invoke (pipeline, ML-API)
 */
TEST (nnstreamer_nnfw_mlapi, invoke_pipeline_00)
{
  gchar *pipeline;
  ml_pipeline_h handle;
  ml_pipeline_src_h src_handle;
  ml_pipeline_sink_h sink_handle;
  ml_tensor_dimension in_dim;
  ml_tensors_info_h info;
  ml_pipeline_state_e state;
  ml_tensors_data_h input;
  float *data;
  size_t data_size;

  gchar *test_model;

  test_model = get_model_file ();
  ASSERT_TRUE (test_model != nullptr);

  pipeline =
      g_strdup_printf ("appsrc name=appsrc ! "
      "other/tensor,dimension=(string)1:1:1:1,type=(string)float32,framerate=(fraction)0/1 ! "
      "tensor_filter framework=nnfw model=%s ! "
      "tensor_sink name=tensor_sink", test_model);

  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* get tensor element using name */
  status = ml_pipeline_src_get_handle (handle, "appsrc", &src_handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  /* register call back function when new data is arrived on sink pad */
  status =
      ml_pipeline_sink_register (handle, "tensor_sink", new_data_cb, NULL,
      &sink_handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  in_dim[0] = in_dim[1] = in_dim[2] = in_dim[3] = 1;
  ml_tensors_info_create (&info);
  ml_tensors_info_set_count (info, 1);
  ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (info, 0, in_dim);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);    /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);


  /* generate data */
  status = ml_tensors_data_create (info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status =
      ml_tensors_data_get_tensor_data (input, 0, (void **) &data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, sizeof (float));
  *data = 10.0;
  status = ml_tensors_data_set_tensor_data (input, 0, data, sizeof (float));
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Push data to the source pad */
  for (int i = 0; i < 5; i++) {
    status =
        ml_pipeline_src_input_data (src_handle, input,
        ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
    EXPECT_EQ (status, ML_ERROR_NONE);
    g_usleep (100000);
  }

  ml_tensors_info_destroy (info);
  ml_tensors_data_destroy (input);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (50000);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
  g_free (test_model);
}

/**
 * @brief Test nnfw subplugin with successful invoke (pipeline, ML-API)
 * @detail Failure case with invalid parameter
 */
TEST (nnstreamer_nnfw_mlapi, invoke_pipeline_01_n)
{
  gchar *pipeline;
  ml_pipeline_h handle;
  ml_pipeline_src_h src_handle;
  ml_tensor_dimension in_dim;
  ml_tensors_info_h info;
  ml_pipeline_state_e state;
  ml_tensors_data_h input;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;
  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  /* Model does not exist. */
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "NULL.tflite", NULL);
  EXPECT_FALSE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  pipeline =
      g_strdup_printf ("appsrc name=appsrc ! "
      "other/tensor,dimension=(string)1:1:1:1,type=(string)float32,framerate=(fraction)0/1 ! "
      "tensor_filter framework=nnfw model=%s ! tensor_sink name=tensor_sink",
      test_model);

  int status = ml_pipeline_construct (NULL, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_STREAMS_PIPE);

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "add.tflite", NULL);
  EXPECT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  pipeline =
      g_strdup_printf ("appsrc name=appsrc ! "
      "other/tensor,dimension=(string)1:1:1:1,type=(string)float32,framerate=(fraction)0/1 ! "
      "tensor_filter framework=nnfw model=%s ! tensor_sink name=tensor_sink",
      test_model);

  status = ml_pipeline_construct (pipeline, NULL, NULL, NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* get tensor element using name */
  status = ml_pipeline_src_get_handle (handle, "appsrc", &src_handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  in_dim[0] = in_dim[1] = in_dim[2] = in_dim[3] = 1;
  ml_tensors_info_create (&info);
  ml_tensors_info_set_count (info, 1);
  ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (info, 0, in_dim);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);    /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);


  /* generate data */
  status = ml_tensors_data_create (info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  /* Push data to the source pad */
  status =
      ml_pipeline_src_input_data (src_handle, input,
      ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
  g_usleep (100000);


  ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_FLOAT32);
  in_dim[0] = 5;
  ml_tensors_info_set_tensor_dimension (info, 0, in_dim);
  input = NULL;

  status = ml_tensors_data_create (info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  /* Push data to the source pad */
  status =
      ml_pipeline_src_input_data (src_handle, input,
      ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
  g_usleep (100000);

  ml_tensors_info_destroy (info);
  ml_tensors_data_destroy (input);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
  g_free (test_model);

}

/**
 * @brief Callback for tensor sink signal.
 */
static void
new_data_cb_2 (const ml_tensors_data_h data, const ml_tensors_info_h info,
    void *user_data)
{
  unsigned int cnt = 0;
  int status;
  float *data_ptr;
  size_t data_size;
  ml_tensor_dimension out_dim;

  ml_tensors_info_get_count (info, &cnt);
  EXPECT_EQ (cnt, 1);

  ml_tensors_info_get_tensor_dimension (info, 0, out_dim);
  EXPECT_EQ (out_dim[0], 2);
  EXPECT_EQ (out_dim[1], 1);
  EXPECT_EQ (out_dim[2], 1);
  EXPECT_EQ (out_dim[3], 1);

  status =
      ml_tensors_data_get_tensor_data (data, 0, (void **) &data_ptr,
      &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, 8);
}

/**
 * @brief Test nnfw subplugin multi-modal (pipeline, ML-API)
 * @detail Invoke a model via Pipeline API, with two input streams into a single tensor 
 */
TEST (nnstreamer_nnfw_mlapi, multimodal_01_p)
{
  gchar *pipeline;
  ml_pipeline_h handle;
  ml_pipeline_src_h src_handle_0, src_handle_1;
  ml_pipeline_sink_h sink_handle;
  ml_tensor_dimension in_dim;
  ml_tensors_info_h info;
  ml_pipeline_state_e state;
  ml_tensors_data_h input_0, input_1;
  float *data0, *data1;
  size_t data_size0, data_size1;
  unsigned int i;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;
  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "text_classification.tflite", NULL);
  EXPECT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  pipeline =
      g_strdup_printf
      ("appsrc name=appsrc_0 ! other/tensor,dimension=(string)128:1:1:1,type=(string)float32,framerate=(fraction)0/1 ! mux.sink_0 "
      "appsrc name=appsrc_1 ! other/tensor,dimension=(string)128:1:1:1,type=(string)float32,framerate=(fraction)0/1 ! mux.sink_1 "
      "tensor_merge mode=linear option=0 sync_mode=nosync name=mux ! "
      "tensor_filter framework=nnfw input=256:1:1:1 inputtype=float32 model=%s ! tensor_sink name=tensor_sink",
      test_model);

  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* get tensor element using name */
  status = ml_pipeline_src_get_handle (handle, "appsrc_0", &src_handle_0);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_pipeline_src_get_handle (handle, "appsrc_1", &src_handle_1);
  EXPECT_EQ (status, ML_ERROR_NONE);
  /* register call back function when new data is arrived on sink pad */
  status =
      ml_pipeline_sink_register (handle, "tensor_sink", new_data_cb_2, NULL,
      &sink_handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  in_dim[0] = 128;
  in_dim[1] = in_dim[2] = in_dim[3] = 1;
  ml_tensors_info_create (&info);
  ml_tensors_info_set_count (info, 1);
  ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (info, 0, in_dim);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (500000);            /* wait for start */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  /* generate data */
  status = ml_tensors_data_create (info, &input_0);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input_0 != NULL);

  status =
      ml_tensors_data_get_tensor_data (input_0, 0, (void **) &data0,
      &data_size0);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size0, sizeof (float) * 128);
  data0[0] = 1.0;
  for (i = 1; i < 128; i++) {
    data0[i] = i + 4;
  }

  status = ml_tensors_data_create (info, &input_1);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input_0 != NULL);
  status =
      ml_tensors_data_get_tensor_data (input_1, 0, (void **) &data1,
      &data_size1);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size1, sizeof (float) * 128);
  for (i = 0; i < 128; i++) {
    data0[i] = 0;
  }

  /* Push data to the source pad */
  status =
      ml_pipeline_src_input_data (src_handle_0, input_0,
      ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_pipeline_src_input_data (src_handle_1, input_1,
      ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (5000000);

  ml_tensors_info_destroy (info);
  ml_tensors_data_destroy (input_0);
  ml_tensors_data_destroy (input_1);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
  g_free (test_model);

}

/**
 * @brief Test nnfw subplugin multi-model (pipeline, ML-API)
 * @detail Invoke two models via Pipeline API, sharing a single input stream 
 */
TEST (nnstreamer_nnfw_mlapi, multimodel_01_p)
{
  gchar *pipeline;
  ml_pipeline_h handle;
  ml_pipeline_src_h src_handle;
  ml_pipeline_sink_h sink_handle_0, sink_handle_1;
  ml_tensor_dimension in_dim;
  ml_tensors_info_h info;
  ml_pipeline_state_e state;
  ml_tensors_data_h input;
  float *data;
  size_t data_size;

  gchar *test_model;

  test_model = get_model_file ();
  ASSERT_TRUE (test_model != nullptr);

  pipeline =
      g_strdup_printf ("appsrc name=appsrc ! "
      "other/tensor,dimension=(string)1:1:1:1,type=(string)float32,framerate=(fraction)0/1 ! tee name=t "
      "t. ! queue ! tensor_filter framework=nnfw model=%s ! tensor_sink name=tensor_sink_0 "
      "t. ! queue ! tensor_filter framework=nnfw model=%s ! tensor_sink name=tensor_sink_1",
      test_model, test_model);

  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* get tensor element using name */
  status = ml_pipeline_src_get_handle (handle, "appsrc", &src_handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* register call back function when new data is arrived on sink pad */
  status =
      ml_pipeline_sink_register (handle, "tensor_sink_0", new_data_cb, NULL,
      &sink_handle_0);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status =
      ml_pipeline_sink_register (handle, "tensor_sink_1", new_data_cb, NULL,
      &sink_handle_1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  in_dim[0] = in_dim[1] = in_dim[2] = in_dim[3] = 1;
  ml_tensors_info_create (&info);
  ml_tensors_info_set_count (info, 1);
  ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (info, 0, in_dim);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (500000);            /* wait for start */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  /* generate data */
  status = ml_tensors_data_create (info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status =
      ml_tensors_data_get_tensor_data (input, 0, (void **) &data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, sizeof (float));
  *data = 10.0;

  status = ml_tensors_data_set_tensor_data (input, 0, data, data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Push data to the source pad */
  status =
      ml_pipeline_src_input_data (src_handle, input,
      ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (500000);

  ml_tensors_info_destroy (info);
  ml_tensors_data_destroy (input);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
  g_free (test_model);

}

/**
 * @brief Test nnfw subplugin multi-model (pipeline, ML-API)
 * @detail Invoke two models which have different framework via Pipeline API, sharing a single input stream 
 */
TEST (nnstreamer_nnfw_mlapi, multimodel_02_p)
{
  gchar *pipeline;
  ml_pipeline_h handle;
  ml_pipeline_src_h src_handle;
  ml_pipeline_sink_h sink_handle_0, sink_handle_1;
  ml_tensor_dimension in_dim;
  ml_tensors_info_h info;
  ml_pipeline_state_e state;
  ml_tensors_data_h input;
  float *data;
  size_t data_size;

  gchar *test_model;

  test_model = get_model_file ();
  ASSERT_TRUE (test_model != nullptr);

  pipeline =
      g_strdup_printf ("appsrc name=appsrc ! "
      "other/tensor,dimension=(string)1:1:1:1,type=(string)float32,framerate=(fraction)0/1 ! tee name=t "
      "t. ! queue ! tensor_filter framework=nnfw model=%s ! tensor_sink name=tensor_sink_0 "
      "t. ! queue ! tensor_filter framework=tensorflow-lite model=%s ! tensor_sink name=tensor_sink_1",
      test_model, test_model);

  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* get tensor element using name */
  status = ml_pipeline_src_get_handle (handle, "appsrc", &src_handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* register call back function when new data is arrived on sink pad */
  status =
      ml_pipeline_sink_register (handle, "tensor_sink_0", new_data_cb, NULL,
      &sink_handle_0);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status =
      ml_pipeline_sink_register (handle, "tensor_sink_1", new_data_cb, NULL,
      &sink_handle_1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  in_dim[0] = in_dim[1] = in_dim[2] = in_dim[3] = 1;
  ml_tensors_info_create (&info);
  ml_tensors_info_set_count (info, 1);
  ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (info, 0, in_dim);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (500000);            /* wait for start */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  /* generate data */
  status = ml_tensors_data_create (info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status =
      ml_tensors_data_get_tensor_data (input, 0, (void **) &data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, sizeof (float));
  *data = 10.0;

  status = ml_tensors_data_set_tensor_data (input, 0, data, data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Push data to the source pad */
  status =
      ml_pipeline_src_input_data (src_handle, input,
      ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (500000);

  ml_tensors_info_destroy (info);
  ml_tensors_data_destroy (input);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (50000);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
  g_free (test_model);

}

/**
 * @brief Main gtest
 */
int
main (int argc, char **argv)
{
  int result;

  testing::InitGoogleTest (&argc, argv);

  /* ignore tizen feature status while running the testcases */
  set_feature_state (1);

  result = RUN_ALL_TESTS ();

  set_feature_state (-1);

  return result;
}
