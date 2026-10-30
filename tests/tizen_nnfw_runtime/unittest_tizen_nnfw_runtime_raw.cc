/**
 * @file        unittest_tizen_nnfw_runtime_raw.cc
 * @date        07 Oct 2019
 * @brief       Unit test for NNFW (ONE) tensor filter plugin, using tensor-filter properties.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <dlfcn.h>
#include <glib.h>
#include <gst/check/gstharness.h>
#include <nnfw.h>
#include <nnstreamer_internal.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>

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
TEST (nnstreamerNnfwRuntimeRawFunctions, getDimension)
{
  int ret;
  void *data = NULL;
  GstTensorsInfo info, res, out;
  gchar *model_file;
  GstTensorFilterProperties prop;
  model_file = get_model_file ();
  ASSERT_TRUE (model_file != nullptr);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  _SetFilterProp (&prop, "nnfw", model_files);
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  ASSERT_NE (sp, (void *) NULL);
  EXPECT_EQ (sp->version, GST_TENSOR_FILTER_FRAMEWORK_V1);

  gst_tensors_info_init (&info);
  gst_tensors_info_init (&res);
  gst_tensors_info_init (&out);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  info.num_tensors = 1;
  info.info[0].type = _NNS_FLOAT32;
  info.info[0].dimension[0] = 1;

  /** get input/output dimension successfully */
  ret = sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &res, &out);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (res.num_tensors, info.num_tensors);
  EXPECT_EQ (res.info[0].type, info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], info.info[0].dimension[0]);

  EXPECT_EQ (out.num_tensors, info.num_tensors);
  EXPECT_EQ (out.info[0].type, info.info[0].type);
  EXPECT_EQ (out.info[0].dimension[0], info.info[0].dimension[0]);

  sp->close (&prop, &data);

  gst_tensors_info_free (&info);
  gst_tensors_info_free (&res);
  gst_tensors_info_free (&out);
  g_free (model_file);
}

/**
 * @brief Set input dimensions with nnfw subplugin
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, setDimension)
{
  int ret;
  void *data = NULL;
  GstTensorsInfo in_info, out_info, res;
  GstTensorMemory input, output;
  gchar *model_file;
  int tensor_size;
  GstTensorFilterProperties prop;
  model_file = get_model_file ();
  ASSERT_TRUE (model_file != nullptr);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  ASSERT_NE (sp, (void *) NULL);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  gst_tensors_info_init (&res);
  _SetFilterProp (&prop, "nnfw", model_files);

  /** set input dimension without open */
  ret = sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  tensor_size = 5;

  res.num_tensors = 1;
  res.info[0].type = _NNS_FLOAT32;
  res.info[0].dimension[0] = tensor_size;

  /** get input/output dimension successfully */
  ret = sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (res.num_tensors, in_info.num_tensors);
  EXPECT_EQ (res.info[0].type, in_info.info[0].type);

  EXPECT_NE (res.info[0].dimension[0], in_info.info[0].dimension[0]);

  ret = sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, &res, &out_info);
  EXPECT_EQ (ret, 0);

  /** the output info SET_INPUT_INFO reports follows the new input */
  EXPECT_EQ (res.num_tensors, out_info.num_tensors);
  EXPECT_EQ (res.info[0].type, out_info.info[0].type);
  EXPECT_TRUE (gst_tensor_dimension_is_equal (
      res.info[0].dimension, out_info.info[0].dimension));

  /** get input/output dimension successfully */
  ret = sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (res.num_tensors, in_info.num_tensors);
  EXPECT_EQ (res.info[0].type, in_info.info[0].type);
  EXPECT_TRUE (gst_tensor_dimension_is_equal (
      res.info[0].dimension, in_info.info[0].dimension));

  EXPECT_EQ (res.num_tensors, out_info.num_tensors);
  EXPECT_EQ (res.info[0].type, out_info.info[0].type);
  EXPECT_TRUE (gst_tensor_dimension_is_equal (
      res.info[0].dimension, out_info.info[0].dimension));

  input.size = gst_tensor_info_get_size (&in_info.info[0]);
  output.size = gst_tensor_info_get_size (&out_info.info[0]);

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  /* generate dummy data */
  for (int idx = 0; idx < tensor_size; idx++)
    ((float *) input.data)[idx] = (float) idx;

  ret = sp->invoke (sp, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);

  for (int idx = 0; idx < tensor_size; idx++)
    EXPECT_FLOAT_EQ (((float *) output.data)[idx], (float) (idx + 2));

  g_free (input.data);
  g_free (output.data);

  sp->close (&prop, &data);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  gst_tensors_info_free (&res);
  g_free (model_file);
}

/**
 * @brief Number of the next nnfw_run () calls that fail.
 */
static int nnfw_run_fail = 0;

/**
 * @brief Make nnfw_run () of the ONE runtime fail on demand.
 * @note The test executable exports this definition, so the nnfw sub-plugin
 *       calls it instead of the runtime's. The runtime need not be in the
 *       global scope, so look up the real one among the sub-plugin's
 *       dependencies.
 */
extern "C" NNFW_STATUS
nnfw_run (nnfw_session *session)
{
  static NNFW_STATUS (*real_run) (nnfw_session *) = NULL;
  const GstTensorFilterFramework *sp;
  Dl_info dl;
  void *handle;

  if (nnfw_run_fail > 0) {
    nnfw_run_fail--;
    return NNFW_STATUS_ERROR;
  }

  if (real_run == NULL) {
    sp = nnstreamer_filter_find ("nnfw");
    if (sp && dladdr ((void *) sp->open, &dl) != 0) {
      handle = dlopen (dl.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
      if (handle) {
        real_run = (NNFW_STATUS (*) (nnfw_session *)) dlsym (handle, "nnfw_run");
        dlclose (handle);
      }
    }
  }

  if (real_run == NULL)
    return NNFW_STATUS_ERROR;

  return real_run (session);
}

/**
 * @brief Set input dimensions with nnfw subplugin when the dummy invoke fails
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, setDimensionInvokeFail_n)
{
  int ret;
  void *data = NULL;
  GstTensorsInfo in_info, out_info, res, cur_in, cur_out;
  GstTensorMemory input, output;
  gchar *model_file;
  guint idx, num;
  GstTensorFilterProperties prop;
  model_file = get_model_file ();
  ASSERT_TRUE (model_file != nullptr);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  ASSERT_NE (sp, (void *) NULL);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  gst_tensors_info_init (&res);
  gst_tensors_info_init (&cur_in);
  gst_tensors_info_init (&cur_out);
  _SetFilterProp (&prop, "nnfw", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  ret = sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  res.num_tensors = 1;
  res.info[0].type = _NNS_FLOAT32;
  res.info[0].dimension[0] = 5;
  EXPECT_FALSE (gst_tensors_info_is_equal (&res, &in_info));

  /** the runtime accepts the new input info, but the dummy invoke fails */
  nnfw_run_fail = 1;
  ret = sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, &res, &cur_out);
  EXPECT_NE (ret, 0);
  EXPECT_EQ (nnfw_run_fail, 0);
  nnfw_run_fail = 0;
  gst_tensors_info_free (&cur_out);
  gst_tensors_info_init (&cur_out);

  /** the input and output info are still those before the change */
  ret = sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &cur_in, &cur_out);
  EXPECT_EQ (ret, 0);
  EXPECT_TRUE (gst_tensors_info_is_equal (&in_info, &cur_in));
  EXPECT_TRUE (gst_tensors_info_is_equal (&out_info, &cur_out));

  /** and the model still runs with them */
  input.size = gst_tensor_info_get_size (&in_info.info[0]);
  output.size = gst_tensor_info_get_size (&out_info.info[0]);
  ASSERT_EQ (input.size, output.size);

  input.data = g_malloc (input.size);
  output.data = g_malloc0 (output.size);

  num = input.size / sizeof (float);
  for (idx = 0; idx < num; idx++)
    ((float *) input.data)[idx] = (float) idx;

  ret = sp->invoke (sp, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);

  for (idx = 0; idx < num; idx++)
    EXPECT_FLOAT_EQ (((float *) output.data)[idx], (float) (idx + 2));

  g_free (input.data);
  g_free (output.data);

  sp->close (&prop, &data);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  gst_tensors_info_free (&res);
  gst_tensors_info_free (&cur_in);
  gst_tensors_info_free (&cur_out);
  g_free (model_file);
}

/**
 * @brief Test nnfw subplugin with successful invoke
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, invoke)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  gchar *model_file;
  GstTensorFilterProperties prop;
  model_file = get_model_file ();
  ASSERT_TRUE (model_file != nullptr);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  ASSERT_NE (sp, (void *) NULL);

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);
  _SetFilterProp (&prop, "nnfw", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke successful */
  *((float *) input.data) = 10.0;
  ret = sp->invoke (sp, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_FLOAT_EQ (*((float *) output.data), 12.0);

  *((float *) input.data) = 1.0;
  ret = sp->invoke (sp, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_FLOAT_EQ (*((float *) output.data), 3.0);

  sp->close (&prop, &data);

  g_free (input.data);
  g_free (output.data);
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
TEST (nnstreamerNnfwRuntimeRawFunctions, invokeAdvanced)
{
  int ret = -1;
  void *data = NULL;
  gchar *model_file, *manifest_file, *data_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar *orig_model = "add.tflite";
  const gchar *new_model = "mobilenet_v1_1.0_224_quant.tflite";
  GstTensorMemory input, output;
  GstTensorsInfo info, res, out;
  char *replace_command;
  gsize data_read;
  size_t max_idx;
  GstTensorFilterProperties prop;

  model_file = manifest_file = data_file = NULL;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  /** nnfw needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  manifest_file = g_build_filename (
      root_path, "tests", "test_models", "models", "metadata", "MANIFEST", NULL);
  data_file = g_build_filename (
      root_path, "tests", "test_models", "data", "orange.raw", NULL);

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  ASSERT_NE (sp, (void *) NULL);

  if (!g_file_test (model_file, G_FILE_TEST_EXISTS)
      || !g_file_test (manifest_file, G_FILE_TEST_EXISTS)
      || !g_file_test (data_file, G_FILE_TEST_EXISTS)) {
    goto failed;
  }

  replace_command = g_strdup_printf ("sed -i '/%s/c\\\"models\" : [ \"%s\" ],' %s",
      orig_model, new_model, manifest_file);
  ret = system (replace_command);
  g_free (replace_command);

  if (ret != 0) {
    goto failed;
  }

  gst_tensors_info_init (&info);
  gst_tensors_info_init (&res);
  gst_tensors_info_init (&out);

  info.num_tensors = 1;
  info.info[0].type = _NNS_UINT8;
  info.info[0].dimension[0] = 3;
  info.info[0].dimension[1] = 224;
  info.info[0].dimension[2] = 224;
  info.info[0].dimension[3] = 1;
  _SetFilterProp (&prop, "nnfw", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** get input/output dimension successfully */
  ret = sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &res, &out);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (res.num_tensors, info.num_tensors);
  EXPECT_EQ (res.info[0].type, info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], info.info[0].dimension[3]);

  input.size = gst_tensor_info_get_size (&res.info[0]);

  info.num_tensors = 1;
  info.info[0].type = _NNS_UINT8;
  info.info[0].dimension[0] = 1001;
  info.info[0].dimension[1] = 1;

  EXPECT_EQ (out.num_tensors, info.num_tensors);
  EXPECT_EQ (out.info[0].type, info.info[0].type);
  EXPECT_EQ (out.info[0].dimension[0], info.info[0].dimension[0]);
  EXPECT_EQ (out.info[0].dimension[1], info.info[0].dimension[1]);
  output.size = gst_tensor_info_get_size (&out.info[0]);

  input.data = NULL;
  output.data = g_malloc (output.size);

  EXPECT_TRUE (g_file_get_contents (data_file, (gchar **) &input.data, &data_read, NULL));
  EXPECT_EQ (data_read, input.size);

  ret = sp->invoke (sp, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);

  /**
   * entry 952 (idx 951) is orange as per tests/test_models/labels/labels.txt
   */
  max_idx = get_argmax ((guint8 *) output.data, output.size);
  EXPECT_EQ (max_idx, 951U);

  g_free (output.data);
  g_free (input.data);
  sp->close (&prop, &data);

  replace_command = g_strdup_printf ("sed -i '/%s/c\\\"models\" : [ \"%s\" ],' %s",
      new_model, orig_model, manifest_file);
  ret = system (replace_command);
  g_free (replace_command);
  gst_tensors_info_free (&info);
  gst_tensors_info_free (&res);
  gst_tensors_info_free (&out);

failed:
  g_free (model_file);
  g_free (manifest_file);
  g_free (data_file);
  ASSERT_EQ (ret, 0);
}

/**
 * @brief Test the static framework info nnfw reports to tensor_filter
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, frameworkInfo)
{
  GstTensorFilterFrameworkInfo info;
  GstTensorFilterProperties prop;
  const gchar *model_files[] = { NULL };
  gboolean has_cpu = FALSE, has_gpu = FALSE, has_npu = FALSE, has_srcn = FALSE;
  gboolean has_neon = FALSE;
  int idx;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  ASSERT_NE (sp, (void *) NULL);
  EXPECT_EQ (sp->version, GST_TENSOR_FILTER_FRAMEWORK_V1);
  EXPECT_TRUE (sp->open && sp->close);

  _SetFilterProp (&prop, "nnfw", model_files);
  memset (&info, 0, sizeof (info));
  EXPECT_EQ (sp->getFrameworkInfo (sp, &prop, NULL, &info), 0);

  EXPECT_STREQ (info.name, "nnfw");
  EXPECT_FALSE (info.allow_in_place);
  EXPECT_FALSE (info.allocate_in_invoke);
  EXPECT_FALSE (info.run_without_model);
  EXPECT_FALSE (info.verify_model_path);
  EXPECT_NE (info.statistics, nullptr);
  EXPECT_EQ (info.accl_default, ACCL_CPU);
#if defined(__aarch64__) || defined(__arm__)
  /** as in V0, NEON is listed on ARM; tensor_filter drops it where the CPU lacks it */
  const gboolean neon = TRUE;
#else
  const gboolean neon = FALSE;
#endif
  EXPECT_EQ (info.accl_auto, neon ? ACCL_CPU_NEON : ACCL_CPU);

  ASSERT_NE (info.hw_list, nullptr);
  for (idx = 0; idx < info.num_hw; idx++) {
    switch (info.hw_list[idx]) {
      case ACCL_CPU:
        has_cpu = TRUE;
        break;
      case ACCL_CPU_NEON:
        has_neon = TRUE;
        break;
      case ACCL_GPU:
        has_gpu = TRUE;
        break;
      case ACCL_NPU:
        has_npu = TRUE;
        break;
      case ACCL_NPU_SRCN:
        has_srcn = TRUE;
        break;
      default:
        ADD_FAILURE () << "Unexpected accelerator " << info.hw_list[idx];
        break;
    }
  }

  EXPECT_TRUE (has_cpu && has_gpu && has_npu && has_srcn);
  EXPECT_EQ (has_neon, neon);
  EXPECT_EQ (info.num_hw, neon ? 5 : 4);
}

/**
 * @brief Test the framework info query without a place to write it
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, frameworkInfo_n)
{
  GstTensorFilterProperties prop;
  const gchar *model_files[] = { NULL };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  ASSERT_NE (sp, (void *) NULL);

  _SetFilterProp (&prop, "nnfw", model_files);
  EXPECT_NE (sp->getFrameworkInfo (sp, &prop, NULL, NULL), 0);
}

/**
 * @brief Test the accelerators tensor_filter finds available for nnfw
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, checkHwAvailability)
{
  ASSERT_NE (nnstreamer_filter_find ("nnfw"), nullptr);

  EXPECT_TRUE (gst_tensor_filter_check_hw_availability ("nnfw", ACCL_AUTO, NULL));
  EXPECT_TRUE (gst_tensor_filter_check_hw_availability ("nnfw", ACCL_DEFAULT, NULL));
  EXPECT_TRUE (gst_tensor_filter_check_hw_availability ("nnfw", ACCL_CPU, NULL));
  EXPECT_TRUE (gst_tensor_filter_check_hw_availability ("nnfw", ACCL_GPU, NULL));
  EXPECT_TRUE (gst_tensor_filter_check_hw_availability ("nnfw", ACCL_NPU, NULL));
  EXPECT_TRUE (gst_tensor_filter_check_hw_availability ("nnfw", ACCL_NPU_SRCN, NULL));

#if defined(__aarch64__) || defined(__arm__)
  /** listed on ARM, as V0's table did, whether or not the CPU has it */
  EXPECT_TRUE (gst_tensor_filter_check_hw_availability ("nnfw", ACCL_CPU_NEON, NULL));
#endif

  /** nnfw takes no custom option to check, which keeps the result */
  EXPECT_TRUE (gst_tensor_filter_check_hw_availability ("nnfw", ACCL_CPU, "any"));
}

/**
 * @brief Test the accelerators tensor_filter finds unavailable for nnfw
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, checkHwAvailability_n)
{
  ASSERT_NE (nnstreamer_filter_find ("nnfw"), nullptr);

#if !defined(__aarch64__) && !defined(__arm__)
  /** ACCL_CPU_NEON is cpu.simd here, which nnfw does not offer */
  EXPECT_FALSE (gst_tensor_filter_check_hw_availability ("nnfw", ACCL_CPU_SIMD, NULL));
#endif
  EXPECT_FALSE (gst_tensor_filter_check_hw_availability ("nnfw", ACCL_NPU_VIVANTE, NULL));
  EXPECT_FALSE (gst_tensor_filter_check_hw_availability ("nnfw", ACCL_NPU_EDGE_TPU, NULL));
  EXPECT_FALSE (gst_tensor_filter_check_hw_availability ("nnfw", ACCL_NPU_MOVIDIUS, NULL));
  EXPECT_FALSE (gst_tensor_filter_check_hw_availability ("nnfw", ACCL_NPU_SLSI, NULL));
}

/**
 * @brief Test nnfw declines every event, as it has no event to handle
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, eventHandler_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  GstTensorFilterProperties prop;
  GstTensorFilterFrameworkEventData event_data;
  const event_ops ops[] = { DESTROY_NOTIFY, RELOAD_MODEL, CUSTOM_PROP, SET_INPUT_PROP,
    SET_OUTPUT_PROP, SET_ACCELERATOR, CHECK_HW_AVAILABILITY, SUSPEND, RESUME };
  guint i;

  model_file = get_model_file ();
  ASSERT_TRUE (model_file != nullptr);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  ASSERT_NE (sp, (void *) NULL);
  _SetFilterProp (&prop, "nnfw", model_files);

  /** tensor_filter probes RELOAD_MODEL this way for the is-updatable property */
  EXPECT_EQ (sp->eventHandler (sp, &prop, NULL, RELOAD_MODEL, NULL), -ENOENT);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  for (i = 0; i < G_N_ELEMENTS (ops); i++) {
    memset (&event_data, 0, sizeof (event_data));
    event_data.model_files = model_files;
    event_data.num_models = 1;
    EXPECT_EQ (sp->eventHandler (sp, &prop, data, ops[i], &event_data), -ENOENT);
  }

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Test the model info query with invalid arguments
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, getModelInfo_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  GstTensorFilterProperties prop;
  GstTensorsInfo in_info, out_info;

  model_file = get_model_file ();
  ASSERT_TRUE (model_file != nullptr);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  ASSERT_NE (sp, (void *) NULL);
  _SetFilterProp (&prop, "nnfw", model_files);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  /** no model opened */
  ret = sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  ret = sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, NULL, &out_info);
  EXPECT_NE (ret, 0);
  ret = sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in_info, NULL);
  EXPECT_NE (ret, 0);
  ret = sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, NULL, &out_info);
  EXPECT_NE (ret, 0);
  ret = sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, &in_info, NULL);
  EXPECT_NE (ret, 0);

  /** an operation nnfw does not know */
  ret = sp->getModelInfo (sp, &prop, data,
      (model_info_ops) (SET_INPUT_INFO + 1), &in_info, &out_info);
  EXPECT_EQ (ret, -ENOENT);

  /** a different number of input tensors */
  ret = sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);
  in_info.num_tensors = 2;
  in_info.info[1].type = _NNS_FLOAT32;
  in_info.info[1].dimension[0] = 1;
  ret = sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);

  sp->close (&prop, &data);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  g_free (model_file);
}

/**
 * @brief Test invoke with no model opened and with no tensor memory
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, invoke_n)
{
  int ret;
  void *data = NULL;
  float in_val = 1.0, out_val = 0.0;
  GstTensorMemory input, output;
  gchar *model_file;
  GstTensorFilterProperties prop;

  model_file = get_model_file ();
  ASSERT_TRUE (model_file != nullptr);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  ASSERT_NE (sp, (void *) NULL);
  _SetFilterProp (&prop, "nnfw", model_files);

  input.data = &in_val;
  output.data = &out_val;
  input.size = output.size = sizeof (float);

  ret = sp->invoke (sp, &prop, NULL, &input, &output);
  EXPECT_NE (ret, 0);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  ret = sp->invoke (sp, &prop, data, NULL, &output);
  EXPECT_NE (ret, 0);
  ret = sp->invoke (sp, &prop, data, &input, NULL);
  EXPECT_NE (ret, 0);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Test the shared statistics count every invoke
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, statistics)
{
  int ret;
  void *data = NULL;
  float in_val = 1.0, out_val = 0.0;
  GstTensorMemory input, output;
  GstTensorFilterFrameworkInfo info;
  gchar *model_file;
  GstTensorFilterProperties prop;
  int64_t invoked;

  model_file = get_model_file ();
  ASSERT_TRUE (model_file != nullptr);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  ASSERT_NE (sp, (void *) NULL);
  _SetFilterProp (&prop, "nnfw", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  memset (&info, 0, sizeof (info));
  EXPECT_EQ (sp->getFrameworkInfo (sp, &prop, data, &info), 0);
  ASSERT_NE (info.statistics, nullptr);
  invoked = info.statistics->total_invoke_num;

  input.data = &in_val;
  output.data = &out_val;
  input.size = output.size = sizeof (float);

  EXPECT_EQ (sp->invoke (sp, &prop, data, &input, &output), 0);
  EXPECT_EQ (sp->invoke (sp, &prop, data, &input, &output), 0);
  EXPECT_FLOAT_EQ (out_val, 3.0);

  EXPECT_EQ (info.statistics->total_invoke_num, invoked + 2);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Open a model with the given accelerators and custom option, then check it adds 2.
 * @return the return value of open()
 */
static int
_open_and_invoke (accl_hw *hw_list, int num_hw, const gchar *custom)
{
  int ret;
  void *data = NULL;
  float in_val = 5.0, out_val = 0.0;
  GstTensorMemory input, output;
  gchar *model_file;
  GstTensorFilterProperties prop;

  model_file = get_model_file ();
  if (model_file == NULL)
    return -ENOENT;

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("nnfw");
  if (sp == NULL) {
    g_free (model_file);
    return -ENOENT;
  }

  _SetFilterProp (&prop, "nnfw", model_files);
  prop.hw_list = hw_list;
  prop.num_hw = num_hw;
  prop.custom_properties = custom;

  ret = sp->open (&prop, &data);
  if (ret == 0) {
    input.data = &in_val;
    output.data = &out_val;
    input.size = output.size = sizeof (float);

    EXPECT_EQ (sp->invoke (sp, &prop, data, &input, &output), 0);
    EXPECT_FLOAT_EQ (out_val, 7.0);
  } else {
    EXPECT_EQ (data, nullptr);
  }

  sp->close (&prop, &data);
  g_free (model_file);
  return ret;
}

/**
 * @brief Test open with no accelerator list, a list, acceleration turned off, and the custom option
 * @note All of them end up on the cpu backend; which backend each accelerator maps to is not observable here.
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, openAccelerator)
{
  accl_hw cpu[] = { ACCL_CPU };
  accl_hw none[] = { ACCL_NONE };

  /** no accelerator given: the default (cpu) backend */
  EXPECT_EQ (_open_and_invoke (NULL, 0, NULL), 0);
  EXPECT_EQ (_open_and_invoke (cpu, 1, NULL), 0);
  /** acceleration turned off: the default backend as well */
  EXPECT_EQ (_open_and_invoke (none, 1, NULL), 0);
  /** the custom option overrides the accelerator */
  EXPECT_EQ (_open_and_invoke (none, 1, "Runtime:cpu"), 0);
  /** an unknown custom option is skipped */
  EXPECT_EQ (_open_and_invoke (cpu, 1, "Unknown:value"), 0);
}

/**
 * @brief Test open fails with a backend nnfw does not have
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, openAccelerator_n)
{
  accl_hw cpu[] = { ACCL_CPU };

  EXPECT_NE (_open_and_invoke (cpu, 1, "Runtime:no_such_backend"), 0);
}

/**
 * @brief Make a tensor_filter running add.tflite with nnfw on the given accelerator, wrapped in a harness.
 * @return the harness, NULL if the model is missing
 */
static GstHarness *
_filter_harness (const gchar *accelerator)
{
  GstElement *filter;
  GstHarness *h;
  gchar *model_file = get_model_file ();

  if (model_file == NULL)
    return NULL;

  filter = gst_element_factory_make ("tensor_filter", NULL);
  if (filter == NULL) {
    g_free (model_file);
    return NULL;
  }

  g_object_set (filter, "framework", "nnfw", "model", model_file, "accelerator",
      accelerator, "custom", "Runtime:cpu", NULL);
  g_free (model_file);

  h = gst_harness_new_with_element (filter, "sink", "src");
  gst_object_unref (filter);

  gst_harness_set_src_caps_str (h,
      "other/tensors,num_tensors=1,types=float32,"
      "dimensions=1:1:1:1,format=static,framerate=(fraction)0/1");
  return h;
}

/**
 * @brief Push one float into the harness and check tensor_filter adds 2.
 */
static void
_push_and_check (GstHarness *h, float value)
{
  GstBuffer *in_buf, *out_buf;
  GstMapInfo map;

  in_buf = gst_buffer_new_allocate (NULL, sizeof (float), NULL);
  ASSERT_TRUE (gst_buffer_map (in_buf, &map, GST_MAP_WRITE));
  *((float *) map.data) = value;
  gst_buffer_unmap (in_buf, &map);

  ASSERT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  out_buf = gst_harness_pull (h);
  ASSERT_NE (out_buf, nullptr);
  ASSERT_TRUE (gst_buffer_map (out_buf, &map, GST_MAP_READ));
  ASSERT_EQ (map.size, sizeof (float));
  EXPECT_FLOAT_EQ (*((float *) map.data), value + 2);
  gst_buffer_unmap (out_buf, &map);
  gst_buffer_unref (out_buf);
}

/**
 * @brief Test tensor_filter runs nnfw through the V1 callbacks
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, tensorFilter)
{
  GstHarness *h;
  gchar *str = NULL;
  gboolean updatable = TRUE;

  h = _filter_harness ("true:cpu");
  ASSERT_NE (h, nullptr);

  _push_and_check (h, 10.0);

  /** tensor_filter reads back the string as set, as it does for V0 */
  g_object_get (h->element, "accelerator", &str, NULL);
  EXPECT_STREQ (str, "true:cpu");
  g_free (str);

  /** nnfw cannot reload a model */
  g_object_set (h->element, "is-updatable", TRUE, NULL);
  g_object_get (h->element, "is-updatable", &updatable, NULL);
  EXPECT_FALSE (updatable);

  _push_and_check (h, 1.0);
  gst_harness_teardown (h);
}

/**
 * @brief Test nnfw keeps its accelerator and custom option once it runs
 */
TEST (nnstreamerNnfwRuntimeRawFunctions, tensorFilterRunning_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = _filter_harness ("true:cpu");
  ASSERT_NE (h, nullptr);

  _push_and_check (h, 10.0);

  /** tensor_filter sends SET_ACCELERATOR and CUSTOM_PROP, which nnfw declines */
  g_object_set (h->element, "accelerator", "true:gpu", NULL);
  g_object_get (h->element, "accelerator", &str, NULL);
  EXPECT_STREQ (str, "true:cpu");
  g_free (str);

  g_object_set (h->element, "custom", "Runtime:acl_cl", NULL);
  g_object_get (h->element, "custom", &str, NULL);
  EXPECT_STREQ (str, "Runtime:cpu");
  g_free (str);

  _push_and_check (h, 1.0);
  gst_harness_teardown (h);
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
    gst_init (&argc, &argv);
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
