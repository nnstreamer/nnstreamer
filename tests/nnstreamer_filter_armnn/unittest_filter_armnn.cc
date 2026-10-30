/**
 * @file        unittest_filter_armnn.cc
 * @date        13 Dec 2019
 * @brief       Unit test for armnn tensor filter plugin.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <fcntl.h>
#include <glib.h>
#include <glib/gstdio.h> /* GStatBuf */
#include <sys/stat.h>
#include <unistd.h>

#include <armnn/ArmNN.hpp>
#include <gst/check/gstharness.h>
#include <gst/gst.h>

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
 * @brief Check the tensor info of add.tflite: one rank-1 float32 tensor
 */
static void
_ExpectAddModelInfo (GstTensorsInfo *info)
{
  EXPECT_EQ (info->num_tensors, 1U);
  EXPECT_EQ (info->info[0].type, _NNS_FLOAT32);
  /** unused dimensions are reported as 0 */
  EXPECT_EQ (info->info[0].dimension[0], 1U);
  EXPECT_EQ (info->info[0].dimension[1], 0U);
  EXPECT_EQ (info->info[0].dimension[2], 0U);
  EXPECT_EQ (info->info[0].dimension[3], 0U);
}

/**
 * @brief Test armnn subplugin existence.
 */
TEST (nnstreamerFilterArmnn, checkExistence)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  ASSERT_NE (sp, (void *) NULL);
  EXPECT_EQ (sp->version, GST_TENSOR_FILTER_FRAMEWORK_V1);
}

/**
 * @brief Test the framework info of armnn subplugin
 */
TEST (nnstreamerFilterArmnn, getFrameworkInfo)
{
  GstTensorFilterFrameworkInfo info;
  GstTensorFilterProperties prop;
  const gchar *model_files[] = { NULL };
  int ret;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  ASSERT_NE (sp, (void *) NULL);
  _SetFilterProp (&prop, "armnn", model_files);

  memset (&info, 0, sizeof (info));
  ret = sp->getFrameworkInfo (sp, &prop, NULL, &info);
  EXPECT_EQ (ret, 0);
  EXPECT_STREQ (info.name, "armnn");
  EXPECT_FALSE (info.allow_in_place);
  EXPECT_FALSE (info.allocate_in_invoke);
  EXPECT_FALSE (info.run_without_model);
  EXPECT_FALSE (info.verify_model_path);
  EXPECT_EQ (info.statistics, nullptr);

  /** the accelerators armnn accepted with v0 checkAvailability */
  ASSERT_NE (info.hw_list, nullptr);
#if defined(__aarch64__) || defined(__arm__)
  ASSERT_EQ (info.num_hw, 3);
  EXPECT_EQ (info.hw_list[0], ACCL_CPU_NEON);
  EXPECT_EQ (info.hw_list[1], ACCL_CPU);
  EXPECT_EQ (info.hw_list[2], ACCL_GPU);
  EXPECT_EQ (info.accl_auto, ACCL_CPU_NEON);
  EXPECT_EQ (info.accl_default, ACCL_CPU_NEON);
#else
  ASSERT_EQ (info.num_hw, 2);
  EXPECT_EQ (info.hw_list[0], ACCL_CPU);
  EXPECT_EQ (info.hw_list[1], ACCL_GPU);
  EXPECT_EQ (info.accl_auto, ACCL_CPU);
  EXPECT_EQ (info.accl_default, ACCL_CPU);
#endif
}

/**
 * @brief Test that armnn does not report accelerators it cannot use
 */
TEST (nnstreamerFilterArmnn, getFrameworkInfo_n)
{
  GstTensorFilterFrameworkInfo info;
  GstTensorFilterProperties prop;
  const gchar *model_files[] = { NULL };
  int idx, ret;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  ASSERT_NE (sp, (void *) NULL);
  _SetFilterProp (&prop, "armnn", model_files);

  memset (&info, 0, sizeof (info));
  ret = sp->getFrameworkInfo (sp, &prop, NULL, &info);
  EXPECT_EQ (ret, 0);
  EXPECT_GT (info.num_hw, 0);
  for (idx = 0; idx < info.num_hw; idx++) {
    EXPECT_NE (info.hw_list[idx], ACCL_NPU);
    EXPECT_NE (info.hw_list[idx], ACCL_NPU_VIVANTE);
#if !defined(__aarch64__) && !defined(__arm__)
    /** v0 never offered cpu.simd, the value cpu.neon has off arm */
    EXPECT_NE (info.hw_list[idx], ACCL_CPU_SIMD);
#endif
  }
}

/**
 * @brief Test armnn subplugin with failing open/close (no model file)
 */
TEST (nnstreamerFilterArmnn, openClose00_n)
{
  int ret;
  const gchar *model_files[] = {
    "temp.armnn",
    NULL,
  };
  GstTensorFilterProperties prop;
  void *data = NULL;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  _SetFilterProp (&prop, "armnn", model_files);
  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Test armnn subplugin with successful open/close
 */
TEST (nnstreamerFilterArmnn, openClose01_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);
  _SetFilterProp (&prop, "armnn", model_files);

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
 * @brief Get input/output dimensions with armnn subplugin
 */
TEST (nnstreamerFilterArmnn, getDimension)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorsInfo in_info, out_info;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** get input/output dimension successfully */
  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  ret = sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  _ExpectAddModelInfo (&in_info);
  _ExpectAddModelInfo (&out_info);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Get input/output dimensions before open
 */
TEST (nnstreamerFilterArmnn, getDimension1_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorsInfo in_info, out_info;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  /** get input/output dimension without open */
  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  ret = sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);

  /** the registered empty instance has no model either */
  ret = sp->getModelInfo (sp, &prop, sp->subplugin_data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  g_free (model_file);
}

/**
 * @brief Set the input dimension, which armnn does not support
 */
TEST (nnstreamerFilterArmnn, getDimension2_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorsInfo in_info, out_info;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };
  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  in_info.num_tensors = 1;
  in_info.info[0].type = _NNS_FLOAT32;
  in_info.info[0].dimension[0] = 2;
  ret = sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, -ENOENT);
  EXPECT_EQ (out_info.num_tensors, 0U);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Test armnn subplugin with successful invoke for tflite
 */
TEST (nnstreamerFilterArmnn, invoke00)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke successful */
  *((float *) input.data) = 10.0;
  ret = sp->invoke (sp, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (*((float *) output.data), 12.0);

  *((float *) input.data) = 1.0;
  ret = sp->invoke (sp, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (*((float *) output.data), 3.0);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Test armnn invoke before open
 */
TEST (nnstreamerFilterArmnn, invoke01_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  /** invoke before open */
  ret = sp->invoke (sp, &prop, data, &input, &output);
  EXPECT_NE (ret, 0);

  g_free (input.data);
  g_free (output.data);
  g_free (model_file);
}

/**
 * @brief Test armnn invoke with invalid param
 */
TEST (nnstreamerFilterArmnn, invoke02_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  _SetFilterProp (&prop, "armnn", model_files);
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke without subplugin data */
  ret = sp->invoke (sp, &prop, NULL, &input, &output);
  EXPECT_NE (ret, 0);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Test armnn invoke with invalid param
 */
TEST (nnstreamerFilterArmnn, invoke03_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorMemory output;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };
  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  output.size = sizeof (float) * 1;

  output.data = g_malloc (output.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke without input */
  ret = sp->invoke (sp, &prop, data, NULL, &output);
  EXPECT_NE (ret, 0);

  g_free (output.data);
  sp->close (&prop, &data);

  g_free (model_file);
}

/**
 * @brief Test armnn invoke with invalid param
 */
TEST (nnstreamerFilterArmnn, invoke04_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorMemory input;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };
  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke without output */
  ret = sp->invoke (sp, &prop, data, &input, NULL);
  EXPECT_NE (ret, 0);

  g_free (input.data);
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Get the path of add.tflite, which adds 2 to a float32 scalar
 */
static gchar *
_GetAddModelPath (void)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  if (root_path == NULL)
    return NULL;

  return g_build_filename (root_path, "tests", "test_models", "models", "add.tflite", NULL);
}

/**
 * @brief Open add.tflite with the given accelerator and check one inference
 */
static void
_OpenAndInvokeAdd (const GstTensorFilterFramework *sp, accl_hw hw)
{
  int ret;
  void *data = NULL;
  float in_val = 10.0f, out_val = 0.0f;
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;
  gchar *model_file = _GetAddModelPath ();
  const gchar *model_files[] = { model_file, NULL };
  accl_hw hw_list[] = { hw };

  ASSERT_NE (model_file, nullptr);
  _SetFilterProp (&prop, "armnn", model_files);
  prop.hw_list = hw_list;
  prop.num_hw = 1;

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0) << "accelerator " << get_accl_hw_str (hw);

  input.data = &in_val;
  input.size = sizeof (float);
  output.data = &out_val;
  output.size = sizeof (float);
  ret = sp->invoke (sp, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0) << "accelerator " << get_accl_hw_str (hw);
  EXPECT_FLOAT_EQ (out_val, 12.0f) << "accelerator " << get_accl_hw_str (hw);

  sp->close (&prop, &data);
  EXPECT_EQ (data, nullptr);
  g_free (model_file);
}

/**
 * @brief Test armnn subplugin with the accelerator list given by tensor_filter
 */
TEST (nnstreamerFilterArmnn, openAccelerator)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  ASSERT_NE (sp, (void *) NULL);

  /** cpu and none run on the reference backend */
  _OpenAndInvokeAdd (sp, ACCL_CPU);
  _OpenAndInvokeAdd (sp, ACCL_NONE);
  /** cpu.neon, auto and default fall back to cpu without NEON */
  _OpenAndInvokeAdd (sp, ACCL_CPU_NEON);
  _OpenAndInvokeAdd (sp, ACCL_AUTO);
  _OpenAndInvokeAdd (sp, ACCL_DEFAULT);
}

/**
 * @brief Test armnn subplugin with an accelerator that armnn does not have
 */
TEST (nnstreamerFilterArmnn, openAcceleratorFallback)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  ASSERT_NE (sp, (void *) NULL);

  /** an unsupported accelerator is not passed to armnn as a backend */
  _OpenAndInvokeAdd (sp, ACCL_NPU);
  _OpenAndInvokeAdd (sp, ACCL_NPU_VIVANTE);
}

/**
 * @brief Check if armnn has the GpuAcc backend on this system
 */
static bool
_HasGpuBackend (void)
{
  armnn::IRuntime::CreationOptions options;
  armnn::IRuntimePtr runtime = armnn::IRuntime::Create (options);

  if (!runtime)
    return false;

  return runtime->GetDeviceSpec ().GetSupportedBackends ().count (armnn::Compute::GpuAcc) > 0;
}

/**
 * @brief Test that the gpu accelerator selects the GpuAcc backend of armnn
 */
TEST (nnstreamerFilterArmnn, openAcceleratorGpu_n)
{
  int ret;
  void *data = NULL;
  GstTensorFilterProperties prop;
  gchar *model_file = _GetAddModelPath ();
  const gchar *model_files[] = { model_file, NULL };
  accl_hw hw_list[] = { ACCL_GPU };
  const bool has_gpu = _HasGpuBackend ();
  ASSERT_NE (model_file, nullptr);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  ASSERT_NE (sp, (void *) NULL);

  _SetFilterProp (&prop, "armnn", model_files);
  prop.hw_list = hw_list;
  prop.num_hw = 1;

  /** without the GpuAcc backend, as in CI, the model cannot be loaded */
  ret = sp->open (&prop, &data);
  if (has_gpu) {
    EXPECT_EQ (ret, 0);
  } else {
    EXPECT_EQ (ret, -EINVAL);
    EXPECT_EQ (data, nullptr);
  }

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Test armnn subplugin open without a model
 */
TEST (nnstreamerFilterArmnn, openNoModel_n)
{
  int ret;
  void *data = NULL;
  const gchar *model_files[] = { NULL };
  GstTensorFilterProperties prop;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  ASSERT_NE (sp, (void *) NULL);

  _SetFilterProp (&prop, "armnn", model_files);
  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, -EINVAL);
  EXPECT_EQ (data, nullptr);

  prop.model_files = NULL;
  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, -EINVAL);
  EXPECT_EQ (data, nullptr);
}

/**
 * @brief Test armnn subplugin open with models armnn cannot load
 */
TEST (nnstreamerFilterArmnn, openInvalidModel_n)
{
  int ret, fd;
  void *data = NULL;
  gchar *tmp_file = NULL;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  ASSERT_NE (sp, (void *) NULL);

  /** a directory is not a model file */
  const gchar *dir_files[] = { root_path, NULL };
  _SetFilterProp (&prop, "armnn", dir_files);
  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, -EINVAL);
  EXPECT_EQ (data, nullptr);

  /** a regular file of a format armnn has no parser for */
  fd = g_file_open_tmp ("nns-armnn-XXXXXX.onnx", &tmp_file, NULL);
  ASSERT_GE (fd, 0);
  close (fd);

  const gchar *onnx_files[] = { tmp_file, NULL };
  _SetFilterProp (&prop, "armnn", onnx_files);
  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, -EINVAL);
  EXPECT_EQ (data, nullptr);

  g_remove (tmp_file);
  g_free (tmp_file);
}

/**
 * @brief Test armnn invoke with tensors of a wrong size
 */
TEST (nnstreamerFilterArmnn, invokeWrongSize_n)
{
  int ret;
  void *data = NULL;
  float in_val[2] = { 10.0f, 0.0f }, out_val[2] = { 0.0f, 0.0f };
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;
  gchar *model_file = _GetAddModelPath ();
  const gchar *model_files[] = { model_file, NULL };
  ASSERT_NE (model_file, nullptr);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  ASSERT_NE (sp, (void *) NULL);

  _SetFilterProp (&prop, "armnn", model_files);
  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  input.data = in_val;
  input.size = sizeof (in_val);
  output.data = out_val;
  output.size = sizeof (float);
  ret = sp->invoke (sp, &prop, data, &input, &output);
  EXPECT_EQ (ret, -EINVAL);

  input.size = sizeof (float);
  output.size = sizeof (out_val);
  ret = sp->invoke (sp, &prop, data, &input, &output);
  EXPECT_EQ (ret, -EINVAL);
  EXPECT_FLOAT_EQ (out_val[0], 0.0f);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Test that armnn handles no event, as with v0
 */
TEST (nnstreamerFilterArmnn, eventHandler_n)
{
  int ret;
  void *data = NULL;
  GstTensorFilterProperties prop;
  GstTensorFilterFrameworkEventData event;
  gchar *model_file = _GetAddModelPath ();
  const gchar *model_files[] = { model_file, NULL };
  ASSERT_NE (model_file, nullptr);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  ASSERT_NE (sp, (void *) NULL);

  _SetFilterProp (&prop, "armnn", model_files);
  memset (&event, 0, sizeof (event));

  /** before open */
  ret = sp->eventHandler (sp, &prop, data, RELOAD_MODEL, &event);
  EXPECT_NE (ret, 0);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** tensor_filter probes RELOAD_MODEL with NULL to set is-updatable */
  ret = sp->eventHandler (sp, &prop, data, RELOAD_MODEL, NULL);
  EXPECT_EQ (ret, -ENOENT);

  event.model_files = model_files;
  event.num_models = 1;
  ret = sp->eventHandler (sp, &prop, data, RELOAD_MODEL, &event);
  EXPECT_EQ (ret, -ENOENT);

  memset (&event, 0, sizeof (event));
  ret = sp->eventHandler (sp, &prop, data, CUSTOM_PROP, &event);
  EXPECT_EQ (ret, -ENOENT);
  ret = sp->eventHandler (sp, &prop, data, SET_ACCELERATOR, &event);
  EXPECT_EQ (ret, -ENOENT);
  ret = sp->eventHandler (sp, &prop, data, DESTROY_NOTIFY, &event);
  EXPECT_EQ (ret, -ENOENT);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Create a harness of tensor_filter with armnn and add.tflite
 */
static GstHarness *
_CreateAddHarness (const gchar *model_file, const gchar *accelerator)
{
  GstHarness *h;
  gchar *desc;

  desc = g_strdup_printf ("tensor_filter framework=armnn model=%s accelerator=%s",
      model_file, accelerator);
  h = gst_harness_new_parse (desc);
  g_free (desc);

  gst_harness_set_src_caps_str (h,
      "other/tensors,num_tensors=(int)1,dimensions=(string)1,types=(string)float32,format=(string)static,framerate=(fraction)0/1");
  return h;
}

/**
 * @brief Test armnn subplugin in a pipeline, through tensor_filter
 */
TEST (nnstreamerFilterArmnn, pipelineInvoke)
{
  GstHarness *h;
  GstElement *filter;
  GstBuffer *in_buf, *out_buf;
  GstMapInfo map;
  gchar *str = NULL;
  gboolean updatable = TRUE;
  float in_val = 10.0f;
  gchar *model_file = _GetAddModelPath ();
  ASSERT_NE (model_file, nullptr);

  h = _CreateAddHarness (model_file, "true:cpu");

  in_buf = gst_buffer_new_allocate (NULL, sizeof (float), NULL);
  gst_buffer_fill (in_buf, 0, &in_val, sizeof (float));
  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  out_buf = gst_harness_pull (h);
  ASSERT_NE (out_buf, nullptr);
  ASSERT_TRUE (gst_buffer_map (out_buf, &map, GST_MAP_READ));
  ASSERT_EQ (map.size, sizeof (float));
  EXPECT_FLOAT_EQ (*((float *) map.data), 12.0f);
  gst_buffer_unmap (out_buf, &map);
  gst_buffer_unref (out_buf);

  filter = gst_harness_find_element (h, "tensor_filter");
  ASSERT_NE (filter, nullptr);
  g_object_get (filter, "accelerator", &str, NULL);
  EXPECT_STREQ (str, "true:cpu");
  g_free (str);

  /** armnn refuses SET_ACCELERATOR, so the opened filter keeps true:cpu */
  g_object_set (filter, "accelerator", "true:gpu", NULL);
  g_object_get (filter, "accelerator", &str, NULL);
  EXPECT_STREQ (str, "true:cpu");
  g_free (str);

  /** armnn cannot reload a model, so the filter stays not updatable */
  g_object_set (filter, "is-updatable", TRUE, NULL);
  g_object_get (filter, "is-updatable", &updatable, NULL);
  EXPECT_FALSE (updatable);

  gst_object_unref (filter);
  gst_harness_teardown (h);
  g_free (model_file);
}

/**
 * @brief Test armnn subplugin in a pipeline with a model it cannot open
 */
TEST (nnstreamerFilterArmnn, pipelineInvoke_n)
{
  GstElement *filter;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *model_file;
  ASSERT_NE (root_path, nullptr);

  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "no-such-model.tflite", NULL);
  filter = gst_element_factory_make ("tensor_filter", NULL);
  ASSERT_NE (filter, nullptr);
  g_object_set (filter, "framework", "armnn", "model", model_file, NULL);

  EXPECT_EQ (gst_element_set_state (filter, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);

  gst_element_set_state (filter, GST_STATE_NULL);
  gst_object_unref (filter);
  g_free (model_file);
}

/**
 * @brief Test that the accelerator property of tensor_filter reaches armnn
 */
TEST (nnstreamerFilterArmnn, pipelineAcceleratorGpu_n)
{
  GstElement *filter;
  GstStateChangeReturn ret;
  const bool has_gpu = _HasGpuBackend ();
  gchar *model_file = _GetAddModelPath ();
  ASSERT_NE (model_file, nullptr);

  filter = gst_element_factory_make ("tensor_filter", NULL);
  ASSERT_NE (filter, nullptr);
  g_object_set (filter, "framework", "armnn", "model", model_file,
      "accelerator", "true:gpu", NULL);

  /** without the GpuAcc backend, as in CI, the model cannot be loaded */
  ret = gst_element_set_state (filter, GST_STATE_PAUSED);
  if (has_gpu)
    EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);
  else
    EXPECT_EQ (ret, GST_STATE_CHANGE_FAILURE);

  gst_element_set_state (filter, GST_STATE_NULL);
  gst_object_unref (filter);
  g_free (model_file);
}

/**
 * @brief get argmax from the array
 */
template <typename T>
size_t
get_argmax (T *array, size_t size)
{
  size_t idx, max_idx = 0;
  T max_value;

  if (size == 0)
    return 0;

  max_value = array[0];
  for (idx = 1; idx < size; idx++) {
    if (max_value < array[idx]) {
      max_idx = idx;
      max_value = array[idx];
    }
  }

  return max_idx;
}

/**
 * @brief Test get_argmax with all-negative values (e.g. raw logits)
 */
TEST (nnstreamerFilterArmnn, argmaxAllNegative)
{
  float values[] = { -3.0f, -1.5f, -0.25f, -2.0f };

  EXPECT_EQ (get_argmax<float> (values, 4), 2U);
  EXPECT_EQ (get_argmax<float> (values, 2), 1U);
  EXPECT_EQ (get_argmax<float> (values, 0), 0U);
}

/**
 * @brief Test armnn subplugin with successful invoke for tflite advanced model
 */
TEST (nnstreamerFilterArmnn, invokeAdvanced)
{
  int ret, fd;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorMemory input, output;
  GstTensorsInfo info, res, other;
  char *data_file;
  ssize_t data_read;
  size_t max_idx;
  gboolean status;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  status = g_file_test (model_file, G_FILE_TEST_EXISTS);
  if (!status) {
    g_free (model_file);
    ASSERT_EQ (status, TRUE);
    return;
  }

  const gchar *model_files[] = {
    model_file,
    NULL,
  };
  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  info.num_tensors = 1;
  info.info[0].type = _NNS_UINT8;
  info.info[0].dimension[0] = 3;
  info.info[0].dimension[1] = 224;
  info.info[0].dimension[2] = 224;
  info.info[0].dimension[3] = 1;

  ret = sp->open (&prop, &data);
  if (ret == -EPERM) { /** TFLite Parser is not included in this instance */
    g_free (model_file);
    return;
  }
  EXPECT_EQ (ret, 0);

  /** get input/output dimension successfully */
  ret = sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &res, &other);
  EXPECT_EQ (ret, 0);
  gst_tensors_info_free (&other);

  EXPECT_EQ (res.num_tensors, info.num_tensors);
  EXPECT_EQ (res.info[0].type, info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], info.info[0].dimension[3]);

  input.size = gst_tensor_info_get_size (&res.info[0]);
  gst_tensors_info_free (&res);

  ret = sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &other, &res);
  EXPECT_EQ (ret, 0);
  gst_tensors_info_free (&other);

  /** the output tensor is rank 2; unused dimensions are reported as 0 */
  info.num_tensors = 1;
  info.info[0].type = _NNS_UINT8;
  info.info[0].dimension[0] = 1001;
  info.info[0].dimension[1] = 1;
  info.info[0].dimension[2] = 0;
  info.info[0].dimension[3] = 0;

  EXPECT_EQ (res.num_tensors, info.num_tensors);
  EXPECT_EQ (res.info[0].type, info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], info.info[0].dimension[3]);

  output.size = gst_tensor_info_get_size (&res.info[0]);
  gst_tensors_info_free (&res);

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  data_file = g_build_filename (
      root_path, "tests", "test_models", "data", "orange.raw", NULL);
  fd = open (data_file, O_RDONLY);

  EXPECT_TRUE (fd >= 0);
  /** Invoke output will not match with fd < 0 - input will be random data */
  if (fd >= 0) {
    data_read = read (fd, input.data, input.size);
    EXPECT_EQ (data_read, input.size);
    close (fd);
  }

  ret = sp->invoke (sp, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);

  /** entry 952 (idx 951) is orange as per tests/test_models/labels/labels.txt
   */
  max_idx = get_argmax<guint8> ((guint8 *) output.data, output.size);
  EXPECT_EQ (max_idx, 951);

  g_free (data_file);
  g_free (output.data);
  g_free (input.data);
  g_free (model_file);
  sp->close (&prop, &data);
}

/**
 * @brief Test armnn subplugin with successful invoke for caffe
 */
TEST (nnstreamerFilterArmnn, invoke01)
{
  int ret;
  void *data = NULL;
  gchar *model_file, *data_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *input_uint8_data = NULL;
  gsize input_uint8_size = 0;
  GstTensorMemory output, input;
  ssize_t max_idx;
  const unsigned int num_labels = 10;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (root_path, "tests", "test_models", "models",
      "lenet_iter_9000.caffemodel", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  data_file = g_build_filename (root_path, "tests", "test_models", "data", "9.raw", NULL);
  ASSERT_TRUE (g_file_test (data_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };
  _SetFilterProp (&prop, "armnn", model_files);

  /** Manually configure the input for test */
  gst_tensors_info_init (&prop.input_meta);
  gst_tensors_info_init (&prop.output_meta);
  prop.output_meta.num_tensors = 1;
  prop.output_meta.info[0].type = _NNS_FLOAT32;
  prop.output_meta.info[0].name = g_strdup ("prob");
  prop.input_meta.num_tensors = 1;
  prop.input_meta.info[0].name = g_strdup ("data");

  EXPECT_TRUE (g_file_get_contents (data_file, &input_uint8_data, &input_uint8_size, NULL));

  /** Convert the data from uint8 to float */
  input.size = input_uint8_size * gst_tensor_get_element_size (_NNS_FLOAT32);
  input.data = g_malloc (input.size);
  for (gsize idx = 0; idx < input_uint8_size; idx++) {
    ((float *) input.data)[idx]
        = static_cast<float> (((guint8 *) input_uint8_data)[idx]);
    ((float *) input.data)[idx] -= 127.5;
    ((float *) input.data)[idx] /= 127.5;
  }

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *) NULL);

  output.size = gst_tensor_get_element_size (_NNS_FLOAT32) * num_labels;
  output.data = g_malloc (output.size);

  ret = sp->open (&prop, &data);
  if (ret == -EPERM) { /** Caffe Parser Not Included in This System */
    g_free (data_file);
    g_free (input.data);
    g_free (input_uint8_data);
    g_free (output.data);
    g_free (prop.output_meta.info[0].name);
    g_free (prop.input_meta.info[0].name);
    g_free (model_file);
    return;
  }
  EXPECT_EQ (ret, 0);

  /** invoke successful */
  ret = sp->invoke (sp, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);

  max_idx = get_argmax<float> ((float *) output.data, num_labels);
  EXPECT_EQ (max_idx, 9);

  sp->close (&prop, &data);

  /** Run the test again but with setting input meta as well */
  gst_tensors_info_init (&prop.input_meta);
  gst_tensors_info_init (&prop.output_meta);
  prop.output_meta.num_tensors = 1;
  prop.output_meta.info[0].type = _NNS_FLOAT32;
  prop.output_meta.info[0].name = g_strdup ("prob");

  prop.input_meta.num_tensors = 1;
  prop.input_meta.info[0].type = _NNS_FLOAT32;
  prop.input_meta.info[0].name = g_strdup ("data");
  prop.input_meta.info[0].dimension[0] = 28;
  prop.input_meta.info[0].dimension[1] = 28;
  prop.input_meta.info[0].dimension[2] = 1;
  prop.input_meta.info[0].dimension[3] = 1;

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke successful */
  ret = sp->invoke (sp, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);

  max_idx = get_argmax<float> ((float *) output.data, num_labels);
  EXPECT_EQ (max_idx, 9);

  sp->close (&prop, &data);

  g_free (data_file);
  g_free (input.data);
  g_free (input_uint8_data);
  g_free (output.data);
  g_free (prop.output_meta.info[0].name);
  g_free (prop.input_meta.info[0].name);
  g_free (model_file);
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
