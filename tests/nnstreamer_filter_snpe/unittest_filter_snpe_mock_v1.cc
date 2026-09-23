/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_snpe_mock_v1.cc
 * @date    22 Sep 2026
 * @brief   Unit tests of the SNPE 1.x tensor_filter sub-plugin on the mock SDK.
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 * These cases run beside the ones of unittest_filter_snpe.cc, which is linked
 * into the same binary and covers the successful paths. What is added here are
 * the paths a resizable output tensor reaches, which no in-tree model has.
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

#include "mock/snpe_mock.h"

/** @brief Bytes of guard placed after an output buffer. */
#define GUARD_SIZE 64
/** @brief Byte the guard is filled with. */
#define GUARD_BYTE 0xA5

/**
 * @brief Set the filter properties of a case.
 */
static void
_MockSetProp (GstTensorFilterProperties *prop, const gchar **models, const gchar *custom)
{
  memset (prop, 0, sizeof (GstTensorFilterProperties));
  prop->fwname = "snpe";
  prop->fw_opened = 0;
  prop->model_files = models;
  prop->num_models = g_strv_length ((gchar **) models);
  prop->custom_properties = custom;
}

/**
 * @brief Build the path of the sample float model.
 */
static gchar *
_MockModelPath (void)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
  gchar *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add2_float.v1.dlc", NULL);

  g_free (root_path);
  return model_file;
}

/**
 * @brief Create an empty file the mock reads a model description from.
 * @return the path of the created file, to be released by the caller
 */
static gchar *
_MockMakeModelFile (const gchar *name_template)
{
  gchar *path = NULL;
  gint fd = g_file_open_tmp (name_template, &path, NULL);

  if (fd < 0)
    return NULL;

  g_close (fd, NULL);
  return path;
}

/**
 * @brief Open the sample model with the given custom properties and close it.
 * @return the result of the open call
 */
static int
_MockOpenClose (const gchar *custom)
{
  void *data = NULL;
  GstTensorFilterProperties prop;
  gchar *model_file = _MockModelPath ();
  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  int ret;

  g_assert (sp != NULL);
  _MockSetProp (&prop, model_files, custom);
  ret = sp->open (&prop, &data);
  sp->close (&prop, &data);
  g_free (model_file);

  return ret;
}

/**
 * @brief Tell whether the guard placed after a buffer is untouched.
 */
static gboolean
_GuardIsIntact (const guint8 *buffer, gsize size)
{
  for (gsize i = 0; i < GUARD_SIZE; i++)
    if (buffer[size + i] != GUARD_BYTE)
      return FALSE;

  return TRUE;
}

/**
 * @brief Positive case: every custom property the sub-plugin accepts.
 */
TEST (nnstreamerFilterSnpeMockV1, customProp00)
{
  snpe_mock_reset ();

  EXPECT_EQ (_MockOpenClose ("Runtime:CPU,CPUFallback:true,OutputTensor:output,"
                             "InputType:float32,OutputType:float32,UserBuffer:false,"
                             "MaxResizableDim:4,Unknown:1"),
      0);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
  EXPECT_EQ (snpe_mock_over_release_count (), 0U);
}

/**
 * @brief Negative case: an unknown runtime name is rejected.
 */
TEST (nnstreamerFilterSnpeMockV1, customPropInvalidRuntime01_n)
{
  snpe_mock_reset ();

  EXPECT_NE (_MockOpenClose ("Runtime:TPU"), 0);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
}

/**
 * @brief Negative case: an unknown CPU fallback value is rejected.
 */
TEST (nnstreamerFilterSnpeMockV1, customPropInvalidCpuFallback02_n)
{
  snpe_mock_reset ();

  EXPECT_NE (_MockOpenClose ("CPUFallback:maybe"), 0);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
}

/**
 * @brief Negative case: an empty output tensor name is rejected.
 */
TEST (nnstreamerFilterSnpeMockV1, customPropEmptyOutputTensorName03_n)
{
  snpe_mock_reset ();

  EXPECT_NE (_MockOpenClose ("OutputTensor:output;;output"), 0);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
}

/**
 * @brief Negative case: a max resizable dim of zero is rejected.
 */
TEST (nnstreamerFilterSnpeMockV1, customPropZeroMaxResizableDim04_n)
{
  snpe_mock_reset ();

  EXPECT_NE (_MockOpenClose ("MaxResizableDim:0"), 0);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
}

/**
 * @brief Negative case: the SNPE instance cannot be built.
 */
TEST (nnstreamerFilterSnpeMockV1, buildFailure05_n)
{
  snpe_mock_reset ();
  snpe_mock_set_failure (SNPE_MOCK_FAIL_BUILD);

  EXPECT_NE (_MockOpenClose (NULL), 0);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
}

/**
 * @brief Negative case: the model file does not exist.
 */
TEST (nnstreamerFilterSnpeMockV1, missingModelFile06_n)
{
  void *data = NULL;
  GstTensorFilterProperties prop;
  const gchar *model_files[] = { "/tmp/nns_snpe_mock_v1_does_not_exist.dlc", NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);
  _MockSetProp (&prop, model_files, NULL);

  snpe_mock_reset ();
  EXPECT_NE (sp->open (&prop, &data), 0);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);

  sp->close (&prop, &data);
}

/**
 * @brief Positive case: output tensor names taken from the filter properties.
 */
TEST (nnstreamerFilterSnpeMockV1, outputMetaNames07)
{
  void *data = NULL;
  GstTensorFilterProperties prop;
  gchar *model_file = _MockModelPath ();
  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);
  _MockSetProp (&prop, model_files, NULL);
  prop.output_meta.num_tensors = 1;
  prop.output_meta.info[0].name = g_strdup (SNPE_MOCK_OUTPUT_NAME);

  snpe_mock_reset ();
  EXPECT_EQ (sp->open (&prop, &data), 0);

  sp->close (&prop, &data);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
  g_free (prop.output_meta.info[0].name);
  g_free (model_file);
}

/**
 * @brief Negative case: an empty output tensor name in the filter properties.
 */
TEST (nnstreamerFilterSnpeMockV1, outputMetaInvalidName08_n)
{
  void *data = NULL;
  GstTensorFilterProperties prop;
  gchar *model_file = _MockModelPath ();
  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);
  _MockSetProp (&prop, model_files, NULL);
  prop.output_meta.num_tensors = 1;
  prop.output_meta.info[0].name = g_strdup ("");

  snpe_mock_reset ();
  EXPECT_NE (sp->open (&prop, &data), 0);

  sp->close (&prop, &data);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
  g_free (prop.output_meta.info[0].name);
  g_free (model_file);
}

/**
 * @brief Positive case: a resizable model whose max dim covers the output.
 */
TEST (nnstreamerFilterSnpeMockV1, resizableOutputFits09)
{
  void *data = NULL;
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;
  GstTensorsInfo in_info, out_info;
  guint8 *guarded_output;
  gchar *custom = g_strdup_printf ("MaxResizableDim:%d", SNPE_MOCK_RESIZABLE_OUTPUT_ELEMENTS);
  gchar *model_file = _MockMakeModelFile ("nns_snpe_mock_resizable_XXXXXX.dlc");
  ASSERT_TRUE (model_file != NULL);

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);
  _MockSetProp (&prop, model_files, custom);

  snpe_mock_reset ();
  ASSERT_EQ (sp->open (&prop, &data), 0);
  ASSERT_EQ (sp->getModelInfo (NULL, NULL, data, GET_IN_OUT_INFO, &in_info, &out_info), 0);

  input.size = gst_tensor_info_get_size (&in_info.info[0]);
  output.size = gst_tensor_info_get_size (&out_info.info[0]);
  input.data = g_malloc0 (input.size);
  guarded_output = (guint8 *) g_malloc0 (output.size + GUARD_SIZE);
  memset (guarded_output + output.size, GUARD_BYTE, GUARD_SIZE);
  output.data = guarded_output;

  EXPECT_EQ (sp->invoke (NULL, NULL, data, &input, &output), 0);
  EXPECT_TRUE (_GuardIsIntact (guarded_output, output.size));
  EXPECT_EQ (((float *) guarded_output)[0], (float) SNPE_MOCK_ADDEND);

  g_free (input.data);
  g_free (guarded_output);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  sp->close (&prop, &data);
  g_remove (model_file);
  g_free (model_file);
  g_free (custom);
}

/**
 * @brief Positive case: the user buffer mode runs the emulated model.
 */
TEST (nnstreamerFilterSnpeMockV1, userBuffer10)
{
  void *data = NULL;
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;
  gchar *model_file = _MockModelPath ();
  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);
  _MockSetProp (&prop, model_files, "UserBuffer:true");

  snpe_mock_reset ();
  ASSERT_EQ (sp->open (&prop, &data), 0);

  output.size = input.size = sizeof (float);
  input.data = g_malloc0 (input.size);
  output.data = g_malloc0 (output.size);
  ((float *) input.data)[0] = 5.0f;

  EXPECT_EQ (sp->invoke (NULL, NULL, data, &input, &output), 0);
  EXPECT_EQ (((float *) output.data)[0], 5.0f + SNPE_MOCK_ADDEND);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
  g_free (model_file);
}

/**
 * @brief Negative case: the user buffer mode needs a resizable dim option.
 */
TEST (nnstreamerFilterSnpeMockV1, userBufferResizableWithoutMaxDim11_n)
{
  void *data = NULL;
  GstTensorFilterProperties prop;
  gchar *model_file = _MockMakeModelFile ("nns_snpe_mock_resizable_XXXXXX.dlc");
  ASSERT_TRUE (model_file != NULL);

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);
  _MockSetProp (&prop, model_files, "UserBuffer:true");

  snpe_mock_reset ();
  EXPECT_NE (sp->open (&prop, &data), 0);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);

  sp->close (&prop, &data);
  g_remove (model_file);
  g_free (model_file);
}
