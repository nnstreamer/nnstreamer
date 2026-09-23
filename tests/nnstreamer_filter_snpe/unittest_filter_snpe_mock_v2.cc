/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_snpe_mock_v2.cc
 * @date    22 Sep 2026
 * @brief   Unit tests of the SNPE 2.x tensor_filter sub-plugin on the mock SDK.
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 * These cases run beside the ones of unittest_filter_snpe.cc, which is linked
 * into the same binary and covers the successful paths. What is added here are
 * the error paths that release resources, which the mock can observe.
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

#include "mock/snpe_mock.h"

/**
 * @brief Build the path of the sample model of the given element type.
 */
static gchar *
_MockModelPath (gboolean is_float_model)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
  gchar *model_file = g_build_filename (root_path, "tests", "test_models",
      "models", is_float_model ? "add2_float.v2.dlc" : "add2_uint8.v2.dlc", NULL);

  g_free (root_path);
  return model_file;
}

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
 * @brief Open the sample model with the given custom properties and close it.
 * @return the result of the open call
 */
static int
_MockOpenClose (gboolean is_float_model, const gchar *custom)
{
  void *data = NULL;
  GstTensorFilterProperties prop;
  gchar *model_file = _MockModelPath (is_float_model);
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
 * @brief Positive case: every custom property the sub-plugin accepts.
 */
TEST (nnstreamerFilterSnpeMockV2, customProp00)
{
  snpe_mock_reset ();

  EXPECT_EQ (_MockOpenClose (TRUE, "Runtime:CPU,OutputTensor:output,InputType:FLOAT32,OutputType:FLOAT32,Unknown:1"),
      0);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
  EXPECT_EQ (snpe_mock_over_release_count (), 0U);
}

/**
 * @brief Negative case: a runtime the mock reports as unavailable.
 */
TEST (nnstreamerFilterSnpeMockV2, unavailableRuntime01_n)
{
  snpe_mock_reset ();

  EXPECT_NE (_MockOpenClose (TRUE, "Runtime:GPU"), 0);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
}

/**
 * @brief Negative case: the SNPE instance cannot be built.
 */
TEST (nnstreamerFilterSnpeMockV2, buildFailure02_n)
{
  snpe_mock_reset ();
  snpe_mock_set_failure (SNPE_MOCK_FAIL_BUILD);

  EXPECT_NE (_MockOpenClose (TRUE, NULL), 0);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
}

/**
 * @brief Negative case: the model file does not exist.
 */
TEST (nnstreamerFilterSnpeMockV2, missingModelFile03_n)
{
  void *data = NULL;
  GstTensorFilterProperties prop;
  const gchar *model_files[] = { "/tmp/nns_snpe_mock_does_not_exist.dlc", NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);
  _MockSetProp (&prop, model_files, NULL);

  snpe_mock_reset ();
  EXPECT_NE (sp->open (&prop, &data), 0);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);

  sp->close (&prop, &data);
}

/**
 * @brief Positive case: a repeated OutputTensor option leaks no string list.
 *
 * Regression test of item F4 of issue #4920: the second OutputTensor option
 * used to overwrite the handle the first one had created.
 */
TEST (nnstreamerFilterSnpeMockV2, customPropRepeatedOutputTensor04)
{
  snpe_mock_reset ();

  EXPECT_EQ (_MockOpenClose (TRUE, "OutputTensor:output,OutputTensor:output"), 0);
  EXPECT_EQ (snpe_mock_live_count (SNPE_MOCK_OBJ_STRING_LIST), 0U);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
}

/**
 * @brief Negative case: an empty output tensor name releases every allocation.
 *
 * Regression test of item F4 of issue #4920: the string vectors the option
 * parser had allocated used to leak when it threw.
 */
TEST (nnstreamerFilterSnpeMockV2, customPropEmptyOutputTensorName05_n)
{
  if (!snpe_mock_ledger_available ())
    GTEST_SKIP () << "the allocation ledger needs the --wrap option of the linker";

  snpe_mock_reset ();

  EXPECT_NE (_MockOpenClose (TRUE, "OutputTensor:output;;output"), 0);
  EXPECT_EQ (snpe_mock_ledger_live_count (), 0U);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
}

/**
 * @brief Negative case: a failing name append releases every allocation.
 *
 * Regression test of item F4 of issue #4920, covering the second throw of the
 * option parser.
 */
TEST (nnstreamerFilterSnpeMockV2, customPropAppendFailure06_n)
{
  if (!snpe_mock_ledger_available ())
    GTEST_SKIP () << "the allocation ledger needs the --wrap option of the linker";

  snpe_mock_reset ();
  snpe_mock_set_failure (SNPE_MOCK_FAIL_STRING_LIST_APPEND);

  EXPECT_NE (_MockOpenClose (TRUE, "OutputTensor:output"), 0);
  EXPECT_EQ (snpe_mock_ledger_live_count (), 0U);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
}

/**
 * @brief Negative case: asking for a quantized type on a float model.
 *
 * Regression test of item F4 of issue #4920: the buffer attributes used to
 * leak when the requested element type did not match the model.
 */
TEST (nnstreamerFilterSnpeMockV2, quantizedTypeOnFloatModel07_n)
{
  snpe_mock_reset ();

  EXPECT_NE (_MockOpenClose (TRUE, "InputType:TF8"), 0);
  EXPECT_EQ (snpe_mock_live_count (SNPE_MOCK_OBJ_BUFFER_ATTRIBUTES), 0U);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);
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
 * @brief Negative case: a model whose element type has no NNStreamer type.
 *
 * Regression test of item F4 of issue #4920, covering the other throw that
 * leaves the buffer attributes behind.
 */
TEST (nnstreamerFilterSnpeMockV2, unsupportedElementType08_n)
{
  void *data = NULL;
  GstTensorFilterProperties prop;
  gchar *model_file = _MockMakeModelFile ("nns_snpe_mock_badenc_XXXXXX.dlc");
  ASSERT_TRUE (model_file != NULL);

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);
  _MockSetProp (&prop, model_files, NULL);

  snpe_mock_reset ();
  EXPECT_NE (sp->open (&prop, &data), 0);
  EXPECT_EQ (snpe_mock_live_count (SNPE_MOCK_OBJ_BUFFER_ATTRIBUTES), 0U);
  EXPECT_EQ (snpe_mock_total_live_count (), 0U);

  sp->close (&prop, &data);
  g_remove (model_file);
  g_free (model_file);
}

/**
 * @brief Positive case: the ledger sees the names an extra array holds.
 *
 * Regression test of the mock itself. The interposer of
 * gst_tensors_info_free() walked only the inline array, so a tensors
 * information holding more than NNS_TENSOR_MEMORY_MAX tensors reported every
 * name beyond it as a block that was never released.
 */
TEST (nnstreamerFilterSnpeMockV2, ledgerExtraTensorNames09)
{
  GstTensorsInfo info;

  if (!snpe_mock_ledger_available ())
    GTEST_SKIP () << "the allocation ledger needs the --wrap option of the linker";

  gst_tensors_info_init (&info);
  info.num_tensors = NNS_TENSOR_MEMORY_MAX + 1;

  snpe_mock_ledger_reset ();
  for (guint i = 0; i < info.num_tensors; i++) {
    gchar name[16];

    /* built here rather than written as a literal, which GLib would inline */
    g_snprintf (name, sizeof (name), "tensor%u", i);
    gst_tensors_info_get_nth_info (&info, i)->name = g_strdup (name);
  }
  ASSERT_EQ (snpe_mock_ledger_live_count (), info.num_tensors);

  gst_tensors_info_free (&info);
  EXPECT_EQ (snpe_mock_ledger_live_count (), 0U);
}
