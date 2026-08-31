/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_litert.cc
 * @date    20 Aug 2026
 * @brief   Unit test for the LiteRT (CompiledModel API) tensor filter sub-plugin
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 * The golden results (e.g., argmax 951 for orange.png) are shared with the
 * tensorflow2-lite subplugin unit tests, which run the very same .tflite
 * model files with the classic Interpreter API. Any divergence between the
 * two subplugins on the same model is therefore caught at unit test level.
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>
#include <unittest_util.h>
#include "nnstreamer_plugin_api.h"
#include "nnstreamer_plugin_api_util.h"

/**
 * @brief internal function to get model file path
 */
static gboolean
_GetModelFilePath (gchar **model_file, int option)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  g_autofree gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
  std::string model_name;

  switch (option) {
    case 0:
      model_name = "mobilenet_v2_1.0_224_quant.tflite";
      break;
    case 1:
      model_name = "mobilenet_v2_1.0_224.tflite";
      break;
    case 2:
      model_name = "simple_32_in_32_out.tflite";
      break;
    case 3:
      model_name = "deeplabv3_257_mv_gpu.tflite";
      break;
    default:
      break;
  }

  *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", model_name.c_str (), NULL);

  return g_file_test (*model_file, G_FILE_TEST_EXISTS);
}

/**
 * @brief internal function to get the orange.png
 */
static gboolean
_GetOrangePngFilePath (gchar **input_file)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  g_autofree gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();

  *input_file = g_build_filename (
      root_path, "tests", "test_models", "data", "orange.png", NULL);

  return g_file_test (*input_file, G_FILE_TEST_EXISTS);
}

/**
 * @brief Set tensor filter properties
 */
static void
_SetFilterProp (GstTensorFilterProperties *prop, const gchar *name,
    const gchar **models, const gchar *custom = NULL)
{
  memset (prop, 0, sizeof (GstTensorFilterProperties));
  prop->fwname = name;
  prop->fw_opened = 0;
  prop->model_files = models;
  prop->num_models = models ? g_strv_length ((gchar **) models) : 0;
  prop->custom_properties = custom;
}

/**
 * @brief Signal to validate the classification result in tensor_sink
 */
static void
check_output (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  GstMemory *mem_res;
  GstMapInfo info_res;
  gboolean mapped;
  UNUSED (element);

  mem_res = gst_buffer_get_memory (buffer, 0);
  mapped = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
  ASSERT_TRUE (mapped);

  gint is_float = (gint) * ((guint8 *) user_data);
  guint idx, max_idx = 0U;

  if (is_float == 0) {
    guint8 *output = (guint8 *) info_res.data;
    guint8 max_value = 0;

    for (idx = 0; idx < info_res.size; ++idx) {
      if (output[idx] > max_value) {
        max_value = output[idx];
        max_idx = idx;
      }
    }
  } else {
    gfloat *output = (gfloat *) info_res.data;
    /** -G_MAXFLOAT: G_MINFLOAT is the smallest positive normalized value,
     *  which would break the max search for all-negative outputs */
    gfloat max_value = -G_MAXFLOAT;

    for (idx = 0; idx < (info_res.size / sizeof (gfloat)); ++idx) {
      if (output[idx] > max_value) {
        max_value = output[idx];
        max_idx = idx;
      }
    }
  }

  gst_memory_unmap (mem_res, &info_res);
  gst_memory_unref (mem_res);

  /** the same golden index as the tensorflow2-lite unit tests */
  EXPECT_EQ (max_idx, 951U) << "The classification result differs from the "
                               "tensorflow2-lite subplugin's golden result "
                               "for the same model file.";
}

/**
 * @brief Signal to count invocations of tensor_sink
 * @note Connect this after check_output. GObject runs same-stage handlers in
 *       connection order, so a satisfied count implies the golden comparison
 *       for that buffer has already run; connecting it first would let the
 *       waiter tear the pipeline down before check_output had a say.
 */
static void
count_output (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  guint *count = (guint *) user_data;
  UNUSED (element);
  UNUSED (buffer);
  g_atomic_int_inc (count);
}

/**
 * @brief Check the litert subplugin is registered.
 */
TEST (nnstreamerFilterLiteRT, checkExistence)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);
}

/**
 * @brief Positive case with open/close.
 */
TEST (nnstreamerFilterLiteRT, openClose00)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case with an invalid model file path.
 */
TEST (nnstreamerFilterLiteRT, openClose00_n)
{
  int ret;
  void *data = NULL;

  const gchar *model_files[] = { "some/invalid/model/path.tflite", NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Negative case with an existing file that is not a valid flatbuffer.
 */
TEST (nnstreamerFilterLiteRT, openClose01_n)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *garbage_file = NULL;
  gint fd;

  fd = g_file_open_tmp ("litert_garbage_XXXXXX.tflite", &garbage_file, NULL);
  ASSERT_GE (fd, 0);
  ASSERT_TRUE (g_file_set_contents (
      garbage_file, "This is not a tflite flatbuffer at all.", -1, NULL));
  g_close (fd, NULL);

  const gchar *model_files[] = { garbage_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);

  g_unlink (garbage_file);
}

/**
 * @brief Negative case with no model file given.
 */
TEST (nnstreamerFilterLiteRT, openClose02_n)
{
  int ret;
  void *data = NULL;

  const gchar *model_files[] = { NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Positive case: getModelInfo of the float mobilenet model.
 */
TEST (nnstreamerFilterLiteRT, getModelInfo00)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  GstTensorsInfo in_info, out_info;
  ret = sp->getModelInfo (NULL, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (in_info.num_tensors, 1U);
  EXPECT_EQ (in_info.info[0].dimension[0], 3U);
  EXPECT_EQ (in_info.info[0].dimension[1], 224U);
  EXPECT_EQ (in_info.info[0].dimension[2], 224U);
  EXPECT_EQ (in_info.info[0].dimension[3], 1U);
  EXPECT_EQ (in_info.info[0].type, _NNS_FLOAT32);

  EXPECT_EQ (out_info.num_tensors, 1U);
  EXPECT_EQ (out_info.info[0].dimension[0], 1001U);
  EXPECT_EQ (out_info.info[0].dimension[1], 1U);
  EXPECT_EQ (out_info.info[0].type, _NNS_FLOAT32);

  sp->close (&prop, &data);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Positive case: getModelInfo of the quantized mobilenet model.
 */
TEST (nnstreamerFilterLiteRT, getModelInfo01)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  GstTensorsInfo in_info, out_info;
  ret = sp->getModelInfo (NULL, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (in_info.num_tensors, 1U);
  EXPECT_EQ (in_info.info[0].dimension[0], 3U);
  EXPECT_EQ (in_info.info[0].dimension[1], 224U);
  EXPECT_EQ (in_info.info[0].dimension[2], 224U);
  EXPECT_EQ (in_info.info[0].type, _NNS_UINT8);

  EXPECT_EQ (out_info.num_tensors, 1U);
  EXPECT_EQ (out_info.info[0].dimension[0], 1001U);
  EXPECT_EQ (out_info.info[0].type, _NNS_UINT8);

  sp->close (&prop, &data);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Negative case: getModelInfo with an unsupported ops.
 */
TEST (nnstreamerFilterLiteRT, getModelInfo02_n)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  GstTensorsInfo in_info, out_info;
  ret = sp->getModelInfo (NULL, &prop, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);

  sp->close (&prop, &data);
}

/**
 * @brief Positive case: direct invoke; output canary must be overwritten.
 */
TEST (nnstreamerFilterLiteRT, invoke00)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  input.size = sizeof (float) * 224 * 224 * 3;
  output.size = sizeof (float) * 1001;
  input.data = g_malloc0 (input.size);
  output.data = g_malloc (output.size);
  memset (output.data, 0xAA, output.size); /* canary */

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  ret = sp->invoke (NULL, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);

  /* mobilenet output is a softmax; it cannot remain the 0xAA canary */
  gboolean changed = FALSE;
  for (gsize i = 0; i < output.size; ++i) {
    if (((guint8 *) output.data)[i] != 0xAA) {
      changed = TRUE;
      break;
    }
  }
  EXPECT_TRUE (changed) << "invoke() succeeded but did not write the output buffer.";

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case: repeated invoke with the same input must be deterministic.
 */
TEST (nnstreamerFilterLiteRT, invokeConsistency)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output1, output2;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  input.size = sizeof (float) * 224 * 224 * 3;
  output1.size = output2.size = sizeof (float) * 1001;
  input.data = g_malloc (input.size);
  output1.data = g_malloc0 (output1.size);
  output2.data = g_malloc0 (output2.size);

  /* a deterministic non-trivial input pattern */
  for (gsize i = 0; i < input.size / sizeof (float); ++i)
    ((float *) input.data)[i] = (float) (i % 255) / 255.0f - 0.5f;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  EXPECT_EQ (sp->invoke (NULL, &prop, data, &input, &output1), 0);
  EXPECT_EQ (sp->invoke (NULL, &prop, data, &input, &output2), 0);
  EXPECT_EQ (memcmp (output1.data, output2.data, output1.size), 0)
      << "Two invocations with identical input produced different outputs.";

  g_free (input.data);
  g_free (output1.data);
  g_free (output2.data);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case: invoke with null tensor memories.
 */
TEST (nnstreamerFilterLiteRT, invoke01_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  input.size = output.size = sizeof (float);
  input.data = g_malloc0 (input.size);
  output.data = g_malloc0 (output.size);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  EXPECT_NE (sp->invoke (NULL, &prop, data, NULL, &output), 0);
  EXPECT_NE (sp->invoke (NULL, &prop, data, &input, NULL), 0);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case: invoke with a 64-byte aligned output buffer that is
 * at/above the zero-copy size gate (deeplabv3 output is 5548116 B, well over
 * the 256 KiB threshold) takes the zero-copy wrap path; the result must
 * still be correct.
 */
TEST (nnstreamerFilterLiteRT, zeroCopyAlignedOutput)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 3));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  input.size = 792588;
  output.size = 5548116;
  input.data = g_malloc0 (input.size);
  ASSERT_EQ (posix_memalign (&output.data, 64, output.size), 0);
  memset (output.data, 0xAA, output.size); /* canary */

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  ret = sp->invoke (NULL, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);

  gboolean changed = FALSE;
  for (gsize i = 0; i < output.size; ++i) {
    if (((guint8 *) output.data)[i] != 0xAA) {
      changed = TRUE;
      break;
    }
  }
  EXPECT_TRUE (changed) << "invoke() succeeded but did not write the output buffer.";

  g_free (input.data);
  free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case: invoke with a deliberately misaligned output buffer
 * takes the memcpy fallback path even though the output is above the
 * zero-copy size gate; the result must still be correct.
 */
TEST (nnstreamerFilterLiteRT, zeroCopyUnalignedOutput)
{
  int ret;
  void *data = NULL;
  void *base = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 3));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  input.size = 792588;
  output.size = 5548116;
  input.data = g_malloc0 (input.size);
  /* 64-byte aligned block, offset by 8 B so output.data itself is not */
  ASSERT_EQ (posix_memalign (&base, 64, output.size + 64), 0);
  output.data = (guint8 *) base + 8;
  memset (output.data, 0xAA, output.size); /* canary */

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  ret = sp->invoke (NULL, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);

  gboolean changed = FALSE;
  for (gsize i = 0; i < output.size; ++i) {
    if (((guint8 *) output.data)[i] != 0xAA) {
      changed = TRUE;
      break;
    }
  }
  EXPECT_TRUE (changed) << "invoke() succeeded but did not write the output buffer.";

  g_free (input.data);
  free (base);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case: the zero-copy path and the memcpy fallback path must
 * produce byte-identical output for the same input. This is the key safety
 * property of routing some invocations directly into caller memory.
 */
TEST (nnstreamerFilterLiteRT, zeroCopyPathEquivalence)
{
  int ret;
  void *data = NULL;
  void *unaligned_base = NULL;
  GstTensorMemory input, aligned_output, unaligned_output;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 3));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  input.size = 792588;
  aligned_output.size = unaligned_output.size = 5548116;
  input.data = g_malloc0 (input.size);

  ASSERT_EQ (posix_memalign (&aligned_output.data, 64, aligned_output.size), 0);
  ASSERT_EQ (posix_memalign (&unaligned_base, 64, unaligned_output.size + 64), 0);
  unaligned_output.data = (guint8 *) unaligned_base + 8;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  EXPECT_EQ (sp->invoke (NULL, &prop, data, &input, &aligned_output), 0);
  EXPECT_EQ (sp->invoke (NULL, &prop, data, &input, &unaligned_output), 0);
  EXPECT_EQ (memcmp (aligned_output.data, unaligned_output.data, aligned_output.size), 0)
      << "The zero-copy path and the memcpy fallback path produced different output.";

  g_free (input.data);
  free (aligned_output.data);
  free (unaligned_base);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case: repeated invoke through the zero-copy path must not
 * leak or corrupt state; every run must succeed and match the first result.
 * The iteration count is kept low because deeplabv3 is a large model and
 * each invoke is comparatively slow.
 */
TEST (nnstreamerFilterLiteRT, zeroCopyRepeatedInvoke)
{
  const guint iterations = 5U;
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;
  g_autofree gchar *first_result = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 3));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  input.size = 792588;
  output.size = 5548116;
  input.data = g_malloc0 (input.size);
  ASSERT_EQ (posix_memalign (&output.data, 64, output.size), 0);
  first_result = (gchar *) g_malloc (output.size);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  for (guint i = 0; i < iterations; ++i) {
    /** Re-poison every round. Comparing one round against the next only shows
     *  they agree, which an untouched buffer also does; the canary is what
     *  says the run actually wrote through the wrapper each time. */
    memset (output.data, 0xAA, output.size);

    ret = sp->invoke (NULL, &prop, data, &input, &output);
    EXPECT_EQ (ret, 0) << "invoke() failed on iteration " << i;
    if (ret != 0)
      break;

    gboolean changed = FALSE;
    for (gsize b = 0; b < output.size; ++b) {
      if (((guint8 *) output.data)[b] != 0xAA) {
        changed = TRUE;
        break;
      }
    }
    EXPECT_TRUE (changed) << "Iteration " << i << " left the output buffer untouched.";

    if (i == 0) {
      memcpy (first_result, output.data, output.size);
    } else {
      EXPECT_EQ (memcmp (first_result, output.data, output.size), 0)
          << "Output diverged on iteration " << i;
    }
  }

  g_free (input.data);
  free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case: an aligned output below the size gate stays correct.
 *
 * The mobilenet output is 4004 B, under litert_wrap_min_bytes, so the gate
 * sends it through the copy even though its alignment would otherwise qualify
 * it. What is pinned here is that the gate does not break such a tensor.
 *
 * It does not pin the gate itself, and no test does: which branch runs is a
 * performance decision that produces identical bytes either way, so removing
 * the size floor leaves this whole suite green. Guarding that would mean
 * making the choice observable from outside the subplugin, which is not worth
 * a constant this well commented. Anyone changing litert_wrap_min_bytes needs
 * to re-measure rather than trust CI.
 */
TEST (nnstreamerFilterLiteRT, zeroCopyBelowThreshold)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  input.size = sizeof (float) * 224 * 224 * 3;
  output.size = sizeof (float) * 1001;
  input.data = g_malloc0 (input.size);
  ASSERT_EQ (posix_memalign (&output.data, 64, output.size), 0);
  memset (output.data, 0xAA, output.size); /* canary */

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  ret = sp->invoke (NULL, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);

  gboolean changed = FALSE;
  for (gsize i = 0; i < output.size; ++i) {
    if (((guint8 *) output.data)[i] != 0xAA) {
      changed = TRUE;
      break;
    }
  }
  EXPECT_TRUE (changed) << "invoke() succeeded but did not write the output buffer.";

  g_free (input.data);
  free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case: a model with many small output tensors, each
 * individually 64-byte aligned. Every output tensor here (4 B) is far below
 * the zero-copy size gate, so this exercises the memcpy fallback path across
 * many simultaneous outputs rather than the zero-copy wrap path.
 */
TEST (nnstreamerFilterLiteRT, zeroCopyManyOutputsAligned)
{
  const guint num_tensors = 32U;
  int ret;
  void *data = NULL;
  GstTensorMemory input[32], output[32];
  void *output_base[32] = { NULL };
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 2));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  for (guint i = 0; i < num_tensors; ++i) {
    input[i].size = sizeof (float);
    input[i].data = g_malloc0 (sizeof (float));
    ((float *) input[i].data)[0] = 16.0f;

    output[i].size = sizeof (float);
    ASSERT_EQ (posix_memalign (&output_base[i], 64, output[i].size), 0);
    output[i].data = output_base[i];
  }

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  ret = sp->invoke (NULL, &prop, data, input, output);
  EXPECT_EQ (ret, 0);

  for (guint i = 0; i < num_tensors; ++i)
    EXPECT_FLOAT_EQ (((float *) output[i].data)[0], 17.0f);

  for (guint i = 0; i < num_tensors; ++i) {
    g_free (input[i].data);
    free (output_base[i]);
  }
  sp->close (&prop, &data);
}

/**
 * @brief Positive case: explicit cpu accelerator via custom property.
 */
TEST (nnstreamerFilterLiteRT, customPropAccelCpu)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files, "Accelerators:cpu");

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case: unknown accelerator name must be rejected.
 */
TEST (nnstreamerFilterLiteRT, customProp00_n)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files, "Accelerators:tpu");

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Negative case: malformed custom property string must be rejected.
 */
TEST (nnstreamerFilterLiteRT, customProp01_n)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files, "JustAKeyWithoutValue");

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Negative case: nonexistent signature key must be rejected.
 */
TEST (nnstreamerFilterLiteRT, customProp02_n)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files, "Signature:no_such_signature_key");

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Negative case to launch gst pipeline: wrong dimension.
 */
TEST (nnstreamerFilterLiteRT, launch00_n)
{
  GstElement *gstpipe;
  GError *err = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  g_autofree gchar *pipeline = g_strdup_printf (
      "videotestsrc num-buffers=10 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=42,height=42,framerate=0/1 ! tensor_converter ! tensor_filter framework=litert model=\"%s\" ! tensor_sink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
}

/**
 * @brief Negative case to launch gst pipeline: wrong data type.
 */
TEST (nnstreamerFilterLiteRT, launch01_n)
{
  GstElement *gstpipe;
  GError *err = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  /* uint8 stream fed into the float32 model */
  g_autofree gchar *pipeline = g_strdup_printf (
      "videotestsrc num-buffers=10 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_filter framework=litert model=\"%s\" ! tensor_sink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
}

/**
 * @brief Positive case: classification result of the float model.
 */
TEST (nnstreamerFilterLiteRT, floatModelResult)
{
  GstElement *gstpipe;
  GError *err = NULL;
  g_autofree gchar *model_file = NULL;
  g_autofree gchar *input_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));
  ASSERT_TRUE (_GetOrangePngFilePath (&input_file));

  g_autofree gchar *pipeline = g_strdup_printf (
      "filesrc location=\"%s\" ! pngdec ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter framework=litert model=\"%s\" ! tensor_sink name=sink",
      input_file, model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  GstElement *sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sink");
  ASSERT_TRUE (sink_handle != nullptr);

  guint8 is_float = 1;
  guint count = 0U;
  g_signal_connect (sink_handle, "new-data", (GCallback) check_output, &is_float);
  g_signal_connect (sink_handle, "new-data", (GCallback) count_output, &count);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10),
      0);
  EXPECT_TRUE (wait_pipeline_process_buffers (&count, 1U, TEST_TIMEOUT_LIMIT_MS));
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_GE (count, 1U);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
}

/**
 * @brief Positive case: classification result of the quantized model.
 */
TEST (nnstreamerFilterLiteRT, quantModelResult)
{
  GstElement *gstpipe;
  GError *err = NULL;
  g_autofree gchar *model_file = NULL;
  g_autofree gchar *input_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 0));
  ASSERT_TRUE (_GetOrangePngFilePath (&input_file));

  g_autofree gchar *pipeline = g_strdup_printf (
      "filesrc location=\"%s\" ! pngdec ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_filter framework=litert model=\"%s\" ! tensor_sink name=sink",
      input_file, model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  GstElement *sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sink");
  ASSERT_TRUE (sink_handle != nullptr);

  guint8 is_float = 0;
  guint count = 0U;
  g_signal_connect (sink_handle, "new-data", (GCallback) check_output, &is_float);
  g_signal_connect (sink_handle, "new-data", (GCallback) count_output, &count);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10),
      0);
  EXPECT_TRUE (wait_pipeline_process_buffers (&count, 1U, TEST_TIMEOUT_LIMIT_MS));
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_GE (count, 1U);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
}

/**
 * @brief Signal to validate the result in tensor_sink of 32 input/output model.
 */
static void
check_output_many (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  GstMemory *mem_res;
  GstMapInfo info_res;
  gboolean mapped;
  UNUSED (element);

  guint *data_received = (guint *) user_data;
  (*data_received)++;

  for (guint i = 0; i < 32; i++) {
    mem_res = gst_tensor_buffer_get_nth_memory (buffer, i);
    mapped = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
    ASSERT_TRUE (mapped);
    gfloat *output = (gfloat *) info_res.data;
    EXPECT_EQ (17.f, *output);
    gst_memory_unmap (mem_res, &info_res);
    gst_memory_unref (mem_res);
  }
}

/**
 * @brief Positive case: model with 32 input/output tensors.
 */
TEST (nnstreamerFilterLiteRT, manyInOutModel)
{
  GstElement *gstpipe;
  GError *err = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 2));

  /* make 32 "t. ! queue ! mux.sink_## " */
  gchar *tee_queue_mux = g_strdup ("");
  for (int i = 0; i < 32; i++) {
    gchar *aux = g_strdup (tee_queue_mux);
    g_free (tee_queue_mux);
    tee_queue_mux = g_strdup_printf ("%s t. ! queue ! mux.sink_%d ", aux, i);
    g_free (aux);
  }

  g_autofree gchar *pipeline = g_strdup_printf (
      "videotestsrc pattern=2 num-buffers=10 is-live=true ! "
      "videoscale ! videoconvert ! video/x-raw,format=GRAY8,width=1,height=1,framerate=30/1 ! "
      "tensor_converter ! tensor_transform mode=typecast option=float32 ! tee name=t "
      "%s"
      "tensor_mux name=mux ! other/tensors,format=static,num_tensors=32 ! "
      "tensor_filter framework=litert model=\"%s\" ! tensor_sink name=sinkx",
      tee_queue_mux, model_file);

  g_free (tee_queue_mux);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  GstElement *sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  ASSERT_TRUE (sink_handle != nullptr);

  guint data_received = 0U;
  g_signal_connect (sink_handle, "new-data", (GCallback) check_output_many, &data_received);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10),
      0);
  g_usleep (1000 * 1000 * 5); /* wait for 5 seconds to check all output is valid */

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_EQ (10U, data_received);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
}

/**
 * @brief Positive case: reopen (RELOAD-like flow) with a different model.
 */
TEST (nnstreamerFilterLiteRT, reopenDifferentModel)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file_float = NULL;
  g_autofree gchar *model_file_quant = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file_float, 1));
  ASSERT_TRUE (_GetModelFilePath (&model_file_quant, 0));

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  const gchar *model_files_float[] = { model_file_float, NULL };
  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files_float);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);
  sp->close (&prop, &data);

  const gchar *model_files_quant[] = { model_file_quant, NULL };
  _SetFilterProp (&prop, "litert", model_files_quant);
  data = NULL;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  GstTensorsInfo in_info, out_info;
  ret = sp->getModelInfo (NULL, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (in_info.info[0].type, _NNS_UINT8);

  sp->close (&prop, &data);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Positive case: a second instance must survive the first instance's close.
 *
 * Both instances share the process-wide LiteRtEnvironment. If close() on the
 * first instance destroyed that shared environment (instead of merely
 * dropping a reference), the second instance's still-open compiled model
 * would be left operating on a freed environment and its next invoke()
 * would fail or crash.
 */
TEST (nnstreamerFilterLiteRT, sharedEnvMultiInstance)
{
  int ret;
  void *data1 = NULL, *data2 = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop1, prop2;
  _SetFilterProp (&prop1, "litert", model_files);
  _SetFilterProp (&prop2, "litert", model_files);

  input.size = sizeof (float) * 224 * 224 * 3;
  output.size = sizeof (float) * 1001;
  input.data = g_malloc0 (input.size);
  output.data = g_malloc (output.size);

  /* two independent instances held open at the same time */
  ret = sp->open (&prop1, &data1);
  ASSERT_EQ (ret, 0);
  ret = sp->open (&prop2, &data2);
  ASSERT_EQ (ret, 0);

  EXPECT_EQ (sp->invoke (NULL, &prop1, data1, &input, &output), 0);
  EXPECT_EQ (sp->invoke (NULL, &prop2, data2, &input, &output), 0);

  /* dropping the first instance must not tear down the shared environment */
  sp->close (&prop1, &data1);

  EXPECT_EQ (sp->invoke (NULL, &prop2, data2, &input, &output), 0)
      << "The second instance failed after the first instance closed; the "
         "shared LiteRtEnvironment was likely destroyed prematurely.";

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop2, &data2);
}

/**
 * @brief Positive case: the shared environment must be recreated after the
 * refcount drops to zero and a new instance is opened.
 */
TEST (nnstreamerFilterLiteRT, sharedEnvReacquire)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  input.size = sizeof (float) * 224 * 224 * 3;
  output.size = sizeof (float) * 1001;
  input.data = g_malloc0 (input.size);
  output.data = g_malloc (output.size);

  /* first instance: acquires, uses, and fully releases the shared env */
  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);
  EXPECT_EQ (sp->invoke (NULL, &prop, data, &input, &output), 0);
  sp->close (&prop, &data);

  /* second instance: must re-acquire (recreate) the shared env from scratch */
  data = NULL;
  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);
  EXPECT_EQ (sp->invoke (NULL, &prop, data, &input, &output), 0)
      << "Reopening after the refcount reached zero failed; the shared "
         "LiteRtEnvironment was likely not recreated on re-acquire.";

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Worker routine for sharedEnvConcurrentOpen: open, invoke, close once.
 * @param[in] model_files NULL-terminated array with a single model path
 * @param[out] ok set to true on success; read by the joining thread only
 */
static void
_sharedEnvWorker (const gchar **model_files, std::atomic<bool> *ok)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  if (sp == nullptr) {
    *ok = false;
    return;
  }

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  GstTensorMemory input, output;
  input.size = sizeof (float) * 224 * 224 * 3;
  output.size = sizeof (float) * 1001;
  input.data = g_malloc0 (input.size);
  output.data = g_malloc (output.size);

  void *data = NULL;
  gboolean success = FALSE;

  if (sp->open (&prop, &data) == 0) {
    success = (sp->invoke (NULL, &prop, data, &input, &output) == 0);
    sp->close (&prop, &data);
  }

  g_free (input.data);
  g_free (output.data);

  *ok = success;
}

/**
 * @brief Positive case: concurrent open/invoke/close from multiple threads
 * must not corrupt the mutex-guarded shared-environment refcount.
 */
TEST (nnstreamerFilterLiteRT, sharedEnvConcurrentOpen)
{
  const guint num_workers = 4U;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };

  std::vector<std::atomic<bool>> results (num_workers);
  for (auto &r : results)
    r = false;

  std::vector<std::thread> workers;
  for (guint i = 0; i < num_workers; ++i)
    workers.emplace_back (_sharedEnvWorker, model_files, &results[i]);

  for (auto &w : workers)
    w.join ();

  for (guint i = 0; i < num_workers; ++i)
    EXPECT_TRUE (results[i].load ()) << "Worker " << i << " failed under concurrent open/close.";
}

static std::atomic<int> _unbalanced_release_count (0);

/**
 * @brief GLib log handler counting the subplugin's unbalanced-release error.
 * @param[in] domain log domain, forwarded to the default handler
 * @param[in] level log level, forwarded to the default handler
 * @param[in] message the log message to inspect
 * @param[in] user_data unused, forwarded to the default handler
 */
static void
_countUnbalancedRelease (const gchar *domain, GLogLevelFlags level,
    const gchar *message, gpointer user_data)
{
  if (message != NULL
      && g_strstr_len (message, -1, "Unbalanced LiteRT environment release") != NULL)
    ++_unbalanced_release_count;

  g_log_default_handler (domain, level, message, user_data);
}

/**
 * @brief Positive case: a failed configure must release exactly the reference
 * it took, leaving an already-open instance usable.
 *
 * Both failures below throw after litert_env_ref() has succeeded, which is the
 * only asymmetric path in the reference count.
 *
 * The reference count itself is what is checked, via the error the subplugin
 * logs when a release finds no reference outstanding. Asserting on invoke()
 * instead would not work: destroying a LiteRtEnvironment while a compiled
 * model built from it is still alive leaves that model usable on the CPU path
 * this runs on, so an over-release is invisible in the output of a pipeline.
 * The invoke() checks below therefore assert the user-visible property (a
 * failed open does not disturb an open instance), not the leak itself.
 *
 * Do not reduce this to a single failing open: the repetition is what makes
 * the log fire. An over-release on the first failure only walks the count
 * down to zero, which is silent. It is the second failure, which finds the
 * count at zero, builds a fresh environment, and then over-releases from one,
 * that reaches a release with nothing outstanding. Verified by injecting a
 * double release: two failures catch it, one does not.
 */
TEST (nnstreamerFilterLiteRT, sharedEnvRefBalanceOnConfigureFailure)
{
  int ret;
  void *data = NULL, *data_bad = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;
  g_autofree gchar *garbage_file = NULL;
  gint fd;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop, prop_bad;
  _SetFilterProp (&prop, "litert", model_files);

  input.size = sizeof (float) * 224 * 224 * 3;
  output.size = sizeof (float) * 1001;
  input.data = g_malloc0 (input.size);
  output.data = g_malloc (output.size);

  /** Everything that can fail fatally is done before the log handler is
   *  installed: a fatal assertion returns from the test on the spot, which
   *  would leave the handler in place for every case that runs after. */
  fd = g_file_open_tmp ("litert_garbage_XXXXXX.tflite", &garbage_file, NULL);
  ASSERT_GE (fd, 0);
  ASSERT_TRUE (g_file_set_contents (
      garbage_file, "This is not a tflite flatbuffer at all.", -1, NULL));
  g_close (fd, NULL);

  const gchar *bad_model_files[] = { garbage_file, NULL };

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  _unbalanced_release_count = 0;
  GLogFunc prev_handler = g_log_set_default_handler (_countUnbalancedRelease, NULL);

  /* fails in resolveSignature(), after the environment reference is taken */
  _SetFilterProp (&prop_bad, "litert", model_files, "Signature:no_such_signature_key");
  EXPECT_NE (sp->open (&prop_bad, &data_bad), 0);

  EXPECT_EQ (sp->invoke (NULL, &prop, data, &input, &output), 0)
      << "A failed signature lookup disturbed an already-open instance.";

  /* fails in LiteRtCreateModelFromFile(), also after the reference is taken */
  data_bad = NULL;
  _SetFilterProp (&prop_bad, "litert", bad_model_files);
  EXPECT_NE (sp->open (&prop_bad, &data_bad), 0);

  EXPECT_EQ (sp->invoke (NULL, &prop, data, &input, &output), 0)
      << "A failed model load disturbed an already-open instance.";

  g_log_set_default_handler (prev_handler, NULL);
  EXPECT_EQ (_unbalanced_release_count.load (), 0)
      << "A failed configure released an environment reference it did not hold.";

  g_unlink (garbage_file);
  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Churn routine for sharedEnvInvokeDuringConfigure: repeatedly build
 * and tear down an instance on the shared environment.
 * @param[in] model_files NULL-terminated array with a single model path
 * @param[in] iterations how many open/close cycles to run
 * @param[out] cycles incremented after each cycle so the caller can overlap it
 * @param[out] ok set to false if any cycle failed; read after join only
 */
static void
_sharedEnvChurn (const gchar **model_files, guint iterations,
    std::atomic<guint> *cycles, std::atomic<bool> *ok)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  if (sp == nullptr) {
    *ok = false;
    *cycles = iterations; /* do not leave the caller spinning */
    return;
  }

  for (guint i = 0; i < iterations; ++i) {
    GstTensorFilterProperties prop;
    void *data = NULL;

    _SetFilterProp (&prop, "litert", model_files);
    if (sp->open (&prop, &data) != 0) {
      *ok = false;
      *cycles = iterations;
      return;
    }
    sp->close (&prop, &data);
    ++(*cycles);
  }
}

/**
 * @brief Positive case: invoking one instance while other instances are built
 * and torn down on the same shared environment must keep producing results.
 *
 * Configuration and teardown hold the environment lock exclusively while
 * invoke holds it in shared mode; this is the case that regresses if that
 * distinction is dropped.
 *
 * The overlap is structural rather than timing-dependent: this thread keeps
 * invoking until the churn thread reports every cycle done, so the two run
 * against the shared environment at the same time however the scheduler
 * interleaves them. An invoke bound guards against a wedged churn thread; it
 * is reported when reached, since it shortens the overlap.
 */
TEST (nnstreamerFilterLiteRT, sharedEnvInvokeDuringConfigure)
{
  const guint iterations = 10U;
  const guint max_invokes = 1000U;
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  input.size = sizeof (float) * 224 * 224 * 3;
  output.size = sizeof (float) * 1001;
  input.data = g_malloc0 (input.size);
  output.data = g_malloc (output.size);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  std::atomic<guint> cycles (0U);
  std::atomic<bool> churn_ok (true);
  std::thread churn (_sharedEnvChurn, model_files, iterations, &cycles, &churn_ok);

  guint invokes = 0U;
  gboolean invoke_ok = TRUE;

  while (invokes < iterations || (cycles.load () < iterations && invokes < max_invokes)) {
    if (sp->invoke (NULL, &prop, data, &input, &output) != 0) {
      invoke_ok = FALSE;
      break;
    }
    ++invokes;
  }

  /** Report a short overlap instead of letting it pass as a full one. Not an
   *  assertion: hitting the bound means invoke is fast relative to a model
   *  compile, which is not a defect. */
  if (invoke_ok && invokes >= max_invokes && cycles.load () < iterations)
    g_message ("sharedEnvInvokeDuringConfigure: reached the %u invoke bound "
               "after %u of %u churn cycles; overlap was partial.",
        max_invokes, cycles.load (), iterations);

  churn.join ();
  EXPECT_TRUE (invoke_ok) << "Invoke " << invokes
                          << " failed while another instance was being configured.";
  EXPECT_TRUE (churn_ok.load ()) << "Concurrent open/close churn failed.";

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
