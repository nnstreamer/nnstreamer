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
    case 4:
      model_name = "dynamic_batch_add_one.tflite";
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

static std::atomic<int> _direct_output_count (0);

/**
 * @brief GLib log handler counting outputs the subplugin elected to write directly.
 * @param[in] domain log domain, forwarded to the default handler
 * @param[in] level log level, forwarded to the default handler
 * @param[in] message the log message to inspect
 * @param[in] user_data unused, forwarded to the default handler
 */
static void
_countDirectOutputs (const gchar *domain, GLogLevelFlags level,
    const gchar *message, gpointer user_data)
{
  if (message != NULL && g_strstr_len (message, -1, "will be written directly") != NULL)
    ++_direct_output_count;

  g_log_default_handler (domain, level, message, user_data);
}

/**
 * @brief Positive case: a large output must actually elect the direct path.
 *
 * The other cases here only prove both paths return the same bytes, which they
 * would keep doing if the direct path stopped being chosen at all. The gate now
 * turns on an answer LiteRT gives at runtime, so an SDK that stopped reporting
 * host memory for the CPU path would silently disable the optimisation with
 * every one of them still green. This reads the decision the subplugin logs
 * and fails if a qualifying tensor no longer elects the wrap.
 */
TEST (nnstreamerFilterLiteRT, zeroCopyDirectPathElected)
{
  void *data = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 3));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);

  /** G_MESSAGES_DEBUG is deliberately not touched. Level filtering happens
   *  inside g_log_default_handler, so a handler installed in front of it sees
   *  g_debug output either way - verified on glib 2.72 - and setting the
   *  variable would only discard whatever the caller had exported. */
  _direct_output_count = 0;
  GLogFunc prev_handler = g_log_set_default_handler (_countDirectOutputs, NULL);

  const int ret = sp->open (&prop, &data);

  g_log_set_default_handler (prev_handler, NULL);

  ASSERT_EQ (ret, 0);
  EXPECT_GT (_direct_output_count.load (), 0)
      << "No output elected the direct path for a model whose output clears "
         "the size gate; the zero-copy path is no longer reachable.";

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
 *
 * Three rounds, not more. The model has to sit above the size gate to reach
 * the wrapped path at all, so it has to be deeplabv3, and each of its invokes
 * costs about 42 s under the CI memcheck run. Three still gives two
 * comparisons against the first result, which is what catches a wrapper going
 * stale; a leak needs a sanitizer rather than more rounds.
 */
TEST (nnstreamerFilterLiteRT, zeroCopyRepeatedInvoke)
{
  const guint iterations = 3U;
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
 * @brief Positive case: dynamic invoke reshapes across batches and produces
 * byte-exact input+1 results, with prop.output_meta refreshed every time.
 */
TEST (nnstreamerFilterLiteRT, dynamicInvokeReshape)
{
  const guint batches[] = { 1U, 3U, 2U, 1U };
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 4));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);
  prop.invoke_dynamic = 1;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  for (guint b = 0; b < G_N_ELEMENTS (batches); ++b) {
    const guint batch = batches[b];
    const guint num_elems = 4U * batch;
    GstTensorMemory input, output;
    g_autofree gfloat *expected = NULL;

    gst_tensors_info_free (&prop.input_meta);
    gst_tensors_info_init (&prop.input_meta);
    prop.input_meta.num_tensors = 1;
    prop.input_meta.info[0].type = _NNS_FLOAT32;
    prop.input_meta.info[0].dimension[0] = 4;
    prop.input_meta.info[0].dimension[1] = batch;

    input.size = sizeof (gfloat) * num_elems;
    input.data = g_malloc (input.size);
    for (guint i = 0; i < num_elems; ++i)
      ((gfloat *) input.data)[i] = (gfloat) i - 1.5f;

    expected = (gfloat *) g_malloc (input.size);
    for (guint i = 0; i < num_elems; ++i)
      expected[i] = ((gfloat *) input.data)[i] + 1.0f;

    /* the dynamic path allocates the output; leave it unset going in */
    output.size = 0;
    output.data = NULL;

    ret = sp->invoke (NULL, &prop, data, &input, &output);
    EXPECT_EQ (ret, 0) << "invoke failed for batch " << batch;
    ASSERT_NE (output.data, nullptr);
    EXPECT_EQ (output.size, sizeof (gfloat) * num_elems);
    EXPECT_EQ (memcmp (output.data, expected, output.size), 0)
        << "Output was not exactly input+1 for batch " << batch;

    EXPECT_EQ (prop.output_meta.info[0].dimension[0], 4U);
    EXPECT_EQ (prop.output_meta.info[0].dimension[1], batch);

    g_free (input.data);
    g_free (output.data);
  }

  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case: repeated dynamic invoke with an unchanged input shape
 * takes the resize-skipping path and must stay stable across repeats.
 */
TEST (nnstreamerFilterLiteRT, dynamicInvokeSameShapeRepeated)
{
  const guint iterations = 5U;
  const guint batch = 2U;
  const guint num_elems = 4U * batch;
  int ret;
  void *data = NULL;
  GstTensorMemory input;
  g_autofree gchar *model_file = NULL;
  g_autofree gfloat *expected = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 4));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);
  prop.invoke_dynamic = 1;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_init (&prop.input_meta);
  prop.input_meta.num_tensors = 1;
  prop.input_meta.info[0].type = _NNS_FLOAT32;
  prop.input_meta.info[0].dimension[0] = 4;
  prop.input_meta.info[0].dimension[1] = batch;

  input.size = sizeof (gfloat) * num_elems;
  input.data = g_malloc (input.size);
  for (guint i = 0; i < num_elems; ++i)
    ((gfloat *) input.data)[i] = (gfloat) i * 0.25f;

  expected = (gfloat *) g_malloc (input.size);
  for (guint i = 0; i < num_elems; ++i)
    expected[i] = ((gfloat *) input.data)[i] + 1.0f;

  for (guint iter = 0; iter < iterations; ++iter) {
    GstTensorMemory output;

    output.size = 0;
    output.data = NULL;

    ret = sp->invoke (NULL, &prop, data, &input, &output);
    EXPECT_EQ (ret, 0) << "invoke failed on iteration " << iter;
    ASSERT_NE (output.data, nullptr);
    EXPECT_EQ (output.size, input.size);
    EXPECT_EQ (memcmp (output.data, expected, output.size), 0)
        << "Output diverged on iteration " << iter;

    g_free (output.data);
  }

  g_free (input.data);
  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case: invoke_dynamic on a static model must fail. The
 * model signature declares no dynamic dimension, so the strict resize inside
 * invoke_dynamic has to reject the reshape.
 */
TEST (nnstreamerFilterLiteRT, dynamicInvokeStaticModel_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  GstTensorsInfo model_in_info, model_out_info;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 1));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);
  prop.invoke_dynamic = 1;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  ret = sp->getModelInfo (NULL, &prop, data, GET_IN_OUT_INFO, &model_in_info, &model_out_info);
  ASSERT_EQ (ret, 0);

  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_init (&prop.input_meta);
  gst_tensors_info_copy (&prop.input_meta, &model_in_info);
  /* the model does not declare this dimension dynamic; the strict resize must reject it */
  prop.input_meta.info[0].dimension[3] += 1;

  /** Size the buffer to the shape being asked for, not the compiled one.
   *  Leaving it at the old size lets fillInputBuffers' size check fail the
   *  invoke instead, so the case would stay green even if the strict resize
   *  one day accepted a fixed-shape model - which is the whole premise it
   *  exists to pin. */
  input.size = gst_tensors_info_get_size (&prop.input_meta, 0);
  input.data = g_malloc0 (input.size);
  output.size = 0;
  output.data = NULL;

  ret = sp->invoke (NULL, &prop, data, &input, &output);
  EXPECT_NE (ret, 0) << "A static model accepted invoke_dynamic's strict resize.";

  g_free (input.data);
  g_free (output.data);
  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_free (&model_in_info);
  gst_tensors_info_free (&model_out_info);
  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case: a rejected reshape must leave the instance usable.
 *
 * Rejecting the only input happens before LiteRT changes anything, so the
 * instance is still good at the shape it already had and the next invoke has
 * to prove it. This is the reachable half of the reshape failure handling:
 * once a resize has succeeded there is no way back, and the instance is
 * dropped instead - which cannot be provoked from here, since the only model
 * with a dynamic dimension has a single input.
 */
TEST (nnstreamerFilterLiteRT, dynamicInvokeRejectedReshapeKeepsInstance)
{
  void *data = NULL;
  GstTensorMemory input, output;
  GstTensorInfo *iinfo;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 4));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);
  prop.invoke_dynamic = 1;

  ASSERT_EQ (sp->open (&prop, &data), 0);

  /** the last axis is fixed at 4, so asking for 5 is a shape the signature
   *  does not admit and the strict resize refuses it */
  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_init (&prop.input_meta);
  prop.input_meta.num_tensors = 1;
  iinfo = gst_tensors_info_get_nth_info (&prop.input_meta, 0);
  iinfo->type = _NNS_FLOAT32;
  iinfo->dimension[0] = 5;
  iinfo->dimension[1] = 1;

  input.size = sizeof (float) * 5;
  input.data = g_malloc0 (input.size);
  output.data = NULL;
  output.size = 0;

  EXPECT_NE (sp->invoke (NULL, &prop, data, &input, &output), 0)
      << "The strict resize accepted a shape the model does not declare.";
  g_free (input.data);
  g_free (output.data);

  /* nothing was resized, so the instance must still run at its own shape */
  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_init (&prop.input_meta);
  prop.input_meta.num_tensors = 1;
  iinfo = gst_tensors_info_get_nth_info (&prop.input_meta, 0);
  iinfo->type = _NNS_FLOAT32;
  iinfo->dimension[0] = 4;
  iinfo->dimension[1] = 1;

  input.size = sizeof (float) * 4;
  input.data = g_malloc (input.size);
  for (guint k = 0; k < 4; ++k)
    ((float *) input.data)[k] = (float) k;
  output.data = NULL;
  output.size = 0;

  EXPECT_EQ (sp->invoke (NULL, &prop, data, &input, &output), 0)
      << "A rejected reshape left the instance unusable.";
  EXPECT_EQ (output.size, sizeof (float) * 4);
  if (output.data != NULL) {
    for (guint k = 0; k < 4; ++k) {
      EXPECT_FLOAT_EQ (((float *) output.data)[k], (float) k + 1.0f);
    }
  }

  g_free (input.data);
  g_free (output.data);
  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

static std::atomic<int> _reshape_count (0);

/**
 * @brief GLib log handler counting the reshapes the subplugin performs.
 * @param[in] domain log domain, forwarded to the default handler
 * @param[in] level log level, forwarded to the default handler
 * @param[in] message the log message to inspect
 * @param[in] user_data unused, forwarded to the default handler
 */
static void
_countReshapes (const gchar *domain, GLogLevelFlags level, const gchar *message, gpointer user_data)
{
  if (message != NULL && g_strstr_len (message, -1, "litert reshaping") != NULL)
    ++_reshape_count;

  g_log_default_handler (domain, level, message, user_data);
}

/**
 * @brief Fill prop.input_meta with one float32 tensor of the given dimensions.
 * @param[in,out] prop the properties whose input_meta is replaced
 * @param[in] dims NNS_TENSOR_RANK_LIMIT entries, zero where the axis is absent
 */
static void
_SetDynamicInputMeta (GstTensorFilterProperties *prop, const guint *dims)
{
  GstTensorInfo *info;

  gst_tensors_info_free (&prop->input_meta);
  gst_tensors_info_init (&prop->input_meta);
  prop->input_meta.num_tensors = 1;
  info = gst_tensors_info_get_nth_info (&prop->input_meta, 0);
  info->type = _NNS_FLOAT32;
  for (guint d = 0; d < NNS_TENSOR_RANK_LIMIT; ++d)
    info->dimension[d] = dims[d];
}

/**
 * @brief Positive case: a shape padded differently is still the same shape.
 *
 * The two sides of the comparison are padded by different rules. The model's
 * own meta comes from convertLayout(), which zero fills past the model's rank,
 * while a pipeline carries explicit trailing 1s - tensor_converter emits
 * 3:224:224:1 from video, and a capsfilter saying dimensions=4:1:1:1 is
 * ordinary. gst_tensor_dimension_is_equal() calls those the same shape.
 *
 * Comparing the arrays element by element instead calls them different, and
 * nothing downstream can tell: the reshape produces exactly the bytes that
 * skipping it would, so the only symptom is that every buffer silently
 * rebuilds the model while holding the shared environment lock exclusively.
 * That is why this reads the subplugin's own reshape log rather than the
 * output, and why it drives the shape the model itself reports rather than
 * one written out by hand - a hand-written shape reproduces convertLayout's
 * padding by accident and agrees no matter how the comparison is done.
 */
TEST (nnstreamerFilterLiteRT, dynamicInvokePaddedShapeSkipsReshape)
{
  const guint rounds = 3U;
  void *data = NULL;
  g_autofree gchar *model_file = NULL;
  GstTensorsInfo in_info, out_info;
  guint padded[NNS_TENSOR_RANK_LIMIT];

  ASSERT_TRUE (_GetModelFilePath (&model_file, 4));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);
  prop.invoke_dynamic = 1;

  ASSERT_EQ (sp->open (&prop, &data), 0);
  ASSERT_EQ (sp->getModelInfo (NULL, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info), 0);

  /* the model's own shape, with every absent axis spelled out as 1 */
  const GstTensorInfo *model_in = gst_tensors_info_get_nth_info (&in_info, 0);
  gsize expected = gst_tensor_info_get_size (model_in);
  for (guint d = 0; d < NNS_TENSOR_RANK_LIMIT; ++d)
    padded[d] = (model_in->dimension[d] == 0) ? 1 : model_in->dimension[d];

  ASSERT_TRUE (gst_tensor_dimension_is_equal (padded, model_in->dimension))
      << "the padded form is not the same shape by the framework's own rule; "
         "this test is built on a false premise";
  ASSERT_NE (0, memcmp (padded, model_in->dimension, sizeof (tensor_dim)))
      << "the padded form is byte-identical to the model's, so this case would "
         "agree under either comparison and prove nothing; the fixture needs an "
         "axis the model leaves unset";

  _reshape_count = 0;
  GLogFunc prev_handler = g_log_set_default_handler (_countReshapes, NULL);

  gboolean invoked_ok = TRUE;
  for (guint r = 0; r < rounds && invoked_ok; ++r) {
    GstTensorMemory input, output;

    _SetDynamicInputMeta (&prop, padded);
    input.size = expected;
    input.data = g_malloc0 (input.size);
    output.data = NULL;
    output.size = 0;

    invoked_ok = (sp->invoke (NULL, &prop, data, &input, &output) == 0);

    g_free (input.data);
    g_free (output.data);
  }

  g_log_set_default_handler (prev_handler, NULL);

  EXPECT_TRUE (invoked_ok) << "An invoke at the model's own shape failed.";
  EXPECT_EQ (_reshape_count.load (), 0)
      << "The model was reshaped for a shape it already had, so a pipeline that "
         "spells out its trailing dimensions rebuilds the model on every buffer.";

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case: a genuinely new shape reshapes once, then settles.
 *
 * The other half of the same decision. Skipping a reshape that is needed would
 * be caught by the output; reshaping when one is not needed would not be, so
 * both directions are pinned here rather than only the visible one.
 */
TEST (nnstreamerFilterLiteRT, dynamicInvokeNewShapeReshapesOnce)
{
  void *data = NULL;
  g_autofree gchar *model_file = NULL;
  guint dims[NNS_TENSOR_RANK_LIMIT] = { 0 };

  ASSERT_TRUE (_GetModelFilePath (&model_file, 4));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);
  prop.invoke_dynamic = 1;

  ASSERT_EQ (sp->open (&prop, &data), 0);

  /* batch 3, which the model compiled at batch 1 does not have */
  dims[0] = 4;
  dims[1] = 3;

  _reshape_count = 0;
  GLogFunc prev_handler = g_log_set_default_handler (_countReshapes, NULL);

  gboolean invoked_ok = TRUE;
  for (guint r = 0; r < 3U && invoked_ok; ++r) {
    GstTensorMemory input, output;

    _SetDynamicInputMeta (&prop, dims);
    input.size = sizeof (float) * 4 * 3;
    input.data = g_malloc0 (input.size);
    output.data = NULL;
    output.size = 0;

    invoked_ok = (sp->invoke (NULL, &prop, data, &input, &output) == 0
                  && output.size == sizeof (float) * 4 * 3);

    g_free (input.data);
    g_free (output.data);
  }

  g_log_set_default_handler (prev_handler, NULL);

  EXPECT_TRUE (invoked_ok) << "Invoking at a new shape failed.";
  EXPECT_EQ (_reshape_count.load (), 1)
      << "Expected exactly one reshape for three invokes at one new shape.";

  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case: a dynamic invoke must reject a null tensor array.
 *
 * cpp_invoke hands input and output straight through, so the guard is the
 * only thing between a null and a dereference. It does test prop itself
 * (`if (prop && prop->invoke_dynamic)`), which routes a null prop to the
 * static invoke instead, so that third of the guard cannot be reached from
 * here and is not claimed below.
 */
TEST (nnstreamerFilterLiteRT, dynamicInvokeNullArg_n)
{
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;
  guint dims[NNS_TENSOR_RANK_LIMIT] = { 0 };

  ASSERT_TRUE (_GetModelFilePath (&model_file, 4));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);
  prop.invoke_dynamic = 1;

  ASSERT_EQ (sp->open (&prop, &data), 0);

  dims[0] = 4;
  dims[1] = 1;
  _SetDynamicInputMeta (&prop, dims);

  input.size = sizeof (float) * 4;
  input.data = g_malloc0 (input.size);
  output.data = NULL;
  output.size = 0;

  EXPECT_NE (sp->invoke (NULL, &prop, data, NULL, &output), 0)
      << "A null input tensor array was accepted.";
  EXPECT_NE (sp->invoke (NULL, &prop, data, &input, NULL), 0)
      << "A null output tensor array was accepted.";

  g_free (input.data);
  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case: a dynamic invoke must reject a null input buffer.
 */
TEST (nnstreamerFilterLiteRT, dynamicInvokeNullInputData_n)
{
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;
  guint dims[NNS_TENSOR_RANK_LIMIT] = { 0 };

  ASSERT_TRUE (_GetModelFilePath (&model_file, 4));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);
  prop.invoke_dynamic = 1;

  ASSERT_EQ (sp->open (&prop, &data), 0);

  dims[0] = 4;
  dims[1] = 1;
  _SetDynamicInputMeta (&prop, dims);

  input.size = sizeof (float) * 4;
  input.data = NULL;
  output.data = NULL;
  output.size = 0;

  EXPECT_NE (sp->invoke (NULL, &prop, data, &input, &output), 0)
      << "A null input buffer was accepted.";

  /* no free for output: the invoke never reaches the point that allocates it */
  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case: the element type must be the model's.
 *
 * int32 and float32 are both 4 bytes wide, so a substitution between them
 * changes no dimension and no byte count. With a flexible sink pad the type
 * in prop->input_meta is whatever the buffer's own meta header says, and the
 * framework sizes its check from that same header, so nothing upstream
 * objects either. Without an explicit check the model reads int32 bits as
 * float32 and returns nonsense with a success code.
 */
TEST (nnstreamerFilterLiteRT, dynamicInvokeInputTypeMismatch_n)
{
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;
  guint dims[NNS_TENSOR_RANK_LIMIT] = { 0 };
  GstTensorInfo *info;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 4));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);
  prop.invoke_dynamic = 1;

  ASSERT_EQ (sp->open (&prop, &data), 0);

  /* the model's own shape, so only the type is out of place */
  dims[0] = 4;
  dims[1] = 1;
  _SetDynamicInputMeta (&prop, dims);
  info = gst_tensors_info_get_nth_info (&prop.input_meta, 0);
  info->type = _NNS_INT32;

  input.size = sizeof (int32_t) * 4;
  input.data = g_malloc0 (input.size);
  output.data = NULL;
  output.size = 0;

  EXPECT_NE (sp->invoke (NULL, &prop, data, &input, &output), 0)
      << "An int32 buffer was accepted by a float32 model, which infers on "
         "reinterpreted bits and reports success.";

  g_free (input.data);
  g_free (output.data);
  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case: the input buffer must match the shape it asks for.
 *
 * The copy is sized by the caller, so the two directions fail differently: a
 * buffer larger than the model's tensor overflows it, while a smaller one
 * leaves the tail of the reused buffer holding the last invoke's data and
 * infers on stale bytes instead of failing.
 *
 * Only the first invoke reshapes. It asks for batch 2, is reshaped, and is
 * then rejected by the size check - which leaves the instance at batch 2, so
 * the second invoke asks for a shape it already has, skips the reshape, and
 * reaches the size check by the shared-lock path.
 *
 * One != rejects both, so the second assertion reaches no branch the first
 * does not. It is there for the loosening: narrowing the check to < would
 * stop refusing oversized buffers and restore the overflow, and this is the
 * case that then turns red.
 */
TEST (nnstreamerFilterLiteRT, dynamicInvokeInputSizeMismatch_n)
{
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;
  guint dims[NNS_TENSOR_RANK_LIMIT] = { 0 };

  ASSERT_TRUE (_GetModelFilePath (&model_file, 4));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);
  prop.invoke_dynamic = 1;

  ASSERT_EQ (sp->open (&prop, &data), 0);

  /* ask for batch 2 but hand over a batch 1 buffer */
  dims[0] = 4;
  dims[1] = 2;
  _SetDynamicInputMeta (&prop, dims);

  input.size = sizeof (float) * 4;
  input.data = g_malloc0 (input.size);
  output.data = NULL;
  output.size = 0;

  EXPECT_NE (sp->invoke (NULL, &prop, data, &input, &output), 0)
      << "An input buffer smaller than the requested shape was accepted.";

  g_free (input.data);

  /** and the other way round. 36 B is not 4 floats times any batch, so it
   *  cannot be mistaken for a shape the model would accept */
  input.size = sizeof (float) * 4 * 2 + 4;
  input.data = g_malloc0 (input.size);

  EXPECT_NE (sp->invoke (NULL, &prop, data, &input, &output), 0)
      << "An input buffer larger than the requested shape was accepted.";

  g_free (input.data);
  g_free (output.data);
  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case: a dynamic invoke may not change the number of input
 * tensors; it must fail before ever touching the input buffer.
 */
TEST (nnstreamerFilterLiteRT, dynamicInvokeTensorCountChange_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input[2], output[2];
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 4));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files);
  prop.invoke_dynamic = 1;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_init (&prop.input_meta);
  prop.input_meta.num_tensors = 2; /* the model has exactly one input tensor */
  prop.input_meta.info[0].type = _NNS_FLOAT32;
  prop.input_meta.info[0].dimension[0] = 4;
  prop.input_meta.info[0].dimension[1] = 1;
  prop.input_meta.info[1].type = _NNS_FLOAT32;
  prop.input_meta.info[1].dimension[0] = 4;
  prop.input_meta.info[1].dimension[1] = 1;

  input[0].size = input[1].size = sizeof (gfloat) * 4;
  input[0].data = g_malloc0 (input[0].size);
  input[1].data = g_malloc0 (input[1].size);
  output[0].size = output[1].size = 0;
  output[0].data = output[1].data = NULL;

  ret = sp->invoke (NULL, &prop, data, input, output);
  EXPECT_NE (ret, 0) << "A dynamic invoke changing the input tensor count was accepted.";

  g_free (input[0].data);
  g_free (input[1].data);
  g_free (output[0].data);
  g_free (output[1].data);
  gst_tensors_info_free (&prop.input_meta);
  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case, regression: the same dynamic-capable model still
 * works through the ordinary (non-dynamic) invoke path, at its default
 * (batch = 1) compiled shape, with the caller allocating the output buffer.
 */
TEST (nnstreamerFilterLiteRT, dynamicInvokeStaticPathUnaffected)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file = NULL;

  ASSERT_TRUE (_GetModelFilePath (&model_file, 4));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("litert");
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, "litert", model_files); /* invoke_dynamic defaults to 0 */

  input.size = sizeof (gfloat) * 4;
  output.size = sizeof (gfloat) * 4;
  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);
  for (guint i = 0; i < 4; ++i)
    ((gfloat *) input.data)[i] = (gfloat) i + 0.5f;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  ret = sp->invoke (NULL, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);

  for (guint i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ (((gfloat *) output.data)[i], ((gfloat *) input.data)[i] + 1.0f);

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
