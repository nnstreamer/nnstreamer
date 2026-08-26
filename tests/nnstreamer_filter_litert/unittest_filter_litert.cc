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
  g_signal_connect (sink_handle, "new-data", (GCallback) check_output, &is_float);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10),
      0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

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
  g_signal_connect (sink_handle, "new-data", (GCallback) check_output, &is_float);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10),
      0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

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
