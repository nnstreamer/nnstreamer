/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_executorch.cc
 * @date    3 Sep 2026
 * @brief   Unit test for the ExecuTorch tensor filter sub-plugin
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 * runTest.sh drives the same two models through pipelines, which cannot reach
 * the sub-plugin's error paths: a pipeline that refuses to negotiate looks the
 * same from the outside whether open() rejected the model or the caps simply
 * did not match. These tests call the framework directly instead.
 *
 * The arithmetic is the point of the fixtures, so the golden values are given
 * exactly: sample_3x4_two_input_two_output.pte computes (x + 1.0, y + 2.0) and
 * sample_4x4x4x4x4_two_input_one_output.pte computes x + y. Every value here is
 * a small integer, well inside the range float32 represents exactly, so the
 * expected results are the arithmetic ones and EXPECT_FLOAT_EQ's four-ULP
 * window never has to absorb anything.
 *
 * The *_add_one.pte fixtures and sample_3x4_multi_type.pte add one to each
 * input in its own type (the bool input of the latter is negated instead), and
 * exist to pin the tensor type mapping. sample_3x4_bfloat16_add_one.pte and
 * sample_17_input_bfloat16_output.pte (17 float32 inputs summed into one
 * bfloat16 output) have a type NNStreamer lacks and must be refused at open.
 *
 * sample_3x4_two_input_two_output_unplanned_output.pte computes the same as
 * sample_3x4_two_input_two_output.pte, but its outputs are left out of memory
 * planning, so ExecuTorch writes them straight into the caller's buffers.
 * sample_3x4_aliased_outputs_unplanned_output.pte returns x + 1 twice, a
 * constant tensor of ones and x itself, with the same export setting; the
 * constant is the only output ExecuTorch reports as memory-planned.
 * sample_3x4_scalar_output_unplanned_output.pte sums its input into a rank-0
 * output, also unplanned.
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>

#include <cstring>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api_util.h>

#define FW_NAME "executorch"

/**
 * @brief Build the path of a .pte fixture under tests/test_models/models.
 */
static gchar *
_GetModelFilePath (const gchar *model_name)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  g_autofree gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();

  return g_build_filename (root_path, "tests", "test_models", "models", model_name, NULL);
}

/**
 * @brief Fill in the filter properties the framework callbacks expect.
 */
static void
_SetFilterProp (GstTensorFilterProperties *prop, const gchar **models)
{
  memset (prop, 0, sizeof (GstTensorFilterProperties));
  prop->fwname = FW_NAME;
  prop->fw_opened = 0;
  prop->model_files = models;
  prop->num_models = models ? g_strv_length ((gchar **) models) : 0;
}

/**
 * @brief Check the executorch sub-plugin is registered.
 */
TEST (nnstreamerFilterExecutorch, checkExistence)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);
}

/**
 * @brief Positive case with open/close.
 */
TEST (nnstreamerFilterExecutorch, openClose00)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file = _GetModelFilePath ("sample_3x4_two_input_two_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case with a model path that does not exist.
 */
TEST (nnstreamerFilterExecutorch, openClose00_n)
{
  int ret;
  void *data = NULL;

  const gchar *model_files[] = { "some/invalid/model/path.pte", NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Negative case with an existing file that is not an ExecuTorch program.
 *
 * This is the shape a stale fixture takes: the file is there and readable, and
 * only Module::load() can tell that the runtime will not accept it.
 */
TEST (nnstreamerFilterExecutorch, openClose01_n)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *garbage_file = NULL;
  gint fd;

  fd = g_file_open_tmp ("executorch_garbage_XXXXXX.pte", &garbage_file, NULL);
  ASSERT_GE (fd, 0);
  ASSERT_TRUE (g_file_set_contents (
      garbage_file, "This is not an ExecuTorch program at all.", -1, NULL));
  g_close (fd, NULL);

  const gchar *model_files[] = { garbage_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);

  g_unlink (garbage_file);
}

/**
 * @brief Negative case with no model file given.
 */
TEST (nnstreamerFilterExecutorch, openClose02_n)
{
  int ret;
  void *data = NULL;

  const gchar *model_files[] = { NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief The sub-plugin reads the tensor metadata out of the .pte itself.
 *
 * The dimensions are reversed against the exported torch shapes, so a (3, 4)
 * input is 4:3 here. runTest.sh depends on that mapping through its caps and
 * would only fail obscurely if it changed.
 */
TEST (nnstreamerFilterExecutorch, getModelInfo00)
{
  int ret;
  void *data = NULL;
  GstTensorsInfo in_info, out_info;
  g_autofree gchar *model_file = _GetModelFilePath ("sample_3x4_two_input_two_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  ret = sp->getModelInfo (NULL, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (in_info.num_tensors, 2U);
  EXPECT_EQ (out_info.num_tensors, 2U);

  for (guint i = 0; i < in_info.num_tensors; i++) {
    GstTensorInfo *info = gst_tensors_info_get_nth_info (&in_info, i);
    EXPECT_EQ (info->type, _NNS_FLOAT32);
    EXPECT_EQ (info->dimension[0], 4U);
    EXPECT_EQ (info->dimension[1], 3U);
  }

  for (guint i = 0; i < out_info.num_tensors; i++) {
    GstTensorInfo *info = gst_tensors_info_get_nth_info (&out_info, i);
    EXPECT_EQ (info->type, _NNS_FLOAT32);
    EXPECT_EQ (info->dimension[0], 4U);
    EXPECT_EQ (info->dimension[1], 3U);
  }

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  sp->close (&prop, &data);
}

/**
 * @brief A rank-5 model keeps all five dimensions.
 */
TEST (nnstreamerFilterExecutorch, getModelInfo01)
{
  int ret;
  void *data = NULL;
  GstTensorsInfo in_info, out_info;
  g_autofree gchar *model_file
      = _GetModelFilePath ("sample_4x4x4x4x4_two_input_one_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  ret = sp->getModelInfo (NULL, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (in_info.num_tensors, 2U);
  EXPECT_EQ (out_info.num_tensors, 1U);

  GstTensorInfo *info = gst_tensors_info_get_nth_info (&out_info, 0);
  EXPECT_EQ (info->type, _NNS_FLOAT32);
  for (guint d = 0; d < 5; d++)
    EXPECT_EQ (info->dimension[d], 4U);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  sp->close (&prop, &data);
}

/**
 * @brief Only GET_IN_OUT_INFO is answered.
 */
TEST (nnstreamerFilterExecutorch, getModelInfo00_n)
{
  int ret;
  void *data = NULL;
  GstTensorsInfo in_info, out_info;
  g_autofree gchar *model_file = _GetModelFilePath ("sample_3x4_two_input_two_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  ret = sp->getModelInfo (NULL, &prop, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case: the two outputs carry the two different offsets.
 *
 * A kernel registration failure shows up here as a non-zero invoke() rather
 * than as a wrong number, which is what the .pc fixup in
 * tools/executorch-install.sh exists to prevent.
 */
TEST (nnstreamerFilterExecutorch, invoke00)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input[2], output[2];
  g_autofree gchar *model_file = _GetModelFilePath ("sample_3x4_two_input_two_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  const gsize num_elems = 3 * 4;
  const gsize tensor_size = sizeof (float) * num_elems;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  for (int i = 0; i < 2; i++) {
    input[i].size = tensor_size;
    input[i].data = g_malloc0 (tensor_size);
    output[i].size = tensor_size;
    output[i].data = g_malloc0 (tensor_size);
  }

  for (gsize i = 0; i < num_elems; i++) {
    ((float *) input[0].data)[i] = (float) i;
    ((float *) input[1].data)[i] = (float) i;
  }

  ret = sp->invoke (NULL, &prop, data, input, output);
  EXPECT_EQ (ret, 0);

  for (gsize i = 0; i < num_elems; i++) {
    EXPECT_FLOAT_EQ (((float *) output[0].data)[i], (float) i + 1.0f);
    EXPECT_FLOAT_EQ (((float *) output[1].data)[i], (float) i + 2.0f);
  }

  for (int i = 0; i < 2; i++) {
    g_free (input[i].data);
    g_free (output[i].data);
  }
  sp->close (&prop, &data);
}

/**
 * @brief Positive case: the rank-5 model adds its two inputs.
 */
TEST (nnstreamerFilterExecutorch, invoke01)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input[2], output[1];
  g_autofree gchar *model_file
      = _GetModelFilePath ("sample_4x4x4x4x4_two_input_one_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  const gsize num_elems = 4 * 4 * 4 * 4 * 4;
  const gsize tensor_size = sizeof (float) * num_elems;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  for (int i = 0; i < 2; i++) {
    input[i].size = tensor_size;
    input[i].data = g_malloc0 (tensor_size);
  }
  output[0].size = tensor_size;
  output[0].data = g_malloc0 (tensor_size);

  for (gsize i = 0; i < num_elems; i++) {
    ((float *) input[0].data)[i] = (float) i;
    ((float *) input[1].data)[i] = (float) (2 * i);
  }

  ret = sp->invoke (NULL, &prop, data, input, output);
  EXPECT_EQ (ret, 0);

  for (gsize i = 0; i < num_elems; i++)
    EXPECT_FLOAT_EQ (((float *) output[0].data)[i], (float) (3 * i));

  for (int i = 0; i < 2; i++)
    g_free (input[i].data);
  g_free (output[0].data);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case: invoke() rejects a NULL input or output buffer.
 */
TEST (nnstreamerFilterExecutorch, invoke00_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input[2], output[2];
  g_autofree gchar *model_file = _GetModelFilePath ("sample_3x4_two_input_two_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  const gsize tensor_size = sizeof (float) * 3 * 4;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  for (int i = 0; i < 2; i++) {
    input[i].size = tensor_size;
    input[i].data = g_malloc0 (tensor_size);
    output[i].size = tensor_size;
    output[i].data = g_malloc0 (tensor_size);
  }

  EXPECT_NE (sp->invoke (NULL, &prop, data, NULL, output), 0);
  EXPECT_NE (sp->invoke (NULL, &prop, data, input, NULL), 0);

  for (int i = 0; i < 2; i++) {
    g_free (input[i].data);
    g_free (output[i].data);
  }
  sp->close (&prop, &data);
}

/**
 * @brief Repeated invocations on one open model must not drift.
 *
 * configure_instance() builds its TensorImpl vector once and invoke() only
 * re-points it at the caller's buffers, so a mistake there would show up on
 * the second call rather than the first.
 */
TEST (nnstreamerFilterExecutorch, invokeRepeated)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input[2], output[2];
  g_autofree gchar *model_file = _GetModelFilePath ("sample_3x4_two_input_two_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  const gsize num_elems = 3 * 4;
  const gsize tensor_size = sizeof (float) * num_elems;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  for (int i = 0; i < 2; i++) {
    input[i].size = tensor_size;
    input[i].data = g_malloc0 (tensor_size);
    output[i].size = tensor_size;
    output[i].data = g_malloc0 (tensor_size);
  }

  for (int round = 0; round < 3; round++) {
    for (gsize i = 0; i < num_elems; i++) {
      ((float *) input[0].data)[i] = (float) (i + round);
      ((float *) input[1].data)[i] = (float) (i + round);
    }

    ret = sp->invoke (NULL, &prop, data, input, output);
    EXPECT_EQ (ret, 0) << "invoke failed on round " << round;
    if (ret != 0)
      break;

    for (gsize i = 0; i < num_elems; i++) {
      EXPECT_FLOAT_EQ (((float *) output[0].data)[i], (float) (i + round) + 1.0f);
      EXPECT_FLOAT_EQ (((float *) output[1].data)[i], (float) (i + round) + 2.0f);
    }
  }

  for (int i = 0; i < 2; i++) {
    g_free (input[i].data);
    g_free (output[i].data);
  }
  sp->close (&prop, &data);
}

/**
 * @brief Every tensor type the sub-plugin maps is reported as such.
 *
 * Before these types were mapped, each of them was reported as _NNS_END, which
 * no caps can express, so such a model opened fine and then never negotiated.
 */
TEST (nnstreamerFilterExecutorch, getModelInfoMultiType)
{
  int ret;
  void *data = NULL;
  GstTensorsInfo in_info, out_info;
  g_autofree gchar *model_file = _GetModelFilePath ("sample_3x4_multi_type.pte");
  const tensor_type expected[] = { _NNS_UINT8, _NNS_INT8, _NNS_INT16,
    _NNS_INT32, _NNS_INT64, _NNS_FLOAT64, _NNS_UINT8 };
  const guint num = G_N_ELEMENTS (expected);

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  ret = sp->getModelInfo (NULL, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (in_info.num_tensors, num);
  EXPECT_EQ (out_info.num_tensors, num);

  for (guint i = 0; i < num; i++) {
    GstTensorInfo *in = gst_tensors_info_get_nth_info (&in_info, i);
    GstTensorInfo *out = gst_tensors_info_get_nth_info (&out_info, i);

    EXPECT_EQ (in->type, expected[i]) << "input " << i;
    EXPECT_EQ (out->type, expected[i]) << "output " << i;
    EXPECT_EQ (in->dimension[0], 4U);
    EXPECT_EQ (in->dimension[1], 3U);
    EXPECT_EQ (out->dimension[0], 4U);
    EXPECT_EQ (out->dimension[1], 3U);
  }

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  sp->close (&prop, &data);
}

/**
 * @brief Each non-float32 type is handed to ExecuTorch in its own layout.
 *
 * A tensor passed with the wrong element size would read past or short of the
 * caller's buffer, so every output value is checked, not just the first.
 */
TEST (nnstreamerFilterExecutorch, invokeMultiType)
{
  int ret;
  void *data = NULL;
  const guint num = 7;
  const gsize num_elems = 3 * 4;
  const gsize elem_size[] = { 1, 1, 2, 4, 8, 8, 1 };
  GstTensorMemory input[7], output[7];
  g_autofree gchar *model_file = _GetModelFilePath ("sample_3x4_multi_type.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  for (guint t = 0; t < num; t++) {
    input[t].size = output[t].size = elem_size[t] * num_elems;
    input[t].data = g_malloc0 (input[t].size);
    output[t].data = g_malloc0 (output[t].size);
  }

  for (gsize i = 0; i < num_elems; i++) {
    ((guint8 *) input[0].data)[i] = (guint8) (200 + i);
    ((gint8 *) input[1].data)[i] = (gint8) ((gint) i - 100);
    ((gint16 *) input[2].data)[i] = (gint16) ((gint) i * 1000 - 30000);
    ((gint32 *) input[3].data)[i] = (gint32) i * 100000 - 2000000000;
    ((gint64 *) input[4].data)[i] = (gint64) i * G_GINT64_CONSTANT (10000000000);
    ((gdouble *) input[5].data)[i] = (gdouble) i * 0.25;
    ((guint8 *) input[6].data)[i] = (guint8) (i % 2);
  }

  ret = sp->invoke (NULL, &prop, data, input, output);
  EXPECT_EQ (ret, 0);

  for (gsize i = 0; i < num_elems; i++) {
    EXPECT_EQ (((guint8 *) output[0].data)[i], (guint8) (201 + i));
    EXPECT_EQ (((gint8 *) output[1].data)[i], (gint8) ((gint) i - 99));
    EXPECT_EQ (((gint16 *) output[2].data)[i], (gint16) ((gint) i * 1000 - 29999));
    EXPECT_EQ (((gint32 *) output[3].data)[i], (gint32) i * 100000 - 1999999999);
    EXPECT_EQ (((gint64 *) output[4].data)[i],
        (gint64) i * G_GINT64_CONSTANT (10000000000) + 1);
    EXPECT_DOUBLE_EQ (((gdouble *) output[5].data)[i], (gdouble) i * 0.25 + 1.0);
    EXPECT_EQ (((guint8 *) output[6].data)[i], (guint8) ((i + 1) % 2));
  }

  for (guint t = 0; t < num; t++) {
    g_free (input[t].data);
    g_free (output[t].data);
  }
  sp->close (&prop, &data);
}

#ifdef FLOAT16_SUPPORT
/**
 * @brief A float16 model is reported and run as float16.
 *
 * The values are IEEE 754 half bit patterns, so the test does not depend on
 * which half type the compiler offers.
 */
TEST (nnstreamerFilterExecutorch, invokeFloat16)
{
  int ret;
  void *data = NULL;
  GstTensorsInfo in_info, out_info;
  GstTensorMemory input, output;
  const gsize num_elems = 3 * 4;
  /* 0.0 to 12.0 in half precision */
  const guint16 half_bits[] = { 0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500,
    0x4600, 0x4700, 0x4800, 0x4880, 0x4900, 0x4980, 0x4A00 };
  g_autofree gchar *model_file = _GetModelFilePath ("sample_3x4_float16_add_one.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  ret = sp->getModelInfo (NULL, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (gst_tensors_info_get_nth_info (&in_info, 0)->type, _NNS_FLOAT16);
  EXPECT_EQ (gst_tensors_info_get_nth_info (&out_info, 0)->type, _NNS_FLOAT16);

  input.size = output.size = sizeof (guint16) * num_elems;
  input.data = g_malloc0 (input.size);
  output.data = g_malloc0 (output.size);

  for (gsize i = 0; i < num_elems; i++)
    ((guint16 *) input.data)[i] = half_bits[i];

  ret = sp->invoke (NULL, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);

  for (gsize i = 0; i < num_elems; i++)
    EXPECT_EQ (((guint16 *) output.data)[i], half_bits[i + 1]) << "element " << i;

  g_free (input.data);
  g_free (output.data);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  sp->close (&prop, &data);
}
#else
/**
 * @brief Without float16 support, a float16 model is refused at open.
 */
TEST (nnstreamerFilterExecutorch, openCloseFloat16_n)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file = _GetModelFilePath ("sample_3x4_float16_add_one.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}
#endif

/**
 * @brief A model with a tensor type NNStreamer lacks is refused at open.
 *
 * It used to open and report _NNS_END, leaving the pipeline to fail later
 * with a negotiation error that did not name the cause.
 */
TEST (nnstreamerFilterExecutorch, openCloseBfloat16_n)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file = _GetModelFilePath ("sample_3x4_bfloat16_add_one.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief A model whose output type NNStreamer lacks is refused at open.
 *
 * Its 17 inputs are all float32, so only the output check can refuse it, and
 * by then the input info has spilled into its extra array, which the refusal
 * must free (seen under valgrind).
 */
TEST (nnstreamerFilterExecutorch, openCloseBfloat16Output_n)
{
  int ret;
  void *data = NULL;
  g_autofree gchar *model_file = _GetModelFilePath ("sample_17_input_bfloat16_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Outputs left out of memory planning are written by ExecuTorch itself.
 *
 * Such outputs have no storage of their own, so the right values can only
 * appear in the caller's buffers if the sub-plugin handed those buffers to
 * ExecuTorch before running the model.
 */
TEST (nnstreamerFilterExecutorch, invokeUnplannedOutput)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input[2], output[2];
  g_autofree gchar *model_file
      = _GetModelFilePath ("sample_3x4_two_input_two_output_unplanned_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  const gsize num_elems = 3 * 4;
  const gsize tensor_size = sizeof (float) * num_elems;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  for (int i = 0; i < 2; i++) {
    input[i].size = tensor_size;
    input[i].data = g_malloc0 (tensor_size);
    output[i].size = tensor_size;
    output[i].data = g_malloc0 (tensor_size);
  }

  for (gsize i = 0; i < num_elems; i++) {
    ((float *) input[0].data)[i] = (float) i;
    ((float *) input[1].data)[i] = (float) i;
  }

  ret = sp->invoke (NULL, &prop, data, input, output);
  EXPECT_EQ (ret, 0);

  for (gsize i = 0; i < num_elems; i++) {
    EXPECT_FLOAT_EQ (((float *) output[0].data)[i], (float) i + 1.0f);
    EXPECT_FLOAT_EQ (((float *) output[1].data)[i], (float) i + 2.0f);
  }

  for (int i = 0; i < 2; i++) {
    g_free (input[i].data);
    g_free (output[i].data);
  }
  sp->close (&prop, &data);
}

/**
 * @brief Each invoke writes into the output buffers given to that invoke.
 *
 * Two sets of output buffers alternate, as pooled buffers do in a pipeline.
 * A pointer kept from an earlier invoke would leave the current set stale and
 * overwrite the other one.
 */
TEST (nnstreamerFilterExecutorch, invokeUnplannedOutputRepeated)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input[2], output[2][2];
  g_autofree gchar *model_file
      = _GetModelFilePath ("sample_3x4_two_input_two_output_unplanned_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  const gsize num_elems = 3 * 4;
  const gsize tensor_size = sizeof (float) * num_elems;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  for (int i = 0; i < 2; i++) {
    input[i].size = tensor_size;
    input[i].data = g_malloc0 (tensor_size);
    for (int s = 0; s < 2; s++) {
      output[s][i].size = tensor_size;
      output[s][i].data = g_malloc0 (tensor_size);
    }
  }

  for (int round = 0; round < 4; round++) {
    const int cur = round % 2;
    const int other = 1 - cur;

    for (gsize i = 0; i < num_elems; i++) {
      ((float *) input[0].data)[i] = (float) (i + 10 * round);
      ((float *) input[1].data)[i] = (float) (i + 10 * round);
    }

    ret = sp->invoke (NULL, &prop, data, input, output[cur]);
    EXPECT_EQ (ret, 0) << "invoke failed on round " << round;
    if (ret != 0)
      break;

    for (gsize i = 0; i < num_elems; i++) {
      /* the other set holds the previous round, or zeros before it was used */
      const float prev = (round == 0) ? 0.0f : (float) (i + 10 * (round - 1));

      EXPECT_FLOAT_EQ (((float *) output[cur][0].data)[i], (float) (i + 10 * round) + 1.0f);
      EXPECT_FLOAT_EQ (((float *) output[cur][1].data)[i], (float) (i + 10 * round) + 2.0f);
      EXPECT_FLOAT_EQ (((float *) output[other][0].data)[i], round == 0 ? 0.0f : prev + 1.0f);
      EXPECT_FLOAT_EQ (((float *) output[other][1].data)[i], round == 0 ? 0.0f : prev + 2.0f);
    }
  }

  for (int i = 0; i < 2; i++) {
    g_free (input[i].data);
    for (int s = 0; s < 2; s++)
      g_free (output[s][i].data);
  }
  sp->close (&prop, &data);
}

/**
 * @brief ExecuTorch writes exactly the caller's region, at any element offset.
 *
 * The output buffers start one element into a larger allocation, so they are
 * not 16-byte aligned, and the bytes around them must stay untouched.
 */
TEST (nnstreamerFilterExecutorch, invokeUnplannedOutputOffset)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input[2], output[2];
  guint8 *block[2];
  g_autofree gchar *model_file
      = _GetModelFilePath ("sample_3x4_two_input_two_output_unplanned_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  const gsize num_elems = 3 * 4;
  const gsize tensor_size = sizeof (float) * num_elems;
  const gsize pad = sizeof (float);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  for (int i = 0; i < 2; i++) {
    input[i].size = tensor_size;
    input[i].data = g_malloc0 (tensor_size);
    block[i] = (guint8 *) g_malloc (tensor_size + 2 * pad);
    memset (block[i], 0xA5, tensor_size + 2 * pad);
    output[i].size = tensor_size;
    output[i].data = block[i] + pad;
  }

  for (gsize i = 0; i < num_elems; i++) {
    ((float *) input[0].data)[i] = (float) i;
    ((float *) input[1].data)[i] = (float) i;
  }

  ret = sp->invoke (NULL, &prop, data, input, output);
  EXPECT_EQ (ret, 0);

  for (int t = 0; t < 2; t++) {
    for (gsize i = 0; i < num_elems; i++)
      EXPECT_FLOAT_EQ (((float *) output[t].data)[i], (float) i + 1.0f + t);
    for (gsize b = 0; b < pad; b++) {
      EXPECT_EQ (block[t][b], 0xA5);
      EXPECT_EQ (block[t][pad + tensor_size + b], 0xA5);
    }
  }

  for (int i = 0; i < 2; i++) {
    g_free (input[i].data);
    g_free (block[i]);
  }
  sp->close (&prop, &data);
}

/**
 * @brief Outputs that share a tensor, are constant or echo the input.
 *
 * sample_3x4_aliased_outputs_unplanned_output.pte returns x + 1 twice, a
 * constant tensor of ones and x itself. The two x + 1 outputs are one tensor,
 * so redirecting both leaves the result in the second buffer only; x is the
 * caller's input buffer; and the constant counts as memory-planned, so it
 * cannot be redirected. Every buffer must still end up with its value, on
 * each invoke with its own buffers.
 */
TEST (nnstreamerFilterExecutorch, invokeUnplannedAliasedOutputs)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output[2][4];
  g_autofree gchar *model_file
      = _GetModelFilePath ("sample_3x4_aliased_outputs_unplanned_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  const gsize num_elems = 3 * 4;
  const gsize tensor_size = sizeof (float) * num_elems;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  input.size = tensor_size;
  input.data = g_malloc0 (tensor_size);
  for (int s = 0; s < 2; s++) {
    for (int t = 0; t < 4; t++) {
      output[s][t].size = tensor_size;
      output[s][t].data = g_malloc0 (tensor_size);
    }
  }

  for (int round = 0; round < 2; round++) {
    for (gsize i = 0; i < num_elems; i++)
      ((float *) input.data)[i] = (float) (i + 10 * round);

    ret = sp->invoke (NULL, &prop, data, &input, output[round]);
    EXPECT_EQ (ret, 0) << "invoke failed on round " << round;
    if (ret != 0)
      break;

    for (gsize i = 0; i < num_elems; i++) {
      const float x = (float) (i + 10 * round);

      EXPECT_FLOAT_EQ (((float *) output[round][0].data)[i], x + 1.0f);
      EXPECT_FLOAT_EQ (((float *) output[round][1].data)[i], x + 1.0f);
      EXPECT_FLOAT_EQ (((float *) output[round][2].data)[i], 1.0f);
      EXPECT_FLOAT_EQ (((float *) output[round][3].data)[i], x);
    }
  }

  g_free (input.data);
  for (int s = 0; s < 2; s++) {
    for (int t = 0; t < 4; t++)
      g_free (output[s][t].data);
  }
  sp->close (&prop, &data);
}

/**
 * @brief A rank-0 output is written into a buffer of its real size.
 *
 * The tensor info of a rank-0 tensor has no dimension at all, so its size
 * there is 0; the buffer check has to use the size ExecuTorch reports. Such a
 * model does not negotiate in a pipeline for the same reason, so only a direct
 * caller of the filter API reaches this path.
 */
TEST (nnstreamerFilterExecutorch, invokeUnplannedScalarOutput)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file
      = _GetModelFilePath ("sample_3x4_scalar_output_unplanned_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  const gsize num_elems = 3 * 4;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  input.size = sizeof (float) * num_elems;
  input.data = g_malloc0 (input.size);
  output.size = sizeof (float);
  output.data = g_malloc0 (output.size);

  for (gsize i = 0; i < num_elems; i++)
    ((float *) input.data)[i] = (float) i;

  ret = sp->invoke (NULL, &prop, data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_FLOAT_EQ (*(float *) output.data, 66.0f);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief A rank-0 output is not written into a buffer claimed to be empty.
 */
TEST (nnstreamerFilterExecutorch, invokeUnplannedScalarShortOutput_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  g_autofree gchar *model_file
      = _GetModelFilePath ("sample_3x4_scalar_output_unplanned_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  input.size = sizeof (float) * 3 * 4;
  input.data = g_malloc0 (input.size);
  output.data = g_malloc0 (sizeof (float));
  output.size = 0;
  *(float *) output.data = -1.0f;

  EXPECT_NE (sp->invoke (NULL, &prop, data, &input, &output), 0);
  EXPECT_FLOAT_EQ (*(float *) output.data, -1.0f);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief A memory-planned output is not copied into a buffer that is too small.
 *
 * The second output buffer claims one element less than the tensor needs. The
 * allocation behind it is full size, so a copy past the claimed size shows up
 * in its last element instead of corrupting the heap. The invoke is refused
 * before the model runs, so the first output is not written either.
 */
TEST (nnstreamerFilterExecutorch, invokeShortOutput_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input[2], output[2];
  g_autofree gchar *model_file = _GetModelFilePath ("sample_3x4_two_input_two_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  const gsize num_elems = 3 * 4;
  const gsize tensor_size = sizeof (float) * num_elems;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  for (int i = 0; i < 2; i++) {
    input[i].size = tensor_size;
    input[i].data = g_malloc0 (tensor_size);
    output[i].size = tensor_size;
    output[i].data = g_malloc0 (tensor_size);
  }
  output[1].size = tensor_size - sizeof (float);
  ((float *) output[1].data)[num_elems - 1] = -1.0f;

  EXPECT_NE (sp->invoke (NULL, &prop, data, input, output), 0);
  EXPECT_FLOAT_EQ (((float *) output[1].data)[num_elems - 1], -1.0f);
  for (gsize i = 0; i < num_elems; i++)
    EXPECT_FLOAT_EQ (((float *) output[0].data)[i], 0.0f);

  for (int i = 0; i < 2; i++) {
    g_free (input[i].data);
    g_free (output[i].data);
  }
  sp->close (&prop, &data);
}

/**
 * @brief An output buffer too small for an unplanned output is refused.
 *
 * ExecuTorch would write the whole tensor into it, so it must not be handed
 * over at all.
 */
TEST (nnstreamerFilterExecutorch, invokeUnplannedShortOutput_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input[2], output[2];
  g_autofree gchar *model_file
      = _GetModelFilePath ("sample_3x4_two_input_two_output_unplanned_output.pte");

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_IS_REGULAR));

  const gchar *model_files[] = { model_file, NULL };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find (FW_NAME);
  ASSERT_TRUE (sp != nullptr);

  GstTensorFilterProperties prop;
  _SetFilterProp (&prop, model_files);

  const gsize num_elems = 3 * 4;
  const gsize tensor_size = sizeof (float) * num_elems;

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  for (int i = 0; i < 2; i++) {
    input[i].size = tensor_size;
    input[i].data = g_malloc0 (tensor_size);
    output[i].size = tensor_size;
    output[i].data = g_malloc0 (tensor_size);
  }
  output[1].size = tensor_size - sizeof (float);
  ((float *) output[1].data)[num_elems - 1] = -1.0f;

  EXPECT_NE (sp->invoke (NULL, &prop, data, input, output), 0);
  EXPECT_FLOAT_EQ (((float *) output[1].data)[num_elems - 1], -1.0f);

  for (int i = 0; i < 2; i++) {
    g_free (input[i].data);
    g_free (output[i].data);
  }
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
