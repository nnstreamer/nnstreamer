/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_snpe.cc
 * @date    16 Sep 2021
 * @brief   Unit test for snpe tensor filter sub-plugin
 * @author  Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <unittest_util.h>

#include <tensor_common.h>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_util.h>

/**
 * @brief internal function to get model filename
 */
static gboolean
_GetModelFilePath (gchar ** model_file, gboolean is_float_model)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
  std::string model_name = is_float_model ? "add2_float.dlc" : "add2_uint8.dlc";

  *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", model_name.c_str (), NULL);

  g_free (root_path);

  return g_file_test (*model_file, G_FILE_TEST_EXISTS);
}

/**
 * @brief Set tensor filter properties
 */
static void
_SetFilterProp (GstTensorFilterProperties *prop, const gchar *name, const gchar **models, gboolean is_float_model)
{
  memset (prop, 0, sizeof (GstTensorFilterProperties));
  prop->fwname = name;
  prop->fw_opened = 0;
  prop->model_files = models;
  prop->num_models = g_strv_length ((gchar **) models);
  if (!is_float_model)
    prop->custom_properties = "InputType:uint8,OutputType:uint8";
}

/**
 * @brief Positive case with successful getModelInfo
 */
TEST (nnstreamerFilterSnpe, getModelInfo00)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  GstTensorFilterProperties prop;
  ASSERT_TRUE (_GetModelFilePath (&model_file, TRUE));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);
  _SetFilterProp (&prop, "snpe", model_files, TRUE);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  GstTensorsInfo in_info, out_info;

  ret = sp->getModelInfo (NULL, NULL, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (in_info.num_tensors, 1U);
  EXPECT_EQ (in_info.info[0].dimension[0], 1U);
  EXPECT_EQ (in_info.info[0].dimension[1], 1U);
  EXPECT_EQ (in_info.info[0].dimension[2], 1U);
  EXPECT_EQ (in_info.info[0].dimension[3], 1U);
  EXPECT_EQ (in_info.info[0].type, _NNS_FLOAT32);

  EXPECT_EQ (out_info.num_tensors, 1U);
  EXPECT_EQ (out_info.info[0].dimension[0], 1U);
  EXPECT_EQ (out_info.info[0].dimension[1], 1U);
  EXPECT_EQ (out_info.info[0].dimension[2], 1U);
  EXPECT_EQ (out_info.info[0].dimension[3], 1U);
  EXPECT_EQ (out_info.info[0].type, _NNS_FLOAT32);

  sp->close (&prop, &data);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  g_free (model_file);
}

/**
 * @brief Negative case calling getModelInfo before open
 */
TEST (nnstreamerFilterSnpe, getModelInfo01_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  GstTensorFilterProperties prop;
  ASSERT_TRUE (_GetModelFilePath (&model_file, TRUE));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);

  GstTensorsInfo in_info, out_info;

  ret = sp->getModelInfo (NULL, NULL, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);
  _SetFilterProp (&prop, "snpe", model_files, TRUE);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Negative case with invalid argument
 */
TEST (nnstreamerFilterSnpe, getModelInfo02_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  GstTensorFilterProperties prop;
  ASSERT_TRUE (_GetModelFilePath (&model_file, TRUE));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);
  _SetFilterProp (&prop, "snpe", model_files, TRUE);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);

  GstTensorsInfo in_info, out_info;

  /* not supported */
  ret = sp->getModelInfo (NULL, NULL, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Test snpe subplugin with successful invoke for sample dlc model (input data type: float)
 */
TEST (nnstreamerFilterSnpe, invoke00)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  gchar *model_file;
  GstTensorFilterProperties prop;

  ASSERT_TRUE (_GetModelFilePath (&model_file, TRUE));
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);
  _SetFilterProp (&prop, "snpe", model_files, TRUE);

  output.size = input.size = sizeof (float) * 1;
  input.data = g_malloc0 (input.size);
  output.data = g_malloc0 (output.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke successful */
  ret = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (static_cast<float *> (output.data)[0], 2.0);

  static_cast<float *> (input.data)[0] = 10.0;
  ret = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (static_cast<float *> (output.data)[0], 12.0);

  static_cast<float *> (input.data)[0] = 1.0;
  ret = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (static_cast<float *> (output.data)[0], 3.0);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Test snpe subplugin with successful invoke for sample dlc model (input data type: uint8)
 */
TEST (nnstreamerFilterSnpe, invoke01)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  gchar *model_file;
  GstTensorFilterProperties prop;

  ASSERT_TRUE (_GetModelFilePath (&model_file, FALSE));
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);
  _SetFilterProp (&prop, "snpe", model_files, FALSE);

  output.size = input.size = sizeof (uint8_t) * 1;
  input.data = g_malloc0 (input.size);
  output.data = g_malloc0 (output.size);

  ret = sp->open (&prop, &data);
  ASSERT_EQ (ret, 0);

  /** invoke successful */
  ret = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (static_cast<uint8_t *> (output.data)[0], 2);

  static_cast<uint8_t *> (input.data)[0] = 10;
  ret = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (static_cast<uint8_t *> (output.data)[0], 12);

  static_cast<uint8_t *> (input.data)[0] = 1;
  ret = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (static_cast<uint8_t *> (output.data)[0], 3);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Negative case with invalid input/output
 */
TEST (nnstreamerFilterSnpe, invoke01_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  gchar *model_file;
  GstTensorFilterProperties prop;

  ASSERT_TRUE (_GetModelFilePath (&model_file, TRUE));
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);
  _SetFilterProp (&prop, "snpe", model_files, TRUE);

  output.size = input.size = sizeof (float) * 1;
  input.data = g_malloc0 (input.size);
  output.data = g_malloc0 (output.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /* catching assertion error */
  EXPECT_DEATH (sp->invoke (NULL, NULL, data, NULL, &output), "");
  EXPECT_DEATH (sp->invoke (NULL, NULL, data, &input, NULL), "");

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Negative test case with invalid private_data
 */
TEST (nnstreamerFilterSnpe, invoke02_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  gchar *model_file;
  GstTensorFilterProperties prop;

  ASSERT_TRUE (_GetModelFilePath (&model_file, TRUE));
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("snpe");
  ASSERT_TRUE (sp != nullptr);
  _SetFilterProp (&prop, "snpe", model_files, TRUE);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, nullptr);

  output.size = input.size = sizeof (float) * 1;
  input.data = g_malloc0 (input.size);
  output.data = g_malloc0 (output.size);

  /* unsuccessful invoke with NULL priv_data */
  ret = sp->invoke (NULL, NULL, NULL, &input, &output);
  EXPECT_NE (ret, 0);

  g_free (model_file);
  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case to launch gst pipeline
 */
TEST (nnstreamerFilterSnpe, launch00)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file;
  ASSERT_TRUE (_GetModelFilePath (&model_file, TRUE));

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=GRAY8,width=1,height=1 ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tensor_filter framework=snpe model=\"%s\" ! tensor_sink name=sink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
}

/**
 * @brief Negative case to launch gst pipeline: wrong input dimension
 */
TEST (nnstreamerFilterSnpe, launch01_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file;
  ASSERT_TRUE (_GetModelFilePath (&model_file, TRUE));

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=3,height=3 ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tensor_filter framework=snpe model=\"%s\" ! tensor_sink name=sink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
}

/**
 * @brief Positive case to launch gst pipeline with user buffer
 */
TEST (nnstreamerFilterSnpe, launch03)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file;
  ASSERT_TRUE (_GetModelFilePath (&model_file, TRUE));

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=GRAY8,width=1,height=1 ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tensor_filter framework=snpe model=\"%s\" custom=UserBuffer:true ! tensor_sink name=sink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
}

/**
 * @brief Negative case to launch gst pipeline: only float type is supported by user buffer
 */
TEST (nnstreamerFilterSnpe, launch04_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file;
  ASSERT_TRUE (_GetModelFilePath (&model_file, TRUE));

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=GRAY8,width=1,height=1 ! tensor_converter ! tensor_filter framework=snpe model=\"%s\" custom=UserBuffer:true,InputType:uint8,OutputType:uint8 ! tensor_sink name=sink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
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
