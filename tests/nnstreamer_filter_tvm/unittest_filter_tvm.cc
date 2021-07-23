/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_tvm.cc
 * @date    30 Apr 2021
 * @brief   Unit test for TVM tensor filter sub-plugin
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 * @notes   Currently the test model can be executed only on x86_64
 *
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <unittest_util.h>

#include <nnstreamer_plugin_api_filter.h>

#if defined(__aarch64__)
#define ARCH "aarch64"
#elif defined(__arm__)
#define ARCH "arm"
#elif defined(__x86_64__)
#define ARCH "x86_64"
#else
#define ARCH "invalid"
#endif

/**
 * @brief Set tensor filter properties
 */
static void
_set_filter_prop (GstTensorFilterProperties *prop, const gchar *name, const gchar **models)
{
  memset (prop, 0, sizeof (GstTensorFilterProperties));
  prop->fwname = name;
  prop->fw_opened = 0;
  prop->model_files = models;
  prop->num_models = g_strv_length ((gchar **) models);
}

/**
 * @brief internal function to get model filename
 */
static void
_get_model_file (gchar ** model_file)
{
  const gchar * root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  if (!root_path) {
    root_path = g_get_current_dir();
  }
  gchar * model_name = g_strdup_printf("tvm_add_one_%s.so_", ARCH);
  *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", model_name, NULL);
  g_free (model_name);
}

/**
 * @brief Signal to validate new output data
 */
static void
_check_output (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  GstMemory *mem_res;
  GstMapInfo info_res;
  gboolean mapped;
  gfloat *output;

  mem_res = gst_buffer_get_memory (buffer, 0);
  mapped = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
  ASSERT_TRUE (mapped);
  output = (gfloat *) info_res.data;

  for (guint i = 0; i < 10; i++) {
    EXPECT_EQ (1, output[i]);
  }
}

/**
 * @brief Negative test case with wrong model file
 */
TEST (nnstreamerFilterTvm, openClose00_n)
{
  int ret;
  void *data = NULL;
  const gchar *model_files[] = {
    "temp.so",
    NULL,
  };
  GstTensorFilterProperties prop;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tvm");
  EXPECT_NE (sp, nullptr);

  _set_filter_prop (&prop, "tvm", model_files);
  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Positive case with open/close
 */
TEST (nnstreamerFilterTvm, openClose00)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  GstTensorFilterProperties prop;
  _get_model_file (&model_file);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tvm");
  EXPECT_NE (sp, nullptr);
  _set_filter_prop (&prop, "tvm", model_files);

  /* close before open */
  sp->close (&prop, &data);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);

  /* double close */
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Positive case with successful getModelInfo
 */
TEST (nnstreamerFilterTvm, getModelInfo00)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  GstTensorFilterProperties prop;
  _get_model_file (&model_file);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tvm");
  EXPECT_NE (sp, nullptr);
  _set_filter_prop (&prop, "tvm", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  GstTensorsInfo in_info, out_info;

  ret = sp->getModelInfo (NULL, NULL, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (in_info.num_tensors, 1U);
  EXPECT_EQ (in_info.info[0].dimension[0], 3U);
  EXPECT_EQ (in_info.info[0].dimension[1], 480U);
  EXPECT_EQ (in_info.info[0].dimension[2], 640U);
  EXPECT_EQ (in_info.info[0].dimension[3], 1U);
  EXPECT_EQ (in_info.info[0].type, _NNS_FLOAT32);
  EXPECT_EQ (out_info.num_tensors, 1U);
  EXPECT_EQ (out_info.info[0].dimension[0], 3U);
  EXPECT_EQ (out_info.info[0].dimension[1], 480U);
  EXPECT_EQ (out_info.info[0].dimension[2], 640U);
  EXPECT_EQ (out_info.info[0].dimension[3], 1U);
  EXPECT_EQ (out_info.info[0].type, _NNS_FLOAT32);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Negative case calling getModelInfo before open
 */
TEST (nnstreamerFilterTvm, getModelInfo00_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  GstTensorFilterProperties prop;
  _get_model_file (&model_file);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tvm");
  EXPECT_NE (sp, nullptr);

  GstTensorsInfo in_info, out_info;

  ret = sp->getModelInfo (NULL, NULL, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);
  _set_filter_prop (&prop, "tvm", model_files);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Negative case with invalid argument
 */
TEST (nnstreamerFilterTvm, getModelInfo01_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  GstTensorFilterProperties prop;
  _get_model_file (&model_file);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tvm");
  EXPECT_NE (sp, nullptr);
  _set_filter_prop (&prop, "tvm", model_files);

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
 * @brief Negative test case with invoke before open
 */
TEST (nnstreamerFilterTvm, invoke00_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  gchar *model_file;
  GstTensorFilterProperties prop;
  _get_model_file (&model_file);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tvm");
  EXPECT_NE (sp, nullptr);
  _set_filter_prop (&prop, "tvm", model_files);

  ret = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_NE (ret, 0);

  g_free (model_file);
  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case with invalid input/output
 */
TEST (nnstreamerFilterTvm, invoke01_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  gchar *model_file;
  GstTensorFilterProperties prop;
  _get_model_file (&model_file);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tvm");
  EXPECT_NE (sp, nullptr);
  _set_filter_prop (&prop, "tvm", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, nullptr);

  output.size = input.size = sizeof (float);

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);
  ((float *) input.data)[0] = 10.0;

  /* catching assertion error */
  EXPECT_DEATH (sp->invoke (NULL, NULL, data, NULL, &output), "");
  EXPECT_DEATH (sp->invoke (NULL, NULL, data, &input, NULL), "");

  g_free (model_file);
  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Negative test case with invalid private_data
 */
TEST (nnstreamerFilterTvm, invoke02_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  gchar *model_file;
  GstTensorFilterProperties prop;
  _get_model_file (&model_file);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tvm");
  EXPECT_NE (sp, nullptr);
  _set_filter_prop (&prop, "tvm", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, nullptr);

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);
  ((float *) input.data)[0] = 10.0;

  /* unsucessful invoke */
  ret = sp->invoke (NULL, NULL, NULL, &input, &output);
  EXPECT_NE (ret, 0);

  g_free (model_file);
  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case with invoke for tvm model
 */
TEST (nnstreamerFilterTvm, invoke00)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  gchar *model_file;
  GstTensorFilterProperties prop;
  _get_model_file (&model_file);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  output.size = input.size = sizeof (float) * 3 * 640 * 480 * 1;

  /* alloc input data without alignment */
  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);
  memset (input.data, 0, input.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tvm");
  EXPECT_NE (sp, nullptr);
  _set_filter_prop (&prop, "tvm", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, (void *) NULL);
  ret = sp->invoke (NULL, NULL, data, &input, &output);

  EXPECT_EQ (ret, 0);
  EXPECT_EQ (static_cast<float *> (output.data)[0], (float) (1.0));

  g_free (model_file);
  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case to launch gst pipeline
 */
TEST (nnstreamerFilterTvm, launch00)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file;
  _get_model_file (&model_file);

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=480,height=640 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-255.0 ! tensor_filter framework=tvm model=\"%s\" ! tensor_sink name=sink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  EXPECT_NE (gstpipe, nullptr);

  GstElement *sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sink");
  EXPECT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) _check_output, NULL);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
}

/**
 * @brief Negative case with invalid model path
 */
TEST (nnstreamerFilterTvm, launch00_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;

  /* model file does not exist */
  gchar *model_file = g_build_filename ("temp.so", NULL);

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=480,height=640 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-255.0 ! tensor_filter framework=tvm model=\"%s\" ! fakesink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
}

/**
 * @brief Negative case with incorrect tensor meta
 */
TEST (nnstreamerFilterTvm, launch01_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file;
  _get_model_file (&model_file);

  /* dimension does not match with the model */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=480 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-255.0 ! tensor_filter framework=tvm model=\"%s\" ! fakesink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

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
