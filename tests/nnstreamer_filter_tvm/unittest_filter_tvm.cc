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
#include <nnstreamer_conf.h>
#include <unittest_util.h>

#include <tensor_common.h>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_util.h>

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
 * @brief Test Fixture class for a tensor-filter TVM functionality
 */
class NNStreamerFilterTVMTest : public ::testing::Test
{
protected:
  const GstTensorFilterFramework *sp;
  const gchar *wrong_model_files[2];
  const gchar *proper_model_files[2];
  gchar *model_file;
  gchar *pipeline;
  GstElement *gstpipe;
  GstElement *sink_handle;
  GstTensorMemory input;
  GstTensorMemory output;

  /**
   * @brief Set the model file for testing
   */
  gchar *SetModelFile ()
  {
    const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
    g_autofree gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
    g_autofree gchar *model_name = g_strdup_printf ("tvm_add_one_%s%s_", ARCH, NNSTREAMER_SO_FILE_EXTENSION);

    return g_build_filename (root_path, "tests", "test_models", "models", model_name, NULL);
  }

public:
  /**
   * @brief Construct a new NNStreamerFilterTVMTest object
   */
  NNStreamerFilterTVMTest ()
    : sp(nullptr), model_file(nullptr), pipeline(nullptr), gstpipe(nullptr), sink_handle(nullptr)
  {
  }

  /**
   * @brief Set tensor filter properties
   */
  void SetFilterProperty (GstTensorFilterProperties *prop, const gchar **models)
  {
    memset (prop, 0, sizeof (GstTensorFilterProperties));
    prop->fwname = "tvm";
    prop->fw_opened = 0;
    prop->model_files = models;
    prop->num_models = g_strv_length ((gchar **) models);
  }

  /**
   * @brief SetUp method for each test case
   */
  void SetUp () override
  {
    wrong_model_files[0] = "temp.so";
    wrong_model_files[1] = NULL;

    model_file = SetModelFile();
    proper_model_files[0] = model_file;
    proper_model_files[1] = NULL;

    input.size = output.size = 0;
    input.data = nullptr;
    output.data = nullptr;

    sp = nnstreamer_filter_find ("tvm");
  }

  /**
   * @brief TearDown method for each test case
   */
  void TearDown () override
  {
    g_free (model_file);
    g_free (pipeline);

    g_clear_object (&gstpipe);
    g_clear_object (&sink_handle);

    g_free (input.data);
    g_free (output.data);
  }

  /**
   * @brief Signal handler to validate new output data
   */
  static void
  CheckOutput (GstElement *element, GstBuffer *buffer, gpointer user_data)
  {
    GstMemory *mem_res;
    GstMapInfo info_res;
    gboolean mapped;
    gfloat *output;
    UNUSED (element);
    UNUSED (user_data);

    mem_res = gst_buffer_peek_memory (buffer, 0);
    mapped = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
    ASSERT_TRUE (mapped);
    output = (gfloat *) info_res.data;

    for (guint i = 0; i < 10; i++) {
      EXPECT_EQ (1, output[i]);
    }

    gst_memory_unmap (mem_res, &info_res);
  }
};

/**
 * @brief Negative test case with wrong model file
 */
TEST_F (NNStreamerFilterTVMTest, openClose00_n)
{
  int ret;
  void *data = NULL;
  GstTensorFilterProperties prop;

  EXPECT_NE (sp, nullptr);

  /* Test */
  SetFilterProperty (&prop, wrong_model_files);
  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Positive case with open/close
 */
TEST_F (NNStreamerFilterTVMTest, openClose00)
{
  int ret;
  void *data = NULL;
  GstTensorFilterProperties prop;

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));
  EXPECT_NE (sp, nullptr);

  /* Test */
  SetFilterProperty (&prop, proper_model_files);

  /* close before open */
  sp->close (&prop, &data);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);

  /* double close */
  sp->close (&prop, &data);
}

/**
 * @brief Positive case with successful getModelInfo
 */
TEST_F (NNStreamerFilterTVMTest, getModelInfo00)
{
  int ret;
  void *data = NULL;
  GstTensorFilterProperties prop;

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));
  EXPECT_NE (sp, nullptr);
  SetFilterProperty (&prop, proper_model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /* Test */
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
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Negative case calling getModelInfo before open
 */
TEST_F (NNStreamerFilterTVMTest, getModelInfo00_n)
{
  int ret;
  void *data = NULL;
  GstTensorFilterProperties prop;
  GstTensorsInfo in_info, out_info;

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));
  EXPECT_NE (sp, nullptr);

  /* Test */
  ret = sp->getModelInfo (NULL, NULL, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);
  SetFilterProperty (&prop, proper_model_files);

  sp->close (&prop, &data);
}

/**
 * @brief Negative case with invalid argument
 */
TEST_F (NNStreamerFilterTVMTest, getModelInfo01_n)
{
  int ret;
  void *data = NULL;
  GstTensorFilterProperties prop;
  GstTensorsInfo in_info, out_info;

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));
  EXPECT_NE (sp, nullptr);
  SetFilterProperty (&prop, proper_model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);

  /* Test: not supported */
  ret = sp->getModelInfo (NULL, NULL, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);

  sp->close (&prop, &data);
}

/**
 * @brief Negative test case with invoke before open
 */
TEST_F (NNStreamerFilterTVMTest, invoke00_n)
{
  int ret;
  void *data = NULL;
  GstTensorFilterProperties prop;

  output.size = input.size = sizeof (float) * 1;
  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));
  EXPECT_NE (sp, nullptr);
  SetFilterProperty (&prop, proper_model_files);

  /* Test */
  ret = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_NE (ret, 0);

  sp->close (&prop, &data);
}

/**
 * @brief Negative case with invalid input/output
 */
TEST_F (NNStreamerFilterTVMTest, invoke01_n)
{
  int ret;
  void *data = NULL;
  GstTensorFilterProperties prop;

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));
  EXPECT_NE (sp, nullptr);
  SetFilterProperty (&prop, proper_model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, nullptr);

  /* Test */
  output.size = input.size = sizeof (float);
  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);
  ((float *) input.data)[0] = 10.0;

  /* catching assertion error */
  EXPECT_DEATH (sp->invoke (NULL, NULL, data, NULL, &output), "");
  EXPECT_DEATH (sp->invoke (NULL, NULL, data, &input, NULL), "");

  sp->close (&prop, &data);
}

/**
 * @brief Negative test case with invalid private_data
 */
TEST_F (NNStreamerFilterTVMTest, invoke02_n)
{
  int ret;
  void *data = NULL;
  GstTensorFilterProperties prop;

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));
  EXPECT_NE (sp, nullptr);
  SetFilterProperty (&prop, proper_model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, nullptr);

  output.size = input.size = sizeof (float) * 1;

  /* Test: unsucessful invoke */
  ret = sp->invoke (NULL, NULL, NULL, &input, &output);
  EXPECT_NE (ret, 0);

  sp->close (&prop, &data);
}

/**
 * @brief Positive case with invoke for tvm model
 */
TEST_F (NNStreamerFilterTVMTest, invoke00)
{
  int ret;
  void *data = NULL;
  GstTensorFilterProperties prop;

  output.size = input.size = sizeof (float) * 3 * 640 * 480 * 1;

  /* alloc input data without alignment */
  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);
  memset (input.data, 0, input.size);

  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));
  EXPECT_NE (sp, nullptr);
  SetFilterProperty (&prop, proper_model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, (void *) NULL);
  ret = sp->invoke (NULL, NULL, data, &input, &output);

  EXPECT_EQ (ret, 0);
  EXPECT_EQ (static_cast<float *> (output.data)[0], (float) (1.0));

  sp->close (&prop, &data);
}

/**
 * @brief Positive case to launch gst pipeline
 */
TEST_F (NNStreamerFilterTVMTest, launch00)
{
  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=480,height=640 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-255.0 ! tensor_filter framework=tvm model=\"%s\" custom=device:CPU,num_input_tensors:1 ! tensor_sink name=sink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sink");
  EXPECT_NE (sink_handle, nullptr);
  g_signal_connect (sink_handle, "new-data", (GCallback) NNStreamerFilterTVMTest::CheckOutput, NULL);

  /* Test */
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief Negative case with invalid model path
 */
TEST_F (NNStreamerFilterTVMTest, launch00_n)
{
  /* model file does not exist */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=480,height=640 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-255.0 ! tensor_filter framework=tvm model=\"%s\" ! fakesink",
      "temp.so");

  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief Negative case with incorrect tensor meta
 */
TEST_F (NNStreamerFilterTVMTest, launch01_n)
{
  /* Test: dimension does not match with the model */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=480 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-255.0 ! tensor_filter framework=tvm model=\"%s\" ! fakesink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief Negative case with invalid custom property (num_input_tensors)
 */
TEST_F (NNStreamerFilterTVMTest, launchInvalidInputNum_n)
{
  /* Test: invalid custom property num_input_tensors should be bigger than 0 */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=480,height=640 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-255.0 ! tensor_filter framework=tvm model=\"%s\" custom=num_input_tensors:0 ! fakesink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief Negative case with invalid custom property (device)
 */
TEST_F (NNStreamerFilterTVMTest, launchInvalidDevice_n)
{
  /* Test: invalid custom property: invalid device */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=480,height=640 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-255.0 ! tensor_filter framework=tvm model=\"%s\" custom=device:INVALID_DEVICE ! fakesink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
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
