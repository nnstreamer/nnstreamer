/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_python3.cc
 * @date    26 Mar 2024
 * @brief   Unit test for Python3 tensor filter sub-plugin
 * @author  Yelin Jeong <yelini.jeong@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <unittest_util.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>

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
 * @brief Test subplugin existence.
 */
TEST (nnstreamerFilterPython3, checkExistence)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);
}

/**
 * @brief Negative test case with invalid model file path
 */
TEST (nnstreamerFilterPython3, openClose00_n)
{
  int ret;
  void *data = NULL;
  const gchar *model_files[] = {
    "some/invalid/model/path.py",
    NULL,
  };
  GstTensorFilterProperties prop;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);

  _SetFilterProp (&prop, "python3", model_files);
  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Positive case with open/close
 */
TEST (nnstreamerFilterPython3, openClose01)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.py", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "python3", model_files);

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
TEST (nnstreamerFilterPython3, getModelInfo00)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.py", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "python3", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  GstTensorsInfo in_info, out_info;

  ret = sp->getModelInfo (NULL, NULL, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  constexpr uint32_t CHANNEL = 3;
  constexpr uint32_t WIDTH = 280;
  constexpr uint32_t HEIGHT = 40;

  EXPECT_EQ (in_info.num_tensors, 1U);
  EXPECT_EQ (in_info.info[0].dimension[0], CHANNEL);
  EXPECT_EQ (in_info.info[0].dimension[1], WIDTH);
  EXPECT_EQ (in_info.info[0].dimension[2], HEIGHT);
  EXPECT_EQ (in_info.info[0].type, _NNS_UINT8);
  EXPECT_EQ (out_info.num_tensors, 1U);
  EXPECT_EQ (out_info.info[0].dimension[0], CHANNEL);
  EXPECT_EQ (out_info.info[0].dimension[1], WIDTH);
  EXPECT_EQ (out_info.info[0].dimension[2], HEIGHT);
  EXPECT_EQ (out_info.info[0].type, _NNS_UINT8);

  sp->close (&prop, &data);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  g_free (model_file);
}

/**
 * @brief Negative case calling getModelInfo before open
 */
TEST (nnstreamerFilterPython3, getModelInfo01_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.py", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);

  GstTensorsInfo in_info, out_info;

  ret = sp->getModelInfo (NULL, NULL, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);
  _SetFilterProp (&prop, "python3", model_files);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Negative case with invalid argument
 */
TEST (nnstreamerFilterPython3, getModelInfo02_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.py", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "python3", model_files);

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
TEST (nnstreamerFilterPython3, invoke00_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  gchar *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.py", NULL);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "python3", model_files);

  ret = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_NE (ret, 0);

  g_free (model_file);
  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Negative test case with invoke before open
 */
TEST (nnstreamerFilterPython3, invoke01)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  gchar *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "passthrough.py", NULL);
  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  output.size = input.size = sizeof (float) * 3 * 280 * 40; // channel * width * height

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  memset (input.data, 0, input.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("python3");
  EXPECT_NE (sp, nullptr);
  _SetFilterProp (&prop, "python3", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, (void *) NULL);

  ret = sp->invoke (NULL, NULL, data, &input, &output);

  EXPECT_EQ (ret, 0);
  EXPECT_EQ (output.size, input.size);

  g_free (model_file);
  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Fixture running filter_output_cases.py in the case given as custom.
 * @note The script takes a uint8 tensor of 8 and gives two uint8 tensors of 4.
 */
class nnstreamerFilterPython3Output : public ::testing::Test
{
  protected:
  const GstTensorFilterFramework *sp;
  GstTensorFilterProperties prop;
  gchar *model_file;
  const gchar *model_files[2];
  void *data;
  guint8 in_data[8];
  guint8 unset[2][4];
  GstTensorMemory input;
  GstTensorMemory output[2];

  /**
   * @brief Prepare the properties and the tensors for a case.
   */
  void SetUp () override
  {
    const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

    model_file = g_build_filename (root_path, "tests", "test_models", "models",
        "filter_output_cases.py", NULL);
    model_files[0] = model_file;
    model_files[1] = NULL;
    data = NULL;

    sp = nnstreamer_filter_find ("python3");
    ASSERT_NE (sp, nullptr);
    _SetFilterProp (&prop, "python3", model_files);

    gst_tensors_info_init (&prop.input_meta);
    prop.input_meta.num_tensors = 1;
    prop.input_meta.info[0].type = _NNS_UINT8;
    gst_tensor_parse_dimension ("8", prop.input_meta.info[0].dimension);

    gst_tensors_info_init (&prop.output_meta);
    prop.output_meta.num_tensors = 2;
    for (guint i = 0; i < 2; i++) {
      prop.output_meta.info[i].type = _NNS_UINT8;
      gst_tensor_parse_dimension ("4", prop.output_meta.info[i].dimension);
    }

    for (guint i = 0; i < 8; i++)
      in_data[i] = (guint8) (i + 10);
    input.data = in_data;
    input.size = sizeof (in_data);

    resetOutput ();
  }

  /**
   * @brief Close the filter and release the properties.
   */
  void TearDown () override
  {
    if (sp)
      sp->close (&prop, &data);
    gst_tensors_info_free (&prop.input_meta);
    gst_tensors_info_free (&prop.output_meta);
    g_free (model_file);
  }

  /**
   * @brief Point the output tensors at buffers the script cannot own.
   */
  void resetOutput ()
  {
    for (guint i = 0; i < 2; i++) {
      output[i].data = unset[i];
      output[i].size = sizeof (unset[i]);
    }
  }

  /**
   * @brief Open the script with the given case.
   */
  int openCase (const gchar *mode)
  {
    prop.custom_properties = mode;
    return sp->open (&prop, &data);
  }

  /**
   * @brief Give an output tensor back to the script, as tensor_filter does.
   */
  void release (void *out)
  {
    GstTensorFilterFrameworkEventData event;

    event.data = out;
    EXPECT_EQ (sp->eventHandler (sp, &prop, data, DESTROY_NOTIFY, &event), 0);
  }

  /**
   * @brief Check that the output tensor holds what is expected and is not the input.
   */
  void expectOutput (guint idx, const guint8 *expected)
  {
    const guint8 *out = (const guint8 *) output[idx].data;

    ASSERT_NE (out, nullptr);
    EXPECT_NE (out, unset[idx]);
    EXPECT_TRUE (out < in_data || out >= in_data + sizeof (in_data));
    EXPECT_EQ (memcmp (out, expected, 4), 0);
  }

  /**
   * @brief Check that a failed invoke left the output tensors untouched.
   */
  void expectNoOutput ()
  {
    EXPECT_EQ (output[0].data, (void *) unset[0]);
    EXPECT_EQ (output[1].data, (void *) unset[1]);
  }

  /**
   * @brief Check that the next invoke of the same script works, i.e., a failure
   *        leaves no error behind. A failing case of the script fails only once.
   */
  void expectRecovered ()
  {
    const guint8 first[4] = { 10, 11, 12, 13 };

    resetOutput ();
    ASSERT_EQ (sp->invoke (sp, &prop, data, &input, output), 0);
    expectOutput (0, first);
    release (output[0].data);
    release (output[1].data);
  }
};

/**
 * @brief Positive case with the outputs the script allocates itself
 */
TEST_F (nnstreamerFilterPython3Output, copy)
{
  const guint8 first[4] = { 10, 11, 12, 13 };
  const guint8 second[4] = { 14, 15, 16, 17 };

  ASSERT_EQ (openCase ("copy"), 0);
  ASSERT_EQ (sp->invoke (sp, &prop, data, &input, output), 0);

  expectOutput (0, first);
  expectOutput (1, second);
  release (output[0].data);
  release (output[1].data);
}

/**
 * @brief Outputs that are views of the input tensor are handed out as a copy
 */
TEST_F (nnstreamerFilterPython3Output, aliasInput)
{
  const guint8 first[4] = { 10, 11, 12, 13 };
  const guint8 second[4] = { 14, 15, 16, 17 };

  ASSERT_EQ (openCase ("alias"), 0);
  ASSERT_EQ (sp->invoke (sp, &prop, data, &input, output), 0);

  expectOutput (0, first);
  expectOutput (1, second);
  release (output[0].data);
  release (output[1].data);
}

/**
 * @brief Strided outputs (positive and negative steps) are made contiguous
 */
TEST_F (nnstreamerFilterPython3Output, nonContiguous)
{
  const guint8 first[4] = { 0, 2, 4, 6 };
  const guint8 second[4] = { 7, 5, 3, 1 };

  ASSERT_EQ (openCase ("strided"), 0);
  ASSERT_EQ (sp->invoke (sp, &prop, data, &input, output), 0);

  expectOutput (0, first);
  expectOutput (1, second);
  release (output[0].data);
  release (output[1].data);
}

/**
 * @brief One array returned for both outputs gives two separate buffers
 */
TEST_F (nnstreamerFilterPython3Output, sameArrayTwice)
{
  const guint8 expected[4] = { 0, 1, 2, 3 };

  ASSERT_EQ (openCase ("same"), 0);
  ASSERT_EQ (sp->invoke (sp, &prop, data, &input, output), 0);

  expectOutput (0, expected);
  expectOutput (1, expected);
  EXPECT_NE (output[0].data, output[1].data);
  release (output[0].data);
  release (output[1].data);
}

/**
 * @brief An array the script keeps is not handed out again while in use
 */
TEST_F (nnstreamerFilterPython3Output, keptArray)
{
  const guint8 first[4] = { 0, 1, 2, 3 };
  const guint8 second[4] = { 14, 15, 16, 17 };
  void *prev[2];

  ASSERT_EQ (openCase ("kept"), 0);
  ASSERT_EQ (sp->invoke (sp, &prop, data, &input, output), 0);
  expectOutput (0, first);
  prev[0] = output[0].data;
  prev[1] = output[1].data;

  resetOutput ();
  ASSERT_EQ (sp->invoke (sp, &prop, data, &input, output), 0);
  expectOutput (0, first);
  expectOutput (1, second);
  EXPECT_NE (output[0].data, prev[0]);

  release (prev[0]);
  release (prev[1]);
  release (output[0].data);
  release (output[1].data);
}

/**
 * @brief Negative case with an output that is not a numpy array
 */
TEST_F (nnstreamerFilterPython3Output, notArray_n)
{
  ASSERT_EQ (openCase ("not_array"), 0);
  EXPECT_NE (sp->invoke (sp, &prop, data, &input, output), 0);
  expectNoOutput ();
  expectRecovered ();
}

/**
 * @brief Negative case with outputs given as a tuple instead of a list
 */
TEST_F (nnstreamerFilterPython3Output, notList_n)
{
  ASSERT_EQ (openCase ("not_list"), 0);
  EXPECT_NE (sp->invoke (sp, &prop, data, &input, output), 0);
  expectNoOutput ();
  expectRecovered ();
}

/**
 * @brief Negative case with fewer outputs than the model gives
 */
TEST_F (nnstreamerFilterPython3Output, wrongCount_n)
{
  ASSERT_EQ (openCase ("count"), 0);
  EXPECT_NE (sp->invoke (sp, &prop, data, &input, output), 0);
  expectNoOutput ();
  expectRecovered ();
}

/**
 * @brief Negative case with a valid first output and an invalid second one
 */
TEST_F (nnstreamerFilterPython3Output, secondInvalid_n)
{
  ASSERT_EQ (openCase ("second_bad"), 0);
  EXPECT_NE (sp->invoke (sp, &prop, data, &input, output), 0);
  expectNoOutput ();
  expectRecovered ();
}

/**
 * @brief Negative case with a script raising an exception in invoke
 */
TEST_F (nnstreamerFilterPython3Output, raise_n)
{
  ASSERT_EQ (openCase ("raise"), 0);
  EXPECT_NE (sp->invoke (sp, &prop, data, &input, output), 0);
  expectNoOutput ();
  expectRecovered ();
}

/**
 * @brief Callback for tensor sink signal, appends the data of every buffer.
 */
static void
_python3_new_data_cb (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  GByteArray *received = (GByteArray *) user_data;
  GstMapInfo map;

  UNUSED (element);
  for (guint i = 0; i < gst_buffer_n_memory (buffer); i++) {
    GstMemory *mem = gst_buffer_peek_memory (buffer, i);

    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
    g_byte_array_append (received, map.data, map.size);
    gst_memory_unmap (mem, &map);
  }
}

/**
 * @brief Push one buffer to tensor_filter running filter_output_cases.py.
 * @return the type of the message ending the stream (EOS or ERROR).
 */
static GstMessageType
_python3_run_pipeline (const gchar *mode, GByteArray *received)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstMessageType result = GST_MESSAGE_UNKNOWN;
  GstElement *pipeline, *src, *sink;
  GstMessage *msg;
  GstBus *bus;
  gchar *model_file, *desc;
  guint8 *in_data = (guint8 *) g_malloc (8);

  for (guint i = 0; i < 8; i++)
    in_data[i] = (guint8) (i + 10);

  model_file = g_build_filename (root_path, "tests", "test_models", "models",
      "filter_output_cases.py", NULL);
  desc = g_strdup_printf (
      "appsrc name=srcx caps=\"other/tensors,num_tensors=1,dimensions=(string)8,types=(string)uint8,format=static,framerate=0/1\" ! "
      "tensor_filter framework=python3 model=\"%s\" custom=%s input=8 inputtype=uint8 "
      "output=4,4 outputtype=uint8,uint8 ! tensor_sink name=sinkx async=false",
      model_file, mode);
  pipeline = gst_parse_launch (desc, NULL);
  g_free (desc);
  g_free (model_file);
  if (!pipeline) {
    g_free (in_data);
    return GST_MESSAGE_ANY;
  }

  src = gst_bin_get_by_name (GST_BIN (pipeline), "srcx");
  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  g_signal_connect (sink, "new-data", (GCallback) _python3_new_data_cb, received);

  if (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT) == 0) {
    gst_app_src_push_buffer (GST_APP_SRC (src), gst_buffer_new_wrapped (in_data, 8));
    gst_app_src_end_of_stream (GST_APP_SRC (src));

    bus = gst_element_get_bus (pipeline);
    msg = gst_bus_timed_pop_filtered (bus, 5 * GST_SECOND,
        (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
    if (msg) {
      result = GST_MESSAGE_TYPE (msg);
      gst_message_unref (msg);
    }
    gst_object_unref (bus);
  } else {
    g_free (in_data);
  }

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (src);
  gst_object_unref (sink);
  gst_object_unref (pipeline);
  return result;
}

/**
 * @brief tensor_filter pushes the copied outputs of a script returning input views
 */
TEST (nnstreamerFilterPython3, pipelineAliasInput)
{
  const guint8 expected[8] = { 10, 11, 12, 13, 14, 15, 16, 17 };
  GByteArray *received = g_byte_array_new ();

  EXPECT_EQ (_python3_run_pipeline ("alias", received), GST_MESSAGE_EOS);
  ASSERT_EQ (received->len, 8U);
  EXPECT_EQ (memcmp (received->data, expected, 8), 0);
  g_byte_array_unref (received);
}

/**
 * @brief tensor_filter stops the stream instead of pushing an invalid output
 */
TEST (nnstreamerFilterPython3, pipelineSecondInvalid_n)
{
  GByteArray *received = g_byte_array_new ();

  EXPECT_EQ (_python3_run_pipeline ("second_bad", received), GST_MESSAGE_ERROR);
  EXPECT_EQ (received->len, 0U);
  g_byte_array_unref (received);
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
