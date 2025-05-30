/**
 * @file	unittest_filter_custom.cc
 * @date	23 May 2025
 * @brief	Unit test for tensor filter custom-easy plugin
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyunil Park <hyunil46.park@samsung.com>
 * @bug		No known bugs.
 */

#include <gtest/gtest.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <unittest_util.h>


/* Static variable to track the number of async callback invocations. */
static int callback_count = 0;

/**
 * @brief Callback function for appsink element to pull sample and print the output text from llamacpp plugin
 */
static GstFlowReturn
new_sample_cb (GstElement *sink, gpointer user_data)
{
  GstSample *sample;
  GstBuffer *buffer;
  GstMapInfo map;

  callback_count++;
  g_signal_emit_by_name (sink, "pull-sample", &sample);
  if (sample) {
    buffer = gst_sample_get_buffer (sample);
    if (gst_buffer_map (buffer, &map, GST_MAP_READ)) {
      g_print ("%.*s", (int) map.size, (char *) map.data);
      /* need to free */
      g_free(map.data);
      gst_buffer_unmap (buffer, &map);
    }
    gst_sample_unref (sample);
    return GST_FLOW_OK;
  }
  return GST_FLOW_ERROR;
}

/**
 * @brief Test fixture class for tensor_filter llama-cpp plugin unit tests.
 */
class NNStreamerFilterLlamaCppTest : public ::testing::Test
{
  protected:
  GstElement *pipeline, *appsrc, *appsink, *tensor_filter;
  GstBuffer *buffer;
  GstMapInfo map;
  gboolean loaded;
  gboolean skip_test;
  gsize data_size;
  
  public:
  /**
   * @brief Constructor a new NNStreamerFilterLlamaCppTest object
   */
  NNStreamerFilterLlamaCppTest ()
  {
    loaded = FALSE;
    skip_test = FALSE;
  }

  /**
   * @brief Set up method for unit test execution is started.
   */
  void SetUp () override
  {
    g_autofree gchar *model = nullptr;
    g_autofree gchar *pipeline_str = nullptr;
    const gchar *data = "Hello my name is";  // input text
    const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

    /* supposed to run test in build directory */
    if (root_path == NULL)
      root_path = "..";

    model = g_build_filename (root_path, "tests", "test_models", "models",
        "llama-2-7b-chat.Q2_K.gguf", NULL);

    if (!g_file_test (model, G_FILE_TEST_EXISTS)) {
      g_critical ("Skipping test due to missing model file. "
                  "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF");
      skip_test = TRUE;
      return;
    }
    
    pipeline_str = g_strdup_printf (
      "appsrc name=appsrc ! application/octet-stream ! tensor_converter ! other/tensors,format=flexible ! "
      "tensor_filter name=tensor_filter framework=llamacpp model=%s invoke-dynamic=TRUE custom=num_predict:32 ! "
      "other/tensors,format=flexible ! tensor_decoder mode=octet_stream ! application/octet-stream ! appsink name=appsink",
      model);
 
    pipeline = gst_parse_launch (pipeline_str, NULL);
    EXPECT_NE (pipeline, nullptr);

    tensor_filter = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_filter");
    EXPECT_NE (tensor_filter, nullptr);

    appsrc = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc");
    EXPECT_NE (appsrc, nullptr);
    appsink = gst_bin_get_by_name (GST_BIN (pipeline), "appsink");
    EXPECT_NE (appsink, nullptr);

    g_object_set (G_OBJECT (appsink), "emit-signals", TRUE, NULL);
    g_signal_connect (appsink, "new-sample", G_CALLBACK (new_sample_cb), NULL);

    data_size = strlen (data);
    buffer = gst_buffer_new_allocate (NULL, data_size, NULL);

    gst_buffer_map (buffer, &map, GST_MAP_WRITE);
    memcpy (map.data, data, data_size);
    gst_buffer_unmap (buffer, &map);

    EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc), buffer), GST_FLOW_OK);
    loaded = TRUE;
  }

  /**
   * @brief Tear down method after unit test execution is completed.
   */
  void TearDown () override
  {
    gst_object_unref (appsrc);
    gst_object_unref (appsink);
    gst_object_unref (tensor_filter);
    gst_object_unref (pipeline);
  }
};

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invoke-async option enabled
 */
TEST_F (NNStreamerFilterLlamaCppTest, InvokAsync_p)
{
  gboolean invoke_async = TRUE;

  if (skip_test) {
    GTEST_SKIP () << "Skipping invokeAsync_p test due to missing model file"
                     "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  callback_count = 0;
  g_object_set (G_OBJECT (tensor_filter), "invoke-async", invoke_async, NULL);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (5000000); // wait for 5 seconds

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (callback_count, 0);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invoke-async option disabled
 */
TEST_F (NNStreamerFilterLlamaCppTest, InvokAsyncDisabled_p)
{
  gboolean invoke_async = FALSE;

  if (skip_test) {
    GTEST_SKIP () << "Skipping InvokAsyncDisabled_p test due to missing model file"
                     "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  callback_count = 0;
  g_object_set (G_OBJECT (tensor_filter), "invoke-async", invoke_async, NULL);
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (callback_count, 0);
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
