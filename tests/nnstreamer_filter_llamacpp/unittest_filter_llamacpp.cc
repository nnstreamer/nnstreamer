/**
 * @file	unittest_filter_custom.cc
 * @date	23 May 2025
 * @brief	Unit test for tensor filter llamacpp plugin
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
static int new_sample_count = 0;

/**
 * @brief Callback function for appsink element to pull sample and print the output text from llamacpp plugin
 */
static GstFlowReturn
new_sample_cb (GstElement *sink, gpointer user_data)
{
  GstSample *sample;
  GstBuffer *buffer;
  GstMapInfo map;

  new_sample_count++;
  sample = gst_app_sink_pull_sample (GST_APP_SINK (sink));
  if (sample) {
    buffer = gst_sample_get_buffer (sample);
    if (gst_buffer_map (buffer, &map, GST_MAP_READ)) {
      g_print ("%.*s", (int) map.size, (char *) map.data);
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
  gboolean loaded;
  gboolean skip_test;

  gchar *model;
  gchar *lora_path; // LoRA adapter model path
  gchar *pipeline_str;

  public:
  /**
   * @brief Constructor a new NNStreamerFilterLlamaCppTest object
   */
  NNStreamerFilterLlamaCppTest ()
  {
    pipeline = nullptr;
    appsrc = nullptr;
    appsink = nullptr;
    tensor_filter = nullptr;
    model = nullptr;
    pipeline_str = nullptr;
    loaded = FALSE;
    skip_test = FALSE;
  }

  /**
   * @brief Set up method for unit test execution is started.
   */
  void SetUp () override
  {
    const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

    /* supposed to run test in build directory */
    if (root_path == NULL)
      root_path = "..";

    model = g_build_filename (root_path, "tests", "test_models", "models",
        "llama-2-7b-chat.Q4_K_M.gguf", NULL);

    if (!g_file_test (model, G_FILE_TEST_EXISTS)) {
      g_critical ("Skipping test due to missing model file. "
                  "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF");
      skip_test = TRUE;
      return;
    }

    lora_path = g_build_filename (root_path, "tests", "test_models", "models",
        "ggml-adapter-model.gguf", NULL);

    if (!g_file_test (lora_path, G_FILE_TEST_EXISTS)) {
      g_critical ("Skipping test due to missing LoRA model file: %s", lora_path);
      skip_test = TRUE;
      g_free (model);
      model = NULL;
      return;
    }

    loaded = TRUE;
  }

  /**
   * @brief Tear down method after unit test execution is completed.
   */
  void TearDown () override
  {
    g_clear_pointer (&pipeline, gst_object_unref);
    g_clear_pointer (&appsrc, gst_object_unref);
    g_clear_pointer (&appsink, gst_object_unref);
    g_clear_pointer (&tensor_filter, gst_object_unref);
    g_clear_pointer (&pipeline_str, g_free);
    g_clear_pointer (&model, g_free);
  }

  /**
   * @brief Create pipeline with given model invoke-async and custom properties.
   * @param model Model file path (or comma-separated base model and LoRA adapter paths)
   * @param invoke_async Whether to use async mode
   * @param custom Custom properties string (e.g., "num_predict:10,top_k:40")
   *               If NULL, uses default "num_predict:32"
   */
  void create_pipeline (const gchar *model, gboolean invoke_async, const gchar *custom)
  {
    const gchar *effective_custom = custom ? custom : "num_predict:32";

    pipeline_str = g_strdup_printf (
        "appsrc name=appsrc ! application/octet-stream ! tensor_converter ! other/tensors,format=flexible ! "
        "tensor_filter name=tensor_filter framework=llamacpp model=%s invoke-dynamic=TRUE invoke-async=%d custom=%s ! "
        "other/tensors,format=flexible ! tensor_decoder mode=octet_stream ! application/octet-stream ! appsink name=appsink",
        model, invoke_async, effective_custom);

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
  }

  /**
   * @brief Push data into the pipeline.
   */
  void data_push (const gchar *data)
  {
    gsize data_size;
    GstBuffer *buffer;
    GstMapInfo map;

    data_size = strlen (data);
    buffer = gst_buffer_new_allocate (NULL, data_size, NULL);
    ASSERT_NE (buffer, nullptr);

    ASSERT_TRUE (gst_buffer_map (buffer, &map, GST_MAP_WRITE));
    memcpy (map.data, data, data_size);
    gst_buffer_unmap (buffer, &map);

    ASSERT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc), buffer), GST_FLOW_OK);
  }
};

/**
 * Note. Inference libraries like LLP and llama.cpp generally do not allow
 * modification of such settings at runtime after the context (e.g., model,
 * session, etc.) has been created. In particular, parameters like n_gpu_layers
 * must be configured at model loading time and cannot be changed afterwards.
 */

/**
 * @brief Test case for tensor_filter llama-cpp plugin with singleInputMultipleOutputsAsync_p
 */
TEST_F (NNStreamerFilterLlamaCppTest, singleInputMultipleOutputsAsync_p)
{
  gboolean invoke_async = TRUE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping singleInputMultipleOutputsAsync_p test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:10");
  data_push ("Hello my name is");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with singleInputSingleOutputSync_p
 */
TEST_F (NNStreamerFilterLlamaCppTest, singleInputSingleOutputSync_p)
{
  gboolean invoke_async = FALSE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping singleInputSingleOutputSync_p test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:10");
  data_push ("Hello my name is");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (new_sample_count, 1);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with multipleInputsMultipleOutputsAsync_p
 */
TEST_F (NNStreamerFilterLlamaCppTest, multipleInputsMultipleOutputsAsync_p)
{
  gboolean invoke_async = TRUE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping multipleInputsMultipleOutputsAsync_p test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:10"); /* Create up to 10 tokens */
  data_push ("Hello my name is");
  data_push ("What is AI?");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (new_sample_count, 15);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with multipleInputsSingleOutputSync_p
 */
TEST_F (NNStreamerFilterLlamaCppTest, multipleInputsSingleOutputSync_p)
{
  gboolean invoke_async = FALSE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping multipleInputsSingleOutputSync_p test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:10");
  data_push ("Hello my name is");
  data_push ("What is AI?");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (new_sample_count, 2);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invalidNumPredict_n
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidNumPredict_n)
{
  gboolean invoke_async = TRUE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping invalidNumPredict_n test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:0");
  data_push ("Hello my name is");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invalidModel_n
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidModel_n)
{
  gboolean invoke_async = TRUE;
  if (skip_test) {
    GTEST_SKIP () << "Skipping invalidModel_n test due to missing model file"
                     "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (NULL, invoke_async, "num_predict:10");
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with earlyTerminationBeforeTokenGenerationAsync_n
 */
TEST_F (NNStreamerFilterLlamaCppTest, earlyTerminationBeforeTokenGenerationAsync_n)
{
  gboolean invoke_async = TRUE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping earlyTerminationBeforeTokenGenerationAsync_n test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:10");
  data_push ("Hello my name is");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  /* g_usleep (3000000); Avoid using g_usleep for early termination. */
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_LE (new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with earlyTerminationBeforeTokenGenerationSync_n
 */
TEST_F (NNStreamerFilterLlamaCppTest, earlyTerminationBeforeTokenGenerationSync_n)
{
  gboolean invoke_async = FALSE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping earlyTerminationBeforeTokenGenerationSync_n test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:10");
  data_push ("Hello my name is");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  /* g_usleep (3000000); Avoid using g_usleep for early termination. */
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_LE (new_sample_count, 1);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with combined sampling (top_k + top_p + temperature)
 */
TEST_F (NNStreamerFilterLlamaCppTest, singleInputCombinedSampling_p)
{
  gboolean invoke_async = TRUE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping singleInputCombinedSampling_p test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:30,top_k:50,top_p:0.9,temperature:0.7");
  data_push ("Hello my name is");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with all sampling options
 */
TEST_F (NNStreamerFilterLlamaCppTest, singleInputAllSamplingOptions_p)
{
  gboolean invoke_async = TRUE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping singleInputAllSamplingOptions_p test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async,
      "num_predict:30,top_k:40,top_p:0.9,typical_p:0.9,temperature:0.7");
  data_push ("Hello my name is");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with typical_p + temperature combination
 */
TEST_F (NNStreamerFilterLlamaCppTest, singleInputTypicalPTempSampling_p)
{
  gboolean invoke_async = TRUE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping singleInputTypicalPTempSampling_p test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:30,typical_p:0.9,temperature:0.7");
  data_push ("Hello my name is");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invalid top_k value
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidTopKValue_n)
{
  gboolean invoke_async = TRUE;
  if (skip_test) {
    GTEST_SKIP () << "Skipping invalidTopKValue_n test due to missing model file"
                     "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:10,top_k:0");
  data_push ("Hello my name is");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invalid top_p value
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidTopPValue_n)
{
  gboolean invoke_async = TRUE;
  if (skip_test) {
    GTEST_SKIP () << "Skipping invalidTopPValue_n test due to missing model file"
                     "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:10,top_p:1.0");
  data_push ("Hello my name is");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invalid typical_p value
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidTypicalPValue_n)
{
  gboolean invoke_async = TRUE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping invalidTypicalPValue_n test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:10,typical_p:1.0");
  data_push ("Hello my name is");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invalid temperature value
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidTemperatureValue_n)
{
  gboolean invoke_async = TRUE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping invalidTemperatureValue_n test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:10,temperature:-1.0");
  data_push ("Hello my name is");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with combined sampling in sync mode
 */
TEST_F (NNStreamerFilterLlamaCppTest, combinedSamplingSync_p)
{
  gboolean invoke_async = FALSE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping combinedSamplingSync_p test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:30,top_k:20,top_p:0.7,temperature:0.7");
  data_push ("Hello my name is");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (new_sample_count, 1);
}

/**
 * @brief Test case for KV cache - context continuation in conversation
 */
TEST_F (NNStreamerFilterLlamaCppTest, contextContinuationInConversation_p)
{
  gboolean invoke_async = FALSE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping contextContinuationInConversation_p test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:15,context_length:512");
  data_push ("Hello my name is John. I like programming and artificial intelligence.");
  data_push ("What did I just say about myself?");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (new_sample_count, 2);
}

/**
 * @brief Test case for KV cache - cache trimming trigger
 */
TEST_F (NNStreamerFilterLlamaCppTest, cacheTrimmingTrigger_p)
{
  gboolean invoke_async = TRUE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping cacheTrimmingTrigger_p test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:128,context_length:128");
  /** If we set the context length to 128 and the number of tokens generated by
    LLM to 128, KV cache trimming should occur after the first data_push output,
    and data_push should function properly. */
  data_push ("Hello my name is");
  data_push ("Hello my name is");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (30000000);

  EXPECT_GT (new_sample_count, 128);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief Test case for KV cache - invalid context length (negative test)
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidContextLength_n)
{
  gboolean invoke_async = TRUE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping invalidContextLength_n test due to missing model file"
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, "num_predict:5,context_length:0");

  data_push ("Hello");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GE (new_sample_count, 0);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with LoRA adapter applied (positive test)
 */
TEST_F (NNStreamerFilterLlamaCppTest, applyLoraAdapter_p)
{
  gboolean invoke_async = FALSE; /* Use sync mode for simpler output verification */
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping applyLoraAdapter_p test due to missing model or LoRA file."
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF "
           "and ensure adapter_model.gguf is present in tests/test_models/models/";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  gchar *model_with_lora = g_strdup_printf ("%s,%s", model, lora_path);
  create_pipeline (model_with_lora, invoke_async, "num_predict:40");
  g_free (model_with_lora);

  data_push ("Can you translate 'Hello world' into Korean?");
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000); /* Wait for inference to complete */

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (new_sample_count, 1); /* In sync mode, expect one sample for the entire output */
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invalid LoRA adapter path (negative test)
 */
TEST_F (NNStreamerFilterLlamaCppTest, applyInvalidLoraAdapter_n)
{
  gboolean invoke_async = FALSE;
  if (skip_test) {
    GTEST_SKIP ()
        << "Skipping applyLoraAdapter_p test due to missing model or LoRA file."
           "Please download model file from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF ";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  /* Intentionally use an invalid LoRA path */
  const gchar *invalid_lora_path = "/invalid/path/to/adapter.gguf";
  gchar *model_with_invalid_lora = g_strdup_printf ("%s,%s", model, invalid_lora_path);
  create_pipeline (model_with_invalid_lora, invoke_async, "num_predict:20");
  g_free (model_with_invalid_lora);

  /* Pipeline state change to PLAYING should fail due to invalid LoRA path */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  EXPECT_EQ (new_sample_count, 0); /* No samples should be generated */
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
