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

#define LLAMACPP_TEST_MODEL "TinyStories-656K-Q2_K.gguf"
#define LLAMACPP_TEST_MODEL_URL \
  "https://huggingface.co/tensorblock/TinyStories-656K-GGUF"
#define LLAMACPP_LORA_TEST_MODEL "llama-2-7b-chat.Q4_K_M.gguf"
#define LLAMACPP_LORA_TEST_MODEL_URL \
  "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF"
#define LLAMACPP_LORA_TEST_ADAPTER "ggml-adapter-model.gguf"

/**
 * @brief Structure to hold LoRA-related file paths
 */
typedef struct {
  gchar *adapter_path;
  gchar *model_path;
} LoRAPaths;

/**
 * @brief Macro to skip testcase if model file is not ready.
 */
#define skip_llamacpp_tc(tc_name)                                                                \
  do {                                                                                           \
    if (skip_test) {                                                                             \
      g_autofree gchar *msg = g_strdup_printf (                                                  \
          "Skipping '%s' due to missing model file '%s'. Please download model file from '%s'.", \
          tc_name, LLAMACPP_TEST_MODEL, LLAMACPP_TEST_MODEL_URL);                                \
      GTEST_SKIP () << msg;                                                                      \
    }                                                                                            \
  } while (0)

/**
 * @brief Macro to skip testcase if lora adapter is not ready.
 * See https://github.com/nnstreamer/nnstreamer/pull/4773 to create adapter.
 */
#define skip_llamacpp_lora_tc(tc_name)                                                           \
  do {                                                                                           \
    if (skip_lora_test) {                                                                        \
      g_autofree gchar *msg = g_strdup_printf (                                                  \
          "Skipping '%s' due to missing model file '%s'. Please download model file from '%s'.", \
          tc_name, LLAMACPP_LORA_TEST_MODEL, LLAMACPP_LORA_TEST_MODEL_URL);                      \
      GTEST_SKIP () << msg;                                                                      \
    }                                                                                            \
  } while (0)

/**
 * @brief Callback function for appsink element to pull sample and print the output text from llamacpp plugin
 */
static GstFlowReturn
new_sample_cb (GstElement *sink, gpointer user_data)
{
  GstSample *sample;
  GstBuffer *buffer;
  GstMapInfo map;

  if (user_data) {
    guint *count = (guint *) user_data;
    (*count)++;
  }

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
  gchar *model;
  LoRAPaths *lora_paths;
  guint *new_sample_count;
  gboolean skip_test;
  gboolean skip_lora_test;

  public:
  /**
   * @brief Constructor a new NNStreamerFilterLlamaCppTest object
   */
  NNStreamerFilterLlamaCppTest ()
  {
    const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

    /* supposed to run test in build directory */
    if (root_path == NULL)
      root_path = "..";

    pipeline = nullptr;
    appsrc = nullptr;
    appsink = nullptr;
    tensor_filter = nullptr;
    new_sample_count = nullptr;

    model = g_build_filename (
        root_path, "tests", "test_models", "models", LLAMACPP_TEST_MODEL, NULL);
    skip_test = !g_file_test (model, G_FILE_TEST_EXISTS);

    /* paths to run lora-related test */
    lora_paths = g_new0 (LoRAPaths, 1);
    lora_paths->adapter_path = g_build_filename (root_path, "tests",
        "test_models", "models", LLAMACPP_LORA_TEST_ADAPTER, NULL);
    lora_paths->model_path = g_build_filename (root_path, "tests",
        "test_models", "models", LLAMACPP_LORA_TEST_MODEL, NULL);
    skip_lora_test = !g_file_test (lora_paths->adapter_path, G_FILE_TEST_EXISTS)
                     || !g_file_test (lora_paths->model_path, G_FILE_TEST_EXISTS);
  }

  /**
   * @brief Destructor of NNStreamerFilterLlamaCppTest class
   */
  ~NNStreamerFilterLlamaCppTest ()
  {
    g_free (model);
    g_free (lora_paths->adapter_path);
    g_free (lora_paths->model_path);
    g_free (lora_paths);
  }

  /**
   * @brief Set up method for unit test execution is started.
   */
  void SetUp () override
  {
    /* Track the number of async callback invocations. */
    new_sample_count = g_new0 (guint, 1);
  }

  /**
   * @brief Tear down method after unit test execution is completed.
   */
  void TearDown () override
  {
    clear_pipeline ();
    g_clear_pointer (&new_sample_count, g_free);
  }

  /**
   * @brief Create pipeline with given model invoke-async and custom properties.
   * @param model Model file path (or comma-separated base model and LoRA adapter paths)
   * @param invoke_async Whether to use async mode
   * @param custom Custom properties string (e.g., "num_predict:10,top_k:40")
   *               If NULL, uses default "num_predict:32"
   * @return TRUE if the pipeline is successfully created.
   */
  gboolean create_pipeline (const gchar *model_path, gboolean invoke_async, const gchar *custom)
  {
    const gchar *effective_custom = custom ? custom : "num_predict:32";
    g_autofree gchar *pipeline_str = g_strdup_printf (
        "appsrc name=appsrc ! application/octet-stream ! tensor_converter ! other/tensors,format=flexible ! "
        "tensor_filter name=tensor_filter framework=llamacpp model=%s invoke-dynamic=TRUE invoke-async=%d custom=%s ! "
        "other/tensors,format=flexible ! tensor_decoder mode=octet_stream ! application/octet-stream ! appsink name=appsink",
        model_path, invoke_async, effective_custom);

    pipeline = gst_parse_launch (pipeline_str, NULL);
    g_return_val_if_fail (pipeline != nullptr, FALSE);

    tensor_filter = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_filter");
    g_return_val_if_fail (tensor_filter != nullptr, FALSE);
    appsrc = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc");
    g_return_val_if_fail (appsrc != nullptr, FALSE);
    appsink = gst_bin_get_by_name (GST_BIN (pipeline), "appsink");
    g_return_val_if_fail (appsink != nullptr, FALSE);

    g_object_set (G_OBJECT (appsink), "emit-signals", TRUE, NULL);
    g_signal_connect (appsink, "new-sample", G_CALLBACK (new_sample_cb), new_sample_count);
    return TRUE;
  }

  /**
   * @brief Release pipeline resources.
   */
  void clear_pipeline ()
  {
    g_clear_pointer (&appsrc, gst_object_unref);
    g_clear_pointer (&appsink, gst_object_unref);
    g_clear_pointer (&tensor_filter, gst_object_unref);
    g_clear_pointer (&pipeline, gst_object_unref);
  }

  /**
   * @brief Push data into the pipeline.
   */
  gboolean data_push (const gchar *data)
  {
    GstBuffer *buffer;
    GstFlowReturn ret;

    buffer = gst_buffer_new_wrapped (g_strdup (data), strlen (data));
    ret = gst_app_src_push_buffer (GST_APP_SRC (appsrc), buffer);
    return (ret == GST_FLOW_OK);
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
  skip_llamacpp_tc ("singleInputMultipleOutputsAsync_p");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:10"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (*new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with singleInputSingleOutputSync_p
 */
TEST_F (NNStreamerFilterLlamaCppTest, singleInputSingleOutputSync_p)
{
  skip_llamacpp_tc ("singleInputSingleOutputSync_p");

  ASSERT_TRUE (create_pipeline (model, FALSE, "num_predict:10"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 1);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with multipleInputsMultipleOutputsAsync_p
 */
TEST_F (NNStreamerFilterLlamaCppTest, multipleInputsMultipleOutputsAsync_p)
{
  skip_llamacpp_tc ("multipleInputsMultipleOutputsAsync_p");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:10")); /* Create up to 10 tokens */
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_TRUE (data_push ("What is AI?"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1500000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (*new_sample_count, 15);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with multipleInputsSingleOutputSync_p
 */
TEST_F (NNStreamerFilterLlamaCppTest, multipleInputsSingleOutputSync_p)
{
  skip_llamacpp_tc ("multipleInputsSingleOutputSync_p");

  ASSERT_TRUE (create_pipeline (model, FALSE, "num_predict:10"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_TRUE (data_push ("What is AI?"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 2);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invalidNumPredict_n
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidNumPredict_n)
{
  skip_llamacpp_tc ("invalidNumPredict_n");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:0"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invalidModel_n
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidModel_n)
{
  skip_llamacpp_tc ("invalidModel_n");

  ASSERT_TRUE (create_pipeline (NULL, TRUE, "num_predict:10"));
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with earlyTerminationBeforeTokenGenerationAsync_n
 */
TEST_F (NNStreamerFilterLlamaCppTest, earlyTerminationBeforeTokenGenerationAsync_n)
{
  skip_llamacpp_tc ("earlyTerminationBeforeTokenGenerationAsync_n");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:10"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_LE (*new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with earlyTerminationBeforeTokenGenerationSync_n
 */
TEST_F (NNStreamerFilterLlamaCppTest, earlyTerminationBeforeTokenGenerationSync_n)
{
  skip_llamacpp_tc ("earlyTerminationBeforeTokenGenerationSync_n");

  ASSERT_TRUE (create_pipeline (model, FALSE, "num_predict:10"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_LE (*new_sample_count, 1);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with combined sampling (top_k + top_p + temperature)
 */
TEST_F (NNStreamerFilterLlamaCppTest, singleInputCombinedSampling_p)
{
  skip_llamacpp_tc ("singleInputCombinedSampling_p");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:30,top_k:50,top_p:0.9,temperature:0.7"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (*new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with all sampling options
 */
TEST_F (NNStreamerFilterLlamaCppTest, singleInputAllSamplingOptions_p)
{
  skip_llamacpp_tc ("singleInputAllSamplingOptions_p");

  ASSERT_TRUE (create_pipeline (model, TRUE,
      "num_predict:30,top_k:40,top_p:0.9,typical_p:0.9,temperature:0.7"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (*new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with typical_p + temperature combination
 */
TEST_F (NNStreamerFilterLlamaCppTest, singleInputTypicalPTempSampling_p)
{
  skip_llamacpp_tc ("singleInputTypicalPTempSampling_p");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:30,typical_p:0.9,temperature:0.7"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (*new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invalid top_k value
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidTopKValue_n)
{
  skip_llamacpp_tc ("invalidTopKValue_n");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:10,top_k:0"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (*new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invalid top_p value
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidTopPValue_n)
{
  skip_llamacpp_tc ("invalidTopPValue_n");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:10,top_p:1.0"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (*new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invalid typical_p value
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidTypicalPValue_n)
{
  skip_llamacpp_tc ("invalidTypicalPValue_n");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:10,typical_p:1.0"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (*new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invalid temperature value
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidTemperatureValue_n)
{
  skip_llamacpp_tc ("invalidTemperatureValue_n");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:10,temperature:-1.0"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (*new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with combined sampling in sync mode
 */
TEST_F (NNStreamerFilterLlamaCppTest, combinedSamplingSync_p)
{
  skip_llamacpp_tc ("combinedSamplingSync_p");

  ASSERT_TRUE (create_pipeline (model, FALSE, "num_predict:30,top_k:20,top_p:0.7,temperature:0.7"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 1);
}

/**
 * @brief Test case for KV cache - context continuation in conversation
 */
TEST_F (NNStreamerFilterLlamaCppTest, contextContinuationInConversation_p)
{
  skip_llamacpp_tc ("contextContinuationInConversation_p");

  ASSERT_TRUE (create_pipeline (model, FALSE, "num_predict:15,context_length:512"));
  EXPECT_TRUE (data_push ("Hello my name is John. I like programming and AI."));
  EXPECT_TRUE (data_push ("What did I just say about myself?"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 2);
}

/**
 * @brief Test case for KV cache - cache trimming trigger
 */
TEST_F (NNStreamerFilterLlamaCppTest, cacheTrimmingTrigger_p)
{
  skip_llamacpp_tc ("cacheTrimmingTrigger_p");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:10,context_length:20"));
  /**
   * If we set the context length to 20 and the number of tokens generated by LLM to 20,
   * KV cache trimming should occur after the first data_push output, and data_push should function properly.
   */
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_TRUE (data_push ("What is AI?"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GT (*new_sample_count, 10);
}

/**
 * @brief Test case for invalid property - invalid context length (negative test)
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidContextLength_n)
{
  skip_llamacpp_tc ("invalidContextLength_n");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:5,context_length:-1"));
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief Test case for invalid property - invalid batch size (negative test)
 */
TEST_F (NNStreamerFilterLlamaCppTest, invalidBatchSize_n)
{
  skip_llamacpp_tc ("invalidBatchSize_n");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:5,batch_size:0"));
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief Test case for context save and load functionality
 */
TEST_F (NNStreamerFilterLlamaCppTest, contextSaveLoad_p)
{
  skip_llamacpp_tc ("contextSaveLoad_p");

  const gchar *context_file = "./context.bin";

  /* First pipeline: save context */
  g_autofree gchar *custom_str1 = g_strdup_printf (
      "num_predict:15,context_length:512,save_ctx:%s", context_file);
  ASSERT_TRUE (create_pipeline (model, FALSE, custom_str1));

  EXPECT_TRUE (data_push ("Hello my name is John. I like programming and AI."));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 1);

  /* Verify context file was created */
  EXPECT_TRUE (g_file_test (context_file, G_FILE_TEST_EXISTS));

  /* Second pipeline: load context */
  clear_pipeline ();
  *new_sample_count = 0;

  g_autofree gchar *custom_str2 = g_strdup_printf (
      "num_predict:20,context_length:512,load_ctx:%s", context_file);
  ASSERT_TRUE (create_pipeline (model, FALSE, custom_str2));

  EXPECT_TRUE (data_push ("What did I just say about myself?"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 1);

  /* Clean up test file */
  g_remove (context_file);
}

/**
 * @brief Test case for loading non-existent context file
 */
TEST_F (NNStreamerFilterLlamaCppTest, contextFileNotFound_n)
{
  skip_llamacpp_tc ("contextFileNotFound_n");

  const gchar *nonexistent_file = "./invalid_context.bin";

  /* Ensure file doesn't exist */
  if (g_file_test (nonexistent_file, G_FILE_TEST_EXISTS)) {
    g_remove (nonexistent_file);
  }

  /* Try to load non-existent context file */
  g_autofree gchar *custom_str = g_strdup_printf (
      "num_predict:10,context_length:512,load_ctx:%s", nonexistent_file);
  ASSERT_TRUE (create_pipeline (model, FALSE, custom_str));

  /* Should work normally with fresh context */
  EXPECT_TRUE (data_push ("Hello my name is John."));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 1); /* Should work with fresh context */
}

/**
 * @brief Test case for saving context to invalid path
 */
TEST_F (NNStreamerFilterLlamaCppTest, contextInvalidSavePath_n)
{
  skip_llamacpp_tc ("contextInvalidSavePath_n");

  const gchar *invalid_path = "./root/invalid_context.bin";

  /* Try to save context to invalid path */
  g_autofree gchar *custom_str = g_strdup_printf (
      "num_predict:10,context_length:512,save_ctx:%s", invalid_path);
  ASSERT_TRUE (create_pipeline (model, FALSE, custom_str));

  /* Should work normally even if save fails */
  EXPECT_TRUE (data_push ("Hello my name is John."));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 1); /* Should work even if save fails */

  /* Verify file was not created */
  EXPECT_FALSE (g_file_test (invalid_path, G_FILE_TEST_EXISTS));
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with LoRA adapter applied (positive test)
 * Please refer to the PR(https://github.com/nnstreamer/nnstreamer/pull/4773) for creating a LoRa adapter.
 */
TEST_F (NNStreamerFilterLlamaCppTest, applyLoraAdapter_p)
{
  skip_llamacpp_lora_tc ("applyLoraAdapter_p");

  g_autofree gchar *model_with_lora
      = g_strdup_printf ("%s,%s", lora_paths->model_path, lora_paths->adapter_path);
  ASSERT_TRUE (create_pipeline (model_with_lora, FALSE, "num_predict:40"));

  EXPECT_TRUE (data_push ("Can you translate 'Hello world' into Korean?"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000); /* Wait for inference to complete */

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 1); /* In sync mode, expect one sample for the entire output */
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with invalid LoRA adapter path (negative test)
 * Please refer to the PR(https://github.com/nnstreamer/nnstreamer/pull/4773) for creating a LoRa adapter.
 */
TEST_F (NNStreamerFilterLlamaCppTest, applyInvalidLoraAdapter_n)
{
  skip_llamacpp_lora_tc ("applyInvalidLoraAdapter_n");

  /* Intentionally use an invalid LoRA path */
  const gchar *invalid_lora_path = "/invalid/path/to/adapter.gguf";
  g_autofree gchar *model_with_invalid_lora
      = g_strdup_printf ("%s,%s", lora_paths->model_path, invalid_lora_path);
  ASSERT_TRUE (create_pipeline (model_with_invalid_lora, FALSE, "num_predict:40"));

  /* Pipeline state change to PLAYING should fail due to invalid LoRA path */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  EXPECT_EQ (*new_sample_count, 0); /* No samples should be generated */
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
