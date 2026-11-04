/**
 * @file	unittest_filter_custom.cc
 * @date	23 May 2025
 * @brief	Unit test for tensor filter llamacpp plugin
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyunil Park <hyunil46.park@samsung.com>
 * @bug		No known bugs.
 */

#include <gtest/gtest.h>
#include <errno.h>
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
 * @brief Test fixture class for tensor_filter llama-cpp plugin unit tests.
 */
class NNStreamerFilterLlamaCppTest : public ::testing::Test
{
  protected:
  GstElement *pipeline, *appsrc, *appsink, *tensor_filter;
  gchar *model;
  LoRAPaths *lora_paths;
  gint *new_sample_count;
  gsize max_sample_size;
  GString *generated_text;
  gboolean pipeline_error;
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
    max_sample_size = 0;
    generated_text = nullptr;
    pipeline_error = FALSE;

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
    new_sample_count = g_new0 (gint, 1);
    max_sample_size = 0;
    generated_text = g_string_new (NULL);
    pipeline_error = FALSE;
  }

  /**
   * @brief Tear down method after unit test execution is completed.
   */
  void TearDown () override
  {
    clear_pipeline ();
    g_clear_pointer (&new_sample_count, g_free);
    if (generated_text) {
      g_string_free (generated_text, TRUE);
      generated_text = nullptr;
    }
  }

  /**
   * @brief Callback for appsink to pull a sample, print the generated text and
   *        record the sample count, the largest payload seen and the text
   *        of all samples in arrival order.
   *
   * A synchronous zero-token generation carries only a string terminator;
   * an asynchronous zero-token generation produces no sample.
   */
  static GstFlowReturn new_sample_cb (GstElement *sink, gpointer user_data)
  {
    NNStreamerFilterLlamaCppTest *self = (NNStreamerFilterLlamaCppTest *) user_data;
    GstSample *sample;
    GstBuffer *buffer;
    GstMapInfo map;

    if (self && self->new_sample_count)
      g_atomic_int_inc (self->new_sample_count);

    sample = gst_app_sink_pull_sample (GST_APP_SINK (sink));
    if (sample) {
      buffer = gst_sample_get_buffer (sample);
      if (gst_buffer_map (buffer, &map, GST_MAP_READ)) {
        g_print ("%.*s", (int) map.size, (char *) map.data);
        if (self && map.size > self->max_sample_size)
          self->max_sample_size = map.size;
        if (self && self->generated_text && map.size > 0)
          g_string_append_len (self->generated_text, (const gchar *) map.data,
              strnlen ((const char *) map.data, map.size));
        gst_buffer_unmap (buffer, &map);
      }
      gst_sample_unref (sample);
      return GST_FLOW_OK;
    }
    return GST_FLOW_ERROR;
  }

  /**
   * @brief Log handler counting the context trimming of the sub-plugin and
   *        passing other messages to the default handler.
   */
  static void trim_log_cb (const gchar *log_domain, GLogLevelFlags log_level,
      const gchar *message, gpointer user_data)
  {
    if (g_str_has_prefix (message, "Context is full"))
      g_atomic_int_inc ((gint *) user_data);
    else
      g_log_default_handler (log_domain, log_level, message, NULL);
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

    /**
     * Do not gate rendering on the clock. tensor_converter timestamps a
     * buffer that carries no PTS from the clock, and while the element's
     * base time is still unset that yields an absolute clock time instead of
     * a running time - a PTS minutes into the future. A synchronizing sink
     * then holds the buffer and the sample never arrives. These tests measure
     * generation, not playback, so clock synchronization has no place here.
     * The converter behaviour itself is tracked as issue #4898; revisit this
     * once that is fixed.
     */
    g_object_set (G_OBJECT (appsink), "emit-signals", TRUE, "sync", FALSE, NULL);
    g_signal_connect (appsink, "new-sample", G_CALLBACK (new_sample_cb), this);
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

  /**
   * @brief Drain the pipeline bus and latch whether the pipeline has failed.
   *
   * A sub-plugin that fails to invoke makes tensor_filter post an
   * "invoke-failure" application message and return GST_FLOW_ERROR, which
   * appsrc turns into a bus error. Without this check such a failure is
   * indistinguishable from a slow inference: both end up as a wait timeout.
   *
   * @return TRUE if the pipeline has posted an error so far.
   */
  gboolean poll_pipeline_error ()
  {
    GstBus *bus;
    GstMessage *msg;

    if (pipeline == nullptr)
      return pipeline_error;

    bus = gst_element_get_bus (pipeline);
    if (bus == nullptr)
      return pipeline_error;

    while ((msg = gst_bus_pop_filtered (bus,
                (GstMessageType) (GST_MESSAGE_ERROR | GST_MESSAGE_APPLICATION)))
           != NULL) {
      if (GST_MESSAGE_TYPE (msg) == GST_MESSAGE_ERROR) {
        GError *err = NULL;

        gst_message_parse_error (msg, &err, NULL);
        g_printerr ("Pipeline error: %s\n", err ? err->message : "unknown");
        g_clear_error (&err);
        pipeline_error = TRUE;
      } else {
        const GstStructure *s = gst_message_get_structure (msg);

        if (s && gst_structure_has_name (s, "nnstreamer-message")
            && g_strcmp0 (gst_structure_get_string (s, "type"), "invoke-failure") == 0) {
          g_printerr ("Invoke failure: %s\n",
              GST_STR_NULL (gst_structure_get_string (s, "message")));
          pipeline_error = TRUE;
        }
      }
      gst_message_unref (msg);
    }
    gst_object_unref (bus);

    return pipeline_error;
  }

  /**
   * @brief Wait until the appsink has received the expected number of samples.
   *        After the count is reached, a short grace period lets unexpected
   *        extra samples arrive so that exact-count assertions can catch them.
   *        The wait is aborted early if the pipeline reports a failure.
   * @param expected Number of samples to wait for.
   * @param timeout_ms Maximum wait time in milliseconds.
   * @return TRUE if the expected count was reached before the timeout.
   */
  gboolean wait_for_samples (guint expected, guint timeout_ms = 10000U)
  {
    guint waited_ms = 0;

    while ((guint) g_atomic_int_get (new_sample_count) < expected && waited_ms < timeout_ms) {
      if (poll_pipeline_error ())
        break;
      g_usleep (10000);
      waited_ms += 10;
    }

    if ((guint) g_atomic_int_get (new_sample_count) < expected)
      return FALSE;

    g_usleep (100000);
    poll_pipeline_error ();
    return TRUE;
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
  EXPECT_TRUE (wait_for_samples (10));

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GE (*new_sample_count, 10);
}

/**
 * @brief Test case for tensor_filter llama-cpp plugin with singleInputSingleOutputSync_p
 *
 * With no sampling option the sampler chain is greedy alone, so a fresh
 * context generates the same tokens for the same prompt on every run.
 * singleInputMultipleOutputsAsync_p already requires ten tokens from this
 * prompt, so the output being more than the terminator is not model luck.
 */
TEST_F (NNStreamerFilterLlamaCppTest, singleInputSingleOutputSync_p)
{
  skip_llamacpp_tc ("singleInputSingleOutputSync_p");

  ASSERT_TRUE (create_pipeline (model, FALSE, "num_predict:10"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 1);
  EXPECT_GT (max_sample_size, 1U);
  /* The text and its terminator, nothing else. */
  EXPECT_EQ (max_sample_size, generated_text->len + 1);
}

/**
 * @brief Test that a synchronous generation is the same text as the tokens an
 *        asynchronous generation dispatches one by one.
 *
 * Both runs use a fresh context and the greedy-only sampler chain, so they
 * decode the same tokens; the comparison holds on any CPU backend and model
 * version as long as generation stays deterministic.
 */
TEST_F (NNStreamerFilterLlamaCppTest, syncGenerationMatchesAsyncTokens_p)
{
  g_autofree gchar *async_text = NULL;
  gboolean async_done;

  skip_llamacpp_tc ("syncGenerationMatchesAsyncTokens_p");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:10"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  /* Only the ten token callbacks produce samples. */
  async_done = wait_for_samples (10);
  EXPECT_FALSE (pipeline_error);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  clear_pipeline ();
  ASSERT_TRUE (async_done);

  async_text = g_strdup (generated_text->str);
  g_string_truncate (generated_text, 0);
  g_atomic_int_set (new_sample_count, 0);
  ASSERT_GT (strlen (async_text), 0U);

  ASSERT_TRUE (create_pipeline (model, FALSE, "num_predict:10"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_EQ (*new_sample_count, 1);
  EXPECT_STREQ (generated_text->str, async_text);
}

/**
 * @brief Test that a synchronous generation with a smaller num_predict is a
 *        strict prefix of one with a larger num_predict.
 *
 * Greedy decoding from a fresh context makes the first five tokens the same in
 * both runs, and singleInputMultipleOutputsAsync_p shows that this prompt does
 * not end within ten tokens.
 */
TEST_F (NNStreamerFilterLlamaCppTest, syncGenerationNumPredictPrefix_p)
{
  g_autofree gchar *short_text = NULL;

  skip_llamacpp_tc ("syncGenerationNumPredictPrefix_p");

  ASSERT_TRUE (create_pipeline (model, FALSE, "num_predict:5"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  clear_pipeline ();

  short_text = g_strdup (generated_text->str);
  g_string_truncate (generated_text, 0);
  g_atomic_int_set (new_sample_count, 0);
  ASSERT_GT (strlen (short_text), 0U);

  ASSERT_TRUE (create_pipeline (model, FALSE, "num_predict:10"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_EQ (*new_sample_count, 1);
  EXPECT_TRUE (g_str_has_prefix (generated_text->str, short_text));
  EXPECT_GT (generated_text->len, strlen (short_text));
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
  EXPECT_TRUE (wait_for_samples (14));

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GE (*new_sample_count, 14);
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
  EXPECT_TRUE (wait_for_samples (2));
  EXPECT_FALSE (pipeline_error);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 2);
}

/**
 * @brief Test case for a generation that produces no token at all.
 *
 * A zero-token generation is a valid result: the filter must still hand one
 * (empty) output to the pipeline so that every input yields exactly one sample.
 * This is the deterministic form of the intermittent stall where the model
 * sampled end-of-generation as the very first token of the second prompt.
 */
TEST_F (NNStreamerFilterLlamaCppTest, zeroTokenGenerationSync_p)
{
  skip_llamacpp_tc ("zeroTokenGenerationSync_p");

  ASSERT_TRUE (create_pipeline (model, FALSE, "num_predict:0"));
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_TRUE (data_push ("What is AI?"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (2));
  EXPECT_FALSE (pipeline_error);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 2);
  /* Both samples carry the string terminator and nothing else. */
  EXPECT_EQ (max_sample_size, 1U);
}

/**
 * @brief A zero-token async generation emits no placeholder for either input.
 */
TEST_F (NNStreamerFilterLlamaCppTest, zeroTokenGenerationAsync_p)
{
  skip_llamacpp_tc ("zeroTokenGenerationAsync_p");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:0"));
  /* No token buffer is available to complete sink preroll. */
  GstElement *sink = gst_bin_get_by_name (GST_BIN (pipeline), "appsink");
  ASSERT_NE (sink, nullptr);
  g_object_set (sink, "async", FALSE, nullptr);
  gst_object_unref (sink);
  EXPECT_TRUE (data_push ("Hello my name is"));
  EXPECT_TRUE (data_push ("What is AI?"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  GstElement *source = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc");
  ASSERT_NE (source, nullptr);
  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (source)), GST_FLOW_OK);
  gst_object_unref (source);
  GstBus *bus = gst_element_get_bus (pipeline);
  GstMessage *message = gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  EXPECT_NE (message, nullptr);
  if (message) {
    EXPECT_EQ (GST_MESSAGE_TYPE (message), GST_MESSAGE_EOS);
    gst_message_unref (message);
  }
  gst_object_unref (bus);
  EXPECT_FALSE (pipeline_error);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 0);
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
 * @brief Test case for a negative num_predict.
 *
 * llama.cpp spells "generate until the context is full" as -1, which this
 * sub-plugin does not implement: the loop would simply never run. Since an
 * empty generation is now a valid result rather than an error, such a
 * misconfiguration would otherwise pass silently, so it is rejected up front.
 */
TEST_F (NNStreamerFilterLlamaCppTest, negativeNumPredict_n)
{
  skip_llamacpp_tc ("negativeNumPredict_n");

  ASSERT_TRUE (create_pipeline (model, TRUE, "num_predict:-1"));
  /**
   * -ESTRPIPE, not merely non-zero: a pipeline that is never fed also fails to
   * reach PLAYING, but with -ETIME once preroll times out. Only the stricter
   * value distinguishes "the sub-plugin rejected the property" from "nothing
   * ever arrived", so this stays a regression test for the rejection itself.
   */
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
      -ESTRPIPE);
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
 * @brief Test that closing an asynchronous filter while it generates or right
 *        after it has finished always returns.
 *
 * The output thread has to see the stop request whichever point of its wait
 * loop it is at. Each round closes as soon as the first token arrives, with no
 * grace period: with num_predict:64 while the thread generates, with
 * num_predict:1 while it decodes its only token and goes back to waiting.
 * A lost wakeup blocks the close in the join of the output thread, which no
 * state-change timeout bounds; the test then hangs until the meson timeout.
 * The window is narrow, so a regression is caught only probabilistically.
 */
TEST_F (NNStreamerFilterLlamaCppTest, closeWhileGeneratingAsync_p)
{
  const gchar *customs[] = { "num_predict:64", "num_predict:1" };
  gint count_at_null;
  gint64 deadline;
  guint round;

  skip_llamacpp_tc ("closeWhileGeneratingAsync_p");

  for (round = 0; round < 20U; round++) {
    ASSERT_TRUE (create_pipeline (model, TRUE, customs[round % 2]));
    EXPECT_TRUE (data_push ("Hello my name is"));
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
        0);
    deadline = g_get_monotonic_time () + UNITTEST_STATECHANGE_TIMEOUT * G_TIME_SPAN_MILLISECOND;
    while (g_atomic_int_get (new_sample_count) < 1 && g_get_monotonic_time () < deadline)
      g_thread_yield ();
    EXPECT_GE (g_atomic_int_get (new_sample_count), 1);
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
    count_at_null = g_atomic_int_get (new_sample_count);
    g_usleep (100000);
    EXPECT_EQ (g_atomic_int_get (new_sample_count), count_at_null);
    EXPECT_FALSE (pipeline_error);
    clear_pipeline ();
    g_atomic_int_set (new_sample_count, 0);
  }
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
  EXPECT_TRUE (wait_for_samples (10));

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GE (*new_sample_count, 10);
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
  EXPECT_TRUE (wait_for_samples (10));

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GE (*new_sample_count, 10);
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
  EXPECT_TRUE (wait_for_samples (10));

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GE (*new_sample_count, 10);
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
  EXPECT_TRUE (wait_for_samples (10));

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GE (*new_sample_count, 10);
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
  EXPECT_TRUE (wait_for_samples (10));

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GE (*new_sample_count, 10);
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
  EXPECT_TRUE (wait_for_samples (10));

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GE (*new_sample_count, 10);
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
  EXPECT_TRUE (wait_for_samples (10));

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GE (*new_sample_count, 10);
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
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);

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
  EXPECT_TRUE (wait_for_samples (2));
  EXPECT_FALSE (pipeline_error);

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
  EXPECT_TRUE (wait_for_samples (9));

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_GE (*new_sample_count, 9);
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
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 1);

  /* Verify context file was created */
  EXPECT_TRUE (g_file_test (context_file, G_FILE_TEST_EXISTS));

  /* Second pipeline: load context */
  clear_pipeline ();
  g_atomic_int_set (new_sample_count, 0);

  g_autofree gchar *custom_str2 = g_strdup_printf (
      "num_predict:20,context_length:512,load_ctx:%s", context_file);
  ASSERT_TRUE (create_pipeline (model, FALSE, custom_str2));

  EXPECT_TRUE (data_push ("What did I just say about myself?"));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 1);

  /* Clean up test file */
  g_remove (context_file);
}

/**
 * @brief Test that a loaded context continues the conversation it was saved from.
 *
 * Greedy decoding makes the reply to the second prompt depend only on the
 * context, so a conversation split by save_ctx / load_ctx must produce the
 * same text as the one run in a single context, and a different text from
 * the second prompt in a fresh context.
 */
TEST_F (NNStreamerFilterLlamaCppTest, contextLoadContinuesConversation_p)
{
  const gchar *context_file = "./context_continue.bin";
  const gchar *first_prompt = "Hello my name is John. I like programming and AI.";
  const gchar *second_prompt = "What did I just say about myself?";
  g_autofree gchar *conversation_text = NULL;
  g_autofree gchar *first_text = NULL;
  g_autofree gchar *loaded_text = NULL;
  g_autofree gchar *custom_save = NULL;
  g_autofree gchar *custom_load = NULL;

  skip_llamacpp_tc ("contextLoadContinuesConversation_p");
  g_remove (context_file);

  /* Both prompts in a single context. */
  ASSERT_TRUE (create_pipeline (model, FALSE, "num_predict:15,context_length:512"));
  EXPECT_TRUE (data_push (first_prompt));
  EXPECT_TRUE (data_push (second_prompt));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (2));
  EXPECT_FALSE (pipeline_error);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 2);
  clear_pipeline ();
  conversation_text = g_strdup (generated_text->str);
  g_string_truncate (generated_text, 0);
  g_atomic_int_set (new_sample_count, 0);

  /* The first prompt, saving the context at close. */
  custom_save = g_strdup_printf ("num_predict:15,context_length:512,save_ctx:%s", context_file);
  ASSERT_TRUE (create_pipeline (model, FALSE, custom_save));
  EXPECT_TRUE (data_push (first_prompt));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  clear_pipeline ();
  ASSERT_TRUE (g_file_test (context_file, G_FILE_TEST_EXISTS));
  first_text = g_strdup (generated_text->str);
  g_string_truncate (generated_text, 0);
  g_atomic_int_set (new_sample_count, 0);
  ASSERT_GT (strlen (first_text), 0U);

  /* The second prompt on the loaded context. */
  custom_load = g_strdup_printf ("num_predict:15,context_length:512,load_ctx:%s", context_file);
  ASSERT_TRUE (create_pipeline (model, FALSE, custom_load));
  EXPECT_TRUE (data_push (second_prompt));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 1);
  clear_pipeline ();
  loaded_text = g_strdup (generated_text->str);
  g_string_truncate (generated_text, 0);
  g_atomic_int_set (new_sample_count, 0);
  g_remove (context_file);

  EXPECT_TRUE (g_str_has_prefix (conversation_text, first_text));
  EXPECT_STREQ (conversation_text + strlen (first_text), loaded_text);

  /* The second prompt in a fresh context. */
  ASSERT_TRUE (create_pipeline (model, FALSE, "num_predict:15,context_length:512"));
  EXPECT_TRUE (data_push (second_prompt));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (*new_sample_count, 1);
  EXPECT_STRNE (generated_text->str, loaded_text);
}

/**
 * @brief Test that the tokens of a loaded context count for the context trimming.
 *
 * The first prompt and its reply take about 210 of the 256 context cells, and
 * the second prompt is about 70 tokens (measured with TinyStories-656K-Q2_K
 * on llama.cpp b6569): it fits only after the trimming that the restored
 * tokens trigger. A conversation split by save_ctx / load_ctx
 * must trim at the same point as the one run in a single context and produce
 * the same text; counting the loaded context as empty overflows it instead.
 * Both runs must log one trimming, so a model or llama.cpp change that lets
 * the second prompt fit untrimmed fails the test instead of passing vacuously.
 */
TEST_F (NNStreamerFilterLlamaCppTest, contextLoadTrimsRestoredTokens_p)
{
  const gchar *context_file = "./context_trim.bin";
  const gchar *custom = "num_predict:10,context_length:256";
  const gchar *story
      = "Once upon a time, there was a little girl named Lily. She loved to play "
        "outside in the sun with her dog Max. One day, Lily and Max went to the "
        "park. They saw a big tree with red apples. Lily wanted to eat an apple, "
        "but the tree was too tall. Max barked at the tree. Then a kind man came "
        "and picked an apple for Lily.";
  const gchar *sequel = "The next day, Lily went back to the park with her mom. "
                        "They had a picnic under the big tree and";
  g_autofree gchar *first_prompt = g_strjoin (" ", story, story, story, NULL);
  g_autofree gchar *second_prompt = g_strjoin (" ", sequel, sequel, sequel, NULL);
  g_autofree gchar *conversation_text = NULL;
  g_autofree gchar *first_text = NULL;
  g_autofree gchar *custom_save = NULL;
  g_autofree gchar *custom_load = NULL;
  gint trims = 0;
  guint handler;

  skip_llamacpp_tc ("contextLoadTrimsRestoredTokens_p");
  g_remove (context_file);

  ASSERT_TRUE (create_pipeline (model, FALSE, custom));
  handler = g_log_set_handler (NULL, G_LOG_LEVEL_DEBUG, trim_log_cb, &trims);
  EXPECT_TRUE (data_push (first_prompt));
  EXPECT_TRUE (data_push (second_prompt));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (2));
  EXPECT_FALSE (pipeline_error);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_log_remove_handler (NULL, handler);
  EXPECT_EQ (*new_sample_count, 2);
  EXPECT_EQ (g_atomic_int_get (&trims), 1);
  clear_pipeline ();
  conversation_text = g_strdup (generated_text->str);
  g_string_truncate (generated_text, 0);
  g_atomic_int_set (new_sample_count, 0);

  custom_save = g_strdup_printf ("%s,save_ctx:%s", custom, context_file);
  ASSERT_TRUE (create_pipeline (model, FALSE, custom_save));
  EXPECT_TRUE (data_push (first_prompt));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  clear_pipeline ();
  ASSERT_TRUE (g_file_test (context_file, G_FILE_TEST_EXISTS));
  first_text = g_strdup (generated_text->str);
  g_string_truncate (generated_text, 0);
  g_atomic_int_set (new_sample_count, 0);

  custom_load = g_strdup_printf ("%s,load_ctx:%s", custom, context_file);
  ASSERT_TRUE (create_pipeline (model, FALSE, custom_load));
  g_atomic_int_set (&trims, 0);
  handler = g_log_set_handler (NULL, G_LOG_LEVEL_DEBUG, trim_log_cb, &trims);
  EXPECT_TRUE (data_push (second_prompt));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_log_remove_handler (NULL, handler);
  EXPECT_EQ (*new_sample_count, 1);
  EXPECT_EQ (g_atomic_int_get (&trims), 1);
  g_remove (context_file);

  EXPECT_TRUE (g_str_has_prefix (conversation_text, first_text));
  EXPECT_STREQ (conversation_text + strlen (first_text), generated_text->str);
}

/**
 * @brief Test that a truncated context file falls back to a fresh context.
 *
 * The file is cut to nothing, to 16 bytes (failing in the header) and to half
 * its size (failing in the memory payload, after part of it has been
 * restored). Each load must give the text of a fresh context.
 */
TEST_F (NNStreamerFilterLlamaCppTest, contextLoadTruncatedFile_n)
{
  const gchar *context_file = "./context_truncated.bin";
  const gchar *prompt = "What did I just say about myself?";
  g_autofree gchar *fresh_text = NULL;
  g_autofree gchar *contents = NULL;
  g_autofree gchar *custom_save = NULL;
  g_autofree gchar *custom_load = NULL;
  gsize length = 0;
  gsize cuts[3];
  guint i;

  skip_llamacpp_tc ("contextLoadTruncatedFile_n");
  g_remove (context_file);

  custom_save = g_strdup_printf ("num_predict:15,context_length:512,save_ctx:%s", context_file);
  ASSERT_TRUE (create_pipeline (model, FALSE, custom_save));
  EXPECT_TRUE (data_push ("Hello my name is John. I like programming and AI."));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  clear_pipeline ();
  g_string_truncate (generated_text, 0);
  g_atomic_int_set (new_sample_count, 0);

  ASSERT_TRUE (g_file_get_contents (context_file, &contents, &length, NULL));
  ASSERT_GT (length, 32U);
  cuts[0] = 0;
  cuts[1] = 16;
  cuts[2] = length / 2;

  ASSERT_TRUE (create_pipeline (model, FALSE, "num_predict:15,context_length:512"));
  EXPECT_TRUE (data_push (prompt));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  clear_pipeline ();
  fresh_text = g_strdup (generated_text->str);
  g_string_truncate (generated_text, 0);
  g_atomic_int_set (new_sample_count, 0);

  custom_load = g_strdup_printf ("num_predict:15,context_length:512,load_ctx:%s", context_file);
  for (i = 0; i < G_N_ELEMENTS (cuts); i++) {
    ASSERT_TRUE (g_file_set_contents (context_file, contents, cuts[i], NULL));
    ASSERT_TRUE (create_pipeline (model, FALSE, custom_load));
    EXPECT_TRUE (data_push (prompt));
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
        0);
    EXPECT_TRUE (wait_for_samples (1));
    EXPECT_FALSE (pipeline_error);
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
    clear_pipeline ();

    EXPECT_EQ (*new_sample_count, 1) << "cut at " << cuts[i] << " of " << length;
    EXPECT_STREQ (generated_text->str, fresh_text)
        << "cut at " << cuts[i] << " of " << length;
    g_string_truncate (generated_text, 0);
    g_atomic_int_set (new_sample_count, 0);
  }
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
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);
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
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);
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
  EXPECT_TRUE (wait_for_samples (1));
  EXPECT_FALSE (pipeline_error);

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
