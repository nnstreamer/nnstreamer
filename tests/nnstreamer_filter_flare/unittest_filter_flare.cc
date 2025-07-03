/**
 * @file	unittest_filter_flare.cc
 * @date	23 May 2025
 * @brief	Unit test for tensor filter flare plugin
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
  g_signal_emit_by_name (sink, "pull-sample", &sample);
  if (sample) {
    buffer = gst_sample_get_buffer (sample);
    if (gst_buffer_map (buffer, &map, GST_MAP_READ)) {
      g_printf ("%.*s", (int) map.size, (char *) map.data);
      gst_buffer_unmap (buffer, &map);
    }
    gst_sample_unref (sample);
    return GST_FLOW_OK;
  }
  return GST_FLOW_ERROR;
}

/**
 * @brief Test fixture class for tensor_filter Flare plugin unit tests.
 */
class NNStreamerFilterFlareCppTest : public ::testing::Test
{
  protected:
  GstElement *pipeline, *appsrc, *appsink, *tensor_filter;
  gboolean loaded;
  gboolean skip_test;

  gchar *model;
  gchar *data_file;
  gchar *pipeline_str;
  gchar *tokenizer;

  public:
  /**
   * @brief Constructor a new NNStreamerFilterFlareCppTest object
   */
  NNStreamerFilterFlareCppTest ()
  {
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


#if 0 /* Flare API uses a hardcoded model path internally. Copy the model to the current directory and then run TC. */
    model = g_build_filename (root_path, "tests", "test_models", "models",
        "sflare_if_4bit_3b.bin", NULL);
#else
    model = g_strdup ("./sflare_if_4bit_3b.bin");
#endif
    if (!g_file_test (model, G_FILE_TEST_EXISTS)) {
      g_critical ("Skipping test due to missing model file. Please download model file");
      skip_test = TRUE;
      return;
    }

    tokenizer = g_build_filename (
        root_path, "tests", "test_models", "data", "tokenizer.json", NULL);
    if (!g_file_test (tokenizer, G_FILE_TEST_EXISTS)) {
      g_critical ("Skipping test due to missing model file. Please download tokenizer file");
      skip_test = TRUE;
      return;
    }
    data_file = g_build_filename (
        root_path, "tests", "test_models", "data", "flare_input.txt", NULL);

    loaded = TRUE;
  }

  /**
   * @brief Tear down method after unit test execution is completed.
   */
  void TearDown () override
  {
    if (appsrc)
      gst_object_unref (appsrc);
    if (appsink)
      gst_object_unref (appsink);
    if (tensor_filter)
      gst_object_unref (tensor_filter);
    if (pipeline)
      gst_object_unref (pipeline);

    g_free (pipeline_str);
    g_free (model);
    g_free (data_file);
    g_free (tokenizer);
  }

  void create_pipeline (const gchar *model, gboolean invoke_async, const gchar *custom)
  {
    pipeline_str = g_strdup_printf (
        "appsrc name=appsrc ! application/octet-stream ! tensor_converter ! other/tensors,format=flexible ! "
        "tensor_filter name=tensor_filter framework=flare model=%s invoke-dynamic=TRUE invoke-async=%d custom=%s ! "
        "other/tensors,format=flexible ! tensor_decoder mode=octet_stream ! application/octet-stream ! appsink name=appsink",
        model, invoke_async, custom);

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

  void data_push (const gchar *data)
  {
    gsize data_size;
    GstBuffer *buffer;
    GstMapInfo map;

    data_size = strlen (data);
    buffer = gst_buffer_new_allocate (NULL, data_size, NULL);

    gst_buffer_map (buffer, &map, GST_MAP_WRITE);
    memcpy (map.data, data, data_size);
    gst_buffer_unmap (buffer, &map);

    EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc), buffer), GST_FLOW_OK);
  }
};

/**
 * Note. Inference libraries like LLP and llama.cpp generally do not allow
 * modification of such settings at runtime after the context (e.g., model,
 * session, etc.) has been created. In particular, parameters like n_gpu_layers
 * must be configured at model loading time and cannot be changed afterwards.
 */

/**
 * @brief Test case for tensor_filter Flare plugin with singleInputSingleOutputSync_p
 */
TEST_F (NNStreamerFilterFlareCppTest, singleInputSingleOutputSync_p)
{
  gboolean invoke_async = FALSE;
  gchar *contents = NULL;
  gsize length = 0;
  g_autofree gchar *custom = g_strdup_printf (
      "tokenizer_path:%s,backend:CPU,output_size:1024,model_type:3B,data_type:W4A32,enable_fsu:false",
      tokenizer);

  if (skip_test) {
    GTEST_SKIP () << "Skipping singleInputSingleOutputSync_p test due to missing model file";
  }
  ASSERT_TRUE (this->loaded);
  new_sample_count = 0;

  ASSERT_TRUE (g_file_get_contents (data_file, &contents, &length, NULL));

  g_critical ("model: %s", model);
  create_pipeline (model, invoke_async, custom);
  data_push (contents);
  /*need to lot's of time */
  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_usleep (40000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (new_sample_count, 1);
  g_free (contents);
}

/**
 * @brief Test case for tensor_filter Flare plugin with multipleInputsSingleOutputSync_p
 */
TEST_F (NNStreamerFilterFlareCppTest, multipleInputsSingleOutputSync_p)
{
  gboolean invoke_async = FALSE;
  gchar *contents = NULL;
  gsize length = 0;
  g_autofree gchar *custom = g_strdup_printf (
      "tokenizer_path:%s,backend:CPU,output_size:1024,model_type:3B,data_type:W4A32,enable_fsu:false",
      tokenizer);

  if (skip_test) {
    GTEST_SKIP () << "Skipping multipleInputsSingleOutputSync_p test due to missing model file";
  }
  ASSERT_TRUE (this->loaded);
  new_sample_count = 0;

  ASSERT_TRUE (g_file_get_contents (data_file, &contents, &length, NULL));

  g_critical ("model: %s", model);
  create_pipeline (model, invoke_async, custom);
  data_push (contents);
  data_push (contents);
  /*need to lot's of time */
  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  /*currently, API can support output_size. */
  g_usleep (80000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (new_sample_count, 2);
  g_free (contents);
}

#if 0 /*Flare call exit(0)*/
/**
 * @brief Test case for tensor_filter Flare plugin with invalidTokenizerPath_n
 */
TEST_F (NNStreamerFilterFlareCppTest, invalidTokenizerPath_n)
{
  gboolean invoke_async = FALSE;
  gchar *contents = NULL;
  gsize length = 0;
  g_autofree gchar *custom = g_strdup_printf ("tokenizer_path:invalidTokenizerPath,backend:CPU,output_size:1024,model_type:3B,data_type:W4A32,enable_fsu:false");

  ASSERT_TRUE (g_file_get_contents(data_file, &contents, &length, NULL));

  if (skip_test) {
    GTEST_SKIP ()
       << "Skipping invalidTokenizerPath_n test due to missing model file";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, NULL);
  data_push (contents);
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (3000000);
  EXPECT_EQ (new_sample_count, 0);
  g_free (contents);  
}
#endif

#if 0 /* Flare use hardcoded model path internally */
/**
 * @brief Test case for tensor_filter Flare plugin with invalidModel_n
 */
TEST_F (NNStreamerFilterFlareCppTest, invalidModel_n)
{
  gboolean invoke_async = TRUE;
  gchar *contents = NULL;
  gsize length = 0;
  g_autofree gchar *custom = g_strdup_printf ("tokenizer_path:%s,backend:CPU,output_size:1024,model_type:3B,data_type:W4A32,enable_fsu:false", tokenizer);
 
 ASSERT_TRUE (g_file_get_contents(data_file, &contents, &length,NULL));

  if (skip_test) {
    GTEST_SKIP ()
       << "Skipping invalidCustom_n test due to missing model file";
  }
  ASSERT_TRUE (this->loaded);

  new_sample_count = 0;
  create_pipeline (model, invoke_async, custom);
  data_push (contents);
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_free(contents);
}
#endif

/**
 * @brief Test case for tensor_filter Flare plugin with earlyTerminationBeforeTokenGenerationSync_n
 */
TEST_F (NNStreamerFilterFlareCppTest, earlyTerminationBeforeTokenGenerationSync_n)
{
  gboolean invoke_async = FALSE;
  gchar *contents = NULL;
  gsize length = 0;
  g_autofree gchar *custom = g_strdup_printf (
      "tokenizer_path:%s,backend:CPU,output_size:1024,model_type:3B,data_type:W4A32,enable_fsu:false",
      tokenizer);

  if (skip_test) {
    GTEST_SKIP () << "Skipping earlyTerminationBeforeTokenGenerationSync_n test due to missing model file";
  }
  ASSERT_TRUE (this->loaded);
  new_sample_count = 0;

  ASSERT_TRUE (g_file_get_contents (data_file, &contents, &length, NULL));

  g_critical ("model: %s", model);
  create_pipeline (model, invoke_async, custom);
  data_push (contents);
  /*need to lot's of time */
  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  /* g_usleep (40000000); the pipeline terminate*/
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (new_sample_count, 0);
  g_free (contents);
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
