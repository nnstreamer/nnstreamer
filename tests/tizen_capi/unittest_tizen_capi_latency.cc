/**
 * @file        unittest_tizen_capi_latency.cc
 * @date        23 July 2020
 * @brief       Unit test to measure Single C-API invoke latency.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug         No known bugs
 */

#define RUN_COUNT 100

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nnstreamer-capi-private.h>
#include <nnstreamer-single.h>
#include <nnstreamer.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>
#include <unittest_util.h>

/**
 * @brief nnstreamer invoke latency testing base class
 */
class nnstreamer_capi_singleshot_latency : public ::testing::Test
{
  protected:
  /**
   * @brief initial setup overload virtual function
   */
  virtual void SetUp ()
  {
    root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
    /* supposed to run test in build directory */
    if (root_path == NULL)
      root_path = "..";

    status = ML_ERROR_NONE;
    single_total_invoke_duration = 0;
    single_invoke_duration_f = 0;
    direct_invoke_duration_f = 0;

    model_file = g_build_filename (
        root_path, "tests", "test_models", "models", new_model, NULL);
    ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

    data_file = g_build_filename (
        root_path, "tests", "test_models", "data", "orange.raw", NULL);
    ASSERT_TRUE (g_file_test (data_file, G_FILE_TEST_EXISTS));

#if defined(ENABLE_NNFW_RUNTIME)
    manifest_file = g_build_filename (root_path, "tests", "test_models",
        "models", "metadata", "MANIFEST", NULL);
    ASSERT_TRUE (g_file_test (manifest_file, G_FILE_TEST_EXISTS));

    replace_command = g_strdup_printf ("sed -i '/%s/c\\\"models\" : [ \"%s\" ],' %s",
        orig_model, new_model, manifest_file);
    status = system (replace_command);
    g_free (replace_command);
    ASSERT_EQ (status, 0);
#endif

    fd = open (data_file, O_RDONLY);
    EXPECT_TRUE (fd >= 0);
  }

  /**
   * @brief final tear down overload virtual function
   */
  virtual void TearDown ()
  {
#if defined(ENABLE_NNFW_RUNTIME)
    if (manifest_file) {
      replace_command = g_strdup_printf ("sed -i '/%s/c\\\"models\" : [ \"%s\" ],' %s",
          new_model, orig_model, manifest_file);
      status = system (replace_command);
      g_free (replace_command);
    }
    g_free (manifest_file);
#endif

    g_free (model_file);
    g_free (data_file);

    if (fd >= 0)
      close (fd);
  }

  /**
   * @brief get the framework internal latency
   */
  void extractInternalInvokeTime (const char *filter)
  {
    /** Find the framework */
    const GstTensorFilterFramework *sp = nnstreamer_filter_find (filter);
    EXPECT_NE (sp, (void *)NULL);

    /** Extract the statictics from the framework */
    EXPECT_TRUE (sp->statistics != NULL);
    if (sp->statistics) {
      direct_invoke_duration_f = (sp->statistics->total_invoke_latency * 1.0f) /
        sp->statistics->total_invoke_num;

      float internalOverheadTime = (sp->statistics->total_overhead_latency * 1.0f) /
        sp->statistics->total_invoke_num;
      g_warning ("Latency introduced by tensor filter extension = %f us.",
          internalOverheadTime);
    }
  }

  /**
   * @brief get argmax from the array
   */
  void matchOutput (void *output_data, size_t size)
  {
    size_t idx, max_idx = 0;
    guint8 *array = (guint8 *)output_data;
    guint8 max_value = 0;
    for (idx = 0; idx < size; idx++) {
      if (max_value < array[idx]) {
        max_idx = idx;
        max_value = array[idx];
      }
    }

    /**
     * entry 952 (idx 951) is orange as per
     * tests/test_models/labels/labels.txt
     */
    EXPECT_EQ (max_idx, 951U);
  }

  /**
   * @brief Resets the data file to the start
   */
  void resetDataFile ()
  {
    status = lseek (fd, 0, SEEK_SET);
    EXPECT_EQ (status, 0);
  }

  /**
   * @brief Benchmark the invoke time for the single API
   */
  void benchmarkSingleInvoke (ml_nnfw_type_e nnfw, const bool no_alloc)
  {
    ml_single_h single;
    ml_tensors_info_h in_info, out_info;
    ml_tensors_data_h input, output;

    /** Open the single handle */
    status = ml_single_open (&single, model_file, NULL, NULL, nnfw, ML_NNFW_HW_ANY);
    ASSERT_EQ (status, ML_ERROR_NONE);

    /** Get input/output data info */
    status = ml_single_get_input_info (single, &in_info);
    EXPECT_EQ (status, ML_ERROR_NONE);

    status = ml_single_get_output_info (single, &out_info);
    EXPECT_EQ (status, ML_ERROR_NONE);

    /** Allocate input data */
    input = output = NULL;
    status = ml_tensors_data_create (in_info, &input);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_TRUE (input != NULL);

    /** Load input data into the buffer */
    status = ml_tensors_data_get_tensor_data (input, 0, (void **)&data, &data_size);
    EXPECT_EQ (status, ML_ERROR_NONE);
    if (fd >= 0) {
      resetDataFile ();
      data_read = read (fd, data, data_size);
      EXPECT_EQ ((size_t) data_read, data_size);
    }

    /** Benchmark the invoke duration */
    for (int idx = 0; idx < RUN_COUNT; ++idx) {
      if (no_alloc) {
        status = ml_tensors_data_create (out_info, &output);
        EXPECT_EQ (status, ML_ERROR_NONE);
        EXPECT_TRUE (output != NULL);
      }

      start = g_get_monotonic_time ();
      if (no_alloc)
        status = ml_single_invoke_fast (single, input, output);
      else
        status = ml_single_invoke (single, input, &output);
      end = g_get_monotonic_time ();
      single_total_invoke_duration += end - start;
      EXPECT_EQ (status, ML_ERROR_NONE);
      EXPECT_TRUE (output != NULL);

      /** Match output with the first run */
      if (idx == 0) {
        status = ml_tensors_data_get_tensor_data (output, 0, (void **)&data,
            &data_size);
        EXPECT_EQ (status, ML_ERROR_NONE);
        matchOutput (data, data_size);
      }

      ml_tensors_data_destroy (output);
    }

    single_invoke_duration_f = (single_total_invoke_duration * 1.0) / RUN_COUNT;
    g_warning ("Time to invoke single = %f us", single_invoke_duration_f);

    /** Close the single handle */
    status = ml_single_close (single);
    EXPECT_EQ (status, ML_ERROR_NONE);

    ml_tensors_data_destroy (input);
    ml_tensors_info_destroy (in_info);
    ml_tensors_info_destroy (out_info);
  }

  /**
   * @brief Benchmark the latency by the single API invoke
   */
  void benchmarkSingleInvokeLatency (ml_nnfw_type_e nnfw, const char * fw, const bool no_alloc)
  {
    /** sleep 30 sec for cooldown from any previous runs */
    sleep (30);

    benchmarkSingleInvoke (nnfw, no_alloc);
    extractInternalInvokeTime (fw);

    g_warning ("Total Latency added by single API over framework %s for invoke"
        "= %f us", fw, single_invoke_duration_f - direct_invoke_duration_f);
  }

  void *data = NULL;
  int status, fd;
  const gchar *root_path;
  const gchar *new_model = "mobilenet_v1_1.0_224_quant.tflite";
  gchar *model_file = NULL, *data_file = NULL;
#if defined(ENABLE_NNFW_RUNTIME)
  const gchar *orig_model = "add.tflite";
  gchar *replace_command = NULL, *manifest_file = NULL;
#endif
  int64_t start, end;
  ssize_t data_read;
  size_t data_size;
  int64_t single_total_invoke_duration;
  float single_invoke_duration_f, direct_invoke_duration_f;
};

#if defined(ENABLE_TENSORFLOW_LITE)
/**
 * @brief Measure latency for NNStreamer single shot (tensorflow-lite)
 * @note Measure the invoke latency added by NNStreamer single shot
 */
TEST_F (nnstreamer_capi_singleshot_latency, benchmarkTensorflowLite)
{
  benchmarkSingleInvokeLatency (ML_NNFW_TYPE_TENSORFLOW_LITE, "tensorflow-lite", false);
}

/**
 * @brief Measure latency for NNStreamer single shot (tensorflow-lite, no output alloc in invoke)
 * @note Measure the invoke latency added by NNStreamer single shot
 */
TEST_F (nnstreamer_capi_singleshot_latency, benchmarkTensorflowLite_no_alloc)
{
  benchmarkSingleInvokeLatency (ML_NNFW_TYPE_TENSORFLOW_LITE, "tensorflow-lite", true);
}
#endif

#if defined(ENABLE_NNFW_RUNTIME)
/**
 * @brief Measure latency for NNStreamer single shot (nnfw-runtime)
 * @note Measure the invoke latency added by NNStreamer single shot
 */
TEST_F (nnstreamer_capi_singleshot_latency, benchmarkNNFWRuntime)
{
  benchmarkSingleInvokeLatency (ML_NNFW_TYPE_NNFW, "nnfw", false);
}
#endif

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

  /* ignore tizen feature status while running the testcases */
  set_feature_state (1);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  set_feature_state (-1);

  return result;
}
