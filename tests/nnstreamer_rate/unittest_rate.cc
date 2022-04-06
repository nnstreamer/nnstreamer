/**
 * @file    unittest_rate.cc
 * @date    02 Dec 2020
 * @brief   Unit test for tensor_rate element
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <unittest_util.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api.h>

#define NNS_TENSOR_RATE_NAME "tensor_rate"

/**
 * @brief Test tensor_rate existence.
 */
TEST (nnstreamerRate, checkExistence)
{
  GstElementFactory *factory;

  factory = gst_element_factory_find (NNS_TENSOR_RATE_NAME);
  EXPECT_TRUE (factory != NULL);
  gst_object_unref (factory);
}

/**
 * @brief Test tensor_rate existence (negative).
 */
TEST (nnstreamerRate, checkExistence_n)
{
  GstElementFactory *factory;
  g_autofree gchar *name = nullptr;

  name = g_strconcat (NNS_TENSOR_RATE_NAME, "_dummy", NULL);
  factory = gst_element_factory_find (name);
  EXPECT_TRUE (factory == NULL);
}

/**
 * @brief Test Fixture class for a tensor_rate element
 */
class NNSRateTest : public testing::Test
{
protected:
  /**
   * @brief Test Mode enumerator
   */
  enum TestMode {
    TENSOR_RATE_MODE_PASSTHROUGH = 0,
    TENSOR_RATE_MODE_NO_THROTTLE,
    TENSOR_RATE_MODE_THROTTLE,
  };

  guint source_num_buffers;
  gchar *target_framerate;
  gchar *framework;
  gchar *modelpath;
  GstElement *pipeline;
  gboolean silent, throttle;
  gchar *source_framerate;
  GstElement *rate;
  TestMode mode;

  const gboolean DEFAULT_SILENT = TRUE;
  const gboolean DEFAULT_THROTTLE = FALSE;
  const guint DEFAULT_SOURCE_NUM_BUFFERS = 300;
  const std::string DEFAULT_SOURCE_FRAMERATE = "30/1";
  const std::string DEFAULT_TARGET_FRAMERATE = "0/1";
  const guint64 DEFAULT_IN = 0;
  const guint64 DEFAULT_OUT = 0;
  const guint64 DEFAULT_DUP = 0;
  const guint64 DEFAULT_DROP = 0;

  /**
   * @brief Construct a new NNSRateTest object
   */
  NNSRateTest() :
    source_num_buffers (0), target_framerate (nullptr), framework (nullptr),
    modelpath (nullptr), pipeline (nullptr), silent (FALSE), throttle (FALSE),
    source_framerate (nullptr), rate (nullptr), mode (TENSOR_RATE_MODE_PASSTHROUGH) {}

  /**
   * @brief Wait until the EOS message is received or the timeout is expired.
   * @param pipeline target pipeline element to watch.
   * @return @c TRUE if EOS message is received in the timeout period. Otherwise FALSE.
   */
  static gboolean
  wait_pipeline_eos (GstElement *pipeline) {
    GstBus *bus = gst_element_get_bus (pipeline);
    gboolean got_eos_message = FALSE;

    if (GST_IS_BUS (bus)) {
      const gulong timeout = G_USEC_PER_SEC * 10;
      const gulong timeout_slice = G_USEC_PER_SEC / 10;
      gulong timeout_accum = 0;
      GstMessage *msg;

      while (!got_eos_message && timeout_accum < timeout) {
        g_usleep (timeout_slice);
        timeout_accum += timeout_slice;

        while ((msg = gst_bus_pop (bus)) != NULL) {
          gst_bus_async_signal_func(bus, msg, NULL);

          switch (GST_MESSAGE_TYPE (msg)) {
            case GST_MESSAGE_EOS:
              got_eos_message = TRUE;
              break;
            default:
              break;
          }
          gst_message_unref(msg);
        }
      }
      gst_object_unref(bus);
    }

    return got_eos_message;
  }

  /**
   * @brief SetUp method for each test case
   */
  void SetUp() override {
    silent = DEFAULT_SILENT;
    throttle = DEFAULT_THROTTLE;
    source_num_buffers = DEFAULT_SOURCE_NUM_BUFFERS;
    source_framerate = const_cast<char *>(DEFAULT_SOURCE_FRAMERATE.c_str());
    target_framerate = const_cast<char *>(DEFAULT_TARGET_FRAMERATE.c_str());
    mode = TENSOR_RATE_MODE_PASSTHROUGH;
  }

  /**
   * @brief TearDown method for each test case
   */
  void TearDown() override {
    gst_object_unref (rate);
    gst_object_unref (pipeline);
  }

  /**
   * @brief Get the rate element in the pipeline.
   * @return GstElement* the rate element
   */
  GstElement* getRateElem() {
    if (!rate)
      rate = gst_bin_get_by_name (GST_BIN (pipeline), "rate");

    return rate;
  }

  /**
   * @brief Make the pipeline description for each mode and construct the pipeline element.
   * @return @c TRUE if success. Otherwise FALSE.
   */
  gboolean setupPipeline() {
    g_autofree gchar *str_pipeline = nullptr;

    switch (mode) {
      case TENSOR_RATE_MODE_PASSTHROUGH:
        str_pipeline = g_strdup_printf (
          "videotestsrc num-buffers=%u ! video/x-raw,framerate=%s ! tensor_converter ! "
          "tensor_rate name=rate framerate=%s throttle=FALSE silent=%s ! fakesink",
          source_num_buffers,
          source_framerate,
          source_framerate,
          silent ? "TRUE" : "FALSE");
        break;
      
      case TENSOR_RATE_MODE_NO_THROTTLE:
        str_pipeline = g_strdup_printf (
          "videotestsrc num-buffers=%u ! video/x-raw,framerate=%s ! tensor_converter ! "
          "tensor_rate name=rate framerate=%s throttle=FALSE silent=%s ! fakesink",
          source_num_buffers,
          source_framerate,
          target_framerate,
          silent ? "TRUE" : "FALSE");
        break;

      case TENSOR_RATE_MODE_THROTTLE:
        str_pipeline = g_strdup_printf (
          "videotestsrc num-buffers=%u ! video/x-raw,framerate=%s ! tensor_converter ! "
          "tensor_filter framework=%s model=%s ! "
          "tensor_rate name=rate framerate=%s throttle=TRUE silent=%s ! fakesink",
          source_num_buffers,
          source_framerate,
          framework,
          modelpath,
          target_framerate,
          silent ? "TRUE" : "FALSE");
        break;

      default:
        return FALSE;
    }
    this->pipeline = gst_parse_launch (str_pipeline, NULL);
  
    return pipeline != NULL ? TRUE : FALSE;
  }
};

/**
 * @brief Test tensor_rate get default property
 */
TEST_F (NNSRateTest, getPropertyDefault)
{
  gboolean silent, throttle;
  guint64 in, out, dup, drop;
  g_autofree gchar *framerate = nullptr;

  ASSERT_TRUE (setupPipeline());

  GstElement *rate = getRateElem();
  ASSERT_TRUE (rate != NULL);

  g_object_get (rate, "silent", &silent, NULL);
  EXPECT_TRUE (silent);

  g_object_get (rate, "throttle", &throttle, NULL);
  EXPECT_FALSE (throttle);

  g_object_get (rate, "framerate", &framerate, NULL);
  EXPECT_STREQ (framerate, DEFAULT_SOURCE_FRAMERATE.c_str());

  g_object_get (rate, "in", &in, NULL);
  EXPECT_EQ (in, DEFAULT_IN);

  g_object_get (rate, "out", &out, NULL);
  EXPECT_EQ (out, DEFAULT_OUT);

  g_object_get (rate, "duplicate", &dup, NULL);
  EXPECT_EQ (dup, DEFAULT_DUP);

  g_object_get (rate, "drop", &drop, NULL);
  EXPECT_EQ (drop, DEFAULT_DROP);
}

/**
 * @brief Test tensor_rate set property
 */
TEST_F (NNSRateTest, setProperty)
{
  gboolean silent, throttle;
  g_autofree gchar *framerate = nullptr;

  ASSERT_TRUE (setupPipeline());

  GstElement *rate = getRateElem();
  ASSERT_TRUE (rate != NULL);

  g_object_set (rate, "silent", (gboolean) FALSE, NULL);
  g_object_get (rate, "silent", &silent, NULL);
  EXPECT_FALSE (silent);

  g_object_set (rate, "throttle", (gboolean) TRUE, NULL);
  g_object_get (rate, "throttle", &throttle, NULL);
  EXPECT_TRUE (throttle);

  g_object_set (rate, "framerate", "15/1", NULL);
  g_object_get (rate, "framerate", &framerate, NULL);
  EXPECT_STREQ ("15/1", framerate);
}

/**
 * @brief Test tensor_rate set property stats (negative)
 */
TEST_F (NNSRateTest, setProperyStats_n)
{
  guint64 in, out, dup, drop;

  ASSERT_TRUE (setupPipeline());

  GstElement *rate = getRateElem();
  ASSERT_TRUE (rate != NULL);

  g_object_set (rate, "in", 10, NULL);
  g_object_get (rate, "in", &in, NULL);
  EXPECT_EQ (in, DEFAULT_IN);

  g_object_set (rate, "out", 10, NULL);
  g_object_get (rate, "out", &out, NULL);
  EXPECT_EQ (out, DEFAULT_OUT);

  g_object_set (rate, "duplicate", 10, NULL);
  g_object_get (rate, "duplicate", &dup, NULL);
  EXPECT_EQ (dup, DEFAULT_DUP);

  g_object_set (rate, "drop", 10, NULL);
  g_object_get (rate, "drop", &drop, NULL);
  EXPECT_EQ (drop, DEFAULT_DROP);
}

/**
 * @brief Test tensor_rate set invalide framerate (negative)
 */
TEST_F (NNSRateTest, setProperyInvalidFramerate_n)
{
  gchar *framerate;

  ASSERT_TRUE (setupPipeline());

  GstElement *rate = getRateElem();
  ASSERT_TRUE (rate != NULL);

  g_object_set (rate, "framerate", "ASDF", NULL);
  g_object_get (rate, "framerate", &framerate, NULL);
  EXPECT_STREQ (framerate, DEFAULT_SOURCE_FRAMERATE.c_str());
  g_free (framerate);

  g_object_set (rate, "framerate", "10/0", NULL);
  g_object_get (rate, "framerate", &framerate, NULL);
  EXPECT_STREQ (framerate, DEFAULT_SOURCE_FRAMERATE.c_str());
  g_free (framerate);
}

/**
 * @brief Test tensor_rate with passthrough mode
 */
TEST_F (NNSRateTest, passthrough)
{
  guint64 in, out, dup, drop;

  ASSERT_TRUE (setupPipeline());

  GstElement *rate = getRateElem();
  ASSERT_TRUE (rate != NULL);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING,
    UNITTEST_STATECHANGE_TIMEOUT), 0);
  
  EXPECT_TRUE (NNSRateTest::wait_pipeline_eos (pipeline));

  g_object_get (rate, "in", &in, NULL);
  g_object_get (rate, "out", &out, NULL);
  g_object_get (rate, "duplicate", &dup, NULL);
  g_object_get (rate, "drop", &drop, NULL);

  EXPECT_EQ (in, source_num_buffers);
  EXPECT_EQ (out, source_num_buffers);
  EXPECT_EQ (0U, dup);
  EXPECT_EQ (0U, drop);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL,
        UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief Test tensor_rate with no-throttling mode
 */
TEST_F (NNSRateTest, noThrottling)
{
  guint64 in, out, dup, drop;

  mode = NNSRateTest::TENSOR_RATE_MODE_NO_THROTTLE;
  target_framerate = g_strdup ("15/1");
  ASSERT_TRUE (setupPipeline());

  GstElement *rate = getRateElem();
  ASSERT_TRUE (rate != NULL);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING,
        UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_TRUE (NNSRateTest::wait_pipeline_eos (pipeline));

  g_object_get (rate, "in", &in, NULL);
  g_object_get (rate, "out", &out, NULL);
  g_object_get (rate, "duplicate", &dup, NULL);
  g_object_get (rate, "drop", &drop, NULL);

  EXPECT_EQ (in, source_num_buffers);
  EXPECT_EQ (0U, dup);

  /** we don't expect the exact values */
  EXPECT_GE (out, (guint64) (((float) source_num_buffers / 2.0) * 0.95));
  EXPECT_LE (out, (guint64) (((float) source_num_buffers / 2.0) * 1.05));

  EXPECT_GE (drop, (guint64) (((float) source_num_buffers / 2.0) * 0.95));
  EXPECT_LE (drop, (guint64) (((float) source_num_buffers / 2.0) * 1.05));

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL,
        UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_free (target_framerate);
}

/**
 * @brief Test tensor_rate with throttling mode
 */
TEST_F (NNSRateTest, throttling)
{
  guint64 in, out, dup, drop;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  if (root_path == NULL)
    root_path = "..";

  gchar *model_file = g_build_filename (root_path, "build", "tests",
      "nnstreamer_example", "libnnstreamer_customfilter_passthrough.so", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  framework = g_strdup ("custom");
  modelpath = model_file;
  mode = NNSRateTest::TENSOR_RATE_MODE_THROTTLE;
  target_framerate = g_strdup ("15/1");
  ASSERT_TRUE (setupPipeline());

  GstElement *rate = getRateElem();
  ASSERT_TRUE (rate != NULL);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING,
        UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (NNSRateTest::wait_pipeline_eos (pipeline));

  g_object_get (rate, "in", &in, NULL);
  g_object_get (rate, "out", &out, NULL);
  g_object_get (rate, "duplicate", &dup, NULL);
  g_object_get (rate, "drop", &drop, NULL);

  /** we don't expect the exact values */
  EXPECT_GE (in, (guint64) (((float) source_num_buffers / 2.0) * 0.95));
  EXPECT_LE (in, (guint64) (((float) source_num_buffers / 2.0) * 1.05));
  EXPECT_GE (out, (guint64) (((float) source_num_buffers / 2.0) * 0.95));
  EXPECT_LE (out, (guint64) (((float) source_num_buffers / 2.0) * 1.05));

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL,
        UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_free (target_framerate);
  g_free (framework);
}

/**
 * @brief gtest main
 */
int main (int argc, char **argv)
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
