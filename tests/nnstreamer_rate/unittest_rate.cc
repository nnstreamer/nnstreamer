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

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api.h>

#include <unittest_util.h>

#define NNS_TENSOR_RATE_NAME "tensor_rate"

/**
 * @brief Test tensor_rate existence.
 */
TEST (nnstreamer_rate, check_existence)
{
  GstElementFactory *factory;

  factory = gst_element_factory_find (NNS_TENSOR_RATE_NAME);
  EXPECT_TRUE (factory != NULL);
  gst_object_unref (factory);
}

/**
 * @brief Test tensor_rate existence (negative).
 */
TEST (nnstreamer_rate, check_existence_n)
{
  GstElementFactory *factory;
  gchar *name;

  name = g_strconcat (NNS_TENSOR_RATE_NAME, "_dummy", NULL);
  factory = gst_element_factory_find (name);
  EXPECT_TRUE (factory == NULL);
  g_free (name);
}

/**
 * @brief Test modes.
 */
typedef enum {
  TENSOR_RATE_MODE_PASSTHROUGH = 0,
  TENSOR_RATE_MODE_NO_THROTTLE,
  TENSOR_RATE_MODE_THROTTLE,
} TestMode;

/**
 * @brief Test options.
 */
typedef struct
{
  guint64 in, out, dup, drop;
  gboolean silent, throttle;
  guint source_num_buffers;
  gchar * source_framerate;
  gchar * target_framerate;
  gchar * framework;
  gchar * model_file;
  TestMode mode;
} TestOption;

/**
 * @brief Data structure for test.
 */
typedef struct
{
  GMainLoop *loop;  /**< main event loop */
  GstElement *pipeline; /**< gst pipeline for test */
} TestData;

/**
 * @brief Data for pipeline and test result.
 */
static TestData test_data;

static gboolean DEFAULT_SILENT = TRUE;
static gboolean DEFAULT_THROTTLE = FALSE;
static guint DEFAULT_SOURCE_NUM_BUFFERS = 300;
static gchar DEFAULT_SOURCE_FRAMERATE[] = "30/1";
static gchar DEFAULT_TARGET_FRAMERATE[] = "0/1";

static guint64 DEFAULT_IN = 0;
static guint64 DEFAULT_OUT = 0;
static guint64 DEFAULT_DUP = 0;
static guint64 DEFAULT_DROP = 0;

/**
 * @brief Prepare test pipeline
 */
static gboolean
_setup_pipeline (TestOption &option)
{
  gchar *str_pipeline;

  switch (option.mode) {
    case TENSOR_RATE_MODE_PASSTHROUGH:
      str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%u ! video/x-raw,framerate=%s ! tensor_converter ! "
        "tensor_rate name=rate framerate=%s throttle=FALSE silent=%s ! fakesink",
        option.source_num_buffers,
        option.source_framerate,
        option.source_framerate,
        option.silent ? "TRUE" : "FALSE"
        );
      break;
    case TENSOR_RATE_MODE_NO_THROTTLE:
      str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%u ! video/x-raw,framerate=%s ! tensor_converter ! "
        "tensor_rate name=rate framerate=%s throttle=FALSE silent=%s ! fakesink",
        option.source_num_buffers,
        option.source_framerate,
        option.target_framerate,
        option.silent ? "TRUE" : "FALSE"
        );
      break;
    case TENSOR_RATE_MODE_THROTTLE:
      str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%u ! video/x-raw,framerate=%s ! tensor_converter ! "
        "tensor_filter framework=%s model=%s ! "
        "tensor_rate name=rate framerate=%s throttle=TRUE silent=%s ! fakesink",
        option.source_num_buffers,
        option.source_framerate,
        option.framework,
        option.model_file,
        option.target_framerate,
        option.silent ? "TRUE" : "FALSE"
        );
      break;
    default:
      return FALSE;
  }

  test_data.pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);

  return test_data.pipeline != NULL ? TRUE : FALSE;
}

/**
 * @brief set default option
 */
static void
_set_default_option (TestOption &option)
{
  option.silent = DEFAULT_SILENT;
  option.throttle = DEFAULT_THROTTLE;
  option.source_num_buffers = DEFAULT_SOURCE_NUM_BUFFERS;
  option.source_framerate = DEFAULT_SOURCE_FRAMERATE;
  option.target_framerate = DEFAULT_TARGET_FRAMERATE;
  option.mode = TENSOR_RATE_MODE_PASSTHROUGH;
}

/**
 * @brief Test tensor_rate get default property
 */
TEST (nnstreamer_rate, get_property_default)
{
  TestOption option;
  GstElement *rate;
  gboolean silent, throttle;
  guint64 in, out, dup, drop;
  gchar *framerate;

  _set_default_option (option);
  option.mode = TENSOR_RATE_MODE_PASSTHROUGH;

  ASSERT_TRUE (_setup_pipeline (option));

  rate = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "rate");
  ASSERT_TRUE (rate != NULL);

  g_object_get (rate, "silent", &silent, NULL);
  EXPECT_TRUE (silent);

  g_object_get (rate, "throttle", &throttle, NULL);
  EXPECT_FALSE (throttle);

  g_object_get (rate, "framerate", &framerate, NULL);
  EXPECT_STREQ (framerate, DEFAULT_SOURCE_FRAMERATE);
  g_free (framerate);

  g_object_get (rate, "in", &in, NULL);
  EXPECT_EQ (in, DEFAULT_IN);

  g_object_get (rate, "out", &out, NULL);
  EXPECT_EQ (out, DEFAULT_OUT);

  g_object_get (rate, "duplicate", &dup, NULL);
  EXPECT_EQ (dup, DEFAULT_DUP);

  g_object_get (rate, "drop", &drop, NULL);
  EXPECT_EQ (drop, DEFAULT_DROP);

  gst_object_unref (rate);
  gst_object_unref (test_data.pipeline);
}

/**
 * @brief Test tensor_rate set property
 */
TEST (nnstreamer_rate, set_property)
{
  TestOption option;
  GstElement *rate;
  gboolean silent, throttle;
  gchar *framerate;

  _set_default_option (option);
  option.mode = TENSOR_RATE_MODE_PASSTHROUGH;

  ASSERT_TRUE (_setup_pipeline (option));

  rate = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "rate");
  ASSERT_TRUE (rate != NULL);

  g_object_set (rate, "silent", (gboolean) FALSE, NULL);
  g_object_get (rate, "silent", &silent, NULL);
  EXPECT_FALSE (silent);

  g_object_set (rate, "throttle", (gboolean) TRUE, NULL);
  g_object_get (rate, "throttle", &throttle, NULL);
  EXPECT_TRUE (throttle);

  g_object_set (rate, "framerate", "15/1", NULL);
  g_object_get (rate, "framerate", &framerate, NULL);
  EXPECT_STREQ (framerate, "15/1");
  g_free (framerate);

  gst_object_unref (rate);
  gst_object_unref (test_data.pipeline);
}

/**
 * @brief Test tensor_rate set property stats (negative)
 */
TEST (nnstreamer_rate, set_propery_stats_n)
{
  TestOption option;
  GstElement *rate;
  guint64 in, out, dup, drop;

  _set_default_option (option);
  option.mode = TENSOR_RATE_MODE_PASSTHROUGH;

  ASSERT_TRUE (_setup_pipeline (option));

  rate = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "rate");
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

  gst_object_unref (rate);
  gst_object_unref (test_data.pipeline);
}

/**
 * @brief Test tensor_rate set invalide framerate (negative)
 */
TEST (nnstreamer_rate, set_propery_invalid_framerate_n)
{
  TestOption option;
  GstElement *rate;
  gchar *framerate;

  _set_default_option (option);
  option.mode = TENSOR_RATE_MODE_PASSTHROUGH;

  ASSERT_TRUE (_setup_pipeline (option));

  rate = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "rate");
  ASSERT_TRUE (rate != NULL);

  g_object_set (rate, "framerate", "ASDF", NULL);
  g_object_get (rate, "framerate", &framerate, NULL);
  EXPECT_STREQ (framerate, DEFAULT_SOURCE_FRAMERATE);
  g_free (framerate);

  g_object_set (rate, "framerate", "10/0", NULL);
  g_object_get (rate, "framerate", &framerate, NULL);
  EXPECT_STREQ (framerate, DEFAULT_SOURCE_FRAMERATE);
  g_free (framerate);

  gst_object_unref (rate);
  gst_object_unref (test_data.pipeline);
}

/**
 * @brief wait until the pipeline gets the eos message.
 */
static gboolean wait_pipeline_eos (GstElement *pipeline)
{
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
 * @brief Test tensor_rate with passthrough mode
 */
TEST (nnstreamer_rate, passthrough)
{
  TestOption option;
  GstElement *rate;
  guint64 in, out, dup, drop;

  _set_default_option (option);
  option.mode = TENSOR_RATE_MODE_PASSTHROUGH;

  ASSERT_TRUE (_setup_pipeline (option));

  rate = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "rate");
  ASSERT_TRUE (rate != NULL);

  EXPECT_EQ (setPipelineStateSync (test_data.pipeline, GST_STATE_PLAYING,
        UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_TRUE (wait_pipeline_eos (test_data.pipeline));

  g_object_get (rate, "in", &in, NULL);
  g_object_get (rate, "out", &out, NULL);
  g_object_get (rate, "duplicate", &dup, NULL);
  g_object_get (rate, "drop", &drop, NULL);

  EXPECT_EQ (in, option.source_num_buffers);
  EXPECT_EQ (out, option.source_num_buffers);
  EXPECT_EQ (dup, 0);
  EXPECT_EQ (drop, 0);

  EXPECT_EQ (setPipelineStateSync (test_data.pipeline, GST_STATE_NULL,
        UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (rate);
  gst_object_unref (test_data.pipeline);
}

/**
 * @brief Test tensor_rate with no-throttling mode
 */
TEST (nnstreamer_rate, no_throttling)
{
  TestOption option;
  GstElement *rate;
  guint64 in, out, dup, drop;

  _set_default_option (option);
  option.mode = TENSOR_RATE_MODE_NO_THROTTLE;
  option.target_framerate = g_strdup ("15/1");

  ASSERT_TRUE (_setup_pipeline (option));

  rate = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "rate");
  ASSERT_TRUE (rate != NULL);

  EXPECT_EQ (setPipelineStateSync (test_data.pipeline, GST_STATE_PLAYING,
        UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_TRUE (wait_pipeline_eos (test_data.pipeline));

  g_object_get (rate, "in", &in, NULL);
  g_object_get (rate, "out", &out, NULL);
  g_object_get (rate, "duplicate", &dup, NULL);
  g_object_get (rate, "drop", &drop, NULL);

  EXPECT_EQ (in, option.source_num_buffers);
  EXPECT_EQ (dup, 0);

  /** we don't expect the exact values */
  EXPECT_GE (out, (guint64) ((option.source_num_buffers / 2) * 0.95));
  EXPECT_LE (out, (guint64) ((option.source_num_buffers / 2) * 1.05));

  EXPECT_GE (drop, (guint64) ((option.source_num_buffers / 2) * 0.95));
  EXPECT_LE (drop, (guint64) ((option.source_num_buffers / 2) * 1.05));

  EXPECT_EQ (setPipelineStateSync (test_data.pipeline, GST_STATE_NULL,
        UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_free (option.target_framerate);

  gst_object_unref (rate);
  gst_object_unref (test_data.pipeline);
}

/**
 * @brief Test tensor_rate with throttling mode
 */
TEST (nnstreamer_rate, throttling)
{
  TestOption option;
  GstElement *rate;
  guint64 in, out, dup, drop;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  if (root_path == NULL)
    root_path = "..";

  gchar *model_file = g_build_filename (root_path, "build", "nnstreamer_example",
      "libnnstreamer_customfilter_passthrough.so", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  option.framework = g_strdup ("custom");
  option.model_file = model_file;

  _set_default_option (option);
  option.mode = TENSOR_RATE_MODE_THROTTLE;
  option.target_framerate = g_strdup ("15/1");

  ASSERT_TRUE (_setup_pipeline (option));

  rate = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "rate");
  ASSERT_TRUE (rate != NULL);

  EXPECT_EQ (setPipelineStateSync (test_data.pipeline, GST_STATE_PLAYING,
        UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_TRUE (wait_pipeline_eos (test_data.pipeline));

  g_object_get (rate, "in", &in, NULL);
  g_object_get (rate, "out", &out, NULL);
  g_object_get (rate, "duplicate", &dup, NULL);
  g_object_get (rate, "drop", &drop, NULL);

  /** we don't expect the exact values */
  EXPECT_GE (in, (guint64) ((option.source_num_buffers / 2) * 0.95));
  EXPECT_LE (in, (guint64) ((option.source_num_buffers / 2) * 1.05));

  EXPECT_GE (out, (guint64) ((option.source_num_buffers / 2) * 0.95));
  EXPECT_LE (out, (guint64) ((option.source_num_buffers / 2) * 1.05));

  EXPECT_EQ (setPipelineStateSync (test_data.pipeline, GST_STATE_NULL,
        UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_free (option.target_framerate);
  g_free (option.model_file);
  g_free (option.framework);

  gst_object_unref (rate);
  gst_object_unref (test_data.pipeline);
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
