/**
 * @file    unittest_filter_rate_change.cc
 * @date    10 May 2023
 * @brief   Unit test for tensor_filter element with dynamic rate changes
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Torsten Schulz <torsten.schulz@gmail.com>
 * @bug     No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <unittest_util.h>

#include <nnstreamer_plugin_api.h>

#define NNS_TENSOR_RATE_NAME "tensor_rate"

class NNSFilterRateChangeTest;

static GstFlowReturn new_data_cb (
    GstElement *element, GstBuffer *buffer, NNSFilterRateChangeTest *_this);

/**
 * @brief Test Fixture class for a tensor_rate element
 */
class NNSFilterRateChangeTest : public testing::Test
{
  protected:
  guint source_num_buffers;
  const gchar *target_framerate_1;
  const gchar *target_framerate_2;
  const gchar *framework;
  gchar *model_path;

  GstElement *pipeline;
  GstElement *throttle;
  GstElement *rate;
  GstElement *tensorsink;

  guint64 num_samples;

  const guint DEFAULT_SOURCE_NUM_BUFFERS = 300;
  const char *DEFAULT_SOURCE_FRAMERATE = "30/1";

  /**
   * @brief Construct a new NNSFilterRateChangeTest object
   */
  NNSFilterRateChangeTest ()
      : source_num_buffers (0), target_framerate_1 (nullptr),
        target_framerate_2 (nullptr), framework (nullptr), model_path (nullptr),
        pipeline (nullptr), throttle (nullptr), tensorsink (nullptr)
  {
  }

  /**
   * @brief Wait until the EOS message is received or the timeout is expired.
   * @param pipeline target pipeline element to watch.
   * @return @c TRUE if EOS message is received in the timeout period. Otherwise FALSE.
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
          gst_bus_async_signal_func (bus, msg, NULL);
          GstObject *src = GST_MESSAGE_SRC (msg);
          gchar *name = GST_IS_OBJECT (src) ? gst_object_get_name (src) :
                                              g_strdup ("no-name");
          const gchar *type_name = GST_IS_OBJECT (src) ? G_OBJECT_TYPE_NAME (src) : "no-type";
          GError *gerror;
          gchar *debug;

          switch (GST_MESSAGE_TYPE (msg)) {
            case GST_MESSAGE_WARNING:
              gst_message_parse_warning (msg, &gerror, &debug);
              g_info ("WARNING from (%s) %s: %s (debug: %s)", type_name, name,
                  gerror->message, debug);
              g_clear_error (&gerror);
              g_free (debug);
              break;
            case GST_MESSAGE_ERROR:
              gst_message_parse_error (msg, &gerror, &debug);
              g_info ("WARNING from (%s) %s: %s (debug: %s)", type_name, name,
                  gerror->message, debug);
              g_clear_error (&gerror);
              g_free (debug);
              break;
            case GST_MESSAGE_EOS:
              got_eos_message = TRUE;
              break;
            default:
              {
                g_info ("GST_MESSAGE UNKNOWN (0x%02X) for (%s) '%s'",
                    GST_MESSAGE_TYPE (msg), type_name, name);
                /* just be quiet by default */
                break;
              }
          }
          gst_message_unref (msg);
        }
      }
      gst_object_unref (bus);
    }

    return got_eos_message;
  }

  /**
   * @brief SetUp method for each test case
   */
  void SetUp () override
  {
    source_num_buffers = DEFAULT_SOURCE_NUM_BUFFERS;
  }

  /**
   * @brief TearDown method for each test case
   */
  void TearDown () override
  {
    gst_object_unref (throttle);
    gst_object_unref (tensorsink);
    gst_object_unref (rate);
    gst_object_unref (pipeline);
  }

  /**
   * @brief Make the pipeline description for each mode and construct the pipeline element.
   * @return @c TRUE if success. Otherwise FALSE.
   */
  gboolean setupPipeline ()
  {
    g_autofree gchar *str_pipeline = nullptr;

    str_pipeline = g_strdup_printf (
        "videotestsrc num-buffers=%u ! video/x-raw,framerate=%s ! "
        "videorate name=rate ! capsfilter name=throttle caps=video/x-raw,framerate=%s ! "
        "tensor_converter ! tensor_filter framework=%s model=%s ! "
        "tensor_sink emit-signal=true name=testsink",
        source_num_buffers, DEFAULT_SOURCE_FRAMERATE, target_framerate_1,
        framework, model_path);

    this->pipeline = gst_parse_launch (str_pipeline, NULL);
    this->tensorsink = gst_bin_get_by_name (GST_BIN (pipeline), "testsink");
    this->throttle = gst_bin_get_by_name (GST_BIN (pipeline), "throttle");
    this->rate = gst_bin_get_by_name (GST_BIN (pipeline), "rate");

    return pipeline != NULL ? TRUE : FALSE;
  }


  /**
   * @brief Run one pipeline test and change framerate midway
   */
  void runPipeline ()
  {
    const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
    if (root_path == NULL)
      root_path = "..";

    gchar *model_file = g_build_filename (root_path, "build", "tests",
        "nnstreamer_example", "libnnstreamer_customfilter_passthrough.so", NULL);
    ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

    framework = "custom";
    model_path = model_file;
    ASSERT_TRUE (setupPipeline ());

    ASSERT_TRUE (this->pipeline != NULL);
    ASSERT_TRUE (this->throttle != NULL);
    ASSERT_TRUE (this->tensorsink != NULL);
    ASSERT_TRUE (this->rate != NULL);

    this->num_samples = 0;

    g_signal_connect (this->tensorsink, "new-data", G_CALLBACK (new_data_cb), this);

    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
        0);
    EXPECT_TRUE (NNSFilterRateChangeTest::wait_pipeline_eos (pipeline));
  }

  public:
  /**
   * @brief Records a received sample in the tensorsink
   */
  void incrementSampleCounter ()
  {
    this->num_samples++;

    if (this->target_framerate_2 && this->num_samples == this->source_num_buffers / 2) {
      g_autofree gchar *str_caps
          = g_strdup_printf ("video/x-raw,framerate=%s", this->target_framerate_2);
      GstCaps *caps = gst_caps_from_string (str_caps);
      g_object_set (this->throttle, "caps", caps, NULL);
      gst_caps_unref (caps);
    }
  }
};

/**
 * @brief Callback function to count buffers arriving at the appsink
 */

static GstFlowReturn
new_data_cb (GstElement *element, GstBuffer *buffer, NNSFilterRateChangeTest *_this)
{
  _this->incrementSampleCounter ();
  return GST_FLOW_OK;
}

/**
 * @brief Test tensor_filter with passthrough framerate filter
 */
TEST_F (NNSFilterRateChangeTest, passthrough)
{
  guint64 in, out, dup, drop;

  this->target_framerate_1 = DEFAULT_SOURCE_FRAMERATE;
  this->target_framerate_2 = NULL;

  this->runPipeline ();

  g_object_get (rate, "in", &in, NULL);
  g_object_get (rate, "out", &out, NULL);
  g_object_get (rate, "duplicate", &dup, NULL);
  g_object_get (rate, "drop", &drop, NULL);

  /** we don't expect the exact values */
  EXPECT_EQ (in, source_num_buffers);
  EXPECT_EQ (out, source_num_buffers);

  EXPECT_EQ (out, this->num_samples);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief Test tensor_filter with a static framerate throttle
 */
TEST_F (NNSFilterRateChangeTest, static_throttle)
{
  guint64 in, out, dup, drop;

  this->target_framerate_1 = "15/1";
  this->target_framerate_2 = NULL;

  this->runPipeline ();

  g_object_get (rate, "in", &in, NULL);
  g_object_get (rate, "out", &out, NULL);
  g_object_get (rate, "duplicate", &dup, NULL);
  g_object_get (rate, "drop", &drop, NULL);

  /** we don't expect the exact values */
  EXPECT_EQ (in, source_num_buffers);
  // ignore possible rounding error
  EXPECT_GE (out, source_num_buffers / 2 - 1);
  EXPECT_LE (out, source_num_buffers / 2 + 1);

  EXPECT_EQ (out, this->num_samples);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief Test tensor_filter with passthrough framerate filter
 */
TEST_F (NNSFilterRateChangeTest, throttling_dynamic_change_dec)
{
  guint64 in, out, dup, drop;

  this->target_framerate_1 = DEFAULT_SOURCE_FRAMERATE;
  this->target_framerate_2 = "15/1";

  this->runPipeline ();

  g_object_get (rate, "in", &in, NULL);
  g_object_get (rate, "out", &out, NULL);
  g_object_get (rate, "duplicate", &dup, NULL);
  g_object_get (rate, "drop", &drop, NULL);

  /** we don't expect the exact values */
  EXPECT_EQ (in, source_num_buffers);
  // ignore possible rounding error
  EXPECT_GE (out, source_num_buffers / 2 + source_num_buffers / 4 - 1);
  EXPECT_LE (out, source_num_buffers / 2 + source_num_buffers / 4 + 1);

  EXPECT_EQ (out, this->num_samples);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
}

/**
 * @brief gtest main
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
