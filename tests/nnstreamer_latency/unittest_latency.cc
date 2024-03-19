/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer Unit tests for latency
 * Copyright 2022 NXP
 */
/**
 * @file    unittest_latency.cc
 * @date    18 Oct 2022
 * @brief   Unit tests for latency
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Julien Vuillaumier <julien.vuillaumier@nxp.com>
 * @bug     No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <unittest_util.h>

#include <nnstreamer_conf.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>


/**
 * @brief Test Fixture class for latency
 *        It creates a pipeline with configurable tensor filter latency based
 *        on framecounter custom filter that has configurable sleep capability.
 */

class NNSLatencyTest : public testing::Test
{
  public:
  static const gchar *custom_dir;

  protected:
  static const guint64 FILTER_LATENCY_DURATION_MS = 500UL;
  static const guint64 FILTER_LATENCY_CONVERGENCE_MS = 5 * FILTER_LATENCY_DURATION_MS;
  static const guint64 PIPELINE_LATENCY_MARGIN_MS = 100UL;
  static const guint64 PIPELINE_STOP_DURATION_MS = 3000UL;
  static const gchar *const SINK_NAME;
  static const gchar *const CUSTOM_MODEL_NAME;

  gboolean latency_report;
  guint64 filter_latency_ms;
  GstElement *pipeline;


  /**
   * @brief SetUp method for each test case
   */
  void SetUp () override
  {
    latency_report = FALSE;
    filter_latency_ms = 0UL;
  }

  /**
   * @brief TearDown method for each test case
   */
  void TearDown () override
  {
    gst_object_unref (pipeline);
    pipeline = nullptr;
  }

  /**
   * @brief Build the pipeline according to parameters
   * @return @gboolean TRUE if success. Otherwise FALSE.
   */
  gboolean setupPipeline ()
  {

    const gchar *latency_str = latency_report ? (const gchar *) "latency_report=1" : "";
    g_autofree const gchar *filter_delay_str
        = filter_latency_ms ?
              g_strdup_printf ("custom=delay-%" G_GUINT64_FORMAT, filter_latency_ms) :
              g_strdup ("");
    g_autofree const gchar *custom_filter_path = nullptr;

    if (custom_dir) {
      custom_filter_path = g_strdup_printf ("%s/%s%s", custom_dir,
          CUSTOM_MODEL_NAME, NNSTREAMER_SO_FILE_EXTENSION);
    } else {
      const gchar *path_from_conf
          = nnsconf_get_fullpath (CUSTOM_MODEL_NAME, NNSCONF_PATH_CUSTOM_FILTERS);
      if (path_from_conf) {
        custom_filter_path = g_strdup (path_from_conf);
      } else {
        custom_filter_path = g_strdup_printf ("%s/%s%s", "./tests/nnstreamer_example",
            CUSTOM_MODEL_NAME, NNSTREAMER_SO_FILE_EXTENSION);
      }
    }

    if (!nnsconf_validate_file (NNSCONF_PATH_CUSTOM_FILTERS, custom_filter_path)) {
      g_warning ("Could not find custom filter %s", custom_filter_path);
      return FALSE;
    }

    g_autofree gchar *pipeline_str
        = g_strdup_printf ("videotestsrc is-live=true do-timestamp=true ! "
                           "video/x-raw,format=RGB,width=16,height=16,framerate=50/1 ! "
                           "queue leaky=2 max-size-buffers=1 ! "
                           "videoconvert ! "
                           "tensor_converter ! "
                           "tensor_filter framework=custom "
                           "model=%s "
                           "%s %s ! "
                           "fakesink name=%s sync=true",
            custom_filter_path, latency_str, filter_delay_str, SINK_NAME);

    g_printf ("pipeline: %s\n", pipeline_str);
    pipeline = gst_parse_launch (pipeline_str, nullptr);

    return pipeline != nullptr ? TRUE : FALSE;
  }

  /**
   * @brief Report pipeline latency
   * @return @gboolean TRUE if success. Otherwise FALSE.
   */
  gboolean pipelineLatency (gboolean *live, GstClockTime *min, GstClockTime *max)
  {
    GstElement *sink = nullptr;
    GstQuery *query = nullptr;
    gboolean ret;

    sink = gst_bin_get_by_name (GST_BIN (pipeline), NNSLatencyTest::SINK_NAME);
    g_return_val_if_fail (GST_IS_ELEMENT (sink), FALSE);

    query = gst_query_new_latency ();

    ret = gst_element_query (sink, query);
    if (ret)
      gst_query_parse_latency (query, live, min, max);

    gst_object_unref (sink);
    gst_query_unref (query);

    return ret;
  }

  /**
   * @brief Start pipeline
   * @return @gboolean TRUE if success. Otherwise FALSE.
   */
  gboolean startPipeline ()
  {
    if (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT))
      return FALSE;

    /* Let latency distribution to stabilize */
    gulong duration_us = FILTER_LATENCY_CONVERGENCE_MS * 1000UL;
    g_usleep (duration_us);
    return TRUE;
  }

  /**
   * @brief Stop pipeline
   * @return @gboolean TRUE if success. Otherwise FALSE.
   */
  gboolean stopPipeline ()
  {
    if (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT))
      return FALSE;

    /* Let pipeline stop */
    gulong duration_us = PIPELINE_STOP_DURATION_MS * 1000UL;
    g_usleep (duration_us);
    return TRUE;
  }
};

const gchar *NNSLatencyTest::custom_dir = nullptr;
const gchar *const NNSLatencyTest::SINK_NAME = "fsink";
const gchar *const NNSLatencyTest::CUSTOM_MODEL_NAME = "libnnscustom_framecounter";


/**
 * @brief Test pipeline latency with no report from tensor filter
 *        Sink should report a pipeline latency (min) that is small
 *        compared to the actual long tensor filter latency.
 */
TEST_F (NNSLatencyTest, noTensorFilterLatencyReport)
{
  GstClockTime min, max;
  gboolean live;

  latency_report = FALSE;
  filter_latency_ms = FILTER_LATENCY_DURATION_MS;

  ASSERT_TRUE (setupPipeline ());

  ASSERT_TRUE (startPipeline ());

  EXPECT_TRUE (pipelineLatency (&live, &min, &max));

  ASSERT_TRUE (stopPipeline ());

  guint64 min_ms = min / GST_MSECOND;
  guint64 threshold_ms = PIPELINE_LATENCY_MARGIN_MS;

  g_printf ("min_ms:%" G_GUINT64_FORMAT " threshold:%" G_GUINT64_FORMAT "\n",
      min_ms, threshold_ms);
  EXPECT_LE (min_ms, threshold_ms);
}

/**
 * @brief Test pipeline latency with report from tensor filter
 *        Sink should report a pipeline latency (min) that is longer
 *        than the tensor filter latency as it has been taken into account.
 */
TEST_F (NNSLatencyTest, TensorFilterLatencyReport)
{
  GstClockTime min, max;
  gboolean live;

  latency_report = TRUE;
  filter_latency_ms = FILTER_LATENCY_DURATION_MS;

  ASSERT_TRUE (setupPipeline ());

  ASSERT_TRUE (startPipeline ());

  EXPECT_TRUE (pipelineLatency (&live, &min, &max));

  ASSERT_TRUE (stopPipeline ());

  guint64 min_ms = min / GST_MSECOND;
  guint64 threshold_min = FILTER_LATENCY_DURATION_MS;
  guint64 threshold_max = FILTER_LATENCY_DURATION_MS + PIPELINE_LATENCY_MARGIN_MS;

  g_printf ("min_ms:%" G_GUINT64_FORMAT " threshold_min:%" G_GUINT64_FORMAT
            " threshold_max:%" G_GUINT64_FORMAT "\n",
      min_ms, threshold_min, threshold_max);
  EXPECT_GE (min_ms, threshold_min);
  EXPECT_LE (min_ms, threshold_max);
}


/**
 * @brief gtest main
 */
int
main (int argc, char **argv)
{
  int result = -1;
  const GOptionEntry main_entries[]
      = { { "customdir", 'd', G_OPTION_FLAG_NONE, G_OPTION_ARG_STRING,
              &NNSLatencyTest::custom_dir, "A directory containing custom sub-plugins to use this test",
              "build/tests/nnstreamer_example" },
          { NULL } };

  GError *error = NULL;
  GOptionContext *optionctx;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  optionctx = g_option_context_new (NULL);
  g_option_context_add_main_entries (optionctx, main_entries, NULL);

  if (!g_option_context_parse (optionctx, &argc, &argv, &error)) {
    g_print ("option parsing failed: %s\n", error->message);
    g_clear_error (&error);
  }

  g_option_context_free (optionctx);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
