/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_mqtt.cc
 * @date        20 May 2021
 * @brief       Unit test for GStreamer MQTT elements
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Wook Song <wook16.song@samsung.com>
 * @bug         No known bugs
 */

#include <glib.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <gtest/gtest.h>
#include <unittest_util.h>

/**
 * @brief Test for mqttsink with wrong URL
 */
TEST (testMqttSink, sinkPushWrongurl_n)
{
  gchar *pipeline;
  GstElement *gstpipe;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc is-live=true ! video/x-raw,format=RGB,width=640,height=480,framerate=5/1 ! mqttsink host=invalid_host pub-topic=test/videotestsrc");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for mqttsink with invalid port
 */
TEST (testMqttSink, sinkPushWrongPort_n)
{
  gchar *pipeline;
  GstElement *gstpipe;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc is-live=true ! video/x-raw,format=RGB,width=640,height=480,framerate=5/1 ! mqttsink port=-1 pub-topic=test/videotestsrc");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

#ifdef __MQTT_BROKER_ENABLED__
/**
 * @brief Test pushing EOS event to mqttsink
 */
TEST (testMqttSink, sinkPushEvent)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  gboolean ret;
  GstEvent *evt;

  ASSERT_TRUE (h != NULL);

  evt = gst_event_new_eos ();
  ASSERT_TRUE (evt != NULL);

  ret = gst_harness_push_event (h, evt);

  EXPECT_EQ (ret, TRUE);

  gst_harness_teardown (h);
}

/**
 * @brief Test for mqttsrc with wrong URL
 */
TEST (testMqttSrc, srcPullWrongurl_n)
{
  GstHarness *h = gst_harness_new ("mqttsrc");
  GstBuffer *out_buffer;

  ASSERT_TRUE (h != NULL);

  g_object_set (h->element, "host", "tcp:://0.0.0.0", "port", "0", NULL);

  out_buffer = gst_harness_try_pull (h);
  EXPECT_TRUE (out_buffer == NULL);

  gst_harness_teardown (h);
}

#endif /* #ifdef __MQTT_BROKER_ENABLED__ */

/**
 * @brief Main GTest
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
