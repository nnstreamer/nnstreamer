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

/**
 * @brief Test for mqttsink with wrong URL
 */
TEST (testMqttSink, sink_push_wrongurl_n)
{
  const static gsize data_size = 1024;
  GstHarness *h = gst_harness_new ("mqttsink");
  GstBuffer *in_buf;
  GstFlowReturn ret;

  ASSERT_TRUE (h != NULL);

  g_object_set (h->element, "host", "tcp:://0.0.0.0", "port", "0",
      "enable-last-sample", (gboolean) FALSE, NULL);
  in_buf = gst_harness_create_buffer (h, data_size);
  ret = gst_harness_push (h, in_buf);

  EXPECT_EQ (ret, GST_FLOW_ERROR);

  gst_harness_teardown (h);
}

/**
 * @brief Test for mqttsrc with wrong URL
 */
TEST (testMqttSrc, src_pull_wrongurl_n)
{
  GstHarness *h = gst_harness_new ("mqttsrc");
  GstBuffer *out_buffer;

  ASSERT_TRUE (h != NULL);

  g_object_set (h->element, "host", "tcp:://0.0.0.0", "port", "0", NULL);

  out_buffer = gst_harness_try_pull (h);
  EXPECT_TRUE (out_buffer == NULL);

  gst_harness_teardown (h);
}

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
