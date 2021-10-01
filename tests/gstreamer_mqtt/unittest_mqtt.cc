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
TEST (testMqttSink, sinkPushWrongurl_n)
{
  const static gsize data_size = 1024;
  GstHarness *h = gst_harness_new_parse ("mqttsink host=invalid_host");
  GstBuffer *in_buf;
  GstFlowReturn ret;

  ASSERT_TRUE (h != NULL);

  in_buf = gst_harness_create_buffer (h, data_size);
  ret = gst_harness_push (h, in_buf);

  EXPECT_EQ (ret, GST_FLOW_ERROR);

  gst_harness_teardown (h);
}

/**
 * @brief Test for mqttsink with invalid port
 */
TEST (testMqttSink, sinkPushWrongPort_n)
{
  const static gsize data_size = 1024;
  GstHarness *h = gst_harness_new_parse ("mqttsink port=-1");
  GstBuffer *in_buf;
  GstFlowReturn ret;

  ASSERT_TRUE (h != NULL);

  in_buf = gst_harness_create_buffer (h, data_size);
  ret = gst_harness_push (h, in_buf);

  EXPECT_EQ (ret, GST_FLOW_ERROR);

  gst_harness_teardown (h);
}

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

/**
 * @brief Test get/set properties of mqttsink
 */
TEST (testMqttSink, sinkGetSetProperties)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  gchar *sprop = NULL;
  gboolean bprop;
  gint iprop;
  gulong ulprop;

  ASSERT_TRUE (h != NULL);

  /** test the default */
  g_object_get (h->element, "debug", &bprop, NULL);
  EXPECT_FALSE (bprop);

  g_object_set (h->element, "debug", true, NULL);
  g_object_get (h->element, "debug", &bprop, NULL);
  EXPECT_TRUE (bprop);

  g_object_set (h->element, "client-id", "testclientid", NULL);
  g_object_get (h->element, "client-id", &sprop, NULL);
  EXPECT_STREQ (sprop, "testclientid");
  g_free (sprop);

  g_object_set (h->element, "host", "hosttest", NULL);
  g_object_get (h->element, "host", &sprop, NULL);
  EXPECT_STREQ (sprop, "hosttest");
  g_free (sprop);

  g_object_set (h->element, "port", "testport", NULL);
  g_object_get (h->element, "port", &sprop, NULL);
  EXPECT_STREQ (sprop, "testport");
  g_free (sprop);

  g_object_set (h->element, "pub-topic", "testtopic", NULL);
  g_object_get (h->element, "pub-topic", &sprop, NULL);
  EXPECT_STREQ (sprop, "testtopic");
  g_free (sprop);

  g_object_set (h->element, "pub-wait-timeout", 9999UL, NULL);
  g_object_get (h->element, "pub-wait-timeout", &ulprop, NULL);
  EXPECT_EQ (ulprop, 9999UL);

  g_object_set (h->element, "cleansession", false, NULL);
  g_object_get (h->element, "cleansession", &bprop, NULL);
  EXPECT_FALSE (bprop);

  g_object_set (h->element, "keep-alive-interval", 9999, NULL);
  g_object_get (h->element, "keep-alive-interval", &iprop, NULL);
  EXPECT_TRUE (iprop == 9999);

  g_object_set (h->element, "max-buffer-size", 1024UL, NULL);
  g_object_get (h->element, "max-buffer-size", &ulprop, NULL);
  EXPECT_EQ (ulprop, 1024UL);

  g_object_set (h->element, "num-buffers", 10, NULL);
  g_object_get (h->element, "num-buffers", &iprop, NULL);
  EXPECT_TRUE (iprop == 10);

  g_object_set (h->element, "mqtt-qos", 1, NULL);
  g_object_get (h->element, "mqtt-qos", &iprop, NULL);
  EXPECT_TRUE (iprop == 1);

  gst_harness_teardown (h);
}

/**
 * @brief Test get/set properties of mqttsrc
 */
TEST (testMqttSrc, srcGetSetProperties)
{
  GstHarness *h = gst_harness_new ("mqttsrc");
  gchar *sprop = NULL;
  gboolean bprop;
  gint iprop;
  gint64 lprop;

  ASSERT_TRUE (h != NULL);

  /** test the default */
  g_object_get (h->element, "debug", &bprop, NULL);
  EXPECT_FALSE (bprop);

  g_object_set (h->element, "debug", true, NULL);
  g_object_get (h->element, "debug", &bprop, NULL);
  EXPECT_TRUE (bprop);

  g_object_set (h->element, "is-live", false, NULL);
  g_object_get (h->element, "is-live", &bprop, NULL);
  EXPECT_FALSE (bprop);

  g_object_set (h->element, "client-id", "testclientid", NULL);
  g_object_get (h->element, "client-id", &sprop, NULL);
  EXPECT_STREQ (sprop, "testclientid");
  g_free (sprop);

  g_object_set (h->element, "host", "hosttest", NULL);
  g_object_get (h->element, "host", &sprop, NULL);
  EXPECT_STREQ (sprop, "hosttest");
  g_free (sprop);

  g_object_set (h->element, "port", "testport", NULL);
  g_object_get (h->element, "port", &sprop, NULL);
  EXPECT_STREQ (sprop, "testport");
  g_free (sprop);

  g_object_set (h->element, "sub-timeout", G_GINT64_CONSTANT (99999999), NULL);
  g_object_get (h->element, "sub-timeout", &lprop, NULL);
  EXPECT_TRUE (lprop == G_GINT64_CONSTANT (99999999));

  g_object_set (h->element, "sub-topic", "testtopic", NULL);
  g_object_get (h->element, "sub-topic", &sprop, NULL);
  EXPECT_STREQ (sprop, "testtopic");
  g_free (sprop);

  g_object_set (h->element, "cleansession", false, NULL);
  g_object_get (h->element, "cleansession", &bprop, NULL);
  EXPECT_FALSE (bprop);

  g_object_set (h->element, "keep-alive-interval", 9999, NULL);
  g_object_get (h->element, "keep-alive-interval", &iprop, NULL);
  EXPECT_TRUE (iprop == 9999);

  g_object_set (h->element, "mqtt-qos", 1, NULL);
  g_object_get (h->element, "mqtt-qos", &iprop, NULL);
  EXPECT_TRUE (iprop == 1);

  gst_harness_teardown (h);
}

/**
 * @brief Test get/set the invalid properties of mqttsink
 */
TEST (testMqttSink, sinkGetSetProperties_n)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  gint iprop;
  guint64 uprop;

  ASSERT_TRUE (h != NULL);

  g_object_set (h->element, "pub-wait-timeout", 0, NULL);
  g_object_get (h->element, "pub-wait-timeout", &uprop, NULL);
  EXPECT_FALSE (uprop == 0);

  g_object_set (h->element, "keep-alive-interval", 0, NULL);
  g_object_get (h->element, "keep-alive-interval", &iprop, NULL);
  EXPECT_FALSE (iprop == 0);

  g_object_set (h->element, "num-buffers", -10, NULL);
  g_object_get (h->element, "num-buffers", &iprop, NULL);
  EXPECT_FALSE (iprop == -10);

  g_object_set (h->element, "mqtt-qos", -1, NULL);
  g_object_get (h->element, "mqtt-qos", &iprop, NULL);
  EXPECT_FALSE (iprop == -1);

  gst_harness_teardown (h);
}

/**
 * @brief Test get/set the invalid properties of mqttsrc
 */
TEST (testMqttSrc, srcGetSetProperties_n)
{
  GstHarness *h = gst_harness_new ("mqttsrc");
  gint iprop;
  gint64 lprop;

  ASSERT_TRUE (h != NULL);

  g_object_set (h->element, "sub-timeout", G_GINT64_CONSTANT (0), NULL);
  g_object_get (h->element, "sub-timeout", &lprop, NULL);
  EXPECT_FALSE (lprop == G_GINT64_CONSTANT (0));

  g_object_set (h->element, "keep-alive-interval", 0, NULL);
  g_object_get (h->element, "keep-alive-interval", &iprop, NULL);
  EXPECT_FALSE (iprop == 0);

  g_object_set (h->element, "num-buffers", -10, NULL);
  g_object_get (h->element, "num-buffers", &iprop, NULL);
  EXPECT_FALSE (iprop == -10);

  g_object_set (h->element, "mqtt-qos", -1, NULL);
  g_object_get (h->element, "mqtt-qos", &iprop, NULL);
  EXPECT_FALSE (iprop == -1);

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
