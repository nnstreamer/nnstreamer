/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_mqtt_w_helper.cc
 * @date        28 May 2021
 * @brief       Unit test for GStreamer MQTT elements using GstMqttTestHelper
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Wook Song <wook16.song@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/base/gstbasesrc.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>

#include <MQTTAsync.h>
#include <unittest_util.h>

#include <atomic>
#include <future>
#include <string>
#include <thread>

#include "GstMqttTestHelper.hh"
#include "mqttcommon.h"
#include "mqttsink.h"

std::unique_ptr<GstMqttTestHelper> GstMqttTestHelper::mInstance;
std::once_flag GstMqttTestHelper::mOnceFlag;

/** The calls a test can hold open on another thread while it changes a property */
enum call_hold_site { HOLD_NONE, HOLD_CREATE, HOLD_EPOCH, HOLD_SEND };

static GMutex call_hold_lock;
static GCond call_hold_cond;
static call_hold_site call_hold_armed = HOLD_NONE;
static bool call_hold_entered = false;
static bool call_hold_released = false;

/**
 * @brief Keep the calling thread inside the call at @a site until released, if a test armed it
 */
static void
_hold_call (call_hold_site site)
{
  g_mutex_lock (&call_hold_lock);
  if (call_hold_armed == site) {
    call_hold_armed = HOLD_NONE;
    call_hold_entered = true;
    g_cond_broadcast (&call_hold_cond);
    while (!call_hold_released)
      g_cond_wait (&call_hold_cond, &call_hold_lock);
  }
  g_mutex_unlock (&call_hold_lock);
}

/**
 * @brief Arm @a site so that its next call waits for _release_call_hold ()
 */
static void
_arm_call_hold (call_hold_site site)
{
  g_mutex_lock (&call_hold_lock);
  call_hold_armed = site;
  call_hold_entered = false;
  call_hold_released = false;
  g_mutex_unlock (&call_hold_lock);
}

/**
 * @brief Wait until a call is held open at the armed site
 * @return false if no call arrived within 10 seconds
 */
static bool
_wait_call_hold (void)
{
  gint64 deadline = g_get_monotonic_time () + 10 * G_TIME_SPAN_SECOND;
  bool entered;

  g_mutex_lock (&call_hold_lock);
  while (!call_hold_entered) {
    if (!g_cond_wait_until (&call_hold_cond, &call_hold_lock, deadline))
      break;
  }
  entered = call_hold_entered;
  g_mutex_unlock (&call_hold_lock);

  return entered;
}

/**
 * @brief Let the call held open by _hold_call () continue
 */
static void
_release_call_hold (void)
{
  g_mutex_lock (&call_hold_lock);
  call_hold_armed = HOLD_NONE;
  call_hold_released = true;
  g_cond_broadcast (&call_hold_cond);
  g_mutex_unlock (&call_hold_lock);
}

/**
 * @brief A mock function for MQTTAsync_create() in paho-mqtt-c
 */
int
MQTTAsync_create (MQTTAsync *handle, const char *serverURI,
    const char *clientId, int persistence_type, void *persistence_context)
{
  _hold_call (HOLD_CREATE);
  GstMqttTestHelper::getInstance ().recordCreate (serverURI, clientId);

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_connect() in paho-mqtt-c
 */
int
MQTTAsync_connect (MQTTAsync handle, const MQTTAsync_connectOptions *options)
{
  MQTTAsync_successData data;
  void *ctx = GstMqttTestHelper::getInstance ().getContext ();
  auto ret = std::async (std::launch::async, options->onSuccess, ctx, &data);

  GstMqttTestHelper::getInstance ().setIsConnected (true);

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_setCallbacks() in paho-mqtt-c
 */
int
MQTTAsync_setCallbacks (MQTTAsync handle, void *context, MQTTAsync_connectionLost *cl,
    MQTTAsync_messageArrived *ma, MQTTAsync_deliveryComplete *dc)
{
  GstMqttTestHelper::getInstance ().init (context);
  GstMqttTestHelper::getInstance ().setCallbacks (cl, ma, dc);

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_send() in paho-mqtt-c
 */
int
MQTTAsync_send (MQTTAsync handle, const char *destinationName, int payloadlen,
    const void *payload, int qos, int retained, MQTTAsync_responseOptions *response)
{
  MQTTAsync_successData data;
  MQTTAsync_failureData failure_data;
  void *ctx = GstMqttTestHelper::getInstance ().getContext ();
  std::future<void> ret;

  _hold_call (HOLD_SEND);
  GstMqttTestHelper::getInstance ().recordSend (destinationName, payload, payloadlen);

  if (GstMqttTestHelper::getInstance ().getFailSend ()) {
    failure_data.code = -1;
    failure_data.message = "";
    ret = std::async (std::launch::async, response->onFailure, ctx, &failure_data);
    return MQTTASYNC_FAILURE;
  }

  ret = std::async (std::launch::async, response->onSuccess, ctx, &data);

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for int MQTTAsync_isConnected() in paho-mqtt-c
 */
int
MQTTAsync_isConnected (MQTTAsync handle)
{
  return GstMqttTestHelper::getInstance ().getIsConnected ();
}

/**
 * @brief A mock function for MQTTAsync_disconnect() in paho-mqtt-c
 */
int
MQTTAsync_disconnect (MQTTAsync handle, const MQTTAsync_disconnectOptions *options)
{
  MQTTAsync_successData data;
  MQTTAsync_failureData fdata;
  void *ctx;
  std::future<void> ret;

  if (!options)
    return MQTTASYNC_SUCCESS;

  ctx = options->context;
  GstMqttTestHelper::getInstance ().setIsConnected (false);
  if (GstMqttTestHelper::getInstance ().getFailDisconnect ()) {
    fdata.code = -1;
    fdata.message = "";
    ret = std::async (std::launch::async, options->onFailure, ctx, &fdata);

    return MQTTASYNC_FAILURE;
  }

  ret = std::async (std::launch::async, options->onSuccess, ctx, &data);

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_destroy() in paho-mqtt-c
 */
void
MQTTAsync_destroy (MQTTAsync *handle)
{
  return;
}

/**
 * @brief A mock function for MQTTAsync_subscribe() in paho-mqtt-c
 */
int
MQTTAsync_subscribe (MQTTAsync handle, const char *topic, int qos,
    MQTTAsync_responseOptions *response)
{
  MQTTAsync_successData data;
  MQTTAsync_failureData fdata;
  std::future<void> ret;
  void *ctx = response->context;

  if (GstMqttTestHelper::getInstance ().getFailSubscribe ()) {
    fdata.code = -1;
    fdata.message = "";
    ret = std::async (std::launch::async, response->onFailure, ctx, &fdata);
    return MQTTASYNC_FAILURE;
  }

  ret = std::async (std::launch::async, response->onSuccess, ctx, &data);
  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_unsubscribe() in paho-mqtt-c
 */
int
MQTTAsync_unsubscribe (MQTTAsync handle, const char *topic, MQTTAsync_responseOptions *response)
{
  MQTTAsync_successData data;
  MQTTAsync_failureData fdata;
  void *ctx = response->context;
  std::future<void> ret;

  if (GstMqttTestHelper::getInstance ().getFailUnsubscribe ()) {
    fdata.code = -1;
    fdata.message = "";
    ret = std::async (std::launch::async, response->onFailure, ctx, &fdata);
    return MQTTASYNC_FAILURE;
  }

  ret = std::async (std::launch::async, response->onSuccess, ctx, &data);
  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A helper function to fill the timestamp information into the header
 */
static void
_set_ts_gst_mqtt_message_hdr (GstElement *elm, GstMQTTMessageHdr *hdr,
    const GstClockTimeDiff diff_sent, const GstClockTime duration)
{
  GstClockTime base_time;
  GstClockTime cur_time;
  GstClockTimeDiff diff;
  GstClock *clock;

  hdr->base_time_epoch = GST_CLOCK_TIME_NONE;
  clock = gst_test_clock_new ();
  base_time = gst_element_get_base_time (elm) + diff_sent;
  cur_time = gst_clock_get_time (clock);
  gst_object_unref (clock);

  diff = GST_CLOCK_DIFF (base_time, cur_time);
  hdr->base_time_epoch = g_get_real_time () * GST_US_TO_NS_MULTIPLIER - diff;
  hdr->sent_time_epoch = hdr->base_time_epoch + diff_sent;

  hdr->pts = 0;
  hdr->dts = 0;
  hdr->duration = duration;
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

  g_object_set (h->element, "ntp-sync", true, NULL);
  g_object_get (h->element, "ntp-sync", &bprop, NULL);
  EXPECT_TRUE (bprop);

  g_object_set (h->element, "ntp-srvs", "time.google.com:123", NULL);
  g_object_get (h->element, "ntp-srvs", &sprop, NULL);
  EXPECT_STREQ (sprop, "time.google.com:123");
  g_free (sprop);

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
 * @brief Test for mqttsink with GstMqttTestHelper (push a GstBuffer)
 */
TEST (testMqttSinkWithHelper, sinkPush0)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstFlowReturn ret;

  g_object_set (h->element, "debug", true, NULL);
  g_object_set (h->element, "ntp-sync", true, NULL);
  gst_harness_add_src_parse (h, "videotestsrc is-live=1 ! queue", TRUE);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  ret = gst_harness_push_from_src (h);

  EXPECT_EQ (ret, GST_FLOW_OK);

  gst_harness_teardown (h);
}

/**
 * @brief Test for mqttsink with GstMqttTestHelper (Push multiple GstBuffers with num-buffers)
 */
TEST (testMqttSinkWithHelper, sinkPush1)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstFlowReturn ret;
  const gint num_buffers = 10;
  gint i;

  g_object_set (h->element, "num-buffers", num_buffers, NULL);
  g_object_set (h->element, "debug", true, NULL);

  gst_harness_add_src_parse (h, "videotestsrc is-live=1 ! queue", TRUE);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  for (i = 0; i < num_buffers; ++i) {
    ret = gst_harness_push_from_src (h);
    EXPECT_EQ (ret, GST_FLOW_OK);
  }

  gst_harness_teardown (h);
}

/**
 * @brief Test for mqttsink with GstMqttTestHelper (MQTTAsync_send failure case)
 */
TEST (testMqttSinkWithHelper, sinkPush0_n)
{
  const static gsize data_size = 1024;
  GstHarness *h = gst_harness_new ("mqttsink");
  GstBuffer *in_buf;
  GstFlowReturn ret;

  ASSERT_TRUE (h != NULL);

  g_object_set (h->element, "debug", true, NULL);

  in_buf = gst_harness_create_buffer (h, data_size);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().setFailSend (true);
  ret = gst_harness_push (h, in_buf);

  EXPECT_EQ (ret, GST_FLOW_ERROR);
  GstMqttTestHelper::getInstance ().setFailSend (false);

  gst_harness_teardown (h);
}

/**
 * @brief Test for mqttsink with GstMqttTestHelper (MQTTAsync_disconnect failure case)
 */
TEST (testMqttSinkWithHelper, sinkPush1_n)
{
  const static gsize data_size = 1024;
  GstHarness *h = gst_harness_new ("mqttsink");
  GstBuffer *in_buf;
  GstFlowReturn ret;

  ASSERT_TRUE (h != NULL);

  g_object_set (h->element, "debug", true, NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();

  in_buf = gst_harness_create_buffer (h, data_size);
  GstMqttTestHelper::getInstance ().setFailDisconnect (true);
  ret = gst_harness_push (h, in_buf);

  EXPECT_EQ (ret, GST_FLOW_OK);
  GstMqttTestHelper::getInstance ().setFailDisconnect (false);

  gst_harness_teardown (h);
}

/**
 * @brief Test for mqttsink with GstMqttTestHelper (Push an empty buffer)
 */
TEST (testMqttSinkWithHelper, sinkPush2_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstFlowReturn ret;

  h = gst_harness_new ("mqttsink");
  ASSERT_TRUE (h != NULL);

  g_object_set (h->element, "debug", true, NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();

  in_buf = gst_buffer_new ();
  ret = gst_harness_push (h, in_buf);

  EXPECT_EQ (ret, GST_FLOW_ERROR);

  gst_harness_teardown (h);
}

/**
 * @brief Test for mqttsink with GstMqttTestHelper (Push GstBuffers more then num-buffers)
 */
TEST (testMqttSinkWithHelper, sinkPush3_n)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstFlowReturn ret;
  const gint num_buffers = 10;
  gint i;

  g_object_set (h->element, "num-buffers", num_buffers, NULL);
  g_object_set (h->element, "debug", true, NULL);

  gst_harness_add_src_parse (h, "videotestsrc is-live=1 ! queue", TRUE);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  for (i = 0; i < num_buffers; ++i) {
    ret = gst_harness_push_from_src (h);

    EXPECT_EQ (ret, GST_FLOW_OK);
  }

  ret = gst_harness_push_from_src (h);
  EXPECT_NE (ret, GST_FLOW_OK);

  gst_harness_teardown (h);
}

/**
 * @brief Test mqttsink with max-buffer-size fitting every pushed buffer (E4 static buffer reuse)
 */
TEST (testMqttSinkWithHelper, sinkPushMaxBufferSizeSmaller)
{
  const gsize sizes[] = { 256, 16, 128, 256 };
  GstHarness *h = gst_harness_new ("mqttsink");
  GstFlowReturn ret;
  guint i;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();

  g_object_set (h->element, "max-buffer-size", 256UL, NULL);

  for (i = 0; i < G_N_ELEMENTS (sizes); ++i) {
    GstBuffer *in_buf = gst_harness_create_buffer (h, sizes[i]);
    guint8 pattern = (guint8) (0x10 * (i + 1));
    GstMapInfo map;
    gsize j;

    ASSERT_TRUE (gst_buffer_map (in_buf, &map, GST_MAP_WRITE));
    for (j = 0; j < map.size; ++j)
      map.data[j] = (guint8) (pattern + j);
    gst_buffer_unmap (in_buf, &map);

    ret = gst_harness_push (h, in_buf);
    EXPECT_EQ (ret, GST_FLOW_OK);
    EXPECT_EQ (GstMqttTestHelper::getInstance ().getSendCount (), (int) (i + 1));
    EXPECT_EQ ((gsize) GstMqttTestHelper::getInstance ().getLastPayloadLen (),
        GST_MQTT_LEN_MSG_HDR + sizes[i]);

    {
      const std::vector<guint8> &payload
          = GstMqttTestHelper::getInstance ().getLastPayload ();

      ASSERT_EQ (payload.size (), (size_t) (GST_MQTT_LEN_MSG_HDR + sizes[i]));
      for (j = 0; j < sizes[i]; ++j)
        EXPECT_EQ (payload[GST_MQTT_LEN_MSG_HDR + j], (guint8) (pattern + j));
    }
  }

  gst_harness_teardown (h);
}

/**
 * @brief Test the header mqttsink prepends to a message: memory sizes, caps, timestamps and send time
 */
TEST (testMqttSinkWithHelper, sinkPushMessageHeader)
{
  const gsize sizes[] = { 24, 40 };
  GstHarness *h = gst_harness_new ("mqttsink");
  GstMQTTMessageHdr hdr;
  GstBuffer *in_buf;
  GstFlowReturn ret;
  gint64 before, after;
  gsize j, offset;
  guint i;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();

  gst_harness_set_src_caps_str (h, "application/octet-stream");

  in_buf = gst_buffer_new ();
  for (i = 0; i < G_N_ELEMENTS (sizes); ++i) {
    GstMemory *mem = gst_allocator_alloc (NULL, sizes[i], NULL);
    GstMapInfo map;

    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));
    memset (map.data, 0xA0 + i, map.size);
    gst_memory_unmap (mem, &map);
    gst_buffer_append_memory (in_buf, mem);
  }
  GST_BUFFER_PTS (in_buf) = GST_SECOND;
  GST_BUFFER_DTS (in_buf) = 900 * GST_MSECOND;
  GST_BUFFER_DURATION (in_buf) = 33 * GST_MSECOND;

  before = g_get_real_time ();
  ret = gst_harness_push (h, in_buf);
  after = g_get_real_time ();
  EXPECT_EQ (ret, GST_FLOW_OK);
  EXPECT_EQ (GstMqttTestHelper::getInstance ().getSendCount (), 1);

  {
    const std::vector<guint8> &payload
        = GstMqttTestHelper::getInstance ().getLastPayload ();

    ASSERT_EQ (payload.size (), (size_t) (GST_MQTT_LEN_MSG_HDR + sizes[0] + sizes[1]));
    memcpy (&hdr, payload.data (), GST_MQTT_LEN_MSG_HDR);

    EXPECT_EQ (hdr.num_mems, 2U);
    EXPECT_EQ (hdr.size_mems[0], sizes[0]);
    EXPECT_EQ (hdr.size_mems[1], sizes[1]);
    EXPECT_STREQ (hdr.gst_caps_str, "application/octet-stream");
    EXPECT_EQ (hdr.pts, (GstClockTime) GST_SECOND);
    EXPECT_EQ (hdr.dts, (GstClockTime) (900 * GST_MSECOND));
    EXPECT_EQ (hdr.duration, (GstClockTime) (33 * GST_MSECOND));
    EXPECT_GE (hdr.sent_time_epoch, before * GST_US_TO_NS_MULTIPLIER);
    EXPECT_LE (hdr.sent_time_epoch, after * GST_US_TO_NS_MULTIPLIER);

    offset = GST_MQTT_LEN_MSG_HDR;
    for (i = 0; i < G_N_ELEMENTS (sizes); ++i) {
      gsize mismatches = 0;

      for (j = 0; j < sizes[i]; ++j) {
        if (payload[offset + j] != (guint8) (0xA0 + i))
          mismatches++;
      }
      EXPECT_EQ (mismatches, 0U) << "memory " << i;
      offset += sizes[i];
    }
  }

  gst_harness_teardown (h);
}

/**
 * @brief Test mqttsink refusing buffers larger than the allocated static message buffer (E4)
 */
TEST (testMqttSinkWithHelper, sinkPushMaxBufferSizeLarger_n)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstFlowReturn ret;
  GstBuffer *in_buf;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();

  g_object_set (h->element, "max-buffer-size", 64UL, NULL);

  in_buf = gst_harness_create_buffer (h, 64);
  ret = gst_harness_push (h, in_buf);
  EXPECT_EQ (ret, GST_FLOW_OK);
  EXPECT_EQ (GstMqttTestHelper::getInstance ().getSendCount (), 1);

  in_buf = gst_harness_create_buffer (h, 65);
  ret = gst_harness_push (h, in_buf);
  EXPECT_EQ (ret, GST_FLOW_ERROR);
  EXPECT_EQ (GstMqttTestHelper::getInstance ().getSendCount (), 1);

  in_buf = gst_harness_create_buffer (h, 1024 * 1024);
  ret = gst_harness_push (h, in_buf);
  EXPECT_EQ (ret, GST_FLOW_ERROR);
  EXPECT_EQ (GstMqttTestHelper::getInstance ().getSendCount (), 1);

  in_buf = gst_harness_create_buffer (h, 32);
  ret = gst_harness_push (h, in_buf);
  EXPECT_EQ (ret, GST_FLOW_OK);
  EXPECT_EQ (GstMqttTestHelper::getInstance ().getSendCount (), 2);

  gst_harness_teardown (h);
}

/**
 * @brief Test mqttsink refusing a first buffer already larger than max-buffer-size (E4)
 */
TEST (testMqttSinkWithHelper, sinkPushFirstBufferTooLarge_n)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstFlowReturn ret;
  GstBuffer *in_buf;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();

  g_object_set (h->element, "max-buffer-size", 16UL, NULL);

  in_buf = gst_harness_create_buffer (h, 17);
  ret = gst_harness_push (h, in_buf);
  EXPECT_EQ (ret, GST_FLOW_ERROR);
  EXPECT_EQ (GstMqttTestHelper::getInstance ().getSendCount (), 0);

  in_buf = gst_harness_create_buffer (h, 16);
  ret = gst_harness_push (h, in_buf);
  EXPECT_EQ (ret, GST_FLOW_OK);
  EXPECT_EQ (GstMqttTestHelper::getInstance ().getSendCount (), 1);

  gst_harness_teardown (h);
}

/**
 * @brief Test mqttsink not growing an already allocated static message buffer at runtime (E4)
 */
TEST (testMqttSinkWithHelper, sinkPushMaxBufferSizeRaised_n)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstFlowReturn ret;
  GstBuffer *in_buf;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();

  g_object_set (h->element, "max-buffer-size", 64UL, NULL);

  in_buf = gst_harness_create_buffer (h, 64);
  ret = gst_harness_push (h, in_buf);
  EXPECT_EQ (ret, GST_FLOW_OK);
  EXPECT_EQ (GstMqttTestHelper::getInstance ().getSendCount (), 1);

  g_object_set (h->element, "max-buffer-size", 4096UL, NULL);

  in_buf = gst_harness_create_buffer (h, 1024);
  ret = gst_harness_push (h, in_buf);
  EXPECT_EQ (ret, GST_FLOW_ERROR);
  EXPECT_EQ (GstMqttTestHelper::getInstance ().getSendCount (), 1);

  gst_harness_teardown (h);
}

/**
 * @brief Test mqttsink refusing a buffer when max-buffer-size + header length wraps around (E4)
 */
TEST (testMqttSinkWithHelper, sinkPushMaxBufferSizeWraps_n)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstFlowReturn ret;
  GstBuffer *in_buf;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();

  g_object_set (h->element, "max-buffer-size", G_MAXULONG - 511UL, NULL);

  in_buf = gst_harness_create_buffer (h, 16);
  ret = gst_harness_push (h, in_buf);
  EXPECT_EQ (ret, GST_FLOW_ERROR);
  EXPECT_EQ (GstMqttTestHelper::getInstance ().getSendCount (), 0);

  gst_harness_teardown (h);
}

/**
 * @brief Test mqttsink's dynamic (default) message buffer re-allocating for both smaller and larger buffers (E4)
 */
TEST (testMqttSinkWithHelper, sinkPushDynamicBufferSize)
{
  const gsize sizes[] = { 16, 4096, 8, 4096 };
  GstHarness *h = gst_harness_new ("mqttsink");
  GstFlowReturn ret;
  guint i;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();

  for (i = 0; i < G_N_ELEMENTS (sizes); ++i) {
    GstBuffer *in_buf = gst_harness_create_buffer (h, sizes[i]);
    guint8 pattern = (guint8) (0x20 + i);
    GstMapInfo map;
    gsize j;

    ASSERT_TRUE (gst_buffer_map (in_buf, &map, GST_MAP_WRITE));
    for (j = 0; j < map.size; ++j)
      map.data[j] = (guint8) (pattern + j);
    gst_buffer_unmap (in_buf, &map);

    ret = gst_harness_push (h, in_buf);
    EXPECT_EQ (ret, GST_FLOW_OK);
    EXPECT_EQ (GstMqttTestHelper::getInstance ().getSendCount (), (int) (i + 1));
    EXPECT_EQ ((gsize) GstMqttTestHelper::getInstance ().getLastPayloadLen (),
        GST_MQTT_LEN_MSG_HDR + sizes[i]);

    {
      const std::vector<guint8> &payload
          = GstMqttTestHelper::getInstance ().getLastPayload ();

      ASSERT_EQ (payload.size (), (size_t) (GST_MQTT_LEN_MSG_HDR + sizes[i]));
      for (j = 0; j < sizes[i]; ++j)
        EXPECT_EQ (payload[GST_MQTT_LEN_MSG_HDR + j], (guint8) (pattern + j));
    }
  }

  gst_harness_teardown (h);
}

/** The NTP host names/ports most recently handed to _capture_epoch_func () */
static guint32 captured_ntp_hnum = 0;
static std::vector<std::string> captured_ntp_hnames;
static std::vector<guint16> captured_ntp_hports;
static bool captured_ntp_terminated = false;

/**
 * @brief A get_epoch_func replacement that records the NTP server list mqttsink hands to it
 */
static int64_t
_capture_epoch_func (uint32_t hnum, char **hnames, uint16_t *hports)
{
  uint32_t i;

  captured_ntp_hnum = hnum;
  captured_ntp_hnames.clear ();
  captured_ntp_hports.clear ();
  for (i = 0; i < hnum; ++i) {
    captured_ntp_hnames.push_back (std::string (hnames[i]));
    captured_ntp_hports.push_back (hports[i]);
  }
  captured_ntp_terminated = (hnames == NULL) || (hnames[hnum] == NULL);

  return g_get_real_time ();
}

/**
 * @brief Test mqttsink's ntp-srvs property parsing a well-formed list of host:port pairs (E5)
 */
TEST (testMqttSink, ntpSrvsParse)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstMqttSink *sink;
  GstBuffer *in_buf;
  GstFlowReturn ret;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();

  g_object_set (h->element, "ntp-srvs", "a.example:123,b.example:456", NULL);

  sink = GST_MQTT_SINK (h->element);
  sink->get_epoch_func = _capture_epoch_func;

  in_buf = gst_harness_create_buffer (h, 4);
  ret = gst_harness_push (h, in_buf);
  EXPECT_EQ (ret, GST_FLOW_OK);

  ASSERT_EQ (captured_ntp_hnum, 2U);
  EXPECT_STREQ (captured_ntp_hnames[0].c_str (), "a.example");
  EXPECT_STREQ (captured_ntp_hnames[1].c_str (), "b.example");
  EXPECT_EQ (captured_ntp_hports[0], 123);
  EXPECT_EQ (captured_ntp_hports[1], 456);
  EXPECT_TRUE (captured_ntp_terminated);

  gst_harness_teardown (h);
}

/**
 * @brief Test mqttsink's ntp-srvs property replacing a previously set server list (E5, the old
 *        vector is released via g_strfreev (); a leak there is only visible to a leak checker)
 */
TEST (testMqttSink, ntpSrvsReplace)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstMqttSink *sink;
  GstBuffer *in_buf;
  GstFlowReturn ret;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();

  g_object_set (h->element, "ntp-srvs", "a:1,b:2,c:3", NULL);
  g_object_set (h->element, "ntp-srvs", "d:4", NULL);

  sink = GST_MQTT_SINK (h->element);
  sink->get_epoch_func = _capture_epoch_func;

  in_buf = gst_harness_create_buffer (h, 4);
  ret = gst_harness_push (h, in_buf);
  EXPECT_EQ (ret, GST_FLOW_OK);

  ASSERT_EQ (captured_ntp_hnum, 1U);
  EXPECT_STREQ (captured_ntp_hnames[0].c_str (), "d");
  EXPECT_EQ (captured_ntp_hports[0], 4);
  EXPECT_TRUE (captured_ntp_terminated);

  gst_harness_teardown (h);
}

/**
 * @brief A helper to set ntp-srvs on a running mqttsink, push a buffer, and check the servers parsed out of it
 */
static void
_check_ntp_srvs (GstHarness *h, const gchar *pairs,
    const std::vector<std::string> &exp_names, const std::vector<guint16> &exp_ports)
{
  const gchar *label = pairs ? pairs : "(null)";
  GstBuffer *in_buf;
  GstFlowReturn ret;
  gchar *sprop = NULL;
  guint i;

  g_object_set (h->element, "ntp-srvs", pairs, NULL);
  g_object_get (h->element, "ntp-srvs", &sprop, NULL);
  EXPECT_STREQ (sprop, pairs);
  g_free (sprop);

  captured_ntp_hnum = G_MAXUINT32;
  in_buf = gst_harness_create_buffer (h, 4);
  ret = gst_harness_push (h, in_buf);
  EXPECT_EQ (ret, GST_FLOW_OK);

  ASSERT_EQ (captured_ntp_hnum, (guint32) exp_names.size ()) << label;
  for (i = 0; i < captured_ntp_hnum; ++i) {
    EXPECT_STREQ (captured_ntp_hnames[i].c_str (), exp_names[i].c_str ()) << label;
    EXPECT_EQ (captured_ntp_hports[i], exp_ports[i]) << label;
  }
  EXPECT_TRUE (captured_ntp_terminated) << label;
}

/**
 * @brief Test mqttsink's ntp-srvs parser surviving pairs without a colon, empty entries, and an empty or NULL list
 */
TEST (testMqttSink, ntpSrvsMalformedPairs_n)
{
  GstHarness *h = gst_harness_new ("mqttsink");

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();
  GST_MQTT_SINK (h->element)->get_epoch_func = _capture_epoch_func;

  _check_ntp_srvs (h, "h:0,h:65536,h:abc,ok:7", { "ok" }, { 7 });
  _check_ntp_srvs (h, "nocolon", {}, {});
  _check_ntp_srvs (h, "a:1,,b:2", { "a", "b" }, { 1, 2 });
  _check_ntp_srvs (h, "a:1,", { "a" }, { 1 });
  _check_ntp_srvs (h, ",c:3", { "c" }, { 3 });
  _check_ntp_srvs (h, ":", {}, {});
  _check_ntp_srvs (h, "d:4,e", { "d" }, { 4 });
  _check_ntp_srvs (h, "", {}, {});
  _check_ntp_srvs (h, "f:6", { "f" }, { 6 });
  _check_ntp_srvs (h, NULL, {}, {});

  gst_harness_teardown (h);
}

/**
 * @brief A get_epoch_func replacement that can be held open at HOLD_EPOCH, then records the list
 */
static int64_t
_holding_epoch_func (uint32_t hnum, char **hnames, uint16_t *hports)
{
  _hold_call (HOLD_EPOCH);
  return _capture_epoch_func (hnum, hnames, hports);
}

/**
 * @brief Push one buffer into the harness, for a test that holds render () open on another thread
 */
static void
_push_one_buffer (GstHarness *h, GstFlowReturn *ret)
{
  *ret = gst_harness_push (h, gst_harness_create_buffer (h, 4));
}

/**
 * @brief Set an element to PLAYING, for a test that holds the state change open on another thread
 */
static void
_set_playing (GstElement *element, GstStateChangeReturn *ret)
{
  *ret = gst_element_set_state (element, GST_STATE_PLAYING);
}

/**
 * @brief Test replacing ntp-srvs while render () is inside get_epoch_func () with the previous list
 */
TEST (testMqttSink, ntpSrvsReplacedDuringRender)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstFlowReturn push_ret = GST_FLOW_ERROR;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();

  g_object_set (h->element, "ntp-srvs", "first.example.org:123,second.example.org:456", NULL);
  GST_MQTT_SINK (h->element)->get_epoch_func = _holding_epoch_func;
  _arm_call_hold (HOLD_EPOCH);

  std::thread pusher (_push_one_buffer, h, &push_ret);

  EXPECT_TRUE (_wait_call_hold ());
  g_object_set (h->element, "ntp-srvs", "third.example.org:789", NULL);
  _release_call_hold ();
  pusher.join ();

  EXPECT_EQ (push_ret, GST_FLOW_OK);
  ASSERT_EQ (captured_ntp_hnum, 2U);
  EXPECT_STREQ (captured_ntp_hnames[0].c_str (), "first.example.org");
  EXPECT_STREQ (captured_ntp_hnames[1].c_str (), "second.example.org");
  EXPECT_EQ (captured_ntp_hports[0], 123);
  EXPECT_EQ (captured_ntp_hports[1], 456);
  EXPECT_TRUE (captured_ntp_terminated);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);
  ASSERT_EQ (captured_ntp_hnum, 1U);
  EXPECT_STREQ (captured_ntp_hnames[0].c_str (), "third.example.org");
  EXPECT_EQ (captured_ntp_hports[0], 789);

  gst_harness_teardown (h);
}

/**
 * @brief Test replacing ntp-srvs while the PAUSED to PLAYING change is inside get_epoch_func () with the previous list
 */
TEST (testMqttSink, ntpSrvsReplacedDuringStateChange)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstStateChangeReturn state_ret = GST_STATE_CHANGE_FAILURE;
  GstClock *clock;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();

  g_object_set (h->element, "async", FALSE, "ntp-srvs",
      "first.example.org:123,second.example.org:456", NULL);
  clock = gst_system_clock_obtain ();
  gst_element_set_clock (h->element, clock);
  gst_object_unref (clock);
  EXPECT_EQ (gst_element_set_state (h->element, GST_STATE_PAUSED), GST_STATE_CHANGE_SUCCESS);

  GST_MQTT_SINK (h->element)->get_epoch_func = _holding_epoch_func;
  _arm_call_hold (HOLD_EPOCH);

  std::thread changer (_set_playing, h->element, &state_ret);

  EXPECT_TRUE (_wait_call_hold ());
  g_object_set (h->element, "ntp-srvs", "third.example.org:789", NULL);
  _release_call_hold ();
  changer.join ();

  EXPECT_EQ (state_ret, GST_STATE_CHANGE_SUCCESS);
  ASSERT_EQ (captured_ntp_hnum, 2U);
  EXPECT_STREQ (captured_ntp_hnames[0].c_str (), "first.example.org");
  EXPECT_STREQ (captured_ntp_hnames[1].c_str (), "second.example.org");
  EXPECT_EQ (captured_ntp_hports[0], 123);
  EXPECT_EQ (captured_ntp_hports[1], 456);
  EXPECT_TRUE (captured_ntp_terminated);

  gst_harness_teardown (h);
}

/**
 * @brief Test replacing pub-topic while render () is inside MQTTAsync_send () with the previous topic
 */
TEST (testMqttSinkWithHelper, pubTopicReplacedDuringSend)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstFlowReturn push_ret = GST_FLOW_ERROR;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();

  g_object_set (h->element, "pub-topic", "first/topic/published/here", NULL);
  _arm_call_hold (HOLD_SEND);

  std::thread pusher (_push_one_buffer, h, &push_ret);

  EXPECT_TRUE (_wait_call_hold ());
  g_object_set (h->element, "pub-topic", "later/topic/published/here", NULL);
  _release_call_hold ();
  pusher.join ();

  EXPECT_EQ (push_ret, GST_FLOW_OK);
  EXPECT_STREQ (GstMqttTestHelper::getInstance ().getLastTopic ().c_str (),
      "first/topic/published/here");

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);
  EXPECT_STREQ (GstMqttTestHelper::getInstance ().getLastTopic ().c_str (),
      "later/topic/published/here");

  gst_harness_teardown (h);
}

/**
 * @brief Set an element to READY, for a test that holds start () open on another thread
 */
static void
_set_ready (GstElement *element, GstStateChangeReturn *ret)
{
  *ret = gst_element_set_state (element, GST_STATE_READY);
}

/**
 * @brief Test replacing client-id and host while start () is inside MQTTAsync_create () with the previous values
 */
TEST (testMqttSinkWithHelper, clientIdReplacedDuringStart)
{
  GstElement *sink = gst_element_factory_make ("mqttsink", NULL);
  GstStateChangeReturn state_ret = GST_STATE_CHANGE_FAILURE;

  ASSERT_TRUE (sink != NULL);
  gst_object_ref_sink (sink);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetCreateRecord ();

  g_object_set (sink, "client-id", "first-client-identifier-in-use", "host",
      "first.broker.example.org", NULL);
  _arm_call_hold (HOLD_CREATE);

  std::thread starter (_set_ready, sink, &state_ret);

  EXPECT_TRUE (_wait_call_hold ());
  g_object_set (sink, "client-id", "later-client-identifier-in-use", "host",
      "later.broker.example.org", NULL);
  _release_call_hold ();
  starter.join ();

  EXPECT_EQ (state_ret, GST_STATE_CHANGE_SUCCESS);
  EXPECT_STREQ (GstMqttTestHelper::getInstance ().getLastClientId ().c_str (),
      "first-client-identifier-in-use");
  EXPECT_STREQ (GstMqttTestHelper::getInstance ().getLastServerUri ().c_str (),
      "first.broker.example.org:1883");

  EXPECT_EQ (gst_element_set_state (sink, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);
  gst_object_unref (sink);
}

/** What _capture_mqttsink_log () saw: the delivery message and how many mqttsink messages it formatted */
static GMutex log_capture_lock;
static std::string log_capture_delivery;
static gint log_capture_count = 0;

/**
 * @brief A GstLogFunction that formats mqttsink's messages the way the default
 * logger does, object path included, without printing them
 */
static void
_capture_mqttsink_log (GstDebugCategory *category, GstDebugLevel level,
    const gchar *file, const gchar *function, gint line, GObject *object,
    GstDebugMessage *message, gpointer user_data)
{
  const gchar *text;
  gchar *path = NULL;

  if (g_strcmp0 (gst_debug_category_get_name (category), GST_MQTT_ELEM_NAME_SINK) != 0)
    return;

  if (object && GST_IS_OBJECT (object))
    path = gst_object_get_path_string (GST_OBJECT (object));
  text = gst_debug_message_get (message);

  g_mutex_lock (&log_capture_lock);
  log_capture_count++;
  if (text && strstr (text, "has been delivered"))
    log_capture_delivery.assign (text);
  g_mutex_unlock (&log_capture_lock);

  g_free (path);
}

/**
 * @brief Send mqttsink's debug messages to _capture_mqttsink_log () instead of the default printer
 * @return the number of default log functions removed, to be passed to _stop_log_capture ()
 */
static guint
_start_log_capture (void)
{
  guint removed;

  g_mutex_lock (&log_capture_lock);
  log_capture_delivery.clear ();
  log_capture_count = 0;
  g_mutex_unlock (&log_capture_lock);

  removed = gst_debug_remove_log_function (gst_debug_log_default);
  gst_debug_add_log_function (_capture_mqttsink_log, NULL, NULL);
  gst_debug_set_threshold_for_name (GST_MQTT_ELEM_NAME_SINK, GST_LEVEL_DEBUG);

  return removed;
}

/**
 * @brief Undo _start_log_capture ()
 */
static void
_stop_log_capture (guint removed)
{
  gst_debug_unset_threshold_for_name (GST_MQTT_ELEM_NAME_SINK);
  gst_debug_remove_log_function (_capture_mqttsink_log);
  if (removed)
    gst_debug_add_log_function (gst_debug_log_default, NULL, NULL);
}

/** Set by a helper thread when its work has returned, so a test can bound the wait */
static GMutex work_done_lock;
static GCond work_done_cond;
static bool work_done = false;

/**
 * @brief Mark the helper thread's work as returned
 */
static void
_mark_work_done (void)
{
  g_mutex_lock (&work_done_lock);
  work_done = true;
  g_cond_broadcast (&work_done_cond);
  g_mutex_unlock (&work_done_lock);
}

/**
 * @brief Wait for _mark_work_done (); abort the test binary if the work does not return in time (a deadlock)
 */
static void
_wait_work_done_or_abort (gint seconds, const gchar *what)
{
  gint64 deadline = g_get_monotonic_time () + seconds * G_TIME_SPAN_SECOND;

  g_mutex_lock (&work_done_lock);
  while (!work_done) {
    if (!g_cond_wait_until (&work_done_cond, &work_done_lock, deadline))
      break;
  }
  if (!work_done)
    g_error ("%s did not return within %d s: a deadlock in mqttsink?", what, seconds);
  work_done = false;
  g_mutex_unlock (&work_done_lock);
}

/** How many buffers _push_buffers () has pushed so far */
static std::atomic<int> push_progress (0);

/**
 * @brief Wait for the pushing thread, aborting only once it has pushed nothing for @a stall_seconds
 */
static void
_wait_pushes_or_abort (gint stall_seconds)
{
  gint last = push_progress.load ();

  g_mutex_lock (&work_done_lock);
  while (!work_done) {
    gint64 deadline = g_get_monotonic_time () + stall_seconds * G_TIME_SPAN_SECOND;

    if (!g_cond_wait_until (&work_done_cond, &work_done_lock, deadline)) {
      gint now = push_progress.load ();

      if (now == last)
        g_error ("the streaming thread pushed nothing for %d s: a deadlock in mqttsink?",
            stall_seconds);
      last = now;
    }
  }
  work_done = false;
  g_mutex_unlock (&work_done_lock);
}

/**
 * @brief Call the delivery-complete callback the way paho does, then mark the work done
 */
static void
_call_delivery_complete (MQTTAsync_deliveryComplete *dc, void *context, MQTTAsync_token token)
{
  dc (context, token);
  _mark_work_done ();
}

/**
 * @brief Test the delivery-complete callback logs the current pub-topic and returns
 */
TEST (testMqttSinkWithHelper, deliveryCompleteLogsTopic)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  MQTTAsync_deliveryComplete *dc;
  guint removed;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();

  g_object_set (h->element, "pub-topic", "delivered/topic/name", NULL);
  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);

  dc = GstMqttTestHelper::getInstance ().getCbDeliveryComplete ();
  ASSERT_TRUE (dc != NULL);

  removed = _start_log_capture ();
  std::thread caller (_call_delivery_complete, dc,
      GstMqttTestHelper::getInstance ().getContext (), 7);
  _wait_work_done_or_abort (30, "the delivery-complete callback");
  caller.join ();
  _stop_log_capture (removed);

#ifndef GST_DISABLE_GST_DEBUG
  EXPECT_NE (log_capture_delivery.find ("delivered/topic/name"), std::string::npos)
      << log_capture_delivery;
  EXPECT_NE (log_capture_delivery.find ("token(7)"), std::string::npos) << log_capture_delivery;
#endif

  gst_harness_teardown (h);
}

/** Stops _churn_string_properties (), counts its rounds, and reports its exit */
static std::atomic<bool> churn_stop (false);
static std::atomic<int> churn_rounds (0);
static std::atomic<bool> churn_finished (false);

/**
 * @brief Keep replacing and reading every string property of @a sink, and
 * calling its delivery-complete callback, until churn_stop is set
 */
static void
_churn_string_properties (GstElement *sink, MQTTAsync_deliveryComplete *dc, void *context)
{
  int i = 0;

  while (!churn_stop.load ()) {
    gchar *topic = g_strdup_printf ("churn/topic/%d", i);
    gchar *srvs = g_strdup_printf (
        "h%d.example.org:%d,other.example.org:123", i, 1 + i % 1000);
    gchar *id = g_strdup_printf ("churn-client-%d", i);
    gchar *t = NULL, *n = NULL, *c = NULL, *a = NULL, *p = NULL;

    g_object_set (sink, "pub-topic", topic, "ntp-srvs", srvs, "client-id", id,
        "host", "churn.example.org", "port", "1883", NULL);
    g_object_get (sink, "pub-topic", &t, "ntp-srvs", &n, "client-id", &c,
        "host", &a, "port", &p, NULL);
    if (dc)
      dc (context, i);

    g_free (topic);
    g_free (srvs);
    g_free (id);
    g_free (t);
    g_free (n);
    g_free (c);
    g_free (a);
    g_free (p);
    i++;
    churn_rounds++;
    g_thread_yield ();
  }

  churn_finished = true;
}

/**
 * @brief Push @a count buffers into the harness, count the ones that did not return GST_FLOW_OK, then mark the work done
 */
static void
_push_buffers (GstHarness *h, int count, std::atomic<int> *failures)
{
  int i;

  for (i = 0; i < count; i++) {
    if (gst_harness_push (h, gst_harness_create_buffer (h, 64)) != GST_FLOW_OK)
      (*failures)++;
    push_progress++;
  }
  _mark_work_done ();
}

/**
 * @brief Test streaming while another thread keeps replacing the string properties, with mqttsink's debug logs formatted
 */
TEST (testMqttSinkWithHelper, propertiesReplacedWhileStreaming)
{
  const int count = 400;
  GstHarness *h = gst_harness_new ("mqttsink");
  std::atomic<int> failures (0);
  guint removed;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().resetSendRecord ();

  g_object_set (h->element, "debug", TRUE, NULL);
  GST_MQTT_SINK (h->element)->get_epoch_func = _capture_epoch_func;
  removed = _start_log_capture ();

  churn_stop = false;
  churn_rounds = 0;
  churn_finished = false;
  std::thread churner (_churn_string_properties, h->element,
      GstMqttTestHelper::getInstance ().getCbDeliveryComplete (),
      GstMqttTestHelper::getInstance ().getContext ());
  for (int waited = 0; churn_rounds.load () == 0; waited++) {
    if (waited >= 30000)
      g_error ("the property churn did not finish a round within 30 s: a deadlock in mqttsink?");
    g_usleep (1000);
  }

  push_progress = 0;
  std::thread pusher (_push_buffers, h, count, &failures);
  _wait_pushes_or_abort (30);
  pusher.join ();

  churn_stop = true;
  for (int waited = 0; !churn_finished.load (); waited++) {
    if (waited >= 30000)
      g_error ("the property churn did not stop within 30 s: a deadlock in mqttsink?");
    g_usleep (1000);
  }
  churner.join ();
  _stop_log_capture (removed);

  EXPECT_EQ (failures.load (), 0);
  EXPECT_EQ (GstMqttTestHelper::getInstance ().getSendCount (), count);
  EXPECT_GT (churn_rounds.load (), 1);
  EXPECT_EQ (GstMqttTestHelper::getInstance ().getLastTopic ().rfind ("churn/topic/", 0), 0U)
      << GstMqttTestHelper::getInstance ().getLastTopic ();
  EXPECT_EQ (captured_ntp_hnum, 2U);
  EXPECT_TRUE (captured_ntp_terminated);
#ifndef GST_DISABLE_GST_DEBUG
  EXPECT_GT (log_capture_count, count);
#endif

  gst_harness_teardown (h);
}

/**
 * @brief Test mqttsink's default pub-topic surviving a stop/start cycle without a double free (E5)
 */
TEST (testMqttSink, defaultPubTopic)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstStateChangeReturn sret;
  gchar *client_id = NULL;
  gchar *expected;
  gchar *topic1 = NULL;
  gchar *topic2 = NULL;

  ASSERT_TRUE (h != NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();

  g_object_get (h->element, "client-id", &client_id, NULL);
  g_object_get (h->element, "pub-topic", &topic1, NULL);
  expected = g_strdup_printf ("%s/topic", client_id);
  EXPECT_STREQ (topic1, expected);

  sret = gst_element_set_state (h->element, GST_STATE_NULL);
  EXPECT_NE (sret, GST_STATE_CHANGE_FAILURE);
  sret = gst_element_set_state (h->element, GST_STATE_PLAYING);
  EXPECT_NE (sret, GST_STATE_CHANGE_FAILURE);
  gst_element_get_state (h->element, NULL, NULL, GST_CLOCK_TIME_NONE);

  g_object_get (h->element, "pub-topic", &topic2, NULL);
  EXPECT_STREQ (topic2, expected);

  g_free (client_id);
  g_free (expected);
  g_free (topic1);
  g_free (topic2);

  gst_harness_teardown (h);
}

/**
 * @brief Test mqttsink keeping a pub-topic set before start instead of generating the default one
 */
TEST (testMqttSink, userPubTopicKept)
{
  GstElement *sink = gst_element_factory_make ("mqttsink", NULL);
  GstHarness *h;
  gchar *topic = NULL;

  ASSERT_TRUE (sink != NULL);
  gst_object_ref_sink (sink);
  g_object_set (sink, "pub-topic", "mytopic", NULL);

  GstMqttTestHelper::getInstance ().initFailFlags ();
  h = gst_harness_new_with_element (sink, "sink", NULL);
  gst_object_unref (sink);
  ASSERT_TRUE (h != NULL);

  g_object_get (h->element, "pub-topic", &topic, NULL);
  EXPECT_STREQ (topic, "mytopic");
  g_free (topic);

  gst_harness_teardown (h);
}

/**
 * @brief A helper function for the generation of a dummy MQTT message
 */
static void
_gen_dummy_mqtt_msg (MQTTAsync_message *msg, GstMQTTMessageHdr *hdr, const gsize len_buf)
{
  gboolean mapped;
  GstBuffer *buf;
  GstMemory *mem;
  GstMapInfo map;

  buf = gst_buffer_new_allocate (NULL, len_buf, NULL);
  ASSERT_FALSE (buf == NULL);

  mem = gst_buffer_get_all_memory (buf);
  ASSERT_FALSE (mem == NULL);

  mapped = gst_memory_map (mem, &map, GST_MAP_READ);
  ASSERT_EQ (mapped, TRUE);

  memcpy (msg->payload, hdr, GST_MQTT_LEN_MSG_HDR);
  memcpy (&((guint8 *) msg->payload)[GST_MQTT_LEN_MSG_HDR], map.data, len_buf);

  gst_memory_unmap (mem, &map);
  gst_memory_unref (mem);
  gst_buffer_unref (buf);
}

/**
 * @brief Test mqttsrc using a proper pipeline description #1
 */
TEST (testMqttSrcWithHelper, srcNormalLaunch0)
{
  const gsize len_buf = 1024;
  gchar *caps_str = g_strdup ("video/x-raw,width=640,height=320,format=RGB");
  gchar *topic_name = g_strdup ("test_topic");
  gchar *str_pipeline
      = g_strdup_printf ("mqttsrc sub-topic=%s debug=true is-live=true num-buffers=%d "
                         "sub-timeout=%" G_GINT64_FORMAT " ! "
                         "capsfilter caps=%s ! videoconvert ! videoscale ! fakesink",
          topic_name, 1, G_TIME_SPAN_MINUTE, caps_str);
  GError *err = NULL;
  GstElement *pipeline;
  GstStateChangeReturn ret;
  GstState cur_state;
  GstMQTTMessageHdr hdr;
  MQTTAsync_message *msg;
  std::future<int> ma_ret;
  std::string err_msg;
  bool err_flag = false;

  pipeline = gst_parse_launch (str_pipeline, &err);
  g_free (str_pipeline);
  if ((!pipeline) || (err)) {
    err_flag = true;
    err_msg = std::string ("Failed to launch the given pipeline");
    goto free_strs;
  }
  GstMqttTestHelper::getInstance ().initFailFlags ();

  msg = (MQTTAsync_message *) g_try_malloc0 (sizeof (*msg));
  if (!msg) {
    err_flag = true;
    err_msg = std::string ("Failed to allocate a MQTTAsync_message");
    goto free_strs;
  }

  _set_ts_gst_mqtt_message_hdr (pipeline, &hdr, GST_SECOND, 500 * GST_MSECOND);
  ret = gst_element_set_state (pipeline, GST_STATE_PAUSED);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_NO_PREROLL);
  EXPECT_EQ (cur_state, GST_STATE_PAUSED);

  memset (hdr.gst_caps_str, '\0', GST_MQTT_MAX_LEN_GST_CAPS_STR);
  memcpy (hdr.gst_caps_str, caps_str,
      MIN (strlen (caps_str), GST_MQTT_MAX_LEN_GST_CAPS_STR - 1));
  hdr.num_mems = 1;
  hdr.size_mems[0] = len_buf;

  msg->payloadlen = GST_MQTT_LEN_MSG_HDR + len_buf;
  msg->payload = (MQTTAsync_message *) g_try_malloc0 (msg->payloadlen);
  if (!msg->payload) {
    err_flag = true;
    err_msg = std::string ("Failed to allocate buffer for MQTT message payload");
    goto free_msg_buf;
  }

  ret = gst_element_set_state (pipeline, GST_STATE_PLAYING);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  _gen_dummy_mqtt_msg (msg, &hdr, len_buf);

  ma_ret = std::async (std::launch::async,
      GstMqttTestHelper::getInstance ().getCbMessageArrived (),
      GstMqttTestHelper::getInstance ().getContext (), topic_name, 0, msg);
  EXPECT_TRUE (ma_ret.get ());

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (cur_state, GST_STATE_PLAYING);

  ret = gst_element_set_state (pipeline, GST_STATE_NULL);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  gst_object_unref (pipeline);

  g_free (msg->payload);
free_msg_buf:
  g_free (msg);
free_strs:
  g_free (caps_str);
  g_free (topic_name);

  if (err_flag)
    FAIL () << err_msg;
}

/**
 * @brief Test mqttsrc using a proper pipeline description #2 (dynamically re-negotiating GstCaps)
 */
TEST (testMqttSrcWithHelper, srcNormalLaunch1)
{
  const gsize len_buf = 1024;
  gchar *caps_str = g_strdup ("video/x-raw,width=640,height=320,format=RGB");
  gchar *topic_name = g_strdup ("test_topic");
  gchar *str_pipeline
      = g_strdup_printf ("mqttsrc sub-topic=%s debug=true is-live=true num-buffers=%d "
                         "sub-timeout=%" G_GINT64_FORMAT " ! "
                         "capsfilter caps=%s ! videoconvert ! videoscale ! fakesink",
          topic_name, 2, G_TIME_SPAN_MINUTE, caps_str);
  GError *err = NULL;
  GstElement *pipeline;
  GstStateChangeReturn ret;
  GstState cur_state;
  GstMQTTMessageHdr hdr;
  MQTTAsync_message *msg;
  std::future<int> ma_ret;
  std::string err_msg;
  bool err_flag = false;

  pipeline = gst_parse_launch (str_pipeline, &err);
  g_free (str_pipeline);
  if ((!pipeline) || (err)) {
    err_flag = true;
    err_msg = std::string ("Failed to launch the given pipeline");
    goto free_strs;
  }
  GstMqttTestHelper::getInstance ().initFailFlags ();

  msg = (MQTTAsync_message *) g_try_malloc0 (sizeof (*msg));
  if (!msg) {
    err_msg = std::string ("Failed to allocate a MQTTAsync_message");
    goto free_strs;
  }

  _set_ts_gst_mqtt_message_hdr (pipeline, &hdr, GST_SECOND, 500 * GST_MSECOND);
  ret = gst_element_set_state (pipeline, GST_STATE_PAUSED);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_NO_PREROLL);
  EXPECT_EQ (cur_state, GST_STATE_PAUSED);

  memset (hdr.gst_caps_str, '\0', GST_MQTT_MAX_LEN_GST_CAPS_STR);
  memcpy (hdr.gst_caps_str, caps_str,
      MIN (strlen (caps_str), GST_MQTT_MAX_LEN_GST_CAPS_STR - 1));
  hdr.num_mems = 1;
  hdr.size_mems[0] = len_buf;

  msg->payloadlen = GST_MQTT_LEN_MSG_HDR + len_buf;
  msg->payload = g_try_malloc0 (msg->payloadlen);
  if (!msg->payload) {
    err_msg = std::string ("Failed to allocate buffer for MQTT message payload");
    err_flag = true;
    goto free_msg_buf;
  }

  ret = gst_element_set_state (pipeline, GST_STATE_PLAYING);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  _gen_dummy_mqtt_msg (msg, &hdr, len_buf);

  ma_ret = std::async (std::launch::async,
      GstMqttTestHelper::getInstance ().getCbMessageArrived (),
      GstMqttTestHelper::getInstance ().getContext (), topic_name, 0, msg);
  EXPECT_TRUE (ma_ret.get ());

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (cur_state, GST_STATE_PLAYING);

  /** Changing caps while the pipeline is in the GST_STATE_PLAYING state */
  g_free (caps_str);
  caps_str = g_strdup ("video/x-raw,width=320,height=160,format=YUY2");
  memset (hdr.gst_caps_str, '\0', GST_MQTT_MAX_LEN_GST_CAPS_STR);
  memcpy (hdr.gst_caps_str, caps_str,
      MIN (strlen (caps_str), GST_MQTT_MAX_LEN_GST_CAPS_STR - 1));
  memcpy (msg->payload, &hdr, GST_MQTT_LEN_MSG_HDR);

  ma_ret = std::async (std::launch::async,
      GstMqttTestHelper::getInstance ().getCbMessageArrived (),
      GstMqttTestHelper::getInstance ().getContext (), topic_name, 0, msg);
  EXPECT_TRUE (ma_ret.get ());

  ret = gst_element_set_state (pipeline, GST_STATE_NULL);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  gst_object_unref (pipeline);

  g_free (msg->payload);
free_msg_buf:
  g_free (msg);
free_strs:
  g_free (caps_str);
  g_free (topic_name);

  if (err_flag)
    FAIL () << err_msg;
}

/**
 * @brief Fail test case for mqttsrc #0 (MQTTAsync_subscribe failure case)
 */
TEST (testMqttSrcWithHelper, srcNormalLaunch0_n)
{
  const gsize len_buf = 1024;
  gchar *caps_str = g_strdup ("video/x-raw,width=640,height=320,format=RGB");
  gchar *topic_name = g_strdup ("test_topic");
  gchar *str_pipeline
      = g_strdup_printf ("mqttsrc sub-topic=%s debug=true is-live=true num-buffers=%d "
                         "sub-timeout=%" G_GINT64_FORMAT " ! "
                         "capsfilter caps=%s ! videoconvert ! videoscale ! fakesink",
          topic_name, 1, G_TIME_SPAN_MINUTE, caps_str);
  GError *err = NULL;
  GstElement *pipeline;
  GstStateChangeReturn ret;
  GstState cur_state;
  GstMQTTMessageHdr hdr;
  MQTTAsync_message *msg;
  std::future<int> ma_ret;
  std::string err_msg;
  bool err_flag = false;

  pipeline = gst_parse_launch (str_pipeline, &err);
  g_free (str_pipeline);
  if ((!pipeline) || (err)) {
    err_flag = true;
    err_msg = std::string ("Failed to launch the given pipeline");
    goto free_strs;
  }

  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().setFailSubscribe (TRUE);

  msg = (MQTTAsync_message *) g_try_malloc0 (sizeof (*msg));
  if (!msg) {
    err_msg = std::string ("Failed to allocate a MQTTAsync_message");
    goto free_strs;
  }

  _set_ts_gst_mqtt_message_hdr (pipeline, &hdr, GST_SECOND, 500 * GST_MSECOND);
  ret = gst_element_set_state (pipeline, GST_STATE_PAUSED);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_NO_PREROLL);
  EXPECT_EQ (cur_state, GST_STATE_PAUSED);

  memset (hdr.gst_caps_str, '\0', GST_MQTT_MAX_LEN_GST_CAPS_STR);
  memcpy (hdr.gst_caps_str, caps_str,
      MIN (strlen (caps_str), GST_MQTT_MAX_LEN_GST_CAPS_STR - 1));
  hdr.num_mems = 1;
  hdr.size_mems[0] = len_buf;

  msg->payloadlen = GST_MQTT_LEN_MSG_HDR + len_buf;
  msg->payload = (MQTTAsync_message *) g_try_malloc0 (msg->payloadlen);
  if (!msg->payload) {
    err_msg = std::string ("Failed to allocate buffer for MQTT message payload");
    err_flag = true;
    goto free_msg_buf;
  }

  ret = gst_element_set_state (pipeline, GST_STATE_PLAYING);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  _gen_dummy_mqtt_msg (msg, &hdr, len_buf);

  ma_ret = std::async (std::launch::async,
      GstMqttTestHelper::getInstance ().getCbMessageArrived (),
      GstMqttTestHelper::getInstance ().getContext (), topic_name, 0, msg);
  EXPECT_TRUE (ma_ret.get ());

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_FAILURE);
  EXPECT_EQ (cur_state, GST_STATE_PAUSED);

  ret = gst_element_set_state (pipeline, GST_STATE_NULL);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  GstMqttTestHelper::getInstance ().setFailSubscribe (FALSE);
  gst_object_unref (pipeline);

  g_free (msg->payload);
free_msg_buf:
  g_free (msg);
free_strs:
  g_free (caps_str);
  g_free (topic_name);

  if (err_flag)
    FAIL () << err_msg;
}

/**
 * @brief Fail test case for mqttsrc #1 (MQTTAsync_disconnect failure case)
 */
TEST (testMqttSrcWithHelper, srcNormalLaunch1_n)
{
  const gsize len_buf = 1024;
  gchar *caps_str = g_strdup ("video/x-raw,width=640,height=320,format=RGB");
  gchar *topic_name = g_strdup ("test_topic");
  gchar *str_pipeline
      = g_strdup_printf ("mqttsrc sub-topic=%s debug=true is-live=true num-buffers=%d "
                         "sub-timeout=%" G_GINT64_FORMAT " ! "
                         "capsfilter caps=%s ! videoconvert ! videoscale ! fakesink",
          topic_name, 1, G_TIME_SPAN_MINUTE, caps_str);
  GError *err = NULL;
  GstElement *pipeline;
  GstStateChangeReturn ret;
  GstState cur_state;
  GstMQTTMessageHdr hdr;
  MQTTAsync_message *msg;
  std::future<int> ma_ret;
  std::string err_msg;
  bool err_flag = false;

  pipeline = gst_parse_launch (str_pipeline, &err);
  g_free (str_pipeline);
  if ((!pipeline) || (err)) {
    err_flag = true;
    err_msg = std::string ("Failed to launch the given pipeline");
    goto free_strs;
  }

  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().setFailDisconnect (TRUE);

  msg = (MQTTAsync_message *) g_try_malloc0 (sizeof (*msg));
  if (!msg) {
    err_msg = std::string ("Failed to allocate a MQTTAsync_message");
    goto free_strs;
  }

  _set_ts_gst_mqtt_message_hdr (pipeline, &hdr, GST_SECOND, 500 * GST_MSECOND);
  ret = gst_element_set_state (pipeline, GST_STATE_PAUSED);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_NO_PREROLL);
  EXPECT_EQ (cur_state, GST_STATE_PAUSED);

  memset (hdr.gst_caps_str, '\0', GST_MQTT_MAX_LEN_GST_CAPS_STR);
  memcpy (hdr.gst_caps_str, caps_str,
      MIN (strlen (caps_str), GST_MQTT_MAX_LEN_GST_CAPS_STR - 1));
  hdr.num_mems = 1;
  hdr.size_mems[0] = len_buf;

  msg->payloadlen = GST_MQTT_LEN_MSG_HDR + len_buf;
  msg->payload = (MQTTAsync_message *) g_try_malloc0 (msg->payloadlen);
  if (!msg->payload) {
    err_msg = std::string ("Failed to allocate buffer for MQTT message payload");
    err_flag = true;
    goto free_msg_buf;
  }

  ret = gst_element_set_state (pipeline, GST_STATE_PLAYING);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  _gen_dummy_mqtt_msg (msg, &hdr, len_buf);

  ma_ret = std::async (std::launch::async,
      GstMqttTestHelper::getInstance ().getCbMessageArrived (),
      GstMqttTestHelper::getInstance ().getContext (), topic_name, 0, msg);
  EXPECT_TRUE (ma_ret.get ());

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (cur_state, GST_STATE_PLAYING);

  GstMqttTestHelper::getInstance ().setFailDisconnect (FALSE);

  ret = gst_element_set_state (pipeline, GST_STATE_NULL);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  gst_object_unref (pipeline);

  g_free (msg->payload);
free_msg_buf:
  g_free (msg);
free_strs:
  g_free (caps_str);
  g_free (topic_name);

  if (err_flag)
    FAIL () << err_msg;
}

/**
 * @brief Fail test case for mqttsrc #2 (MQTTAsync_unsubscribe failure case)
 */
TEST (testMqttSrcWithHelper, srcNormalLaunch2)
{
  const gsize len_buf = 1024;
  gchar *caps_str = g_strdup ("video/x-raw,width=640,height=320,format=RGB");
  gchar *topic_name = g_strdup ("test_topic");
  gchar *str_pipeline
      = g_strdup_printf ("mqttsrc sub-topic=%s debug=true is-live=true num-buffers=%d "
                         "sub-timeout=%" G_GINT64_FORMAT " ! "
                         "capsfilter caps=%s ! videoconvert ! videoscale ! fakesink",
          topic_name, 1, G_TIME_SPAN_MINUTE, caps_str);
  GError *err = NULL;
  GstElement *pipeline;
  GstStateChangeReturn ret;
  GstState cur_state;
  GstMQTTMessageHdr hdr;
  MQTTAsync_message *msg;
  std::future<int> ma_ret;
  std::string err_msg;
  bool err_flag = false;

  pipeline = gst_parse_launch (str_pipeline, &err);
  g_free (str_pipeline);
  if ((!pipeline) || (err)) {
    err_flag = true;
    err_msg = std::string ("Failed to launch the given pipeline");
    goto free_strs;
  }

  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().setFailUnsubscribe (TRUE);

  msg = (MQTTAsync_message *) g_try_malloc0 (sizeof (*msg));
  if (!msg) {
    err_msg = std::string ("Failed to allocate a MQTTAsync_message");
    goto free_strs;
  }

  _set_ts_gst_mqtt_message_hdr (pipeline, &hdr, GST_SECOND, 500 * GST_MSECOND);
  ret = gst_element_set_state (pipeline, GST_STATE_PAUSED);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_NO_PREROLL);
  EXPECT_EQ (cur_state, GST_STATE_PAUSED);

  memset (hdr.gst_caps_str, '\0', GST_MQTT_MAX_LEN_GST_CAPS_STR);
  memcpy (hdr.gst_caps_str, caps_str,
      MIN (strlen (caps_str), GST_MQTT_MAX_LEN_GST_CAPS_STR - 1));
  hdr.num_mems = 1;
  hdr.size_mems[0] = len_buf;

  msg->payloadlen = GST_MQTT_LEN_MSG_HDR + len_buf;
  msg->payload = (MQTTAsync_message *) g_try_malloc0 (msg->payloadlen);
  if (!msg->payload) {
    err_msg = std::string ("Failed to allocate buffer for MQTT message payload");
    err_flag = true;
    goto free_msg_buf;
  }

  ret = gst_element_set_state (pipeline, GST_STATE_PLAYING);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  _gen_dummy_mqtt_msg (msg, &hdr, len_buf);

  ma_ret = std::async (std::launch::async,
      GstMqttTestHelper::getInstance ().getCbMessageArrived (),
      GstMqttTestHelper::getInstance ().getContext (), topic_name, 0, msg);
  EXPECT_TRUE (ma_ret.get ());

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_FAILURE);
  EXPECT_EQ (cur_state, GST_STATE_PAUSED);

  GstMqttTestHelper::getInstance ().setFailUnsubscribe (FALSE);

  ret = gst_element_set_state (pipeline, GST_STATE_NULL);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  gst_object_unref (pipeline);

  g_free (msg->payload);
free_msg_buf:
  g_free (msg);
free_strs:
  g_free (caps_str);
  g_free (topic_name);

  if (err_flag)
    FAIL () << err_msg;
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
