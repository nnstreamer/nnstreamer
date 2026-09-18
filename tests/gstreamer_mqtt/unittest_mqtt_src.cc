/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_mqtt_src.cc
 * @date        18 Sep 2026
 * @brief       Unit tests for mqttsrc with the paho-mqtt calls mocked
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 *
 * The mocks here are deliberately separate from the ones in
 * unittest_mqtt_w_helper.cc: these cases drive the message-arrived callback
 * with headers a broken or hostile publisher can send, and they have to
 * observe what the element does with the memory paho hands it.
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>

#include <MQTTAsync.h>
#include <string.h>

#include <string>
#include <thread>
#include <vector>

#include "mqttcommon.h"

#define DEFAULT_SUB_TIMEOUT (G_GINT64_CONSTANT (1000000))
#define SETTLE_TIME_US (300 * G_TIME_SPAN_MILLISECOND)
#define WAIT_TIMEOUT_US (5 * G_TIME_SPAN_SECOND)

/**
 * @brief The mocked paho call a test case can hold open on another thread
 */
typedef enum {
  HOLD_NONE = 0,
  HOLD_CREATE,
  HOLD_SUBSCRIBE,
} hold_site_e;

/**
 * @brief The state shared between the mocked paho calls and the test cases
 */
typedef struct {
  GMutex lock;
  GCond cond;

  void *context;
  MQTTAsync_messageArrived *message_arrived;
  MQTTAsync_connectionLost *connection_lost;

  gboolean async_connect;
  gboolean fail_create;
  gboolean fail_connect;
  gboolean fail_subscribe;
  gboolean fail_unsubscribe;
  gboolean fail_reconnect;

  std::string last_client_id;
  std::string last_server_uri;
  std::string last_sub_topic;
  std::string last_unsub_topic;

  guint free_message_count;
  guint free_topic_count;
  guint subscribe_count;

  hold_site_e hold_site;
  gboolean hold_entered;
  gboolean hold_released;
} mqtt_mock_s;

static mqtt_mock_s g_mock;
static std::vector<std::thread> g_mock_threads;

/**
 * @brief Reset the mock state before a test case runs
 */
static void
_mock_reset (void)
{
  g_mutex_lock (&g_mock.lock);
  g_mock.context = NULL;
  g_mock.message_arrived = NULL;
  g_mock.connection_lost = NULL;
  g_mock.async_connect = FALSE;
  g_mock.fail_create = FALSE;
  g_mock.fail_connect = FALSE;
  g_mock.fail_subscribe = FALSE;
  g_mock.fail_unsubscribe = FALSE;
  g_mock.fail_reconnect = FALSE;
  g_mock.last_client_id.clear ();
  g_mock.last_server_uri.clear ();
  g_mock.last_sub_topic.clear ();
  g_mock.last_unsub_topic.clear ();
  g_mock.free_message_count = 0;
  g_mock.free_topic_count = 0;
  g_mock.subscribe_count = 0;
  g_mock.hold_site = HOLD_NONE;
  g_mock.hold_entered = FALSE;
  g_mock.hold_released = FALSE;
  g_mutex_unlock (&g_mock.lock);
}

/**
 * @brief Join the threads the mocked calls have spawned
 */
static void
_mock_join_threads (void)
{
  for (size_t i = 0; i < g_mock_threads.size (); ++i) {
    if (g_mock_threads[i].joinable ())
      g_mock_threads[i].join ();
  }
  g_mock_threads.clear ();
}

/**
 * @brief Make the next call at the given site block until the test releases it
 */
static void
_hold_arm (hold_site_e site)
{
  g_mutex_lock (&g_mock.lock);
  g_mock.hold_site = site;
  g_mock.hold_entered = FALSE;
  g_mock.hold_released = FALSE;
  g_mutex_unlock (&g_mock.lock);
}

/**
 * @brief Block inside a mocked call while the site is armed
 */
static void
_hold_enter (hold_site_e site)
{
  g_mutex_lock (&g_mock.lock);
  if (g_mock.hold_site == site) {
    g_mock.hold_entered = TRUE;
    g_cond_broadcast (&g_mock.cond);
    while (!g_mock.hold_released)
      g_cond_wait (&g_mock.cond, &g_mock.lock);
  }
  g_mutex_unlock (&g_mock.lock);
}

/**
 * @brief Wait until another thread is blocked inside the armed call
 */
static gboolean
_hold_wait_entered (void)
{
  gint64 end_time = g_get_monotonic_time () + WAIT_TIMEOUT_US;
  gboolean entered;

  g_mutex_lock (&g_mock.lock);
  while (!g_mock.hold_entered) {
    if (!g_cond_wait_until (&g_mock.cond, &g_mock.lock, end_time))
      break;
  }
  entered = g_mock.hold_entered;
  g_mutex_unlock (&g_mock.lock);

  return entered;
}

/**
 * @brief Let the held call continue
 */
static void
_hold_release (void)
{
  g_mutex_lock (&g_mock.lock);
  g_mock.hold_site = HOLD_NONE;
  g_mock.hold_released = TRUE;
  g_cond_broadcast (&g_mock.cond);
  g_mutex_unlock (&g_mock.lock);
}

/**
 * @brief Wait until the element has called MQTTAsync_subscribe ()
 */
static gboolean
_wait_for_subscribe (void)
{
  gint64 end_time = g_get_monotonic_time () + WAIT_TIMEOUT_US;
  gboolean subscribed;

  g_mutex_lock (&g_mock.lock);
  while (g_mock.subscribe_count == 0) {
    if (!g_cond_wait_until (&g_mock.cond, &g_mock.lock, end_time))
      break;
  }
  subscribed = (g_mock.subscribe_count > 0);
  g_mutex_unlock (&g_mock.lock);

  return subscribed;
}

/**
 * @brief Run a success callback the way the paho client thread would
 */
static void
_run_success_cb (MQTTAsync_onSuccess *cb, void *context)
{
  MQTTAsync_successData data;

  memset (&data, 0, sizeof (data));
  cb (context, &data);
}

/**
 * @brief Run a failure callback the way the paho client thread would
 */
static void
_run_failure_cb (MQTTAsync_onFailure *cb, void *context)
{
  MQTTAsync_failureData data;

  memset (&data, 0, sizeof (data));
  data.code = -1;
  data.message = "mocked failure";
  cb (context, &data);
}

/**
 * @brief Run a callback on a thread of its own, as the paho client does
 */
static void
_spawn_cb_thread (MQTTAsync_onSuccess *cb, void *context)
{
  g_mutex_lock (&g_mock.lock);
  g_mock_threads.push_back (std::thread (_run_success_cb, cb, context));
  g_mutex_unlock (&g_mock.lock);
}

/**
 * @brief A mock function for MQTTAsync_create () in paho-mqtt-c
 */
int
MQTTAsync_create (MQTTAsync *handle, const char *serverURI,
    const char *clientId, int persistence_type, void *persistence_context)
{
  (void) persistence_type;
  (void) persistence_context;

  gboolean fail;

  _hold_enter (HOLD_CREATE);

  g_mutex_lock (&g_mock.lock);
  g_mock.last_server_uri.assign (serverURI ? serverURI : "");
  g_mock.last_client_id.assign (clientId ? clientId : "");
  fail = g_mock.fail_create;
  g_mutex_unlock (&g_mock.lock);

  if (fail)
    return MQTTASYNC_FAILURE;

  *handle = (MQTTAsync) &g_mock;

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_setCallbacks () in paho-mqtt-c
 */
int
MQTTAsync_setCallbacks (MQTTAsync handle, void *context, MQTTAsync_connectionLost *cl,
    MQTTAsync_messageArrived *ma, MQTTAsync_deliveryComplete *dc)
{
  (void) handle;
  (void) dc;

  g_mutex_lock (&g_mock.lock);
  g_mock.context = context;
  g_mock.message_arrived = ma;
  g_mock.connection_lost = cl;
  g_mutex_unlock (&g_mock.lock);

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_connect () in paho-mqtt-c
 */
int
MQTTAsync_connect (MQTTAsync handle, const MQTTAsync_connectOptions *options)
{
  gboolean async;
  gboolean fail;

  (void) handle;

  g_mutex_lock (&g_mock.lock);
  async = g_mock.async_connect;
  fail = g_mock.fail_connect;
  g_mutex_unlock (&g_mock.lock);

  if (fail) {
    _run_failure_cb (options->onFailure, options->context);
    return MQTTASYNC_FAILURE;
  }

  if (async) {
    _spawn_cb_thread (options->onSuccess, options->context);
  } else {
    _run_success_cb (options->onSuccess, options->context);
  }

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_subscribe () in paho-mqtt-c
 */
int
MQTTAsync_subscribe (MQTTAsync handle, const char *topic, int qos,
    MQTTAsync_responseOptions *response)
{
  gboolean fail;

  (void) handle;
  (void) qos;

  _hold_enter (HOLD_SUBSCRIBE);

  g_mutex_lock (&g_mock.lock);
  g_mock.last_sub_topic.assign (topic ? topic : "");
  g_mock.subscribe_count++;
  fail = g_mock.fail_subscribe;
  g_cond_broadcast (&g_mock.cond);
  g_mutex_unlock (&g_mock.lock);

  if (fail) {
    _run_failure_cb (response->onFailure, response->context);
    return MQTTASYNC_FAILURE;
  }

  _run_success_cb (response->onSuccess, response->context);

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_unsubscribe () in paho-mqtt-c
 */
int
MQTTAsync_unsubscribe (MQTTAsync handle, const char *topic, MQTTAsync_responseOptions *response)
{
  gboolean fail;

  (void) handle;

  g_mutex_lock (&g_mock.lock);
  g_mock.last_unsub_topic.assign (topic ? topic : "");
  fail = g_mock.fail_unsubscribe;
  g_mutex_unlock (&g_mock.lock);

  if (fail) {
    _run_failure_cb (response->onFailure, response->context);
    return MQTTASYNC_FAILURE;
  }

  _run_success_cb (response->onSuccess, response->context);

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_reconnect () in paho-mqtt-c
 */
int
MQTTAsync_reconnect (MQTTAsync handle)
{
  gboolean fail;

  (void) handle;

  g_mutex_lock (&g_mock.lock);
  fail = g_mock.fail_reconnect;
  g_mutex_unlock (&g_mock.lock);

  return fail ? MQTTASYNC_FAILURE : MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_disconnect () in paho-mqtt-c
 */
int
MQTTAsync_disconnect (MQTTAsync handle, const MQTTAsync_disconnectOptions *options)
{
  (void) handle;

  if (!options)
    return MQTTASYNC_SUCCESS;

  /** The element waits for this callback with its own lock held */
  _spawn_cb_thread (options->onSuccess, options->context);

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_isConnected () in paho-mqtt-c
 */
int
MQTTAsync_isConnected (MQTTAsync handle)
{
  (void) handle;
  return 1;
}

/**
 * @brief A mock function for MQTTAsync_destroy () in paho-mqtt-c
 */
void
MQTTAsync_destroy (MQTTAsync *handle)
{
  *handle = NULL;
}

/**
 * @brief A mock function for MQTTAsync_free () in paho-mqtt-c
 */
void
MQTTAsync_free (void *memory)
{
  g_mutex_lock (&g_mock.lock);
  g_mock.free_topic_count++;
  g_mutex_unlock (&g_mock.lock);

  g_free (memory);
}

/**
 * @brief A mock function for MQTTAsync_freeMessage () in paho-mqtt-c
 */
void
MQTTAsync_freeMessage (MQTTAsync_message **message)
{
  g_mutex_lock (&g_mock.lock);
  g_mock.free_message_count++;
  g_mutex_unlock (&g_mock.lock);

  g_free ((*message)->payload);
  g_free (*message);
  *message = NULL;
}

/**
 * @brief A log function that reads the logged object under its own lock
 */
static void
_path_log_func (GstDebugCategory *category, GstDebugLevel level,
    const gchar *file, const gchar *function, gint line, GObject *object,
    GstDebugMessage *message, gpointer user_data)
{
  (void) category;
  (void) level;
  (void) file;
  (void) function;
  (void) line;
  (void) message;
  (void) user_data;

  if (object && GST_IS_OBJECT (object)) {
    GST_OBJECT_LOCK (object);
    (void) GST_OBJECT_NAME (object);
    GST_OBJECT_UNLOCK (object);
  }
}

/**
 * @brief The fixture a test case drives the element with
 */
typedef struct {
  GstElement *pipeline;
  GstElement *src;
  GstPad *srcpad;
  gulong probe_id;
  GstBus *bus;

  GMutex lock;
  GCond cond;
  guint num_buffers;
  guint num_memories;
  GstCaps *last_caps;
} src_fixture_s;

/**
 * @brief Count the buffers the element pushes and keep the caps they carry
 */
static GstPadProbeReturn
_buffer_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  src_fixture_s *fixture = (src_fixture_s *) user_data;
  GstCaps *caps = gst_pad_get_current_caps (pad);

  g_mutex_lock (&fixture->lock);
  fixture->num_buffers++;
  fixture->num_memories = gst_buffer_n_memory (GST_PAD_PROBE_INFO_BUFFER (info));
  if (caps)
    gst_caps_replace (&fixture->last_caps, caps);
  g_cond_broadcast (&fixture->cond);
  g_mutex_unlock (&fixture->lock);

  if (caps)
    gst_caps_unref (caps);

  return GST_PAD_PROBE_OK;
}

/**
 * @brief Build a pipeline that feeds the element's output into a fakesink
 */
static gboolean
_fixture_setup (src_fixture_s *fixture, const gchar *topic, const gchar *filter_caps)
{
  gchar *desc;

  memset (fixture, 0, sizeof (*fixture));
  g_mutex_init (&fixture->lock);
  g_cond_init (&fixture->cond);

  _mock_reset ();

  desc = g_strdup_printf ("mqttsrc name=src sub-topic=%s sub-timeout=%" G_GINT64_FORMAT
                          " ! capsfilter caps=%s ! fakesink sync=false",
      topic, DEFAULT_SUB_TIMEOUT, filter_caps);
  fixture->pipeline = gst_parse_launch (desc, NULL);
  g_free (desc);
  if (!fixture->pipeline)
    return FALSE;

  fixture->src = gst_bin_get_by_name (GST_BIN (fixture->pipeline), "src");
  if (!fixture->src)
    return FALSE;

  fixture->srcpad = gst_element_get_static_pad (fixture->src, "src");
  fixture->probe_id = gst_pad_add_probe (
      fixture->srcpad, GST_PAD_PROBE_TYPE_BUFFER, _buffer_probe, fixture, NULL);
  fixture->bus = gst_element_get_bus (fixture->pipeline);

  return TRUE;
}

/**
 * @brief Tell the element that the connection to the broker is gone
 */
static void
_drop_connection (void)
{
  MQTTAsync_connectionLost *cb;
  void *context;

  g_mutex_lock (&g_mock.lock);
  cb = g_mock.connection_lost;
  context = g_mock.context;
  g_mutex_unlock (&g_mock.lock);

  if (cb)
    cb (context, NULL);
}

/**
 * @brief Release everything the fixture holds
 */
static void
_fixture_teardown (src_fixture_s *fixture)
{
  if (fixture->pipeline) {
    /**
     * create () waits for the subscription with no way out other than an
     * error, so an element torn down while it waits would never stop its
     * streaming thread. Report the connection as lost first, which is what a
     * broker going away does.
     */
    _drop_connection ();
    gst_element_set_state (fixture->pipeline, GST_STATE_NULL);
    gst_element_get_state (fixture->pipeline, NULL, NULL, GST_CLOCK_TIME_NONE);
  }

  _mock_join_threads ();

  if (fixture->probe_id)
    gst_pad_remove_probe (fixture->srcpad, fixture->probe_id);
  if (fixture->srcpad)
    gst_object_unref (fixture->srcpad);
  if (fixture->src)
    gst_object_unref (fixture->src);
  if (fixture->bus)
    gst_object_unref (fixture->bus);
  if (fixture->pipeline)
    gst_object_unref (fixture->pipeline);

  gst_caps_replace (&fixture->last_caps, NULL);
  g_mutex_clear (&fixture->lock);
  g_cond_clear (&fixture->cond);
}

/**
 * @brief Get the number of buffers the element has pushed so far
 */
static guint
_fixture_num_buffers (src_fixture_s *fixture)
{
  guint num;

  g_mutex_lock (&fixture->lock);
  num = fixture->num_buffers;
  g_mutex_unlock (&fixture->lock);

  return num;
}

/**
 * @brief Get the number of memory blocks the last pushed buffer carries
 */
static guint
_fixture_num_memories (src_fixture_s *fixture)
{
  guint num;

  g_mutex_lock (&fixture->lock);
  num = fixture->num_memories;
  g_mutex_unlock (&fixture->lock);

  return num;
}

/**
 * @brief Wait until the element has pushed the given number of buffers
 */
static gboolean
_fixture_wait_buffers (src_fixture_s *fixture, guint expected)
{
  gint64 end_time = g_get_monotonic_time () + WAIT_TIMEOUT_US;
  gboolean reached;

  g_mutex_lock (&fixture->lock);
  while (fixture->num_buffers < expected) {
    if (!g_cond_wait_until (&fixture->cond, &fixture->lock, end_time))
      break;
  }
  reached = (fixture->num_buffers >= expected);
  g_mutex_unlock (&fixture->lock);

  return reached;
}

/**
 * @brief Take a copy of the caps the last pushed buffer was sent with
 */
static GstCaps *
_fixture_last_caps (src_fixture_s *fixture)
{
  GstCaps *caps = NULL;

  g_mutex_lock (&fixture->lock);
  if (fixture->last_caps)
    caps = gst_caps_ref (fixture->last_caps);
  g_mutex_unlock (&fixture->lock);

  return caps;
}

/**
 * @brief Start the pipeline and wait until the element has subscribed
 */
static gboolean
_fixture_play (src_fixture_s *fixture)
{
  GstStateChangeReturn ret;

  ret = gst_element_set_state (fixture->pipeline, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE)
    return FALSE;

  return _wait_for_subscribe ();
}

/**
 * @brief Fill the timestamp fields of a header so that the element keeps the buffer
 */
static void
_set_header_timestamps (GstElement *elm, GstMQTTMessageHdr *hdr)
{
  GstClockTimeDiff diff;
  GstClockTime base_time;
  GstClockTime cur_time;
  GstClock *clock;

  clock = gst_element_get_clock (elm);
  base_time = gst_element_get_base_time (elm) + GST_SECOND;
  cur_time = clock ? gst_clock_get_time (clock) : 0;
  if (clock)
    gst_object_unref (clock);

  diff = GST_CLOCK_DIFF (base_time, cur_time);
  hdr->base_time_epoch = g_get_real_time () * GST_US_TO_NS_MULTIPLIER - diff;
  hdr->sent_time_epoch = hdr->base_time_epoch + GST_SECOND;
  hdr->pts = 0;
  hdr->dts = 0;
  hdr->duration = 500 * GST_MSECOND;
}

/**
 * @brief Put a caps string into the header field without a terminator of its own
 */
static void
_set_header_caps (GstMQTTMessageHdr *hdr, const gchar *caps_str, const gchar *trailer)
{
  gsize len = strlen (caps_str);
  gsize field_offset = (gsize) ((guint8 *) hdr->gst_caps_str - (guint8 *) hdr);
  guint8 *field_end = hdr->_reserved_hdr + field_offset + GST_MQTT_MAX_LEN_GST_CAPS_STR;

  memset (hdr->gst_caps_str, '\0', GST_MQTT_MAX_LEN_GST_CAPS_STR);
  memcpy (hdr->gst_caps_str, caps_str, MIN (len, GST_MQTT_MAX_LEN_GST_CAPS_STR));

  if (trailer) {
    gsize room = GST_MQTT_LEN_MSG_HDR - field_offset - GST_MQTT_MAX_LEN_GST_CAPS_STR;

    memcpy (field_end, trailer, MIN (strlen (trailer) + 1, room));
  }
}

/**
 * @brief Build a message the way mqttsink would put it on the wire
 */
static MQTTAsync_message *
_new_message (const GstMQTTMessageHdr *hdr, gsize data_len)
{
  MQTTAsync_message *msg = (MQTTAsync_message *) g_malloc0 (sizeof (*msg));

  msg->payloadlen = (int) (GST_MQTT_LEN_MSG_HDR + data_len);
  msg->payload = g_malloc0 (msg->payloadlen);
  memcpy (msg->payload, hdr, GST_MQTT_LEN_MSG_HDR);

  return msg;
}

/**
 * @brief Hand a message to the element the way the paho client thread would
 */
static int
_deliver (MQTTAsync_message *msg)
{
  MQTTAsync_messageArrived *cb;
  void *context;

  g_mutex_lock (&g_mock.lock);
  cb = g_mock.message_arrived;
  context = g_mock.context;
  g_mutex_unlock (&g_mock.lock);

  if (!cb)
    return 0;

  return cb (context, g_strdup ("test_topic"), 0, msg);
}

/**
 * @brief Get the number of messages the element has released
 */
static guint
_freed_messages (void)
{
  guint num;

  g_mutex_lock (&g_mock.lock);
  num = g_mock.free_message_count;
  g_mutex_unlock (&g_mock.lock);

  return num;
}

/**
 * @brief Get the number of topic names the element has released
 */
static guint
_freed_topics (void)
{
  guint num;

  g_mutex_lock (&g_mock.lock);
  num = g_mock.free_topic_count;
  g_mutex_unlock (&g_mock.lock);

  return num;
}

/**
 * @brief Build a caps string that fills the whole header field, with no room for a terminator
 */
static gchar *
_full_length_caps_str (void)
{
  GString *str = g_string_new ("video/x-raw,format=RGB,width=640,height=320,pad=(string)");

  while (str->len < GST_MQTT_MAX_LEN_GST_CAPS_STR)
    g_string_append_c (str, 'A');

  return g_string_free (str, FALSE);
}

/**
 * @brief mqttsrc pushes a buffer built from a well-formed message
 */
TEST (testMqttSrc, messageArrived)
{
  src_fixture_s fixture;
  GstMQTTMessageHdr hdr = {};
  MQTTAsync_message *msg;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  _set_header_timestamps (fixture.src, &hdr);
  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = 2;
  hdr.size_mems[0] = 512;
  hdr.size_mems[1] = 512;
  msg = _new_message (&hdr, 1024);

  EXPECT_TRUE (_deliver (msg));
  EXPECT_TRUE (_fixture_wait_buffers (&fixture, 1));

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc refuses a message that declares more memory blocks than its header holds
 */
TEST (testMqttSrc, messageTooManyMems_n)
{
  src_fixture_s fixture;
  GstMQTTMessageHdr hdr = {};
  MQTTAsync_message *msg;
  guint i;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  _set_header_timestamps (fixture.src, &hdr);
  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = GST_MQTT_MAX_NUM_MEMS + 1;
  for (i = 0; i < GST_MQTT_MAX_NUM_MEMS; ++i)
    hdr.size_mems[i] = 16;
  msg = _new_message (&hdr, 16 * (GST_MQTT_MAX_NUM_MEMS + 1));

  EXPECT_TRUE (_deliver (msg));
  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (_fixture_num_buffers (&fixture), 0U);
  EXPECT_EQ (_freed_messages (), 1U);

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc takes a message that declares as many memory blocks as fit its header
 */
TEST (testMqttSrc, messageWithMaxMems)
{
  src_fixture_s fixture;
  GstMQTTMessageHdr hdr = {};
  MQTTAsync_message *msg;
  guint i;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  _set_header_timestamps (fixture.src, &hdr);
  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = GST_MQTT_MAX_NUM_MEMS;
  for (i = 0; i < GST_MQTT_MAX_NUM_MEMS; ++i)
    hdr.size_mems[i] = 64;
  msg = _new_message (&hdr, 64 * GST_MQTT_MAX_NUM_MEMS);

  EXPECT_TRUE (_deliver (msg));
  ASSERT_TRUE (_fixture_wait_buffers (&fixture, 1));
  EXPECT_EQ (_fixture_num_memories (&fixture), (guint) GST_MQTT_MAX_NUM_MEMS);

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc refuses a message whose memory block reaches past its payload
 */
TEST (testMqttSrc, messageMemPastPayload_n)
{
  src_fixture_s fixture;
  GstMQTTMessageHdr hdr = {};
  MQTTAsync_message *msg;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  _set_header_timestamps (fixture.src, &hdr);
  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = 1;
  hdr.size_mems[0] = 2048;
  msg = _new_message (&hdr, 512);

  EXPECT_TRUE (_deliver (msg));
  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (_fixture_num_buffers (&fixture), 0U);
  EXPECT_EQ (_freed_messages (), 1U);

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc refuses a message whose second memory block reaches past its payload
 */
TEST (testMqttSrc, messageLastMemPastPayload_n)
{
  src_fixture_s fixture;
  GstMQTTMessageHdr hdr = {};
  MQTTAsync_message *msg;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  _set_header_timestamps (fixture.src, &hdr);
  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = 2;
  hdr.size_mems[0] = 512;
  hdr.size_mems[1] = 513;
  msg = _new_message (&hdr, 1024);

  EXPECT_TRUE (_deliver (msg));
  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (_fixture_num_buffers (&fixture), 0U);
  EXPECT_EQ (_freed_messages (), 1U);

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc refuses a message that is shorter than the header it carries
 */
TEST (testMqttSrc, messageShorterThanHeader_n)
{
  src_fixture_s fixture;
  MQTTAsync_message *msg;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  msg = (MQTTAsync_message *) g_malloc0 (sizeof (*msg));
  msg->payloadlen = GST_MQTT_LEN_MSG_HDR / 2;
  msg->payload = g_malloc0 (msg->payloadlen);

  EXPECT_TRUE (_deliver (msg));
  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (_fixture_num_buffers (&fixture), 0U);
  EXPECT_EQ (_freed_messages (), 1U);

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc parses no more than the caps field of a header that has no terminator
 */
TEST (testMqttSrc, messageCapsWithoutTerminator)
{
  src_fixture_s fixture;
  GstMQTTMessageHdr hdr = {};
  MQTTAsync_message *msg;
  gchar *caps_str = _full_length_caps_str ();
  GstCaps *caps;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  _set_header_timestamps (fixture.src, &hdr);
  _set_header_caps (&hdr, caps_str, ",framerate=(fraction)30/1");
  hdr.num_mems = 1;
  hdr.size_mems[0] = 512;
  msg = _new_message (&hdr, 512);

  EXPECT_TRUE (_deliver (msg));
  ASSERT_TRUE (_fixture_wait_buffers (&fixture, 1));

  caps = _fixture_last_caps (&fixture);
  ASSERT_TRUE (caps != NULL);
  EXPECT_FALSE (gst_structure_has_field (gst_caps_get_structure (caps, 0), "framerate"));
  EXPECT_STREQ (gst_structure_get_string (gst_caps_get_structure (caps, 0), "pad"),
      caps_str + strlen ("video/x-raw,format=RGB,width=640,height=320,pad=(string)"));
  gst_caps_unref (caps);

  g_free (caps_str);
  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc keeps its caps when the header carries a caps string it cannot parse
 */
TEST (testMqttSrc, messageCapsUnparsable_n)
{
  src_fixture_s fixture;
  GstMQTTMessageHdr hdr = {};
  MQTTAsync_message *msg;
  GstCaps *caps;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  _set_header_timestamps (fixture.src, &hdr);
  _set_header_caps (&hdr, ",,,,", NULL);
  hdr.num_mems = 1;
  hdr.size_mems[0] = 512;
  msg = _new_message (&hdr, 512);

  EXPECT_TRUE (_deliver (msg));
  ASSERT_TRUE (_fixture_wait_buffers (&fixture, 1));

  caps = _fixture_last_caps (&fixture);
  EXPECT_TRUE (caps == NULL || gst_caps_is_any (caps));
  if (caps)
    gst_caps_unref (caps);

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc releases the message and the topic name of every message it takes
 */
TEST (testMqttSrc, messageReleased)
{
  src_fixture_s fixture;
  GstMQTTMessageHdr hdr = {};
  MQTTAsync_message *msg;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  _set_header_timestamps (fixture.src, &hdr);
  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = 1;
  hdr.size_mems[0] = 1024;
  msg = _new_message (&hdr, 1024);

  EXPECT_TRUE (_deliver (msg));
  ASSERT_TRUE (_fixture_wait_buffers (&fixture, 1));
  EXPECT_EQ (_freed_topics (), 1U);

  /** the message outlives the callback: it is released with the buffer */
  gst_element_set_state (fixture.pipeline, GST_STATE_NULL);
  gst_element_get_state (fixture.pipeline, NULL, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (_freed_messages (), 1U);

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc releases a message that arrives while it is not subscribed
 */
TEST (testMqttSrc, messageWhileNotSubscribed_n)
{
  src_fixture_s fixture;
  GstMQTTMessageHdr hdr = {};
  MQTTAsync_message *msg;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));

  g_mutex_lock (&g_mock.lock);
  g_mock.fail_subscribe = TRUE;
  g_mutex_unlock (&g_mock.lock);

  ASSERT_TRUE (_fixture_play (&fixture));

  _set_header_timestamps (fixture.src, &hdr);
  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = 1;
  hdr.size_mems[0] = 1024;
  msg = _new_message (&hdr, 1024);

  EXPECT_TRUE (_deliver (msg));
  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (_fixture_num_buffers (&fixture), 0U);
  EXPECT_EQ (_freed_messages (), 1U);
  EXPECT_EQ (_freed_topics (), 1U);

  _fixture_teardown (&fixture);
}

/**
 * @brief The topic mqttsrc subscribes with survives a sub-topic replaced during the call
 */
TEST (testMqttSrc, subTopicReplacedDuringSubscribe)
{
  src_fixture_s fixture;
  std::string subscribed;

  ASSERT_TRUE (_fixture_setup (&fixture, "topic_aaaaaaaaaaaa", "video/x-raw"));

  g_mutex_lock (&g_mock.lock);
  g_mock.async_connect = TRUE;
  g_mutex_unlock (&g_mock.lock);

  _hold_arm (HOLD_SUBSCRIBE);
  EXPECT_NE (gst_element_set_state (fixture.pipeline, GST_STATE_PLAYING),
      GST_STATE_CHANGE_FAILURE);
  ASSERT_TRUE (_hold_wait_entered ());

  g_object_set (fixture.src, "sub-topic", "topic_bbbbbbbbbbbb", NULL);
  _hold_release ();

  ASSERT_TRUE (_wait_for_subscribe ());
  g_mutex_lock (&g_mock.lock);
  subscribed = g_mock.last_sub_topic;
  g_mutex_unlock (&g_mock.lock);

  EXPECT_STREQ (subscribed.c_str (), "topic_aaaaaaaaaaaa");

  _fixture_teardown (&fixture);
}

/**
 * @brief The client id and the host mqttsrc starts with survive a replacement during the call
 */
TEST (testMqttSrc, clientIdReplacedDuringStart)
{
  src_fixture_s fixture;
  std::string client_id;
  std::string server_uri;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  g_object_set (fixture.src, "client-id", "client_aaaaaaaaaaaa", "host",
      "host_aaaaaaaaaaaa", "port", "1111", NULL);

  _hold_arm (HOLD_CREATE);
  std::thread starter (gst_element_set_state, fixture.pipeline, GST_STATE_PLAYING);
  ASSERT_TRUE (_hold_wait_entered ());

  g_object_set (fixture.src, "client-id", "client_bbbbbbbbbbbb", "host",
      "host_bbbbbbbbbbbb", "port", "2222", NULL);
  _hold_release ();
  starter.join ();

  g_mutex_lock (&g_mock.lock);
  client_id = g_mock.last_client_id;
  server_uri = g_mock.last_server_uri;
  g_mutex_unlock (&g_mock.lock);

  EXPECT_STREQ (client_id.c_str (), "client_aaaaaaaaaaaa");
  EXPECT_STREQ (server_uri.c_str (), "host_aaaaaaaaaaaa:1111");

  _fixture_teardown (&fixture);
}

/**
 * @brief The string properties can be read and replaced while the element streams
 */
TEST (testMqttSrc, propertiesReplacedWhileStreaming)
{
  src_fixture_s fixture;
  GstMQTTMessageHdr hdr = {};
  guint i;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));
  g_object_set (fixture.src, "debug", TRUE, NULL);

  /**
   * The element logs the topic on its streaming thread and on the paho thread.
   * The default log function reads the object's name under its lock, so a log
   * call made while the object lock is held deadlocks rather than merely
   * printing; this log function does the same. It stays installed once the
   * case is over: removing a log function orphans the list node that GStreamer
   * keeps, which a memory checker then reports as a leak of this file.
   */
  gst_debug_add_log_function (_path_log_func, NULL, NULL);
  gst_debug_set_threshold_for_name (GST_MQTT_ELEM_NAME_SRC, GST_LEVEL_DEBUG);

  _set_header_timestamps (fixture.src, &hdr);
  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = 1;
  hdr.size_mems[0] = 512;

  for (i = 0; i < 64; ++i) {
    gchar *topic = g_strdup_printf ("topic_%04u_padding_padding", i);
    gchar *readback = NULL;

    g_object_set (fixture.src, "sub-topic", topic, "client-id", topic, "host",
        topic, "port", topic, NULL);
    g_object_get (fixture.src, "sub-topic", &readback, NULL);
    EXPECT_TRUE (readback != NULL);
    g_free (readback);
    g_free (topic);

    EXPECT_TRUE (_deliver (_new_message (&hdr, 512)));
  }

  EXPECT_TRUE (_fixture_wait_buffers (&fixture, 1));

  gst_debug_set_threshold_for_name (GST_MQTT_ELEM_NAME_SRC, GST_LEVEL_NONE);

  _fixture_teardown (&fixture);
}

/**
 * @brief The topic mqttsrc unsubscribes with is the one the property held
 */
TEST (testMqttSrc, unsubscribeFailure_n)
{
  src_fixture_s fixture;
  std::string unsubscribed;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  g_mutex_lock (&g_mock.lock);
  g_mock.fail_unsubscribe = TRUE;
  g_mutex_unlock (&g_mock.lock);

  EXPECT_NE (gst_element_set_state (fixture.pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);

  g_mutex_lock (&g_mock.lock);
  unsubscribed = g_mock.last_unsub_topic;
  g_mutex_unlock (&g_mock.lock);

  EXPECT_STREQ (unsubscribed.c_str (), "test_topic");

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc refuses to play again when it cannot reconnect to the broker
 */
TEST (testMqttSrc, reconnectFailure_n)
{
  src_fixture_s fixture;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  EXPECT_NE (gst_element_set_state (fixture.pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);

  g_mutex_lock (&g_mock.lock);
  g_mock.fail_reconnect = TRUE;
  g_mutex_unlock (&g_mock.lock);
  _drop_connection ();

  EXPECT_EQ (gst_element_set_state (fixture.pipeline, GST_STATE_PLAYING),
      GST_STATE_CHANGE_FAILURE);

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc fails to start when the client cannot be created
 */
TEST (testMqttSrc, startFailure_n)
{
  src_fixture_s fixture;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));

  g_mutex_lock (&g_mock.lock);
  g_mock.fail_create = TRUE;
  g_mutex_unlock (&g_mock.lock);

  EXPECT_EQ (gst_element_set_state (fixture.pipeline, GST_STATE_PLAYING),
      GST_STATE_CHANGE_FAILURE);

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc fails to start when it cannot connect to the broker
 */
TEST (testMqttSrc, connectFailure_n)
{
  src_fixture_s fixture;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));

  g_mutex_lock (&g_mock.lock);
  g_mock.fail_connect = TRUE;
  g_mutex_unlock (&g_mock.lock);

  EXPECT_EQ (gst_element_set_state (fixture.pipeline, GST_STATE_PLAYING),
      GST_STATE_CHANGE_FAILURE);

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc drops a message that carries no timestamp it can use
 */
TEST (testMqttSrc, messageFromThePastDropped_n)
{
  src_fixture_s fixture;
  GstMQTTMessageHdr hdr = {};
  MQTTAsync_message *msg;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));
  g_object_set (fixture.src, "debug", TRUE, NULL);

  _set_header_timestamps (fixture.src, &hdr);
  hdr.sent_time_epoch = 0;
  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = 1;
  hdr.size_mems[0] = 512;
  msg = _new_message (&hdr, 512);

  EXPECT_TRUE (_deliver (msg));
  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (_fixture_num_buffers (&fixture), 0U);

  _fixture_teardown (&fixture);
}

/**
 * @brief The string properties read back what was set into them
 */
TEST (testMqttSrc, getSetStringProperties)
{
  GstElement *elm = gst_element_factory_make ("mqttsrc", NULL);
  gchar *value = NULL;

  ASSERT_TRUE (elm != NULL);

  g_object_set (elm, "sub-topic", "a_topic", "client-id", "an_id", "host",
      "a_host", "port", "1234", NULL);

  g_object_get (elm, "sub-topic", &value, NULL);
  EXPECT_STREQ (value, "a_topic");
  g_free (value);

  g_object_get (elm, "client-id", &value, NULL);
  EXPECT_STREQ (value, "an_id");
  g_free (value);

  g_object_get (elm, "host", &value, NULL);
  EXPECT_STREQ (value, "a_host");
  g_free (value);

  g_object_get (elm, "port", &value, NULL);
  EXPECT_STREQ (value, "1234");
  g_free (value);

  gst_object_unref (elm);
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

  g_mutex_init (&g_mock.lock);
  g_cond_init (&g_mock.cond);
  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
