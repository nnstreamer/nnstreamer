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
#include <glib/gstdio.h>
#include <gst/gst.h>

#include <MQTTAsync.h>
#include <errno.h>
#include <string.h>

#include <initializer_list>
#include <string>
#include <thread>
#include <vector>

#include "mqttcommon.h"
#include "mqttsrc.h"

#define DEFAULT_SUB_TIMEOUT (G_GINT64_CONSTANT (1000000))
#define SETTLE_TIME_US (300 * G_TIME_SPAN_MILLISECOND)
#define WAIT_TIMEOUT_US (5 * G_TIME_SPAN_SECOND)
#define LONG_SUB_TIMEOUT (G_GINT64_CONSTANT (10000000))
#define QUEUE_POLL_PASSED_US (1500 * G_TIME_SPAN_MILLISECOND)

/** TLS settings copied synchronously when the mocked client connects. */
static bool last_connect_tls;
static int last_connect_verify;
static int last_connect_auth;
static std::string last_connect_ca;
static std::string last_connect_cert;
static std::string last_connect_key;
static std::string last_connect_key_password;

/** @brief Record options while they are valid, without borrowing their storage. */
static void
_record_tls_options (const MQTTAsync_connectOptions *options)
{
  const MQTTAsync_SSLOptions *ssl = options->ssl;
  last_connect_tls = ssl != nullptr;
  last_connect_verify = ssl ? ssl->verify : 0;
  last_connect_auth = ssl ? ssl->enableServerCertAuth : 0;
  last_connect_ca = ssl && ssl->trustStore ? ssl->trustStore : "";
  last_connect_cert = ssl && ssl->keyStore ? ssl->keyStore : "";
  last_connect_key = ssl && ssl->privateKey ? ssl->privateKey : "";
  last_connect_key_password = ssl && ssl->privateKeyPassword ? ssl->privateKeyPassword : "";
}

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
  gboolean fail_disconnect;
  gboolean omit_disconnect_callback;
  MQTTAsync_onSuccess *disconnect_callback;
  void *disconnect_context;
  gboolean deliver_during_destroy;
  gboolean destroy_callback_done;
  gboolean destroy_drained_callback;
  guint destroy_count;
  gboolean model_disconnect_lock;
  gboolean receive_lock_held;
  gboolean disconnect_entered;
  gboolean disconnect_acquired_lock;

  std::string last_client_id;
  std::string last_server_uri;
  gboolean has_username;
  std::string last_username;
  gboolean has_password;
  std::string last_password;
  std::string required_password;
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
static GMutex mock_paho_mutex;
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
  g_mock.fail_disconnect = FALSE;
  g_mock.omit_disconnect_callback = FALSE;
  g_mock.disconnect_callback = NULL;
  g_mock.disconnect_context = NULL;
  g_mock.deliver_during_destroy = FALSE;
  g_mock.destroy_callback_done = FALSE;
  g_mock.destroy_drained_callback = FALSE;
  g_mock.destroy_count = 0;
  g_mock.model_disconnect_lock = FALSE;
  g_mock.receive_lock_held = FALSE;
  g_mock.disconnect_entered = FALSE;
  g_mock.disconnect_acquired_lock = FALSE;
  g_mock.last_client_id.clear ();
  g_mock.last_server_uri.clear ();
  g_mock.has_username = FALSE;
  g_mock.last_username.clear ();
  g_mock.has_password = FALSE;
  g_mock.last_password.clear ();
  g_mock.required_password.clear ();
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
  _record_tls_options (options);
  gboolean async;
  gboolean fail;

  (void) handle;

  g_mutex_lock (&g_mock.lock);
  async = g_mock.async_connect;
  fail = g_mock.fail_connect;
  g_mock.has_username = (options->username != NULL);
  g_mock.last_username.assign (options->username ? options->username : "");
  g_mock.has_password = (options->password != NULL);
  g_mock.last_password.assign (options->password ? options->password : "");
  if (!g_mock.required_password.empty ()
      && (options->password == NULL || g_mock.required_password != options->password))
    fail = TRUE;
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
  gboolean fail, omit_callback, model_lock;

  g_mutex_lock (&g_mock.lock);
  fail = g_mock.fail_disconnect;
  omit_callback = g_mock.omit_disconnect_callback;
  model_lock = g_mock.model_disconnect_lock;
  g_mock.disconnect_entered = TRUE;
  g_cond_broadcast (&g_mock.cond);
  g_mock.disconnect_callback = options ? options->onSuccess : NULL;
  g_mock.disconnect_context = options ? options->context : NULL;
  g_mutex_unlock (&g_mock.lock);

  if (model_lock) {
    gint64 deadline = g_get_monotonic_time () + 2 * G_TIME_SPAN_SECOND;
    gboolean acquired;
    /** Paho 1.3.14 takes its library lock before checking the request.
     * Bound this model so a regression fails instead of hanging the suite. */
    while (!(acquired = g_mutex_trylock (&mock_paho_mutex)) && g_get_monotonic_time () < deadline)
      g_usleep (1000);
    g_mutex_lock (&g_mock.lock);
    g_mock.disconnect_acquired_lock = acquired;
    g_mutex_unlock (&g_mock.lock);
    if (!acquired)
      return MQTTASYNC_FAILURE;
    g_mutex_unlock (&mock_paho_mutex);
  }

  if (fail)
    return MQTTASYNC_FAILURE;

  if (!options || omit_callback)
    return MQTTASYNC_SUCCESS;

  /* A fast completion must remain visible when disconnect returns. */
  if (model_lock)
    _run_success_cb (options->onSuccess, options->context);
  else
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
 * @brief Deliver a message whose callback must finish before client destruction.
 */
static void
_deliver_during_destroy (MQTTAsync_messageArrived *callback, void *context,
    MQTTAsync_onSuccess *disconnect_callback, void *disconnect_context)
{
  MQTTAsync_message *message = g_new0 (MQTTAsync_message, 1);
  GstMQTTMessageHdr *header;

  /* A disconnect completion can arrive after the caller's wait timed out. */
  if (disconnect_callback)
    _run_success_cb (disconnect_callback, disconnect_context);

  message->payloadlen = GST_MQTT_LEN_MSG_HDR + 1;
  message->payload = g_malloc0 (message->payloadlen);
  header = (GstMQTTMessageHdr *) message->payload;
  header->num_mems = 1;
  header->size_mems[0] = 1;
  g_strlcpy (header->gst_caps_str, "application/octet-stream", sizeof (header->gst_caps_str));
  callback (context, g_strdup ("test_topic"), 0, message);

  g_mutex_lock (&g_mock.lock);
  g_mock.destroy_callback_done = TRUE;
  g_cond_broadcast (&g_mock.cond);
  g_mutex_unlock (&g_mock.lock);
}

/**
 * @brief Model paho destruction waiting for a receive callback to release its lock.
 * @details The bounded wait records a lock inversion without hanging the test
 *          process; the callback is joined after stop releases the element lock.
 */
void
MQTTAsync_destroy (MQTTAsync *handle)
{
  g_mutex_lock (&g_mock.lock);
  g_mock.destroy_count++;
  if (g_mock.deliver_during_destroy) {
    gint64 deadline = g_get_monotonic_time () + WAIT_TIMEOUT_US;

    g_mock.destroy_callback_done = FALSE;
    g_mock_threads.push_back (std::thread (_deliver_during_destroy, g_mock.message_arrived,
        g_mock.context, g_mock.omit_disconnect_callback ? g_mock.disconnect_callback : NULL,
        g_mock.disconnect_context));
    while (!g_mock.destroy_callback_done) {
      if (!g_cond_wait_until (&g_mock.cond, &g_mock.lock, deadline))
        break;
    }
    g_mock.destroy_drained_callback = g_mock.destroy_callback_done;
  }
  g_mutex_unlock (&g_mock.lock);
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
 * @brief Build a message shorter than the header it should carry
 */
static MQTTAsync_message *
_new_short_message (void)
{
  MQTTAsync_message *msg = (MQTTAsync_message *) g_malloc0 (sizeof (*msg));

  msg->payloadlen = GST_MQTT_LEN_MSG_HDR / 2;
  msg->payload = g_malloc0 (msg->payloadlen);

  return msg;
}

/**
 * @brief Build a message with no payload at all
 */
static MQTTAsync_message *
_new_empty_message (void)
{
  return (MQTTAsync_message *) g_malloc0 (sizeof (MQTTAsync_message));
}

/**
 * @brief Build a message that declares more memory blocks than its header holds
 */
static MQTTAsync_message *
_new_too_many_mems_message (void)
{
  GstMQTTMessageHdr hdr = {};
  guint i;

  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = GST_MQTT_MAX_NUM_MEMS + 1;
  for (i = 0; i < GST_MQTT_MAX_NUM_MEMS; ++i)
    hdr.size_mems[i] = 16;

  return _new_message (&hdr, 16 * (GST_MQTT_MAX_NUM_MEMS + 1));
}

/**
 * @brief Build a message whose memory block reaches past its payload
 */
static MQTTAsync_message *
_new_mem_past_payload_message (void)
{
  GstMQTTMessageHdr hdr = {};

  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = 1;
  hdr.size_mems[0] = 2048;

  return _new_message (&hdr, 512);
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

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  EXPECT_TRUE (_deliver (_new_too_many_mems_message ()));
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

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  EXPECT_TRUE (_deliver (_new_mem_past_payload_message ()));
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
 * @brief A message dropped for a memory block past its payload does not
 *        replace the caps of the stream
 */
TEST (testMqttSrc, messageMemPastPayloadKeepsCaps_n)
{
  src_fixture_s fixture;
  GstMQTTMessageHdr hdr = {};
  GstCaps *expected;
  GstMqttSrc *src;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));
  src = GST_MQTT_SRC (fixture.src);

  _set_header_timestamps (fixture.src, &hdr);
  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = 1;
  hdr.size_mems[0] = 512;
  EXPECT_TRUE (_deliver (_new_message (&hdr, 512)));
  ASSERT_TRUE (_fixture_wait_buffers (&fixture, 1));

  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=320,height=160", NULL);
  hdr.size_mems[0] = 2048;
  EXPECT_TRUE (_deliver (_new_message (&hdr, 512)));

  expected = gst_caps_from_string ("video/x-raw,format=RGB,width=640,height=320");
  ASSERT_TRUE (src->caps != NULL);
  EXPECT_TRUE (gst_caps_is_equal (src->caps, expected));
  gst_caps_unref (expected);

  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (_fixture_num_buffers (&fixture), 1U);

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc refuses a message that is shorter than the header it carries
 */
TEST (testMqttSrc, messageShorterThanHeader_n)
{
  src_fixture_s fixture;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  EXPECT_TRUE (_deliver (_new_short_message ()));
  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (_fixture_num_buffers (&fixture), 0U);
  EXPECT_EQ (_freed_messages (), 1U);

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc drops an empty payload before wrapping its null data pointer,
 *        without latching an error
 */
TEST (testMqttSrc, messageEmptyPayload_n)
{
  src_fixture_s fixture;
  GstMqttSrc *src;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  EXPECT_TRUE (_deliver (_new_empty_message ()));

  src = GST_MQTT_SRC (fixture.src);
  EXPECT_EQ (src->err, nullptr);
  EXPECT_EQ (_fixture_num_buffers (&fixture), 0U);
  EXPECT_EQ (_freed_messages (), 1U);
  EXPECT_EQ (_freed_topics (), 1U);

  _fixture_teardown (&fixture);
}

/**
 * @brief a non-null pointer does not make a zero-length payload a complete header
 */
TEST (testMqttSrc, messageEmptyAllocatedPayload_n)
{
  src_fixture_s fixture;
  MQTTAsync_message *msg;
  GstMqttSrc *src;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));

  msg = (MQTTAsync_message *) g_malloc0 (sizeof (*msg));
  msg->payload = g_malloc0 (1);
  EXPECT_TRUE (_deliver (msg));

  src = GST_MQTT_SRC (fixture.src);
  EXPECT_EQ (src->err, nullptr);
  EXPECT_EQ (_fixture_num_buffers (&fixture), 0U);
  EXPECT_EQ (_freed_messages (), 1U);
  EXPECT_EQ (_freed_topics (), 1U);

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
 * @brief Start mqttsrc in a new fixture with the given credentials and record what it gave to paho
 */
static gboolean
_fixture_start (src_fixture_s *fixture, const gchar *username,
    const gchar *password, const gchar *required_password)
{
  if (!_fixture_setup (fixture, "test_topic", "video/x-raw"))
    return FALSE;

  g_mutex_lock (&g_mock.lock);
  g_mock.required_password.assign (required_password ? required_password : "");
  g_mutex_unlock (&g_mock.lock);

  if (username)
    g_object_set (fixture->src, "username", username, NULL);
  if (password)
    g_object_set (fixture->src, "password", password, NULL);

  return (gst_element_set_state (fixture->pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE);
}

/**
 * @brief mqttsrc generates a random client id when client-id is not set
 */
TEST (testMqttSrc, defaultClientIdRandom)
{
  src_fixture_s fixture;
  std::string ids[2];
  gchar *prop = NULL;

  for (int i = 0; i < 2; i++) {
    ASSERT_TRUE (_fixture_start (&fixture, NULL, NULL, NULL));
    g_mutex_lock (&g_mock.lock);
    ids[i] = g_mock.last_client_id;
    g_mutex_unlock (&g_mock.lock);

    g_object_get (fixture.src, "client-id", &prop, NULL);
    EXPECT_STREQ (prop, ids[i].c_str ());
    g_free (prop);
    _fixture_teardown (&fixture);

    ASSERT_EQ (ids[i].size (), 22U);
    EXPECT_EQ (ids[i].compare (0, 6, "nnssrc"), 0);
    for (size_t c = 6; c < ids[i].size (); c++)
      EXPECT_TRUE (g_ascii_isdigit (ids[i][c]) || (ids[i][c] >= 'a' && ids[i][c] <= 'f'));
  }

  EXPECT_STRNE (ids[0].c_str (), ids[1].c_str ());
}

/**
 * @brief mqttsrc keeps the client id it generated when it starts again
 */
TEST (testMqttSrc, defaultClientIdKeptOnRestart)
{
  src_fixture_s fixture;
  std::string first;
  std::string second;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  EXPECT_NE (gst_element_set_state (fixture.pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  g_mutex_lock (&g_mock.lock);
  first = g_mock.last_client_id;
  g_mock.last_client_id.clear ();
  g_mutex_unlock (&g_mock.lock);

  EXPECT_EQ (gst_element_set_state (fixture.pipeline, GST_STATE_READY), GST_STATE_CHANGE_SUCCESS);
  EXPECT_NE (gst_element_set_state (fixture.pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  g_mutex_lock (&g_mock.lock);
  second = g_mock.last_client_id;
  g_mutex_unlock (&g_mock.lock);

  EXPECT_EQ (first.compare (0, 6, "nnssrc"), 0);
  EXPECT_STREQ (second.c_str (), first.c_str ());

  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc connects with the credentials set, and with none by default
 */
TEST (testMqttSrc, credentialsGivenToConnect)
{
  src_fixture_s fixture;
  gchar *username = NULL;

  ASSERT_TRUE (_fixture_start (&fixture, NULL, NULL, NULL));
  g_object_get (fixture.src, "username", &username, NULL);
  EXPECT_TRUE (username == NULL);
  g_mutex_lock (&g_mock.lock);
  EXPECT_FALSE (g_mock.has_username);
  EXPECT_FALSE (g_mock.has_password);
  g_mutex_unlock (&g_mock.lock);
  _fixture_teardown (&fixture);

  ASSERT_TRUE (_fixture_start (&fixture, "an-mqtt-user", "an-mqtt-password", NULL));
  g_object_get (fixture.src, "username", &username, NULL);
  EXPECT_STREQ (username, "an-mqtt-user");
  g_free (username);
  g_mutex_lock (&g_mock.lock);
  EXPECT_TRUE (g_mock.has_username);
  EXPECT_STREQ (g_mock.last_username.c_str (), "an-mqtt-user");
  EXPECT_TRUE (g_mock.has_password);
  EXPECT_STREQ (g_mock.last_password.c_str (), "an-mqtt-password");
  g_mutex_unlock (&g_mock.lock);
  _fixture_teardown (&fixture);
}

/**
 * @brief A new_client_id_func replacement for a system without a secure random source
 */
static gchar *
_no_random_client_id (const gchar *prefix)
{
  (void) prefix;
  return NULL;
}

/**
 * @brief mqttsrc refuses to start without a random source, and starts with an explicit client-id
 */
TEST (testMqttSrc, startWithoutRandomSource_n)
{
  src_fixture_s fixture;
  GstMessage *msg;
  GError *err = NULL;
  gchar *prop = NULL;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  GST_MQTT_SRC (fixture.src)->new_client_id_func = _no_random_client_id;

  EXPECT_EQ (gst_element_set_state (fixture.pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  EXPECT_LT (GST_STATE (fixture.src), GST_STATE_PAUSED);
  g_mutex_lock (&g_mock.lock);
  EXPECT_TRUE (g_mock.last_client_id.empty ());
  g_mutex_unlock (&g_mock.lock);

  msg = gst_bus_pop_filtered (fixture.bus, GST_MESSAGE_ERROR);
  ASSERT_TRUE (msg != NULL);
  gst_message_parse_error (msg, &err, NULL);
  EXPECT_TRUE (g_error_matches (err, GST_RESOURCE_ERROR, GST_RESOURCE_ERROR_FAILED));
  g_error_free (err);
  gst_message_unref (msg);

  /* No id is made up in place of the random one */
  g_object_get (fixture.src, "client-id", &prop, NULL);
  EXPECT_EQ (std::string (prop).find ("nnssrc"), std::string::npos);
  g_free (prop);

  g_object_set (fixture.src, "client-id", "an-explicit-client-id", NULL);
  EXPECT_NE (gst_element_set_state (fixture.pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  g_mutex_lock (&g_mock.lock);
  EXPECT_STREQ (g_mock.last_client_id.c_str (), "an-explicit-client-id");
  g_mutex_unlock (&g_mock.lock);

  _fixture_teardown (&fixture);
}

/**
 * @brief Credentials changed while mqttsrc is stopped apply from its next start
 */
TEST (testMqttSrc, credentialsClearedOnRestart)
{
  src_fixture_s fixture;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  g_object_set (fixture.src, "username", "an-mqtt-user", "password",
      "an-mqtt-password", NULL);
  EXPECT_NE (gst_element_set_state (fixture.pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  g_mutex_lock (&g_mock.lock);
  EXPECT_TRUE (g_mock.has_username);
  EXPECT_TRUE (g_mock.has_password);
  g_mutex_unlock (&g_mock.lock);

  EXPECT_EQ (gst_element_set_state (fixture.pipeline, GST_STATE_READY), GST_STATE_CHANGE_SUCCESS);
  g_object_set (fixture.src, "username", NULL, "password", NULL, NULL);
  EXPECT_NE (gst_element_set_state (fixture.pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  g_mutex_lock (&g_mock.lock);
  EXPECT_FALSE (g_mock.has_username);
  EXPECT_FALSE (g_mock.has_password);
  g_mutex_unlock (&g_mock.lock);

  _fixture_teardown (&fixture);
}

/**
 * @brief The password of mqttsrc cannot be read back
 */
TEST (testMqttSrc, passwordNotReadable_n)
{
  GstElement *elm = gst_element_factory_make ("mqttsrc", NULL);
  GParamSpec *pspec;

  ASSERT_TRUE (elm != NULL);
  pspec = g_object_class_find_property (G_OBJECT_GET_CLASS (elm), "password");
  ASSERT_TRUE (pspec != NULL);
  EXPECT_FALSE (pspec->flags & G_PARAM_READABLE);
  EXPECT_TRUE (pspec->flags & G_PARAM_WRITABLE);

  gst_object_unref (elm);
}

/**
 * @brief mqttsrc fails to start when the broker rejects the password, and starts with the right one
 */
TEST (testMqttSrc, credentialsRejected_n)
{
  src_fixture_s fixture;

  EXPECT_FALSE (_fixture_start (&fixture, "an-mqtt-user", NULL, "the-right-password"));
  g_mutex_lock (&g_mock.lock);
  EXPECT_TRUE (g_mock.has_username);
  EXPECT_FALSE (g_mock.has_password);
  g_mutex_unlock (&g_mock.lock);
  _fixture_teardown (&fixture);

  EXPECT_FALSE (_fixture_start (
      &fixture, "an-mqtt-user", "a-wrong-password", "the-right-password"));
  g_mutex_lock (&g_mock.lock);
  EXPECT_STREQ (g_mock.last_password.c_str (), "a-wrong-password");
  g_mutex_unlock (&g_mock.lock);
  _fixture_teardown (&fixture);

  EXPECT_TRUE (_fixture_start (
      &fixture, "an-mqtt-user", "the-right-password", "the-right-password"));
  _fixture_teardown (&fixture);
}

/**
 * @brief mqttsrc refuses to start with a password but no username, before creating a client
 */
TEST (testMqttSrc, passwordWithoutUsername_n)
{
  src_fixture_s fixture;
  GstMessage *msg;
  GError *err = NULL;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  g_object_set (fixture.src, "password", "an-mqtt-password", NULL);
  EXPECT_EQ (gst_element_set_state (fixture.pipeline, GST_STATE_PLAYING),
      GST_STATE_CHANGE_FAILURE);
  g_mutex_lock (&g_mock.lock);
  EXPECT_TRUE (g_mock.last_client_id.empty ());
  g_mutex_unlock (&g_mock.lock);

  msg = gst_bus_pop_filtered (fixture.bus, GST_MESSAGE_ERROR);
  ASSERT_TRUE (msg != NULL);
  gst_message_parse_error (msg, &err, NULL);
  EXPECT_TRUE (g_error_matches (err, GST_RESOURCE_ERROR, GST_RESOURCE_ERROR_SETTINGS));
  g_error_free (err);
  gst_message_unref (msg);
  _fixture_teardown (&fixture);

  /* The same password starts with a username */
  EXPECT_TRUE (_fixture_start (&fixture, "an-mqtt-user", "an-mqtt-password", NULL));
  g_mutex_lock (&g_mock.lock);
  EXPECT_STREQ (g_mock.last_password.c_str (), "an-mqtt-password");
  g_mutex_unlock (&g_mock.lock);
  _fixture_teardown (&fixture);
}

/**
 * @brief The timestamps of the buffers the element pushes
 */
typedef struct {
  GMutex lock;
  GCond cond;
  guint count;
  GstClockTime pts;
  GstClockTime dts;
  gulong probe_id;
} ts_record_s;

/**
 * @brief Keep the timestamps of the last buffer the element pushes
 */
static GstPadProbeReturn
_ts_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  ts_record_s *rec = (ts_record_s *) user_data;
  GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER (info);
  (void) pad;

  g_mutex_lock (&rec->lock);
  rec->count++;
  rec->pts = GST_BUFFER_PTS (buf);
  rec->dts = GST_BUFFER_DTS (buf);
  g_cond_broadcast (&rec->cond);
  g_mutex_unlock (&rec->lock);

  return GST_PAD_PROBE_OK;
}

/**
 * @brief Start recording the timestamps of the buffers the element pushes
 */
static void
_ts_record_start (ts_record_s *rec, src_fixture_s *fixture)
{
  memset (rec, 0, sizeof (*rec));
  g_mutex_init (&rec->lock);
  g_cond_init (&rec->cond);
  rec->pts = rec->dts = GST_CLOCK_TIME_NONE;
  rec->probe_id = gst_pad_add_probe (
      fixture->srcpad, GST_PAD_PROBE_TYPE_BUFFER, _ts_probe, rec, NULL);
}

/**
 * @brief Stop recording; call it after the fixture is torn down
 */
static void
_ts_record_stop (ts_record_s *rec)
{
  g_mutex_clear (&rec->lock);
  g_cond_clear (&rec->cond);
}

/**
 * @brief Wait until the given number of buffers has been recorded
 */
static gboolean
_ts_record_wait (ts_record_s *rec, guint expected)
{
  gint64 end_time = g_get_monotonic_time () + WAIT_TIMEOUT_US;
  gboolean reached;

  g_mutex_lock (&rec->lock);
  while (rec->count < expected) {
    if (!g_cond_wait_until (&rec->cond, &rec->lock, end_time))
      break;
  }
  reached = (rec->count >= expected);
  g_mutex_unlock (&rec->lock);

  return reached;
}

/**
 * @brief Get the epoch the element maps its running time onto
 */
static gint64
_local_epoch (src_fixture_s *fixture)
{
  return ((GstMqttSrc *) fixture->src)->base_time_epoch;
}

/**
 * @brief Build a message carrying the given wire-header timestamps
 */
static MQTTAsync_message *
_new_timed_message (gint64 base_time_epoch, gint64 sent_time_epoch,
    GstClockTime pts, GstClockTime dts)
{
  GstMQTTMessageHdr hdr = {};

  hdr.base_time_epoch = base_time_epoch;
  hdr.sent_time_epoch = sent_time_epoch;
  hdr.pts = pts;
  hdr.dts = dts;
  hdr.duration = 33 * GST_MSECOND;
  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = 1;
  hdr.size_mems[0] = 512;

  return _new_message (&hdr, 512);
}

/**
 * @brief Deliver a message from a publisher that went to PLAYING 5 s before
 *        the element, stamped 100 ms into the element's running time
 */
static gboolean
_deliver_valid_message (src_fixture_s *fixture)
{
  gint64 epoch = _local_epoch (fixture);

  return _deliver (_new_timed_message (
      epoch - 5 * GST_SECOND, epoch + 1, 5100 * GST_MSECOND, 5 * GST_SECOND));
}

/**
 * @brief mqttsrc moves wire timestamps onto its own running time unchanged
 * @note GstBaseSrc offsets a live stream so that its first buffer lands on the
 *       running time it is pushed at, so the cases check the PTS-DTS distance.
 */
TEST (testMqttSrc, messageTimestampsShifted)
{
  src_fixture_s fixture;
  ts_record_s rec;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  _ts_record_start (&rec, &fixture);
  ASSERT_TRUE (_fixture_play (&fixture));
  ASSERT_NE (_local_epoch (&fixture), (gint64) GST_CLOCK_TIME_NONE);

  EXPECT_TRUE (_deliver_valid_message (&fixture));
  EXPECT_TRUE (_ts_record_wait (&rec, 1));
  EXPECT_EQ (rec.pts - rec.dts, 100 * GST_MSECOND);

  _fixture_teardown (&fixture);
  _ts_record_stop (&rec);
}

/**
 * @brief mqttsrc drops a DTS that would fall before its running time starts
 *        instead of wrapping it into a far-future sync point
 */
TEST (testMqttSrc, messageDtsBeforeStart)
{
  src_fixture_s fixture;
  ts_record_s rec;
  gint64 epoch;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  _ts_record_start (&rec, &fixture);
  ASSERT_TRUE (_fixture_play (&fixture));
  epoch = _local_epoch (&fixture);

  EXPECT_TRUE (_deliver (_new_timed_message (
      epoch - 5 * GST_SECOND, epoch + 1, 5100 * GST_MSECOND, 4 * GST_SECOND)));
  EXPECT_TRUE (_ts_record_wait (&rec, 1));
  EXPECT_TRUE (GST_CLOCK_TIME_IS_VALID (rec.pts));
  EXPECT_EQ (rec.dts, GST_CLOCK_TIME_NONE);

  _fixture_teardown (&fixture);
  _ts_record_stop (&rec);
}

/**
 * @brief A publisher epoch whose distance from the local one overflows is
 *        dropped and does not hold back the messages after it
 */
TEST (testMqttSrc, messageBaseEpochOverflow_n)
{
  src_fixture_s fixture;
  ts_record_s rec;
  gint64 epoch;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  _ts_record_start (&rec, &fixture);
  ASSERT_TRUE (_fixture_play (&fixture));
  epoch = _local_epoch (&fixture);

  EXPECT_TRUE (_deliver (_new_timed_message (G_MININT64, epoch + 1, 0, 0)));
  EXPECT_TRUE (_deliver_valid_message (&fixture));
  EXPECT_TRUE (_ts_record_wait (&rec, 1));
  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (rec.count, 1U);
  EXPECT_EQ (rec.pts - rec.dts, 100 * GST_MSECOND);

  _fixture_teardown (&fixture);
  _ts_record_stop (&rec);
}

/**
 * @brief A PTS that overflows once moved onto the local running time is
 *        dropped and does not hold back the messages after it
 */
TEST (testMqttSrc, messagePtsOverflow_n)
{
  src_fixture_s fixture;
  ts_record_s rec;
  gint64 epoch;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  _ts_record_start (&rec, &fixture);
  ASSERT_TRUE (_fixture_play (&fixture));
  epoch = _local_epoch (&fixture);

  EXPECT_TRUE (_deliver (_new_timed_message (
      epoch + GST_SECOND, epoch + GST_SECOND, G_MAXINT64, GST_CLOCK_TIME_NONE)));
  EXPECT_TRUE (_deliver_valid_message (&fixture));
  EXPECT_TRUE (_ts_record_wait (&rec, 1));
  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (rec.count, 1U);
  EXPECT_EQ (rec.pts - rec.dts, 100 * GST_MSECOND);

  _fixture_teardown (&fixture);
  _ts_record_stop (&rec);
}

/**
 * @brief A PTS beyond the signed time range is refused instead of wrapping
 *        into a plausible-looking one
 */
TEST (testMqttSrc, messagePtsBeyondRange_n)
{
  src_fixture_s fixture;
  ts_record_s rec;
  gint64 epoch;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  _ts_record_start (&rec, &fixture);
  ASSERT_TRUE (_fixture_play (&fixture));
  epoch = _local_epoch (&fixture);

  EXPECT_TRUE (_deliver (_new_timed_message (epoch + 200 * GST_MSECOND,
      epoch + 1, G_MAXUINT64 - 1, GST_CLOCK_TIME_NONE)));
  EXPECT_TRUE (_deliver_valid_message (&fixture));
  EXPECT_TRUE (_ts_record_wait (&rec, 1));
  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (rec.count, 1U);
  EXPECT_EQ (rec.pts - rec.dts, 100 * GST_MSECOND);

  _fixture_teardown (&fixture);
  _ts_record_stop (&rec);
}

/**
 * @brief A message that arrives before the element knows its epoch is
 *        dropped and does not hold back the messages after it
 */
TEST (testMqttSrc, messageBeforePlaying_n)
{
  src_fixture_s fixture;
  ts_record_s rec;
  GstMQTTMessageHdr hdr = {};

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  _ts_record_start (&rec, &fixture);
  ASSERT_NE (gst_element_set_state (fixture.pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  ASSERT_TRUE (_wait_for_subscribe ());
  ASSERT_EQ (_local_epoch (&fixture), (gint64) GST_CLOCK_TIME_NONE);

  _set_header_timestamps (fixture.src, &hdr);
  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = 1;
  hdr.size_mems[0] = 512;
  EXPECT_TRUE (_deliver (_new_message (&hdr, 512)));

  ASSERT_TRUE (_fixture_play (&fixture));
  EXPECT_TRUE (_deliver_valid_message (&fixture));
  EXPECT_TRUE (_ts_record_wait (&rec, 1));
  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (rec.count, 1U);
  EXPECT_EQ (rec.pts - rec.dts, 100 * GST_MSECOND);

  _fixture_teardown (&fixture);
  _ts_record_stop (&rec);
}

/**
 * @brief A DTS beyond the signed time range is dropped on its own and the
 *        buffer keeps its PTS
 */
TEST (testMqttSrc, messageDtsBeyondRange)
{
  src_fixture_s fixture;
  ts_record_s rec;
  gint64 epoch;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  _ts_record_start (&rec, &fixture);
  ASSERT_TRUE (_fixture_play (&fixture));
  epoch = _local_epoch (&fixture);

  EXPECT_TRUE (_deliver (_new_timed_message (
      epoch - 5 * GST_SECOND, epoch + 1, 5100 * GST_MSECOND, G_MAXUINT64 - 1)));
  EXPECT_TRUE (_ts_record_wait (&rec, 1));
  EXPECT_TRUE (GST_CLOCK_TIME_IS_VALID (rec.pts));
  EXPECT_EQ (rec.dts, GST_CLOCK_TIME_NONE);

  _fixture_teardown (&fixture);
  _ts_record_stop (&rec);
}

/**
 * @brief A DTS that overflows once moved onto the local running time is
 *        dropped on its own and the buffer keeps its PTS
 */
TEST (testMqttSrc, messageDtsOverflow)
{
  src_fixture_s fixture;
  ts_record_s rec;
  gint64 epoch;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  _ts_record_start (&rec, &fixture);
  ASSERT_TRUE (_fixture_play (&fixture));
  epoch = _local_epoch (&fixture);

  EXPECT_TRUE (_deliver (
      _new_timed_message (epoch + 100 * GST_MSECOND, epoch + 1, 0, G_MAXINT64)));
  EXPECT_TRUE (_ts_record_wait (&rec, 1));
  EXPECT_TRUE (GST_CLOCK_TIME_IS_VALID (rec.pts));
  EXPECT_EQ (rec.dts, GST_CLOCK_TIME_NONE);

  _fixture_teardown (&fixture);
  _ts_record_stop (&rec);
}

/**
 * @brief A clockless pipeline never learns the local epoch; its messages
 *        still flow with the timestamps they always had
 */
TEST (testMqttSrc, messageClocklessPipeline)
{
  src_fixture_s fixture;
  ts_record_s rec;
  gint64 hdr_epoch = g_get_real_time () * GST_US_TO_NS_MULTIPLIER;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  _ts_record_start (&rec, &fixture);
  gst_pipeline_use_clock (GST_PIPELINE (fixture.pipeline), NULL);
  ASSERT_TRUE (_fixture_play (&fixture));
  ASSERT_EQ (_local_epoch (&fixture), (gint64) GST_CLOCK_TIME_NONE);

  EXPECT_TRUE (_deliver (
      _new_timed_message (hdr_epoch, hdr_epoch, 5100 * GST_MSECOND, 5 * GST_SECOND)));
  EXPECT_TRUE (_ts_record_wait (&rec, 1));
  EXPECT_EQ (rec.pts, 5100 * GST_MSECOND + hdr_epoch + 1);
  EXPECT_EQ (rec.dts, 5 * GST_SECOND + hdr_epoch + 1);

  _fixture_teardown (&fixture);
  _ts_record_stop (&rec);
}

/**
 * @brief A function that builds a malformed message
 */
typedef MQTTAsync_message *(*bad_message_builder) (void);

/**
 * @brief Deliver a malformed message, let the element find its queue empty at
 *        least once, then check that a valid message still goes through
 */
static void
_expect_stream_survives (bad_message_builder new_bad_message)
{
  src_fixture_s fixture;
  GstMessage *error;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  g_object_set (fixture.src, "sub-timeout", LONG_SUB_TIMEOUT, NULL);
  ASSERT_TRUE (_fixture_play (&fixture));

  EXPECT_TRUE (_deliver (new_bad_message ()));
  EXPECT_EQ (GST_MQTT_SRC (fixture.src)->err, nullptr);
  EXPECT_EQ (_freed_messages (), 1U);

  g_usleep (QUEUE_POLL_PASSED_US);
  EXPECT_EQ (_fixture_num_buffers (&fixture), 0U);
  EXPECT_TRUE (_deliver_valid_message (&fixture));
  EXPECT_TRUE (_fixture_wait_buffers (&fixture, 1));

  error = gst_bus_pop_filtered (fixture.bus, GST_MESSAGE_ERROR);
  EXPECT_TRUE (error == NULL);
  if (error)
    gst_message_unref (error);

  _fixture_teardown (&fixture);
}

/**
 * @brief A message shorter than its header does not end the stream
 */
TEST (testMqttSrc, messageShortThenValid_n)
{
  _expect_stream_survives (_new_short_message);
}

/**
 * @brief An empty message does not end the stream
 */
TEST (testMqttSrc, messageEmptyThenValid_n)
{
  _expect_stream_survives (_new_empty_message);
}

/**
 * @brief A message declaring too many memory blocks does not end the stream
 */
TEST (testMqttSrc, messageTooManyMemsThenValid_n)
{
  _expect_stream_survives (_new_too_many_mems_message);
}

/**
 * @brief A message whose memory block reaches past its payload does not end the stream
 */
TEST (testMqttSrc, messageMemPastPayloadThenValid_n)
{
  _expect_stream_survives (_new_mem_past_payload_message);
}

/**
 * @brief A malformed message does not hide a connection lost after it, which
 *        still stops the stream with an error
 */
TEST (testMqttSrc, connectionLostAfterBadMessage_n)
{
  src_fixture_s fixture;
  GstMessage *error;
  GstMqttSrc *src;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  g_object_set (fixture.src, "sub-timeout", LONG_SUB_TIMEOUT, NULL);
  ASSERT_TRUE (_fixture_play (&fixture));
  src = GST_MQTT_SRC (fixture.src);

  EXPECT_TRUE (_deliver (_new_short_message ()));
  EXPECT_EQ (src->err, nullptr);

  _drop_connection ();
  ASSERT_NE (src->err, nullptr);
  EXPECT_EQ (src->err->code, EHOSTDOWN);

  error = gst_bus_timed_pop_filtered (
      fixture.bus, WAIT_TIMEOUT_US * GST_USECOND, GST_MESSAGE_ERROR);
  EXPECT_TRUE (error != NULL);
  if (error)
    gst_message_unref (error);

  _fixture_teardown (&fixture);
}

/**
 * @brief The gate a test case holds the element's streaming thread with
 */
typedef struct {
  GMutex lock;
  GCond cond;
  gboolean entered;
  gboolean released;
  std::vector<gsize> sizes;
  std::vector<std::string> caps;
  gulong probe_id;
} flood_gate_s;

/**
 * @brief Record the size and the caps of a pushed buffer and hold it until the gate opens
 */
static GstPadProbeReturn
_flood_gate_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  flood_gate_s *gate = (flood_gate_s *) user_data;
  GstCaps *caps = gst_pad_get_current_caps (pad);
  gchar *caps_str = caps ? gst_caps_to_string (caps) : NULL;

  if (caps)
    gst_caps_unref (caps);

  g_mutex_lock (&gate->lock);
  gate->sizes.push_back (gst_buffer_get_size (GST_PAD_PROBE_INFO_BUFFER (info)));
  gate->caps.push_back (caps_str ? caps_str : "");
  gate->entered = TRUE;
  g_cond_broadcast (&gate->cond);
  while (!gate->released)
    g_cond_wait (&gate->cond, &gate->lock);
  g_mutex_unlock (&gate->lock);

  g_free (caps_str);

  return GST_PAD_PROBE_OK;
}

/**
 * @brief Close the gate on the element's source pad
 */
static void
_flood_gate_start (flood_gate_s *gate, src_fixture_s *fixture)
{
  g_mutex_init (&gate->lock);
  g_cond_init (&gate->cond);
  gate->entered = FALSE;
  gate->released = FALSE;
  gate->probe_id = gst_pad_add_probe (
      fixture->srcpad, GST_PAD_PROBE_TYPE_BUFFER, _flood_gate_probe, gate, NULL);
}

/**
 * @brief Wait until the streaming thread is held with the first buffer
 */
static gboolean
_flood_gate_wait_entered (flood_gate_s *gate)
{
  gint64 end_time = g_get_monotonic_time () + WAIT_TIMEOUT_US;
  gboolean entered;

  g_mutex_lock (&gate->lock);
  while (!gate->entered) {
    if (!g_cond_wait_until (&gate->cond, &gate->lock, end_time))
      break;
  }
  entered = gate->entered;
  g_mutex_unlock (&gate->lock);

  return entered;
}

/**
 * @brief Open the gate so that the streaming thread runs again
 */
static void
_flood_gate_open (flood_gate_s *gate)
{
  g_mutex_lock (&gate->lock);
  gate->released = TRUE;
  g_cond_broadcast (&gate->cond);
  g_mutex_unlock (&gate->lock);
}

/**
 * @brief Wait until the given number of buffers have passed the gate and
 *        take a copy of the sizes of the buffers pushed so far
 */
static std::vector<gsize>
_flood_gate_wait_sizes (flood_gate_s *gate, gsize expected)
{
  gint64 end_time = g_get_monotonic_time () + WAIT_TIMEOUT_US;
  std::vector<gsize> sizes;

  g_mutex_lock (&gate->lock);
  while (gate->sizes.size () < expected) {
    if (!g_cond_wait_until (&gate->cond, &gate->lock, end_time))
      break;
  }
  sizes = gate->sizes;
  g_mutex_unlock (&gate->lock);

  return sizes;
}

/**
 * @brief Open the gate for good and take it off the element's source pad
 */
static void
_flood_gate_stop (flood_gate_s *gate, src_fixture_s *fixture)
{
  _flood_gate_open (gate);
  gst_pad_remove_probe (fixture->srcpad, gate->probe_id);
}

/**
 * @brief Release what the gate holds, once the pipeline is torn down
 */
static void
_flood_gate_clear (flood_gate_s *gate)
{
  g_mutex_clear (&gate->lock);
  g_cond_clear (&gate->cond);
}

#define FLOOD_BASE_SIZE (100U)

/**
 * @brief Build the buffer sizes a test case expects from the message indices
 */
static std::vector<gsize>
_flood_sizes (std::initializer_list<guint> indices)
{
  std::vector<gsize> sizes;

  for (guint idx : indices)
    sizes.push_back (FLOOD_BASE_SIZE + idx);

  return sizes;
}

/**
 * @brief Deliver messages that their payload sizes tell apart: FLOOD_BASE_SIZE + index
 */
static void
_deliver_flood (src_fixture_s *fixture, guint first, guint last)
{
  GstMQTTMessageHdr hdr = {};
  guint i;

  _set_header_timestamps (fixture->src, &hdr);
  _set_header_caps (&hdr, "video/x-raw,format=RGB,width=640,height=320", NULL);
  hdr.num_mems = 1;
  for (i = first; i <= last; ++i) {
    hdr.size_mems[0] = FLOOD_BASE_SIZE + i;
    EXPECT_TRUE (_deliver (_new_message (&hdr, FLOOD_BASE_SIZE + i)));
  }
}

/**
 * @brief The 'max-buffers' property of mqttsrc is 0 by default and reads back what was set
 */
TEST (testMqttSrc, maxBuffersProperty)
{
  GstElement *elm = gst_element_factory_make ("mqttsrc", NULL);
  guint value = 1;

  ASSERT_TRUE (elm != NULL);

  g_object_get (elm, "max-buffers", &value, NULL);
  EXPECT_EQ (value, 0U);

  g_object_set (elm, "max-buffers", 5U, NULL);
  g_object_get (elm, "max-buffers", &value, NULL);
  EXPECT_EQ (value, 5U);

  gst_object_unref (elm);
}

/**
 * @brief Without 'max-buffers', mqttsrc keeps every message a stalled pipeline has not taken yet
 */
TEST (testMqttSrc, messageFloodWithoutLimit)
{
  src_fixture_s fixture;
  flood_gate_s gate;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));
  _flood_gate_start (&gate, &fixture);

  _deliver_flood (&fixture, 0, 0);
  EXPECT_TRUE (_flood_gate_wait_entered (&gate));
  _deliver_flood (&fixture, 1, 9);
  EXPECT_EQ (_freed_messages (), 0U);

  _flood_gate_open (&gate);
  EXPECT_EQ (_flood_gate_wait_sizes (&gate, 10),
      _flood_sizes ({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }));

  _flood_gate_stop (&gate, &fixture);
  _fixture_teardown (&fixture);
  _flood_gate_clear (&gate);
}

/**
 * @brief mqttsrc drops nothing while the waiting messages fit in 'max-buffers'
 */
TEST (testMqttSrc, messagesWithinMaxBuffers)
{
  src_fixture_s fixture;
  flood_gate_s gate;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  g_object_set (fixture.src, "max-buffers", 3U, NULL);
  ASSERT_TRUE (_fixture_play (&fixture));
  _flood_gate_start (&gate, &fixture);

  _deliver_flood (&fixture, 0, 0);
  EXPECT_TRUE (_flood_gate_wait_entered (&gate));
  _deliver_flood (&fixture, 1, 3);
  EXPECT_EQ (_freed_messages (), 0U);

  _flood_gate_open (&gate);
  EXPECT_EQ (_flood_gate_wait_sizes (&gate, 4), _flood_sizes ({ 0, 1, 2, 3 }));

  _flood_gate_stop (&gate, &fixture);
  _fixture_teardown (&fixture);
  _flood_gate_clear (&gate);
}

/**
 * @brief With 'max-buffers', mqttsrc drops the oldest waiting messages of a
 *        publisher that outruns the pipeline and releases them at once
 */
TEST (testMqttSrc, messageFloodDropsOldest_n)
{
  src_fixture_s fixture;
  flood_gate_s gate;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  g_object_set (fixture.src, "max-buffers", 3U, NULL);
  ASSERT_TRUE (_fixture_play (&fixture));
  _flood_gate_start (&gate, &fixture);

  _deliver_flood (&fixture, 0, 0);
  EXPECT_TRUE (_flood_gate_wait_entered (&gate));
  _deliver_flood (&fixture, 1, 9);
  /** the messages 1 to 6 made room for the three after them */
  EXPECT_EQ (_freed_messages (), 6U);

  _flood_gate_open (&gate);
  EXPECT_EQ (_flood_gate_wait_sizes (&gate, 4).size (), 4U);
  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (_flood_gate_wait_sizes (&gate, 4), _flood_sizes ({ 0, 7, 8, 9 }));

  _flood_gate_stop (&gate, &fixture);
  _fixture_teardown (&fixture);
  _flood_gate_clear (&gate);
}

/**
 * @brief A 'max-buffers' set while more messages are waiting takes effect
 *        with the next message
 */
TEST (testMqttSrc, maxBuffersLoweredWhileQueued_n)
{
  src_fixture_s fixture;
  flood_gate_s gate;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));
  _flood_gate_start (&gate, &fixture);

  _deliver_flood (&fixture, 0, 0);
  EXPECT_TRUE (_flood_gate_wait_entered (&gate));
  _deliver_flood (&fixture, 1, 5);
  EXPECT_EQ (_freed_messages (), 0U);

  g_object_set (fixture.src, "max-buffers", 2U, NULL);
  _deliver_flood (&fixture, 6, 6);
  EXPECT_EQ (_freed_messages (), 4U);

  _flood_gate_open (&gate);
  EXPECT_EQ (_flood_gate_wait_sizes (&gate, 3).size (), 3U);
  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (_flood_gate_wait_sizes (&gate, 3), _flood_sizes ({ 0, 5, 6 }));

  _flood_gate_stop (&gate, &fixture);
  _fixture_teardown (&fixture);
  _flood_gate_clear (&gate);
}

/**
 * @brief The context of the thread that keeps changing 'max-buffers'
 */
typedef struct {
  GstElement *src;
  gint stop;
} max_buffers_churner_s;

/**
 * @brief Switch 'max-buffers' between no limit and 2 until told to stop
 */
static void
_max_buffers_churner (max_buffers_churner_s *churner)
{
  guint value = 0;

  while (!g_atomic_int_get (&churner->stop)) {
    value = (value == 0) ? 2 : 0;
    g_object_set (churner->src, "max-buffers", value, NULL);
    g_usleep (10);
  }
}

/**
 * @brief 'max-buffers' changed by another thread while messages arrive keeps
 *        the queue bound by the value set last
 */
TEST (testMqttSrc, maxBuffersChangedDuringDelivery)
{
  src_fixture_s fixture;
  flood_gate_s gate;
  max_buffers_churner_s churner;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  ASSERT_TRUE (_fixture_play (&fixture));
  _flood_gate_start (&gate, &fixture);

  _deliver_flood (&fixture, 0, 0);
  EXPECT_TRUE (_flood_gate_wait_entered (&gate));

  churner.src = fixture.src;
  churner.stop = 0;
  std::thread churn_thread (_max_buffers_churner, &churner);
  _deliver_flood (&fixture, 1, 200);
  g_atomic_int_set (&churner.stop, 1);
  churn_thread.join ();

  g_object_set (fixture.src, "max-buffers", 2U, NULL);
  _deliver_flood (&fixture, 201, 201);
  /** only the latest two of the 201 messages after the first one are kept */
  EXPECT_EQ (_freed_messages (), 199U);

  _flood_gate_open (&gate);
  EXPECT_EQ (_flood_gate_wait_sizes (&gate, 3).size (), 3U);
  g_usleep (SETTLE_TIME_US);
  EXPECT_EQ (_flood_gate_wait_sizes (&gate, 3), _flood_sizes ({ 0, 200, 201 }));

  _flood_gate_stop (&gate, &fixture);
  _fixture_teardown (&fixture);
  _flood_gate_clear (&gate);
}

#define CAPS_A "video/x-raw,format=RGB,width=640,height=320"
#define CAPS_B "video/x-raw,format=RGB,width=320,height=240"

/**
 * @brief The caps events the element pushes and the threads it pushes them from
 */
typedef struct {
  GMutex lock;
  std::vector<std::string> caps;
  std::vector<GThread *> threads;
  gulong probe_id;
} caps_event_record_s;

/**
 * @brief Record a caps event the element pushes
 */
static GstPadProbeReturn
_caps_event_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  caps_event_record_s *rec = (caps_event_record_s *) user_data;
  GstEvent *event = GST_PAD_PROBE_INFO_EVENT (info);
  GstCaps *caps;
  gchar *caps_str;
  (void) pad;

  if (GST_EVENT_TYPE (event) != GST_EVENT_CAPS)
    return GST_PAD_PROBE_OK;

  gst_event_parse_caps (event, &caps);
  caps_str = gst_caps_to_string (caps);

  g_mutex_lock (&rec->lock);
  rec->caps.push_back (caps_str);
  rec->threads.push_back (g_thread_self ());
  g_mutex_unlock (&rec->lock);

  g_free (caps_str);

  return GST_PAD_PROBE_OK;
}

/**
 * @brief Start recording the caps events of the element
 */
static void
_caps_event_record_start (caps_event_record_s *rec, src_fixture_s *fixture)
{
  g_mutex_init (&rec->lock);
  rec->probe_id = gst_pad_add_probe (fixture->srcpad,
      GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, _caps_event_probe, rec, NULL);
}

/**
 * @brief Take a copy of the caps events recorded so far
 */
static std::vector<std::string>
_caps_event_record_get (caps_event_record_s *rec)
{
  std::vector<std::string> caps;

  g_mutex_lock (&rec->lock);
  caps = rec->caps;
  g_mutex_unlock (&rec->lock);

  return caps;
}

/**
 * @brief Check whether a caps event has been pushed from the given thread
 */
static gboolean
_caps_event_record_has_thread (caps_event_record_s *rec, GThread *thread)
{
  gboolean found = FALSE;

  g_mutex_lock (&rec->lock);
  for (GThread *each : rec->threads)
    found |= (each == thread);
  g_mutex_unlock (&rec->lock);

  return found;
}

/**
 * @brief Stop recording before the fixture is torn down
 */
static void
_caps_event_record_stop (caps_event_record_s *rec, src_fixture_s *fixture)
{
  gst_pad_remove_probe (fixture->srcpad, rec->probe_id);
}

/**
 * @brief Release what the record holds, once the pipeline is torn down
 */
static void
_caps_event_record_clear (caps_event_record_s *rec)
{
  g_mutex_clear (&rec->lock);
}

/**
 * @brief Check whether a recorded caps string describes the given caps
 */
static gboolean
_caps_str_equal (const std::string &recorded, const gchar *expected)
{
  GstCaps *a = gst_caps_from_string (recorded.c_str ());
  GstCaps *b = gst_caps_from_string (expected);
  gboolean equal = (a && b && gst_caps_is_equal (a, b));

  if (a)
    gst_caps_unref (a);
  if (b)
    gst_caps_unref (b);

  return equal;
}

/**
 * @brief Deliver a message with the given caps that its payload size tells apart: FLOOD_BASE_SIZE + index
 */
static void
_deliver_with_caps (src_fixture_s *fixture, const gchar *caps_str, guint idx)
{
  GstMQTTMessageHdr hdr = {};

  _set_header_timestamps (fixture->src, &hdr);
  _set_header_caps (&hdr, caps_str, NULL);
  hdr.num_mems = 1;
  hdr.size_mems[0] = FLOOD_BASE_SIZE + idx;
  EXPECT_TRUE (_deliver (_new_message (&hdr, FLOOD_BASE_SIZE + idx)));
}

/**
 * @brief A caps change goes out with the message that carries it: the
 *        messages queued before it keep the old caps, and the caps event is
 *        pushed on the streaming thread, not on the thread of the paho callback
 */
TEST (testMqttSrc, messageCapsChangeAfterQueuedMessages)
{
  src_fixture_s fixture;
  flood_gate_s gate;
  caps_event_record_s rec;
  std::vector<std::string> events;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  _caps_event_record_start (&rec, &fixture);
  ASSERT_TRUE (_fixture_play (&fixture));
  _flood_gate_start (&gate, &fixture);

  _deliver_with_caps (&fixture, CAPS_A, 0);
  EXPECT_TRUE (_flood_gate_wait_entered (&gate));
  _deliver_with_caps (&fixture, CAPS_A, 1);
  _deliver_with_caps (&fixture, CAPS_B, 2);
  _deliver_with_caps (&fixture, CAPS_B, 3);
  g_usleep (SETTLE_TIME_US);
  /** the messages wait in the queue, so the caps have not changed yet */
  EXPECT_EQ (_caps_event_record_get (&rec).size (), 1U);

  _flood_gate_open (&gate);
  EXPECT_EQ (_flood_gate_wait_sizes (&gate, 4), _flood_sizes ({ 0, 1, 2, 3 }));

  g_mutex_lock (&gate.lock);
  EXPECT_EQ (gate.caps.size (), 4U);
  if (gate.caps.size () == 4U) {
    EXPECT_TRUE (_caps_str_equal (gate.caps[0], CAPS_A));
    EXPECT_TRUE (_caps_str_equal (gate.caps[1], CAPS_A));
    EXPECT_TRUE (_caps_str_equal (gate.caps[2], CAPS_B));
    EXPECT_TRUE (_caps_str_equal (gate.caps[3], CAPS_B));
  }
  g_mutex_unlock (&gate.lock);

  events = _caps_event_record_get (&rec);
  EXPECT_EQ (events.size (), 2U);
  if (events.size () == 2U) {
    EXPECT_TRUE (_caps_str_equal (events[0], CAPS_A));
    EXPECT_TRUE (_caps_str_equal (events[1], CAPS_B));
  }
  EXPECT_FALSE (_caps_event_record_has_thread (&rec, g_thread_self ()));

  _caps_event_record_stop (&rec, &fixture);
  _flood_gate_stop (&gate, &fixture);
  _fixture_teardown (&fixture);
  _flood_gate_clear (&gate);
  _caps_event_record_clear (&rec);
}

/**
 * @brief A caps change survives the drop of the message that brought it, and
 *        a publisher may go back to caps it used before
 */
TEST (testMqttSrc, messageCapsChangeDropped_n)
{
  src_fixture_s fixture;
  flood_gate_s gate;
  caps_event_record_s rec;
  std::vector<std::string> events;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  g_object_set (fixture.src, "max-buffers", 2U, NULL);
  _caps_event_record_start (&rec, &fixture);
  ASSERT_TRUE (_fixture_play (&fixture));
  _flood_gate_start (&gate, &fixture);

  _deliver_with_caps (&fixture, CAPS_A, 0);
  EXPECT_TRUE (_flood_gate_wait_entered (&gate));
  _deliver_with_caps (&fixture, CAPS_A, 1);
  /** message 2 is the first with CAPS_B and is dropped with message 1 */
  _deliver_with_caps (&fixture, CAPS_B, 2);
  _deliver_with_caps (&fixture, CAPS_B, 3);
  _deliver_with_caps (&fixture, CAPS_B, 4);
  EXPECT_EQ (_freed_messages (), 2U);

  _flood_gate_open (&gate);
  EXPECT_EQ (_flood_gate_wait_sizes (&gate, 3), _flood_sizes ({ 0, 3, 4 }));

  _deliver_with_caps (&fixture, CAPS_A, 5);
  EXPECT_EQ (_flood_gate_wait_sizes (&gate, 4), _flood_sizes ({ 0, 3, 4, 5 }));

  g_mutex_lock (&gate.lock);
  EXPECT_EQ (gate.caps.size (), 4U);
  if (gate.caps.size () == 4U) {
    EXPECT_TRUE (_caps_str_equal (gate.caps[0], CAPS_A));
    EXPECT_TRUE (_caps_str_equal (gate.caps[1], CAPS_B));
    EXPECT_TRUE (_caps_str_equal (gate.caps[2], CAPS_B));
    EXPECT_TRUE (_caps_str_equal (gate.caps[3], CAPS_A));
  }
  g_mutex_unlock (&gate.lock);

  events = _caps_event_record_get (&rec);
  EXPECT_EQ (events.size (), 3U);
  if (events.size () == 3U) {
    EXPECT_TRUE (_caps_str_equal (events[0], CAPS_A));
    EXPECT_TRUE (_caps_str_equal (events[1], CAPS_B));
    EXPECT_TRUE (_caps_str_equal (events[2], CAPS_A));
  }

  _caps_event_record_stop (&rec, &fixture);
  _flood_gate_stop (&gate, &fixture);
  _fixture_teardown (&fixture);
  _flood_gate_clear (&gate);
  _caps_event_record_clear (&rec);
}

/**
 * @brief Messages whose caps strings differ only in spelling do not renegotiate
 */
TEST (testMqttSrc, messageEquivalentCapsKept)
{
  src_fixture_s fixture;
  caps_event_record_s rec;
  std::vector<std::string> events;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  _caps_event_record_start (&rec, &fixture);
  ASSERT_TRUE (_fixture_play (&fixture));

  _deliver_with_caps (&fixture, CAPS_A, 0);
  _deliver_with_caps (&fixture, CAPS_A, 1);
  _deliver_with_caps (&fixture,
      "video/x-raw, format=(string)RGB, width=(int)640, height=(int)320", 2);
  _deliver_with_caps (&fixture, CAPS_A, 3);
  EXPECT_TRUE (_fixture_wait_buffers (&fixture, 4));

  events = _caps_event_record_get (&rec);
  EXPECT_EQ (events.size (), 1U);
  if (events.size () == 1U) {
    EXPECT_TRUE (_caps_str_equal (events[0], CAPS_A));
  }

  _caps_event_record_stop (&rec, &fixture);
  _fixture_teardown (&fixture);
  _caps_event_record_clear (&rec);
}

/**
 * @brief A message with caps mqttsrc cannot parse goes out with the caps of the message before it
 */
TEST (testMqttSrc, messageCapsUnparsableAfterValid_n)
{
  src_fixture_s fixture;
  flood_gate_s gate;
  caps_event_record_s rec;
  std::vector<std::string> events;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  _caps_event_record_start (&rec, &fixture);
  ASSERT_TRUE (_fixture_play (&fixture));
  _flood_gate_start (&gate, &fixture);
  _flood_gate_open (&gate);

  _deliver_with_caps (&fixture, CAPS_A, 0);
  _deliver_with_caps (&fixture, ",,,,", 1);
  _deliver_with_caps (&fixture, CAPS_B, 2);
  _deliver_with_caps (&fixture, ",,,,", 3);
  EXPECT_EQ (_flood_gate_wait_sizes (&gate, 4), _flood_sizes ({ 0, 1, 2, 3 }));

  g_mutex_lock (&gate.lock);
  EXPECT_EQ (gate.caps.size (), 4U);
  if (gate.caps.size () == 4U) {
    EXPECT_TRUE (_caps_str_equal (gate.caps[1], CAPS_A));
    EXPECT_TRUE (_caps_str_equal (gate.caps[3], CAPS_B));
  }
  g_mutex_unlock (&gate.lock);

  events = _caps_event_record_get (&rec);
  EXPECT_EQ (events.size (), 2U);
  if (events.size () == 2U) {
    EXPECT_TRUE (_caps_str_equal (events[0], CAPS_A));
    EXPECT_TRUE (_caps_str_equal (events[1], CAPS_B));
  }

  _caps_event_record_stop (&rec, &fixture);
  _flood_gate_stop (&gate, &fixture);
  _fixture_teardown (&fixture);
  _flood_gate_clear (&gate);
  _caps_event_record_clear (&rec);
}

/**
 * @brief mqttsrc sets its caps again after a restart, even if the publisher keeps them
 */
TEST (testMqttSrc, messageCapsAfterRestart)
{
  src_fixture_s fixture;
  caps_event_record_s rec;
  std::vector<std::string> events;
  GstCaps *caps;

  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  _caps_event_record_start (&rec, &fixture);
  ASSERT_TRUE (_fixture_play (&fixture));

  _deliver_with_caps (&fixture, CAPS_A, 0);
  EXPECT_TRUE (_fixture_wait_buffers (&fixture, 1));

  EXPECT_NE (gst_element_set_state (fixture.pipeline, GST_STATE_READY), GST_STATE_CHANGE_FAILURE);
  EXPECT_NE (gst_element_get_state (fixture.pipeline, NULL, NULL, GST_CLOCK_TIME_NONE),
      GST_STATE_CHANGE_FAILURE);
  caps = gst_pad_get_current_caps (fixture.srcpad);
  EXPECT_TRUE (caps == NULL);
  if (caps)
    gst_caps_unref (caps);

  g_mutex_lock (&g_mock.lock);
  g_mock.subscribe_count = 0;
  g_mutex_unlock (&g_mock.lock);
  /** A live source calls create () only once it plays, after the subscription */
  EXPECT_NE (gst_element_set_state (fixture.pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  EXPECT_TRUE (_wait_for_subscribe ());
  EXPECT_TRUE (_fixture_play (&fixture));

  /** The caps of the previous run are not taken for a message without its own */
  _deliver_with_caps (&fixture, ",,,,", 1);
  EXPECT_TRUE (_fixture_wait_buffers (&fixture, 2));
  caps = gst_pad_get_current_caps (fixture.srcpad);
  EXPECT_TRUE (caps == NULL);
  if (caps)
    gst_caps_unref (caps);

  _deliver_with_caps (&fixture, CAPS_A, 2);
  EXPECT_TRUE (_fixture_wait_buffers (&fixture, 3));

  events = _caps_event_record_get (&rec);
  EXPECT_EQ (events.size (), 2U);
  if (events.size () == 2U) {
    EXPECT_TRUE (_caps_str_equal (events[1], CAPS_A));
  }
  caps = gst_pad_get_current_caps (fixture.srcpad);
  EXPECT_TRUE (caps != NULL);
  if (caps)
    gst_caps_unref (caps);

  _caps_event_record_stop (&rec, &fixture);
  _fixture_teardown (&fixture);
  _caps_event_record_clear (&rec);
}


/** @brief A TLS URI cannot bypass the verified, explicit TLS settings. */
TEST (testMqttSrc, tlsSchemeWithoutSwitch_n)
{
  for (const char *host : { "ssl://localhost", "mqtts://localhost", "wss://localhost" }) {
    src_fixture_s fixture;
    ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
    GstElement *element = fixture.src;
    GstElement *pipeline = fixture.pipeline;
    g_object_set (element, "host", host, NULL);
    EXPECT_EQ (gst_element_set_state (pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
    _fixture_teardown (&fixture);
  }
}

/** @brief TLS defaults to off, and a key password is write-only. */
TEST (testMqttSrc, tlsProperties)
{
  src_fixture_s fixture;
  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  GstElement *element = fixture.src;
  GstElement *pipeline = fixture.pipeline;
  gboolean enabled = TRUE;
  gchar *ca = nullptr;
  g_object_get (element, "tls", &enabled, "tls-ca-file", &ca, NULL);
  EXPECT_FALSE (enabled);
  EXPECT_EQ (ca, nullptr);
  GParamSpec *pspec = g_object_class_find_property (
      G_OBJECT_GET_CLASS (element), "tls-key-password");
  ASSERT_NE (pspec, nullptr);
  EXPECT_FALSE (pspec->flags & G_PARAM_READABLE);
  EXPECT_TRUE (pspec->flags & G_PARAM_WRITABLE);
  EXPECT_NE (gst_element_set_state (pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  EXPECT_FALSE (last_connect_tls);
  _fixture_teardown (&fixture);
}

/** @brief Unsupported TLS or invalid files fail before connecting. */
TEST (testMqttSrc, tlsInvalidSettings_n)
{
  src_fixture_s fixture;
  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  GstElement *element = fixture.src;
  GstElement *pipeline = fixture.pipeline;
  g_object_set (element, "tls", TRUE, "tls-ca-file", "/nonexistent/mqtt-ca.pem", NULL);
  EXPECT_EQ (gst_element_set_state (pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  _fixture_teardown (&fixture);
}

#ifdef ENABLE_MQTT_TLS
/** @brief Pass verified TLS and copied client credentials to the Paho client. */
TEST (testMqttSrc, tlsOptionsPassedToConnect)
{
  src_fixture_s fixture;
  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  GstElement *element = fixture.src;
  GstElement *pipeline = fixture.pipeline;
  gchar *file = nullptr;
  int fd = g_file_open_tmp ("nnstreamer-mqtt-tls-XXXXXX", &file, NULL);
  ASSERT_GE (fd, 0);
  close (fd);
  g_object_set (element, "tls", TRUE, "tls-ca-file", file, "tls-cert-file",
      file, "tls-key-file", file, "tls-key-password", "test-key-password", NULL);
  EXPECT_NE (gst_element_set_state (pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  EXPECT_TRUE (last_connect_tls);
  EXPECT_EQ (last_connect_verify, 1);
  EXPECT_EQ (last_connect_auth, 1);
  EXPECT_STREQ (last_connect_ca.c_str (), file);
  EXPECT_STREQ (last_connect_cert.c_str (), file);
  EXPECT_STREQ (last_connect_key.c_str (), file);
  EXPECT_STREQ (last_connect_key_password.c_str (), "test-key-password");
  EXPECT_EQ (g_mock.last_server_uri.find ("ssl://"), 0U);
  /* Changing a property must not invalidate Paho's reconnect options. */
  g_object_set (element, "tls-ca-file", NULL, "tls-key-password", NULL, NULL);
  GstMqttSrc *self = GST_MQTT_SRC (element);
  ASSERT_NE (self->mqtt_conn_opts.ssl, nullptr);
  EXPECT_STREQ (self->mqtt_conn_opts.ssl->trustStore, file);
  EXPECT_STREQ (self->mqtt_conn_opts.ssl->privateKeyPassword, "test-key-password");
  EXPECT_NE (gst_element_set_state (pipeline, GST_STATE_READY), GST_STATE_CHANGE_FAILURE);
  g_object_set (element, "tls", FALSE, NULL);
  EXPECT_NE (gst_element_set_state (pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  EXPECT_FALSE (last_connect_tls);
  EXPECT_EQ (self->mqtt_conn_opts.ssl, nullptr);
  _fixture_teardown (&fixture);
  g_unlink (file);
  g_free (file);
}

/** @brief Refuse a client private key without a client certificate. */
TEST (testMqttSrc, tlsKeyWithoutCertificate_n)
{
  src_fixture_s fixture;
  ASSERT_TRUE (_fixture_setup (&fixture, "test_topic", "video/x-raw"));
  GstElement *element = fixture.src;
  GstElement *pipeline = fixture.pipeline;
  g_object_set (element, "tls", TRUE, "tls-key-file", "/nonexistent/key.pem", NULL);
  EXPECT_EQ (gst_element_set_state (pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);
  _fixture_teardown (&fixture);
}
#endif

/**
 * @brief Exercise the source lifecycle while destruction drains a receive callback.
 */
static void
_check_disconnect_callback (gboolean fail, gboolean omit_callback, guint cycles)
{
  g_autoptr (GstElement) src = gst_element_factory_make ("mqttsrc", NULL);
  GstMqttSrc *self;
  GstBaseSrcClass *klass;

  ASSERT_NE (src, nullptr);
  gst_object_ref_sink (src);
  self = GST_MQTT_SRC (src);
  klass = GST_BASE_SRC_GET_CLASS (src);
  _mock_reset ();

  for (guint i = 0; i < cycles; i++) {
    ASSERT_TRUE (klass->start (GST_BASE_SRC (src)));

    /* Drive the lifecycle hooks without a streaming task or a real broker. */
    g_mutex_lock (&self->mqtt_src_mutex);
    self->is_subscribed = TRUE;
    g_mutex_unlock (&self->mqtt_src_mutex);
    g_mutex_lock (&g_mock.lock);
    g_mock.fail_disconnect = fail;
    g_mock.omit_disconnect_callback = omit_callback;
    g_mock.deliver_during_destroy = TRUE;
    g_mutex_unlock (&g_mock.lock);

    EXPECT_TRUE (klass->stop (GST_BASE_SRC (src)));
    _mock_join_threads ();

    g_mutex_lock (&g_mock.lock);
    gboolean drained = g_mock.destroy_drained_callback;
    guint destroyed = g_mock.destroy_count;
    guint messages = g_mock.free_message_count;
    guint topics = g_mock.free_topic_count;
    g_mutex_unlock (&g_mock.lock);
    EXPECT_TRUE (drained) << "destroy could not drain the message callback";
    EXPECT_EQ (destroyed, i + 1);
    EXPECT_EQ (messages, i + 1);
    EXPECT_EQ (topics, i + 1);
    EXPECT_EQ (g_async_queue_length (self->aqueue), 0);
    EXPECT_EQ (self->mqtt_client_handle, nullptr);
    EXPECT_FALSE (self->is_connected);
    EXPECT_FALSE (self->is_subscribed);
    EXPECT_FALSE (self->is_disconnect_finished);
  }
}

/**
 * @brief A receive callback can complete while stop destroys the client.
 */
TEST (testMqttSrc, stopDrainsMessageCallback)
{
  _check_disconnect_callback (FALSE, FALSE, 1);
}

/**
 * @brief A source can start again after draining a callback at shutdown.
 */
TEST (testMqttSrc, restartAfterDrainingMessageCallback)
{
  _check_disconnect_callback (FALSE, FALSE, 2);
}

/**
 * @brief A refused disconnect still destroys the client and drains its callback.
 */
TEST (testMqttSrc, disconnectFailureDrainsMessageCallback_n)
{
  _check_disconnect_callback (TRUE, FALSE, 1);
}

/**
 * @brief A missing disconnect response does not block destruction of the client.
 */
TEST (testMqttSrc, disconnectTimeoutDrainsMessageCallback_n)
{
  _check_disconnect_callback (FALSE, TRUE, 1);
}

/**
 * @brief Stop while a receive callback holds the Paho-style library mutex.
 */
static void
_check_disconnect_library_lock (gboolean fail, guint cycles)
{
  g_autoptr (GstElement) src = gst_element_factory_make ("mqttsrc", NULL);
  ASSERT_NE (src, nullptr);
  gst_object_ref_sink (src);
  GstMqttSrc *self = GST_MQTT_SRC (src);
  GstBaseSrcClass *klass = GST_BASE_SRC_GET_CLASS (src);
  _mock_reset ();

  for (guint i = 0; i < cycles; i++) {
    ASSERT_TRUE (klass->start (GST_BASE_SRC (src)));
    g_mutex_lock (&self->mqtt_src_mutex);
    self->is_subscribed = TRUE;
    g_mutex_unlock (&self->mqtt_src_mutex);
    g_mutex_lock (&g_mock.lock);
    g_mock.model_disconnect_lock = TRUE;
    g_mock.fail_disconnect = fail;
    g_mock.receive_lock_held = FALSE;
    g_mock.disconnect_entered = FALSE;
    g_mock.disconnect_acquired_lock = FALSE;
    g_mutex_unlock (&g_mock.lock);

    std::thread receive ([] {
      g_mutex_lock (&mock_paho_mutex);
      g_mutex_lock (&g_mock.lock);
      g_mock.receive_lock_held = TRUE;
      g_cond_broadcast (&g_mock.cond);
      gint64 deadline = g_get_monotonic_time () + WAIT_TIMEOUT_US;
      while (!g_mock.disconnect_entered) {
        if (!g_cond_wait_until (&g_mock.cond, &g_mock.lock, deadline))
          break;
      }
      MQTTAsync_messageArrived *callback = g_mock.message_arrived;
      void *context = g_mock.context;
      g_mutex_unlock (&g_mock.lock);
      _deliver_during_destroy (callback, context, NULL, NULL);
      g_mutex_unlock (&mock_paho_mutex);
    });

    g_mutex_lock (&g_mock.lock);
    gint64 deadline = g_get_monotonic_time () + WAIT_TIMEOUT_US;
    while (!g_mock.receive_lock_held) {
      if (!g_cond_wait_until (&g_mock.cond, &g_mock.lock, deadline))
        break;
    }
    gboolean held = g_mock.receive_lock_held;
    g_mutex_unlock (&g_mock.lock);
    EXPECT_TRUE (held);

    EXPECT_TRUE (klass->stop (GST_BASE_SRC (src)));
    receive.join ();
    _mock_join_threads ();
    /* Delivery before disconnect can leave a buffer owned by the queue. */
    GstBuffer *buffer;
    while ((buffer = (GstBuffer *) g_async_queue_try_pop (self->aqueue)))
      gst_buffer_unref (buffer);
    g_mutex_lock (&g_mock.lock);
    gboolean acquired = g_mock.disconnect_acquired_lock;
    guint destroyed = g_mock.destroy_count;
    guint messages = g_mock.free_message_count;
    guint topics = g_mock.free_topic_count;
    g_mutex_unlock (&g_mock.lock);
    EXPECT_TRUE (acquired) << "disconnect could not acquire the callback's library lock";
    EXPECT_EQ (destroyed, i + 1);
    EXPECT_EQ (messages, i + 1);
    EXPECT_EQ (topics, i + 1);
    EXPECT_EQ (self->mqtt_client_handle, nullptr);
    EXPECT_FALSE (self->is_connected);
    EXPECT_FALSE (self->is_subscribed);
    EXPECT_FALSE (self->is_disconnect_finished);
  }
}

/** @brief Receive delivery already holds the library lock before disconnect. */
TEST (testMqttSrc, disconnectWhileReceiveHoldsLibraryLock)
{
  _check_disconnect_library_lock (FALSE, 1);
}

/** @brief A rejected disconnect still releases and destroys the client. */
TEST (testMqttSrc, disconnectFailureWhileReceiveHoldsLibraryLock_n)
{
  _check_disconnect_library_lock (TRUE, 1);
}

/** @brief Restart preserves the lock order and clears a fast completion. */
TEST (testMqttSrc, restartWhileReceiveHoldsLibraryLock)
{
  _check_disconnect_library_lock (FALSE, 2);
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
