/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_mqtt.cc
 * @date        20 May 2021
 * @brief       Unit test for GStreamer MQTT elements
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Wook Song <wook16.song@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/base/gstbasesink.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <unittest_util.h>

#ifdef __MQTT_BROKER_ENABLED__
#include <atomic>
#include <signal.h>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

/** @brief How long mqttsrc and mqttsink wait for the broker in start () */
#define MQTT_CONN_TIMEOUT_SEC (5)
#define SMOKE_FRAMES (20)
#define SMOKE_WIDTH (32)
#define SMOKE_HEIGHT (24)
#define SMOKE_FRAME_SIZE (SMOKE_WIDTH * SMOKE_HEIGHT * 3U)
#define SMOKE_USER "nnstreamer"
#define SMOKE_PASSWORD "test-password"
#endif /* __MQTT_BROKER_ENABLED__ */

/**
 * @brief An empty buffer list is a successful no-op without a connection.
 * @note An uninitialized return value may still read as GST_FLOW_OK in a plain
 *       run. The branch is in this frame, not in gtest, so that the Valgrind
 *       job's memcheck gate attributes such a read to this file and fails.
 */
TEST (testMqttSink, emptyBufferList)
{
  GstElement *sink = gst_element_factory_make ("mqttsink", nullptr);
  ASSERT_NE (sink, nullptr);
  GstBaseSinkClass *klass = GST_BASE_SINK_GET_CLASS (sink);
  ASSERT_NE (klass->render_list, nullptr);
  GstBufferList *list = gst_buffer_list_new ();
  GstFlowReturn ret = klass->render_list (GST_BASE_SINK (sink), list);
  if (ret != GST_FLOW_OK)
    ADD_FAILURE () << "render_list returned " << gst_flow_get_name (ret);
  gst_buffer_list_unref (list);
  gst_object_unref (sink);
}

/** @brief A nonempty list still reports the unconnected renderer's failure. */
TEST (testMqttSink, disconnectedBufferList_n)
{
  GstElement *sink = gst_element_factory_make ("mqttsink", nullptr);
  ASSERT_NE (sink, nullptr);
  GstBaseSinkClass *klass = GST_BASE_SINK_GET_CLASS (sink);
  ASSERT_NE (klass->render_list, nullptr);
  GstBufferList *list = gst_buffer_list_new ();
  gst_buffer_list_add (list, gst_buffer_new_allocate (nullptr, 1, nullptr));
  EXPECT_NE (klass->render_list (GST_BASE_SINK (sink), list), GST_FLOW_OK);
  gst_buffer_list_unref (list);
  gst_object_unref (sink);
}

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

/** @brief Time limits stretch by this factor under the memory checker. */
static gint64
_time_scale (void)
{
  const gchar *preload = g_getenv ("LD_PRELOAD");

  return (preload && strstr (preload, "vgpreload")) ? 10 : 1;
}

/** @brief Run a command to completion and report whether it exited with 0. */
static bool
_run_command (const std::vector<std::string> &args)
{
  std::vector<gchar *> argv;
  gchar *err_out = nullptr;
  GError *error = nullptr;
  gint status = 0;
  bool ok;

  for (const std::string &arg : args)
    argv.push_back (const_cast<gchar *> (arg.c_str ()));
  argv.push_back (nullptr);

  ok = g_spawn_sync (nullptr, argv.data (), nullptr,
           (GSpawnFlags) (G_SPAWN_SEARCH_PATH | G_SPAWN_STDOUT_TO_DEV_NULL),
           nullptr, nullptr, nullptr, &err_out, &status, &error)
       && WIFEXITED (status) && WEXITSTATUS (status) == 0;
  if (!ok)
    ADD_FAILURE () << args[0] << " failed: " << (error ? error->message : "")
                   << (err_out ? err_out : "");
  g_clear_error (&error);
  g_free (err_out);
  return ok;
}

/** @brief Find a TCP port on the loopback interface that nobody listens on. */
static guint
_free_port (void)
{
  struct sockaddr_in addr = {};
  socklen_t len = sizeof (addr);
  guint port = 0;
  int fd = socket (AF_INET, SOCK_STREAM, 0);

  if (fd < 0)
    return 0;
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl (INADDR_LOOPBACK);
  if (bind (fd, (struct sockaddr *) &addr, sizeof (addr)) == 0
      && getsockname (fd, (struct sockaddr *) &addr, &len) == 0)
    port = ntohs (addr.sin_port);
  close (fd);
  return port;
}

/** @brief Count the buffers that reach a fakesink and those of a wrong size. */
typedef struct {
  std::atomic<int> received;
  std::atomic<int> wrong_size;
} smoke_count_s;

/** @brief fakesink handoff callback of the smoke cases. */
static void
_smoke_handoff (GstElement *, GstBuffer *buf, GstPad *, gpointer user_data)
{
  smoke_count_s *count = (smoke_count_s *) user_data;

  count->received++;
  if (gst_buffer_get_size (buf) != SMOKE_FRAME_SIZE)
    count->wrong_size++;
}

/** @brief What a pipeline reported while it tried to start. */
typedef struct {
  GstStateChangeReturn ret;
  gint64 elapsed;
  bool has_error;
  GQuark domain;
  gint code;
  std::string message;
} start_result_s;

/**
 * @brief A mosquitto broker of the test's own on a free loopback port.
 * @details Each case starts one with the listener and the authentication it
 *          needs, and reads its log to tell why a client was refused.
 */
class MqttBrokerTest : public ::testing::Test
{
  protected:
  gchar *dir = nullptr;
  std::string log_path;
  std::string port;
  GPid pid = 0;

  /** @brief Make the directory that holds the configuration and the log. */
  void SetUp () override
  {
    dir = g_dir_make_tmp ("nnstreamer-mqtt-broker-XXXXXX", nullptr);
    ASSERT_NE (dir, nullptr);
    log_path = path ("broker.log");
  }

  /** @brief Stop the broker and remove what the case wrote. */
  void TearDown () override
  {
    stopBroker ();
    if (dir) {
      GDir *d = g_dir_open (dir, 0, nullptr);
      const gchar *name;

      while (d && (name = g_dir_read_name (d)) != nullptr)
        g_unlink (path (name).c_str ());
      if (d)
        g_dir_close (d);
      g_rmdir (dir);
      g_free (dir);
    }
  }

  /** @brief The path of a file in the case's directory. */
  std::string path (const gchar *name)
  {
    gchar *p = g_build_filename (dir, name, nullptr);
    std::string ret (p);

    g_free (p);
    return ret;
  }

  /** @brief The broker log as it is now. */
  std::string brokerLog ()
  {
    gchar *contents = nullptr;
    std::string ret;

    if (g_file_get_contents (log_path.c_str (), &contents, nullptr, nullptr))
      ret = contents;
    g_free (contents);
    return ret;
  }

  /** @brief How many times a text appears in the broker log. */
  int logCount (const std::string &text)
  {
    std::string log = brokerLog ();
    int count = 0;

    for (size_t pos = log.find (text); pos != std::string::npos;
        pos = log.find (text, pos + text.size ()))
      count++;
    return count;
  }

  /** @brief Wait until the broker log has a text, or the time limit passes. */
  bool waitForLog (const std::string &text, gint64 limit_sec)
  {
    gint64 end = g_get_monotonic_time () + limit_sec * _time_scale () * G_TIME_SPAN_SECOND;

    while (logCount (text) == 0) {
      if (g_get_monotonic_time () > end)
        return false;
      g_usleep (10000);
    }
    return true;
  }

  /** @brief Start mosquitto with the given lines added to its configuration. */
  bool startBroker (const std::string &conf_lines)
  {
    gchar *program = g_find_program_in_path ("mosquitto");
    std::string conf_path = path ("mosquitto.conf");
    std::string conf;
    GError *error = nullptr;
    bool spawned;

    if (!program && g_file_test ("/usr/sbin/mosquitto", G_FILE_TEST_IS_EXECUTABLE))
      program = g_strdup ("/usr/sbin/mosquitto");
    if (!program) {
      ADD_FAILURE () << "mosquitto is not installed.";
      return false;
    }

    port = std::to_string (_free_port ());
    if (port == "0") {
      g_free (program);
      ADD_FAILURE () << "No free port on the loopback interface.";
      return false;
    }
    conf = "per_listener_settings false\n"
           "persistence false\n"
           "log_dest file "
           + log_path
           + "\n"
             "log_type error\nlog_type warning\nlog_type notice\n"
             "log_type information\nlog_type subscribe\n"
             "connection_messages true\n"
             "listener "
           + port + " 127.0.0.1\n" + conf_lines;
    if (!g_file_set_contents (conf_path.c_str (), conf.c_str (), -1, nullptr)) {
      g_free (program);
      ADD_FAILURE () << "Cannot write " << conf_path;
      return false;
    }

    {
      gchar *argv[] = { program, (gchar *) "-c", (gchar *) conf_path.c_str (), nullptr };
      spawned = g_spawn_async (nullptr, argv, nullptr,
          (GSpawnFlags) (G_SPAWN_DO_NOT_REAP_CHILD | G_SPAWN_STDOUT_TO_DEV_NULL),
          nullptr, nullptr, &pid, &error);
    }
    g_free (program);
    if (!spawned) {
      ADD_FAILURE () << "Cannot start mosquitto: " << error->message;
      g_clear_error (&error);
      pid = 0;
      return false;
    }
    if (!waitForLog (" running", 10)) {
      ADD_FAILURE () << "mosquitto did not start:\n" << brokerLog ();
      return false;
    }
    return true;
  }

  /** @brief Stop the broker started by startBroker (). */
  void stopBroker ()
  {
    gint64 end = g_get_monotonic_time () + 5 * G_TIME_SPAN_SECOND;
    int status;

    if (pid <= 0)
      return;
    kill (pid, SIGTERM);
    while (waitpid (pid, &status, WNOHANG) == 0) {
      if (g_get_monotonic_time () > end) {
        kill (pid, SIGKILL);
        waitpid (pid, &status, 0);
        break;
      }
      g_usleep (10000);
    }
    g_spawn_close_pid (pid);
    pid = 0;
  }

  /** @brief Add a user to a new password file for the broker. */
  bool makePasswordFile ()
  {
    gchar *program = g_find_program_in_path ("mosquitto_passwd");
    bool ok;

    if (!program) {
      ADD_FAILURE () << "mosquitto_passwd is not installed.";
      return false;
    }
    ok = _run_command ({ program, "-c", "-b", path ("passwd"), SMOKE_USER, SMOKE_PASSWORD });
    g_free (program);
    return ok;
  }

  /**
   * @brief Make a CA, a server certificate it signs for "localhost" only,
   *        and a second CA that signs nothing.
   */
  bool makeCertificates ()
  {
    const std::vector<std::string> ec
        = { "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes" };
    std::string ext = path ("server.ext");
    std::vector<std::string> ca = { "openssl", "req", "-x509" };
    std::vector<std::string> other = { "openssl", "req", "-x509" };
    std::vector<std::string> csr = { "openssl", "req" };

    if (!g_file_set_contents (ext.c_str (), "subjectAltName=DNS:localhost\n", -1, nullptr))
      return false;
    ca.insert (ca.end (), ec.begin (), ec.end ());
    ca.insert (ca.end (), { "-days", "2", "-subj", "/CN=nnstreamer-test-ca",
                              "-keyout", path ("ca.key"), "-out", path ("ca.pem") });
    other.insert (other.end (), ec.begin (), ec.end ());
    other.insert (other.end (),
        { "-days", "2", "-subj", "/CN=nnstreamer-other-ca", "-keyout",
            path ("other-ca.key"), "-out", path ("other-ca.pem") });
    csr.insert (csr.end (), ec.begin (), ec.end ());
    csr.insert (csr.end (), { "-subj", "/CN=localhost", "-keyout",
                                path ("server.key"), "-out", path ("server.csr") });
    return _run_command (ca) && _run_command (other) && _run_command (csr)
           && _run_command ({ "openssl", "x509", "-req", "-in", path ("server.csr"),
               "-CA", path ("ca.pem"), "-CAkey", path ("ca.key"), "-set_serial",
               "1", "-days", "2", "-extfile", ext, "-out", path ("server.pem") });
  }

  /** @brief Start a broker that requires the test user's password. */
  bool startAuthBroker ()
  {
    return makePasswordFile ()
           && startBroker ("allow_anonymous false\npassword_file " + path ("passwd") + "\n");
  }

  /** @brief Start a broker whose only listener speaks TLS. */
  bool startTlsBroker ()
  {
    return makeCertificates ()
           && startBroker ("allow_anonymous true\ncafile " + path ("ca.pem")
                           + "\ncertfile " + path ("server.pem") + "\nkeyfile "
                           + path ("server.key") + "\n");
  }

  /**
   * @brief Publish SMOKE_FRAMES frames with mqttsink and count what an mqttsrc
   *        in the same process receives through the broker.
   * @param props Properties both elements get, such as credentials or TLS.
   */
  void runSmoke (const std::string &props)
  {
    const std::string topic = "nnstreamer/smoke/" + port;
    smoke_count_s count;
    GstElement *sub, *pub;
    gint64 start;
    gint64 limit = 3 * _time_scale () * G_TIME_SPAN_SECOND;

    count.received = 0;
    count.wrong_size = 0;
    /* Until mqttsrc can unlock create (), stopping waits for sub-timeout (#5185 C2). */
    sub = gst_parse_launch (
        ("mqttsrc client-id=smoke-sub sub-timeout=" + std::to_string (_time_scale () * G_USEC_PER_SEC)
            + " port=" + port + " sub-topic=" + topic + " " + props
            + " ! fakesink name=sink sync=false signal-handoffs=true")
            .c_str (),
        nullptr);
    ASSERT_NE (sub, nullptr);
    pub = gst_parse_launch (
        ("videotestsrc num-buffers=" + std::to_string (SMOKE_FRAMES)
            + " ! video/x-raw,format=RGB,width=" + std::to_string (SMOKE_WIDTH)
            + ",height=" + std::to_string (SMOKE_HEIGHT) + ",framerate=30/1 ! mqttsink client-id=smoke-pub port="
            + port + " pub-topic=" + topic + " " + props)
            .c_str (),
        nullptr);
    if (!pub) {
      gst_object_unref (sub);
      FAIL () << "Cannot make the publisher pipeline.";
    }
    {
      GstElement *sink = gst_bin_get_by_name (GST_BIN (sub), "sink");
      g_signal_connect (sink, "handoff", G_CALLBACK (_smoke_handoff), &count);
      gst_object_unref (sink);
    }

    /* Connect first: mqttsrc gives up when nothing arrives within sub-timeout. */
    EXPECT_EQ (gst_element_set_state (pub, GST_STATE_READY), GST_STATE_CHANGE_SUCCESS);
    EXPECT_NE (gst_element_set_state (sub, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
    /* Publish only once the broker has the subscription, or frames get lost. */
    EXPECT_TRUE (waitForLog (" " + topic, 10)) << brokerLog ();
    EXPECT_NE (gst_element_set_state (pub, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
    {
      GstBus *bus = gst_element_get_bus (pub);
      GstMessage *msg = gst_bus_timed_pop_filtered (bus, 10 * _time_scale () * GST_SECOND,
          (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));

      EXPECT_NE (msg, nullptr);
      if (msg) {
        EXPECT_EQ (GST_MESSAGE_TYPE (msg), GST_MESSAGE_EOS);
        gst_message_unref (msg);
      }
      gst_object_unref (bus);
    }
    {
      gint64 end = g_get_monotonic_time () + 10 * _time_scale () * G_TIME_SPAN_SECOND;

      while (count.received < SMOKE_FRAMES && g_get_monotonic_time () < end)
        g_usleep (10000);
    }

    start = g_get_monotonic_time ();
    EXPECT_EQ (gst_element_set_state (sub, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);
    EXPECT_LE (g_get_monotonic_time () - start, limit) << "mqttsrc took too long to stop.";
    start = g_get_monotonic_time ();
    EXPECT_EQ (gst_element_set_state (pub, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);
    EXPECT_LE (g_get_monotonic_time () - start, limit) << "mqttsink took too long to stop.";

    EXPECT_EQ (count.received.load (), SMOKE_FRAMES);
    EXPECT_EQ (count.wrong_size.load (), 0);
    EXPECT_EQ (logCount ("as smoke-sub "), 1) << brokerLog ();
    EXPECT_EQ (logCount ("as smoke-pub "), 1) << brokerLog ();
    gst_object_unref (sub);
    gst_object_unref (pub);
  }

  /**
   * @brief Try to start a pipeline and collect the first error it posted.
   * @param desc The pipeline; its MQTT element starts on the way to @a target.
   */
  start_result_s tryStart (const std::string &desc, GstState target)
  {
    start_result_s res = { GST_STATE_CHANGE_FAILURE, 0, false, 0, 0, "" };
    GstElement *pipeline = gst_parse_launch (desc.c_str (), nullptr);
    GstBus *bus;
    GstMessage *msg;
    gint64 start;

    if (!pipeline) {
      ADD_FAILURE () << "Cannot make " << desc;
      return res;
    }
    bus = gst_element_get_bus (pipeline);
    start = g_get_monotonic_time ();
    res.ret = gst_element_set_state (pipeline, target);
    res.elapsed = g_get_monotonic_time () - start;
    msg = gst_bus_pop_filtered (bus, GST_MESSAGE_ERROR);
    if (msg) {
      GError *error = nullptr;

      gst_message_parse_error (msg, &error, nullptr);
      res.has_error = true;
      res.domain = error->domain;
      res.code = error->code;
      res.message = error->message;
      g_error_free (error);
      gst_message_unref (msg);
    }
    gst_element_set_state (pipeline, GST_STATE_NULL);
    gst_bus_set_flushing (bus, TRUE);
    gst_object_unref (bus);
    gst_object_unref (pipeline);
    return res;
  }

  /**
   * @brief Check that each element fails to start against the broker after
   *        connecting to it, and never becomes a client of it.
   * @param props Properties that make the connection fail.
   * @param refusal A broker log line the attempt must add, or "" for none.
   */
  void expectConnectionRefused (const std::string &props, const std::string &refusal)
  {
    /* GstBaseSrc posts an error when start () fails, GstBaseSink does not. */
    const struct {
      const gchar *id;
      std::string desc;
      GstState target;
      bool posts_error;
    } kinds[] = {
      { "refused-sink", "fakesrc ! mqttsink client-id=refused-sink port=" + port + " " + props,
          GST_STATE_READY, false },
      { "refused-src",
          "mqttsrc client-id=refused-src sub-topic=t port=" + port + " " + props + " ! fakesink",
          GST_STATE_PAUSED, true },
    };
    gint64 limit = (MQTT_CONN_TIMEOUT_SEC + 3) * _time_scale () * G_TIME_SPAN_SECOND;

    for (const auto &kind : kinds) {
      int connections = logCount ("New connection from ");
      int refusals = refusal.empty () ? 0 : logCount (refusal);
      start_result_s res = tryStart (kind.desc, kind.target);

      EXPECT_EQ (res.ret, GST_STATE_CHANGE_FAILURE) << kind.desc;
      EXPECT_LE (res.elapsed, limit) << kind.desc;
      if (kind.posts_error) {
        EXPECT_TRUE (res.has_error) << kind.desc;
      }
      /* Not a refusal of the settings before any connection was tried */
      EXPECT_FALSE (res.has_error && res.domain == GST_RESOURCE_ERROR
                    && res.code == GST_RESOURCE_ERROR_SETTINGS)
          << kind.desc << ": " << res.message;
      EXPECT_GT (logCount ("New connection from "), connections) << brokerLog ();
      if (!refusal.empty ()) {
        EXPECT_GT (logCount (refusal), refusals) << brokerLog ();
      }
      EXPECT_EQ (logCount (std::string ("as ") + kind.id + " "), 0) << brokerLog ();
    }
  }
};

/** @brief Frames go from mqttsink to mqttsrc through a broker. */
TEST_F (MqttBrokerTest, smoke)
{
  ASSERT_TRUE (startBroker ("allow_anonymous true\n"));
  runSmoke ("");
}

/** @brief A user the broker knows publishes and subscribes. */
TEST_F (MqttBrokerTest, authenticatedSmoke)
{
  ASSERT_TRUE (startAuthBroker ());
  runSmoke ("username=" SMOKE_USER " password=" SMOKE_PASSWORD);
}

/** @brief A password without a username is refused before connecting. */
TEST_F (MqttBrokerTest, passwordWithoutUsername_n)
{
  ASSERT_TRUE (startAuthBroker ());
  const struct {
    std::string desc;
    GstState target;
  } kinds[] = {
    { "fakesrc ! mqttsink port=" + port + " password=" SMOKE_PASSWORD, GST_STATE_READY },
    { "mqttsrc sub-topic=t port=" + port + " password=" SMOKE_PASSWORD " ! fakesink", GST_STATE_PAUSED },
  };

  for (const auto &kind : kinds) {
    start_result_s res = tryStart (kind.desc, kind.target);

    EXPECT_EQ (res.ret, GST_STATE_CHANGE_FAILURE) << kind.desc;
    EXPECT_TRUE (res.has_error) << kind.desc;
    EXPECT_EQ (res.domain, GST_RESOURCE_ERROR) << kind.desc;
    EXPECT_EQ (res.code, GST_RESOURCE_ERROR_SETTINGS) << kind.desc;
    EXPECT_NE (res.message.find ("without a username"), std::string::npos) << res.message;
  }
  EXPECT_EQ (logCount ("New connection from "), 0) << brokerLog ();
}

/** @brief A wrong password fails start, and the broker says why. */
TEST_F (MqttBrokerTest, wrongPassword_n)
{
  ASSERT_TRUE (startAuthBroker ());
  expectConnectionRefused ("username=" SMOKE_USER " password=wrong-password", "not authorised");
}

/** @brief Without credentials, the broker refuses an anonymous client. */
TEST_F (MqttBrokerTest, anonymousRefused_n)
{
  ASSERT_TRUE (startAuthBroker ());
  expectConnectionRefused ("", "not authorised");
}

#ifdef ENABLE_MQTT_TLS
/** @brief Frames go through a broker the client verifies with the test CA. */
TEST_F (MqttBrokerTest, tlsSmoke)
{
  ASSERT_TRUE (startTlsBroker ());
  runSmoke ("host=localhost tls=true tls-ca-file=" + path ("ca.pem"));
}

/** @brief A certificate for another host name fails start. */
TEST_F (MqttBrokerTest, tlsHostnameMismatch_n)
{
  ASSERT_TRUE (startTlsBroker ());
  /* The certificate names localhost only, not this address. */
  expectConnectionRefused ("host=127.0.0.1 tls=true tls-ca-file=" + path ("ca.pem"), "");
}

/** @brief A server certificate that the given CA did not sign fails start. */
TEST_F (MqttBrokerTest, tlsUnknownCa_n)
{
  ASSERT_TRUE (startTlsBroker ());
  expectConnectionRefused (
      "host=localhost tls=true tls-ca-file=" + path ("other-ca.pem"), "");
}

/** @brief A plain connection to the TLS listener fails start. */
TEST_F (MqttBrokerTest, tlsDisabledOnTlsPort_n)
{
  ASSERT_TRUE (startTlsBroker ());
  expectConnectionRefused ("host=localhost tls=false", "");
}
#endif /* ENABLE_MQTT_TLS */

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
