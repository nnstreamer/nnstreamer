/**
 * @file    unittest_grpc.cc
 * @date    28 Oct 2020
 * @brief   Unit test for gRPC tensor source/sink plugin
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api.h>

#define NNS_GRPC_PLUGIN_NAME "nnstreamer_grpc"
#define NNS_GRPC_TENSOR_SRC_NAME "tensor_src_grpc"
#define NNS_GRPC_TENSOR_SINK_NAME "tensor_sink_grpc"

/**
 * @brief Test gRPC tensor_src/sink existence.
 */
TEST (nnstreamer_grpc, check_existence)
{
  GstPlugin *plugin;
  GstElementFactory *factory;

  plugin = gst_registry_find_plugin (gst_registry_get (),
      NNS_GRPC_PLUGIN_NAME);
  EXPECT_TRUE (plugin != NULL);
  gst_object_unref (plugin);

  factory = gst_element_factory_find (NNS_GRPC_TENSOR_SRC_NAME);
  EXPECT_TRUE (factory != NULL);
  gst_object_unref (factory);

  factory = gst_element_factory_find (NNS_GRPC_TENSOR_SINK_NAME);
  EXPECT_TRUE (factory != NULL);
  gst_object_unref (factory);
}

/**
 * @brief Test gRPC tensor_src/sink existence (negative).
 */
TEST (nnstreamer_grpc, check_existence_n)
{
  GstPlugin *plugin;
  GstElementFactory *factory;
  gchar *name;

  name = g_strconcat (NNS_GRPC_PLUGIN_NAME, "_dummy", NULL);
  plugin = gst_registry_find_plugin (gst_registry_get (), name);
  EXPECT_TRUE (plugin == NULL);
  g_free (name);

  name = g_strconcat (NNS_GRPC_TENSOR_SRC_NAME, "_dummy", NULL);
  factory = gst_element_factory_find (name);
  EXPECT_TRUE (factory == NULL);
  g_free (name);

  name = g_strconcat (NNS_GRPC_TENSOR_SINK_NAME, "_dummy", NULL);
  factory = gst_element_factory_find (name);
  EXPECT_TRUE (factory == NULL);
  g_free (name);
}

/**
 * @brief Test modes.
 */
typedef enum {
  GRPC_MODE_BOTH = 0,
  GRPC_MODE_SRC,
  GRPC_MODE_SINK
} TestMode;

/**
 * @brief Test options.
 */
typedef struct
{
  gboolean server;
  gchar* host;
  guint port;
  guint fps;
  tensor_type type;
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

static gboolean DEFAULT_SERVER = TRUE;
static char DEFAULT_HOST[] = "localhost";
static guint DEFAULT_PORT = 55115;
static guint DEFAULT_FPS = 10;
static guint DEFAULT_OUT = 0;
static tensor_type DEFAULT_TYPE = _NNS_UINT8;

/**
 * @brief Prepare test pipeline
 */
static gboolean
_setup_pipeline (TestOption &option)
{
  gchar *str_pipeline;

  switch (option.mode) {
    case GRPC_MODE_SRC:
      str_pipeline = g_strdup_printf (
        "tensor_src_grpc name=src server=%s host=%s port=%u ! "
        "other/tensor,dimension=(string)1:1:1:1,type=(string)%s,framerate=(fraction)%u/1 ! "
        "fakesink",
        option.server ? "TRUE" : "FALSE", option.host, option.port,
        gst_tensor_get_type_string (option.type), option.fps);
      break;
    case GRPC_MODE_SINK:
      str_pipeline = g_strdup_printf (
        "videotestsrc ! video/x-raw,format=RGB,width=640,height=480,framerate=%u/1 !"
        "tensor_converter ! tensor_sink_grpc name=sink server=%s host=%s port=%u",
        option.fps, option.server ? "TRUE" : "FALSE", option.host, option.port);
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
  option.server = DEFAULT_SERVER;
  option.host = DEFAULT_HOST;
  option.port = DEFAULT_PORT;
  option.fps = DEFAULT_FPS;
  option.type = DEFAULT_TYPE;
  option.mode = GRPC_MODE_BOTH;
}

/**
 * @brief Test gRPC tensor_src get default property
 */
TEST (nnstreamer_grpc, src_get_property_default)
{
  TestOption option;
  GstElement *src;
  gboolean silent, server;
  guint port, out;
  gchar *host;

  _set_default_option (option);
  option.mode = GRPC_MODE_SRC;

  ASSERT_TRUE (_setup_pipeline (option));

  src = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "src");
  ASSERT_TRUE (src != NULL);

  g_object_get (src, "silent", &silent, NULL);
  EXPECT_TRUE (silent);

  g_object_get (src, "server", &server, NULL);
  EXPECT_EQ (server, DEFAULT_SERVER);

  g_object_get (src, "host", &host, NULL);
  EXPECT_STREQ (host, DEFAULT_HOST);
  g_free (host);

  g_object_get (src, "port", &port, NULL);
  EXPECT_EQ (port, DEFAULT_PORT);

  g_object_get (src, "out", &out, NULL);
  EXPECT_EQ (out, DEFAULT_OUT);

  gst_object_unref (src);
  gst_object_unref (test_data.pipeline);
}

/**
 * @brief Test gRPC tensor_sink get default property
 */
TEST (nnstreamer_grpc, sink_get_property_default)
{
  TestOption option;
  GstElement *sink;
  gboolean silent, server;
  guint port, out;
  gchar *host;

  _set_default_option (option);
  option.mode = GRPC_MODE_SINK;

  ASSERT_TRUE (_setup_pipeline (option));

  sink = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "sink");
  ASSERT_TRUE (sink != NULL);

  g_object_get (sink, "silent", &silent, NULL);
  EXPECT_TRUE (silent);

  g_object_get (sink, "server", &server, NULL);
  EXPECT_EQ (server, DEFAULT_SERVER);

  g_object_get (sink, "host", &host, NULL);
  EXPECT_STREQ (host, DEFAULT_HOST);
  g_free (host);

  g_object_get (sink, "port", &port, NULL);
  EXPECT_EQ (port, DEFAULT_PORT);

  g_object_get (sink, "out", &out, NULL);
  EXPECT_EQ (out, DEFAULT_OUT);

  gst_object_unref (sink);
  gst_object_unref (test_data.pipeline);
}

/**
 * @brief Test gRPC tensor_src set property
 */
TEST (nnstreamer_grpc, src_set_property)
{
  TestOption option;
  GstElement *src;
  gboolean silent, server;
  guint port;
  gchar *host;

  _set_default_option (option);
  option.mode = GRPC_MODE_SRC;

  ASSERT_TRUE (_setup_pipeline (option));

  src = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "src");
  ASSERT_TRUE (src != NULL);

  g_object_set (src, "silent", (gboolean) FALSE, NULL);
  g_object_get (src, "silent", &silent, NULL);
  EXPECT_TRUE (!silent);

  g_object_set (src, "server", (gboolean) FALSE, NULL);
  g_object_get (src, "server", &server, NULL);
  EXPECT_TRUE (!server);

  g_object_set (src, "host", "1.1.1.1", NULL);
  g_object_get (src, "host", &host, NULL);
  EXPECT_STREQ (host, "1.1.1.1");
  g_free (host);

  g_object_set (src, "port", 1000, NULL);
  g_object_get (src, "port", &port, NULL);
  EXPECT_EQ (port, 1000);

  gst_object_unref (src);
  gst_object_unref (test_data.pipeline);
}

/**
 * @brief Test gRPC tensor_sink set property
 */
TEST (nnstreamer_grpc, sink_set_property)
{
  TestOption option;
  GstElement *sink;
  gboolean silent, server;
  guint port;
  gchar *host;

  _set_default_option (option);
  option.mode = GRPC_MODE_SINK;

  ASSERT_TRUE (_setup_pipeline (option));

  sink = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "sink");
  ASSERT_TRUE (sink != NULL);

  g_object_set (sink, "silent", (gboolean) FALSE, NULL);
  g_object_get (sink, "silent", &silent, NULL);
  EXPECT_TRUE (!silent);

  g_object_set (sink, "server", (gboolean) FALSE, NULL);
  g_object_get (sink, "server", &server, NULL);
  EXPECT_TRUE (!server);

  g_object_set (sink, "host", "1.1.1.1", NULL);
  g_object_get (sink, "host", &host, NULL);
  EXPECT_STREQ (host, "1.1.1.1");
  g_free (host);

  g_object_set (sink, "port", 1000, NULL);
  g_object_get (sink, "port", &port, NULL);
  EXPECT_EQ (port, 1000);

  gst_object_unref (sink);
  gst_object_unref (test_data.pipeline);
}

/**
 * @brief Test gRPC tensor_src invalid host
 */
TEST (nnstreamer_grpc, src_invalid_host_n)
{
  TestOption option;
  GstElement *src;
  gchar *host;

  _set_default_option (option);
  option.mode = GRPC_MODE_SRC;

  ASSERT_TRUE (_setup_pipeline (option));

  src = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "src");
  ASSERT_TRUE (src != NULL);

  g_object_set (src, "host", "invalid host", NULL);
  g_object_get (src, "host", &host, NULL);
  EXPECT_STREQ (host, DEFAULT_HOST);
  g_free (host);

  gst_object_unref (src);
  gst_object_unref (test_data.pipeline);
}

/**
 * @brief Test gRPC tensor_sink invalid host
 */
TEST (nnstreamer_grpc, sink_invalid_host_n)
{
  TestOption option;
  GstElement *sink;
  gchar *host;

  _set_default_option (option);
  option.mode = GRPC_MODE_SINK;

  ASSERT_TRUE (_setup_pipeline (option));

  sink = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "sink");
  ASSERT_TRUE (sink != NULL);

  g_object_set (sink, "host", "invalid host", NULL);
  g_object_get (sink, "host", &host, NULL);
  EXPECT_STREQ (host, DEFAULT_HOST);
  g_free (host);

  gst_object_unref (sink);
  gst_object_unref (test_data.pipeline);
}

/**
 * @brief Test gRPC tensor_src invalid port
 */
TEST (nnstreamer_grpc, src_invalid_port_n)
{
  TestOption option;
  GstElement *src;
  guint port;

  _set_default_option (option);
  option.mode = GRPC_MODE_SRC;

  ASSERT_TRUE (_setup_pipeline (option));

  src = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "src");
  ASSERT_TRUE (src != NULL);

  g_object_set (src, "port", G_MAXUSHORT + 1, NULL);
  g_object_get (src, "port", &port, NULL);
  EXPECT_EQ (port, DEFAULT_PORT);

  g_object_set (src, "port", "5555", NULL);
  g_object_get (src, "port", &port, NULL);
  EXPECT_EQ (port, DEFAULT_PORT);

  gst_object_unref (src);
  gst_object_unref (test_data.pipeline);
}

/**
 * @brief Test gRPC tensor_sink invalid port
 */
TEST (nnstreamer_grpc, sink_invalid_port_n)
{
  TestOption option;
  GstElement *sink;
  guint port;

  _set_default_option (option);
  option.mode = GRPC_MODE_SINK;

  ASSERT_TRUE (_setup_pipeline (option));

  sink = gst_bin_get_by_name (GST_BIN (test_data.pipeline), "sink");
  ASSERT_TRUE (sink != NULL);

  g_object_set (sink, "port", G_MAXUSHORT + 1, NULL);
  g_object_get (sink, "port", &port, NULL);
  EXPECT_EQ (port, DEFAULT_PORT);

  g_object_set (sink, "port", "5555", NULL);
  g_object_get (sink, "port", &port, NULL);
  EXPECT_EQ (port, DEFAULT_PORT);

  gst_object_unref (sink);
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
