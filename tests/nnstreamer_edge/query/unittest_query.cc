/**
 * @file        unittest_query.cc
 * @date        27 Aug 2021
 * @brief       Unit test for tensor_query
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>
#include <tensor_common.h>
#include <unittest_util.h>
#include "../gst/nnstreamer/tensor_query/tensor_query_common.h"

/**
 * @brief Test for tensor_query_server get and set properties
 */
TEST (tensorQuery, serverProperties0)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GstElement *srv_handle;
  gint int_val;
  guint uint_val;
  gchar *str_val;
  guint src_port;

  src_port = get_available_port ();

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("tensor_query_serversrc host=127.0.0.1 name=serversrc port=%u ! "
                              "other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! "
                              "tensor_query_serversink name=serversink",
      src_port);
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  /* Get properties of query server source */
  srv_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "serversrc");
  EXPECT_NE (srv_handle, nullptr);

  g_object_get (srv_handle, "host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.1", str_val);
  g_free (str_val);

  g_object_get (srv_handle, "port", &uint_val, NULL);
  EXPECT_EQ (src_port, uint_val);

  g_object_get (srv_handle, "connect-type", &int_val, NULL);
  EXPECT_EQ (0, int_val);

  g_object_get (srv_handle, "timeout", &uint_val, NULL);
  EXPECT_EQ (10U, uint_val);

  /* Set properties of query server source */
  g_object_set (srv_handle, "host", "127.0.0.2", NULL);
  g_object_get (srv_handle, "host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.2", str_val);
  g_free (str_val);

  g_object_set (srv_handle, "port", 5001U, NULL);
  g_object_get (srv_handle, "port", &uint_val, NULL);
  EXPECT_EQ (5001U, uint_val);

  g_object_set (srv_handle, "dest-host", "127.0.0.2", NULL);
  g_object_get (srv_handle, "dest-host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.2", str_val);
  g_free (str_val);

  g_object_set (srv_handle, "dest-port", 5001U, NULL);
  g_object_get (srv_handle, "dest-port", &uint_val, NULL);
  EXPECT_EQ (5001U, uint_val);

  g_object_set (srv_handle, "connect-type", 0, NULL);
  g_object_get (srv_handle, "connect-type", &int_val, NULL);
  EXPECT_EQ (0, int_val);


  g_object_set (srv_handle, "timeout", 20U, NULL);
  g_object_get (srv_handle, "timeout", &uint_val, NULL);
  EXPECT_EQ (20U, uint_val);

  g_object_set (srv_handle, "topic", "TEMP_TEST_TOPIC", NULL);
  g_object_get (srv_handle, "topic", &str_val, NULL);
  EXPECT_STREQ ("TEMP_TEST_TOPIC", str_val);
  g_free (str_val);

  g_object_set (srv_handle, "id", 12345U, NULL);
  g_object_get (srv_handle, "id", &uint_val, NULL);
  EXPECT_EQ (12345U, uint_val);

  gst_object_unref (srv_handle);

  /* Get properties of query server sink */
  srv_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "serversink");
  EXPECT_NE (srv_handle, nullptr);

  g_object_get (srv_handle, "connect-type", &int_val, NULL);
  EXPECT_EQ (0, int_val);

  g_object_get (srv_handle, "timeout", &uint_val, NULL);
  EXPECT_EQ (10U, uint_val);

  g_object_set (srv_handle, "connect-type", 0, NULL);
  g_object_get (srv_handle, "connect-type", &int_val, NULL);
  EXPECT_EQ (0, int_val);

  g_object_set (srv_handle, "timeout", 20U, NULL);
  g_object_get (srv_handle, "timeout", &uint_val, NULL);
  EXPECT_EQ (20U, uint_val);

  g_object_set (srv_handle, "id", 12345U, NULL);
  g_object_get (srv_handle, "id", &uint_val, NULL);
  EXPECT_EQ (12345U, uint_val);

  g_object_set (srv_handle, "limit", 10, NULL);
  g_object_get (srv_handle, "limit", &int_val, NULL);
  EXPECT_EQ (10, int_val);

  gst_object_unref (srv_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_query_server with invalid host name.
 */
TEST (tensorQuery, serverProperties2_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  guint src_port;

  src_port = get_available_port ();

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("tensor_query_serversrc name=serversrc host=f.a.i.l port=%u ! "
                              "other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! "
                              "tensor_query_serversink sync=false async=false",
      src_port);
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_query_client get and set properties
 */
TEST (tensorQuery, clientProperties0)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GstElement *client_handle;
  nns_edge_connect_type_e connect_type;
  guint uint_val;
  gchar *str_val;
  gboolean bool_val;

  /* Create a query client pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB !"
      "tensor_converter ! tensor_query_client name=client connect-type=TCP ! tensor_sink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  /* Get properties of query client */
  client_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "client");
  EXPECT_NE (client_handle, nullptr);

  g_object_get (client_handle, "host", &str_val, NULL);
  EXPECT_STREQ ("localhost", str_val);
  g_free (str_val);

  g_object_get (client_handle, "port", &uint_val, NULL);
  EXPECT_EQ (3001U, uint_val);

  g_object_get (client_handle, "connect-type", &connect_type, NULL);
  EXPECT_EQ (connect_type, NNS_EDGE_CONNECT_TYPE_TCP);

  g_object_get (client_handle, "silent", &bool_val, NULL);
  EXPECT_EQ (TRUE, bool_val);

  /* Set properties of query client */
  g_object_set (client_handle, "host", "127.0.0.2", NULL);
  g_object_get (client_handle, "host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.2", str_val);
  g_free (str_val);

  g_object_set (client_handle, "port", 5001U, NULL);
  g_object_get (client_handle, "port", &uint_val, NULL);
  EXPECT_EQ (5001U, uint_val);

  g_object_set (client_handle, "timeout", 1000U, NULL);
  g_object_get (client_handle, "timeout", &uint_val, NULL);
  EXPECT_EQ (1000U, uint_val);

  g_object_set (client_handle, "silent", FALSE, NULL);
  g_object_get (client_handle, "silent", &bool_val, NULL);
  EXPECT_EQ (FALSE, bool_val);

  gst_object_unref (client_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_query_server run.
 */
TEST (tensorQuery, serverRun)
{
  gchar *pipeline;
  GstElement *gstpipe;
  guint src_port;

  src_port = get_available_port ();

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("tensor_query_serversrc name=serversrc host=127.0.0.1 port=%u ! "
                              "other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! "
                              "tensor_query_serversink sync=false async=false",
      src_port);
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PAUSED, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Run tensor query client without server
 */
TEST (tensorQuery, clientAlone_n)
{
  gchar *pipeline;
  GstElement *gstpipe;

  /* Create a query client pipeline */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB !"
                              "tensor_converter ! tensor_query_client ! tensor_sink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
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
