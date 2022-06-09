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

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "tensor_query_serversrc host=127.0.0.1 name=serversrc ! "
      "other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! "
      "tensor_query_serversink host=127.0.0.1 name=serversink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  /* Get properties of query server source */
  srv_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "serversrc");
  EXPECT_NE (srv_handle, nullptr);

  g_object_get (srv_handle, "host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.1", str_val);
  g_free (str_val);

  g_object_get (srv_handle, "port", &uint_val, NULL);
  EXPECT_EQ (3001U, uint_val);

  g_object_get (srv_handle, "protocol", &int_val, NULL);
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

  g_object_set (srv_handle, "protocol", 1, NULL);
  g_object_get (srv_handle, "protocol", &int_val, NULL);
  EXPECT_EQ (1, int_val);


  g_object_set (srv_handle, "timeout", 20U, NULL);
  g_object_get (srv_handle, "timeout", &uint_val, NULL);
  EXPECT_EQ (20U, uint_val);

  gst_object_unref (srv_handle);

  /* Get properties of query server sink */
  srv_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "serversink");
  EXPECT_NE (srv_handle, nullptr);
  g_object_get (srv_handle, "host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.1", str_val);
  g_free (str_val);

  g_object_get (srv_handle, "port", &uint_val, NULL);
  EXPECT_EQ (3000U, uint_val);

  g_object_get (srv_handle, "protocol", &int_val, NULL);
  EXPECT_EQ (0, int_val);

  g_object_get (srv_handle, "timeout", &uint_val, NULL);
  EXPECT_EQ (10U, uint_val);

  /* Set properties of query server sink */
  g_object_set (srv_handle, "host", "127.0.0.2", NULL);
  g_object_get (srv_handle, "host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.2", str_val);
  g_free (str_val);

  g_object_set (srv_handle, "port", 5000U, NULL);
  g_object_get (srv_handle, "port", &uint_val, NULL);
  EXPECT_EQ (5000U, uint_val);

  g_object_set (srv_handle, "protocol", 1, NULL);
  g_object_get (srv_handle, "protocol", &int_val, NULL);
  EXPECT_EQ (1, int_val);


  g_object_set (srv_handle, "timeout", 20U, NULL);
  g_object_get (srv_handle, "timeout", &uint_val, NULL);
  EXPECT_EQ (20U, uint_val);

  gst_object_unref (srv_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_query_server with same port.
 */
TEST (tensorQuery, serverProperties1_n)
{
  gchar *pipeline;
  GstElement *gstpipe;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "tensor_query_serversrc name=serversrc port=3000 ! "
      "other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! "
      "tensor_query_serversink port=3000 sync=false async=false");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

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

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "tensor_query_serversrc name=serversrc host=f.a.i.l ! "
      "other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! "
      "tensor_query_serversink sync=false async=false");
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
  TensorQueryProtocol protocol;
  guint uint_val;
  gchar *str_val;
  gboolean bool_val;

  /* Create a query client pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB !"
      "tensor_converter ! tensor_query_client name=client protocol=TCP src-host=127.0.0.1 sink-host=127.0.0.1 ! tensor_sink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  /* Get properties of query client */
  client_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "client");
  EXPECT_NE (client_handle, nullptr);

  g_object_get (client_handle, "src-host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.1", str_val);
  g_free (str_val);

  g_object_get (client_handle, "src-port", &uint_val, NULL);
  EXPECT_EQ (3001U, uint_val);

  g_object_get (client_handle, "sink-host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.1", str_val);
  g_free (str_val);

  g_object_get (client_handle, "sink-port", &uint_val, NULL);
  EXPECT_EQ (3000U, uint_val);

  g_object_get (client_handle, "protocol", &protocol, NULL);
  EXPECT_EQ (protocol, _TENSOR_QUERY_PROTOCOL_TCP);

  g_object_get (client_handle, "silent", &bool_val, NULL);
  EXPECT_EQ (TRUE, bool_val);

  /* Set properties of query client */
  g_object_set (client_handle, "src-host", "127.0.0.2", NULL);
  g_object_get (client_handle, "src-host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.2", str_val);
  g_free (str_val);

  g_object_set (client_handle, "src-port", 5001U, NULL);
  g_object_get (client_handle, "src-port", &uint_val, NULL);
  EXPECT_EQ (5001U, uint_val);

  g_object_set (client_handle, "sink-host", "127.0.0.2", NULL);
  g_object_get (client_handle, "sink-host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.2", str_val);
  g_free (str_val);

  g_object_set (client_handle, "sink-port", 5001U, NULL);
  g_object_get (client_handle, "sink-port", &uint_val, NULL);
  EXPECT_EQ (5001U, uint_val);

  g_object_set (client_handle, "silent", FALSE, NULL);
  g_object_get (client_handle, "silent", &bool_val, NULL);
  EXPECT_EQ (FALSE, bool_val);

  gst_object_unref (client_handle);
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
  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB !"
      "tensor_converter ! tensor_query_client ! tensor_sink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for nnstreamer_query_server_init
 */
TEST (tensorQueryCommon, serverInit0)
{
  query_server_handle server_data = NULL;
  server_data = nnstreamer_query_server_data_new ();
  EXPECT_NE ((void *) NULL, server_data);

  EXPECT_EQ (0, nnstreamer_query_server_init (server_data, _TENSOR_QUERY_PROTOCOL_TCP, "localhost", 3001, TRUE));
  nnstreamer_query_server_data_free (server_data);
}

/**
 * @brief Test for nnstreamer_query_server_init with invalid parameter.
 */
TEST (tensorQueryCommon, serverInit1_n)
{
  EXPECT_NE (0, nnstreamer_query_server_init (NULL, _TENSOR_QUERY_PROTOCOL_TCP, "localhost", 3001, TRUE));
}

/**
 * @brief Test for nnstreamer_query_server_init with invalid parameter.
 */
TEST (tensorQueryCommon, serverInit2_n)
{
  query_server_handle server_data = NULL;
  server_data = nnstreamer_query_server_data_new ();
  EXPECT_NE ((void *) NULL, server_data);

  EXPECT_NE (0, nnstreamer_query_server_init (server_data, _TENSOR_QUERY_PROTOCOL_END, "localhost", 3001, TRUE));

  nnstreamer_query_server_data_free (server_data);
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
