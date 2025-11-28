/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_edge.cc
 * @date        21 Jul 2022
 * @brief       Unit test for NNStreamer edge element
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Yechan Choi <yechan9.choi@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include "../../../gst/edge/edge_sink.h"
#include "../../../gst/edge/edge_src.h"
#include "nnstreamer_log.h"
#include "unittest_util.h"

static int data_received;
static const char *CUSTOM_LIB_PATH = "libnnstreamer-edge-custom-test.so";

/**
 * @brief Test for edgesink get and set properties.
 */
TEST (edgeSink, properties0)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GstElement *edge_handle;
  gint int_val;
  guint uint_val;
  gchar *str_val;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! "
                              "video/x-raw,width=320,height=240,format=RGB,framerate=10/1 ! "
                              "tensor_converter ! edgesink name=sinkx port=0");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  edge_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  EXPECT_NE (edge_handle, nullptr);

  /* Set/Get properties of edgesink */
  g_object_set (edge_handle, "host", "127.0.0.2", NULL);
  g_object_get (edge_handle, "host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.2", str_val);
  g_free (str_val);

  g_object_set (edge_handle, "port", 5001U, NULL);
  g_object_get (edge_handle, "port", &uint_val, NULL);
  EXPECT_EQ (5001U, uint_val);

  g_object_set (edge_handle, "dest-host", "127.0.0.2", NULL);
  g_object_get (edge_handle, "dest-host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.2", str_val);
  g_free (str_val);

  g_object_set (edge_handle, "dest-port", 5001U, NULL);
  g_object_get (edge_handle, "dest-port", &uint_val, NULL);
  EXPECT_EQ (5001U, uint_val);

  g_object_set (edge_handle, "connect-type", 0, NULL);
  g_object_get (edge_handle, "connect-type", &int_val, NULL);
  EXPECT_EQ (0, int_val);

  g_object_set (edge_handle, "topic", "TEMP_TEST_TOPIC", NULL);
  g_object_get (edge_handle, "topic", &str_val, NULL);
  EXPECT_STREQ ("TEMP_TEST_TOPIC", str_val);
  g_free (str_val);

  gst_object_unref (edge_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for edgesink with invalid host name.
 */
TEST (edgeSink, properties2_n)
{
  gchar *pipeline;
  GstElement *gstpipe;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! "
      "video/x-raw,width=320,height=240,format=RGB,framerate=10/1 ! "
      "tensor_converter ! edgesink host=f.a.i.l name=sinkx port=0");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for edgesrc get and set properties.
 */
TEST (edgeSrc, properties0)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GstElement *edge_handle;
  gint int_val;
  guint uint_val;
  gchar *str_val;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("edgesrc name=srcx ! "
                              "other/tensors,num_tensors=1,dimensions=3:320:240:1,types=uint8,format=static,framerate=30/1 ! "
                              "tensor_sink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  edge_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "srcx");
  EXPECT_NE (edge_handle, nullptr);

  /* Set/Get properties of edgesrc */
  g_object_set (edge_handle, "dest-host", "127.0.0.2", NULL);
  g_object_get (edge_handle, "dest-host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.2", str_val);
  g_free (str_val);

  g_object_set (edge_handle, "dest-port", 5001U, NULL);
  g_object_get (edge_handle, "dest-port", &uint_val, NULL);
  EXPECT_EQ (5001U, uint_val);

  g_object_set (edge_handle, "connect-type", 0, NULL);
  g_object_get (edge_handle, "connect-type", &int_val, NULL);
  EXPECT_EQ (0, int_val);

  g_object_set (edge_handle, "topic", "TEMP_TEST_TOPIC", NULL);
  g_object_get (edge_handle, "topic", &str_val, NULL);
  EXPECT_STREQ ("TEMP_TEST_TOPIC", str_val);
  g_free (str_val);

  gst_object_unref (edge_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for edgesrc with invalid host name.
 */
TEST (edgeSrc, properties2_n)
{
  gchar *pipeline;
  GstElement *gstpipe;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("edgesrc host=f.a.i.l port=0 name=srcx ! "
                              "other/tensors,num_tensors=1,dimensions=3:320:240:1,types=uint8,format=static,framerate=30/1 ! "
                              "tensor_sink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test data for edgesink/src (dimension 3:4:2)
 */
const gint test_frames[48] = { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109,
  1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122,
  1123, 1124, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211,
  1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224 };

/**
 * @brief Callback for tensor sink signal.
 */
static void
new_data_cb (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  GstMemory *mem_res;
  GstMapInfo info_res;
  gint *output, i;
  gboolean ret;

  data_received++;
  mem_res = gst_buffer_get_memory (buffer, 0);
  ret = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
  ASSERT_TRUE (ret);
  output = (gint *) info_res.data;

  for (i = 0; i < 48; i++) {
    EXPECT_EQ (test_frames[i], output[i]);
  }
  gst_memory_unmap (mem_res, &info_res);
  gst_memory_unref (mem_res);
}

/**
 * @brief Test for edgesink and edgesrc.
 */
TEST (edgeSinkSrc, runNormal)
{
  gchar *sink_pipeline, *src_pipeline;
  GstElement *sink_gstpipe, *src_gstpipe;
  GstElement *appsrc_handle, *sink_handle, *edge_handle;
  guint port;
  GstBuffer *buf;
  GstMemory *mem;
  GstMapInfo info;
  int ret;

  /* Create a nnstreamer pipeline */
  port = get_available_port ();
  sink_pipeline = g_strdup_printf (
      "appsrc name=appsrc ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! edgesink name=sinkx port=%u async=false",
      port);
  sink_gstpipe = gst_parse_launch (sink_pipeline, NULL);
  EXPECT_NE (sink_gstpipe, nullptr);

  edge_handle = gst_bin_get_by_name (GST_BIN (sink_gstpipe), "sinkx");
  EXPECT_NE (edge_handle, nullptr);
  g_object_get (edge_handle, "port", &port, NULL);

  appsrc_handle = gst_bin_get_by_name (GST_BIN (sink_gstpipe), "appsrc");
  EXPECT_NE (appsrc_handle, nullptr);

  src_pipeline = g_strdup_printf ("edgesrc dest-port=%u name=srcx ! "
                                  "other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! "
                                  "tensor_sink name=sinkx async=false",
      port);
  src_gstpipe = gst_parse_launch (src_pipeline, NULL);
  EXPECT_NE (src_gstpipe, nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (src_gstpipe), "sinkx");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, NULL);

  buf = gst_buffer_new ();
  mem = gst_allocator_alloc (NULL, 192, NULL);
  ret = gst_memory_map (mem, &info, GST_MAP_WRITE);
  ASSERT_TRUE (ret);
  memcpy (info.data, test_frames, 192);
  gst_memory_unmap (mem, &info);
  gst_buffer_append_memory (buf, mem);
  data_received = 0;

  EXPECT_EQ (setPipelineStateSync (sink_gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
      0);
  g_usleep (1000000);

  buf = gst_buffer_ref (buf);
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle), buf), GST_FLOW_OK);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (src_gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
      0);
  g_usleep (100000);

  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle), buf), GST_FLOW_OK);
  g_usleep (100000);

  gst_object_unref (src_gstpipe);
  g_free (src_pipeline);

  gst_object_unref (appsrc_handle);
  gst_object_unref (edge_handle);
  gst_object_unref (sink_handle);
  gst_object_unref (sink_gstpipe);
  g_free (sink_pipeline);
}

/**
 * @brief Test for edgesink custom connection.
 */
TEST (edgeCustom, sinkNormal)
{
  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! "
      "video/x-raw,width=320,height=240,format=RGB,framerate=10/1 ! "
      "tensor_converter ! edgesink connect-type=CUSTOM custom-lib=%s name=sinkx port=0",
      CUSTOM_LIB_PATH);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Ensure edgesink releases its handle after stop.
 */
TEST (edgeCustom, sinkReleasesHandle)
{
  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;
  GstElement *edge_handle = nullptr;
  GstEdgeSink *sink = nullptr;

  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! "
      "video/x-raw,width=320,height=240,format=RGB,framerate=10/1 ! "
      "tensor_converter ! edgesink connect-type=CUSTOM custom-lib=%s name=sinkx port=0",
      CUSTOM_LIB_PATH);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  ASSERT_NE (gstpipe, nullptr);

  edge_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  ASSERT_NE (edge_handle, nullptr);
  sink = GST_EDGESINK (edge_handle);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_NE (sink->edge_h, (nns_edge_h) NULL);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_READY, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (sink->edge_h, (nns_edge_h) NULL);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (edge_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for edgesink custom connection with invalid property.
 */
TEST (edgeCustom, sinkInvalidProp_n)
{
  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! "
      "video/x-raw,width=320,height=240,format=RGB,framerate=10/1 ! "
      "tensor_converter ! edgesink connect-type=CUSTOM name=sinkx port=0");
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for edgesink custom connection with invalid property.
 */
TEST (edgeCustom, sinkInvalidProp2_n)
{
  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! "
      "video/x-raw,width=320,height=240,format=RGB,framerate=10/1 ! "
      "tensor_converter ! edgesink connect-type=CUSTOM custom-lib=libINVALID.so name=sinkx port=0");
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for edgesrc custom connection.
 */
TEST (edgeCustom, srcNormal)
{
  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("edgesrc connect-type=CUSTOM custom-lib=%s name=srcx ! "
                              "other/tensors,num_tensors=1,dimensions=3:320:240:1,types=uint8,format=static,framerate=30/1 ! "
                              "tensor_sink",
      CUSTOM_LIB_PATH);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Ensure edgesrc releases its handle after stop.
 */
TEST (edgeCustom, srcReleasesHandle)
{
  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;
  GstElement *edge_handle = nullptr;
  GstEdgeSrc *src = nullptr;

  pipeline = g_strdup_printf ("edgesrc connect-type=CUSTOM custom-lib=%s name=srcx ! "
                              "other/tensors,num_tensors=1,dimensions=3:320:240:1,types=uint8,format=static,framerate=30/1 ! "
                              "tensor_sink",
      CUSTOM_LIB_PATH);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  ASSERT_NE (gstpipe, nullptr);

  edge_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "srcx");
  ASSERT_NE (edge_handle, nullptr);
  src = GST_EDGESRC (edge_handle);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_NE (src->edge_h, (nns_edge_h) NULL);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_READY, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (src->edge_h, (nns_edge_h) NULL);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (edge_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for edgesrc custom connection with invalid property.
 */
TEST (edgeCustom, srcInvalidProp_n)
{
  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("edgesrc connect-type=CUSTOM name=srcx ! "
                              "other/tensors,num_tensors=1,dimensions=3:320:240:1,types=uint8,format=static,framerate=30/1 ! "
                              "tensor_sink");
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for edgesrc custom connection with invalid property.
 */
TEST (edgeCustom, srcInvalidProp2_n)
{
  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "edgesrc connect-type=CUSTOM custom-lib=libINVALID.so name=srcx ! "
      "other/tensors,num_tensors=1,dimensions=3:320:240:1,types=uint8,format=static,framerate=30/1 ! "
      "tensor_sink");
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

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
