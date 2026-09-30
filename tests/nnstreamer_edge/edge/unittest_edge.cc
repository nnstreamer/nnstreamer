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
#include "edge_sink.h"
#include "edge_src.h"
#include "nnstreamer_log.h"
#include "unittest_util.h"

static int data_received;
static const char *CUSTOM_LIB_PATH = "libnnstreamer-edge-custom-test.so";

/**
 * @brief Look for the warning that the given element posts for rejected custom-props options.
 * @note Consumes the queued warning messages up to and including the matching one.
 */
static gboolean
_pop_custom_props_warning (GstElement *gstpipe, const gchar *elem_name)
{
  GstBus *bus;
  GstMessage *msg;
  GError *err = nullptr;
  gboolean matched = FALSE;

  bus = gst_element_get_bus (gstpipe);
  while ((msg = gst_bus_pop_filtered (bus, GST_MESSAGE_WARNING)) != nullptr) {
    gst_message_parse_warning (msg, &err, nullptr);
    matched = g_error_matches (err, GST_RESOURCE_ERROR, GST_RESOURCE_ERROR_SETTINGS)
              && g_strcmp0 (GST_OBJECT_NAME (GST_MESSAGE_SRC (msg)), elem_name) == 0
              && err->message
              && g_strstr_len (err->message, -1, "custom-props") != nullptr;
    g_clear_error (&err);
    gst_message_unref (msg);
    if (matched)
      break;
  }
  gst_object_unref (bus);

  return matched;
}

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

  g_object_set (edge_handle, "custom-props", "custom1:prop1,custom2:prop2", NULL);
  g_object_get (edge_handle, "custom-props", &str_val, NULL);
  EXPECT_STREQ ("custom1:prop1,custom2:prop2", str_val);
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

  g_object_set (edge_handle, "custom-props", "custom1:prop1,custom2:prop2", NULL);
  g_object_get (edge_handle, "custom-props", &str_val, NULL);
  EXPECT_STREQ ("custom1:prop1,custom2:prop2", str_val);
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

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
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

  EXPECT_EQ (setPipelineStateSync (src_gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (src_gstpipe);
  g_free (src_pipeline);

  gst_object_unref (appsrc_handle);
  gst_object_unref (edge_handle);
  gst_object_unref (sink_handle);
  EXPECT_EQ (setPipelineStateSync (sink_gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
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
 * @brief Ensure edgesink releases its handle when the pipeline goes to NULL state.
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
  sink = GST_EDGESINK_CAST (edge_handle);

  /* setPipelineStateSync () returning makes the lock-free edge_h reads safe (happens-before). */
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_NE (sink->edge_h, (nns_edge_h) NULL);

  /* GstBaseSink calls stop() on READY to NULL, the handle should survive READY. */
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_READY, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_NE (sink->edge_h, (nns_edge_h) NULL);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (sink->edge_h, (nns_edge_h) NULL);

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
  GstElement *edge_handle = nullptr;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! "
      "video/x-raw,width=320,height=240,format=RGB,framerate=10/1 ! "
      "tensor_converter ! edgesink connect-type=CUSTOM name=sinkx port=0");
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  /* Without custom-lib, start() bails out early; assert the handle was never created. */
  edge_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  ASSERT_NE (edge_handle, nullptr);
  EXPECT_EQ (GST_EDGESINK_CAST (edge_handle)->edge_h, (nns_edge_h) NULL);
  gst_object_unref (edge_handle);

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
  GstElement *edge_handle = nullptr;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! "
      "video/x-raw,width=320,height=240,format=RGB,framerate=10/1 ! "
      "tensor_converter ! edgesink connect-type=CUSTOM custom-lib=libINVALID.so name=sinkx port=0");
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  /* On load failure the library frees the handle without clearing the out-param; start() must NULL it. */
  edge_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  ASSERT_NE (edge_handle, nullptr);
  EXPECT_EQ (GST_EDGESINK_CAST (edge_handle)->edge_h, (nns_edge_h) NULL);
  gst_object_unref (edge_handle);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Ensure custom-props values reach the edge handle when edgesink starts.
 */
TEST (edgeCustom, sinkCustomProps)
{
  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;
  GstElement *edge_handle = nullptr;
  GstEdgeSink *sink = nullptr;
  char *val = nullptr;

  /* Value with ':' and whitespace around tokens: split on the first ':' only, then strip. */
  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! "
      "video/x-raw,width=320,height=240,format=RGB,framerate=10/1 ! "
      "tensor_converter ! edgesink connect-type=CUSTOM custom-lib=%s "
      "custom-props=\" PEER_ADDRESS : tcp://127.0.0.1:1883 , QUEUE_SIZE:5:OLD\" name=sinkx port=0",
      CUSTOM_LIB_PATH);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  ASSERT_NE (gstpipe, nullptr);

  edge_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  ASSERT_NE (edge_handle, nullptr);
  sink = GST_EDGESINK_CAST (edge_handle);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  ASSERT_NE (sink->edge_h, (nns_edge_h) NULL);

  EXPECT_EQ (nns_edge_get_info (sink->edge_h, "PEER_ADDRESS", &val), NNS_EDGE_ERROR_NONE);
  EXPECT_STREQ ("tcp://127.0.0.1:1883", val);
  g_free (val);

  /* All options were valid, no warning should be posted. */
  EXPECT_FALSE (_pop_custom_props_warning (gstpipe, "sinkx"));

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (edge_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Malformed custom-props tokens must be skipped without crash and must not block start.
 */
TEST (edgeCustom, sinkCustomPropsMalformed_n)
{
  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;
  GstElement *edge_handle = nullptr;
  GstEdgeSink *sink = nullptr;
  char *val = nullptr;

  /* Colon-less token, empty tokens, empty key, and empty value with a trailing comma. */
  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! "
      "video/x-raw,width=320,height=240,format=RGB,framerate=10/1 ! "
      "tensor_converter ! edgesink connect-type=CUSTOM custom-lib=%s "
      "custom-props=\"foo,,topic:test_topic, : ,c:,:v,\" name=sinkx port=0",
      CUSTOM_LIB_PATH);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  ASSERT_NE (gstpipe, nullptr);

  edge_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  ASSERT_NE (edge_handle, nullptr);
  sink = GST_EDGESINK_CAST (edge_handle);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  ASSERT_NE (sink->edge_h, (nns_edge_h) NULL);

  /* The valid token amid the malformed ones must still be applied. */
  EXPECT_EQ (nns_edge_get_info (sink->edge_h, "TOPIC", &val), NNS_EDGE_ERROR_NONE);
  EXPECT_STREQ ("test_topic", val);
  g_free (val);

  /* Rejected options must be reported on the bus. */
  EXPECT_TRUE (_pop_custom_props_warning (gstpipe, "sinkx"));

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (edge_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief A well-formed option that the edge library refuses must be reported too.
 */
TEST (edgeCustom, sinkCustomPropsRejected_n)
{
  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;

  /* ID is a valid key:value pair, but nnstreamer-edge does not allow updating it. */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! "
                              "video/x-raw,width=320,height=240,format=RGB,framerate=10/1 ! "
                              "tensor_converter ! edgesink connect-type=CUSTOM custom-lib=%s "
                              "custom-props=\"ID:not_allowed\" name=sinkx port=0",
      CUSTOM_LIB_PATH);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  ASSERT_NE (gstpipe, nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_TRUE (_pop_custom_props_warning (gstpipe, "sinkx"));

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

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
                              "other/tensors,num_tensors=1,dimensions=11:1:1:1,types=uint8,format=static,framerate=30/1 ! "
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
 * @brief Ensure custom-props values reach the edge handle when edgesrc starts.
 */
TEST (edgeCustom, srcCustomProps)
{
  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;
  GstElement *edge_handle = nullptr;
  GstEdgeSrc *src = nullptr;
  char *val = nullptr;

  pipeline = g_strdup_printf ("edgesrc connect-type=CUSTOM custom-lib=%s "
                              "custom-props=\" PEER_ADDRESS : tcp://127.0.0.1:1883 , QUEUE_SIZE:5:OLD\" name=srcx ! "
                              "other/tensors,num_tensors=1,dimensions=11:1:1:1,types=uint8,format=static,framerate=30/1 ! "
                              "tensor_sink",
      CUSTOM_LIB_PATH);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  ASSERT_NE (gstpipe, nullptr);

  edge_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "srcx");
  ASSERT_NE (edge_handle, nullptr);
  src = GST_EDGESRC_CAST (edge_handle);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  ASSERT_NE (src->edge_h, (nns_edge_h) NULL);

  EXPECT_EQ (nns_edge_get_info (src->edge_h, "PEER_ADDRESS", &val), NNS_EDGE_ERROR_NONE);
  EXPECT_STREQ ("tcp://127.0.0.1:1883", val);
  g_free (val);

  /* All options were valid, no warning should be posted. */
  EXPECT_FALSE (_pop_custom_props_warning (gstpipe, "srcx"));

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (edge_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Malformed custom-props tokens must be skipped without crash and must not block start.
 */
TEST (edgeCustom, srcCustomPropsMalformed_n)
{
  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;

  pipeline = g_strdup_printf ("edgesrc connect-type=CUSTOM custom-lib=%s "
                              "custom-props=\"foo,,topic:test_topic, : ,c:,:v,\" name=srcx ! "
                              "other/tensors,num_tensors=1,dimensions=11:1:1:1,types=uint8,format=static,framerate=30/1 ! "
                              "tensor_sink",
      CUSTOM_LIB_PATH);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  ASSERT_NE (gstpipe, nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  /* Rejected options must be reported on the bus. */
  EXPECT_TRUE (_pop_custom_props_warning (gstpipe, "srcx"));

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Ensure edgesrc releases its handle when stopping (PAUSED to READY).
 */
TEST (edgeCustom, srcReleasesHandle)
{
  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;
  GstElement *edge_handle = nullptr;
  GstEdgeSrc *src = nullptr;

  pipeline = g_strdup_printf ("edgesrc connect-type=CUSTOM custom-lib=%s name=srcx ! "
                              "other/tensors,num_tensors=1,dimensions=11:1:1:1,types=uint8,format=static,framerate=30/1 ! "
                              "tensor_sink",
      CUSTOM_LIB_PATH);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  ASSERT_NE (gstpipe, nullptr);

  edge_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "srcx");
  ASSERT_NE (edge_handle, nullptr);
  src = GST_EDGESRC_CAST (edge_handle);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_NE (src->edge_h, (nns_edge_h) NULL);

  /* GstBaseSrc calls stop() on PAUSED to READY, the handle should be released in READY. */
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_READY, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (src->edge_h, (nns_edge_h) NULL);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (src->edge_h, (nns_edge_h) NULL);

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
  GstElement *edge_handle = nullptr;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("edgesrc connect-type=CUSTOM name=srcx ! "
                              "other/tensors,num_tensors=1,dimensions=3:320:240:1,types=uint8,format=static,framerate=30/1 ! "
                              "tensor_sink");
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  /* Without custom-lib, start() bails out early; assert the handle was never created. */
  edge_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "srcx");
  ASSERT_NE (edge_handle, nullptr);
  EXPECT_EQ (GST_EDGESRC_CAST (edge_handle)->edge_h, (nns_edge_h) NULL);
  gst_object_unref (edge_handle);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
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
  GstElement *edge_handle = nullptr;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "edgesrc connect-type=CUSTOM custom-lib=libINVALID.so name=srcx ! "
      "other/tensors,num_tensors=1,dimensions=3:320:240:1,types=uint8,format=static,framerate=30/1 ! "
      "tensor_sink");
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  /* On load failure the library frees the handle without clearing the out-param; start() must NULL it. */
  edge_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "srcx");
  ASSERT_NE (edge_handle, nullptr);
  EXPECT_EQ (GST_EDGESRC_CAST (edge_handle)->edge_h, (nns_edge_h) NULL);
  gst_object_unref (edge_handle);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Count the buffers a fakesink receives.
 */
static void
_count_handoff (GstElement *element, GstBuffer *buffer, GstPad *pad, gpointer user_data)
{
  g_atomic_int_inc ((gint *) user_data);
}

/**
 * @brief Publish a buffer with edgesink and count what edgesrc pushes downstream.
 * @param sink_caps The caps the publisher sends under.
 * @param mems The memories of the buffer to publish (transfer full).
 * @param src_caps The caps edgesrc negotiates with downstream.
 * @param[out] received The number of buffers the sink after edgesrc received.
 * @return TRUE if the edgesrc pipeline posted an error.
 */
static gboolean
_publish_to_edgesrc (const gchar *sink_caps, GstMemory **mems, guint num_mems,
    const gchar *src_caps, guint *received)
{
  gchar *pipeline;
  GstElement *sink_gstpipe, *src_gstpipe, *appsrc, *edge_handle, *sink;
  GstBuffer *buf;
  GstBus *bus;
  GstMessage *msg = nullptr;
  guint port, i, count = 0;

  port = get_available_port ();
  pipeline = g_strdup_printf ("appsrc name=appsrc ! %s ! edgesink name=sinkx port=%u async=false",
      sink_caps, port);
  sink_gstpipe = gst_parse_launch (pipeline, NULL);
  g_free (pipeline);
  EXPECT_NE (sink_gstpipe, nullptr);

  edge_handle = gst_bin_get_by_name (GST_BIN (sink_gstpipe), "sinkx");
  g_object_get (edge_handle, "port", &port, NULL);
  gst_object_unref (edge_handle);

  pipeline = g_strdup_printf ("edgesrc dest-port=%u ! %s ! fakesink name=sinkx signal-handoffs=true sync=false async=false",
      port, src_caps);
  src_gstpipe = gst_parse_launch (pipeline, NULL);
  g_free (pipeline);
  EXPECT_NE (src_gstpipe, nullptr);

  sink = gst_bin_get_by_name (GST_BIN (src_gstpipe), "sinkx");
  g_signal_connect (sink, "handoff", (GCallback) _count_handoff, &count);
  gst_object_unref (sink);

  buf = gst_buffer_new ();
  for (i = 0; i < num_mems; i++)
    gst_buffer_append_memory (buf, mems[i]);

  EXPECT_EQ (setPipelineStateSync (sink_gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
      0);
  EXPECT_EQ (setPipelineStateSync (src_gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
      0);

  /* Publish until the sink receives the buffer or edgesrc posts an error; edgesink drops data sent before the subscriber is registered. */
  appsrc = gst_bin_get_by_name (GST_BIN (sink_gstpipe), "appsrc");
  bus = gst_element_get_bus (src_gstpipe);
  for (i = 0; i < 30 && !msg && g_atomic_int_get ((gint *) &count) == 0; i++) {
    EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc), gst_buffer_ref (buf)), GST_FLOW_OK);
    msg = gst_bus_timed_pop_filtered (bus, 100 * GST_MSECOND, GST_MESSAGE_ERROR);
  }
  gst_object_unref (bus);
  gst_object_unref (appsrc);
  gst_buffer_unref (buf);
  if (msg)
    gst_message_unref (msg);
  *received = (guint) g_atomic_int_get (&count);

  EXPECT_EQ (setPipelineStateSync (src_gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (src_gstpipe);
  EXPECT_EQ (setPipelineStateSync (sink_gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (sink_gstpipe);

  return (msg != nullptr);
}

/**
 * @brief Allocate a zero-filled memory.
 */
static GstMemory *
_alloc_mem (gsize size)
{
  gpointer data = g_malloc0 (size);

  return gst_memory_new_wrapped ((GstMemoryFlags) 0, data, size, 0, size, data, g_free);
}

/**
 * @brief Allocate a uint8 tensor memory with a tensor-meta header describing @a dim0 bytes of data.
 * @param extra The number of bytes to add to (or, if negative, cut from) the data described by the header.
 */
static GstMemory *
_alloc_flex_mem (guint dim0, gint extra)
{
  GstTensorMetaInfo meta;
  gsize hsize, size;
  gpointer data;

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_UINT8;
  meta.dimension[0] = dim0;
  meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  hsize = gst_tensor_meta_info_get_header_size (&meta);
  size = hsize + dim0 + extra;
  data = g_malloc0 (MAX (size, hsize));
  gst_tensor_meta_info_update_header (&meta, data);

  return gst_memory_new_wrapped ((GstMemoryFlags) 0, data, size, 0, size, data, g_free);
}

#define EDGE_TEST_CAPS_192 \
  "other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1"
#define EDGE_TEST_CAPS_96 \
  "other/tensor,dimension=(string)3:4:2:1,type=(string)int32,framerate=(fraction)0/1"
#define EDGE_TEST_CAPS_96_NO_RATE \
  "other/tensor,dimension=(string)3:4:2:1,type=(string)int32"
#define EDGE_TEST_CAPS_96X2 \
  "other/tensors,num_tensors=2,dimensions=(string)3:4:2:1.3:4:2:1,types=(string)int32.int32,framerate=(fraction)0/1"
#define EDGE_TEST_CAPS_FLEX \
  "other/tensors,format=flexible,framerate=(fraction)0/1"

/**
 * @brief edgesrc pushes the published data that matches its caps.
 */
TEST (edgeSinkSrc, acceptsMatchingData)
{
  GstMemory *mems[1] = { _alloc_mem (192) };
  guint received = 0;

  EXPECT_FALSE (_publish_to_edgesrc (
      EDGE_TEST_CAPS_192, mems, 1, EDGE_TEST_CAPS_192, &received));
  EXPECT_GE (received, 1U);
}

/**
 * @brief edgesrc refuses published data whose memory size does not match its caps.
 */
TEST (edgeSinkSrc, refusesWrongSize_n)
{
  GstMemory *mems[1] = { _alloc_mem (192) };
  guint received = 0;

  EXPECT_TRUE (_publish_to_edgesrc (EDGE_TEST_CAPS_192, mems, 1, EDGE_TEST_CAPS_96, &received));
  EXPECT_EQ (received, 0U);
}

/**
 * @brief edgesrc refuses published data whose memory count does not match its caps, even if the total size does.
 */
TEST (edgeSinkSrc, refusesWrongCount_n)
{
  GstMemory *mems[2] = { _alloc_mem (96), _alloc_mem (96) };
  guint received = 0;

  EXPECT_TRUE (_publish_to_edgesrc (
      EDGE_TEST_CAPS_96X2, mems, 2, EDGE_TEST_CAPS_192, &received));
  EXPECT_EQ (received, 0U);
}

/**
 * @brief edgesrc pushes matching data also when its tensor caps have no framerate.
 */
TEST (edgeSinkSrc, acceptsMatchingDataWithoutFramerate)
{
  GstMemory *mems[1] = { _alloc_mem (96) };
  guint received = 0;

  EXPECT_FALSE (_publish_to_edgesrc (
      EDGE_TEST_CAPS_96, mems, 1, EDGE_TEST_CAPS_96_NO_RATE, &received));
  EXPECT_GE (received, 1U);
}

/**
 * @brief edgesrc refuses wrong-size data also when its tensor caps have no framerate.
 */
TEST (edgeSinkSrc, refusesWrongSizeWithoutFramerate_n)
{
  GstMemory *mems[1] = { _alloc_mem (192) };
  guint received = 0;

  EXPECT_TRUE (_publish_to_edgesrc (
      EDGE_TEST_CAPS_192, mems, 1, EDGE_TEST_CAPS_96_NO_RATE, &received));
  EXPECT_EQ (received, 0U);
}

/**
 * @brief edgesrc pushes flexible tensors whose header describes the data they carry.
 */
TEST (edgeSinkSrc, acceptsFlexibleData)
{
  GstMemory *mems[2] = { _alloc_flex_mem (4, 0), _alloc_flex_mem (10, 0) };
  guint received = 0;

  EXPECT_FALSE (_publish_to_edgesrc (
      EDGE_TEST_CAPS_FLEX, mems, 2, EDGE_TEST_CAPS_FLEX, &received));
  EXPECT_GE (received, 1U);
}

/**
 * @brief edgesrc refuses flexible tensors shorter than the data their header describes.
 */
TEST (edgeSinkSrc, refusesTruncatedFlexibleData_n)
{
  GstMemory *mems[2] = { _alloc_flex_mem (4, 0), _alloc_flex_mem (10, -1) };
  guint received = 0;

  EXPECT_TRUE (_publish_to_edgesrc (
      EDGE_TEST_CAPS_FLEX, mems, 2, EDGE_TEST_CAPS_FLEX, &received));
  EXPECT_EQ (received, 0U);
}

/**
 * @brief edgesrc max-buffers is 0 (no limit) by default and can be set.
 */
TEST (edgeSrc, maxBuffersProperty)
{
  GstElement *element = gst_element_factory_make ("edgesrc", nullptr);
  guint val = 1U;

  ASSERT_NE (element, nullptr);

  g_object_get (element, "max-buffers", &val, NULL);
  EXPECT_EQ (val, 0U);

  g_object_set (element, "max-buffers", 5U, NULL);
  g_object_get (element, "max-buffers", &val, NULL);
  EXPECT_EQ (val, 5U);

  gst_object_unref (element);
}

/**
 * @brief The first bytes of the buffers a fakesink received, in order.
 */
typedef struct {
  GMutex lock;
  GArray *ids;
} IdLog;

/**
 * @brief Record the first byte of a buffer a fakesink receives.
 */
static void
_log_id (GstElement *element, GstBuffer *buffer, GstPad *pad, gpointer user_data)
{
  IdLog *log = (IdLog *) user_data;
  guint8 id = 0xff;

  gst_buffer_extract (buffer, 0, &id, 1);
  g_mutex_lock (&log->lock);
  g_array_append_val (log->ids, id);
  g_mutex_unlock (&log->lock);
}

/**
 * @brief Wait until the id log has @a num entries, then a little longer to catch extra buffers.
 */
static void
_wait_ids (IdLog *log, guint num, guint timeout_ms)
{
  guint len, waited;

  for (waited = 0; waited <= timeout_ms; waited += 10) {
    g_mutex_lock (&log->lock);
    len = log->ids->len;
    g_mutex_unlock (&log->lock);
    if (len >= num)
      break;
    g_usleep (10000);
  }

  g_usleep (200000);
}

/**
 * @brief Adds up the data edgesrc reports as dropped from its receive queue.
 */
static gint edgesrc_dropped;

/**
 * @brief Debug log function adding up the "Dropped N ..." messages of edgesrc.
 */
static void
_count_dropped (GstDebugCategory *category, GstDebugLevel level,
    const gchar *file, const gchar *function, gint line, GObject *object,
    GstDebugMessage *message, gpointer user_data)
{
  const gchar *text;

  if (g_strcmp0 (gst_debug_category_get_name (category), "edgesrc") != 0)
    return;

  text = gst_debug_message_get (message);
  if (text && g_str_has_prefix (text, "Dropped "))
    g_atomic_int_add (&edgesrc_dropped, (gint) g_ascii_strtoull (text + 8, NULL, 10));
}

/**
 * @brief Event callback of a raw nnstreamer-edge publisher; counts the subscribers that connected.
 */
static int
_raw_pub_event_cb (nns_edge_event_h event_h, void *user_data)
{
  nns_edge_event_e type;

  if (nns_edge_event_get_type (event_h, &type) == NNS_EDGE_ERROR_NONE
      && type == NNS_EDGE_EVENT_CONNECTION_COMPLETED)
    g_atomic_int_inc ((gint *) user_data);

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Pad probe callback that keeps buffers blocked until the probe is removed.
 */
static GstPadProbeReturn
_block_buffers (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  return GST_PAD_PROBE_OK;
}

#define EDGE_TEST_CAPS_4 \
  "other/tensor,dimension=(string)4,type=(string)uint8,framerate=(fraction)0/1"

/**
 * @brief Publish ten data (ids 0 to 9) from a raw publisher to edgesrc while its src pad is blocked, then unblock it.
 * @param max_buffers The max-buffers property of edgesrc.
 * @param[out] log The ids of the data edgesrc pushed, in order.
 * @param expected The number of data edgesrc is expected to push.
 * @return The number of data edgesrc reported as dropped.
 */
static guint
_flood_edgesrc (guint max_buffers, IdLog *log, guint expected)
{
  gchar *pipeline, *port_str;
  GstElement *gstpipe, *element;
  GstPad *pad;
  nns_edge_h pub_h = nullptr;
  nns_edge_data_h data_h;
  gulong probe_id;
  guint i, port, waited, timeout_ms, dropped;
  gint connected = 0;
  guint8 *mem;
  static gsize log_function_added = 0;

  /* Removing a log function leaks the old list in GStreamer, so add it once and keep it. */
  if (g_once_init_enter (&log_function_added)) {
    gst_debug_add_log_function (_count_dropped, nullptr, nullptr);
    g_once_init_leave (&log_function_added, 1);
  }
  g_atomic_int_set (&edgesrc_dropped, 0);
  gst_debug_set_threshold_for_name ("edgesrc", GST_LEVEL_DEBUG);

  port = get_available_port ();
  EXPECT_EQ (nns_edge_create_handle ("rawpub", NNS_EDGE_CONNECT_TYPE_TCP,
                 NNS_EDGE_NODE_TYPE_PUB, &pub_h),
      NNS_EDGE_ERROR_NONE);
  nns_edge_set_event_callback (pub_h, _raw_pub_event_cb, &connected);
  nns_edge_set_info (pub_h, "HOST", "127.0.0.1");
  port_str = g_strdup_printf ("%u", port);
  nns_edge_set_info (pub_h, "PORT", port_str);
  g_free (port_str);
  nns_edge_set_info (pub_h, "CAPS", EDGE_TEST_CAPS_4);
  EXPECT_EQ (nns_edge_start (pub_h), NNS_EDGE_ERROR_NONE);

  pipeline = g_strdup_printf ("edgesrc name=srcx dest-host=127.0.0.1 dest-port=%u max-buffers=%u ! %s ! "
                              "fakesink name=sinkx signal-handoffs=true sync=false async=false",
      port, max_buffers, EDGE_TEST_CAPS_4);
  gstpipe = gst_parse_launch (pipeline, NULL);
  g_free (pipeline);
  EXPECT_NE (gstpipe, nullptr);
  if (!gstpipe) {
    nns_edge_release_handle (pub_h);
    gst_debug_unset_threshold_for_name ("edgesrc");
    return 0;
  }

  element = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  g_signal_connect (element, "handoff", (GCallback) _log_id, log);
  gst_object_unref (element);

  element = gst_bin_get_by_name (GST_BIN (gstpipe), "srcx");
  pad = gst_element_get_static_pad (element, "src");
  gst_object_unref (element);
  probe_id = gst_pad_add_probe (pad,
      (GstPadProbeType) (GST_PAD_PROBE_TYPE_BLOCK | GST_PAD_PROBE_TYPE_BUFFER),
      _block_buffers, nullptr, nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  for (waited = 0; waited < 5000 && g_atomic_int_get (&connected) == 0; waited += 10)
    g_usleep (10000);
  EXPECT_GT (g_atomic_int_get (&connected), 0);

  /* edgesrc takes one data and blocks pushing it; the others wait in its queue. */
  for (i = 0; i < 10; i++) {
    mem = (guint8 *) g_malloc0 (4);
    mem[0] = (guint8) i;
    EXPECT_EQ (nns_edge_data_create (&data_h), NNS_EDGE_ERROR_NONE);
    nns_edge_data_add (data_h, mem, 4, g_free);
    EXPECT_EQ (nns_edge_send (pub_h, data_h), NNS_EDGE_ERROR_NONE);
    nns_edge_data_destroy (data_h);
  }

  timeout_ms = (max_buffers > 0) ? 5000U : 500U;
  for (waited = 0; waited <= timeout_ms; waited += 10) {
    if ((guint) g_atomic_int_get (&edgesrc_dropped) >= 10 - 1 - max_buffers)
      break;
    g_usleep (10000);
  }
  g_usleep (200000);
  dropped = (guint) g_atomic_int_get (&edgesrc_dropped);

  gst_pad_remove_probe (pad, probe_id);
  gst_object_unref (pad);
  _wait_ids (log, expected, 5000);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (gstpipe);
  nns_edge_release_handle (pub_h);

  gst_debug_unset_threshold_for_name ("edgesrc");

  return dropped;
}

/**
 * @brief Without max-buffers, edgesrc keeps every data that arrives while the pipeline is busy.
 */
TEST (edgeSinkSrc, keepsAllDataByDefault)
{
  IdLog log;
  guint i;

  g_mutex_init (&log.lock);
  log.ids = g_array_new (FALSE, FALSE, sizeof (guint8));

  EXPECT_EQ (_flood_edgesrc (0, &log, 10), 0U);

  EXPECT_EQ (log.ids->len, 10U);
  for (i = 0; i < log.ids->len; i++)
    EXPECT_EQ (g_array_index (log.ids, guint8, i), i);

  g_array_free (log.ids, TRUE);
  g_mutex_clear (&log.lock);
}

/**
 * @brief edgesrc with max-buffers keeps the latest data of a publisher out-running the pipeline.
 */
TEST (edgeSinkSrc, maxBuffersDropsOldest_n)
{
  IdLog log;
  guint dropped, len;

#ifdef GST_DISABLE_GST_DEBUG
  GTEST_SKIP () << "The dropped data are counted with the GStreamer debug log.";
#endif

  g_mutex_init (&log.lock);
  log.ids = g_array_new (FALSE, FALSE, sizeof (guint8));

  /**
   * Usually one data is taken before the pad blocks, 3 stay queued and 6 are dropped.
   * If all ten arrive before the element waits on its queue, 7 are dropped and 3 pushed.
   * Either way nothing is lost unaccounted and the last three pushed are the latest.
   */
  dropped = _flood_edgesrc (3, &log, 3);
  EXPECT_GE (dropped, 6U);
  EXPECT_LE (dropped, 7U);

  len = log.ids->len;
  EXPECT_EQ (len + dropped, 10U);
  ASSERT_GE (len, 3U);
  EXPECT_EQ (g_array_index (log.ids, guint8, len - 3), 7U);
  EXPECT_EQ (g_array_index (log.ids, guint8, len - 2), 8U);
  EXPECT_EQ (g_array_index (log.ids, guint8, len - 1), 9U);

  g_array_free (log.ids, TRUE);
  g_mutex_clear (&log.lock);
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
