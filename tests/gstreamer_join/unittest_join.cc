/**
 * @file        unittest_join.cc
 * @date        10 Nov 2020
 * @brief       Unit test for gstreamer join element
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <unittest_util.h>

static int data_received;

/**
 * @brief Test data for join (2 frames with dimension 3:4:2:2)
 */
const gint test_frames[2][48]
    = { { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112,
            1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124,
            1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212,
            1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224 },
        { 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113,
            2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2201,
            2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213,
            2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224 } };

/**
 * @brief Callback for tensor sink signal.
 */
static void
new_data_cb (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  GstMemory *mem_res;
  GstMapInfo info_res;
  gboolean mapped;
  gint *output, i;
  gint index = *(gint *)user_data;

  data_received++;
  /* Index 100 means a callback that is not allowed. */
  EXPECT_NE (100, index);
  mem_res = gst_buffer_get_memory (buffer, 0);
  mapped = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
  ASSERT_TRUE (mapped);
  output = (gint *)info_res.data;

  for (i = 0; i < 48; i++) {
    EXPECT_EQ (test_frames[index][i], output[i]);
  }
  gst_memory_unmap (mem_res, &info_res);
  gst_memory_unref (mem_res);
}

/**
 * @brief Test join element with appsrc
 */
TEST (join, normal0)
{
  gint idx, n_pads;
  GstBuffer *buf_0, *buf_1, *buf_3, *buf_4;
  GstElement *appsrc_handle_0, *appsrc_handle_1, *sink_handle, *join_handle;
  GstPad *active_pad;
  gchar *active_name;

  gchar *str_pipeline = g_strdup (
      "appsrc name=appsrc_0 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_0 "
      "appsrc name=appsrc_1 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_1 "
      "join name=join ! other/tensor,dimension=(string)3:4:2:2, type=(string)int32, framerate=(fraction)0/1 ! "
      "tensor_sink name=sinkx async=false");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);

  appsrc_handle_0 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_0");
  EXPECT_NE (appsrc_handle_0, nullptr);

  appsrc_handle_1 = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc_1");
  EXPECT_NE (appsrc_handle_1, nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback)new_data_cb, (gpointer)&idx);

  buf_0 = gst_buffer_new_wrapped (g_memdup (test_frames[0], 192), 192);
  buf_3 = gst_buffer_copy (buf_0);

  buf_1 = gst_buffer_new_wrapped (g_memdup (test_frames[1], 192), 192);
  buf_4 = gst_buffer_copy (buf_1);

  data_received = 0;
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  idx = 0;
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle_0), buf_0), GST_FLOW_OK);
  g_usleep (100000);

  idx = 1;
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle_1), buf_1), GST_FLOW_OK);
  g_usleep (100000);

  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle_1), buf_4), GST_FLOW_OK);
  g_usleep (100000);

  g_object_get (join_handle, "active-pad", &active_pad, NULL);
  EXPECT_NE (nullptr, active_pad);
  active_name = gst_pad_get_name (active_pad);
  EXPECT_STREQ ("sink_1", active_name);
  gst_object_unref (active_pad);
  g_free (active_name);

  idx = 0;
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle_0), buf_3), GST_FLOW_OK);
  g_usleep (100000);

  g_object_get (join_handle, "active-pad", &active_pad, NULL);
  EXPECT_NE (nullptr, active_pad);
  active_name = gst_pad_get_name (active_pad);
  EXPECT_STREQ ("sink_0", active_name);
  gst_object_unref (active_pad);
  g_free (active_name);

  g_object_get (join_handle, "n-pads", &n_pads, NULL);
  EXPECT_EQ (2, n_pads);

  gst_object_unref (sink_handle);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);
  EXPECT_EQ (4, data_received);

  gst_object_unref (appsrc_handle_0);
  gst_object_unref (appsrc_handle_1);
  gst_object_unref (join_handle);
  gst_object_unref (pipeline);
}

/**
 * @brief Test get property with invalid parameter
 */
TEST (join, prop0_n)
{
  GstElement *join_handle;
  gchar *str_val = NULL;

  gchar *str_pipeline = g_strdup (
      "appsrc name=appsrc_0 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_0 "
      "appsrc name=appsrc_1 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_1 "
      "join name=join ! other/tensor,dimension=(string)3:4:2:2, type=(string)int32, framerate=(fraction)0/1 ! "
      "tensor_sink name=sinkx async=false");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);

  g_object_get (G_OBJECT (join_handle), "invalid_prop", &str_val, NULL);
  EXPECT_TRUE (str_val == NULL);

  gst_object_unref (join_handle);
  gst_object_unref (pipeline);
}

/**
 * @brief Test get property with NULL parameter
 */
TEST (join, prop1_n)
{
  GstElement *join_handle;
  gchar *str_val = NULL;

  gchar *str_pipeline = g_strdup (
      "appsrc name=appsrc_0 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_0 "
      "appsrc name=appsrc_1 ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! join.sink_1 "
      "join name=join ! other/tensor,dimension=(string)3:4:2:2, type=(string)int32, framerate=(fraction)0/1 ! "
      "tensor_sink name=sinkx async=false");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  join_handle = gst_bin_get_by_name (GST_BIN (pipeline), "join");
  ASSERT_NE (join_handle, nullptr);

  g_object_get (G_OBJECT (join_handle), NULL, &str_val, NULL);
  EXPECT_TRUE (str_val == NULL);

  gst_object_unref (join_handle);
  gst_object_unref (pipeline);
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
