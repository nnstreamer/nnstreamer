/**
 * @file    unittest_filter_mvncsdk2.cc
 * @date    10 Jan 2020
 * @brief   Unit test for the tensor filter sub-plugin for MVNCSDK2
 * @see     https://github.com/nnsuite/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs.
 */

#include <string.h>
#include <gtest/gtest.h>
#include <gst/gst.h>
#include <gst/check/gstcheck.h>
#include <gst/check/gsttestclock.h>
#include <gst/check/gstharness.h>
#include <glib/gstdio.h>
#include <tensor_common.h>
#include <nnstreamer_plugin_api_filter.h>

#include "NCSDKTensorFilterTestHelper.hh"

/**
 * @brief Testing valid pipeline launching and its state changing
 */
TEST (pipeline_mvncsdk2_filter, launch_normal)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *pipeline;
  gchar *test_model;
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;

  if (root_path == NULL) {
    root_path = "..";
  }
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "google_lenet_ncsdk_caffe_1.graph", NULL);
  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=BGR,width=224,height=224 "
      "! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-104.0069877 "
      "! tensor_filter framework=movidius-ncsdk2 model=\"%s\" ! fakesink",
      test_model);

  NCSDKTensorFilterTestHelper::getInstance ().init (GOOGLE_LENET);

  gstpipe = gst_parse_launch (pipeline, &err);
  if (gstpipe) {
    GstState state;
    GstStateChangeReturn ret;
    bool test = false;

    ret = gst_element_get_state (gstpipe, &state, nullptr, GST_SECOND);
    EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
    EXPECT_EQ (state, GST_STATE_NULL);

    ret = gst_element_set_state (gstpipe, GST_STATE_READY);
    EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);

    ret = gst_element_get_state (gstpipe, &state, nullptr, GST_SECOND);
    EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
    EXPECT_EQ (state, GST_STATE_READY);

    ret = gst_element_set_state (gstpipe, GST_STATE_PLAYING);
    /* Run the pipeline for three seconds */
    g_usleep (3 * G_USEC_PER_SEC);
    if ((ret == GST_STATE_CHANGE_ASYNC) || (ret == GST_STATE_CHANGE_SUCCESS)) {
      test = true;
    }
    EXPECT_EQ (test, true);

    ret = gst_element_set_state (gstpipe, GST_STATE_NULL);
    EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);

    ret = gst_element_get_state (gstpipe, &state, nullptr, GST_SECOND);
    EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
    EXPECT_EQ (state, GST_STATE_NULL);

    gst_object_unref (gstpipe);
  } else {
    status = -1;
    g_printerr("Failed to launch the pipeline, %s : %s\n", pipeline,
        (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, 0);
  g_free (test_model);
  g_free (pipeline);

  NCSDKTensorFilterTestHelper::getInstance ().release ();
}

/**
 * @brief Main function for unit test.
 */
int
main (int argc, char **argv)
{
  int ret = -1;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);


  try {
    ret = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return ret;
}
