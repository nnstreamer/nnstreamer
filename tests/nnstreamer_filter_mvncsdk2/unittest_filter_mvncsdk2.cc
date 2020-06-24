/**
 * @file    unittest_filter_mvncsdk2.cc
 * @date    10 Jan 2020
 * @brief   Unit test for the tensor filter sub-plugin for MVNCSDK2
 * @see     https://github.com/nnstreamer/nnstreamer
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
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
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
      "! tensor_filter name=tfilter framework=movidius-ncsdk2 model=\"%s\" ! fakesink",
      test_model);

  NCSDKTensorFilterTestHelper::getInstance ().init (GOOGLE_LENET);

  gstpipe = gst_parse_launch (pipeline, &err);
  if (gstpipe) {
    GstState state;
    GstStateChangeReturn ret;
    GstElement *filter;
    bool test = false;
    guint changed;
    gchar *fw_name;

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

    /* Check framework auto option */
    pipeline = replace_string (pipeline, "movidius-ncsdk2", "auto", NULL, &changed);
    EXPECT_EQ (changed, 1U);

    gstpipe = gst_parse_launch (pipeline, &err);
    EXPECT_TRUE (gstpipe != NULL);

    filter = gst_bin_get_by_name (GST_BIN (gstpipe), "tfilter");
    ASSERT_NE (filter, nullptr);

    /* Check framework */
    g_object_get (filter, "framework", &fw_name, NULL);
    EXPECT_STREQ (fw_name, "movidius-ncsdk2");

    g_free (fw_name);
    gst_object_unref (filter);
    gst_object_unref (gstpipe);
  } else {
    status = -1;
    g_printerr ("Failed to launch the pipeline, %s : %s\n", pipeline,
        (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, 0);
  g_free (test_model);
  g_free (pipeline);

  NCSDKTensorFilterTestHelper::getInstance ().release ();
}

#define TEST_PIPELINE_LAUNCH_NORMAL_FAILURE(idx, fail_stage) \
    TEST (pipeline_mvncsdk2_filter, launch_normal_ ##idx##_n) { \
      const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH"); \
      gchar *pipeline; \
      gchar *test_model; \
      GstElement *gstpipe; \
      GError *err = NULL; \
      int status = 0; \
      \
      NCSDKTensorFilterTestHelper::getInstance ().init (GOOGLE_LENET); \
      NCSDKTensorFilterTestHelper::getInstance () \
          .setFailStage (fail_stage); \
      \
      if (root_path == NULL) { \
        root_path = ".."; \
      } \
      test_model = g_build_filename (root_path, "tests", "test_models", \
          "models", "google_lenet_ncsdk_caffe_1.graph", NULL); \
      \
      pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=BGR,width=224,height=224 " \
          "! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-104.0069877 " \
          "! tensor_filter framework=movidius-ncsdk2 model=\"%s\" ! fakesink", \
          test_model); \
      gstpipe = gst_parse_launch (pipeline, &err); \
      if (gstpipe) { \
        GstStateChangeReturn ret; \
        bool test = false; \
        \
        ret = gst_element_set_state (gstpipe, GST_STATE_PLAYING); \
        \
        g_usleep (1 * G_USEC_PER_SEC); \
        if ((ret == GST_STATE_CHANGE_ASYNC) || \
            (ret == GST_STATE_CHANGE_SUCCESS)) { \
          test = true;\
        } \
        EXPECT_NE (test, true); \
        EXPECT_EQ (ret, GST_STATE_CHANGE_FAILURE); \
        \
        gst_object_unref (gstpipe); \
      } else { \
        status = -1; \
        g_printerr ("Failed to launch the pipeline, %s : %s\n", pipeline, \
            (err) ? err->message : "unknown reason"); \
        g_clear_error (&err); \
      } \
      EXPECT_EQ (status, 0); \
      g_free (test_model); \
      g_free (pipeline); \
      \
      NCSDKTensorFilterTestHelper::getInstance ().release (); \
    };\


/** @brief Testing failure cases (in the case of wrong SDK version) */
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE(0, fail_stage_t::WRONG_SDK_VER);

/** @brief Testing failure cases (in the case of fialure in getting version information) */
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE(1, fail_stage_t::FAIL_GLBL_GET_OPT);

/** @brief Testing failure cases (in the case of fialure in handling device handles) */
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE(2, fail_stage_t::FAIL_DEV_CREATE);
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE(3, fail_stage_t::FAIL_DEV_OPEN);

/** @brief Testing failure cases (in the case of fialure in handling graph handles) */
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE(4, fail_stage_t::FAIL_GRAPH_CREATE);
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE(5, fail_stage_t::FAIL_GRAPH_ALLOC);
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE(6,
    fail_stage_t::FAIL_GRAPH_GET_INPUT_TENSOR_DESC);
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE(7,
    fail_stage_t::FAIL_GRAPH_GET_OUTPUT_TENSOR_DESC);

/** @brief Testing failure cases (in the case of fialure in handling FIFO handles) */
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE(8, fail_stage_t::FAIL_FIFO_CREATE_INPUT);
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE(9, fail_stage_t::FAIL_FIFO_CREATE_OUTPUT);
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE(10, fail_stage_t::FAIL_FIFO_ALLOC_INPUT);
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE(11, fail_stage_t::FAIL_FIFO_ALLOC_OUTPUT);

/** @todo: Failure in invoke () incurs assertion so that the whole tests would be stopped. */

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
