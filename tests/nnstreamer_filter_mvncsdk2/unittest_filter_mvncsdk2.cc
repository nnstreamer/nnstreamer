/**
 * @file    unittest_filter_mvncsdk2.cc
 * @date    10 Jan 2020
 * @brief   Unit test for the tensor filter sub-plugin for MVNCSDK2
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs.
 */

#include <gtest/gtest.h>
#include <glib/gstdio.h>
#include <gst/check/gstcheck.h>
#include <gst/check/gstharness.h>
#include <gst/check/gsttestclock.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api_filter.h>
#include <string.h>
#include <tensor_common.h>

#include "NCSDKTensorFilterTestHelper.hh"

/**
 * @brief Testing valid pipeline launching and its state changing
 */
TEST (pipelineMvncsdk2Filter, launchNormal)
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
  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=BGR,width=224,height=224 "
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

#define TEST_PIPELINE_LAUNCH_NORMAL_FAILURE(idx, fail_stage)                                                  \
  TEST (pipelineMvncsdk2Filter, launchNormal##idx##_n)                                                        \
  {                                                                                                           \
    const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");                                        \
    gchar *pipeline;                                                                                          \
    gchar *test_model;                                                                                        \
    GstElement *gstpipe;                                                                                      \
    GError *err = NULL;                                                                                       \
    int status = 0;                                                                                           \
                                                                                                              \
    NCSDKTensorFilterTestHelper::getInstance ().init (GOOGLE_LENET);                                          \
    NCSDKTensorFilterTestHelper::getInstance ().setFailStage (fail_stage);                                    \
                                                                                                              \
    if (root_path == NULL) {                                                                                  \
      root_path = "..";                                                                                       \
    }                                                                                                         \
    test_model = g_build_filename (root_path, "tests", "test_models",                                         \
        "models", "google_lenet_ncsdk_caffe_1.graph", NULL);                                                  \
                                                                                                              \
    pipeline = g_strdup_printf (                                                                              \
        "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=BGR,width=224,height=224 " \
        "! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-104.0069877 "     \
        "! tensor_filter framework=movidius-ncsdk2 model=\"%s\" ! fakesink",                                  \
        test_model);                                                                                          \
    gstpipe = gst_parse_launch (pipeline, &err);                                                              \
    if (gstpipe) {                                                                                            \
      GstStateChangeReturn ret;                                                                               \
      bool test = false;                                                                                      \
                                                                                                              \
      ret = gst_element_set_state (gstpipe, GST_STATE_PLAYING);                                               \
                                                                                                              \
      g_usleep (1 * G_USEC_PER_SEC);                                                                          \
      if ((ret == GST_STATE_CHANGE_ASYNC) || (ret == GST_STATE_CHANGE_SUCCESS)) {                             \
        test = true;                                                                                          \
      }                                                                                                       \
      EXPECT_NE (test, true);                                                                                 \
      EXPECT_EQ (ret, GST_STATE_CHANGE_FAILURE);                                                              \
                                                                                                              \
      gst_object_unref (gstpipe);                                                                             \
    } else {                                                                                                  \
      status = -1;                                                                                            \
      g_printerr ("Failed to launch the pipeline, %s : %s\n", pipeline,                                       \
          (err) ? err->message : "unknown reason");                                                           \
      g_clear_error (&err);                                                                                   \
    }                                                                                                         \
    EXPECT_EQ (status, 0);                                                                                    \
    g_free (test_model);                                                                                      \
    g_free (pipeline);                                                                                        \
                                                                                                              \
    NCSDKTensorFilterTestHelper::getInstance ().release ();                                                   \
  };


/** @brief Testing failure cases (in the case of wrong SDK version) */
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE (0, fail_stage_t::WRONG_SDK_VER);

/** @brief Testing failure cases (in the case of failure in getting version information) */
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE (1, fail_stage_t::FAIL_GLBL_GET_OPT);

/** @brief Testing failure cases (in the case of failure in handling device handles) */
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE (2, fail_stage_t::FAIL_DEV_CREATE);
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE (3, fail_stage_t::FAIL_DEV_OPEN);

/** @brief Testing failure cases (in the case of failure in handling graph handles) */
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE (4, fail_stage_t::FAIL_GRAPH_CREATE);
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE (5, fail_stage_t::FAIL_GRAPH_ALLOC);
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE (6, fail_stage_t::FAIL_GRAPH_GET_INPUT_TENSOR_DESC);
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE (7, fail_stage_t::FAIL_GRAPH_GET_OUTPUT_TENSOR_DESC);

/** @brief Testing failure cases (in the case of failure in handling FIFO handles) */
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE (8, fail_stage_t::FAIL_FIFO_CREATE_INPUT);
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE (9, fail_stage_t::FAIL_FIFO_CREATE_OUTPUT);
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE (10, fail_stage_t::FAIL_FIFO_ALLOC_INPUT);
TEST_PIPELINE_LAUNCH_NORMAL_FAILURE (11, fail_stage_t::FAIL_FIFO_ALLOC_OUTPUT);

#define MVNCSDK2_IN_CAPS_STR                                           \
  "other/tensors,format=static,num_tensors=1,framerate=(fraction)0/1," \
  "dimensions=(string)3:224:224:1,types=(string)float32"

#define MVNCSDK2_IN_BUF_SIZE                                             \
  (GOOGLE_LENET_IN_DIM_C * GOOGLE_LENET_IN_DIM_W * GOOGLE_LENET_IN_DIM_H \
      * GOOGLE_LENET_IN_DIM_N * sizeof (float))

/**
 * @brief Build a harness around a tensor_filter bound to the mocked device.
 * @return The harness, or NULL if the description cannot be parsed.
 */
static GstHarness *
_mvncsdk2_harness_new (void)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstHarness *h;
  gchar *desc;
  gchar *test_model;

  if (root_path == NULL) {
    root_path = "..";
  }

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "google_lenet_ncsdk_caffe_1.graph", NULL);
  desc = g_strdup_printf ("tensor_filter framework=movidius-ncsdk2 model=\"%s\"", test_model);
  h = gst_harness_new_parse (desc);
  g_free (desc);
  g_free (test_model);

  if (h != NULL) {
    gst_harness_set_src_caps_str (h, MVNCSDK2_IN_CAPS_STR);
  }

  return h;
}

/**
 * @brief Push a single input tensor into the given harness.
 */
static GstFlowReturn
_mvncsdk2_push (GstHarness *h)
{
  return gst_harness_push (h, gst_harness_create_buffer (h, MVNCSDK2_IN_BUF_SIZE));
}

/**
 * @brief Check that the sub-plugin survives a failure in invoke ()
 * @details The framework keeps fw_opened set when invoke () returns -1, so the
 *          sub-plugin has to keep its private data and the device handles in
 *          it. Closing them here used to leave the private data NULL, and the
 *          next buffer dereferenced it.
 */
static void
_mvncsdk2_run_invoke_failure (fail_stage_t stage)
{
  GstHarness *h;

  NCSDKTensorFilterTestHelper::getInstance ().init (GOOGLE_LENET);

  h = _mvncsdk2_harness_new ();
  if (h == NULL) {
    /* Leaving the mock initialized would take the following cases down too. */
    ADD_FAILURE () << "Failed to parse the tensor_filter description";
    NCSDKTensorFilterTestHelper::getInstance ().release ();
    return;
  }

  EXPECT_EQ (_mvncsdk2_push (h), GST_FLOW_OK);

  NCSDKTensorFilterTestHelper::getInstance ().setFailStage (stage);
  EXPECT_EQ (_mvncsdk2_push (h), GST_FLOW_ERROR);

  NCSDKTensorFilterTestHelper::getInstance ().setFailStage (fail_stage_t::NONE);
  EXPECT_EQ (_mvncsdk2_push (h), GST_FLOW_OK);

  gst_harness_teardown (h);

  NCSDKTensorFilterTestHelper::getInstance ().release ();
}

/** @brief Testing an invoke () failure while writing the input FIFO */
TEST (pipelineMvncsdk2Filter, invokeFailure0_n)
{
  _mvncsdk2_run_invoke_failure (fail_stage_t::FAIL_FIFO_WRT_ELEM);
}

/** @brief Testing an invoke () failure while queueing the inference */
TEST (pipelineMvncsdk2Filter, invokeFailure1_n)
{
  _mvncsdk2_run_invoke_failure (fail_stage_t::FAIL_GRAPH_Q_INFER);
}

/** @brief Testing an invoke () failure while reading the output FIFO */
TEST (pipelineMvncsdk2Filter, invokeFailure2_n)
{
  _mvncsdk2_run_invoke_failure (fail_stage_t::FAIL_FIFO_RD_ELEM);
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
