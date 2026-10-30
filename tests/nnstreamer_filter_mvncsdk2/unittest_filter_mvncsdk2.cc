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
#include <gmodule.h>
#include <gst/check/gstcheck.h>
#include <gst/check/gstharness.h>
#include <gst/check/gsttestclock.h>
#include <gst/gst.h>
#include <nnstreamer_conf.h>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_subplugin.h>
#include <string.h>
#include <tensor_common.h>

#include "../gst/nnstreamer/tensor_filter/tensor_filter_common.h"
#include "NCSDKTensorFilterTestHelper.hh"

static const gchar MVNCSDK2_FW_NAME[] = "movidius-ncsdk2";

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
 * @param props : Extra tensor_filter properties to set
 * @param caps : The caps of the input
 * @return The harness, or NULL if the description cannot be parsed.
 */
static GstHarness *
_mvncsdk2_harness_new (const gchar *props = "", const gchar *caps = MVNCSDK2_IN_CAPS_STR)
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
  desc = g_strdup_printf ("tensor_filter framework=movidius-ncsdk2 model=\"%s\" %s",
      test_model, props);
  h = gst_harness_new_parse (desc);
  g_free (desc);
  g_free (test_model);

  if (h != NULL) {
    gst_harness_set_src_caps_str (h, caps);
  }

  return h;
}

/**
 * @brief Push one input tensor opening with @a tag into the given harness.
 * @details The mocked device carries the tag through to the result, so a test
 *          can tell which input a result belongs to.
 */
static GstFlowReturn
_mvncsdk2_push (GstHarness *h, guint32 tag)
{
  GstBuffer *buf = gst_harness_create_buffer (h, MVNCSDK2_IN_BUF_SIZE);
  GstMapInfo map;

  if (!gst_buffer_map (buf, &map, GST_MAP_WRITE)) {
    ADD_FAILURE () << "Failed to map the input buffer";
    gst_buffer_unref (buf);
    return GST_FLOW_ERROR;
  }
  memset (map.data, 0, map.size);
  memcpy (map.data, &tag, sizeof (tag));
  gst_buffer_unmap (buf, &map);

  return gst_harness_push (h, buf);
}

/**
 * @brief Take the tag out of the oldest result the harness holds.
 * @return The tag, or 0 if there is no result to take.
 */
static guint32
_mvncsdk2_pull (GstHarness *h)
{
  GstBuffer *buf;
  GstMapInfo map;
  guint32 tag = 0;

  if (gst_harness_buffers_in_queue (h) == 0) {
    ADD_FAILURE () << "No result to pull";
    return 0;
  }

  buf = gst_harness_pull (h);
  if (gst_buffer_map (buf, &map, GST_MAP_READ)) {
    memcpy (&tag, map.data, sizeof (tag));
    gst_buffer_unmap (buf, &map);
  } else {
    ADD_FAILURE () << "Failed to map the result buffer";
  }
  gst_buffer_unref (buf);

  return tag;
}

/**
 * @brief Check that the sub-plugin survives a failure in invoke () and then
 *        refuses to run rather than return a result of the wrong input.
 * @details The framework keeps fw_opened set when invoke () returns -1, so the
 *          sub-plugin has to keep its private data and the device handles in
 *          it; closing them here used to leave the private data NULL and the
 *          next buffer dereferenced it. It must not carry on inferring either:
 *          the failure may have left an element in either FIFO, and a later
 *          result could then belong to an earlier frame.
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

  EXPECT_EQ (_mvncsdk2_push (h, 1), GST_FLOW_OK);
  EXPECT_EQ (_mvncsdk2_pull (h), 1U);

  NCSDKTensorFilterTestHelper::getInstance ().setFailStage (stage);
  EXPECT_EQ (_mvncsdk2_push (h, 2), GST_FLOW_ERROR);

  /* The framework still holds this instance, so its handles have to be alive. */
  EXPECT_EQ (NCSDKTensorFilterTestHelper::getInstance ().getNumFifoDestroy (), 0U);

  NCSDKTensorFilterTestHelper::getInstance ().setFailStage (fail_stage_t::NONE);
  EXPECT_EQ (_mvncsdk2_push (h, 3), GST_FLOW_ERROR);
  if (gst_harness_buffers_in_queue (h) != 0) {
    ADD_FAILURE () << "The refused inference still produced a result, tagged "
                   << _mvncsdk2_pull (h) << " (input 3 was given)";
  }

  gst_harness_teardown (h);

  NCSDKTensorFilterTestHelper::getInstance ().release ();
}

/**
 * @brief Testing that a result belongs to the input it was inferred from
 * @details This is what the refusal above protects. Without it the case below
 *          would see the tag of an earlier frame here.
 */
TEST (pipelineMvncsdk2Filter, invokeResultTag)
{
  GstHarness *h;
  guint32 tag;

  NCSDKTensorFilterTestHelper::getInstance ().init (GOOGLE_LENET);

  h = _mvncsdk2_harness_new ();
  if (h == NULL) {
    ADD_FAILURE () << "Failed to parse the tensor_filter description";
    NCSDKTensorFilterTestHelper::getInstance ().release ();
    return;
  }

  for (tag = 1; tag <= 3; tag++) {
    EXPECT_EQ (_mvncsdk2_push (h, tag), GST_FLOW_OK);
    EXPECT_EQ (_mvncsdk2_pull (h), tag);
  }

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
 * @brief Get the tensor_filter a harness made by _mvncsdk2_harness_new () holds.
 * @param h : The harness made by _mvncsdk2_harness_new ()
 * @return The tensor_filter named "tfilter"; unref it after use.
 */
static GstElement *
_mvncsdk2_harness_get_filter (GstHarness *h)
{
  if (GST_IS_BIN (h->element))
    return gst_bin_get_by_name (GST_BIN (h->element), "tfilter");

  return GST_ELEMENT (gst_object_ref (h->element));
}

/**
 * @brief Testing that a requested accelerator does not get in the way
 * @details The sub-plugin ignores the accelerator, and tensor_filter maps a
 *          request onto the list the sub-plugin reports. The property reads
 *          back the string as set, as it did with the v0 sub-plugin.
 */
TEST (pipelineMvncsdk2Filter, invokeAccelerator)
{
  const gchar *accls[] = { "true:npu.movidius", "true:npu", "true:gpu",
    "true:auto", "true:default", "false" };
  GstHarness *h;
  GstElement *filter;
  gchar *props, *accl;
  guint i;

  for (i = 0; i < G_N_ELEMENTS (accls); i++) {
    NCSDKTensorFilterTestHelper::getInstance ().init (GOOGLE_LENET);

    props = g_strdup_printf ("name=tfilter accelerator=%s", accls[i]);
    h = _mvncsdk2_harness_new (props);
    g_free (props);
    if (h == NULL) {
      ADD_FAILURE ()
          << "Failed to parse the tensor_filter description with " << accls[i];
      NCSDKTensorFilterTestHelper::getInstance ().release ();
      continue;
    }

    EXPECT_EQ (_mvncsdk2_push (h, i + 1), GST_FLOW_OK) << accls[i];
    EXPECT_EQ (_mvncsdk2_pull (h), i + 1) << accls[i];

    filter = _mvncsdk2_harness_get_filter (h);
    accl = NULL;
    g_object_get (filter, "accelerator", &accl, NULL);
    EXPECT_STREQ (accl, accls[i]);
    g_free (accl);
    gst_object_unref (filter);

    gst_harness_teardown (h);

    NCSDKTensorFilterTestHelper::getInstance ().release ();
  }
}

/**
 * @brief Testing that an input the model does not take is refused
 * @details tensor_filter takes the input information getModelInfo () reports
 *          when it opens the model, and refuses caps that do not match it.
 */
TEST (pipelineMvncsdk2Filter, invokeDimensionMismatch_n)
{
  GstHarness *h;
  GstElement *filter;
  gchar *dims = NULL;

  NCSDKTensorFilterTestHelper::getInstance ().init (GOOGLE_LENET);

  h = _mvncsdk2_harness_new ("name=tfilter",
      "other/tensors,format=static,num_tensors=1,framerate=(fraction)0/1,"
      "dimensions=(string)3:112:112:1,types=(string)float32");
  if (h == NULL) {
    ADD_FAILURE () << "Failed to parse the tensor_filter description";
    NCSDKTensorFilterTestHelper::getInstance ().release ();
    return;
  }

  EXPECT_EQ (_mvncsdk2_push (h, 1), GST_FLOW_NOT_NEGOTIATED);
  EXPECT_EQ (gst_harness_buffers_in_queue (h), 0U);

  /* The model was opened, and its input is what the caps were checked against. */
  filter = _mvncsdk2_harness_get_filter (h);
  if (filter != NULL) {
    g_object_get (filter, "input", &dims, NULL);
    EXPECT_STREQ (dims, "3:224:224:1");
    g_free (dims);
    gst_object_unref (filter);
  } else {
    ADD_FAILURE () << "Failed to get the tensor_filter";
  }

  gst_harness_teardown (h);

  NCSDKTensorFilterTestHelper::getInstance ().release ();
}

/**
 * @brief Testing that is-updatable is turned off for an opened model
 * @details The sub-plugin cannot reload a model, so a model change while it
 *          runs must not be taken as a reload.
 */
TEST (pipelineMvncsdk2Filter, isUpdatable_n)
{
  GstHarness *h;
  GstElement *filter;
  gboolean updatable = TRUE;

  NCSDKTensorFilterTestHelper::getInstance ().init (GOOGLE_LENET);

  h = _mvncsdk2_harness_new ("name=tfilter is-updatable=true");
  if (h == NULL) {
    ADD_FAILURE () << "Failed to parse the tensor_filter description";
    NCSDKTensorFilterTestHelper::getInstance ().release ();
    return;
  }

  EXPECT_EQ (_mvncsdk2_push (h, 1), GST_FLOW_OK);
  EXPECT_EQ (_mvncsdk2_pull (h), 1U);

  filter = _mvncsdk2_harness_get_filter (h);
  if (filter != NULL) {
    g_object_set (filter, "is-updatable", TRUE, NULL);
    g_object_get (filter, "is-updatable", &updatable, NULL);
    EXPECT_FALSE (updatable);
    gst_object_unref (filter);
  } else {
    ADD_FAILURE () << "Failed to get the tensor_filter";
  }

  gst_harness_teardown (h);

  NCSDKTensorFilterTestHelper::getInstance ().release ();
}

/**
 * @brief Open the sub-plugin directly on the mocked device.
 * @param fw : The movidius-ncsdk2 sub-plugin to open
 * @param[out] prop : The properties to open with; free model_files with g_strfreev ()
 * @return The private data of the opened instance, or NULL on failure.
 */
static void *
_mvncsdk2_open (const GstTensorFilterFramework *fw, GstTensorFilterProperties *prop)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar **models = g_new0 (gchar *, 2);
  void *private_data = NULL;

  if (root_path == NULL) {
    root_path = "..";
  }

  models[0] = g_build_filename (root_path, "tests", "test_models", "models",
      "google_lenet_ncsdk_caffe_1.graph", NULL);

  memset (prop, 0, sizeof (*prop));
  prop->fwname = MVNCSDK2_FW_NAME;
  prop->model_files = (const char **) models;
  prop->num_models = 1;

  if (fw->open (prop, &private_data) != 0) {
    ADD_FAILURE () << "Failed to open " << MVNCSDK2_FW_NAME;
    return NULL;
  }

  EXPECT_NE (private_data, nullptr);

  return private_data;
}

/**
 * @brief Testing that unloading the sub-plugin unregisters it
 * @details The v0 name member aliases invoke in the v1 layout, so reading the
 *          name from the framework struct would unregister nothing.
 */
TEST (filterMvncsdk2, finiUnregister)
{
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (MVNCSDK2_FW_NAME);
  const gchar *path;
  GModule *module;
  gpointer init_sym = NULL;
  gpointer fini_sym = NULL;

  ASSERT_NE (fw, nullptr);
  path = nnsconf_get_fullpath (MVNCSDK2_FW_NAME, NNSCONF_PATH_FILTERS);
  ASSERT_NE (path, nullptr);
  module = g_module_open (path, G_MODULE_BIND_LOCAL);
  ASSERT_NE (module, nullptr);

  if (g_module_symbol (module, "init_filter_mvncsdk2", &init_sym)
      && g_module_symbol (module, "fini_filter_mvncsdk2", &fini_sym)) {
    ((void (*) (void)) fini_sym) ();
    /* Nothing is left to remove once fini () has unregistered it. */
    EXPECT_FALSE (unregister_subplugin (NNS_SUBPLUGIN_FILTER, MVNCSDK2_FW_NAME));

    /* Register it again for the other cases. */
    ((void (*) (void)) init_sym) ();
    EXPECT_EQ (nnstreamer_filter_find (MVNCSDK2_FW_NAME), fw);
  } else {
    ADD_FAILURE () << "Failed to find the init/fini functions in " << path;
  }

  g_module_close (module);
}

/**
 * @brief Testing the information the v1 sub-plugin reports without a model
 */
TEST (filterMvncsdk2, frameworkInfo)
{
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (MVNCSDK2_FW_NAME);
  GstTensorFilterProperties prop;
  GstTensorFilterFrameworkInfo info;

  ASSERT_NE (fw, nullptr);
  ASSERT_EQ (fw->version, GST_TENSOR_FILTER_FRAMEWORK_V1);
  ASSERT_NE (fw->getFrameworkInfo, nullptr);
  EXPECT_NE (fw->getModelInfo, nullptr);
  EXPECT_NE (fw->eventHandler, nullptr);

  memset (&prop, 0, sizeof (prop));
  memset (&info, 0, sizeof (info));
  EXPECT_EQ (fw->getFrameworkInfo (fw, &prop, NULL, &info), 0);

  EXPECT_STREQ (info.name, MVNCSDK2_FW_NAME);
  EXPECT_FALSE (info.allow_in_place);
  EXPECT_FALSE (info.allocate_in_invoke);
  EXPECT_FALSE (info.run_without_model);
  EXPECT_FALSE (info.verify_model_path);
  ASSERT_EQ (info.num_hw, 2);
  ASSERT_NE (info.hw_list, nullptr);
  EXPECT_EQ (info.hw_list[0], ACCL_NPU);
  EXPECT_EQ (info.hw_list[1], ACCL_NPU_MOVIDIUS);
  EXPECT_EQ (info.accl_auto, ACCL_NPU_MOVIDIUS);
  EXPECT_EQ (info.accl_default, ACCL_NPU_MOVIDIUS);
}

/**
 * @brief Testing the accelerators the sub-plugin accepts
 */
TEST (filterMvncsdk2, checkHwAvailability)
{
  EXPECT_TRUE (gst_tensor_filter_check_hw_availability (MVNCSDK2_FW_NAME, ACCL_NPU, NULL));
  EXPECT_TRUE (gst_tensor_filter_check_hw_availability (
      MVNCSDK2_FW_NAME, ACCL_NPU_MOVIDIUS, NULL));
}

/**
 * @brief Testing the accelerators the sub-plugin refuses
 */
TEST (filterMvncsdk2, checkHwAvailability_n)
{
  EXPECT_FALSE (gst_tensor_filter_check_hw_availability (MVNCSDK2_FW_NAME, ACCL_CPU, NULL));
  EXPECT_FALSE (gst_tensor_filter_check_hw_availability (MVNCSDK2_FW_NAME, ACCL_GPU, NULL));
  EXPECT_FALSE (gst_tensor_filter_check_hw_availability (
      MVNCSDK2_FW_NAME, ACCL_NPU_VIVANTE, NULL));
}

/**
 * @brief Testing the tensor information of an opened model
 */
TEST (filterMvncsdk2, getModelInfo)
{
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (MVNCSDK2_FW_NAME);
  GstTensorFilterProperties prop;
  GstTensorsInfo in_info, out_info;
  GstTensorInfo *info;
  void *private_data;

  ASSERT_NE (fw, nullptr);
  ASSERT_EQ (fw->version, GST_TENSOR_FILTER_FRAMEWORK_V1);
  NCSDKTensorFilterTestHelper::getInstance ().init (GOOGLE_LENET);

  private_data = _mvncsdk2_open (fw, &prop);
  if (private_data != NULL) {
    gst_tensors_info_init (&in_info);
    gst_tensors_info_init (&out_info);

    EXPECT_EQ (fw->getModelInfo (fw, &prop, private_data, GET_IN_OUT_INFO, &in_info, &out_info),
        0);

    EXPECT_EQ (in_info.num_tensors, 1U);
    info = gst_tensors_info_get_nth_info (&in_info, 0);
    EXPECT_EQ (info->type, _NNS_FLOAT32);
    EXPECT_EQ (info->dimension[0], (guint) GOOGLE_LENET_IN_DIM_C);
    EXPECT_EQ (info->dimension[1], (guint) GOOGLE_LENET_IN_DIM_W);
    EXPECT_EQ (info->dimension[2], (guint) GOOGLE_LENET_IN_DIM_H);
    EXPECT_EQ (info->dimension[3], (guint) GOOGLE_LENET_IN_DIM_N);

    EXPECT_EQ (out_info.num_tensors, 1U);
    info = gst_tensors_info_get_nth_info (&out_info, 0);
    EXPECT_EQ (info->type, _NNS_FLOAT32);
    EXPECT_EQ (info->dimension[0], (guint) GOOGLE_LENET_OUT_DIM_C);
    EXPECT_EQ (info->dimension[1], (guint) GOOGLE_LENET_OUT_DIM_W);
    EXPECT_EQ (info->dimension[2], (guint) GOOGLE_LENET_OUT_DIM_H);
    EXPECT_EQ (info->dimension[3], (guint) GOOGLE_LENET_OUT_DIM_N);

    gst_tensors_info_free (&in_info);
    gst_tensors_info_free (&out_info);

    fw->close (&prop, &private_data);
    EXPECT_EQ (private_data, nullptr);
  }

  g_strfreev ((gchar **) prop.model_files);
  NCSDKTensorFilterTestHelper::getInstance ().release ();
}

/**
 * @brief Testing that the sub-plugin refuses to set the input information
 */
TEST (filterMvncsdk2, getModelInfoSetInput_n)
{
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (MVNCSDK2_FW_NAME);
  GstTensorFilterProperties prop;
  GstTensorsInfo in_info, out_info;
  void *private_data;

  ASSERT_NE (fw, nullptr);
  ASSERT_EQ (fw->version, GST_TENSOR_FILTER_FRAMEWORK_V1);
  NCSDKTensorFilterTestHelper::getInstance ().init (GOOGLE_LENET);

  private_data = _mvncsdk2_open (fw, &prop);
  if (private_data != NULL) {
    gst_tensors_info_init (&in_info);
    gst_tensors_info_init (&out_info);

    EXPECT_EQ (fw->getModelInfo (fw, &prop, private_data, SET_INPUT_INFO, &in_info, &out_info),
        -ENOENT);

    gst_tensors_info_free (&in_info);
    gst_tensors_info_free (&out_info);

    fw->close (&prop, &private_data);
  }

  g_strfreev ((gchar **) prop.model_files);
  NCSDKTensorFilterTestHelper::getInstance ().release ();
}

/**
 * @brief Testing that the sub-plugin refuses the tensor information without a model
 */
TEST (filterMvncsdk2, getModelInfoNotOpened_n)
{
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (MVNCSDK2_FW_NAME);
  GstTensorFilterProperties prop;
  GstTensorsInfo in_info, out_info;

  ASSERT_NE (fw, nullptr);
  ASSERT_EQ (fw->version, GST_TENSOR_FILTER_FRAMEWORK_V1);

  memset (&prop, 0, sizeof (prop));
  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  EXPECT_EQ (fw->getModelInfo (fw, &prop, NULL, GET_IN_OUT_INFO, &in_info, &out_info), -EINVAL);
  EXPECT_EQ (in_info.num_tensors, 0U);
  EXPECT_EQ (out_info.num_tensors, 0U);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Testing that the sub-plugin handles no event
 * @details A RELOAD_MODEL answer other than -ENOENT would make tensor_filter
 *          keep is-updatable set on an opened instance and then reload into a
 *          sub-plugin that cannot do so.
 */
TEST (filterMvncsdk2, eventHandler_n)
{
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (MVNCSDK2_FW_NAME);
  const event_ops ops[] = { DESTROY_NOTIFY, RELOAD_MODEL, CUSTOM_PROP, SET_INPUT_PROP,
    SET_OUTPUT_PROP, SET_ACCELERATOR, CHECK_HW_AVAILABILITY, SUSPEND, RESUME };
  GstTensorFilterFrameworkEventData data;
  GstTensorFilterProperties prop;
  void *private_data;
  guint i;

  ASSERT_NE (fw, nullptr);
  ASSERT_EQ (fw->version, GST_TENSOR_FILTER_FRAMEWORK_V1);
  NCSDKTensorFilterTestHelper::getInstance ().init (GOOGLE_LENET);

  private_data = _mvncsdk2_open (fw, &prop);
  if (private_data != NULL) {
    for (i = 0; i < G_N_ELEMENTS (ops); i++) {
      memset (&data, 0, sizeof (data));
      EXPECT_EQ (fw->eventHandler (fw, &prop, private_data, ops[i], &data), -ENOENT)
          << "event " << ops[i];
      EXPECT_EQ (fw->eventHandler (fw, &prop, NULL, ops[i], &data), -ENOENT)
          << "event " << ops[i] << " without a model";
    }

    fw->close (&prop, &private_data);
  }

  g_strfreev ((gchar **) prop.model_files);
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
