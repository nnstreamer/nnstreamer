/**
 * @file        unittest_tizen_capi.cpp
 * @date        13 Mar 2019
 * @brief       Unit test for Tizen CAPI of NNStreamer. Basis of TCT in the future.
 * @see         https://github.com/nnsuite/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <tizen-api.h>
#include <gtest/gtest.h>
#include <glib.h>

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_construct_destruct, dummy_01)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! fakesink";
  nns_pipeline_h handle;
  int status = nns_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_destroy (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_construct_destruct, dummy_02)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224 ! tensor_converter ! fakesink";
  nns_pipeline_h handle;
  int status = nns_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_destroy (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_construct_destruct, dummy_03)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224 ! tensor_converter ! valve name=valvex ! tensor_sink name=sinkx";
  nns_pipeline_h handle;
  int status = nns_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_destroy (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_playstop, dummy_01)
{
  const char *pipeline = "videotestsrc is-live=true num-buffers=30 framerate=60/1 ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224 ! tensor_converter ! valve name=valvex ! valve name=valvey ! input-selector name=is01 ! tensor_sink name=sinkx";
  nns_pipeline_h handle;
  nns_pipeline_state state;
  int status = nns_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_start (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  status = nns_pipeline_getstate(handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, NNS_PIPELINE_UNKNOWN);
  EXPECT_NE (state, NNS_PIPELINE_NULL);

  g_usleep(50000); /* 50ms. Let a few frames flow. */
  status = nns_pipeline_getstate(handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  EXPECT_EQ (state, NNS_PIPELINE_PLAYING);

  status = nns_pipeline_stop (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  g_usleep(50000); /* 50ms. Let a few frames flow. */

  status = nns_pipeline_getstate(handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  EXPECT_EQ (state, NNS_PIPELINE_PAUSED);

  status = nns_pipeline_destroy (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
}

/**
 * @brief Main gtest
 */
int main (int argc, char **argv)
{
  testing::InitGoogleTest (&argc, argv);

  return RUN_ALL_TESTS ();
}
