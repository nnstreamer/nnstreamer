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
#include <glib/gstdio.h> /* GStatBuf */

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
 * @brief Test NNStreamer pipeline construct with non-existent filter
 */
TEST (nnstreamer_capi_construct_destruct, failed_01)
{
  const char *pipeline = "nonexistsrc ! fakesink";
  nns_pipeline_h handle;
  int status = nns_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, NNS_ERROR_PIPELINE_FAIL);
}

/**
 * @brief Test NNStreamer pipeline construct with erroneous pipeline
 */
TEST (nnstreamer_capi_construct_destruct, failed_02)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! audioconvert ! fakesink";
  nns_pipeline_h handle;
  int status = nns_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, NNS_ERROR_PIPELINE_FAIL);
}


/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_playstop, dummy_01)
{
  const char *pipeline = "videotestsrc is-live=true num-buffers=30 ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! valve name=valvex ! valve name=valvey ! input-selector name=is01 ! tensor_sink name=sinkx";
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
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_playstop, dummy_02)
{
  const char *pipeline = "videotestsrc is-live=true num-buffers=30 ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! valve name=valvex ! valve name=valvey ! input-selector name=is01 ! tensor_sink name=sinkx";
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

  /* Resume playing */
  status = nns_pipeline_start (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  EXPECT_NE (state, NNS_PIPELINE_UNKNOWN);
  EXPECT_NE (state, NNS_PIPELINE_NULL);

  g_usleep(50000); /* 50ms. Enough to empty the queue */
  status = nns_pipeline_stop (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_getstate(handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  EXPECT_EQ (state, NNS_PIPELINE_PAUSED);

  status = nns_pipeline_destroy (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_valve, test01)
{
  const gchar *_tmpdir = g_get_tmp_dir();
  const gchar *_dirname = "nns-tizen-XXXXXX";
  gchar *fullpath = g_build_path ("/", _tmpdir, _dirname, NULL);
  gchar *dir = g_mkdtemp((gchar *) fullpath);
  gchar *file1 = g_build_path("/", dir, "valve1", NULL);
  gchar *pipeline = g_strdup_printf("videotestsrc is-live=true num-buffers=20 ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=16,height=16,framerate=60/1 ! tensor_converter ! queue ! valve name=valve1 ! filesink location=\"%s\"",
    file1);
  GStatBuf buf;

  nns_pipeline_h handle;
  nns_pipeline_state state;
  nns_valve_h valve1;

  int status = nns_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  EXPECT_TRUE (dir != NULL);

  status = nns_pipeline_valve_gethandle (handle, "valve1", &valve1);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_start (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_valve_control (valve1, 1); /* close */
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_getstate(handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, NNS_PIPELINE_UNKNOWN);
  EXPECT_NE (state, NNS_PIPELINE_NULL);

  g_usleep(100000); /* 100ms. Let a few frames flow. */
  status = nns_pipeline_stop (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = g_lstat (file1, &buf);
  EXPECT_EQ (status, 0);
  EXPECT_EQ (buf.st_size, 0);

  status = nns_pipeline_start (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_valve_control (valve1, 0); /* open */
  EXPECT_EQ (status, NNS_ERROR_NONE);


  g_usleep(50000); /* 50ms. Let a few frames flow. */
  status = nns_pipeline_stop (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_destroy (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = g_lstat (file1, &buf);
  EXPECT_EQ (status, 0);
  EXPECT_GE (buf.st_size, 2048); /* At least two frames during 50ms */
  EXPECT_LE (buf.st_size, 4096); /* At most four frames during 50ms */

  g_free (fullpath);
  g_free (file1);
  g_free (pipeline);
}

/**
 * @brief Main gtest
 */
int main (int argc, char **argv)
{
  testing::InitGoogleTest (&argc, argv);

  return RUN_ALL_TESTS ();
}
