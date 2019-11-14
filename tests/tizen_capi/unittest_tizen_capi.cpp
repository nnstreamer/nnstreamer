/**
 * @file        unittest_tizen_capi.cpp
 * @date        13 Mar 2019
 * @brief       Unit test for Tizen CAPI of NNStreamer. Basis of TCT in the future.
 * @see         https://github.com/nnsuite/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <nnstreamer.h>
#include <nnstreamer-single.h>
#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h> /* GStatBuf */
#include <nnstreamer-capi-private.h>

#define SINGLE_DEF_TIMEOUT_MSEC 10000

/**
 * @brief Struct to check the pipeline state changes.
 */
typedef struct
{
  gboolean paused;
  gboolean playing;
} TestPipeState;

#if defined (__TIZEN__)
/**
 * @brief Test NNStreamer pipeline construct with Tizen cam
 * @details Failure case to check permission (camera privilege)
 */
TEST (nnstreamer_capi_construct_destruct, tizen_cam_fail_01_n)
{
  ml_pipeline_h handle;
  gchar *pipeline;
  int status;

  pipeline = g_strdup_printf ("%s ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=240 ! tensor_converter ! tensor_sink",
      ML_TIZEN_CAM_VIDEO_SRC);

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_PERMISSION_DENIED);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline construct with Tizen cam
 * @details Failure case to check permission (camera privilege)
 */
TEST (nnstreamer_capi_construct_destruct, tizen_cam_fail_02_n)
{
  ml_pipeline_h handle;
  gchar *pipeline;
  int status;

  pipeline = g_strdup_printf ("%s ! audioconvert ! audio/x-raw,format=S16LE,rate=16000 ! tensor_converter ! tensor_sink",
      ML_TIZEN_CAM_AUDIO_SRC);

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_PERMISSION_DENIED);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline construct with Tizen internal API.
 */
TEST (nnstreamer_capi_construct_destruct, tizen_internal_01_p)
{
  ml_pipeline_h handle;
  gchar *pipeline;
  int status;

  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=320,height=240 ! tensor_converter ! tensor_sink");

  status = ml_pipeline_construct_internal (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline construct with Tizen internal API.
 */
TEST (nnstreamer_capi_construct_destruct, tizen_internal_02_p)
{
  ml_pipeline_h handle;
  gchar *pipeline;
  int status;

  pipeline = g_strdup_printf ("audiotestsrc ! audioconvert ! audio/x-raw,format=S16LE,rate=16000 ! tensor_converter ! tensor_sink");

  status = ml_pipeline_construct_internal (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}
#endif /* __TIZEN__ */

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_construct_destruct, dummy_01)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! fakesink";
  ml_pipeline_h handle;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_construct_destruct, dummy_02)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224 ! tensor_converter ! fakesink";
  ml_pipeline_h handle;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_construct_destruct, dummy_03)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224 ! tensor_converter ! valve name=valvex ! tensor_sink name=sinkx";
  ml_pipeline_h handle;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline construct with non-existent filter
 */
TEST (nnstreamer_capi_construct_destruct, failure_01_n)
{
  const char *pipeline = "nonexistsrc ! fakesink";
  ml_pipeline_h handle;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_STREAMS_PIPE);
}

/**
 * @brief Test NNStreamer pipeline construct with erroneous pipeline
 */
TEST (nnstreamer_capi_construct_destruct, failure_02_n)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! audioconvert ! fakesink";
  ml_pipeline_h handle;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_STREAMS_PIPE);
}

#define wait_for_start(handle, state, status) \
do {\
  int counter = 0;\
  while ((state == ML_PIPELINE_STATE_PAUSED || \
          state == ML_PIPELINE_STATE_READY) && counter < 20) {\
    g_usleep (50000);\
    counter ++;\
    status = ml_pipeline_get_state (handle, &state);\
    EXPECT_EQ (status, ML_ERROR_NONE);\
  }\
} while (0)\

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_playstop, dummy_01)
{
  const char *pipeline = "videotestsrc is-live=true ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! valve name=valvex ! valve name=valvey ! input-selector name=is01 ! tensor_sink name=sinkx";
  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  g_usleep (50000); /* 50ms is good for general systems, but not enough for emulators to start gst pipeline. Let a few frames flow. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  wait_for_start (handle, state, status);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PLAYING);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (50000); /* 50ms. Let a few frames flow. */

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PAUSED);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}


/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_playstop, dummy_02)
{
  const char *pipeline = "videotestsrc is-live=true ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! valve name=valvex ! valve name=valvey ! input-selector name=is01 ! tensor_sink name=sinkx";
  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  g_usleep (50000); /* 50ms is good for general systems, but not enough for emulators to start gst pipeline. Let a few frames flow. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  wait_for_start (handle, state, status);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PLAYING);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (50000); /* 50ms. Let a few frames flow. */

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PAUSED);

  /* Resume playing */
  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  g_usleep (50000); /* 50ms. Enough to empty the queue */
  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PAUSED);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_valve, test01)
{
  const gchar *_tmpdir = g_get_tmp_dir ();
  const gchar *_dirname = "nns-tizen-XXXXXX";
  gchar *fullpath = g_build_path ("/", _tmpdir, _dirname, NULL);
  gchar *dir = g_mkdtemp ((gchar *) fullpath);
  gchar *file1 = g_build_path ("/", dir, "valve1", NULL);
  gchar *pipeline =
      g_strdup_printf
      ("videotestsrc is-live=true ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=16,height=16,framerate=10/1 ! tensor_converter ! queue ! valve name=valve1 ! filesink location=\"%s\"",
      file1);
  GStatBuf buf;

  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  ml_pipeline_valve_h valve1;

  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_TRUE (dir != NULL);

  status = ml_pipeline_valve_get_handle (handle, "valve1", &valve1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_valve_set_open (valve1, false); /* close */
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (50000); /* 50ms. Wait for the pipeline stgart. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  wait_for_start (handle, state, status);
  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = g_lstat (file1, &buf);
  EXPECT_EQ (status, 0);
  EXPECT_EQ (buf.st_size, 0);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_valve_set_open (valve1, true); /* open */
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_valve_release_handle (valve1); /* release valve handle */
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (500000); /* 500ms. Let a few frames flow. (10Hz x 0.5s --> 5)*/

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = g_lstat (file1, &buf);
  EXPECT_EQ (status, 0);
  EXPECT_GE (buf.st_size, 2048); /* At least two frames during 500ms */
  EXPECT_LE (buf.st_size, 6144); /* At most six frames during 500ms */
  EXPECT_EQ (buf.st_size % 1024, 0); /* It should be divided by 1024 */

  g_free (fullpath);
  g_free (file1);
  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline valve
 * @detail Failure case to handle valve element with invalid param.
 */
TEST (nnstreamer_capi_valve, failure_01_n)
{
  ml_pipeline_h handle;
  ml_pipeline_valve_h valve_h;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : pipe */
  status = ml_pipeline_valve_get_handle (NULL, "valvex", &valve_h);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : name */
  status = ml_pipeline_valve_get_handle (handle, NULL, &valve_h);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : wrong name */
  status = ml_pipeline_valve_get_handle (handle, "wrongname", &valve_h);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : invalid type */
  status = ml_pipeline_valve_get_handle (handle, "sinkx", &valve_h);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : handle */
  status = ml_pipeline_valve_get_handle (handle, "valvex", NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

G_LOCK_DEFINE_STATIC(callback_lock);
/**
 * @brief A tensor-sink callback for sink handle in a pipeline
 */
static void
test_sink_callback_dm01 (const ml_tensors_data_h data,
    const ml_tensors_info_h info, void *user_data)
{
  gchar *filepath = (gchar *) user_data;
  unsigned int i, num = 0;
  void *data_ptr;
  size_t data_size;
  int status;
  FILE *fp = g_fopen (filepath, "a");

  if (fp == NULL)
    return;

  G_LOCK(callback_lock);
  ml_tensors_info_get_count (info, &num);

  for (i = 0; i < num; i++) {
    status = ml_tensors_data_get_tensor_data (data, i, &data_ptr, &data_size);
    if (status == ML_ERROR_NONE)
      fwrite (data_ptr, data_size, 1, fp);
  }
  G_UNLOCK(callback_lock);

  fclose (fp);
}

/**
 * @brief A tensor-sink callback for sink handle in a pipeline
 */
static void
test_sink_callback_count (const ml_tensors_data_h data,
    const ml_tensors_info_h info, void *user_data)
{
  guint *count = (guint *) user_data;

  G_LOCK(callback_lock);
  *count = *count + 1;
  G_UNLOCK(callback_lock);
}

/**
 * @brief Pipeline state changed callback
 */
static void
test_pipe_state_callback (ml_pipeline_state_e state, void *user_data)
{
  TestPipeState *pipe_state;

  G_LOCK(callback_lock);
  pipe_state = (TestPipeState *) user_data;

  switch (state) {
    case ML_PIPELINE_STATE_PAUSED:
      pipe_state->paused = TRUE;
      break;
    case ML_PIPELINE_STATE_PLAYING:
      pipe_state->playing = TRUE;
      break;
    default:
      break;
  }
  G_UNLOCK(callback_lock);
}

/**
 * @brief compare the two files.
 */
static int
file_cmp (const gchar * f1, const gchar * f2)
{
  gboolean r;
  gchar *content1, *content2;
  gsize len1, len2;
  int cmp = 0;

  r = g_file_get_contents (f1, &content1, &len1, NULL);
  if (r != TRUE)
    return -1;

  r = g_file_get_contents (f2, &content2, &len2, NULL);
  if (r != TRUE) {
    g_free (content1);
    return -2;
  }

  if (len1 == len2) {
    cmp = memcmp (content1, content2, len1);
  } else {
    cmp = 1;
  }

  g_free (content1);
  g_free (content2);

  return cmp;
}

/**
 * @brief Test NNStreamer pipeline sink
 */
TEST (nnstreamer_capi_sink, dummy_01)
{
  const gchar *_tmpdir = g_get_tmp_dir ();
  const gchar *_dirname = "nns-tizen-XXXXXX";
  gchar *fullpath = g_build_path ("/", _tmpdir, _dirname, NULL);
  gchar *dir = g_mkdtemp ((gchar *) fullpath);

  ASSERT_NE (dir, (gchar *) NULL);

  gchar *file1 = g_build_path ("/", dir, "original", NULL);
  gchar *file2 = g_build_path ("/", dir, "sink", NULL);
  gchar *pipeline =
      g_strdup_printf
      ("videotestsrc num-buffers=3 ! videoconvert ! video/x-raw,format=BGRx,width=64,height=48,famerate=60/1 ! tee name=t t. ! queue ! filesink location=\"%s\"  t. ! queue ! tensor_converter ! tensor_sink name=sinkx",
      file1);
  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  ml_pipeline_sink_h sinkhandle;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_dm01, file2, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (sinkhandle != NULL);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  g_usleep (100000); /* 100ms. Let a few frames flow. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PLAYING);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PAUSED);

  status = ml_pipeline_sink_unregister (sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);

  /* File Comparison to check the integrity */
  EXPECT_EQ (file_cmp (file1, file2), 0);
}

/**
 * @brief Test NNStreamer pipeline sink
 */
TEST (nnstreamer_capi_sink, dummy_02)
{
  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  ml_pipeline_sink_h sinkhandle;
  gchar *pipeline;
  int status;
  guint *count_sink;
  TestPipeState *pipe_state;

  /* pipeline with appsink */
  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! tensor_converter ! appsink name=sinkx");

  count_sink = (guint *) g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink != NULL);
  *count_sink = 0;

  pipe_state = (TestPipeState *) g_new0 (TestPipeState, 1);
  ASSERT_TRUE (pipe_state != NULL);

  status = ml_pipeline_construct (pipeline, test_pipe_state_callback, pipe_state, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (sinkhandle != NULL);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (100000); /* 100ms. Let a few frames flow. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PLAYING);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PAUSED);

  status = ml_pipeline_sink_unregister (sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_TRUE (*count_sink > 0U);
  EXPECT_TRUE (pipe_state->paused);
  EXPECT_TRUE (pipe_state->playing);

  g_free (pipeline);
  g_free (count_sink);
  g_free (pipe_state);
}

/**
 * @brief Test NNStreamer pipeline sink
 */
TEST (nnstreamer_capi_sink, register_duplicated)
{
  ml_pipeline_h handle;
  ml_pipeline_sink_h sinkhandle0, sinkhandle1;
  gchar *pipeline;
  int status;
  guint *count_sink0, *count_sink1;
  TestPipeState *pipe_state;

  /* pipeline with appsink */
  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! tensor_converter ! appsink name=sinkx");

  count_sink0 = (guint *) g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink0 != NULL);
  *count_sink0 = 0;

  count_sink1 = (guint *) g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink1 != NULL);
  *count_sink1 = 0;

  pipe_state = (TestPipeState *) g_new0 (TestPipeState, 1);
  ASSERT_TRUE (pipe_state != NULL);

  status = ml_pipeline_construct (pipeline, test_pipe_state_callback, pipe_state, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_count, count_sink0, &sinkhandle0);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (sinkhandle0 != NULL);

  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_count, count_sink1, &sinkhandle1);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (sinkhandle1 != NULL);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (100000); /* 100ms. Let a few frames flow. */

  status = ml_pipeline_sink_unregister (sinkhandle0);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_unregister (sinkhandle1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_TRUE (*count_sink0 > 0U);
  EXPECT_TRUE (*count_sink1 > 0U);
  EXPECT_TRUE (pipe_state->paused);
  EXPECT_TRUE (pipe_state->playing);

  g_free (pipeline);
  g_free (count_sink0);
  g_free (count_sink1);
  g_free (pipe_state);
}

/**
 * @brief Test NNStreamer pipeline sink
 * @detail Failure case to register callback with invalid param.
 */
TEST (nnstreamer_capi_sink, failure_01_n)
{
  ml_pipeline_h handle;
  ml_pipeline_sink_h sinkhandle;
  gchar *pipeline;
  int status;
  guint *count_sink;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  count_sink = (guint *) g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink != NULL);
  *count_sink = 0;

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : pipe */
  status = ml_pipeline_sink_register (NULL, "sinkx", test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : name */
  status = ml_pipeline_sink_register (handle, NULL, test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : wrong name */
  status = ml_pipeline_sink_register (handle, "wrongname", test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : invalid type */
  status = ml_pipeline_sink_register (handle, "valvex", test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : callback */
  status = ml_pipeline_sink_register (handle, "sinkx", NULL, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : handle */
  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_count, count_sink, NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (sinkhandle != NULL);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (100000); /* 100ms. Let a few frames flow. */

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_unregister (sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_TRUE (*count_sink > 0U);

  g_free (pipeline);
  g_free (count_sink);
}

/**
 * @brief Test NNStreamer pipeline src
 */
TEST (nnstreamer_capi_src, dummy_01)
{
  const gchar *_tmpdir = g_get_tmp_dir ();
  const gchar *_dirname = "nns-tizen-XXXXXX";
  gchar *fullpath = g_build_path ("/", _tmpdir, _dirname, NULL);
  gchar *dir = g_mkdtemp ((gchar *) fullpath);
  gchar *file1 = g_build_path ("/", dir, "output", NULL);
  gchar *pipeline =
      g_strdup_printf
      ("appsrc name=srcx ! other/tensor,dimension=(string)4:1:1:1,type=(string)uint8,framerate=(fraction)0/1 ! filesink location=\"%s\"",
      file1);
  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  ml_pipeline_src_h srchandle;
  int status;
  ml_tensors_info_h info;
  ml_tensors_data_h data1, data2;
  unsigned int count = 0;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;
  ml_tensor_dimension dim = { 0, };

  int i;
  uint8_t *uintarray1[10];
  uint8_t *uintarray2[10];
  uint8_t *content;
  gsize len;

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (dir != NULL);
  for (i = 0; i < 10; i++) {
    uintarray1[i] = (uint8_t *) g_malloc (4);
    ASSERT_TRUE (uintarray1[i] != NULL);
    uintarray1[i][0] = i + 4;
    uintarray1[i][1] = i + 1;
    uintarray1[i][2] = i + 3;
    uintarray1[i][3] = i + 2;

    uintarray2[i] = (uint8_t *) g_malloc (4);
    ASSERT_TRUE (uintarray2[i] != NULL);
    uintarray2[i][0] = i + 3;
    uintarray2[i][1] = i + 2;
    uintarray2[i][2] = i + 1;
    uintarray2[i][3] = i + 4;
    /* These will be free'ed by gstreamer (ML_PIPELINE_BUF_POLICY_AUTO_FREE) */
    /** @todo Check whether gstreamer really deallocates this */
  }

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  status = ml_pipeline_src_get_handle (handle, "srcx", &srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_get_tensors_info (srchandle, &info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_get_count (info, &count);
  EXPECT_EQ (count, 1U);

  ml_tensors_info_get_tensor_type (info, 0, &type);
  EXPECT_EQ (type, ML_TENSOR_TYPE_UINT8);

  ml_tensors_info_get_tensor_dimension (info, 0, dim);
  EXPECT_EQ (dim[0], 4U);
  EXPECT_EQ (dim[1], 1U);
  EXPECT_EQ (dim[2], 1U);
  EXPECT_EQ (dim[3], 1U);

  status = ml_tensors_data_create (info, &data1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_destroy (info);

  status = ml_tensors_data_set_tensor_data (data1, 0, uintarray1[0], 4);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_input_data (srchandle, data1, ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (50000); /* 50ms. Wait a bit. */

  status = ml_pipeline_src_input_data (srchandle, data1, ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (50000); /* 50ms. Wait a bit. */

  status = ml_pipeline_src_release_handle (srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_get_handle (handle, "srcx", &srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_get_tensors_info (srchandle, &info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_get_count (info, &count);
  EXPECT_EQ (count, 1U);

  ml_tensors_info_get_tensor_type (info, 0, &type);
  EXPECT_EQ (type, ML_TENSOR_TYPE_UINT8);

  ml_tensors_info_get_tensor_dimension (info, 0, dim);
  EXPECT_EQ (dim[0], 4U);
  EXPECT_EQ (dim[1], 1U);
  EXPECT_EQ (dim[2], 1U);
  EXPECT_EQ (dim[3], 1U);

  for (i = 0; i < 10; i++) {
    status = ml_tensors_data_set_tensor_data (data1, 0, uintarray1[i], 4);
    EXPECT_EQ (status, ML_ERROR_NONE);

    status = ml_pipeline_src_input_data (srchandle, data1, ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
    EXPECT_EQ (status, ML_ERROR_NONE);

    status = ml_tensors_data_create (info, &data2);
    EXPECT_EQ (status, ML_ERROR_NONE);

    status = ml_tensors_data_set_tensor_data (data2, 0, uintarray2[i], 4);
    EXPECT_EQ (status, ML_ERROR_NONE);

    status = ml_pipeline_src_input_data (srchandle, data2, ML_PIPELINE_BUF_POLICY_AUTO_FREE);
    EXPECT_EQ (status, ML_ERROR_NONE);

    g_usleep (50000); /* 50ms. Wait a bit. */
  }

  status = ml_pipeline_src_release_handle (srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (50000); /* Wait for the pipeline to flush all */

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);

  EXPECT_TRUE (g_file_get_contents (file1, (gchar **) &content, &len, NULL));
  EXPECT_EQ (len, 8U * 11);

  for (i = 0; i < 10; i++) {
    EXPECT_EQ (content[i * 8 + 0 + 8], i + 4);
    EXPECT_EQ (content[i * 8 + 1 + 8], i + 1);
    EXPECT_EQ (content[i * 8 + 2 + 8], i + 3);
    EXPECT_EQ (content[i * 8 + 3 + 8], i + 2);
    EXPECT_EQ (content[i * 8 + 4 + 8], i + 3);
    EXPECT_EQ (content[i * 8 + 5 + 8], i + 2);
    EXPECT_EQ (content[i * 8 + 6 + 8], i + 1);
    EXPECT_EQ (content[i * 8 + 7 + 8], i + 4);
  }

  g_free (content);
  ml_tensors_info_destroy (info);
  ml_tensors_data_destroy (data1);
}

/**
 * @brief Test NNStreamer pipeline src
 * @detail Failure case when pipeline is NULL.
 */
TEST (nnstreamer_capi_src, failure_01_n)
{
  int status;
  ml_pipeline_src_h srchandle;

  status = ml_pipeline_src_get_handle (NULL, "dummy", &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test NNStreamer pipeline src
 * @detail Failure case when the name of source node is wrong.
 */
TEST (nnstreamer_capi_src, failure_02_n)
{
  const char *pipeline = "appsrc name=mysource ! other/tensor,dimension=(string)4:1:1:1,type=(string)uint8,framerate=(fraction)0/1 ! valve name=valvex ! tensor_sink";
  ml_pipeline_h handle;
  ml_pipeline_src_h srchandle;

  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : pipe */
  status = ml_pipeline_src_get_handle (NULL, "mysource", &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : name */
  status = ml_pipeline_src_get_handle (handle, NULL, &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : wrong name */
  status = ml_pipeline_src_get_handle (handle, "wrongname", &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : invalid type */
  status = ml_pipeline_src_get_handle (handle, "valvex", &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : handle */
  status = ml_pipeline_src_get_handle (handle, "mysource", NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline src
 * @detail Failure case when the number of tensors is 0 or bigger than ML_TENSOR_SIZE_LIMIT;
 */
TEST (nnstreamer_capi_src, failure_03_n)
{
  const char *pipeline = "appsrc name=srcx ! other/tensor,dimension=(string)4:1:1:1,type=(string)uint8,framerate=(fraction)0/1 ! tensor_sink";
  ml_pipeline_h handle;
  ml_pipeline_src_h srchandle;
  ml_tensors_data_h data;
  ml_tensors_info_h info;

  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_get_handle (handle, "srcx", &srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_get_tensors_info (srchandle, &info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_create (info, &data);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* null data */
  status = ml_pipeline_src_input_data (srchandle, NULL, ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_src_release_handle (srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_destroy (data);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline switch
 */
TEST (nnstreamer_capi_switch, dummy_01)
{
  ml_pipeline_h handle;
  ml_pipeline_switch_h switchhandle;
  ml_pipeline_sink_h sinkhandle;
  ml_pipeline_switch_e type;
  ml_pipeline_state_e state;
  gchar *pipeline;
  int status;
  guint *count_sink;
  TestPipeState *pipe_state;
  gchar **node_list = NULL;

  pipeline = g_strdup ("input-selector name=ins ! tensor_converter ! tensor_sink name=sinkx "
      "videotestsrc is-live=true ! videoconvert ! ins.sink_0 "
      "videotestsrc num-buffers=3 is-live=true ! videoconvert ! ins.sink_1");

  count_sink = (guint *) g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink != NULL);
  *count_sink = 0;

  pipe_state = (TestPipeState *) g_new0 (TestPipeState, 1);
  ASSERT_TRUE (pipe_state != NULL);

  status = ml_pipeline_construct (pipeline, test_pipe_state_callback, pipe_state, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_switch_get_handle (handle, "ins", &type, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_PIPELINE_SWITCH_INPUT_SELECTOR);

  status = ml_pipeline_switch_get_pad_list (switchhandle, &node_list);
  EXPECT_EQ (status, ML_ERROR_NONE);

  if (node_list) {
    gchar *name = NULL;
    guint idx = 0;

    while ((name = node_list[idx]) != NULL) {
      EXPECT_TRUE (g_str_equal (name, "sink_0") || g_str_equal (name, "sink_1"));
      idx++;
      g_free (name);
    }

    EXPECT_EQ (idx, 2U);
    g_free (node_list);
  }

  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (sinkhandle != NULL);

  status = ml_pipeline_switch_select (switchhandle, "sink_1");
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (50000);
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  wait_for_start (handle, state, status);

  g_usleep (300000); /* 300ms. Let a few frames flow. */

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_unregister (sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_switch_release_handle (switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_EQ (*count_sink, 3U);

  EXPECT_TRUE (pipe_state->paused);
  EXPECT_TRUE (pipe_state->playing);

  g_free (pipeline);
  g_free (count_sink);
  g_free (pipe_state);
}

/**
 * @brief Test NNStreamer pipeline switch
 */
TEST (nnstreamer_capi_switch, dummy_02)
{
  ml_pipeline_h handle;
  ml_pipeline_switch_h switchhandle;
  ml_pipeline_sink_h sinkhandle0, sinkhandle1;
  ml_pipeline_switch_e type;
  gchar *pipeline;
  int status;
  guint *count_sink0, *count_sink1;
  gchar **node_list = NULL;

  /**
   * Prerolling problem
   * For running the test, set async=false in the sink element when using an output selector.
   * The pipeline state can be changed to paused after all sink element receive buffer.
   */
  pipeline = g_strdup ("videotestsrc is-live=true ! videoconvert ! tensor_converter ! output-selector name=outs "
      "outs.src_0 ! tensor_sink name=sink0 async=false "
      "outs.src_1 ! tensor_sink name=sink1 async=false");

  count_sink0 = (guint *) g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink0 != NULL);
  *count_sink0 = 0;

  count_sink1 = (guint *) g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink1 != NULL);
  *count_sink1 = 0;

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_switch_get_handle (handle, "outs", &type, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_PIPELINE_SWITCH_OUTPUT_SELECTOR);

  status = ml_pipeline_switch_get_pad_list (switchhandle, &node_list);
  EXPECT_EQ (status, ML_ERROR_NONE);

  if (node_list) {
    gchar *name = NULL;
    guint idx = 0;

    while ((name = node_list[idx]) != NULL) {
      EXPECT_TRUE (g_str_equal (name, "src_0") || g_str_equal (name, "src_1"));
      idx++;
      g_free (name);
    }

    EXPECT_EQ (idx, 2U);
    g_free (node_list);
  }

  status = ml_pipeline_sink_register (handle, "sink0", test_sink_callback_count, count_sink0, &sinkhandle0);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (sinkhandle0 != NULL);

  status = ml_pipeline_sink_register (handle, "sink1", test_sink_callback_count, count_sink1, &sinkhandle1);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (sinkhandle1 != NULL);

  status = ml_pipeline_switch_select (switchhandle, "src_1");
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (200000); /* 200ms. Let a few frames flow. */

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_unregister (sinkhandle0);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_unregister (sinkhandle1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_switch_release_handle (switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_EQ (*count_sink0, 0U);
  EXPECT_TRUE (*count_sink1 > 0U);

  g_free (pipeline);
  g_free (count_sink0);
  g_free (count_sink1);
}

/**
 * @brief Test NNStreamer pipeline switch
 * @detail Failure case to handle input-selector element with invalid param.
 */
TEST (nnstreamer_capi_switch, failure_01_n)
{
  ml_pipeline_h handle;
  ml_pipeline_switch_h switchhandle;
  ml_pipeline_switch_e type;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("input-selector name=ins ! tensor_converter ! tensor_sink name=sinkx "
      "videotestsrc is-live=true ! videoconvert ! ins.sink_0 "
      "videotestsrc num-buffers=3 ! videoconvert ! ins.sink_1");

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : pipe */
  status = ml_pipeline_switch_get_handle (NULL, "ins", &type, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : name */
  status = ml_pipeline_switch_get_handle (handle, NULL, &type, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : wrong name */
  status = ml_pipeline_switch_get_handle (handle, "wrongname", &type, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : invalid type */
  status = ml_pipeline_switch_get_handle (handle, "sinkx", &type, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : handle */
  status = ml_pipeline_switch_get_handle (handle, "ins", &type, NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* succesfully get switch handle if the param type is null */
  status = ml_pipeline_switch_get_handle (handle, "ins", NULL, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : handle */
  status = ml_pipeline_switch_select (NULL, "invalidpadname");
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : pad name */
  status = ml_pipeline_switch_select (switchhandle, NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : wrong pad name */
  status = ml_pipeline_switch_select (switchhandle, "wrongpadname");
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_switch_release_handle (switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer Utility for checking availability of NNFW
 * @todo After adding sub-plugin for NNFW, this testcase should be fixed.
 */
TEST (nnstreamer_capi_util, availability_00)
{
  bool result;
  int status = ml_check_nnfw_availability (ML_NNFW_TYPE_NNFW, ML_NNFW_HW_ANY, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_NNFW, ML_NNFW_HW_AUTO, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_NNFW, ML_NNFW_HW_NPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);
}

/**
 * @brief Test NNStreamer Utility for checking tensors info handle
 */
TEST (nnstreamer_capi_util, tensors_info)
{
  ml_tensors_info_h info;
  ml_tensor_dimension in_dim, out_dim;
  ml_tensor_type_e out_type;
  gchar *out_name;
  size_t data_size;
  int status;

  status = ml_tensors_info_create (&info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  in_dim[0] = 3;
  in_dim[1] = 300;
  in_dim[2] = 300;
  in_dim[3] = 1;

  /* add tensor info */
  status = ml_tensors_info_set_count (info, 2);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_UINT8);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_dimension (info, 0, in_dim);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_set_tensor_type (info, 1, ML_TENSOR_TYPE_FLOAT64);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_dimension (info, 1, in_dim);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_name (info, 1, "tensor-name-test");
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_set_tensor_type (info, 2, ML_TENSOR_TYPE_UINT64);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
  status = ml_tensors_info_set_tensor_dimension (info, 2, in_dim);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* get tensor info */
  status = ml_tensors_info_get_tensor_type (info, 0, &out_type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (out_type, ML_TENSOR_TYPE_UINT8);

  status = ml_tensors_info_get_tensor_dimension (info, 0, out_dim);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (out_dim[0], 3U);
  EXPECT_EQ (out_dim[1], 300U);
  EXPECT_EQ (out_dim[2], 300U);
  EXPECT_EQ (out_dim[3], 1U);

  status = ml_tensors_info_get_tensor_name (info, 0, &out_name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (out_name == NULL);

  status = ml_tensors_info_get_tensor_type (info, 1, &out_type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (out_type, ML_TENSOR_TYPE_FLOAT64);

  status = ml_tensors_info_get_tensor_dimension (info, 1, out_dim);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (out_dim[0], 3U);
  EXPECT_EQ (out_dim[1], 300U);
  EXPECT_EQ (out_dim[2], 300U);
  EXPECT_EQ (out_dim[3], 1U);

  status = ml_tensors_info_get_tensor_name (info, 1, &out_name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (out_name && g_str_equal (out_name, "tensor-name-test"));

  status = ml_tensors_info_get_tensor_type (info, 2, &out_type);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_get_tensor_dimension (info, 2, out_dim);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_get_tensor_name (info, 2, &out_name);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* get tensor size */
  status = ml_tensors_info_get_tensor_size (info, 0, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (data_size == (3 * 300 * 300));

  status = ml_tensors_info_get_tensor_size (info, 1, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (data_size == (3 * 300 * 300 * 8));

  status = ml_tensors_info_get_tensor_size (info, -1, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (data_size == ((3 * 300 * 300) + (3 * 300 * 300 * 8)));

  status = ml_tensors_info_get_tensor_size (info, 2, &data_size);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

#ifdef ENABLE_TENSORFLOW_LITE
/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 */
TEST (nnstreamer_capi_singleshot, invoke_01)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_info_h in_res, out_res;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim, res_dim;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;
  unsigned int count = 0;
  char *name = NULL;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);
  ml_tensors_info_create (&in_res);
  ml_tensors_info_create (&out_res);

  in_dim[0] = 3;
  in_dim[1] = 224;
  in_dim[2] = 224;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = 1001;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* input tensor in filter */
  status = ml_single_get_input_info (single, &in_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (in_res, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_name (in_res, 0, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (name == NULL);

  status = ml_tensors_info_get_tensor_type (in_res, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_UINT8);

  ml_tensors_info_get_tensor_dimension (in_res, 0, res_dim);
  EXPECT_TRUE (in_dim[0] == res_dim[0]);
  EXPECT_TRUE (in_dim[1] == res_dim[1]);
  EXPECT_TRUE (in_dim[2] == res_dim[2]);
  EXPECT_TRUE (in_dim[3] == res_dim[3]);

  /* output tensor in filter */
  status = ml_single_get_output_info (single, &out_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (out_res, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_name (out_res, 0, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (name == NULL);

  status = ml_tensors_info_get_tensor_type (out_res, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_UINT8);

  ml_tensors_info_get_tensor_dimension (out_res, 0, res_dim);
  EXPECT_TRUE (out_dim[0] == res_dim[0]);
  EXPECT_TRUE (out_dim[1] == res_dim[1]);
  EXPECT_TRUE (out_dim[2] == res_dim[2]);
  EXPECT_TRUE (out_dim[3] == res_dim[3]);

  input = output = NULL;

  /* generate dummy data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_single_set_timeout (single, SINGLE_DEF_TIMEOUT_MSEC);
  EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED || status == ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
  ml_tensors_info_destroy (in_res);
  ml_tensors_info_destroy (out_res);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Start pipeline without tensor info
 */
TEST (nnstreamer_capi_singleshot, invoke_02)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  in_dim[0] = 3;
  in_dim[1] = 224;
  in_dim[2] = 224;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = 1001;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NONE);

  input = output = NULL;

  /* generate dummy data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_single_set_timeout (single, SINGLE_DEF_TIMEOUT_MSEC);
  EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED || status == ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @note Measure the loading time and total time for the run
 */
TEST (nnstreamer_capi_singleshot, benchmark_time)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim;
  int status, count;
  unsigned long open_duration=0, invoke_duration=0, close_duration=0;
  gint64 start, end;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  in_dim[0] = 3;
  in_dim[1] = 224;
  in_dim[2] = 224;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = 1001;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  input = output = NULL;

  /* generate dummy data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  /** Initial run to warm up the cache */
  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  count = 1;
  for (int i=0; i<count; i++) {
    start = g_get_real_time();
    status = ml_single_open (&single, test_model, in_info, out_info,
        ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
    end = g_get_real_time();
    open_duration += end - start;
    EXPECT_EQ (status, ML_ERROR_NONE);

    status = ml_single_set_timeout (single, SINGLE_DEF_TIMEOUT_MSEC);
    EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED || status == ML_ERROR_NONE);

    start = g_get_real_time();
    status = ml_single_invoke (single, input, &output);
    end = g_get_real_time();
    invoke_duration += end - start;
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_TRUE (output != NULL);

    start = g_get_real_time();
    status = ml_single_close (single);
    end = g_get_real_time();
    close_duration = end - start;
    EXPECT_EQ (status, ML_ERROR_NONE);
  }

  g_warning ("Time to open single = %f us", (open_duration * 1.0)/count);
  g_warning ("Time to invoke single = %f us", (invoke_duration * 1.0)/count);
  g_warning ("Time to close single = %f us", (close_duration * 1.0)/count);

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);

  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}
#endif /* ENABLE_TENSORFLOW_LITE */

/**
 * @brief Test NNStreamer single shot (custom filter)
 * @detail Run pipeline with custom filter, handle multi tensors.
 */
TEST (nnstreamer_capi_singleshot, invoke_03)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim;
  int status;
  unsigned int i;
  void *data_ptr;
  size_t data_size;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "build", "nnstreamer_example", "custom_example_passthrough",
      "libnnstreamer_customfilter_passthrough_variable.so", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  ml_tensors_info_set_count (in_info, 2);

  in_dim[0] = 10;
  in_dim[1] = 1;
  in_dim[2] = 1;
  in_dim[3] = 1;

  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT16);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  ml_tensors_info_set_tensor_type (in_info, 1, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 1, in_dim);

  ml_tensors_info_clone (out_info, in_info);

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_CUSTOM_FILTER, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NONE);

  input = output = NULL;

  /* generate input data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  ASSERT_TRUE (input != NULL);

  for (i = 0; i < 10; i++) {
    int16_t i16 = (int16_t) (i + 1);
    float f32 = (float) (i + .1);

    status = ml_tensors_data_get_tensor_data (input, 0, &data_ptr, &data_size);
    EXPECT_EQ (status, ML_ERROR_NONE);
    ((int16_t *) data_ptr)[i] = i16;

    status = ml_tensors_data_get_tensor_data (input, 1, &data_ptr, &data_size);
    EXPECT_EQ (status, ML_ERROR_NONE);
    ((float *) data_ptr)[i] = f32;
  }

  status = ml_single_set_timeout (single, SINGLE_DEF_TIMEOUT_MSEC);
  EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED || status == ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  for (i = 0; i < 10; i++) {
    int16_t i16 = (int16_t) (i + 1);
    float f32 = (float) (i + .1);

    status = ml_tensors_data_get_tensor_data (output, 0, &data_ptr, &data_size);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_EQ (((int16_t *) data_ptr)[i], i16);

    status = ml_tensors_data_get_tensor_data (output, 1, &data_ptr, &data_size);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_FLOAT_EQ (((float *) data_ptr)[i], f32);
  }

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

#ifdef ENABLE_TENSORFLOW
/**
 * @brief Test NNStreamer single shot (tensorflow)
 * @detail Run pipeline with tensorflow speech command model.
 */
TEST (nnstreamer_capi_singleshot, invoke_04)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_info_h in_res, out_res;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim, res_dim;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;
  unsigned int count = 0;
  char *name = NULL;
  int status, max_score_index;
  float score, max_score;
  void *data_ptr;
  size_t data_size;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model, *test_file;
  gchar *contents = NULL;
  gsize len = 0;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "conv_actions_frozen.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  test_file = g_build_filename (root_path, "tests", "test_models", "data",
      "yes.wav", NULL);
  ASSERT_TRUE (g_file_test (test_file, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);
  ml_tensors_info_create (&in_res);
  ml_tensors_info_create (&out_res);

  in_dim[0] = 1;
  in_dim[1] = 16022;
  in_dim[2] = 1;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_name (in_info, 0, "wav_data");
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT16);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = 12;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_name (out_info, 0, "labels_softmax");
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  ASSERT_TRUE (g_file_get_contents (test_file, &contents, &len, NULL));
  status = ml_tensors_info_get_tensor_size (in_info, 0, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  ASSERT_TRUE (len == data_size);

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_TENSORFLOW, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* input tensor in filter */
  status = ml_single_get_input_info (single, &in_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (in_res, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_name (in_res, 0, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (g_str_equal (name, "wav_data"));

  status = ml_tensors_info_get_tensor_type (in_res, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_INT16);

  ml_tensors_info_get_tensor_dimension (in_res, 0, res_dim);
  EXPECT_TRUE (in_dim[0] == res_dim[0]);
  EXPECT_TRUE (in_dim[1] == res_dim[1]);
  EXPECT_TRUE (in_dim[2] == res_dim[2]);
  EXPECT_TRUE (in_dim[3] == res_dim[3]);

  /* output tensor in filter */
  status = ml_single_get_output_info (single, &out_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (out_res, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_name (out_res, 0, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (g_str_equal (name, "labels_softmax"));

  status = ml_tensors_info_get_tensor_type (out_res, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_FLOAT32);

  ml_tensors_info_get_tensor_dimension (out_res, 0, res_dim);
  EXPECT_TRUE (out_dim[0] == res_dim[0]);
  EXPECT_TRUE (out_dim[1] == res_dim[1]);
  EXPECT_TRUE (out_dim[2] == res_dim[2]);
  EXPECT_TRUE (out_dim[3] == res_dim[3]);

  input = output = NULL;

  /* generate input data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_tensors_data_set_tensor_data (input, 0, contents, len);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_set_timeout (single, SINGLE_DEF_TIMEOUT_MSEC);
  EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED || status == ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  /* check result (max score index is 2) */
  status = ml_tensors_data_get_tensor_data (output, 1, &data_ptr, &data_size);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_data_get_tensor_data (output, 0, &data_ptr, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);

  max_score = .0;
  max_score_index = 0;
  for (gint i = 0; i < 12; i++) {
    score = ((float *) data_ptr)[i];
    if (score > max_score) {
      max_score = score;
      max_score_index = i;
    }
  }

  EXPECT_EQ (max_score_index, 2);

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  g_free (test_file);
  g_free (contents);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
  ml_tensors_info_destroy (in_res);
  ml_tensors_info_destroy (out_res);
}
#else
/**
 * @brief Test NNStreamer single shot (tensorflow is not supported)
 */
TEST (nnstreamer_capi_singleshot, unavailable_fw_tf_n)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensor_dimension in_dim, out_dim;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "conv_actions_frozen.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  in_dim[0] = 1;
  in_dim[1] = 16022;
  in_dim[2] = 1;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_name (in_info, 0, "wav_data");
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT16);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = 12;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_name (out_info, 0, "labels_softmax");
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  /* tensorflow is not supported */
  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_TENSORFLOW, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NOT_SUPPORTED);

  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
  g_free (test_model);
}
#endif /* ENABLE_TENSORFLOW */

#ifdef ENABLE_TENSORFLOW_LITE
/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Failure case with invalid param.
 */
TEST (nnstreamer_capi_singleshot, failure_01_n)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensor_dimension in_dim, out_dim;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  /* invalid file path */
  status = ml_single_open (&single, "wrong_file_name", in_info, out_info,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* null file path */
  status = ml_single_open (&single, NULL, in_info, out_info,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid handle */
  status = ml_single_open (NULL, test_model, in_info, out_info,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid input tensor info */
  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  in_dim[0] = 3;
  in_dim[1] = 224;
  in_dim[2] = 224;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  /* invalid output tensor info */
  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  out_dim[0] = 1001;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  /* invalid file extension */
  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_TENSORFLOW, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid handle */
  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* Successfully opened unknown fw type (tf-lite) */
  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_ANY, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Structure containing values to run single shot
 */
typedef struct {
  gchar *test_model;
  guint num_runs;
  guint timeout;
  guint min_time_to_run;
  gboolean expect;
  ml_single_h *single;
} single_shot_thread_data;

/**
 * @brief Open and run on single shot API with provided data
 */
static void *
single_shot_loop_test (void *arg)
{
  guint i;
  int status = ML_ERROR_NONE;
  ml_single_h single;
  single_shot_thread_data *ss_data = (single_shot_thread_data *) arg;
  int timeout_cond, no_error_cond;

  status = ml_single_open (&single, ss_data->test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (ss_data->expect)
    EXPECT_EQ (status, ML_ERROR_NONE);
  ss_data->single = &single;

  /* set timeout */
  if (ss_data->timeout != 0) {
    status = ml_single_set_timeout (single, ss_data->timeout);
    if (ss_data->expect)
      EXPECT_NE (status, ML_ERROR_INVALID_PARAMETER);
    if (status == ML_ERROR_NOT_SUPPORTED)
      ss_data->timeout = 0;
  }

  ml_tensors_info_h in_info;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim;

  ml_tensors_info_create (&in_info);

  in_dim[0] = 3;
  in_dim[1] = 224;
  in_dim[2] = 224;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  input = output = NULL;

  /* generate dummy data */
  status = ml_tensors_data_create (in_info, &input);
  if (ss_data->expect) {
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_TRUE (input != NULL);
  }

  for (i=0; i<ss_data->num_runs; i++) {
    status = ml_single_invoke (single, input, &output);
    if (ss_data->expect) {
      no_error_cond = status == ML_ERROR_NONE && output != NULL;
      if (ss_data->timeout < ss_data->min_time_to_run) {
        /** Default timeout can return timed out with many parallel runs */
        timeout_cond = output == NULL &&
          (status == ML_ERROR_TIMED_OUT || status == ML_ERROR_TRY_AGAIN);
        EXPECT_TRUE (timeout_cond || no_error_cond);
      } else {
        EXPECT_TRUE (no_error_cond);
      }
    }
    output = NULL;
  }

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);
  ml_tensors_info_destroy (in_info);

  status = ml_single_close (single);
  if (ss_data->expect)
    EXPECT_EQ (status, ML_ERROR_NONE);

  return NULL;
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Testcase with timeout.
 */
TEST (nnstreamer_capi_singleshot, invoke_timeout)
{
  ml_single_h single;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* set timeout 5 ms */
  status = ml_single_set_timeout (single, 5);
  /* test timeout if supported (gstreamer ver >= 1.10) */
  if (status == ML_ERROR_NONE) {
    ml_tensors_info_h in_info;
    ml_tensors_data_h input, output;
    ml_tensor_dimension in_dim;

    ml_tensors_info_create (&in_info);

    in_dim[0] = 3;
    in_dim[1] = 224;
    in_dim[2] = 224;
    in_dim[3] = 1;
    ml_tensors_info_set_count (in_info, 1);
    ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
    ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

    input = output = NULL;

    /* generate dummy data */
    status = ml_tensors_data_create (in_info, &input);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_TRUE (input != NULL);

    status = ml_single_invoke (single, input, &output);
    EXPECT_EQ (status, ML_ERROR_TIMED_OUT);
    EXPECT_TRUE (output == NULL);

    /* check the old buffer is dropped */
    status = ml_single_invoke (single, input, &output);
    /* try_again implies that previous invoke hasn't finished yet */
    EXPECT_TRUE (status == ML_ERROR_TIMED_OUT || status == ML_ERROR_TRY_AGAIN);
    EXPECT_TRUE (output == NULL);

    /* set timeout 10 s */
    status = ml_single_set_timeout (single, SINGLE_DEF_TIMEOUT_MSEC);
    /* clear out previous buffers */
    g_usleep (SINGLE_DEF_TIMEOUT_MSEC * 1000);    /** 10 sec */

    status = ml_single_invoke (single, input, &output);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_TRUE (output != NULL);

    ml_tensors_data_destroy (output);
    ml_tensors_data_destroy (input);
    ml_tensors_info_destroy (in_info);
  }

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Testcase with multiple runs in parallel. Some of the
 *         running instances will timeout, however others will not.
 */
TEST (nnstreamer_capi_singleshot, parallel_runs)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;
  const gint num_threads = 3;
  const gint num_cases = 3;
  pthread_t thread[num_threads * num_cases];
  single_shot_thread_data ss_data[num_cases];
  guint i, j;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  for (i=0; i<num_cases; i++) {
    ss_data[i].test_model = test_model;
    ss_data[i].num_runs = 3;
    ss_data[i].min_time_to_run = 10;    /** 10 msec */
    ss_data[i].expect = TRUE;
  }

  /** Default timeout runs */
  ss_data[0].timeout = 0;
  /** small timeout runs */
  ss_data[1].timeout = 5;
  /** large timeout runs - increases with each run as tests run in parallel */
  ss_data[2].timeout = SINGLE_DEF_TIMEOUT_MSEC * num_cases * num_threads;

  /**
   * make thread running things in background, each with different timeout,
   * some fails, some runs, all opens pipelines by themselves in parallel
   */
  for (j=0; j<num_cases; j++) {
    for (i=0; i<num_threads; i++) {
      pthread_create (&thread[i+j * num_threads], NULL, single_shot_loop_test,
          (void *) &ss_data[j]);
    }
  }

  for (i=0; i<num_threads * num_cases; i++) {
    pthread_join(thread[i], NULL);
  }

  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Close the single handle while running. This test shuuld not crash.
 *         This closes the single handle twice, while opens it once
 */
TEST (nnstreamer_capi_singleshot, close_while_running)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;
  pthread_t thread;
  single_shot_thread_data ss_data;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ss_data.test_model = test_model;
  ss_data.num_runs = 10;
  ss_data.min_time_to_run = 10;    /** 10 msec */
  ss_data.expect = FALSE;
  ss_data.timeout = SINGLE_DEF_TIMEOUT_MSEC;
  ss_data.single = NULL;

  pthread_create (&thread, NULL, single_shot_loop_test, (void *) &ss_data);

  /** Start the thread and let the code start */
  g_usleep (100000);    /** 100 msec */

  /**
   * Call single API functions while its running. One run takes 100ms on average.
   * So, these calls would in the middle of running and should not crash
   * However, their status can be of failure, if the thread is closed earlier
   */
  if (ss_data.single) {
    ml_single_set_timeout (*ss_data.single, ss_data.timeout);
    ml_single_close (*ss_data.single);
  }

  pthread_join(thread, NULL);

  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Try setting dimensions for input tensor.
 */
TEST (nnstreamer_capi_singleshot, set_input_info_fail)
{
  int status;
  ml_single_h single;
  ml_tensors_info_h in_info;
  ml_tensor_dimension in_dim;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_set_input_info (single, NULL);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_info_create (&in_info);
  in_dim[0] = 3;
  in_dim[1] = 4;
  in_dim[2] = 4;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  /** mobilenet model does not support setting different input dimension */
  status = ml_single_set_input_info (single, in_info);
  EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED ||
      status == ML_ERROR_INVALID_PARAMETER);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Try setting number of input tensors and its type
 */
TEST (nnstreamer_capi_singleshot, set_input_info_fail_01)
{
  ml_single_h single;
  ml_tensors_info_h in_info;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;
  unsigned int count = 0;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  /** add.tflite adds value 2 to all the values in the input */
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_get_input_info (single, &in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (in_info, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /** changing the count of number of tensors is not allowed */
  ml_tensors_info_set_count (in_info, count + 1);
  status = ml_single_set_input_info (single, in_info);
  EXPECT_NE (status, ML_ERROR_NONE);
  ml_tensors_info_set_count (in_info, count);

  status = ml_tensors_info_get_tensor_type (in_info, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_FLOAT32);

  /** changing the type of input tensors is not allowed */
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT32);
  status = ml_single_set_input_info (single, in_info);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Try setting dimension to the same value. This model does not allow
 *         changing the dimension to a different. However, setting the same
 *         value for dimension should be successful.
 */
TEST (nnstreamer_capi_singleshot, set_input_info_success)
{
  int status;
  ml_single_h single;
  ml_tensors_info_h in_info;
  ml_tensor_dimension in_dim;
  ml_tensors_data_h input, output;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_set_input_info (single, NULL);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_info_create (&in_info);
  in_dim[0] = 3;
  in_dim[1] = 224;
  in_dim[2] = 224;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  /** set the same original input dimension */
  status = ml_single_set_input_info (single, in_info);
  EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED ||
      status == ML_ERROR_NONE);

  /* generate dummy data */
  input = output = NULL;
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Change the number of input tensors, run the model and verify output
 */
TEST (nnstreamer_capi_singleshot, set_input_info_success_01)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_info_h in_res, out_res;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim, res_dim;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;
  unsigned int count = 0;
  int status, tensor_size;
  size_t data_size;
  float *data;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  /** add.tflite adds value 2 to all the values in the input */
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);
  ml_tensors_info_create (&in_res);
  ml_tensors_info_create (&out_res);

  tensor_size = 5;
  in_dim[0] = tensor_size;
  in_dim[1] = 1;
  in_dim[2] = 1;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = tensor_size;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_get_input_info (single, &in_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /**
   * 1. start with a model file with different input dimensions
   * 2. change the input for the model file
   * 3. run the model file with the updated input dimensions
   * 4. verify the output
   */

  ml_tensors_info_get_tensor_dimension (in_res, 0, res_dim);
  EXPECT_FALSE (in_dim[0] == res_dim[0]);
  EXPECT_TRUE (in_dim[1] == res_dim[1]);
  EXPECT_TRUE (in_dim[2] == res_dim[2]);
  EXPECT_TRUE (in_dim[3] == res_dim[3]);

  /** set the same original input dimension */
  status = ml_single_set_input_info (single, in_info);
  EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED ||
      status == ML_ERROR_NONE);
  if (status == ML_ERROR_NONE) {
    /* input tensor in filter */
    status = ml_single_get_input_info (single, &in_res);
    EXPECT_EQ (status, ML_ERROR_NONE);

    status = ml_tensors_info_get_count (in_res, &count);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_EQ (count, 1U);

    status = ml_tensors_info_get_tensor_type (in_res, 0, &type);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_EQ (type, ML_TENSOR_TYPE_FLOAT32);

    ml_tensors_info_get_tensor_dimension (in_res, 0, res_dim);
    EXPECT_TRUE (in_dim[0] == res_dim[0]);
    EXPECT_TRUE (in_dim[1] == res_dim[1]);
    EXPECT_TRUE (in_dim[2] == res_dim[2]);
    EXPECT_TRUE (in_dim[3] == res_dim[3]);

    /* output tensor in filter */
    status = ml_single_get_output_info (single, &out_res);
    EXPECT_EQ (status, ML_ERROR_NONE);

    status = ml_tensors_info_get_count (out_res, &count);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_EQ (count, 1U);

    status = ml_tensors_info_get_tensor_type (out_res, 0, &type);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_EQ (type, ML_TENSOR_TYPE_FLOAT32);

    ml_tensors_info_get_tensor_dimension (out_res, 0, res_dim);
    EXPECT_TRUE (out_dim[0] == res_dim[0]);
    EXPECT_TRUE (out_dim[1] == res_dim[1]);
    EXPECT_TRUE (out_dim[2] == res_dim[2]);
    EXPECT_TRUE (out_dim[3] == res_dim[3]);

    input = output = NULL;

    /* generate dummy data */
    status = ml_tensors_data_create (in_info, &input);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_TRUE (input != NULL);

    status = ml_tensors_data_get_tensor_data (input, 0, (void **) &data, &data_size);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_EQ (data_size, tensor_size * sizeof (int));
    for (int idx = 0; idx < tensor_size; idx++)
      data[idx] = idx;

    status = ml_single_invoke (single, input, &output);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_TRUE (output != NULL);

    status = ml_tensors_data_get_tensor_data (input, 0, (void **) &data, &data_size);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_EQ (data_size, tensor_size * sizeof (int));
    for (int idx = 0; idx < tensor_size; idx++)
      EXPECT_EQ (data[idx], idx);

    status = ml_tensors_data_get_tensor_data (output, 0, (void **) &data, &data_size);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_EQ (data_size, tensor_size * sizeof (int));
    for (int idx = 0; idx < tensor_size; idx++)
      EXPECT_EQ (data[idx], idx+2);

    ml_tensors_data_destroy (output);
    ml_tensors_data_destroy (input);
  }

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
  ml_tensors_info_destroy (in_res);
  ml_tensors_info_destroy (out_res);
}

#endif /* ENABLE_TENSORFLOW_LITE */

/**
 * @brief Main gtest
 */
int
main (int argc, char **argv)
{
  int result;

  testing::InitGoogleTest (&argc, argv);

  /* ignore tizen feature status while running the testcases */
  set_feature_state (1);

  result = RUN_ALL_TESTS ();

  set_feature_state (-1);

  return result;
}
