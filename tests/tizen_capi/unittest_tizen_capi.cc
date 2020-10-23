/**
 * @file        unittest_tizen_capi.cc
 * @date        13 Mar 2019
 * @brief       Unit test for Tizen CAPI of NNStreamer. Basis of TCT in the future.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <nnstreamer.h>
#include <nnstreamer-single.h>
#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h> /* GStatBuf */
#include <nnstreamer-capi-private.h>
#include <nnstreamer_conf.h> /* NNSTREAMER_SO_FILE_EXTENSION */
#include <nnstreamer_plugin_api.h>
#include <unittest_util.h>

static const unsigned int SINGLE_DEF_TIMEOUT_MSEC = 10000U;

#if defined (ENABLE_TENSORFLOW_LITE)
constexpr bool is_enabled_tensorflow_lite = true;
#else
constexpr bool is_enabled_tensorflow_lite = false;
#endif /* defined (ENABLE_TENSORFLOW_LITE) */

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
  ml_pipeline_valve_h valve_h;
  int status;

  /* invalid param : pipe */
  status = ml_pipeline_valve_get_handle (NULL, "valvex", &valve_h);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test NNStreamer pipeline valve
 * @detail Failure case to handle valve element with invalid param.
 */
TEST (nnstreamer_capi_valve, failure_02_n)
{
  ml_pipeline_h handle;
  ml_pipeline_valve_h valve_h;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : name */
  status = ml_pipeline_valve_get_handle (handle, NULL, &valve_h);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}
/**
 * @brief Test NNStreamer pipeline valve
 * @detail Failure case to handle valve element with invalid param.
 */
TEST (nnstreamer_capi_valve, failure_03_n)
{
  ml_pipeline_h handle;
  ml_pipeline_valve_h valve_h;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : wrong name */
  status = ml_pipeline_valve_get_handle (handle, "wrongname", &valve_h);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline valve
 * @detail Failure case to handle valve element with invalid param.
 */
TEST (nnstreamer_capi_valve, failure_04_n)
{
  ml_pipeline_h handle;
  ml_pipeline_valve_h valve_h;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : invalid type */
  status = ml_pipeline_valve_get_handle (handle, "sinkx", &valve_h);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline valve
 * @detail Failure case to handle valve element with invalid param.
 */
TEST (nnstreamer_capi_valve, failure_05_n)
{
  ml_pipeline_h handle;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

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
  gchar *content1 = NULL;
  gchar *content2 = NULL;
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
 * @brief Wait until the change in pipeline status is done
 * @return ML_ERROR_NONE success, ML_ERROR_UNKNOWN if failed, ML_ERROR_TIMED_OUT if timeout happens.
 */
static int
waitPipelineStateChange (ml_pipeline_h handle, ml_pipeline_state_e state,
    guint timeout_ms)
{
  int status = ML_ERROR_UNKNOWN;
  guint counter = 0;
  ml_pipeline_state_e cur_state = ML_PIPELINE_STATE_NULL;

  do {
    status = ml_pipeline_get_state (handle, &cur_state);
    EXPECT_EQ (status, ML_ERROR_NONE);
    if (cur_state == ML_PIPELINE_STATE_UNKNOWN)
      return ML_ERROR_UNKNOWN;
    if (cur_state == state)
      return ML_ERROR_NONE;
    g_usleep (10000);
  } while ((timeout_ms / 10) >= counter++);

  return ML_ERROR_TIMED_OUT;
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
      ("videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,format=BGRx,width=64,height=48,famerate=30/1 ! tee name=t t. ! queue ! filesink location=\"%s\" buffer-mode=unbuffered t. ! queue ! tensor_converter ! tensor_sink name=sinkx",
      file1);
  ml_pipeline_h handle;
  ml_pipeline_sink_h sinkhandle;
  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_dm01, file2, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (sinkhandle != NULL);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = waitPipelineStateChange (handle, ML_PIPELINE_STATE_PLAYING, 200);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* 200ms. Give enough time for three frames to flow. */
  g_usleep (200000);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */

  status = ml_pipeline_sink_unregister (sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);

  /* File Comparison to check the integrity */
  EXPECT_EQ (file_cmp (file1, file2), 0);

  g_free (fullpath);
  g_free (file1);
  g_free (file2);
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
  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! tensor_converter ! appsink name=sinkx sync=false");

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
  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! tensor_converter ! appsink name=sinkx sync=false");
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
  ml_pipeline_sink_h sinkhandle;
  int status;
  guint *count_sink;

  count_sink = (guint *) g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink != NULL);
  *count_sink = 0;

  /* invalid param : pipe */
  status = ml_pipeline_sink_register (NULL, "sinkx", test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (count_sink);
}

/**
 * @brief Test NNStreamer pipeline sink
 * @detail Failure case to register callback with invalid param.
 */
TEST (nnstreamer_capi_sink, failure_02_n)
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

  /* invalid param : name */
  status = ml_pipeline_sink_register (handle, NULL, test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (pipeline);
  g_free (count_sink);
}

/**
 * @brief Test NNStreamer pipeline sink
 * @detail Failure case to register callback with invalid param.
 */
TEST (nnstreamer_capi_sink, failure_03_n)
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

  /* invalid param : wrong name */
  status = ml_pipeline_sink_register (handle, "wrongname", test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (pipeline);
  g_free (count_sink);
}

/**
 * @brief Test NNStreamer pipeline sink
 * @detail Failure case to register callback with invalid param.
 */
TEST (nnstreamer_capi_sink, failure_04_n)
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

  /* invalid param : invalid type */
  status = ml_pipeline_sink_register (handle, "valvex", test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (pipeline);
  g_free (count_sink);
}

/**
 * @brief Test NNStreamer pipeline sink
 * @detail Failure case to register callback with invalid param.
 */
TEST (nnstreamer_capi_sink, failure_05_n)
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

  /* invalid param : callback */
  status = ml_pipeline_sink_register (handle, "sinkx", NULL, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (pipeline);
  g_free (count_sink);
}

/**
 * @brief Test NNStreamer pipeline sink
 * @detail Failure case to register callback with invalid param.
 */
TEST (nnstreamer_capi_sink, failure_06_n)
{
  ml_pipeline_h handle;
  gchar *pipeline;
  int status;
  guint *count_sink;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  count_sink = (guint *) g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink != NULL);
  *count_sink = 0;

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : handle */
  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_count, count_sink, NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

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
      ("appsrc name=srcx ! other/tensor,dimension=(string)4:1:1:1,type=(string)uint8,framerate=(fraction)0/1 ! filesink location=\"%s\" buffer-mode=unbuffered",
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
  uint8_t *content = NULL;
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
  EXPECT_TRUE (content != nullptr);

  if (content && len == 88U) {
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
  }

  g_free (content);
  ml_tensors_info_destroy (info);
  ml_tensors_data_destroy (data1);

  for (i = 0; i < 10; i++) {
    g_free (uintarray1[i]);
    g_free (uintarray2[i]);
  }

  g_free (fullpath);
  g_free (file1);
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

  /* invalid param : name */
  status = ml_pipeline_src_get_handle (handle, NULL, &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline src
 * @detail Failure case when the name of source node is wrong.
 */
TEST (nnstreamer_capi_src, failure_03_n)
{
  const char *pipeline = "appsrc name=mysource ! other/tensor,dimension=(string)4:1:1:1,type=(string)uint8,framerate=(fraction)0/1 ! valve name=valvex ! tensor_sink";
  ml_pipeline_h handle;
  ml_pipeline_src_h srchandle;

  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : wrong name */
  status = ml_pipeline_src_get_handle (handle, "wrongname", &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline src
 * @detail Failure case when the name of source node is wrong.
 */
TEST (nnstreamer_capi_src, failure_04_n)
{
  const char *pipeline = "appsrc name=mysource ! other/tensor,dimension=(string)4:1:1:1,type=(string)uint8,framerate=(fraction)0/1 ! valve name=valvex ! tensor_sink";
  ml_pipeline_h handle;
  ml_pipeline_src_h srchandle;

  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : invalid type */
  status = ml_pipeline_src_get_handle (handle, "valvex", &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline src
 * @detail Failure case when the name of source node is wrong.
 */
TEST (nnstreamer_capi_src, failure_05_n)
{
  const char *pipeline = "appsrc name=mysource ! other/tensor,dimension=(string)4:1:1:1,type=(string)uint8,framerate=(fraction)0/1 ! valve name=valvex ! tensor_sink";
  ml_pipeline_h handle;

  int status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

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
TEST (nnstreamer_capi_src, failure_06_n)
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

  status = ml_tensors_info_destroy (info);
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
  EXPECT_EQ (state, ML_PIPELINE_STATE_PLAYING);

  EXPECT_TRUE (wait_pipeline_process_buffers (count_sink, 3, SINGLE_DEF_TIMEOUT_MSEC));

  g_usleep (300000); /* To check if more frames are coming in  */

  EXPECT_EQ (*count_sink, 3U);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_unregister (sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_switch_release_handle (switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

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
  ml_pipeline_switch_h switchhandle;
  ml_pipeline_switch_e type;
  int status;

  /* invalid param : pipe */
  status = ml_pipeline_switch_get_handle (NULL, "ins", &type, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test NNStreamer pipeline switch
 * @detail Failure case to handle input-selector element with invalid param.
 */
TEST (nnstreamer_capi_switch, failure_02_n)
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

  /* invalid param : name */
  status = ml_pipeline_switch_get_handle (handle, NULL, &type, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline switch
 * @detail Failure case to handle input-selector element with invalid param.
 */
TEST (nnstreamer_capi_switch, failure_03_n)
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

  /* invalid param : wrong name */
  status = ml_pipeline_switch_get_handle (handle, "wrongname", &type, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline switch
 * @detail Failure case to handle input-selector element with invalid param.
 */
TEST (nnstreamer_capi_switch, failure_04_n)
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

  /* invalid param : invalid type */
  status = ml_pipeline_switch_get_handle (handle, "sinkx", &type, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline switch
 * @detail Failure case to handle input-selector element with invalid param.
 */
TEST (nnstreamer_capi_switch, failure_05_n)
{
  ml_pipeline_h handle;
  ml_pipeline_switch_e type;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("input-selector name=ins ! tensor_converter ! tensor_sink name=sinkx "
      "videotestsrc is-live=true ! videoconvert ! ins.sink_0 "
      "videotestsrc num-buffers=3 ! videoconvert ! ins.sink_1");

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : handle */
  status = ml_pipeline_switch_get_handle (handle, "ins", &type, NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline switch
 * @detail Failure case to handle input-selector element with invalid param.
 */
TEST (nnstreamer_capi_switch, failure_06_n)
{
  ml_pipeline_h handle;
  ml_pipeline_switch_h switchhandle;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("input-selector name=ins ! tensor_converter ! tensor_sink name=sinkx "
      "videotestsrc is-live=true ! videoconvert ! ins.sink_0 "
      "videotestsrc num-buffers=3 ! videoconvert ! ins.sink_1");

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* succesfully get switch handle if the param type is null */
  status = ml_pipeline_switch_get_handle (handle, "ins", NULL, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : handle */
  status = ml_pipeline_switch_select (NULL, "invalidpadname");
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_switch_release_handle (switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline switch
 * @detail Failure case to handle input-selector element with invalid param.
 */
TEST (nnstreamer_capi_switch, failure_07_n)
{
  ml_pipeline_h handle;
  ml_pipeline_switch_h switchhandle;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("input-selector name=ins ! tensor_converter ! tensor_sink name=sinkx "
      "videotestsrc is-live=true ! videoconvert ! ins.sink_0 "
      "videotestsrc num-buffers=3 ! videoconvert ! ins.sink_1");

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* succesfully get switch handle if the param type is null */
  status = ml_pipeline_switch_get_handle (handle, "ins", NULL, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : pad name */
  status = ml_pipeline_switch_select (switchhandle, NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_switch_release_handle (switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline switch
 * @detail Failure case to handle input-selector element with invalid param.
 */
TEST (nnstreamer_capi_switch, failure_08_n)
{
  ml_pipeline_h handle;
  ml_pipeline_switch_h switchhandle;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("input-selector name=ins ! tensor_converter ! tensor_sink name=sinkx "
      "videotestsrc is-live=true ! videoconvert ! ins.sink_0 "
      "videotestsrc num-buffers=3 ! videoconvert ! ins.sink_1");

  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* succesfully get switch handle if the param type is null */
  status = ml_pipeline_switch_get_handle (handle, "ins", NULL, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

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
 * @brief Test NNStreamer Utility for checking plugin availability (invalid param)
 */
TEST (nnstreamer_capi_util, plugin_availability_fail_invalid_01_n)
{
  int status;

  status = ml_check_plugin_availability (NULL, "tensor_filter");
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer Utility for checking plugin availability (invalid param)
 */
TEST (nnstreamer_capi_util, plugin_availability_fail_invalid_02_n)
{
  int status;

  status = ml_check_plugin_availability (NULL, "tensor_filter");
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer Utility for checking nnfw availability (invalid param)
 */
TEST (nnstreamer_capi_util, nnfw_availability_fail_invalid_01_n)
{
  int status;

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY, NULL);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer Utility for checking nnfw availability (invalid param)
 */
TEST (nnstreamer_capi_util, nnfw_availability_fail_invalid_02_n)
{
  bool result;
  int status;

  /* any is unknown nnfw type */
  status = ml_check_nnfw_availability (ML_NNFW_TYPE_ANY, ML_NNFW_HW_ANY, &result);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer Utility for checking availability of NNFW
 */
TEST (nnstreamer_capi_util, availability_00)
{
  bool result;
  int status = ml_check_nnfw_availability (ML_NNFW_TYPE_NNFW, ML_NNFW_HW_ANY, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
#ifdef ENABLE_NNFW_RUNTIME
  EXPECT_EQ (result, true);
#else   /* ENABLE_NNFW_RUNTIME */
  EXPECT_EQ (result, false);
#endif  /* ENABLE_NNFW_RUNTIME */

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_NNFW, ML_NNFW_HW_AUTO, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
#ifdef ENABLE_NNFW_RUNTIME
  EXPECT_EQ (result, true);
#else   /* ENABLE_NNFW_RUNTIME */
  EXPECT_EQ (result, false);
#endif  /* ENABLE_NNFW_RUNTIME */

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_NNFW, ML_NNFW_HW_NPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
#ifdef ENABLE_NNFW_RUNTIME
  EXPECT_EQ (result, true);
#else   /* ENABLE_NNFW_RUNTIME */
  EXPECT_EQ (result, false);
#endif  /* ENABLE_NNFW_RUNTIME */
}

/**
 * @brief Test NNStreamer Utility for checking availability of Tensorflow-lite backend
 */
TEST (nnstreamer_capi_util, availability_01)
{
  bool result;
  int status;

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, is_enabled_tensorflow_lite);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_AUTO, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, is_enabled_tensorflow_lite);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_CPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, is_enabled_tensorflow_lite);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_CPU_NEON, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, is_enabled_tensorflow_lite);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_CPU_SIMD, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, is_enabled_tensorflow_lite);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_GPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, is_enabled_tensorflow_lite);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_NPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, is_enabled_tensorflow_lite);
}

/**
 * @brief Test NNStreamer Utility for checking availability of Tensorflow-lite backend
 */
TEST (nnstreamer_capi_util, availability_fail_01_n)
{
  bool result;
  int status;

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_NPU_MOVIDIUS, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_NPU_EDGE_TPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_NPU_VIVANTE, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_NPU_SR, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);
}

#ifdef ENABLE_TENSORFLOW
/**
 * @brief Test NNStreamer Utility for checking availability of Tensorflow backend
 */
TEST (nnstreamer_capi_util, availability_02)
{
  bool result;
  int status;

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW, ML_NNFW_HW_ANY, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW, ML_NNFW_HW_AUTO, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);
}

/**
 * @brief Test NNStreamer Utility for checking availability of Tensorflow backend
 */
TEST (nnstreamer_capi_util, availability_fail_02_n)
{
  bool result;
  int status;

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW, ML_NNFW_HW_CPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_TENSORFLOW, ML_NNFW_HW_GPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);
}
#endif /** ENABLE_TENSORFLOW */

/**
 * @brief Test NNStreamer Utility for checking availability of custom backend
 */
TEST (nnstreamer_capi_util, availability_03)
{
  bool result;
  int status;

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_CUSTOM_FILTER, ML_NNFW_HW_ANY, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_CUSTOM_FILTER, ML_NNFW_HW_AUTO, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);
}

/**
 * @brief Test NNStreamer Utility for checking availability of custom backend
 */
TEST (nnstreamer_capi_util, availability_fail_03_n)
{
  bool result;
  int status;

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_CUSTOM_FILTER, ML_NNFW_HW_CPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_CUSTOM_FILTER, ML_NNFW_HW_GPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);
}

#ifdef ENABLE_NNFW_RUNTIME
/**
 * @brief Test NNStreamer Utility for checking availability of NNFW
 */
TEST (nnstreamer_capi_util, availability_04)
{
  bool result;
  int status;

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_NNFW, ML_NNFW_HW_ANY, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_NNFW, ML_NNFW_HW_AUTO, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_NNFW, ML_NNFW_HW_CPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_NNFW, ML_NNFW_HW_GPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_NNFW, ML_NNFW_HW_NPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);
}

/**
 * @brief Test NNStreamer Utility for checking availability of NNFW
 */
TEST (nnstreamer_capi_util, availability_fail_04_n)
{
  bool result;
  int status;

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_NNFW, ML_NNFW_HW_NPU_SR, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_NNFW, ML_NNFW_HW_NPU_MOVIDIUS, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);
}
#endif /** ENABLE_NNFW_RUNTIME */

#ifdef ENABLE_MOVIDIUS_NCSDK2
/**
 * @brief Test NNStreamer Utility for checking availability of NCSDK2
 */
TEST (nnstreamer_capi_util, availability_05)
{
  bool result;
  int status;

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_MVNC, ML_NNFW_HW_ANY, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_MVNC, ML_NNFW_HW_AUTO, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_MVNC, ML_NNFW_HW_NPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_MVNC, ML_NNFW_HW_NPU_MOVIDIUS, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);
}

/**
 * @brief Test NNStreamer Utility for checking availability of NCSDK2
 */
TEST (nnstreamer_capi_util, availability_fail_05_n)
{
  bool result;
  int status;

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_MVNC, ML_NNFW_HW_CPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_MVNC, ML_NNFW_HW_GPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);
}
#endif /** ENABLE_MOVIDIUS_NCSDK2 */

#ifdef ENABLE_ARMNN
/**
 * @brief Test NNStreamer Utility for checking availability of ARMNN
 */
TEST (nnstreamer_capi_util, availability_06)
{
  bool result;
  int status;

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_ARMNN, ML_NNFW_HW_ANY, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_ARMNN, ML_NNFW_HW_AUTO, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_ARMNN, ML_NNFW_HW_CPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_ARMNN, ML_NNFW_HW_CPU_NEON, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_ARMNN, ML_NNFW_HW_GPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, true);
}

/**
 * @brief Test NNStreamer Utility for checking availability of ARMNN
 */
TEST (nnstreamer_capi_util, availability_fail_06_n)
{
  bool result;
  int status;

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_ARMNN, ML_NNFW_HW_NPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);

  status = ml_check_nnfw_availability (ML_NNFW_TYPE_ARMNN, ML_NNFW_HW_NPU_EDGE_TPU, &result);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (result, false);
}
#endif /** ENABLE_ARMNN */

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
  g_free (out_name);

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

/**
 * @brief Test utility functions
 */
TEST (nnstreamer_capi_util, compare_info)
{
  ml_tensors_info_h info1, info2;
  ml_tensor_dimension dim;
  int status;

  status = ml_tensors_info_create (&info1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_create (&info2);
  EXPECT_EQ (status, ML_ERROR_NONE);

  dim[0] = 3;
  dim[1] = 4;
  dim[2] = 4;
  dim[3] = 1;

  ml_tensors_info_set_count (info1, 1);
  ml_tensors_info_set_tensor_type (info1, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (info1, 0, dim);

  ml_tensors_info_set_count (info2, 1);
  ml_tensors_info_set_tensor_type (info2, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (info2, 0, dim);

  /* compare info */
  EXPECT_TRUE (ml_tensors_info_is_equal (info1, info2));

  /* change type */
  ml_tensors_info_set_tensor_type (info2, 0, ML_TENSOR_TYPE_UINT16);
  EXPECT_FALSE (ml_tensors_info_is_equal (info1, info2));

  /* validate info */
  EXPECT_TRUE (ml_tensors_info_is_valid (info2));

  /* validate invalid dimension */
  dim[3] = 0;
  ml_tensors_info_set_tensor_dimension (info2, 0, dim);
  EXPECT_FALSE (ml_tensors_info_is_valid (info2));

  status = ml_tensors_info_destroy (info1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_destroy (info2);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_create_1_n)
{
  int status = ml_tensors_info_create (nullptr);
  ASSERT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (internal)
 */
TEST (nnstreamer_capi_util, info_create_2_n)
{
  ml_tensors_info_h i;
  int status = ml_tensors_info_create_from_gst (&i, nullptr);
  ASSERT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (internal)
 */
TEST (nnstreamer_capi_util, info_create_3_n)
{
  GstTensorsInfo gi;
  int status = ml_tensors_info_create_from_gst (nullptr, &gi);
  ASSERT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_destroy_n)
{
  int status = ml_tensors_info_destroy (nullptr);
  ASSERT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (internal)
 */
TEST (nnstreamer_capi_util, info_init_n)
{
  int status = ml_tensors_info_initialize (nullptr);
  ASSERT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_valid_01_n)
{
  bool valid;
  int status = ml_tensors_info_validate (nullptr, &valid);
  ASSERT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_valid_02_n)
{
  ml_tensors_info_h info;
  ml_tensor_dimension dim = { 2, 2, 2, 2 };
  int status;

  status = ml_tensors_info_create (&info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_set_count (info, 1);
  ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (info, 0, dim);

  status = ml_tensors_info_validate (info, nullptr);
  ASSERT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (internal)
 */
TEST (nnstreamer_capi_util, info_comp_01_n)
{
  ml_tensors_info_h info;
  ml_tensor_dimension dim = { 2, 2, 2, 2 };
  bool equal;
  int status;

  status = ml_tensors_info_create (&info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_set_count (info, 1);
  ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (info, 0, dim);

  status = ml_tensors_info_compare (nullptr, info, &equal);
  ASSERT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (internal)
 */
TEST (nnstreamer_capi_util, info_comp_02_n)
{
  ml_tensors_info_h info;
  ml_tensor_dimension dim = { 2, 2, 2, 2 };

  bool equal;
  int status;

  status = ml_tensors_info_create (&info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_set_count (info, 1);
  ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (info, 0, dim);

  status = ml_tensors_info_compare (info, nullptr, &equal);
  ASSERT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (internal)
 */
TEST (nnstreamer_capi_util, info_comp_03_n)
{
  ml_tensors_info_h info1, info2;
  ml_tensor_dimension dim = { 2, 2, 2, 2 };
  int status;

  status = ml_tensors_info_create (&info1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_set_count (info1, 1);
  ml_tensors_info_set_tensor_type (info1, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (info1, 0, dim);

  status = ml_tensors_info_create (&info2);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_set_count (info2, 1);
  ml_tensors_info_set_tensor_type (info2, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (info2, 0, dim);

  status = ml_tensors_info_compare (info1, info2, nullptr);
  ASSERT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_destroy (info2);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (internal)
 */
TEST (nnstreamer_capi_util, info_comp_0)
{
  bool equal;
  ml_tensors_info_h info1, info2;
  ml_tensors_info_s *is;
  int status = ml_tensors_info_create (&info1);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_create (&info2);
  ASSERT_EQ (status, ML_ERROR_NONE);

  is = (ml_tensors_info_s *) info1;
  is->num_tensors = 1;
  is = (ml_tensors_info_s *) info2;
  is->num_tensors = 2;

  status = ml_tensors_info_compare (info1, info2, &equal);
  ASSERT_EQ (status, ML_ERROR_NONE);
  ASSERT_FALSE (equal);

  status = ml_tensors_info_destroy (info1);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_destroy (info2);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_set_count_n)
{
  int status = ml_tensors_info_set_count (nullptr, 1);
  ASSERT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_get_count_n)
{
  int status = ml_tensors_info_get_count (nullptr, nullptr);
  ASSERT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_set_tname_0_n)
{
  int status = ml_tensors_info_set_tensor_name (nullptr, 0, "fail");
  ASSERT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_set_tname_1_n)
{
  ml_tensors_info_h info;
  int status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 3);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_set_tensor_name (info, 3, "fail");
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_set_tname_1)
{
  ml_tensors_info_h info;
  int status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_set_tensor_name (info, 0, "first");
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_name (info, 0, "second");
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_get_tname_01_n)
{
  int status;
  ml_tensors_info_h info;
  char *name = NULL;

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_tensor_name (nullptr, 0, &name);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_get_tname_02_n)
{
  int status;
  ml_tensors_info_h info;

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_tensor_name (info, 0, nullptr);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_get_tname_03_n)
{
  int status;
  ml_tensors_info_h info;
  char *name = NULL;

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_tensor_name (info, 2, &name);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_set_ttype_01_n)
{
  int status;

  status = ml_tensors_info_set_tensor_type (nullptr, 0, ML_TENSOR_TYPE_INT16);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_set_ttype_02_n)
{
  int status;
  ml_tensors_info_h info;

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_UNKNOWN);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_set_ttype_03_n)
{
  int status;
  ml_tensors_info_h info;

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_set_tensor_type (info, 2, ML_TENSOR_TYPE_INT16);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_get_ttype_01_n)
{
  int status;
  ml_tensor_type_e type;

  status = ml_tensors_info_get_tensor_type (nullptr, 0, &type);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_get_ttype_02_n)
{
  int status;
  ml_tensors_info_h info;

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_tensor_type (info, 0, nullptr);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_get_ttype_03_n)
{
  int status;
  ml_tensors_info_h info;
  ml_tensor_type_e type;

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_tensor_type (info, 2, &type);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_set_tdimension_01_n)
{
  int status;
  ml_tensor_dimension dim = { 2, 2, 2, 2 };

  status = ml_tensors_info_set_tensor_dimension (nullptr, 0, dim);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_set_tdimension_02_n)
{
  int status;
  ml_tensors_info_h info;
  ml_tensor_dimension dim = { 1, 2, 3, 4 };

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_set_tensor_dimension (info, 2, dim);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_get_tdimension_01_n)
{
  int status;
  ml_tensor_dimension dim;

  status = ml_tensors_info_get_tensor_dimension (nullptr, 0, dim);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_get_tdimension_02_n)
{
  int status;
  ml_tensors_info_h info;
  ml_tensor_dimension dim;

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_tensor_dimension (info, 2, dim);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_get_tsize_01_n)
{
  int status;
  ml_tensors_info_h info;

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_tensor_size (info, 0, nullptr);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_get_tsize_02_n)
{
  int status;
  size_t data_size;

  status = ml_tensors_info_get_tensor_size (nullptr, 0, &data_size);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_get_tsize_03_n)
{
  int status;
  size_t data_size;
  ml_tensors_info_h info;

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_tensor_size (info, 2, &data_size);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
  status = ml_tensors_info_get_tensor_size (info, 0, &data_size);
  EXPECT_TRUE (data_size == 0);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_clone_01_n)
{
  int status;
  ml_tensors_info_h src;

  status = ml_tensors_info_create (&src);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_clone (nullptr, src);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (src);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, info_clone_02_n)
{
  int status;
  ml_tensors_info_h desc;

  status = ml_tensors_info_create (&desc);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_clone (desc, nullptr);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (desc);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, data_create_01_n)
{
  int status;
  ml_tensors_data_h data = nullptr;

  status = ml_tensors_data_create (nullptr, &data);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, data_create_02_n)
{
  int status;
  ml_tensors_info_h info;

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);


  status = ml_tensors_data_create (info, nullptr);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, data_create_03_n)
{
  int status;
  ml_tensors_info_h info;
  ml_tensors_data_h data = nullptr;

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);

  /* invalid info */
  status = ml_tensors_data_create (info, &data);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_destroy (data);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (internal)
 */
TEST (nnstreamer_capi_util, data_create_internal_n)
{
  int status;

  status = ml_tensors_data_create_no_alloc (NULL, NULL);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, data_get_tdata_01_n)
{
  int status;
  size_t data_size;
  void *raw_data;

  status = ml_tensors_data_get_tensor_data (nullptr, 0, &raw_data, &data_size);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, data_get_tdata_02_n)
{
  int status;
  size_t data_size;
  ml_tensors_info_h info;
  ml_tensors_data_h data;
  ml_tensor_dimension dim = { 2, 2, 2, 2 };

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_UINT8);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_dimension (info, 0, dim);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_data_create (info, &data);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_get_tensor_data (data, 0, nullptr, &data_size);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_data_destroy (data);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, data_get_tdata_03_n)
{
  int status;
  void *raw_data;
  ml_tensors_info_h info;
  ml_tensors_data_h data;
  ml_tensor_dimension dim = { 2, 2, 2, 2 };

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_UINT8);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_dimension (info, 0, dim);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_data_create (info, &data);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_get_tensor_data (data, 0, &raw_data, nullptr);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_data_destroy (data);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, data_get_tdata_04_n)
{
  int status;
  size_t data_size;
  void *raw_data;
  ml_tensors_info_h info;
  ml_tensors_data_h data;
  ml_tensor_dimension dim = { 2, 2, 2, 2 };

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_UINT8);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_dimension (info, 0, dim);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_data_create (info, &data);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_get_tensor_data (data, 2, &raw_data, &data_size);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_data_destroy (data);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, data_set_tdata_01_n)
{
  int status;
  void *raw_data;

  raw_data = g_malloc (1024); /* larger than tensor */

  status = ml_tensors_data_set_tensor_data (nullptr, 0, raw_data, 16);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (raw_data);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, data_set_tdata_02_n)
{
  int status;
  ml_tensors_info_h info;
  ml_tensors_data_h data;
  ml_tensor_dimension dim = { 2, 2, 2, 2 };

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_UINT8);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_dimension (info, 0, dim);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_data_create (info, &data);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_set_tensor_data (data, 0, nullptr, 16);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_data_destroy (data);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, data_set_tdata_03_n)
{
  int status;
  void *raw_data;
  ml_tensors_info_h info;
  ml_tensors_data_h data;
  ml_tensor_dimension dim = { 2, 2, 2, 2 };

  raw_data = g_malloc (1024); /* larger than tensor */

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_UINT8);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_dimension (info, 0, dim);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_data_create (info, &data);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_set_tensor_data (data, 2, raw_data, 16);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_data_destroy (data);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  g_free (raw_data);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, data_set_tdata_04_n)
{
  int status;
  void *raw_data;
  ml_tensors_info_h info;
  ml_tensors_data_h data;
  ml_tensor_dimension dim = { 2, 2, 2, 2 };

  raw_data = g_malloc (1024); /* larger than tensor */

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_UINT8);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_dimension (info, 0, dim);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_data_create (info, &data);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_set_tensor_data (data, 0, raw_data, 0);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_data_destroy (data);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  g_free (raw_data);
}

/**
 * @brief Test utility functions (public)
 */
TEST (nnstreamer_capi_util, data_set_tdata_05_n)
{
  int status;
  void *raw_data;
  ml_tensors_info_h info;
  ml_tensors_data_h data;
  ml_tensor_dimension dim = { 2, 2, 2, 2 };

  raw_data = g_malloc (1024); /* larger than tensor */

  status = ml_tensors_info_create (&info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (info, 1);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_type (info, 0, ML_TENSOR_TYPE_UINT8);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_dimension (info, 0, dim);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_data_create (info, &data);
  ASSERT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_set_tensor_data (data, 0, raw_data, 1024);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_data_destroy (data);
  ASSERT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_destroy (info);
  ASSERT_EQ (status, ML_ERROR_NONE);
  g_free (raw_data);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 */
TEST (nnstreamer_capi_singleshot, invoke_invalid_param_01_n)
{
  ml_single_h single;
  int status;
  ml_tensors_info_h in_info;
  ml_tensors_data_h input, output;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  status = ml_single_get_input_info (single, &in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_invoke (NULL, input, &output);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_single_invoke (single, NULL, &output);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_single_invoke (single, input, NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_data_destroy (input);
  ml_tensors_info_destroy (in_info);

skip_test:
  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 */
TEST (nnstreamer_capi_singleshot, invoke_invalid_param_02_n)
{
  ml_single_h single;
  int status;
  ml_tensors_info_h in_info;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  ml_tensors_info_create (&in_info);

  in_dim[0] = 3;
  in_dim[1] = 224;
  in_dim[2] = 224;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  /* handle null data */
  status = ml_tensors_data_create_no_alloc (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_data_destroy (input);

  /* set invalid type to test wrong data size */
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT32);

  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_data_destroy (input);

  /* set invalid input tensor number */
  ml_tensors_info_set_count (in_info, 2);
  ml_tensors_info_set_tensor_type (in_info, 1, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (in_info, 1, in_dim);

  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_data_destroy (input);
  ml_tensors_info_destroy (in_info);

skip_test:
  g_free (test_model);
}

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

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
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

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

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

  ml_tensors_info_destroy (in_res);
  ml_tensors_info_destroy (out_res);

skip_test:
  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Start pipeline without tensor info
 */
TEST (nnstreamer_capi_singleshot, invoke_02)
{
  ml_single_h single;
  ml_tensors_info_h in_info;
  ml_tensors_data_h input, output;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  status = ml_single_get_input_info (single, &in_info);
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

  ml_tensors_info_destroy (in_info);
skip_test:
  g_free (test_model);
}

/**
 * @brief Measure the loading time and total time for the run
 */
static void
benchmark_single (const gboolean no_alloc, const gboolean no_timeout,
    const int count)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim;
  int status;
  unsigned long open_duration=0, invoke_duration=0, close_duration=0;
  gint64 start, end;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
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

  /** Initial run to warm up the cache */
  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }
  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  for (int i = 0; i < count; i++) {
    start = g_get_monotonic_time ();
    status = ml_single_open (&single, test_model, in_info, out_info,
        ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
    end = g_get_monotonic_time ();
    open_duration += end - start;
    ASSERT_EQ (status, ML_ERROR_NONE);

    if (!no_timeout) {
      status = ml_single_set_timeout (single, SINGLE_DEF_TIMEOUT_MSEC);
      EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED || status == ML_ERROR_NONE);
    }

    /* generate dummy data */
    input = output = NULL;

    status = ml_tensors_data_create (in_info, &input);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_TRUE (input != NULL);

    if (no_alloc) {
      status = ml_tensors_data_create (out_info, &output);
      EXPECT_EQ (status, ML_ERROR_NONE);
      EXPECT_TRUE (output != NULL);
    }

    start = g_get_monotonic_time ();
    if (no_alloc)
      status = ml_single_invoke_fast (single, input, output);
    else
      status = ml_single_invoke (single, input, &output);
    end = g_get_monotonic_time ();
    invoke_duration += end - start;
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_TRUE (output != NULL);

    start = g_get_monotonic_time ();
    status = ml_single_close (single);
    end = g_get_monotonic_time ();
    close_duration = end - start;
    EXPECT_EQ (status, ML_ERROR_NONE);

    ml_tensors_data_destroy (input);
    ml_tensors_data_destroy (output);
  }

  g_warning ("Time to open single = %f us", (open_duration * 1.0)/count);
  g_warning ("Time to invoke single = %f us", (invoke_duration * 1.0)/count);
  g_warning ("Time to close single = %f us", (close_duration * 1.0)/count);

skip_test:
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
  g_warning ("Benchmark (no timeout)");
  benchmark_single (FALSE, TRUE, 1);

  g_warning ("Benchmark (no alloc, no timeout)");
  benchmark_single (TRUE, TRUE, 1);
}

/**
 * @brief Test NNStreamer single shot (custom filter)
 * @detail Run pipeline with custom filter, handle multi tensors.
 */
TEST (nnstreamer_capi_singleshot, invoke_03)
{
  const gchar cf_name[] = "libnnstreamer_customfilter_passthrough_variable" \
      NNSTREAMER_SO_FILE_EXTENSION;
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

  test_model = g_build_filename (root_path, "nnstreamer_example", cf_name, NULL);
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
  ASSERT_EQ (status, ML_ERROR_NONE);

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

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
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
  ASSERT_EQ (status, ML_ERROR_NONE);

  /* input tensor in filter */
  status = ml_single_get_input_info (single, &in_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (in_res, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_name (in_res, 0, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_STREQ (name, "wav_data");
  g_free (name);

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
  EXPECT_STREQ (name, "labels_softmax");
  g_free (name);

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

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
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

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Failure case with invalid param.
 */
TEST (nnstreamer_capi_singleshot, open_fail_01_n)
{
  ml_single_h single;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  /* invalid file path */
  status = ml_single_open (&single, "wrong_file_name", NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* null file path */
  status = ml_single_open (&single, NULL, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid handle */
  status = ml_single_open (NULL, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid file extension */
  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid handle */
  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* Successfully opened unknown fw type (tf-lite) */
  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_ANY, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

skip_test:
  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Failure case with invalid tensor info.
 */
TEST (nnstreamer_capi_singleshot, open_fail_02_n)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensor_dimension in_dim, out_dim;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  /* invalid input tensor info */
  status = ml_single_open (&single, test_model, in_info, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid output tensor info */
  status = ml_single_open (&single, test_model, NULL, out_info,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  in_dim[0] = 3;
  in_dim[1] = 100;
  in_dim[2] = 100;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  /* invalid input dimension (model does not support dynamic dimension) */
  status = ml_single_open (&single, test_model, in_info, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
  } else {
    EXPECT_NE (status, ML_ERROR_INVALID_PARAMETER);
  }

  in_dim[1] = in_dim[2] = 224;
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT16);

  /* invalid input type */
  status = ml_single_open (&single, test_model, in_info, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
  } else {
    EXPECT_NE (status, ML_ERROR_INVALID_PARAMETER);
  }

  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_UINT8);

  out_dim[0] = 1;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_UINT8);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  /* invalid output dimension */
  status = ml_single_open (&single, test_model, NULL, out_info,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
  } else {
    EXPECT_NE (status, ML_ERROR_INVALID_PARAMETER);
  }

  out_dim[0] = 1001;
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_UINT16);

  /* invalid output type */
  status = ml_single_open (&single, test_model, NULL, out_info,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
  } else {
    EXPECT_NE (status, ML_ERROR_INVALID_PARAMETER);
  }

  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_UINT8);

  /* Successfully opened unknown fw type (tf-lite) */
  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_ANY, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

skip_test:
  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Open model (dynamic dimension is supported)
 */
TEST (nnstreamer_capi_singleshot, open_dynamic)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensor_dimension in_dim, out_dim;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;
  unsigned int count = 0;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  /* dynamic dimension supported */
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);

  in_dim[0] = 5;
  in_dim[1] = 1;
  in_dim[2] = 1;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  /* open with input tensor info (1:1:1:1 > 5:1:1:1) */
  status = ml_single_open (&single, test_model, in_info, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  /* validate output info */
  status = ml_single_get_output_info (single, &out_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (out_info, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_type (out_info, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_FLOAT32);

  ml_tensors_info_get_tensor_dimension (out_info, 0, out_dim);
  EXPECT_EQ (out_dim[0], 5U);
  EXPECT_EQ (out_dim[1], 1U);
  EXPECT_EQ (out_dim[2], 1U);
  EXPECT_EQ (out_dim[3], 1U);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_destroy (out_info);
skip_test:
  ml_tensors_info_destroy (in_info);
  g_free (test_model);
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
  if (ss_data->expect) {
    EXPECT_EQ (status, ML_ERROR_NONE);
    if (status != ML_ERROR_NONE)
      return NULL;
  }
  ss_data->single = &single;

  /* set timeout */
  if (ss_data->timeout != 0) {
    status = ml_single_set_timeout (single, ss_data->timeout);
    if (ss_data->expect) {
      EXPECT_NE (status, ML_ERROR_INVALID_PARAMETER);
    }
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
  if (ss_data->expect) {
    EXPECT_EQ (status, ML_ERROR_NONE);
  }

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

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

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

skip_test:
  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Testcase with multiple runs in parallel. Some of the
 *         running instances will timeout, however others will not.
 */
TEST (nnstreamer_capi_singleshot, parallel_runs)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;
  const gint num_threads = 3;
  const gint num_cases = 3;
  pthread_t thread[num_threads * num_cases];
  single_shot_thread_data ss_data[num_cases];
  guint i, j;

  /* Skip this test if enable-tensorflow-lite is false */
  if (!is_enabled_tensorflow_lite)
    return;

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
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;
  pthread_t thread;
  single_shot_thread_data ss_data;

  /* Skip this test if enable-tensorflow-lite is false */
  if (!is_enabled_tensorflow_lite)
    return;

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
TEST (nnstreamer_capi_singleshot, set_input_info_fail_01_n)
{
  int status;
  ml_single_h single;
  ml_tensors_info_h in_info;
  ml_tensor_dimension in_dim;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

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

  status = ml_tensors_info_destroy (in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

skip_test:
  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Try setting number of input tensors and its type
 */
TEST (nnstreamer_capi_singleshot, set_input_info_fail_02_n)
{
  ml_single_h single;
  ml_tensors_info_h in_info;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;
  unsigned int count = 0;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
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
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

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

  status = ml_tensors_info_destroy (in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

skip_test:
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

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

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

  status = ml_tensors_info_destroy (in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

skip_test:
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
  ml_tensors_info_h in_res = nullptr, out_res = nullptr;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim, res_dim;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;
  unsigned int count = 0;
  int status, tensor_size;
  size_t data_size;
  float *data;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
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
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

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
    ml_tensors_info_destroy (in_res);
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

skip_test:
  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
  ml_tensors_info_destroy (in_res);
  ml_tensors_info_destroy (out_res);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Update property 'layout' for input tensor
 */
TEST (nnstreamer_capi_singleshot, property_01_p)
{
  ml_single_h single;
  ml_tensors_info_h in_info;
  ml_tensors_data_h input, output;
  int status;
  char *prop_value;
  void *data;
  size_t data_size;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  /* get layout */
  status = ml_single_get_property (single, "inputlayout", &prop_value);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_STREQ (prop_value, "ANY");
  g_free (prop_value);

  /* get updatable */
  status = ml_single_get_property (single, "is-updatable", &prop_value);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_STREQ (prop_value, "false");
  g_free (prop_value);

  /* get input info */
  status = ml_single_get_input_info (single, &in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invoke */
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

  status = ml_tensors_data_get_tensor_data (output, 0, (void **) &data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, 1001U);

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);
  ml_tensors_info_destroy (in_info);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

skip_test:
  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Failure case to set invalid property
 */
TEST (nnstreamer_capi_singleshot, property_02_n)
{
  ml_single_h single;
  int status;
  char *prop_value = NULL;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  /* get invalid property */
  status = ml_single_get_property (single, "unknown_prop", &prop_value);
  EXPECT_NE (status, ML_ERROR_NONE);
  g_free (prop_value);

  /* set invalid property */
  status = ml_single_set_property (single, "unknown_prop", "INVALID");
  EXPECT_NE (status, ML_ERROR_NONE);

  /* null params */
  status = ml_single_set_property (single, "input", NULL);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_single_set_property (single, NULL, "INVALID");
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_single_get_property (single, "input", NULL);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_single_get_property (single, NULL, &prop_value);
  EXPECT_NE (status, ML_ERROR_NONE);
  g_free (prop_value);

  /* dimension should be valid */
  status = ml_single_get_property (single, "input", &prop_value);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_STREQ (prop_value, "3:224:224:1");
  g_free (prop_value);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

skip_test:
  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Failure case to set meta property
 */
TEST (nnstreamer_capi_singleshot, property_03_n)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensor_dimension in_dim, out_dim;
  ml_tensors_data_h input, output;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;
  int status;
  unsigned int count = 0;
  char *name = NULL;
  char *prop_value = NULL;
  void *data;
  size_t data_size;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  /* failed to set dimension */
  status = ml_single_set_property (single, "input", "3:4:4:1");
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_single_get_property (single, "input", &prop_value);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_STREQ (prop_value, "3:224:224:1");
  g_free (prop_value);

  /* input tensor in filter */
  status = ml_single_get_input_info (single, &in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (in_info, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_name (in_info, 0, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (name == NULL);

  status = ml_tensors_info_get_tensor_type (in_info, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_UINT8);

  ml_tensors_info_get_tensor_dimension (in_info, 0, in_dim);
  EXPECT_EQ (in_dim[0], 3U);
  EXPECT_EQ (in_dim[1], 224U);
  EXPECT_EQ (in_dim[2], 224U);
  EXPECT_EQ (in_dim[3], 1U);

  /* output tensor in filter */
  status = ml_single_get_output_info (single, &out_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (out_info, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_name (out_info, 0, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (name == NULL);

  status = ml_tensors_info_get_tensor_type (out_info, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_UINT8);

  ml_tensors_info_get_tensor_dimension (out_info, 0, out_dim);
  EXPECT_EQ (out_dim[0], 1001U);
  EXPECT_EQ (out_dim[1], 1U);
  EXPECT_EQ (out_dim[2], 1U);
  EXPECT_EQ (out_dim[3], 1U);

  /* invoke */
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

  status = ml_tensors_data_get_tensor_data (output, 0, (void **) &data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, 1001U);

  ml_tensors_data_destroy (input);
  ml_tensors_data_destroy (output);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

skip_test:
  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Update dimension for input tensor
 */
TEST (nnstreamer_capi_singleshot, property_04_p)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim;
  char *prop_value;
  int status;
  size_t data_size;
  float *data;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
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
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  status = ml_single_set_property (single, "input", "5:1:1:1");
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_get_property (single, "input", &prop_value);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_STREQ (prop_value, "5:1:1:1");
  g_free (prop_value);

  /* validate in/out info */
  status = ml_single_get_input_info (single, &in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_get_tensor_dimension (in_info, 0, in_dim);
  EXPECT_EQ (in_dim[0], 5U);
  EXPECT_EQ (in_dim[1], 1U);
  EXPECT_EQ (in_dim[2], 1U);
  EXPECT_EQ (in_dim[3], 1U);

  status = ml_single_get_output_info (single, &out_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_get_tensor_dimension (out_info, 0, out_dim);
  EXPECT_EQ (out_dim[0], 5U);
  EXPECT_EQ (out_dim[1], 1U);
  EXPECT_EQ (out_dim[2], 1U);
  EXPECT_EQ (out_dim[3], 1U);

  /* invoke */
  input = output = NULL;

  /* generate dummy data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_tensors_data_get_tensor_data (input, 0, (void **) &data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, 5 * sizeof (float));
  for (int idx = 0; idx < 5; idx++)
    data[idx] = idx;

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  status = ml_tensors_data_get_tensor_data (output, 0, (void **) &data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (data_size, 5 * sizeof (float));
  for (int idx = 0; idx < 5; idx++)
    EXPECT_EQ (data[idx], idx + 2);

  ml_tensors_data_destroy (input);
  ml_tensors_data_destroy (output);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

skip_test:
  g_free (test_model);
}

#ifdef ENABLE_NNFW_RUNTIME
/**
 * @brief Test NNStreamer single shot (nnfw backend)
 */
TEST (nnstreamer_capi_singleshot, invoke_05)
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

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  in_dim[0] = 1;
  in_dim[1] = 1;
  in_dim[2] = 1;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = 1;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_NNFW, ML_NNFW_HW_ANY);
  ASSERT_EQ (status, ML_ERROR_NONE);

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

  status = ml_tensors_info_get_tensor_name (out_res, 0, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (name == NULL);

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
 * @brief Test NNStreamer single shot (nnfw backend using model path)
 */
TEST (nnstreamer_capi_singleshot, open_dir)
{
  ml_single_h single;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_NNFW, ML_NNFW_HW_ANY);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
}

#endif  /* ENABLE_NNFW_RUNTIME */

#ifdef ENABLE_ARMNN
/**
 * @brief Test NNStreamer single shot (caffe/armnn)
 * @detail Run pipeline with caffe lenet model.
 */
TEST (nnstreamer_capi_singleshot, invoke_06)
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

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model, *test_file;
  guint8 *contents_uint8 = NULL;
  gfloat *contents_float = NULL;
  gsize len = 0;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "lenet_iter_9000.caffemodel", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  test_file = g_build_filename (root_path, "tests", "test_models", "data",
      "9.raw", NULL);
  ASSERT_TRUE (g_file_test (test_file, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  in_dim[0] = 28;
  in_dim[1] = 28;
  in_dim[2] = 1;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_name (in_info, 0, "data");
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = 10;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_name (out_info, 0, "prob");
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  ASSERT_TRUE (g_file_get_contents (test_file, (gchar **) &contents_uint8, &len,
        NULL));
  status = ml_tensors_info_get_tensor_size (in_info, 0, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ASSERT_TRUE (len == data_size / sizeof (float));

  /** Convert uint8 data with range [0, 255] to float with range [-1, 1] */
  contents_float = (gfloat *) g_malloc (data_size);
  for (unsigned int idx=0; idx < len; idx ++) {
    contents_float[idx] = static_cast<float> (contents_uint8[idx]);
    contents_float[idx] -= 127.5;
    contents_float[idx] /= 127.5;
  }

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_ARMNN, ML_NNFW_HW_ANY);
  ASSERT_EQ (status, ML_ERROR_NONE);

  /* input tensor in filter */
  status = ml_single_get_input_info (single, &in_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_info_get_count (in_res, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1U);

  status = ml_tensors_info_get_tensor_name (in_res, 0, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_STREQ (name, "data");
  g_free (name);

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

  status = ml_tensors_info_get_tensor_name (out_res, 0, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_STREQ (name, "prob");
  g_free (name);

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

  status = ml_tensors_data_set_tensor_data (input, 0, contents_float, data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_set_timeout (single, SINGLE_DEF_TIMEOUT_MSEC);
  EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED || status == ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  status = ml_tensors_data_get_tensor_data (output, 1, &data_ptr, &data_size);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_tensors_data_get_tensor_data (output, 0, &data_ptr, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);

  max_score = .0;
  max_score_index = 0;
  for (gint i = 0; i < 10; i++) {
    score = ((float *) data_ptr)[i];
    if (score > max_score) {
      max_score = score;
      max_score_index = i;
    }
  }

  EXPECT_EQ (max_score_index, 9);

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  g_free (test_file);
  g_free (contents_uint8);
  g_free (contents_float);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
  ml_tensors_info_destroy (in_res);
  ml_tensors_info_destroy (out_res);
}

/**
 * @brief Test NNStreamer single shot (tflite/armnn)
 * @detail Run pipeline with tflite basic model.
 */
TEST (nnstreamer_capi_singleshot, invoke_07)
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
  void *data_ptr;
  size_t data_size;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  in_dim[0] = 1;
  in_dim[1] = 1;
  in_dim[2] = 1;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = 1;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_ARMNN, ML_NNFW_HW_ANY);
  ASSERT_EQ (status, ML_ERROR_NONE);

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

  status = ml_tensors_info_get_tensor_name (out_res, 0, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (name == NULL);

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

  status = ml_tensors_data_get_tensor_data (input, 0, &data_ptr, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  ((float *) data_ptr)[0] = 10.0;

  status = ml_single_set_timeout (single, SINGLE_DEF_TIMEOUT_MSEC);
  EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED || status == ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  status = ml_tensors_data_get_tensor_data (output, 0, &data_ptr, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (((float *) data_ptr)[0], 12.0);

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
 * @brief Test NNStreamer single shot (caffe/armnn)
 * @detail Failure open with invalid param.
 */
TEST (nnstreamer_capi_singleshot, open_fail_03_n)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensor_dimension in_dim, out_dim;
  int status;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "lenet_iter_9000.caffemodel", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  /** Set the correct input/output info */
  in_dim[0] = 28;
  in_dim[1] = 28;
  in_dim[2] = 1;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_name (in_info, 0, "data");
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = 10;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_name (out_info, 0, "prob");
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  /** Modify the input or output name to be wrong and open */
  ml_tensors_info_set_tensor_name (in_info, 0, "data1");
  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_ARMNN, ML_NNFW_HW_ANY);
  EXPECT_NE (status, ML_ERROR_NONE);
  ml_tensors_info_set_tensor_name (in_info, 0, "data");

  ml_tensors_info_set_tensor_name (out_info, 0, "prob1");
  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_ARMNN, ML_NNFW_HW_ANY);
  EXPECT_NE (status, ML_ERROR_NONE);
  ml_tensors_info_set_tensor_name (out_info, 0, "prob");

  /**
   * Modify the input dim to be wrong and open
   * output dim is not used for caffe, so wrong output dim will pass open
   * but will fail at invoke (check nnstreamer_capi_singleshot.invoke_07_n)
   */
  ml_tensors_info_set_tensor_dimension (in_info, 0, out_dim);
  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_ARMNN, ML_NNFW_HW_ANY);
  EXPECT_NE (status, ML_ERROR_NONE);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test NNStreamer single shot (caffe/armnn)
 * @detail Failure invoke with invalid param.
 */
TEST (nnstreamer_capi_singleshot, invoke_08_n)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim;
  int status;
  size_t data_size;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;
  gfloat *contents_float = NULL;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "lenet_iter_9000.caffemodel", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  in_dim[0] = 28;
  in_dim[1] = 28;
  in_dim[2] = 1;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_name (in_info, 0, "data");
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = 10;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_name (out_info, 0, "prob");
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);
  ml_tensors_info_set_tensor_dimension (out_info, 0, in_dim);

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_ARMNN, ML_NNFW_HW_ANY);
  ASSERT_EQ (status, ML_ERROR_NONE);

  input = output = NULL;

  /* generate input data with wrong info */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_tensors_info_get_tensor_size (in_info, 0, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  contents_float = (gfloat *) g_malloc (data_size);
  status = ml_tensors_data_set_tensor_data (input, 0, contents_float, data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_set_timeout (single, SINGLE_DEF_TIMEOUT_MSEC);
  EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED || status == ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_NE (status, ML_ERROR_NONE);
  EXPECT_TRUE (output == NULL);

  ml_tensors_data_destroy (input);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  g_free (contents_float);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test NNStreamer single shot (caffe/armnn)
 * @detail Failure invoke with invalid param.
 */
TEST (nnstreamer_capi_singleshot, invoke_09_n)
{
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_info_h in_res, out_res;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim;
  int status;
  size_t data_size;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;
  gfloat *contents_float = NULL;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "lenet_iter_9000.caffemodel", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);
  ml_tensors_info_create (&in_res);
  ml_tensors_info_create (&out_res);

  in_dim[0] = 28;
  in_dim[1] = 28;
  in_dim[2] = 1;
  in_dim[3] = 1;
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_name (in_info, 0, "data");
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  out_dim[0] = 10;
  out_dim[1] = 1;
  out_dim[2] = 1;
  out_dim[3] = 1;
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_name (out_info, 0, "prob");
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_ARMNN, ML_NNFW_HW_ANY);
  ASSERT_EQ (status, ML_ERROR_NONE);

  input = output = NULL;

  /* generate input data with wrong info */
  status = ml_tensors_data_create (out_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_tensors_info_get_tensor_size (out_info, 0, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  contents_float = (gfloat *) g_malloc (data_size);
  status = ml_tensors_data_set_tensor_data (input, 0, contents_float, data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_set_timeout (single, SINGLE_DEF_TIMEOUT_MSEC);
  EXPECT_TRUE (status == ML_ERROR_NOT_SUPPORTED || status == ML_ERROR_NONE);

  status = ml_single_invoke (single, input, &output);
  EXPECT_NE (status, ML_ERROR_NONE);
  EXPECT_TRUE (output == NULL);

  ml_tensors_data_destroy (input);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  g_free (contents_float);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
  ml_tensors_info_destroy (in_res);
  ml_tensors_info_destroy (out_res);
}
#endif  /* ENABLE_ARMNN */

/**
 * @brief Test NNStreamer single shot (custom filter)
 * @detail Run pipeline with custom filter with allocate in invoke, handle multi tensors.
 */
TEST (nnstreamer_capi_singleshot, invoke_10_p)
{
  const gchar cf_name[] = "libnnstreamer_customfilter_scaler_allocator" \
      NNSTREAMER_SO_FILE_EXTENSION;
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

  test_model = g_build_filename (root_path, "nnstreamer_example", cf_name,
      NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  ml_tensors_info_set_count (in_info, 1);

  in_dim[0] = 10;
  in_dim[1] = 1;
  in_dim[2] = 1;
  in_dim[3] = 1;

  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT16);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  ml_tensors_info_clone (out_info, in_info);

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_CUSTOM_FILTER, ML_NNFW_HW_ANY);
  ASSERT_EQ (status, ML_ERROR_NONE);

  input = output = NULL;

  /* generate input data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  ASSERT_TRUE (input != NULL);

  status = ml_tensors_data_get_tensor_data (input, 0, &data_ptr, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  for (i = 0; i < 10; i++) {
    ((int16_t *) data_ptr)[i] = (int16_t) (i + 1);
  }

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  /**
   * since the output data was allocated by the tensor filter element in the single API,
   * closing this single handle will also delete the data
   */
  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_destroy (input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);

  /** This will destroy twice resulting in UB */
  // EXPECT_DEATH (ml_tensors_data_destroy (output), ".*");
}

/**
 * @brief Test NNStreamer single shot (custom filter)
 * @detail Run pipeline with custom filter with allocate in invoke, handle multi tensors.
 */
TEST (nnstreamer_capi_singleshot, invoke_11_p)
{
  const gchar cf_name[] = "libnnstreamer_customfilter_scaler_allocator" \
      NNSTREAMER_SO_FILE_EXTENSION;
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

  test_model = g_build_filename (root_path, "nnstreamer_example", cf_name,
      NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  ml_tensors_info_set_count (in_info, 1);

  in_dim[0] = 10;
  in_dim[1] = 1;
  in_dim[2] = 1;
  in_dim[3] = 1;

  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT16);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  ml_tensors_info_clone (out_info, in_info);

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_CUSTOM_FILTER, ML_NNFW_HW_ANY);
  ASSERT_EQ (status, ML_ERROR_NONE);

  input = output = NULL;

  /* generate input data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  ASSERT_TRUE (input != NULL);

  status = ml_tensors_data_get_tensor_data (input, 0, &data_ptr, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  for (i = 0; i < 10; i++) {
    ((int16_t *) data_ptr)[i] = (int16_t) (i + 1);
  }

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  status = ml_tensors_data_destroy (input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /** Access data before destroy works */
  status = ml_tensors_data_get_tensor_data (output, 0, &data_ptr, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /** User can destroy by themselves */
  status = ml_tensors_data_destroy (output);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /** Close handle works normally */
  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test NNStreamer single shot (custom filter)
 * @detail Run pipeline with custom filter with allocate in invoke, handle multi tensors.
 */
TEST (nnstreamer_capi_singleshot, invoke_12_p)
{
  const gchar cf_name[] = "libnnstreamer_customfilter_scaler_allocator" \
      NNSTREAMER_SO_FILE_EXTENSION;
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_h input, output1, output2;
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

  test_model = g_build_filename (root_path, "nnstreamer_example", cf_name,
      NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  ml_tensors_info_set_count (in_info, 1);

  in_dim[0] = 10;
  in_dim[1] = 1;
  in_dim[2] = 1;
  in_dim[3] = 1;

  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT16);
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);

  ml_tensors_info_clone (out_info, in_info);

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_CUSTOM_FILTER, ML_NNFW_HW_ANY);
  ASSERT_EQ (status, ML_ERROR_NONE);

  input = output1 = output2 = NULL;

  /* generate input data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  ASSERT_TRUE (input != NULL);

  status = ml_tensors_data_get_tensor_data (input, 0, &data_ptr, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  for (i = 0; i < 10; i++) {
    ((int16_t *) data_ptr)[i] = (int16_t) (i + 1);
  }

  status = ml_single_invoke (single, input, &output1);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output1 != NULL);

  status = ml_single_invoke (single, input, &output2);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output2 != NULL);

  status = ml_tensors_data_destroy (input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /** Destroy one data by user */
  status = ml_tensors_data_destroy (output1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /** Destroy the other data by closing the handle */
  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);

  /** This will destroy twice resulting in UB */
  // EXPECT_DEATH (ml_tensors_data_destroy (output2), ".*");
}

/**
 * @brief Test NNStreamer single shot (custom filter)
 * @detail Change the number of input tensors, run the model and verify output
 */
TEST (nnstreamer_capi_singleshot, set_input_info_success_02)
{
  const gchar cf_name[] = "libnnstreamer_customfilter_passthrough_variable" \
      NNSTREAMER_SO_FILE_EXTENSION;
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_info_h in_res = nullptr, out_res = nullptr;
  ml_tensors_data_h input, output;
  ml_tensor_dimension in_dim, out_dim, res_dim;
  ml_tensor_type_e type = ML_TENSOR_TYPE_UNKNOWN;
  unsigned int count = 0;
  int status, tensor_size;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  /* custom-passthrough */
  test_model = g_build_filename (root_path, "nnstreamer_example", cf_name,
      NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

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

  status = ml_single_open (&single, test_model, in_info, out_info,
      ML_NNFW_TYPE_CUSTOM_FILTER, ML_NNFW_HW_ANY);
  ASSERT_EQ (status, ML_ERROR_NONE);

  /* Run the model once with the original input/output info */
  input = output = NULL;

  /* generate dummy data */
  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (input != NULL);

  status = ml_single_invoke (single, input, &output);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (output != NULL);

  ml_tensors_data_destroy (output);
  ml_tensors_data_destroy (input);

  /** modify input/output info and run again */
  in_dim[0] = 10;
  ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);
  out_dim[0] = 10;
  ml_tensors_info_set_tensor_dimension (out_info, 0, out_dim);

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
    ml_tensors_info_destroy (in_res);
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

    status = ml_single_invoke (single, input, &output);
    EXPECT_EQ (status, ML_ERROR_NONE);
    EXPECT_TRUE (output != NULL);

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

/**
 * @brief Test NNStreamer single shot (tflite)
 * @detail run the `ml_single_invoke_dynamic` api works properly.
 */
TEST (nnstreamer_capi_singleshot, invoke_dynamic_success_01_p)
{
  ml_single_h single;
  int status;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_h input, output;
  size_t data_size;

  unsigned int tmp_count;
  ml_tensor_type_e tmp_type = ML_TENSOR_TYPE_UNKNOWN;
  ml_tensor_dimension tmp_dim;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  /* dynamic dimension supported */
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  status = ml_single_get_input_info (single, &in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);
  {
    float tmp_input[] = { 1.0 };
    float *output_buf;
    status = ml_tensors_data_set_tensor_data (input, 0, tmp_input,
        1 * sizeof (float));

    ml_tensors_info_get_count (in_info, &tmp_count);
    ml_tensors_info_get_tensor_type (in_info, 0, &tmp_type);
    ml_tensors_info_get_tensor_dimension (in_info, 0, tmp_dim);

    EXPECT_EQ (tmp_count, 1U);
    EXPECT_EQ (tmp_type, ML_TENSOR_TYPE_FLOAT32);
    EXPECT_EQ (tmp_dim[0], 1U);
    EXPECT_EQ (tmp_dim[1], 1U);
    EXPECT_EQ (tmp_dim[2], 1U);
    EXPECT_EQ (tmp_dim[3], 1U);

    status =
        ml_single_invoke_dynamic (single, input, in_info, &output, &out_info);
    EXPECT_EQ (status, ML_ERROR_NONE);

    ml_tensors_data_get_tensor_data (output, 0, (void **) &output_buf,
        &data_size);

    EXPECT_FLOAT_EQ (output_buf[0], 3.0f);
    EXPECT_EQ (data_size, sizeof (float));

    ml_tensors_info_get_count (out_info, &tmp_count);
    ml_tensors_info_get_tensor_type (out_info, 0, &tmp_type);
    ml_tensors_info_get_tensor_dimension (out_info, 0, tmp_dim);

    EXPECT_EQ (tmp_count, 1U);
    EXPECT_EQ (tmp_type, ML_TENSOR_TYPE_FLOAT32);
    EXPECT_EQ (tmp_dim[0], 1U);
    EXPECT_EQ (tmp_dim[1], 1U);
    EXPECT_EQ (tmp_dim[2], 1U);
    EXPECT_EQ (tmp_dim[3], 1U);

    ml_tensors_data_destroy (output);
    ml_tensors_data_destroy (input);
    ml_tensors_info_destroy (in_info);
    ml_tensors_info_destroy (out_info);
  }

  status = ml_single_set_property (single, "input", "5:1:1:1");
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_get_input_info (single, &in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  {
    float tmp_input2[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    float *output_buf2;
    status = ml_tensors_data_set_tensor_data (input, 0, tmp_input2,
        5 * sizeof (float));

    ml_tensors_info_get_count (in_info, &tmp_count);
    ml_tensors_info_get_tensor_type (in_info, 0, &tmp_type);
    ml_tensors_info_get_tensor_dimension (in_info, 0, tmp_dim);

    EXPECT_EQ (tmp_count, 1U);
    EXPECT_EQ (tmp_type, ML_TENSOR_TYPE_FLOAT32);
    EXPECT_EQ (tmp_dim[0], 5U);
    EXPECT_EQ (tmp_dim[1], 1U);
    EXPECT_EQ (tmp_dim[2], 1U);
    EXPECT_EQ (tmp_dim[3], 1U);

    status =
        ml_single_invoke_dynamic (single, input, in_info, &output, &out_info);
    EXPECT_EQ (status, ML_ERROR_NONE);

    ml_tensors_data_get_tensor_data (output, 0, (void **) &output_buf2,
        &data_size);

    EXPECT_FLOAT_EQ (output_buf2[0], 3.0f);
    EXPECT_FLOAT_EQ (output_buf2[1], 4.0f);
    EXPECT_FLOAT_EQ (output_buf2[2], 5.0f);
    EXPECT_FLOAT_EQ (output_buf2[3], 6.0f);
    EXPECT_FLOAT_EQ (output_buf2[4], 7.0f);
    EXPECT_EQ (data_size, 5 * sizeof (float));

    ml_tensors_info_get_count (out_info, &tmp_count);
    ml_tensors_info_get_tensor_type (out_info, 0, &tmp_type);
    ml_tensors_info_get_tensor_dimension (out_info, 0, tmp_dim);

    EXPECT_EQ (tmp_count, 1U);
    EXPECT_EQ (tmp_type, ML_TENSOR_TYPE_FLOAT32);
    EXPECT_EQ (tmp_dim[0], 5U);
    EXPECT_EQ (tmp_dim[1], 1U);
    EXPECT_EQ (tmp_dim[2], 1U);
    EXPECT_EQ (tmp_dim[3], 1U);

    status = ml_single_close (single);
    EXPECT_EQ (status, ML_ERROR_NONE);

    ml_tensors_data_destroy (output);
    ml_tensors_data_destroy (input);
    ml_tensors_info_destroy (in_info);
    ml_tensors_info_destroy (out_info);
  }
skip_test:
  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tflite)
 * @detail run the `ml_single_invoke_dynamic` api works properly.
 */
TEST (nnstreamer_capi_singleshot, invoke_dynamic_success_02_p)
{
  ml_single_h single;
  int status;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_h input, output;
  size_t data_size;

  unsigned int tmp_count;
  ml_tensor_type_e tmp_type = ML_TENSOR_TYPE_UNKNOWN;
  ml_tensor_dimension tmp_dim, in_dim;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  /* dynamic dimension supported */
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  status = ml_single_get_input_info (single, &in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  {
    float tmp_input[] = { 1.0 };
    float *output_buf;

    status = ml_tensors_data_set_tensor_data (input, 0, tmp_input,
        1 * sizeof (float));

    ml_tensors_info_get_count (in_info, &tmp_count);
    ml_tensors_info_get_tensor_type (in_info, 0, &tmp_type);
    ml_tensors_info_get_tensor_dimension (in_info, 0, tmp_dim);

    EXPECT_EQ (tmp_count, 1U);
    EXPECT_EQ (tmp_type, ML_TENSOR_TYPE_FLOAT32);
    EXPECT_EQ (tmp_dim[0], 1U);
    EXPECT_EQ (tmp_dim[1], 1U);
    EXPECT_EQ (tmp_dim[2], 1U);
    EXPECT_EQ (tmp_dim[3], 1U);

    status =
        ml_single_invoke_dynamic (single, input, in_info, &output, &out_info);
    EXPECT_EQ (status, ML_ERROR_NONE);

    ml_tensors_data_get_tensor_data (output, 0, (void **) &output_buf,
        &data_size);
    EXPECT_FLOAT_EQ (output_buf[0], 3.0f);
    EXPECT_EQ (data_size, sizeof (float));

    ml_tensors_info_get_count (out_info, &tmp_count);
    ml_tensors_info_get_tensor_type (out_info, 0, &tmp_type);
    ml_tensors_info_get_tensor_dimension (out_info, 0, tmp_dim);

    EXPECT_EQ (tmp_count, 1U);
    EXPECT_EQ (tmp_type, ML_TENSOR_TYPE_FLOAT32);
    EXPECT_EQ (tmp_dim[0], 1U);
    EXPECT_EQ (tmp_dim[1], 1U);
    EXPECT_EQ (tmp_dim[2], 1U);
    EXPECT_EQ (tmp_dim[3], 1U);

    ml_tensors_data_destroy (output);
    ml_tensors_data_destroy (input);
    ml_tensors_info_destroy (in_info);
    ml_tensors_info_destroy (out_info);
  }

  status = ml_single_get_input_info (single, &in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  in_dim[0] = 5;
  in_dim[1] = 1;
  in_dim[2] = 1;
  in_dim[3] = 1;

  status = ml_tensors_info_set_tensor_dimension (in_info, 0, in_dim);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  {
    float tmp_input2[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    float *output_buf2;
    status = ml_tensors_data_set_tensor_data (input, 0, tmp_input2,
        5 * sizeof (float));

    ml_tensors_info_get_count (in_info, &tmp_count);
    ml_tensors_info_get_tensor_type (in_info, 0, &tmp_type);
    ml_tensors_info_get_tensor_dimension (in_info, 0, tmp_dim);

    EXPECT_EQ (tmp_count, 1U);
    EXPECT_EQ (tmp_type, ML_TENSOR_TYPE_FLOAT32);
    EXPECT_EQ (tmp_dim[0], 5U);
    EXPECT_EQ (tmp_dim[1], 1U);
    EXPECT_EQ (tmp_dim[2], 1U);
    EXPECT_EQ (tmp_dim[3], 1U);

    status =
        ml_single_invoke_dynamic (single, input, in_info, &output, &out_info);
    EXPECT_EQ (status, ML_ERROR_NONE);

    ml_tensors_data_get_tensor_data (output, 0, (void **) &output_buf2,
        &data_size);

    EXPECT_FLOAT_EQ (output_buf2[0], 3.0f);
    EXPECT_FLOAT_EQ (output_buf2[1], 4.0f);
    EXPECT_FLOAT_EQ (output_buf2[2], 5.0f);
    EXPECT_FLOAT_EQ (output_buf2[3], 6.0f);
    EXPECT_FLOAT_EQ (output_buf2[4], 7.0f);
    EXPECT_EQ (data_size, 5 * sizeof (float));

    ml_tensors_info_get_count (out_info, &tmp_count);
    ml_tensors_info_get_tensor_type (out_info, 0, &tmp_type);
    ml_tensors_info_get_tensor_dimension (out_info, 0, tmp_dim);

    EXPECT_EQ (tmp_count, 1U);
    EXPECT_EQ (tmp_type, ML_TENSOR_TYPE_FLOAT32);
    EXPECT_EQ (tmp_dim[0], 5U);
    EXPECT_EQ (tmp_dim[1], 1U);
    EXPECT_EQ (tmp_dim[2], 1U);
    EXPECT_EQ (tmp_dim[3], 1U);

    status = ml_single_close (single);
    EXPECT_EQ (status, ML_ERROR_NONE);

    ml_tensors_data_destroy (output);
    ml_tensors_data_destroy (input);
    ml_tensors_info_destroy (in_info);
    ml_tensors_info_destroy (out_info);
  }

skip_test:
  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tflite)
 * @detail check the `ml_single_invoke_dynamic` api handles exception cases well.
 */
TEST (nnstreamer_capi_singleshot, invoke_dynamic_fail_n)
{
  ml_single_h single;
  int status;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_h input, output;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  /* dynamic dimension supported */
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  if (is_enabled_tensorflow_lite) {
    ASSERT_EQ (status, ML_ERROR_NONE);
  } else {
    ASSERT_NE (status, ML_ERROR_NONE);
    goto skip_test;
  }

  status = ml_single_get_input_info (single, &in_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_tensors_data_create (in_info, &input);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_single_invoke_dynamic (NULL, input, in_info, &output, &out_info);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_single_invoke_dynamic (single, NULL, in_info, &output, &out_info);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_single_invoke_dynamic (single, input, NULL, &output, &out_info);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_single_invoke_dynamic (single, input, in_info, NULL, &out_info);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_single_invoke_dynamic (single, input, in_info, &output, NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_data_destroy (input);
  ml_tensors_info_destroy (in_info);

skip_test:
  g_free (test_model);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_handle()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_handle_00_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h  = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_handle()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_handle_01_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_handle (handle, nullptr, &vsrc_h);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "WRONG_PROPERTY_NAME", &vsrc_h);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_release_handle()` API and check its results.
 */
TEST (nnstreamer_capi_element, release_handle_02_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  ml_pipeline_element_h selector_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "is01", &selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_release_handle()` API and check its results.
 */
TEST (nnstreamer_capi_element, release_handle_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_bool()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_bool_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h selector_h = nullptr;
  gchar *pipeline;
  int status;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "is01", &selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_bool(selector_h, "sync-streams", FALSE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_bool(selector_h, "sync-streams", TRUE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_bool()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_bool_02_n)
{
  int status;

  /* Test Code */
  status = ml_pipeline_element_set_property_bool(nullptr, "sync-streams", FALSE);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_bool()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_bool_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h selector_h = nullptr;
  gchar *pipeline;
  int status;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "is01", &selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_bool(selector_h, "WRONG_PROPERTY", TRUE);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_bool()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_bool_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vscale_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vscale", &vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_bool (vscale_h, "sharpness", 10);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_bool()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_bool_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h selector_h = nullptr;
  gchar *pipeline;
  int ret_sync_streams;
  int status;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "is01", &selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_bool(selector_h, "sync-streams", FALSE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_bool (selector_h, "sync-streams", &ret_sync_streams);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (ret_sync_streams, FALSE);

  status = ml_pipeline_element_set_property_bool(selector_h, "sync-streams", TRUE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_property_bool (selector_h, "sync-streams", &ret_sync_streams);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (ret_sync_streams, TRUE);

  status = ml_pipeline_element_release_handle (selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_bool()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_bool_02_n)
{
  int ret_sync_streams;
  int status;

  /* Test Code */
  status = ml_pipeline_element_get_property_bool (nullptr, "sync-streams", &ret_sync_streams);
  ASSERT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_bool()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_bool_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h selector_h = nullptr;
  gchar *pipeline;
  int ret_sync_streams;
  int status;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "is01", &selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_bool(selector_h, "sync-streams", FALSE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_bool (selector_h, "WRONG_NAME", &ret_sync_streams);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_bool()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_bool_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h selector_h = nullptr;
  gchar *pipeline;
  int status;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "is01", &selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_bool(selector_h, "sync-streams", FALSE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_bool (selector_h, "sync-streams", nullptr);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_bool()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_bool_05_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h udpsrc_h = nullptr;
  int status;
  int wrong_type;
  gchar *pipeline;

  pipeline = g_strdup("udpsrc name=usrc port=5555 caps=application/x-rtp ! queue ! fakesink");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "usrc", &udpsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_uint64 (udpsrc_h, "timeout", 123456789123456789ULL);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_bool (udpsrc_h, "timeout", &wrong_type);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (udpsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_string()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_string_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h demux_h = nullptr;
  gchar *pipeline;
  int status;

  pipeline = g_strdup("videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! videorate max-rate=1 ! " \
    "tensor_converter ! tensor_mux ! tensor_demux name=demux ! tensor_sink");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "demux", &demux_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_string (demux_h, "tensorpick", "1,2");
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (demux_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_string()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_string_02_n)
{
  int status;

  /* Test Code */
  status = ml_pipeline_element_set_property_string (nullptr, "tensorpick", "1,2");
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_string()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_string_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h demux_h = nullptr;
  gchar *pipeline;
  int status;

  pipeline = g_strdup("videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! videorate max-rate=1 ! " \
    "tensor_converter ! tensor_mux ! tensor_demux name=demux ! tensor_sink");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "demux", &demux_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_string (demux_h, "WRONG_NAME", "1,2");
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (demux_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_string()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_string_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h selector_h = nullptr;
  gchar *pipeline;
  int status;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "is01", &selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_string(selector_h, "sync-streams", "TRUE");
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_string()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_string_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h demux_h = nullptr;
  gchar *pipeline;
  gchar *ret_tensorpick;
  int status;

  pipeline = g_strdup("videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! videorate max-rate=1 ! " \
    "tensor_converter ! tensor_mux ! tensor_demux name=demux ! tensor_sink");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "demux", &demux_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_string (demux_h, "tensorpick", "1,2");
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_string (demux_h, "tensorpick", &ret_tensorpick);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (g_str_equal (ret_tensorpick, "1,2"));
  g_free (ret_tensorpick);

  status = ml_pipeline_element_set_property_string (demux_h, "tensorpick", "1");
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_property_string (demux_h, "tensorpick", &ret_tensorpick);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (g_str_equal (ret_tensorpick, "1"));
  g_free (ret_tensorpick);

  status = ml_pipeline_element_release_handle (demux_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_string()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_string_02_n)
{
  int status;
  gchar *ret_tensorpick;

  /* Test Code */
  status = ml_pipeline_element_get_property_string (nullptr, "tensorpick", &ret_tensorpick);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_string()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_string_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h demux_h = nullptr;
  gchar *pipeline;
  gchar *ret_tensorpick;
  int status;

  pipeline = g_strdup("videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! videorate max-rate=1 ! " \
    "tensor_converter ! tensor_mux ! tensor_demux name=demux ! tensor_sink");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "demux", &demux_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_string (demux_h, "tensorpick", "1,2");
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_string (demux_h, "WRONG_NAME", &ret_tensorpick);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (demux_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_string()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_string_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h demux_h = nullptr;
  gchar *pipeline;
  int status;

  pipeline = g_strdup("videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! videorate max-rate=1 ! " \
    "tensor_converter ! tensor_mux ! tensor_demux name=demux ! tensor_sink");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "demux", &demux_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_string (demux_h, "tensorpick", "1,2");
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_string (demux_h, "tensorpick", nullptr);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (demux_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_string()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_string_05_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h selector_h = nullptr;
  gchar *pipeline;
  gchar *ret_wrong_type;
  int status;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "is01", &selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_bool(selector_h, "sync-streams", FALSE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_string (selector_h, "sync-streams", &ret_wrong_type);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (selector_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_int32()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_int32_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_int32 (vsrc_h, "kx", 10);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_int32 (vsrc_h, "kx", -1234);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_int32()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_int32_02_n)
{
  int status;

  /* Test Code */
  status = ml_pipeline_element_set_property_int32 (nullptr, "kx", 10);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_int32()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_int32_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_int32 (vsrc_h, "WRONG_NAME", 10);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_int32()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_int32_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h demux_h = nullptr;
  gchar *pipeline;
  int status;

  pipeline = g_strdup("videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! videorate max-rate=1 ! " \
    "tensor_converter ! tensor_mux ! tensor_demux name=demux ! tensor_sink");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "demux", &demux_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_int32 (demux_h, "tensorpick", 1);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (demux_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_int32()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_int32_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  int32_t ret_kx;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_int32 (vsrc_h, "kx", 10);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_int32 (vsrc_h, "kx", &ret_kx);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (ret_kx == 10);

  status = ml_pipeline_element_set_property_int32 (vsrc_h, "kx", -1234);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_property_int32 (vsrc_h, "kx", &ret_kx);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (ret_kx == -1234);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_int32()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_int32_02_n)
{
  int status;
  int32_t ret_kx;

  /* Test Code */
  status = ml_pipeline_element_get_property_int32 (nullptr, "kx", &ret_kx);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_int32()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_int32_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  int32_t ret_kx;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_int32 (vsrc_h, "kx", 10);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_int32 (vsrc_h, "WRONG_NAME", &ret_kx);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_int32()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_int32_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_int32 (vsrc_h, "kx", 10);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_int32 (vsrc_h, "kx", nullptr);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_int32()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_int32_05_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vscale_h = nullptr;
  int status;
  int wrong_type;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vscale", &vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_double (vscale_h, "sharpness", 0.72);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_int32 (vscale_h, "sharpness", &wrong_type);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_int64()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_int64_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_int64 (vsrc_h, "timestamp-offset", 1234567891234LL);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_int64 (vsrc_h, "timestamp-offset", 10LL);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_int64()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_int64_02_n)
{
  int status;

  /* Test Code */
  status = ml_pipeline_element_set_property_int64 (nullptr, "timestamp-offset", 1234567891234LL);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_int64()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_int64_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_int64 (vsrc_h, "WRONG_NAME", 1234567891234LL);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_int64()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_int64_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_int64 (vsrc_h, "foreground-color", 123456);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_int64()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_int64_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  int64_t ret_timestame_offset;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_int64 (vsrc_h, "timestamp-offset", 1234567891234LL);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_int64 (vsrc_h, "timestamp-offset", &ret_timestame_offset);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (ret_timestame_offset == 1234567891234LL);

  status = ml_pipeline_element_set_property_int64 (vsrc_h, "timestamp-offset", 10LL);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_property_int64 (vsrc_h, "timestamp-offset", &ret_timestame_offset);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (ret_timestame_offset == 10LL);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_int64()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_int64_02_n)
{
  int status;
  int64_t ret_timestame_offset;

  /* Test Code */
  status = ml_pipeline_element_get_property_int64 (nullptr, "timestamp-offset", &ret_timestame_offset);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_int64()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_int64_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  int64_t ret_timestame_offset;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_int64 (vsrc_h, "timestamp-offset", 1234567891234LL);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_int64 (vsrc_h, "WRONG_NAME", &ret_timestame_offset);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_int64()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_int64_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  int64_t wrong_type;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_uint32 (vsrc_h, "foreground-color", 123456U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_int64 (vsrc_h, "foreground-color", &wrong_type);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_int64()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_int64_05_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_int64 (vsrc_h, "timestamp-offset", 1234567891234LL);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_int64 (vsrc_h, "timestamp-offset", nullptr);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_uint32()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_uint32_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_uint32 (vsrc_h, "foreground-color", 123456U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_uint32 (vsrc_h, "foreground-color", 4294967295U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_uint32()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_uint32_02_n)
{
  int status;

  /* Test Code */
  status = ml_pipeline_element_set_property_uint32 (nullptr, "foreground-color", 123456U);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_uint32()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_uint32_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_uint32 (vsrc_h, "WRONG_NAME", 123456U);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_uint32()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_uint32_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_uint32 (vsrc_h, "kx", 10U);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_uint32()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_uint32_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  uint32_t ret_foreground_color;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_uint32 (vsrc_h, "foreground-color", 123456U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_uint32 (vsrc_h, "foreground-color", &ret_foreground_color);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (ret_foreground_color == 123456U);

  status = ml_pipeline_element_set_property_uint32 (vsrc_h, "foreground-color", 4294967295U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_property_uint32 (vsrc_h, "foreground-color", &ret_foreground_color);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (ret_foreground_color == 4294967295U);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_uint32()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_uint32_02_n)
{
  int status;
  uint32_t ret_foreground_color;

  /* Test Code */
  status = ml_pipeline_element_get_property_uint32 (nullptr, "foreground-color", &ret_foreground_color);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_uint32()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_uint32_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  uint32_t ret_foreground_color;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_uint32 (vsrc_h, "foreground-color", 123456U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_uint32 (vsrc_h, "WRONG_NAME", &ret_foreground_color);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_uint32()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_uint32_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  uint32_t ret_wrong_type;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_int32 (vsrc_h, "kx", 10);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_uint32 (vsrc_h, "kx", &ret_wrong_type);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_uint32()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_uint32_05_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_uint32 (vsrc_h, "foreground-color", 123456U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_uint32 (vsrc_h, "foreground-color", nullptr);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_uint64()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_uint64_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h udpsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("udpsrc name=usrc port=5555 caps=application/x-rtp ! queue ! fakesink");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "usrc", &udpsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_uint64 (udpsrc_h, "timeout", 123456789123456789ULL);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_uint64 (udpsrc_h, "timeout", 987654321ULL);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (udpsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_uint64()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_uint64_02_n)
{
  int status;

  /* Test Code */
  status = ml_pipeline_element_set_property_uint64 (nullptr, "timeout", 123456789123456789ULL);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_uint64()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_uint64_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h udpsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("udpsrc name=usrc port=5555 caps=application/x-rtp ! queue ! fakesink");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "usrc", &udpsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_uint64 (udpsrc_h, "WRONG_NAME", 123456789123456789ULL);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (udpsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_uint64()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_uint64_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_uint64 (vsrc_h, "timestamp-offset", 12ULL);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_uint64()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_uint64_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h udpsrc_h = nullptr;
  int status;
  uint64_t ret_timeout;
  gchar *pipeline;

  pipeline = g_strdup("udpsrc name=usrc port=5555 caps=application/x-rtp ! queue ! fakesink");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "usrc", &udpsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_uint64 (udpsrc_h, "timeout", 123456789123456789ULL);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_uint64 (udpsrc_h, "timeout", &ret_timeout);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (ret_timeout == 123456789123456789ULL);

  status = ml_pipeline_element_set_property_uint64 (udpsrc_h, "timeout", 987654321ULL);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_property_uint64 (udpsrc_h, "timeout", &ret_timeout);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (ret_timeout == 987654321ULL);

  status = ml_pipeline_element_release_handle (udpsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_uint64()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_uint64_02_n)
{
  int status;
  uint64_t ret_timeout;

  /* Test Code */
  status = ml_pipeline_element_get_property_uint64 (nullptr, "timeout", &ret_timeout);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_uint64()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_uint64_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h udpsrc_h = nullptr;
  int status;
  uint64_t ret_timeout;
  gchar *pipeline;

  pipeline = g_strdup("udpsrc name=usrc port=5555 caps=application/x-rtp ! queue ! fakesink");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "usrc", &udpsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_uint64 (udpsrc_h, "timeout", 123456789123456789ULL);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_uint64 (udpsrc_h, "WRONG_NAME", &ret_timeout);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (udpsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_uint64()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_uint64_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  int status;
  uint64_t wrong_type;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_int64 (vsrc_h, "timestamp-offset", 1234567891234LL);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_uint64 (vsrc_h, "timestamp-offset", &wrong_type);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_uint64()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_uint64_05_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h udpsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("udpsrc name=usrc port=5555 caps=application/x-rtp ! queue ! fakesink");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "usrc", &udpsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_uint64 (udpsrc_h, "timeout", 123456789123456789ULL);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_uint64 (udpsrc_h, "timeout", nullptr);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (udpsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_double()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_double_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vscale_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vscale", &vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_double (vscale_h, "sharpness", 0.72);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_double (vscale_h, "sharpness", 1.43);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_double()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_double_02_n)
{
  int status;

  /* Test Code */
  status = ml_pipeline_element_set_property_double (nullptr, "sharpness", 0.72);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_double()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_double_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vscale_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vscale", &vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_double (vscale_h, "WRONG_NAME", 1.43);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_double()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_double_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vscale_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vscale", &vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_double (vscale_h, "method", 3.0);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_double()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_double_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vscale_h = nullptr;
  int status;
  double ret_sharpness;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vscale", &vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_double (vscale_h, "sharpness", 0.72);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_double (vscale_h, "sharpness", &ret_sharpness);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (ret_sharpness, 0.72);

  status = ml_pipeline_element_set_property_double (vscale_h, "sharpness", 1.43);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_property_double (vscale_h, "sharpness", &ret_sharpness);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (ret_sharpness, 1.43);

  status = ml_pipeline_element_release_handle (vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_double()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_double_02_n)
{
  int status;
  double ret_sharpness;

  /* Test Code */
  status = ml_pipeline_element_get_property_double (nullptr, "sharpness", &ret_sharpness);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_double()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_double_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vscale_h = nullptr;
  int status;
  double ret_sharpness;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vscale", &vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_double (vscale_h, "sharpness", 0.72);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_double (vscale_h, "WRONG_NAME", &ret_sharpness);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_double()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_double_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vscale_h = nullptr;
  int status;
  double wrong_type;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vscale", &vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_enum (vscale_h, "method", 3U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_double (vscale_h, "method", &wrong_type);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_double()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_double_05_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vscale_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vscale", &vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_double (vscale_h, "sharpness", 0.72);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_double (vscale_h, "sharpness", nullptr);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_enum()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_enum_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vscale_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vscale", &vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_enum (vscale_h, "method", 3U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_enum (vscale_h, "method", 5U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_int32 (vscale_h, "method", 4);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_uint32 (vscale_h, "method", 2U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_enum()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_enum_02_n)
{
  int status;

  /* Test Code */
  status = ml_pipeline_element_set_property_enum (nullptr, "method", 3U);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_enum()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_enum_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vscale_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vscale", &vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_enum (vscale_h, "WRONG_NAME", 3U);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_set_property_enum()` API and check its results.
 */
TEST (nnstreamer_capi_element, set_property_enum_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h udpsrc_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("udpsrc name=usrc port=5555 caps=application/x-rtp ! queue ! fakesink");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "usrc", &udpsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_set_property_enum (udpsrc_h, "timeout", 12345);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (udpsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_enum()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_enum_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vscale_h = nullptr;
  int status;
  uint32_t ret_method;
  int32_t ret_signed_method;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vscale", &vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_enum (vscale_h, "method", 3U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_enum (vscale_h, "method", &ret_method);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (ret_method, 3U);

  status = ml_pipeline_element_set_property_enum (vscale_h, "method", 5U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_property_enum (vscale_h, "method", &ret_method);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (ret_method, 5U);

  status = ml_pipeline_element_set_property_uint32 (vscale_h, "method", 2U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_property_uint32 (vscale_h, "method", &ret_method);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (ret_method, 2U);

  status = ml_pipeline_element_set_property_int32 (vscale_h, "method", 4);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_property_int32 (vscale_h, "method", &ret_signed_method);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (ret_signed_method, 4);

  status = ml_pipeline_element_release_handle (vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_enum()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_enum_02_n)
{

  int status;
  uint32_t ret_method;

  /* Test Code */
  status = ml_pipeline_element_get_property_enum (nullptr, "method", &ret_method);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_enum()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_enum_03_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vscale_h = nullptr;
  int status;
  uint32_t ret_method;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vscale", &vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_enum (vscale_h, "method", 3U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_enum (vscale_h, "WRONG_NAME", &ret_method);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_enum()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_enum_04_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h demux_h = nullptr;
  gchar *pipeline;
  uint32_t ret_wrong_type;
  int status;

  pipeline = g_strdup("videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! videorate max-rate=1 ! " \
    "tensor_converter ! tensor_mux ! tensor_demux name=demux ! tensor_sink");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "demux", &demux_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_string (demux_h, "tensorpick", "1,2");
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_enum (demux_h, "tensorpick", &ret_wrong_type);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (demux_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Run the `ml_pipeline_element_get_property_enum()` API and check its results.
 */
TEST (nnstreamer_capi_element, get_property_enum_05_n)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vscale_h = nullptr;
  int status;
  gchar *pipeline;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale name=vscale ! " \
    "video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! " \
    "valve name=valvex ! input-selector name=is01 ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vscale", &vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_set_property_enum (vscale_h, "method", 3U);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test Code */
  status = ml_pipeline_element_get_property_enum (vscale_h, "method", nullptr);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (vscale_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Construct the pipeline and run it during updating elements' property.
 */
TEST (nnstreamer_capi_element, scenario_01_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_element_h vsrc_h = nullptr;
  ml_pipeline_state_e state;
  gchar *pipeline;
  int status;

  pipeline = g_strdup("videotestsrc name=vsrc is-live=true ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! " \
    "tensor_converter ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "vsrc", &vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Test code: Set the videotestsrc pattern */
  status = ml_pipeline_element_set_property_enum (vsrc_h, "pattern", 4);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (50000);

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  wait_for_start (handle, state, status);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PLAYING);

  /* Stop playing */
  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (50000);

  /* Test code: Set the new videotestsrc pattern */
  status = ml_pipeline_element_set_property_enum (vsrc_h, "pattern", 12);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* Resume playing */
  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (50000);

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  wait_for_start (handle, state, status);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PLAYING);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (50000);

  status = ml_pipeline_element_release_handle (vsrc_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free(pipeline);
}

/**
 * @brief Test case of Element Property Control.
 * @detail Construct the pipeline and run it during updating elements' property.
 */
TEST (nnstreamer_capi_element, scenario_02_p)
{
  ml_pipeline_h handle = nullptr;
  ml_pipeline_sink_h sinkhandle = nullptr;
  ml_pipeline_element_h asink_h = nullptr;
  gchar *pipeline;
  guint *count_sink;
  int status;

  pipeline = g_strdup ("videotestsrc is-live=true ! videoconvert ! tensor_converter ! appsink name=sinkx sync=false");

  count_sink = (guint *) g_malloc (sizeof (guint));
  ASSERT_TRUE (count_sink != NULL);
  *count_sink = 0;

  status = ml_pipeline_construct (pipeline, nullptr, nullptr, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_get_handle (handle, "sinkx", &asink_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_count, count_sink, &sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (sinkhandle != NULL);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (100000);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (*count_sink > 0U);

  /* Test Code */
  *count_sink = 0;

  status = ml_pipeline_element_set_property_bool (asink_h, "emit-signals", FALSE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (100000);

  /* Since `emit-signals` property of appsink is set as FALSE, *count_sink should be 0 */
  EXPECT_TRUE (*count_sink == 0U);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_unregister (sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_element_release_handle (asink_h);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
  g_free (count_sink);
}

/**
 * @brief Test for internal function 'ml_tensors_info_copy_from_gst'.
 */
TEST (nnstreamer_capi_internal, copy_from_gst)
{
  int status;
  ml_tensors_info_h ml_info;
  ml_tensor_type_e type;
  ml_tensor_dimension dim;
  char *name;
  unsigned int count;
  GstTensorsInfo gst_info;
  guint i;

  gst_tensors_info_init (&gst_info);
  gst_info.num_tensors = 2;
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    gst_info.info[0].dimension[i] = i + 1;
    gst_info.info[1].dimension[i] = i + 1;
  }

  status = ml_tensors_info_create (&ml_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_copy_from_gst ((ml_tensors_info_s *) ml_info, &gst_info);
  status = ml_tensors_info_get_count (ml_info, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 2U);
  status = ml_tensors_info_get_tensor_dimension (ml_info, 0, dim);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (dim[0], 1U);
  EXPECT_EQ (dim[1], 2U);
  EXPECT_EQ (dim[2], 3U);
  EXPECT_EQ (dim[3], 4U);

  gst_info.info[0].type = _NNS_INT32;
  gst_info.info[1].type = _NNS_UINT32;
  ml_tensors_info_copy_from_gst ((ml_tensors_info_s *) ml_info, &gst_info);
  status = ml_tensors_info_get_tensor_type (ml_info, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_INT32);
  status = ml_tensors_info_get_tensor_type (ml_info, 1, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_UINT32);

  gst_info.info[0].type = _NNS_INT16;
  gst_info.info[1].type = _NNS_UINT16;
  ml_tensors_info_copy_from_gst ((ml_tensors_info_s *) ml_info, &gst_info);
  status = ml_tensors_info_get_tensor_type (ml_info, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_INT16);
  status = ml_tensors_info_get_tensor_type (ml_info, 1, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_UINT16);

  gst_info.info[0].type = _NNS_INT8;
  gst_info.info[1].type = _NNS_UINT8;
  ml_tensors_info_copy_from_gst ((ml_tensors_info_s *) ml_info, &gst_info);
  status = ml_tensors_info_get_tensor_type (ml_info, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_INT8);
  status = ml_tensors_info_get_tensor_type (ml_info, 1, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_UINT8);

  gst_info.info[0].type = _NNS_INT64;
  gst_info.info[1].type = _NNS_UINT64;
  ml_tensors_info_copy_from_gst ((ml_tensors_info_s *) ml_info, &gst_info);
  status = ml_tensors_info_get_tensor_type (ml_info, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_INT64);
  status = ml_tensors_info_get_tensor_type (ml_info, 1, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_UINT64);

  gst_info.info[0].type = _NNS_FLOAT64;
  gst_info.info[1].type = _NNS_FLOAT32;
  ml_tensors_info_copy_from_gst ((ml_tensors_info_s *) ml_info, &gst_info);
  status = ml_tensors_info_get_tensor_type (ml_info, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_FLOAT64);
  status = ml_tensors_info_get_tensor_type (ml_info, 1, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_FLOAT32);

  gst_info.info[0].name = g_strdup ("tn1");
  gst_info.info[1].name = g_strdup ("tn2");
  ml_tensors_info_copy_from_gst ((ml_tensors_info_s *) ml_info, &gst_info);
  status = ml_tensors_info_get_tensor_name (ml_info, 0, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_STREQ (name, "tn1");
  g_free (name);
  status = ml_tensors_info_get_tensor_name (ml_info, 1, &name);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_STREQ (name, "tn2");
  g_free (name);

  status = ml_tensors_info_destroy (ml_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  gst_tensors_info_free (&gst_info);
}

/**
 * @brief Test for internal function 'ml_tensors_info_copy_from_ml'.
 */
TEST (nnstreamer_capi_internal, copy_from_ml)
{
  int status;
  ml_tensors_info_h ml_info;
  ml_tensor_dimension dim = { 1, 2, 3, 4 };
  GstTensorsInfo gst_info;

  gst_tensors_info_init (&gst_info);

  status = ml_tensors_info_create (&ml_info);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_count (ml_info, 2);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_dimension (ml_info, 0, dim);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_dimension (ml_info, 1, dim);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_copy_from_ml (&gst_info, (ml_tensors_info_s *) ml_info);
  EXPECT_EQ (gst_info.num_tensors, 2U);
  EXPECT_EQ (gst_info.info[0].dimension[0], 1U);
  EXPECT_EQ (gst_info.info[0].dimension[1], 2U);
  EXPECT_EQ (gst_info.info[0].dimension[2], 3U);
  EXPECT_EQ (gst_info.info[0].dimension[3], 4U);

  status = ml_tensors_info_set_tensor_type (ml_info, 0, ML_TENSOR_TYPE_INT32);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_type (ml_info, 1, ML_TENSOR_TYPE_UINT32);
  EXPECT_EQ (status, ML_ERROR_NONE);
  ml_tensors_info_copy_from_ml (&gst_info, (ml_tensors_info_s *) ml_info);
  EXPECT_EQ (gst_info.info[0].type, _NNS_INT32);
  EXPECT_EQ (gst_info.info[1].type, _NNS_UINT32);

  status = ml_tensors_info_set_tensor_type (ml_info, 0, ML_TENSOR_TYPE_INT16);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_type (ml_info, 1, ML_TENSOR_TYPE_UINT16);
  EXPECT_EQ (status, ML_ERROR_NONE);
  ml_tensors_info_copy_from_ml (&gst_info, (ml_tensors_info_s *) ml_info);
  EXPECT_EQ (gst_info.info[0].type, _NNS_INT16);
  EXPECT_EQ (gst_info.info[1].type, _NNS_UINT16);

  status = ml_tensors_info_set_tensor_type (ml_info, 0, ML_TENSOR_TYPE_INT8);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_type (ml_info, 1, ML_TENSOR_TYPE_UINT8);
  EXPECT_EQ (status, ML_ERROR_NONE);
  ml_tensors_info_copy_from_ml (&gst_info, (ml_tensors_info_s *) ml_info);
  EXPECT_EQ (gst_info.info[0].type, _NNS_INT8);
  EXPECT_EQ (gst_info.info[1].type, _NNS_UINT8);

  status = ml_tensors_info_set_tensor_type (ml_info, 0, ML_TENSOR_TYPE_INT64);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_type (ml_info, 1, ML_TENSOR_TYPE_UINT64);
  EXPECT_EQ (status, ML_ERROR_NONE);
  ml_tensors_info_copy_from_ml (&gst_info, (ml_tensors_info_s *) ml_info);
  EXPECT_EQ (gst_info.info[0].type, _NNS_INT64);
  EXPECT_EQ (gst_info.info[1].type, _NNS_UINT64);

  status = ml_tensors_info_set_tensor_type (ml_info, 0, ML_TENSOR_TYPE_FLOAT64);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_type (ml_info, 1, ML_TENSOR_TYPE_FLOAT32);
  EXPECT_EQ (status, ML_ERROR_NONE);
  ml_tensors_info_copy_from_ml (&gst_info, (ml_tensors_info_s *) ml_info);
  EXPECT_EQ (gst_info.info[0].type, _NNS_FLOAT64);
  EXPECT_EQ (gst_info.info[1].type, _NNS_FLOAT32);

  status = ml_tensors_info_set_tensor_name (ml_info, 0, "tn1");
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_tensors_info_set_tensor_name (ml_info, 1, "tn2");
  EXPECT_EQ (status, ML_ERROR_NONE);
  ml_tensors_info_copy_from_ml (&gst_info, (ml_tensors_info_s *) ml_info);
  EXPECT_STREQ (gst_info.info[0].name, "tn1");
  EXPECT_STREQ (gst_info.info[1].name, "tn2");

  status = ml_tensors_info_destroy (ml_info);
  EXPECT_EQ (status, ML_ERROR_NONE);

  gst_tensors_info_free (&gst_info);
}

/**
 * @brief Test for internal function 'ml_validate_model_file'.
 * @detail Invalid params.
 */
TEST (nnstreamer_capi_internal, validate_model_file_01_n)
{
  const gchar cf_name[] = "libnnstreamer_customfilter_passthrough_variable" \
      NNSTREAMER_SO_FILE_EXTENSION;
  int status;
  ml_nnfw_type_e nnfw = ML_NNFW_TYPE_CUSTOM_FILTER;
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "nnstreamer_example", cf_name,
      NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  status = ml_validate_model_file (NULL, 1, &nnfw);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_validate_model_file (&test_model, 0, &nnfw);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_validate_model_file (&test_model, 1, NULL);
  EXPECT_NE (status, ML_ERROR_NONE);

  g_free (test_model);
}

/**
 * @brief Test for internal function 'ml_validate_model_file'.
 * @detail Invalid file extension.
 */
TEST (nnstreamer_capi_internal, validate_model_file_02_n)
{
  const gchar cf_name[] = "libnnstreamer_customfilter_passthrough_variable" \
      NNSTREAMER_SO_FILE_EXTENSION;
  int status;
  ml_nnfw_type_e nnfw;
  const gchar *sroot_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar *broot_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model1, *test_model2;
  gchar *test_models[2];

  /* supposed to run test in build directory */
  if (sroot_path == NULL)
    sroot_path = "..";
  if (broot_path == NULL)
    broot_path = ".";

  test_model1 = g_build_filename (broot_path, "nnstreamer_example", cf_name,
      NULL);
  test_model2 = g_build_filename (sroot_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model1, G_FILE_TEST_EXISTS));
  ASSERT_TRUE (g_file_test (test_model2, G_FILE_TEST_EXISTS));

  test_models[0] = test_model1;
  test_models[1] = test_model2;

  nnfw = ML_NNFW_TYPE_CUSTOM_FILTER;
  status = ml_validate_model_file (&test_model2, 1, &nnfw);
  EXPECT_NE (status, ML_ERROR_NONE);

  nnfw = ML_NNFW_TYPE_TENSORFLOW_LITE;
  status = ml_validate_model_file (&test_model1, 1, &nnfw);
  EXPECT_NE (status, ML_ERROR_NONE);

  nnfw = ML_NNFW_TYPE_TENSORFLOW;
  status = ml_validate_model_file (&test_model1, 1, &nnfw);
  EXPECT_NE (status, ML_ERROR_NONE);

  /* snap only for android */
  nnfw = ML_NNFW_TYPE_SNAP;
  status = ml_validate_model_file (&test_model1, 1, &nnfw);
  EXPECT_NE (status, ML_ERROR_NONE);

  nnfw = ML_NNFW_TYPE_VIVANTE;
  status = ml_validate_model_file (test_models, 1, &nnfw);
  EXPECT_NE (status, ML_ERROR_NONE);

  /** @todo currently mvnc, openvino and edgetpu always return failure */
  nnfw = ML_NNFW_TYPE_MVNC;
  status = ml_validate_model_file (&test_model1, 1, &nnfw);
  EXPECT_NE (status, ML_ERROR_NONE);

  nnfw = ML_NNFW_TYPE_OPENVINO;
  status = ml_validate_model_file (&test_model1, 1, &nnfw);
  EXPECT_NE (status, ML_ERROR_NONE);

  nnfw = ML_NNFW_TYPE_EDGE_TPU;
  status = ml_validate_model_file (&test_model1, 1, &nnfw);
  EXPECT_NE (status, ML_ERROR_NONE);

  nnfw = ML_NNFW_TYPE_ARMNN;
  status = ml_validate_model_file (&test_model1, 1, &nnfw);
  EXPECT_NE (status, ML_ERROR_NONE);

  g_free (test_model1);
  g_free (test_model2);
}

/**
 * @brief Test for internal function 'ml_validate_model_file'.
 * @detail Invalid model path.
 */
TEST (nnstreamer_capi_internal, validate_model_file_03_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  int status;
  ml_nnfw_type_e nnfw;
  gchar *test_dir1, *test_dir2;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  /* test model path */
  test_dir1 = g_build_filename (root_path, "tests", "test_models", "models", NULL);

  /* invalid dir */
  test_dir2 = g_build_filename (test_dir1, "invaliddir", NULL);

  nnfw = ML_NNFW_TYPE_TENSORFLOW_LITE;
  status = ml_validate_model_file (&test_dir1, 1, &nnfw);
  EXPECT_NE (status, ML_ERROR_NONE);

  nnfw = ML_NNFW_TYPE_TENSORFLOW;
  status = ml_validate_model_file (&test_dir1, 1, &nnfw);
  EXPECT_NE (status, ML_ERROR_NONE);

  nnfw = ML_NNFW_TYPE_NNFW;
  status = ml_validate_model_file (&test_dir2, 1, &nnfw);
  EXPECT_NE (status, ML_ERROR_NONE);

  /* only NNFW supports dir path */
  nnfw = ML_NNFW_TYPE_NNFW;
  status = ml_validate_model_file (&test_dir1, 1, &nnfw);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_dir1);
  g_free (test_dir2);
}

/**
 * @brief Invoke callback for custom-easy filter.
 */
static int
test_custom_easy_cb (const ml_tensors_data_h in, ml_tensors_data_h out,
    void *user_data)
{
  /* test code, set data size. */
  if (user_data) {
    void *raw_data = NULL;
    size_t *data_size = (size_t *) user_data;

    ml_tensors_data_get_tensor_data (out, 0, &raw_data, data_size);
  }

  return 0;
}

/**
 * @brief Test for custom-easy registration.
 */
TEST (nnstreamer_capi_custom, register_filter_01_p)
{
  const char test_custom_filter[] = "test-custom-filter";
  ml_pipeline_h pipe;
  ml_pipeline_src_h src;
  ml_pipeline_sink_h sink;
  ml_custom_easy_filter_h custom;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_h in_data;
  ml_tensor_dimension dim = { 2, 1, 1, 1 };
  int status;
  gchar *pipeline =
      g_strdup_printf
      ("appsrc name=srcx ! other/tensor,dimension=(string)2:1:1:1,type=(string)int8,framerate=(fraction)0/1 ! tensor_filter framework=custom-easy model=%s ! tensor_sink name=sinkx",
      test_custom_filter);
  guint *count_sink = (guint *) g_malloc0 (sizeof (guint));
  size_t *filter_data_size = (size_t *) g_malloc0 (sizeof (size_t));
  size_t data_size;
  guint i;

  ml_tensors_info_create (&in_info);
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, dim);

  ml_tensors_info_create (&out_info);
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, dim);
  ml_tensors_info_get_tensor_size(out_info, 0, &data_size);

  /* test code for custom filter */
  status = ml_pipeline_custom_easy_filter_register (test_custom_filter,
      in_info, out_info, test_custom_easy_cb, filter_data_size, &custom);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_construct (pipeline, NULL, NULL, &pipe);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (pipe, "sinkx", test_sink_callback_count,
      count_sink, &sink);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_get_handle (pipe, "srcx", &src);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (pipe);
  EXPECT_EQ (status, ML_ERROR_NONE);

  for (i = 0; i < 5; i++) {
    status = ml_tensors_data_create (in_info, &in_data);
    EXPECT_EQ (status, ML_ERROR_NONE);

    status = ml_pipeline_src_input_data (src, in_data, ML_PIPELINE_BUF_POLICY_AUTO_FREE);
    EXPECT_EQ (status, ML_ERROR_NONE);

    g_usleep (50000); /* 50ms. Wait a bit. */
  }

  status = ml_pipeline_stop (pipe);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_release_handle (src);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_unregister (sink);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (pipe);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_custom_easy_filter_unregister (custom);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* check received data in sink node */
  EXPECT_TRUE (*count_sink > 0U);
  EXPECT_TRUE (*filter_data_size > 0U && *filter_data_size == data_size);

  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
  g_free (pipeline);
  g_free (count_sink);
  g_free (filter_data_size);
}

/**
 * @brief Test for custom-easy registration.
 * @detail Invalid params.
 */
TEST (nnstreamer_capi_custom, register_filter_02_n)
{
  ml_custom_easy_filter_h custom;
  ml_tensors_info_h in_info, out_info;
  int status;
  ml_tensor_dimension dim = { 2, 1, 1, 1 };

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);


  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, dim);

  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, dim);

  /* test code with null param */
  status = ml_pipeline_custom_easy_filter_register (NULL,
      in_info, out_info, test_custom_easy_cb, NULL, &custom);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test for custom-easy registration.
 * @detail Invalid params.
 */
TEST (nnstreamer_capi_custom, register_filter_03_n)
{
  const char test_custom_filter[] = "test-custom-filter";
  ml_custom_easy_filter_h custom;
  ml_tensors_info_h out_info;
  int status;
  ml_tensor_dimension dim = { 2, 1, 1, 1 };

  ml_tensors_info_create (&out_info);

  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, dim);

  /* test code with null param */
  status = ml_pipeline_custom_easy_filter_register (test_custom_filter,
      NULL, out_info, test_custom_easy_cb, NULL, &custom);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test for custom-easy registration.
 * @detail Invalid params.
 */
TEST (nnstreamer_capi_custom, register_filter_04_n)
{
  const char test_custom_filter[] = "test-custom-filter";
  ml_custom_easy_filter_h custom;
  ml_tensors_info_h in_info;
  int status;
  ml_tensor_dimension dim = { 2, 1, 1, 1 };

  ml_tensors_info_create (&in_info);

  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, dim);

  /* test code with null param */
  status = ml_pipeline_custom_easy_filter_register (test_custom_filter,
      in_info, NULL, test_custom_easy_cb, NULL, &custom);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_info_destroy (in_info);
}

/**
 * @brief Test for custom-easy registration.
 * @detail Invalid params.
 */
TEST (nnstreamer_capi_custom, register_filter_05_n)
{
  const char test_custom_filter[] = "test-custom-filter";
  ml_custom_easy_filter_h custom;
  ml_tensors_info_h in_info, out_info;
  int status;
  ml_tensor_dimension dim = { 2, 1, 1, 1 };

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, dim);

  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, dim);

  /* test code with null param */
  status = ml_pipeline_custom_easy_filter_register (test_custom_filter,
      in_info, out_info, NULL, NULL, &custom);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test for custom-easy registration.
 * @detail Invalid params.
 */
TEST (nnstreamer_capi_custom, register_filter_06_n)
{
  const char test_custom_filter[] = "test-custom-filter";
  ml_tensors_info_h in_info, out_info;
  int status;
  ml_tensor_dimension dim = { 2, 1, 1, 1 };

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, dim);

  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, dim);

  /* test code with null param */
  status = ml_pipeline_custom_easy_filter_register (test_custom_filter,
      in_info, out_info, test_custom_easy_cb, NULL, NULL);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test for custom-easy registration.
 * @detail Invalid params.
 */
TEST (nnstreamer_capi_custom, register_filter_07_n)
{
  int status;

  /* test code with null param */
  status = ml_pipeline_custom_easy_filter_unregister (NULL);
  EXPECT_NE (status, ML_ERROR_NONE);
}

/**
 * @brief Test for custom-easy registration.
 * @detail Invalid params.
 */
TEST (nnstreamer_capi_custom, register_filter_08_n)
{
  const char test_custom_filter[] = "test-custom-filter";
  ml_custom_easy_filter_h custom;
  ml_tensors_info_h in_info, out_info;
  ml_tensor_dimension dim = { 2, 1, 1, 1 };
  int status;

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  /* test code with invalid output info */
  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, dim);

  status = ml_pipeline_custom_easy_filter_register (test_custom_filter,
      in_info, out_info, test_custom_easy_cb, NULL, &custom);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test for custom-easy registration.
 * @detail Invalid params.
 */
TEST (nnstreamer_capi_custom, register_filter_09_n)
{
  const char test_custom_filter[] = "test-custom-filter";
  ml_custom_easy_filter_h custom;
  ml_tensors_info_h in_info, out_info;
  ml_tensor_dimension dim = { 2, 1, 1, 1 };
  int status;

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  /* test code with invalid input info */
  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, dim);

  status = ml_pipeline_custom_easy_filter_register (test_custom_filter,
      in_info, out_info, test_custom_easy_cb, NULL, &custom);
  EXPECT_NE (status, ML_ERROR_NONE);

  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Test for custom-easy registration.
 * @detail Invalid params.
 */
TEST (nnstreamer_capi_custom, register_filter_10_n)
{
  const char test_custom_filter[] = "test-custom-filter";
  ml_custom_easy_filter_h custom, custom2;
  ml_tensors_info_h in_info, out_info;
  ml_tensor_dimension dim = { 2, 1, 1, 1 };
  int status;

  ml_tensors_info_create (&in_info);
  ml_tensors_info_create (&out_info);

  ml_tensors_info_set_count (in_info, 1);
  ml_tensors_info_set_tensor_type (in_info, 0, ML_TENSOR_TYPE_INT8);
  ml_tensors_info_set_tensor_dimension (in_info, 0, dim);

  ml_tensors_info_set_count (out_info, 1);
  ml_tensors_info_set_tensor_type (out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension (out_info, 0, dim);

  /* test code for duplicated name */
  status = ml_pipeline_custom_easy_filter_register (test_custom_filter,
      in_info, out_info, test_custom_easy_cb, NULL, &custom);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_custom_easy_filter_register (test_custom_filter,
      in_info, out_info, test_custom_easy_cb, NULL, &custom2);
  EXPECT_NE (status, ML_ERROR_NONE);

  status = ml_pipeline_custom_easy_filter_unregister (custom);
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
}

/**
 * @brief Main gtest
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

  /* ignore tizen feature status while running the testcases */
  set_feature_state (SUPPORTED);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  set_feature_state (NOT_CHECKED_YET);

  return result;
}
