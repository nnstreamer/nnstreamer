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

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_construct_destruct, dummy_01)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! fakesink";
  ml_pipeline_h handle;
  int status = ml_pipeline_construct (pipeline, &handle);
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
  int status = ml_pipeline_construct (pipeline, &handle);
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
  int status = ml_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline construct with non-existent filter
 */
TEST (nnstreamer_capi_construct_destruct, failed_01)
{
  const char *pipeline = "nonexistsrc ! fakesink";
  ml_pipeline_h handle;
  int status = ml_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, ML_ERROR_STREAMS_PIPE);
}

/**
 * @brief Test NNStreamer pipeline construct with erroneous pipeline
 */
TEST (nnstreamer_capi_construct_destruct, failed_02)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! audioconvert ! fakesink";
  ml_pipeline_h handle;
  int status = ml_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, ML_ERROR_STREAMS_PIPE);
}

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_playstop, dummy_01)
{
  const char *pipeline = "videotestsrc is-live=true num-buffers=30 ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! valve name=valvex ! valve name=valvey ! input-selector name=is01 ! tensor_sink name=sinkx";
  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  int status = ml_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  g_usleep (50000); /* 50ms. Let a few frames flow. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
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
  const char *pipeline = "videotestsrc is-live=true num-buffers=30 ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! valve name=valvex ! valve name=valvey ! input-selector name=is01 ! tensor_sink name=sinkx";
  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  int status = ml_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  g_usleep (50000); /* 50ms. Let a few frames flow. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
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
      ("videotestsrc is-live=true num-buffers=20 ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=16,height=16,framerate=60/1 ! tensor_converter ! queue ! valve name=valve1 ! filesink location=\"%s\"",
      file1);
  GStatBuf buf;

  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  ml_pipeline_valve_h valve1;

  int status = ml_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_TRUE (dir != NULL);

  status = ml_pipeline_valve_get_handle (handle, "valve1", &valve1);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_valve_control (valve1, 1); /* close */
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  g_usleep (100000); /* 100ms. Let a few frames flow. */
  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = g_lstat (file1, &buf);
  EXPECT_EQ (status, 0);
  EXPECT_EQ (buf.st_size, 0);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_valve_control (valve1, 0); /* open */
  EXPECT_EQ (status, ML_ERROR_NONE);
  status = ml_pipeline_valve_put_handle (valve1); /* release valve handle */
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (50000); /* 50ms. Let a few frames flow. */

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = g_lstat (file1, &buf);
  EXPECT_EQ (status, 0);
  EXPECT_GE (buf.st_size, 2048); /* At least two frames during 50ms */
  EXPECT_LE (buf.st_size, 4096); /* At most four frames during 50ms */

  g_free (fullpath);
  g_free (file1);
  g_free (pipeline);
}

/**
 * @brief Test NNStreamer pipeline valve
 * @detail Failure case to handle valve element with invalid param.
 */
TEST (nnstreamer_capi_valve, failure_01)
{
  ml_pipeline_h handle;
  ml_pipeline_valve_h valve_h;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  status = ml_pipeline_construct (pipeline, &handle);
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

/**
 * @brief A tensor-sink callback for sink handle in a pipeline
 */
static void
test_sink_callback_dm01 (const ml_tensors_data_s * data,
    const ml_tensors_info_s * info, void *pdata)
{
  gchar *filepath = (gchar *) pdata;
  FILE *fp = g_fopen (filepath, "a");
  if (fp == NULL)
    return;

  int i, num = info->num_tensors;

  for (i = 0; i < num; i++) {
    fwrite (data->tensors[i].tensor, data->tensors[i].size, 1, fp);
  }

  fclose (fp);
}

/**
 * @brief A tensor-sink callback for sink handle in a pipeline
 */
static void
test_sink_callback_count (const ml_tensors_data_s * data,
    const ml_tensors_info_s * info, void *pdata)
{
  guint *count = (guint *) pdata;

  *count = *count + 1;
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
  int status = ml_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_dm01,
      &sinkhandle, file2);

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

  /* pipeline with appsink */
  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! appsink name=sinkx");

  count_sink = (guint *) g_malloc (sizeof (guint));
  *count_sink = 0;

  status = ml_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_count, &sinkhandle, count_sink);
  EXPECT_EQ (status, ML_ERROR_NONE);

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

  g_free (pipeline);
  g_free (count_sink);
}

/**
 * @brief Test NNStreamer pipeline sink
 * @detail Failure case to register callback with invalid param.
 */
TEST (nnstreamer_capi_sink, failure_01)
{
  ml_pipeline_h handle;
  ml_pipeline_sink_h sinkhandle;
  gchar *pipeline;
  int status;
  guint *count_sink;

  pipeline = g_strdup ("videotestsrc num-buffers=3 ! videoconvert ! valve name=valvex ! tensor_converter ! tensor_sink name=sinkx");

  count_sink = (guint *) g_malloc (sizeof (guint));
  *count_sink = 0;

  status = ml_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : pipe */
  status = ml_pipeline_sink_register (NULL, "sinkx", test_sink_callback_count, &sinkhandle, count_sink);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : name */
  status = ml_pipeline_sink_register (handle, NULL, test_sink_callback_count, &sinkhandle, count_sink);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : wrong name */
  status = ml_pipeline_sink_register (handle, "wrongname", test_sink_callback_count, &sinkhandle, count_sink);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : invalid type */
  status = ml_pipeline_sink_register (handle, "valvex", test_sink_callback_count, &sinkhandle, count_sink);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : callback */
  status = ml_pipeline_sink_register (handle, "sinkx", NULL, &sinkhandle, count_sink);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : handle */
  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_count, NULL, count_sink);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_count, &sinkhandle, count_sink);
  EXPECT_EQ (status, ML_ERROR_NONE);

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

static char uintarray[10][4];
static char *uia_index[10];

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
      ("appsrc name=srcx ! other/tensor,dimension=4:1:1:1,type=uint8,framerate=0/1 ! filesink location=\"%s\"",
      file1);
  ml_pipeline_h handle;
  ml_pipeline_state_e state;
  ml_pipeline_src_h srchandle;
  int status;
  ml_tensors_info_s tensorsinfo;
  ml_tensors_data_s data1, data2;

  int i;
  char *uintarray2[10];
  uint8_t *content;
  gboolean r;
  gsize len;

  status = ml_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (dir != NULL);
  for (i = 0; i < 10; i++) {
    uia_index[i] = &uintarray[i][0];

    uintarray[i][0] = i;
    uintarray[i][1] = i + 1;
    uintarray[i][2] = i + 3;
    uintarray[i][3] = i + 2;

    uintarray2[i] = (char *) g_malloc (4);
    uintarray2[i][0] = i + 3;
    uintarray2[i][1] = i + 2;
    uintarray2[i][2] = i + 1;
    uintarray2[i][3] = i;
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

  status = ml_pipeline_src_get_handle (handle, "srcx", &tensorsinfo, &srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_EQ (tensorsinfo.num_tensors, 1U);
  EXPECT_EQ (tensorsinfo.info[0].type, ML_TENSOR_TYPE_UINT8);
  EXPECT_EQ (tensorsinfo.info[0].dimension[0], 4U);
  EXPECT_EQ (tensorsinfo.info[0].dimension[1], 1U);
  EXPECT_EQ (tensorsinfo.info[0].dimension[2], 1U);
  EXPECT_EQ (tensorsinfo.info[0].dimension[3], 1U);

  tensorsinfo.num_tensors = 1;
  tensorsinfo.info[0].type = ML_TENSOR_TYPE_UINT8;
  tensorsinfo.info[0].dimension[0] = 4;
  tensorsinfo.info[0].dimension[1] = 1;
  tensorsinfo.info[0].dimension[2] = 1;
  tensorsinfo.info[0].dimension[3] = 1;

  data1.num_tensors = 1;
  data1.tensors[0].tensor = uia_index[0];
  data1.tensors[0].size = 4;

  status = ml_pipeline_src_input_data (srchandle, &data1, ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_input_data (srchandle, &data1, ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_put_handle (srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_get_handle (handle, "srcx", &tensorsinfo, &srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_EQ (tensorsinfo.num_tensors, 1U);
  EXPECT_EQ (tensorsinfo.info[0].type, ML_TENSOR_TYPE_UINT8);
  EXPECT_EQ (tensorsinfo.info[0].dimension[0], 4U);
  EXPECT_EQ (tensorsinfo.info[0].dimension[1], 1U);
  EXPECT_EQ (tensorsinfo.info[0].dimension[2], 1U);
  EXPECT_EQ (tensorsinfo.info[0].dimension[3], 1U);

  for (i = 0; i < 10; i++) {
    data1.num_tensors = 1;
    data1.tensors[0].tensor = uia_index[i];
    data1.tensors[0].size = 4;
    status = ml_pipeline_src_input_data (srchandle, &data1, ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
    EXPECT_EQ (status, ML_ERROR_NONE);

    data2.num_tensors = 1;
    data2.tensors[0].tensor = uintarray2[i];
    data2.tensors[0].size = 4;
    status = ml_pipeline_src_input_data (srchandle, &data2, ML_PIPELINE_BUF_POLICY_AUTO_FREE);
    EXPECT_EQ (status, ML_ERROR_NONE);
  }

  status = ml_pipeline_src_put_handle (srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (50000); /* Wait for the pipeline to flush all */

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);

  r = g_file_get_contents (file1, (gchar **) &content, &len, NULL);
  EXPECT_EQ (r, TRUE);

  EXPECT_EQ (len, 8U * 11);

  for (i = 0; i < 10; i++) {
    EXPECT_EQ (content[i * 8 + 0 + 8], i);
    EXPECT_EQ (content[i * 8 + 1 + 8], i + 1);
    EXPECT_EQ (content[i * 8 + 2 + 8], i + 3);
    EXPECT_EQ (content[i * 8 + 3 + 8], i + 2);
    EXPECT_EQ (content[i * 8 + 4 + 8], i + 3);
    EXPECT_EQ (content[i * 8 + 5 + 8], i + 2);
    EXPECT_EQ (content[i * 8 + 6 + 8], i + 1);
    EXPECT_EQ (content[i * 8 + 7 + 8], i);
  }

  g_free (content);
}

/**
 * @brief Test NNStreamer pipeline src
 * @detail Failure case when pipeline is NULL.
 */
TEST (nnstreamer_capi_src, failure_01)
{
  int status;
  ml_tensors_info_s tensorsinfo;
  ml_pipeline_src_h srchandle;

  status = ml_pipeline_src_get_handle (NULL, "dummy", &tensorsinfo, &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test NNStreamer pipeline src
 * @detail Failure case when the name of source node is wrong.
 */
TEST (nnstreamer_capi_src, failure_02)
{
  const char *pipeline = "appsrc is-live=true name=mysource ! valve name=valvex ! filesink";
  ml_pipeline_h handle;
  ml_tensors_info_s tensorsinfo;
  ml_pipeline_src_h srchandle;

  int status = ml_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* invalid param : pipe */
  status = ml_pipeline_src_get_handle (NULL, "mysource", &tensorsinfo, &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : name */
  status = ml_pipeline_src_get_handle (handle, NULL, &tensorsinfo, &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : wrong name */
  status = ml_pipeline_src_get_handle (handle, "wrongname", &tensorsinfo, &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : invalid type */
  status = ml_pipeline_src_get_handle (handle, "valvex", &tensorsinfo, &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : info */
  status = ml_pipeline_src_get_handle (handle, "mysource", NULL, &srchandle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid param : handle */
  status = ml_pipeline_src_get_handle (handle, "mysource", &tensorsinfo, NULL);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline src
 * @detail Failure case when the number of tensors is 0 or bigger than ML_TENSOR_SIZE_LIMIT;
 */
TEST (nnstreamer_capi_src, failure_03)
{
  const int num_tensors = ML_TENSOR_SIZE_LIMIT + 1;
  const int num_dims = 4;

  const char *pipeline = "appsrc name=srcx ! other/tensor,dimension=4:1:1:1,type=uint8,framerate=0/1 ! tensor_sink";
  ml_pipeline_h handle;
  ml_tensors_info_s tensorsinfo;
  ml_pipeline_src_h srchandle;
  ml_tensors_data_s data;

  for (int i = 0; i < ML_TENSOR_SIZE_LIMIT; ++i) {
    data.tensors[i].tensor = g_malloc0 (sizeof (char) * num_dims);
    data.tensors[i].size = num_dims;
  }

  int status = ml_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_src_get_handle (handle, "srcx", &tensorsinfo, &srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* null data */
  status = ml_pipeline_src_input_data (srchandle, NULL, ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid number of tensors (max size) */
  data.num_tensors = num_tensors;
  status = ml_pipeline_src_input_data (srchandle, &data, ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid number of tensors (size is 0) */
  data.num_tensors = 0;
  status = ml_pipeline_src_input_data (srchandle, &data, ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  status = ml_pipeline_src_put_handle (srchandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  for (int i = 0; i < ML_TENSOR_SIZE_LIMIT; ++i)
    g_free (data.tensors[i].tensor);
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
  gchar *pipeline;
  int status;
  guint *count_sink;
  gchar **node_list = NULL;

  pipeline = g_strdup ("input-selector name=ins ! tensor_converter ! tensor_sink name=sinkx "
      "videotestsrc is-live=true ! videoconvert ! ins.sink_0 "
      "videotestsrc num-buffers=3 ! videoconvert ! ins.sink_1");

  count_sink = (guint *) g_malloc (sizeof (guint));
  *count_sink = 0;

  status = ml_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_switch_get_handle (handle, "ins", &type, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_PIPELINE_SWITCH_INPUT_SELECTOR);

  status = ml_pipeline_switch_nodelist (switchhandle, &node_list);
  EXPECT_EQ (status, ML_ERROR_NONE);

  if (node_list) {
    gchar *name = NULL;
    guint idx = 0;

    while ((name = node_list[idx]) != NULL) {
      EXPECT_TRUE (g_str_equal (name, "sink_0") || g_str_equal (name, "sink_1"));
      idx++;
    }

    EXPECT_EQ (idx, 2U);
  }

  status = ml_pipeline_sink_register (handle, "sinkx", test_sink_callback_count, &sinkhandle, count_sink);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_switch_select (switchhandle, "sink_1");
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_usleep (300000); /* 300ms. Let a few frames flow. */

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_unregister (sinkhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_switch_put_handle (switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_EQ (*count_sink, 3U);

  g_free (pipeline);
  g_free (count_sink);
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
  *count_sink0 = 0;

  count_sink1 = (guint *) g_malloc (sizeof (guint));
  *count_sink1 = 0;

  status = ml_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_switch_get_handle (handle, "outs", &type, &switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_PIPELINE_SWITCH_OUTPUT_SELECTOR);

  status = ml_pipeline_switch_nodelist (switchhandle, &node_list);
  EXPECT_EQ (status, ML_ERROR_NONE);

  if (node_list) {
    gchar *name = NULL;
    guint idx = 0;

    while ((name = node_list[idx]) != NULL) {
      EXPECT_TRUE (g_str_equal (name, "src_0") || g_str_equal (name, "src_1"));
      idx++;
    }

    EXPECT_EQ (idx, 2U);
  }

  status = ml_pipeline_sink_register (handle, "sink0", test_sink_callback_count, &sinkhandle0, count_sink0);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (handle, "sink1", test_sink_callback_count, &sinkhandle1, count_sink1);
  EXPECT_EQ (status, ML_ERROR_NONE);

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

  status = ml_pipeline_switch_put_handle (switchhandle);
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
TEST (nnstreamer_capi_switch, failure_01)
{
  ml_pipeline_h handle;
  ml_pipeline_switch_h switchhandle;
  ml_pipeline_switch_e type;
  gchar *pipeline;
  int status;

  pipeline = g_strdup ("input-selector name=ins ! tensor_converter ! tensor_sink name=sinkx "
      "videotestsrc is-live=true ! videoconvert ! ins.sink_0 "
      "videotestsrc num-buffers=3 ! videoconvert ! ins.sink_1");

  status = ml_pipeline_construct (pipeline, &handle);
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

  status = ml_pipeline_switch_put_handle (switchhandle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 */
TEST (nnstreamer_capi_singleshot, invoke_01)
{
  ml_single_h single;
  ml_tensors_info_s in_info, out_info;
  ml_tensors_info_s in_res, out_res;
  ml_tensors_data_s *input, *output1, *output2;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  ml_util_initialize_tensors_info (&in_info);
  ml_util_initialize_tensors_info (&out_info);
  ml_util_initialize_tensors_info (&in_res);
  ml_util_initialize_tensors_info (&out_res);

  ASSERT_TRUE (root_path != NULL);
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);

  in_info.num_tensors = 1;
  in_info.info[0].type = ML_TENSOR_TYPE_UINT8;
  in_info.info[0].dimension[0] = 3;
  in_info.info[0].dimension[1] = 224;
  in_info.info[0].dimension[2] = 224;
  in_info.info[0].dimension[3] = 1;

  out_info.num_tensors = 1;
  out_info.info[0].type = ML_TENSOR_TYPE_UINT8;
  out_info.info[0].dimension[0] = 1001;
  out_info.info[0].dimension[1] = 1;
  out_info.info[0].dimension[2] = 1;
  out_info.info[0].dimension[3] = 1;

  status = ml_single_open (&single, test_model, &in_info, &out_info,
      ML_NNFW_TENSORFLOW_LITE, ML_NNFW_HW_DO_NOT_CARE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* input tensor in filter */
  status = ml_single_get_input_info (single, &in_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_TRUE (in_info.num_tensors == in_res.num_tensors);
  for (guint idx = 0; idx < in_res.num_tensors; idx++) {
    EXPECT_TRUE (in_info.info[idx].type == in_res.info[idx].type);
    EXPECT_TRUE (in_info.info[idx].dimension[0] == in_res.info[idx].dimension[0]);
    EXPECT_TRUE (in_info.info[idx].dimension[1] == in_res.info[idx].dimension[1]);
    EXPECT_TRUE (in_info.info[idx].dimension[2] == in_res.info[idx].dimension[2]);
    EXPECT_TRUE (in_info.info[idx].dimension[3] == in_res.info[idx].dimension[3]);
  }

  /* output tensor in filter */
  status = ml_single_get_output_info (single, &out_res);
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_TRUE (out_info.num_tensors == out_res.num_tensors);
  for (guint idx = 0; idx < out_res.num_tensors; idx++) {
    EXPECT_TRUE (out_info.info[idx].type == out_res.info[idx].type);
    EXPECT_TRUE (out_info.info[idx].dimension[0] == out_res.info[idx].dimension[0]);
    EXPECT_TRUE (out_info.info[idx].dimension[1] == out_res.info[idx].dimension[1]);
    EXPECT_TRUE (out_info.info[idx].dimension[2] == out_res.info[idx].dimension[2]);
    EXPECT_TRUE (out_info.info[idx].dimension[3] == out_res.info[idx].dimension[3]);
  }

  /* generate dummy data */
  input = ml_util_allocate_tensors_data (&in_info);
  EXPECT_TRUE (input != NULL);

  status = ml_util_get_last_error ();
  EXPECT_EQ (status, ML_ERROR_NONE);

  output1 = ml_single_inference (single, input, NULL);
  EXPECT_TRUE (output1 != NULL);

  status = ml_util_get_last_error ();
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_util_free_tensors_data (&output1);

  output2 = ml_util_allocate_tensors_data (&out_info);
  EXPECT_TRUE (output2 != NULL);

  status = ml_util_get_last_error ();
  EXPECT_EQ (status, ML_ERROR_NONE);

  output1 = ml_single_inference (single, input, output2);
  EXPECT_TRUE (output1 != NULL);
  EXPECT_TRUE (output1 == output2);

  status = ml_util_get_last_error ();
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_util_free_tensors_data (&output2);
  ml_util_free_tensors_data (&input);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
  ml_util_free_tensors_info (&in_res);
  ml_util_free_tensors_info (&out_res);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Start pipeline without tensor info
 */
TEST (nnstreamer_capi_singleshot, invoke_02)
{
  ml_single_h single;
  ml_tensors_info_s in_info, out_info;
  ml_tensors_data_s *input, *output1, *output2;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  ml_util_initialize_tensors_info (&in_info);
  ml_util_initialize_tensors_info (&out_info);

  ASSERT_TRUE (root_path != NULL);
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);

  in_info.num_tensors = 1;
  in_info.info[0].type = ML_TENSOR_TYPE_UINT8;
  in_info.info[0].dimension[0] = 3;
  in_info.info[0].dimension[1] = 224;
  in_info.info[0].dimension[2] = 224;
  in_info.info[0].dimension[3] = 1;

  out_info.num_tensors = 1;
  out_info.info[0].type = ML_TENSOR_TYPE_UINT8;
  out_info.info[0].dimension[0] = 1001;
  out_info.info[0].dimension[1] = 1;
  out_info.info[0].dimension[2] = 1;
  out_info.info[0].dimension[3] = 1;

  status = ml_single_open (&single, test_model, NULL, NULL,
      ML_NNFW_TENSORFLOW_LITE, ML_NNFW_HW_DO_NOT_CARE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* generate dummy data */
  input = ml_util_allocate_tensors_data (&in_info);
  EXPECT_TRUE (input != NULL);

  status = ml_util_get_last_error ();
  EXPECT_EQ (status, ML_ERROR_NONE);

  output1 = ml_single_inference (single, input, NULL);
  EXPECT_TRUE (output1 != NULL);

  status = ml_util_get_last_error ();
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_util_free_tensors_data (&output1);

  output2 = ml_util_allocate_tensors_data (&out_info);
  EXPECT_TRUE (output2 != NULL);

  status = ml_util_get_last_error ();
  EXPECT_EQ (status, ML_ERROR_NONE);

  output1 = ml_single_inference (single, input, output2);
  EXPECT_TRUE (output1 != NULL);
  EXPECT_TRUE (output1 == output2);

  status = ml_util_get_last_error ();
  EXPECT_EQ (status, ML_ERROR_NONE);

  ml_util_free_tensors_data (&output2);
  ml_util_free_tensors_data (&input);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (custom filter)
 * @detail Run pipeline with custom filter, handle multi tensors.
 */
TEST (nnstreamer_capi_singleshot, invoke_03)
{
  ml_single_h single;
  ml_tensors_info_s in_info, out_info;
  ml_tensors_data_s *input, *output1, *output2;
  int i, status;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  ml_util_initialize_tensors_info (&in_info);
  ml_util_initialize_tensors_info (&out_info);

  ASSERT_TRUE (root_path != NULL);
  test_model = g_build_filename (root_path, "build", "nnstreamer_example", "custom_example_passthrough",
      "libnnstreamer_customfilter_passthrough_variable.so", NULL);

  in_info.num_tensors = 2;
  in_info.info[0].type = ML_TENSOR_TYPE_INT16;
  in_info.info[0].dimension[0] = 10;
  in_info.info[0].dimension[1] = 1;
  in_info.info[0].dimension[2] = 1;
  in_info.info[0].dimension[3] = 1;
  in_info.info[1].type = ML_TENSOR_TYPE_FLOAT32;
  in_info.info[1].dimension[0] = 10;
  in_info.info[1].dimension[1] = 1;
  in_info.info[1].dimension[2] = 1;
  in_info.info[1].dimension[3] = 1;

  ml_util_copy_tensors_info (&out_info, &in_info);

  status = ml_single_open (&single, test_model, &in_info, &out_info,
      ML_NNFW_CUSTOM_FILTER, ML_NNFW_HW_DO_NOT_CARE);
  EXPECT_EQ (status, ML_ERROR_NONE);

  /* generate input data */
  input = ml_util_allocate_tensors_data (&in_info);
  ASSERT_TRUE (input != NULL);
  EXPECT_TRUE (input->num_tensors == 2U);

  for (i = 0; i < 10; i++) {
    int16_t i16 = (int16_t) (i + 1);
    float f32 = (float) (i + .1);

    ((int16_t *) input->tensors[0].tensor)[i] = i16;
    ((float *) input->tensors[1].tensor)[i] = f32;
  }

  status = ml_util_get_last_error ();
  EXPECT_EQ (status, ML_ERROR_NONE);

  output1 = ml_single_inference (single, input, NULL);
  EXPECT_TRUE (output1 != NULL);

  status = ml_util_get_last_error ();
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_TRUE (output1->tensors[0].size == ml_util_get_tensor_size (&in_info.info[0]));
  EXPECT_TRUE (output1->tensors[1].size == ml_util_get_tensor_size (&in_info.info[1]));

  for (i = 0; i < 10; i++) {
    int16_t i16 = (int16_t) (i + 1);
    float f32 = (float) (i + .1);

    EXPECT_EQ (((int16_t *) output1->tensors[0].tensor)[i], i16);
    EXPECT_FLOAT_EQ (((float *) output1->tensors[1].tensor)[i], f32);
  }

  ml_util_free_tensors_data (&output1);

  output2 = ml_util_allocate_tensors_data (&out_info);
  EXPECT_TRUE (output2 != NULL);

  status = ml_util_get_last_error ();
  EXPECT_EQ (status, ML_ERROR_NONE);

  output1 = ml_single_inference (single, input, output2);
  EXPECT_TRUE (output1 != NULL);
  EXPECT_TRUE (output1 == output2);

  status = ml_util_get_last_error ();
  EXPECT_EQ (status, ML_ERROR_NONE);

  EXPECT_TRUE (output1->tensors[0].size == ml_util_get_tensor_size (&in_info.info[0]));
  EXPECT_TRUE (output1->tensors[1].size == ml_util_get_tensor_size (&in_info.info[1]));

  for (i = 0; i < 10; i++) {
    int16_t i16 = (int16_t) (i + 1);
    float f32 = (float) (i + .1);

    EXPECT_EQ (((int16_t *) output1->tensors[0].tensor)[i], i16);
    EXPECT_FLOAT_EQ (((float *) output1->tensors[1].tensor)[i], f32);
  }

  ml_util_free_tensors_data (&output2);
  ml_util_free_tensors_data (&input);

  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (test_model);
}

/**
 * @brief Test NNStreamer single shot (tensorflow-lite)
 * @detail Failure case with invalid param.
 */
TEST (nnstreamer_capi_singleshot, failure_01)
{
  ml_single_h single;
  ml_tensors_info_s in_info, out_info;
  int status;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  ml_util_initialize_tensors_info (&in_info);
  ml_util_initialize_tensors_info (&out_info);

  ASSERT_TRUE (root_path != NULL);
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);

  /* invalid file path */
  status = ml_single_open (&single, "wrong_file_name", &in_info, &out_info,
      ML_NNFW_TENSORFLOW_LITE, ML_NNFW_HW_DO_NOT_CARE);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* null file path */
  status = ml_single_open (&single, NULL, &in_info, &out_info,
      ML_NNFW_TENSORFLOW_LITE, ML_NNFW_HW_DO_NOT_CARE);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid handle */
  status = ml_single_open (NULL, test_model, &in_info, &out_info,
      ML_NNFW_TENSORFLOW_LITE, ML_NNFW_HW_DO_NOT_CARE);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  /* invalid input tensor info */
  status = ml_single_open (&single, test_model, &in_info, &out_info,
      ML_NNFW_TENSORFLOW_LITE, ML_NNFW_HW_DO_NOT_CARE);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  in_info.num_tensors = 1;
  in_info.info[0].type = ML_TENSOR_TYPE_UINT8;
  in_info.info[0].dimension[0] = 3;
  in_info.info[0].dimension[1] = 224;
  in_info.info[0].dimension[2] = 224;
  in_info.info[0].dimension[3] = 1;

  /* invalid output tensor info */
  status = ml_single_open (&single, test_model, &in_info, &out_info,
      ML_NNFW_TENSORFLOW_LITE, ML_NNFW_HW_DO_NOT_CARE);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  out_info.num_tensors = 1;
  out_info.info[0].type = ML_TENSOR_TYPE_UINT8;
  out_info.info[0].dimension[0] = 1001;
  out_info.info[0].dimension[1] = 1;
  out_info.info[0].dimension[2] = 1;
  out_info.info[0].dimension[3] = 1;

  /* unknown fw type */
  status = ml_single_open (&single, test_model, &in_info, &out_info,
      ML_NNFW_UNKNOWN, ML_NNFW_HW_DO_NOT_CARE);
  EXPECT_EQ (status, ML_ERROR_NOT_SUPPORTED);

  /* invalid handle */
  status = ml_single_close (single);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (test_model);
}

/**
 * @brief Main gtest
 */
int
main (int argc, char **argv)
{
  testing::InitGoogleTest (&argc, argv);

  return RUN_ALL_TESTS ();
}
