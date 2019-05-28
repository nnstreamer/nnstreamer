/**
 * @file        unittest_tizen_capi.cpp
 * @date        13 Mar 2019
 * @brief       Unit test for Tizen CAPI of NNStreamer. Basis of TCT in the future.
 * @see         https://github.com/nnsuite/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <nnstreamer.h>
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
  EXPECT_EQ (status, NNS_ERROR_STREAMS_PIPE);
}

/**
 * @brief Test NNStreamer pipeline construct with erroneous pipeline
 */
TEST (nnstreamer_capi_construct_destruct, failed_02)
{
  const char *pipeline = "videotestsrc num_buffers=2 ! audioconvert ! fakesink";
  nns_pipeline_h handle;
  int status = nns_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, NNS_ERROR_STREAMS_PIPE);
}

/**
 * @brief Test NNStreamer pipeline construct & destruct
 */
TEST (nnstreamer_capi_playstop, dummy_01)
{
  const char *pipeline = "videotestsrc is-live=true num-buffers=30 ! videoconvert ! videoscale ! video/x-raw,format=RGBx,width=224,height=224,framerate=60/1 ! tensor_converter ! valve name=valvex ! valve name=valvey ! input-selector name=is01 ! tensor_sink name=sinkx";
  nns_pipeline_h handle;
  nns_pipeline_state_e state;
  int status = nns_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_start (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  status = nns_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, NNS_PIPELINE_UNKNOWN);
  EXPECT_NE (state, NNS_PIPELINE_NULL);

  g_usleep (50000); /* 50ms. Let a few frames flow. */
  status = nns_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  EXPECT_EQ (state, NNS_PIPELINE_PLAYING);

  status = nns_pipeline_stop (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  g_usleep (50000); /* 50ms. Let a few frames flow. */

  status = nns_pipeline_get_state (handle, &state);
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
  nns_pipeline_state_e state;
  int status = nns_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_start (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  status = nns_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, NNS_PIPELINE_UNKNOWN);
  EXPECT_NE (state, NNS_PIPELINE_NULL);

  g_usleep (50000); /* 50ms. Let a few frames flow. */
  status = nns_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  EXPECT_EQ (state, NNS_PIPELINE_PLAYING);

  status = nns_pipeline_stop (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  g_usleep (50000); /* 50ms. Let a few frames flow. */

  status = nns_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  EXPECT_EQ (state, NNS_PIPELINE_PAUSED);

  /* Resume playing */
  status = nns_pipeline_start (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  EXPECT_NE (state, NNS_PIPELINE_UNKNOWN);
  EXPECT_NE (state, NNS_PIPELINE_NULL);

  g_usleep (50000); /* 50ms. Enough to empty the queue */
  status = nns_pipeline_stop (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_get_state (handle, &state);
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

  nns_pipeline_h handle;
  nns_pipeline_state_e state;
  nns_valve_h valve1;

  int status = nns_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  EXPECT_TRUE (dir != NULL);

  status = nns_pipeline_valve_get_handle (handle, "valve1", &valve1);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_start (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_valve_control (valve1, 1); /* close */
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, NNS_PIPELINE_UNKNOWN);
  EXPECT_NE (state, NNS_PIPELINE_NULL);

  g_usleep (100000); /* 100ms. Let a few frames flow. */
  status = nns_pipeline_stop (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = g_lstat (file1, &buf);
  EXPECT_EQ (status, 0);
  EXPECT_EQ (buf.st_size, 0);

  status = nns_pipeline_start (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_valve_control (valve1, 0); /* open */
  EXPECT_EQ (status, NNS_ERROR_NONE);
  status = nns_pipeline_valve_put_handle (valve1); /* release valve handle */
  EXPECT_EQ (status, NNS_ERROR_NONE);

  g_usleep (50000); /* 50ms. Let a few frames flow. */

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
 * @brief A tensor-sink callback for sink handle in a pipeline
 */
static void
nns_sink_callback_dm01 (const char *buf[], const size_t size[],
    const nns_tensors_info_s * tensorsinfo, void *pdata)
{
  gchar *filepath = (gchar *) pdata;
  FILE *fp = g_fopen (filepath, "a");
  if (fp == NULL)
    return;

  int i, num = tensorsinfo->num_tensors;

  for (i = 0; i < num; i++) {
    fwrite (buf[i], size[i], 1, fp);
  }

  fclose (fp);
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
  nns_pipeline_h handle;
  nns_pipeline_state_e state;
  nns_sink_h sinkhandle;
  int status = nns_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_sink_register (handle, "sinkx", nns_sink_callback_dm01,
      &sinkhandle, file2);

  status = nns_pipeline_start (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  status = nns_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, NNS_PIPELINE_UNKNOWN);
  EXPECT_NE (state, NNS_PIPELINE_NULL);

  g_usleep (100000); /* 100ms. Let a few frames flow. */
  status = nns_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  EXPECT_EQ (state, NNS_PIPELINE_PLAYING);

  status = nns_pipeline_stop (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */

  status = nns_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  EXPECT_EQ (state, NNS_PIPELINE_PAUSED);

  status = nns_pipeline_sink_unregister (sinkhandle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_destroy (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  g_free (pipeline);

  /* File Comparison to check the integrity */
  EXPECT_EQ (file_cmp (file1, file2), 0);
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
  nns_pipeline_h handle;
  nns_pipeline_state_e state;
  nns_src_h srchandle;
  int status = nns_pipeline_construct (pipeline, &handle);
  nns_tensors_info_s tensorsinfo;

  int i;
  char *uintarray2[10];
  uint8_t *content;
  gboolean r;
  gsize len;
  const size_t size[1] = { 4 };

  EXPECT_EQ (status, NNS_ERROR_NONE);
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
    /* These will be free'ed by gstreamer (NNS_BUF_FREE_BY_NNSTREAMER) */
    /** @todo Check whether gstreamer really deallocates this */
  }

  status = nns_pipeline_start (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  status = nns_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, NNS_ERROR_NONE); /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, NNS_PIPELINE_UNKNOWN);
  EXPECT_NE (state, NNS_PIPELINE_NULL);

  status = nns_pipeline_src_get_handle (handle, "srcx", &tensorsinfo, &srchandle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  EXPECT_EQ (tensorsinfo.num_tensors, 1);
  EXPECT_EQ (tensorsinfo.info[0].type, NNS_UINT8);
  EXPECT_EQ (tensorsinfo.info[0].dimension[0], 4);
  EXPECT_EQ (tensorsinfo.info[0].dimension[1], 1);
  EXPECT_EQ (tensorsinfo.info[0].dimension[2], 1);
  EXPECT_EQ (tensorsinfo.info[0].dimension[3], 1);

  tensorsinfo.num_tensors = 1;
  tensorsinfo.info[0].type = NNS_UINT8;
  tensorsinfo.info[0].dimension[0] = 4;
  tensorsinfo.info[0].dimension[1] = 1;
  tensorsinfo.info[0].dimension[2] = 1;
  tensorsinfo.info[0].dimension[3] = 1;

  status = nns_pipeline_src_input_data (srchandle, NNS_BUF_DO_NOT_FREE1,
      &(uia_index[0]), size, 1);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  status = nns_pipeline_src_input_data (srchandle, NNS_BUF_DO_NOT_FREE1,
      &(uia_index[0]), size, 1);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_src_put_handle (srchandle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
  status = nns_pipeline_src_get_handle (handle, "srcx", &tensorsinfo, &srchandle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  EXPECT_EQ (tensorsinfo.num_tensors, 1);
  EXPECT_EQ (tensorsinfo.info[0].type, NNS_UINT8);
  EXPECT_EQ (tensorsinfo.info[0].dimension[0], 4);
  EXPECT_EQ (tensorsinfo.info[0].dimension[1], 1);
  EXPECT_EQ (tensorsinfo.info[0].dimension[2], 1);
  EXPECT_EQ (tensorsinfo.info[0].dimension[3], 1);


  for (i = 0; i < 10; i++) {
    status = nns_pipeline_src_input_data (srchandle, NNS_BUF_DO_NOT_FREE1,
        &(uia_index[i]), size, 1);
    EXPECT_EQ (status, NNS_ERROR_NONE);
    status = nns_pipeline_src_input_data (srchandle, NNS_BUF_FREE_BY_NNSTREAMER,
        &(uintarray2[i]), size, 1);
    EXPECT_EQ (status, NNS_ERROR_NONE);
  }

  status = nns_pipeline_src_put_handle (srchandle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  g_usleep (50000); /* Wait for the pipeline to flush all */

  status = nns_pipeline_destroy (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  g_free (pipeline);

  r = g_file_get_contents (file1, (gchar **) &content, &len, NULL);
  EXPECT_EQ (r, TRUE);

  EXPECT_EQ (len, 8 * 11);

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
  nns_tensors_info_s tensorsinfo;
  nns_src_h srchandle;

  status = nns_pipeline_src_get_handle (NULL, "dummy", &tensorsinfo, &srchandle);
  EXPECT_EQ (status, NNS_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Test NNStreamer pipeline src
 * @detail Failure case when the name of source node is wrong.
 */
TEST (nnstreamer_capi_src, failure_02)
{
  const char *pipeline = "appsrc is-live=true name=mysource ! filesink";
  nns_pipeline_h handle;
  nns_tensors_info_s tensorsinfo;
  nns_src_h srchandle;

  int status = nns_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_src_get_handle (handle, "wrongname", &tensorsinfo, &srchandle);
  EXPECT_EQ (status, NNS_ERROR_INVALID_PARAMETER);

  status = nns_pipeline_destroy (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);
}

/**
 * @brief Test NNStreamer pipeline src
 * @detail Failure case when the number of tensors is 0 or bigger than NNS_TENSOR_SIZE_LIMIT;
 */
TEST (nnstreamer_capi_src, failure_03)
{
  const int num_tensors = NNS_TENSOR_SIZE_LIMIT + 1;
  const int num_dims = 4;

  const char *pipeline = "appsrc name=srcx ! other/tensor,dimension=4:1:1:1,type=uint8,framerate=0/1 ! tensor_sink";
  nns_pipeline_h handle;
  nns_tensors_info_s tensorsinfo;
  nns_src_h srchandle;
  const size_t tensor_size[1] = { num_dims };
  char *pbuffer[num_tensors];

  for (int i = 0; i < num_tensors; ++i)
    pbuffer[i] = (char *) g_malloc0 (sizeof (char) * num_dims);

  int status = nns_pipeline_construct (pipeline, &handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_start (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_src_get_handle (handle, "srcx", &tensorsinfo, &srchandle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_src_input_data (srchandle, NNS_BUF_DO_NOT_FREE1,
      &(pbuffer[0]), tensor_size, num_tensors);
  EXPECT_EQ (status, NNS_ERROR_INVALID_PARAMETER);

  status = nns_pipeline_src_input_data (srchandle, NNS_BUF_DO_NOT_FREE1,
      &(pbuffer[0]), tensor_size, 0);
  EXPECT_EQ (status, NNS_ERROR_INVALID_PARAMETER);

  status = nns_pipeline_src_put_handle (srchandle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_stop (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  status = nns_pipeline_destroy (handle);
  EXPECT_EQ (status, NNS_ERROR_NONE);

  for (int i = 0; i < num_tensors; ++i)
    g_free (pbuffer[i]);
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
