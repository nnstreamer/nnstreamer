/**
 * @file        unittest_datareposrc.cc
 * @date        21 Apr 2023
 * @brief       Unit test for datareposrc
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Hyunil Park <hyunil46.park@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>
#include <unittest_util.h>

static const gchar filename[] = "mnist.data";
static const gchar json[] = "mnist.json";

/**
 * @brief Get file path
 */
static gchar *
get_file_path (const gchar *filename)
{
  const gchar *root_path = NULL;
  gchar *file_path = NULL;

  root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  /** supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  file_path = g_build_filename (
      root_path, "tests", "test_models", "data", "datarepo", filename, NULL);

  return file_path;
}

/**
 * @brief Bus callback function
 */
static gboolean
bus_callback (GstBus *bus, GstMessage *message, gpointer data)
{
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_EOS:
      g_main_loop_quit ((GMainLoop *) data);
      break;
    default:
      break;
  }

  return TRUE;
}

/**
 * @brief create video test file
 */
static void
create_video_test_file ()
{
  GstBus *bus;
  GMainLoop *loop;

  loop = g_main_loop_new (NULL, FALSE);

  gchar *str_pipeline = g_strdup ("gst-launch-1.0 videotestsrc num-buffers=10 ! "
                                  "datareposink location=video1.raw json=video1.json");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (pipeline);
  g_main_loop_unref (loop);
}

/**
 * @brief create audio test file
 */
static void
create_audio_test_file ()
{
  GstBus *bus;
  GMainLoop *loop;

  loop = g_main_loop_new (NULL, FALSE);

  gchar *str_pipeline = g_strdup (
      "gst-launch-1.0 audiotestsrc samplesperbuffer=44100 num-buffers=1 ! "
      "audio/x-raw, format=S16LE, layout=interleaved, rate=44100, channels=1 ! "
      "datareposink location=audio1.raw json=audio1.json");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (pipeline);
  g_main_loop_unref (loop);
}

/**
 * @brief create image test file
 */
static void
create_image_test_file ()
{
  GstBus *bus;
  GMainLoop *loop;

  loop = g_main_loop_new (NULL, FALSE);

  gchar *str_pipeline = g_strdup ("gst-launch-1.0 videotestsrc num-buffers=5 ! pngenc ! "
                                  "datareposink location=img_%02d.png json=img.json");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (pipeline);
  g_main_loop_unref (loop);
}

/**
 * @brief Test for reading image files
 */
TEST (datareposrc, readImageFiles)
{
  GstBus *bus;
  GMainLoop *loop;

  create_image_test_file ();

  loop = g_main_loop_new (NULL, FALSE);

  gchar *str_pipeline
      = g_strdup ("gst-launch-1.0 datareposrc "
                  "location=img_%02d.png "
                  "json=img.json "
                  "start-sample-index=0 stop-sample-index=4 ! pngdec ! fakesink");
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  gst_object_unref (bus);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (pipeline);
  g_main_loop_unref (loop);
}

/**
 * @brief Test for reading a video raw file
 */
TEST (datareposrc, readVideoRaw)
{
  GstBus *bus;
  GMainLoop *loop;

  create_video_test_file ();

  loop = g_main_loop_new (NULL, FALSE);

  gchar *str_pipeline = g_strdup (
      "gst-launch-1.0 datareposrc "
      "location=video1.raw "
      "json=video1.json ! "
      "video/x-raw, format=RGBx, width=320, height=240, framerate=30/1 ! fakesink");
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  gst_object_unref (bus);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (pipeline);
  g_main_loop_unref (loop);
}

/**
 * @brief Test for reading a video raw file
 */
TEST (datareposrc, readAudioRaw)
{
  GstBus *bus;
  GMainLoop *loop;

  create_audio_test_file ();

  loop = g_main_loop_new (NULL, FALSE);

  gchar *str_pipeline = g_strdup (
      "gst-launch-1.0 datareposrc "
      "location=audio1.raw "
      "json=audio1.json ! "
      "audio/x-raw, format=S16LE, rate=44100, channels=1, layout=interleaved ! fakesink ");
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  gst_object_unref (bus);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (pipeline);
  g_main_loop_unref (loop);
}

/**
 * @brief Test for reading a file with invalid param (JSON path)
 */
TEST (datareposrc, invalidJsonPath0_n)
{
  GstElement *datareposrc = NULL;

  gchar *str_pipeline = g_strdup (
      "gst-launch-1.0 datareposrc name=datareposrc ! "
      "video/x-raw, format=RGBx, width=320, height=240, framerate=30/1 ! fakesink");
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  g_object_set (GST_OBJECT (datareposrc), "location", "video1.raw", NULL);
  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "json", NULL, NULL);

  /* state chagne failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);


  gst_object_unref (pipeline);
}

/**
 * @brief Test for reading a file with invalid param (JSON path)
 */
TEST (datareposrc, invalidJsonPath1_n)
{
  GstElement *datareposrc = NULL;

  gchar *str_pipeline = g_strdup (
      "gst-launch-1.0 datareposrc name=datareposrc ! "
      "video/x-raw, format=RGBx, width=320, height=240, framerate=30/1 ! fakesink");
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  g_object_set (GST_OBJECT (datareposrc), "location", "video1.raw", NULL);
  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "json", "no_search_file", NULL);

  /* state chagne failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  gst_object_unref (pipeline);
}

/**
 * @brief Test for reading a file with invalid param (File path)
 */
TEST (datareposrc, invalidFilePath0_n)
{
  GstElement *datareposrc = NULL;

  gchar *str_pipeline = g_strdup (
      "gst-launch-1.0 datareposrc name=datareposrc ! "
      "video/x-raw, format=RGBx, width=320, height=240, framerate=30/1 ! fakesink");
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  g_object_set (GST_OBJECT (datareposrc), "json", "video1.json", NULL);
  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "location", NULL, NULL);

  /* state chagne failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (pipeline);
}

/**
 * @brief Test for reading a file with invalid param (File path)
 */
TEST (datareposrc, invalidFilePath1_n)
{
  GstElement *datareposrc = NULL;

  gchar *str_pipeline = g_strdup (
      "gst-launch-1.0 datareposrc name=datareposrc ! "
      "video/x-raw, format=RGBx, width=320, height=240, framerate=30/1 ! fakesink");
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  g_object_set (GST_OBJECT (datareposrc), "json", "video1.json", NULL);
  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "location", "no_search_file", NULL);

  /* state chagne failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (pipeline);
}

/**
 * @brief Test for reading a tensors file
 * the number of total sample(mnist.data) is 10 (0~9)
 * the number tensors is 2 and indices (0,1), default is (0,1)
 * the default epochs is 1,
 * the default shuffle is TRUE.
 * can remove start-sample-index, epochs, tensors-sequence, shuffle property.
 */
TEST (datareposrc, readTensors)
{
  GstBus *bus;
  GMainLoop *loop;
  gchar *file_path = NULL;
  gchar *json_path = NULL;
  GstElement *datareposrc = NULL;
  gchar *get_str;
  guint get_value;

  loop = g_main_loop_new (NULL, FALSE);

  file_path = get_file_path (filename);
  json_path = get_file_path (json);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc name=datareposrc location=%s json=%s "
      "start-sample-index=0 stop-sample-index=9 epochs=2 tensors-sequence=0,1 !"
      "other/tensors, format=static, num_tensors=2, framerate=0/1, "
      "dimensions=1:1:784:1.1:1:10:1, types=float32.float32 ! fakesink",
      file_path, json_path);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  EXPECT_NE (datareposrc, nullptr);

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  gst_object_unref (bus);

  g_object_get (datareposrc, "location", &get_str, NULL);
  EXPECT_STREQ (get_str, file_path);

  g_object_get (datareposrc, "json", &get_str, NULL);
  EXPECT_STREQ (get_str, json_path);

  g_object_get (datareposrc, "tensors-sequence", &get_str, NULL);
  EXPECT_STREQ (get_str, "0,1");

  g_object_get (datareposrc, "is-shuffle", &get_value, NULL);
  ASSERT_EQ (get_value, 1U);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  gst_object_unref (pipeline);
  g_main_loop_unref (loop);
  g_free (file_path);
  g_free (json_path);
}

/**
 * @brief Test for reading a file with invalid param (start-sample-index)
 * the number of total sample(mnist.data) is 1000 (0~999)
 */
TEST (datareposrc, invalidStartSampleIndex0_n)
{
  GstElement *datareposrc = NULL;
  int idx_out_of_range = 1000;
  gchar *file_path = NULL;
  gchar *json_path = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc name=datareposrc location=%s json=%s "
      "stop-sample-index=9 epochs=2 tensors-sequence=0,1 !"
      "other/tensors, format=static, num_tensors=2, framerate=0/1, "
      "dimensions=1:1:784:1.1:1:10:1, types=float32.float32 ! fakesink",
      file_path, json_path);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "start-sample-index", idx_out_of_range, NULL);

  /* state chagne failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (pipeline);
}

/**
 * @brief Test for reading a file with invalid param (start-sample-index)
 * the number of total sample(mnist.data) is 10 (0~9)
 */
TEST (datareposrc, invalidStartSampleIndex1_n)
{
  GstElement *datareposrc = NULL;
  gint idx_out_of_range = -1;
  guint get_value;
  gchar *file_path = NULL;
  gchar *json_path = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc name=datareposrc location=%s json=%s "
      "stop-sample-index=9 epochs=2 tensors-sequence=0,1 !"
      "other/tensors, format=static, num_tensors=2, framerate=0/1, "
      "dimensions=1:1:784:1.1:1:10:1, types=float32.float32 ! fakesink",
      file_path, json_path);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "start-sample-index", idx_out_of_range, NULL);
  /** value "-1" of type 'gint' is invalid or out of range for property
     'start-sample-index' of type 'gint' default value is set */
  g_object_get (GST_OBJECT (datareposrc), "start-sample-index", &get_value, NULL);
  EXPECT_EQ (get_value, 0U);

  gst_object_unref (pipeline);
}

/**
 * @brief Test for reading a file with invalid param (stop-sample-index)
 * the number of total sample(mnist.data) is 1000 (0~999)
 */
TEST (datareposrc, invalidStopSampleIndex0_n)
{
  GstElement *datareposrc = NULL;
  guint idx_out_of_range = 1000;
  gchar *file_path = NULL;
  gchar *json_path = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc name=datareposrc location=%s json=%s "
      "start-sample-index=0 epochs=2 tensors-sequence=0,1 !"
      "other/tensors, format=static, num_tensors=2, framerate=0/1, "
      "dimensions=1:1:784:1.1:1:10:1, types=float32.float32 ! fakesink",
      file_path, json_path);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  g_object_set (GST_OBJECT (datareposrc), "stop-sample-index", idx_out_of_range, NULL);

  /* state chagne failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (pipeline);
}

/**
 * @brief Test for reading a file with invalid param (start-sample-index)
 * the number of total sample(mnist.data) is 10 (0~9)
 */
TEST (datareposrc, invalidStopSampleIndex1_n)
{
  GstElement *datareposrc = NULL;
  gint idx_out_of_range = -1;
  guint get_value;
  gchar *file_path = NULL;
  gchar *json_path = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc name=datareposrc location=%s json=%s "
      "start-sample-index=0 epochs=2 tensors-sequence=0,1 !"
      "other/tensors, format=static, num_tensors=2, framerate=0/1, "
      "dimensions=1:1:784:1.1:1:10:1, types=float32.float32 ! fakesink",
      file_path, json_path);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "stop-sample-index", idx_out_of_range, NULL);
  /** value "-1" of type 'gint' is invalid or out of range for property
     'start-sample-index' of type 'gint' default value is set */
  g_object_get (GST_OBJECT (datareposrc), "stop-sample-index", &get_value, NULL);
  EXPECT_EQ (get_value, 0U);

  gst_object_unref (pipeline);
}

/**
 * @brief Test for reading a file with invalid param (epochs)
 */
TEST (datareposrc, invalidEpochs0_n)
{
  GstElement *datareposrc = NULL;
  gint invalid_epochs = -1;
  guint get_value;
  gchar *file_path = NULL;
  gchar *json_path = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc name=datareposrc location=%s json=%s "
      "start-sample-index=0 stop-sample-index=9 tensors-sequence=0,1 ! "
      "other/tensors, format=static, num_tensors=2, framerate=0/1, "
      "dimensions=1:1:784:1.1:1:10:1, types=float32.float32 ! fakesink",
      file_path, json_path);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "epochs", invalid_epochs, NULL);
  /** value "-1" of type 'gint' is invalid or out of range for property
     'start-sample-index' of type 'gint' default value is set */
  g_object_get (GST_OBJECT (datareposrc), "epochs", &get_value, NULL);
  EXPECT_EQ (get_value, 1U);

  gst_object_unref (pipeline);
}

/**
 * @brief Test for reading a file with invalid param (epochs)
 */
TEST (datareposrc, invalidEpochs1_n)
{
  GstElement *datareposrc = NULL;
  guint invalid_epochs = 0;
  gchar *file_path = NULL;
  gchar *json_path = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc name=datareposrc location=%s json=%s "
      "start-sample-index=0 stop-sample-index=9 tensors-sequence=0,1 ! "
      "other/tensors, format=static, num_tensors=2, framerate=0/1, "
      "dimensions=1:1:784:1.1:1:10:1, types=float32.float32 ! fakesink",
      file_path, json_path);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "epochs", invalid_epochs, NULL);

  /* state chagne failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (pipeline);
}

/**
 * @brief Test for reading a file with invalid param (tensors-sequence)
 * the number tensors is 2 and indices (0,1)
 */
TEST (datareposrc, invalidTensorsSequence0_n)
{
  GstElement *datareposrc = NULL;
  gchar *file_path = NULL;
  gchar *json_path = NULL;

  file_path = get_file_path (filename);
  json_path = get_file_path (json);

  gchar *str_pipeline = g_strdup_printf (
      "gst-launch-1.0 datareposrc name=datareposrc location=%s json=%s "
      "start-sample-index=0 stop-sample-index=9 ! "
      "other/tensors, format=static, num_tensors=2, framerate=0/1, "
      "dimensions=1:1:784:1.1:1:10:1, types=float32.float32 ! fakesink",
      file_path, json_path);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (file_path);
  g_free (json_path);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "tensors-sequence", "1,0,2", NULL);

  /* state chagne failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (pipeline);
}

/**
 * @brief remove test file
 */
static void
remove_test_file (void)
{
  gchar *filename = NULL;
  int i;

  g_remove ("audio1.json");
  g_remove ("audio1.raw");
  g_remove ("video1.json");
  g_remove ("video1.raw");
  g_remove ("img.son");

  for (i = 0; i < 5; i++) {
    filename = g_strdup_printf ("img_%02d.png", i);
    g_remove (filename);
    g_free (filename);
  }
}

/**
 * @brief Main GTest
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

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  remove_test_file ();

  return result;
}
