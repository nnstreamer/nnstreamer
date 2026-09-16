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
#include <nnstreamer_plugin_api_util.h>
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
    case GST_MESSAGE_ERROR:
      g_main_loop_quit ((GMainLoop *) data);
      break;
    default:
      break;
  }

  return TRUE;
}

/**
 * @brief Callback for tensor sink signal.
 */
static void
new_data_cb (GstElement *element, GstBuffer *buffer, gint *user_data)
{
  (*user_data)++;
  return;
}

/**
 * @brief create sparse tensors file
 */
static void
create_sparse_tensors_test_file (gint file_index)
{
  GstBus *bus;
  GMainLoop *loop;
  g_autofree gchar *file_path = get_file_path (filename);
  g_autofree gchar *json_path = get_file_path (json);
  g_autofree gchar *str_pipeline = g_strdup_printf (
      "datareposrc location=%s json=%s start-sample-index=0 stop-sample-index=9 ! "
      "tensor_sparse_enc ! other/tensors,format=sparse,framerate=0/1 ! "
      "datareposink location=sparse%d.data json=sparse%d.json",
      file_path, json_path, file_index, file_index);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);
}

/**
 * @brief create flexible tensors file
 */
static void
create_flexible_tensors_test_file (gint fps, gint file_index)
{
  GstBus *bus;
  GMainLoop *loop;
  gint rate_n = fps;
  g_autofree gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=10 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=176,height=144,framerate=%d/1 ! tensor_converter ! join0.sink_0 "
      "videotestsrc num-buffers=10 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=320,height=240,framerate=%d/1 ! tensor_converter ! join0.sink_1 "
      "videotestsrc num-buffers=10 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=640,height=480,framerate=%d/1 ! tensor_converter ! join0.sink_2 "
      "join name=join0 ! other/tensors,format=flexible ! "
      "datareposink location=flexible%d.data json=flexible%d.json",
      rate_n, rate_n, rate_n, file_index, file_index);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);
}

/**
 * @brief create video test file
 */
static void
create_video_test_file ()
{
  GstBus *bus;
  GMainLoop *loop;
  const gchar *str_pipeline = "videotestsrc num-buffers=10 ! "
                              "datareposink location=video1.raw json=video1.json";

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);
}

/**
 * @brief create audio test file
 */
static void
create_audio_test_file (gint file_index)
{
  GstBus *bus;
  GMainLoop *loop;
  g_autofree gchar *str_pipeline = g_strdup_printf (
      "audiotestsrc samplesperbuffer=44100 num-buffers=1 ! "
      "audio/x-raw, format=S16LE, layout=interleaved, rate=44100, channels=1 ! "
      "datareposink location=audio%d.raw json=audio%d.json",
      file_index, file_index);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  loop = g_main_loop_new (NULL, FALSE);
  ASSERT_NE (pipeline, nullptr);

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  g_clear_pointer (&pipeline, gst_object_unref);
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
  const gchar *str_pipeline = "videotestsrc num-buffers=5 ! pngenc ! "
                              "datareposink location=img_%02d.png json=img.json";

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);
}

/**
 * @brief Test for reading image files
 */
TEST (datareposrc, readImageFiles)
{
  gint buffer_count = 0, i;
  GCallback handler = G_CALLBACK (new_data_cb);
  GstElement *tensor_sink;
  GstBus *bus;
  GMainLoop *loop;
  const gchar *str_pipeline
      = "datareposrc location=img_%02d.png json=img.json start-sample-index=0 stop-sample-index=4 !"
        "pngdec ! tensor_converter ! tensor_sink name=tensor_sink0";

  create_image_test_file ();
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  tensor_sink = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_sink0");
  ASSERT_NE (tensor_sink, nullptr);
  g_signal_connect (tensor_sink, "new-data", (GCallback) handler, &buffer_count);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  EXPECT_NE (buffer_count, 0);
  handler = NULL;

  g_clear_pointer (&tensor_sink, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);

  for (i = 0; i < 5; i++) {
    g_autofree gchar *filename = g_strdup_printf ("img_%02d.png", i);
    g_remove (filename);
  }
}

/**
 * @brief Test for reading a video raw file
 */
TEST (datareposrc, readVideoRaw)
{
  gint buffer_count = 0;
  GstElement *tensor_sink;
  GstBus *bus;
  GMainLoop *loop;
  GCallback handler = G_CALLBACK (new_data_cb);
  const gchar *str_pipeline
      = "datareposrc location=video1.raw json=video1.json ! tensor_converter ! tensor_sink name=tensor_sink0";

  create_video_test_file ();
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  tensor_sink = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_sink0");
  ASSERT_NE (tensor_sink, nullptr);
  g_signal_connect (tensor_sink, "new-data", (GCallback) handler, &buffer_count);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  EXPECT_NE (buffer_count, 0);
  handler = NULL;

  g_clear_pointer (&tensor_sink, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);

  g_remove ("video1.json");
  g_remove ("video1.raw");
}

/**
 * @brief Test for reading a video raw file
 */
TEST (datareposrc, readAudioRaw)
{
  gchar *data_1 = NULL, *data_2 = NULL;
  gsize size_1, size_2;
  GstBus *bus;
  GMainLoop *loop;
  gint ret = -1;
  gint file_index = 1;
  const gchar *str_pipeline
      = "datareposrc location=audio1.raw json=audio1.json ! tee name=t "
        "t. ! queue ! datareposink location=result.raw json=result.json "
        "t. ! queue ! tensor_sink";

  create_audio_test_file (file_index);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);

  if (!g_file_get_contents ("aduio1.raw", &data_1, &size_1, NULL)) {
    goto error;
  }

  if (!g_file_get_contents ("result.raw", &data_2, &size_2, NULL)) {
    goto error;
  }
  EXPECT_EQ (size_1, size_2);
  g_clear_pointer (&data_1, g_free);
  g_clear_pointer (&data_2, g_free);

  if (!g_file_get_contents ("audio1.json", &data_1, &size_1, NULL)) {
    goto error;
  }

  if (!g_file_get_contents ("result.json", &data_2, &size_2, NULL)) {
    goto error;
  }
  ret = g_strcmp0 (data_1, data_2);
  EXPECT_EQ (ret, 0);
error:
  g_clear_pointer (&data_1, g_free);
  g_clear_pointer (&data_2, g_free);
  g_remove ("audio1.json");
  g_remove ("audio1.raw");
  g_remove ("result.json");
  g_remove ("result.raw");
}

/**
 * @brief Test for reading a file with invalid param (JSON path)
 */
TEST (datareposrc, invalidJsonPath0_n)
{
  GstElement *datareposrc = NULL;
  const gchar *str_pipeline = "datareposrc name=datareposrc ! fakesink";

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  g_object_set (GST_OBJECT (datareposrc), "location", "video1.raw", NULL);
  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "json", NULL, NULL);

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_clear_pointer (&datareposrc, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
}

/**
 * @brief Test for reading a file with invalid param (JSON path)
 */
TEST (datareposrc, invalidJsonPath1_n)
{
  GstElement *datareposrc = NULL;
  const gchar *str_pipeline = "datareposrc name=datareposrc ! fakesink";

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  g_object_set (GST_OBJECT (datareposrc), "location", "video1.raw", NULL);
  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "json", "no_search_file", NULL);

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_clear_pointer (&datareposrc, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
}

/**
 * @brief Test for reading a file with invalid param (File path)
 */
TEST (datareposrc, invalidFilePath0_n)
{
  GstElement *datareposrc = NULL;
  const gchar *str_pipeline = "datareposrc name=datareposrc ! fakesink";

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  g_object_set (GST_OBJECT (datareposrc), "json", "video1.json", NULL);
  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "location", NULL, NULL);

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_clear_pointer (&datareposrc, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
}

/**
 * @brief Test for reading a file with invalid param (File path)
 */
TEST (datareposrc, invalidFilePath1_n)
{
  GstElement *datareposrc = NULL;
  const gchar *str_pipeline = "datareposrc name=datareposrc ! fakesink";

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  g_object_set (GST_OBJECT (datareposrc), "json", "video1.json", NULL);
  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "location", "no_search_file", NULL);

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_clear_pointer (&datareposrc, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
}

/**
 * @brief Test for reading a file with invalid param (caps)
 */
TEST (datareposrc, invalidCapsWithoutJSON_n)
{
  GstElement *datareposrc = NULL;
  const gchar *str_pipeline = "datareposrc name=datareposrc ! fakesink";

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  g_object_set (GST_OBJECT (datareposrc), "location", "video1.raw", NULL);
  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "caps", NULL, NULL);

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_clear_pointer (&datareposrc, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
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
  g_autofree gchar *file_path = get_file_path (filename);
  g_autofree gchar *json_path = get_file_path (json);
  GstElement *datareposrc = NULL;
  gchar *get_str;
  guint get_value;
  g_autofree gchar *str_pipeline = g_strdup_printf (
      "datareposrc name=datareposrc location=%s json=%s "
      "start-sample-index=0 stop-sample-index=9 epochs=2 tensors-sequence=0,1 ! "
      "fakesink",
      file_path, json_path);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  EXPECT_NE (datareposrc, nullptr);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  g_object_get (datareposrc, "location", &get_str, NULL);
  EXPECT_STREQ (get_str, file_path);
  g_clear_pointer (&get_str, g_free);

  g_object_get (datareposrc, "json", &get_str, NULL);
  EXPECT_STREQ (get_str, json_path);
  g_clear_pointer (&get_str, g_free);

  g_object_get (datareposrc, "tensors-sequence", &get_str, NULL);
  EXPECT_STREQ (get_str, "0,1");
  g_clear_pointer (&get_str, g_free);

  g_object_get (datareposrc, "is-shuffle", &get_value, NULL);
  ASSERT_EQ (get_value, 1U);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_main_loop_run (loop);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_clear_pointer (&datareposrc, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);
}

/**
 * @brief Test for reading a file composed of flexible tensors
 * the default shuffle is TRUE.
 */
TEST (datareposrc, readFlexibleTensors)
{
  gchar *data_1 = NULL, *data_2 = NULL;
  gsize size_1, size_2;
  gint fps = 10, ret = -1;
  GstBus *bus;
  const gchar *str_pipeline = NULL;
  GMainLoop *loop;
  gint file_index = 0;
  str_pipeline = "datareposrc location=flexible0.data json=flexible0.json ! tee name=t "
                 "t. ! queue ! datareposink location=result.data json=result.json "
                 "t. ! queue ! tensor_sink";

  create_flexible_tensors_test_file (fps, file_index);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);

  if (!g_file_get_contents ("flexible.raw", &data_1, &size_1, NULL)) {
    goto error;
  }

  if (!g_file_get_contents ("result.raw", &data_2, &size_2, NULL)) {
    goto error;
  }
  EXPECT_EQ (size_1, size_2);
  g_clear_pointer (&data_1, g_free);
  g_clear_pointer (&data_2, g_free);

  if (!g_file_get_contents ("flexible.json", &data_1, &size_1, NULL)) {
    goto error;
  }

  if (!g_file_get_contents ("result.json", &data_2, &size_2, NULL)) {
    goto error;
  }
  ret = g_strcmp0 (data_1, data_2);
  EXPECT_EQ (ret, 0);
error:
  g_clear_pointer (&data_1, g_free);
  g_clear_pointer (&data_2, g_free);
  g_remove ("flexible0.json");
  g_remove ("flexible0.data");
  g_remove ("result.json");
  g_remove ("result.data");
}


/**
 * @brief Framerate Test for reading a file composed of flexible tensors
 */
TEST (datareposrc, fps30ReadFlexibleTensors)
{
  gint fps = 30;
  gint total_samples = 30;
  guint64 start_time, end_time;
  gdouble elapsed_time, stream_duration;
  GstElement *tensor_sink;
  GstBus *bus;
  GMainLoop *loop;
  gint file_index = 1;
  gint buffer_count = 0, no_sync_count = 0;
  const gchar *str_pipeline
      = "datareposrc location=flexible1.data json=flexible1.json ! queue ! tensor_sink name=tensor_sink0 sync=true";

  create_flexible_tensors_test_file (fps, file_index);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  tensor_sink = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_sink0");
  ASSERT_NE (tensor_sink, nullptr);
  g_signal_connect (tensor_sink, "new-data", G_CALLBACK (new_data_cb), &buffer_count);

  start_time = g_get_monotonic_time ();

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  end_time = g_get_monotonic_time ();
  elapsed_time = (end_time - start_time) / (double) G_USEC_PER_SEC;

  /* join ends the stream only after every source does, so nothing is lost. */
  ASSERT_EQ (buffer_count, total_samples);
  stream_duration = (buffer_count - 1) / (gdouble) fps;

  g_print ("Elapsed time: %.6f second (%d buffers)\n", elapsed_time, buffer_count);
  EXPECT_LT (stream_duration * 0.9, elapsed_time);

  g_clear_pointer (&tensor_sink, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);

  pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  tensor_sink = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_sink0");
  ASSERT_NE (tensor_sink, nullptr);
  g_object_set (GST_OBJECT (tensor_sink), "sync", FALSE, NULL);
  g_signal_connect (tensor_sink, "new-data", G_CALLBACK (new_data_cb), &no_sync_count);

  start_time = g_get_monotonic_time ();

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  end_time = g_get_monotonic_time ();
  elapsed_time = (end_time - start_time) / (double) G_USEC_PER_SEC;

  g_print ("Elapsed time: %.6f second (%d buffers)\n", elapsed_time, no_sync_count);
  EXPECT_EQ (no_sync_count, total_samples);
  /* Without sync the same samples are read without waiting for the clock. */
  EXPECT_LT (elapsed_time, stream_duration * 0.5);

  g_clear_pointer (&tensor_sink, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);

  g_remove ("flexible1.json");
  g_remove ("flexible1.data");
}

/**
 * @brief Test for reading a file composed of sparse tensors
 * the default shuffle is TRUE.
 */
TEST (datareposrc, readSparseTensors)
{
  g_autofree gchar *sparse_data = NULL, *sample_data = NULL;
  gsize size, org_size = 31760;
  gint buffer_count = 0;
  GstElement *tensor_sink;
  GstBus *bus;
  const gchar *str_pipeline = NULL;
  GMainLoop *loop;
  gint file_index = 0;
  GCallback handler = G_CALLBACK (new_data_cb);
  str_pipeline = "datareposrc location=sparse0.data json=sparse0.json ! tensor_sparse_dec ! "
                 "other/tensors, format=static, num_tensors=2, framerate=0/1, "
                 "dimensions=1:1:784:1.1:1:10:1, types=\"float32,float32\" ! tee name= t "
                 "t. ! queue ! filesink location=sample0.data "
                 "t. ! queue ! tensor_sink name=tensor_sink0";

  create_sparse_tensors_test_file (file_index);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  tensor_sink = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_sink0");
  ASSERT_NE (tensor_sink, nullptr);

  g_signal_connect (tensor_sink, "new-data", (GCallback) handler, &buffer_count);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  EXPECT_NE (buffer_count, 0);
  handler = NULL;

  g_clear_pointer (&tensor_sink, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);

  if (!g_file_get_contents ("sparse.data", &sparse_data, &size, NULL)) {
    goto error;
  }
  EXPECT_LT (size, org_size);

  if (!g_file_get_contents ("sample.data", &sample_data, &size, NULL)) {
    goto error;
  }
  EXPECT_EQ (size, org_size);
error:
  g_remove ("sparse0.json");
  g_remove ("sparse0.data");
  g_remove ("sample0.data");
}

/**
 * @brief Test for reading a tensors file with Caps property
 */
TEST (datareposrc, readTensorsNoJSONWithCapsParam)
{
  GstBus *bus;
  GMainLoop *loop;
  GstElement *datareposrc = NULL;
  gchar *get_str;
  guint get_value;
  g_autofree gchar *file_path = get_file_path (filename);
  g_autofree gchar *str_pipeline = g_strdup_printf (
      "datareposrc name=datareposrc location=%s "
      "start-sample-index=0 stop-sample-index=9 epochs=2 tensors-sequence=0,1 "
      "caps =\"other/tensors, format=(string)static, framerate=(fraction)0/1, "
      "num_tensors=(int)2, dimensions=(string)1:1:784:1.1:1:10:1, types=(string)float32.float32\" ! "
      "fakesink",
      file_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  EXPECT_NE (datareposrc, nullptr);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  g_object_get (datareposrc, "location", &get_str, NULL);
  EXPECT_STREQ (get_str, file_path);
  g_clear_pointer (&get_str, g_free);

  g_object_get (datareposrc, "tensors-sequence", &get_str, NULL);
  EXPECT_STREQ (get_str, "0,1");
  g_clear_pointer (&get_str, g_free);

  g_object_get (datareposrc, "is-shuffle", &get_value, NULL);
  ASSERT_EQ (get_value, 1U);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_main_loop_run (loop);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_clear_pointer (&datareposrc, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);
}

/**
 * @brief Test for reading a file with invalid param (start-sample-index)
 * the number of total sample(mnist.data) is 1000 (0~999)
 */
TEST (datareposrc, invalidStartSampleIndex0_n)
{
  GstElement *datareposrc = NULL;
  int idx_out_of_range = 1000;
  g_autofree gchar *file_path = get_file_path (filename);
  g_autofree gchar *json_path = get_file_path (json);
  g_autofree gchar *str_pipeline
      = g_strdup_printf ("datareposrc name=datareposrc location=%s json=%s "
                         "stop-sample-index=9 epochs=2 tensors-sequence=0,1 ! fakesink",
          file_path, json_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);
  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "start-sample-index", idx_out_of_range, NULL);

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_clear_pointer (&datareposrc, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
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
  g_autofree gchar *file_path = get_file_path (filename);
  g_autofree gchar *json_path = get_file_path (json);
  g_autofree gchar *str_pipeline
      = g_strdup_printf ("datareposrc name=datareposrc location=%s json=%s "
                         "stop-sample-index=9 epochs=2 tensors-sequence=0,1 ! fakesink",
          file_path, json_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);
  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "start-sample-index", idx_out_of_range, NULL);
  /** value "-1" of type 'gint' is invalid or out of range for property
     'start-sample-index' of type 'gint' default value is set */
  g_object_get (GST_OBJECT (datareposrc), "start-sample-index", &get_value, NULL);
  EXPECT_EQ (get_value, 0U);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_clear_pointer (&datareposrc, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
}

/**
 * @brief Test for reading a file with invalid param (stop-sample-index)
 * the number of total sample(mnist.data) is 1000 (0~999)
 */
TEST (datareposrc, invalidStopSampleIndex0_n)
{
  GstElement *datareposrc = NULL;
  guint idx_out_of_range = 1000;
  g_autofree gchar *file_path = get_file_path (filename);
  g_autofree gchar *json_path = get_file_path (json);
  g_autofree gchar *str_pipeline
      = g_strdup_printf ("datareposrc name=datareposrc location=%s json=%s "
                         "start-sample-index=0 epochs=2 tensors-sequence=0,1 ! fakesink",
          file_path, json_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);
  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  g_object_set (GST_OBJECT (datareposrc), "stop-sample-index", idx_out_of_range, NULL);

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_clear_pointer (&datareposrc, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
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
  g_autofree gchar *file_path = get_file_path (filename);
  g_autofree gchar *json_path = get_file_path (json);
  g_autofree gchar *str_pipeline
      = g_strdup_printf ("datareposrc name=datareposrc location=%s json=%s "
                         "start-sample-index=0 epochs=2 tensors-sequence=0,1 ! fakesink",
          file_path, json_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);
  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "stop-sample-index", idx_out_of_range, NULL);
  /** value "-1" of type 'gint' is invalid or out of range for property
     'start-sample-index' of type 'gint' default value is set */
  g_object_get (GST_OBJECT (datareposrc), "stop-sample-index", &get_value, NULL);
  EXPECT_EQ (get_value, 0U);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_clear_pointer (&datareposrc, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
}

/**
 * @brief Test for reading a file with invalid param (epochs)
 */
TEST (datareposrc, invalidEpochs0_n)
{
  GstElement *datareposrc = NULL;
  gint invalid_epochs = -1;
  guint get_value;
  g_autofree gchar *file_path = get_file_path (filename);
  g_autofree gchar *json_path = get_file_path (json);
  g_autofree gchar *str_pipeline = g_strdup_printf (
      "datareposrc name=datareposrc location=%s json=%s "
      "start-sample-index=0 stop-sample-index=9 tensors-sequence=0,1 ! fakesink",
      file_path, json_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);
  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "epochs", invalid_epochs, NULL);
  /** value "-1" of type 'gint' is invalid or out of range for property
     'start-sample-index' of type 'gint' default value is set */
  g_object_get (GST_OBJECT (datareposrc), "epochs", &get_value, NULL);
  EXPECT_EQ (get_value, 1U);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_clear_pointer (&datareposrc, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
}

/**
 * @brief Test for reading a file with invalid param (epochs)
 */
TEST (datareposrc, invalidEpochs1_n)
{
  GstElement *datareposrc = NULL;
  guint invalid_epochs = 0;
  g_autofree gchar *file_path = get_file_path (filename);
  g_autofree gchar *json_path = get_file_path (json);
  g_autofree gchar *str_pipeline = g_strdup_printf (
      "datareposrc name=datareposrc location=%s json=%s "
      "start-sample-index=0 stop-sample-index=9 tensors-sequence=0,1 ! fakesink",
      file_path, json_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);
  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "epochs", invalid_epochs, NULL);

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_clear_pointer (&datareposrc, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
}

/**
 * @brief Test for reading a file with invalid param (tensors-sequence)
 * the number tensors is 2 and indices (0,1)
 */
TEST (datareposrc, invalidTensorsSequence0_n)
{
  GstElement *datareposrc = NULL;
  g_autofree gchar *file_path = get_file_path (filename);
  g_autofree gchar *json_path = get_file_path (json);
  g_autofree gchar *str_pipeline
      = g_strdup_printf ("datareposrc name=datareposrc location=%s json=%s "
                         "start-sample-index=0 stop-sample-index=9 ! fakesink",
          file_path, json_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);
  datareposrc = gst_bin_get_by_name (GST_BIN (pipeline), "datareposrc");
  ASSERT_NE (datareposrc, nullptr);

  /* set invalid param */
  g_object_set (GST_OBJECT (datareposrc), "tensors-sequence", "1,0,2", NULL);

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_clear_pointer (&datareposrc, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
}

/**
 * @brief Test for reading a file composed of non-flexible tensors
 * the default shuffle is TRUE.
 */
TEST (datareposrc, readInvalidFlexibleTensors_n)
{
  gint buffer_count = 0;
  gint fps = 10;
  GstBus *bus;
  GMainLoop *loop;
  GCallback handler = G_CALLBACK (new_data_cb);
  const gchar *str_pipeline
      = "datareposrc location=audio2.raw json=flexible2.json ! tensor_sink name=tensor_sink0";
  GstElement *tensor_sink;
  gint file_index = 2;

  create_flexible_tensors_test_file (fps, file_index);
  create_audio_test_file (file_index);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  tensor_sink = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_sink0");
  ASSERT_NE (tensor_sink, nullptr);
  g_signal_connect (tensor_sink, "new-data", (GCallback) handler, &buffer_count);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  /* EXPECT_EQ not checked due to internal data stream error */
  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  /* Internal data stream error */
  EXPECT_EQ (buffer_count, 0);
  handler = NULL;

  g_clear_pointer (&tensor_sink, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);

  g_remove ("audio2.json");
  g_remove ("audio2.raw");
  g_remove ("flexible2.json");
  g_remove ("flexible2.data");
}

/**
 * @brief Test for reading a file composed of non-sparse tensors
 * the default shuffle is TRUE.
 */
TEST (datareposrc, readInvalidSparseTensors_n)
{
  gint buffer_count = 0;
  GstBus *bus;
  GMainLoop *loop;
  GCallback handler = G_CALLBACK (new_data_cb);
  const gchar *str_pipeline
      = "datareposrc location=audio3.raw json=sparse3.json ! tensor_sink name=tensor_sink0";
  GstElement *tensor_sink;
  gint file_index = 3;

  create_sparse_tensors_test_file (file_index);
  create_audio_test_file (file_index);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  tensor_sink = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_sink0");
  ASSERT_NE (tensor_sink, nullptr);
  g_signal_connect (tensor_sink, "new-data", (GCallback) handler, &buffer_count);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  g_clear_pointer (&bus, gst_object_unref);

  /* EXPECT_EQ not checked due to internal data stream error */
  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  /* Internal data stream error */
  EXPECT_EQ (buffer_count, 0);
  handler = NULL;

  g_clear_pointer (&tensor_sink, gst_object_unref);
  g_clear_pointer (&pipeline, gst_object_unref);
  g_main_loop_unref (loop);

  g_remove ("audio3.json");
  g_remove ("audio3.raw");
  g_remove ("sparse3.json");
  g_remove ("sparse3.data");
}

#define REPO_DATA "crafted.data"
#define REPO_JSON "crafted.json"
#define REPO_FLEX_CAPS "other/tensors,format=flexible,framerate=0/1"
#define REPO_PAYLOAD_SIZE 4U
#define REPO_TENSOR_SIZE (128U + REPO_PAYLOAD_SIZE)

/**
 * @brief Outcome of a pipeline run in the crafted-repository tests.
 */
typedef struct {
  GMainLoop *loop;
  GstMessageType type; /**< first EOS or ERROR, GST_MESSAGE_UNKNOWN on timeout */
  gchar *error_src; /**< name of the element that posted the first error */
  GQuark error_domain; /**< domain of the first error */
  gint error_code; /**< code of the first error */
  guint buffers; /**< buffers that reached the sink */
  guint last_mems; /**< memories in the last buffer */
  guint first_sample; /**< sample index of the first buffer (no shuffle) */
  guint tensors; /**< tensors per sample, payload is checked if not zero */
  guint mismatches; /**< memories whose size or payload is unexpected */
  guint json_logs; /**< messages json-glib logged during the run */
} repo_run_s;

/**
 * @brief Payload byte of a tensor in the crafted repository.
 */
static guint8
_repo_payload (guint sample, guint tensor)
{
  return (guint8) (sample * 16 + tensor);
}

/**
 * @brief Write a repository of flexible uint8 tensors, 4 bytes each.
 * @return the size of the data file.
 */
static gsize
_write_flexible_repo (guint num_samples, guint tensors_per_sample)
{
  GstTensorMetaInfo meta;
  gsize size = (gsize) num_samples * tensors_per_sample * REPO_TENSOR_SIZE;
  g_autofree guint8 *data = (guint8 *) g_malloc0 (size);
  guint8 *pos = data;
  guint s, t;

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_UINT8;
  meta.dimension[0] = REPO_PAYLOAD_SIZE;
  meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  EXPECT_EQ (gst_tensor_meta_info_get_header_size (&meta), 128U);

  for (s = 0; s < num_samples; s++) {
    for (t = 0; t < tensors_per_sample; t++) {
      EXPECT_TRUE (gst_tensor_meta_info_update_header (&meta, pos));
      memset (pos + 128, _repo_payload (s, t), REPO_PAYLOAD_SIZE);
      pos += REPO_TENSOR_SIZE;
    }
  }

  EXPECT_TRUE (g_file_set_contents (REPO_DATA, (const gchar *) data, size, NULL));
  return size;
}

/**
 * @brief Build a JSON array body "first, first + step, ..." of @a n entries.
 */
static gchar *
_repo_seq (guint n, guint64 first, guint64 step)
{
  GString *str = g_string_new (NULL);
  guint i;

  for (i = 0; i < n; i++)
    g_string_append_printf (str, "%s%" G_GUINT64_FORMAT, i ? "," : "", first + step * i);

  return g_string_free (str, FALSE);
}

/**
 * @brief Write the JSON of a flexible repository with the given arrays.
 */
static void
_write_flexible_json (const gchar *path, guint total_samples,
    const gchar *sample_offset, const gchar *tensor_size, const gchar *tensor_count)
{
  g_autofree gchar *json = g_strdup_printf ("{\"gst_caps\":\"%s\",\"total_samples\":%u,\"sample_offset\":[%s],"
                                            "\"tensor_size\":[%s],\"tensor_count\":[%s]}",
      REPO_FLEX_CAPS, total_samples, sample_offset, tensor_size, tensor_count);

  EXPECT_TRUE (g_file_set_contents (path, json, -1, NULL));
}

/**
 * @brief Write the JSON of a valid flexible repository.
 */
static void
_write_valid_flexible_json (const gchar *path, guint num_samples, guint tensors_per_sample)
{
  g_autofree gchar *offsets
      = _repo_seq (num_samples, 0, (guint64) tensors_per_sample * REPO_TENSOR_SIZE);
  g_autofree gchar *sizes
      = _repo_seq (num_samples * tensors_per_sample, REPO_TENSOR_SIZE, 0);
  g_autofree gchar *counts = _repo_seq (num_samples, 0, tensors_per_sample);

  _write_flexible_json (path, num_samples, offsets, sizes, counts);
}

/**
 * @brief Bus callback recording the first EOS or ERROR.
 */
static gboolean
_repo_bus_cb (GstBus *bus, GstMessage *message, gpointer user_data)
{
  repo_run_s *run = (repo_run_s *) user_data;
  GError *err = NULL;

  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_ERROR:
      if (run->type == GST_MESSAGE_UNKNOWN) {
        gst_message_parse_error (message, &err, NULL);
        run->type = GST_MESSAGE_ERROR;
        run->error_src = g_strdup (GST_OBJECT_NAME (GST_MESSAGE_SRC (message)));
        run->error_domain = err->domain;
        run->error_code = err->code;
        g_error_free (err);
      }
      g_main_loop_quit (run->loop);
      break;
    case GST_MESSAGE_EOS:
      if (run->type == GST_MESSAGE_UNKNOWN)
        run->type = GST_MESSAGE_EOS;
      g_main_loop_quit (run->loop);
      break;
    default:
      break;
  }

  return TRUE;
}

/**
 * @brief Timeout callback stopping a pipeline run that never ends.
 */
static gboolean
_repo_timeout_cb (gpointer user_data)
{
  g_main_loop_quit ((GMainLoop *) user_data);
  return G_SOURCE_REMOVE;
}

/**
 * @brief fakesink handoff callback counting buffers and checking the payload.
 */
static void
_repo_handoff_cb (GstElement *element, GstBuffer *buffer, GstPad *pad, gpointer user_data)
{
  repo_run_s *run = (repo_run_s *) user_data;
  guint sample = run->first_sample + run->buffers;
  guint i, n = gst_buffer_n_memory (buffer);
  GstMapInfo map;

  run->buffers++;
  run->last_mems = n;

  if (run->tensors == 0)
    return;

  if (n != run->tensors) {
    run->mismatches++;
    return;
  }

  for (i = 0; i < n; i++) {
    GstMemory *mem = gst_buffer_peek_memory (buffer, i);

    if (!gst_memory_map (mem, &map, GST_MAP_READ)) {
      run->mismatches++;
      continue;
    }
    if (map.size != REPO_TENSOR_SIZE || map.data[128] != _repo_payload (sample, i))
      run->mismatches++;
    gst_memory_unmap (mem, &map);
  }
}

/**
 * @brief Log handler counting what json-glib logs, e.g. an array index out of range.
 */
static void
_repo_json_log_cb (const gchar *domain, GLogLevelFlags level,
    const gchar *message, gpointer user_data)
{
  (*(guint *) user_data)++;
}

/**
 * @brief Run a pipeline with a fakesink named sink0 until EOS, ERROR or timeout.
 */
static void
_run_repo_pipeline (GstElement *pipeline, repo_run_s *run)
{
  GstElement *sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink0");
  GstBus *bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  guint watch_id, timeout_id, log_id;

  ASSERT_NE (sink, nullptr);
  g_object_set (sink, "signal-handoffs", TRUE, NULL);
  g_signal_connect (sink, "handoff", G_CALLBACK (_repo_handoff_cb), run);

  run->loop = g_main_loop_new (NULL, FALSE);
  run->type = GST_MESSAGE_UNKNOWN;
  watch_id = gst_bus_add_watch (bus, _repo_bus_cb, run);
  timeout_id = g_timeout_add (10000, _repo_timeout_cb, run->loop);
  log_id = g_log_set_handler ("Json",
      (GLogLevelFlags) (G_LOG_LEVEL_CRITICAL | G_LOG_LEVEL_WARNING),
      _repo_json_log_cb, &run->json_logs);

  if (gst_element_set_state (pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
    run->type = GST_MESSAGE_ANY;
  else
    g_main_loop_run (run->loop);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_log_remove_handler ("Json", log_id);
  g_source_remove (watch_id);
  if (run->type != GST_MESSAGE_UNKNOWN)
    g_source_remove (timeout_id);

  g_clear_pointer (&run->loop, g_main_loop_unref);
  gst_object_unref (bus);
  gst_object_unref (sink);
}

/**
 * @brief Read samples [start, stop] of the crafted repository without shuffle.
 */
static void
_read_crafted_repo (guint start, guint stop, repo_run_s *run)
{
  g_autofree gchar *str_pipeline = g_strdup_printf (
      "datareposrc name=src0 location=" REPO_DATA " json=" REPO_JSON
      " is-shuffle=false start-sample-index=%u stop-sample-index=%u ! fakesink name=sink0",
      start, stop);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);

  ASSERT_NE (pipeline, nullptr);
  run->first_sample = start;
  _run_repo_pipeline (pipeline, run);
  gst_object_unref (pipeline);
}

/**
 * @brief Expect that datareposrc refused the repository before pushing anything.
 */
static void
_expect_refused (repo_run_s *run)
{
  EXPECT_EQ (run->type, GST_MESSAGE_ERROR);
  EXPECT_STREQ (run->error_src, "src0");
  EXPECT_EQ (run->error_domain, GST_STREAM_ERROR);
  EXPECT_EQ (run->error_code, GST_STREAM_ERROR_FORMAT);
  EXPECT_EQ (run->buffers, 0U);
  EXPECT_EQ (run->json_logs, 0U);
  g_clear_pointer (&run->error_src, g_free);
  g_remove (REPO_DATA);
  g_remove (REPO_JSON);
}

/**
 * @brief Read a well-formed crafted flexible repository.
 * The last tensor ends exactly at the end of the data file.
 */
TEST (datareposrc, readCraftedFlexibleTensors)
{
  repo_run_s run = {};

  EXPECT_EQ (_write_flexible_repo (4, 2), 8 * REPO_TENSOR_SIZE);
  _write_valid_flexible_json (REPO_JSON, 4, 2);

  run.tensors = 2;
  _read_crafted_repo (0, 3, &run);

  EXPECT_EQ (run.type, GST_MESSAGE_EOS);
  EXPECT_EQ (run.buffers, 4U);
  EXPECT_EQ (run.last_mems, 2U);
  EXPECT_EQ (run.mismatches, 0U);
  EXPECT_EQ (run.json_logs, 0U);
  g_remove (REPO_DATA);
  g_remove (REPO_JSON);
}

/**
 * @brief A sample may carry up to NNS_TENSOR_SIZE_LIMIT flexible tensors.
 */
TEST (datareposrc, readCraftedFlexibleMaxTensors)
{
  repo_run_s run = {};

  _write_flexible_repo (1, NNS_TENSOR_SIZE_LIMIT);
  _write_valid_flexible_json (REPO_JSON, 1, NNS_TENSOR_SIZE_LIMIT);

  _read_crafted_repo (0, 0, &run);

  EXPECT_EQ (run.type, GST_MESSAGE_EOS);
  EXPECT_EQ (run.buffers, 1U);
  g_remove (REPO_DATA);
  g_remove (REPO_JSON);
}

/**
 * @brief A sample with more than NNS_TENSOR_SIZE_LIMIT tensors is refused.
 */
TEST (datareposrc, readCraftedFlexibleTooManyTensors_n)
{
  repo_run_s run = {};

  _write_flexible_repo (1, NNS_TENSOR_SIZE_LIMIT + 1);
  _write_valid_flexible_json (REPO_JSON, 1, NNS_TENSOR_SIZE_LIMIT + 1);

  _read_crafted_repo (0, 0, &run);
  _expect_refused (&run);
}

/**
 * @brief A decreasing tensor_count must not wrap the number of tensors.
 */
TEST (datareposrc, readCraftedFlexibleDecreasingCount_n)
{
  repo_run_s run = {};
  g_autofree gchar *sizes = _repo_seq (8, REPO_TENSOR_SIZE, 0);

  _write_flexible_repo (4, 2);
  _write_flexible_json (REPO_JSON, 4, "0,264,528,792", sizes, "0,4,2,6");

  _read_crafted_repo (1, 1, &run);
  _expect_refused (&run);
}

/**
 * @brief A tensor_count past the tensor_size array is refused.
 */
TEST (datareposrc, readCraftedFlexibleCountPastSizes_n)
{
  repo_run_s run = {};
  g_autofree gchar *sizes = _repo_seq (8, REPO_TENSOR_SIZE, 0);

  _write_flexible_repo (4, 2);
  _write_flexible_json (REPO_JSON, 4, "0,264,528,792", sizes, "0,2,4,9");

  _read_crafted_repo (2, 2, &run);
  _expect_refused (&run);
}

/**
 * @brief A negative sample_offset is refused.
 */
TEST (datareposrc, readCraftedFlexibleNegativeOffset_n)
{
  repo_run_s run = {};
  g_autofree gchar *sizes = _repo_seq (8, REPO_TENSOR_SIZE, 0);

  _write_flexible_repo (4, 2);
  _write_flexible_json (REPO_JSON, 4, "0,264,-528,792", sizes, "0,2,4,6");

  _read_crafted_repo (2, 2, &run);
  _expect_refused (&run);
}

/**
 * @brief A negative tensor_count is refused.
 */
TEST (datareposrc, readCraftedFlexibleNegativeCount_n)
{
  repo_run_s run = {};
  g_autofree gchar *sizes = _repo_seq (8, REPO_TENSOR_SIZE, 0);

  _write_flexible_repo (4, 2);
  _write_flexible_json (REPO_JSON, 4, "0,264,528,792", sizes, "0,2,-4,6");

  _read_crafted_repo (2, 2, &run);
  _expect_refused (&run);
}

/**
 * @brief A negative tensor_size is refused.
 */
TEST (datareposrc, readCraftedFlexibleNegativeSize_n)
{
  repo_run_s run = {};

  _write_flexible_repo (4, 2);
  _write_flexible_json (REPO_JSON, 4, "0,264,528,792",
      "-132,132,132,132,132,132,132,132", "0,2,4,6");

  _read_crafted_repo (0, 0, &run);
  _expect_refused (&run);
}

/**
 * @brief A tensor_count that does not fit in 32 bits is refused instead of truncated.
 * 4294967298 truncates to 2, which would describe the second sample of this repository.
 */
TEST (datareposrc, readCraftedFlexibleNextCountBeyondUint_n)
{
  repo_run_s run = {};
  g_autofree gchar *sizes = _repo_seq (8, REPO_TENSOR_SIZE, 0);

  _write_flexible_repo (4, 2);
  _write_flexible_json (REPO_JSON, 4, "0,264,528,792", sizes, "0,4294967298,4,6");

  _read_crafted_repo (0, 0, &run);
  _expect_refused (&run);
}

/**
 * @brief The tensor_count of the sample itself is refused when it does not fit in 32 bits.
 */
TEST (datareposrc, readCraftedFlexibleCountBeyondUint_n)
{
  repo_run_s run = {};
  g_autofree gchar *sizes = _repo_seq (8, REPO_TENSOR_SIZE, 0);

  _write_flexible_repo (4, 2);
  _write_flexible_json (REPO_JSON, 4, "0,264,528,792", sizes, "0,4294967298,4,6");

  _read_crafted_repo (1, 1, &run);
  _expect_refused (&run);
}

/**
 * @brief A tensor smaller than the flexible meta header is refused.
 */
TEST (datareposrc, readCraftedFlexibleSizeBelowHeader_n)
{
  repo_run_s run = {};

  _write_flexible_repo (4, 2);
  _write_flexible_json (REPO_JSON, 4, "0,264,528,792",
      "127,132,132,132,132,132,132,132", "0,2,4,6");

  _read_crafted_repo (0, 0, &run);
  _expect_refused (&run);
}

/**
 * @brief A tensor size larger than any file is refused instead of allocated.
 */
TEST (datareposrc, readCraftedFlexibleHugeSize_n)
{
  repo_run_s run = {};

  _write_flexible_repo (4, 2);
  _write_flexible_json (REPO_JSON, 4, "0,264,528,792",
      "1000000000000000,132,132,132,132,132,132,132", "0,2,4,6");

  _read_crafted_repo (0, 0, &run);
  _expect_refused (&run);
}

/**
 * @brief A tensor that runs past the end of the data file by one byte is refused.
 */
TEST (datareposrc, readCraftedFlexibleSizePastEnd_n)
{
  repo_run_s run = {};

  _write_flexible_repo (4, 2);
  _write_flexible_json (REPO_JSON, 4, "0,264,528,792",
      "132,132,132,132,132,132,132,133", "0,2,4,6");

  _read_crafted_repo (3, 3, &run);
  _expect_refused (&run);
}

/**
 * @brief A sample offset past the end of the data file is refused.
 */
TEST (datareposrc, readCraftedFlexibleOffsetPastEnd_n)
{
  repo_run_s run = {};
  g_autofree gchar *sizes = _repo_seq (8, REPO_TENSOR_SIZE, 0);

  _write_flexible_repo (4, 2);
  _write_flexible_json (REPO_JSON, 4, "0,264,528,1057", sizes, "0,2,4,6");

  _read_crafted_repo (3, 3, &run);
  _expect_refused (&run);
}

/**
 * @brief A sample without a sample_offset entry is refused.
 */
TEST (datareposrc, readCraftedFlexibleOffsetMissing_n)
{
  repo_run_s run = {};
  g_autofree gchar *sizes = _repo_seq (8, REPO_TENSOR_SIZE, 0);

  _write_flexible_repo (4, 2);
  _write_flexible_json (REPO_JSON, 4, "0,264", sizes, "0,2,4,6");

  _read_crafted_repo (3, 3, &run);
  _expect_refused (&run);
}

/**
 * @brief A sample without a tensor_count entry is refused.
 */
TEST (datareposrc, readCraftedFlexibleCountMissing_n)
{
  repo_run_s run = {};
  g_autofree gchar *sizes = _repo_seq (8, REPO_TENSOR_SIZE, 0);

  _write_flexible_repo (4, 2);
  _write_flexible_json (REPO_JSON, 4, "0,264,528,792", sizes, "0,2");

  _read_crafted_repo (3, 3, &run);
  _expect_refused (&run);
}

/**
 * @brief Flexible caps without a JSON file give no sample layout to read.
 */
TEST (datareposrc, readFlexibleNoJSONWithCapsParam_n)
{
  repo_run_s run = {};
  GstElement *pipeline = gst_parse_launch (
      "datareposrc name=src0 location=" REPO_DATA " is-shuffle=false stop-sample-index=3 "
      "caps=\"" REPO_FLEX_CAPS "\" ! fakesink name=sink0",
      NULL);

  ASSERT_NE (pipeline, nullptr);
  _write_flexible_repo (4, 2);

  _run_repo_pipeline (pipeline, &run);
  gst_object_unref (pipeline);
  _expect_refused (&run);
}

/**
 * @brief Build a bare datareposrc pipeline whose properties are set later.
 */
static GstElement *
_repo_bare_pipeline (GstElement **src)
{
  GstElement *pipeline
      = gst_parse_launch ("datareposrc name=src0 ! fakesink name=sink0", NULL);

  if (pipeline)
    *src = gst_bin_get_by_name (GST_BIN (pipeline), "src0");
  return pipeline;
}

/**
 * @brief A JSON that fails to load must not leave the arrays of the previous one behind.
 */
TEST (datareposrc, readFlexibleAfterFailedJsonReload_n)
{
  repo_run_s run = {};
  GstElement *src = NULL;
  GstElement *pipeline = _repo_bare_pipeline (&src);
  GstCaps *caps = gst_caps_from_string (REPO_FLEX_CAPS);

  ASSERT_NE (pipeline, nullptr);
  ASSERT_NE (src, nullptr);
  _write_flexible_repo (4, 2);
  _write_valid_flexible_json (REPO_JSON, 4, 2);
  EXPECT_TRUE (g_file_set_contents ("broken.json", "{ not a json", -1, NULL));

  g_object_set (src, "json", REPO_JSON, NULL);
  g_object_set (src, "json", "broken.json", NULL);
  g_object_set (src, "caps", caps, NULL);
  g_object_set (src, "location", REPO_DATA, "is-shuffle", FALSE, NULL);

  _run_repo_pipeline (pipeline, &run);
  gst_caps_unref (caps);
  gst_object_unref (src);
  gst_object_unref (pipeline);
  g_remove ("broken.json");
  _expect_refused (&run);
}

/**
 * @brief Loading a second valid JSON replaces the sample layout of the first.
 */
TEST (datareposrc, readFlexibleAfterJsonReload)
{
  repo_run_s run = {};
  GstElement *src = NULL;
  GstElement *pipeline = _repo_bare_pipeline (&src);

  ASSERT_NE (pipeline, nullptr);
  ASSERT_NE (src, nullptr);
  _write_flexible_repo (4, 2);
  _write_flexible_json ("first.json", 4, "0,0,0,0", "1,1,1,1,1,1,1,1", "0,0,0,0");
  _write_valid_flexible_json (REPO_JSON, 4, 2);

  g_object_set (src, "json", "first.json", NULL);
  g_object_set (src, "json", REPO_JSON, NULL);
  g_object_set (src, "location", REPO_DATA, "is-shuffle", FALSE, NULL);

  run.tensors = 2;
  _run_repo_pipeline (pipeline, &run);
  gst_object_unref (src);
  gst_object_unref (pipeline);

  EXPECT_EQ (run.type, GST_MESSAGE_EOS);
  EXPECT_EQ (run.buffers, 4U);
  EXPECT_EQ (run.mismatches, 0U);
  EXPECT_EQ (run.json_logs, 0U);
  g_remove ("first.json");
  g_remove (REPO_DATA);
  g_remove (REPO_JSON);
}

/**
 * @brief Setting json while the element is not stopped is refused and does not reload it.
 */
TEST (datareposrc, setJsonWhilePaused_n)
{
  repo_run_s run = {};
  GstElement *src = NULL;
  GstElement *pipeline = _repo_bare_pipeline (&src);
  g_autofree gchar *json_path = NULL;

  ASSERT_NE (pipeline, nullptr);
  ASSERT_NE (src, nullptr);
  _write_flexible_repo (4, 2);
  _write_valid_flexible_json (REPO_JSON, 4, 2);

  g_object_set (src, "json", REPO_JSON, "location", REPO_DATA, "is-shuffle", FALSE, NULL);
  EXPECT_NE (gst_element_set_state (pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_FAILURE);

  /* The same path now holds a file that cannot be loaded. */
  EXPECT_TRUE (g_file_set_contents (REPO_JSON, "{ not a json", -1, NULL));
  g_object_set (src, "json", REPO_JSON, NULL);
  g_object_get (src, "json", &json_path, NULL);
  EXPECT_STREQ (json_path, REPO_JSON);

  run.tensors = 2;
  _run_repo_pipeline (pipeline, &run);
  gst_object_unref (src);
  gst_object_unref (pipeline);

  EXPECT_EQ (run.type, GST_MESSAGE_EOS);
  EXPECT_EQ (run.buffers, 4U);
  EXPECT_EQ (run.mismatches, 0U);
  EXPECT_EQ (run.json_logs, 0U);
  g_remove (REPO_DATA);
  g_remove (REPO_JSON);
}

/**
 * @brief Write an octet repository whose one sample is @a sample_size bytes.
 */
static void
_write_octet_json (gsize sample_size)
{
  g_autofree gchar *json = g_strdup_printf (
      "{\"gst_caps\":\"application/octet-stream\",\"total_samples\":1,\"sample_size\":%zu}",
      sample_size);

  EXPECT_TRUE (g_file_set_contents (REPO_JSON, json, -1, NULL));
}

/**
 * @brief A sample exactly as large as the data file is read.
 */
TEST (datareposrc, readOctetSampleOfFileSize)
{
  repo_run_s run = {};
  gsize size = _write_flexible_repo (4, 2);

  _write_octet_json (size);
  _read_crafted_repo (0, 0, &run);

  EXPECT_EQ (run.type, GST_MESSAGE_EOS);
  EXPECT_EQ (run.buffers, 1U);
  g_remove (REPO_DATA);
  g_remove (REPO_JSON);
}

/**
 * @brief A sample larger than the data file is refused instead of allocated.
 */
TEST (datareposrc, readOctetSampleLargerThanFile_n)
{
  repo_run_s run = {};
  gsize size = _write_flexible_repo (4, 2);

  _write_octet_json (size + 1);
  _read_crafted_repo (0, 0, &run);
  _expect_refused (&run);
}

/**
 * @brief Static tensors larger than the data file are refused instead of allocated.
 */
TEST (datareposrc, readTensorsNoJSONSampleLargerThanFile_n)
{
  repo_run_s run = {};
  g_autofree gchar *file_path = get_file_path (filename);
  g_autofree gchar *str_pipeline = g_strdup_printf (
      "datareposrc name=src0 location=%s stop-sample-index=1 "
      "caps=\"other/tensors, format=(string)static, framerate=(fraction)0/1, "
      "num_tensors=(int)1, dimensions=(string)100000:100000:1:1, types=(string)float32\" ! "
      "fakesink name=sink0",
      file_path);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);

  ASSERT_NE (pipeline, nullptr);
  _run_repo_pipeline (pipeline, &run);
  gst_object_unref (pipeline);
  _expect_refused (&run);
}

/**
 * @brief Main GTest
 */
int
main (int argc, char **argv)
{
  int result = -1;
  gchar *work_dir;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  /* These tests write fixed file names, which unittest_datareposink also uses. */
  /* Enter after InitGoogleTest, which anchors --gtest_output to the start directory. */
  work_dir = enterPrivateWorkDir ();

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  leavePrivateWorkDir (&work_dir);

  return result;
}
