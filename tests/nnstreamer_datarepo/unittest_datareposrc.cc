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
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (pipeline);
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
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (pipeline);
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
  const gchar *str_pipeline = "videotestsrc num-buffers=5 ! pngenc ! "
                              "datareposink location=img_%02d.png json=img.json";

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  loop = g_main_loop_new (NULL, FALSE);
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
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  EXPECT_NE (buffer_count, 0);
  handler = NULL;

  gst_object_unref (tensor_sink);
  gst_object_unref (pipeline);
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
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  EXPECT_NE (buffer_count, 0);
  handler = NULL;

  gst_object_unref (tensor_sink);
  gst_object_unref (pipeline);
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
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  gst_object_unref (pipeline);
  g_main_loop_unref (loop);

  if (!g_file_get_contents ("aduio1.raw", &data_1, &size_1, NULL)) {
    goto error;
  }

  if (!g_file_get_contents ("result.raw", &data_2, &size_2, NULL)) {
    goto error;
  }
  EXPECT_EQ (size_1, size_2);
  g_free (data_1);
  g_free (data_2);
  data_1 = data_2 = NULL;

  if (!g_file_get_contents ("audio1.json", &data_1, &size_1, NULL)) {
    goto error;
  }

  if (!g_file_get_contents ("result.json", &data_2, &size_2, NULL)) {
    goto error;
  }
  ret = g_strcmp0 (data_1, data_2);
  EXPECT_EQ (ret, 0);
error:
  g_free (data_1);
  g_free (data_2);
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
  gst_object_unref (datareposrc);
  gst_object_unref (pipeline);
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
  gst_object_unref (datareposrc);
  gst_object_unref (pipeline);
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
  gst_object_unref (datareposrc);
  gst_object_unref (pipeline);
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
  gst_object_unref (datareposrc);
  gst_object_unref (pipeline);
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
  gst_object_unref (datareposrc);
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
  gst_object_unref (bus);

  g_object_get (datareposrc, "location", &get_str, NULL);
  EXPECT_STREQ (get_str, file_path);
  g_free (get_str);

  g_object_get (datareposrc, "json", &get_str, NULL);
  EXPECT_STREQ (get_str, json_path);
  g_free (get_str);

  g_object_get (datareposrc, "tensors-sequence", &get_str, NULL);
  EXPECT_STREQ (get_str, "0,1");
  g_free (get_str);

  g_object_get (datareposrc, "is-shuffle", &get_value, NULL);
  ASSERT_EQ (get_value, 1U);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_main_loop_run (loop);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (datareposrc);
  gst_object_unref (pipeline);
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
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  gst_object_unref (pipeline);
  g_main_loop_unref (loop);

  if (!g_file_get_contents ("flexible.raw", &data_1, &size_1, NULL)) {
    goto error;
  }

  if (!g_file_get_contents ("result.raw", &data_2, &size_2, NULL)) {
    goto error;
  }
  EXPECT_EQ (size_1, size_2);
  g_free (data_1);
  g_free (data_2);
  data_1 = data_2 = NULL;

  if (!g_file_get_contents ("flexible.json", &data_1, &size_1, NULL)) {
    goto error;
  }

  if (!g_file_get_contents ("result.json", &data_2, &size_2, NULL)) {
    goto error;
  }
  ret = g_strcmp0 (data_1, data_2);
  EXPECT_EQ (ret, 0);
error:
  g_free (data_1);
  g_free (data_2);
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
  guint64 start_time, end_time;
  gdouble elapsed_time;
  GstElement *tensor_sink;
  GstBus *bus;
  GMainLoop *loop;
  gint file_index = 1;
  const gchar *str_pipeline
      = "datareposrc location=flexible1.data json=flexible1.json ! queue ! tensor_sink name=tensor_sink0 sync=true";

  create_flexible_tensors_test_file (fps, file_index);
  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  gst_object_unref (bus);

  start_time = g_get_monotonic_time ();

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  end_time = g_get_monotonic_time ();
  elapsed_time = (end_time - start_time) / (double) G_USEC_PER_SEC;

  g_print ("Elapsed time: %.6f second\n", elapsed_time);
  EXPECT_LT (0.8, elapsed_time);

  gst_object_unref (pipeline);
  g_main_loop_unref (loop);

  pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  gst_object_unref (bus);

  tensor_sink = gst_bin_get_by_name (GST_BIN (pipeline), "tensor_sink0");
  g_object_set (GST_OBJECT (tensor_sink), "sync", FALSE, NULL);

  start_time = g_get_monotonic_time ();

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  end_time = g_get_monotonic_time ();
  elapsed_time = (end_time - start_time) / (double) G_USEC_PER_SEC;

  g_print ("Elapsed time: %.6f second\n", elapsed_time);
  EXPECT_LT (elapsed_time, 0.05);

  gst_object_unref (tensor_sink);
  gst_object_unref (pipeline);
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
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  EXPECT_NE (buffer_count, 0);
  handler = NULL;

  gst_object_unref (tensor_sink);
  gst_object_unref (pipeline);
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
  gst_object_unref (bus);

  g_object_get (datareposrc, "location", &get_str, NULL);
  EXPECT_STREQ (get_str, file_path);
  g_free (get_str);

  g_object_get (datareposrc, "tensors-sequence", &get_str, NULL);
  EXPECT_STREQ (get_str, "0,1");
  g_free (get_str);

  g_object_get (datareposrc, "is-shuffle", &get_value, NULL);
  ASSERT_EQ (get_value, 1U);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_main_loop_run (loop);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (datareposrc);
  gst_object_unref (pipeline);
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
  gst_object_unref (datareposrc);
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
  gst_object_unref (datareposrc);
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
  gst_object_unref (datareposrc);
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
  gst_object_unref (datareposrc);
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
  gst_object_unref (datareposrc);
  gst_object_unref (pipeline);
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
  gst_object_unref (datareposrc);
  gst_object_unref (pipeline);
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
  gst_object_unref (datareposrc);
  gst_object_unref (pipeline);
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
  gst_object_unref (bus);

  /* EXPECT_EQ not checked due to internal data stream error */
  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  /* Internal data stream error */
  EXPECT_EQ (buffer_count, 0);
  handler = NULL;

  gst_object_unref (tensor_sink);
  gst_object_unref (pipeline);
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
  gst_object_unref (bus);

  /* EXPECT_EQ not checked due to internal data stream error */
  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  /* Internal data stream error */
  EXPECT_EQ (buffer_count, 0);
  handler = NULL;

  gst_object_unref (tensor_sink);
  gst_object_unref (pipeline);
  g_main_loop_unref (loop);

  g_remove ("audio3.json");
  g_remove ("audio3.raw");
  g_remove ("sparse3.json");
  g_remove ("sparse3.data");
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

  return result;
}
