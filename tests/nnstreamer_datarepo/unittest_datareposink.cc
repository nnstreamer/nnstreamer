/**
 * @file        unittest_datareposink.cc
 * @date        21 Apr 2023
 * @brief       Unit test for datareposink
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
 * @brief Test for writing image files
 */
TEST (datareposink, writeImageFiles)
{
  GFile *file = NULL;
  gchar *contents = NULL;
  gchar *filename = NULL;
  GstBus *bus;
  GMainLoop *loop;
  gint i = 0;
  gboolean ret;
  const gchar *str_pipeline
      = "videotestsrc num-buffers=5 ! pngenc ! datareposink location=image_%02d.png json=image.json";

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

  /* Confirm file creation */
  for (i = 0; i < 5; i++) {
    filename = g_strdup_printf ("image_%02d.png", i);
    file = g_file_new_for_path (filename);
    g_free (filename);

    ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
    g_object_unref (file);
    g_free (contents);
    ASSERT_EQ (ret, TRUE);
  }

  /* Confirm file creation */
  file = g_file_new_for_path ("image.json");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_object_unref (file);
  g_free (contents);
  ASSERT_EQ (ret, TRUE);

  g_remove ("image.json");
  for (i = 0; i < 5; i++) {
    filename = g_strdup_printf ("image_%02d.png", i);
    g_remove (filename);
    g_free (filename);
  }
}

/**
 * @brief Test for writing an audio raw file
 */
TEST (datareposink, writeAudioRaw)
{
  GFile *file = NULL;
  gchar *contents = NULL;
  GstBus *bus;
  GMainLoop *loop;
  gboolean ret;
  const gchar *str_pipeline
      = "audiotestsrc samplesperbuffer=44100 num-buffers=1 ! "
        "audio/x-raw, format=S16LE, layout=interleaved, rate=44100, channels=1 ! "
        "datareposink location=audio.raw json=audio.json";

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

  /* Confirm file creation */
  file = g_file_new_for_path ("audio.raw");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_object_unref (file);
  g_free (contents);
  ASSERT_EQ (ret, TRUE);

  /* Confirm file creation */
  file = g_file_new_for_path ("audio.json");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_object_unref (file);
  g_free (contents);
  ASSERT_EQ (ret, TRUE);

  g_remove ("audio.json");
  g_remove ("audio.raw");
}

/**
 * @brief Test for writing a video raw file
 */
TEST (datareposink, writeVideoRaw)
{
  GFile *file = NULL;
  gchar *contents = NULL;
  GstBus *bus;
  GMainLoop *loop;
  gboolean ret;
  const gchar *str_pipeline
      = "videotestsrc num-buffers=10 ! datareposink location=video.raw json=video.json";

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

  /* Confirm file creation */
  file = g_file_new_for_path ("video.raw");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_object_unref (file);
  g_free (contents);
  ASSERT_EQ (ret, TRUE);

  /* Confirm file creation */
  file = g_file_new_for_path ("video.json");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_object_unref (file);
  g_free (contents);
  ASSERT_EQ (ret, TRUE);

  g_remove ("video.raw");
  g_remove ("video.json");
}

/**
 * @brief Test for writing a Tensors file
 */
TEST (datareposink, writeTensors)
{
  GFile *file = NULL;
  gchar *contents = NULL;
  GstBus *bus;
  GMainLoop *loop;
  gchar *file_path = NULL;
  gchar *json_path = NULL;
  GstElement *datareposink = NULL;
  gchar *get_str = NULL;
  gboolean ret;

  loop = g_main_loop_new (NULL, FALSE);

  file_path = get_file_path (filename);
  json_path = get_file_path (json);

  gchar *str_pipeline = g_strdup_printf (
      "datareposrc location=%s json=%s start-sample-index=0 stop-sample-index=9 ! "
      "datareposink name=datareposink location=mnist.data json=mnist.json",
      file_path, json_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  ASSERT_NE (pipeline, nullptr);

  datareposink = gst_bin_get_by_name (GST_BIN (pipeline), "datareposink");
  EXPECT_NE (datareposink, nullptr);

  g_object_get (datareposink, "location", &get_str, NULL);
  EXPECT_STREQ (get_str, "mnist.data");

  g_object_get (datareposink, "json", &get_str, NULL);
  EXPECT_STREQ (get_str, "mnist.json");

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (pipeline);
  g_main_loop_unref (loop);

  /* Confirm file creation */
  file = g_file_new_for_path ("mnist.data");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_object_unref (file);
  g_free (contents);
  ASSERT_EQ (ret, TRUE);

  /* Confirm file creation */
  file = g_file_new_for_path ("mnist.json");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_free (contents);
  g_object_unref (file);
  g_free (file_path);
  g_free (json_path);
  ASSERT_EQ (ret, TRUE);

  g_remove ("mnist.data");
  g_remove ("mnist.json");
}

/**
 * @brief Test for writing flexible tensors
 */
TEST (datareposink, writeFlexibleTensors)
{
  GFile *file = NULL;
  gchar *contents = NULL;
  GstBus *bus;
  GMainLoop *loop;
  gboolean ret;
  const gchar *str_pipeline
      = "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! "
        "video/x-raw,format=RGB,width=176,height=144,framerate=10/1 ! tensor_converter ! join0.sink_0 "
        "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! "
        "video/x-raw,format=RGB,width=320,height=240,framerate=10/1 ! tensor_converter ! join0.sink_1 "
        "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! "
        "video/x-raw,format=RGB,width=640,height=480,framerate=10/1 ! tensor_converter ! join0.sink_2 "
        "join name=join0 ! other/tensors,format=flexible ! "
        "datareposink location=flexible.data json=flexible.json";

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

  /* Confirm file creation */
  file = g_file_new_for_path ("flexible.data");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_object_unref (file);
  g_free (contents);
  ASSERT_EQ (ret, TRUE);

  /* Confirm file creation */
  file = g_file_new_for_path ("flexible.json");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_object_unref (file);
  g_free (contents);
  ASSERT_EQ (ret, TRUE);

  g_remove ("flexible.data");
  g_remove ("flexible.json");
}

/**
 * @brief Test for writing a file with invalid param (JSON path)
 */
TEST (datareposink, invalidJsonPath0_n)
{
  GstElement *datareposink = NULL;

  const gchar *str_pipeline
      = "videotestsrc num-buffers=10 ! pngenc ! datareposink name=datareposink";

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  datareposink = gst_bin_get_by_name (GST_BIN (pipeline), "datareposink");
  EXPECT_NE (datareposink, nullptr);

  g_object_set (GST_OBJECT (datareposink), "location", "video.raw", NULL);
  /* set invalid param */
  g_object_set (GST_OBJECT (datareposink), "json", NULL, NULL);

  /* state chagne failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (pipeline);
}

/**
 * @brief Test for writing a file with invalid param (file path)
 */
TEST (datareposink, invalidFilePath0_n)
{
  GstElement *datareposink = NULL;

  const gchar *str_pipeline
      = "videotestsrc num-buffers=10 ! pngenc ! datareposink name=datareposink";

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  datareposink = gst_bin_get_by_name (GST_BIN (pipeline), "datareposink");
  EXPECT_NE (datareposink, nullptr);

  g_object_set (GST_OBJECT (datareposink), "json", "image.json", NULL);
  /* set invalid param */
  g_object_set (GST_OBJECT (datareposink), "location", NULL, NULL);

  /* state chagne failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (pipeline);
}

/**
 * @brief Test for writing a file with video compression format
 */
TEST (datareposink, unsupportedVideoCaps0_n)
{
  const gchar *str_pipeline
      = "videotestsrc ! vp8enc ! datareposink location=video.raw json=video.json";

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  /* Could not to to GST_STATE_PLAYING state due to caps negotiation failure */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (pipeline);
}

/**
 * @brief Test for writing a file with audio compression format
 */
TEST (datareposink, unsupportedAudioCaps0_n)
{
  const gchar *str_pipeline = "audiotestsrc ! audio/x-raw,rate=44100,channels=2 ! "
                              "wavenc ! datareposink location=audio.raw json=audio.json";

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  /* Could not to to GST_STATE_PLAYING state due to caps negotiation failure */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (pipeline);
}

/**
 * @brief Test for writing flexible tensors
 */
TEST (datareposink, writeFlexibleTensors_n)
{
  GFile *file = NULL;
  GstBus *bus;
  GMainLoop *loop;
  GstElement *pipeline;
  GFileInfo *file_info = NULL;
  gint64 size = 0;
  int i;
  gchar *filename = NULL;

  create_image_test_file ();

  /* Insert non-Flexible Tensor data after negotiating with flexible caps. */
  const gchar *str_pipeline
      = "multifilesrc location=img_%02d.png caps=other/tensors,format=flexible ! "
        "datareposink location=flexible.data json=flexible.json";

  pipeline = gst_parse_launch (str_pipeline, NULL);
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

  /* Confirm file creation */
  file = g_file_new_for_path ("flexible.data");
  ASSERT_NE (file, nullptr);
  file_info = g_file_query_info (
      file, G_FILE_ATTRIBUTE_STANDARD_SIZE, G_FILE_QUERY_INFO_NONE, NULL, NULL);
  ASSERT_NE (file_info, nullptr);
  size = g_file_info_get_size (file_info);
  ASSERT_EQ (size, 0);
  g_object_unref (file_info);
  g_object_unref (file);

  for (i = 0; i < 5; i++) {
    filename = g_strdup_printf ("img_%02d.png", i);
    g_remove (filename);
    g_free (filename);
  }

  g_remove ("img.json");
  g_remove ("flexible.json");
  g_remove ("flexible.data");
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
