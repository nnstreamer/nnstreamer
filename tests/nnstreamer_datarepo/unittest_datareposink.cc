/**
 * @file        unittest_datareposink.cc
 * @date        21 Apr 2023
 * @brief       Unit test for datareposink
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Hyunil Park <hyunil46.park@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <fcntl.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api_util.h>
#include <sys/stat.h>
#include <unistd.h>
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

  gchar *str_pipeline = g_strdup ("videotestsrc num-buffers=5 ! pngenc ! "
                                  "datareposink location=img_%02d.png json=img.json");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_clear_pointer (&str_pipeline, g_free);
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
    g_clear_pointer (&filename, g_free);

    ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
    g_clear_pointer (&contents, g_free);
    g_clear_pointer (&file, g_object_unref);
    ASSERT_EQ (ret, TRUE);
  }

  /* Confirm file creation */
  file = g_file_new_for_path ("image.json");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_clear_pointer (&contents, g_free);
  g_clear_pointer (&file, g_object_unref);
  ASSERT_EQ (ret, TRUE);

  g_remove ("image.json");
  for (i = 0; i < 5; i++) {
    filename = g_strdup_printf ("image_%02d.png", i);
    g_remove (filename);
    g_clear_pointer (&filename, g_free);
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
  g_clear_pointer (&contents, g_free);
  g_clear_pointer (&file, g_object_unref);
  ASSERT_EQ (ret, TRUE);

  /* Confirm file creation */
  file = g_file_new_for_path ("audio.json");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_clear_pointer (&contents, g_free);
  g_clear_pointer (&file, g_object_unref);
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
  g_clear_pointer (&contents, g_free);
  g_clear_pointer (&file, g_object_unref);
  ASSERT_EQ (ret, TRUE);

  /* Confirm file creation */
  file = g_file_new_for_path ("video.json");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_clear_pointer (&contents, g_free);
  g_clear_pointer (&file, g_object_unref);
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
  g_clear_pointer (&str_pipeline, g_free);
  g_clear_pointer (&file_path, g_free);
  g_clear_pointer (&json_path, g_free);
  ASSERT_NE (pipeline, nullptr);

  datareposink = gst_bin_get_by_name (GST_BIN (pipeline), "datareposink");
  EXPECT_NE (datareposink, nullptr);

  g_object_get (datareposink, "location", &get_str, NULL);
  EXPECT_STREQ (get_str, "mnist.data");
  g_clear_pointer (&get_str, g_free);

  g_object_get (datareposink, "json", &get_str, NULL);
  EXPECT_STREQ (get_str, "mnist.json");
  g_clear_pointer (&get_str, g_free);
  gst_object_unref (datareposink);

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  ASSERT_NE (bus, nullptr);
  gst_bus_add_watch (bus, bus_callback, loop);
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  g_main_loop_unref (loop);
  gst_object_unref (pipeline);

  /* Confirm file creation */
  file = g_file_new_for_path ("mnist.data");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_clear_pointer (&contents, g_free);
  g_clear_pointer (&file, g_object_unref);
  ASSERT_EQ (ret, TRUE);

  /* Confirm file creation */
  file = g_file_new_for_path ("mnist.json");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_clear_pointer (&contents, g_free);
  g_clear_pointer (&file, g_object_unref);
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
  const gchar *found;
  gint total_samples = 0;
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
  g_clear_pointer (&contents, g_free);
  g_clear_pointer (&file, g_object_unref);
  ASSERT_EQ (ret, TRUE);

  /* Confirm file creation */
  file = g_file_new_for_path ("flexible.json");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  /* every buffer of the three sources should reach the sink */
  found = contents ? g_strstr_len (contents, -1, "\"total_samples\"") : NULL;
  EXPECT_TRUE (found && sscanf (found, "\"total_samples\"%*[ :]%d", &total_samples) == 1);
  EXPECT_EQ (total_samples, 9);
  g_clear_pointer (&contents, g_free);
  g_clear_pointer (&file, g_object_unref);
  ASSERT_EQ (ret, TRUE);

  g_remove ("flexible.data");
  g_remove ("flexible.json");
}


/**
 * @brief Test for writing sparse tensors
 */
TEST (datareposink, writeSparseTensors)
{
  GFile *file = NULL;
  gchar *contents = NULL;
  GstBus *bus;
  GMainLoop *loop;
  gchar *file_path = NULL;
  gchar *json_path = NULL;
  gboolean ret;
  gint64 size, org_size = 31760;
  GFileInfo *info = NULL;

  loop = g_main_loop_new (NULL, FALSE);

  file_path = get_file_path (filename);
  json_path = get_file_path (json);

  gchar *str_pipeline = g_strdup_printf (
      "datareposrc location=%s json=%s start-sample-index=0 stop-sample-index=9 ! "
      "tensor_sparse_enc ! other/tensors,format=sparse,framerate=0/1 ! "
      "datareposink location=sparse.data json=sparse.json",
      file_path, json_path);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  g_clear_pointer (&str_pipeline, g_free);
  g_clear_pointer (&file_path, g_free);
  g_clear_pointer (&json_path, g_free);
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

  /* Confirm file creation */
  file = g_file_new_for_path ("sparse.json");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  g_clear_pointer (&contents, g_free);
  g_clear_pointer (&file, g_object_unref);
  ASSERT_EQ (ret, TRUE);

  /* Confirm file creation */
  file = g_file_new_for_path ("sparse.data");
  ret = g_file_load_contents (file, NULL, &contents, NULL, NULL, NULL);
  info = g_file_query_info (
      file, G_FILE_ATTRIBUTE_STANDARD_SIZE, G_FILE_QUERY_INFO_NONE, NULL, NULL);
  size = g_file_info_get_size (info);
  g_clear_pointer (&contents, g_free);
  g_clear_pointer (&file, g_object_unref);
  g_clear_pointer (&info, g_object_unref);
  ASSERT_EQ (ret, TRUE);

  /* The size of one mnist sample is 3176 bytes, the number of samples for test is 10. */
  EXPECT_LT (size, org_size);

  g_remove ("sparse.data");
  g_remove ("sparse.json");
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

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (datareposink);
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

  /* state change failure is expected */
  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (datareposink);
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
 * @brief Write three png images with the given location and report whether the pipeline reached PLAYING.
 */
static gboolean
write_three_images (const gchar *location)
{
  GstElement *pipeline, *sink;
  GstBus *bus;
  GMainLoop *loop;
  gboolean playing;

  pipeline = gst_parse_launch (
      "videotestsrc num-buffers=3 ! pngenc ! datareposink name=sink json=fmt.json", NULL);
  if (pipeline == NULL)
    return FALSE;

  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");
  g_object_set (sink, "location", location, NULL);
  gst_object_unref (sink);

  loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  gst_bus_add_watch (bus, bus_callback, loop);
  gst_object_unref (bus);

  playing = (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT)
             == 0);
  if (playing)
    g_main_loop_run (loop);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (pipeline);
  g_main_loop_unref (loop);
  g_remove ("fmt.json");

  return playing;
}

/**
 * @brief Test the integer placeholders accepted in the image location
 */
TEST (datareposink, writeImageLocationPlaceholders)
{
  const struct {
    const gchar *location;
    const gchar *names[3];
  } cases[] = {
    { "fmt_%d.png", { "fmt_0.png", "fmt_1.png", "fmt_2.png" } },
    { "fmt_%04d.png", { "fmt_0000.png", "fmt_0001.png", "fmt_0002.png" } },
    { "fmt_%02ld.png", { "fmt_00.png", "fmt_01.png", "fmt_02.png" } },
    { "fmt_%03llu.png", { "fmt_000.png", "fmt_001.png", "fmt_002.png" } },
    { "fmt_%3i.png", { "fmt_  0.png", "fmt_  1.png", "fmt_  2.png" } },
    { "fmt_%%_%u.png", { "fmt_%_0.png", "fmt_%_1.png", "fmt_%_2.png" } },
    { "fmt_%#x.png", { "fmt_0.png", "fmt_0x1.png", "fmt_0x2.png" } },
    { "fmt_%X.png", { "fmt_0.png", "fmt_1.png", "fmt_2.png" } },
    { "fmt_%o.png", { "fmt_0.png", "fmt_1.png", "fmt_2.png" } },
    { "fmt_%-3d.png", { "fmt_0  .png", "fmt_1  .png", "fmt_2  .png" } },
    { "fmt_%+d.png", { "fmt_+0.png", "fmt_+1.png", "fmt_+2.png" } },
    { "fmt_% d.png", { "fmt_ 0.png", "fmt_ 1.png", "fmt_ 2.png" } },
    { "fmt_%.3d.png", { "fmt_000.png", "fmt_001.png", "fmt_002.png" } },
    { "fmt_%5.2d.png", { "fmt_   00.png", "fmt_   01.png", "fmt_   02.png" } },
    { "fmt_%.003d.png", { "fmt_000.png", "fmt_001.png", "fmt_002.png" } },
    { "fmt_%.0003u.png", { "fmt_000.png", "fmt_001.png", "fmt_002.png" } },
    { "fmt_%1$u_%1$u.png", { "fmt_0_0.png", "fmt_1_1.png", "fmt_2_2.png" } },
    { "fmt_%1$02d_%1$#x.png", { "fmt_00_0.png", "fmt_01_0x1.png", "fmt_02_0x2.png" } },
    { "fmt_%hd.png", { "fmt_0.png", "fmt_1.png", "fmt_2.png" } },
    { "fmt_%hhu.png", { "fmt_0.png", "fmt_1.png", "fmt_2.png" } },
    { "fmt_%zu.png", { "fmt_0.png", "fmt_1.png", "fmt_2.png" } },
    { "fmt_%1$02d.png", { "fmt_00.png", "fmt_01.png", "fmt_02.png" } },
  };

  for (const auto &c : cases) {
    EXPECT_TRUE (write_three_images (c.location)) << c.location;

    for (const auto name : c.names) {
      EXPECT_TRUE (g_file_test (name, G_FILE_TEST_IS_REGULAR)) << c.location;
      g_remove (name);
    }
  }
}

/**
 * @brief Test that a location without a placeholder keeps overwriting one file
 */
TEST (datareposink, writeImageLocationConstant)
{
  EXPECT_TRUE (write_three_images ("fmt_100%%.png"));
  EXPECT_TRUE (g_file_test ("fmt_100%.png", G_FILE_TEST_IS_REGULAR));
  g_remove ("fmt_100%.png");
}

/**
 * @brief Test that the image location is never used as a printf format string
 */
TEST (datareposink, writeImageLocationInvalid_n)
{
  const gchar *locations[] = {
    "fmt_%s.png",
    "fmt_%n.png",
    "fmt_%s%s%s%s%n.png",
    "fmt_%p.png",
    "fmt_%c.png",
    "fmt_%f.png",
    "fmt_%m.png",
    "fmt_%*d.png",
    "fmt_%.*d.png",
    "fmt_%2$d.png",
    "fmt_%5-d.png",
    "fmt_%lllu.png",
    "fmt_%hhhd.png",
    "fmt_%hld.png",
    "fmt_%1000d.png",
    "fmt_%.1000d.png",
    "fmt_%.0001000d.png",
    "fmt_%1$d_%d.png",
    "fmt_%d_%1$d.png",
    "fmt_%999999999d.png",
    "fmt_%d_%d.png",
    "fmt_%d_%%_%d.png",
    "fmt_%",
  };

  for (const auto location : locations)
    EXPECT_FALSE (write_three_images (location)) << location;
}

/**
 * @brief Data for changing the location of datareposink while it writes images.
 */
typedef struct {
  GstElement *sink; /**< datareposink */
  const gchar *location; /**< location to set before the second image */
  guint count; /**< number of images seen */
  gboolean error; /**< an error message was posted */
  GMainLoop *loop; /**< main loop */
} LocationChangeData;

/**
 * @brief Set the new location of datareposink before the second image reaches it.
 */
static void
change_location_cb (GstElement *identity, GstBuffer *buffer, LocationChangeData *data)
{
  if (++data->count == 2)
    g_object_set (data->sink, "location", data->location, NULL);
}

/**
 * @brief Record whether the pipeline posted an error, and quit at the end.
 */
static gboolean
location_change_bus_cb (GstBus *bus, GstMessage *message, gpointer user_data)
{
  LocationChangeData *data = (LocationChangeData *) user_data;

  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_ERROR:
      data->error = TRUE;
      g_main_loop_quit (data->loop);
      break;
    case GST_MESSAGE_EOS:
      g_main_loop_quit (data->loop);
      break;
    default:
      break;
  }

  return TRUE;
}

/**
 * @brief Write three images with fmt_%d.png, changing the location before the second one.
 */
static gboolean
write_images_changing_location (const gchar *location)
{
  LocationChangeData data = { NULL, location, 0, FALSE, NULL };
  GstElement *pipeline, *identity;
  GstBus *bus;

  pipeline = gst_parse_launch ("videotestsrc num-buffers=3 ! pngenc ! "
                               "identity name=id signal-handoffs=true ! "
                               "datareposink name=sink location=fmt_%d.png json=fmt.json",
      NULL);
  if (pipeline == NULL)
    return FALSE;

  data.sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");
  identity = gst_bin_get_by_name (GST_BIN (pipeline), "id");
  g_signal_connect (identity, "handoff", G_CALLBACK (change_location_cb), &data);
  gst_object_unref (identity);

  data.loop = g_main_loop_new (NULL, FALSE);
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  gst_bus_add_watch (bus, location_change_bus_cb, &data);

  if (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT) == 0)
    g_main_loop_run (data.loop);
  else
    data.error = TRUE;

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_bus_remove_watch (bus);
  gst_object_unref (bus);
  gst_object_unref (data.sink);
  gst_object_unref (pipeline);
  g_main_loop_unref (data.loop);
  g_remove ("fmt.json");

  return !data.error;
}

/**
 * @brief Test that a valid location set while writing images is used for the next image
 */
TEST (datareposink, writeImageLocationChanged)
{
  EXPECT_TRUE (write_images_changing_location ("alt_%02d.png"));

  EXPECT_TRUE (g_file_test ("fmt_0.png", G_FILE_TEST_IS_REGULAR));
  EXPECT_FALSE (g_file_test ("fmt_1.png", G_FILE_TEST_EXISTS));
  EXPECT_TRUE (g_file_test ("alt_01.png", G_FILE_TEST_IS_REGULAR));
  EXPECT_TRUE (g_file_test ("alt_02.png", G_FILE_TEST_IS_REGULAR));

  g_remove ("fmt_0.png");
  g_remove ("alt_01.png");
  g_remove ("alt_02.png");
}

/**
 * @brief Test that an invalid location set while writing images stops the pipeline
 */
TEST (datareposink, writeImageLocationChangedInvalid_n)
{
  EXPECT_FALSE (write_images_changing_location ("alt_%s%n.png"));

  EXPECT_TRUE (g_file_test ("fmt_0.png", G_FILE_TEST_IS_REGULAR));
  EXPECT_FALSE (g_file_test ("fmt_1.png", G_FILE_TEST_EXISTS));

  g_remove ("fmt_0.png");
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
  g_usleep (100000); /** wait 0.1 sec before forcing stop */

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (pipeline);
  g_main_loop_unref (loop);

  /* Confirm file creation */
  file = g_file_new_for_path ("flexible.data");
  ASSERT_NE (file, nullptr);

  for (int retry_count = 0; retry_count < 5 && file_info == NULL; retry_count++) {
    file_info = g_file_query_info (file, G_FILE_ATTRIBUTE_STANDARD_SIZE,
        G_FILE_QUERY_INFO_NONE, NULL, NULL);
    if (file_info == NULL) {
      g_usleep (50000);
    }
  }

  ASSERT_NE (file_info, nullptr);
  size = g_file_info_get_size (file_info);
  ASSERT_EQ (size, 0);
  g_clear_pointer (&file_info, g_object_unref);
  g_clear_pointer (&file, g_object_unref);

  for (i = 0; i < 5; i++) {
    filename = g_strdup_printf ("img_%02d.png", i);
    g_remove (filename);
    g_clear_pointer (&filename, g_free);
  }

  g_remove ("img.json");
  g_remove ("flexible.json");
  g_remove ("flexible.data");
}

/**
 * @brief Test for writing flexible tensors
 */
TEST (datareposink, writeSparseTensors_n)
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
      = "multifilesrc location=img_%02d.png ! other/tensors,format=sparse,framerate=0/1 ! "
        "datareposink location=sparse.data json=sparse.json";

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
  file = g_file_new_for_path ("sparse.data");
  ASSERT_NE (file, nullptr);

  for (int retry_count = 0; retry_count < 5 && file_info == NULL; retry_count++) {
    file_info = g_file_query_info (file, G_FILE_ATTRIBUTE_STANDARD_SIZE,
        G_FILE_QUERY_INFO_NONE, NULL, NULL);
    if (file_info == NULL) {
      g_usleep (50000);
    }
  }

  ASSERT_NE (file_info, nullptr);
  size = g_file_info_get_size (file_info);
  ASSERT_EQ (size, 0);
  g_clear_pointer (&file_info, g_object_unref);
  g_clear_pointer (&file, g_object_unref);

  for (i = 0; i < 5; i++) {
    filename = g_strdup_printf ("img_%02d.png", i);
    g_remove (filename);
    g_clear_pointer (&filename, g_free);
  }

  g_remove ("img.json");
  g_remove ("sparse.json");
  g_remove ("sparse.data");
}

/**
 * @brief Bus callback recording the first EOS or ERROR.
 */
static gboolean
_short_bus_cb (GstBus *bus, GstMessage *message, gpointer user_data)
{
  GstMessageType *type = (GstMessageType *) user_data;

  if (GST_MESSAGE_TYPE (message) == GST_MESSAGE_EOS
      || GST_MESSAGE_TYPE (message) == GST_MESSAGE_ERROR) {
    if (*type == GST_MESSAGE_UNKNOWN)
      *type = GST_MESSAGE_TYPE (message);
  }

  return TRUE;
}

/**
 * @brief Write one flexible tensor (128-byte header + 4 bytes) of which the memory maps only @a mem_size bytes.
 * @return the size of the data file written by datareposink.
 */
static gint64
_write_short_flexible_memory (gsize mem_size, GstMessageType *type)
{
  const gsize tensor_size = 128 + 4;
  GstElement *pipeline, *src;
  GstBus *bus;
  GstBuffer *buf;
  GstFlowReturn flow;
  GstTensorMetaInfo meta;
  GStatBuf st;
  guint8 *data = (guint8 *) g_malloc0 (tensor_size);
  gint64 size = -1;
  guint i;

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_UINT8;
  meta.dimension[0] = 4;
  meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  EXPECT_TRUE (gst_tensor_meta_info_update_header (&meta, data));

  buf = gst_buffer_new ();
  gst_buffer_append_memory (buf, gst_memory_new_wrapped ((GstMemoryFlags) 0, data,
                                     tensor_size, 0, mem_size, data, g_free));

  pipeline = gst_parse_launch ("appsrc name=src0 caps=other/tensors,format=flexible,framerate=0/1 ! "
                               "datareposink location=short.data json=short.json",
      NULL);
  EXPECT_NE (pipeline, nullptr);
  if (!pipeline) {
    gst_buffer_unref (buf);
    return size;
  }

  src = gst_bin_get_by_name (GST_BIN (pipeline), "src0");
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  *type = GST_MESSAGE_UNKNOWN;
  gst_bus_add_signal_watch (bus);
  g_signal_connect (bus, "message", G_CALLBACK (_short_bus_cb), type);

  EXPECT_NE (gst_element_set_state (pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
  /* push-buffer does not take the buffer. */
  g_signal_emit_by_name (src, "push-buffer", buf, &flow);
  gst_buffer_unref (buf);
  g_signal_emit_by_name (src, "end-of-stream", &flow);

  for (i = 0; i < 500 && *type == GST_MESSAGE_UNKNOWN; i++) {
    g_main_context_iteration (NULL, FALSE);
    g_usleep (10000);
  }

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_bus_remove_signal_watch (bus);
  gst_object_unref (bus);
  gst_object_unref (src);
  gst_object_unref (pipeline);

  if (g_stat ("short.data", &st) == 0)
    size = st.st_size;

  g_remove ("short.data");
  g_remove ("short.json");
  return size;
}

/**
 * @brief A flexible tensor whose memory holds the whole meta header is written.
 */
TEST (datareposink, writeFlexibleTensorMemory)
{
  GstMessageType type;

  EXPECT_EQ (_write_short_flexible_memory (132, &type), 132);
  EXPECT_EQ (type, GST_MESSAGE_EOS);
}

/**
 * @brief A flexible tensor memory shorter than the meta header is refused.
 * The bytes behind the mapping hold a valid header, so reading past the memory would accept it.
 */
TEST (datareposink, writeFlexibleTensorShortMemory_n)
{
  GstMessageType type;

  EXPECT_EQ (_write_short_flexible_memory (8, &type), 0);
  EXPECT_EQ (type, GST_MESSAGE_ERROR);
}

/**
 * @brief Run @a pipeline until EOS or ERROR; the caller sets it to NULL.
 * @return the first EOS or ERROR message type.
 */
static GstMessageType
_run_to_eos (GstElement *pipeline)
{
  GstMessageType type = GST_MESSAGE_UNKNOWN;
  GstBus *bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  guint i;

  gst_bus_add_signal_watch (bus);
  g_signal_connect (bus, "message", G_CALLBACK (_short_bus_cb), &type);

  EXPECT_NE (gst_element_set_state (pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
  for (i = 0; i < 500 && type == GST_MESSAGE_UNKNOWN; i++) {
    g_main_context_iteration (NULL, FALSE);
    g_usleep (10000);
  }

  gst_bus_remove_signal_watch (bus);
  gst_object_unref (bus);

  return type;
}

/**
 * @brief An image sink never opens a data file, so its stop must not close fd 0.
 */
TEST (datareposink, imageStopKeepsFd0_n)
{
  struct stat marker, st;
  gint saved = dup (0);
  gint fd = g_open ("fd0.marker", O_RDWR | O_CREAT | O_TRUNC, 0644);
  GstElement *pipeline;
  gint i;

  EXPECT_GE (fd, 0);
  if (fd > 0) {
    EXPECT_EQ (dup2 (fd, 0), 0);
    close (fd);
  }
  EXPECT_EQ (fstat (0, &marker), 0);

  pipeline = gst_parse_launch ("videotestsrc num-buffers=2 ! pngenc ! "
                               "datareposink location=fd0_%02d.png json=fd0.json",
      NULL);
  EXPECT_NE (pipeline, nullptr);
  if (pipeline) {
    EXPECT_EQ (_run_to_eos (pipeline), GST_MESSAGE_EOS);
    gst_element_set_state (pipeline, GST_STATE_NULL);
    gst_object_unref (pipeline);
  }

  EXPECT_EQ (fstat (0, &st), 0);
  EXPECT_EQ (st.st_dev, marker.st_dev);
  EXPECT_EQ (st.st_ino, marker.st_ino);

  if (saved >= 0) {
    dup2 (saved, 0);
    close (saved);
  } else {
    close (0);
  }

  g_remove ("fd0.marker");
  g_remove ("fd0.json");
  for (i = 0; i < 2; i++) {
    gchar *name = g_strdup_printf ("fd0_%02d.png", i);
    g_remove (name);
    g_free (name);
  }
}

/**
 * @brief A data file opened on fd 0 (stdin closed by the process) is written.
 */
TEST (datareposink, writeVideoRawOnFd0)
{
  struct stat st, fd0;
  gint saved;
  GstElement *pipeline = gst_parse_launch ("videotestsrc num-buffers=3 ! "
                                           "video/x-raw,format=RGB,width=4,height=4 ! "
                                           "datareposink location=fd0.raw json=fd0raw.json",
      NULL);
  ASSERT_NE (pipeline, nullptr);

  /* Close fd 0 after the pipeline has its own descriptors, so the data file takes it. */
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PAUSED, UNITTEST_STATECHANGE_TIMEOUT), 0);
  saved = dup (0);
  close (0);

  EXPECT_EQ (_run_to_eos (pipeline), GST_MESSAGE_EOS);
  EXPECT_EQ (fstat (0, &fd0), 0);
  EXPECT_EQ (stat ("fd0.raw", &st), 0);
  EXPECT_EQ (fd0.st_dev, st.st_dev);
  EXPECT_EQ (fd0.st_ino, st.st_ino);
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (pipeline);

  if (saved >= 0) {
    dup2 (saved, 0);
    close (saved);
  }

  ASSERT_EQ (stat ("fd0.raw", &st), 0);
  EXPECT_EQ (st.st_size, 3 * 4 * 4 * 3);

  g_remove ("fd0.raw");
  g_remove ("fd0raw.json");
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

  /* These tests write fixed file names, which unittest_datareposrc also uses. */
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
