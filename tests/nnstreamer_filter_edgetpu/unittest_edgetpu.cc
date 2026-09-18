/**
 * @file        unittest_edgetpu.cc
 * @date        16 Dec 2019
 * @brief       Unit test for tensor_filter::edgetpu.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h> /* GStatBuf */
#include <gst/gst.h>
#include <tensor_common.h>

/**
 * @brief Standard positive case with a small tensorflow-lite model
 */
TEST (edgetpuTfliteDirect, run01)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model = g_build_filename (root_path, "tests", "test_models",
      "models", "mobilenet_v1_1.0_224_quant.tflite", NULL);

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=edgetpu model=\"%s\" custom=device_type:dummy ! fakesink",
      test_model);
  gstpipe = gst_parse_launch (pipeline, &err);
  if (gstpipe) {
    status = 0;

    EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
    g_usleep (500000);
    EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_NULL), GST_STATE_CHANGE_FAILURE);
    g_usleep (100000);

    gst_object_unref (gstpipe);
  } else {
    status = -1;
    g_printerr ("GST PARSE LAUNCH FAILED: [%s], %s\n", pipeline,
        (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, 0);
  g_free (test_model);
  g_free (pipeline);
}

/**
 * @brief Negative case with incorrect path
 */
TEST (edgetpuTfliteDirect, error01_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model = g_build_filename (root_path, "tests", "test_models",
      "models", "does_not_exists.0_224_quant.tflite", NULL);

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=edgetpu model=\"%s\" custom=device_type:dummy ! fakesink",
      test_model);
  gstpipe = gst_parse_launch (pipeline, &err);

  if (gstpipe) {
    status = 0;

    EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_PLAYING), GST_STATE_CHANGE_SUCCESS);
    EXPECT_EQ (gst_element_set_state (gstpipe, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
    g_usleep (500000);
    EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_NULL), GST_STATE_CHANGE_FAILURE);
    g_usleep (100000);

    gst_object_unref (gstpipe);
  } else {
    status = -1;
    g_printerr ("GST PARSE LAUNCH FAILED: [%s], %s\n", pipeline,
        (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, 0);

  g_free (test_model);
  g_free (pipeline);
}

/**
 * @brief Negative case with incorrect tensor meta
 */
TEST (edgetpuTfliteDirect, error02_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model = g_build_filename (root_path, "tests", "test_models",
      "models", "mobilenet_v1_1.0_224_quant.tflite", NULL);

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=240,height=224 ! tensor_converter ! tensor_filter framework=edgetpu model=\"%s\" custom=device_type:dummy ! fakesink",
      test_model);
  gstpipe = gst_parse_launch (pipeline, &err);
  if (gstpipe) {
    status = 0;
    GstState state, pending;

    EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_PLAYING), GST_STATE_CHANGE_SUCCESS);
    g_usleep (500000);
    EXPECT_EQ (gst_element_get_state (gstpipe, &state, &pending, GST_SECOND / 4),
        GST_STATE_CHANGE_FAILURE); /* This should fail: dimension mismatched. */

    EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_NULL), GST_STATE_CHANGE_FAILURE);
    g_usleep (100000);

    gst_object_unref (gstpipe);
  } else {
    status = -1;
    g_printerr ("GST PARSE LAUNCH FAILED: [%s], %s\n", pipeline,
        (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, 0);
  g_free (test_model);
  g_free (pipeline);
}

/**
 * @brief A flatbuffer carrying the TFLite file identifier and an unsupported
 *        schema version.
 * @details tflite::FlatBufferModel accepts it, while tflite::InterpreterBuilder
 *          refuses it on the version check and leaves the interpreter unset.
 *          The layout is a root table offset, the "TFL3" identifier, a vtable
 *          describing a single field, and the table holding that field.
 */
static const guint8 unsupported_schema_model[] = {
  0x14, 0x00, 0x00, 0x00, /* offset of the root table */
  'T', 'F', 'L', '3', /* file identifier */
  0x06, 0x00, /* vtable: size of the vtable itself */
  0x08, 0x00, /* vtable: inline size of the table */
  0x04, 0x00, /* vtable: offset of 'version' within the table */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* padding */
  0x0c, 0x00, 0x00, 0x00, /* table: offset back to the vtable */
  0x2a, 0x00, 0x00, 0x00, /* version = 42 */
};

/**
 * @brief Write @a unsupported_schema_model into a temporary file.
 * @return The path of the written file, or NULL on failure. Free it with g_free().
 */
static gchar *
_edgetpu_write_tmp_model (void)
{
  gchar *path = NULL;
  gint fd;

  fd = g_file_open_tmp ("nnstreamer-edgetpu-XXXXXX.tflite", &path, NULL);
  if (fd == -1)
    return NULL;
  g_close (fd, NULL);

  if (!g_file_set_contents (path, (const gchar *) unsupported_schema_model,
          sizeof (unsupported_schema_model), NULL)) {
    g_remove (path);
    g_free (path);
    return NULL;
  }

  return path;
}

/**
 * @brief Run an edgetpu pipeline with the given model and 'custom' property.
 * @details The sub-plugin verifies the model path before the framework is
 *          opened, so the model has to exist for the 'custom' property to be
 *          parsed at all.
 * @return TRUE if the pipeline reaches the playing state.
 */
static gboolean
_edgetpu_pipeline_plays (const gchar *model, const gchar *custom)
{
  GstElement *gstpipe;
  GError *err = NULL;
  gboolean played;
  gchar *pipeline;

  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=edgetpu model=\"%s\" custom=\"%s\" ! fakesink",
      model, custom);

  gstpipe = gst_parse_launch (pipeline, &err);
  if (gstpipe == NULL) {
    /* Not the refusal the caller is after: the description itself is broken. */
    ADD_FAILURE () << "GST PARSE LAUNCH FAILED: [" << pipeline << "], "
                   << ((err) ? err->message : "unknown reason");
    g_clear_error (&err);
    g_free (pipeline);
    return FALSE;
  }
  g_clear_error (&err); /* a description may parse and still leave a warning */

  played = (gst_element_set_state (gstpipe, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE);
  if (played) {
    GstState state;

    g_usleep (500000);
    played = (gst_element_get_state (gstpipe, &state, NULL, GST_SECOND / 4)
              != GST_STATE_CHANGE_FAILURE);
  }

  gst_element_set_state (gstpipe, GST_STATE_NULL);
  g_usleep (100000);
  gst_object_unref (gstpipe);
  g_free (pipeline);

  return played;
}

/**
 * @brief Build the path of the tflite model that edgetpuTfliteDirect.run01 uses.
 */
static gchar *
_edgetpu_dummy_model (void)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  if (root_path == NULL)
    root_path = "..";

  return g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
}

/**
 * @brief Negative case with a 'custom' property holding a key without a value
 * @details run01 plays the very same pipeline with 'device_type:dummy', so a
 *          key without a value has to fall back to the default device type,
 *          which no test machine has.
 */
TEST (edgetpuTfliteDirect, customPropNoValue_n)
{
  gchar *test_model = _edgetpu_dummy_model ();

  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_IS_REGULAR));
  EXPECT_FALSE (_edgetpu_pipeline_plays (test_model, "device_type"));

  g_free (test_model);
}

/**
 * @brief Negative case with a 'custom' property that holds no token at all
 * @details The first comma-separated field is empty, so 'device_type:dummy'
 *          is never looked at and the default device type is taken.
 */
TEST (edgetpuTfliteDirect, customPropNoToken_n)
{
  gchar *test_model = _edgetpu_dummy_model ();

  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_IS_REGULAR));
  EXPECT_FALSE (_edgetpu_pipeline_plays (test_model, ",device_type:dummy"));

  g_free (test_model);
}

/**
 * @brief Negative case with a model that no interpreter can be built from
 */
TEST (edgetpuTfliteDirect, error03_n)
{
  gchar *test_model = _edgetpu_write_tmp_model ();

  ASSERT_TRUE (test_model != NULL);
  EXPECT_FALSE (_edgetpu_pipeline_plays (test_model, "device_type:dummy"));

  g_remove (test_model);
  g_free (test_model);
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

  /* Force the binary to use dlog_print of untitest-util by calling it directly */
  ml_logd ("Edge TPU test starts w/ dummy backend.");

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
