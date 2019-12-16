/**
 * @file        unittest_edgetpu.cpp
 * @date        16 Dec 2019
 * @brief       Unit test for tensor_filter::edgetpu.
 * @see         https://github.com/nnsuite/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h> /* GStatBuf */
#include <tensor_common.h>
#include <gst/gst.h>

/**
 * @brief Standard positive case with a small tensorflow-lite model
 */
TEST (edgetpu_tflite_direct, run_01)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model = g_build_filename (root_path, "tests", "test_models", "models", "mobilenet_v1_1.0_224_quant.tflite", NULL);

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoformat ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=edgetpu model=\"%s\" ! fakesink",
      test_model);
  gstpipe = gst_parse_launch (pipeline, &err);
  if (gstpipe) {
    status = 0;

    EXPECT_NE(gst_element_set_state (gstpipe, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
    g_usleep(500000);
    EXPECT_NE(gst_element_set_state (gstpipe, GST_STATE_NULL), GST_STATE_CHANGE_FAILURE);
    g_usleep(100000);

    gst_object_unref (gstpipe);
  } else {
    status = -1;
    g_printerr("GST PARSE LAUNCH FAILED: [%s], %s\n",
      pipeline, (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, 0);
  g_free (test_model);
  g_free (pipeline);
}

/**
 * @brief Negative case with incorrect path
 */
TEST (edgetpu_tflite_direct, error_01_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model = g_build_filename (root_path, "tests", "test_models", "models", "does_not_exists.0_224_quant.tflite", NULL);

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoformat ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=edgetpu model=\"%s\" ! fakesink",
      test_model);
  gstpipe = gst_parse_launch (pipeline, &err);

  if (gstpipe) {
    status = 0;

    EXPECT_NE(gst_element_set_state (gstpipe, GST_STATE_PLAYING), GST_STATE_CHANGE_SUCCESS);
    EXPECT_EQ(gst_element_set_state (gstpipe, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
    g_usleep(500000);
    EXPECT_NE(gst_element_set_state (gstpipe, GST_STATE_NULL), GST_STATE_CHANGE_FAILURE);
    g_usleep(100000);

    gst_object_unref (gstpipe);
  } else {
    status = -1;
    g_printerr("GST PARSE LAUNCH FAILED: [%s], %s\n",
      pipeline, (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, 0);

  g_free (test_model);
  g_free (pipeline);
}

/**
 * @brief Negative case with incorrect tensor meta
 */
TEST (edgetpu_tflite_direct, error_02_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model = g_build_filename (root_path, "tests", "test_models", "models", "mobilenet_v1_1.0_224_quant.tflite", NULL);

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoformat ! videorate ! video/x-raw,format=RGB,width=240,height=224 ! tensor_converter ! tensor_filter framework=edgetpu model=\"%s\" ! fakesink",
      test_model);
  gstpipe = gst_parse_launch (pipeline, &err);
  if (gstpipe) {
    status = 0;
    GstState state, pending;

    EXPECT_NE(gst_element_set_state (gstpipe, GST_STATE_PLAYING), GST_STATE_CHANGE_SUCCESS);
    g_usleep(500000);
    EXPECT_EQ(gst_element_get_state (gstpipe, &state, &pending, GST_SECOND/4), GST_STATE_CHANGE_FAILURE); /* This should fail: dimension mismatched. */

    EXPECT_NE(gst_element_set_state (gstpipe, GST_STATE_NULL), GST_STATE_CHANGE_FAILURE);
    g_usleep(100000);

    gst_object_unref (gstpipe);
  } else {
    status = -1;
    g_printerr("GST PARSE LAUNCH FAILED: [%s], %s\n",
      pipeline, (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, 0);
  g_free (test_model);
  g_free (pipeline);
}

/**
 * @brief Main GTest
 */
int main (int argc, char **argv)
{
  int result;

  testing::InitGoogleTest (&argc, argv);

  gst_init (&argc, &argv);
  result = RUN_ALL_TESTS ();

  return result;
}
