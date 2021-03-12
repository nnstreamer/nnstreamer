/**
 * @file        unittest_cpp_methods.cc
 * @date        15 Jan 2019
 * @brief       Unit test cases for tensor_filter::cpp
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>

#include <unittest_util.h>
#include "cppfilter_test.hh"

static char *path_to_lib = NULL;

/** @brief Positive case for the simpliest execution path */
TEST (cppFilterOnDemand, basic01)
{
  filter_basic basic ("basic_01");
  EXPECT_EQ (basic._register (), 0);
  EXPECT_EQ (basic._unregister (), 0);
}

/** @brief Negative case for the simpliest execution path */
TEST (cppFilterOnDemand, basic02_n)
{
  filter_basic basic ("basic_02");
  EXPECT_NE (basic._unregister (), 0);
  EXPECT_EQ (basic._register (), 0);
  EXPECT_NE (basic._register (), 0);
  EXPECT_EQ (basic._unregister (), 0);
  EXPECT_NE (basic._unregister (), 0);
}

/** @brief Negative case for the simpliest execution path w/ static calls */
TEST (cppFilterOnDemand, basic03_n)
{
  filter_basic basic ("basic_03");
  EXPECT_NE (filter_basic::__unregister ("basic_03"), 0);
  EXPECT_EQ (filter_basic::__register (&basic), 0);
  EXPECT_NE (filter_basic::__register (&basic), 0);
  EXPECT_EQ (filter_basic::__unregister ("basic_03"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_03"), 0);
}

/** @brief Negative case for the simpliest execution path w/ static calls */
TEST (cppFilterOnDemand, basic04_n)
{
  filter_basic basic ("basic_04");
  EXPECT_NE (filter_basic::__unregister ("basic_xx"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_03"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_04"), 0);
  EXPECT_EQ (filter_basic::__register (&basic), 0);
  EXPECT_NE (filter_basic::__register (&basic), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_xx"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_03"), 0);
  EXPECT_EQ (filter_basic::__unregister ("basic_04"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_03"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_04"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_xx"), 0);
}

/** @brief Actual GST Pipeline with cpp on demand */
TEST (cppFilterOnDemand, pipeline01)
{
  filter_basic basic ("pl01");
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);
  EXPECT_EQ (basic._register (), 0);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=5 ! videoconvert ! videoscale ! "
      "video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tee name=t "
      "t. ! queue name=q1 ! tensor_filter framework=cpp model=pl01 ! filesink location=%s "
      "t. ! queue name=q2 ! filesink location=%s ",
      tmp1, tmp2);

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  if (pipeline) {
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
        0);

    g_usleep (100000);

    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
    g_usleep (100000);

    gst_object_unref (pipeline);

    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp1), 0);
  }
  g_free (str_pipeline);

  g_free (tmp1);
  g_free (tmp2);
  EXPECT_EQ (basic._unregister (), 0);
}

/** @brief Negative case for the simpliest execution path */
TEST (cppFilterOnDemand, unregstered01_n)
{
  filter_basic basic ("basic_01");
  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=3 ! videoconvert ! videoscale ! "
      "video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! "
      "tensor_filter framework=cpp model=XXbasic_01 ! fakesink");

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  EXPECT_EQ (basic._register (), 0);
  gst_object_unref (pipeline);

  pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);
  if (pipeline)
    gst_object_unref (pipeline);
  EXPECT_EQ (basic._unregister (), 0);

  basic._unregister ();
  g_free (str_pipeline);
  EXPECT_NE (basic._unregister (), 0);
}

/** @brief gtest method */
TEST (cppFilterObj, base01_n)
{
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();
  char *tmp3 = getTempFilename ();

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);
  EXPECT_NE (tmp3, nullptr);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=5 ! videoconvert ! videoscale ! "
      "video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tee name=t "
      "t. ! queue name=q1 ! tensor_filter framework=cpp model=basic_so_01,%slibcppfilter_test.so ! filesink location=%s "
      "t. ! queue name=q2 ! filesink location=%s "
      "t. ! queue ! tensor_filter framework=cpp model=basic_so_03,%slibcppfilter_test.so ! filesink location=%s",
      path_to_lib, tmp1, tmp2, path_to_lib, tmp3);

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  if (pipeline) {
    EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
        0);

    gst_object_unref (pipeline);
  }
  g_free (str_pipeline);

  g_free (tmp1);
  g_free (tmp2);
  g_free (tmp3);
}

/** @brief gtest method */
TEST (cppFilterObj, base02_n)
{
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();
  char *tmp3 = getTempFilename ();

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);
  EXPECT_NE (tmp3, nullptr);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=5 ! videoconvert ! videoscale ! "
      "video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tee name=t "
      "t. ! queue name=q1 ! tensor_filter framework=cpp model=basic_so_01,%slibcppfilter_test.so ! filesink location=%s "
      "t. ! queue name=q2 ! filesink location=%s "
      "t. ! queue ! tensor_filter framework=cpp model=basic_so_03,%slibcppfilter_test.so ! filesink location=%s",
      path_to_lib, tmp1, tmp2, path_to_lib, tmp3);

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  if (pipeline) {
    EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
        0);

    gst_object_unref (pipeline);
  }
  g_free (str_pipeline);

  g_free (tmp1);
  g_free (tmp2);
  g_free (tmp3);
}

/** @brief gtest method */
TEST (cppFilterObj, base03)
{
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();
  char *tmp3 = getTempFilename ();
  char *tmp4 = getTempFilename ();
  char *tmp5 = getTempFilename ();

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);
  EXPECT_NE (tmp3, nullptr);
  EXPECT_NE (tmp4, nullptr);
  EXPECT_NE (tmp5, nullptr);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=5 ! videoconvert ! videoscale ! "
      "video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tee name=t "
      "t. ! queue ! tensor_filter framework=cpp model=basic_so_01,%slibcppfilter_test.so ! filesink location=%s sync=true "
      "t. ! queue ! filesink location=%s sync=true "
      "t. ! queue ! tensor_filter framework=cpp model=basic_so_02,%slibcppfilter_test.so ! filesink location=%s sync=true "
      "videotestsrc num-buffers=5 ! videoconvert ! videoscale ! "
      "video/x-raw,width=16,height=16,format=RGB ! tensor_converter ! tee name=t2 "
      "t2. ! queue ! tensor_filter framework=cpp model=basic_so2,%slibcppfilter_test.so ! filesink location=%s sync=true "
      "t2. ! queue ! filesink location=%s sync=true ",
      path_to_lib, tmp1, tmp2, path_to_lib, tmp3, path_to_lib, tmp4, tmp5);

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  if (pipeline) {
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),
        0);

    g_usleep (300000);

    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

    gst_object_unref (pipeline);
    g_usleep (300000);

    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp1), 0);
    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp3), 0);
    EXPECT_EQ (filter_basic2::resultCompare (tmp5, tmp4), 0);
  }
  g_free (str_pipeline);

  g_free (tmp1);
  g_free (tmp2);
  g_free (tmp3);
  g_free (tmp4);
  g_free (tmp5);
}

/**
 * @brief Main GTest
 */
int
main (int argc, char **argv)
{
  int result = 0;
  int delete_path = 0;

  if (argc > 3 && !g_strcmp0 (argv[1], "-libpath")) {
    path_to_lib = argv[2];
  } else {
    gchar *dir = g_path_get_dirname (argv[0]);
    path_to_lib = g_strdup_printf ("%s/", dir);
    delete_path = 1;
    g_free (dir);
    g_printerr ("LIBPATH = %s\n", path_to_lib);
  }

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

  if (delete_path)
    g_free (path_to_lib);
  return result;
}
