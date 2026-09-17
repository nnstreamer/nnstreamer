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
#include "../../gst/nnstreamer/tensor_filter/tensor_filter.h"
#include "cppfilter_test.hh"

static char *path_to_lib = NULL;

/** @brief Test C++ filter allocating its own output, otherwise the same as filter_basic */
class filter_self_alloc : public filter_basic
{
  public:
  /** @brief Construct the test filter with the given filter name */
  filter_self_alloc (const char *str) : filter_basic (str)
  {
  }

  /** @brief Tell the framework the filter allocates the output buffer */
  bool isAllocatedBeforeInvoke ()
  {
    return false;
  }

  /** @brief Allocate the output, then fill it as filter_basic does */
  int invoke (const GstTensorMemory *in, GstTensorMemory *out)
  {
    out[0].data = g_malloc (out[0].size);
    return filter_basic::invoke (in, out);
  }
};

/**
 * @brief Prepare the properties a tensor_filter opens a filter_basic-shaped cpp model with
 */
static void
_init_basic_prop (GstTensorFilterProperties *prop, const gchar **models)
{
  filter_basic shape ("b10b11_shape");

  memset (prop, 0, sizeof (*prop));
  prop->fwname = "cpp";
  prop->model_files = models;
  prop->num_models = 1;
  shape.getInputDim (&prop->input_meta);
  shape.getOutputDim (&prop->output_meta);
}

/**
 * @brief Ask the tensor_filter framework whether an opened cpp filter allocates its output in invoke
 */
static gboolean
_allocates_in_invoke (void *private_data)
{
  GstTensorFilterPrivate priv;

  memset (&priv, 0, sizeof (priv));
  priv.fw = nnstreamer_filter_find ("cpp");
  priv.privateData = private_data;

  return gst_tensor_filter_allocate_in_invoke (&priv);
}

/** @brief Positive case for the simplest execution path */
TEST (cppFilterOnDemand, basic01)
{
  filter_basic basic ("basic_01");
  EXPECT_EQ (basic._register (), 0);
  EXPECT_EQ (basic._unregister (), 0);
}

/** @brief Negative case for the simplest execution path */
TEST (cppFilterOnDemand, basic02_n)
{
  filter_basic basic ("basic_02");
  EXPECT_NE (basic._unregister (), 0);
  EXPECT_EQ (basic._register (), 0);
  EXPECT_NE (basic._register (), 0);
  EXPECT_EQ (basic._unregister (), 0);
  EXPECT_NE (basic._unregister (), 0);
}

/** @brief Negative case for the simplest execution path w/ static calls */
TEST (cppFilterOnDemand, basic03_n)
{
  filter_basic basic ("basic_03");
  EXPECT_NE (filter_basic::__unregister ("basic_03"), 0);
  EXPECT_EQ (filter_basic::__register (&basic), 0);
  EXPECT_NE (filter_basic::__register (&basic), 0);
  EXPECT_EQ (filter_basic::__unregister ("basic_03"), 0);
  EXPECT_NE (filter_basic::__unregister ("basic_03"), 0);
}

/** @brief Negative case for the simplest execution path w/ static calls */
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

/** @brief Wait until the pipeline saving the file */
static void
_wait_save_files (const gchar *file, gsize expected_len)
{
  gchar *content = NULL;
  gsize len;

  _wait_pipeline_save_files (file, content, len, expected_len, 1000U);

  g_free (content);
  content = NULL;
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
      "t. ! queue name=q1 ! tensor_filter framework=cpp model=pl01 ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue name=q2 ! filesink location=%s buffer-mode=unbuffered sync=false async=false",
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

    _wait_save_files (tmp1, 480);
    _wait_save_files (tmp2, 240);

    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
    g_usleep (100000);

    gst_object_unref (pipeline);

    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp1), 0);
  }
  g_free (str_pipeline);

  removeTempFile (&tmp1);
  removeTempFile (&tmp2);

  EXPECT_EQ (basic._unregister (), 0);
}

/** @brief Negative case for the simplest execution path */
TEST (cppFilterOnDemand, unregistered01_n)
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
      "t. ! queue name=q1 ! tensor_filter framework=cpp model=basic_so_01,%slibcppfilter_test.so ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue name=q2 ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! tensor_filter framework=cpp model=basic_so_03,%slibcppfilter_test.so ! filesink location=%s buffer-mode=unbuffered sync=false async=false",
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
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

    gst_object_unref (pipeline);
  }
  g_free (str_pipeline);

  removeTempFile (&tmp1);
  removeTempFile (&tmp2);
  removeTempFile (&tmp3);
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
      "t. ! queue name=q1 ! tensor_filter framework=cpp model=basic_so_01,%slibcppfilter_test.so ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue name=q2 ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! tensor_filter framework=cpp model=basic_so_03,%slibcppfilter_test.so ! filesink location=%s buffer-mode=unbuffered sync=false async=false",
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
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

    gst_object_unref (pipeline);
  }
  g_free (str_pipeline);

  removeTempFile (&tmp1);
  removeTempFile (&tmp2);
  removeTempFile (&tmp3);
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
      "t. ! queue ! tensor_filter framework=cpp model=basic_so_01,%slibcppfilter_test.so ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! tensor_filter framework=cpp model=basic_so_02,%slibcppfilter_test.so ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "videotestsrc num-buffers=5 ! videoconvert ! videoscale ! "
      "video/x-raw,width=16,height=16,format=RGB ! tensor_converter ! tee name=t2 "
      "t2. ! queue ! tensor_filter framework=cpp model=basic_so2,%slibcppfilter_test.so ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t2. ! queue ! filesink location=%s buffer-mode=unbuffered sync=false async=false ",
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
    _wait_save_files (tmp1, 480);
    _wait_save_files (tmp2, 240);
    _wait_save_files (tmp3, 480);
    _wait_save_files (tmp4, 7680);
    _wait_save_files (tmp5, 3840);
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

    gst_object_unref (pipeline);
    g_usleep (300000);

    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp1), 0);
    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp3), 0);
    EXPECT_EQ (filter_basic2::resultCompare (tmp5, tmp4), 0);
  }
  g_free (str_pipeline);

  removeTempFile (&tmp1);
  removeTempFile (&tmp2);
  removeTempFile (&tmp3);
  removeTempFile (&tmp4);
  removeTempFile (&tmp5);
}

/** @brief Each opened cpp filter keeps its own output allocation contract */
TEST (cppFilterShared, allocContractPerFilter)
{
  filter_basic pre ("b10_pre");
  filter_self_alloc self ("b10_self");
  const gchar *pre_models[] = { "b10_pre" };
  const gchar *self_models[] = { "b10_self" };
  GstTensorFilterProperties prop_pre, prop_self, prop_pre2;
  void *pd_pre = NULL, *pd_self = NULL, *pd_pre2 = NULL;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (pre._register (), 0);
  EXPECT_EQ (self._register (), 0);
  _init_basic_prop (&prop_pre, pre_models);
  _init_basic_prop (&prop_self, self_models);
  _init_basic_prop (&prop_pre2, pre_models);

  EXPECT_EQ (fw->open (&prop_pre, &pd_pre), 0);
  EXPECT_EQ (fw->open (&prop_self, &pd_self), 0);
  EXPECT_FALSE (_allocates_in_invoke (pd_pre));
  EXPECT_TRUE (_allocates_in_invoke (pd_self));

  EXPECT_EQ (fw->open (&prop_pre2, &pd_pre2), 0);
  EXPECT_FALSE (_allocates_in_invoke (pd_pre));
  EXPECT_TRUE (_allocates_in_invoke (pd_self));
  EXPECT_FALSE (_allocates_in_invoke (pd_pre2));

  fw->close (&prop_pre2, &pd_pre2);
  fw->close (&prop_self, &pd_self);
  fw->close (&prop_pre, &pd_pre);
  EXPECT_EQ (self._unregister (), 0);
  EXPECT_EQ (pre._unregister (), 0);
}

/** @brief A failed open does not change the contract of an opened cpp filter */
TEST (cppFilterShared, allocContractFailedOpen_n)
{
  filter_self_alloc self ("b10_self_n");
  const gchar *self_models[] = { "b10_self_n" };
  const gchar *unknown_models[] = { "b10_unknown_n" };
  GstTensorFilterProperties prop_self, prop_unknown;
  void *pd_self = NULL, *pd_unknown = NULL;
  const GstTensorFilterFramework *fw = nnstreamer_filter_find ("cpp");

  ASSERT_NE (fw, nullptr);
  EXPECT_EQ (self._register (), 0);
  _init_basic_prop (&prop_self, self_models);
  _init_basic_prop (&prop_unknown, unknown_models);

  EXPECT_EQ (fw->open (&prop_self, &pd_self), 0);
  EXPECT_NE (fw->open (&prop_unknown, &pd_unknown), 0);
  EXPECT_EQ (pd_unknown, nullptr);
  EXPECT_TRUE (_allocates_in_invoke (pd_self));

  fw->close (&prop_self, &pd_self);
  EXPECT_EQ (self._unregister (), 0);
}

/** @brief A preallocated and a self-allocating cpp filter both produce their output in one pipeline */
TEST (cppFilterShared, allocContractPipeline)
{
  filter_basic pre ("b10_pl_pre");
  filter_self_alloc self ("b10_pl_self");
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();
  char *tmp3 = getTempFilename ();

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);
  EXPECT_NE (tmp3, nullptr);
  EXPECT_EQ (pre._register (), 0);
  EXPECT_EQ (self._register (), 0);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=5 ! videoconvert ! videoscale ! "
      "video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tee name=t "
      "t. ! queue ! tensor_filter framework=cpp model=b10_pl_pre ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! tensor_filter framework=cpp model=b10_pl_self ! filesink location=%s buffer-mode=unbuffered sync=false async=false",
      tmp1, tmp2, tmp3);

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
    _wait_save_files (tmp1, 480);
    _wait_save_files (tmp2, 240);
    _wait_save_files (tmp3, 480);
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

    gst_object_unref (pipeline);

    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp1), 0);
    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp3), 0);
  }
  g_free (str_pipeline);

  removeTempFile (&tmp1);
  removeTempFile (&tmp2);
  removeTempFile (&tmp3);

  EXPECT_EQ (self._unregister (), 0);
  EXPECT_EQ (pre._unregister (), 0);
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
