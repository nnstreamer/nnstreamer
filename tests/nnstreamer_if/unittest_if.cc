/**
 * @file        unittest_if.cc
 * @date        15 Oct 2020
 * @brief       Unit test for tensor_if
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <errno.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <tensor_common.h>
#include <unittest_util.h>
#include "../gst/nnstreamer/elements/gsttensor_if.h"

#define TEST_TIMEOUT_MS (20000U)

static int data_received = 0;

/**
 * @brief nnstreamer tensor_if testing base class
 */
class tensor_if_run : public ::testing::Test
{
  protected:
  /**
   * @brief  Sets up the base fixture
   */
  void SetUp () override
  {
    gchar *content = NULL;
    gsize len;
    gchar *smpte_pipeline = g_strdup_printf (
        "videotestsrc name=vsrc num-buffers=1 pattern=13 ! videoconvert ! videoscale ! "
        "video/x-raw,format=RGB,width=160,height=120 ! filesink location=smpte.golden");
    gchar *gamut_pipeline = g_strdup_printf (
        "videotestsrc name=vsrc num-buffers=1 pattern=15 ! videoconvert ! videoscale ! "
        "video/x-raw,format=RGB,width=160,height=120 ! filesink location=gamut.golden");
    GstElement *gstpipe = gst_parse_launch (smpte_pipeline, NULL);

    setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
    _wait_pipeline_save_files ("./smpte.golden", content, len, 57600, TEST_TIMEOUT_MS);

    setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
    g_free (content);
    gst_object_unref (gstpipe);

    gstpipe = gst_parse_launch (gamut_pipeline, NULL);
    setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
    _wait_pipeline_save_files ("./gamut.golden", content, len, 57600, TEST_TIMEOUT_MS);
    g_free (content);

    setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
    g_usleep (10000);
    gst_object_unref (gstpipe);
    g_free (smpte_pipeline);
    g_free (gamut_pipeline);
  }

  /**
   * @brief tear down the base fixture
   */
  void TearDown () override
  {
    g_remove ("smpte.golden");
    g_remove ("gamut.golden");
  }
};

/**
 * @brief Test for tensor_if get and set properties
 */
TEST (tensorIfProp, properties0)
{
  gchar *pipeline;
  GstElement *gstpipe;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc num-buffers=1 pattern=13 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! "
      "tensor_if name=tif compared-value=A_VALUE compared-value-option=1:2:1:1,1 "
      "supplied-value=100 operator=GE then=PASSTHROUGH else=SKIP ! tensor_sink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  GstElement *tif_handle;
  gint int_val;
  gchar *str_val;
  gboolean bool_val;

  tif_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "tif");
  EXPECT_NE (tif_handle, nullptr);

  /* Get properties */
  g_object_get (tif_handle, "compared-value", &int_val, NULL);
  EXPECT_EQ (TIFCV_A_VALUE, int_val);

  g_object_get (tif_handle, "compared-value-option", &str_val, NULL);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal ("1:2:1:1,1", str_val));
  g_free (str_val);

  g_object_get (tif_handle, "supplied-value", &str_val, NULL);
  EXPECT_STREQ ("100", str_val);
  g_free (str_val);

  g_object_get (tif_handle, "operator", &int_val, NULL);
  EXPECT_EQ (TIFOP_GE, int_val);

  g_object_get (tif_handle, "then", &int_val, NULL);
  EXPECT_EQ (TIFB_PASSTHROUGH, int_val);

  g_object_get (tif_handle, "else", &int_val, NULL);
  EXPECT_EQ (TIFB_SKIP, int_val);

  /* Set properties */
  g_object_set (tif_handle, "compared-value", TIFCV_TENSOR_AVERAGE_VALUE, NULL);
  g_object_get (tif_handle, "compared-value", &int_val, NULL);
  EXPECT_EQ (TIFCV_TENSOR_AVERAGE_VALUE, int_val);

  g_object_set (tif_handle, "compared-value-option", "0", NULL);
  g_object_get (tif_handle, "compared-value-option", &str_val, NULL);
  EXPECT_STREQ ("0", str_val);
  g_free (str_val);

  /* Check float type */
  g_object_set (tif_handle, "supplied-value", "1.541234", NULL);
  g_object_get (tif_handle, "supplied-value", &str_val, NULL);
  EXPECT_DOUBLE_EQ (1.541234, g_ascii_strtod (str_val, NULL));
  g_free (str_val);

  g_object_set (tif_handle, "operator", TIFOP_RANGE_INCLUSIVE, NULL);
  g_object_get (tif_handle, "operator", &int_val, NULL);
  EXPECT_EQ (TIFOP_RANGE_INCLUSIVE, int_val);

  /* Check 2 input parameter */
  g_object_set (tif_handle, "supplied-value", "30,100", NULL);
  g_object_get (tif_handle, "supplied-value", &str_val, NULL);
  EXPECT_STREQ ("30,100", str_val);
  g_free (str_val);

  g_object_set (tif_handle, "then", TIFB_TENSORPICK, NULL);
  g_object_get (tif_handle, "then", &int_val, NULL);
  EXPECT_EQ (TIFB_TENSORPICK, int_val);

  /* Check behavior option */
  g_object_set (tif_handle, "then-option", "0", NULL);
  g_object_get (tif_handle, "then-option", &str_val, NULL);
  EXPECT_STREQ ("0", str_val);
  g_free (str_val);

  g_object_set (tif_handle, "else", TIFB_TENSORPICK, NULL);
  g_object_get (tif_handle, "else", &int_val, NULL);
  EXPECT_EQ (TIFB_TENSORPICK, int_val);

  g_object_set (tif_handle, "else-option", "0", NULL);
  g_object_get (tif_handle, "else-option", &str_val, NULL);
  EXPECT_STREQ ("0", str_val);
  g_free (str_val);

  g_object_set (tif_handle, "silent", TRUE, NULL);
  g_object_get (tif_handle, "silent", &bool_val, NULL);
  EXPECT_EQ (TRUE, bool_val);

  gst_object_unref (tif_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for invalid properties of tensor_if
 */
TEST (tensorIfProp, properties1_n)
{
  gchar *pipeline;
  GstElement *gstpipe;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc num-buffers=1 pattern=13 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! "
      "tensor_if name=tif compared-value=A_VALUE compared-value-option=0:2:1:1,0 "
      "supplied-value=100 operator=GE then=PASSTHROUGH else=SKIP ! tensor_sink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  GstElement *tif_handle;
  gchar *str_val = NULL;

  tif_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "tif");
  EXPECT_NE (tif_handle, nullptr);

  /* Set properties */
  g_object_set (tif_handle, "invalid-prop", "invalid-value", NULL);
  g_object_get (tif_handle, "invalid_prop", &str_val, NULL);
  /* getting unknown property, str should be null */
  EXPECT_TRUE (str_val == NULL);

  gst_object_unref (tif_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for invalid tensor index of tensor_if compared value option
 */
TEST (tensorIfProp, properties2_n)
{
  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=1 pattern=13 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! "
      "tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0,1 supplied-value=100 "
      "operator=GT then=PASSTHROUGH else=SKIP ! fakesink");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_usleep (100000);

  gst_object_unref (pipeline);
  g_free (str_pipeline);
}

/**
 * @brief Test for invalid tensor index of tensor_if compared value option
 */
TEST (tensorIfProp, properties3_n)
{
  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=1 pattern=13 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! "
      "tensor_if name=tif compared-value=TENSOR_AVERAGE_VALUE compared-value-option=1 supplied-value=100 "
      "operator=GT then=PASSTHROUGH else=SKIP ! fakesink");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_usleep (100000);

  gst_object_unref (pipeline);
  g_free (str_pipeline);
}

/**
 * @brief Test for invalid value of tensor_if compared value option
 */
TEST (tensorIfProp, properties4_n)
{
  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=1 pattern=13 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! "
      "tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0 supplied-value=100 "
      "operator=GT then=PASSTHROUGH else=SKIP ! fakesink");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_usleep (100000);

  gst_object_unref (pipeline);
  g_free (str_pipeline);
}

/**
 * @brief Test for invalid value of tensor_if compared value option
 */
TEST (tensorIfProp, properties5_n)
{
  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=2 pattern=13 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! mux.sink_0 "
      "videotestsrc num-buffers=2 pattern=15 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! mux.sink_1 "
      "tensor_mux name=mux ! tensor_if name=tif compared-value=TENSOR_AVERAGE_VALUE compared-value-option=0,1 supplied-value=100 "
      "operator=LT then=TENSORPICK then-option=1 else=TENSORPICK else-option=2 "
      "tif.src_0 ! queue ! fakesink "
      "tif.src_1 ! queue ! fakesink");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_usleep (100000);

  gst_object_unref (pipeline);
  g_free (str_pipeline);
}

static guint glib_critical_cnt = 0;

/**
 * @brief Log handler counting the critical messages reported by glib itself
 */
static void
_count_glib_critical (const gchar *, GLogLevelFlags, const gchar *, gpointer)
{
  glib_critical_cnt++;
}

static guint invalid_index_log_cnt = 0;

/**
 * @brief Log handler counting the invalid index reports of the option parsers
 */
static void
_count_invalid_index_log (const gchar *, GLogLevelFlags, const gchar *message, gpointer)
{
  if (message && g_strrstr (message, "Invalid tensor index"))
    invalid_index_log_cnt++;
}

/**
 * @brief Set the option properties of @a tif and count the invalid index reports.
 */
static guint
_set_options_with_errno (GstElement *tif, const gchar *cv_option, const gchar *option, int err)
{
  GLogFunc prev_handler;

  invalid_index_log_cnt = 0;
  prev_handler = g_log_set_default_handler (_count_invalid_index_log, NULL);

  errno = err;
  g_object_set (tif, "then-option", option, NULL);
  errno = err;
  g_object_set (tif, "else-option", option, NULL);
  errno = err;
  g_object_set (tif, "compared-value-option", cv_option, NULL);

  g_log_set_default_handler (prev_handler, NULL);

  return invalid_index_log_cnt;
}

/**
 * @brief Whether ml_loge () reaches the log domain the handler above hooks.
 * @details ml_loge () is g_critical () in a Linux distro build, but dlog on
 *          Tizen and logcat on Android, where the handler counts nothing. The
 *          decision is made at compile time on purpose: asking the parser under
 *          test would let a future loss of the index report skip the cases
 *          below instead of failing them.
 */
#if defined(__TIZEN__) || defined(__ANDROID__)
#define ML_LOGE_REACHES_GLIB 0
#else
#define ML_LOGE_REACHES_GLIB 1
#endif

/**
 * @brief Test that the option parsers of tensor_if ignore a stale errno
 */
TEST (tensorIfProp, optionStaleErrno)
{
#if !ML_LOGE_REACHES_GLIB
  GTEST_SKIP () << "ml_loge () does not reach the GLib log domain here";
#endif
  GstElement *tif = gst_element_factory_make ("tensor_if", NULL);
  gchar *str_val = NULL;

  ASSERT_NE (tif, nullptr);

  EXPECT_EQ (0U, _set_options_with_errno (tif, "1:2:1:1,1", "0,1", ERANGE));

  g_object_get (tif, "then-option", &str_val, NULL);
  EXPECT_STREQ ("0,1", str_val);
  g_free (str_val);

  g_object_get (tif, "else-option", &str_val, NULL);
  EXPECT_STREQ ("0,1", str_val);
  g_free (str_val);

  g_object_get (tif, "compared-value-option", &str_val, NULL);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal ("1:2:1:1,1", str_val));
  g_free (str_val);

  gst_object_unref (tif);
}

/**
 * @brief Test that the option parsers of tensor_if still report a real overflow (negative)
 * @note compared-value-option may name a custom callback, so it is not reported.
 */
TEST (tensorIfProp, optionOverflow_n)
{
#if !ML_LOGE_REACHES_GLIB
  GTEST_SKIP () << "ml_loge () does not reach the GLib log domain here";
#endif
  GstElement *tif = gst_element_factory_make ("tensor_if", NULL);
  gchar *str_val = NULL;

  ASSERT_NE (tif, nullptr);

  EXPECT_EQ (0U, _set_options_with_errno (tif, "1:2:1:1,1", "0,1", 0));
  EXPECT_EQ (2U, _set_options_with_errno (tif, "1:2:1:1,99999999999999999999",
                     "99999999999999999999", 0));

  g_object_get (tif, "then-option", &str_val, NULL);
  EXPECT_STREQ ("0,1", str_val);
  g_free (str_val);

  g_object_get (tif, "else-option", &str_val, NULL);
  EXPECT_STREQ ("0,1", str_val);
  g_free (str_val);

  g_object_get (tif, "compared-value-option", &str_val, NULL);
  EXPECT_STREQ ("", str_val);
  g_free (str_val);

  gst_object_unref (tif);
}

/**
 * @brief Test that the name of a custom callback is not reported as an invalid index
 */
TEST (tensorIfProp, optionCustomName)
{
#if !ML_LOGE_REACHES_GLIB
  GTEST_SKIP () << "ml_loge () does not reach the GLib log domain here";
#endif
  GstElement *tif = gst_element_factory_make ("tensor_if", NULL);
  gchar *str_val = NULL;

  ASSERT_NE (tif, nullptr);

  EXPECT_EQ (0U, _set_options_with_errno (tif, "tifx", "0", 0));

  g_object_get (tif, "compared-value-option", &str_val, NULL);
  EXPECT_STREQ ("", str_val);
  g_free (str_val);

  gst_object_unref (tif);
}

/**
 * @brief Check that @a option of tensor_if refuses @a value and keeps "0,1".
 */
static void
_expect_refused_option (GstElement *tif, const gchar *option, const gchar *value)
{
  gchar *str_val = NULL;

  g_object_set (tif, option, "0,1", NULL);
  g_object_set (tif, option, value, NULL);
  g_object_get (tif, option, &str_val, NULL);
  EXPECT_STREQ ("0,1", str_val) << option << "=" << value;
  g_free (str_val);
}

/**
 * @brief Test that tensor_if keeps a tensorpick option with a token that is not an index (negative)
 */
TEST (tensorIfProp, optionNotIndex_n)
{
  const gchar *invalid[] = { "0,abc", "0,1,", ",1", "1x", "4294967297",
    "2147483648", "-1", "+1", "0x1", "1 2", NULL };
  GstElement *tif = gst_element_factory_make ("tensor_if", NULL);
  guint i;

  ASSERT_NE (tif, nullptr);

  for (i = 0; invalid[i] != NULL; i++) {
    _expect_refused_option (tif, "then-option", invalid[i]);
    _expect_refused_option (tif, "else-option", invalid[i]);
  }

  gst_object_unref (tif);
}

/**
 * @brief Test that tensor_if accepts the bounds of a tensorpick index and blanks around it
 */
TEST (tensorIfProp, optionIndexBounds)
{
  GstElement *tif = gst_element_factory_make ("tensor_if", NULL);
  gchar *str_val = NULL;

  ASSERT_NE (tif, nullptr);

  g_object_set (tif, "then-option", " 2 , 0 ", NULL);
  g_object_get (tif, "then-option", &str_val, NULL);
  EXPECT_STREQ ("2,0", str_val);
  g_free (str_val);

  g_object_set (tif, "else-option", "2147483647,007", NULL);
  g_object_get (tif, "else-option", &str_val, NULL);
  EXPECT_STREQ ("2147483647,7", str_val);
  g_free (str_val);

  g_object_set (tif, "then-option", "", NULL);
  g_object_get (tif, "then-option", &str_val, NULL);
  EXPECT_STREQ ("", str_val);
  g_free (str_val);

  gst_object_unref (tif);
}

/**
 * @brief Test that a compared-value option that is not an index list leaves no index (negative)
 */
TEST (tensorIfProp, cvOptionNotIndex_n)
{
  const gchar *invalid[] = { "1:2:x:1,1", "1:2:1:1,x", "1:2:1:1,", "1:2:1:1,-1",
    "1::1:1,1", "1:2:1:1,4294967297", "", "x", ",1", " ,1", NULL };
  GstElement *tif = gst_element_factory_make ("tensor_if", NULL);
  gchar *str_val = NULL;
  guint i;

  ASSERT_NE (tif, nullptr);

  for (i = 0; invalid[i] != NULL; i++) {
    g_object_set (tif, "compared-value-option", "1:2:1:1,1", NULL);
    g_object_set (tif, "compared-value-option", invalid[i], NULL);
    g_object_get (tif, "compared-value-option", &str_val, NULL);
    EXPECT_STREQ ("", str_val) << invalid[i];
    g_free (str_val);
  }

  gst_object_unref (tif);
}

/**
 * @brief Test that a compared-value option allows blanks around its indices
 */
TEST (tensorIfProp, cvOptionBlanks)
{
  GstElement *tif = gst_element_factory_make ("tensor_if", NULL);
  gchar *str_val = NULL;

  ASSERT_NE (tif, nullptr);

  g_object_set (tif, "compared-value-option", " 1 : 2 : 1 : 1 , 1 ", NULL);
  g_object_get (tif, "compared-value-option", &str_val, NULL);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal ("1:2:1:1,1", str_val));
  g_free (str_val);

  g_object_set (tif, "compared-value-option", " 1 ", NULL);
  g_object_get (tif, "compared-value-option", &str_val, NULL);
  EXPECT_STREQ ("1", str_val);
  g_free (str_val);

  gst_object_unref (tif);
}

/**
 * @brief Test that tensor_if keeps a compared-value option it refuses (negative)
 */
TEST (tensorIfProp, optionTooManyFields_n)
{
  GstElement *tif = gst_element_factory_make ("tensor_if", NULL);
  gchar *str_val = NULL;

  ASSERT_NE (tif, nullptr);

  g_object_set (tif, "compared-value-option", "1:2:1:1,1", NULL);
  /* it should be in the form of 'IDX_DIM0: ... :INDEX_DIM_LAST,nth-tensor' */
  g_object_set (tif, "compared-value-option", "1:2,3,4", NULL);

  g_object_get (tif, "compared-value-option", &str_val, NULL);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal ("1:2:1:1,1", str_val));
  g_free (str_val);

  gst_object_unref (tif);
}

/**
 * @brief Test for the supplied value property of tensor_if
 */
TEST (tensorIfProp, suppliedValue)
{
  gchar *pipeline;
  GstElement *gstpipe, *tif_handle;
  gchar *str_val = NULL;
  gchar **strv;

  pipeline = g_strdup_printf (
      "videotestsrc num-buffers=1 pattern=13 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! "
      "tensor_if name=tif compared-value=A_VALUE compared-value-option=1:2:1:1,1 "
      "supplied-value=100 operator=GE then=PASSTHROUGH else=SKIP ! tensor_sink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  ASSERT_NE (gstpipe, nullptr);

  tif_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "tif");
  ASSERT_NE (tif_handle, nullptr);

  /* the maximum number of the supplied values */
  g_object_set (tif_handle, "supplied-value", "10,100", NULL);
  g_object_get (tif_handle, "supplied-value", &str_val, NULL);
  EXPECT_STREQ ("10,100", str_val);
  g_free (str_val);

  g_object_set (tif_handle, "supplied-value", "1.5,2.5", NULL);
  g_object_get (tif_handle, "supplied-value", &str_val, NULL);
  strv = g_strsplit (str_val, ",", -1);
  ASSERT_EQ (2U, g_strv_length (strv));
  EXPECT_DOUBLE_EQ (1.5, g_ascii_strtod (strv[0], NULL));
  EXPECT_DOUBLE_EQ (2.5, g_ascii_strtod (strv[1], NULL));
  g_strfreev (strv);
  g_free (str_val);

  g_object_set (tif_handle, "supplied-value", "", NULL);
  g_object_get (tif_handle, "supplied-value", &str_val, NULL);
  EXPECT_STREQ ("", str_val);
  g_free (str_val);

  gst_object_unref (tif_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_if supplied value with more values than it can hold
 */
TEST (tensorIfProp, suppliedValue1_n)
{
  gchar *pipeline;
  GstElement *gstpipe, *tif_handle;
  gchar *str_val = NULL;

  pipeline = g_strdup_printf (
      "videotestsrc num-buffers=1 pattern=13 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! "
      "tensor_if name=tif compared-value=A_VALUE compared-value-option=1:2:1:1,1 "
      "supplied-value=100 operator=GE then=PASSTHROUGH else=SKIP ! tensor_sink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  ASSERT_NE (gstpipe, nullptr);

  tif_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "tif");
  ASSERT_NE (tif_handle, nullptr);

  g_object_set (tif_handle, "then-option", "1", NULL);
  g_object_set (tif_handle, "else-option", "2", NULL);
  g_object_set (tif_handle, "supplied-value", "10,100", NULL);

  /* the values beyond the second one used to be written past the array */
  g_object_set (tif_handle, "supplied-value", "1,2,3,4,5,6,7,8", NULL);

  g_object_get (tif_handle, "supplied-value", &str_val, NULL);
  EXPECT_STREQ ("10,100", str_val);
  g_free (str_val);

  /* the members stored next to the supplied value should be intact */
  g_object_get (tif_handle, "compared-value-option", &str_val, NULL);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal ("1:2:1:1,1", str_val));
  g_free (str_val);

  g_object_get (tif_handle, "then-option", &str_val, NULL);
  EXPECT_STREQ ("1", str_val);
  g_free (str_val);

  g_object_get (tif_handle, "else-option", &str_val, NULL);
  EXPECT_STREQ ("2", str_val);
  g_free (str_val);

  gst_object_unref (tif_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_if supplied value with a token that is not a number (negative)
 */
TEST (tensorIfProp, suppliedValueNotNumber_n)
{
  const gchar *invalid[] = { "10,abc", "abc", "1x", "10,", ",10", "1.5x", "1.5,abc", "1e",
    "e1", "9223372036854775808", "0x10", "1 0", "1e999", "-1e999", "1e-999", NULL };
  GstElement *tif = gst_element_factory_make ("tensor_if", NULL);
  gchar *str_val = NULL;
  guint i;

  ASSERT_NE (tif, nullptr);

  for (i = 0; invalid[i] != NULL; i++) {
    g_object_set (tif, "supplied-value", "10,100", NULL);
    g_object_set (tif, "supplied-value", invalid[i], NULL);
    g_object_get (tif, "supplied-value", &str_val, NULL);
    EXPECT_STREQ ("10,100", str_val) << invalid[i];
    g_free (str_val);
  }

  gst_object_unref (tif);
}

/**
 * @brief Test for tensor_if supplied value at the bounds of its types and with blanks
 */
TEST (tensorIfProp, suppliedValueBounds)
{
  GstElement *tif = gst_element_factory_make ("tensor_if", NULL);
  gchar *str_val = NULL;
  gchar **strv;

  ASSERT_NE (tif, nullptr);

  g_object_set (tif, "supplied-value", " -10 , 100 ", NULL);
  g_object_get (tif, "supplied-value", &str_val, NULL);
  EXPECT_STREQ ("-10,100", str_val);
  g_free (str_val);

  /* the getter prints a long, which cannot hold these on a 32-bit target */
  g_object_set (tif, "supplied-value", "-9223372036854775808,9223372036854775807", NULL);
  g_object_get (tif, "supplied-value", &str_val, NULL);
  EXPECT_STRNE ("-10,100", str_val);
  g_free (str_val);

  g_object_set (tif, "supplied-value", " -1.5 , 2e3 ", NULL);
  g_object_get (tif, "supplied-value", &str_val, NULL);
  strv = g_strsplit (str_val, ",", -1);
  ASSERT_EQ (2U, g_strv_length (strv));
  EXPECT_DOUBLE_EQ (-1.5, g_ascii_strtod (strv[0], NULL));
  EXPECT_DOUBLE_EQ (2000.0, g_ascii_strtod (strv[1], NULL));
  g_strfreev (strv);
  g_free (str_val);

  /* an exponent alone makes the values floating-point */
  g_object_set (tif, "supplied-value", "2E3,3e-1", NULL);
  g_object_get (tif, "supplied-value", &str_val, NULL);
  strv = g_strsplit (str_val, ",", -1);
  ASSERT_EQ (2U, g_strv_length (strv));
  EXPECT_DOUBLE_EQ (2000.0, g_ascii_strtod (strv[0], NULL));
  EXPECT_DOUBLE_EQ (0.3, g_ascii_strtod (strv[1], NULL));
  g_strfreev (strv);
  g_free (str_val);

  g_object_set (tif, "supplied-value", "1.7e308", NULL);
  g_object_get (tif, "supplied-value", &str_val, NULL);
  EXPECT_DOUBLE_EQ (1.7e308, g_ascii_strtod (str_val, NULL));
  g_free (str_val);

  gst_object_unref (tif);
}

/**
 * @brief Test for tensor_if supplied value set to null
 */
TEST (tensorIfProp, suppliedValue2_n)
{
  gchar *pipeline;
  GstElement *gstpipe, *tif_handle;
  gchar *str_val = NULL;
  guint handler_id;

  pipeline = g_strdup_printf (
      "videotestsrc num-buffers=1 pattern=13 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! "
      "tensor_if name=tif compared-value=A_VALUE compared-value-option=1:2:1:1,1 "
      "supplied-value=100 operator=GE then=PASSTHROUGH else=SKIP ! tensor_sink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  ASSERT_NE (gstpipe, nullptr);

  tif_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "tif");
  ASSERT_NE (tif_handle, nullptr);

  glib_critical_cnt = 0;
  handler_id = g_log_set_handler ("GLib",
      (GLogLevelFlags) (G_LOG_LEVEL_CRITICAL | G_LOG_FLAG_FATAL | G_LOG_FLAG_RECURSION),
      _count_glib_critical, NULL);
  g_object_set (tif_handle, "supplied-value", NULL, NULL);
  g_log_remove_handler ("GLib", handler_id);

  /* the null value should be rejected before glib is fed with it */
  EXPECT_EQ (0U, glib_critical_cnt);

  g_object_get (tif_handle, "supplied-value", &str_val, NULL);
  EXPECT_STREQ ("100", str_val);
  g_free (str_val);

  gst_object_unref (tif_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test tensor_if behavior: PASSTHROUGH, SKIP
 */
TEST_F (tensor_if_run, action_0)
{
  gchar *content1 = NULL;
  gchar *content2 = NULL;
  gsize len1 = 0, len2 = 0;
  char *tmp = getTempFilename ();
  GstElement *tif_handle;

  EXPECT_NE (tmp, nullptr);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=1 pattern=13 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! "
      "tensor_if name=tif silent=false compared-value=A_VALUE compared-value-option=0:0:0:0,0 supplied-value=100 "
      "operator=GT then=PASSTHROUGH else=SKIP ! "
      "filesink location=%s buffer-mode=unbuffered",
      tmp);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_TRUE (g_file_get_contents ("./smpte.golden", &content1, &len1, NULL));
  _wait_pipeline_save_files (tmp, content2, len2, len1, TEST_TIMEOUT_MS);
  EXPECT_TRUE (len1 > 0 && len1 == len2);
  EXPECT_EQ (memcmp (content1, content2, len1), 0);
  g_free (content1);
  g_free (content2);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  tif_handle = gst_bin_get_by_name (GST_BIN (pipeline), "tif");
  EXPECT_NE (tif_handle, nullptr);
  g_object_set (tif_handle, "operator", TIFOP_LT, NULL);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  gst_object_unref (tif_handle);
  gst_object_unref (pipeline);

  g_free (str_pipeline);
  removeTempFile (&tmp);
}

/**
 * @brief Test tensor_if other/tensors stream test
 */
TEST_F (tensor_if_run, action_1)
{
  gchar *content1 = NULL;
  gchar *content2 = NULL;
  gsize len1 = 0, len2 = 0;
  char *tmp_true = getTempFilename ();
  char *tmp_false = getTempFilename ();

  EXPECT_NE (tmp_true, nullptr);
  EXPECT_NE (tmp_false, nullptr);

  /* videotestsrc pattern 12 alternate between black and white.*/
  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=2 pattern=12 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! mux.sink_0 "
      "videotestsrc num-buffers=2 pattern=13 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! mux.sink_1 "
      "videotestsrc num-buffers=2 pattern=15 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! mux.sink_2 "
      "tensor_mux name=mux ! tensor_if name=tif compared-value=TENSOR_AVERAGE_VALUE compared-value-option=0 supplied-value=100 "
      "operator=LT then=TENSORPICK then-option=1 else=TENSORPICK else-option=2 "
      "tif.src_0 ! queue ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "tif.src_1 ! queue ! filesink location=%s buffer-mode=unbuffered sync=false async=false",
      tmp_true, tmp_false);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  /* True action result */
  EXPECT_TRUE (g_file_get_contents ("./smpte.golden", &content1, &len1, NULL));
  _wait_pipeline_save_files (tmp_true, content2, len2, len1, TEST_TIMEOUT_MS);
  EXPECT_TRUE (len1 > 0 && len1 == len2);
  EXPECT_EQ (memcmp (content1, content2, len1), 0);
  g_free (content1);
  g_free (content2);

  /* False action result */
  EXPECT_TRUE (g_file_get_contents ("./gamut.golden", &content1, &len1, NULL));
  _wait_pipeline_save_files (tmp_false, content2, len2, len1, TEST_TIMEOUT_MS);
  EXPECT_TRUE (len1 > 0 && len1 == len2);
  EXPECT_EQ (memcmp (content1, content2, len1), 0);
  g_free (content1);
  g_free (content2);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  gst_object_unref (pipeline);

  g_free (str_pipeline);

  removeTempFile (&tmp_true);
  removeTempFile (&tmp_false);
}

#define change_transform_type(pipe, type, size)                                                  \
  do {                                                                                           \
    g_object_set (transform_handle, "option", type, NULL);                                       \
    EXPECT_EQ (setPipelineStateSync (pipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0); \
    g_usleep (100000);                                                                           \
    _wait_pipeline_save_files (tmp1, content1, len1, size, TEST_TIMEOUT_MS);                     \
    _wait_pipeline_save_files (tmp2, content2, len2, size, TEST_TIMEOUT_MS);                     \
    EXPECT_TRUE (len1 > 0 && len1 == len2);                                                      \
    EXPECT_EQ (memcmp (content1, content2, len1), 0);                                            \
    g_free (content1);                                                                           \
    g_free (content2);                                                                           \
    EXPECT_EQ (setPipelineStateSync (pipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);    \
    g_usleep (100000);                                                                           \
  } while (0);

/**
 * @brief Test tensor_if compared value with all tensor data type
 */
TEST_F (tensor_if_run, action_2)
{
  gchar *content1 = NULL;
  gchar *content2 = NULL;
  gsize len1 = 0, len2 = 0;
  gchar *tmp1 = getTempFilename ();
  gchar *tmp2 = getTempFilename ();
  GstElement *transform_handle;

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=1 pattern=13 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=160,height=120 ! "
      "tensor_converter ! tensor_transform mode=clamp option=0:127 ! tensor_transform name=trans mode=typecast option=uint8 ! "
      "tee name=t ! queue ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! tensor_if compared-value=A_VALUE compared-value-option=0:0:0:0,0 supplied-value=0,127 "
      "operator=RANGE_INCLUSIVE then=PASSTHROUGH else=SKIP ! filesink location=%s buffer-mode=unbuffered sync=false async=false",
      tmp1, tmp2);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  transform_handle = gst_bin_get_by_name (GST_BIN (pipeline), "trans");
  EXPECT_NE (transform_handle, nullptr);

  change_transform_type (pipeline, "uint8", 57600 * sizeof (uint8_t));
  change_transform_type (pipeline, "uint16", 57600 * sizeof (uint16_t));
  change_transform_type (pipeline, "uint32", 57600 * sizeof (uint32_t));
  change_transform_type (pipeline, "uint64", 57600 * sizeof (uint64_t));
  change_transform_type (pipeline, "int8", 57600 * sizeof (int8_t));
  change_transform_type (pipeline, "int16", 57600 * sizeof (int16_t));
  change_transform_type (pipeline, "int32", 57600 * sizeof (int32_t));
  change_transform_type (pipeline, "int64", 57600 * sizeof (int64_t));
  change_transform_type (pipeline, "float32", 57600 * sizeof (float));
  change_transform_type (pipeline, "float64", 57600 * sizeof (double));

  gst_object_unref (transform_handle);
  gst_object_unref (pipeline);
  g_free (str_pipeline);

  removeTempFile (&tmp1);
  removeTempFile (&tmp2);
}

/**
 * @brief Test Tensor_if compared-value-option with undefined dimension properties
 */
TEST_F (tensor_if_run, action_3)
{
  gchar *content1 = NULL;
  gchar *content2 = NULL;
  gsize len1 = 0, len2 = 0;
  char *tmp = getTempFilename ();
  GstElement *tif_handle;

  EXPECT_NE (tmp, nullptr);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=1 pattern=13 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! "
      "tensor_if name=tif silent=false compared-value=A_VALUE compared-value-option=0:0,0 supplied-value=100 "
      "operator=GT then=PASSTHROUGH else=SKIP ! "
      "filesink location=%s buffer-mode=unbuffered",
      tmp);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_TRUE (g_file_get_contents ("./smpte.golden", &content1, &len1, NULL));
  _wait_pipeline_save_files (tmp, content2, len2, len1, TEST_TIMEOUT_MS);
  EXPECT_TRUE (len1 > 0 && len1 == len2);
  EXPECT_EQ (memcmp (content1, content2, len1), 0);
  g_free (content1);
  g_free (content2);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  tif_handle = gst_bin_get_by_name (GST_BIN (pipeline), "tif");
  EXPECT_NE (tif_handle, nullptr);
  g_object_set (tif_handle, "operator", TIFOP_LT, NULL);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  gst_object_unref (tif_handle);
  gst_object_unref (pipeline);

  g_free (str_pipeline);
  removeTempFile (&tmp);
}

/**
 * @brief Test data for tensor_if (2 frames with dimension 3:4:2:2)
 */
const gint test_frames[2][48]
    = { { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112,
            1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124,
            1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212,
            1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224 },
        { 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113,
            2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2201,
            2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213,
            2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224 } };

/**
 * @brief Callback for tensor sink signal.
 */
static void
new_data_cb (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  GstMemory *mem_res;
  GstMapInfo info_res;
  gint *output, i;
  gint index = *(gint *) user_data;
  gboolean ret;

  data_received++;
  /* Index 100 means a callback that is not allowed. */
  ASSERT_NE (100, index);
  mem_res = gst_buffer_get_memory (buffer, 0);
  ret = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
  ASSERT_TRUE (ret);
  output = (gint *) info_res.data;

  for (i = 0; i < 48; i++) {
    EXPECT_EQ (test_frames[index][i], output[i]);
  }
  gst_memory_unmap (mem_res, &info_res);
  gst_memory_unref (mem_res);
}

/**
 * @brief Test behavior: PASSTHROUGH, SKIP with tensor stream using appsrc
 */
TEST (tensorIfAppsrc, action0)
{
  GstBuffer *buf_0, *buf_1;
  GstMemory *mem;
  GstMapInfo info;
  GstElement *appsrc_handle, *sink_handle, *tif_handle;
  GstCaps *caps;
  gint idx;
  gchar *caps_name;
  GstStructure *structure;
  GstPad *pad;
  gboolean ret;
  gchar *str_pipeline = g_strdup (
      "appsrc name=appsrc ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! "
      "tensor_if name=tif compared-value=A_VALUE compared-value-option=1:1:1:1,0 supplied-value=1217 "
      "operator=EQ then=PASSTHROUGH else=SKIP ! "
      "other/tensors,num_tensors=1,dimensions=(string)3:4:2:2, types=(string)int32, framerate=(fraction)0/1 ! "
      "tensor_sink name=sinkx async=false");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);


  appsrc_handle = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc");
  EXPECT_NE (appsrc_handle, nullptr);

  tif_handle = gst_bin_get_by_name (GST_BIN (pipeline), "tif");
  EXPECT_NE (tif_handle, nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, (gpointer) &idx);

  buf_0 = gst_buffer_new ();
  mem = gst_allocator_alloc (NULL, 192, NULL);
  ret = gst_memory_map (mem, &info, GST_MAP_WRITE);
  ASSERT_TRUE (ret);
  memcpy (info.data, test_frames[0], 192);
  gst_memory_unmap (mem, &info);
  gst_buffer_append_memory (buf_0, mem);
  buf_1 = gst_buffer_copy (buf_0);

  data_received = 0;

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  idx = 0;
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle), buf_0), GST_FLOW_OK);
  g_usleep (100000);

  g_object_set (tif_handle, "supplied-value", "2000", NULL);

  idx = 100;
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle), buf_1), GST_FLOW_OK);
  g_usleep (100000);

  /** get negotiated caps */
  pad = gst_element_get_static_pad (sink_handle, "sink");
  EXPECT_NE (pad, nullptr);
  caps = gst_pad_get_current_caps (pad);
  EXPECT_NE (pad, nullptr);
  structure = gst_caps_get_structure (caps, 0);
  EXPECT_NE (structure, nullptr);
  caps_name = g_strdup (gst_structure_get_name (structure));

  EXPECT_STREQ ("other/tensors", caps_name);
  g_free (caps_name);
  gst_caps_unref (caps);
  gst_object_unref (pad);
  gst_object_unref (sink_handle);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (1, data_received);

  gst_object_unref (appsrc_handle);
  gst_object_unref (tif_handle);
  gst_object_unref (pipeline);
  g_free (str_pipeline);
}

/**
 * @brief Test that a new supplied value fully replaces the previous one
 */
TEST (tensorIfAppsrc, suppliedValueReset)
{
  GstBuffer *buf_0, *buf_1;
  GstMemory *mem;
  GstMapInfo info;
  GstElement *appsrc_handle, *sink_handle, *tif_handle;
  gint idx;
  gboolean ret;
  gchar *str_pipeline = g_strdup (
      "appsrc name=appsrc ! other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1 ! "
      "tensor_if name=tif compared-value=A_VALUE compared-value-option=1:1:1:1,0 supplied-value=1000,2000 "
      "operator=RANGE_INCLUSIVE then=PASSTHROUGH else=SKIP ! "
      "other/tensors,num_tensors=1,dimensions=(string)3:4:2:2, types=(string)int32, framerate=(fraction)0/1 ! "
      "tensor_sink name=sinkx async=false");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_NE (pipeline, nullptr);

  appsrc_handle = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc");
  ASSERT_NE (appsrc_handle, nullptr);

  tif_handle = gst_bin_get_by_name (GST_BIN (pipeline), "tif");
  ASSERT_NE (tif_handle, nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, (gpointer) &idx);

  buf_0 = gst_buffer_new ();
  mem = gst_allocator_alloc (NULL, 192, NULL);
  ret = gst_memory_map (mem, &info, GST_MAP_WRITE);
  ASSERT_TRUE (ret);
  memcpy (info.data, test_frames[0], 192);
  gst_memory_unmap (mem, &info);
  gst_buffer_append_memory (buf_0, mem);
  buf_1 = gst_buffer_copy (buf_0);

  data_received = 0;

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  /* the compared value 1217 is within [1000, 2000] */
  idx = 0;
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle), buf_0), GST_FLOW_OK);
  g_usleep (100000);

  /* the second operand of the previous value should not survive */
  g_object_set (tif_handle, "supplied-value", "1000", NULL);

  idx = 100;
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle), buf_1), GST_FLOW_OK);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (1, data_received);

  gst_object_unref (sink_handle);
  gst_object_unref (appsrc_handle);
  gst_object_unref (tif_handle);
  gst_object_unref (pipeline);
  g_free (str_pipeline);
}

/**
 * @brief Test behavior: TENSORPICK with tensors stream using appsrc
 */
TEST (tensorIfAppsrc, action1)
{
  GstBuffer *buf_0, *buf_1;
  GstMemory *mem;
  GstMapInfo info;
  GstElement *appsrc_handle, *sink_handle, *tif_handle;
  gint i, idx;

  gchar *str_pipeline = g_strdup (
      "appsrc name=appsrc ! other/tensors,num_tensors=2,dimensions=(string)3:4:2:2.3:4:2:2, types=(string)int32.int32,framerate=(fraction)0/1 ! "
      "tensor_if name=tif compared-value=TENSOR_AVERAGE_VALUE compared-value-option=0 supplied-value=1162.5 "
      "operator=EQ then=TENSORPICK then-option=0 else=TENSORPICK else-option=1 "
      "tif.src_0 ! queue ! tensor_sink name=sink_true async=false "
      "tif.src_1 ! queue ! tensor_sink name=sink_false async=false");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  appsrc_handle = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc");
  EXPECT_NE (appsrc_handle, nullptr);

  tif_handle = gst_bin_get_by_name (GST_BIN (pipeline), "tif");
  EXPECT_NE (tif_handle, nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sink_true");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, (gpointer) &idx);
  gst_object_unref (sink_handle);

  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sink_false");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, (gpointer) &idx);
  gst_object_unref (sink_handle);

  buf_0 = gst_buffer_new ();
  for (i = 0; i < 2; i++) {
    gboolean ret;
    mem = gst_allocator_alloc (NULL, 192, NULL);
    ret = gst_memory_map (mem, &info, GST_MAP_WRITE);
    ASSERT_TRUE (ret);
    memcpy (info.data, test_frames[i], 192);
    gst_memory_unmap (mem, &info);
    gst_buffer_append_memory (buf_0, mem);
  }
  buf_1 = gst_buffer_copy (buf_0);

  data_received = 0;
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  idx = 0;
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle), buf_0), GST_FLOW_OK);
  g_usleep (100000);

  g_object_set (tif_handle, "supplied-value", "2000", NULL);

  idx = 1;
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle), buf_1), GST_FLOW_OK);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (2, data_received);

  gst_object_unref (appsrc_handle);
  gst_object_unref (tif_handle);
  gst_object_unref (pipeline);
  g_free (str_pipeline);
}

/**
 * @brief custom callback function
 */
static gboolean
tensor_if_custom_cb (const GstTensorsInfo *info, const GstTensorMemory *input,
    void *user_data, gboolean *result)
{
  gint *output, i, idx;

  if (!info || !input || !result)
    return FALSE;

  idx = *(gint *) user_data;
  output = (gint *) input[idx].data;
  *result = TRUE;

  for (i = 0; i < 48; i++) {
    if (test_frames[idx][i] != output[i]) {
      *result = FALSE;
      break;
    }
  }

  return TRUE;
}

/**
 * @brief Test behavior: custom callback
 */
TEST (tensorIfCustom, normal0)
{
  GstBuffer *buf_0, *buf_1;
  GstMemory *mem;
  GstMapInfo info;
  GstElement *appsrc_handle, *sink_handle, *tif_handle;
  gint i, idx;
  gchar *str_val;

  gchar *str_pipeline = g_strdup (
      "appsrc name=appsrc ! other/tensors,num_tensors=2,dimensions=(string)3:4:2:2.3:4:2:2, types=(string)int32.int32,framerate=(fraction)0/1 ! "
      "tensor_if name=tif compared-value=CUSTOM compared-value-option=tifx then=TENSORPICK then-option=0 else=TENSORPICK else-option=1 "
      "tif.src_0 ! queue ! tensor_sink name=sink_true async=false "
      "tif.src_1 ! queue ! tensor_sink name=sink_false async=false");

  EXPECT_EQ (0,
      nnstreamer_if_custom_register ("tifx", tensor_if_custom_cb, (gpointer) &idx));

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  appsrc_handle = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc");
  EXPECT_NE (appsrc_handle, nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sink_true");
  EXPECT_NE (sink_handle, nullptr);

  tif_handle = gst_bin_get_by_name (GST_BIN (pipeline), "tif");
  EXPECT_NE (tif_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, (gpointer) &idx);
  gst_object_unref (sink_handle);

  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sink_false");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback) new_data_cb, (gpointer) &idx);
  gst_object_unref (sink_handle);

  g_object_get (tif_handle, "compared-value-option", &str_val, NULL);
  EXPECT_STREQ ("tifx", str_val);
  g_free (str_val);

  buf_0 = gst_buffer_new ();
  for (i = 0; i < 2; i++) {
    gboolean ret;
    mem = gst_allocator_alloc (NULL, 192, NULL);
    ret = gst_memory_map (mem, &info, GST_MAP_WRITE);
    ASSERT_TRUE (ret);
    memcpy (info.data, test_frames[i], 192);
    gst_memory_unmap (mem, &info);
    gst_buffer_append_memory (buf_0, mem);
  }
  buf_1 = gst_buffer_copy (buf_0);

  data_received = 0;
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  idx = 0;
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle), buf_0), GST_FLOW_OK);
  g_usleep (100000);

  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle), buf_1), GST_FLOW_OK);
  g_usleep (100000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (2, data_received);

  EXPECT_EQ (0, nnstreamer_if_custom_unregister ("tifx"));
  gst_object_unref (appsrc_handle);
  gst_object_unref (tif_handle);
  gst_object_unref (pipeline);
  g_free (str_pipeline);
}

/**
 * @brief Test behavior: custom callback, change the order of compared value option.
 */
TEST (tensorIfCustom, normal1)
{
  GstElement *tif_handle;
  gchar *str_val;
  gint int_val;
  gchar *str_pipeline = g_strdup (
      "appsrc name=appsrc ! other/tensors,num_tensors=2,dimensions=(string)3:4:2:2.3:4:2:2, types=(string)int32.int32,framerate=(fraction)0/1 ! "
      "tensor_if name=tif compared-value-option=tifx compared-value=CUSTOM  then=TENSORPICK then-option=0 else=TENSORPICK else-option=1 "
      "tif.src_0 ! queue ! tensor_sink name=sink_true async=false "
      "tif.src_1 ! queue ! tensor_sink name=sink_false async=false");

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  tif_handle = gst_bin_get_by_name (GST_BIN (pipeline), "tif");
  EXPECT_NE (tif_handle, nullptr);

  g_object_get (tif_handle, "compared-value-option", &str_val, NULL);
  EXPECT_STREQ ("tifx", str_val);
  g_free (str_val);

  /* Get properties */
  g_object_get (tif_handle, "compared-value", &int_val, NULL);
  EXPECT_EQ (TIFCV_CUSTOM, int_val);

  gst_object_unref (tif_handle);
  gst_object_unref (pipeline);
  g_free (str_pipeline);
}

/**
 * @brief Register custom callback with NULL parameter
 */
TEST (tensorIfCustom, invalidParam0_n)
{
  EXPECT_NE (0, nnstreamer_if_custom_register (NULL, tensor_if_custom_cb, NULL));
  EXPECT_NE (0, nnstreamer_if_custom_register ("tifx", NULL, NULL));
}

/**
 * @brief Register custom callback twice with same name
 */
TEST (tensorIfCustom, invalidParam1_n)
{
  EXPECT_EQ (0, nnstreamer_if_custom_register ("tifx", tensor_if_custom_cb, NULL));
  EXPECT_NE (0, nnstreamer_if_custom_register ("tifx", tensor_if_custom_cb, NULL));
  EXPECT_EQ (0, nnstreamer_if_custom_unregister ("tifx"));
}

/**
 * @brief Unregister custom callback with NULL parameter
 */
TEST (tensorIfCustom, invalidParam2_n)
{
  EXPECT_NE (0, nnstreamer_if_custom_unregister (NULL));
}

/**
 * @brief Unregister custom callback which is not registered
 */
TEST (tensorIfCustom, invalidParam3_n)
{
  EXPECT_NE (0, nnstreamer_if_custom_unregister ("tifx"));
}

/**
 * @brief Callback counting the buffers that reach the sink.
 */
static void
count_data_cb (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  guint *received = (guint *) user_data;

  *received = *received + 1;
}

/**
 * @brief Caps and size of the test frame test_frames[0].
 */
#define TEST_FRAME_CAPS \
  "other/tensor,dimension=(string)3:4:2:2,type=(string)int32,framerate=(fraction)0/1"
#define TEST_FRAME_SIZE (192U)

/**
 * @brief Push the first bytes of test_frames[0] as one buffer through a tensor_if.
 * @param caps the caps the buffer is pushed with
 * @param size the size of the buffer, at most TEST_FRAME_SIZE
 * @param if_props the properties of the tensor_if element under test
 * @param error_src name of the element the first bus error came from, or NULL
 * @param error the first bus error, or NULL
 * @param received number of the buffers that reached the sink
 * @return the first bus message type, GST_MESSAGE_ANY if the pipeline could not
 * be built and GST_MESSAGE_UNKNOWN if neither an error nor EOS arrived in time.
 */
static GstMessageType
_push_if_frame (const gchar *caps, gsize size, const gchar *if_props,
    gchar **error_src, GError **error, guint *received)
{
  GstElement *pipeline, *appsrc_handle, *sink_handle;
  GstBuffer *buf;
  GstBus *bus;
  GstMessage *msg;
  GstMessageType type = GST_MESSAGE_UNKNOWN;
  gchar *str_pipeline;

  *error_src = NULL;
  *error = NULL;
  *received = 0;

  str_pipeline = g_strdup_printf ("appsrc name=appsrc ! %s ! tensor_if name=tif %s ! "
                                  "tensor_sink name=sinkx async=false",
      caps, if_props);

  pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  if (pipeline == NULL)
    return GST_MESSAGE_ANY;

  appsrc_handle = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc");
  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  g_signal_connect (sink_handle, "new-data", (GCallback) count_data_cb, received);

  buf = gst_buffer_new_allocate (NULL, size, NULL);
  gst_buffer_fill (buf, 0, test_frames[0], size);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);
  gst_app_src_push_buffer (GST_APP_SRC (appsrc_handle), buf);
  gst_app_src_end_of_stream (GST_APP_SRC (appsrc_handle));

  bus = gst_element_get_bus (pipeline);
  msg = gst_bus_timed_pop_filtered (bus, 5 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
  if (msg) {
    type = GST_MESSAGE_TYPE (msg);
    if (type == GST_MESSAGE_ERROR) {
      *error_src = g_strdup (GST_OBJECT_NAME (GST_MESSAGE_SRC (msg)));
      gst_message_parse_error (msg, error, NULL);
    }
    gst_message_unref (msg);
  }
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);

  gst_object_unref (sink_handle);
  gst_object_unref (appsrc_handle);
  gst_object_unref (pipeline);

  return type;
}

/**
 * @brief Push one buffer through tensor_if compared-value=A_VALUE.
 * @param caps the caps the buffer is pushed with
 * @param size the size of the buffer
 * @param cv_option the compared-value-option to test
 * @param supplied_value the value the element compares the picked element with
 * @param error_src name of the element the first bus error came from, or NULL
 * @param error the first bus error, or NULL
 * @param received number of the buffers that reached the sink
 * @return the first bus message type, as _push_if_frame() returns it.
 */
static GstMessageType
_push_a_value_frame (const gchar *caps, gsize size, const gchar *cv_option,
    const gchar *supplied_value, gchar **error_src, GError **error, guint *received)
{
  GstMessageType type;
  gchar *if_props = g_strdup_printf ("compared-value=A_VALUE compared-value-option=%s supplied-value=%s "
                                     "operator=EQ then=PASSTHROUGH else=SKIP",
      cv_option, supplied_value);

  type = _push_if_frame (caps, size, if_props, error_src, error, received);
  g_free (if_props);

  return type;
}

/**
 * @brief Check that tensor_if refused the buffer instead of reading out of bounds.
 */
static void
_expect_refused_cv_option (const gchar *caps, gsize size, const gchar *cv_option)
{
  gchar *error_src = NULL;
  GError *error = NULL;
  guint received = 0;

  EXPECT_EQ (GST_MESSAGE_ERROR, _push_a_value_frame (caps, size, cv_option,
                                    "1224", &error_src, &error, &received));
  EXPECT_STREQ ("tif", error_src);
  ASSERT_NE (error, nullptr);
  EXPECT_EQ (GST_STREAM_ERROR, error->domain);
  EXPECT_EQ (GST_STREAM_ERROR_WRONG_TYPE, error->code);
  EXPECT_EQ (0U, received);

  g_clear_error (&error);
  g_free (error_src);
}

/**
 * @brief Check that caps declaring a tensor whose byte size overflows are refused.
 * @note The size of such a tensor cannot be computed, so the caps fail to
 * negotiate before tensor_if computes the offset of the compared element.
 */
static void
_expect_refused_overflow_caps (const gchar *caps, gsize size, const gchar *cv_option)
{
  gchar *error_src = NULL;
  GError *error = NULL;
  guint received = 0;

  EXPECT_EQ (GST_MESSAGE_ERROR, _push_a_value_frame (caps, size, cv_option,
                                    "1224", &error_src, &error, &received));
  EXPECT_EQ (0U, received);

  g_clear_error (&error);
  g_free (error_src);
}

/**
 * @brief Compare the elements at both ends of the tensor.
 * @note Every element of the test frame is unique, so the buffer reaches the
 * sink only if the element the option describes is the one that was read.
 */
TEST (tensorIfAppsrc, comparedValueBothEnds)
{
  gchar *error_src = NULL;
  GError *error = NULL;
  guint received = 0;

  /* test_frames[0][0] */
  EXPECT_EQ (GST_MESSAGE_EOS, _push_a_value_frame (TEST_FRAME_CAPS, TEST_FRAME_SIZE,
                                  "0:0:0:0,0", "1101", &error_src, &error, &received));
  EXPECT_EQ (1U, received);

  /* test_frames[0][2 + 3 * 3 + 1 * 12 + 1 * 24], the last offset that fits */
  EXPECT_EQ (GST_MESSAGE_EOS, _push_a_value_frame (TEST_FRAME_CAPS, TEST_FRAME_SIZE,
                                  "2:3:1:1,0", "1224", &error_src, &error, &received));
  EXPECT_EQ (1U, received);

  g_clear_error (&error);
  g_free (error_src);
}

/**
 * @brief The element index of the first dimension is not in the tensor.
 */
TEST (tensorIfAppsrc, comparedValueIndexOverFirstDim_n)
{
  _expect_refused_cv_option (TEST_FRAME_CAPS, TEST_FRAME_SIZE, "3:0:0:0,0");
}

/**
 * @brief The element index of the last dimension is one past the tensor.
 */
TEST (tensorIfAppsrc, comparedValueIndexOverLastDim_n)
{
  _expect_refused_cv_option (TEST_FRAME_CAPS, TEST_FRAME_SIZE, "0:0:0:2,0");
}

/**
 * @brief The element index stays inside the tensor but not inside its dimension.
 */
TEST (tensorIfAppsrc, comparedValueIndexOverInnerDim_n)
{
  _expect_refused_cv_option (TEST_FRAME_CAPS, TEST_FRAME_SIZE, "0:4:0:0,0");
}

/**
 * @brief The element index refers to a dimension the tensor does not have.
 */
TEST (tensorIfAppsrc, comparedValueIndexOverRank_n)
{
  _expect_refused_cv_option (TEST_FRAME_CAPS, TEST_FRAME_SIZE, "0:0:0:0:1,0");
}

/**
 * @brief A negative element index wraps around the unsigned dimension index.
 */
TEST (tensorIfAppsrc, comparedValueNegativeIndex_n)
{
  _expect_refused_cv_option (TEST_FRAME_CAPS, TEST_FRAME_SIZE, "-1:0:0:0,0");
}

/**
 * @brief An element index that is not a number used to be read as index 0.
 */
TEST (tensorIfAppsrc, comparedValueNotIndex_n)
{
  _expect_refused_cv_option (TEST_FRAME_CAPS, TEST_FRAME_SIZE, "2:3:x:1,0");
  _expect_refused_cv_option (TEST_FRAME_CAPS, TEST_FRAME_SIZE, "2:3:1:1,x");
  _expect_refused_cv_option (TEST_FRAME_CAPS, TEST_FRAME_SIZE, ",0");
}

/**
 * @brief Blanks around the element indices select the same element.
 */
TEST (tensorIfAppsrc, comparedValueBlanks)
{
  gchar *error_src = NULL;
  GError *error = NULL;
  guint received = 0;

  EXPECT_EQ (GST_MESSAGE_EOS,
      _push_a_value_frame (TEST_FRAME_CAPS, TEST_FRAME_SIZE,
          "\" 2 : 3 : 1 : 1 , 0 \"", "1224", &error_src, &error, &received));
  EXPECT_EQ (1U, received);

  g_clear_error (&error);
  g_free (error_src);
}

/**
 * @brief A tensor index of TENSOR_AVERAGE_VALUE that is not a number used to be read as tensor 0.
 */
TEST (tensorIfAppsrc, tensorAverageNotIndex_n)
{
  gchar *error_src = NULL;
  GError *error = NULL;
  guint received = 0;

  EXPECT_EQ (GST_MESSAGE_ERROR, _push_if_frame (TEST_FRAME_CAPS, TEST_FRAME_SIZE,
                                    "compared-value=TENSOR_AVERAGE_VALUE compared-value-option=x "
                                    "supplied-value=0 operator=GE then=PASSTHROUGH else=SKIP",
                                    &error_src, &error, &received));
  EXPECT_STREQ ("tif", error_src);
  EXPECT_EQ (0U, received);

  g_clear_error (&error);
  g_free (error_src);
}

/**
 * @brief An element index far past its dimension, which wrapped the 32-bit offset of main.
 */
TEST (tensorIfAppsrc, comparedValueHugeIndex_n)
{
  _expect_refused_cv_option (TEST_FRAME_CAPS, TEST_FRAME_SIZE, "2000000000:0:0:0,0");
}

/**
 * @brief The caps declare more data than the buffer carries.
 * @note The element is inside the dimensions, so only the size of the mapped
 * memory can refuse it.
 */
TEST (tensorIfAppsrc, comparedValueBufferShorterThanCaps_n)
{
  _expect_refused_cv_option ("other/tensor,dimension=(string)4:2:1,type=(string)uint8,framerate=(fraction)0/1",
      4, "0:1:0,0");
}

/**
 * @brief The byte offset of the element is 2^64 - 1, one short of wrapping to 0.
 * @note (2^32 - 1) * 2 + (2^32 - 1)^2 = 2^64 - 1, so a size check that adds the
 * element size to the offset wraps and reads the byte before the buffer.
 */
TEST (tensorIfAppsrc, comparedValueOffsetWrapsAround_n)
{
  _expect_refused_overflow_caps ("other/tensor,dimension=(string)4294967295:4294967295:2,type=(string)uint8,framerate=(fraction)0/1",
      4, "0:2:1,0");
}

#if GLIB_CHECK_VERSION(2, 48, 0)
/**
 * @brief The offset arithmetic overflows and wraps back into the buffer.
 * @note 2^31 * 2^31 * 4 wraps to 0, so without the overflow check the last
 * index adds nothing and the first byte of the buffer is compared silently.
 */
TEST (tensorIfAppsrc, comparedValueOffsetOverflow_n)
{
  _expect_refused_overflow_caps ("other/tensor,dimension=(string)2147483648:2147483648:4:2,type=(string)uint8,framerate=(fraction)0/1",
      4, "0:0:0:1,0");
}
#endif

/**
 * @brief Custom callback that cannot tell the condition of the buffer.
 */
static gboolean
tensor_if_custom_fail_cb (const GstTensorsInfo *info,
    const GstTensorMemory *input, void *user_data, gboolean *result)
{
  return FALSE;
}

/**
 * @brief The element reports the buffer a custom callback could not handle.
 */
TEST (tensorIfAppsrc, customCallbackFailure_n)
{
  gchar *error_src = NULL;
  GError *error = NULL;
  guint received = 0;
  GstMessageType type;

  ASSERT_EQ (0, nnstreamer_if_custom_register ("tif_fail", tensor_if_custom_fail_cb, NULL));

  type = _push_if_frame (TEST_FRAME_CAPS, TEST_FRAME_SIZE,
      "compared-value=CUSTOM compared-value-option=tif_fail then=PASSTHROUGH else=SKIP",
      &error_src, &error, &received);
  EXPECT_EQ (0, nnstreamer_if_custom_unregister ("tif_fail"));

  EXPECT_EQ (GST_MESSAGE_ERROR, type);
  EXPECT_STREQ ("tif", error_src);
  ASSERT_NE (error, nullptr);
  EXPECT_EQ (GST_STREAM_ERROR, error->domain);
  EXPECT_EQ (GST_STREAM_ERROR_WRONG_TYPE, error->code);
  EXPECT_EQ (0U, received);

  g_clear_error (&error);
  g_free (error_src);
}

/**
 * @brief Number of tensors a stream carries to reach GstTensorsInfo::extra.
 */
#define EXTRA_NUM_TENSORS ((guint) (NNS_TENSOR_MEMORY_MAX + 4))

/**
 * @brief Start a standalone tensor_if and hand out its sink pad.
 * @param tensor_if the element to start, filled in by this function
 * @return the sink pad of the element, which the caller should unref
 */
static GstPad *
_start_tensor_if (GstElement **tensor_if)
{
  GstElement *element;
  GstPad *sinkpad;

  element = gst_element_factory_make ("tensor_if", NULL);
  if (element == NULL)
    return NULL;

  g_object_set (element, "compared-value", TIFCV_A_VALUE,
      "compared-value-option", "0:0:0:0,0", "supplied-value", "0", "operator",
      TIFOP_GE, "then", TIFB_PASSTHROUGH, "else", TIFB_SKIP, NULL);

  gst_element_set_state (element, GST_STATE_PLAYING);
  sinkpad = gst_element_get_static_pad (element, "sink");
  gst_pad_send_event (sinkpad, gst_event_new_stream_start ("tensorif-extra"));

  *tensor_if = element;
  return sinkpad;
}

/**
 * @brief Build a buffer of uint8 tensors, each holding 4 zeroed elements.
 * @param info the tensors info describing the tensors of the buffer
 * @return the buffer, which the caller should unref
 */
static GstBuffer *
_buffer_with_tensors (GstTensorsInfo *info)
{
  GstBuffer *buffer = gst_buffer_new ();
  guint i;

  for (i = 0; i < info->num_tensors; i++) {
    GstTensorInfo *_info = gst_tensors_info_get_nth_info (info, i);
    GstMemory *mem = gst_allocator_alloc (NULL, gst_tensor_info_get_size (_info), NULL);
    GstMapInfo map;

    EXPECT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));
    memset (map.data, 0, map.size);
    gst_memory_unmap (mem, &map);

    EXPECT_TRUE (gst_tensor_buffer_append_memory (buffer, mem, _info));
  }

  return buffer;
}

/**
 * @brief Renegotiate a stream of more tensors than NNS_TENSOR_MEMORY_MAX, which
 *        reparses the input configuration of tensor_if.
 */
TEST (tensorIfExtraTensors, capsRenegotiation)
{
  GstElement *tensor_if = NULL;
  GstPad *sinkpad;
  GstCaps *caps;

  sinkpad = _start_tensor_if (&tensor_if);
  ASSERT_NE (sinkpad, nullptr);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_caps (caps)));
  gst_caps_unref (caps);

  EXPECT_EQ (GST_TENSOR_IF (tensor_if)->in_config.info.num_tensors, EXTRA_NUM_TENSORS);
  EXPECT_NE (GST_TENSOR_IF (tensor_if)->in_config.info.extra, nullptr);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS + 1, EXTRA_NUM_TENSORS + 1);
  EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_caps (caps)));
  gst_caps_unref (caps);

  EXPECT_EQ (GST_TENSOR_IF (tensor_if)->in_config.info.num_tensors, EXTRA_NUM_TENSORS + 1);

  gst_element_set_state (tensor_if, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (tensor_if);
}

/**
 * @brief Negotiate a stream whose tensors are not all described.
 */
TEST (tensorIfExtraTensors, capsRenegotiation_n)
{
  GstElement *tensor_if = NULL;
  GstPad *sinkpad;
  GstCaps *caps;

  sinkpad = _start_tensor_if (&tensor_if);
  ASSERT_NE (sinkpad, nullptr);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS - 1);
  EXPECT_FALSE (gst_pad_send_event (sinkpad, gst_event_new_caps (caps)));
  gst_caps_unref (caps);

  EXPECT_EQ (GST_TENSOR_IF (tensor_if)->in_config.info.num_tensors, 0U);

  gst_element_set_state (tensor_if, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (tensor_if);
}

/**
 * @brief Pass a buffer of more tensors than NNS_TENSOR_MEMORY_MAX through,
 * which fills the output configuration of tensor_if with extra tensors.
 */
TEST (tensorIfExtraTensors, passthroughBuffer)
{
  GstElement *tensor_if = NULL;
  GstPad *sinkpad;
  GstCaps *caps;
  GstBuffer *buffer;
  GstSegment segment;

  sinkpad = _start_tensor_if (&tensor_if);
  ASSERT_NE (sinkpad, nullptr);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_caps (caps)));
  gst_caps_unref (caps);

  gst_segment_init (&segment, GST_FORMAT_TIME);
  EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_segment (&segment)));

  buffer = _buffer_with_tensors (&GST_TENSOR_IF (tensor_if)->in_config.info);
  ASSERT_EQ (gst_tensor_buffer_get_count (buffer), EXTRA_NUM_TENSORS);

  /* the then-pad is created by the chain and has no peer to push to */
  EXPECT_EQ (gst_pad_chain (sinkpad, buffer), GST_FLOW_NOT_LINKED);
  EXPECT_EQ (GST_TENSOR_IF (tensor_if)->out_config[0].info.num_tensors, EXTRA_NUM_TENSORS);
  EXPECT_NE (GST_TENSOR_IF (tensor_if)->out_config[0].info.extra, nullptr);

  gst_element_set_state (tensor_if, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (tensor_if);
}

/**
 * @brief Chain a buffer into a standalone tensor_if negotiated with the given
 *        caps, and tell whether the element posted an error message.
 * @param caps the caps to negotiate, or NULL to chain before any caps
 * @param buffer the buffer to chain, which this function takes
 * @param posted set to TRUE if the element posted an error message
 * @return the flow return of the chain
 */
static GstFlowReturn
_chain_tensor_if (GstCaps *caps, GstBuffer *buffer, gboolean *posted)
{
  GstElement *tensor_if = NULL;
  GstPad *sinkpad;
  GstBus *bus;
  GstMessage *msg;
  GstSegment segment;
  GstFlowReturn ret;

  sinkpad = _start_tensor_if (&tensor_if);
  if (sinkpad == NULL) {
    gst_buffer_unref (buffer);
    return GST_FLOW_CUSTOM_ERROR;
  }

  bus = gst_bus_new ();
  gst_element_set_bus (tensor_if, bus);

  if (caps) {
    EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_caps (caps)));
  }

  gst_segment_init (&segment, GST_FORMAT_TIME);
  EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_segment (&segment)));

  ret = gst_pad_chain (sinkpad, buffer);

  msg = gst_bus_pop_filtered (bus, GST_MESSAGE_ERROR);
  *posted = (msg != NULL);
  if (msg)
    gst_message_unref (msg);

  gst_element_set_state (tensor_if, GST_STATE_NULL);
  gst_element_set_bus (tensor_if, NULL);
  gst_object_unref (bus);
  gst_object_unref (sinkpad);
  gst_object_unref (tensor_if);
  return ret;
}

/**
 * @brief Build a buffer of the given number of zeroed memories.
 * @param num_mems the number of memories
 * @param size the size of each memory
 * @return the buffer, which the caller should unref
 */
static GstBuffer *
_buffer_with_memories (guint num_mems, gsize size)
{
  GstBuffer *buffer = gst_buffer_new ();
  guint i;

  for (i = 0; i < num_mems; i++)
    gst_buffer_append_memory (buffer, gst_allocator_alloc (NULL, size, NULL));

  gst_buffer_memset (buffer, 0, 0, gst_buffer_get_size (buffer));
  return buffer;
}

/**
 * @brief Chain a buffer holding as many tensors as the caps declare.
 */
TEST (tensorIfTensorCount, matchingBuffer)
{
  GstCaps *caps = caps_with_tensors (2, 2);
  gboolean posted = TRUE;

  /* the then-pad is created by the chain and has no peer to push to */
  EXPECT_EQ (_chain_tensor_if (caps, _buffer_with_memories (2, 4), &posted), GST_FLOW_NOT_LINKED);
  EXPECT_FALSE (posted);

  gst_caps_unref (caps);
}

/**
 * @brief Chain a buffer holding more tensors than NNS_TENSOR_MEMORY_MAX, as
 *        many as the caps declare.
 */
TEST (tensorIfTensorCount, matchingExtraBuffer)
{
  GstCaps *caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  GstTensorsConfig config;
  GstBuffer *buffer;
  gboolean posted = TRUE;

  ASSERT_TRUE (gst_tensors_config_from_caps (&config, caps, TRUE));
  buffer = _buffer_with_tensors (&config.info);
  gst_tensors_config_free (&config);
  ASSERT_EQ (gst_tensor_buffer_get_count (buffer), EXTRA_NUM_TENSORS);

  EXPECT_EQ (_chain_tensor_if (caps, buffer, &posted), GST_FLOW_NOT_LINKED);
  EXPECT_FALSE (posted);

  gst_caps_unref (caps);
}

/**
 * @brief Chain a buffer carrying the two tensors of the caps in one memory,
 *        which used to abort the process.
 */
TEST (tensorIfTensorCount, fewerMemories_n)
{
  GstCaps *caps = caps_with_tensors (2, 2);
  gboolean posted = FALSE;

  EXPECT_EQ (_chain_tensor_if (caps, _buffer_with_memories (1, 8), &posted), GST_FLOW_ERROR);
  EXPECT_TRUE (posted);

  gst_caps_unref (caps);
}

/**
 * @brief Chain a buffer of more memories than the caps declare tensors.
 */
TEST (tensorIfTensorCount, moreMemories_n)
{
  GstCaps *caps = caps_with_tensors (2, 2);
  gboolean posted = FALSE;

  EXPECT_EQ (_chain_tensor_if (caps, _buffer_with_memories (3, 4), &posted), GST_FLOW_ERROR);
  EXPECT_TRUE (posted);

  gst_caps_unref (caps);
}

/**
 * @brief Chain a buffer before the caps are negotiated.
 */
TEST (tensorIfTensorCount, bufferBeforeCaps_n)
{
  gboolean posted = FALSE;

  EXPECT_EQ (_chain_tensor_if (NULL, _buffer_with_memories (1, 4), &posted), GST_FLOW_ERROR);
  EXPECT_TRUE (posted);
}

/**
 * @brief Size of the tensors of the allocator below, which is not less than the
 *        header of a flexible tensor so that picking such a tensor maps it.
 */
#define IF_MAP_HOOK_TENSOR_SIZE (128U)

/**
 * @brief Memory of the allocator below, a uint8 tensor.
 */
typedef struct {
  GstMemory mem;
  guint8 data[IF_MAP_HOOK_TENSOR_SIZE];
} IfMapHookMemory;

/**
 * @brief Allocator calling a hook from one of the maps of its memories, or refusing one.
 */
typedef struct {
  GstAllocator parent;
  guint maps; /**< number of maps so far */
  guint hook_at; /**< the map to call the hook from, counted from 1 */
  guint refuse_at; /**< the map to refuse, counted from 1 */
  GFunc hook; /**< called with hook_data as its first argument */
  gpointer hook_data;
} IfMapHookAllocator;

/**
 * @brief Class of IfMapHookAllocator.
 */
typedef struct {
  GstAllocatorClass parent_class;
} IfMapHookAllocatorClass;

G_DEFINE_TYPE (IfMapHookAllocator, if_map_hook_allocator, GST_TYPE_ALLOCATOR);

/**
 * @brief Map a memory of IfMapHookAllocator.
 */
static gpointer
if_map_hook_memory_map (GstMemory *mem, gsize, GstMapFlags)
{
  IfMapHookAllocator *self = (IfMapHookAllocator *) mem->allocator;

  if (++self->maps == self->hook_at && self->hook)
    self->hook (self->hook_data, NULL);

  if (self->maps == self->refuse_at)
    return NULL;

  return ((IfMapHookMemory *) mem)->data;
}

/**
 * @brief Unmap a memory of IfMapHookAllocator.
 */
static void
if_map_hook_memory_unmap (GstMemory *)
{
}

/**
 * @brief Free a memory of IfMapHookAllocator.
 */
static void
if_map_hook_allocator_free (GstAllocator *, GstMemory *mem)
{
  g_free (mem);
}

/**
 * @brief Initialize the class of IfMapHookAllocator.
 */
static void
if_map_hook_allocator_class_init (IfMapHookAllocatorClass *klass)
{
  GST_ALLOCATOR_CLASS (klass)->free = if_map_hook_allocator_free;
}

/**
 * @brief Initialize an IfMapHookAllocator.
 */
static void
if_map_hook_allocator_init (IfMapHookAllocator *self)
{
  GstAllocator *allocator = GST_ALLOCATOR_CAST (self);

  allocator->mem_type = "IfMapHook";
  allocator->mem_map = if_map_hook_memory_map;
  allocator->mem_unmap = if_map_hook_memory_unmap;
  GST_OBJECT_FLAG_SET (allocator, GST_ALLOCATOR_FLAG_CUSTOM_ALLOC);
}

/**
 * @brief Build a buffer of two zeroed tensors of the given allocator.
 */
static GstBuffer *
_buffer_with_hook_memories (IfMapHookAllocator *allocator)
{
  GstBuffer *buffer = gst_buffer_new ();
  guint i;

  for (i = 0; i < 2; i++) {
    IfMapHookMemory *mem = g_new0 (IfMapHookMemory, 1);

    gst_memory_init (GST_MEMORY_CAST (mem), GST_MEMORY_FLAG_NO_SHARE,
        GST_ALLOCATOR_CAST (allocator), NULL, sizeof (mem->data), 0, 0,
        sizeof (mem->data));
    gst_buffer_append_memory (buffer, GST_MEMORY_CAST (mem));
  }

  return buffer;
}

/**
 * @brief How long a property access is given to finish while the chain is held.
 */
#define PROP_ACCESS_WAIT_MS (100U)

/**
 * @brief A property access made from another thread while tensor_if handles a buffer.
 */
typedef struct {
  GstElement *tensor_if;
  const gchar *name; /**< the string property to access */
  const gchar *value; /**< the value to set, NULL to get the property */
  GThread *thread;
  gint started; /**< set once the thread runs */
  gint done; /**< set once the access returned */
  gboolean done_in_chain; /**< the access returned while the chain was held */
} IfPropAccess;

/**
 * @brief Thread accessing the property of IfPropAccess.
 */
static gpointer
_prop_access_thread (gpointer user_data)
{
  IfPropAccess *access = (IfPropAccess *) user_data;

  g_atomic_int_set (&access->started, 1);
  if (access->value) {
    g_object_set (access->tensor_if, access->name, access->value, NULL);
  } else {
    gchar *value = NULL;

    g_object_get (access->tensor_if, access->name, &value, NULL);
    g_free (value);
  }

  g_atomic_int_set (&access->done, 1);
  return NULL;
}

/**
 * @brief Map hook starting the property access and giving it time to finish.
 */
static void
_prop_access_hook (gpointer user_data, gpointer)
{
  IfPropAccess *access = (IfPropAccess *) user_data;
  guint i;

  access->thread = g_thread_new ("tif-prop", _prop_access_thread, access);
  while (!g_atomic_int_get (&access->started))
    g_usleep (1000);

  for (i = 0; i < PROP_ACCESS_WAIT_MS && !g_atomic_int_get (&access->done); i++)
    g_usleep (1000);

  access->done_in_chain = (g_atomic_int_get (&access->done) != 0);
}

/**
 * @brief Negotiate two uint8 tensors of the given size on the sink pad of a
 *        standalone tensor_if.
 */
static void
_negotiate_two_tensors (GstPad *sinkpad, guint size)
{
  gchar *dimensions = g_strdup_printf ("%u:1:1:1,%u:1:1:1", size, size);
  GstCaps *caps = gst_caps_new_simple ("other/tensors", "format", G_TYPE_STRING, "static",
      "num_tensors", G_TYPE_INT, 2, "dimensions", G_TYPE_STRING, dimensions, "types",
      G_TYPE_STRING, "uint8,uint8", "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  GstSegment segment;

  g_free (dimensions);
  EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_caps (caps)));
  gst_caps_unref (caps);
  gst_segment_init (&segment, GST_FORMAT_TIME);
  EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_segment (&segment)));
}

/**
 * @brief Chain a buffer of two tensors into a tensor_if picking tensors, and
 *        access a property from another thread in the middle of the chain.
 * @param hook_at the map to access the property from: 1 is the map reading the
 *        compared value, 2 and later are the maps of the picked tensors
 * @param then_option the then-option the element starts with
 * @param name the string property to access
 * @param value the value to set, NULL to get the property
 * @param picked set to the number of tensors the chain picked
 * @param after set to the value of the property after the chain, which the caller frees
 * @return TRUE if the access returned while the chain was still using the properties
 */
static gboolean
_prop_access_during_chain (guint hook_at, const gchar *then_option,
    const gchar *name, const gchar *value, guint *picked, gchar **after)
{
  IfPropAccess access = { NULL, name, value, NULL, 0, 0, FALSE };
  IfMapHookAllocator *allocator;
  GstPad *sinkpad;

  *picked = 0;
  *after = NULL;

  sinkpad = _start_tensor_if (&access.tensor_if);
  if (sinkpad == NULL)
    return TRUE;

  g_object_set (access.tensor_if, "then", TIFB_TENSORPICK, "then-option", then_option, NULL);

  allocator = (IfMapHookAllocator *) g_object_new (if_map_hook_allocator_get_type (), NULL);
  allocator->hook_at = hook_at;
  allocator->hook = _prop_access_hook;
  allocator->hook_data = &access;

  _negotiate_two_tensors (sinkpad, IF_MAP_HOOK_TENSOR_SIZE);
  EXPECT_EQ (gst_pad_chain (sinkpad, _buffer_with_hook_memories (allocator)), GST_FLOW_NOT_LINKED);

  /* the hook should have run, or the result tells nothing */
  EXPECT_GE (allocator->maps, hook_at);
  EXPECT_NE (access.thread, nullptr);
  if (access.thread)
    g_thread_join (access.thread);
  else
    access.done_in_chain = TRUE;

  *picked = GST_TENSOR_IF (access.tensor_if)->out_config[TIFSP_THEN_PAD].info.num_tensors;
  g_object_get (access.tensor_if, name, after, NULL);

  gst_element_set_state (access.tensor_if, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (access.tensor_if);
  gst_object_unref (allocator);

  return access.done_in_chain;
}

/**
 * @brief Set the properties the condition reads while the chain reads the
 *        compared value, which used to free the lists under the chain.
 */
TEST (tensorIfPropRace, setWhileCheckingCondition)
{
  const gchar *props[][2] = { { "compared-value-option", "0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0,1" },
    { "supplied-value", "1" }, { "then-option", "1" }, { "else-option", "1" } };
  guint i, picked;
  gchar *after;

  for (i = 0; i < G_N_ELEMENTS (props); i++) {
    EXPECT_FALSE (_prop_access_during_chain (1, "0", props[i][0], props[i][1], &picked, &after))
        << props[i][0];
    EXPECT_EQ (picked, 1U);
    EXPECT_STREQ (after, props[i][1]) << props[i][0];
    g_free (after);
  }
}

/**
 * @brief Replace then-option while the chain walks it to pick the tensors.
 */
TEST (tensorIfPropRace, setWhilePickingTensors)
{
  guint picked;
  gchar *after;

  /* at the first picked tensor, the buffer keeps the option it started with */
  EXPECT_FALSE (_prop_access_during_chain (2, "0,1", "then-option", "1", &picked, &after));
  EXPECT_EQ (picked, 2U);
  EXPECT_STREQ (after, "1");
  g_free (after);

  /* at the last picked tensor */
  EXPECT_FALSE (_prop_access_during_chain (3, "0,1", "then-option", "0", &picked, &after));
  EXPECT_EQ (picked, 2U);
  EXPECT_STREQ (after, "0");
  g_free (after);
}

/**
 * @brief Read the properties while the chain uses them.
 */
TEST (tensorIfPropRace, getWhileStreaming)
{
  guint picked;
  gchar *after;

  EXPECT_FALSE (_prop_access_during_chain (
      1, "0,1", "compared-value-option", NULL, &picked, &after));
  EXPECT_STREQ (after, "0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0,0");
  g_free (after);

  EXPECT_FALSE (_prop_access_during_chain (2, "0,1", "then-option", NULL, &picked, &after));
  EXPECT_EQ (picked, 2U);
  EXPECT_STREQ (after, "0,1");
  g_free (after);
}

/**
 * @brief Set an invalid then-option while the chain walks the list, which
 *        waits for the chain as well and leaves the option as it was.
 */
TEST (tensorIfPropRace, setInvalidWhilePickingTensors_n)
{
  guint picked;
  gchar *after;

  EXPECT_FALSE (_prop_access_during_chain (2, "0,1", "then-option", "0,x", &picked, &after));
  EXPECT_EQ (picked, 2U);
  EXPECT_STREQ (after, "0,1");
  g_free (after);
}

/**
 * @brief Custom condition changing and reading the properties of its tensor_if.
 */
static gboolean
_custom_cb_touching_props (const GstTensorsInfo *, const GstTensorMemory *,
    void *user_data, gboolean *result)
{
  GstElement *tensor_if = *(GstElement **) user_data;
  gchar *value = NULL;

  g_object_set (tensor_if, "then-option", "1", NULL);
  g_object_get (tensor_if, "then-option", &value, NULL);
  *result = (g_strcmp0 (value, "1") == 0);
  g_free (value);

  return TRUE;
}

/**
 * @brief Let the custom condition change then-option of its own element; the
 *        callback runs without the lock and the buffer takes the new option.
 */
TEST (tensorIfPropRace, customCallbackSetsProperty)
{
  GstElement *tensor_if = NULL;
  GstPad *sinkpad;

  ASSERT_EQ (0, nnstreamer_if_custom_register (
                    "tif_touch_props", _custom_cb_touching_props, &tensor_if));

  sinkpad = _start_tensor_if (&tensor_if);
  ASSERT_NE (sinkpad, nullptr);
  g_object_set (tensor_if, "compared-value", TIFCV_CUSTOM, "compared-value-option",
      "tif_touch_props", "then", TIFB_TENSORPICK, "then-option", "0,1", NULL);

  _negotiate_two_tensors (sinkpad, 4);
  EXPECT_EQ (gst_pad_chain (sinkpad, _buffer_with_memories (2, 4)), GST_FLOW_NOT_LINKED);
  EXPECT_EQ (GST_TENSOR_IF (tensor_if)->out_config[TIFSP_THEN_PAD].info.num_tensors, 1U);

  gst_element_set_state (tensor_if, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (tensor_if);
  EXPECT_EQ (0, nnstreamer_if_custom_unregister ("tif_touch_props"));
}

/**
 * @brief Custom condition which is always true.
 */
static gboolean
_custom_cb_true (const GstTensorsInfo *, const GstTensorMemory *, void *, gboolean *result)
{
  *result = TRUE;
  return TRUE;
}

/**
 * @brief Tell whether the properties of a tensor_if can still be written and
 *        read, which blocks if a failed chain kept them locked.
 */
static gboolean
_props_are_released (GstElement *tensor_if)
{
  gchar *value = NULL;
  gboolean released;

  g_object_set (tensor_if, "else-option", "1", NULL);
  g_object_get (tensor_if, "else-option", &value, NULL);
  released = (g_strcmp0 (value, "1") == 0);
  g_free (value);

  return released;
}

/**
 * @brief Chain a buffer into a tensor_if whose custom condition is not registered.
 */
TEST (tensorIfPropRace, customNotConfigured_n)
{
  GstElement *tensor_if = NULL;
  GstPad *sinkpad;

  sinkpad = _start_tensor_if (&tensor_if);
  ASSERT_NE (sinkpad, nullptr);
  g_object_set (tensor_if, "compared-value", TIFCV_CUSTOM,
      "compared-value-option", "tif_not_registered", NULL);

  _negotiate_two_tensors (sinkpad, 4);
  EXPECT_EQ (gst_pad_chain (sinkpad, _buffer_with_memories (2, 4)), GST_FLOW_ERROR);
  EXPECT_TRUE (_props_are_released (tensor_if));

  gst_element_set_state (tensor_if, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (tensor_if);
}

/**
 * @brief Chain a buffer whose second tensor cannot be mapped for the custom condition.
 */
TEST (tensorIfPropRace, customMapFailure_n)
{
  GstElement *tensor_if = NULL;
  GstPad *sinkpad;
  IfMapHookAllocator *allocator;

  ASSERT_EQ (0, nnstreamer_if_custom_register ("tif_true", _custom_cb_true, NULL));

  sinkpad = _start_tensor_if (&tensor_if);
  ASSERT_NE (sinkpad, nullptr);
  g_object_set (tensor_if, "compared-value", TIFCV_CUSTOM,
      "compared-value-option", "tif_true", NULL);

  allocator = (IfMapHookAllocator *) g_object_new (if_map_hook_allocator_get_type (), NULL);
  allocator->refuse_at = 2;

  _negotiate_two_tensors (sinkpad, IF_MAP_HOOK_TENSOR_SIZE);
  EXPECT_EQ (gst_pad_chain (sinkpad, _buffer_with_hook_memories (allocator)), GST_FLOW_ERROR);
  EXPECT_EQ (allocator->maps, 2U);
  EXPECT_TRUE (_props_are_released (tensor_if));

  gst_element_set_state (tensor_if, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (tensor_if);
  gst_object_unref (allocator);
  EXPECT_EQ (0, nnstreamer_if_custom_unregister ("tif_true"));
}

/**
 * @brief Change a property of tensor_if when it adds its source pad.
 */
static void
_pad_added_sets_property (GstElement *tensor_if, GstPad *, gpointer user_data)
{
  g_object_set (tensor_if, "then-option", "1", NULL);
  (*(guint *) user_data)++;
}

/**
 * @brief Change then-option from the pad-added signal, which the chain emits
 *        after it is done with the properties.
 */
TEST (tensorIfPropRace, padAddedSetsProperty)
{
  GstElement *tensor_if = NULL;
  GstPad *sinkpad;
  guint added = 0;
  gchar *value = NULL;

  sinkpad = _start_tensor_if (&tensor_if);
  ASSERT_NE (sinkpad, nullptr);
  g_object_set (tensor_if, "then", TIFB_TENSORPICK, "then-option", "0,1", NULL);
  g_signal_connect (tensor_if, "pad-added", G_CALLBACK (_pad_added_sets_property), &added);

  _negotiate_two_tensors (sinkpad, 4);
  EXPECT_EQ (gst_pad_chain (sinkpad, _buffer_with_memories (2, 4)), GST_FLOW_NOT_LINKED);
  EXPECT_EQ (added, 1U);
  EXPECT_EQ (GST_TENSOR_IF (tensor_if)->out_config[TIFSP_THEN_PAD].info.num_tensors, 2U);
  g_object_get (tensor_if, "then-option", &value, NULL);
  EXPECT_STREQ (value, "1");
  g_free (value);

  gst_element_set_state (tensor_if, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (tensor_if);
}

/**
 * @brief Buffer probe changing a property of the tensor_if pushing the buffer.
 */
static GstPadProbeReturn
_push_probe_sets_property (GstPad *pad, GstPadProbeInfo *, gpointer user_data)
{
  GstElement *tensor_if = gst_pad_get_parent_element (pad);

  g_object_set (tensor_if, "then-option", "1", NULL);
  gst_object_unref (tensor_if);
  (*(guint *) user_data)++;

  return GST_PAD_PROBE_OK;
}

/**
 * @brief Add the probe above to the source pad a tensor_if adds.
 */
static void
_pad_added_adds_probe (GstElement *, GstPad *pad, gpointer user_data)
{
  gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER, _push_probe_sets_property,
      user_data, NULL);
}

/**
 * @brief Change then-option from a probe on the buffer tensor_if pushes, as an
 *        element downstream may do; the buffer is pushed without the lock.
 */
TEST (tensorIfPropRace, pushProbeSetsProperty)
{
  GstElement *tensor_if = NULL;
  GstPad *sinkpad;
  guint pushed = 0;
  gchar *value = NULL;

  sinkpad = _start_tensor_if (&tensor_if);
  ASSERT_NE (sinkpad, nullptr);
  g_object_set (tensor_if, "then", TIFB_TENSORPICK, "then-option", "0,1", NULL);
  g_signal_connect (tensor_if, "pad-added", G_CALLBACK (_pad_added_adds_probe), &pushed);

  _negotiate_two_tensors (sinkpad, 4);
  EXPECT_EQ (gst_pad_chain (sinkpad, _buffer_with_memories (2, 4)), GST_FLOW_NOT_LINKED);
  EXPECT_EQ (pushed, 1U);
  g_object_get (tensor_if, "then-option", &value, NULL);
  EXPECT_STREQ (value, "1");
  g_free (value);

  gst_element_set_state (tensor_if, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (tensor_if);
}

/**
 * @brief The properties a pad-removed handler read from its tensor_if.
 */
typedef struct {
  guint removed; /**< number of the pads removed */
  gchar *then_option; /**< then-option at the last removal */
  gchar *cv_option; /**< compared-value-option at the last removal */
} IfPadRemovedProps;

/**
 * @brief Read the properties of tensor_if when it removes a pad.
 */
static void
_pad_removed_reads_property (GstElement *tensor_if, GstPad *, gpointer user_data)
{
  IfPadRemovedProps *props = (IfPadRemovedProps *) user_data;

  g_free (props->then_option);
  g_free (props->cv_option);
  g_object_get (tensor_if, "then-option", &props->then_option,
      "compared-value-option", &props->cv_option, NULL);
  props->removed++;
}

/**
 * @brief Read the properties from the pad-removed signal of a tensor_if being
 *        disposed. The source pad goes before the options are freed and the
 *        sink pad after, when the lock still works and the options read empty.
 */
TEST (tensorIfPropRace, padRemovedReadsProperty)
{
  GstElement *tensor_if = NULL;
  GstPad *sinkpad;
  IfPadRemovedProps props = { 0, NULL, NULL };

  sinkpad = _start_tensor_if (&tensor_if);
  ASSERT_NE (sinkpad, nullptr);
  g_object_set (tensor_if, "then", TIFB_TENSORPICK, "then-option", "0,1", NULL);
  g_signal_connect (tensor_if, "pad-removed",
      G_CALLBACK (_pad_removed_reads_property), &props);

  _negotiate_two_tensors (sinkpad, 4);
  EXPECT_EQ (gst_pad_chain (sinkpad, _buffer_with_memories (2, 4)), GST_FLOW_NOT_LINKED);

  gst_element_set_state (tensor_if, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (tensor_if);

  EXPECT_EQ (props.removed, 2U);
  EXPECT_STREQ (props.then_option, "");
  EXPECT_STREQ (props.cv_option, "");
  g_free (props.then_option);
  g_free (props.cv_option);
}

/**
 * @brief Bus sync handler reading a property of the element posting an error.
 */
static GstBusSyncReply
_sync_handler_reads_property (GstBus *, GstMessage *message, gpointer user_data)
{
  if (GST_MESSAGE_TYPE (message) == GST_MESSAGE_ERROR) {
    gchar *value = NULL;

    g_object_get (GST_MESSAGE_SRC (message), "compared-value-option", &value, NULL);
    g_free (value);
    (*(guint *) user_data)++;
  }

  return GST_BUS_PASS;
}

/**
 * @brief Fail the condition with a bus sync handler reading the properties;
 *        the error is posted after the chain released the properties.
 */
TEST (tensorIfPropRace, errorHandlerReadsProperty_n)
{
  GstElement *tensor_if = NULL;
  GstPad *sinkpad;
  GstBus *bus;
  guint errors = 0;

  sinkpad = _start_tensor_if (&tensor_if);
  ASSERT_NE (sinkpad, nullptr);
  /* the caps declare 4 elements in the first dimension */
  g_object_set (tensor_if, "compared-value-option", "4:0:0:0,0", NULL);

  bus = gst_bus_new ();
  gst_bus_set_sync_handler (bus, _sync_handler_reads_property, &errors, NULL);
  gst_element_set_bus (tensor_if, bus);

  _negotiate_two_tensors (sinkpad, 4);
  EXPECT_EQ (gst_pad_chain (sinkpad, _buffer_with_memories (2, 4)), GST_FLOW_ERROR);
  EXPECT_EQ (errors, 1U);

  gst_element_set_state (tensor_if, GST_STATE_NULL);
  gst_element_set_bus (tensor_if, NULL);
  gst_object_unref (bus);
  gst_object_unref (sinkpad);
  gst_object_unref (tensor_if);
}

/**
 * @brief Number of the buffers of the test changing the properties from another thread.
 */
#define PROP_RACE_BUFFERS (300U)

static gint prop_toggle_stop = 0;
static gint prop_toggle_count = 0;

/**
 * @brief Thread replacing the properties of a tensor_if until it is told to stop.
 */
static gpointer
_prop_toggle_thread (gpointer user_data)
{
  GstElement *tensor_if = (GstElement *) user_data;
  guint i = 0;

  while (!g_atomic_int_get (&prop_toggle_stop)) {
    gboolean odd = (++i % 2 != 0);

    g_object_set (tensor_if, "then-option", odd ? "1,0" : "0", "else-option",
        odd ? "0" : "0,1", "compared-value-option",
        odd ? "1:0:0:0,1" : "0:0:0:0,0", "supplied-value", odd ? "1" : "0", NULL);
    g_atomic_int_inc (&prop_toggle_count);
    /* yield, a spinning thread starves the streaming one under valgrind */
    g_usleep (10);
  }

  return NULL;
}

/**
 * @brief Keep replacing the option lists and the supplied value from another
 *        thread while buffers flow.
 */
TEST (tensorIfPropRace, setFromAnotherThread)
{
  GstElement *tensor_if = NULL;
  GstPad *sinkpad;
  GThread *thread;
  guint i;

  sinkpad = _start_tensor_if (&tensor_if);
  ASSERT_NE (sinkpad, nullptr);
  g_object_set (tensor_if, "then", TIFB_TENSORPICK, "then-option", "0", "else",
      TIFB_TENSORPICK, "else-option", "0,1", NULL);

  _negotiate_two_tensors (sinkpad, 4);

  g_atomic_int_set (&prop_toggle_stop, 0);
  g_atomic_int_set (&prop_toggle_count, 0);
  thread = g_thread_new ("tif-toggle", _prop_toggle_thread, tensor_if);
  while (g_atomic_int_get (&prop_toggle_count) == 0)
    g_usleep (1000);

  for (i = 0; i < PROP_RACE_BUFFERS; i++) {
    GstFlowReturn ret = gst_pad_chain (sinkpad, _buffer_with_memories (2, 4));

    if (ret != GST_FLOW_NOT_LINKED) {
      ADD_FAILURE () << "buffer " << i << " returned " << gst_flow_get_name (ret);
      break;
    }
  }

  g_atomic_int_set (&prop_toggle_stop, 1);
  g_thread_join (thread);

  gst_element_set_state (tensor_if, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (tensor_if);
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
