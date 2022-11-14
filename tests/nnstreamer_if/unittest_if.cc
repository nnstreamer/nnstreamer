/**
 * @file        unittest_if.cc
 * @date        15 Oct 2020
 * @brief       Unit test for tensor_if
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <tensor_common.h>
#include <unittest_util.h>
#include "../gst/nnstreamer/elements/gsttensor_if.h"

#define TEST_TIMEOUT_MS (20000U)

static int data_received;

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
      "tensor_if name=tif compared-value=A_VALUE compared-value-option=0:2:1:1,0 "
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
  EXPECT_STREQ ("0:2:1:1,0", str_val);
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

/**
 * @brief Test tensor_if behavior: PASSTHROUGH, SKIP
 */
TEST_F (tensor_if_run, action_0)
{
  gchar *content1 = NULL;
  gchar *content2 = NULL;
  gsize len1, len2;
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
  EXPECT_EQ (len1, len2);
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
  g_remove (tmp);
  g_free (tmp);
}

/**
 * @brief Test tensor_if other/tensors stream test
 */
TEST_F (tensor_if_run, action_1)
{
  gchar *content1 = NULL;
  gchar *content2 = NULL;
  gsize len1, len2;
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
  EXPECT_EQ (len1, len2);
  EXPECT_EQ (memcmp (content1, content2, len1), 0);
  g_free (content1);
  g_free (content2);


  /* False action result */
  EXPECT_TRUE (g_file_get_contents ("./gamut.golden", &content1, &len1, NULL));
  _wait_pipeline_save_files (tmp_false, content2, len2, len1, TEST_TIMEOUT_MS);
  EXPECT_EQ (len1, len2);
  EXPECT_EQ (memcmp (content1, content2, len1), 0);
  g_free (content1);
  g_free (content2);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  gst_object_unref (pipeline);

  g_free (str_pipeline);

  g_remove (tmp_true);
  g_remove (tmp_false);
  g_free (tmp_true);
  g_free (tmp_false);
}

#define change_transform_type(type, size)                                                         \
  do {                                                                                            \
    g_object_set (transform_handle, "option", type, NULL);                                        \
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT),  \
        0);                                                                                       \
    g_usleep (100000);                                                                            \
    _wait_pipeline_save_files (tmp1, content1, len1, size, TEST_TIMEOUT_MS);                      \
    _wait_pipeline_save_files (tmp2, content2, len2, size, TEST_TIMEOUT_MS);                      \
    EXPECT_EQ (len1, len2);                                                                       \
    EXPECT_EQ (memcmp (content1, content2, len1), 0);                                             \
    g_free (content1);                                                                            \
    g_free (content2);                                                                            \
                                                                                                  \
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0); \
    g_usleep (100000);                                                                            \
                                                                                                  \
  } while (0);

/**
 * @brief Test tensor_if compared value with all tensor data type
 */
TEST_F (tensor_if_run, action_2)
{
  gchar *content1 = NULL;
  gchar *content2 = NULL;
  gsize len1, len2;
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

  change_transform_type ("uint8", 57600 * sizeof (uint8_t));
  change_transform_type ("uint16", 57600 * sizeof (uint16_t));
  change_transform_type ("uint32", 57600 * sizeof (uint32_t));
  change_transform_type ("uint64", 57600 * sizeof (uint64_t));
  change_transform_type ("int8", 57600 * sizeof (int8_t));
  change_transform_type ("int16", 57600 * sizeof (int16_t));
  change_transform_type ("int32", 57600 * sizeof (int32_t));
  change_transform_type ("int64", 57600 * sizeof (int64_t));
  change_transform_type ("float32", 57600 * sizeof (float));
  change_transform_type ("float64", 57600 * sizeof (double));

  gst_object_unref (transform_handle);
  gst_object_unref (pipeline);
  g_free (str_pipeline);

  g_remove (tmp1);
  g_remove (tmp2);
  g_free (tmp1);
  g_free (tmp2);
}

/**
 * @brief Test Tensor_if compared-value-option with undefined dimension properties
 */
TEST_F (tensor_if_run, action_3)
{
  gchar *content1 = NULL;
  gchar *content2 = NULL;
  gsize len1, len2;
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
  EXPECT_EQ (len1, len2);
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
  g_remove (tmp);
  g_free (tmp);
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
  gint index = *(gint *)user_data;
  gboolean ret;

  data_received++;
  /* Index 100 means a callback that is not allowed. */
  EXPECT_NE (100, index);
  mem_res = gst_buffer_get_memory (buffer, 0);
  ret = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
  ASSERT_TRUE (ret);
  output = (gint *)info_res.data;

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

  g_signal_connect (sink_handle, "new-data", (GCallback)new_data_cb, (gpointer)&idx);

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

  g_signal_connect (sink_handle, "new-data", (GCallback)new_data_cb, (gpointer)&idx);
  gst_object_unref (sink_handle);

  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sink_false");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback)new_data_cb, (gpointer)&idx);
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

  idx = *(gint *)user_data;
  output = (gint *)input[idx].data;
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
      nnstreamer_if_custom_register ("tifx", tensor_if_custom_cb, (gpointer)&idx));

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  appsrc_handle = gst_bin_get_by_name (GST_BIN (pipeline), "appsrc");
  EXPECT_NE (appsrc_handle, nullptr);

  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sink_true");
  EXPECT_NE (sink_handle, nullptr);

  tif_handle = gst_bin_get_by_name (GST_BIN (pipeline), "tif");
  EXPECT_NE (tif_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback)new_data_cb, (gpointer)&idx);
  gst_object_unref (sink_handle);

  sink_handle = gst_bin_get_by_name (GST_BIN (pipeline), "sink_false");
  EXPECT_NE (sink_handle, nullptr);

  g_signal_connect (sink_handle, "new-data", (GCallback)new_data_cb, (gpointer)&idx);
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
