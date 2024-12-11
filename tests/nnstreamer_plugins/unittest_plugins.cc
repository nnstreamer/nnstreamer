/**
 * @file	unittest_plugins.cc
 * @date	7 November 2018
 * @brief	Unit test for nnstreamer plugins. (testcases to check data conversion or buffer transfer)
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs.
 */

#include <gtest/gtest.h>
#include <glib/gstdio.h>
#include <gst/check/gstcheck.h>
#include <gst/check/gstharness.h>
#include <gst/check/gsttestclock.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api_converter.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_subplugin.h>
#include <string.h>
#include <tensor_common.h>
#include <tensor_meta.h>
#include <unistd.h>

#include "../gst/nnstreamer/elements/gsttensor_sparseutil.h"
#include "../gst/nnstreamer/elements/gsttensor_transform.h"
#include "../unittest_util.h"

#if defined(ENABLE_TENSORFLOW_LITE) || defined(ENABLE_TENSORFLOW2_LITE)
#define TEST_REQUIRE_TFLITE(Case, Name) TEST (Case, Name)
#else
#define TEST_REQUIRE_TFLITE(Case, Name) TEST (Case, DISABLED_##Name)
#endif

/**
 * @brief Macro for default value of the transform's 'acceleration' property
 */
#ifdef HAVE_ORC
#define DEFAULT_VAL_PROP_ACCELERATION TRUE
#else
#define DEFAULT_VAL_PROP_ACCELERATION FALSE
#endif

#define str(s) #s
#define TEST_TRANSFORM_TYPECAST(                                               \
    name, num_bufs, size, from_t, from_nns_t, to_t, str_to_t, to_nns_t, accel) \
  TEST (testTensorTransform, name)                                             \
  {                                                                            \
    const guint num_buffers = num_bufs;                                        \
    const guint array_size = size;                                             \
                                                                               \
    GstHarness *h;                                                             \
    GstBuffer *in_buf, *out_buf;                                               \
    GstTensorsConfig config;                                                   \
    GstMemory *mem;                                                            \
    GstMapInfo info;                                                           \
    guint i, b;                                                                \
    gsize data_in_size, data_out_size;                                         \
                                                                               \
    h = gst_harness_new ("tensor_transform");                                  \
                                                                               \
    g_object_set (h->element, "mode", GTT_TYPECAST, "option", str_to_t, NULL); \
    g_object_set (h->element, "acceleration", (gboolean) accel, NULL);         \
    /** input tensor info */                                                   \
    gst_tensors_config_init (&config);                                         \
    config.info.num_tensors = 1U;                                              \
    config.info.info[0].type = from_nns_t;                                     \
    gst_tensor_parse_dimension (str (size), config.info.info[0].dimension);    \
    config.rate_n = 0;                                                         \
    config.rate_d = 1;                                                         \
                                                                               \
    gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));      \
    data_in_size = gst_tensors_info_get_size (&config.info, 0);                \
                                                                               \
    config.info.info[0].type = to_nns_t;                                       \
    data_out_size = gst_tensors_info_get_size (&config.info, 0);               \
                                                                               \
    /** push buffers */                                                        \
    for (b = 0; b < num_buffers; b++) {                                        \
      /** set input buffer */                                                  \
      in_buf = gst_harness_create_buffer (h, data_in_size);                    \
                                                                               \
      mem = gst_buffer_peek_memory (in_buf, 0);                                \
      ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));                \
                                                                               \
      for (i = 0; i < array_size; i++) {                                       \
        from_t value = (i + 1) * (b + 1);                                      \
        ((from_t *) info.data)[i] = value;                                     \
      }                                                                        \
                                                                               \
      gst_memory_unmap (mem, &info);                                           \
                                                                               \
      EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);                   \
                                                                               \
      /** get output buffer */                                                 \
      out_buf = gst_harness_pull (h);                                          \
                                                                               \
      ASSERT_TRUE (out_buf != NULL);                                           \
      ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);                           \
      ASSERT_EQ (gst_buffer_get_size (out_buf), data_out_size);                \
                                                                               \
      mem = gst_buffer_peek_memory (out_buf, 0);                               \
      ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));                 \
                                                                               \
      for (i = 0; i < array_size; i++) {                                       \
        to_t expected = (i + 1) * (b + 1);                                     \
        EXPECT_EQ (((to_t *) info.data)[i], expected);                         \
      }                                                                        \
                                                                               \
      gst_memory_unmap (mem, &info);                                           \
      gst_buffer_unref (out_buf);                                              \
    }                                                                          \
    EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);                 \
    gst_harness_teardown (h);                                                  \
  }

#define GET_MODEL_PATH(model_name)                                      \
  do {                                                                  \
    const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");  \
                                                                        \
    if (root_path == NULL)                                              \
      root_path = "..";                                                 \
                                                                        \
    test_model = g_build_filename (                                     \
        root_path, "tests", "test_models", "models", model_name, NULL); \
    ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));         \
  } while (0)

/**
 * @brief Macro for tensor filter auto option test
 */
#define TEST_TENSOR_FILTER_AUTO_OPTION_P(gstpipe, fw_name)       \
  do {                                                           \
    GstElement *filter;                                          \
    gchar *prop_string;                                          \
                                                                 \
    filter = gst_bin_get_by_name (GST_BIN (gstpipe), "tfilter"); \
    EXPECT_NE (filter, nullptr);                                 \
    g_object_get (filter, "framework", &prop_string, NULL);      \
    EXPECT_STREQ (prop_string, fw_name);                         \
                                                                 \
    g_free (prop_string);                                        \
    gst_object_unref (filter);                                   \
  } while (0)

/**
 * @brief Macro for check erroneous pipeline
 */
#define TEST_TENSOR_FILTER_AUTO_OPTION_N(gstpipe, fw_name)                  \
  do {                                                                      \
    GstStateChangeReturn ret;                                               \
                                                                            \
    if (fw_name) {                                                          \
      GstElement *filter;                                                   \
      gchar *prop_string;                                                   \
      filter = gst_bin_get_by_name (GST_BIN (gstpipe), "tfilter");          \
      EXPECT_NE (filter, nullptr);                                          \
      g_object_get (filter, "framework", &prop_string, NULL);               \
      EXPECT_STREQ (prop_string, fw_name);                                  \
      g_free (prop_string);                                                 \
      gst_object_unref (filter);                                            \
    }                                                                       \
    gst_element_set_state (gstpipe, GST_STATE_PLAYING);                     \
    g_usleep (100000);                                                      \
    ret = gst_element_get_state (gstpipe, NULL, NULL, GST_CLOCK_TIME_NONE); \
    EXPECT_TRUE (ret == GST_STATE_CHANGE_FAILURE);                          \
                                                                            \
  } while (0)

#define wait_for_element_state(element, state)                                  \
  do {                                                                          \
    GstState cur_state = GST_STATE_VOID_PENDING;                                \
    GstStateChangeReturn ret;                                                   \
    gint counter = 0;                                                           \
    ret = gst_element_set_state (element, state);                               \
    EXPECT_TRUE (ret != GST_STATE_CHANGE_FAILURE);                              \
    while (cur_state != state && counter < 20) {                                \
      g_usleep (50000);                                                         \
      counter++;                                                                \
      ret = gst_element_get_state (element, &cur_state, NULL, 5 * GST_MSECOND); \
      EXPECT_TRUE (ret != GST_STATE_CHANGE_FAILURE);                            \
    }                                                                           \
    EXPECT_TRUE (cur_state == state);                                           \
    g_usleep (50000);                                                           \
  } while (0)

/**
 * @brief wait for output buffer on GstHarness sinkpad.
 */
static guint
_harness_wait_for_output_buffer (GstHarness *h, guint expected)
{
  guint received, count;

  received = count = 0;
  do {
    g_usleep (100000);
    received = gst_harness_buffers_received (h);
    count++;
  } while (received < expected && count < 30);

  return received;
}

/**
 * @brief Test for setting/getting properties of tensor_transform
 */
TEST (testTensorTransform, properties01)
{
  const gboolean default_silent = TRUE;
  const gboolean default_accl = DEFAULT_VAL_PROP_ACCELERATION;
  const gint default_mode = GTT_TYPECAST; /* typecast */
  const gchar default_option[] = "uint32";
  gchar *str_launch_line;
  gint res_mode;
  gchar *res_option = NULL;
  gboolean silent, res_silent;
  gboolean accl;
  GstHarness *hrnss;
  GstElement *transform;

  hrnss = gst_harness_new_empty ();
  ASSERT_TRUE (hrnss != NULL);

  str_launch_line = g_strdup_printf (
      "tensor_transform mode=%d option=%s", default_mode, default_option);
  gst_harness_add_parse (hrnss, str_launch_line);
  g_free (str_launch_line);
  transform = gst_harness_find_element (hrnss, "tensor_transform");
  ASSERT_TRUE (transform != NULL);

  /** default silent is TRUE */
  g_object_get (transform, "silent", &silent, NULL);
  EXPECT_EQ (default_silent, silent);

  g_object_set (transform, "silent", !default_silent, NULL);
  g_object_get (transform, "silent", &res_silent, NULL);
  /** expect FALSE, which is !default_silent */
  EXPECT_FALSE (res_silent);

  /**
   * If HAVE_ORC is set, default acceleration is TRUE.
   * Otherwise the default value is FALSE.
   */
  g_object_get (transform, "acceleration", &accl, NULL);
  EXPECT_EQ (default_accl, accl);

#ifdef HAVE_ORC
  g_object_set (transform, "acceleration", !default_accl, NULL);
  g_object_get (transform, "acceleration", &accl, NULL);
  /** expect FALSE, which is !default_accl */
  EXPECT_FALSE (accl);
#endif

  /** We do not need to test setting properties for 'mode' and 'option' */
  g_object_get (transform, "mode", &res_mode, NULL);
  EXPECT_EQ (default_mode, res_mode);

  g_object_get (transform, "option", &res_option, NULL);
  EXPECT_STREQ (default_option, res_option);
  g_free (res_option);

  g_object_unref (transform);
  gst_harness_teardown (hrnss);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, properties02_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* invalid option (unknown mode) */
  g_object_set (h->element, "mode", "unknown", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, dimchgProperties0_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of ^([0-9]|1[0-5]):([0-9]|1[0-5]) */
  g_object_set (h->element, "mode", GTT_DIMCHG, "option", "20:21", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, dimchgProperties1_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of ^([0-9]|1[0-5]):([0-9]|1[0-5]) */
  g_object_set (h->element, "mode", GTT_DIMCHG, "option", "1,2", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, dimchgProperties2_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* Option is not given */
  g_object_set (h->element, "mode", GTT_DIMCHG, NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, dimchgProperties3_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of ^([0-9]|1[0-5]):([0-9]|1[0-5]) */
  g_object_set (h->element, "mode", GTT_DIMCHG, "option", "0:2,1:3", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, arithmeticProperties0_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of [typecast:TYPE,]add|mul|div:NUMBER..., */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "typecast", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, arithmeticProperties1_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of [typecast:TYPE,]add|mul|div:NUMBER..., */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "typecast:unknown", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, arithmeticProperties2_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of [typecast:TYPE,]add|mul|div:NUMBER..., */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "typecast:char", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, arithmeticProperties3_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of [typecast:TYPE,]add|mul|div:NUMBER..., */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "typecast:double", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, arithmeticProperties4_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of [typecast:TYPE,]add|mul|div:NUMBER..., */
  g_object_set (h->element, "mode", "typecast:int8", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, arithmeticProperties5_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of [typecast:TYPE,]add|mul|div:NUMBER..., */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:0xF", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, arithmeticProperties6_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of [typecast:TYPE,]add|mul|div:NUMBER..., */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:1U", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, arithmeticProperties7_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of [typecast:TYPE,]add|mul|div:NUMBER..., */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "+2", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, arithmeticProperties8_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of [typecast:TYPE,]add|mul|div:NUMBER..., */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "*2", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, arithmeticProperties9_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of [typecast:TYPE,]add|mul|div:NUMBER..., */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "/2", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, arithmeticProperties10_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of [typecast:TYPE,]add|mul|div:NUMBER..., */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "max", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, arithmeticProperties11_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* Option is not given */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, arithmeticProperties12_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "per-channel:false", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, arithmeticProperties13_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option",
      "per-channel:invalid,add:1@2", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, transposeProperties0_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of NEW_IDX_DIM0:NEW_IDX_DIM1:NEW_IDX_DIM2:3 */
  g_object_set (h->element, "mode", GTT_TRANSPOSE, "option", "5:2:4:3", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, transposeProperties1_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of NEW_IDX_DIM0:NEW_IDX_DIM1:NEW_IDX_DIM2:3 */
  g_object_set (h->element, "mode", GTT_TRANSPOSE, "option", "2:3:1:0", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, transposeProperties2_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of NEW_IDX_DIM0:NEW_IDX_DIM1:NEW_IDX_DIM2:3 */
  g_object_set (h->element, "mode", GTT_TRANSPOSE, "option", "0:3", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, transposeProperties3_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* Option is not given */
  g_object_set (h->element, "mode", GTT_TRANSPOSE, NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, clmapProperties0_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* it should be in the form of [CLAMP_MIN:CLAMP_MAX] */
  g_object_set (h->element, "mode", GTT_CLAMP, "option", "50:20", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, clmapProperties1_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* it should be in the form of [CLAMP_MIN:CLAMP_MAX] */
  g_object_set (h->element, "mode", GTT_CLAMP, "option", "+-1", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, clmapProperties2_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* it should be in the form of [CLAMP_MIN:CLAMP_MAX] */
  g_object_set (h->element, "mode", GTT_CLAMP, "option", "+1:-2", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, clmapProperties3_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* overflow case */
  g_object_set (h->element, "mode", GTT_CLAMP, "option", "1:1.7e309", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, clmapProperties4_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* underflow case */
  g_object_set (h->element, "mode", GTT_CLAMP, "option", "-1.7e309:1", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, clmapProperties5_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* Option is not given */
  g_object_set (h->element, "mode", GTT_CLAMP, NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, standProperties0_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* invalid option (stand mode) */
  g_object_set (h->element, "mode", GTT_STAND, "option", "invalid", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, standProperties1_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of (default|dc-average)[:TYPE][,per-channel:(false|true)] */
  g_object_set (h->element, "mode", GTT_STAND, "option", "dc-average:unknown", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, standProperties2_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of (default|dc-average)[:TYPE][,per-channel:(false|true)] */
  g_object_set (h->element, "mode", GTT_STAND, "option", "dc-average,per-channel", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, standProperties3_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of (default|dc-average)[:TYPE][,per-channel:(false|true)] */
  g_object_set (h->element, "mode", GTT_STAND, "option",
      "dc-average:uint8,per-channel:yes", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, standProperties4_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* It should be in the form of (default|dc-average)[:TYPE][,per-channel:(false|true)] */
  g_object_set (h->element, "mode", GTT_STAND, "option", "dc-average:uint8,true", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (testTensorTransform, standProperties5_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* Option is not given */
  g_object_set (h->element, "mode", GTT_STAND, NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform typecast (uint8 > uint32)
 */
TEST_TRANSFORM_TYPECAST (typecast_1, 3U, 5U, uint8_t, _NNS_UINT8, uint32_t,
    "uint32", _NNS_UINT32, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, uint8 > uint32)
 */
TEST_TRANSFORM_TYPECAST (typecast_1_accel, 3U, 5U, uint8_t, _NNS_UINT8,
    uint32_t, "uint32", _NNS_UINT32, TRUE)

/**
 * @brief Test for tensor_transform typecast (uint32 > float64)
 */
TEST_TRANSFORM_TYPECAST (typecast_2, 3U, 5U, uint32_t, _NNS_UINT32, double,
    "float64", _NNS_FLOAT64, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, uint32 > float64)
 */
TEST_TRANSFORM_TYPECAST (typecast_2_accel, 3U, 5U, uint32_t, _NNS_UINT32,
    double, "float64", _NNS_FLOAT64, TRUE)

/**
 * @brief Test for tensor_transform typecast (int32 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_3, 3U, 5U, int32_t, _NNS_INT32, float,
    "float32", _NNS_FLOAT32, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, int32 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_3_accel, 3U, 5U, int32_t, _NNS_INT32, float,
    "float32", _NNS_FLOAT32, TRUE)

/**
 * @brief Test for tensor_transform typecast (int8 > float32)
 */
TEST_TRANSFORM_TYPECAST (
    typecast_4, 3U, 5U, int8_t, _NNS_INT8, float, "float32", _NNS_FLOAT32, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, int8 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_4_accel, 3U, 5U, int8_t, _NNS_INT8, float,
    "float32", _NNS_FLOAT32, TRUE)

/**
 * @brief Test for tensor_transform typecast (uint8 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_5, 3U, 5U, uint8_t, _NNS_UINT8, float,
    "float32", _NNS_FLOAT32, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, uint8 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_5_accel, 3U, 5U, uint8_t, _NNS_UINT8, float,
    "float32", _NNS_FLOAT32, TRUE)

/**
 * @brief Test for tensor_transform typecast (int16 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_6, 3U, 5U, int16_t, _NNS_INT16, float,
    "float32", _NNS_FLOAT32, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, int16 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_6_accel, 3U, 5U, int16_t, _NNS_INT16, float,
    "float32", _NNS_FLOAT32, TRUE)

/**
 * @brief Test for tensor_transform typecast (uint16 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_7, 3U, 5U, uint16_t, _NNS_UINT16, float,
    "float32", _NNS_FLOAT32, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, uint16 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_7_accel, 3U, 5U, uint16_t, _NNS_UINT16, float,
    "float32", _NNS_FLOAT32, TRUE)

/**
 * @brief Test for tensor_transform typecast (uint64 -> int64)
 */
TEST_TRANSFORM_TYPECAST (typecast_8, 3U, 5U, uint64_t, _NNS_UINT64, int64_t,
    "int64", _NNS_INT64, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, uint64 -> int64)
 */
TEST_TRANSFORM_TYPECAST (typecast_8_accel, 3U, 5U, uint64_t, _NNS_UINT64,
    int64_t, "int64", _NNS_INT64, TRUE)

/**
 * @brief Test for tensor_transform typecast (float -> uint32)
 */
TEST_TRANSFORM_TYPECAST (typecast_9, 3U, 5U, float, _NNS_FLOAT32, uint32_t,
    "uint32", _NNS_UINT32, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, float -> uint32)
 */
TEST_TRANSFORM_TYPECAST (typecast_9_accel, 3U, 5U, float, _NNS_FLOAT32,
    uint32_t, "uint32", _NNS_UINT32, TRUE)

/**
 * @brief Test for tensor_transform typecast (uint8 -> int8)
 */
TEST_TRANSFORM_TYPECAST (
    typecast_10, 3U, 5U, uint8_t, _NNS_UINT8, int8_t, "int8", _NNS_INT8, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, uint8 -> int8)
 */
TEST_TRANSFORM_TYPECAST (typecast_10_accel, 3U, 5U, uint8_t, _NNS_UINT8, int8_t,
    "int8", _NNS_INT8, TRUE)

/**
 * @brief Test for tensor_transform typecast (uint32 -> int16)
 */
TEST_TRANSFORM_TYPECAST (typecast_11, 3U, 5U, uint32_t, _NNS_UINT32, int16_t,
    "int16", _NNS_INT16, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, uint32 -> int16)
 */
TEST_TRANSFORM_TYPECAST (typecast_11_accel, 3U, 5U, uint32_t, _NNS_UINT32,
    int16_t, "int16", _NNS_INT16, TRUE)

/**
 * @brief Test for tensor_transform typecast (float -> uint8)
 */
TEST_TRANSFORM_TYPECAST (typecast_12, 3U, 5U, float, _NNS_FLOAT32, uint8_t,
    "uint8", _NNS_UINT8, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, float -> uint8)
 */
TEST_TRANSFORM_TYPECAST (typecast_12_accel, 3U, 5U, float, _NNS_FLOAT32,
    uint8_t, "uint8", _NNS_UINT8, TRUE)

/**
 * @brief Test for tensor_transform typecast (double -> uint16)
 */
TEST_TRANSFORM_TYPECAST (typecast_13, 3U, 5U, double, _NNS_FLOAT64, uint16_t,
    "uint16", _NNS_UINT16, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, double -> uint16)
 */
TEST_TRANSFORM_TYPECAST (typecast_13_accel, 3U, 5U, double, _NNS_FLOAT64,
    uint16_t, "uint16", _NNS_UINT16, TRUE)

/**
 * @brief Test for tensor_transform typecast (double -> uint64)
 */
TEST_TRANSFORM_TYPECAST (typecast_14, 3U, 5U, double, _NNS_FLOAT64, uint64_t,
    "uint64", _NNS_UINT64, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, double -> uint64)
 */
TEST_TRANSFORM_TYPECAST (typecast_14_accel, 3U, 5U, double, _NNS_FLOAT64,
    uint64_t, "uint64", _NNS_UINT64, TRUE)

/**
 * @brief Test for tensor_transform arithmetic (float32, add .5)
 */
TEST (testTensorTransform, arithmetic1)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:.5", NULL);
  g_object_set (h->element, "acceleration", (gboolean) FALSE, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (b = 0; b < num_buffers; b++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    for (i = 0; i < array_size; i++) {
      float value = (i + 1) * (b + 1) + .2;
      ((float *) info.data)[i] = value;
    }

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
    ASSERT_EQ (gst_buffer_get_size (out_buf), data_size);

    mem = gst_buffer_peek_memory (out_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

    for (i = 0; i < array_size; i++) {
      float expected = (i + 1) * (b + 1) + .2 + .5;
      EXPECT_FLOAT_EQ (((float *) info.data)[i], expected);
    }

    gst_memory_unmap (mem, &info);
    gst_buffer_unref (out_buf);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic (acceleration, float32, add .5)
 */
TEST (testTensorTransform, arithmetic1Accel)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:.5", NULL);
  g_object_set (h->element, "acceleration", (gboolean) TRUE, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (b = 0; b < num_buffers; b++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    for (i = 0; i < array_size; i++) {
      float value = (i + 1) * (b + 1) + .2;
      ((float *) info.data)[i] = value;
    }

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
    ASSERT_EQ (gst_buffer_get_size (out_buf), data_size);

    mem = gst_buffer_peek_memory (out_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

    for (i = 0; i < array_size; i++) {
      float expected = (i + 1) * (b + 1) + .2 + .5;
      EXPECT_FLOAT_EQ (((float *) info.data)[i], expected);
    }

    gst_memory_unmap (mem, &info);
    gst_buffer_unref (out_buf);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic (float64, mul .5)
 */
TEST (testTensorTransform, arithmetic2)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "mul:.5", NULL);
  g_object_set (h->element, "acceleration", (gboolean) FALSE, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_FLOAT64;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (b = 0; b < num_buffers; b++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    for (i = 0; i < array_size; i++) {
      double value = (i + 1) * (b + 1) + .2;
      ((double *) info.data)[i] = value;
    }

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
    ASSERT_EQ (gst_buffer_get_size (out_buf), data_size);

    mem = gst_buffer_peek_memory (out_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

    for (i = 0; i < array_size; i++) {
      double expected = ((i + 1) * (b + 1) + .2) * .5;
      EXPECT_DOUBLE_EQ (((double *) info.data)[i], expected);
    }

    gst_memory_unmap (mem, &info);
    gst_buffer_unref (out_buf);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic (acceleration, float64, mul .5)
 */
TEST (testTensorTransform, arithmetic2Accel)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "mul:.5", NULL);
  g_object_set (h->element, "acceleration", (gboolean) TRUE, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_FLOAT64;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (b = 0; b < num_buffers; b++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    for (i = 0; i < array_size; i++) {
      double value = (i + 1) * (b + 1) + .2;
      ((double *) info.data)[i] = value;
    }

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
    ASSERT_EQ (gst_buffer_get_size (out_buf), data_size);

    mem = gst_buffer_peek_memory (out_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

    for (i = 0; i < array_size; i++) {
      double expected = ((i + 1) * (b + 1) + .2) * .5;
      EXPECT_DOUBLE_EQ (((double *) info.data)[i], expected);
    }

    gst_memory_unmap (mem, &info);
    gst_buffer_unref (out_buf);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic (typecast uint8 > float32, add .5, mul .2)
 */
TEST (testTensorTransform, arithmetic3)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option",
      "typecast:float32,add:.5,mul:0.2", NULL);
  g_object_set (h->element, "acceleration", (gboolean) FALSE, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  config.info.info[0].type = _NNS_FLOAT32;
  data_out_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (b = 0; b < num_buffers; b++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_in_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    for (i = 0; i < array_size; i++) {
      uint8_t value = (i + 1) * (b + 1);
      ((uint8_t *) info.data)[i] = value;
    }

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
    ASSERT_EQ (gst_buffer_get_size (out_buf), data_out_size);

    mem = gst_buffer_peek_memory (out_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

    for (i = 0; i < array_size; i++) {
      float expected = ((i + 1) * (b + 1) + .5) * .2;
      EXPECT_FLOAT_EQ (((float *) info.data)[i], expected);
    }

    gst_memory_unmap (mem, &info);
    gst_buffer_unref (out_buf);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic (acceleration, typecast uint8 > float32, add .5, mul .2)
 */
TEST (testTensorTransform, arithmetic3Accel)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option",
      "typecast:float32,add:.5,mul:0.2", NULL);
  g_object_set (h->element, "acceleration", (gboolean) TRUE, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  config.info.info[0].type = _NNS_FLOAT32;
  data_out_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (b = 0; b < num_buffers; b++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_in_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    for (i = 0; i < array_size; i++) {
      uint8_t value = (i + 1) * (b + 1);
      ((uint8_t *) info.data)[i] = value;
    }

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
    ASSERT_EQ (gst_buffer_get_size (out_buf), data_out_size);

    mem = gst_buffer_peek_memory (out_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

    for (i = 0; i < array_size; i++) {
      float expected = ((i + 1) * (b + 1) + .5) * .2;
      EXPECT_FLOAT_EQ (((float *) info.data)[i], expected);
    }

    gst_memory_unmap (mem, &info);
    gst_buffer_unref (out_buf);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic (typecast uint8 > float64, add .2, add .1, final typecast uint16 will be ignored)
 */
TEST (testTensorTransform, arithmetic4)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option",
      "typecast:float64,add:0.2,add:0.1,typecast:uint16", NULL);
  g_object_set (h->element, "acceleration", (gboolean) FALSE, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  config.info.info[0].type = _NNS_FLOAT64;
  data_out_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (b = 0; b < num_buffers; b++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_in_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    for (i = 0; i < array_size; i++) {
      uint8_t value = (i + 1) * (b + 1);
      ((uint8_t *) info.data)[i] = value;
    }

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
    ASSERT_EQ (gst_buffer_get_size (out_buf), data_out_size);

    mem = gst_buffer_peek_memory (out_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

    for (i = 0; i < array_size; i++) {
      double expected = (i + 1) * (b + 1) + .3;
      EXPECT_DOUBLE_EQ (((double *) info.data)[i], expected);
    }

    gst_memory_unmap (mem, &info);
    gst_buffer_unref (out_buf);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic.
 * - option : acceleration, typecast uint8 > float64, add .2, add .1
 * - final typecast uint16 will be ignored.
 */
TEST (testTensorTransform, arithmetic4Accel)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option",
      "typecast:float64,add:0.2,add:0.1,typecast:uint16", NULL);
  g_object_set (h->element, "acceleration", (gboolean) TRUE, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  config.info.info[0].type = _NNS_FLOAT64;
  data_out_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (b = 0; b < num_buffers; b++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_in_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    for (i = 0; i < array_size; i++) {
      uint8_t value = (i + 1) * (b + 1);
      ((uint8_t *) info.data)[i] = value;
    }

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
    ASSERT_EQ (gst_buffer_get_size (out_buf), data_out_size);

    mem = gst_buffer_peek_memory (out_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

    for (i = 0; i < array_size; i++) {
      double expected = (i + 1) * (b + 1) + .3;
      EXPECT_DOUBLE_EQ (((double *) info.data)[i], expected);
    }

    gst_memory_unmap (mem, &info);
    gst_buffer_unref (out_buf);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic (typecast uint8 > int32, mul 2, div 2, add -1)
 */
TEST (testTensorTransform, arithmetic5)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option",
      "typecast:int32,mul:2,div:2,add:-1", NULL);
  g_object_set (h->element, "acceleration", (gboolean) FALSE, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  config.info.info[0].type = _NNS_INT32;
  data_out_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (b = 0; b < num_buffers; b++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_in_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    for (i = 0; i < array_size; i++) {
      uint8_t value = (i + 1) * (b + 1);
      ((uint8_t *) info.data)[i] = value;
    }

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
    ASSERT_EQ (gst_buffer_get_size (out_buf), data_out_size);

    mem = gst_buffer_peek_memory (out_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

    for (i = 0; i < array_size; i++) {
      int32_t expected = (i + 1) * (b + 1) - 1;
      EXPECT_EQ (((int32_t *) info.data)[i], expected);
    }

    gst_memory_unmap (mem, &info);
    gst_buffer_unref (out_buf);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic (acceleration, typecast uint8 > int32, mul 2, div 2, add -1)
 */
TEST (testTensorTransform, arithmetic5Accel)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option",
      "typecast:int32,mul:2,div:2,add:-1", NULL);
  g_object_set (h->element, "acceleration", (gboolean) TRUE, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  config.info.info[0].type = _NNS_INT32;
  data_out_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (b = 0; b < num_buffers; b++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_in_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    for (i = 0; i < array_size; i++) {
      uint8_t value = (i + 1) * (b + 1);
      ((uint8_t *) info.data)[i] = value;
    }

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
    ASSERT_EQ (gst_buffer_get_size (out_buf), data_out_size);

    mem = gst_buffer_peek_memory (out_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

    for (i = 0; i < array_size; i++) {
      int32_t expected = (i + 1) * (b + 1) - 1;
      EXPECT_EQ (((int32_t *) info.data)[i], expected);
    }

    gst_memory_unmap (mem, &info);
    gst_buffer_unref (out_buf);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic, per-channel
 */
TEST (testTensorTransform, arithmeticPerChannel)
{
  const guint num_buffers = 3;
  const guint array_size = 5; /* channel size : 5 */

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option",
      "typecast:int32,per-channel:true@0,mul:1@0,mul:2@1,mul:3@2,mul:4@3,mul:5@4,add:-1@0,add:-2@1,add:-3@2,add:-4@3,add:-5@4",
      NULL);
  g_object_set (h->element, "acceleration", (gboolean) FALSE, NULL);

  /**
   * 1  2  3  4  5 -> 0  2  6 12 20
   * 2  4  6  8 10 -> 1  6 15 28 45
   * 3  6  9 12 15 -> 2 10 24 44 70
   */

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  config.info.info[0].type = _NNS_INT32;
  data_out_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (b = 0; b < num_buffers; b++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_in_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    for (i = 0; i < array_size; i++) {
      uint8_t value = (i + 1) * (b + 1);
      ((uint8_t *) info.data)[i] = value;
    }

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
    ASSERT_EQ (gst_buffer_get_size (out_buf), data_out_size);

    mem = gst_buffer_peek_memory (out_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

    for (i = 0; i < array_size; i++) {
      int32_t expected = ((i + 1) * (b + 1)) * (i + 1) - (i + 1);
      EXPECT_EQ (((int32_t *) info.data)[i], expected);
    }

    gst_memory_unmap (mem, &info);
    gst_buffer_unref (out_buf);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic (changing option string dynamically)
 */
TEST (testTensorTransform, arithmeticChangeOptionString)
{
  const guint array_size = 5;
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i;
  gsize data_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:.5", NULL);
  g_object_set (h->element, "acceleration", (gboolean) FALSE, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_size = gst_tensors_info_get_size (&config.info, 0);
  in_buf = gst_harness_create_buffer (h, data_size);

  mem = gst_buffer_peek_memory (in_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

  for (i = 0; i < array_size; i++) {
    float value = (i + 1) * (i * 3 + 1) + .2;
    ((float *) info.data)[i] = value;
  }

  gst_memory_unmap (mem, &info);

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  /* get output buffer */
  out_buf = gst_harness_pull (h);

  ASSERT_TRUE (out_buf != NULL);
  ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
  ASSERT_EQ (gst_buffer_get_size (out_buf), data_size);

  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

  for (i = 0; i < array_size; i++) {
    float expected = (i + 1) * (i * 3 + 1) + .2 + .5;
    EXPECT_FLOAT_EQ (((float *) info.data)[i], expected);
  }

  gst_memory_unmap (mem, &info);
  gst_buffer_unref (out_buf);

  /** Change the option string during runtime */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "mul:20", NULL);
  in_buf = gst_harness_create_buffer (h, data_size);

  mem = gst_buffer_peek_memory (in_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

  for (i = 0; i < array_size; i++) {
    float value = (i + 1) * (i * 3 + 1) + .9;
    ((float *) info.data)[i] = value;
  }

  gst_memory_unmap (mem, &info);

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  /* get output buffer */
  out_buf = gst_harness_pull (h);

  ASSERT_TRUE (out_buf != NULL);
  ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
  ASSERT_EQ (gst_buffer_get_size (out_buf), data_size);

  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

  for (i = 0; i < array_size; i++) {
    float expected = ((i + 1) * (i * 3 + 1) + .9) * 20;
    EXPECT_FLOAT_EQ (((float *) info.data)[i], expected);
  }

  gst_memory_unmap (mem, &info);
  gst_buffer_unref (out_buf);

  EXPECT_EQ (gst_harness_buffers_received (h), 2U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic (flex tensor)
 */
TEST (testTensorTransform, arithmeticFlexTensor)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorInfo in_info, out_info;
  GstCaps *caps;
  GstMemory *mem;
  GstMapInfo map;
  guint i, b;
  uint8_t *_input;
  float *_output;
  gsize data_in_size, data_out_size, hsize;
  GstTensorMetaInfo meta;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option",
      "typecast:float32,add:.5,mul:0.2", NULL);

  /* in/out tensor info */
  gst_tensor_info_init (&in_info);
  in_info.type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", in_info.dimension);
  data_in_size = gst_tensor_info_get_size (&in_info);

  gst_tensor_info_copy (&out_info, &in_info);
  out_info.type = _NNS_FLOAT32;
  data_out_size = gst_tensor_info_get_size (&out_info);

  /* set caps (flex-tensor) */
  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);

  gst_harness_set_src_caps (h, gst_caps_copy (caps));
  gst_harness_set_sink_caps (h, caps);

  /* push buffers */
  for (b = 0; b < num_buffers; b++) {
    /* set input buffer */
    gst_tensor_info_convert_to_meta (&in_info, &meta);
    hsize = gst_tensor_meta_info_get_header_size (&meta);

    in_buf = gst_harness_create_buffer (h, data_in_size + hsize);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));
    gst_tensor_meta_info_update_header (&meta, map.data);

    _input = (uint8_t *) (map.data + hsize);
    for (i = 0; i < array_size; i++) {
      uint8_t value = (i + 1) * (b + 1);
      _input[i] = value;
    }

    gst_memory_unmap (mem, &map);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);

    mem = gst_buffer_peek_memory (out_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));

    gst_tensor_meta_info_parse_header (&meta, map.data);
    EXPECT_EQ (meta.type, _NNS_FLOAT32);
    EXPECT_EQ (meta.dimension[0], 5U);

    hsize = gst_tensor_meta_info_get_header_size (&meta);
    ASSERT_EQ (gst_buffer_get_size (out_buf), data_out_size + hsize);

    _output = (float *) (map.data + hsize);
    for (i = 0; i < array_size; i++) {
      float expected = ((i + 1) * (b + 1) + .5) * .2;
      EXPECT_FLOAT_EQ (_output[i], expected);
    }

    gst_memory_unmap (mem, &map);
    gst_buffer_unref (out_buf);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);
  gst_harness_teardown (h);
}

/**
 * @brief Test data for tensor_aggregator (2 frames with dimension 3:4:2:2 or 3:2:2:2:2)
 */
const gint aggr_test_frames[2][48]
    = { { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112,
            1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124,
            1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212,
            1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224 },
        { 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113,
            2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2201,
            2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213,
            2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224 } };

/**
 * @brief Test for tensor aggregator properties
 */
TEST (testTensorAggregator, properties)
{
  GstHarness *h;
  guint fr_val, res_fr_val;
  gboolean concat, res_concat;
  gboolean silent, res_silent;

  h = gst_harness_new ("tensor_aggregator");

  /* default frames-in is 1 */
  g_object_get (h->element, "frames-in", &fr_val, NULL);
  EXPECT_EQ (fr_val, 1U);

  fr_val = 2;
  g_object_set (h->element, "frames-in", fr_val, NULL);
  g_object_get (h->element, "frames-in", &res_fr_val, NULL);
  EXPECT_EQ (res_fr_val, fr_val);

  /* default frames-out is 1 */
  g_object_get (h->element, "frames-out", &fr_val, NULL);
  EXPECT_EQ (fr_val, 1U);

  fr_val = 2;
  g_object_set (h->element, "frames-out", fr_val, NULL);
  g_object_get (h->element, "frames-out", &res_fr_val, NULL);
  EXPECT_EQ (res_fr_val, fr_val);

  /* default frames-flush is 0 */
  g_object_get (h->element, "frames-flush", &fr_val, NULL);
  EXPECT_EQ (fr_val, 0U);

  fr_val = 2;
  g_object_set (h->element, "frames-flush", fr_val, NULL);
  g_object_get (h->element, "frames-flush", &res_fr_val, NULL);
  EXPECT_EQ (res_fr_val, fr_val);

  /* default frames-dim is (NNS_TENSOR_RANK_LIMIT - 1) */
  g_object_get (h->element, "frames-dim", &fr_val, NULL);
  EXPECT_EQ (fr_val, (guint) (NNS_TENSOR_RANK_LIMIT - 1));

  fr_val = 1;
  g_object_set (h->element, "frames-dim", fr_val, NULL);
  g_object_get (h->element, "frames-dim", &res_fr_val, NULL);
  EXPECT_EQ (res_fr_val, fr_val);

  /* default concat is TRUE */
  g_object_get (h->element, "concat", &concat, NULL);
  EXPECT_EQ (concat, TRUE);

  g_object_set (h->element, "concat", !concat, NULL);
  g_object_get (h->element, "concat", &res_concat, NULL);
  EXPECT_EQ (res_concat, !concat);

  /* default silent is TRUE */
  g_object_get (h->element, "silent", &silent, NULL);
  EXPECT_EQ (silent, TRUE);

  g_object_set (h->element, "silent", !silent, NULL);
  g_object_get (h->element, "silent", &res_silent, NULL);
  EXPECT_EQ (res_silent, !silent);

  gst_harness_teardown (h);
}

/**
 * @brief Internal function for tensor-aggregator test, push buffer to harness pad.
 */
static void
_aggregator_test_push_buffer (GstHarness *h, const gint *data, const gsize data_size)
{
  GstBuffer *buf;
  GstMemory *mem;
  GstMapInfo map;

  buf = gst_harness_create_buffer (h, data_size);

  mem = gst_buffer_peek_memory (buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));
  memcpy (map.data, data, data_size);
  gst_memory_unmap (mem, &map);

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_OK);
}

/**
 * @brief Internal function for tensor-aggregator test, check output data.
 */
static void
_aggregator_test_check_output (GstHarness *h, const gint *expected, const gint length)
{
  GstBuffer *output;
  GstMemory *mem;
  GstMapInfo map;
  gint i;

  output = gst_harness_pull (h);
  mem = gst_buffer_peek_memory (output, 0);
  ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
  ASSERT_TRUE (map.size == sizeof (gint) * length);

  for (i = 0; i < length; i++)
    EXPECT_EQ (((gint *) map.data)[i], expected[i]);

  gst_memory_unmap (mem, &map);
  gst_buffer_unref (output);
}

/**
 * @brief Test for tensor_aggregator (concatenate 2 frames with frames-dim 3, out-dimension 3:4:2:4)
 */
TEST (testTensorAggregator, aggregate1)
{
  GstHarness *h;
  GstTensorsConfig config;
  guint i;
  gsize data_in_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "frames-out", 2, "frames-dim", 3, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    _aggregator_test_push_buffer (h, aggr_test_frames[i], data_in_size);
  }

  /* get output buffer */
  const gint expected[96] = { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108,
    1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120,
    1121, 1122, 1123, 1124, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208,
    1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221,
    1222, 1223, 1224, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110,
    2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123,
    2124, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212,
    2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224 };

  _aggregator_test_check_output (h, expected, 96);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (concatenate 2 frames with frames-dim 2, out-dimension 3:4:4:2)
 */
TEST (testTensorAggregator, aggregate2)
{
  GstHarness *h;
  GstTensorsConfig config;
  guint i;
  gsize data_in_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "frames-out", 2, "frames-dim", 2, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    _aggregator_test_push_buffer (h, aggr_test_frames[i], data_in_size);
  }

  /* get output buffer */
  const gint expected[96] = { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108,
    1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120,
    1121, 1122, 1123, 1124, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108,
    2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121,
    2122, 2123, 2124, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210,
    1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223,
    1224, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212,
    2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224 };

  _aggregator_test_check_output (h, expected, 96);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (concatenate 2 frames with frames-dim 1, out-dimension 3:8:2:2)
 */
TEST (testTensorAggregator, aggregate3)
{
  GstHarness *h;
  GstTensorsConfig config;
  guint i;
  gsize data_in_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "frames-out", 2, "frames-dim", 1, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    _aggregator_test_push_buffer (h, aggr_test_frames[i], data_in_size);
  }

  /* get output buffer */
  const gint expected[96] = { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108,
    1109, 1110, 1111, 1112, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108,
    2109, 2110, 2111, 2112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120,
    1121, 1122, 1123, 1124, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121,
    2122, 2123, 2124, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210,
    1211, 1212, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211,
    2212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224,
    2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224 };

  _aggregator_test_check_output (h, expected, 96);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (concatenate 2 frames with frames-dim 0, out-dimension 6:4:2:2)
 */
TEST (testTensorAggregator, aggregate4)
{
  GstHarness *h;
  GstTensorsConfig config;
  guint i;
  gsize data_in_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "frames-out", 2, "frames-dim", 0, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    _aggregator_test_push_buffer (h, aggr_test_frames[i], data_in_size);
  }

  /* get output buffer */
  const gint expected[96] = { 1101, 1102, 1103, 2101, 2102, 2103, 1104, 1105,
    1106, 2104, 2105, 2106, 1107, 1108, 1109, 2107, 2108, 2109, 1110, 1111,
    1112, 2110, 2111, 2112, 1113, 1114, 1115, 2113, 2114, 2115, 1116, 1117,
    1118, 2116, 2117, 2118, 1119, 1120, 1121, 2119, 2120, 2121, 1122, 1123, 1124,
    2122, 2123, 2124, 1201, 1202, 1203, 2201, 2202, 2203, 1204, 1205, 1206, 2204,
    2205, 2206, 1207, 1208, 1209, 2207, 2208, 2209, 1210, 1211, 1212, 2210, 2211,
    2212, 1213, 1214, 1215, 2213, 2214, 2215, 1216, 1217, 1218, 2216, 2217, 2218,
    1219, 1220, 1221, 2219, 2220, 2221, 1222, 1223, 1224, 2222, 2223, 2224 };

  _aggregator_test_check_output (h, expected, 96);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (no-concat, same in-out frames)
 */
TEST (testTensorAggregator, aggregate5)
{
  GstHarness *h;
  GstTensorsConfig config;
  guint i;
  gsize data_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "concat", (gboolean) FALSE, NULL);

  /* in/out tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    _aggregator_test_push_buffer (h, aggr_test_frames[i], data_size);
    _aggregator_test_check_output (h, aggr_test_frames[i], 48);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), 2U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (concatenate 2 frames with frames-dim 4, out-dimension 3:2:2:2:4)
 */
TEST (testTensorAggregator, aggregate6)
{
  GstHarness *h;
  GstTensorsConfig config;
  guint i;
  gsize data_in_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "frames-out", 2, "frames-dim", 4, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:2:2:2:2", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    _aggregator_test_push_buffer (h, aggr_test_frames[i], data_in_size);
  }

  /* get output buffer */
  const gint expected[96] = { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108,
    1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120,
    1121, 1122, 1123, 1124, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208,
    1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221,
    1222, 1223, 1224, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110,
    2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123,
    2124, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212,
    2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224 };

  _aggregator_test_check_output (h, expected, 96);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (concatenate 2 frames with frames-dim 3, out-dimension 3:2:2:4:2)
 */
TEST (testTensorAggregator, aggregate7)
{
  GstHarness *h;
  GstTensorsConfig config;
  guint i;
  gsize data_in_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "frames-out", 2, "frames-dim", 3, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:2:2:2:2", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    _aggregator_test_push_buffer (h, aggr_test_frames[i], data_in_size);
  }

  /* get output buffer */
  const gint expected[96] = { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108,
    1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120,
    1121, 1122, 1123, 1124, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108,
    2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121,
    2122, 2123, 2124, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210,
    1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223,
    1224, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212,
    2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224 };

  _aggregator_test_check_output (h, expected, 96);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (concatenate 2 frames with frames-dim 2, out-dimension 3:2:4:2:2)
 */
TEST (testTensorAggregator, aggregate8)
{
  GstHarness *h;
  GstTensorsConfig config;
  guint i;
  gsize data_in_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "frames-out", 2, "frames-dim", 2, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:2:2:2:2", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    _aggregator_test_push_buffer (h, aggr_test_frames[i], data_in_size);
  }

  /* get output buffer */
  const gint expected[96] = { 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108,
    1109, 1110, 1111, 1112, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108,
    2109, 2110, 2111, 2112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120,
    1121, 1122, 1123, 1124, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121,
    2122, 2123, 2124, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210,
    1211, 1212, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211,
    2212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224,
    2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224 };

  _aggregator_test_check_output (h, expected, 96);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (concatenate 2 frames with frames-dim 1, out-dimension 3:4:2:2:2)
 */
TEST (testTensorAggregator, aggregate9)
{
  GstHarness *h;
  GstTensorsConfig config;
  guint i;
  gsize data_in_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "frames-out", 2, "frames-dim", 1, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:2:2:2:2", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    _aggregator_test_push_buffer (h, aggr_test_frames[i], data_in_size);
  }

  /* get output buffer */
  const gint expected[96] = { 1101, 1102, 1103, 1104, 1105, 1106, 2101, 2102,
    2103, 2104, 2105, 2106, 1107, 1108, 1109, 1110, 1111, 1112, 2107, 2108,
    2109, 2110, 2111, 2112, 1113, 1114, 1115, 1116, 1117, 1118, 2113, 2114,
    2115, 2116, 2117, 2118, 1119, 1120, 1121, 1122, 1123, 1124, 2119, 2120, 2121,
    2122, 2123, 2124, 1201, 1202, 1203, 1204, 1205, 1206, 2201, 2202, 2203, 2204,
    2205, 2206, 1207, 1208, 1209, 1210, 1211, 1212, 2207, 2208, 2209, 2210, 2211,
    2212, 1213, 1214, 1215, 1216, 1217, 1218, 2213, 2214, 2215, 2216, 2217, 2218,
    1219, 1220, 1221, 1222, 1223, 1224, 2219, 2220, 2221, 2222, 2223, 2224 };

  _aggregator_test_check_output (h, expected, 96);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (concatenate 2 frames with frames-dim 0, out-dimension 6:2:2:2:2)
 */
TEST (testTensorAggregator, aggregate10)
{
  GstHarness *h;
  GstTensorsConfig config;
  guint i;
  gsize data_in_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "frames-out", 2, "frames-dim", 0, NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:2:2:2:2", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_in_size = gst_tensors_info_get_size (&config.info, 0);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    _aggregator_test_push_buffer (h, aggr_test_frames[i], data_in_size);
  }

  /* get output buffer */
  const gint expected[96] = { 1101, 1102, 1103, 2101, 2102, 2103, 1104, 1105,
    1106, 2104, 2105, 2106, 1107, 1108, 1109, 2107, 2108, 2109, 1110, 1111,
    1112, 2110, 2111, 2112, 1113, 1114, 1115, 2113, 2114, 2115, 1116, 1117,
    1118, 2116, 2117, 2118, 1119, 1120, 1121, 2119, 2120, 2121, 1122, 1123, 1124,
    2122, 2123, 2124, 1201, 1202, 1203, 2201, 2202, 2203, 1204, 1205, 1206, 2204,
    2205, 2206, 1207, 1208, 1209, 2207, 2208, 2209, 1210, 1211, 1212, 2210, 2211,
    2212, 1213, 1214, 1215, 2213, 2214, 2215, 1216, 1217, 1218, 2216, 2217, 2218,
    1219, 1220, 1221, 2219, 2220, 2221, 1222, 1223, 1224, 2222, 2223, 2224 };

  _aggregator_test_check_output (h, expected, 96);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (flush old data in aggregator)
 */
TEST (testTensorAggregator, flushData)
{
  const gint test_data[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  const gint frames_out = 6;
  GstHarness *h;
  GstTensorsConfig config;
  guint received;
  gsize data_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "frames-in", 10, "frames-out", frames_out,
      "frames-flush", frames_out, "frames-dim", 0, NULL);

  /* set input tensor info and pad caps */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("10", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_size = gst_tensors_info_get_size (&config.info, 0);

  /* push 1st buffer (4 frames remained in aggregator) */
  _aggregator_test_push_buffer (h, test_data, data_size);
  received = _harness_wait_for_output_buffer (h, 1U);
  EXPECT_EQ (received, 1U);
  _aggregator_test_check_output (h, test_data, frames_out);

  /* flush data */
  gst_element_send_event (h->element, gst_event_new_flush_start ());
  gst_element_send_event (h->element, gst_event_new_flush_stop (TRUE));

  /* push buffer after flushing the data */
  _aggregator_test_push_buffer (h, test_data, data_size);
  received = _harness_wait_for_output_buffer (h, 2U);
  EXPECT_EQ (received, 2U);
  _aggregator_test_check_output (h, test_data, frames_out);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (supposed multi clients using tensor-meta)
 */
TEST (testTensorAggregator, multiClients)
{
  GstHarness *h;
  GstBuffer *input1, *input2, *output;
  GstMetaQuery *meta;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo map;
  guint i, received;
  gsize data_size;
  const gint data1[4] = { 1, 1, 1, 1 };
  const gint data2[4] = { 2, 2, 2, 2 };

  h = gst_harness_new ("tensor_aggregator");

  /* input 4 frames / output 5 frames */
  g_object_set (h->element, "frames-in", 4, "frames-out", 5, "frames-dim", 0, NULL);

  /* set input tensor info and pad caps */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("4", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_size = gst_tensors_info_get_size (&config.info, 0);

  /* create buffers */
  input1 = gst_harness_create_buffer (h, data_size);
  mem = gst_buffer_peek_memory (input1, 0);
  ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));
  memcpy (map.data, data1, data_size);
  gst_memory_unmap (mem, &map);
  meta = gst_buffer_add_meta_query (input1);
  meta->client_id = 0xBADC0FEEU;

  input2 = gst_harness_create_buffer (h, data_size);
  mem = gst_buffer_peek_memory (input2, 0);
  ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));
  memcpy (map.data, data2, data_size);
  gst_memory_unmap (mem, &map);
  meta = gst_buffer_add_meta_query (input2);
  meta->client_id = 0xBADF00DU;

  /* push buffers (1 > 2 > 1) */
  EXPECT_EQ (gst_harness_push (h, gst_buffer_copy_deep (input1)), GST_FLOW_OK);
  g_usleep (10000);
  EXPECT_EQ (gst_harness_push (h, gst_buffer_copy_deep (input2)), GST_FLOW_OK);
  g_usleep (10000);
  EXPECT_EQ (gst_harness_push (h, gst_buffer_copy_deep (input1)), GST_FLOW_OK);
  g_usleep (10000);

  /* total 12 frames (different client-id) are sent, 1 output buffer should be in harness pad. */
  received = _harness_wait_for_output_buffer (h, 1U);
  EXPECT_EQ (received, 1U);

  /* check 1st output buffer. */
  if (received == 1U) {
    output = gst_harness_pull (h);
    meta = gst_buffer_get_meta_query (output);
    EXPECT_TRUE (meta && (meta->client_id == 0xBADC0FEEU));

    mem = gst_buffer_peek_memory (output, 0);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
    ASSERT_TRUE (map.size == sizeof (gint) * 5);

    for (i = 0; i < 5; i++) {
      EXPECT_EQ (((gint *) map.data)[i], 1);
    }

    gst_memory_unmap (mem, &map);
    gst_buffer_unref (output);
  }

  /* push buffer (client2) again, now 2nd output buffer is in harness pad. */
  EXPECT_EQ (gst_harness_push (h, gst_buffer_copy_deep (input2)), GST_FLOW_OK);
  g_usleep (10000);

  received = _harness_wait_for_output_buffer (h, 2U);
  EXPECT_EQ (received, 2U);

  /* check 2nd output buffer. */
  if (received == 2U) {
    output = gst_harness_pull (h);
    meta = gst_buffer_get_meta_query (output);
    EXPECT_TRUE (meta && (meta->client_id == 0xBADF00DU));

    mem = gst_buffer_peek_memory (output, 0);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
    ASSERT_TRUE (map.size == sizeof (gint) * 5);

    for (i = 0; i < 5; i++) {
      EXPECT_EQ (((gint *) map.data)[i], 2);
    }

    gst_memory_unmap (mem, &map);
    gst_buffer_unref (output);
  }

  gst_buffer_unref (input1);
  gst_buffer_unref (input2);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (testTensorConverter, bytesToMulti1)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  GstMemory *mem;
  GstMapInfo info;
  guint i;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "3:4:2:2,3:4:2:2", "input-type",
      "int32,int32", NULL);

  /* in/out caps and tensors info */
  caps = gst_caps_from_string ("application/octet-stream");
  gst_harness_set_src_caps (h, caps);

  gst_tensors_config_init (&config);
  config.rate_n = 0;
  config.rate_d = 1;
  config.info.num_tensors = 2;

  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", config.info.info[0].dimension);
  config.info.info[1].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", config.info.info[1].dimension);

  data_size = gst_tensors_info_get_size (&config.info, -1);

  /* push buffers */
  in_buf = gst_harness_create_buffer (h, data_size);

  mem = gst_buffer_peek_memory (in_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

  memcpy (info.data, aggr_test_frames, data_size);

  gst_memory_unmap (mem, &info);

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  /* get output buffer */
  out_buf = gst_harness_pull (h);

  ASSERT_TRUE (out_buf != NULL);
  ASSERT_EQ (gst_buffer_n_memory (out_buf), 2U);
  ASSERT_EQ (gst_buffer_get_size (out_buf), data_size);

  /* 1st tensor */
  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));
  for (i = 0; i < 48; i++)
    EXPECT_EQ (((gint *) info.data)[i], aggr_test_frames[0][i]);
  gst_memory_unmap (mem, &info);

  /* 2nd tensor */
  mem = gst_buffer_peek_memory (out_buf, 1);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));
  for (i = 0; i < 48; i++)
    EXPECT_EQ (((gint *) info.data)[i], aggr_test_frames[1][i]);
  gst_memory_unmap (mem, &info);

  gst_buffer_unref (out_buf);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (testTensorConverter, bytesToMulti2)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  GstMemory *mem;
  GstMapInfo info;
  guint i;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "3:4:2:1,3:4:2:1,3:4:2:1,3:4:2:1",
      "input-type", "int32,int32,int32,int32", NULL);

  /* in/out caps and tensors info */
  caps = gst_caps_from_string ("application/octet-stream");
  gst_harness_set_src_caps (h, caps);

  gst_tensors_config_init (&config);
  config.rate_n = 0;
  config.rate_d = 1;
  config.info.num_tensors = 4;

  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:1", config.info.info[0].dimension);
  config.info.info[1].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:1", config.info.info[1].dimension);
  config.info.info[2].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:1", config.info.info[2].dimension);
  config.info.info[3].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:1", config.info.info[3].dimension);

  data_size = gst_tensors_info_get_size (&config.info, -1);

  /* push buffers */
  in_buf = gst_harness_create_buffer (h, data_size);

  mem = gst_buffer_peek_memory (in_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

  memcpy (info.data, aggr_test_frames, data_size);

  gst_memory_unmap (mem, &info);

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  /* get output buffer */
  out_buf = gst_harness_pull (h);

  ASSERT_TRUE (out_buf != NULL);
  ASSERT_EQ (gst_buffer_n_memory (out_buf), 4U);
  ASSERT_EQ (gst_buffer_get_size (out_buf), data_size);

  /* 1st tensor */
  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));
  for (i = 0; i < 24; i++)
    EXPECT_EQ (((gint *) info.data)[i], aggr_test_frames[0][i]);
  gst_memory_unmap (mem, &info);

  /* 2nd tensor */
  mem = gst_buffer_peek_memory (out_buf, 1);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));
  for (i = 24; i < 48; i++)
    EXPECT_EQ (((gint *) info.data)[i - 24], aggr_test_frames[0][i]);
  gst_memory_unmap (mem, &info);

  /* 3rd tensor */
  mem = gst_buffer_peek_memory (out_buf, 2);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));
  for (i = 0; i < 24; i++)
    EXPECT_EQ (((gint *) info.data)[i], aggr_test_frames[1][i]);
  gst_memory_unmap (mem, &info);

  /* 4th tensor */
  mem = gst_buffer_peek_memory (out_buf, 3);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));
  for (i = 24; i < 48; i++)
    EXPECT_EQ (((gint *) info.data)[i - 24], aggr_test_frames[1][i]);
  gst_memory_unmap (mem, &info);

  gst_buffer_unref (out_buf);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (testTensorConverter, bytesToMultiInvalidDim01_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "2:2:2:2", "input-type", "int32,uint64", NULL);

  /* in/out caps and tensors info */
  caps = gst_caps_from_string ("application/octet-stream");
  gst_harness_set_src_caps (h, caps);

  gst_tensors_config_init (&config);
  config.rate_n = 0;
  config.rate_d = 1;
  config.info.num_tensors = 2;

  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("2:2:2:2", config.info.info[0].dimension);
  config.info.info[1].type = _NNS_UINT64;
  gst_tensor_parse_dimension ("2:2:2:2", config.info.info[1].dimension);

  data_size = gst_tensors_info_get_size (&config.info, -1);

  /* push buffers */
  in_buf = gst_harness_create_buffer (h, data_size);
  EXPECT_EQ (GST_FLOW_NOT_NEGOTIATED, gst_harness_push (h, in_buf));

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (testTensorConverter, bytesToMultiInvalidDim02_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "2:2:2:2,2:0:1", "input-type",
      "int32,float32", NULL);

  /* in/out caps and tensors info */
  caps = gst_caps_from_string ("application/octet-stream");
  gst_harness_set_src_caps (h, caps);

  gst_tensors_config_init (&config);
  config.rate_n = 0;
  config.rate_d = 1;
  config.info.num_tensors = 2;

  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("2:2:2:2", config.info.info[0].dimension);
  config.info.info[1].type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("2:2:1:1", config.info.info[1].dimension);

  data_size = gst_tensors_info_get_size (&config.info, -1);

  /* push buffers */
  in_buf = gst_harness_create_buffer (h, data_size);
  EXPECT_EQ (GST_FLOW_NOT_NEGOTIATED, gst_harness_push (h, in_buf));

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (testTensorConverter, bytesToMultiInvalidType01_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-type", "int64", "input-dim", "2:2:2:2,2:2:2:2", NULL);

  /* in/out caps and tensors info */
  caps = gst_caps_from_string ("application/octet-stream");
  gst_harness_set_src_caps (h, caps);

  gst_tensors_config_init (&config);
  config.rate_n = 0;
  config.rate_d = 1;
  config.info.num_tensors = 2;

  config.info.info[0].type = _NNS_INT64;
  gst_tensor_parse_dimension ("2:2:2:2", config.info.info[0].dimension);
  config.info.info[1].type = _NNS_UINT64;
  gst_tensor_parse_dimension ("2:2:2:2", config.info.info[1].dimension);

  data_size = gst_tensors_info_get_size (&config.info, -1);

  /* push buffers */
  in_buf = gst_harness_create_buffer (h, data_size);
  EXPECT_EQ (GST_FLOW_NOT_NEGOTIATED, gst_harness_push (h, in_buf));

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (testTensorConverter, bytesToMultiInvalidType02_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "2:2:2:2,2:2:1:1", "input-type",
      "int16,invalid", NULL);

  /* in/out caps and tensors info */
  caps = gst_caps_from_string ("application/octet-stream");
  gst_harness_set_src_caps (h, caps);

  gst_tensors_config_init (&config);
  config.rate_n = 0;
  config.rate_d = 1;
  config.info.num_tensors = 2;

  config.info.info[0].type = _NNS_INT16;
  gst_tensor_parse_dimension ("2:2:2:2", config.info.info[0].dimension);
  config.info.info[1].type = _NNS_INT16;
  gst_tensor_parse_dimension ("2:2:1:1", config.info.info[1].dimension);

  data_size = gst_tensors_info_get_size (&config.info, -1);

  /* push buffers */
  in_buf = gst_harness_create_buffer (h, data_size);
  EXPECT_EQ (GST_FLOW_NOT_NEGOTIATED, gst_harness_push (h, in_buf));

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (testTensorConverter, bytesToMultiInvalidType03_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "1:1:1:1,2:1:1:1,3", "input-type",
      "int16,uint16", NULL);

  /* in/out caps and tensors info */
  caps = gst_caps_from_string ("application/octet-stream");
  gst_harness_set_src_caps (h, caps);

  gst_tensors_config_init (&config);
  config.rate_n = 0;
  config.rate_d = 1;
  config.info.num_tensors = 3;

  config.info.info[0].type = _NNS_INT16;
  gst_tensor_parse_dimension ("1:1:1:1", config.info.info[0].dimension);
  config.info.info[1].type = _NNS_UINT16;
  gst_tensor_parse_dimension ("2:1:1:1", config.info.info[1].dimension);
  config.info.info[1].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:1:1:1", config.info.info[1].dimension);

  data_size = gst_tensors_info_get_size (&config.info, -1);

  /* push buffers */
  in_buf = gst_harness_create_buffer (h, data_size);
  EXPECT_DEATH (gst_harness_push (h, in_buf), "");

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_buffer_unref (in_buf);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (testTensorConverter, bytesToMultiInvalidSize_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "2:2:2:2,1:1:1:1", "input-type",
      "float32,float64", NULL);

  /* in/out caps and tensors info */
  caps = gst_caps_from_string ("application/octet-stream");
  gst_harness_set_src_caps (h, caps);

  gst_tensors_config_init (&config);
  config.rate_n = 0;
  config.rate_d = 1;
  config.info.num_tensors = 2;

  config.info.info[0].type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("2:2:2:2", config.info.info[0].dimension);
  config.info.info[1].type = _NNS_FLOAT64;
  gst_tensor_parse_dimension ("2:2:1:1", config.info.info[1].dimension);

  data_size = gst_tensors_info_get_size (&config.info, -1);

  /* push buffers */
  in_buf = gst_harness_create_buffer (h, data_size);
  EXPECT_DEATH (gst_harness_push (h, in_buf), "");

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_buffer_unref (in_buf);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (testTensorConverter, bytesToMultiInvalidFrames_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "2:2:2:2,1:1:1:1", "input-type",
      "float32,float64", "frames-per-tensor", "2", NULL);

  /* in/out caps and tensors info */
  caps = gst_caps_from_string ("application/octet-stream");
  gst_harness_set_src_caps (h, caps);

  gst_tensors_config_init (&config);
  config.rate_n = 0;
  config.rate_d = 1;
  config.info.num_tensors = 2;

  config.info.info[0].type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("2:2:2:2", config.info.info[0].dimension);
  config.info.info[1].type = _NNS_FLOAT64;
  gst_tensor_parse_dimension ("1:1:1:1", config.info.info[1].dimension);

  data_size = gst_tensors_info_get_size (&config.info, -1);

  /* push buffers */
  in_buf = gst_harness_create_buffer (h, data_size);
  EXPECT_EQ (GST_FLOW_NOT_NEGOTIATED, gst_harness_push (h, in_buf));

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes (multi memories) to static tensor, no properties)
 */
TEST (testTensorConverter, bytesToStatic)
{
  GstHarness *h;
  GstCaps *caps;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo map;
  gint *input;
  guint i, received;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  /* in/out caps */
  gst_tensors_config_init (&config);
  config.rate_n = 0;
  config.rate_d = 1;
  config.info.num_tensors = 1;

  config.info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", config.info.info[0].dimension);

  caps = gst_tensors_caps_from_config (&config);
  gst_harness_set_sink_caps (h, caps);

  caps = gst_caps_from_string ("application/octet-stream");
  gst_harness_set_src_caps (h, caps);

  data_size = gst_tensors_info_get_size (&config.info, -1);

  /* 1st memory */
  in_buf = gst_harness_create_buffer (h, data_size / 2);
  ASSERT_TRUE (gst_buffer_map (in_buf, &map, GST_MAP_WRITE));
  input = (gint *) map.data;
  for (i = 0; i < 24; i++)
    input[i] = aggr_test_frames[0][i];
  gst_buffer_unmap (in_buf, &map);

  /* 2nd memory */
  input = (gint *) g_malloc0 (data_size / 2);
  for (i = 0; i < 24; i++)
    input[i] = aggr_test_frames[0][i + 24];
  mem = gst_memory_new_wrapped (
      (GstMemoryFlags) 0, input, data_size / 2, 0, data_size / 2, input, g_free);
  gst_buffer_append_memory (in_buf, mem);

  /* push buffer and compare result */
  EXPECT_EQ (GST_FLOW_OK, gst_harness_push (h, in_buf));
  received = _harness_wait_for_output_buffer (h, i);
  EXPECT_EQ (received, 1U);

  out_buf = gst_harness_pull (h);
  EXPECT_EQ (gst_buffer_n_memory (out_buf), 1U);
  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));
  for (i = 0; i < 48; i++)
    EXPECT_EQ (((gint *) map.data)[i], aggr_test_frames[0][i]);
  gst_memory_unmap (mem, &map);

  gst_buffer_unref (out_buf);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to flex tensor)
 */
TEST (testTensorConverter, bytesToFlex)
{
  GstHarness *h;
  GstCaps *caps;
  GstBuffer *in_buf, *out_buf;
  GstMemory *mem;
  GstTensorMetaInfo meta;
  guint i, received;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  /* in/out caps */
  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  gst_harness_set_sink_caps (h, caps);

  caps = gst_caps_from_string ("application/octet-stream");
  gst_harness_set_src_caps (h, caps);

  /* push buffers */
  for (i = 1; i <= 3; i++) {
    data_size = i * 10U;

    in_buf = gst_harness_create_buffer (h, data_size);
    EXPECT_EQ (GST_FLOW_OK, gst_harness_push (h, in_buf));

    received = _harness_wait_for_output_buffer (h, i);
    EXPECT_EQ (received, i);

    out_buf = gst_harness_pull (h);
    EXPECT_EQ (gst_buffer_n_memory (out_buf), 1U);
    mem = gst_buffer_peek_memory (out_buf, 0);
    gst_tensor_meta_info_parse_memory (&meta, mem);

    EXPECT_EQ (meta.type, _NNS_UINT8);
    EXPECT_EQ (meta.dimension[0], data_size);
    EXPECT_LE (meta.dimension[1], 1U);
    EXPECT_EQ ((media_type) meta.media_type, _NNS_OCTET);

    data_size = gst_tensor_meta_info_get_header_size (&meta);
    data_size += gst_tensor_meta_info_get_data_size (&meta);

    EXPECT_EQ (gst_buffer_get_size (out_buf), data_size);
    gst_buffer_unref (out_buf);
  }

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to flex tensor with invalid condition)
 */
TEST (testTensorConverter, bytesToFlexInvalidFrames_n)
{
  GstHarness *h;
  GstCaps *caps;
  GstBuffer *in_buf;

  h = gst_harness_new ("tensor_converter");

  /* cannot configure multi tensors if output is flex tensor. */
  g_object_set (h->element, "frames-per-tensor", "2", NULL);

  /* in/out caps */
  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  gst_harness_set_sink_caps (h, caps);

  caps = gst_caps_from_string ("application/octet-stream");
  gst_harness_set_src_caps (h, caps);

  /* push buffers */
  in_buf = gst_harness_create_buffer (h, 100U);
  EXPECT_EQ (GST_FLOW_NOT_NEGOTIATED, gst_harness_push (h, in_buf));

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (flexible to static tensor)
 */
TEST (testTensorConverter, flexToStaticTensor)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstMemory *mem;
  GstMapInfo map;
  GstTensorMetaInfo meta;
  GstTensorsInfo info;
  GstCaps *caps;
  guint8 *data;
  guint i, j, received, *value;
  gsize data_size, hsize;

  h = gst_harness_new ("tensor_converter");

  /* in/out caps and tensors info */
  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  gst_harness_set_src_caps (h, caps);

  gst_tensors_info_init (&info);
  info.num_tensors = 2;

  info.info[0].type = _NNS_UINT32;
  gst_tensor_parse_dimension ("10:1:1:1", info.info[0].dimension);
  info.info[1].type = _NNS_UINT32;
  gst_tensor_parse_dimension ("20:1:1:1", info.info[1].dimension);

  /* input buffer */
  in_buf = gst_buffer_new ();

  /* 1st mem block */
  gst_tensor_info_convert_to_meta (&info.info[0], &meta);
  hsize = gst_tensor_meta_info_get_header_size (&meta);
  data_size = hsize + gst_tensor_meta_info_get_data_size (&meta);

  data = (guint8 *) g_malloc0 (data_size);
  gst_tensor_meta_info_update_header (&meta, data);
  value = (guint *) (data + hsize);
  for (i = 0; i < 10; i++)
    value[i] = i * 10;

  mem = gst_memory_new_wrapped (
      (GstMemoryFlags) 0, data, data_size, 0, data_size, data, g_free);
  gst_buffer_append_memory (in_buf, mem);

  /* 2nd mem block */
  gst_tensor_info_convert_to_meta (&info.info[1], &meta);
  hsize = gst_tensor_meta_info_get_header_size (&meta);
  data_size = hsize + gst_tensor_meta_info_get_data_size (&meta);

  data = (guint8 *) g_malloc0 (data_size);
  gst_tensor_meta_info_update_header (&meta, data);
  value = (guint *) (data + hsize);
  for (i = 0; i < 20; i++)
    value[i] = i * 20;

  mem = gst_memory_new_wrapped (
      (GstMemoryFlags) 0, data, data_size, 0, data_size, data, g_free);
  gst_buffer_append_memory (in_buf, mem);

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  /* wait for output buffer */
  received = _harness_wait_for_output_buffer (h, 1U);
  ASSERT_EQ (received, 1U);

  /* get output buffer */
  out_buf = gst_harness_pull (h);
  EXPECT_EQ (gst_buffer_n_memory (out_buf), 2U);

  for (i = 0; i < gst_buffer_n_memory (out_buf); i++) {
    mem = gst_buffer_peek_memory (out_buf, i);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
    EXPECT_EQ (map.size, gst_tensors_info_get_size (&info, i));
    value = (guint *) map.data;

    for (j = 0; j < (i + 1) * 10; j++)
      EXPECT_EQ (value[j], j * (i + 1) * 10);

    gst_memory_unmap (mem, &map);
  }

  gst_buffer_unref (out_buf);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (flexible to static tensor)
 */
TEST (testTensorConverter, flexToStaticInvalidBuffer1_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstMemory *mem;
  GstTensorMetaInfo meta;
  GstTensorInfo info;
  GstCaps *caps;
  gpointer data;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "3:4:2:2,3:4:2:2", "input-type",
      "int32,int32", NULL);

  /* in/out caps and tensors info */
  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  gst_harness_set_src_caps (h, caps);

  gst_tensor_info_init (&info);
  info.type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", info.dimension);

  /* push buffer (invalid number) */
  in_buf = gst_buffer_new ();

  gst_tensor_info_convert_to_meta (&info, &meta);
  data_size = gst_tensor_meta_info_get_header_size (&meta);
  data_size += gst_tensor_meta_info_get_data_size (&meta);

  data = g_malloc0 (data_size);
  gst_tensor_meta_info_update_header (&meta, data);

  mem = gst_memory_new_wrapped (
      (GstMemoryFlags) 0, data, data_size, 0, data_size, data, g_free);
  gst_buffer_append_memory (in_buf, mem);

  EXPECT_NE (gst_harness_push (h, in_buf), GST_FLOW_OK);

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (flexible to static tensor)
 */
TEST (testTensorConverter, flexToStaticInvalidBuffer2_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstMemory *mem;
  GstTensorMetaInfo meta;
  GstTensorsInfo info;
  GstCaps *caps;
  gpointer data;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "3:4:2:2,3:4:2:2", "input-type",
      "int32,int32", NULL);

  /* in/out caps and tensors info */
  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  gst_harness_set_src_caps (h, caps);

  gst_tensors_info_init (&info);
  info.num_tensors = 2;

  info.info[0].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", info.info[0].dimension);
  info.info[1].type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", info.info[1].dimension);

  /* push buffer (invalid buffer size) */
  in_buf = gst_buffer_new ();

  /* 1st mem block */
  gst_tensor_info_convert_to_meta (&info.info[0], &meta);
  data_size = gst_tensor_meta_info_get_header_size (&meta);
  data_size += gst_tensor_meta_info_get_data_size (&meta);

  data = g_malloc0 (data_size);
  gst_tensor_meta_info_update_header (&meta, data);

  mem = gst_memory_new_wrapped (
      (GstMemoryFlags) 0, data, data_size, 0, data_size, data, g_free);
  gst_buffer_append_memory (in_buf, mem);

  /* 2nd mem block (invalid size) */
  gst_tensor_info_convert_to_meta (&info.info[1], &meta);
  data_size = gst_tensor_meta_info_get_header_size (&meta);
  data_size += gst_tensor_meta_info_get_data_size (&meta) / 2;

  data = g_malloc0 (data_size);
  gst_tensor_meta_info_update_header (&meta, data);

  mem = gst_memory_new_wrapped (
      (GstMemoryFlags) 0, data, data_size, 0, data_size, data, g_free);
  gst_buffer_append_memory (in_buf, mem);

  EXPECT_NE (gst_harness_push (h, in_buf), GST_FLOW_OK);

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

#ifdef HAVE_ORC
#include "nnstreamer-orc.h"

/**
 * @brief Test for tensor_transform orc functions (add constant value)
 */
TEST (testTensorTransform, orcAdd)
{
  const guint array_size = 10;
  guint i;

  /* add constant s8 */
  int8_t data_s8[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_s8[i] = (gint) i - 1;
  }

  nns_orc_add_c_s8 (data_s8, -20, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s8[i], (gint) i - 1 - 20);
  }

  for (i = 0; i < array_size; i++) {
    data_s8[i] = (gint) i + 1;
  }

  nns_orc_add_c_s8 (data_s8, 20, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s8[i], (gint) i + 1 + 20);
  }

  /* add constant u8 */
  uint8_t data_u8[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_u8[i] = i + 1;
  }

  nns_orc_add_c_u8 (data_u8, 3, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_u8[i], i + 1 + 3);
  }

  /* add constant s16 */
  int16_t data_s16[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_s16[i] = (gint) i - 1;
  }

  nns_orc_add_c_s16 (data_s16, -16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s16[i], (gint) i - 1 - 16);
  }

  for (i = 0; i < array_size; i++) {
    data_s16[i] = (gint) i + 1;
  }

  nns_orc_add_c_s16 (data_s16, 16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s16[i], (gint) i + 1 + 16);
  }

  /* add constant u16 */
  uint16_t data_u16[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_u16[i] = i + 1;
  }

  nns_orc_add_c_u16 (data_u16, 17, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_u16[i], i + 1 + 17);
  }

  /* add constant s32 */
  int32_t data_s32[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_s32[i] = (gint) i + 1;
  }

  nns_orc_add_c_s32 (data_s32, -32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s32[i], (gint) i + 1 - 32);
  }

  for (i = 0; i < array_size; i++) {
    data_s32[i] = (gint) i + 1;
  }

  nns_orc_add_c_s32 (data_s32, 32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s32[i], (gint) i + 1 + 32);
  }

  /* add constant u32 */
  uint32_t data_u32[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_u32[i] = i + 1;
  }

  nns_orc_add_c_u32 (data_u32, 33, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_u32[i], i + 1 + 33);
  }

  /* add constant s64 */
  int64_t data_s64[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_s64[i] = (gint) i + 1;
  }

  nns_orc_add_c_s64 (data_s64, -61, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s64[i], (gint) i + 1 - 61);
  }

  for (i = 0; i < array_size; i++) {
    data_s64[i] = (gint) i + 1;
  }

  nns_orc_add_c_s64 (data_s64, 61, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s64[i], (gint) i + 1 + 61);
  }

  /* add constant u64 */
  uint64_t data_u64[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_u64[i] = i + 1;
  }

  nns_orc_add_c_u64 (data_u64, 62, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_u64[i], i + 1 + 62);
  }

  /* add constant f32 */
  float data_f32[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_f32[i] = i - .1;
  }

  nns_orc_add_c_f32 (data_f32, -10.2, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (data_f32[i], i - .1 - 10.2);
  }

  for (i = 0; i < array_size; i++) {
    data_f32[i] = i + .1;
  }

  nns_orc_add_c_f32 (data_f32, 10.2, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (data_f32[i], i + .1 + 10.2);
  }

  /* add constant f64 */
  double data_f64[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_f64[i] = i - .1;
  }

  nns_orc_add_c_f64 (data_f64, -20.5, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (data_f64[i], i - .1 - 20.5);
  }

  for (i = 0; i < array_size; i++) {
    data_f64[i] = i + .2;
  }

  nns_orc_add_c_f64 (data_f64, 20.5, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (data_f64[i], i + .2 + 20.5);
  }
}

/**
 * @brief Test for tensor_transform orc functions (mul constant value)
 */
TEST (testTensorTransform, orcMul)
{
  const guint array_size = 10;
  guint i;

  /* mul constant s8 */
  int8_t data_s8[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_s8[i] = (gint) i + 1;
  }

  nns_orc_mul_c_s8 (data_s8, -3, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s8[i], (gint) (i + 1) * (-3));
  }

  for (i = 0; i < array_size; i++) {
    data_s8[i] = (gint) i + 1;
  }

  nns_orc_mul_c_s8 (data_s8, 5, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s8[i], (gint) (i + 1) * 5);
  }

  /* mul constant u8 */
  uint8_t data_u8[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_u8[i] = i + 1;
  }

  nns_orc_mul_c_u8 (data_u8, 3, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_u8[i], (i + 1) * 3);
  }

  /* mul constant s16 */
  int16_t data_s16[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_s16[i] = (gint) i + 1;
  }

  nns_orc_mul_c_s16 (data_s16, -16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s16[i], (gint) (i + 1) * (-16));
  }

  for (i = 0; i < array_size; i++) {
    data_s16[i] = (gint) i + 1;
  }

  nns_orc_mul_c_s16 (data_s16, 16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s16[i], (gint) (i + 1) * 16);
  }

  /* mul constant u16 */
  uint16_t data_u16[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_u16[i] = i + 1;
  }

  nns_orc_mul_c_u16 (data_u16, 17, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_u16[i], (i + 1) * 17);
  }

  /* mul constant s32 */
  int32_t data_s32[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_s32[i] = (gint) i + 1;
  }

  nns_orc_mul_c_s32 (data_s32, -32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s32[i], (gint) (i + 1) * (-32));
  }

  for (i = 0; i < array_size; i++) {
    data_s32[i] = (gint) i + 1;
  }

  nns_orc_mul_c_s32 (data_s32, 32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s32[i], (gint) (i + 1) * 32);
  }

  /* mul constant u32 */
  uint32_t data_u32[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_u32[i] = i + 1;
  }

  nns_orc_mul_c_u32 (data_u32, 33, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_u32[i], (i + 1) * 33);
  }

  /* mul constant s64 */
  int64_t data_s64[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_s64[i] = (gint) i + 1;
  }

  nns_orc_mul_c_s64 (data_s64, -61, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s64[i], (gint) (i + 1) * (-61));
  }

  for (i = 0; i < array_size; i++) {
    data_s64[i] = (gint) i + 1;
  }

  nns_orc_mul_c_s64 (data_s64, 61, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_s64[i], (gint) (i + 1) * 61);
  }

  /* mul constant u64 */
  uint64_t data_u64[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_u64[i] = i + 1;
  }

  nns_orc_mul_c_u64 (data_u64, 62, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_u64[i], (i + 1) * 62);
  }

  /* mul constant f32 */
  float data_f32[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_f32[i] = i + 1 - .1;
  }

  nns_orc_mul_c_f32 (data_f32, -10.2, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (data_f32[i], (i + 1 - .1) * (-10.2));
  }

  for (i = 0; i < array_size; i++) {
    data_f32[i] = i + .1;
  }

  nns_orc_mul_c_f32 (data_f32, 10.2, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (data_f32[i], (i + .1) * 10.2);
  }

  /* mul constant f64 */
  double data_f64[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_f64[i] = i + 1 - .1;
  }

  nns_orc_mul_c_f64 (data_f64, -20.5, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (data_f64[i], (i + 1 - .1) * (-20.5));
  }

  for (i = 0; i < array_size; i++) {
    data_f64[i] = i + .2;
  }

  nns_orc_mul_c_f64 (data_f64, 20.5, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (data_f64[i], (i + .2) * 20.5);
  }
}

/**
 * @brief Test for tensor_transform orc functions (div constant value)
 */
TEST (testTensorTransform, orcDiv)
{
  const guint array_size = 10;
  guint i;

  /* div constant f32 */
  float data_f32[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_f32[i] = i + 1 - .1;
  }

  nns_orc_div_c_f32 (data_f32, -2.2, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (data_f32[i], (i + 1 - .1) / (-2.2));
  }

  for (i = 0; i < array_size; i++) {
    data_f32[i] = i + 10.1;
  }

  nns_orc_div_c_f32 (data_f32, 10.2, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (data_f32[i], (i + 10.1) / 10.2);
  }

  /* div constant f64 */
  double data_f64[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_f64[i] = i + 1 - .1;
  }

  nns_orc_div_c_f64 (data_f64, -10.5, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (data_f64[i], (i + 1 - .1) / (-10.5));
  }

  for (i = 0; i < array_size; i++) {
    data_f64[i] = i + .2;
  }

  nns_orc_div_c_f64 (data_f64, 5.5, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (data_f64[i], (i + .2) / 5.5);
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert s8 to other type)
 */
TEST (testTensorTransform, orcConvS8)
{
  const guint array_size = 10;
  guint i;

  int8_t data_s8[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_s8[i] = ((int8_t) (i + 1)) * -1;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = {
    0,
  };

  nns_orc_conv_s8_to_s8 (res_s8, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_s8[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = {
    0,
  };

  nns_orc_conv_s8_to_u8 (res_u8, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u8[i], (uint8_t) data_s8[i]);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = {
    0,
  };

  nns_orc_conv_s8_to_s16 (res_s16, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_s8[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = {
    0,
  };

  nns_orc_conv_s8_to_u16 (res_u16, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u16[i], (uint16_t) data_s8[i]);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = {
    0,
  };

  nns_orc_conv_s8_to_s32 (res_s32, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_s8[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = {
    0,
  };

  nns_orc_conv_s8_to_u32 (res_u32, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u32[i], (uint32_t) data_s8[i]);
  }

  /* convert s64 */
  int64_t res_s64[array_size] = {
    0,
  };

  nns_orc_conv_s8_to_s64 (res_s64, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_s8[i]);
  }

  /* convert u64 */
  uint64_t res_u64[array_size] = {
    0,
  };

  nns_orc_conv_s8_to_u64 (res_u64, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u64[i], (uint64_t) data_s8[i]);
  }

  /* convert f32 */
  float res_f32[array_size] = {
    0,
  };

  nns_orc_conv_s8_to_f32 (res_f32, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_s8[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = {
    0,
  };

  nns_orc_conv_s8_to_f64 (res_f64, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_s8[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert u8 to other type)
 */
TEST (testTensorTransform, orcConvU8)
{
  const guint array_size = 10;
  guint i;

  uint8_t data_u8[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_u8[i] = G_MAXUINT8 - i;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = {
    0,
  };

  nns_orc_conv_u8_to_s8 (res_s8, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_u8[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = {
    0,
  };

  nns_orc_conv_u8_to_u8 (res_u8, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u8[i], (uint8_t) data_u8[i]);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = {
    0,
  };

  nns_orc_conv_u8_to_s16 (res_s16, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_u8[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = {
    0,
  };

  nns_orc_conv_u8_to_u16 (res_u16, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u16[i], (uint16_t) data_u8[i]);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = {
    0,
  };

  nns_orc_conv_u8_to_s32 (res_s32, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_u8[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = {
    0,
  };

  nns_orc_conv_u8_to_u32 (res_u32, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u32[i], (uint32_t) data_u8[i]);
  }

  /* convert s64 */
  int64_t res_s64[array_size] = {
    0,
  };

  nns_orc_conv_u8_to_s64 (res_s64, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_u8[i]);
  }

  /* convert u64 */
  uint64_t res_u64[array_size] = {
    0,
  };

  nns_orc_conv_u8_to_u64 (res_u64, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u64[i], (uint64_t) data_u8[i]);
  }

  /* convert f32 */
  float res_f32[array_size] = {
    0,
  };

  nns_orc_conv_u8_to_f32 (res_f32, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_u8[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = {
    0,
  };

  nns_orc_conv_u8_to_f64 (res_f64, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_u8[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert s16 to other type)
 */
TEST (testTensorTransform, orcConvS16)
{
  const guint array_size = 10;
  guint i;

  int16_t data_s16[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_s16[i] = ((int16_t) (i + 1)) * -1;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = {
    0,
  };

  nns_orc_conv_s16_to_s8 (res_s8, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_s16[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = {
    0,
  };

  nns_orc_conv_s16_to_u8 (res_u8, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u8[i], (uint8_t) data_s16[i]);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = {
    0,
  };

  nns_orc_conv_s16_to_s16 (res_s16, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_s16[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = {
    0,
  };

  nns_orc_conv_s16_to_u16 (res_u16, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u16[i], (uint16_t) data_s16[i]);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = {
    0,
  };

  nns_orc_conv_s16_to_s32 (res_s32, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_s16[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = {
    0,
  };

  nns_orc_conv_s16_to_u32 (res_u32, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u32[i], (uint32_t) data_s16[i]);
  }

  /* convert s64 */
  int64_t res_s64[array_size] = {
    0,
  };

  nns_orc_conv_s16_to_s64 (res_s64, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_s16[i]);
  }

  /* convert u64 */
  uint64_t res_u64[array_size] = {
    0,
  };

  nns_orc_conv_s16_to_u64 (res_u64, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u64[i], (uint64_t) data_s16[i]);
  }

  /* convert f32 */
  float res_f32[array_size] = {
    0,
  };

  nns_orc_conv_s16_to_f32 (res_f32, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_s16[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = {
    0,
  };

  nns_orc_conv_s16_to_f64 (res_f64, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_s16[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert u16 to other type)
 */
TEST (testTensorTransform, orcConvU16)
{
  const guint array_size = 10;
  guint i;

  uint16_t data_u16[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_u16[i] = G_MAXUINT16 - i;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = {
    0,
  };

  nns_orc_conv_u16_to_s8 (res_s8, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_u16[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = {
    0,
  };

  nns_orc_conv_u16_to_u8 (res_u8, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u8[i], (uint8_t) data_u16[i]);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = {
    0,
  };

  nns_orc_conv_u16_to_s16 (res_s16, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_u16[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = {
    0,
  };

  nns_orc_conv_u16_to_u16 (res_u16, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u16[i], (uint16_t) data_u16[i]);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = {
    0,
  };

  nns_orc_conv_u16_to_s32 (res_s32, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_u16[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = {
    0,
  };

  nns_orc_conv_u16_to_u32 (res_u32, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u32[i], (uint32_t) data_u16[i]);
  }

  /* convert s64 */
  int64_t res_s64[array_size] = {
    0,
  };

  nns_orc_conv_u16_to_s64 (res_s64, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_u16[i]);
  }

  /* convert u64 */
  uint64_t res_u64[array_size] = {
    0,
  };

  nns_orc_conv_u16_to_u64 (res_u64, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u64[i], (uint64_t) data_u16[i]);
  }

  /* convert f32 */
  float res_f32[array_size] = {
    0,
  };

  nns_orc_conv_u16_to_f32 (res_f32, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_u16[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = {
    0,
  };

  nns_orc_conv_u16_to_f64 (res_f64, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_u16[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert s32 to other type)
 */
TEST (testTensorTransform, orcConvS32)
{
  const guint array_size = 10;
  guint i;

  int32_t data_s32[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_s32[i] = ((int32_t) (i + 1)) * -1;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = {
    0,
  };

  nns_orc_conv_s32_to_s8 (res_s8, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_s32[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = {
    0,
  };

  nns_orc_conv_s32_to_u8 (res_u8, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u8[i], (uint8_t) data_s32[i]);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = {
    0,
  };

  nns_orc_conv_s32_to_s16 (res_s16, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_s32[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = {
    0,
  };

  nns_orc_conv_s32_to_u16 (res_u16, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u16[i], (uint16_t) data_s32[i]);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = {
    0,
  };

  nns_orc_conv_s32_to_s32 (res_s32, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_s32[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = {
    0,
  };

  nns_orc_conv_s32_to_u32 (res_u32, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u32[i], (uint32_t) data_s32[i]);
  }

  /* convert s64 */
  int64_t res_s64[array_size] = {
    0,
  };

  nns_orc_conv_s32_to_s64 (res_s64, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_s32[i]);
  }

  /* convert u64 */
  uint64_t res_u64[array_size] = {
    0,
  };

  nns_orc_conv_s32_to_u64 (res_u64, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u64[i], (uint64_t) data_s32[i]);
  }

  /* convert f32 */
  float res_f32[array_size] = {
    0,
  };

  nns_orc_conv_s32_to_f32 (res_f32, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_s32[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = {
    0,
  };

  nns_orc_conv_s32_to_f64 (res_f64, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_s32[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert u32 to other type)
 */
TEST (testTensorTransform, orcConvU32)
{
  const guint array_size = 10;
  guint i;

  uint32_t data_u32[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_u32[i] = G_MAXUINT32 - i;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = {
    0,
  };

  nns_orc_conv_u32_to_s8 (res_s8, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_u32[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = {
    0,
  };

  nns_orc_conv_u32_to_u8 (res_u8, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u8[i], (uint8_t) data_u32[i]);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = {
    0,
  };

  nns_orc_conv_u32_to_s16 (res_s16, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_u32[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = {
    0,
  };

  nns_orc_conv_u32_to_u16 (res_u16, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u16[i], (uint16_t) data_u32[i]);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = {
    0,
  };

  nns_orc_conv_u32_to_s32 (res_s32, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_u32[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = {
    0,
  };

  nns_orc_conv_u32_to_u32 (res_u32, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u32[i], (uint32_t) data_u32[i]);
  }

  /* convert s64 */
  int64_t res_s64[array_size] = {
    0,
  };

  nns_orc_conv_u32_to_s64 (res_s64, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_u32[i]);
  }

  /* convert u64 */
  uint64_t res_u64[array_size] = {
    0,
  };

  nns_orc_conv_u32_to_u64 (res_u64, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u64[i], (uint64_t) data_u32[i]);
  }

  /* convert f32 */
  float res_f32[array_size] = {
    0,
  };

  nns_orc_conv_u32_to_f32 (res_f32, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) ((int32_t) data_u32[i]));
  }

  /* convert f64 */
  double res_f64[array_size] = {
    0,
  };

  nns_orc_conv_u32_to_f64 (res_f64, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) ((int32_t) data_u32[i]));
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert s64 to other type)
 */
TEST (testTensorTransform, orcConvS64)
{
  const guint array_size = 10;
  guint i;

  int64_t data_s64[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_s64[i] = ((int64_t) (i + 1)) * -1;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = {
    0,
  };

  nns_orc_conv_s64_to_s8 (res_s8, data_s64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_s64[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = {
    0,
  };

  nns_orc_conv_s64_to_u8 (res_u8, data_s64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u8[i], (uint8_t) data_s64[i]);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = {
    0,
  };

  nns_orc_conv_s64_to_s16 (res_s16, data_s64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_s64[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = {
    0,
  };

  nns_orc_conv_s64_to_u16 (res_u16, data_s64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u16[i], (uint16_t) data_s64[i]);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = {
    0,
  };

  nns_orc_conv_s64_to_s32 (res_s32, data_s64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_s64[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = {
    0,
  };

  nns_orc_conv_s64_to_u32 (res_u32, data_s64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u32[i], (uint32_t) data_s64[i]);
  }

  /* convert s64 */
  int64_t res_s64[array_size] = {
    0,
  };

  nns_orc_conv_s64_to_s64 (res_s64, data_s64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_s64[i]);
  }

  /* convert u64 */
  uint64_t res_u64[array_size] = {
    0,
  };

  nns_orc_conv_s64_to_u64 (res_u64, data_s64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u64[i], (uint64_t) data_s64[i]);
  }

  /* convert f32 */
  float res_f32[array_size] = {
    0,
  };

  nns_orc_conv_s64_to_f32 (res_f32, data_s64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_s64[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = {
    0,
  };

  nns_orc_conv_s64_to_f64 (res_f64, data_s64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_s64[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert u64 to other type)
 */
TEST (testTensorTransform, orcConvU64)
{
  const guint array_size = 10;
  guint i;

  uint64_t data_u64[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_u64[i] = G_MAXUINT64 - i;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = {
    0,
  };

  nns_orc_conv_u64_to_s8 (res_s8, data_u64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_u64[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = {
    0,
  };

  nns_orc_conv_u64_to_u8 (res_u8, data_u64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u8[i], (uint8_t) data_u64[i]);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = {
    0,
  };

  nns_orc_conv_u64_to_s16 (res_s16, data_u64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_u64[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = {
    0,
  };

  nns_orc_conv_u64_to_u16 (res_u16, data_u64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u16[i], (uint16_t) data_u64[i]);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = {
    0,
  };

  nns_orc_conv_u64_to_s32 (res_s32, data_u64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_u64[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = {
    0,
  };

  nns_orc_conv_u64_to_u32 (res_u32, data_u64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u32[i], (uint32_t) data_u64[i]);
  }

  /* convert s64 */
  int64_t res_s64[array_size] = {
    0,
  };

  nns_orc_conv_u64_to_s64 (res_s64, data_u64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_u64[i]);
  }

  /* convert u64 */
  uint64_t res_u64[array_size] = {
    0,
  };

  nns_orc_conv_u64_to_u64 (res_u64, data_u64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u64[i], (uint64_t) data_u64[i]);
  }

  /* convert f32 */
  float res_f32[array_size] = {
    0,
  };

  nns_orc_conv_u64_to_f32 (res_f32, data_u64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) ((int64_t) data_u64[i]));
  }

  /* convert f64 */
  double res_f64[array_size] = {
    0,
  };

  nns_orc_conv_u64_to_f64 (res_f64, data_u64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) ((int64_t) data_u64[i]));
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert f32 to other type)
 */
TEST (testTensorTransform, orcConvF32)
{
  const guint array_size = 10;
  guint i;

  float data_f32[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_f32[i] = (((float) i) + 1.) * -1.;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = {
    0,
  };

  nns_orc_conv_f32_to_s8 (res_s8, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_f32[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = {
    0,
  };

  nns_orc_conv_f32_to_u8 (res_u8, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    int8_t val = (int8_t) data_f32[i];
    EXPECT_EQ (res_u8[i], (uint8_t) val);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = {
    0,
  };

  nns_orc_conv_f32_to_s16 (res_s16, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_f32[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = {
    0,
  };

  nns_orc_conv_f32_to_u16 (res_u16, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    int16_t val = (int16_t) data_f32[i];
    EXPECT_EQ (res_u16[i], (uint16_t) val);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = {
    0,
  };

  nns_orc_conv_f32_to_s32 (res_s32, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_f32[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = {
    0,
  };

  nns_orc_conv_f32_to_u32 (res_u32, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    int32_t val = (int32_t) data_f32[i];
    EXPECT_EQ (res_u32[i], (uint32_t) val);
  }

  /* convert s64 */
  int64_t res_s64[array_size] = {
    0,
  };

  nns_orc_conv_f32_to_s64 (res_s64, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_f32[i]);
  }

  /* convert u64 */
  uint64_t res_u64[array_size] = {
    0,
  };

  nns_orc_conv_f32_to_u64 (res_u64, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    int64_t val = (int64_t) data_f32[i];
    EXPECT_EQ (res_u64[i], (uint64_t) val);
  }

  /* convert f32 */
  float res_f32[array_size] = {
    0,
  };

  nns_orc_conv_f32_to_f32 (res_f32, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_f32[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = {
    0,
  };

  nns_orc_conv_f32_to_f64 (res_f64, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_f32[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert f64 to other type)
 */
TEST (testTensorTransform, orcConvF64)
{
  const guint array_size = 10;
  guint i;

  double data_f64[array_size] = {
    0,
  };

  for (i = 0; i < array_size; i++) {
    data_f64[i] = (((double) i) + 1.) * -1.;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = {
    0,
  };

  nns_orc_conv_f64_to_s8 (res_s8, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_f64[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = {
    0,
  };

  nns_orc_conv_f64_to_u8 (res_u8, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    int8_t val = (int8_t) data_f64[i];
    EXPECT_EQ (res_u8[i], (uint8_t) val);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = {
    0,
  };

  nns_orc_conv_f64_to_s16 (res_s16, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_f64[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = {
    0,
  };

  nns_orc_conv_f64_to_u16 (res_u16, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    int16_t val = (int16_t) data_f64[i];
    EXPECT_EQ (res_u16[i], (uint16_t) val);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = {
    0,
  };

  nns_orc_conv_f64_to_s32 (res_s32, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_f64[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = {
    0,
  };

  nns_orc_conv_f64_to_u32 (res_u32, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    int32_t val = (int32_t) data_f64[i];
    EXPECT_EQ (res_u32[i], (uint32_t) val);
  }

  /* convert s64 */
  int64_t res_s64[array_size] = {
    0,
  };

  nns_orc_conv_f64_to_s64 (res_s64, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_f64[i]);
  }

  /* convert u64 */
  uint64_t res_u64[array_size] = {
    0,
  };

  nns_orc_conv_f64_to_u64 (res_u64, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    int64_t val = (int64_t) data_f64[i];
    EXPECT_EQ (res_u64[i], (uint64_t) val);
  }

  /* convert f32 */
  float res_f32[array_size] = {
    0,
  };

  nns_orc_conv_f64_to_f32 (res_f32, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_f64[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = {
    0,
  };

  nns_orc_conv_f64_to_f64 (res_f64, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_f64[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (performance)
 */
TEST (testTensorTransform, orcPerformance)
{
  const guint array_size = 80000;
  guint i;
  gint64 start_ts, stop_ts, diff_loop, diff_orc;
  uint8_t *data_u8 = (uint8_t *) g_malloc0 (sizeof (uint8_t) * array_size);
  float *data_float = (float *) g_malloc0 (sizeof (float) * array_size);
  gboolean ret = true;

  if (!(ret = (data_u8 != NULL)))
    goto error;

  if (!(ret = (data_float != NULL)))
    goto error;

  /* orc add u8 */
  start_ts = g_get_real_time ();
  nns_orc_add_c_u8 (data_u8, 2, array_size);
  stop_ts = g_get_real_time ();

  diff_orc = stop_ts - start_ts;
  _print_log ("add u8 orc: %" G_GINT64_FORMAT, diff_orc);

  for (i = 0; i < array_size; ++i) {
    EXPECT_EQ (data_u8[i], 2);
  }

  /* loop */
  start_ts = g_get_real_time ();
  for (i = 0; i < array_size; ++i) {
    data_u8[i] += 2;
  }
  stop_ts = g_get_real_time ();

  diff_loop = stop_ts - start_ts;
  _print_log ("add u8 loop: %" G_GINT64_FORMAT, diff_loop);

  /* orc mul u8 */
  start_ts = g_get_real_time ();
  nns_orc_mul_c_u8 (data_u8, 2, array_size);
  stop_ts = g_get_real_time ();

  diff_orc = stop_ts - start_ts;
  _print_log ("mul u8 orc: %" G_GINT64_FORMAT, diff_orc);

  for (i = 0; i < array_size; ++i) {
    EXPECT_EQ (data_u8[i], 8);
  }

  /* loop */
  start_ts = g_get_real_time ();
  for (i = 0; i < array_size; ++i) {
    data_u8[i] *= 2;
  }
  stop_ts = g_get_real_time ();

  diff_loop = stop_ts - start_ts;
  _print_log ("mul u8 loop: %" G_GINT64_FORMAT, diff_loop);

  /* orc typecast to float */
  start_ts = g_get_real_time ();
  nns_orc_conv_u8_to_f32 (data_float, data_u8, array_size);
  stop_ts = g_get_real_time ();

  diff_orc = stop_ts - start_ts;
  _print_log ("conv u8 orc: %" G_GINT64_FORMAT, diff_orc);

  for (i = 0; i < array_size; ++i) {
    EXPECT_FLOAT_EQ (data_float[i], 16.);
  }

  /* loop */
  start_ts = g_get_real_time ();
  for (i = 0; i < array_size; ++i) {
    data_float[i] = (float) data_u8[i];
  }
  stop_ts = g_get_real_time ();

  diff_loop = stop_ts - start_ts;
  _print_log ("conv u8 loop: %" G_GINT64_FORMAT, diff_loop);

  /* orc div f32 */
  start_ts = g_get_real_time ();
  nns_orc_div_c_f32 (data_float, 2., array_size);
  stop_ts = g_get_real_time ();

  diff_orc = stop_ts - start_ts;
  _print_log ("div f32 orc: %" G_GINT64_FORMAT, diff_orc);

  for (i = 0; i < array_size; ++i) {
    EXPECT_FLOAT_EQ (data_float[i], 8.);
  }

  /* loop */
  start_ts = g_get_real_time ();
  for (i = 0; i < array_size; ++i) {
    data_float[i] /= 2.;
  }
  stop_ts = g_get_real_time ();

  diff_loop = stop_ts - start_ts;
  _print_log ("div f32 loop: %" G_GINT64_FORMAT, diff_loop);

  /* orc mul f32 */
  start_ts = g_get_real_time ();
  nns_orc_mul_c_f32 (data_float, 2., array_size);
  stop_ts = g_get_real_time ();

  diff_orc = stop_ts - start_ts;
  _print_log ("mul f32 orc: %" G_GINT64_FORMAT, diff_orc);

  for (i = 0; i < array_size; ++i) {
    EXPECT_FLOAT_EQ (data_float[i], 8.);
  }

  /* loop */
  start_ts = g_get_real_time ();
  for (i = 0; i < array_size; ++i) {
    data_float[i] *= 2.;
  }
  stop_ts = g_get_real_time ();

  diff_loop = stop_ts - start_ts;
  _print_log ("mul f32 loop: %" G_GINT64_FORMAT, diff_loop);

  /* orc add f32 */
  start_ts = g_get_real_time ();
  nns_orc_add_c_f32 (data_float, 2., array_size);
  stop_ts = g_get_real_time ();

  diff_orc = stop_ts - start_ts;
  _print_log ("add f32 orc: %" G_GINT64_FORMAT, diff_orc);

  for (i = 0; i < array_size; ++i) {
    EXPECT_FLOAT_EQ (data_float[i], 18.);
  }

  /* loop */
  start_ts = g_get_real_time ();
  for (i = 0; i < array_size; ++i) {
    data_float[i] += 2.;
  }
  stop_ts = g_get_real_time ();

  diff_loop = stop_ts - start_ts;
  _print_log ("add f32 loop: %" G_GINT64_FORMAT, diff_loop);

  /* init data for tc combined */
  for (i = 0; i < array_size; ++i) {
    data_u8[i] = 1;
  }

  /* orc typecast - add - mul */
  start_ts = g_get_real_time ();
  nns_orc_conv_u8_to_f32 (data_float, data_u8, array_size);
  nns_orc_add_c_f32 (data_float, .2, array_size);
  nns_orc_mul_c_f32 (data_float, 1.2, array_size);
  stop_ts = g_get_real_time ();

  diff_orc = stop_ts - start_ts;
  _print_log ("combined orc: %" G_GINT64_FORMAT, diff_orc);

  for (i = 0; i < array_size; ++i) {
    EXPECT_FLOAT_EQ (data_float[i], (1 + .2) * 1.2);
  }

  /* loop */
  start_ts = g_get_real_time ();
  for (i = 0; i < array_size; ++i) {
    data_float[i] = (float) data_u8[i];
    data_float[i] += .2;
    data_float[i] *= 1.2;
  }
  stop_ts = g_get_real_time ();

  diff_loop = stop_ts - start_ts;
  _print_log ("combined loop: %" G_GINT64_FORMAT, diff_loop);

error:
  g_free (data_u8);
  g_free (data_float);

  ASSERT_TRUE (ret);
}
#endif /* HAVE_ORC */

/**
 * @brief caps negotiation with tensor-filter.
 */
TEST_REQUIRE_TFLITE (testTensorTransform, negotiationFilter)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  gsize in_size, out_size;
  GstTensorsConfig config;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  g_autofree gchar *test_model = g_build_filename (root_path, "tests",
      "test_models", "models", "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  /**
   * tensor-filter information
   * input type uint8 dimension 3:224:224:1
   * output type uint8 dimension 1001:1
   */
  g_autofree gchar *pipeline = g_strdup_printf (
      "tensor_transform mode=typecast option=uint8 ! tensor_filter framework=tensorflow-lite model=%s ! "
      "other/tensors,num_tensors=1,dimensions=(string)\"1001:1:1:1:1\" ! "
      "tensor_transform mode=typecast option=int8",
      test_model);

  h = gst_harness_new_parse (pipeline);
  ASSERT_TRUE (h != NULL);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT32;
  gst_tensor_parse_dimension ("3:224:224", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  /* push buffer (dummy input RGB 224x224, output 1001) */
  in_size = gst_tensors_info_get_size (&config.info, 0);
  out_size = 1001;

  in_buf = gst_harness_create_buffer (h, in_size);
  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  /* get output buffer */
  out_buf = gst_harness_pull (h);
  EXPECT_EQ (gst_buffer_n_memory (out_buf), 1U);
  EXPECT_EQ (gst_buffer_get_size (out_buf), out_size);
  gst_buffer_unref (out_buf);

  gst_harness_teardown (h);
}

/**
 * @brief Test to re-open tf-lite model file in tensor-filter.
 */
TEST_REQUIRE_TFLITE (testTensorFilter, reopenTFlite01)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  gsize in_size, out_size;
  GstTensorsConfig config;
  gchar *str_launch_line, *prop_string;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  h = gst_harness_new_empty ();
  ASSERT_TRUE (h != NULL);

  str_launch_line = g_strdup_printf (
      "tensor_filter framework=tensorflow-lite model=%s", test_model);
  gst_harness_add_parse (h, str_launch_line);
  g_free (str_launch_line);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:224:224:1", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  /* playing state */
  wait_for_element_state (h->element, GST_STATE_PLAYING);

  /* paused state */
  wait_for_element_state (h->element, GST_STATE_PAUSED);

  /* set same model file */
  gst_harness_set (h, "tensor_filter", "framework", "tensorflow-lite", "model",
      test_model, NULL);

  /* playing state */
  wait_for_element_state (h->element, GST_STATE_PLAYING);

  /* get properties */
  gst_harness_get (h, "tensor_filter", "framework", &prop_string, NULL);
  EXPECT_STREQ (prop_string, "tensorflow-lite");
  g_free (prop_string);

  gst_harness_get (h, "tensor_filter", "model", &prop_string, NULL);
  EXPECT_STREQ (prop_string, test_model);
  g_free (prop_string);

  /* push buffer (dummy input RGB 224x224, output 1001) */
  in_size = 3 * 224 * 224;
  out_size = 1001;

  in_buf = gst_harness_create_buffer (h, in_size);
  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  /* get output buffer */
  out_buf = gst_harness_pull (h);
  EXPECT_EQ (gst_buffer_n_memory (out_buf), 1U);
  EXPECT_EQ (gst_buffer_get_size (out_buf), out_size);
  gst_buffer_unref (out_buf);

  gst_harness_teardown (h);
  g_free (test_model);
}

/**
 * @brief Test to re-open tf-lite model file directly with nnfw struct.
 */
TEST_REQUIRE_TFLITE (testTensorFilter, reopenTFlite02)
{
  const gchar fw_name[] = "tensorflow-lite";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    test_model,
    NULL,
  };

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);

  prop->fwname = fw_name;
  prop->model_files = model_files;
  prop->num_models = 1;

  /* open tf-lite model */
  EXPECT_TRUE (fw->open (prop, &private_data) == 0);

  /* re-open tf-lite model */
  EXPECT_TRUE (fw->open (prop, &private_data) > 0);

  /* close tf-lite model */
  fw->close (prop, &private_data);

  g_free (prop);
  g_free (test_model);
}

/**
 * @brief Test to reload tf-lite model set_property of model/is-updatable
 */
TEST_REQUIRE_TFLITE (testTensorFilter, reloadTFliteSetProperty)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  gsize in_size, out_size;
  GstTensorsConfig config;
  gboolean prop_updatable;
  gchar *str_launch_line, *prop_string;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model, *test_model2;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  test_model2 = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v2_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model2, G_FILE_TEST_EXISTS));

  h = gst_harness_new_empty ();
  ASSERT_TRUE (h != NULL);

  str_launch_line = g_strdup_printf ("tensor_filter framework=tensorflow-lite "
                                     "is-updatable=true model=%s",
      test_model);
  gst_harness_add_parse (h, str_launch_line);
  g_free (str_launch_line);

  /* input tensor info */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:224:224:1", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  /* playing state */
  wait_for_element_state (h->element, GST_STATE_PLAYING);

  /* paused state */
  wait_for_element_state (h->element, GST_STATE_PAUSED);

  /* get properties */
  gst_harness_get (h, "tensor_filter", "framework", &prop_string, NULL);
  EXPECT_STREQ (prop_string, "tensorflow-lite");
  g_free (prop_string);

  gst_harness_get (h, "tensor_filter", "model", &prop_string, NULL);
  EXPECT_STREQ (prop_string, test_model);
  g_free (prop_string);

  gst_harness_get (h, "tensor_filter", "is-updatable", &prop_updatable, NULL);
  EXPECT_TRUE (prop_updatable);

  /* playing state */
  wait_for_element_state (h->element, GST_STATE_PLAYING);

  /* push buffer (dummy input RGB 224x224, output 1001) */
  in_size = 3 * 224 * 224;
  out_size = 1001;

  in_buf = gst_harness_create_buffer (h, in_size);
  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  /* get output buffer */
  out_buf = gst_harness_pull (h);
  EXPECT_EQ (gst_buffer_n_memory (out_buf), 1U);
  EXPECT_EQ (gst_buffer_get_size (out_buf), out_size);
  gst_buffer_unref (out_buf);

  /* set second model file */
  gst_harness_set (h, "tensor_filter", "model", test_model2, NULL);

  gst_harness_get (h, "tensor_filter", "model", &prop_string, NULL);
  EXPECT_STREQ (prop_string, test_model2);
  g_free (prop_string);

  /* push buffer again */
  in_buf = gst_harness_create_buffer (h, in_size);
  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  /* get output buffer */
  out_buf = gst_harness_pull (h);
  EXPECT_EQ (gst_buffer_n_memory (out_buf), 1U);
  EXPECT_EQ (gst_buffer_get_size (out_buf), out_size);
  gst_buffer_unref (out_buf);

  gst_harness_teardown (h);
  g_free (test_model);
  g_free (test_model2);
}

/**
 * @brief Test to reload tf-lite; model does not exist (negative)
 */
TEST_REQUIRE_TFLITE (testTensorFilter, reloadTFliteModelNotFound_n)
{
  const gchar fw_name[] = "tensorflow-lite";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close && fw->reloadModel);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);

  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    test_model,
    NULL,
  };

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);

  prop->fwname = fw_name;
  prop->model_files = model_files;
  prop->num_models = 1;

  /* open tf-lite model */
  EXPECT_TRUE (fw->open (prop, &private_data) == 0);

  g_free (test_model);
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v2_1.0_224_quant.tflite", NULL);
  ((gchar **) model_files)[0] = test_model; /* remove const for the test */

  /* reload tf-lite model */
  EXPECT_TRUE (fw->reloadModel (prop, &private_data) == 0);

  g_free (test_model);
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "model_does_not_exist.tflite", NULL);
  ((gchar **) model_files)[0] = test_model; /* remove const for the test */

  /* reload tf-lite model which does not exist */
  EXPECT_FALSE (fw->reloadModel (prop, &private_data) == 0);

  /* close tf-lite model */
  fw->close (prop, &private_data);

  g_free (prop);
  g_free (test_model);
}

/**
 * @brief Test to reload tf-lite; model has wrong dimension (negative)
 */
TEST_REQUIRE_TFLITE (testTensorFilter, reloadTFliteModelWrongDims_n)
{
  const gchar fw_name[] = "tensorflow-lite";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close && fw->reloadModel);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);

  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    test_model,
    NULL,
  };

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);

  prop->fwname = fw_name;
  prop->model_files = model_files;
  prop->num_models = 1;

  /* open tf-lite model */
  EXPECT_TRUE (fw->open (prop, &private_data) == 0);

  g_free (test_model);
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "add.tflite", NULL); /* it has unmatched dimension with mobilenet v1 */
  ((gchar **) model_files)[0] = test_model; /* remove const for the test */

  /* reload tf-lite model with unmatched dims */
  EXPECT_FALSE (fw->reloadModel (prop, &private_data) == 0);

  /* close tf-lite model */
  fw->close (prop, &private_data);

  g_free (prop);
  g_free (test_model);
}

/**
 * @brief Test to reload tf-lite; same model does not exist (negative)
 */
TEST_REQUIRE_TFLITE (testTensorFilter, reloadTFliteSameModelNotFound_n)
{
  const gchar fw_name[] = "tensorflow-lite";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;
  gchar *test_model_renamed;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close && fw->reloadModel);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  test_model_renamed = g_build_filename (root_path, "tests", "test_models",
      "models", "mobilenet_v1_renamed.tflite", NULL);

  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    test_model,
    NULL,
  };

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);

  prop->fwname = fw_name;
  prop->model_files = model_files;
  prop->num_models = 1;

  /* open tf-lite model */
  EXPECT_TRUE (fw->open (prop, &private_data) == 0);

  /* reload tf-lite model again */
  EXPECT_TRUE (fw->reloadModel (prop, &private_data) == 0);

  /* rename the model */
  ASSERT_TRUE (g_rename (test_model, test_model_renamed) == 0);

  /* reload tf-lite model which does not exist */
  EXPECT_FALSE (fw->reloadModel (prop, &private_data) == 0);

  /* test model rollback */
  ASSERT_TRUE (g_rename (test_model_renamed, test_model) == 0);

  /* close tf-lite model */
  fw->close (prop, &private_data);

  g_free (prop);
  g_free (test_model);
  g_free (test_model_renamed);
}

/**
 * @brief Test to reload tf-lite; same model has wrong dimension (negative)
 */
TEST_REQUIRE_TFLITE (testTensorFilter, reloadTFliteSameModelWrongDims_n)
{
  const gchar fw_name[] = "tensorflow-lite";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;

  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *test_model;
  gchar *test_model_backup;
  gchar *test_model_renamed;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close && fw->reloadModel);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  test_model_backup = g_build_filename (root_path, "tests", "test_models",
      "models", "mobilenet_v1_backup.tflite", NULL);
  test_model_renamed = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);

  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    test_model,
    NULL,
  };

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);

  prop->fwname = fw_name;
  prop->model_files = model_files;
  prop->num_models = 1;

  /* open tf-lite model */
  EXPECT_TRUE (fw->open (prop, &private_data) == 0);

  /* reload tf-lite model again */
  EXPECT_TRUE (fw->reloadModel (prop, &private_data) == 0);

  /* rename the model */
  ASSERT_TRUE (g_rename (test_model, test_model_backup) == 0);
  ASSERT_TRUE (g_rename (test_model_renamed, test_model) == 0);

  /* reload tf-lite model with unmatched dims */
  EXPECT_FALSE (fw->reloadModel (prop, &private_data) == 0);

  /* test model rollback */
  ASSERT_TRUE (g_rename (test_model, test_model_renamed) == 0);
  ASSERT_TRUE (g_rename (test_model_backup, test_model) == 0);

  /* close tf-lite model */
  fw->close (prop, &private_data);

  g_free (prop);
  g_free (test_model);
  g_free (test_model_backup);
  g_free (test_model_renamed);
}

/**
 * @brief Test framework auto detection option in tensor-filter.
 */
TEST_REQUIRE_TFLITE (testTensorFilter, frameworkAutoExtTFlite01)
{
  gchar *test_model, *str_launch_line;
  GstElement *gstpipe;
  const gchar fw_name[] = "tensorflow-lite";
  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.tflite");

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter name=tfilter framework=auto model=%s ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name);
  g_free (test_model);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details The order of tensor filter options has changed.
 */
TEST_REQUIRE_TFLITE (testTensorFilter, frameworkAutoExtTFlite02)
{
  gchar *test_model, *str_launch_line;
  GstElement *gstpipe;
  const gchar fw_name[] = "tensorflow-lite";
  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.tflite");

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter name=tfilter model=%s framework=auto ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name);
  g_free (test_model);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Test if options are insensitive to the case
 */
TEST_REQUIRE_TFLITE (testTensorFilter, frameworkAutoExtTFlite03)
{
  gchar *test_model, *str_launch_line;
  GstElement *gstpipe;
  const gchar fw_name[] = "tensorflow-lite";
  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.tflite");

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter name=tfilter model=%s framework=AutO ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name);
  g_free (test_model);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Negative case when model file does not exist
 */
TEST_REQUIRE_TFLITE (testTensorFilter, frameworkAutoExtTFliteModelNotFound_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = NULL;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstElement *gstpipe;

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (
      root_path, "tests", "test_models", "models", "mirage.tflite", NULL);

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=auto model=%s ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name);

  g_free (test_model);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Negative case with not supported extension
 */
TEST_REQUIRE_TFLITE (testTensorFilter, frameworkAutoExtTFliteNotSupportedExt_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = NULL;
  GstElement *gstpipe;
  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.invalid");

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=auto model=%s ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name);

  g_free (test_model);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Negative case when permission of model file is not given.
 */
TEST_REQUIRE_TFLITE (testTensorFilter, frameworkAutoExtTFliteNoPermission_n)
{
  int ret;
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = NULL;
  GstElement *gstpipe;

  /** If the user is root, skip this test */
  if (geteuid () == 0)
    return;

  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.tflite");

  ret = g_chmod (test_model, 0000);
  EXPECT_TRUE (ret == 0);

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=auto model=%s ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name);

  ret = g_chmod (test_model, 0664);
  EXPECT_TRUE (ret == 0);

  g_free (test_model);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Negative case with invalid framework name
 */
TEST_REQUIRE_TFLITE (testTensorFilter, frameworkAutoExtTFliteInvalidFWName_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = NULL;
  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.tflite");
  GstElement *gstpipe;

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=auta model=%s ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name);

  g_free (test_model);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Negative case with invalid dimension of tensor filter
 */
TEST_REQUIRE_TFLITE (testTensorFilter, frameworkAutoExtTFliteWrongDimension_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = "tensorflow-lite";
  GstElement *gstpipe;
  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.tflite");

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter name=tfilter framework=auto model=%s input=784:1 ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name);

  g_free (test_model);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Negative case with invalid input type of tensor filter
 */
TEST_REQUIRE_TFLITE (testTensorFilter, frameworkAutoExtTFliteWrongInputType_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = "tensorflow-lite";
  GstElement *gstpipe;
  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.tflite");

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter name=tfilter framework=auto model=%s  inputtype=float32 ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name);

  g_free (test_model);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection without specifying the option in tensor-filter.
 */
TEST_REQUIRE_TFLITE (testTensorFilter, frameworkAutoNoFw)
{
  gchar *test_model, *str_launch_line;
  GstElement *gstpipe;
  const gchar fw_name[] = "tensorflow-lite";
  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.tflite");

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter name=tfilter model=%s ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name);
  g_free (test_model);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Negative case when model file does not exist
 */
TEST_REQUIRE_TFLITE (testTensorFilter, frameworkAutoNoFwModelNotFound_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = NULL;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstElement *gstpipe;

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (
      root_path, "tests", "test_models", "models", "mirage.tflite", NULL);

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter model=%s ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name);

  g_free (test_model);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection without specifying the option in tensor-filter.
 * @details Negative case with not supported extension
 */
TEST_REQUIRE_TFLITE (testTensorFilter, frameworkAutoNoFwNotSupportedExt_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = NULL;
  GstElement *gstpipe;
  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.invalid");

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter model=%s ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name);

  g_free (test_model);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection without specifying the option in tensor-filter.
 * @details Negative case when permission of model file is not given.
 */
TEST_REQUIRE_TFLITE (testTensorFilter, frameworkAutoNoFwNoPermission_n)
{
  int ret;
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = NULL;
  GstElement *gstpipe;

  /** If the user is root, skip this test */
  if (geteuid () == 0)
    return;

  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.tflite");

  ret = g_chmod (test_model, 0000);
  EXPECT_TRUE (ret == 0);

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter model=%s ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name);

  ret = g_chmod (test_model, 0664);
  EXPECT_TRUE (ret == 0);

  g_free (test_model);
  gst_object_unref (gstpipe);
}

#if !defined(ENABLE_TENSORFLOW_LITE) && !defined(ENABLE_TENSORFLOW2_LITE) \
    && defined(ENABLE_NNFW_RUNTIME)
/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Check if nnfw (second priority) is detected automatically
 */
TEST (testTensorFilter, frameworkAutoExtTfliteNnfw04)
{
  gchar *test_model, *str_launch_line;
  GstElement *gstpipe;
  const gchar fw_name[] = "nnfw";
  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.tflite");

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter name=tfilter framework=auto model=%s ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name);
  g_free (test_model);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection without specifying the option in tensor-filter.
 * @details Check if nnfw (second priority) is detected automatically
 */
TEST (testTensorFilter, frameworkAutoWoOptExtTfliteNnfw)
{
  gchar *test_model, *str_launch_line;
  GstElement *gstpipe;
  const gchar fw_name[] = "nnfw";
  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.tflite");

  str_launch_line = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter name=tfilter model=%s ! tensor_sink",
      test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name);
  g_free (test_model);
  gst_object_unref (gstpipe);
}

#endif /* !defined(ENABLE_TENSORFLOW_LTIE) && defined(ENABLE_NNFW_RUNTIME) */

#ifdef ENABLE_TENSORFLOW
/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Check if tensoflow is detected automatically
 */
TEST (testTensorFilter, frameworkAutoExtPb01)
{
  gchar *test_model, *str_launch_line, *data_path;
  GstElement *gstpipe;
  const gchar fw_name[] = "tensorflow";
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (
      root_path, "tests", "test_models", "models", "mnist.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  data_path = g_build_filename (root_path, "tests", "test_models", "data", "9.raw", NULL);
  ASSERT_TRUE (g_file_test (data_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf (
      "filesrc location=%s ! application/octet-stream ! tensor_converter input-dim=784:1 input-type=uint8 ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter name=tfilter framework=auto model=%s input=784:1 inputtype=float32 inputname=input output=10:1 outputtype=float32 outputname=softmax ! tensor_sink",
      data_path, test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name);
  g_free (test_model);
  gst_object_unref (gstpipe);
  g_free (data_path);
}

/**
 * @brief Test framework auto detection without specifying the option in tensor-filter.
 * @details Check if tensoflow is detected automatically
 */
TEST (testTensorFilter, frameworkAutoWoOptExtPb)
{
  gchar *test_model, *str_launch_line, *data_path;
  GstElement *gstpipe;
  const gchar fw_name[] = "tensorflow";
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (
      root_path, "tests", "test_models", "models", "mnist.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  data_path = g_build_filename (root_path, "tests", "test_models", "data", "9.raw", NULL);
  ASSERT_TRUE (g_file_test (data_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf (
      "filesrc location=%s ! application/octet-stream ! tensor_converter input-dim=784:1 input-type=uint8 ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter name=tfilter model=%s input=784:1 inputtype=float32 inputname=input output=10:1 outputtype=float32 outputname=softmax ! tensor_sink",
      data_path, test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name);
  g_free (test_model);
  gst_object_unref (gstpipe);
  g_free (data_path);
}
#else
/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Negative case whtn tensorflow is not enabled
 */
TEST (testTensorFilter, frameworkAutoExtPbTfDisabled_n)
{
  gchar *test_model, *str_launch_line, *data_path;
  const gchar *fw_name = NULL;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstElement *gstpipe;

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (
      root_path, "tests", "test_models", "models", "mnist.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  data_path = g_build_filename (root_path, "tests", "test_models", "data", "9.raw", NULL);
  ASSERT_TRUE (g_file_test (data_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf (
      "filesrc location=%s ! application/octet-stream ! tensor_converter input-dim=784:1 input-type=uint8 ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter name=tfilter framework=auto model=%s input=784:1 inputtype=float32 inputname=input output=10:1 outputtype=float32 outputname=softmax ! tensor_sink",
      data_path, test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name);

  g_free (test_model);
  g_free (data_path);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection without specifying the option in tensor-filter.
 * @details Negative case whtn tensorflow is not enabled
 */
TEST (testTensorFilter, frameworkAutoWoOptExtPbTfDisabled_n)
{
  gchar *test_model, *str_launch_line, *data_path;
  const gchar *fw_name = NULL;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstElement *gstpipe;

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (
      root_path, "tests", "test_models", "models", "mnist.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  data_path = g_build_filename (root_path, "tests", "test_models", "data", "9.raw", NULL);
  ASSERT_TRUE (g_file_test (data_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf (
      "filesrc location=%s ! application/octet-stream ! tensor_converter input-dim=784:1 input-type=uint8 ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter name=tfilter model=%s input=784:1 inputtype=float32 inputname=input output=10:1 outputtype=float32 outputname=softmax ! tensor_sink",
      data_path, test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name);

  g_free (test_model);
  g_free (data_path);
  gst_object_unref (gstpipe);
}
#endif /* ENABLE_TENSORFLOW */

#ifdef ENABLE_CAFFE2
/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Check if caffe2 is detected automatically
 */
TEST (testTensorFilter, frameworkAutoExtPb03)
{
  gchar *test_model, *str_launch_line, *test_model_2, *data_path;
  GstElement *gstpipe;
  const gchar fw_name[] = "caffe2";
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (
      root_path, "tests", "test_models", "models", "caffe2_init_net.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  test_model_2 = g_build_filename (root_path, "tests", "test_models", "models",
      "caffe2_predict_net.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model_2, G_FILE_TEST_EXISTS));
  data_path = g_build_filename (root_path, "tests", "test_models", "data", "5", NULL);
  ASSERT_TRUE (g_file_test (data_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf (
      "filesrc location=%s blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=32:32:3:1 input-type=float32 ! tensor_filter name=tfilter framework=caffe2 model=%s,%s inputname=data input=32:32:3:1 inputtype=float32 output=10:1 outputtype=float32 outputname=softmax ! fakesink",
      data_path, test_model, test_model_2);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name);
  g_free (test_model);
  g_free (test_model_2);
  g_free (data_path);
  gst_object_unref (gstpipe);
}

#else
/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Check if caffe2 is not enabled
 */
TEST (testTensorFilter, frameworkAutoExtPbCaffe2Disabled_n)
{
  gchar *test_model, *str_launch_line, *test_model_2, *data_path;
  const gchar *fw_name = NULL;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstElement *gstpipe;

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (
      root_path, "tests", "test_models", "models", "caffe2_init_net.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  test_model_2 = g_build_filename (root_path, "tests", "test_models", "models",
      "caffe2_predict_net.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model_2, G_FILE_TEST_EXISTS));
  data_path = g_build_filename (root_path, "tests", "test_models", "data", "5", NULL);
  ASSERT_TRUE (g_file_test (data_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf (
      "filesrc location=%s blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=32:32:3:1 input-type=float32 ! tensor_filter name=tfilter framework=caffe2 model=%s,%s inputname=data input=32:32:3:1 inputtype=float32 output=10:1 outputtype=float32 outputname=softmax ! fakesink",
      data_path, test_model, test_model_2);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name);

  g_free (test_model);
  g_free (test_model_2);
  g_free (data_path);
  gst_object_unref (gstpipe);
}
#endif /* ENABLE_CAFFE2 */

#ifdef ENABLE_PYTORCH
/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Check if pytorch is detected automatically
 */
TEST (testTensorFilter, frameworkAutoExtPt01)
{
  gchar *test_model, *str_launch_line, *image_path;
  GstElement *gstpipe;
  const gchar fw_name[] = "pytorch";
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (
      root_path, "tests", "test_models", "models", "pytorch_lenet5.pt", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  image_path = g_build_filename (root_path, "tests", "test_models", "data", "9.png", NULL);
  ASSERT_TRUE (g_file_test (image_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf (
      "filesrc location=%s ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter name=tfilter framework=auto model=%s input=1:28:28:1 inputtype=uint8 output=10:1:1:1 outputtype=uint8 ! tensor_sink",
      image_path, test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name);
  g_free (test_model);
  g_free (image_path);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection without specifying the option in tensor-filter.
 * @details Check if pytorch is detected automatically
 */
TEST (testTensorFilter, frameworkAutoWoOptExtPt01)
{
  gchar *test_model, *str_launch_line, *image_path;
  GstElement *gstpipe;
  const gchar fw_name[] = "pytorch";
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (
      root_path, "tests", "test_models", "models", "pytorch_lenet5.pt", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  image_path = g_build_filename (root_path, "tests", "test_models", "data", "9.png", NULL);
  ASSERT_TRUE (g_file_test (image_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf (
      "filesrc location=%s ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter name=tfilter model=%s input=1:28:28:1 inputtype=uint8 output=10:1:1:1 outputtype=uint8 ! tensor_sink",
      image_path, test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name);

  g_free (test_model);
  g_free (image_path);
  gst_object_unref (gstpipe);
}

#else
/**
 * @brief Test framework auto detection option in tensor-filter.
 * @details Check if pytorch is not enabled
 */
TEST (testTensorFilter, frameworkAutoExtPtPytorchDisabled_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = NULL;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstElement *gstpipe;

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (
      root_path, "tests", "test_models", "models", "pytorch_lenet5.pt", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  gchar *image_path
      = g_build_filename (root_path, "tests", "test_models", "data", "9.png", NULL);
  ASSERT_TRUE (g_file_test (image_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf (
      "filesrc location=%s ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter framework=auto model=%s input=1:28:28:1 inputtype=uint8 output=10:1:1:1 outputtype=uint8 ! tensor_sink",
      image_path, test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name);

  g_free (image_path);
  g_free (test_model);
  gst_object_unref (gstpipe);
}

/**
 * @brief Test framework auto detection without specifying the option in tensor-filter.
 * @details Check if pytorch is not enabled
 */
TEST (testTensorFilter, frameworkAutoWoOptExtPtPytorchDisabled_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = NULL;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstElement *gstpipe;

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (
      root_path, "tests", "test_models", "models", "pytorch_lenet5.pt", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  gchar *image_path
      = g_build_filename (root_path, "tests", "test_models", "data", "9.png", NULL);
  ASSERT_TRUE (g_file_test (image_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf (
      "filesrc location=%s ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter model=%s input=1:28:28:1 inputtype=uint8 output=10:1:1:1 outputtype=uint8 ! tensor_sink",
      image_path, test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name);

  g_free (image_path);
  g_free (test_model);
  gst_object_unref (gstpipe);
}
#endif /* ENABLE_PYTORCH */

/**
 * @brief Test for inputranks and outputranks property of the tensor_filter
 * @details Given dimension string, check its rank value.
 */
TEST_REQUIRE_TFLITE (testTensorFilter, propertyRank01)
{
  gchar *str_launch_line;
  GstHarness *hrnss;
  GstElement *filter;
  gchar *test_model;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  hrnss = gst_harness_new_empty ();
  ASSERT_TRUE (hrnss != NULL);

  str_launch_line = g_strdup_printf ("tensor_filter framework=auto model=%s input=3:224:224 inputtype=uint8 \
      output=1001:1:1:1 outputtype=uint8 ",
      test_model);
  gst_harness_add_parse (hrnss, str_launch_line);
  g_free (str_launch_line);

  filter = gst_harness_find_element (hrnss, "tensor_filter");
  ASSERT_TRUE (filter != NULL);

  /* Check input dimension '3:224:224' */
  gchar *input_dim;
  g_object_get (filter, "input", &input_dim, NULL);
  EXPECT_STREQ (input_dim, "3:224:224");
  g_free (input_dim);

  /* Rank should be 3 since dimension string of the input is explicitly '3:224:224'. */
  gchar *input_ranks;
  g_object_get (filter, "inputranks", &input_ranks, NULL);
  EXPECT_STREQ (input_ranks, "3");
  g_free (input_ranks);

  gchar *output_dim;
  g_object_get (filter, "output", &output_dim, NULL);
  EXPECT_STREQ (output_dim, "1001:1:1:1");
  g_free (output_dim);

  /* Rank should be 4 since dimension string of the output is explicitly '1000:1:1:1'. */
  gchar *output_ranks;
  g_object_get (filter, "outputranks", &output_ranks, NULL);
  EXPECT_STREQ (output_ranks, "4");
  g_free (output_ranks);

  g_object_unref (filter);
  gst_harness_teardown (hrnss);
}

/**
 * @brief Test for inputranks and outputranks property of the tensor_filter
 * @details Given dimension string, check its rank value.
 */
TEST_REQUIRE_TFLITE (testTensorFilter, propertyRank02)
{
  gchar *str_launch_line;
  GstHarness *hrnss;
  GstElement *filter;
  gchar *test_model;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  hrnss = gst_harness_new_empty ();
  ASSERT_TRUE (hrnss != NULL);

  str_launch_line = g_strdup_printf ("tensor_filter framework=auto model=%s ", test_model);
  gst_harness_add_parse (hrnss, str_launch_line);
  g_free (str_launch_line);

  filter = gst_harness_find_element (hrnss, "tensor_filter");
  ASSERT_TRUE (filter != NULL);

  gchar *input_dim;
  g_object_get (filter, "input", &input_dim, NULL);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal (input_dim, "3:224:224:1"));
  g_free (input_dim);

  /* Rank should be 4 since input dimension string is not given. */
  gchar *input_ranks;
  g_object_get (filter, "inputranks", &input_ranks, NULL);
  EXPECT_STREQ (input_ranks, "4");
  g_free (input_ranks);

  gchar *output_dim;
  g_object_get (filter, "output", &output_dim, NULL);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal (output_dim, "1001:1"));
  g_free (output_dim);

  /* Rank should be 2 since output dimension string is not given. */
  gchar *output_ranks;
  g_object_get (filter, "outputranks", &output_ranks, NULL);
  EXPECT_STREQ (output_ranks, "2");
  g_free (output_ranks);

  g_object_unref (filter);
  gst_harness_teardown (hrnss);
}

/**
 * @brief Test for inputranks and outputranks property of the tensor_filter
 * @details Given dimension string, check its rank value.
 */
TEST_REQUIRE_TFLITE (testTensorFilter, propertyRank03_n)
{
  gchar *str_launch_line;
  GstHarness *hrnss;
  GstElement *filter;
  gchar *test_model;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  hrnss = gst_harness_new_empty ();
  ASSERT_TRUE (hrnss != NULL);

  str_launch_line = g_strdup_printf ("tensor_filter framework=auto model=%s input=3:224:224 inputtype=uint8 \
      output=1001:1 outputtype=uint8 ",
      test_model);
  gst_harness_add_parse (hrnss, str_launch_line);
  g_free (str_launch_line);

  filter = gst_harness_find_element (hrnss, "tensor_filter");
  ASSERT_TRUE (filter != NULL);

  /* The input dimension string should be '3:224:224' since it is given in the pipeline. */
  gchar *input_dim;
  g_object_get (filter, "input", &input_dim, NULL);
  EXPECT_STRNE (input_dim, "3:224:224:1");
  g_free (input_dim);

  /* The input dimension string should be '1001:1' since it is given in the pipeline. */
  gchar *output_dim;
  g_object_get (filter, "output", &output_dim, NULL);
  EXPECT_STRNE (output_dim, "1001:1:1:1");
  g_free (output_dim);

  /* Rank should be 2 since dimension string of the output is explicitly '1000:1:1:1'. */
  gchar *output_ranks;
  g_object_get (filter, "outputranks", &output_ranks, NULL);
  EXPECT_STREQ (output_ranks, "2");
  g_free (output_ranks);

  g_object_unref (filter);
  gst_harness_teardown (hrnss);
}

/**
 * @brief Test for flex tensor in tensor_filter
 */
TEST_REQUIRE_TFLITE (testTensorFilter, flexInvalidBuffer1_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstMemory *mem;
  GstTensorMetaInfo meta;
  GstTensorsInfo info;
  GstCaps *caps;
  gpointer data;
  gsize data_size;
  gchar *pipeline;
  gchar *test_model;

  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.tflite");

  h = gst_harness_new_empty ();
  ASSERT_TRUE (h != NULL);

  pipeline = g_strdup_printf ("tensor_filter framework=tensorflow-lite model=%s", test_model);
  gst_harness_add_parse (h, pipeline);
  g_free (pipeline);

  /* set caps (flex-tensor) */
  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  gst_harness_set_src_caps (h, caps);

  gst_tensors_info_init (&info);
  info.num_tensors = 2;

  info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:224:224:1", info.info[0].dimension);
  info.info[1].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:8", info.info[1].dimension);

  /* push buffer (invalid number) */
  in_buf = gst_buffer_new ();

  /* 1st mem block */
  gst_tensor_info_convert_to_meta (&info.info[0], &meta);
  data_size = gst_tensor_meta_info_get_header_size (&meta);
  data_size += gst_tensor_meta_info_get_data_size (&meta);

  data = g_malloc0 (data_size);
  gst_tensor_meta_info_update_header (&meta, data);

  mem = gst_memory_new_wrapped (
      (GstMemoryFlags) 0, data, data_size, 0, data_size, data, g_free);
  gst_buffer_append_memory (in_buf, mem);

  /* 2nd mem block (invalid, unnecessary block) */
  gst_tensor_info_convert_to_meta (&info.info[1], &meta);
  data_size = gst_tensor_meta_info_get_header_size (&meta);
  data_size += gst_tensor_meta_info_get_data_size (&meta);

  data = g_malloc0 (data_size);
  gst_tensor_meta_info_update_header (&meta, data);

  mem = gst_memory_new_wrapped (
      (GstMemoryFlags) 0, data, data_size, 0, data_size, data, g_free);
  gst_buffer_append_memory (in_buf, mem);

  EXPECT_NE (gst_harness_push (h, in_buf), GST_FLOW_OK);

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for flex tensor in tensor_filter
 */
TEST_REQUIRE_TFLITE (testTensorFilter, flexInvalidBuffer2_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstMemory *mem;
  GstTensorMetaInfo meta;
  GstTensorInfo info;
  GstCaps *caps;
  gpointer data;
  gsize data_size;
  gchar *pipeline;
  gchar *test_model;

  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.tflite");

  h = gst_harness_new_empty ();
  ASSERT_TRUE (h != NULL);

  pipeline = g_strdup_printf ("tensor_filter framework=tensorflow-lite model=%s", test_model);
  gst_harness_add_parse (h, pipeline);
  g_free (pipeline);

  /* set caps (flex-tensor) */
  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  gst_harness_set_src_caps (h, caps);

  gst_tensor_info_init (&info);
  info.type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:224:224:1", info.dimension);

  /* push buffer (invalid size) */
  in_buf = gst_buffer_new ();

  gst_tensor_info_convert_to_meta (&info, &meta);
  data_size = gst_tensor_meta_info_get_header_size (&meta);
  data_size += gst_tensor_meta_info_get_data_size (&meta) / 2;

  data = g_malloc0 (data_size);
  gst_tensor_meta_info_update_header (&meta, data);

  mem = gst_memory_new_wrapped (
      (GstMemoryFlags) 0, data, data_size, 0, data_size, data, g_free);
  gst_buffer_append_memory (in_buf, mem);

  EXPECT_NE (gst_harness_push (h, in_buf), GST_FLOW_OK);

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for flex tensor in tensor_filter
 */
TEST_REQUIRE_TFLITE (testTensorFilter, flexToFlex)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstMemory *mem;
  GstTensorMetaInfo meta;
  GstTensorInfo info;
  GstCaps *caps;
  gpointer data;
  gsize data_size;
  gchar *pipeline;
  gchar *test_model;
  guint received;

  GET_MODEL_PATH ("mobilenet_v1_1.0_224_quant.tflite");

  h = gst_harness_new_empty ();
  ASSERT_TRUE (h != NULL);

  pipeline = g_strdup_printf ("tensor_filter framework=tensorflow-lite model=%s", test_model);
  gst_harness_add_parse (h, pipeline);
  g_free (pipeline);

  /* set caps (flex-tensor) */
  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);

  gst_harness_set_src_caps (h, gst_caps_copy (caps));
  gst_harness_set_sink_caps (h, caps);

  gst_tensor_info_init (&info);
  info.type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:224:224:1", info.dimension);

  /* push buffer (uint8, 3:224:224:1) */
  in_buf = gst_buffer_new ();

  gst_tensor_info_convert_to_meta (&info, &meta);
  data_size = gst_tensor_meta_info_get_header_size (&meta);
  data_size += gst_tensor_meta_info_get_data_size (&meta);

  data = g_malloc0 (data_size);
  gst_tensor_meta_info_update_header (&meta, data);

  mem = gst_memory_new_wrapped (
      (GstMemoryFlags) 0, data, data_size, 0, data_size, data, g_free);
  gst_buffer_append_memory (in_buf, mem);

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  /* wait for output buffer */
  received = _harness_wait_for_output_buffer (h, 1U);
  EXPECT_EQ (received, 1U);

  /* get output buffer (uint8, 1001:1) */
  if (received) {
    out_buf = gst_harness_pull (h);
    EXPECT_EQ (gst_buffer_n_memory (out_buf), 1U);

    mem = gst_buffer_peek_memory (out_buf, 0);
    gst_tensor_meta_info_parse_memory (&meta, mem);

    EXPECT_EQ (meta.type, _NNS_UINT8);
    EXPECT_EQ (meta.dimension[0], 1001U);
    EXPECT_EQ (meta.dimension[1], 1U);
    EXPECT_EQ ((media_type) meta.media_type, _NNS_TENSOR);

    data_size = gst_tensor_meta_info_get_header_size (&meta);
    data_size += gst_tensor_meta_info_get_data_size (&meta);

    EXPECT_EQ (gst_buffer_get_size (out_buf), data_size);
    gst_buffer_unref (out_buf);
  }

  gst_harness_teardown (h);
}

#if defined(ENABLE_PROTOBUF) && defined(ENABLE_FLATBUF)
/**
 * @brief Test for flatbuf, flexbuf and protobuf (tensors -> serialized buf -> tensors)
 */
TEST (testStreamBuffers, tensorsNormal)
{
  const gchar *mode_name[3] = { "flatbuf", "flexbuf", "protobuf" };
  GstBuffer *dec_out_buf = NULL, *conv_out_buf = NULL;
  GstTensorsConfig config, check_config;
  GstMemory *mem;
  GstTensorMemory input[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo info;
  guint mode_idx, i, j;
  const GstTensorDecoderDef *fb_dec;
  const NNStreamerExternalConverter *fb_conv;

  for (mode_idx = 0; mode_idx < 3; mode_idx++) {
    /** Find converter and decoder subplugins */
    fb_dec = nnstreamer_decoder_find (mode_name[mode_idx]);
    fb_conv = nnstreamer_converter_find (mode_name[mode_idx]);
    ASSERT_TRUE (fb_dec);
    ASSERT_TRUE (fb_conv);

    /** Prepare input */
    gst_tensors_config_init (&config);
    gst_tensors_config_init (&check_config);
    config.rate_n = 0;
    config.rate_d = 1;
    config.info.num_tensors = 2;

    config.info.info[0].type = _NNS_INT32;
    gst_tensor_parse_dimension ("3:4:2:2", config.info.info[0].dimension);
    config.info.info[1].name = g_strdup ("2nd_tensor");
    config.info.info[1].type = _NNS_INT32;
    gst_tensor_parse_dimension ("3:4:2:2", config.info.info[1].dimension);

    for (i = 0; i < config.info.num_tensors; i++) {
      input[i].size = gst_tensors_info_get_size (&config.info, i);
      input[i].data = g_malloc0 (input[0].size);
      memcpy (input[i].data, aggr_test_frames[i], input[i].size);
    }

    /** Decode tensors to serialized buffers */
    dec_out_buf = gst_buffer_new ();
    fb_dec->decode (NULL, &config, input, dec_out_buf);

    for (i = 0; i < config.info.num_tensors; i++) {
      g_free (input[i].data);
    }

    EXPECT_TRUE (dec_out_buf != NULL);
    EXPECT_EQ (gst_buffer_n_memory (dec_out_buf), 1U);

    /** Convert flatbuf to tensors */
    conv_out_buf = fb_conv->convert (dec_out_buf, &check_config, NULL);
    EXPECT_EQ (gst_buffer_n_memory (conv_out_buf), 2U);

    /** Check tensors config. */
    EXPECT_TRUE (check_config.info.info[0].name == NULL);
    EXPECT_STREQ ("2nd_tensor", check_config.info.info[1].name);
    EXPECT_TRUE (gst_tensors_config_is_equal (&config, &check_config));
    /** Check data */
    for (i = 0; i < config.info.num_tensors; i++) {
      mem = gst_buffer_peek_memory (conv_out_buf, i);
      ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));
      for (j = 0; j < 48; j++)
        EXPECT_EQ (((gint *) info.data)[j], aggr_test_frames[i][j]);
      gst_memory_unmap (mem, &info);
    }

    gst_tensors_config_free (&config);
    gst_tensors_config_free (&check_config);
    gst_buffer_unref (dec_out_buf);
    gst_buffer_unref (conv_out_buf);
  }
}

/**
 * @brief Test for decoder subplugins with invalid parameter
 */
TEST (testDecoderSubplugins, flatbufInvalidParam0_n)
{
  const gchar *mode_name = "flatbuf";
  GstBuffer *dec_out_buf = NULL;
  GstTensorMemory input[NNS_TENSOR_SIZE_LIMIT];
  const GstTensorDecoderDef *fb_dec;

  /** Find decoder subplugins */
  fb_dec = nnstreamer_decoder_find (mode_name);
  ASSERT_TRUE (fb_dec);

  dec_out_buf = gst_buffer_new ();
  EXPECT_EQ (GST_FLOW_ERROR, fb_dec->decode (NULL, NULL, input, dec_out_buf));

  gst_buffer_unref (dec_out_buf);
}

/**
 * @brief Test for decoder subplugins with invalid parameter
 */
TEST (testDecoderSubplugins, flatbufInvalidParam1_n)
{
  const gchar *mode_name = "flatbuf";
  GstBuffer *dec_out_buf = NULL;
  GstTensorsConfig config;
  const GstTensorDecoderDef *fb_dec;

  /** Find  decoder subplugins */
  fb_dec = nnstreamer_decoder_find (mode_name);
  ASSERT_TRUE (fb_dec);

  gst_tensors_config_init (&config);
  dec_out_buf = gst_buffer_new ();
  EXPECT_EQ (GST_FLOW_ERROR, fb_dec->decode (NULL, &config, NULL, dec_out_buf));

  gst_tensors_config_free (&config);
  gst_buffer_unref (dec_out_buf);
}

/**
 * @brief Test for decoder subplugins with invalid parameter
 */
TEST (testDecoderSubplugins, flatbufInvalidParam2_n)
{
  const gchar *mode_name = "flatbuf";
  GstTensorsConfig config;
  GstTensorMemory input[NNS_TENSOR_SIZE_LIMIT];
  const GstTensorDecoderDef *fb_dec;

  /** Find  decoder subplugins */
  fb_dec = nnstreamer_decoder_find (mode_name);
  ASSERT_TRUE (fb_dec);

  gst_tensors_config_init (&config);
  EXPECT_EQ (GST_FLOW_ERROR, fb_dec->decode (NULL, &config, input, NULL));
}

/**
 * @brief Test for converter subplugins with invalid parameter
 */
TEST (testConverterSubplugins, flatbufInvalidParam0_n)
{
  const gchar *mode_name = "flatbuf";
  GstBuffer *conv_out_buf = NULL;
  GstTensorsConfig config;
  const NNStreamerExternalConverter *fb_conv;

  /** Find converter subplugins */
  fb_conv = nnstreamer_converter_find (mode_name);
  ASSERT_TRUE (fb_conv);

  gst_tensors_config_init (&config);
  conv_out_buf = fb_conv->convert (NULL, &config, NULL);

  EXPECT_TRUE (NULL == conv_out_buf);
  gst_tensors_config_free (&config);
}

/**
 * @brief Test for converter subplugins with invalid parameter
 */
TEST (testConverterSubplugins, flatbufInvalidParam1_n)
{
  const gchar *mode_name = "flatbuf";
  GstBuffer *in_buf = NULL, *conv_out_buf = NULL;
  GstTensorsConfig config;
  const NNStreamerExternalConverter *fb_conv;

  /** Find converter subplugins */
  fb_conv = nnstreamer_converter_find (mode_name);
  ASSERT_TRUE (fb_conv);

  /** Prepare input */
  gst_tensors_config_init (&config);
  in_buf = gst_buffer_new ();
  conv_out_buf = fb_conv->convert (in_buf, NULL, NULL);

  EXPECT_TRUE (NULL == conv_out_buf);
  gst_tensors_config_free (&config);
  gst_buffer_unref (in_buf);
}

/**
 * @brief Test for decoder subplugins with invalid parameter
 */
TEST (testDecoderSubplugins, protobufInvalidParam0_n)
{
  const gchar *mode_name = "protobuf";
  GstBuffer *dec_out_buf = NULL;
  GstTensorMemory input[NNS_TENSOR_SIZE_LIMIT];
  const GstTensorDecoderDef *pb_dec;

  /** Find decoder subplugins */
  pb_dec = nnstreamer_decoder_find (mode_name);
  ASSERT_TRUE (pb_dec);

  dec_out_buf = gst_buffer_new ();
  EXPECT_EQ (GST_FLOW_ERROR, pb_dec->decode (NULL, NULL, input, dec_out_buf));

  gst_buffer_unref (dec_out_buf);
}

/**
 * @brief Test for decoder subplugins with invalid parameter
 */
TEST (testDecoderSubplugins, protobufInvalidParam1_n)
{
  const gchar *mode_name = "protobuf";
  GstBuffer *dec_out_buf = NULL;
  GstTensorsConfig config;
  const GstTensorDecoderDef *pb_dec;

  /** Find  decoder subplugins */
  pb_dec = nnstreamer_decoder_find (mode_name);
  ASSERT_TRUE (pb_dec);

  gst_tensors_config_init (&config);
  dec_out_buf = gst_buffer_new ();
  EXPECT_EQ (GST_FLOW_ERROR, pb_dec->decode (NULL, &config, NULL, dec_out_buf));

  gst_tensors_config_free (&config);
  gst_buffer_unref (dec_out_buf);
}

/**
 * @brief Test for decoder subplugins with invalid parameter
 */
TEST (testDecoderSubplugins, protobufInvalidParam2_n)
{
  const gchar *mode_name = "protobuf";
  GstTensorsConfig config;
  GstTensorMemory input[NNS_TENSOR_SIZE_LIMIT];
  const GstTensorDecoderDef *pb_dec;

  /** Find  decoder subplugins */
  pb_dec = nnstreamer_decoder_find (mode_name);
  ASSERT_TRUE (pb_dec);

  gst_tensors_config_init (&config);
  EXPECT_EQ (GST_FLOW_ERROR, pb_dec->decode (NULL, &config, input, NULL));
}

/**
 * @brief Test for converter subplugins with invalid parameter
 */
TEST (testConverterSubplugins, protobufInvalidParam0_n)
{
  const gchar *mode_name = "protobuf";
  GstBuffer *conv_out_buf = NULL;
  GstTensorsConfig config;
  const NNStreamerExternalConverter *pb_conv;

  /** Find converter subplugins */
  pb_conv = nnstreamer_converter_find (mode_name);
  ASSERT_TRUE (pb_conv);

  gst_tensors_config_init (&config);
  conv_out_buf = pb_conv->convert (NULL, &config, NULL);

  EXPECT_TRUE (NULL == conv_out_buf);
  gst_tensors_config_free (&config);
}

/**
 * @brief Test for converter subplugins with invalid parameter
 */
TEST (testConverterSubplugins, protobufInvalidParam1_n)
{
  const gchar *mode_name = "protobuf";
  GstBuffer *in_buf = NULL, *conv_out_buf = NULL;
  GstTensorsConfig config;
  const NNStreamerExternalConverter *pb_conv;

  /** Find converter subplugins */
  pb_conv = nnstreamer_converter_find (mode_name);
  ASSERT_TRUE (pb_conv);

  gst_tensors_config_init (&config);
  in_buf = gst_buffer_new ();
  conv_out_buf = pb_conv->convert (in_buf, NULL, NULL);

  EXPECT_TRUE (NULL == conv_out_buf);
  gst_tensors_config_free (&config);
  gst_buffer_unref (in_buf);
}

/**
 * @brief Test for decoder subplugins with invalid parameter
 */
TEST (testDecoderSubplugins, flexbufInvalidParam0_n)
{
  const gchar *mode_name = "flexbuf";
  GstBuffer *dec_out_buf = NULL;
  GstTensorMemory input[NNS_TENSOR_SIZE_LIMIT];
  const GstTensorDecoderDef *flx_dec;

  /** Find decoder subplugins */
  flx_dec = nnstreamer_decoder_find (mode_name);
  ASSERT_TRUE (flx_dec);

  dec_out_buf = gst_buffer_new ();
  EXPECT_EQ (GST_FLOW_ERROR, flx_dec->decode (NULL, NULL, input, dec_out_buf));

  gst_buffer_unref (dec_out_buf);
}

/**
 * @brief Test for decoder subplugins with invalid parameter
 */
TEST (testDecoderSubplugins, flexbufInvalidParam1_n)
{
  const gchar *mode_name = "flexbuf";
  GstBuffer *dec_out_buf = NULL;
  GstTensorsConfig config;
  const GstTensorDecoderDef *flx_dec;

  /** Find  decoder subplugins */
  flx_dec = nnstreamer_decoder_find (mode_name);
  ASSERT_TRUE (flx_dec);

  gst_tensors_config_init (&config);

  dec_out_buf = gst_buffer_new ();
  EXPECT_EQ (GST_FLOW_ERROR, flx_dec->decode (NULL, &config, NULL, dec_out_buf));

  gst_tensors_config_free (&config);
  gst_buffer_unref (dec_out_buf);
}

/**
 * @brief Test for decoder subplugins with invalid parameter
 */
TEST (testDecoderSubplugins, flexbufInvalidParam2_n)
{
  const gchar *mode_name = "flexbuf";
  GstTensorsConfig config;
  GstTensorMemory input[NNS_TENSOR_SIZE_LIMIT];
  const GstTensorDecoderDef *flx_dec;

  /** Find  decoder subplugins */
  flx_dec = nnstreamer_decoder_find (mode_name);
  ASSERT_TRUE (flx_dec);

  gst_tensors_config_init (&config);
  EXPECT_EQ (GST_FLOW_ERROR, flx_dec->decode (NULL, &config, input, NULL));
}

/**
 * @brief Test for converter subplugins with invalid parameter
 */
TEST (testConverterSubplugins, flexbufInvalidParam0_n)
{
  const gchar *mode_name = "flexbuf";
  GstBuffer *conv_out_buf = NULL;
  GstTensorsConfig config;
  const NNStreamerExternalConverter *flx_conv;

  /** Find converter subplugins */
  flx_conv = nnstreamer_converter_find (mode_name);
  ASSERT_TRUE (flx_conv);

  gst_tensors_config_init (&config);
  conv_out_buf = flx_conv->convert (NULL, &config, NULL);

  EXPECT_TRUE (NULL == conv_out_buf);
  gst_tensors_config_free (&config);
}

/**
 * @brief Test for converter subplugins with invalid parameter
 */
TEST (testConverterSubplugins, flexbufInvalidParam1_n)
{
  const gchar *mode_name = "flexbuf";
  GstBuffer *in_buf = NULL, *conv_out_buf = NULL;
  GstTensorsConfig config;
  const NNStreamerExternalConverter *flx_conv;

  /** Find converter subplugins */
  flx_conv = nnstreamer_converter_find (mode_name);
  ASSERT_TRUE (flx_conv);

  gst_tensors_config_init (&config);
  in_buf = gst_buffer_new ();
  conv_out_buf = flx_conv->convert (in_buf, NULL, NULL);

  EXPECT_TRUE (NULL == conv_out_buf);
  gst_tensors_config_free (&config);
  gst_buffer_unref (in_buf);
}
#endif /** ENABLE_FLATBUF && ENABLE_PROTOBUF */

/**
 * @brief Data structure for tensor-crop test.
 */
typedef struct {
  GstHarness *crop;
  GstHarness *raw;
  GstHarness *info;
  GstHarness *raw_q;
  GstHarness *info_q;

  GstTensorInfo raw_info;
  tensor_format raw_format;
  guint received;
  gpointer raw_data;
  gsize raw_size;
  GstClockTime ts_raw;
  tensor_type info_type;
  gpointer info_data;
  guint info_num;
  gsize info_size;
  GstClockTime ts_info;
} crop_test_data_s;

/**
 * @brief Initialize tensor-crop test data.
 * After calling this function, you should set raw-pad caps.
 */
static void
_crop_test_init (crop_test_data_s *crop_test)
{
  GstPad *raw_sink, *info_sink, *raw_src, *info_src;
  GstCaps *caps;

  crop_test->crop = gst_harness_new_with_padnames ("tensor_crop", NULL, "src");
  crop_test->raw = gst_harness_new_with_element (crop_test->crop->element, "raw", NULL);
  crop_test->info = gst_harness_new_with_element (crop_test->crop->element, "info", NULL);
  crop_test->raw_q = gst_harness_new ("queue");
  crop_test->info_q = gst_harness_new ("queue");

  raw_sink = GST_PAD_PEER (crop_test->raw->srcpad);
  info_sink = GST_PAD_PEER (crop_test->info->srcpad);
  raw_src = GST_PAD_PEER (crop_test->raw_q->sinkpad);
  info_src = GST_PAD_PEER (crop_test->info_q->sinkpad);

  gst_pad_unlink (crop_test->raw->srcpad, raw_sink);
  gst_pad_unlink (crop_test->info->srcpad, info_sink);
  gst_pad_unlink (raw_src, crop_test->raw_q->sinkpad);
  gst_pad_unlink (info_src, crop_test->info_q->sinkpad);
  gst_pad_link (raw_src, raw_sink);
  gst_pad_link (info_src, info_sink);

  /* caps for crop info (flex tensor) */
  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  gst_harness_set_src_caps (crop_test->info_q, caps);

  gst_tensor_info_init (&crop_test->raw_info);
  crop_test->raw_format = _NNS_TENSOR_FORMAT_STATIC;
  crop_test->received = 0;
  crop_test->raw_data = NULL;
  crop_test->raw_size = 0;
  crop_test->ts_raw = GST_CLOCK_TIME_NONE;
  crop_test->info_type = _NNS_END;
  crop_test->info_data = NULL;
  crop_test->info_num = 0;
  crop_test->info_size = 0;
  crop_test->ts_info = GST_CLOCK_TIME_NONE;
}

/**
 * @brief Free tensor-crop test data.
 */
static void
_crop_test_free (crop_test_data_s *crop_test)
{
  gst_harness_teardown (crop_test->raw);
  gst_harness_teardown (crop_test->info);
  gst_harness_teardown (crop_test->raw_q);
  gst_harness_teardown (crop_test->info_q);
  gst_harness_teardown (crop_test->crop);

  g_free (crop_test->raw_data);
  g_free (crop_test->info_data);
  gst_tensor_info_free (&crop_test->raw_info);
}

/**
 * @brief Macro to push raw buffer to tensor_crop.
 */
#define _crop_test_push_raw_buffer(ctd, ts)                                           \
  do {                                                                                \
    GstBuffer *rb = gst_buffer_new ();                                                \
    GstMemory *mem;                                                                   \
    mem = gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY, (ctd)->raw_data,          \
        (ctd)->raw_size, 0, (ctd)->raw_size, NULL, NULL);                             \
    if ((ctd)->raw_format == _NNS_TENSOR_FORMAT_FLEXIBLE) {                           \
      GstTensorMetaInfo meta;                                                         \
      gst_tensor_info_convert_to_meta (&(ctd)->raw_info, &meta);                      \
      gst_buffer_append_memory (rb, gst_tensor_meta_info_append_header (&meta, mem)); \
      gst_memory_unref (mem);                                                         \
    } else {                                                                          \
      gst_buffer_append_memory (rb, mem);                                             \
    }                                                                                 \
    if ((ts) != GST_CLOCK_TIME_NONE)                                                  \
      GST_BUFFER_TIMESTAMP (rb) = (ts);                                               \
    EXPECT_EQ (gst_harness_push ((ctd)->raw_q, rb), GST_FLOW_OK);                     \
  } while (0)

/**
 * @brief Macro to push info buffer to tensor_crop.
 */
#define _crop_test_push_info_buffer(ctd, ts)                                        \
  do {                                                                              \
    GstBuffer *ib = gst_buffer_new ();                                              \
    GstMemory *mem;                                                                 \
    GstTensorMetaInfo meta;                                                         \
    gst_tensor_meta_info_init (&meta);                                              \
    meta.type = (ctd)->info_type;                                                   \
    meta.dimension[0] = 4U;                                                         \
    meta.dimension[1] = (ctd)->info_num;                                            \
    meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;                                      \
    mem = gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY, (ctd)->info_data,       \
        (ctd)->info_size, 0, (ctd)->info_size, NULL, NULL);                         \
    gst_buffer_append_memory (ib, gst_tensor_meta_info_append_header (&meta, mem)); \
    gst_memory_unref (mem);                                                         \
    if ((ts) != GST_CLOCK_TIME_NONE)                                                \
      GST_BUFFER_TIMESTAMP (ib) = (ts);                                             \
    EXPECT_EQ (gst_harness_push ((ctd)->info_q, ib), GST_FLOW_OK);                  \
  } while (0)

/**
 * @brief Push raw and info buffer to tensor_crop.
 */
static void
_crop_test_push_buffer (crop_test_data_s *crop_test)
{
  GstTensorsConfig config;

  /* caps for raw data */
  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0] = crop_test->raw_info;
  config.info.format = crop_test->raw_format;
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (crop_test->raw_q, gst_tensors_caps_from_config (&config));

  /* push raw buffer */
  _crop_test_push_raw_buffer (crop_test, crop_test->ts_raw);

  /* push info buffer (default mode region [x, y, w, h] * num) */
  _crop_test_push_info_buffer (crop_test, crop_test->ts_info);

  /* wait for output buffer */
  crop_test->received
      = _harness_wait_for_output_buffer (crop_test->crop, (crop_test->received + 1));
}

/**
 * @brief Internal function to check cropped buffer.
 * raw buffer uint32 [1, 2, ..., 40] dimension 1:10:4:1
 * info buffer uint32 [3, 0, 3, 1] [2, 1, 7, 2]
 */
static void
_crop_test_compare_res1 (crop_test_data_s *crop_test)
{
  GstBuffer *out_buf;
  GstMemory *mem;
  GstMapInfo map;
  GstTensorMetaInfo meta;
  gsize hsize;
  guint i;
  guint *cropped;

  out_buf = gst_harness_pull (crop_test->crop);
  ASSERT_EQ (gst_buffer_n_memory (out_buf), 2U);

  /* 1st cropped data [3, 0, 3, 1] */
  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));

  gst_tensor_meta_info_parse_header (&meta, map.data);
  EXPECT_EQ (meta.type, _NNS_UINT32);
  EXPECT_EQ (meta.dimension[0], 1U);
  EXPECT_EQ (meta.dimension[1], 3U);
  EXPECT_EQ (meta.dimension[2], 1U);

  hsize = gst_tensor_meta_info_get_header_size (&meta);
  cropped = (guint *) (map.data + hsize);
  /* expected [4, 5, 6] */
  EXPECT_EQ (map.size - hsize, sizeof (guint) * 3U);
  EXPECT_EQ (cropped[0], 4U);
  EXPECT_EQ (cropped[1], 5U);
  EXPECT_EQ (cropped[2], 6U);

  gst_memory_unmap (mem, &map);

  /* 2nd cropped data [2, 1, 7, 2] */
  mem = gst_buffer_peek_memory (out_buf, 1);
  ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));

  gst_tensor_meta_info_parse_header (&meta, map.data);
  EXPECT_EQ (meta.dimension[0], 1U);
  EXPECT_EQ (meta.dimension[1], 7U);
  EXPECT_EQ (meta.dimension[2], 2U);

  hsize = gst_tensor_meta_info_get_header_size (&meta);
  cropped = (guint *) (map.data + hsize);
  /* expected [13, 14, ..., 19, 23, 24, ..., 29] */
  EXPECT_EQ (map.size - hsize, sizeof (guint) * 14U);
  for (i = 0; i < 2; i++) {
    EXPECT_EQ (cropped[0 + 7 * i], 3U + (10U * (i + 1)));
    EXPECT_EQ (cropped[1 + 7 * i], 4U + (10U * (i + 1)));
    EXPECT_EQ (cropped[2 + 7 * i], 5U + (10U * (i + 1)));
    EXPECT_EQ (cropped[3 + 7 * i], 6U + (10U * (i + 1)));
    EXPECT_EQ (cropped[4 + 7 * i], 7U + (10U * (i + 1)));
    EXPECT_EQ (cropped[5 + 7 * i], 8U + (10U * (i + 1)));
    EXPECT_EQ (cropped[6 + 7 * i], 9U + (10U * (i + 1)));
  }

  gst_memory_unmap (mem, &map);
  gst_buffer_unref (out_buf);
}

/**
 * @brief Internal function to check cropped buffer.
 * raw buffer uint32 [1, 2, ..., 40] dimension 2:5:4:1
 * info buffer uint32 [2, 0, 3, 1] [1, 1, 5, 2]
 */
static void
_crop_test_compare_res2 (crop_test_data_s *crop_test)
{
  GstBuffer *out_buf;
  GstMemory *mem;
  GstMapInfo map;
  GstTensorMetaInfo meta;
  gsize hsize;
  guint i;
  guint *cropped;

  out_buf = gst_harness_pull (crop_test->crop);
  ASSERT_EQ (gst_buffer_n_memory (out_buf), 2U);

  /* 1st cropped data [2, 0, 3, 1] */
  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));

  gst_tensor_meta_info_parse_header (&meta, map.data);
  EXPECT_EQ (meta.type, _NNS_UINT32);
  EXPECT_EQ (meta.dimension[0], 2U);
  EXPECT_EQ (meta.dimension[1], 3U);
  EXPECT_EQ (meta.dimension[2], 1U);

  hsize = gst_tensor_meta_info_get_header_size (&meta);
  cropped = (guint *) (map.data + hsize);
  /* expected [5, 6, 7, ..., 10] */
  EXPECT_EQ (map.size - hsize, sizeof (guint) * 6U);
  EXPECT_EQ (cropped[0], 5U);
  EXPECT_EQ (cropped[1], 6U);
  EXPECT_EQ (cropped[2], 7U);
  EXPECT_EQ (cropped[3], 8U);
  EXPECT_EQ (cropped[4], 9U);
  EXPECT_EQ (cropped[5], 10U);

  gst_memory_unmap (mem, &map);

  /* 2nd cropped data [1, 1, 5, 2] -> [1, 1, 4, 2] */
  mem = gst_buffer_peek_memory (out_buf, 1);
  ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));

  gst_tensor_meta_info_parse_header (&meta, map.data);
  EXPECT_EQ (meta.dimension[0], 2U);
  EXPECT_EQ (meta.dimension[1], 4U);
  EXPECT_EQ (meta.dimension[2], 2U);

  hsize = gst_tensor_meta_info_get_header_size (&meta);
  cropped = (guint *) (map.data + hsize);
  /* expected [13, 14, ..., 20, 23, 24, ..., 30] */
  EXPECT_EQ (map.size - hsize, sizeof (guint) * 16U);
  for (i = 0; i < 2; i++) {
    EXPECT_EQ (cropped[0 + 8 * i], 3U + (10U * (i + 1)));
    EXPECT_EQ (cropped[1 + 8 * i], 4U + (10U * (i + 1)));
    EXPECT_EQ (cropped[2 + 8 * i], 5U + (10U * (i + 1)));
    EXPECT_EQ (cropped[3 + 8 * i], 6U + (10U * (i + 1)));
    EXPECT_EQ (cropped[4 + 8 * i], 7U + (10U * (i + 1)));
    EXPECT_EQ (cropped[5 + 8 * i], 8U + (10U * (i + 1)));
    EXPECT_EQ (cropped[6 + 8 * i], 9U + (10U * (i + 1)));
    EXPECT_EQ (cropped[7 + 8 * i], 10U + (10U * (i + 1)));
  }

  gst_memory_unmap (mem, &map);
  gst_buffer_unref (out_buf);
}

/**
 * @brief Test for tensor_crop, cropping raw data with crop info.
 */
TEST (testTensorCrop, cropTensor)
{
  crop_test_data_s crop_test;
  guint i;
  guint *_data, *_info;

  _crop_test_init (&crop_test);

  /* prepare test data */
  crop_test.raw_info.type = _NNS_UINT32;

  crop_test.raw_size = sizeof (guint) * 40U;
  crop_test.raw_data = g_malloc0 (crop_test.raw_size);
  _data = (guint *) crop_test.raw_data;

  for (i = 0; i < 40; i++)
    _data[i] = i + 1;

  crop_test.info_type = _NNS_UINT32;
  crop_test.info_size = sizeof (guint) * 8U;
  crop_test.info_num = 2U;
  crop_test.info_data = g_malloc0 (crop_test.info_size);
  _info = (guint *) crop_test.info_data;

  /* crop info (1 ch / [3, 0, 3, 1] [2, 1, 7, 2]) */
  _info[0] = 3U;
  _info[1] = 0U;
  _info[2] = 3U;
  _info[3] = 1U;
  _info[4] = 2U;
  _info[5] = 1U;
  _info[6] = 7U;
  _info[7] = 2U;

  gst_tensor_parse_dimension ("1:10:4:1", crop_test.raw_info.dimension);
  _crop_test_push_buffer (&crop_test);
  EXPECT_EQ (crop_test.received, 1U);

  if (crop_test.received > 0)
    _crop_test_compare_res1 (&crop_test);

  /* crop info (2 ch / [2, 0, 3, 1] [1, 1, 5, 2]) */
  _info[0] = 2U;
  _info[1] = 0U;
  _info[2] = 3U;
  _info[3] = 1U;
  _info[4] = 1U;
  _info[5] = 1U;
  _info[6] = 5U;
  _info[7] = 2U;

  gst_tensor_parse_dimension ("2:5:4:1", crop_test.raw_info.dimension);
  _crop_test_push_buffer (&crop_test);
  EXPECT_EQ (crop_test.received, 2U);

  if (crop_test.received > 1)
    _crop_test_compare_res2 (&crop_test);

  _crop_test_free (&crop_test);
}

/**
 * @brief Test for tensor_crop, invalid property name.
 */
TEST (testTensorCrop, invalidProperty_n)
{
  crop_test_data_s crop_test;
  gboolean value_bool, res_bool;
  gchar *value_str = NULL;

  _crop_test_init (&crop_test);

  g_object_get (crop_test.crop->element, "silent", &value_bool, NULL);
  g_object_set (crop_test.crop->element, "silent", !value_bool, NULL);
  g_object_get (crop_test.crop->element, "silent", &res_bool, NULL);
  EXPECT_EQ (res_bool, !value_bool);

  g_object_set (crop_test.crop->element, "invalid-prop", &value_str, NULL);
  EXPECT_FALSE (value_str != NULL);

  _crop_test_free (&crop_test);
}

/**
 * @brief Test for tensor_crop, seek event is not available.
 */
TEST (testTensorCrop, eventSeek_n)
{
  crop_test_data_s crop_test;
  GstEvent *event;

  _crop_test_init (&crop_test);

  event = gst_event_new_seek (1, GST_FORMAT_TIME, GST_SEEK_FLAG_FLUSH,
      GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_SET, 2 * GST_SECOND);
  EXPECT_FALSE (gst_harness_push_upstream_event (crop_test.crop, event));

  _crop_test_free (&crop_test);
}

/**
 * @brief Test for tensor_crop, push invalid raw buffer.
 */
TEST (testTensorCrop, rawInvalidSize_n)
{
  crop_test_data_s crop_test;

  _crop_test_init (&crop_test);

  crop_test.raw_info.type = _NNS_UINT32;
  gst_tensor_parse_dimension ("20:1:1:1", crop_test.raw_info.dimension);

  crop_test.raw_size = sizeof (guint) * 10U;
  crop_test.raw_data = g_malloc0 (crop_test.raw_size);

  crop_test.info_type = _NNS_UINT16;
  crop_test.info_size = gst_tensor_get_element_size (crop_test.info_type) * 8U;
  crop_test.info_num = 2U;
  crop_test.info_data = g_malloc0 (crop_test.info_size);

  /* raw buffer has invalid size */
  _crop_test_push_buffer (&crop_test);
  EXPECT_EQ (crop_test.received, 0U);

  _crop_test_free (&crop_test);
}

/**
 * @brief Test for tensor_crop, push invalid info buffer.
 */
TEST (testTensorCrop, infoInvalidSize_n)
{
  crop_test_data_s crop_test;

  _crop_test_init (&crop_test);

  crop_test.raw_info.type = _NNS_UINT32;
  gst_tensor_parse_dimension ("10:1:1:1", crop_test.raw_info.dimension);

  crop_test.raw_size = sizeof (guint) * 10U;
  crop_test.raw_data = g_malloc0 (crop_test.raw_size);

  crop_test.info_type = _NNS_INT8;
  crop_test.info_size = gst_tensor_get_element_size (crop_test.info_type) * 7U;
  crop_test.info_num = 2U;
  crop_test.info_data = g_malloc0 (crop_test.info_size);

  /* info buffer has invalid size */
  _crop_test_push_buffer (&crop_test);
  EXPECT_EQ (crop_test.received, 0U);

  _crop_test_free (&crop_test);
}

/**
 * @brief Test for tensor_crop, push delayed raw buffer.
 */
TEST (testTensorCrop, rawDelayed_n)
{
  crop_test_data_s crop_test;
  gint lateness;
  guint i;
  guint *_data;
  guint8 *_info;

  _crop_test_init (&crop_test);

  /* set lateness 300ms */
  g_object_set (crop_test.crop->element, "lateness", 300, NULL);
  g_object_get (crop_test.crop->element, "lateness", &lateness, NULL);
  EXPECT_EQ (lateness, 300);

  crop_test.raw_format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  crop_test.raw_info.type = _NNS_UINT32;
  gst_tensor_parse_dimension ("1:10:4:1", crop_test.raw_info.dimension);

  crop_test.raw_size = sizeof (guint) * 40U;
  crop_test.raw_data = g_malloc0 (crop_test.raw_size);
  _data = (guint *) crop_test.raw_data;

  crop_test.info_type = _NNS_UINT8;
  crop_test.info_size = gst_tensor_get_element_size (crop_test.info_type) * 8U;
  crop_test.info_num = 2U;
  crop_test.info_data = g_malloc0 (crop_test.info_size);
  _info = (guint8 *) crop_test.info_data;

  /* crop info (1 ch / [3, 0, 3, 1] [2, 1, 7, 2]) */
  _info[0] = 3U;
  _info[1] = 0U;
  _info[2] = 3U;
  _info[3] = 1U;
  _info[4] = 2U;
  _info[5] = 1U;
  _info[6] = 7U;
  _info[7] = 2U;

  /* delayed raw buffer */
  crop_test.ts_raw = 10U * GST_MSECOND;
  crop_test.ts_info = 400U * GST_MSECOND;

  /* raw buffer is dropped, no result buffer. */
  _crop_test_push_buffer (&crop_test);
  EXPECT_EQ (crop_test.received, 0U);

  /* fill raw buffer and push valid buffer */
  for (i = 0; i < 40; i++)
    _data[i] = i + 1;

  _crop_test_push_raw_buffer (&crop_test, 300U * GST_MSECOND);

  crop_test.received = _harness_wait_for_output_buffer (crop_test.crop, 1U);
  EXPECT_EQ (crop_test.received, 1U);

  if (crop_test.received > 0)
    _crop_test_compare_res1 (&crop_test);

  _crop_test_free (&crop_test);
}

/**
 * @brief Test for tensor_crop, push delayed info buffer.
 */
TEST (testTensorCrop, infoDelayed_n)
{
  crop_test_data_s crop_test;
  gint lateness;
  guint i;
  guint *_data;
  guint8 *_info;

  _crop_test_init (&crop_test);

  /* set lateness 100ms */
  g_object_set (crop_test.crop->element, "lateness", 100, NULL);
  g_object_get (crop_test.crop->element, "lateness", &lateness, NULL);
  EXPECT_EQ (lateness, 100);

  crop_test.raw_info.type = _NNS_UINT32;
  gst_tensor_parse_dimension ("2:5:4:1", crop_test.raw_info.dimension);

  crop_test.raw_size = sizeof (guint) * 40U;
  crop_test.raw_data = g_malloc0 (crop_test.raw_size);
  _data = (guint *) crop_test.raw_data;

  for (i = 0; i < 40; i++)
    _data[i] = i + 1;

  crop_test.info_type = _NNS_UINT8;
  crop_test.info_size = gst_tensor_get_element_size (crop_test.info_type) * 8U;
  crop_test.info_num = 2U;
  crop_test.info_data = g_malloc0 (crop_test.info_size);
  _info = (guint8 *) crop_test.info_data;

  /* delayed info buffer */
  crop_test.ts_raw = 200U * GST_MSECOND;
  crop_test.ts_info = 10U * GST_MSECOND;

  /* info buffer is dropped, no result buffer. */
  _crop_test_push_buffer (&crop_test);
  EXPECT_EQ (crop_test.received, 0U);

  /* crop info (2 ch / [2, 0, 3, 1] [1, 1, 5, 2]) */
  _info[0] = 2U;
  _info[1] = 0U;
  _info[2] = 3U;
  _info[3] = 1U;
  _info[4] = 1U;
  _info[5] = 1U;
  _info[6] = 5U;
  _info[7] = 2U;

  _crop_test_push_info_buffer (&crop_test, 220U * GST_MSECOND);

  crop_test.received = _harness_wait_for_output_buffer (crop_test.crop, 1U);
  EXPECT_EQ (crop_test.received, 1U);

  if (crop_test.received > 0)
    _crop_test_compare_res2 (&crop_test);

  _crop_test_free (&crop_test);
}

/**
 * @brief Macro to test sparse tensor conversion for each data type.
 */
#define RUN_SPARSE_CONVERT_TEST(ttype, dtype)                                   \
  do {                                                                          \
    failed = false;                                                             \
    const gint sparse_test_data[40] = {                                         \
      0,                                                                        \
      0,                                                                        \
      1,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      1,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      1,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      1,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      1,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      0,                                                                        \
      1,                                                                        \
    };                                                                          \
    GstMemory *sparse, *dense, *origin;                                         \
    GstMapInfo map;                                                             \
    GstTensorInfo info;                                                         \
    GstTensorMetaInfo meta;                                                     \
    guint i;                                                                    \
    gpointer data;                                                              \
    gsize data_size;                                                            \
    gst_tensor_info_init (&info);                                               \
    info.type = ttype;                                                          \
    gst_tensor_parse_dimension ("40", info.dimension);                          \
    gst_tensor_info_convert_to_meta (&info, &meta);                             \
    data_size = gst_tensor_info_get_size (&info);                               \
    data = g_malloc0 (data_size);                                               \
    for (i = 0; i < 40U; i++)                                                   \
      ((dtype *) data)[i] = (dtype) sparse_test_data[i];                        \
    origin = gst_memory_new_wrapped (                                           \
        GST_MEMORY_FLAG_READONLY, data, data_size, 0, data_size, data, g_free); \
    sparse = gst_tensor_sparse_from_dense (&meta, origin);                      \
    EXPECT_TRUE (sparse != NULL);                                               \
    dense = gst_tensor_sparse_to_dense (&meta, sparse);                         \
    EXPECT_TRUE (dense != NULL);                                                \
    ASSERT_TRUE (gst_memory_map (dense, &map, GST_MAP_READ));                   \
    for (i = 0; i < 40U; i++)                                                   \
      if (((dtype *) data)[i] != ((dtype *) map.data)[i])                       \
        failed = true;                                                          \
    gst_memory_unmap (dense, &map);                                             \
    gst_tensor_info_free (&info);                                               \
    gst_memory_unref (sparse);                                                  \
    gst_memory_unref (dense);                                                   \
    gst_memory_unref (origin);                                                  \
  } while (0)

/**
 * @brief Test for tensor_sparse util, sparse tensor for various data type.
 */
TEST (testTensorSparse, utilConvert)
{
  gboolean failed;
  RUN_SPARSE_CONVERT_TEST (_NNS_INT32, int32_t);
  EXPECT_FALSE (failed);
  RUN_SPARSE_CONVERT_TEST (_NNS_UINT32, uint32_t);
  EXPECT_FALSE (failed);
  RUN_SPARSE_CONVERT_TEST (_NNS_INT16, int16_t);
  EXPECT_FALSE (failed);
  RUN_SPARSE_CONVERT_TEST (_NNS_UINT16, uint16_t);
  EXPECT_FALSE (failed);
  RUN_SPARSE_CONVERT_TEST (_NNS_INT8, int8_t);
  EXPECT_FALSE (failed);
  RUN_SPARSE_CONVERT_TEST (_NNS_UINT8, uint8_t);
  EXPECT_FALSE (failed);
  RUN_SPARSE_CONVERT_TEST (_NNS_INT64, int64_t);
  EXPECT_FALSE (failed);
  RUN_SPARSE_CONVERT_TEST (_NNS_UINT64, uint64_t);
  EXPECT_FALSE (failed);
  RUN_SPARSE_CONVERT_TEST (_NNS_FLOAT64, double);
  EXPECT_FALSE (failed);
  RUN_SPARSE_CONVERT_TEST (_NNS_FLOAT32, float);
  EXPECT_FALSE (failed);
}

/**
 * @brief Test for tensor_sparse util, invalid tensor-meta.
 */
TEST (testTensorSparse, utilInvalidMeta_n)
{
  GstTensorMetaInfo meta;
  GstMemory *in, *out;
  guint *data;
  gsize data_size = 20000U;

  /* temporal data, unspecified tensor info. */
  gst_tensor_meta_info_init (&meta);
  data = (guint *) g_malloc0 (data_size);
  in = gst_memory_new_wrapped (
      GST_MEMORY_FLAG_READONLY, data, data_size, 0, data_size, data, g_free);

  out = gst_tensor_sparse_from_dense (&meta, in);
  EXPECT_FALSE (out != NULL);

  out = gst_tensor_sparse_to_dense (&meta, in);
  EXPECT_FALSE (out != NULL);

  gst_memory_unref (in);
}

/**
 * @brief Test for tensor_sparse_enc, invalid property name.
 */
TEST (testTensorSparse, encInvalidProperty_n)
{
  GstHarness *h;
  gboolean value_bool, res_bool;
  gchar *value_str = NULL;

  h = gst_harness_new ("tensor_sparse_enc");

  g_object_get (h->element, "silent", &value_bool, NULL);
  g_object_set (h->element, "silent", !value_bool, NULL);
  g_object_get (h->element, "silent", &res_bool, NULL);
  EXPECT_EQ (res_bool, !value_bool);

  g_object_set (h->element, "invalid-prop", &value_str, NULL);
  EXPECT_FALSE (value_str != NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_sparse_dec, invalid property name.
 */
TEST (testTensorSparse, decInvalidProperty_n)
{
  GstHarness *h;
  gboolean value_bool, res_bool;
  gchar *value_str = NULL;

  h = gst_harness_new ("tensor_sparse_dec");

  g_object_get (h->element, "silent", &value_bool, NULL);
  g_object_set (h->element, "silent", !value_bool, NULL);
  g_object_get (h->element, "silent", &res_bool, NULL);
  EXPECT_EQ (res_bool, !value_bool);

  g_object_set (h->element, "invalid-prop", &value_str, NULL);
  EXPECT_FALSE (value_str != NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Main function for unit test.
 */
int
main (int argc, char **argv)
{
  int ret = -1;
  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  try {
    ret = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return ret;
}
