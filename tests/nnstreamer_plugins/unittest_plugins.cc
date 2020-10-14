/**
 * @file	unittest_plugins.cc
 * @date	7 November 2018
 * @brief	Unit test for nnstreamer plugins. (testcases to check data conversion or buffer transfer)
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs.
 */

#include <string.h>
#include <gtest/gtest.h>
#include <gst/gst.h>
#include <gst/check/gstcheck.h>
#include <gst/check/gsttestclock.h>
#include <gst/check/gstharness.h>
#include <glib/gstdio.h>
#include <tensor_common.h>
#include <nnstreamer_plugin_api_filter.h>

#include "../gst/nnstreamer/tensor_transform/tensor_transform.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief Macro for default value of the transform's 'acceleration' property
 */
#ifdef HAVE_ORC
#define DEFAULT_VAL_PROP_ACCELERATION TRUE
#else
#define DEFAULT_VAL_PROP_ACCELERATION FALSE
#endif

/**
 * @brief Macro for debug message.
 */
#define _print_log(...) \
  do { \
    if (DBG) \
      g_message (__VA_ARGS__); \
  } while (0)

#define str(s) #s
#define TEST_TRANSFORM_TYPECAST(name, num_bufs, size, from_t, from_nns_t, to_t, str_to_t, to_nns_t, accel) \
    TEST (test_tensor_transform, name)  \
    { \
      const guint num_buffers = num_bufs; \
      const guint array_size = size; \
      \
      GstHarness *h;  \
      GstBuffer *in_buf, *out_buf;  \
      GstTensorConfig config; \
      GstMemory *mem; \
      GstMapInfo info;  \
      guint i, b; \
      gsize data_in_size, data_out_size;  \
      \
      h = gst_harness_new ("tensor_transform"); \
      \
      g_object_set (h->element, "mode", GTT_TYPECAST, "option", str_to_t, NULL);  \
      g_object_set (h->element, "acceleration", (gboolean) accel, NULL);  \
      /** input tensor info */ \
      config.info.type = from_nns_t; \
      gst_tensor_parse_dimension (str(size), config.info.dimension); \
      config.rate_n = 0; \
      config.rate_d = 1; \
      \
      gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));  \
      data_in_size = gst_tensor_info_get_size (&config.info); \
      \
      config.info.type = to_nns_t; \
      data_out_size = gst_tensor_info_get_size (&config.info); \
      \
      /** push buffers */  \
      for (b = 0; b < num_buffers; b++) { \
        /** set input buffer */ \
        in_buf = gst_harness_create_buffer (h, data_in_size); \
        \
        mem = gst_buffer_peek_memory (in_buf, 0); \
        ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE)); \
        \
        for (i = 0; i < array_size; i++) {  \
          from_t value = (i + 1) * (b + 1);  \
          ((from_t *) info.data)[i] = value; \
        } \
        \
        gst_memory_unmap (mem, &info);  \
        \
        EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);  \
        \
        /** get output buffer */ \
        out_buf = gst_harness_pull (h); \
        \
        ASSERT_TRUE (out_buf != NULL);  \
        ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U); \
        ASSERT_EQ (gst_buffer_get_size (out_buf), data_out_size); \
        \
        mem = gst_buffer_peek_memory (out_buf, 0);  \
        ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));  \
        \
        for (i = 0; i < array_size; i++) {  \
          to_t expected = (i + 1) * (b + 1);  \
          EXPECT_EQ (((to_t *) info.data)[i], expected);  \
        } \
        \
        gst_memory_unmap (mem, &info);  \
        gst_buffer_unref (out_buf); \
      } \
      EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);  \
      gst_harness_teardown (h); \
    }

#define GET_MODEL_PATH(model_name) do { \
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH"); \
  \
  if (root_path == NULL) \
    root_path = ".."; \
  \
  test_model = g_build_filename (root_path, "tests", "test_models", "models", \
    #model_name, NULL); \
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS)); \
  } while (0);

/**
 * @brief Macro for tensor filter auto option test
 */
#define TEST_TENSOR_FILTER_AUTO_OPTION_P(gstpipe, fw_name) do { \
  GstElement *filter; \
  gchar *prop_string; \
  \
  filter = gst_bin_get_by_name (GST_BIN (gstpipe), "tfilter"); \
  EXPECT_NE (filter, nullptr); \
  g_object_get (filter, "framework", &prop_string, NULL); \
  EXPECT_STREQ (prop_string, fw_name); \
  \
  g_free (prop_string); \
  g_free (test_model); \
  gst_object_unref (filter); \
  gst_object_unref (gstpipe); \
} while (0);

/**
 * @brief Macro for check errorneous pipeline
 */
#define TEST_TENSOR_FILTER_AUTO_OPTION_N(gstpipe, fw_name) do { \
  int status = 0; \
  GstStateChangeReturn ret; \
  \
  status = 0; \
  if ( fw_name ) { \
    GstElement *filter; \
    gchar *prop_string; \
    filter = gst_bin_get_by_name (GST_BIN (gstpipe), "tfilter"); \
    EXPECT_NE (filter, nullptr); \
    g_object_get (filter, "framework", &prop_string, NULL); \
    EXPECT_STREQ (prop_string, fw_name); \
    gst_object_unref (filter); \
  } \
  gst_element_set_state (gstpipe, GST_STATE_PLAYING); \
  g_usleep(100000); \
  ret = gst_element_get_state (gstpipe, NULL, NULL, GST_CLOCK_TIME_NONE); \
  EXPECT_TRUE (ret == GST_STATE_CHANGE_FAILURE); \
  \
  gst_object_unref (gstpipe); \
  \
  EXPECT_EQ (status, 0); \
  \
} while (0);

#define wait_for_element_state(element,state) do { \
  GstState cur_state = GST_STATE_VOID_PENDING; \
  GstStateChangeReturn ret; \
  gint counter = 0;\
  ret = gst_element_set_state (element, state); \
  EXPECT_TRUE (ret != GST_STATE_CHANGE_FAILURE); \
  while (cur_state != state && counter < 20) { \
    g_usleep (50000); \
    counter++; \
    ret = gst_element_get_state (element, &cur_state, NULL, 5 * GST_MSECOND); \
    EXPECT_TRUE (ret != GST_STATE_CHANGE_FAILURE); \
  } \
  EXPECT_TRUE (cur_state == state); \
  g_usleep (50000); \
} while (0)

/**
 * @brief Test for setting/getting properties of tensor_transform
 */
TEST (test_tensor_transform, properties_01)
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

  str_launch_line = g_strdup_printf ("tensor_transform mode=%d option=%s",
      default_mode, default_option);
  gst_harness_add_parse (hrnss,  str_launch_line);
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
TEST (test_tensor_transform, properties_02_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");

  /* invalid option (from:to) */
  g_object_set (h->element, "mode", GTT_DIMCHG, "option", "10:11", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (test_tensor_transform, properties_03_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");

  /* invalid option (typecast:type) */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "typecast", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  /* invalid option (typecast:type) */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "typecast:unknown", NULL);
  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (test_tensor_transform, properties_04_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");

  /* invalid option (from:to) */
  g_object_set (h->element, "mode", GTT_TRANSPOSE, "option", "5:2:4:3", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for invalid properties of tensor_transform
 */
TEST (test_tensor_transform, properties_05_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");

  /* invalid option (stand mode) */
  g_object_set (h->element, "mode", GTT_STAND, "option", "invalid", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform typecast (uint8 > uint32)
 */
TEST_TRANSFORM_TYPECAST (typecast_1, 3U, 5U, uint8_t, _NNS_UINT8, uint32_t, "uint32", _NNS_UINT32, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, uint8 > uint32)
 */
TEST_TRANSFORM_TYPECAST (typecast_1_accel, 3U, 5U, uint8_t, _NNS_UINT8, uint32_t, "uint32", _NNS_UINT32, TRUE)

/**
 * @brief Test for tensor_transform typecast (uint32 > float64)
 */
TEST_TRANSFORM_TYPECAST (typecast_2, 3U, 5U, uint32_t, _NNS_UINT32, double, "float64", _NNS_FLOAT64, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, uint32 > float64)
 */
TEST_TRANSFORM_TYPECAST (typecast_2_accel, 3U, 5U, uint32_t, _NNS_UINT32, double, "float64", _NNS_FLOAT64, TRUE)

/**
 * @brief Test for tensor_transform typecast (int32 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_3, 3U, 5U, int32_t, _NNS_INT32, float, "float32", _NNS_FLOAT32, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, int32 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_3_accel, 3U, 5U, int32_t, _NNS_INT32, float, "float32", _NNS_FLOAT32, TRUE)

/**
 * @brief Test for tensor_transform typecast (int8 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_4, 3U, 5U, int8_t, _NNS_INT8, float, "float32", _NNS_FLOAT32, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, int8 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_4_accel, 3U, 5U, int8_t, _NNS_INT8, float, "float32", _NNS_FLOAT32, TRUE)

/**
 * @brief Test for tensor_transform typecast (uint8 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_5, 3U, 5U, uint8_t, _NNS_UINT8, float, "float32", _NNS_FLOAT32, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, uint8 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_5_accel, 3U, 5U, uint8_t, _NNS_UINT8, float, "float32", _NNS_FLOAT32, TRUE)

/**
 * @brief Test for tensor_transform typecast (int16 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_6, 3U, 5U, int16_t, _NNS_INT16, float, "float32", _NNS_FLOAT32, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, int16 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_6_accel, 3U, 5U, int16_t, _NNS_INT16, float, "float32", _NNS_FLOAT32, TRUE)

/**
 * @brief Test for tensor_transform typecast (uint16 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_7, 3U, 5U, uint16_t, _NNS_UINT16, float, "float32", _NNS_FLOAT32, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, uint16 > float32)
 */
TEST_TRANSFORM_TYPECAST (typecast_7_accel, 3U, 5U, uint16_t, _NNS_UINT16, float, "float32", _NNS_FLOAT32, TRUE)

/**
 * @brief Test for tensor_transform typecast (uint64 -> int64)
 */
TEST_TRANSFORM_TYPECAST (typecast_8, 3U, 5U, uint64_t, _NNS_UINT64, int64_t, "int64", _NNS_INT64, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, uint64 -> int64)
 */
TEST_TRANSFORM_TYPECAST (typecast_8_accel, 3U, 5U, uint64_t, _NNS_UINT64, int64_t, "int64", _NNS_INT64, TRUE)

/**
 * @brief Test for tensor_transform typecast (float -> uint32)
 */
TEST_TRANSFORM_TYPECAST (typecast_9, 3U, 5U, float, _NNS_FLOAT32, uint32_t, "uint32", _NNS_UINT32, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, float -> uint32)
 */
TEST_TRANSFORM_TYPECAST (typecast_9_accel, 3U, 5U, float, _NNS_FLOAT32, uint32_t, "uint32", _NNS_UINT32, TRUE)

/**
 * @brief Test for tensor_transform typecast (uint8 -> int8)
 */
TEST_TRANSFORM_TYPECAST (typecast_10, 3U, 5U, uint8_t, _NNS_UINT8, int8_t, "int8", _NNS_INT8, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, uint8 -> int8)
 */
TEST_TRANSFORM_TYPECAST (typecast_10_accel, 3U, 5U, uint8_t, _NNS_UINT8, int8_t, "int8", _NNS_INT8, TRUE)

/**
 * @brief Test for tensor_transform typecast (uint32 -> int16)
 */
TEST_TRANSFORM_TYPECAST (typecast_11, 3U, 5U, uint32_t, _NNS_UINT32, int16_t, "int16", _NNS_INT16, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, uint32 -> int16)
 */
TEST_TRANSFORM_TYPECAST (typecast_11_accel, 3U, 5U, uint32_t, _NNS_UINT32, int16_t, "int16", _NNS_INT16, TRUE)

/**
 * @brief Test for tensor_transform typecast (float -> uint8)
 */
TEST_TRANSFORM_TYPECAST (typecast_12, 3U, 5U, float, _NNS_FLOAT32, uint8_t, "uint8", _NNS_UINT8, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, float -> uint8)
 */
TEST_TRANSFORM_TYPECAST (typecast_12_accel, 3U, 5U, float, _NNS_FLOAT32, uint8_t, "uint8", _NNS_UINT8, TRUE)

/**
 * @brief Test for tensor_transform typecast (double -> uint16)
 */
TEST_TRANSFORM_TYPECAST (typecast_13, 3U, 5U, double, _NNS_FLOAT64, uint16_t, "uint16", _NNS_UINT16, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, double -> uint16)
 */
TEST_TRANSFORM_TYPECAST (typecast_13_accel, 3U, 5U, double, _NNS_FLOAT64, uint16_t, "uint16", _NNS_UINT16, TRUE)

/**
 * @brief Test for tensor_transform typecast (double -> uint64)
 */
TEST_TRANSFORM_TYPECAST (typecast_14, 3U, 5U, double, _NNS_FLOAT64, uint64_t, "uint64", _NNS_UINT64, FALSE)

/**
 * @brief Test for tensor_transform typecast (acceleration, double -> uint64)
 */
TEST_TRANSFORM_TYPECAST (typecast_14_accel, 3U, 5U, double, _NNS_FLOAT64, uint64_t, "uint64", _NNS_UINT64, TRUE)

/**
 * @brief Test for tensor_transform arithmetic (float32, add .5)
 */
TEST (test_tensor_transform, arithmetic_1)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:.5", NULL);
  g_object_set (h->element, "acceleration", (gboolean) FALSE, NULL);

  /* input tensor info */
  config.info.type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("5", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_size = gst_tensor_info_get_size (&config.info);

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
TEST (test_tensor_transform, arithmetic_1_accel)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:.5", NULL);
  g_object_set (h->element, "acceleration", (gboolean) TRUE, NULL);

  /* input tensor info */
  config.info.type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("5", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_size = gst_tensor_info_get_size (&config.info);

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
TEST (test_tensor_transform, arithmetic_2)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "mul:.5", NULL);
  g_object_set (h->element, "acceleration", (gboolean) FALSE, NULL);

  /* input tensor info */
  config.info.type = _NNS_FLOAT64;
  gst_tensor_parse_dimension ("5", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_size = gst_tensor_info_get_size (&config.info);

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
TEST (test_tensor_transform, arithmetic_2_accel)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "mul:.5", NULL);
  g_object_set (h->element, "acceleration", (gboolean) TRUE, NULL);

  /* input tensor info */
  config.info.type = _NNS_FLOAT64;
  gst_tensor_parse_dimension ("5", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_size = gst_tensor_info_get_size (&config.info);

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
TEST (test_tensor_transform, arithmetic_3)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC,
      "option", "typecast:float32,add:.5,mul:0.2", NULL);
  g_object_set (h->element, "acceleration", (gboolean) FALSE, NULL);

  /* input tensor info */
  config.info.type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_in_size = gst_tensor_info_get_size (&config.info);

  config.info.type = _NNS_UINT32;
  data_out_size = gst_tensor_info_get_size (&config.info);

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
TEST (test_tensor_transform, arithmetic_3_accel)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC,
      "option", "typecast:float32,add:.5,mul:0.2", NULL);
  g_object_set (h->element, "acceleration", (gboolean) TRUE, NULL);

  /* input tensor info */
  config.info.type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_in_size = gst_tensor_info_get_size (&config.info);

  config.info.type = _NNS_UINT32;
  data_out_size = gst_tensor_info_get_size (&config.info);

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
TEST (test_tensor_transform, arithmetic_4)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC,
      "option", "typecast:float64,add:0.2,add:0.1,typecast:uint16", NULL);
  g_object_set (h->element, "acceleration", (gboolean) FALSE, NULL);

  /* input tensor info */
  config.info.type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_in_size = gst_tensor_info_get_size (&config.info);

  config.info.type = _NNS_FLOAT64;
  data_out_size = gst_tensor_info_get_size (&config.info);

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
 * @brief Test for tensor_transform arithmetic (acceleration, typecast uint8 > float64, add .2, add .1, final typecast uint16 will be ignored)
 */
TEST (test_tensor_transform, arithmetic_4_accel)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC,
      "option", "typecast:float64,add:0.2,add:0.1,typecast:uint16", NULL);
  g_object_set (h->element, "acceleration", (gboolean) TRUE, NULL);

  /* input tensor info */
  config.info.type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_in_size = gst_tensor_info_get_size (&config.info);

  config.info.type = _NNS_FLOAT64;
  data_out_size = gst_tensor_info_get_size (&config.info);

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
TEST (test_tensor_transform, arithmetic_5)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC,
      "option", "typecast:int32,mul:2,div:2,add:-1", NULL);
  g_object_set (h->element, "acceleration", (gboolean) FALSE, NULL);

  /* input tensor info */
  config.info.type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_in_size = gst_tensor_info_get_size (&config.info);

  config.info.type = _NNS_INT32;
  data_out_size = gst_tensor_info_get_size (&config.info);

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
TEST (test_tensor_transform, arithmetic_5_accel)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, b;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC,
      "option", "typecast:int32,mul:2,div:2,add:-1", NULL);
  g_object_set (h->element, "acceleration", (gboolean) TRUE, NULL);

  /* input tensor info */
  config.info.type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_in_size = gst_tensor_info_get_size (&config.info);

  config.info.type = _NNS_INT32;
  data_out_size = gst_tensor_info_get_size (&config.info);

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
 * @brief Test for tensor_transform arithmetic (changing option string dynamically)
 */
TEST (test_tensor_transform, arithmetic_change_option_string)
{
  const guint array_size = 5;
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i;
  gsize data_size;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:.5", NULL);
  g_object_set (h->element, "acceleration", (gboolean) FALSE, NULL);

  /* input tensor info */
  config.info.type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("5", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_size = gst_tensor_info_get_size (&config.info);
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
 * @brief Test data for tensor_aggregator (2 frames with dimension 3:4:2:2)
 */
const gint aggr_test_frames[2][48] = {
  {
    1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112,
    1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124,
    1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212,
    1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224
  },
  {
    2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112,
    2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124,
    2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212,
    2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224
  }
};

/**
 * @brief Test for tensor aggregator properties
 */
TEST (test_tensor_aggregator, properties)
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
 * @brief Test for tensor_aggregator (concatenate 2 frames with frames-dim 3, out-dimension 3:4:2:4)
 */
TEST (test_tensor_aggregator, aggregate_1)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "frames-out", 2, "frames-dim", 3, NULL);

  /* input tensor info */
  config.info.type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_in_size = gst_tensor_info_get_size (&config.info);

  gst_tensor_parse_dimension ("3:4:2:4", config.info.dimension);
  data_out_size = gst_tensor_info_get_size (&config.info);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_in_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    memcpy (info.data, aggr_test_frames[i], data_in_size);

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);
  }

  /* get output buffer */
  out_buf = gst_harness_pull (h);

  ASSERT_TRUE (out_buf != NULL);
  ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
  ASSERT_EQ (gst_buffer_get_size (out_buf), data_out_size);

  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

  const gint expected[96] = {
    1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112,
    1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124,
    1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212,
    1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224,
    2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112,
    2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124,
    2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212,
    2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224
  };

  for (i = 0; i < 96; i++) {
    EXPECT_EQ (((gint *) info.data)[i], expected[i]);
  }

  gst_memory_unmap (mem, &info);
  gst_buffer_unref (out_buf);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (concatenate 2 frames with frames-dim 2, out-dimension 3:4:4:2)
 */
TEST (test_tensor_aggregator, aggregate_2)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "frames-out", 2, "frames-dim", 2, NULL);

  /* input tensor info */
  config.info.type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_in_size = gst_tensor_info_get_size (&config.info);

  gst_tensor_parse_dimension ("3:4:4:2", config.info.dimension);
  data_out_size = gst_tensor_info_get_size (&config.info);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_in_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    memcpy (info.data, aggr_test_frames[i], data_in_size);

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);
  }

  /* get output buffer */
  out_buf = gst_harness_pull (h);

  ASSERT_TRUE (out_buf != NULL);
  ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
  ASSERT_EQ (gst_buffer_get_size (out_buf), data_out_size);

  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

  const gint expected[96] = {
    1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112,
    1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124,
    2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112,
    2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124,
    1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212,
    1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224,
    2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212,
    2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224
  };

  for (i = 0; i < 96; i++) {
    EXPECT_EQ (((gint *) info.data)[i], expected[i]);
  }

  gst_memory_unmap (mem, &info);
  gst_buffer_unref (out_buf);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (concatenate 2 frames with frames-dim 1, out-dimension 3:8:2:2)
 */
TEST (test_tensor_aggregator, aggregate_3)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "frames-out", 2, "frames-dim", 1, NULL);

  /* input tensor info */
  config.info.type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_in_size = gst_tensor_info_get_size (&config.info);

  gst_tensor_parse_dimension ("3:8:2:2", config.info.dimension);
  data_out_size = gst_tensor_info_get_size (&config.info);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_in_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    memcpy (info.data, aggr_test_frames[i], data_in_size);

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);
  }

  /* get output buffer */
  out_buf = gst_harness_pull (h);

  ASSERT_TRUE (out_buf != NULL);
  ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
  ASSERT_EQ (gst_buffer_get_size (out_buf), data_out_size);

  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

  const gint expected[96] = {
    1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112,
    2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112,
    1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124,
    2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124,
    1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212,
    2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212,
    1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224,
    2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224
  };

  for (i = 0; i < 96; i++) {
    EXPECT_EQ (((gint *) info.data)[i], expected[i]);
  }

  gst_memory_unmap (mem, &info);
  gst_buffer_unref (out_buf);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (concatenate 2 frames with frames-dim 0, out-dimension 6:4:2:2)
 */
TEST (test_tensor_aggregator, aggregate_4)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i;
  gsize data_in_size, data_out_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "frames-out", 2, "frames-dim", 0, NULL);

  /* input tensor info */
  config.info.type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_in_size = gst_tensor_info_get_size (&config.info);

  gst_tensor_parse_dimension ("6:4:2:2", config.info.dimension);
  data_out_size = gst_tensor_info_get_size (&config.info);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_in_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    memcpy (info.data, aggr_test_frames[i], data_in_size);

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);
  }

  /* get output buffer */
  out_buf = gst_harness_pull (h);

  ASSERT_TRUE (out_buf != NULL);
  ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
  ASSERT_EQ (gst_buffer_get_size (out_buf), data_out_size);

  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

  const gint expected[96] = {
    1101, 1102, 1103, 2101, 2102, 2103, 1104, 1105, 1106, 2104, 2105, 2106,
    1107, 1108, 1109, 2107, 2108, 2109, 1110, 1111, 1112, 2110, 2111, 2112,
    1113, 1114, 1115, 2113, 2114, 2115, 1116, 1117, 1118, 2116, 2117, 2118,
    1119, 1120, 1121, 2119, 2120, 2121, 1122, 1123, 1124, 2122, 2123, 2124,
    1201, 1202, 1203, 2201, 2202, 2203, 1204, 1205, 1206, 2204, 2205, 2206,
    1207, 1208, 1209, 2207, 2208, 2209, 1210, 1211, 1212, 2210, 2211, 2212,
    1213, 1214, 1215, 2213, 2214, 2215, 1216, 1217, 1218, 2216, 2217, 2218,
    1219, 1220, 1221, 2219, 2220, 2221, 1222, 1223, 1224, 2222, 2223, 2224
  };

  for (i = 0; i < 96; i++) {
    EXPECT_EQ (((gint *) info.data)[i], expected[i]);
  }

  gst_memory_unmap (mem, &info);
  gst_buffer_unref (out_buf);

  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_aggregator (no-concat, same in-out frames)
 */
TEST (test_tensor_aggregator, aggregate_5)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i, j;
  gsize data_size;

  h = gst_harness_new ("tensor_aggregator");

  g_object_set (h->element, "concat", (gboolean) FALSE, NULL);

  /* in/out tensor info */
  config.info.type = _NNS_INT32;
  gst_tensor_parse_dimension ("3:4:2:2", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));
  data_size = gst_tensor_info_get_size (&config.info);

  /* push buffers */
  for (i = 0; i < 2; i++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));

    memcpy (info.data, aggr_test_frames[i], data_size);

    gst_memory_unmap (mem, &info);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_buffer_n_memory (out_buf), 1U);
    ASSERT_EQ (gst_buffer_get_size (out_buf), data_size);

    mem = gst_buffer_peek_memory (out_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));

    for (j = 0; j < 48; j++) {
      EXPECT_EQ (((gint *) info.data)[j], aggr_test_frames[i][j]);
    }

    gst_memory_unmap (mem, &info);
    gst_buffer_unref (out_buf);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), 2U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (test_tensor_converter, bytes_to_multi_1)
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

  g_object_set (h->element, "input-dim", "3:4:2:2,3:4:2:2",
      "input-type", "int32,int32", NULL);

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
TEST (test_tensor_converter, bytes_to_multi_2)
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
TEST (test_tensor_converter, bytes_to_multi_invalid_dim_01_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "2:2:2:2",
      "input-type", "int32,uint64", NULL);

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
  EXPECT_DEATH (gst_harness_push (h, in_buf), "");

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (test_tensor_converter, bytes_to_multi_invalid_dim_02_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "2:2:2:2,2:0:1",
      "input-type", "int32,float32", NULL);

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
  EXPECT_DEATH (gst_harness_push (h, in_buf), "");

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (test_tensor_converter, bytes_to_multi_invalid_type_01_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-type", "int64",
      "input-dim", "2:2:2:2,2:2:2:2", NULL);

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
  EXPECT_DEATH (gst_harness_push (h, in_buf), "");

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (test_tensor_converter, bytes_to_multi_invalid_type_02_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "2:2:2:2,2:2:1:1",
      "input-type", "int16,invalid", NULL);

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
  EXPECT_DEATH (gst_harness_push (h, in_buf), "");

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (test_tensor_converter, bytes_to_multi_invalid_type_03_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "1:1:1:1,2:1:1:1,3",
      "input-type", "int16,uint16", NULL);

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
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (test_tensor_converter, bytes_to_multi_invalid_size_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "2:2:2:2,1:1:1:1",
      "input-type", "float32,float64", NULL);

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
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_converter (bytes to multi tensors)
 */
TEST (test_tensor_converter, bytes_to_multi_invalid_frames_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  GstCaps *caps;
  gsize data_size;

  h = gst_harness_new ("tensor_converter");

  g_object_set (h->element, "input-dim", "2:2:2:2,1:1:1:1",
      "input-type", "float32,float64", "frames-per-tensor", "2", NULL);

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
  EXPECT_DEATH (gst_harness_push (h, in_buf), "");

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);
  gst_harness_teardown (h);
}

#ifdef HAVE_ORC
#include "transform-orc.h"

/**
 * @brief Test for tensor_transform orc functions (add constant value)
 */
TEST (test_tensor_transform, orc_add)
{
  const guint array_size = 10;
  guint i;

  /* add constant s8 */
  int8_t data_s8[array_size] = { 0, };

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
  uint8_t data_u8[array_size] = { 0, };

  for (i = 0; i < array_size; i++) {
    data_u8[i] = i + 1;
  }

  nns_orc_add_c_u8 (data_u8, 3, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_u8[i], i + 1 + 3);
  }

  /* add constant s16 */
  int16_t data_s16[array_size] = { 0, };

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
  uint16_t data_u16[array_size] = { 0, };

  for (i = 0; i < array_size; i++) {
    data_u16[i] = i + 1;
  }

  nns_orc_add_c_u16 (data_u16, 17, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_u16[i], i + 1 + 17);
  }

  /* add constant s32 */
  int32_t data_s32[array_size] = { 0, };

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
  uint32_t data_u32[array_size] = { 0, };

  for (i = 0; i < array_size; i++) {
    data_u32[i] = i + 1;
  }

  nns_orc_add_c_u32 (data_u32, 33, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_u32[i], i + 1 + 33);
  }

  /* add constant f32 */
  float data_f32[array_size] = { 0, };

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
  double data_f64[array_size] = { 0, };

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
TEST (test_tensor_transform, orc_mul)
{
  const guint array_size = 10;
  guint i;

  /* mul constant s8 */
  int8_t data_s8[array_size] = { 0, };

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
  uint8_t data_u8[array_size] = { 0, };

  for (i = 0; i < array_size; i++) {
    data_u8[i] = i + 1;
  }

  nns_orc_mul_c_u8 (data_u8, 3, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_u8[i], (i + 1) * 3);
  }

  /* mul constant s16 */
  int16_t data_s16[array_size] = { 0, };

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
  uint16_t data_u16[array_size] = { 0, };

  for (i = 0; i < array_size; i++) {
    data_u16[i] = i + 1;
  }

  nns_orc_mul_c_u16 (data_u16, 17, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_u16[i], (i + 1) * 17);
  }

  /* mul constant s32 */
  int32_t data_s32[array_size] = { 0, };

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
  uint32_t data_u32[array_size] = { 0, };

  for (i = 0; i < array_size; i++) {
    data_u32[i] = i + 1;
  }

  nns_orc_mul_c_u32 (data_u32, 33, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (data_u32[i], (i + 1) * 33);
  }

  /* mul constant f32 */
  float data_f32[array_size] = { 0, };

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
  double data_f64[array_size] = { 0, };

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
TEST (test_tensor_transform, orc_div)
{
  const guint array_size = 10;
  guint i;

  /* div constant f32 */
  float data_f32[array_size] = { 0, };

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
  double data_f64[array_size] = { 0, };

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
TEST (test_tensor_transform, orc_conv_s8)
{
  const guint array_size = 10;
  guint i;

  int8_t data_s8[array_size] = { 0, };

  for (i = 0; i < array_size; i++) {
    data_s8[i] = ((int8_t) (i + 1)) * -1;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = { 0, };

  nns_orc_conv_s8_to_s8 (res_s8, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_s8[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = { 0, };

  nns_orc_conv_s8_to_u8 (res_u8, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u8[i], (uint8_t) data_s8[i]);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = { 0, };

  nns_orc_conv_s8_to_s16 (res_s16, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_s8[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = { 0, };

  nns_orc_conv_s8_to_u16 (res_u16, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u16[i], (uint16_t) data_s8[i]);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = { 0, };

  nns_orc_conv_s8_to_s32 (res_s32, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_s8[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = { 0, };

  nns_orc_conv_s8_to_u32 (res_u32, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u32[i], (uint32_t) data_s8[i]);
  }

  /* convert f32 */
  float res_f32[array_size] = { 0, };

  nns_orc_conv_s8_to_f32 (res_f32, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_s8[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = { 0, };

  nns_orc_conv_s8_to_f64 (res_f64, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_s8[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert u8 to other type)
 */
TEST (test_tensor_transform, orc_conv_u8)
{
  const guint array_size = 10;
  guint i;

  uint8_t data_u8[array_size] = { 0, };

  for (i = 0; i < array_size; i++) {
    data_u8[i] = G_MAXUINT8 - i;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = { 0, };

  nns_orc_conv_u8_to_s8 (res_s8, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_u8[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = { 0, };

  nns_orc_conv_u8_to_u8 (res_u8, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u8[i], (uint8_t) data_u8[i]);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = { 0, };

  nns_orc_conv_u8_to_s16 (res_s16, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_u8[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = { 0, };

  nns_orc_conv_u8_to_u16 (res_u16, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u16[i], (uint16_t) data_u8[i]);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = { 0, };

  nns_orc_conv_u8_to_s32 (res_s32, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_u8[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = { 0, };

  nns_orc_conv_u8_to_u32 (res_u32, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u32[i], (uint32_t) data_u8[i]);
  }

  /* convert f32 */
  float res_f32[array_size] = { 0, };

  nns_orc_conv_u8_to_f32 (res_f32, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_u8[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = { 0, };

  nns_orc_conv_u8_to_f64 (res_f64, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_u8[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert s16 to other type)
 */
TEST (test_tensor_transform, orc_conv_s16)
{
  const guint array_size = 10;
  guint i;

  int16_t data_s16[array_size] = { 0, };

  for (i = 0; i < array_size; i++) {
    data_s16[i] = ((int16_t) (i + 1)) * -1;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = { 0, };

  nns_orc_conv_s16_to_s8 (res_s8, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_s16[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = { 0, };

  nns_orc_conv_s16_to_u8 (res_u8, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u8[i], (uint8_t) data_s16[i]);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = { 0, };

  nns_orc_conv_s16_to_s16 (res_s16, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_s16[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = { 0, };

  nns_orc_conv_s16_to_u16 (res_u16, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u16[i], (uint16_t) data_s16[i]);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = { 0, };

  nns_orc_conv_s16_to_s32 (res_s32, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_s16[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = { 0, };

  nns_orc_conv_s16_to_u32 (res_u32, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u32[i], (uint32_t) data_s16[i]);
  }

  /* convert f32 */
  float res_f32[array_size] = { 0, };

  nns_orc_conv_s16_to_f32 (res_f32, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_s16[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = { 0, };

  nns_orc_conv_s16_to_f64 (res_f64, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_s16[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert u16 to other type)
 */
TEST (test_tensor_transform, orc_conv_u16)
{
  const guint array_size = 10;
  guint i;

  uint16_t data_u16[array_size] = { 0, };

  for (i = 0; i < array_size; i++) {
    data_u16[i] = G_MAXUINT16 - i;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = { 0, };

  nns_orc_conv_u16_to_s8 (res_s8, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_u16[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = { 0, };

  nns_orc_conv_u16_to_u8 (res_u8, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u8[i], (uint8_t) data_u16[i]);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = { 0, };

  nns_orc_conv_u16_to_s16 (res_s16, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_u16[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = { 0, };

  nns_orc_conv_u16_to_u16 (res_u16, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u16[i], (uint16_t) data_u16[i]);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = { 0, };

  nns_orc_conv_u16_to_s32 (res_s32, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_u16[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = { 0, };

  nns_orc_conv_u16_to_u32 (res_u32, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u32[i], (uint32_t) data_u16[i]);
  }

  /* convert f32 */
  float res_f32[array_size] = { 0, };

  nns_orc_conv_u16_to_f32 (res_f32, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_u16[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = { 0, };

  nns_orc_conv_u16_to_f64 (res_f64, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_u16[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert s32 to other type)
 */
TEST (test_tensor_transform, orc_conv_s32)
{
  const guint array_size = 10;
  guint i;

  int32_t data_s32[array_size] = { 0, };

  for (i = 0; i < array_size; i++) {
    data_s32[i] = ((int32_t) (i + 1)) * -1;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = { 0, };

  nns_orc_conv_s32_to_s8 (res_s8, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_s32[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = { 0, };

  nns_orc_conv_s32_to_u8 (res_u8, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u8[i], (uint8_t) data_s32[i]);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = { 0, };

  nns_orc_conv_s32_to_s16 (res_s16, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_s32[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = { 0, };

  nns_orc_conv_s32_to_u16 (res_u16, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u16[i], (uint16_t) data_s32[i]);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = { 0, };

  nns_orc_conv_s32_to_s32 (res_s32, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_s32[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = { 0, };

  nns_orc_conv_s32_to_u32 (res_u32, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u32[i], (uint32_t) data_s32[i]);
  }

  /* convert f32 */
  float res_f32[array_size] = { 0, };

  nns_orc_conv_s32_to_f32 (res_f32, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_s32[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = { 0, };

  nns_orc_conv_s32_to_f64 (res_f64, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_s32[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert u32 to other type)
 */
TEST (test_tensor_transform, orc_conv_u32)
{
  const guint array_size = 10;
  guint i;

  uint32_t data_u32[array_size] = { 0, };

  for (i = 0; i < array_size; i++) {
    data_u32[i] = G_MAXUINT32 - i;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = { 0, };

  nns_orc_conv_u32_to_s8 (res_s8, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_u32[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = { 0, };

  nns_orc_conv_u32_to_u8 (res_u8, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u8[i], (uint8_t) data_u32[i]);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = { 0, };

  nns_orc_conv_u32_to_s16 (res_s16, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_u32[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = { 0, };

  nns_orc_conv_u32_to_u16 (res_u16, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u16[i], (uint16_t) data_u32[i]);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = { 0, };

  nns_orc_conv_u32_to_s32 (res_s32, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_u32[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = { 0, };

  nns_orc_conv_u32_to_u32 (res_u32, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_u32[i], (uint32_t) data_u32[i]);
  }

  /* convert f32 */
  float res_f32[array_size] = { 0, };

  nns_orc_conv_u32_to_f32 (res_f32, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) ((int32_t) data_u32[i]));
  }

  /* convert f64 */
  double res_f64[array_size] = { 0, };

  nns_orc_conv_u32_to_f64 (res_f64, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) ((int32_t) data_u32[i]));
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert f32 to other type)
 */
TEST (test_tensor_transform, orc_conv_f32)
{
  const guint array_size = 10;
  guint i;

  float data_f32[array_size] = { 0, };

  for (i = 0; i < array_size; i++) {
    data_f32[i] = (((float) i) + 1.) * -1.;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = { 0, };

  nns_orc_conv_f32_to_s8 (res_s8, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_f32[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = { 0, };

  nns_orc_conv_f32_to_u8 (res_u8, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    int8_t val = (int8_t) data_f32[i];
    EXPECT_EQ (res_u8[i], (uint8_t) val);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = { 0, };

  nns_orc_conv_f32_to_s16 (res_s16, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_f32[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = { 0, };

  nns_orc_conv_f32_to_u16 (res_u16, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    int16_t val = (int16_t) data_f32[i];
    EXPECT_EQ (res_u16[i], (uint16_t) val);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = { 0, };

  nns_orc_conv_f32_to_s32 (res_s32, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_f32[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = { 0, };

  nns_orc_conv_f32_to_u32 (res_u32, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    int32_t val = (int32_t) data_f32[i];
    EXPECT_EQ (res_u32[i], (uint32_t) val);
  }

  /* convert f32 */
  float res_f32[array_size] = { 0, };

  nns_orc_conv_f32_to_f32 (res_f32, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_f32[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = { 0, };

  nns_orc_conv_f32_to_f64 (res_f64, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_f32[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (convert f64 to other type)
 */
TEST (test_tensor_transform, orc_conv_f64)
{
  const guint array_size = 10;
  guint i;

  double data_f64[array_size] = { 0, };

  for (i = 0; i < array_size; i++) {
    data_f64[i] = (((double) i) + 1.) * -1.;
  }

  /* convert s8 */
  int8_t res_s8[array_size] = { 0, };

  nns_orc_conv_f64_to_s8 (res_s8, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s8[i], (int8_t) data_f64[i]);
  }

  /* convert u8 */
  uint8_t res_u8[array_size] = { 0, };

  nns_orc_conv_f64_to_u8 (res_u8, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    int8_t val = (int8_t) data_f64[i];
    EXPECT_EQ (res_u8[i], (uint8_t) val);
  }

  /* convert s16 */
  int16_t res_s16[array_size] = { 0, };

  nns_orc_conv_f64_to_s16 (res_s16, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s16[i], (int16_t) data_f64[i]);
  }

  /* convert u16 */
  uint16_t res_u16[array_size] = { 0, };

  nns_orc_conv_f64_to_u16 (res_u16, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    int16_t val = (int16_t) data_f64[i];
    EXPECT_EQ (res_u16[i], (uint16_t) val);
  }

  /* convert s32 */
  int32_t res_s32[array_size] = { 0, };

  nns_orc_conv_f64_to_s32 (res_s32, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s32[i], (int32_t) data_f64[i]);
  }

  /* convert u32 */
  uint32_t res_u32[array_size] = { 0, };

  nns_orc_conv_f64_to_u32 (res_u32, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    int32_t val = (int32_t) data_f64[i];
    EXPECT_EQ (res_u32[i], (uint32_t) val);
  }

  /* convert f32 */
  float res_f32[array_size] = { 0, };

  nns_orc_conv_f64_to_f32 (res_f32, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_FLOAT_EQ (res_f32[i], (float) data_f64[i]);
  }

  /* convert f64 */
  double res_f64[array_size] = { 0, };

  nns_orc_conv_f64_to_f64 (res_f64, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_DOUBLE_EQ (res_f64[i], (double) data_f64[i]);
  }
}

/**
 * @brief Test for tensor_transform orc functions (performance)
 */
TEST (test_tensor_transform, orc_performance)
{
  const guint array_size = 80000;
  guint i;
  gint64 start_ts, stop_ts, diff_loop, diff_orc;
  uint8_t *data_u8 = (uint8_t *) g_malloc0 (sizeof (uint8_t) * array_size);
  float *data_float = (float *) g_malloc0 (sizeof (float) * array_size);

  ASSERT_TRUE (data_u8 != NULL);
  ASSERT_TRUE (data_float != NULL);

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

  g_free (data_u8);
  g_free (data_float);
}
#endif /* HAVE_ORC */

#ifdef ENABLE_TENSORFLOW_LITE
/**
 * @brief Test to re-open tf-lite model file in tensor-filter.
 */
TEST (test_tensor_filter, reopen_tflite_01_p)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  gsize in_size, out_size;
  GstTensorConfig config;
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

  str_launch_line = g_strdup_printf ("tensor_filter framework=tensorflow-lite model=%s", test_model);
  gst_harness_add_parse (h,  str_launch_line);
  g_free (str_launch_line);

  /* input tensor info */
  config.info.type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:224:224:1", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));

  /* playing state */
  wait_for_element_state (h->element, GST_STATE_PLAYING);

  /* paused state */
  wait_for_element_state (h->element, GST_STATE_PAUSED);

  /* set same model file */
  gst_harness_set (h, "tensor_filter", "framework", "tensorflow-lite", "model", test_model, NULL);

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
TEST (test_tensor_filter, reopen_tflite_02_p)
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
    test_model, NULL,
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
TEST (test_tensor_filter, reload_tflite_set_property)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  gsize in_size, out_size;
  GstTensorConfig config;
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
      "is-updatable=true model=%s", test_model);
  gst_harness_add_parse (h,  str_launch_line);
  g_free (str_launch_line);

  /* input tensor info */
  config.info.type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:224:224:1", config.info.dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensor_caps_from_config (&config));

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
TEST (test_tensor_filter, reload_tflite_model_not_found_n)
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
    test_model, NULL,
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
  ((gchar **)model_files)[0] = test_model; /* remove const for the test */

  /* reload tf-lite model */
  EXPECT_TRUE (fw->reloadModel (prop, &private_data) == 0);

  g_free (test_model);
  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "model_does_not_exist.tflite", NULL);
  ((gchar **)model_files)[0] = test_model; /* remove const for the test */

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
TEST (test_tensor_filter, reload_tflite_model_wrong_dims_n)
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
    test_model, NULL,
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
  ((gchar **)model_files)[0] = test_model; /* remove const for the test */

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
TEST (test_tensor_filter, reload_tflite_same_model_not_found_n)
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
  test_model_renamed = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_renamed.tflite", NULL);

  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    test_model, NULL,
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
TEST (test_tensor_filter, reload_tflite_same_model_wrong_dims_n)
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
  test_model_backup = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_backup.tflite", NULL);
  test_model_renamed = g_build_filename (root_path, "tests", "test_models", "models",
      "add.tflite", NULL);

  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    test_model, NULL,
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
 * @brief Test framework auto detecion option in tensor-filter.
 */
TEST (test_tensor_filter, framework_auto_ext_tflite_01)
{
  gchar *test_model, *str_launch_line;
  GstElement *gstpipe;
  const gchar fw_name[] = "tensorflow-lite";
  GET_MODEL_PATH (mobilenet_v1_1.0_224_quant.tflite)

  str_launch_line = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter name=tfilter framework=auto model=%s ! tensor_sink", test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name)
}

/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details The order of tensor filter options has changed.
 */
TEST (test_tensor_filter, framework_auto_ext_tflite_02)
{
  gchar *test_model, *str_launch_line;
  GstElement *gstpipe;
  const gchar fw_name[] = "tensorflow-lite";
  GET_MODEL_PATH (mobilenet_v1_1.0_224_quant.tflite)

  str_launch_line = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter name=tfilter model=%s framework=auto ! tensor_sink", test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name)
}

/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details Test if options are insensitive to the case
 */
TEST (test_tensor_filter, framework_auto_ext_tflite_03)
{
  gchar *test_model, *str_launch_line;
  GstElement *gstpipe;
  const gchar fw_name[] = "tensorflow-lite";
  GET_MODEL_PATH (mobilenet_v1_1.0_224_quant.tflite)

  str_launch_line = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter name=tfilter model=%s framework=AutO ! tensor_sink", test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name)
}

/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details Negative case when model file does not exist
 */
TEST (test_tensor_filter, framework_auto_ext_tflite_model_not_found_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = NULL;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstElement *gstpipe;

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models", "mirage.tflite", NULL);

  str_launch_line = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=auto model=%s ! tensor_sink", test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name)

  g_free (test_model);
}

/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details Negative case with not supported extension
 */
TEST (test_tensor_filter, framework_auto_ext_tflite_not_supported_ext_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = NULL;
  GstElement *gstpipe;
  GET_MODEL_PATH (mobilenet_v1_1.0_224_quant.invalid)

  str_launch_line = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=auto model=%s ! tensor_sink", test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name)

  g_free (test_model);
}

/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details Negative case when permission of model file is not given.
 */
TEST (test_tensor_filter, framework_auto_ext_tflite_no_permission_n)
{
  int ret;
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = NULL;
  GstElement *gstpipe;

  GET_MODEL_PATH (mobilenet_v1_1.0_224_quant.tflite)

  ret = g_chmod (test_model, 0000);
  EXPECT_TRUE (ret == 0);

  str_launch_line = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=auto model=%s ! tensor_sink", test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name)

  ret = g_chmod (test_model, 0664);
  EXPECT_TRUE (ret == 0);

  g_free (test_model);
}

/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details Negative case with invalid framework name
 */
TEST (test_tensor_filter, framework_auto_ext_tflite_invalid_fw_name_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = NULL;
  GET_MODEL_PATH (mobilenet_v1_1.0_224_quant.tflite)
  GstElement *gstpipe;

  str_launch_line = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=auta model=%s ! tensor_sink", test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name)

  g_free (test_model);
}

/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details Negative case with invalid dimension of tensor filter
 */
TEST (test_tensor_filter, framework_auto_ext_tflite_wrong_dimension_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = "tensorflow-lite";
  GstElement *gstpipe;
  GET_MODEL_PATH (mobilenet_v1_1.0_224_quant.tflite)

  str_launch_line = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter name=tfilter framework=auto model=%s input=784:1 ! tensor_sink", test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name)

  g_free (test_model);
}

/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details Negative case with invalid input type of tensor filter
 */
TEST (test_tensor_filter, framework_auto_ext_tflite_wrong_inputtype_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = "tensorflow-lite";
  GstElement *gstpipe;
  GET_MODEL_PATH (mobilenet_v1_1.0_224_quant.tflite)

  str_launch_line = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter name=tfilter framework=auto model=%s  inputtype=float32 ! tensor_sink", test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name)

  g_free (test_model);
}

#elif ENABLE_NNFW_RUNTIME
/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details Check if nnfw (second priority) is detected automatically
 */
TEST (test_tensor_filter, framework_auto_ext_tflite_nnfw_04)
{
  gchar *test_model, *str_launch_line;
  GstElement *gstpipe;
  const gchar fw_name[] = "nnfw";
  GET_MODEL_PATH (mobilenet_v1_1.0_224_quant.tflite)

  str_launch_line = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter name=tfilter framework=auto model=%s ! tensor_sink", test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name)
}
#endif /* ENABLE_TENSORFLOW_LITE */

#ifdef ENABLE_TENSORFLOW
/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details Check if tensoflow is detected automatically
 */
TEST (test_tensor_filter, framework_auto_ext_pb_01)
{
  gchar *test_model, *str_launch_line, *data_path;
  GstElement *gstpipe;
  const gchar fw_name[] = "tensorflow";
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
    "mnist.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  data_path = g_build_filename (root_path, "tests", "test_models", "data", "9.raw", NULL);
  ASSERT_TRUE (g_file_test (data_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf ("filesrc location=%s ! application/octet-stream ! tensor_converter input-dim=784:1 input-type=uint8 ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter name=tfilter framework=auto model=%s input=784:1 inputtype=float32 inputname=input output=10:1 outputtype=float32 outputname=softmax ! tensor_sink", data_path, test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name)

  g_free (data_path);
}
#else
/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details Negative case whtn tensorflow is not enabled
 */
TEST (test_tensor_filter, framework_auto_ext_pb_tf_disabled_n)
{
  gchar *test_model, *str_launch_line, *data_path;
  const gchar *fw_name = NULL;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstElement *gstpipe;

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
    "mnist.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  data_path = g_build_filename (root_path, "tests", "test_models", "data", "9.raw", NULL);
  ASSERT_TRUE (g_file_test (data_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf ("filesrc location=%s ! application/octet-stream ! tensor_converter input-dim=784:1 input-type=uint8 ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter name=tfilter framework=auto model=%s input=784:1 inputtype=float32 inputname=input output=10:1 outputtype=float32 outputname=softmax ! tensor_sink", data_path, test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name)

  g_free (test_model);
  g_free (data_path);
}
#endif /* ENABLE_TENSORFLOW */

#ifdef ENABLE_CAFFE2
/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details Check if caffe2 is detected automatically
 */
TEST (test_tensor_filter, framework_auto_ext_pb_03)
{
  gchar *test_model, *str_launch_line, *test_model_2, *data_path;
  GstElement *gstpipe;
  const gchar fw_name[] = "caffe2";
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
    "caffe2_init_net.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  test_model_2 = g_build_filename (root_path, "tests", "test_models", "models", "caffe2_predict_net.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model_2, G_FILE_TEST_EXISTS));
  data_path = g_build_filename (root_path, "tests", "test_models", "data", "5", NULL);
  ASSERT_TRUE (g_file_test (data_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf ("filesrc location=%s blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=32:32:3:1 input-type=float32 ! tensor_filter name=tfilter framework=caffe2 model=%s,%s inputname=data input=32:32:3:1 inputtype=float32 output=10:1 outputtype=float32 outputname=softmax ! fakesink", data_path, test_model, test_model_2);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name)

  g_free (test_model_2);
  g_free (data_path);
}

#else
/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details Check if caffe2 is not enabled
 */
TEST (test_tensor_filter, framework_auto_ext_pb_caffe2_disabled_n)
{
  gchar *test_model, *str_launch_line, *test_model_2, *data_path;
  const gchar *fw_name = NULL;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstElement *gstpipe;

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
    "caffe2_init_net.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  test_model_2 = g_build_filename (root_path, "tests", "test_models", "models", "caffe2_predict_net.pb", NULL);
  ASSERT_TRUE (g_file_test (test_model_2, G_FILE_TEST_EXISTS));
  data_path = g_build_filename (root_path, "tests", "test_models", "data", "5", NULL);
  ASSERT_TRUE (g_file_test (data_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf ("filesrc location=%s blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=32:32:3:1 input-type=float32 ! tensor_filter name=tfilter framework=caffe2 model=%s,%s inputname=data input=32:32:3:1 inputtype=float32 output=10:1 outputtype=float32 outputname=softmax ! fakesink", data_path, test_model, test_model_2);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name)

  g_free (test_model);
  g_free (test_model_2);
  g_free (data_path);
}
#endif /* ENABLE_CAFFE2 */

#ifdef ENABLE_PYTORCH
/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details Check if pytorch is detected automatically
 */
TEST (test_tensor_filter, framework_auto_ext_pt_01)
{
  gchar *test_model, *str_launch_line, *image_path;
  GstElement *gstpipe;
  const gchar fw_name[] = "pytorch";
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
    "pytorch_lenet5.pt", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  image_path = g_build_filename (root_path, "tests", "test_models", "data", "9.png", NULL);
  ASSERT_TRUE (g_file_test (image_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf ("filesrc location=%s ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter name=tfilter framework=auto model=%s input=1:28:28:1 inputtype=uint8 output=10:1:1:1 outputtype=uint8 ! tensor_sink", image_path, test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  EXPECT_TRUE (gstpipe != nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_P (gstpipe, fw_name)

  g_free (image_path);
}

#else
/**
 * @brief Test framework auto detecion option in tensor-filter.
 * @details Check if pytorch is not enabled
 */
TEST (test_tensor_filter, framework_auto_ext_pt_pytorch_disabled_n)
{
  gchar *test_model, *str_launch_line;
  const gchar *fw_name = NULL;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstElement *gstpipe;

  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
    "pytorch_lenet5.pt", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));
  gchar *image_path = g_build_filename (root_path, "tests", "test_models", "data", "9.png", NULL);
  ASSERT_TRUE (g_file_test (image_path, G_FILE_TEST_EXISTS));

  str_launch_line = g_strdup_printf ("filesrc location=%s ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter framework=auto model=%s input=1:28:28:1 inputtype=uint8 output=10:1:1:1 outputtype=uint8 ! tensor_sink", image_path, test_model);
  gstpipe = gst_parse_launch (str_launch_line, NULL);
  g_free (str_launch_line);
  ASSERT_NE (gstpipe, nullptr);
  TEST_TENSOR_FILTER_AUTO_OPTION_N (gstpipe, fw_name)

  g_free (image_path);
  g_free (test_model);
}
#endif /* ENABLE_PYTORCH */

/**
 * @brief Test for inputranks and outputranks property of the tensor_filter
 * @details Given dimension string, check its rank value.
 */
TEST (test_tensor_filter, property_rank_01_p)
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
      output=1001:1:1:1 outputtype=uint8 ", test_model);
  gst_harness_add_parse (hrnss,  str_launch_line);
  g_free (str_launch_line);

  filter = gst_harness_find_element (hrnss, "tensor_filter");
  ASSERT_TRUE (filter != NULL);

  gchar * input_dim;
  g_object_get (filter, "input", &input_dim, NULL);
  EXPECT_STREQ (input_dim, "3:224:224:1");
  g_free (input_dim);

  /* Rank should be 3 since dimension string of the output is explicitly '3:224:224'. */
  gchar * input_ranks;
  g_object_get (filter, "inputranks", &input_ranks, NULL);
  EXPECT_STREQ (input_ranks, "3");
  g_free (input_ranks);

  gchar * output_dim;
  g_object_get (filter, "output", &output_dim, NULL);
  EXPECT_STREQ (output_dim, "1001:1:1:1");
  g_free (output_dim);

  /* Rank should be 4 since dimension string of the output is explicitly '1000:1:1:1'. */
  gchar * output_ranks;
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
TEST (test_tensor_filter, property_rank_02_p)
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
  gst_harness_add_parse (hrnss,  str_launch_line);
  g_free (str_launch_line);

  filter = gst_harness_find_element (hrnss, "tensor_filter");
  ASSERT_TRUE (filter != NULL);

  gchar * input_dim;
  g_object_get (filter, "input", &input_dim, NULL);
  EXPECT_STREQ (input_dim, "3:224:224:1");
  g_free (input_dim);

  gchar * input_ranks;
  g_object_get (filter, "inputranks", &input_ranks, NULL);
  EXPECT_STREQ (input_ranks, "3");
  g_free (input_ranks);

  gchar * output_dim;
  g_object_get (filter, "output", &output_dim, NULL);
  EXPECT_STREQ (output_dim, "1001:1:1:1");
  g_free (output_dim);

  /* Rank should be 1 since dimension string is not given. */
  gchar * output_ranks;
  g_object_get (filter, "outputranks", &output_ranks, NULL);
  EXPECT_STREQ (output_ranks, "1");
  g_free (output_ranks);

  g_object_unref (filter);
  gst_harness_teardown (hrnss);
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
