/**
 * @file	unittest_plugins.cpp
 * @date	7 November 2018
 * @brief	Unit test for nnstreamer plugins. (testcases to check data conversion or buffer transfer)
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs.
 */

#include <string.h>
#include <gtest/gtest.h>
#include <gst/gst.h>
#include <gst/check/gstcheck.h>
#include <gst/check/gsttestclock.h>
#include <gst/check/gstharness.h>
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
#define _print_log(...) if (DBG) g_message (__VA_ARGS__)

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

/**
 * @brief Test for setting/getting properties of tensor_transform
 */
TEST (test_tensor_transform, properties)
{
  const gboolean default_silent = TRUE;
  const gboolean default_accl = DEFAULT_VAL_PROP_ACCELERATION;
  const gint default_mode = 1; /* typecast */
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
    data_s8[i] = (i + 1) * -1;
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
    data_s16[i] = (i + 1) * -1;
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
    data_s32[i] = (i + 1) * -1;
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
    data_f32[i] = (i + 1.) * -1.;
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
    data_f64[i] = (i + 1.) * -1.;
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
 * @brief Test to re-open tf-lite model file in tensor-filter.
 */
TEST (test_tensor_filter, reopen_tflite_01_p)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  gsize in_size, out_size;
  GstTensorConfig config;
  gchar *str_launch_line, *prop_string;

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
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
  EXPECT_TRUE (g_str_equal (prop_string, "tensorflow-lite"));
  g_free (prop_string);

  gst_harness_get (h, "tensor_filter", "model", &prop_string, NULL);
  EXPECT_TRUE (g_str_equal (prop_string, test_model));
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

  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  gchar *test_model;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  ASSERT_TRUE (g_file_test (test_model, G_FILE_TEST_EXISTS));

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);

  prop->fwname = fw_name;
  prop->model_file = test_model;

  ASSERT_TRUE (fw && fw->open && fw->close);

  /* open tf-lite model */
  EXPECT_TRUE (fw->open (prop, &private_data) == 0);

  /* re-open tf-lite model */
  EXPECT_TRUE (fw->open (prop, &private_data) > 0);

  /* close tf-lite model */
  fw->close (prop, &private_data);

  g_free (prop);
  g_free (test_model);
}
#endif /* ENABLE_TENSORFLOW_LITE */

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
