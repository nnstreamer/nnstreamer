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
#include <nnstreamer_conf.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_plugin_api_converter.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_subplugin.h>
#include <nnstreamer_util.h>
#include <string.h>
#include <tensor_common.h>
#include <tensor_decoder_custom.h>
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
 * @brief Log function collecting messages of the tensor_transform category
 */
static void
_collect_transform_log (GstDebugCategory *category, GstDebugLevel level,
    const gchar *file, const gchar *function, gint line, GObject *object,
    GstDebugMessage *message, gpointer user_data)
{
  GString *log = (GString *) user_data;

  UNUSED (level);
  UNUSED (file);
  UNUSED (function);
  UNUSED (line);
  UNUSED (object);

  if (g_strcmp0 (gst_debug_category_get_name (category), "tensor_transform") == 0) {
    g_string_append (log, gst_debug_message_get (message));
    g_string_append (log, "\n");
  }
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
 * @brief Test for setting the 'apply' property of tensor_transform
 */
TEST (testTensorTransform, applyProperty)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");

  g_object_get (h->element, "apply", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  g_object_set (h->element, "apply", "1,2", NULL);
  g_object_get (h->element, "apply", &str, NULL);
  EXPECT_STREQ (str, "1,2");
  g_free (str);

  /* setting the property again replaces the old list */
  g_object_set (h->element, "apply", "3", NULL);
  g_object_get (h->element, "apply", &str, NULL);
  EXPECT_STREQ (str, "3");
  g_free (str);

  gst_harness_teardown (h);
}

/**
 * @brief Test for setting the 'apply' property of tensor_transform with invalid value
 */
TEST (testTensorTransform, applyPropertyInvalid_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "apply", "1,invalid,3", NULL);
  g_object_get (h->element, "apply", &str, NULL);
  EXPECT_STREQ (str, "1,3");
  g_free (str);

  g_object_set (h->element, "apply", "invalid", NULL);
  g_object_get (h->element, "apply", &str, NULL);
  EXPECT_STREQ (str, "");
  g_free (str);

  gst_harness_teardown (h);
}

/**
 * @brief Test for changing the 'apply' property between the buffers
 */
TEST (testTensorTransform, applyChangedWhileStreaming)
{
  const guint num_tensors = 2U;
  const guint array_size = 64U;
  const gchar *apply_values[] = { "1", "0", "invalid" };
  /* the operator is applied to the tensor selected by each value above */
  const gboolean applied[3][2] = { { FALSE, TRUE }, { TRUE, FALSE }, { TRUE, TRUE } };

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstTensorInfo *_info;
  GstMemory *mem;
  GstMapInfo map;
  guint i, j, p;
  gsize dsize;
  float *_data;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:1", NULL);

  gst_tensors_config_init (&config);
  config.info.num_tensors = num_tensors;
  config.rate_n = 0;
  config.rate_d = 1;

  for (i = 0; i < num_tensors; i++) {
    _info = gst_tensors_info_get_nth_info (&config.info, i);
    _info->type = _NNS_FLOAT32;
    gst_tensor_parse_dimension ("64", _info->dimension);
  }

  dsize = gst_tensors_info_get_size (&config.info, 0);
  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  for (p = 0; p < G_N_ELEMENTS (apply_values); p++) {
    g_object_set (h->element, "apply", apply_values[p], NULL);

    /* set input buffer */
    in_buf = gst_buffer_new ();

    for (i = 0; i < num_tensors; i++) {
      mem = gst_allocator_alloc (NULL, dsize, NULL);
      ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));

      _data = (float *) map.data;
      for (j = 0; j < array_size; j++)
        _data[j] = (float) (i * 100 + j);

      gst_memory_unmap (mem, &map);
      ASSERT_TRUE (gst_tensor_buffer_append_memory (
          in_buf, mem, gst_tensors_info_get_nth_info (&config.info, i)));
    }

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_tensor_buffer_get_count (out_buf), num_tensors);

    for (i = 0; i < num_tensors; i++) {
      float diff = applied[p][i] ? 1.0f : 0.0f;

      mem = gst_tensor_buffer_get_nth_memory (out_buf, i);
      ASSERT_TRUE (mem != NULL);
      ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
      ASSERT_EQ (map.size, dsize);

      _data = (float *) map.data;
      for (j = 0; j < array_size; j++)
        EXPECT_FLOAT_EQ (_data[j], (float) (i * 100 + j) + diff);

      gst_memory_unmap (mem, &map);
      gst_memory_unref (mem);
    }

    gst_buffer_unref (out_buf);
  }

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
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
 * @brief Test set_caps failure when mode/option is never configured (#4103)
 */
TEST (testTensorTransform, setCapsNotConfigured_n)
{
  GstHarness *h;
  GstTensorsConfig config;
  GstBuffer *in_buf;
  GString *log;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  log = g_string_new (NULL);
  gst_debug_add_log_function (_collect_transform_log, log, NULL);
  gst_debug_set_threshold_for_name ("tensor_transform", GST_LEVEL_WARNING);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  in_buf = gst_harness_create_buffer (h, 5);
  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_NOT_NEGOTIATED);

  gst_debug_remove_log_function (_collect_transform_log);
  gst_debug_unset_threshold_for_name ("tensor_transform");

#ifndef GST_DISABLE_GST_DEBUG
  EXPECT_TRUE (strstr (log->str, "Transform is not configured") != NULL);
  EXPECT_TRUE (strstr (log->str, "mode=unknown") != NULL);
#endif

  gst_harness_teardown (h);
  g_string_free (log, TRUE);
}

/**
 * @brief Test set_caps failure when mode is set without option (#4103).
 *        Before the fix, this case passed set_caps and crashed with a
 *        CRITICAL assertion on the first buffer.
 */
TEST (testTensorTransform, setCapsModeWithoutOption_n)
{
  GstHarness *h;
  GstTensorsConfig config;
  GstBuffer *in_buf;
  GString *log;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_TYPECAST, NULL);

  log = g_string_new (NULL);
  gst_debug_add_log_function (_collect_transform_log, log, NULL);
  gst_debug_set_threshold_for_name ("tensor_transform", GST_LEVEL_WARNING);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  in_buf = gst_harness_create_buffer (h, 5);
  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_NOT_NEGOTIATED);

  gst_debug_remove_log_function (_collect_transform_log);
  gst_debug_unset_threshold_for_name ("tensor_transform");

#ifndef GST_DISABLE_GST_DEBUG
  EXPECT_TRUE (strstr (log->str, "Transform is not configured") != NULL);
  EXPECT_TRUE (strstr (log->str, "mode=typecast") != NULL);
#endif

  gst_harness_teardown (h);
  g_string_free (log, TRUE);
}

/**
 * @brief Test set_caps failure when the given option fails to parse (#4103)
 */
TEST (testTensorTransform, setCapsInvalidOption_n)
{
  GstHarness *h;
  GstTensorsConfig config;
  GstBuffer *in_buf;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* "casttype" is a typo of "typecast"; the option fails to parse */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option",
      "casttype:uint64,mul:65535", NULL);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  in_buf = gst_harness_create_buffer (h, 5);
  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_NOT_NEGOTIATED);

  gst_harness_teardown (h);
}

/**
 * @brief Test transform_caps warning when downstream caps are incompatible
 *        with the configured mode/option (#4103)
 */
TEST (testTensorTransform, transformCapsIncompatibleDownstream_n)
{
  GstHarness *h;
  GstTensorsConfig config;
  GstBuffer *in_buf;
  GString *log;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_TYPECAST, "option", "float32", NULL);

  log = g_string_new (NULL);
  gst_debug_add_log_function (_collect_transform_log, log, NULL);
  gst_debug_set_threshold_for_name ("tensor_transform", GST_LEVEL_WARNING);

  gst_harness_set_sink_caps_str (h, "other/tensors,format=static,num_tensors=1,types=uint8,"
                                    "dimensions=5:1:1:1,framerate=0/1");

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  in_buf = gst_harness_create_buffer (h, 5);
  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_NOT_NEGOTIATED);

  gst_debug_remove_log_function (_collect_transform_log);
  gst_debug_unset_threshold_for_name ("tensor_transform");

#ifndef GST_DISABLE_GST_DEBUG
  EXPECT_TRUE (strstr (log->str, "downstream element cannot accept") != NULL);
  EXPECT_TRUE (strstr (log->str, "option=float32") != NULL);
#endif

  gst_harness_teardown (h);
  g_string_free (log, TRUE);
}

/**
 * @brief Positive control: configuring mode/option after the harness is
 *        created and linked should still negotiate and transform (#4103)
 */
TEST (testTensorTransform, setCapsConfiguredAfterLink)
{
  GstHarness *h;
  GstTensorsConfig config;
  GstBuffer *in_buf, *out_buf;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_TYPECAST, "option", "float32", NULL);

  gst_harness_set_sink_caps_str (h, "other/tensors,format=static,num_tensors=1,types=float32,"
                                    "dimensions=5:1:1:1,framerate=0/1");

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  in_buf = gst_harness_create_buffer (h, 5);
  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  out_buf = gst_harness_pull (h);
  ASSERT_TRUE (out_buf != NULL);
  EXPECT_EQ (gst_buffer_get_size (out_buf), 5 * sizeof (float));
  gst_buffer_unref (out_buf);

  gst_harness_teardown (h);
}

/**
 * @brief Caps negotiation through a chain of tensor_transform and
 *        tensor_filter, both of which query their peers from transform_caps.
 *        Guards against caps-query recursion between the two. The
 *        tensor_transform debug threshold is raised to WARNING because the
 *        downstream peer query in transform_caps is now skipped unless that
 *        threshold is at least WARNING, so raising it is required to
 *        actually exercise the peer-query path (#4103).
 */
TEST (testTensorTransform, negotiationChainWithFilter)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *model_file, *str_pipeline, *lib_name;
  GstElement *pipeline;
  GstBus *bus;
  GstMessage *msg;

  if (root_path == NULL)
    root_path = "..";

  lib_name = g_strdup_printf ("libnnstreamer_customfilter_passthrough_variable%s",
      NNSTREAMER_SO_FILE_EXTENSION);
  model_file = g_build_filename (
      root_path, "build", "tests", "nnstreamer_example", lib_name, NULL);
  g_free (lib_name);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  gst_debug_set_threshold_for_name ("tensor_transform", GST_LEVEL_WARNING);

  str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=3 ! video/x-raw,format=RGB,width=8,height=8,framerate=30/1 ! "
      "tensor_converter ! tensor_transform mode=typecast option=float32 ! "
      "tensor_filter framework=custom model=%s ! "
      "tensor_transform mode=arithmetic option=add:1.0 ! "
      "tensor_filter framework=custom model=%s ! "
      "tensor_transform mode=typecast option=uint8 ! "
      "other/tensors,format=static,num_tensors=1,types=uint8 ! fakesink",
      model_file, model_file);
  pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  g_free (model_file);
  ASSERT_TRUE (pipeline != nullptr);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  bus = gst_element_get_bus (pipeline);
  msg = gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  ASSERT_TRUE (msg != nullptr);
  EXPECT_EQ (GST_MESSAGE_TYPE (msg), GST_MESSAGE_EOS);
  gst_message_unref (msg);
  gst_object_unref (bus);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_debug_unset_threshold_for_name ("tensor_transform");
  gst_object_unref (pipeline);
}

/**
 * @brief Test transform_caps/set_caps when mode is configured but has no
 *        option; downstream caps are the real, untransformed caps. Before
 *        the fix, transform_caps advertised padding's default dimension
 *        change and negotiation failed before set_caps ever ran, so the
 *        "not configured" message never appeared (#4103).
 */
TEST (testTensorTransform, setCapsUnconfiguredPaddingWithDownstreamCaps_n)
{
  GstHarness *h;
  GstTensorsConfig config;
  GstBuffer *in_buf;
  GString *log;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* Mode is set, but no option is given: filter->loaded stays FALSE. */
  g_object_set (h->element, "mode", GTT_PADDING, NULL);

  gst_harness_set_sink_caps_str (h, "other/tensors,format=static,num_tensors=1,types=uint8,"
                                    "dimensions=5:1:1:1,framerate=0/1");

  log = g_string_new (NULL);
  gst_debug_add_log_function (_collect_transform_log, log, NULL);
  gst_debug_set_threshold_for_name ("tensor_transform", GST_LEVEL_WARNING);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  in_buf = gst_harness_create_buffer (h, 5);
  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_NOT_NEGOTIATED);

  gst_debug_remove_log_function (_collect_transform_log);
  gst_debug_unset_threshold_for_name ("tensor_transform");

#ifndef GST_DISABLE_GST_DEBUG
  EXPECT_TRUE (strstr (log->str, "Transform is not configured") != NULL);
  EXPECT_TRUE (strstr (log->str, "mode=padding") != NULL);
#endif

  gst_harness_teardown (h);
  g_string_free (log, TRUE);
}

/**
 * @brief Test set_caps failure after a mode change makes the previously
 *        parsed option invalid, resetting filter->loaded to FALSE (#4103).
 */
TEST (testTensorTransform, setCapsAfterModeChangeUnparsable_n)
{
  GstHarness *h;
  GstTensorsConfig config;
  GstBuffer *in_buf;
  GString *log;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* "float32" is not a valid dimchg option: the element is unconfigured again */
  g_object_set (h->element, "mode", GTT_TYPECAST, "option", "float32", NULL);
  g_object_set (h->element, "mode", GTT_DIMCHG, NULL);

  log = g_string_new (NULL);
  gst_debug_add_log_function (_collect_transform_log, log, NULL);
  gst_debug_set_threshold_for_name ("tensor_transform", GST_LEVEL_WARNING);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  in_buf = gst_harness_create_buffer (h, 5);
  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_NOT_NEGOTIATED);

  gst_debug_remove_log_function (_collect_transform_log);
  gst_debug_unset_threshold_for_name ("tensor_transform");

#ifndef GST_DISABLE_GST_DEBUG
  EXPECT_TRUE (strstr (log->str, "Transform is not configured") != NULL);
  EXPECT_TRUE (strstr (log->str, "mode=dimchg") != NULL);
#endif

  gst_harness_teardown (h);
  g_string_free (log, TRUE);
}

/**
 * @brief Test transform_caps warning when a caps query's filter cannot be
 *        satisfied by the transformed caps, for both pad directions (#4103).
 */
TEST (testTensorTransform, transformCapsFilterIntersectionEmpty_n)
{
  GstHarness *h;
  GstTensorsConfig config;
  GString *log;
  GstPad *pad;
  GstCaps *filter, *res;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_TYPECAST, "option", "float32", NULL);

  gst_harness_set_sink_caps_str (h, "other/tensors,format=static,num_tensors=1,types=float32,"
                                    "dimensions=5:1:1:1,framerate=0/1");

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  log = g_string_new (NULL);
  gst_debug_add_log_function (_collect_transform_log, log, NULL);
  gst_debug_set_threshold_for_name ("tensor_transform", GST_LEVEL_WARNING);

  pad = gst_element_get_static_pad (h->element, "src");
  filter = gst_caps_from_string ("other/tensors,format=static,num_tensors=1,"
                                 "types=float16,dimensions=5:1:1:1");
  res = gst_pad_query_caps (pad, filter);
  EXPECT_TRUE (gst_caps_is_empty (res));
  gst_caps_unref (res);
  gst_caps_unref (filter);
  gst_object_unref (pad);

  pad = gst_element_get_static_pad (h->element, "sink");
  filter = gst_caps_from_string ("other/tensors,format=static,num_tensors=1,"
                                 "types=uint8,dimensions=7:1:1:1");
  res = gst_pad_query_caps (pad, filter);
  EXPECT_TRUE (gst_caps_is_empty (res));
  gst_caps_unref (res);
  gst_caps_unref (filter);
  gst_object_unref (pad);

  gst_debug_remove_log_function (_collect_transform_log);
  gst_debug_unset_threshold_for_name ("tensor_transform");

#ifndef GST_DISABLE_GST_DEBUG
  EXPECT_TRUE (strstr (log->str, "cannot produce src caps") != NULL);
  EXPECT_TRUE (strstr (log->str, "cannot produce sink caps") != NULL);
#endif

  gst_harness_teardown (h);
  g_string_free (log, TRUE);
}

/**
 * @brief Positive control: elements are linked before tensor_transform's
 *        mode/option are set, matching the app pattern from issue #4102.
 *        Linking must succeed while the element is still unconfigured, and
 *        the pipeline must still negotiate and transform once mode/option
 *        are set afterward (#4103).
 */
TEST (testTensorTransform, linkThenConfigure)
{
  GstElement *pipeline, *src, *capsfilter1, *converter, *transform;
  GstElement *capsfilter2, *sink;
  GstCaps *caps;
  GstBus *bus;
  GstMessage *msg;

  pipeline = gst_pipeline_new (NULL);
  ASSERT_TRUE (pipeline != nullptr);

  src = gst_element_factory_make ("videotestsrc", NULL);
  capsfilter1 = gst_element_factory_make ("capsfilter", NULL);
  converter = gst_element_factory_make ("tensor_converter", NULL);
  transform = gst_element_factory_make ("tensor_transform", NULL);
  capsfilter2 = gst_element_factory_make ("capsfilter", NULL);
  sink = gst_element_factory_make ("fakesink", NULL);
  ASSERT_TRUE (src != nullptr);
  ASSERT_TRUE (capsfilter1 != nullptr);
  ASSERT_TRUE (converter != nullptr);
  ASSERT_TRUE (transform != nullptr);
  ASSERT_TRUE (capsfilter2 != nullptr);
  ASSERT_TRUE (sink != nullptr);

  g_object_set (src, "num-buffers", 3, NULL);

  caps = gst_caps_from_string ("video/x-raw,format=RGB,width=8,height=8,framerate=30/1");
  g_object_set (capsfilter1, "caps", caps, NULL);
  gst_caps_unref (caps);

  caps = gst_caps_from_string ("other/tensors,format=static,num_tensors=1,"
                               "types=float32,dimensions=3:8:8:1");
  g_object_set (capsfilter2, "caps", caps, NULL);
  gst_caps_unref (caps);

  gst_bin_add_many (GST_BIN (pipeline), src, capsfilter1, converter, transform,
      capsfilter2, sink, NULL);

  /* Linking must succeed while tensor_transform is still unconfigured. */
  ASSERT_TRUE (gst_element_link_many (
      src, capsfilter1, converter, transform, capsfilter2, sink, NULL));

  g_object_set (transform, "mode", GTT_TYPECAST, "option", "float32", NULL);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  bus = gst_element_get_bus (pipeline);
  msg = gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  ASSERT_TRUE (msg != nullptr);
  EXPECT_EQ (GST_MESSAGE_TYPE (msg), GST_MESSAGE_EOS);
  gst_message_unref (msg);
  gst_object_unref (bus);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (pipeline);
}

/**
 * @brief Positive control: mode/option can be changed while the pipeline is
 *        streaming. Re-setting the same mode must keep the element
 *        configured, and an invalid option is rejected while the previously
 *        parsed option is kept in effect (#4103).
 */
TEST (testTensorTransform, optionChangeWhileStreaming)
{
  GstHarness *h;
  GstTensorsConfig config;
  GstBuffer *in_buf, *out_buf;
  GstMemory *mem;
  GstMapInfo info;
  guint i;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:1", NULL);
  g_object_set (h->element, "acceleration", (gboolean) FALSE, NULL);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  /* First buffer: 0..4 -> 1..5 with option "add:1". */
  in_buf = gst_harness_create_buffer (h, 5);
  mem = gst_buffer_peek_memory (in_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));
  for (i = 0; i < 5; i++)
    info.data[i] = i;
  gst_memory_unmap (mem, &info);

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);
  out_buf = gst_harness_pull (h);
  ASSERT_TRUE (out_buf != NULL);
  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));
  for (i = 0; i < 5; i++)
    EXPECT_EQ (info.data[i], i + 1);
  gst_memory_unmap (mem, &info);
  gst_buffer_unref (out_buf);

  /* Change option to "add:2" and re-set the same mode: stays configured. */
  g_object_set (h->element, "option", "add:2", NULL);
  g_object_set (h->element, "mode", GTT_ARITHMETIC, NULL);

  in_buf = gst_harness_create_buffer (h, 5);
  mem = gst_buffer_peek_memory (in_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));
  for (i = 0; i < 5; i++)
    info.data[i] = i;
  gst_memory_unmap (mem, &info);

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);
  out_buf = gst_harness_pull (h);
  ASSERT_TRUE (out_buf != NULL);
  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));
  for (i = 0; i < 5; i++)
    EXPECT_EQ (info.data[i], i + 2);
  gst_memory_unmap (mem, &info);
  gst_buffer_unref (out_buf);

  /* Invalid option is rejected; the previous "add:2" stays in effect. */
  g_object_set (h->element, "option", "nonsense", NULL);

  in_buf = gst_harness_create_buffer (h, 5);
  mem = gst_buffer_peek_memory (in_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));
  for (i = 0; i < 5; i++)
    info.data[i] = i;
  gst_memory_unmap (mem, &info);

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);
  out_buf = gst_harness_pull (h);
  ASSERT_TRUE (out_buf != NULL);
  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));
  for (i = 0; i < 5; i++)
    EXPECT_EQ (info.data[i], i + 2);
  gst_memory_unmap (mem, &info);
  gst_buffer_unref (out_buf);

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
 * @brief Push a single buffer of the given size and return the flow status.
 */
static GstFlowReturn
_push_tensor_of_size (GstHarness *h, gsize size)
{
  GstBuffer *buf = gst_harness_create_buffer (h, size);
  GstMemory *mem = gst_buffer_peek_memory (buf, 0);
  GstMapInfo info;

  if (gst_memory_map (mem, &info, GST_MAP_WRITE)) {
    memset (info.data, 1, info.size);
    gst_memory_unmap (mem, &info);
  }

  return gst_harness_push (h, buf);
}

/**
 * @brief Test for tensor_transform, a static tensor shorter than the caps
 */
TEST (testTensorTransform, pushShortTensor_n)
{
  GstHarness *h;
  GstTensorsConfig config;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:1", NULL);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:4:4:1", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  /* the caps describe 48 bytes */
  EXPECT_EQ (_push_tensor_of_size (h, 24U), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform, a static tensor larger than the caps
 */
TEST (testTensorTransform, pushLongTensor)
{
  GstHarness *h;
  GstBuffer *out_buf;
  GstTensorsConfig config;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:1", NULL);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:4:4:1", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  /* the caps describe 48 bytes, a memory holding more of them is not an error */
  EXPECT_EQ (_push_tensor_of_size (h, 96U), GST_FLOW_OK);

  out_buf = gst_harness_pull (h);
  ASSERT_TRUE (out_buf != NULL);
  EXPECT_EQ (gst_buffer_get_size (out_buf), 48U);
  gst_buffer_unref (out_buf);

  gst_harness_teardown (h);
}

/**
 * @brief Build a uint8 flexible tensor memory of the given dimension.
 * @param dim_str dimension the meta header describes
 * @param with_data allocate the data the header describes, not the header alone
 * @param size bytes the memory exposes, 0 for every allocated byte
 */
static GstMemory *
_new_flex_memory (const gchar *dim_str, gboolean with_data, gsize size)
{
  GstTensorMetaInfo meta;
  GstTensorInfo info;
  guint8 *data;
  gsize hsize, alloc;

  gst_tensor_info_init (&info);
  info.type = _NNS_UINT8;
  gst_tensor_parse_dimension (dim_str, info.dimension);
  gst_tensor_info_convert_to_meta (&info, &meta);

  hsize = gst_tensor_meta_info_get_header_size (&meta);
  alloc = hsize + (with_data ? gst_tensor_info_get_size (&info) : 0);
  if (size == 0)
    size = alloc;
  g_assert (size <= alloc);

  data = (guint8 *) g_malloc0 (alloc);
  gst_tensor_meta_info_update_header (&meta, data);

  return gst_memory_new_wrapped ((GstMemoryFlags) 0, data, alloc, 0, size, data, g_free);
}

/**
 * @brief Test for tensor_transform, a flexible tensor without a complete header
 */
TEST (testTensorTransform, pushShortFlexibleHeader_n)
{
  GstHarness *h;
  GstBuffer *buf;
  GstCaps *caps;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:1", NULL);

  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  gst_harness_set_src_caps (h, caps);

  /**
   * The memory exposes 8 bytes of a fully allocated tensor, so an element that
   * parses the header anyway reads a valid one and keeps going, which is what
   * makes the refusal below the only possible outcome. The second memory keeps
   * the buffer from being re-split by the header.
   */
  buf = gst_buffer_new ();
  gst_buffer_append_memory (buf, _new_flex_memory ("3:4:4:1", TRUE, 8U));
  gst_buffer_append_memory (buf, _new_flex_memory ("3:4:4:1", TRUE, 0U));

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform, a flexible tensor shorter than its header says
 */
TEST (testTensorTransform, pushShortFlexibleData_n)
{
  GstHarness *h;
  GstBuffer *buf;
  GstCaps *caps;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:1", NULL);

  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  gst_harness_set_src_caps (h, caps);

  /* the header describes 48 bytes of data, nothing beyond it is allocated */
  buf = gst_buffer_new ();
  gst_buffer_append_memory (buf, _new_flex_memory ("3:4:4:1", FALSE, 0U));
  gst_buffer_append_memory (buf, _new_flex_memory ("3:4:4:1", TRUE, 0U));

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform, a tensor whose header says it is flexible
 * @details tensor_crop and a dynamic tensor_filter stamp that format on every
 *          header they append, so only a sparse payload may be refused.
 */
TEST (testTensorTransform, pushFlexibleFormatTensor)
{
  GstHarness *h;
  GstBuffer *buf, *out_buf;
  GstCaps *caps;
  GstMemory *mem;
  GstMapInfo info;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:1", NULL);

  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  gst_harness_set_src_caps (h, caps);

  buf = gst_buffer_new ();
  mem = _new_flex_memory ("3:4:4:1", TRUE, 0U);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));
  ((uint32_t *) info.data)[19] = (uint32_t) _NNS_TENSOR_FORMAT_FLEXIBLE;
  gst_memory_unmap (mem, &info);
  gst_buffer_append_memory (buf, mem);
  gst_buffer_append_memory (buf, _new_flex_memory ("3:4:4:1", TRUE, 0U));

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_OK);

  out_buf = gst_harness_pull (h);
  ASSERT_TRUE (out_buf != NULL);
  gst_buffer_unref (out_buf);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform, a flexible tensor declaring a sparse payload
 */
TEST (testTensorTransform, pushSparseFlexibleTensor_n)
{
  GstHarness *h;
  GstBuffer *buf;
  GstCaps *caps;
  GstMemory *mem;
  GstMapInfo info;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:1", NULL);

  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  gst_harness_set_src_caps (h, caps);

  /**
   * The memory is large enough for the dense size the element compares
   * against, so nothing but the format tells the two payloads apart.
   */
  buf = gst_buffer_new ();
  mem = _new_flex_memory ("3:4:4:1", TRUE, 0U);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));
  ((uint32_t *) info.data)[19] = (uint32_t) _NNS_TENSOR_FORMAT_SPARSE;
  gst_memory_unmap (mem, &info);
  gst_buffer_append_memory (buf, mem);
  gst_buffer_append_memory (buf, _new_flex_memory ("3:4:4:1", TRUE, 0U));

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Push a uint8 tensor through the dimchg mode and compare the result
 *        with the reference permutation of the given dimension.
 * @param dim_str dimension of the input tensor
 * @param from index of the dimension to be moved
 * @param to index the dimension is moved to
 */
static void
_test_dimchg (const gchar *dim_str, guint from, guint to)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config, out_config;
  GstCaps *caps;
  GstMemory *mem;
  GstMapInfo info;
  uint32_t *dim, expected[NNS_TENSOR_RANK_LIMIT];
  gchar *option;
  guint i, b, f, m, r;
  guint below = 1, moved, between = 1, above = 1;
  gsize data_size;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  option = g_strdup_printf ("%u:%u", from, to);
  g_object_set (h->element, "mode", GTT_DIMCHG, "option", option, NULL);
  g_free (option);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension (dim_str, config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;
  dim = config.info.info[0].dimension;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_size = gst_tensors_info_get_size (&config.info, 0);
  ASSERT_LE (data_size, 256U);

  in_buf = gst_harness_create_buffer (h, data_size);
  mem = gst_buffer_peek_memory (in_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));
  for (i = 0; i < data_size; i++)
    ((uint8_t *) info.data)[i] = (uint8_t) i;
  gst_memory_unmap (mem, &info);

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  out_buf = gst_harness_pull (h);
  ASSERT_TRUE (out_buf != NULL);
  ASSERT_EQ (gst_buffer_get_size (out_buf), data_size);

  /* the negotiated output dimension moves 'from' to 'to' as well */
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    expected[i] = (i < from || i > to) ? dim[i] : (i == to ? dim[from] : dim[i + 1]);

  caps = gst_pad_get_current_caps (h->sinkpad);
  ASSERT_TRUE (caps != NULL);
  gst_tensors_config_init (&out_config);
  ASSERT_TRUE (gst_tensors_config_from_structure (
      &out_config, gst_caps_get_structure (caps, 0)));
  gst_caps_unref (caps);
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    EXPECT_EQ (out_config.info.info[0].dimension[i], expected[i]);
  gst_tensors_config_free (&out_config);

  /* the moved dimension splits the tensor into below/between/above blocks */
  for (i = 0; i < from; i++)
    below *= dim[i];
  moved = dim[from];
  for (i = from + 1; i <= to; i++)
    between *= dim[i];
  for (i = to + 1; i < NNS_TENSOR_RANK_LIMIT; i++)
    if (dim[i] > 0)
      above *= dim[i];

  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));
  for (r = 0; r < above; r++)
    for (f = 0; f < moved; f++)
      for (m = 0; m < between; m++)
        for (b = 0; b < below; b++)
          EXPECT_EQ (((uint8_t *) info.data)[b + below * (m + between * (f + moved * r))],
              (uint8_t) (b + below * (f + moved * (m + between * r))));
  gst_memory_unmap (mem, &info);
  gst_buffer_unref (out_buf);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform dimchg, moving the first dimension
 */
TEST (testTensorTransform, dimchg)
{
  _test_dimchg ("2:3:4:1", 0, 2);
  _test_dimchg ("2:3:4:5", 0, 3);
}

/**
 * @brief Test for tensor_transform dimchg, moving a dimension other than the first
 */
TEST (testTensorTransform, dimchgFromNonZeroDim)
{
  _test_dimchg ("2:3:4:1", 1, 2);
  _test_dimchg ("2:3:4:5", 1, 2);
  _test_dimchg ("2:3:4:5", 1, 3);
  _test_dimchg ("2:3:4:5", 2, 3);
  _test_dimchg ("2:3:4:2:2:2", 2, 4);
}

/**
 * @brief Test for tensor_transform dimchg of a flexible tensor
 */
TEST (testTensorTransform, dimchgFlexible)
{
  GstHarness *h;
  GstBuffer *buf, *out_buf;
  GstCaps *caps;
  GstMemory *mem;
  GstTensorMetaInfo meta;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_DIMCHG, "option", "1:2", NULL);

  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  gst_harness_set_src_caps (h, caps);

  buf = gst_buffer_new ();
  gst_buffer_append_memory (buf, _new_flex_memory ("4:3:2", TRUE, 0U));
  gst_buffer_append_memory (buf, _new_flex_memory ("4:3:2", TRUE, 0U));

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_OK);

  out_buf = gst_harness_pull (h);
  ASSERT_TRUE (out_buf != NULL);
  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_tensor_meta_info_parse_memory (&meta, mem));
  EXPECT_EQ (meta.dimension[0], 4U);
  EXPECT_EQ (meta.dimension[1], 2U);
  EXPECT_EQ (meta.dimension[2], 3U);
  gst_buffer_unref (out_buf);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform dimchg, a flexible tensor of a lower rank
 */
TEST (testTensorTransform, dimchgFlexibleShortRank_n)
{
  GstHarness *h;
  GstBuffer *buf;
  GstCaps *caps;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_DIMCHG, "option", "1:2", NULL);

  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
  gst_harness_set_src_caps (h, caps);

  /* the tensor has no third dimension for the second one to move to */
  buf = gst_buffer_new ();
  gst_buffer_append_memory (buf, _new_flex_memory ("4:3", TRUE, 0U));
  gst_buffer_append_memory (buf, _new_flex_memory ("4:3", TRUE, 0U));

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
}

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
 * @brief Test for tensor_transform arithmetic, per-channel with acceleration
 */
TEST (testTensorTransform, arithmeticPerChannelAccel)
{
  const guint array_size = 6; /* 3 channels of 2 pixels */

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo info;
  guint i;
  gsize data_size;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option",
      "per-channel:true@0,add:10@1", NULL);
  g_object_set (h->element, "acceleration", (gboolean) TRUE, NULL);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:2:1:1", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_size = gst_tensors_info_get_size (&config.info, 0);

  in_buf = gst_harness_create_buffer (h, data_size);
  mem = gst_buffer_peek_memory (in_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_WRITE));
  for (i = 0; i < array_size; i++)
    ((uint8_t *) info.data)[i] = (uint8_t) i;
  gst_memory_unmap (mem, &info);

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  out_buf = gst_harness_pull (h);
  ASSERT_TRUE (out_buf != NULL);
  ASSERT_EQ (gst_buffer_get_size (out_buf), data_size);

  mem = gst_buffer_peek_memory (out_buf, 0);
  ASSERT_TRUE (gst_memory_map (mem, &info, GST_MAP_READ));
  for (i = 0; i < array_size; i++) {
    uint8_t expected = (uint8_t) (i + ((i % 3 == 1) ? 10 : 0));
    EXPECT_EQ (((uint8_t *) info.data)[i], expected);
  }
  gst_memory_unmap (mem, &info);
  gst_buffer_unref (out_buf);

  gst_harness_teardown (h);
}

/**
 * @brief Configure tensor_transform for the per-channel arithmetic tests.
 */
static GstHarness *
_arith_per_channel_harness (const gchar *option, gboolean accel)
{
  GstHarness *h;
  GstTensorsConfig config;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  if (!h)
    return NULL;

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", option, NULL);
  g_object_set (h->element, "acceleration", accel, NULL);

  g_object_get (h->element, "option", &str, NULL);
  if (!str) {
    gst_harness_teardown (h);
    return NULL;
  }
  g_free (str);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1U;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:4:4:1", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  return h;
}

/**
 * @brief Test for tensor_transform arithmetic, the documented channel dimension range
 */
TEST (testTensorTransform, arithmeticPerChannelDimBound)
{
  GstHarness *h;
  gchar *str = NULL;

  /* the option regex spells the rank limit out, keep the two in step */
  ASSERT_EQ (NNS_TENSOR_RANK_LIMIT, 16);

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option",
      "per-channel:true@15,add:1", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_STREQ (str, "per-channel:true@15,add:1");
  g_free (str);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic, channel dimension out of the rank
 */
TEST (testTensorTransform, arithmeticPerChannelDimOutOfRank_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option",
      "per-channel:true@16,add:1", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic, channel dimension wrapping the rank
 */
TEST (testTensorTransform, arithmeticPerChannelDimOverflow_n)
{
  GstHarness *h;
  gchar *str = NULL;

  h = gst_harness_new ("tensor_transform");
  ASSERT_TRUE (NULL != h);

  /* a dimension index out of the rank, which truncates to 0 in 32 bits */
  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option",
      "per-channel:true@4294967296,add:1", NULL);

  g_object_get (h->element, "option", &str, NULL);
  EXPECT_TRUE (str == NULL);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic, channel dimension not in the tensor
 */
TEST (testTensorTransform, arithmeticPerChannelDimNotInTensor_n)
{
  GstHarness *h;

  h = _arith_per_channel_harness ("per-channel:true@5,add:1", FALSE);
  ASSERT_TRUE (NULL != h);

  EXPECT_EQ (_push_tensor_of_size (h, 48U), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic, channel dimension not in the tensor
 */
TEST (testTensorTransform, arithmeticPerChannelDimNotInTensorAccel_n)
{
  GstHarness *h;

  h = _arith_per_channel_harness ("per-channel:true@5,add:1", TRUE);
  ASSERT_TRUE (NULL != h);

  EXPECT_EQ (_push_tensor_of_size (h, 48U), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic, channel index out of the channels
 */
TEST (testTensorTransform, arithmeticPerChannelIndexOutOfRange_n)
{
  GstHarness *h;

  h = _arith_per_channel_harness ("per-channel:true@0,add:1@10", FALSE);
  ASSERT_TRUE (NULL != h);

  EXPECT_EQ (_push_tensor_of_size (h, 48U), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic, channel index out of the channels
 */
TEST (testTensorTransform, arithmeticPerChannelIndexOutOfRangeAccel_n)
{
  GstHarness *h;

  h = _arith_per_channel_harness ("per-channel:true@0,add:1@10", TRUE);
  ASSERT_TRUE (NULL != h);

  EXPECT_EQ (_push_tensor_of_size (h, 48U), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic, channel index wrapping the int range
 */
TEST (testTensorTransform, arithmeticPerChannelIndexOverflow_n)
{
  GstHarness *h;

  /* 4294967296 is 0 when it is truncated to a 32bit index */
  h = _arith_per_channel_harness ("per-channel:true@0,add:1@4294967296", FALSE);
  ASSERT_TRUE (NULL != h);

  EXPECT_EQ (_push_tensor_of_size (h, 48U), GST_FLOW_ERROR);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

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
 * @brief Test for tensor_transform arithmetic (static to flexible tensor)
 */
TEST (testTensorTransform, arithmeticStaticToFlexTensor)
{
  const guint num_buffers = 3;
  const guint array_size = 5;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig in_config, out_config;
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
  gst_tensors_config_init (&in_config);
  gst_tensors_config_init (&out_config);

  in_config.info.num_tensors = 1U;
  in_config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("5", in_config.info.info[0].dimension);
  in_config.rate_n = 10;
  in_config.rate_d = 1;

  gst_tensors_config_copy (&out_config, &in_config);
  out_config.info.info[0].type = _NNS_FLOAT32;

  data_in_size = gst_tensors_info_get_size (&in_config.info, 0);
  data_out_size = gst_tensors_info_get_size (&out_config.info, 0);

  /* set input caps (static tensor) */
  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&in_config));

  /* set output caps (flexible tensor) */
  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
  gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 10, 1, NULL);
  gst_harness_set_sink_caps (h, caps);

  /* push buffers */
  for (b = 0; b < num_buffers; b++) {
    /* set input buffer */
    in_buf = gst_harness_create_buffer (h, data_in_size);

    mem = gst_buffer_peek_memory (in_buf, 0);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));

    _input = (uint8_t *) map.data;
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
 * @brief The number of tensors to test extra tensors (more than NNS_TENSOR_MEMORY_MAX).
 */
#define TEST_EXTRA_TENSORS_NUM (18U)

/**
 * @brief The number of elements of each tensor to test extra tensors.
 */
#define TEST_EXTRA_TENSORS_SIZE (64U)

/**
 * @brief Prepare tensors config to test extra tensors.
 */
static void
_setup_extra_tensors_config (GstTensorsConfig *config)
{
  GstTensorInfo *_info;
  guint i;

  gst_tensors_config_init (config);
  config->info.num_tensors = TEST_EXTRA_TENSORS_NUM;
  config->rate_n = 0;
  config->rate_d = 1;

  for (i = 0; i < TEST_EXTRA_TENSORS_NUM; i++) {
    _info = gst_tensors_info_get_nth_info (&config->info, i);
    _info->type = _NNS_FLOAT32;
    gst_tensor_parse_dimension ("64", _info->dimension);
  }
}

/**
 * @brief Test for tensor_transform arithmetic (more tensors than NNS_TENSOR_MEMORY_MAX)
 */
TEST (testTensorTransform, arithmeticExtraTensors)
{
  const guint num_buffers = 2U;

  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo map;
  guint i, j, b;
  gint refcount;
  gsize dsize;
  float *_data;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:1", NULL);

  _setup_extra_tensors_config (&config);
  dsize = gst_tensors_info_get_size (&config.info, 0);

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  for (b = 0; b < num_buffers; b++) {
    /* set input buffer */
    in_buf = gst_buffer_new ();

    for (i = 0; i < TEST_EXTRA_TENSORS_NUM; i++) {
      mem = gst_allocator_alloc (NULL, dsize, NULL);
      ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));

      _data = (float *) map.data;
      for (j = 0; j < TEST_EXTRA_TENSORS_SIZE; j++)
        _data[j] = (float) (b * 10000 + i * 100 + j);

      gst_memory_unmap (mem, &map);
      ASSERT_TRUE (gst_tensor_buffer_append_memory (
          in_buf, mem, gst_tensors_info_get_nth_info (&config.info, i)));
    }

    /* keep the reference to check the ownership of the input buffer */
    gst_buffer_ref (in_buf);

    EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

    refcount = GST_MINI_OBJECT_REFCOUNT_VALUE (in_buf);
    EXPECT_EQ (refcount, 1);
    gst_buffer_unref (in_buf);

    /* get output buffer */
    out_buf = gst_harness_pull (h);

    ASSERT_TRUE (out_buf != NULL);
    ASSERT_EQ (gst_tensor_buffer_get_count (out_buf), TEST_EXTRA_TENSORS_NUM);

    for (i = 0; i < TEST_EXTRA_TENSORS_NUM; i++) {
      mem = gst_tensor_buffer_get_nth_memory (out_buf, i);
      ASSERT_TRUE (mem != NULL);
      ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
      ASSERT_EQ (map.size, dsize);

      _data = (float *) map.data;
      for (j = 0; j < TEST_EXTRA_TENSORS_SIZE; j++)
        EXPECT_FLOAT_EQ (_data[j], (float) (b * 10000 + i * 100 + j + 1));

      gst_memory_unmap (mem, &map);
      gst_memory_unref (mem);
    }

    gst_buffer_unref (out_buf);
  }

  EXPECT_EQ (gst_harness_buffers_received (h), num_buffers);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic with 'apply' property (more tensors than NNS_TENSOR_MEMORY_MAX)
 */
TEST (testTensorTransform, arithmeticExtraTensorsApply)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo map;
  guint i, j;
  gint refcount;
  gsize dsize;
  float *_data;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:1", "apply",
      "0,16,17", NULL);

  _setup_extra_tensors_config (&config);
  dsize = gst_tensors_info_get_size (&config.info, 0);

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  /* set input buffer */
  in_buf = gst_buffer_new ();

  for (i = 0; i < TEST_EXTRA_TENSORS_NUM; i++) {
    mem = gst_allocator_alloc (NULL, dsize, NULL);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));

    _data = (float *) map.data;
    for (j = 0; j < TEST_EXTRA_TENSORS_SIZE; j++)
      _data[j] = (float) (i * 100 + j);

    gst_memory_unmap (mem, &map);
    ASSERT_TRUE (gst_tensor_buffer_append_memory (
        in_buf, mem, gst_tensors_info_get_nth_info (&config.info, i)));
  }

  /* keep the reference to check the ownership of the input buffer */
  gst_buffer_ref (in_buf);

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  refcount = GST_MINI_OBJECT_REFCOUNT_VALUE (in_buf);
  EXPECT_EQ (refcount, 1);
  gst_buffer_unref (in_buf);

  /* get output buffer */
  out_buf = gst_harness_pull (h);

  ASSERT_TRUE (out_buf != NULL);
  ASSERT_EQ (gst_tensor_buffer_get_count (out_buf), TEST_EXTRA_TENSORS_NUM);

  for (i = 0; i < TEST_EXTRA_TENSORS_NUM; i++) {
    /* the operator is applied to the tensor 0, 16 and 17 */
    float diff = (i == 0U || i >= 16U) ? 1.0f : 0.0f;

    mem = gst_tensor_buffer_get_nth_memory (out_buf, i);
    ASSERT_TRUE (mem != NULL);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
    ASSERT_EQ (map.size, dsize);

    _data = (float *) map.data;
    for (j = 0; j < TEST_EXTRA_TENSORS_SIZE; j++)
      EXPECT_FLOAT_EQ (_data[j], (float) (i * 100 + j) + diff);

    gst_memory_unmap (mem, &map);
    gst_memory_unref (mem);
  }

  gst_buffer_unref (out_buf);
  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform arithmetic with 'apply' property which selects no tensor
 */
TEST (testTensorTransform, arithmeticExtraTensorsApplyNone)
{
  GstHarness *h;
  GstBuffer *in_buf, *out_buf;
  GstTensorsConfig config;
  GstMemory *mem;
  GstMapInfo map;
  guint i, j;
  gint refcount;
  gsize dsize;
  float *_data;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:1", NULL);

  /* no tensor of the stream is selected */
  g_object_set (h->element, "apply", "99", NULL);

  _setup_extra_tensors_config (&config);
  dsize = gst_tensors_info_get_size (&config.info, 0);

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  /* set input buffer */
  in_buf = gst_buffer_new ();

  for (i = 0; i < TEST_EXTRA_TENSORS_NUM; i++) {
    mem = gst_allocator_alloc (NULL, dsize, NULL);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));

    _data = (float *) map.data;
    for (j = 0; j < TEST_EXTRA_TENSORS_SIZE; j++)
      _data[j] = (float) (i * 100 + j);

    gst_memory_unmap (mem, &map);
    ASSERT_TRUE (gst_tensor_buffer_append_memory (
        in_buf, mem, gst_tensors_info_get_nth_info (&config.info, i)));
  }

  /* keep the reference to check the ownership of the input buffer */
  gst_buffer_ref (in_buf);

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);

  refcount = GST_MINI_OBJECT_REFCOUNT_VALUE (in_buf);
  EXPECT_EQ (refcount, 1);
  gst_buffer_unref (in_buf);

  /* get output buffer, every tensor is passed through */
  out_buf = gst_harness_pull (h);

  ASSERT_TRUE (out_buf != NULL);
  ASSERT_EQ (gst_tensor_buffer_get_count (out_buf), TEST_EXTRA_TENSORS_NUM);

  for (i = 0; i < TEST_EXTRA_TENSORS_NUM; i++) {
    mem = gst_tensor_buffer_get_nth_memory (out_buf, i);
    ASSERT_TRUE (mem != NULL);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
    ASSERT_EQ (map.size, dsize);

    _data = (float *) map.data;
    for (j = 0; j < TEST_EXTRA_TENSORS_SIZE; j++)
      EXPECT_FLOAT_EQ (_data[j], (float) (i * 100 + j));

    gst_memory_unmap (mem, &map);
    gst_memory_unref (mem);
  }

  gst_buffer_unref (out_buf);
  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_transform with insufficient buffer size (more tensors than NNS_TENSOR_MEMORY_MAX)
 */
TEST (testTensorTransform, arithmeticExtraTensorsInvalidSize_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstTensorsConfig config;
  gint refcount;
  gsize dsize;

  h = gst_harness_new ("tensor_transform");

  g_object_set (h->element, "mode", GTT_ARITHMETIC, "option", "add:1", NULL);

  _setup_extra_tensors_config (&config);
  dsize = gst_tensors_info_get_size (&config.info, 0);

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  /* the buffer has the data of a single tensor */
  in_buf = gst_harness_create_buffer (h, dsize);
  gst_buffer_ref (in_buf);

  EXPECT_NE (gst_harness_push (h, in_buf), GST_FLOW_OK);

  refcount = GST_MINI_OBJECT_REFCOUNT_VALUE (in_buf);
  EXPECT_EQ (refcount, 1);
  gst_buffer_unref (in_buf);

  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_tensors_config_free (&config);
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
#include <type_traits>

#include "nnstreamer-orc.h"

/**
 * @brief Deduce the element type an orc constant-operand function works on.
 *
 * Declared only, and used in unevaluated context, to recover the buffer type
 * from an orc prototype without spelling out its ORC_RESTRICT qualifier.
 */
template <typename T> T test_orc_elem_type (void (*orc_func) (T *, int, int));

/**
 * @brief Buffer element types for the 64-bit orc test cases.
 *
 * orcc emits its own orc_intNN typedefs, and their C99 branch is not taken in
 * C++, where __STDC_VERSION__ is undefined, so on LP64 orc_int64 falls back to
 * long. That is the same type as int64_t on Linux but not on macOS, where
 * int64_t is long long; the two are distinct types to the compiler even though
 * both are 64 bits wide, so buffers spelled int64_t cannot be passed to the orc
 * entry points there. Taking the types from the prototypes themselves keeps
 * these buffers correct under either convention by construction, rather than
 * naming a type that only happens to match on one of them. New orc test code
 * should use them too; int64_t/uint64_t buffers compile on Linux but break the
 * macOS build.
 */
typedef decltype (test_orc_elem_type (nns_orc_add_c_s64)) orc_s64_elem;
typedef decltype (test_orc_elem_type (nns_orc_add_c_u64)) orc_u64_elem;

static_assert (sizeof (orc_s64_elem) == 8 && sizeof (orc_u64_elem) == 8,
    "orc 64-bit buffers must be 64 bits wide");
static_assert (std::is_signed<orc_s64_elem>::value && std::is_unsigned<orc_u64_elem>::value,
    "orc 64-bit buffer signedness must match the s64/u64 prototypes");

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
  orc_s64_elem data_s64[array_size] = {
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
  orc_u64_elem data_u64[array_size] = {
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
  orc_s64_elem data_s64[array_size] = {
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
  orc_u64_elem data_u64[array_size] = {
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
  orc_s64_elem res_s64[array_size] = {
    0,
  };

  nns_orc_conv_s8_to_s64 (res_s64, data_s8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_s8[i]);
  }

  /* convert u64 */
  orc_u64_elem res_u64[array_size] = {
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
  orc_s64_elem res_s64[array_size] = {
    0,
  };

  nns_orc_conv_u8_to_s64 (res_s64, data_u8, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_u8[i]);
  }

  /* convert u64 */
  orc_u64_elem res_u64[array_size] = {
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
  orc_s64_elem res_s64[array_size] = {
    0,
  };

  nns_orc_conv_s16_to_s64 (res_s64, data_s16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_s16[i]);
  }

  /* convert u64 */
  orc_u64_elem res_u64[array_size] = {
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
  orc_s64_elem res_s64[array_size] = {
    0,
  };

  nns_orc_conv_u16_to_s64 (res_s64, data_u16, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_u16[i]);
  }

  /* convert u64 */
  orc_u64_elem res_u64[array_size] = {
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
  orc_s64_elem res_s64[array_size] = {
    0,
  };

  nns_orc_conv_s32_to_s64 (res_s64, data_s32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_s32[i]);
  }

  /* convert u64 */
  orc_u64_elem res_u64[array_size] = {
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
  orc_s64_elem res_s64[array_size] = {
    0,
  };

  nns_orc_conv_u32_to_s64 (res_s64, data_u32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_u32[i]);
  }

  /* convert u64 */
  orc_u64_elem res_u64[array_size] = {
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

  orc_s64_elem data_s64[array_size] = {
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
  orc_s64_elem res_s64[array_size] = {
    0,
  };

  nns_orc_conv_s64_to_s64 (res_s64, data_s64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_s64[i]);
  }

  /* convert u64 */
  orc_u64_elem res_u64[array_size] = {
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

  orc_u64_elem data_u64[array_size] = {
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
  orc_s64_elem res_s64[array_size] = {
    0,
  };

  nns_orc_conv_u64_to_s64 (res_s64, data_u64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_u64[i]);
  }

  /* convert u64 */
  orc_u64_elem res_u64[array_size] = {
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
  orc_s64_elem res_s64[array_size] = {
    0,
  };

  nns_orc_conv_f32_to_s64 (res_s64, data_f32, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_f32[i]);
  }

  /* convert u64 */
  orc_u64_elem res_u64[array_size] = {
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
  orc_s64_elem res_s64[array_size] = {
    0,
  };

  nns_orc_conv_f64_to_s64 (res_s64, data_f64, array_size);

  for (i = 0; i < array_size; i++) {
    EXPECT_EQ (res_s64[i], (int64_t) data_f64[i]);
  }

  /* convert u64 */
  orc_u64_elem res_u64[array_size] = {
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
 * @brief Mock C++ subplugin (tensor_filter_subplugin) to test open/close paths
 *        of the C++ subplugin API wrapper (tensor_filter_support_cc.cc).
 */
class cpp_mock_subplugin : public nnstreamer::tensor_filter_subplugin
{
  public:
  enum throw_point_t {
    THROW_NONE = 0,
    THROW_BEFORE_ACQUIRE,
    THROW_AFTER_ACQUIRE,
    THROW_AFTER_RELEASE,
  };

  static const char *mock_name;
  static cpp_mock_subplugin *registered;
  static guint instances_alive;
  static guint resources_alive;
  static throw_point_t throw_point;

  /** @brief constructor */
  cpp_mock_subplugin ()
  {
    instances_alive++;
  }

  /** @brief destructor */
  ~cpp_mock_subplugin ()
  {
    if (resource_held)
      resources_alive--;
    instances_alive--;
  }

  /** @brief mandatory method */
  tensor_filter_subplugin &getEmptyInstance () override
  {
    return *(new cpp_mock_subplugin ());
  }

  /** @brief mandatory method; acquires a resource and throws at the point set by throw_point */
  void configure_instance (const GstTensorFilterProperties *prop) override
  {
    UNUSED (prop);
    if (throw_point == THROW_BEFORE_ACQUIRE)
      throw std::invalid_argument ("Configuration failure for testing");
    resources_alive++;
    resource_held = true;
    if (throw_point == THROW_AFTER_ACQUIRE)
      throw std::invalid_argument ("Configuration failure for testing");
    if (throw_point == THROW_AFTER_RELEASE) {
      resources_alive--;
      resource_held = false;
      throw std::invalid_argument ("Configuration failure for testing");
    }
  }

  /** @brief mandatory method */
  void invoke (const GstTensorMemory *input, GstTensorMemory *output) override
  {
    UNUSED (input);
    UNUSED (output);
  }

  /** @brief mandatory method */
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info) override
  {
    info.name = mock_name;
    info.allow_in_place = FALSE;
    info.allocate_in_invoke = FALSE;
    info.run_without_model = TRUE;
    info.verify_model_path = FALSE;
    info.hw_list = nullptr;
    info.num_hw = 0;
  }

  /** @brief mandatory method */
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info) override
  {
    UNUSED (ops);
    UNUSED (in_info);
    UNUSED (out_info);
    return -ENOENT;
  }

  /** @brief register this mock subplugin */
  static void init ()
  {
    registered = register_subplugin<cpp_mock_subplugin> ();
  }

  /** @brief unregister this mock subplugin */
  static void fini ()
  {
    unregister_subplugin<cpp_mock_subplugin> (registered);
    registered = nullptr;
  }

  private:
  bool resource_held = false;
};

const char *cpp_mock_subplugin::mock_name = "cpp_mock_subplugin";
cpp_mock_subplugin *cpp_mock_subplugin::registered = nullptr;
guint cpp_mock_subplugin::instances_alive = 0;
guint cpp_mock_subplugin::resources_alive = 0;
cpp_mock_subplugin::throw_point_t cpp_mock_subplugin::throw_point
    = cpp_mock_subplugin::THROW_NONE;

/**
 * @brief Test fixture registering/unregistering the mock C++ subplugin.
 */
class testTensorFilterCppSubplugin : public ::testing::Test
{
  protected:
  /** @brief reset counters and register the mock subplugin */
  void SetUp () override
  {
    cpp_mock_subplugin::instances_alive = 0;
    cpp_mock_subplugin::resources_alive = 0;
    cpp_mock_subplugin::throw_point = cpp_mock_subplugin::THROW_NONE;
    cpp_mock_subplugin::init ();
  }

  /** @brief unregister the mock subplugin and verify no instance leaks */
  void TearDown () override
  {
    cpp_mock_subplugin::throw_point = cpp_mock_subplugin::THROW_NONE;
    cpp_mock_subplugin::fini ();
    EXPECT_EQ (cpp_mock_subplugin::instances_alive, 0U);
    EXPECT_EQ (cpp_mock_subplugin::resources_alive, 0U);
  }
};

/**
 * @brief Test that a C++ sub-plugin declaring run_without_model is opened by
 *        the element without a model property.
 * @details This drives cpp_getFrameworkInfo() with no private data, the path
 *          every C++ sub-plugin takes while tensor_filter verifies the model.
 */
TEST_F (testTensorFilterCppSubplugin, elementOpenWithoutModel)
{
  GstElement *filter = gst_element_factory_make ("tensor_filter", NULL);

  ASSERT_TRUE (filter != NULL);
  g_object_set (filter, "framework", cpp_mock_subplugin::mock_name, NULL);

  EXPECT_EQ (gst_element_set_state (filter, GST_STATE_PAUSED), GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (cpp_mock_subplugin::resources_alive, 1U);

  gst_element_set_state (filter, GST_STATE_NULL);
  gst_object_unref (filter);
}

/**
 * @brief Test C++ subplugin open: when configure_instance() throws at any
 *        point, repeated open attempts must neither crash nor leak the
 *        spawned instance and the resources it acquired before throwing.
 */
TEST_F (testTensorFilterCppSubplugin, openFailNoLeak_n)
{
  const GstTensorFilterFramework *fw;
  GstTensorFilterProperties prop;
  gpointer private_data;
  guint i;
  const cpp_mock_subplugin::throw_point_t points[] = { cpp_mock_subplugin::THROW_BEFORE_ACQUIRE,
    cpp_mock_subplugin::THROW_AFTER_ACQUIRE, cpp_mock_subplugin::THROW_AFTER_RELEASE };

  fw = nnstreamer_filter_find (cpp_mock_subplugin::mock_name);
  ASSERT_TRUE (fw && fw->open && fw->close);

  memset (&prop, 0, sizeof (prop));
  prop.fwname = cpp_mock_subplugin::mock_name;

  EXPECT_EQ (cpp_mock_subplugin::instances_alive, 1U);

  for (cpp_mock_subplugin::throw_point_t point : points) {
    cpp_mock_subplugin::throw_point = point;
    for (i = 0; i < 100U; i++) {
      private_data = nullptr;
      EXPECT_EQ (fw->open (&prop, &private_data), -EINVAL);
      EXPECT_TRUE (private_data == nullptr);
      ASSERT_EQ (cpp_mock_subplugin::instances_alive, 1U);
      ASSERT_EQ (cpp_mock_subplugin::resources_alive, 0U);
    }
  }
}

/**
 * @brief Test C++ subplugin open/close: on success, the ownership of the
 *        configured instance is transferred to *private_data and the
 *        instance is deleted by close.
 */
TEST_F (testTensorFilterCppSubplugin, openCloseOwnership_p)
{
  const GstTensorFilterFramework *fw;
  GstTensorFilterProperties prop;
  gpointer private_data = nullptr;

  fw = nnstreamer_filter_find (cpp_mock_subplugin::mock_name);
  ASSERT_TRUE (fw && fw->open && fw->close);

  memset (&prop, 0, sizeof (prop));
  prop.fwname = cpp_mock_subplugin::mock_name;

  EXPECT_EQ (cpp_mock_subplugin::instances_alive, 1U);

  EXPECT_EQ (fw->open (&prop, &private_data), 0);
  EXPECT_TRUE (private_data != nullptr);
  EXPECT_EQ (cpp_mock_subplugin::instances_alive, 2U);
  EXPECT_EQ (cpp_mock_subplugin::resources_alive, 1U);

  fw->close (&prop, &private_data);
  EXPECT_EQ (cpp_mock_subplugin::instances_alive, 1U);
  EXPECT_EQ (cpp_mock_subplugin::resources_alive, 0U);
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
 * @brief Caps of the dense tensor stream used by the tensor_sparse tests.
 */
#define SPARSE_DENSE_CAPS_STR                  \
  "other/tensors,format=static,num_tensors=1," \
  "dimensions=(string)40:1:1:1,types=(string)int32,framerate=0/1"

/**
 * @brief Data for the src-pad probe watching the buffer pushed by tensor_sparse.
 */
typedef struct {
  GstBuffer *buffer;
  guint received;
} sparse_probe_data_s;

/**
 * @brief Src-pad probe holding two extra references of the pushed buffer.
 * @details Two references are taken on purpose. gst_pad_push() consumes the
 * reference owned by the chain function, so a single extra reference would be
 * dropped by the very double unref this probe is meant to detect and reading
 * the refcount back would be a use-after-free. With two, the buffer stays
 * alive either way and the refcount alone tells whether it was unreffed twice.
 */
static GstPadProbeReturn
_sparse_probe_ref_buffer (GstPad *pad, GstPadProbeInfo *info, gpointer udata)
{
  sparse_probe_data_s *pdata = (sparse_probe_data_s *) udata;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);

  UNUSED (pad);

  if (pdata->received++ == 0U) {
    pdata->buffer = gst_buffer_ref (buffer);
    gst_buffer_ref (buffer);
  }

  return GST_PAD_PROBE_OK;
}

/**
 * @brief Create a dense tensor memory of 40 int32 elements for the sparse tests.
 */
static GstMemory *
_sparse_new_dense_memory (GstTensorInfo *info)
{
  gpointer data;
  gsize data_size;
  guint i;

  gst_tensor_info_init (info);
  info->type = _NNS_INT32;
  gst_tensor_parse_dimension ("40", info->dimension);

  data_size = gst_tensor_info_get_size (info);
  data = g_malloc0 (data_size);
  for (i = 0; i < 40U; i++)
    ((gint32 *) data)[i] = (i % 7U == 0U) ? (gint32) (i + 1) : 0;

  return gst_memory_new_wrapped (
      GST_MEMORY_FLAG_READONLY, data, data_size, 0, data_size, data, g_free);
}

/**
 * @brief Wrap a single memory into a new buffer.
 */
static GstBuffer *
_sparse_new_buffer (GstMemory *mem)
{
  GstBuffer *buf = gst_buffer_new ();

  gst_buffer_append_memory (buf, mem);
  return buf;
}

/**
 * @brief Compare a dense memory with the reference data of the sparse tests.
 */
static void
_sparse_check_dense_memory (GstMemory *mem)
{
  GstMapInfo map;
  guint i;

  ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
  ASSERT_EQ (map.size, 40U * sizeof (gint32));
  for (i = 0; i < 40U; i++)
    EXPECT_EQ (((gint32 *) map.data)[i], (i % 7U == 0U) ? (gint32) (i + 1) : 0);
  gst_memory_unmap (mem, &map);
}

/**
 * @brief Create a sparse tensor memory holding the reference data.
 */
static GstMemory *
_sparse_new_sparse_memory (GstTensorInfo *info)
{
  GstMemory *dense, *sparse;
  GstTensorMetaInfo meta;

  dense = _sparse_new_dense_memory (info);
  gst_tensor_info_convert_to_meta (info, &meta);
  meta.format = _NNS_TENSOR_FORMAT_SPARSE;
  meta.media_type = _NNS_TENSOR;

  sparse = gst_tensor_sparse_from_dense (&meta, dense);
  gst_memory_unref (dense);

  return sparse;
}

/**
 * @brief Test for tensor_sparse_enc, encoded buffer is pushed to the src pad.
 */
TEST (testTensorSparse, encPushBuffer)
{
  GstHarness *h;
  GstBuffer *out;
  GstMemory *dense;
  GstTensorInfo info;
  GstTensorMetaInfo meta;

  h = gst_harness_new ("tensor_sparse_enc");
  ASSERT_TRUE (h != NULL);

  gst_harness_set_src_caps_str (h, SPARSE_DENSE_CAPS_STR);

  ASSERT_EQ (gst_harness_push (h, _sparse_new_buffer (_sparse_new_dense_memory (&info))),
      GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 1U);

  out = gst_harness_pull (h);
  ASSERT_TRUE (out != NULL);
  ASSERT_EQ (gst_buffer_n_memory (out), 1U);

  gst_tensor_meta_info_init (&meta);
  dense = gst_tensor_sparse_to_dense (&meta, gst_buffer_peek_memory (out, 0));
  ASSERT_TRUE (dense != NULL);
  _sparse_check_dense_memory (dense);

  gst_memory_unref (dense);
  gst_buffer_unref (out);
  gst_tensor_info_free (&info);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_sparse_enc, the output buffer is not unreffed twice
 *        when the downstream push fails.
 */
TEST (testTensorSparse, encPushFailure_n)
{
  GstHarness *h;
  GstPad *srcpad;
  GstTensorInfo info;
  sparse_probe_data_s pdata = { NULL, 0U };

  h = gst_harness_new ("tensor_sparse_enc");
  ASSERT_TRUE (h != NULL);

  srcpad = gst_element_get_static_pad (h->element, "src");
  ASSERT_TRUE (srcpad != NULL);
  gst_pad_add_probe (
      srcpad, GST_PAD_PROBE_TYPE_BUFFER, _sparse_probe_ref_buffer, &pdata, NULL);

  gst_harness_set_src_caps_str (h, SPARSE_DENSE_CAPS_STR);

  /* A deactivated peer fails the push, as a shutdown or seek does. */
  gst_pad_set_active (h->sinkpad, FALSE);

  EXPECT_EQ (gst_harness_push (h, _sparse_new_buffer (_sparse_new_dense_memory (&info))),
      GST_FLOW_FLUSHING);

  ASSERT_EQ (pdata.received, 1U);
  ASSERT_TRUE (pdata.buffer != NULL);
  EXPECT_EQ (GST_MINI_OBJECT_REFCOUNT_VALUE (pdata.buffer), 2);

  gst_buffer_unref (pdata.buffer);
  gst_buffer_unref (pdata.buffer);
  gst_object_unref (srcpad);
  gst_tensor_info_free (&info);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_sparse_dec, decoded buffer is pushed to the src pad.
 */
TEST (testTensorSparse, decPushBuffer)
{
  GstHarness *h;
  GstBuffer *out;
  GstMemory *sparse;
  GstTensorInfo info;

  h = gst_harness_new ("tensor_sparse_dec");
  ASSERT_TRUE (h != NULL);

  gst_harness_set_sink_caps_str (h, SPARSE_DENSE_CAPS_STR);
  gst_harness_set_src_caps_str (h, "other/tensors,format=sparse,framerate=0/1");

  sparse = _sparse_new_sparse_memory (&info);
  ASSERT_TRUE (sparse != NULL);

  ASSERT_EQ (gst_harness_push (h, _sparse_new_buffer (sparse)), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 1U);

  out = gst_harness_pull (h);
  ASSERT_TRUE (out != NULL);
  ASSERT_EQ (gst_buffer_n_memory (out), 1U);
  _sparse_check_dense_memory (gst_buffer_peek_memory (out, 0));

  gst_buffer_unref (out);
  gst_tensor_info_free (&info);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_sparse_dec, the output buffer is not unreffed twice
 *        when the downstream push fails.
 */
TEST (testTensorSparse, decPushFailure_n)
{
  GstHarness *h;
  GstPad *srcpad;
  GstMemory *sparse;
  GstTensorInfo info;
  sparse_probe_data_s pdata = { NULL, 0U };

  h = gst_harness_new ("tensor_sparse_dec");
  ASSERT_TRUE (h != NULL);

  srcpad = gst_element_get_static_pad (h->element, "src");
  ASSERT_TRUE (srcpad != NULL);
  gst_pad_add_probe (
      srcpad, GST_PAD_PROBE_TYPE_BUFFER, _sparse_probe_ref_buffer, &pdata, NULL);

  gst_harness_set_sink_caps_str (h, SPARSE_DENSE_CAPS_STR);
  gst_harness_set_src_caps_str (h, "other/tensors,format=sparse,framerate=0/1");

  sparse = _sparse_new_sparse_memory (&info);
  ASSERT_TRUE (sparse != NULL);

  /* A deactivated peer fails the push, as a shutdown or seek does. */
  gst_pad_set_active (h->sinkpad, FALSE);

  EXPECT_EQ (gst_harness_push (h, _sparse_new_buffer (sparse)), GST_FLOW_FLUSHING);

  ASSERT_EQ (pdata.received, 1U);
  ASSERT_TRUE (pdata.buffer != NULL);
  EXPECT_EQ (GST_MINI_OBJECT_REFCOUNT_VALUE (pdata.buffer), 2);

  gst_buffer_unref (pdata.buffer);
  gst_buffer_unref (pdata.buffer);
  gst_object_unref (srcpad);
  gst_tensor_info_free (&info);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_sparse_dec, the decoded buffer is dropped when it does
 *        not match the negotiated downstream config.
 */
TEST (testTensorSparse, decConfigMismatch_n)
{
  GstHarness *h;
  GstMemory *sparse;
  GstTensorInfo info;

  h = gst_harness_new ("tensor_sparse_dec");
  ASSERT_TRUE (h != NULL);

  /* Downstream expects 20 elements while the buffer decodes into 40. */
  gst_harness_set_sink_caps_str (h,
      "other/tensors,format=static,num_tensors=1,"
      "dimensions=(string)20:1:1:1,types=(string)int32,framerate=0/1");
  gst_harness_set_src_caps_str (h, "other/tensors,format=sparse,framerate=0/1");

  sparse = _sparse_new_sparse_memory (&info);
  ASSERT_TRUE (sparse != NULL);

  EXPECT_EQ (gst_harness_push (h, _sparse_new_buffer (sparse)), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_tensor_info_free (&info);
  gst_harness_teardown (h);
}

/**
 * @brief Number of GStreamer critical logs of a tensor_sparse test.
 */
static guint sparse_gst_critical_count;

/**
 * @brief Last critical message of the GStreamer domain, for diagnostics.
 */
static gchar sparse_gst_critical_msg[256];

/**
 * @brief Log handler counting the critical logs of the GStreamer domain.
 * @details A buffer released one time too many makes gst_mini_object_unref()
 * log 'assertion refcount > 0 failed' there, so a zero count is what tells the
 * error paths apart from a double unref.
 */
static void
_sparse_count_gst_critical (const gchar *domain, GLogLevelFlags level,
    const gchar *message, gpointer udata)
{
  UNUSED (domain);
  UNUSED (udata);

  if (level & G_LOG_LEVEL_CRITICAL) {
    sparse_gst_critical_count++;
    g_strlcpy (sparse_gst_critical_msg, message, sizeof (sparse_gst_critical_msg));
  }
}

/**
 * @brief Start counting the critical logs of the GStreamer domain.
 * @details The counter is proven live before it is used, so that a handler that
 * silently stops matching cannot turn the zero-critical assertions into no-ops.
 */
static guint
_sparse_watch_gst_critical (void)
{
  guint handler = g_log_set_handler ("GStreamer",
      (GLogLevelFlags) (G_LOG_LEVEL_CRITICAL | G_LOG_FLAG_FATAL | G_LOG_FLAG_RECURSION),
      _sparse_count_gst_critical, NULL);

  sparse_gst_critical_count = 0;
  g_log ("GStreamer", G_LOG_LEVEL_CRITICAL, "tensor_sparse test: counter self-check");
  EXPECT_EQ (sparse_gst_critical_count, 1U);

  sparse_gst_critical_count = 0;
  sparse_gst_critical_msg[0] = '\0';
  return handler;
}

/**
 * @brief Test for tensor_sparse_dec, a buffer that carries no valid meta header
 *        is rejected and its output buffer is released exactly once.
 */
TEST (testTensorSparse, decInvalidSparseData_n)
{
  GstHarness *h;
  GstBuffer *in;
  guint handler, i;
  const gsize data_size = 200U;

  h = gst_harness_new ("tensor_sparse_dec");
  ASSERT_TRUE (h != NULL);

  /* Sink caps avoid the unfixed-caps critical, which would be counted below. */
  gst_harness_set_sink_caps_str (h, SPARSE_DENSE_CAPS_STR);
  gst_harness_set_src_caps_str (h, "other/tensors,format=sparse,framerate=0/1");

  /* Two memories: gst_tensor_buffer_from_config() passes the buffer through. */
  in = gst_buffer_new ();
  for (i = 0; i < 2U; i++) {
    gpointer data = g_malloc0 (data_size);

    gst_buffer_append_memory (in, gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY,
                                      data, data_size, 0, data_size, data, g_free));
  }

  handler = _sparse_watch_gst_critical ();
  EXPECT_EQ (gst_harness_push (h, in), GST_FLOW_ERROR);
  g_log_remove_handler ("GStreamer", handler);

  EXPECT_EQ (sparse_gst_critical_count, 0U) << sparse_gst_critical_msg;
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_sparse_enc, a tensor type the encoder cannot handle is
 *        rejected and its output buffer is released exactly once.
 */
TEST (testTensorSparse, encUnsupportedType_n)
{
  GstHarness *h;
  GstMemory *mem;
  gpointer data;
  guint handler;
  const gsize data_size = 40U * 2U;

  h = gst_harness_new ("tensor_sparse_enc");
  ASSERT_TRUE (h != NULL);

  /* float16 has no case in gst_tensor_sparse_from_dense(). */
  gst_harness_set_src_caps_str (h,
      "other/tensors,format=static,num_tensors=1,"
      "dimensions=(string)40:1:1:1,types=(string)float16,framerate=0/1");

  data = g_malloc0 (data_size);
  mem = gst_memory_new_wrapped (
      GST_MEMORY_FLAG_READONLY, data, data_size, 0, data_size, data, g_free);

  handler = _sparse_watch_gst_critical ();
  EXPECT_EQ (gst_harness_push (h, _sparse_new_buffer (mem)), GST_FLOW_ERROR);
  g_log_remove_handler ("GStreamer", handler);

  EXPECT_EQ (sparse_gst_critical_count, 0U) << sparse_gst_critical_msg;
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Build a sparse int32 tensor memory from the given header fields.
 * @details Unlike _sparse_new_sparse_memory(), the header fields and the size
 * of the memory are set independently of each other, so that a header
 * describing more data than the memory holds can be handed to the decoder.
 * @param dimension the dimension the meta info declares
 * @param nnz the number of non-zero elements the meta info declares
 * @param indices the indices to write after the values, NULL to leave them 0
 * @param size the number of bytes the memory actually holds
 */
static GstMemory *
_sparse_new_raw_memory_dim (const gchar *dimension, guint nnz, const guint *indices, gsize size)
{
  GstTensorMetaInfo meta;
  guint8 *data;
  gsize header_size;

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_INT32;
  gst_tensor_parse_dimension (dimension, meta.dimension);
  meta.format = _NNS_TENSOR_FORMAT_SPARSE;
  meta.media_type = _NNS_TENSOR;
  meta.sparse_info.nnz = nnz;

  header_size = gst_tensor_meta_info_get_header_size (&meta);
  data = (guint8 *) g_malloc0 (size);

  if (size >= header_size) {
    gst_tensor_meta_info_update_header (&meta, data);

    if (indices && size >= header_size + (gsize) nnz * (sizeof (gint32) + sizeof (guint)))
      memcpy (data + header_size + (gsize) nnz * sizeof (gint32), indices,
          (gsize) nnz * sizeof (guint));
  }

  return gst_memory_new_wrapped ((GstMemoryFlags) 0, data, size, 0, size, data, g_free);
}

/**
 * @brief Build a sparse int32 tensor memory of a one-dimensional tensor.
 */
static GstMemory *
_sparse_new_raw_memory (guint element_count, guint nnz, const guint *indices, gsize size)
{
  GstMemory *mem;
  gchar *dim_str = g_strdup_printf ("%u", element_count);

  mem = _sparse_new_raw_memory_dim (dim_str, nnz, indices, size);
  g_free (dim_str);

  return mem;
}

/**
 * @brief Test for tensor_sparse util, a memory too short to hold a meta header.
 * @details Without the length check gst_tensor_meta_info_parse_header() reads
 * 88 bytes out of the 64 the memory holds, which only valgrind reports.
 */
TEST (testTensorSparse, utilToDenseShortHeader_n)
{
  GstTensorMetaInfo meta;
  GstMemory *in, *out;

  in = _sparse_new_raw_memory (40U, 0U, NULL, 64U);
  ASSERT_TRUE (in != NULL);

  out = gst_tensor_sparse_to_dense (&meta, in);
  EXPECT_TRUE (out == NULL);

  gst_memory_unref (in);
}

/**
 * @brief Fill the given bytes with the start of a valid sparse int32 header.
 * @details The header declares 40 elements and no non-zero one, and is cut to
 * the given size, so that a memory can hold every field the parse reads while
 * being shorter than the header it describes.
 * @param data the bytes to fill
 * @param size the number of header bytes to copy, at most the header size
 */
static void
_sparse_fill_truncated_header (guint8 *data, gsize size)
{
  GstTensorMetaInfo meta;
  guint8 header[128];

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_INT32;
  gst_tensor_parse_dimension ("40", meta.dimension);
  meta.format = _NNS_TENSOR_FORMAT_SPARSE;
  meta.media_type = _NNS_TENSOR;
  meta.sparse_info.nnz = 0U;

  ASSERT_LE (gst_tensor_meta_info_get_header_size (&meta), sizeof (header));
  ASSERT_LE (size, sizeof (header));
  ASSERT_TRUE (gst_tensor_meta_info_update_header (&meta, header));
  memcpy (data, header, size);
}

/**
 * @brief Test for tensor_sparse util, a memory that holds every field of a
 *        header but is shorter than the header itself.
 * @details gst_tensor_meta_info_parse_header() reads 88 bytes, so a 100-byte
 * memory parses as a valid header whose payload starts at byte 128, past the
 * end of the memory. Without the length check map.size - header_size
 * underflows, the payload bound passes, and the memory decodes into a whole
 * tensor. utilToDenseShortHeader_n covers a memory shorter than what the parse
 * reads; this covers the one in between.
 */
TEST (testTensorSparse, utilToDenseTruncatedHeader_n)
{
  GstTensorMetaInfo meta;
  GstMemory *in, *out;
  guint8 *data;
  const gsize size = 100U;

  data = (guint8 *) g_malloc0 (size);
  _sparse_fill_truncated_header (data, size);
  in = gst_memory_new_wrapped ((GstMemoryFlags) 0, data, size, 0, size, data, g_free);

  out = gst_tensor_sparse_to_dense (&meta, in);
  EXPECT_TRUE (out == NULL);
  if (out)
    gst_memory_unref (out);

  gst_memory_unref (in);
}

/**
 * @brief Test for tensor_sparse util, a header declaring more data than the
 *        memory holds.
 */
TEST (testTensorSparse, utilToDenseShortPayload_n)
{
  GstTensorMetaInfo meta;
  GstMemory *in, *out;

  /* 1000 non-zero elements need 128 + 1000 * 8 bytes, the memory holds 200. */
  in = _sparse_new_raw_memory (40U, 1000U, NULL, 200U);
  ASSERT_TRUE (in != NULL);

  out = gst_tensor_sparse_to_dense (&meta, in);
  EXPECT_TRUE (out == NULL);

  gst_memory_unref (in);
}

/**
 * @brief Test for tensor_sparse util, a non-zero count that overflows the
 *        offset of the index array.
 */
TEST (testTensorSparse, utilToDenseNnzOverflow_n)
{
  GstTensorMetaInfo meta;
  GstMemory *in, *out;

  in = _sparse_new_raw_memory (40U, G_MAXUINT32, NULL, 136U);
  ASSERT_TRUE (in != NULL);

  out = gst_tensor_sparse_to_dense (&meta, in);
  EXPECT_TRUE (out == NULL);

  gst_memory_unref (in);
}

/**
 * @brief Test for tensor_sparse util, an index past the end of the dense tensor.
 * @details The index is the offset of the write into the decoded tensor, so
 * without the bound check this writes about 2 GB past a 160-byte allocation.
 */
TEST (testTensorSparse, utilToDenseIndexOutOfRange_n)
{
  GstTensorMetaInfo meta;
  GstMemory *in, *out;
  const guint indices[] = { 0x20000000U };

  in = _sparse_new_raw_memory (40U, 1U, indices, 136U);
  ASSERT_TRUE (in != NULL);

  out = gst_tensor_sparse_to_dense (&meta, in);
  EXPECT_TRUE (out == NULL);

  gst_memory_unref (in);
}

/**
 * @brief Test for tensor_sparse util, a header of a version this build cannot
 *        size.
 * @details gst_tensor_meta_info_validate() asks only for the magic of the
 * version marker, while gst_tensor_meta_info_get_header_size() answers 128 for
 * version 1 and 0 for every other. A header of an unknown version therefore
 * validated and then placed the payload at offset 0, so the header itself was
 * decoded as tensor data. Reported as item A7 of #4920, and refused here
 * because the offset of the payload is what this function is bounding.
 */
TEST (testTensorSparse, utilToDenseUnknownVersion_n)
{
  GstTensorMetaInfo meta;
  GstMemory *in, *out;
  GstMapInfo map;

  /* no non-zero element, so only the version decides the result */
  in = _sparse_new_raw_memory (40U, 0U, NULL, 136U);
  ASSERT_TRUE (in != NULL);

  /* the magic of the version marker, with a major version of 0 */
  ASSERT_TRUE (gst_memory_map (in, &map, GST_MAP_WRITE));
  ((uint32_t *) map.data)[1] = 0xDE000000U;
  gst_memory_unmap (in, &map);

  out = gst_tensor_sparse_to_dense (&meta, in);
  EXPECT_TRUE (out == NULL);

  gst_memory_unref (in);
}

/**
 * @brief Test for tensor_sparse util, a dimension whose byte size has overflowed.
 * @details gst_tensor_meta_info_get_data_size() multiplies the element count by
 * the element size in a gsize, so a dimension can name more elements than the
 * size of the tensor it computes: 3340214413 x 1380655685 int32 elements are
 * 2^62 + 1, whose 2^64 + 4 bytes wrap to 4. Bounding the indices by the element
 * count alone would let an index of 2^62 through and write it into a four-byte
 * allocation. The same wrap happens far sooner where gsize is 32 bits, which
 * the armv7l build of the Tizen target is.
 */
TEST (testTensorSparse, utilToDenseDimensionOverflow_n)
{
  GstTensorMetaInfo meta;
  GstMemory *in, *out;
  const guint indices[] = { 0x20000000U };

  in = _sparse_new_raw_memory_dim ("3340214413:1380655685", 1U, indices, 136U);
  ASSERT_TRUE (in != NULL);

  out = gst_tensor_sparse_to_dense (&meta, in);
  EXPECT_TRUE (out == NULL);

  gst_memory_unref (in);
}

/**
 * @brief Test for tensor_sparse util, the last valid index is still decoded.
 */
TEST (testTensorSparse, utilToDenseLastIndex)
{
  GstTensorMetaInfo meta;
  GstMemory *in, *out;
  GstMapInfo map;
  const guint indices[] = { 39U };

  in = _sparse_new_raw_memory (40U, 1U, indices, 136U);
  ASSERT_TRUE (in != NULL);

  out = gst_tensor_sparse_to_dense (&meta, in);
  ASSERT_TRUE (out != NULL);

  ASSERT_TRUE (gst_memory_map (out, &map, GST_MAP_READ));
  EXPECT_EQ (map.size, 40U * sizeof (gint32));
  gst_memory_unmap (out, &map);

  gst_memory_unref (out);
  gst_memory_unref (in);
}

/**
 * @brief Test for tensor_sparse util, a dense memory shorter than the meta info.
 * @details Without the length check the encoder reads 64 MB out of the 64 bytes
 * the memory holds.
 */
TEST (testTensorSparse, utilFromDenseShortMemory_n)
{
  GstTensorMetaInfo meta;
  GstMemory *in, *out;
  guint8 *data;
  const gsize data_size = 64U;

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_INT32;
  gst_tensor_parse_dimension ("16777216", meta.dimension);
  meta.media_type = _NNS_TENSOR;

  data = (guint8 *) g_malloc0 (data_size);
  in = gst_memory_new_wrapped (
      GST_MEMORY_FLAG_READONLY, data, data_size, 0, data_size, data, g_free);

  out = gst_tensor_sparse_from_dense (&meta, in);
  EXPECT_TRUE (out == NULL);

  gst_memory_unref (in);
}

/**
 * @brief Test for tensor_sparse_dec, a flexible buffer whose last bytes cannot
 *        hold a whole tensor.
 * @details gst_tensor_buffer_from_config() hands the bytes after the last
 * tensor it can size to the caller as one more memory, and leaves it to the
 * caller to refuse them. Here they are a 100-byte tail carrying a valid header,
 * which the decoder has to refuse after it has already decoded the tensor
 * before it. Without the length check the tail decodes into a second tensor,
 * the buffer no longer matches the negotiated config, and it is dropped with
 * GST_FLOW_OK instead of being reported. The split is checked first, so that
 * the case cannot keep passing by another route if that contract changes.
 */
TEST (testTensorSparse, decTrailingRemainder_n)
{
  GstHarness *h;
  GstMemory *sparse, *all;
  GstBuffer *in, *split;
  GstCaps *caps;
  GstMapInfo smap, amap;
  GstTensorInfo info;
  GstTensorsConfig config;
  gsize sparse_size;
  guint handler;
  const gsize tail = 100U;

  h = gst_harness_new ("tensor_sparse_dec");
  ASSERT_TRUE (h != NULL);

  gst_harness_set_sink_caps_str (h, SPARSE_DENSE_CAPS_STR);
  gst_harness_set_src_caps_str (h, "other/tensors,format=sparse,framerate=0/1");

  sparse = _sparse_new_sparse_memory (&info);
  ASSERT_TRUE (sparse != NULL);
  ASSERT_TRUE (gst_memory_map (sparse, &smap, GST_MAP_READ));

  /* one memory: a whole sparse tensor, then a tail shorter than a header */
  all = gst_allocator_alloc (NULL, smap.size + tail, NULL);
  ASSERT_TRUE (gst_memory_map (all, &amap, GST_MAP_WRITE));
  memcpy (amap.data, smap.data, smap.size);
  _sparse_fill_truncated_header (amap.data + smap.size, tail);
  sparse_size = smap.size;
  gst_memory_unmap (all, &amap);
  gst_memory_unmap (sparse, &smap);
  gst_memory_unref (sparse);

  in = gst_buffer_new ();
  gst_buffer_append_memory (in, all);

  /* the premise: the decoder is handed the tensor and the tail as two memories */
  caps = gst_caps_from_string ("other/tensors,format=sparse,framerate=0/1");
  ASSERT_TRUE (gst_tensors_config_from_caps (&config, caps, TRUE));
  gst_caps_unref (caps);
  split = gst_tensor_buffer_from_config (gst_buffer_ref (in), &config);
  ASSERT_TRUE (split != NULL);
  ASSERT_EQ (gst_buffer_n_memory (split), 2U);
  EXPECT_EQ (gst_memory_get_sizes (gst_buffer_peek_memory (split, 0), NULL, NULL), sparse_size);
  EXPECT_EQ (gst_memory_get_sizes (gst_buffer_peek_memory (split, 1), NULL, NULL), tail);
  gst_buffer_unref (split);
  gst_tensors_config_free (&config);

  handler = _sparse_watch_gst_critical ();
  EXPECT_EQ (gst_harness_push (h, in), GST_FLOW_ERROR);
  g_log_remove_handler ("GStreamer", handler);

  EXPECT_EQ (sparse_gst_critical_count, 0U) << sparse_gst_critical_msg;
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_tensor_info_free (&info);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_sparse_dec, a buffer that cannot be split into the
 *        tensors of the negotiated config.
 * @details The single memory carries two meta headers, the second of which
 * declares more data than is left, so gst_tensor_buffer_from_config() fails.
 */
TEST (testTensorSparse, decBufferFromConfigFailure_n)
{
  GstHarness *h;
  GstBuffer *in;
  GstMemory *mem;
  GstMapInfo map;
  GstTensorMetaInfo meta;
  guint handler;
  guint8 *data;
  const gsize data_size = 300U;

  h = gst_harness_new ("tensor_sparse_dec");
  ASSERT_TRUE (h != NULL);

  gst_harness_set_sink_caps_str (h, SPARSE_DENSE_CAPS_STR);
  gst_harness_set_src_caps_str (h, "other/tensors,format=sparse,framerate=0/1");

  data = (guint8 *) g_malloc0 (data_size);
  mem = gst_memory_new_wrapped (
      (GstMemoryFlags) 0, data, data_size, 0, data_size, data, g_free);
  ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_WRITE));

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_INT32;
  gst_tensor_parse_dimension ("40", meta.dimension);
  meta.format = _NNS_TENSOR_FORMAT_SPARSE;
  meta.media_type = _NNS_TENSOR;

  /* 128 + 2 * 8 bytes, so the second header starts at 144. */
  meta.sparse_info.nnz = 2U;
  gst_tensor_meta_info_update_header (&meta, map.data);

  /* 128 + 1000 * 8 bytes, far past the 156 bytes that are left. */
  meta.sparse_info.nnz = 1000U;
  gst_tensor_meta_info_update_header (&meta, map.data + 144U);

  gst_memory_unmap (mem, &map);

  in = gst_buffer_new ();
  gst_buffer_append_memory (in, mem);

  handler = _sparse_watch_gst_critical ();
  EXPECT_EQ (gst_harness_push (h, in), GST_FLOW_ERROR);
  g_log_remove_handler ("GStreamer", handler);

  EXPECT_EQ (sparse_gst_critical_count, 0U) << sparse_gst_critical_msg;
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_sparse_enc, a buffer smaller than the negotiated caps.
 */
TEST (testTensorSparse, encBufferFromConfigFailure_n)
{
  GstHarness *h;
  GstMemory *mem;
  gpointer data;
  guint handler;
  const gsize data_size = 16U;

  h = gst_harness_new ("tensor_sparse_enc");
  ASSERT_TRUE (h != NULL);

  /* Two tensors of 160 bytes are declared, one memory of 16 bytes is pushed. */
  gst_harness_set_src_caps_str (h, "other/tensors,format=static,num_tensors=2,"
                                   "dimensions=(string)40:1:1:1.40:1:1:1,types=(string)int32.int32,"
                                   "framerate=0/1");

  data = g_malloc0 (data_size);
  mem = gst_memory_new_wrapped (
      GST_MEMORY_FLAG_READONLY, data, data_size, 0, data_size, data, g_free);

  handler = _sparse_watch_gst_critical ();
  EXPECT_EQ (gst_harness_push (h, _sparse_new_buffer (mem)), GST_FLOW_ERROR);
  g_log_remove_handler ("GStreamer", handler);

  EXPECT_EQ (sparse_gst_critical_count, 0U) << sparse_gst_critical_msg;
  EXPECT_EQ (gst_harness_buffers_received (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Number of the glib critical logs raised by gst_pad_set_caps().
 */
static guint sparse_setcaps_critical_count;

/**
 * @brief Log handler counting the caps assertions of gst_pad_set_caps().
 * @details gst_pad_set_caps() is a static inline of gstcompat.h, so it is
 * compiled into the caller rather than into GStreamer, and nnstreamer defines
 * no G_LOG_DOMAIN of its own; its assertion therefore lands in the default
 * domain, where the elements' own error messages are. Matching the name of the
 * function is what keeps the two apart.
 */
static void
_sparse_count_setcaps_critical (const gchar *domain, GLogLevelFlags level,
    const gchar *message, gpointer udata)
{
  UNUSED (domain);
  UNUSED (udata);

  if ((level & G_LOG_LEVEL_CRITICAL) && message && strstr (message, "gst_pad_set_caps") != NULL)
    sparse_setcaps_critical_count++;
}

/**
 * @brief Start counting the caps assertions, proving the counter live first.
 */
static guint
_sparse_watch_setcaps_critical (void)
{
  guint handler = g_log_set_handler (NULL,
      (GLogLevelFlags) (G_LOG_LEVEL_CRITICAL | G_LOG_FLAG_FATAL | G_LOG_FLAG_RECURSION),
      _sparse_count_setcaps_critical, NULL);

  sparse_setcaps_critical_count = 0;
  g_critical ("tensor_sparse test: gst_pad_set_caps counter self-check");
  EXPECT_EQ (sparse_setcaps_critical_count, 1U);

  sparse_setcaps_critical_count = 0;
  return handler;
}

/**
 * @brief Test for tensor_sparse_dec, caps that cannot be read into a config are
 *        refused rather than kept.
 * @details gst_harness_push_event() reports success either way, so what the
 * refusal changes is that the pad does not end up carrying caps the element
 * cannot work with.
 */
TEST (testTensorSparse, decUnreadableCaps_n)
{
  GstHarness *h;
  GstPad *sinkpad;
  GstCaps *caps, *current;

  h = gst_harness_new ("tensor_sparse_dec");
  ASSERT_TRUE (h != NULL);

  /* a sparse stream with no framerate, which no config can be built from */
  caps = gst_caps_from_string ("other/tensors,format=sparse");
  gst_harness_push_event (h, gst_event_new_caps (caps));
  gst_caps_unref (caps);

  sinkpad = gst_element_get_static_pad (h->element, "sink");
  ASSERT_TRUE (sinkpad != NULL);

  current = gst_pad_get_current_caps (sinkpad);
  EXPECT_TRUE (current == NULL);
  if (current)
    gst_caps_unref (current);

  gst_object_unref (sinkpad);
  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_sparse_dec, a downstream that leaves the decoded caps
 *        unfixed is negotiated without a caps assertion.
 * @details The tensors a sparse buffer decodes into are described by the buffer
 * itself, so a downstream that does not constrain them leaves the config read
 * back from the peer unfixed, and the caps built from it are not fixed either.
 * They used to go straight into gst_pad_set_caps(), which rejects them with an
 * assertion. The SSAT cases of tests/nnstreamer_sparse carry a caps filter that
 * fixes the caps, which is why nothing had caught this.
 */
TEST (testTensorSparse, decUnfixedSrcCaps_n)
{
  GstHarness *h;
  guint handler;

  h = gst_harness_new ("tensor_sparse_dec");
  ASSERT_TRUE (h != NULL);

  handler = _sparse_watch_setcaps_critical ();
  /* no sink caps, so the peer of the src pad keeps its template caps */
  gst_harness_set_src_caps_str (h, "other/tensors,format=sparse,framerate=0/1");
  g_log_remove_handler (NULL, handler);

  EXPECT_EQ (sparse_setcaps_critical_count, 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Test for tensor_sparse_dec, a pipeline that never negotiates fails
 *        without raising a critical.
 * @details tensor_sparse_enc does not carry the framerate of the stream into
 * its sparse src caps, so a decoder linked to it directly is negotiated with a
 * config it cannot validate. gst_tensor_buffer_from_config() then returns NULL
 * for every buffer, which the chain used to count the tensors of and unref, so
 * glib reported two critical logs of the GStreamer domain per buffer and an
 * empty buffer went downstream. The SSAT cases of tests/nnstreamer_sparse cover
 * the same pipeline with the caps filter that makes it work.
 */
TEST (testTensorSparse, decUnnegotiatedPipeline_n)
{
  GstElement *pipeline;
  GstBus *bus;
  GstMessage *msg;
  guint handler;
  const gchar *str_pipeline
      = "videotestsrc num-buffers=1 ! "
        "video/x-raw,format=RGB,width=10,height=10,framerate=0/1 ! videoconvert ! "
        "tensor_converter ! tensor_sparse_enc ! tensor_sparse_dec ! fakesink";

  pipeline = gst_parse_launch (str_pipeline, NULL);
  ASSERT_TRUE (pipeline != nullptr);

  handler = _sparse_watch_gst_critical ();

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);

  bus = gst_element_get_bus (pipeline);
  msg = gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  /* the pipeline has to settle either way, so a time-out is a failure of its own */
  EXPECT_TRUE (msg != nullptr);
  if (msg)
    gst_message_unref (msg);
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  g_log_remove_handler ("GStreamer", handler);

  EXPECT_EQ (sparse_gst_critical_count, 0U) << sparse_gst_critical_msg;

  gst_object_unref (pipeline);
}

/**
 * @brief The number of tensors of the tensor_sparse extra-tensor test.
 */
#define SPARSE_EXTRA_TENSORS_NUM (18U)

/**
 * @brief Create a sparse tensor memory as long as the dense tensor it encodes.
 * @details Four non-zero elements of 40 int32 encode into 128 + 4 * 8 bytes,
 * exactly the 160 bytes the static tensor info declares. The two sizes have to
 * agree because gst_tensor_buffer_append_memory() records only the dense size
 * of an extra tensor, which truncates a longer sparse memory (item A6 of #4920,
 * issue #4934).
 */
static GstMemory *
_sparse_new_extra_memory (GstTensorInfo *info)
{
  GstMemory *dense, *sparse;
  GstTensorMetaInfo meta;
  gpointer data;
  gsize data_size;
  guint i;

  gst_tensor_info_init (info);
  info->type = _NNS_INT32;
  gst_tensor_parse_dimension ("40", info->dimension);

  data_size = gst_tensor_info_get_size (info);
  data = g_malloc0 (data_size);
  for (i = 0; i < 40U; i++)
    ((gint32 *) data)[i] = (i % 10U == 0U) ? (gint32) (i + 1) : 0;

  dense = gst_memory_new_wrapped (
      GST_MEMORY_FLAG_READONLY, data, data_size, 0, data_size, data, g_free);

  gst_tensor_info_convert_to_meta (info, &meta);
  meta.format = _NNS_TENSOR_FORMAT_SPARSE;
  meta.media_type = _NNS_TENSOR;

  sparse = gst_tensor_sparse_from_dense (&meta, dense);
  gst_memory_unref (dense);

  return sparse;
}

/**
 * @brief Test for tensor_sparse_dec, a buffer of more tensors than
 *        NNS_TENSOR_MEMORY_MAX is decoded and its tensors info is released.
 * @details The decoder allocates GstTensorsInfo.extra for the 17th tensor, so
 * this is the case in which the tensors info of the chain function has to be
 * freed. The leak itself is reported by the valgrind step of the CI.
 */
TEST (testTensorSparse, decExtraTensors)
{
  GstHarness *h;
  GstBuffer *in, *out;
  GstMemory *mem;
  GstMapInfo map;
  GstTensorsConfig config;
  GstTensorInfo *_info;
  guint i, j;

  h = gst_harness_new ("tensor_sparse_dec");
  ASSERT_TRUE (h != NULL);

  gst_tensors_config_init (&config);
  config.info.num_tensors = SPARSE_EXTRA_TENSORS_NUM;
  config.rate_n = 0;
  config.rate_d = 1;

  for (i = 0; i < SPARSE_EXTRA_TENSORS_NUM; i++) {
    _info = gst_tensors_info_get_nth_info (&config.info, i);
    _info->type = _NNS_INT32;
    gst_tensor_parse_dimension ("40", _info->dimension);
  }

  gst_harness_set_sink_caps (h, gst_tensors_caps_from_config (&config));
  gst_harness_set_src_caps_str (h, "other/tensors,format=sparse,framerate=0/1");

  in = gst_buffer_new ();
  for (i = 0; i < SPARSE_EXTRA_TENSORS_NUM; i++) {
    GstTensorInfo info;

    mem = _sparse_new_extra_memory (&info);
    ASSERT_TRUE (mem != NULL);
    ASSERT_TRUE (gst_tensor_buffer_append_memory (
        in, mem, gst_tensors_info_get_nth_info (&config.info, i)));
    gst_tensor_info_free (&info);
  }
  ASSERT_EQ (gst_tensor_buffer_get_count (in), SPARSE_EXTRA_TENSORS_NUM);

  ASSERT_EQ (gst_harness_push (h, in), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 1U);

  out = gst_harness_pull (h);
  ASSERT_TRUE (out != NULL);
  ASSERT_EQ (gst_tensor_buffer_get_count (out), SPARSE_EXTRA_TENSORS_NUM);

  for (i = 0; i < SPARSE_EXTRA_TENSORS_NUM; i++) {
    mem = gst_tensor_buffer_get_nth_memory (out, i);
    ASSERT_TRUE (mem != NULL);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
    ASSERT_EQ (map.size, 40U * sizeof (gint32));
    for (j = 0; j < 40U; j++)
      EXPECT_EQ (((gint32 *) map.data)[j], (j % 10U == 0U) ? (gint32) (j + 1) : 0);
    gst_memory_unmap (mem, &map);
    gst_memory_unref (mem);
  }

  gst_buffer_unref (out);
  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
}

/**
 * @brief Rendezvous used to set a property from the test thread exactly while
 *        the streaming thread is adding a source pad.
 */
typedef struct {
  GMutex lock;
  GCond cond;
  gboolean pad_added;
  gboolean prop_set;
  guint pads_added;
  guint pads_at_no_more;
} padRaceSync;

/**
 * @brief Handler of "pad-added", which runs on the streaming thread.
 */
static void
_pad_race_pad_added (GstElement *element, GstPad *pad, gpointer user_data)
{
  padRaceSync *sync = (padRaceSync *) user_data;
  gint64 until = g_get_monotonic_time () + TEST_TIMEOUT_LIMIT;

  UNUSED (element);
  UNUSED (pad);

  g_mutex_lock (&sync->lock);
  sync->pad_added = TRUE;
  sync->pads_added++;
  g_cond_broadcast (&sync->cond);
  while (!sync->prop_set) {
    if (!g_cond_wait_until (&sync->cond, &sync->lock, until))
      break;
  }
  g_mutex_unlock (&sync->lock);
}

/**
 * @brief Set the given property while the streaming thread is blocked in
 *        _pad_race_pad_added(), then release the streaming thread.
 * @return TRUE if the element reported a new pad before the time-out
 */
static gboolean
_pad_race_set_property (padRaceSync *sync, GstElement *element,
    const gchar *name, const gchar *value)
{
  gint64 until = g_get_monotonic_time () + TEST_TIMEOUT_LIMIT;
  gboolean pad_added;

  g_mutex_lock (&sync->lock);
  while (!sync->pad_added) {
    if (!g_cond_wait_until (&sync->cond, &sync->lock, until))
      break;
  }
  pad_added = sync->pad_added;
  g_mutex_unlock (&sync->lock);

  if (pad_added)
    g_object_set (G_OBJECT (element), name, value, NULL);

  g_mutex_lock (&sync->lock);
  sync->prop_set = TRUE;
  g_cond_broadcast (&sync->cond);
  g_mutex_unlock (&sync->lock);

  return pad_added;
}

/**
 * @brief Handler of "no-more-pads", which runs on the streaming thread.
 */
static void
_pad_race_no_more_pads (GstElement *element, gpointer user_data)
{
  padRaceSync *sync = (padRaceSync *) user_data;

  UNUSED (element);

  g_mutex_lock (&sync->lock);
  sync->pads_at_no_more = sync->pads_added;
  g_cond_broadcast (&sync->cond);
  g_mutex_unlock (&sync->lock);
}

/**
 * @brief Wait until the element reports that it has no more pads.
 * @return the number of pads it had added by then, 0 if it never reported
 */
static guint
_pad_race_wait_no_more_pads (padRaceSync *sync)
{
  gint64 until = g_get_monotonic_time () + TEST_TIMEOUT_LIMIT;
  guint pads;

  g_mutex_lock (&sync->lock);
  while (sync->pads_at_no_more == 0) {
    if (!g_cond_wait_until (&sync->cond, &sync->lock, until))
      break;
  }
  pads = sync->pads_at_no_more;
  g_mutex_unlock (&sync->lock);

  return pads;
}

/**
 * @brief Initialize the rendezvous.
 */
static void
_pad_race_sync_init (padRaceSync *sync)
{
  g_mutex_init (&sync->lock);
  g_cond_init (&sync->cond);
  sync->pad_added = FALSE;
  sync->prop_set = FALSE;
  sync->pads_added = 0;
  sync->pads_at_no_more = 0;
}

/**
 * @brief Release the rendezvous.
 */
static void
_pad_race_sync_clear (padRaceSync *sync)
{
  g_mutex_clear (&sync->lock);
  g_cond_clear (&sync->cond);
}

/**
 * @brief Set tensorpick while tensor_demux is creating its first source pad.
 * @details The application thread may touch a property at any time, and the
 *          first buffer keeps the streaming thread inside the pad creation for
 *          a long while (caps negotiation and the delayed link downstream).
 */
TEST (testTensorDemux, setTensorpickWhileAddingPad)
{
  gchar *pipeline_desc;
  GstElement *pipeline, *demux, *sink;
  padRaceSync sync;
  guint data_received = 0;
  gchar *tensorpick = NULL;

  _pad_race_sync_init (&sync);

  pipeline_desc = g_strdup (
      "videotestsrc num-buffers=3 ! "
      "video/x-raw,format=RGB,width=16,height=16,framerate=30/1 ! "
      "tensor_converter ! tensor_demux name=demux ! tensor_sink name=sinkx");
  pipeline = gst_parse_launch (pipeline_desc, NULL);
  g_free (pipeline_desc);
  ASSERT_TRUE (pipeline != NULL);

  demux = gst_bin_get_by_name (GST_BIN (pipeline), "demux");
  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_TRUE (demux != NULL);
  ASSERT_TRUE (sink != NULL);

  g_signal_connect (demux, "pad-added", G_CALLBACK (_pad_race_pad_added), &sync);
  g_signal_connect (sink, "new-data", G_CALLBACK (count_output), &data_received);

  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  EXPECT_TRUE (_pad_race_set_property (&sync, demux, "tensorpick", "0"));
  EXPECT_TRUE (wait_pipeline_process_buffers (&data_received, 1, TEST_TIMEOUT_LIMIT_MS));

  g_object_get (demux, "tensorpick", &tensorpick, NULL);
  EXPECT_STREQ (tensorpick, "0");
  g_free (tensorpick);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (sink);
  gst_object_unref (demux);
  gst_object_unref (pipeline);
  _pad_race_sync_clear (&sync);
}

/**
 * @brief Setting tensorpick again replaces the previous selection.
 */
TEST (testTensorDemux, setTensorpickTwice)
{
  GstElement *demux = gst_element_factory_make ("tensor_demux", NULL);
  gchar *tensorpick = NULL;

  ASSERT_TRUE (demux != NULL);
  gst_object_ref_sink (demux);

  g_object_set (demux, "tensorpick", "0,1", NULL);
  g_object_set (demux, "tensorpick", "2", NULL);

  g_object_get (demux, "tensorpick", &tensorpick, NULL);
  EXPECT_STREQ (tensorpick, "2");
  g_free (tensorpick);

  gst_object_unref (demux);
}

/**
 * @brief Set tensorseg while tensor_split is creating its first source pad.
 */
TEST (testTensorSplit, setTensorsegWhileAddingPad)
{
  gchar *pipeline_desc;
  GstElement *pipeline, *split, *sink;
  padRaceSync sync;
  guint data_received = 0;

  _pad_race_sync_init (&sync);

  pipeline_desc = g_strdup ("videotestsrc num-buffers=5 ! "
                            "video/x-raw,format=RGB,width=4,height=4,framerate=30/1 ! "
                            "tensor_converter ! tensor_split name=split tensorseg=3:4:4 ! "
                            "tensor_sink name=sinkx");
  pipeline = gst_parse_launch (pipeline_desc, NULL);
  g_free (pipeline_desc);
  ASSERT_TRUE (pipeline != NULL);

  split = gst_bin_get_by_name (GST_BIN (pipeline), "split");
  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_TRUE (split != NULL);
  ASSERT_TRUE (sink != NULL);

  g_signal_connect (split, "pad-added", G_CALLBACK (_pad_race_pad_added), &sync);
  g_signal_connect (sink, "new-data", G_CALLBACK (count_output), &data_received);

  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  EXPECT_TRUE (_pad_race_set_property (&sync, split, "tensorseg", "1:4:4,2:4:4"));
  EXPECT_TRUE (wait_pipeline_process_buffers (&data_received, 1, TEST_TIMEOUT_LIMIT_MS));

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (sink);
  gst_object_unref (split);
  gst_object_unref (pipeline);
  _pad_race_sync_clear (&sync);
}

/**
 * @brief Setting tensorseg again replaces the previous rule.
 */
TEST (testTensorSplit, setTensorsegTwice)
{
  GstElement *split = gst_element_factory_make ("tensor_split", NULL);
  gchar *tensorseg = NULL;
  gchar **strv;

  ASSERT_TRUE (split != NULL);
  gst_object_ref_sink (split);

  g_object_set (split, "tensorseg", "1:4:4,1:4:4", NULL);
  g_object_set (split, "tensorseg", "1:4:4,1:4:4,1:4:4", NULL);

  g_object_get (split, "tensorseg", &tensorseg, NULL);
  strv = g_strsplit (tensorseg, ",", -1);
  EXPECT_EQ (g_strv_length (strv), 3U);
  g_strfreev (strv);
  g_free (tensorseg);

  g_object_set (split, "tensorseg", "2:4:4", NULL);
  g_object_get (split, "tensorseg", &tensorseg, NULL);
  strv = g_strsplit (tensorseg, ",", -1);
  EXPECT_EQ (g_strv_length (strv), 1U);
  g_strfreev (strv);
  g_free (tensorseg);

  gst_object_unref (split);
}

/**
 * @brief A tensorseg rule longer than the tensor rank limit is truncated.
 */
TEST (testTensorSplit, setTensorsegOverRank_n)
{
  GstElement *split = gst_element_factory_make ("tensor_split", NULL);
  gchar *tensorseg = NULL;
  guint i;
  GString *param = g_string_new (NULL);
  GString *expected = g_string_new (NULL);

  ASSERT_TRUE (split != NULL);
  gst_object_ref_sink (split);

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT + 4; i++) {
    if (i > 0)
      g_string_append_c (param, ':');
    g_string_append_printf (param, "%u", i + 1);
  }
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    if (i > 0)
      g_string_append_c (expected, ':');
    g_string_append_printf (expected, "%u", i + 1);
  }

  g_object_set (split, "tensorseg", param->str, NULL);

  g_object_get (split, "tensorseg", &tensorseg, NULL);
  EXPECT_STREQ (tensorseg, expected->str);
  g_free (tensorseg);

  g_string_free (param, TRUE);
  g_string_free (expected, TRUE);
  gst_object_unref (split);
}

/**
 * @brief Setting tensorpick again replaces the previous selection.
 */
TEST (testTensorSplit, setTensorpickTwice)
{
  GstElement *split = gst_element_factory_make ("tensor_split", NULL);
  gchar *tensorpick = NULL;

  ASSERT_TRUE (split != NULL);
  gst_object_ref_sink (split);

  g_object_set (split, "tensorpick", "0,1", NULL);
  g_object_set (split, "tensorpick", "2", NULL);

  g_object_get (split, "tensorpick", &tensorpick, NULL);
  EXPECT_STREQ (tensorpick, "2");
  g_free (tensorpick);

  gst_object_unref (split);
}

/**
 * @brief Set tensorpick while tensor_split is creating its first source pad.
 * @details The pad creation decides whether to report no-more-pads by comparing
 *          the pick count with the pads created so far. Reading the property
 *          instead of the snapshot the buffer is being split by ends that
 *          report after src_0, and src_1 is then added behind it.
 */
TEST (testTensorSplit, setTensorpickWhileAddingPad)
{
  gchar *pipeline_desc;
  GstElement *pipeline, *split, *sink;
  padRaceSync sync;
  guint data_received = 0;
  gchar *tensorpick = NULL;

  _pad_race_sync_init (&sync);

  pipeline_desc = g_strdup ("videotestsrc num-buffers=5 ! "
                            "video/x-raw,format=RGB,width=4,height=4,framerate=30/1 ! "
                            "tensor_converter ! tensor_split name=split tensorseg=1:4:4,2:4:4 "
                            "tensorpick=0,1 ! tensor_sink name=sinkx");
  pipeline = gst_parse_launch (pipeline_desc, NULL);
  g_free (pipeline_desc);
  ASSERT_TRUE (pipeline != NULL);

  split = gst_bin_get_by_name (GST_BIN (pipeline), "split");
  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  ASSERT_TRUE (split != NULL);
  ASSERT_TRUE (sink != NULL);

  g_signal_connect (split, "pad-added", G_CALLBACK (_pad_race_pad_added), &sync);
  g_signal_connect (split, "no-more-pads", G_CALLBACK (_pad_race_no_more_pads), &sync);
  g_signal_connect (sink, "new-data", G_CALLBACK (count_output), &data_received);

  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  EXPECT_TRUE (_pad_race_set_property (&sync, split, "tensorpick", "0"));
  EXPECT_TRUE (wait_pipeline_process_buffers (&data_received, 1, TEST_TIMEOUT_LIMIT_MS));
  EXPECT_EQ (_pad_race_wait_no_more_pads (&sync), 2U);

  g_object_get (split, "tensorpick", &tensorpick, NULL);
  EXPECT_STREQ (tensorpick, "0");
  g_free (tensorpick);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (sink);
  gst_object_unref (split);
  gst_object_unref (pipeline);
  _pad_race_sync_clear (&sync);
}

/**
 * @brief A buffer arriving before tensorseg is set is rejected and released.
 */
TEST (testTensorSplit, pushWithoutTensorseg_n)
{
  GstHarness *h = gst_harness_new_with_padnames ("tensor_split", "sink", NULL);
  GstTensorsConfig config;
  GstBuffer *buf;

  ASSERT_TRUE (h != NULL);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:4:4", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;
  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  buf = gst_harness_create_buffer (h, gst_tensors_info_get_size (&config.info, 0));
  gst_buffer_ref (buf);
  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);
  EXPECT_EQ (GST_MINI_OBJECT_REFCOUNT_VALUE (buf), 1);
  gst_buffer_unref (buf);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
}

/**
 * @brief A tensorseg asking for more bytes than the incoming tensor is refused.
 * @details Nothing bounds the copy against the input, and with HAVE_ORC the
 *          copy runs through orc_memcpy(), where neither a sanitizer nor
 *          valgrind sees the over-read - so the element has to say no itself.
 */
TEST (testTensorSplit, pushOversizedTensorseg_n)
{
  GstHarness *h = gst_harness_new_with_padnames ("tensor_split", "sink", NULL);
  GstTensorsConfig config;
  GstBuffer *buf;

  ASSERT_TRUE (h != NULL);

  g_object_set (h->element, "tensorseg", "6:4:4", NULL);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:4:4", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;
  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  buf = gst_harness_create_buffer (h, gst_tensors_info_get_size (&config.info, 0));
  gst_buffer_ref (buf);
  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);
  EXPECT_EQ (GST_MINI_OBJECT_REFCOUNT_VALUE (buf), 1);
  gst_buffer_unref (buf);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
}

/**
 * @brief Disposing twice does not free the tensorpick list twice.
 * @details GObject allows dispose to run more than once, so it has to leave
 *          the member it released in a state the next run can survive.
 */
TEST (testTensorDemux, disposeTwice)
{
  GstElement *demux = gst_element_factory_make ("tensor_demux", NULL);

  ASSERT_TRUE (demux != NULL);
  gst_object_ref_sink (demux);

  g_object_set (demux, "tensorpick", "0,1", NULL);
  g_object_run_dispose (G_OBJECT (demux));
  g_object_run_dispose (G_OBJECT (demux));

  gst_object_unref (demux);
}

/**
 * @brief A tensorpick naming a tensor the buffer does not have is refused.
 * @details This is the only way into the chain function's error exit, so it is
 *          also what puts a buffer on the path that used to leak the split
 *          pick string on the way out.
 */
TEST (testTensorDemux, pushOutOfRangeTensorpick_n)
{
  GstHarness *h = gst_harness_new_with_padnames ("tensor_demux", "sink", NULL);
  GstTensorsConfig config;
  GstBuffer *buf;

  ASSERT_TRUE (h != NULL);

  g_object_set (h->element, "tensorpick", "5", NULL);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:4:4", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;
  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  buf = gst_harness_create_buffer (h, gst_tensors_info_get_size (&config.info, 0));
  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
}

/**
 * @brief A source pad with no segment rule left for it is refused.
 * @details The rule array is replaced on every tensorseg set, so it can shrink
 *          under source pads that already exist. The pad created after that
 *          takes the next ordinal, which is then past the end of the array.
 */
TEST (testTensorSplit, addPadBeyondTensorseg_n)
{
  GstHarness *h = gst_harness_new_with_padnames ("tensor_split", "sink", NULL);
  GstTensorsConfig config;
  gsize data_size;

  ASSERT_TRUE (h != NULL);

  gst_tensors_config_init (&config);
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("2:8:8", config.info.info[0].dimension);
  config.rate_n = 0;
  config.rate_d = 1;
  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_size = gst_tensors_info_get_size (&config.info, 0);

  /* picking the second of two rules puts src_0 at ordinal 0 of two */
  g_object_set (h->element, "tensorseg", "1:8:8,1:8:8", "tensorpick", "1", NULL);
  EXPECT_NE (gst_harness_push (h, gst_harness_create_buffer (h, data_size)), GST_FLOW_ERROR);

  /* one rule now, but src_0 already took ordinal 0 */
  g_object_set (h->element, "tensorseg", "2:8:8", "tensorpick", "0", NULL);
  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size)), GST_FLOW_ERROR);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
}

/* set by _record_critical() when a critical is logged */
static gboolean tensor_split_logged_critical;

/**
 * @brief Log handler that records whether a critical was issued.
 */
static void
_record_critical (const gchar *domain, GLogLevelFlags level,
    const gchar *message, gpointer user_data)
{
  UNUSED (domain);
  UNUSED (message);
  UNUSED (user_data);

  if (level & G_LOG_LEVEL_CRITICAL)
    tensor_split_logged_critical = TRUE;
}

/**
 * @brief Releasing a tensor_split that never got a tensorseg says nothing.
 * @details finalize releases the rule array, which is unset here; releasing an
 *          unset one the wrong way logs a critical rather than doing nothing.
 */
TEST (testTensorSplit, finalizeWithoutTensorseg)
{
  GstElement *split = gst_element_factory_make ("tensor_split", NULL);
  guint handler;
  GLogLevelFlags fatal_mask;

  ASSERT_TRUE (split != NULL);
  gst_object_ref_sink (split);

  /* g_array_unref() reports invalid arguments in the GLib domain. */
  handler = g_log_set_handler ("GLib",
      (GLogLevelFlags) (G_LOG_LEVEL_CRITICAL | G_LOG_FLAG_FATAL | G_LOG_FLAG_RECURSION),
      _record_critical, NULL);
  tensor_split_logged_critical = FALSE;
  /* Only the intentional probe may bypass G_DEBUG=fatal-criticals. */
  fatal_mask = g_log_set_always_fatal ((GLogLevelFlags) G_LOG_FATAL_MASK);
  g_log ("GLib", G_LOG_LEVEL_CRITICAL, "tensor_split test: handler self-check");
  g_log_set_always_fatal (fatal_mask);
  EXPECT_TRUE (tensor_split_logged_critical);

  tensor_split_logged_critical = FALSE;
  gst_object_unref (split);
  g_log_remove_handler ("GLib", handler);

  EXPECT_FALSE (tensor_split_logged_critical);
}

/**
 * @brief The dimension of the tensor the decoder input cases negotiate
 */
#define TEST_DECODER_DIM "3:16:16"

/* how many times the custom decoder below was called, and with what size */
static guint decoder_custom_invoked;
static gsize decoder_custom_size;

/**
 * @brief Custom decoder recording what the element handed over
 */
static int
_decoder_custom_cb (const GstTensorMemory *input,
    const GstTensorsConfig *config, void *data, GstBuffer *out_buf)
{
  UNUSED (config);
  UNUSED (data);

  decoder_custom_invoked++;
  decoder_custom_size = input[0].size;

  gst_buffer_append_memory (out_buf, gst_allocator_alloc (NULL, 4, NULL));

  return GST_FLOW_OK;
}

/**
 * @brief Get a harness of a tensor_decoder negotiated with the given config
 */
static GstHarness *
_get_decoder_harness (const gchar *mode, const gchar *option1, GstTensorsConfig *config)
{
  GstElement *dec = gst_element_factory_make ("tensor_decoder", NULL);
  GstHarness *h;

  if (!dec)
    return NULL;
  gst_object_ref_sink (dec);

  /* the mode should be selected before the caps are negotiated */
  g_object_set (dec, "mode", mode, NULL);
  if (option1)
    g_object_set (dec, "option1", option1, NULL);

  h = gst_harness_new_with_element (dec, "sink", "src");
  gst_object_unref (dec);

  if (h)
    gst_harness_set_src_caps (h, gst_tensors_caps_from_config (config));

  return h;
}

/**
 * @brief Get a static tensors config of a single uint8 tensor
 */
static void
_get_decoder_config (GstTensorsConfig *config)
{
  gst_tensors_config_init (config);
  config->info.num_tensors = 1;
  config->info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension (TEST_DECODER_DIM, config->info.info[0].dimension);
  config->rate_n = 0;
  config->rate_d = 1;
}

/**
 * @brief Get a flexible tensor buffer, of which the meta info describes
 *        @a data_size bytes while the memory holds @a mem_size bytes
 */
static GstBuffer *
_get_flex_decoder_buffer (GstHarness *h, gsize data_size, gsize mem_size)
{
  GstTensorMetaInfo meta;
  GstBuffer *buf;
  GstMapInfo map;

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_UINT8;
  meta.dimension[0] = (uint32_t) data_size;
  meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  buf = gst_harness_create_buffer (h, mem_size);

  if (mem_size >= gst_tensor_meta_info_get_header_size (&meta)) {
    if (!gst_buffer_map (buf, &map, GST_MAP_WRITE)) {
      gst_buffer_unref (buf);
      return NULL;
    }
    gst_tensor_meta_info_update_header (&meta, map.data);
    gst_buffer_unmap (buf, &map);
  }

  return buf;
}

/**
 * @brief The negotiated tensor is handed to the decoder sub-plugin as it is.
 */
TEST (testTensorDecoder, pushInputSize)
{
  GstTensorsConfig config;
  GstHarness *h;
  gsize data_size;

  ASSERT_EQ (0, nnstreamer_decoder_custom_register ("tdec_size", _decoder_custom_cb, NULL));

  _get_decoder_config (&config);
  h = _get_decoder_harness ("custom-code", "tdec_size", &config);
  ASSERT_TRUE (h != NULL);

  data_size = gst_tensors_info_get_size (&config.info, 0);
  decoder_custom_invoked = 0;

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size)), GST_FLOW_OK);
  EXPECT_EQ (decoder_custom_invoked, 1U);
  EXPECT_EQ (decoder_custom_size, data_size);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  EXPECT_EQ (0, nnstreamer_decoder_custom_unregister ("tdec_size"));
}

/**
 * @brief A buffer larger than the negotiated tensor is refused.
 * @details The sub-plugin sizes its output from the caps while it reads as
 *          many bytes as the incoming memory holds, so the surplus of an
 *          oversized buffer is written past the end of that output.
 */
TEST (testTensorDecoder, pushInputSizeTooLarge_n)
{
  GstTensorsConfig config;
  GstHarness *h;
  gsize data_size;

  ASSERT_EQ (0, nnstreamer_decoder_custom_register ("tdec_size", _decoder_custom_cb, NULL));

  _get_decoder_config (&config);
  h = _get_decoder_harness ("custom-code", "tdec_size", &config);
  ASSERT_TRUE (h != NULL);

  data_size = gst_tensors_info_get_size (&config.info, 0);
  decoder_custom_invoked = 0;

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size + 1)), GST_FLOW_ERROR);
  EXPECT_EQ (decoder_custom_invoked, 0U);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  EXPECT_EQ (0, nnstreamer_decoder_custom_unregister ("tdec_size"));
}

/**
 * @brief A buffer smaller than the negotiated tensor is refused.
 */
TEST (testTensorDecoder, pushInputSizeTooSmall_n)
{
  GstTensorsConfig config;
  GstHarness *h;
  gsize data_size;

  ASSERT_EQ (0, nnstreamer_decoder_custom_register ("tdec_size", _decoder_custom_cb, NULL));

  _get_decoder_config (&config);
  h = _get_decoder_harness ("custom-code", "tdec_size", &config);
  ASSERT_TRUE (h != NULL);

  data_size = gst_tensors_info_get_size (&config.info, 0);
  decoder_custom_invoked = 0;

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size / 2)), GST_FLOW_ERROR);
  EXPECT_EQ (decoder_custom_invoked, 0U);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  EXPECT_EQ (0, nnstreamer_decoder_custom_unregister ("tdec_size"));
}

/**
 * @brief A buffer holding more memory chunks than the caps have tensors is refused.
 * @details The element used to assert on this, which aborts the process for a
 *          buffer an application or a remote peer has built.
 */
TEST (testTensorDecoder, pushInputMemoryCount_n)
{
  GstTensorsConfig config;
  GstHarness *h;
  GstBuffer *buf;
  gsize data_size;

  ASSERT_EQ (0, nnstreamer_decoder_custom_register ("tdec_size", _decoder_custom_cb, NULL));

  _get_decoder_config (&config);
  h = _get_decoder_harness ("custom-code", "tdec_size", &config);
  ASSERT_TRUE (h != NULL);

  data_size = gst_tensors_info_get_size (&config.info, 0);
  decoder_custom_invoked = 0;

  buf = gst_harness_create_buffer (h, data_size / 2);
  gst_buffer_append_memory (buf, gst_allocator_alloc (NULL, data_size / 2, NULL));

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);
  EXPECT_EQ (decoder_custom_invoked, 0U);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  EXPECT_EQ (0, nnstreamer_decoder_custom_unregister ("tdec_size"));
}

/**
 * @brief A buffer holding fewer memory chunks than the caps have tensors is refused.
 */
TEST (testTensorDecoder, pushInputMemoryCountShort_n)
{
  GstTensorsConfig config;
  GstHarness *h;
  gsize data_size;

  ASSERT_EQ (0, nnstreamer_decoder_custom_register ("tdec_size", _decoder_custom_cb, NULL));

  _get_decoder_config (&config);
  config.info.num_tensors = 2;
  gst_tensor_parse_dimension (TEST_DECODER_DIM, config.info.info[1].dimension);
  config.info.info[1].type = _NNS_UINT8;

  h = _get_decoder_harness ("custom-code", "tdec_size", &config);
  ASSERT_TRUE (h != NULL);

  data_size = gst_tensors_info_get_size (&config.info, -1);
  decoder_custom_invoked = 0;

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size)), GST_FLOW_ERROR);
  EXPECT_EQ (decoder_custom_invoked, 0U);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  EXPECT_EQ (0, nnstreamer_decoder_custom_unregister ("tdec_size"));
}

/**
 * @brief A flexible tensor is handed over with its meta header.
 */
TEST (testTensorDecoder, pushFlexibleInput)
{
  GstTensorsConfig config;
  GstTensorMetaInfo meta;
  GstHarness *h;
  gsize hsize;

  ASSERT_EQ (0, nnstreamer_decoder_custom_register ("tdec_size", _decoder_custom_cb, NULL));

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  config.rate_n = 0;
  config.rate_d = 1;

  h = _get_decoder_harness ("custom-code", "tdec_size", &config);
  ASSERT_TRUE (h != NULL);

  gst_tensor_meta_info_init (&meta);
  hsize = gst_tensor_meta_info_get_header_size (&meta);
  decoder_custom_invoked = 0;

  EXPECT_EQ (gst_harness_push (h, _get_flex_decoder_buffer (h, 64, hsize + 64)), GST_FLOW_OK);
  EXPECT_EQ (decoder_custom_invoked, 1U);
  EXPECT_EQ (decoder_custom_size, hsize + 64);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  EXPECT_EQ (0, nnstreamer_decoder_custom_unregister ("tdec_size"));
}

/**
 * @brief A flexible tensor longer than its own meta info is accepted.
 * @details A tensor that describes itself is allowed to carry slack, so the
 *          header and the data have to fit in the memory, not to fill it.
 */
TEST (testTensorDecoder, pushFlexibleInputOversized)
{
  GstTensorsConfig config;
  GstTensorMetaInfo meta;
  GstHarness *h;
  gsize hsize;

  ASSERT_EQ (0, nnstreamer_decoder_custom_register ("tdec_size", _decoder_custom_cb, NULL));

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  config.rate_n = 0;
  config.rate_d = 1;

  h = _get_decoder_harness ("custom-code", "tdec_size", &config);
  ASSERT_TRUE (h != NULL);

  gst_tensor_meta_info_init (&meta);
  hsize = gst_tensor_meta_info_get_header_size (&meta);
  decoder_custom_invoked = 0;

  EXPECT_EQ (gst_harness_push (h, _get_flex_decoder_buffer (h, 64, hsize + 128)), GST_FLOW_OK);
  EXPECT_EQ (decoder_custom_invoked, 1U);
  EXPECT_EQ (decoder_custom_size, hsize + 128);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  EXPECT_EQ (0, nnstreamer_decoder_custom_unregister ("tdec_size"));
}

/**
 * @brief A flexible tensor shorter than its own meta info is refused.
 * @details The sub-plugins take the data size out of the meta header without
 *          knowing how long the memory holding it is.
 */
TEST (testTensorDecoder, pushFlexibleInputTruncated_n)
{
  GstTensorsConfig config;
  GstTensorMetaInfo meta;
  GstHarness *h;
  gsize hsize;

  ASSERT_EQ (0, nnstreamer_decoder_custom_register ("tdec_size", _decoder_custom_cb, NULL));

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  config.rate_n = 0;
  config.rate_d = 1;

  h = _get_decoder_harness ("custom-code", "tdec_size", &config);
  ASSERT_TRUE (h != NULL);

  gst_tensor_meta_info_init (&meta);
  hsize = gst_tensor_meta_info_get_header_size (&meta);
  decoder_custom_invoked = 0;

  EXPECT_EQ (gst_harness_push (h, _get_flex_decoder_buffer (h, 64, hsize + 32)), GST_FLOW_ERROR);
  EXPECT_EQ (decoder_custom_invoked, 0U);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  EXPECT_EQ (0, nnstreamer_decoder_custom_unregister ("tdec_size"));
}

/**
 * @brief A flexible tensor long enough for a meta header but not holding one is refused.
 * @details The memory passes the length guard, so this is the branch where the
 *          content of the header itself decides.
 */
TEST (testTensorDecoder, pushFlexibleInputBrokenHeader_n)
{
  GstTensorsConfig config;
  GstTensorMetaInfo meta;
  GstHarness *h;
  GstBuffer *buf;
  gsize hsize;

  ASSERT_EQ (0, nnstreamer_decoder_custom_register ("tdec_size", _decoder_custom_cb, NULL));

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  config.rate_n = 0;
  config.rate_d = 1;

  h = _get_decoder_harness ("custom-code", "tdec_size", &config);
  ASSERT_TRUE (h != NULL);

  gst_tensor_meta_info_init (&meta);
  hsize = gst_tensor_meta_info_get_header_size (&meta);
  decoder_custom_invoked = 0;

  buf = gst_harness_create_buffer (h, hsize + 64);
  gst_buffer_memset (buf, 0, 0x11, hsize + 64);

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);
  EXPECT_EQ (decoder_custom_invoked, 0U);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  EXPECT_EQ (0, nnstreamer_decoder_custom_unregister ("tdec_size"));
}

/**
 * @brief A flexible tensor declaring an unknown meta version is refused.
 * @details The magic and the fields still validate, but the header layout of
 *          another major version is unknown, so the data cannot be located.
 */
TEST (testTensorDecoder, pushFlexibleInputUnknownVersion_n)
{
  GstTensorsConfig config;
  GstTensorMetaInfo meta;
  GstHarness *h;
  GstBuffer *buf;
  GstMapInfo map;
  gsize hsize;
  guint major = 0;

  ASSERT_EQ (0, nnstreamer_decoder_custom_register ("tdec_size", _decoder_custom_cb, NULL));

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  config.rate_n = 0;
  config.rate_d = 1;

  h = _get_decoder_harness ("custom-code", "tdec_size", &config);
  ASSERT_TRUE (h != NULL);

  gst_tensor_meta_info_init (&meta);
  hsize = gst_tensor_meta_info_get_header_size (&meta);
  decoder_custom_invoked = 0;

  buf = _get_flex_decoder_buffer (h, 64, hsize + 64);
  ASSERT_TRUE (buf != NULL);

  /**
   * Keep the magic and raise the major of the version. The version marker and
   * the major live in the second word of the header; both encodings are
   * private to nnstreamer_plugin_api_util_impl.c, so the assertions below pin
   * what this literal has to mean: a header that still validates, of a version
   * whose layout cannot be sized. Without them a changed encoding would leave
   * the case passing on the parse failure before the branch under test.
   */
  ASSERT_TRUE (gst_buffer_map (buf, &map, GST_MAP_WRITE));
  ((uint32_t *) map.data)[1] = 0xDE002000U;

  EXPECT_TRUE (gst_tensor_meta_info_parse_header (&meta, map.data));
  EXPECT_TRUE (gst_tensor_meta_info_get_version (&meta, &major, NULL));
  EXPECT_NE (major, 1U);
  EXPECT_EQ (gst_tensor_meta_info_get_header_size (&meta), 0U);
  gst_buffer_unmap (buf, &map);

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_ERROR);
  EXPECT_EQ (decoder_custom_invoked, 0U);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  EXPECT_EQ (0, nnstreamer_decoder_custom_unregister ("tdec_size"));
}

/**
 * @brief A flexible tensor too short to hold a meta header is refused.
 */
TEST (testTensorDecoder, pushFlexibleInputNoHeader_n)
{
  GstTensorsConfig config;
  GstHarness *h;

  ASSERT_EQ (0, nnstreamer_decoder_custom_register ("tdec_size", _decoder_custom_cb, NULL));

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  config.rate_n = 0;
  config.rate_d = 1;

  h = _get_decoder_harness ("custom-code", "tdec_size", &config);
  ASSERT_TRUE (h != NULL);

  decoder_custom_invoked = 0;

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 16)), GST_FLOW_ERROR);
  EXPECT_EQ (decoder_custom_invoked, 0U);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  EXPECT_EQ (0, nnstreamer_decoder_custom_unregister ("tdec_size"));
}

/**
 * @brief An oversized buffer does not overrun the video frame of direct_video.
 * @details direct_video copies as many bytes as the incoming memory holds into
 *          an output frame sized from the negotiated dimensions.
 */
TEST (testTensorDecoder, pushDirectVideoInputSizeTooLarge_n)
{
  GstTensorsConfig config;
  GstHarness *h;
  gsize data_size;

  _get_decoder_config (&config);
  h = _get_decoder_harness ("direct_video", NULL, &config);
  ASSERT_TRUE (h != NULL);

  data_size = gst_tensors_info_get_size (&config.info, 0);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size * 64)), GST_FLOW_ERROR);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
}

#define TEST_DECODER_MOCK_NAME "tdec_mock"

/**
 * @brief Set while the mock decoder sub-plugin describes no output caps.
 */
static gboolean decoder_mock_refuses;

/**
 * @brief Number of GStreamer critical logs of a tensor_decoder test.
 */
static guint decoder_gst_critical_count;

/**
 * @brief Last critical message of the GStreamer domain, for diagnostics.
 */
static gchar decoder_gst_critical_msg[256];

/**
 * @brief Log handler counting the critical logs of the GStreamer domain.
 * @details A sub-plugin describing no output leaves the element with no caps
 * to intersect, and GStreamer answers every NULL handed to its caps helpers
 * with a critical, so a zero count is what tells a refusal apart from the
 * element walking on through the NULL.
 */
static void
_decoder_count_gst_critical (const gchar *domain, GLogLevelFlags level,
    const gchar *message, gpointer udata)
{
  UNUSED (domain);
  UNUSED (udata);

  if (level & G_LOG_LEVEL_CRITICAL) {
    decoder_gst_critical_count++;
    g_strlcpy (decoder_gst_critical_msg, message, sizeof (decoder_gst_critical_msg));
  }
}

/**
 * @brief Start counting the critical logs of the GStreamer domain.
 * @details The counter is proven live before it is used, so that a handler that
 * silently stops matching cannot turn the zero-critical assertions into no-ops.
 */
static guint
_decoder_watch_gst_critical (void)
{
  guint handler = g_log_set_handler ("GStreamer",
      (GLogLevelFlags) (G_LOG_LEVEL_CRITICAL | G_LOG_FLAG_FATAL | G_LOG_FLAG_RECURSION),
      _decoder_count_gst_critical, NULL);

  decoder_gst_critical_count = 0;
  g_log ("GStreamer", G_LOG_LEVEL_CRITICAL, "tensor_decoder test: counter self-check");
  EXPECT_EQ (decoder_gst_critical_count, 1U);

  decoder_gst_critical_count = 0;
  decoder_gst_critical_msg[0] = '\0';
  return handler;
}

/**
 * @brief Mock decoder sub-plugin, object initialization.
 */
static int
_decoder_mock_init (void **pdata)
{
  *pdata = NULL;
  return TRUE;
}

/**
 * @brief Mock decoder sub-plugin, object destruction.
 */
static void
_decoder_mock_exit (void **pdata)
{
  UNUSED (pdata);
}

/**
 * @brief Mock decoder sub-plugin, describing no output while it refuses.
 */
static GstCaps *
_decoder_mock_get_out_caps (void **pdata, const GstTensorsConfig *config)
{
  UNUSED (pdata);
  UNUSED (config);

  if (decoder_mock_refuses)
    return NULL;

  return gst_caps_from_string ("application/octet-stream");
}

/**
 * @brief Mock decoder sub-plugin, emitting a fixed-size output.
 */
static GstFlowReturn
_decoder_mock_decode (void **pdata, const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  UNUSED (pdata);
  UNUSED (config);
  UNUSED (input);

  gst_buffer_append_memory (outbuf, gst_allocator_alloc (NULL, 4, NULL));
  return GST_FLOW_OK;
}

/**
 * @brief Register the mock decoder sub-plugin.
 */
static GstTensorDecoderDef *
_decoder_mock_register (void)
{
  GstTensorDecoderDef *sub = g_new0 (GstTensorDecoderDef, 1);

  sub->modename = g_strdup (TEST_DECODER_MOCK_NAME);
  sub->init = _decoder_mock_init;
  sub->exit = _decoder_mock_exit;
  sub->getOutCaps = _decoder_mock_get_out_caps;
  sub->decode = _decoder_mock_decode;

  if (!nnstreamer_decoder_probe (sub)) {
    g_free (sub->modename);
    g_free (sub);
    return NULL;
  }

  return sub;
}

/**
 * @brief Unregister the mock decoder sub-plugin and release it.
 */
static void
_decoder_mock_unregister (GstTensorDecoderDef *sub)
{
  nnstreamer_decoder_exit (TEST_DECODER_MOCK_NAME);
  g_free (sub->modename);
  g_free (sub);
}

/**
 * @brief A sub-plugin that describes no output is refused, not dereferenced.
 * @details The element intersects and unrefs whatever getOutCaps () hands back,
 *          so a sub-plugin refusing the config used to take a NULL through the
 *          whole of the caps fixation.
 */
TEST (testTensorDecoder, subpluginRefusesOutCaps_n)
{
  GstTensorDecoderDef *sub;
  GstTensorsConfig config;
  GstHarness *h;
  guint handler;

  sub = _decoder_mock_register ();
  ASSERT_TRUE (sub != NULL);

  decoder_mock_refuses = TRUE;
  handler = _decoder_watch_gst_critical ();

  _get_decoder_config (&config);
  h = _get_decoder_harness (TEST_DECODER_MOCK_NAME, NULL, &config);
  if (h == NULL) {
    g_log_remove_handler ("GStreamer", handler);
    gst_tensors_config_free (&config);
    _decoder_mock_unregister (sub);
  }
  ASSERT_TRUE (h != NULL);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_NOT_NEGOTIATED);
  EXPECT_EQ (decoder_gst_critical_count, 0U) << decoder_gst_critical_msg;

  g_log_remove_handler ("GStreamer", handler);
  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  _decoder_mock_unregister (sub);
}

/**
 * @brief A sub-plugin refusing a renegotiated config is refused, not dereferenced.
 * @details The element asks the sub-plugin again when a new caps event does not
 *          match what it was configured with, which is a second place the NULL
 *          used to reach gst_caps_unref ().
 */
TEST (testTensorDecoder, subpluginRefusesNewConfig_n)
{
  GstTensorDecoderDef *sub;
  GstTensorsConfig config;
  GstHarness *h;
  gsize data_size;
  guint handler;

  sub = _decoder_mock_register ();
  ASSERT_TRUE (sub != NULL);

  decoder_mock_refuses = FALSE;

  _get_decoder_config (&config);
  data_size = gst_tensors_info_get_size (&config.info, 0);
  h = _get_decoder_harness (TEST_DECODER_MOCK_NAME, NULL, &config);
  if (h == NULL) {
    gst_tensors_config_free (&config);
    _decoder_mock_unregister (sub);
  }
  ASSERT_TRUE (h != NULL);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size)), GST_FLOW_OK);

  handler = _decoder_watch_gst_critical ();
  decoder_mock_refuses = TRUE;

  gst_tensor_parse_dimension ("3:64:64:1", config.info.info[0].dimension);
  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));

  EXPECT_NE (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_OK);
  EXPECT_EQ (decoder_gst_critical_count, 0U) << decoder_gst_critical_msg;

  g_log_remove_handler ("GStreamer", handler);
  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  _decoder_mock_unregister (sub);
}

/**
 * @brief A sub-plugin that describes an output is negotiated as before.
 * @details The refusal must be told apart from an ordinary caps query, whose
 *          config the element cannot read yet.
 */
TEST (testTensorDecoder, subpluginAcceptsOutCaps)
{
  GstTensorDecoderDef *sub;
  GstTensorsConfig config;
  GstHarness *h;
  gsize data_size;
  guint handler;

  sub = _decoder_mock_register ();
  ASSERT_TRUE (sub != NULL);

  decoder_mock_refuses = FALSE;
  handler = _decoder_watch_gst_critical ();

  _get_decoder_config (&config);
  data_size = gst_tensors_info_get_size (&config.info, 0);
  h = _get_decoder_harness (TEST_DECODER_MOCK_NAME, NULL, &config);
  if (h == NULL) {
    g_log_remove_handler ("GStreamer", handler);
    gst_tensors_config_free (&config);
    _decoder_mock_unregister (sub);
  }
  ASSERT_TRUE (h != NULL);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size)), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  EXPECT_EQ (decoder_gst_critical_count, 0U) << decoder_gst_critical_msg;

  g_log_remove_handler ("GStreamer", handler);
  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  _decoder_mock_unregister (sub);
}

/**
 * @brief The refusal of a config carrying extra tensors walks the release path.
 * @details The config the element parses out of a caps query allocates the
 *          records of the 17th and later tensors on the heap, and the refusal
 *          leaves the element holding it. A single-tensor config keeps that
 *          allocation empty, so it takes a stream this wide to walk the path
 *          the release is there for.
 */
TEST (testTensorDecoder, subpluginRefusesExtraTensors_n)
{
  GstTensorDecoderDef *sub;
  GstTensorsConfig config;
  GstHarness *h;
  guint i, handler;

  sub = _decoder_mock_register ();
  ASSERT_TRUE (sub != NULL);

  decoder_mock_refuses = TRUE;
  handler = _decoder_watch_gst_critical ();

  gst_tensors_config_init (&config);
  config.info.num_tensors = NNS_TENSOR_MEMORY_MAX + 1;
  for (i = 0; i < config.info.num_tensors; i++) {
    GstTensorInfo *info = gst_tensors_info_get_nth_info (&config.info, i);
    info->type = _NNS_UINT8;
    gst_tensor_parse_dimension (TEST_DECODER_DIM, info->dimension);
  }
  config.rate_n = 0;
  config.rate_d = 1;

  h = _get_decoder_harness (TEST_DECODER_MOCK_NAME, NULL, &config);
  if (h == NULL) {
    g_log_remove_handler ("GStreamer", handler);
    gst_tensors_config_free (&config);
    _decoder_mock_unregister (sub);
  }
  ASSERT_TRUE (h != NULL);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_NOT_NEGOTIATED);
  EXPECT_EQ (decoder_gst_critical_count, 0U) << decoder_gst_critical_msg;

  g_log_remove_handler ("GStreamer", handler);
  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  _decoder_mock_unregister (sub);
}

/**
 * @brief A caps query the element cannot read a config from still offers caps.
 * @details Only a config the sub-plugin has seen and turned down is a refusal.
 *          A query carrying no fixed config says nothing about what the
 *          sub-plugin would accept, so answering it with the refusal would
 *          stop every decoder from negotiating at all. The mock refuses
 *          everything it is asked, which is what tells the two apart.
 */
TEST (testTensorDecoder, capsQueryWithoutConfig)
{
  GstTensorDecoderDef *sub;
  GstElement *dec;
  GstHarness *h;
  GstCaps *queried;

  sub = _decoder_mock_register ();
  ASSERT_TRUE (sub != NULL);

  decoder_mock_refuses = TRUE;

  dec = gst_element_factory_make ("tensor_decoder", NULL);
  if (dec == NULL)
    _decoder_mock_unregister (sub);
  ASSERT_TRUE (dec != NULL);
  gst_object_ref_sink (dec);
  g_object_set (dec, "mode", TEST_DECODER_MOCK_NAME, NULL);

  h = gst_harness_new_with_element (dec, "sink", "src");
  gst_object_unref (dec);
  if (h == NULL)
    _decoder_mock_unregister (sub);
  ASSERT_TRUE (h != NULL);

  /* no caps event yet, so the sink pad still carries the unfixed template */
  queried = gst_pad_peer_query_caps (h->sinkpad, NULL);
  ASSERT_TRUE (queried != NULL);
  EXPECT_FALSE (gst_caps_is_empty (queried));

  gst_caps_unref (queried);
  gst_harness_teardown (h);
  _decoder_mock_unregister (sub);
}

/**
 * @brief Hand the decoder a caps pair straight through its set_caps vfunc.
 * @details GstBaseTransform refuses a stream before set_caps whenever
 *          transform_caps already has, so a decoder that was negotiated once
 *          is the only way to ask set_caps about a config on its own.
 */
static gboolean
_decoder_set_caps (GstHarness *h, const GstTensorsConfig *config, const gchar *out_str)
{
  GstBaseTransform *trans = GST_BASE_TRANSFORM (h->element);
  GstCaps *incaps = gst_tensors_caps_from_config (config);
  GstCaps *outcaps = gst_caps_from_string (out_str);
  gboolean ret;

  ret = GST_BASE_TRANSFORM_GET_CLASS (trans)->set_caps (trans, incaps, outcaps);

  gst_caps_unref (incaps);
  gst_caps_unref (outcaps);
  return ret;
}

/**
 * @brief set_caps fails when the sub-plugin refuses the renegotiated config.
 * @details The refusal made gst_tensordec_configure () return FALSE, but
 *          set_caps then returned what the flag held from the negotiation
 *          before, which reported success for a config it had just refused.
 */
TEST (testTensorDecoder, setCapsRefusedRenegotiation_n)
{
  GstTensorDecoderDef *sub;
  GstTensorsConfig config;
  GstHarness *h;
  gsize data_size;

  sub = _decoder_mock_register ();
  ASSERT_TRUE (sub != NULL);

  decoder_mock_refuses = FALSE;

  _get_decoder_config (&config);
  data_size = gst_tensors_info_get_size (&config.info, 0);
  h = _get_decoder_harness (TEST_DECODER_MOCK_NAME, NULL, &config);
  if (h == NULL) {
    gst_tensors_config_free (&config);
    _decoder_mock_unregister (sub);
  }
  ASSERT_TRUE (h != NULL);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size)), GST_FLOW_OK);

  decoder_mock_refuses = TRUE;
  gst_tensor_parse_dimension ("3:64:64:1", config.info.info[0].dimension);
  EXPECT_FALSE (_decoder_set_caps (h, &config, "application/octet-stream"));

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  _decoder_mock_unregister (sub);
}

/**
 * @brief set_caps fails when the output caps are not what the sub-plugin makes.
 * @details The same stale flag answered for this branch too: the mismatch
 *          was logged and the earlier success returned anyway. The refusal
 *          must not stop the stream either, since the stored config still
 *          describes the pad caps, which a refused caps event leaves as they
 *          were.
 */
TEST (testTensorDecoder, setCapsIncompatibleOutput_n)
{
  GstTensorDecoderDef *sub;
  GstTensorsConfig config;
  GstHarness *h;
  gsize data_size;

  sub = _decoder_mock_register ();
  ASSERT_TRUE (sub != NULL);

  decoder_mock_refuses = FALSE;

  _get_decoder_config (&config);
  data_size = gst_tensors_info_get_size (&config.info, 0);
  h = _get_decoder_harness (TEST_DECODER_MOCK_NAME, NULL, &config);
  if (h == NULL) {
    gst_tensors_config_free (&config);
    _decoder_mock_unregister (sub);
  }
  ASSERT_TRUE (h != NULL);

  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size)), GST_FLOW_OK);

  EXPECT_FALSE (_decoder_set_caps (h, &config, "video/x-raw"));
  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size)), GST_FLOW_OK);
  EXPECT_TRUE (_decoder_set_caps (h, &config, "application/octet-stream"));

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  _decoder_mock_unregister (sub);
}

/**
 * @brief A stream that goes back to its caps after a refused change flows again.
 * @details Caps without dimensions reach set_caps, since no config can be
 *          read out of them any earlier, and are refused there. A refused caps
 *          event is not stored on the pad, and GstBaseTransform does not call
 *          set_caps for caps equal to the pad's current ones, so the element
 *          has to still hold the earlier negotiation when those come back.
 */
TEST (testTensorDecoder, recoverAfterRefusedCaps)
{
  GstTensorDecoderDef *sub;
  GstTensorsConfig config;
  GstElement *dec;
  GstHarness *h;
  gsize data_size;

  sub = _decoder_mock_register ();
  ASSERT_TRUE (sub != NULL);

  decoder_mock_refuses = FALSE;

  dec = gst_element_factory_make ("tensor_decoder", NULL);
  if (dec == NULL)
    _decoder_mock_unregister (sub);
  ASSERT_TRUE (dec != NULL);
  gst_object_ref_sink (dec);
  g_object_set (dec, "mode", TEST_DECODER_MOCK_NAME, NULL);

  h = gst_harness_new_with_element (dec, "sink", "src");
  gst_object_unref (dec);
  if (h == NULL)
    _decoder_mock_unregister (sub);
  ASSERT_TRUE (h != NULL);

  _get_decoder_config (&config);
  data_size = gst_tensors_info_get_size (&config.info, 0);
  gst_harness_set_sink_caps_str (h, "application/octet-stream");

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size)), GST_FLOW_OK);

  gst_harness_set_src_caps_str (
      h, "other/tensors,format=static,num_tensors=1,types=uint8,framerate=0/1");

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size)), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 2U);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  _decoder_mock_unregister (sub);
}

/**
 * @brief Caps nothing can be read out of are refused, not fixated as ANY.
 * @details Without dimensions the element cannot tell its output, and a
 *          downstream that takes anything does not tell it either, so the
 *          fixation is left with ANY, which has no fixed form.
 */
TEST (testTensorDecoder, unreadableCapsToAnyDownstream_n)
{
  GstTensorDecoderDef *sub;
  GstElement *dec;
  GstHarness *h;
  guint handler;

  sub = _decoder_mock_register ();
  ASSERT_TRUE (sub != NULL);

  decoder_mock_refuses = FALSE;
  handler = _decoder_watch_gst_critical ();

  dec = gst_element_factory_make ("tensor_decoder", NULL);
  if (dec == NULL) {
    g_log_remove_handler ("GStreamer", handler);
    _decoder_mock_unregister (sub);
  }
  ASSERT_TRUE (dec != NULL);
  gst_object_ref_sink (dec);
  g_object_set (dec, "mode", TEST_DECODER_MOCK_NAME, NULL);

  h = gst_harness_new_with_element (dec, "sink", "src");
  gst_object_unref (dec);
  if (h == NULL) {
    g_log_remove_handler ("GStreamer", handler);
    _decoder_mock_unregister (sub);
  }
  ASSERT_TRUE (h != NULL);

  gst_harness_set_src_caps_str (
      h, "other/tensors,format=static,num_tensors=1,types=uint8,framerate=0/1");
  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, 4)), GST_FLOW_NOT_NEGOTIATED);
  EXPECT_EQ (decoder_gst_critical_count, 0U) << decoder_gst_critical_msg;

  g_log_remove_handler ("GStreamer", handler);
  gst_harness_teardown (h);
  _decoder_mock_unregister (sub);
}

/**
 * @brief A stream wider than the inline tensor records is decoded as before.
 * @details Negotiation stores the config twice, at fixation and again at
 *          set_caps, so the config holding the extra records is replaced and
 *          released before the first buffer reads the one that replaced it.
 */
TEST (testTensorDecoder, subpluginAcceptsExtraTensors)
{
  GstTensorDecoderDef *sub;
  GstTensorsConfig config;
  GstHarness *h;
  GstBuffer *buf;
  GstMemory *mem;
  GstMapInfo map;
  gsize data_size;
  guint i, handler;

  sub = _decoder_mock_register ();
  ASSERT_TRUE (sub != NULL);

  decoder_mock_refuses = FALSE;
  handler = _decoder_watch_gst_critical ();

  gst_tensors_config_init (&config);
  config.info.num_tensors = NNS_TENSOR_MEMORY_MAX + 1;
  for (i = 0; i < config.info.num_tensors; i++) {
    GstTensorInfo *info = gst_tensors_info_get_nth_info (&config.info, i);
    info->type = _NNS_UINT8;
    gst_tensor_parse_dimension (TEST_DECODER_DIM, info->dimension);
  }
  config.rate_n = 0;
  config.rate_d = 1;
  data_size = gst_tensors_info_get_size (&config.info, 0);

  h = _get_decoder_harness (TEST_DECODER_MOCK_NAME, NULL, &config);
  if (h == NULL) {
    g_log_remove_handler ("GStreamer", handler);
    gst_tensors_config_free (&config);
    _decoder_mock_unregister (sub);
  }
  ASSERT_TRUE (h != NULL);

  buf = gst_buffer_new ();
  for (i = 0; i < config.info.num_tensors; i++) {
    mem = gst_allocator_alloc (NULL, data_size, NULL);
    if (gst_memory_map (mem, &map, GST_MAP_WRITE)) {
      memset (map.data, 0, map.size);
      gst_memory_unmap (mem, &map);
    }
    EXPECT_TRUE (gst_tensor_buffer_append_memory (
        buf, mem, gst_tensors_info_get_nth_info (&config.info, i)));
  }

  EXPECT_EQ (gst_harness_push (h, buf), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 1U);
  EXPECT_EQ (decoder_gst_critical_count, 0U) << decoder_gst_critical_msg;

  g_log_remove_handler ("GStreamer", handler);
  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
  _decoder_mock_unregister (sub);
}

/**
 * @brief The video format a decoder negotiated for the stream, or NULL.
 */
static gchar *
_decoder_output_format (GstHarness *h)
{
  GstCaps *caps = gst_pad_get_current_caps (h->sinkpad);
  gchar *format = NULL;

  if (caps) {
    const gchar *f = gst_structure_get_string (gst_caps_get_structure (caps, 0), "format");
    format = g_strdup (f);
    gst_caps_unref (caps);
  }

  return format;
}

/**
 * @brief A resolution change keeps the video format option1 chose.
 * @details Renegotiating a new tensor config re-initialises the sub-plugin,
 *          which starts with none of the options, so direct_video describes
 *          RGB for a 3-channel tensor against the BGR caps just fixated. A
 *          set_caps that answered with a stale success let that pass; one
 *          that refuses it renegotiates the stream to RGB with its red and
 *          blue swapped, or fails where the downstream takes only BGR.
 */
TEST (testTensorDecoder, directVideoResolutionChangeKeepsFormat)
{
  GstTensorsConfig config;
  GstElement *dec;
  GstHarness *h;
  gsize data_size;
  gchar *format;

  dec = gst_element_factory_make ("tensor_decoder", NULL);
  ASSERT_TRUE (dec != NULL);
  gst_object_ref_sink (dec);
  g_object_set (dec, "mode", "direct_video", "option1", "BGR", NULL);

  h = gst_harness_new_with_element (dec, "sink", "src");
  gst_object_unref (dec);
  ASSERT_TRUE (h != NULL);

  gst_harness_set_sink_caps_str (h, "video/x-raw");

  _get_decoder_config (&config);
  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_size = gst_tensors_info_get_size (&config.info, 0);
  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size)), GST_FLOW_OK);

  format = _decoder_output_format (h);
  EXPECT_STREQ (format, "BGR");
  g_free (format);

  gst_tensor_parse_dimension ("3:8:8", config.info.info[0].dimension);
  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  data_size = gst_tensors_info_get_size (&config.info, 0);
  EXPECT_EQ (gst_harness_push (h, gst_harness_create_buffer (h, data_size)), GST_FLOW_OK);
  EXPECT_EQ (gst_harness_buffers_received (h), 2U);

  format = _decoder_output_format (h);
  EXPECT_STREQ (format, "BGR");
  g_free (format);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
}

/**
 * @brief What a tensor_merge test saw arrive at its tensor_sink.
 */
typedef struct {
  guint received; /**< the number of buffers, bumped by the streaming thread */
  gsize size; /**< the size of the last buffer */
  guint8 head[12]; /**< the first bytes of the last buffer */
  guint8 tail[12]; /**< the last bytes of the last buffer */
  gchar error_src[32]; /**< the element an error was posted by, if any */
  GQuark error_domain; /**< the domain of that error */
  gint error_code; /**< the code of that error */
} mergeOutput;

/**
 * @brief Record which element ended the run, so a refusal by something else in
 *        the pipeline cannot stand in for the one the case is about.
 */
static void
_record_merge_message (GstMessage *msg, mergeOutput *out)
{
  const gchar *name = NULL;
  GError *err = NULL;

  if (msg == NULL || GST_MESSAGE_TYPE (msg) != GST_MESSAGE_ERROR)
    return;

  if (GST_MESSAGE_SRC (msg))
    name = GST_OBJECT_NAME (GST_MESSAGE_SRC (msg));
  g_strlcpy (out->error_src, name ? name : "", sizeof (out->error_src));

  gst_message_parse_error (msg, &err, NULL);
  if (err) {
    out->error_domain = err->domain;
    out->error_code = err->code;
    g_error_free (err);
  }
}

/**
 * @brief tensor_sink handler recording the merged buffer.
 */
static void
_record_merge_output (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  mergeOutput *out = (mergeOutput *) user_data;
  GstMapInfo info;

  UNUSED (element);

  if (gst_buffer_map (buffer, &info, GST_MAP_READ)) {
    gsize taken = MIN (info.size, sizeof (out->head));

    out->size = info.size;
    memcpy (out->head, info.data, taken);
    memcpy (out->tail, info.data + info.size - taken, taken);
    gst_buffer_unmap (buffer, &info);
  }

  g_atomic_int_inc (&out->received);
}

/**
 * @brief Run a tensor_merge pipeline until it ends and report what came out.
 * @details The three ways this can end are told apart on purpose: a case that
 *          expects a refusal has to fail, not pass, when the description no
 *          longer builds or when the element hangs instead of refusing.
 * @param[out] out what the tensor_sink named 'sinkx' received
 * @return GST_MESSAGE_EOS or GST_MESSAGE_ERROR as the pipeline posted it,
 *         GST_MESSAGE_ANY if it could not be built, GST_MESSAGE_UNKNOWN if it
 *         posted neither within the time limit
 */
static GstMessageType
_run_merge_pipeline (const gchar *desc, mergeOutput *out)
{
  GstElement *pipeline, *sink;
  GstBus *bus;
  GstMessage *msg;
  GstMessageType type;

  memset (out, 0, sizeof (mergeOutput));

  pipeline = gst_parse_launch (desc, NULL);
  if (pipeline == NULL)
    return GST_MESSAGE_ANY;

  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  if (sink == NULL) {
    gst_object_unref (pipeline);
    return GST_MESSAGE_ANY;
  }
  g_signal_connect (sink, "new-data", G_CALLBACK (_record_merge_output), out);

  setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT);

  bus = gst_element_get_bus (pipeline);
  msg = gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  type = (msg != NULL) ? GST_MESSAGE_TYPE (msg) : GST_MESSAGE_UNKNOWN;
  _record_merge_message (msg, out);
  if (msg)
    gst_message_unref (msg);
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (sink);
  gst_object_unref (pipeline);

  return type;
}

/**
 * @brief Two identical streams merged along the channel direction.
 * @details The bytes pin the interleave the copy loop performs: one pixel of
 *          the black stream, then one pixel of the white stream.
 */
TEST (testTensorMerge, linearFirstDirection)
{
  mergeOutput out;

  EXPECT_EQ (_run_merge_pipeline ("tensor_merge name=merge mode=linear option=0 ! tensor_sink name=sinkx "
                                  "videotestsrc num-buffers=1 pattern=black ! "
                                  "video/x-raw,format=RGB,width=20,height=20,framerate=30/1 ! tensor_converter ! merge.sink_0 "
                                  "videotestsrc num-buffers=1 pattern=white ! "
                                  "video/x-raw,format=RGB,width=20,height=20,framerate=30/1 ! tensor_converter ! merge.sink_1",
                 &out),
      GST_MESSAGE_EOS);

  EXPECT_EQ (out.received, 1U);
  EXPECT_EQ (out.size, 2400U);
  for (guint i = 0; i < 3; i++) {
    EXPECT_EQ (out.head[i], 0);
    EXPECT_EQ (out.head[i + 3], 255);
  }
}

/**
 * @brief Two streams of different heights merged along the height direction.
 * @details Nothing else in the tree merges streams that differ in the merge
 *          direction with option=2: the existing height cases all carry equal
 *          sizes, so the per-input chunk of that copy loop is never told apart
 *          from input 0's. The bytes at both ends pin the two chunks.
 */
TEST (testTensorMerge, linearThirdDirection)
{
  mergeOutput out;

  EXPECT_EQ (_run_merge_pipeline ("tensor_merge name=merge mode=linear option=2 ! tensor_sink name=sinkx "
                                  "videotestsrc num-buffers=1 pattern=black ! "
                                  "video/x-raw,format=RGB,width=20,height=20,framerate=30/1 ! tensor_converter ! merge.sink_0 "
                                  "videotestsrc num-buffers=1 pattern=white ! "
                                  "video/x-raw,format=RGB,width=20,height=10,framerate=30/1 ! tensor_converter ! merge.sink_1",
                 &out),
      GST_MESSAGE_EOS);

  EXPECT_EQ (out.received, 1U);
  EXPECT_EQ (out.size, (gsize) (3 * 20 * 30));
  for (guint i = 0; i < sizeof (out.head); i++) {
    EXPECT_EQ (out.head[i], 0);
    EXPECT_EQ (out.tail[i], 255);
  }
}

/**
 * @brief An input whose other dimensions differ is refused.
 * @details The copy loop walks every input with the dimensions of input 0, so
 *          a smaller input is read past its end and the output, sized from the
 *          input sizes, is written past its end (1200 B read from 300 B, 2400 B
 *          written into 1500 B). The mismatch used to be reported and ignored.
 */
TEST (testTensorMerge, dimensionMismatch_n)
{
  mergeOutput out;

  EXPECT_EQ (_run_merge_pipeline ("tensor_merge name=merge mode=linear option=0 ! tensor_sink name=sinkx "
                                  "videotestsrc num-buffers=1 ! "
                                  "video/x-raw,format=RGB,width=20,height=20,framerate=30/1 ! tensor_converter ! merge.sink_0 "
                                  "videotestsrc num-buffers=1 ! "
                                  "video/x-raw,format=RGB,width=10,height=10,framerate=30/1 ! tensor_converter ! merge.sink_1",
                 &out),
      GST_MESSAGE_ERROR);

  EXPECT_STREQ (out.error_src, "merge");
  EXPECT_EQ (out.error_domain, (GQuark) GST_CORE_ERROR);
  EXPECT_EQ (out.error_code, GST_CORE_ERROR_NEGOTIATION);
  EXPECT_EQ (out.received, 0U);
}

/**
 * @brief An input of another type is refused.
 * @details The element size of input 0 decides the stride for every input, so
 *          a float32 input 0 makes the loop read four times the uint8 input.
 */
TEST (testTensorMerge, typeMismatch_n)
{
  mergeOutput out;

  EXPECT_EQ (_run_merge_pipeline (
                 "tensor_merge name=merge mode=linear option=0 ! tensor_sink name=sinkx "
                 "videotestsrc num-buffers=1 ! "
                 "video/x-raw,format=RGB,width=20,height=20,framerate=30/1 ! tensor_converter ! "
                 "tensor_transform mode=typecast option=float32 ! merge.sink_0 "
                 "videotestsrc num-buffers=1 ! "
                 "video/x-raw,format=RGB,width=20,height=20,framerate=30/1 ! tensor_converter ! merge.sink_1",
                 &out),
      GST_MESSAGE_ERROR);

  EXPECT_STREQ (out.error_src, "merge");
  EXPECT_EQ (out.error_domain, (GQuark) GST_CORE_ERROR);
  EXPECT_EQ (out.error_code, GST_CORE_ERROR_NEGOTIATION);
  EXPECT_EQ (out.received, 0U);
}

/**
 * @brief A sink pad that renegotiates to another dimension is refused.
 * @details The element negotiates its source caps once and never looks at a
 *          later caps event, so the check that compares the inputs with each
 *          other runs only on the first buffer. The copy still strides input 1
 *          with the dimensions of input 0 afterwards, which is the same
 *          overrun by another route.
 */
TEST (testTensorMerge, renegotiatedDimension_n)
{
  mergeOutput out;

  EXPECT_EQ (_run_merge_pipeline (
                 "tensor_merge name=merge mode=linear option=0 ! tensor_sink name=sinkx "
                 "videotestsrc num-buffers=4 ! "
                 "video/x-raw,format=RGB,width=20,height=20,framerate=30/1 ! tensor_converter ! merge.sink_0 "
                 "concat name=c ! merge.sink_1 "
                 "videotestsrc num-buffers=2 ! "
                 "video/x-raw,format=RGB,width=20,height=20,framerate=30/1 ! tensor_converter ! c. "
                 "videotestsrc num-buffers=2 ! "
                 "video/x-raw,format=RGB,width=10,height=10,framerate=30/1 ! tensor_converter ! c.",
                 &out),
      GST_MESSAGE_ERROR);

  EXPECT_STREQ (out.error_src, "merge");
  EXPECT_EQ (out.error_domain, (GQuark) GST_STREAM_ERROR);
  EXPECT_EQ (out.error_code, GST_STREAM_ERROR_WRONG_TYPE);
  /* the two 20x20 buffers concat hands over first are merged, the third is not */
  EXPECT_EQ (out.received, 2U);
  EXPECT_EQ (out.size, 2400U);
}

/**
 * @brief Feed two appsrc buffers of a chosen size into tensor_merge.
 * @details appsrc lets the caps and the memory disagree, which no in-tree
 *          source does. Each buffer is pushed with the caps of the dimension
 *          given for it, so the element sees a self-consistent stream.
 * @param[out] out what the tensor_sink named 'sinkx' received
 * @return GST_MESSAGE_EOS or GST_MESSAGE_ERROR as the pipeline posted it,
 *         GST_MESSAGE_UNKNOWN if it posted neither within the time limit
 */
static GstMessageType
_run_merge_appsrc (const gchar *option, const gchar *dim0, gsize size0,
    const gchar *dim1, gsize size1, mergeOutput *out)
{
  GstElement *pipeline, *src[2], *sink;
  GstTensorsConfig config;
  GstBus *bus;
  GstMessage *msg;
  GstFlowReturn ret;
  GstMessageType type;
  const gchar *dim[2] = { dim0, dim1 };
  gsize size[2] = { size0, size1 };
  gchar *desc;
  guint i;

  memset (out, 0, sizeof (mergeOutput));

  desc = g_strdup_printf ("tensor_merge name=merge mode=linear option=%s ! "
                          "tensor_sink name=sinkx "
                          "appsrc name=src0 ! merge.sink_0 appsrc name=src1 ! merge.sink_1",
      option);
  pipeline = gst_parse_launch (desc, NULL);
  g_free (desc);
  if (pipeline == NULL)
    return GST_MESSAGE_ANY;

  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");
  if (sink == NULL) {
    gst_object_unref (pipeline);
    return GST_MESSAGE_ANY;
  }
  g_signal_connect (sink, "new-data", G_CALLBACK (_record_merge_output), out);

  for (i = 0; i < 2; i++) {
    gchar *name = g_strdup_printf ("src%u", i);
    GstCaps *caps;

    src[i] = gst_bin_get_by_name (GST_BIN (pipeline), name);
    g_free (name);

    gst_tensors_config_init (&config);
    config.info.num_tensors = 1;
    config.info.info[0].type = _NNS_UINT8;
    gst_tensor_parse_dimension (dim[i], config.info.info[0].dimension);
    config.rate_n = 30;
    config.rate_d = 1;
    caps = gst_tensor_caps_from_config (&config);
    g_object_set (src[i], "caps", caps, NULL);
    gst_caps_unref (caps);
    gst_tensors_config_free (&config);
  }

  /* the sink prerolls on a merged buffer, so it may never reach PLAYING */
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  for (i = 0; i < 2; i++) {
    /* zeroed: append_memory() sniffs a meta header from the first bytes */
    GstBuffer *buf = gst_buffer_new_allocate (NULL, size[i], NULL);

    gst_buffer_memset (buf, 0, 0, size[i]);
    /* the push-buffer action signal is transfer-none, unlike the C entry point */
    g_signal_emit_by_name (src[i], "push-buffer", buf, &ret);
    gst_buffer_unref (buf);
    if (ret != GST_FLOW_OK)
      break;
  }

  bus = gst_element_get_bus (pipeline);
  msg = gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  type = (msg != NULL) ? GST_MESSAGE_TYPE (msg) : GST_MESSAGE_UNKNOWN;
  _record_merge_message (msg, out);
  if (msg)
    gst_message_unref (msg);
  gst_object_unref (bus);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (sink);
  gst_object_unref (src[1]);
  gst_object_unref (src[0]);
  gst_object_unref (pipeline);

  return type;
}

/**
 * @brief A buffer smaller than the tensor its caps declare is refused.
 * @details The dimensions agree here, so nothing at negotiation time can see
 *          it; only the incoming memory tells the element the copy would run
 *          past the end. Every direction is tried because the copy bounds are
 *          written out once per direction, and so is the size they have to
 *          agree with.
 */
TEST (testTensorMerge, undersizedMemory_n)
{
  mergeOutput out;

  for (const gchar *option : { "0", "1", "2", "3" }) {
    EXPECT_EQ (_run_merge_appsrc (option, "3:20:20:1", 3 * 20 * 20, "3:20:20:1",
                   3 * 10 * 10, &out),
        GST_MESSAGE_ERROR)
        << "option=" << option;
    EXPECT_STREQ (out.error_src, "merge") << "option=" << option;
    EXPECT_EQ (out.error_domain, (GQuark) GST_STREAM_ERROR) << "option=" << option;
    EXPECT_EQ (out.error_code, GST_STREAM_ERROR_WRONG_TYPE) << "option=" << option;
    EXPECT_EQ (out.received, 0U) << "option=" << option;
  }
}

/**
 * @brief An input of a lower rank is refused in the batch direction.
 * @details The dimension array ends at the first zero, so a rank 3 tensor has
 *          dimension[3] == 0 and the batch-direction stride is zero: the input
 *          is never copied and its share of the output, which is sized from
 *          the input sizes, goes downstream uninitialised.
 */
TEST (testTensorMerge, lowerRankBatchDirection_n)
{
  mergeOutput out;

  EXPECT_EQ (_run_merge_appsrc ("3", "3:4:4:1", 3 * 4 * 4, "3:4:4", 3 * 4 * 4, &out),
      GST_MESSAGE_ERROR);
  EXPECT_STREQ (out.error_src, "merge");
  EXPECT_EQ (out.error_domain, (GQuark) GST_STREAM_ERROR);
  EXPECT_EQ (out.error_code, GST_STREAM_ERROR_WRONG_TYPE);
  EXPECT_EQ (out.received, 0U);
}

/**
 * @brief A mode the element does not know is refused by the element itself.
 * @details The source caps cannot be built without a mode, and the reason has
 *          to come from here: the flow error alone only makes the source post
 *          a generic stream error, which says nothing about what is wrong.
 */
TEST (testTensorMerge, unknownMode_n)
{
  mergeOutput out;

  EXPECT_EQ (_run_merge_pipeline ("tensor_merge name=merge mode=nosuchmode ! tensor_sink name=sinkx "
                                  "videotestsrc num-buffers=1 ! "
                                  "video/x-raw,format=RGB,width=20,height=20,framerate=30/1 ! tensor_converter ! merge.sink_0",
                 &out),
      GST_MESSAGE_ERROR);

  EXPECT_STREQ (out.error_src, "merge");
  EXPECT_EQ (out.error_domain, (GQuark) GST_CORE_ERROR);
  EXPECT_EQ (out.error_code, GST_CORE_ERROR_NEGOTIATION);
  EXPECT_EQ (out.received, 0U);
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
