/**
 * @file        unittest_query.cc
 * @date        27 Aug 2021
 * @brief       Unit test for tensor_query
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
#include "../gst/nnstreamer/tensor_query/tensor_query_common.h"

static const char *CUSTOM_LIB_PATH = "./libnnstreamer-edge-custom-test.so";

/**
 * @brief Test for tensor_query_server get and set properties
 */
TEST (tensorQuery, serverProperties0)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GstElement *srv_handle;
  gint int_val;
  guint uint_val;
  gchar *str_val;
  guint src_port;

  src_port = get_available_port ();

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("tensor_query_serversrc host=127.0.0.1 name=serversrc port=%u ! "
                              "other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! "
                              "tensor_query_serversink name=serversink",
      src_port);
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  /* Get properties of query server source */
  srv_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "serversrc");
  EXPECT_NE (srv_handle, nullptr);

  g_object_get (srv_handle, "host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.1", str_val);
  g_free (str_val);

  g_object_get (srv_handle, "port", &uint_val, NULL);
  EXPECT_EQ (src_port, uint_val);

  g_object_get (srv_handle, "connect-type", &int_val, NULL);
  EXPECT_EQ (0, int_val);

  g_object_get (srv_handle, "timeout", &uint_val, NULL);
  EXPECT_EQ (10U, uint_val);

  /* Set properties of query server source */
  g_object_set (srv_handle, "host", "127.0.0.2", NULL);
  g_object_get (srv_handle, "host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.2", str_val);
  g_free (str_val);

  g_object_set (srv_handle, "port", 5001U, NULL);
  g_object_get (srv_handle, "port", &uint_val, NULL);
  EXPECT_EQ (5001U, uint_val);

  g_object_set (srv_handle, "dest-host", "127.0.0.2", NULL);
  g_object_get (srv_handle, "dest-host", &str_val, NULL);
  EXPECT_STREQ ("127.0.0.2", str_val);
  g_free (str_val);

  g_object_set (srv_handle, "dest-port", 5001U, NULL);
  g_object_get (srv_handle, "dest-port", &uint_val, NULL);
  EXPECT_EQ (5001U, uint_val);

  g_object_set (srv_handle, "connect-type", 0, NULL);
  g_object_get (srv_handle, "connect-type", &int_val, NULL);
  EXPECT_EQ (0, int_val);


  g_object_set (srv_handle, "timeout", 20U, NULL);
  g_object_get (srv_handle, "timeout", &uint_val, NULL);
  EXPECT_EQ (20U, uint_val);

  g_object_set (srv_handle, "topic", "TEMP_TEST_TOPIC", NULL);
  g_object_get (srv_handle, "topic", &str_val, NULL);
  EXPECT_STREQ ("TEMP_TEST_TOPIC", str_val);
  g_free (str_val);

  g_object_set (srv_handle, "id", 12345U, NULL);
  g_object_get (srv_handle, "id", &uint_val, NULL);
  EXPECT_EQ (12345U, uint_val);

  gst_object_unref (srv_handle);

  /* Get properties of query server sink */
  srv_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "serversink");
  EXPECT_NE (srv_handle, nullptr);

  g_object_get (srv_handle, "connect-type", &int_val, NULL);
  EXPECT_EQ (0, int_val);

  g_object_get (srv_handle, "timeout", &uint_val, NULL);
  EXPECT_EQ (10U, uint_val);

  g_object_set (srv_handle, "connect-type", 0, NULL);
  g_object_get (srv_handle, "connect-type", &int_val, NULL);
  EXPECT_EQ (0, int_val);

  g_object_set (srv_handle, "timeout", 20U, NULL);
  g_object_get (srv_handle, "timeout", &uint_val, NULL);
  EXPECT_EQ (20U, uint_val);

  g_object_set (srv_handle, "id", 12345U, NULL);
  g_object_get (srv_handle, "id", &uint_val, NULL);
  EXPECT_EQ (12345U, uint_val);

  g_object_set (srv_handle, "limit", 10, NULL);
  g_object_get (srv_handle, "limit", &int_val, NULL);
  EXPECT_EQ (10, int_val);

  gst_object_unref (srv_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_query_server with invalid host name.
 */
TEST (tensorQuery, serverProperties2_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  guint src_port;

  src_port = get_available_port ();

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("tensor_query_serversrc name=serversrc host=f.a.i.l port=%u ! "
                              "other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! "
                              "tensor_query_serversink sync=false async=false",
      src_port);
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_query_server run.
 */
TEST (tensorQuery, serverRun)
{
  gchar *pipeline;
  GstElement *gstpipe;
  guint src_port;

  src_port = get_available_port ();

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("tensor_query_serversrc name=serversrc host=127.0.0.1 port=%u ! "
                              "other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! "
                              "tensor_query_serversink sync=false async=false",
      src_port);
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PAUSED, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_READY, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_READY, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PAUSED, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_READY, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PAUSED, UNITTEST_STATECHANGE_TIMEOUT), 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Run tensor query client without server
 */
TEST (tensorQuery, clientAlone_n)
{
  gchar *pipeline;
  GstElement *gstpipe;

  /* Create a query client pipeline */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB !"
                              "tensor_converter ! tensor_query_client ! tensor_sink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}


/**
 * @brief Test for tensor_query_server custom connection
 */
TEST (tensorQuery, serverCustomNormal)
{
  /** @todo TDD: Enable this test later. */
  GTEST_SKIP ();

  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;
  guint src_port;

  src_port = get_available_port ();

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "tensor_query_serversrc connect-type=CUSTOM custom-lib=%s name=serversrc port=%u ! "
      "other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! "
      "tensor_query_serversink connect-type=CUSTOM custom-lib=%s name=serversink",
      CUSTOM_LIB_PATH, src_port, CUSTOM_LIB_PATH);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);


  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_query_server custom connection with invalid property.
 */
TEST (tensorQuery, serverCustomInvalidProp_n)
{
  /** @todo TDD: Enable this test later. */
  GTEST_SKIP ();

  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;
  guint src_port;

  src_port = get_available_port ();

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "tensor_query_serversrc connect-type=CUSTOM name=serversrc port=%u ! "
      "other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! "
      "tensor_query_serversink connect-type=CUSTOM custom-lib=%s name=serversink",
      src_port, CUSTOM_LIB_PATH);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);


  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_query_server custom connection with invalid property.
 */
TEST (tensorQuery, serverCustomInvalidProp2_n)
{
  /** @todo TDD: Enable this test later. */
  GTEST_SKIP ();

  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;
  guint src_port = 0;

  src_port = get_available_port ();

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "tensor_query_serversrc connect-type=CUSTOM custom-lib=%s name=serversrc port=%u ! "
      "other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! "
      "tensor_query_serversink connect-type=CUSTOM  name=serversink",
      CUSTOM_LIB_PATH, src_port);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);


  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_query_server custom connection with invalid property.
 */
TEST (tensorQuery, serverCustomInvalidProp3_n)
{
  /** @todo TDD: Enable this test later. */
  GTEST_SKIP ();

  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;
  guint src_port = 0;

  src_port = get_available_port ();

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "tensor_query_serversrc connect-type=CUSTOM custom-lib=INVALID.so name=serversrc port=%u ! "
      "other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! "
      "tensor_query_serversink connect-type=CUSTOM custom-lib=%s name=serversink",
      src_port, CUSTOM_LIB_PATH);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);


  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_query_server custom connection with invalid property.
 */
TEST (tensorQuery, serverCustomInvalidProp4_n)
{
  /** @todo TDD: Enable this test later. */
  GTEST_SKIP ();

  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;
  guint src_port = 0;

  src_port = get_available_port ();

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "tensor_query_serversrc connect-type=CUSTOM custom-lib=%s name=serversrc port=%u ! "
      "other/tensors,num_tensors=1,dimensions=3:300:300:1,types=uint8 ! "
      "tensor_query_serversink connect-type=CUSTOM custom-lib=INVALID.so name=serversink",
      CUSTOM_LIB_PATH, src_port);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_query_client custom connection
 */
TEST (tensorQuery, customClientNormal)
{
  /** @todo TDD: Enable this test later. */
  GTEST_SKIP ();

  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;

  /* Create a query client pipeline */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB !"
                              "tensor_converter ! tensor_query_client connect-type=CUSTOM custom-lib=%s "
                              "name=client connect-type=TCP ! tensor_sink",
      CUSTOM_LIB_PATH);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_query_client custom connection with invalid property.
 */
TEST (tensorQuery, customClientInvalidProp_n)
{
  /** @todo TDD: Enable this test later. */
  GTEST_SKIP ();

  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;

  /* Create a query client pipeline */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB !"
                              "tensor_converter ! tensor_query_client connect-type=CUSTOM "
                              "name=client connect-type=TCP ! tensor_sink");
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test for tensor_query_client custom connection with invalid property.
 */
TEST (tensorQuery, customClientInvalidProp2_n)
{
  /** @todo TDD: Enable this test later. */
  GTEST_SKIP ();

  gchar *pipeline = nullptr;
  GstElement *gstpipe = nullptr;

  /* Create a query client pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB !"
      "tensor_converter ! tensor_query_client connect-type=CUSTOM custom-lib=INVALID.so "
      "name=client connect-type=TCP ! tensor_sink");
  gstpipe = gst_parse_launch (pipeline, nullptr);
  EXPECT_NE (gstpipe, nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Create edge data holding zero-filled memories of the given sizes.
 */
static nns_edge_data_h
_make_edge_data (const gsize *sizes, guint num)
{
  nns_edge_data_h data_h = nullptr;
  guint i;

  if (nns_edge_data_create (&data_h) != NNS_EDGE_ERROR_NONE)
    return nullptr;

  for (i = 0; i < num; i++)
    nns_edge_data_add (data_h, g_malloc0 (sizes[i]), sizes[i], g_free);

  return data_h;
}

/**
 * @brief Add a uint8 tensor memory with a tensor-meta header to the edge data.
 * @param extra The number of bytes to add to (or, if negative, cut from) the data described by the header.
 */
static void
_add_meta_memory (nns_edge_data_h data_h, tensor_format format, guint dim0,
    guint nnz, gint extra)
{
  GstTensorMetaInfo meta;
  gsize hsize, size;
  gpointer mem;

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_UINT8;
  meta.dimension[0] = dim0;
  meta.format = format;
  meta.sparse_info.nnz = nnz;

  hsize = gst_tensor_meta_info_get_header_size (&meta);
  size = hsize + gst_tensor_meta_info_get_data_size (&meta) + extra;

  mem = g_malloc0 (MAX (size, hsize));
  gst_tensor_meta_info_update_header (&meta, mem);
  nns_edge_data_add (data_h, mem, size, g_free);
}

/**
 * @brief Fill a static config of two tensors, uint8 4 (4 bytes) and float32 2:2 (16 bytes).
 */
static void
_make_static_config (GstTensorsConfig *config)
{
  GstTensorInfo *info;

  gst_tensors_config_init (config);
  config->rate_n = 0;
  config->rate_d = 1;
  config->info.num_tensors = 2;

  info = gst_tensors_info_get_nth_info (&config->info, 0);
  info->type = _NNS_UINT8;
  gst_tensor_parse_dimension ("4", info->dimension);

  info = gst_tensors_info_get_nth_info (&config->info, 1);
  info->type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("2:2", info->dimension);
}

/**
 * @brief Edge data whose memories match a static config is accepted.
 */
TEST (tensorQueryValidate, staticMatch)
{
  GstTensorsConfig config;
  const gsize sizes[] = { 4, 16 };
  nns_edge_data_h data_h;

  _make_static_config (&config);
  data_h = _make_edge_data (sizes, 2);
  ASSERT_NE (data_h, nullptr);

  EXPECT_TRUE (gst_tensor_query_validate_edge_data (data_h, &config));

  nns_edge_data_destroy (data_h);
  gst_tensors_config_free (&config);
}

/**
 * @brief Edge data with fewer or more memories than a static config describes is refused.
 */
TEST (tensorQueryValidate, staticCount_n)
{
  GstTensorsConfig config;
  const gsize sizes[] = { 4, 16, 4 };
  nns_edge_data_h data_h;

  _make_static_config (&config);

  data_h = _make_edge_data (sizes, 1);
  ASSERT_NE (data_h, nullptr);
  EXPECT_FALSE (gst_tensor_query_validate_edge_data (data_h, &config));
  nns_edge_data_destroy (data_h);

  data_h = _make_edge_data (sizes, 3);
  ASSERT_NE (data_h, nullptr);
  EXPECT_FALSE (gst_tensor_query_validate_edge_data (data_h, &config));
  nns_edge_data_destroy (data_h);

  gst_tensors_config_free (&config);
}

/**
 * @brief Edge data with a memory smaller or larger than a static config describes is refused.
 */
TEST (tensorQueryValidate, staticSize_n)
{
  GstTensorsConfig config;
  const gsize smaller[] = { 4, 15 };
  const gsize larger[] = { 5, 16 };
  nns_edge_data_h data_h;

  _make_static_config (&config);

  data_h = _make_edge_data (smaller, 2);
  ASSERT_NE (data_h, nullptr);
  EXPECT_FALSE (gst_tensor_query_validate_edge_data (data_h, &config));
  nns_edge_data_destroy (data_h);

  data_h = _make_edge_data (larger, 2);
  ASSERT_NE (data_h, nullptr);
  EXPECT_FALSE (gst_tensor_query_validate_edge_data (data_h, &config));
  nns_edge_data_destroy (data_h);

  gst_tensors_config_free (&config);
}

/**
 * @brief Flexible and sparse memories whose header describes the data they carry are accepted.
 */
TEST (tensorQueryValidate, flexibleAndSparse)
{
  GstTensorsConfig config;
  nns_edge_data_h data_h;

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  ASSERT_EQ (nns_edge_data_create (&data_h), NNS_EDGE_ERROR_NONE);
  _add_meta_memory (data_h, _NNS_TENSOR_FORMAT_FLEXIBLE, 4, 0, 0);
  _add_meta_memory (data_h, _NNS_TENSOR_FORMAT_FLEXIBLE, 10, 0, 3);
  EXPECT_TRUE (gst_tensor_query_validate_edge_data (data_h, &config));
  nns_edge_data_destroy (data_h);

  config.info.format = _NNS_TENSOR_FORMAT_SPARSE;
  ASSERT_EQ (nns_edge_data_create (&data_h), NNS_EDGE_ERROR_NONE);
  _add_meta_memory (data_h, _NNS_TENSOR_FORMAT_SPARSE, 100, 2, 0);
  EXPECT_TRUE (gst_tensor_query_validate_edge_data (data_h, &config));
  nns_edge_data_destroy (data_h);

  gst_tensors_config_free (&config);
}

/**
 * @brief Flexible memories without a valid header are refused.
 */
TEST (tensorQueryValidate, flexibleHeader_n)
{
  GstTensorsConfig config;
  const gsize short_size[] = { 64 };
  const gsize no_header[] = { 256 };
  nns_edge_data_h data_h;

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  data_h = _make_edge_data (short_size, 1);
  ASSERT_NE (data_h, nullptr);
  EXPECT_FALSE (gst_tensor_query_validate_edge_data (data_h, &config));
  nns_edge_data_destroy (data_h);

  data_h = _make_edge_data (no_header, 1);
  ASSERT_NE (data_h, nullptr);
  EXPECT_FALSE (gst_tensor_query_validate_edge_data (data_h, &config));
  nns_edge_data_destroy (data_h);

  gst_tensors_config_free (&config);
}

/**
 * @brief Flexible and sparse memories shorter than the data their header describes are refused.
 */
TEST (tensorQueryValidate, flexibleTruncated_n)
{
  GstTensorsConfig config;
  nns_edge_data_h data_h;

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  /* The first memory is valid, the second one lacks a byte. */
  ASSERT_EQ (nns_edge_data_create (&data_h), NNS_EDGE_ERROR_NONE);
  _add_meta_memory (data_h, _NNS_TENSOR_FORMAT_FLEXIBLE, 4, 0, 0);
  _add_meta_memory (data_h, _NNS_TENSOR_FORMAT_FLEXIBLE, 10, 0, -1);
  EXPECT_FALSE (gst_tensor_query_validate_edge_data (data_h, &config));
  nns_edge_data_destroy (data_h);

  config.info.format = _NNS_TENSOR_FORMAT_SPARSE;
  ASSERT_EQ (nns_edge_data_create (&data_h), NNS_EDGE_ERROR_NONE);
  _add_meta_memory (data_h, _NNS_TENSOR_FORMAT_SPARSE, 100, 2, -1);
  EXPECT_FALSE (gst_tensor_query_validate_edge_data (data_h, &config));
  nns_edge_data_destroy (data_h);

  gst_tensors_config_free (&config);
}

/**
 * @brief Edge data without memories, or invalid parameters, are refused.
 */
TEST (tensorQueryValidate, invalidParam_n)
{
  GstTensorsConfig config;
  const gsize sizes[] = { 4, 16 };
  nns_edge_data_h data_h;

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  ASSERT_EQ (nns_edge_data_create (&data_h), NNS_EDGE_ERROR_NONE);
  EXPECT_FALSE (gst_tensor_query_validate_edge_data (data_h, &config));
  nns_edge_data_destroy (data_h);

  EXPECT_FALSE (gst_tensor_query_validate_edge_data (nullptr, &config));
  gst_tensors_config_free (&config);

  data_h = _make_edge_data (sizes, 2);
  ASSERT_NE (data_h, nullptr);
  EXPECT_FALSE (gst_tensor_query_validate_edge_data (data_h, nullptr));
  nns_edge_data_destroy (data_h);
}

/**
 * @brief Run gst_tensor_query_config_from_caps() on the caps string and return the number of tensors it describes.
 * @param[out] is_tensor The return value of gst_tensor_query_config_from_caps().
 */
static guint
_config_from_caps_string (const gchar *str, gboolean *is_tensor)
{
  GstTensorsConfig config;
  GstCaps *caps = gst_caps_from_string (str);
  guint num;

  *is_tensor = gst_tensor_query_config_from_caps (caps, &config);
  num = config.info.num_tensors;

  gst_tensors_config_free (&config);
  gst_caps_unref (caps);
  return num;
}

/**
 * @brief Fixed tensor caps are checked with their config, with or without a framerate.
 */
TEST (tensorQueryValidate, configFromCaps)
{
  gboolean is_tensor = FALSE;

  EXPECT_EQ (_config_from_caps_string ("other/tensor,dimension=(string)4,type=(string)uint8,framerate=(fraction)0/1",
                 &is_tensor),
      1U);
  EXPECT_TRUE (is_tensor);

  EXPECT_EQ (_config_from_caps_string (
                 "other/tensor,dimension=(string)4,type=(string)uint8", &is_tensor),
      1U);
  EXPECT_TRUE (is_tensor);

  _config_from_caps_string ("other/tensors,format=flexible", &is_tensor);
  EXPECT_TRUE (is_tensor);
}

/**
 * @brief Tensor caps that do not describe valid tensors give an empty config, so that every data is refused.
 */
TEST (tensorQueryValidate, configFromIncompleteCaps_n)
{
  gboolean is_tensor = FALSE;

  EXPECT_EQ (_config_from_caps_string (
                 "other/tensor,type=(string)uint8,framerate=(fraction)0/1", &is_tensor),
      0U);
  EXPECT_TRUE (is_tensor);

  EXPECT_EQ (_config_from_caps_string ("other/tensors,format=static,num_tensors=2,dimensions=(string)4,types=(string)uint8",
                 &is_tensor),
      0U);
  EXPECT_TRUE (is_tensor);
}

/**
 * @brief Non-tensor, unfixed and missing caps are not checked as tensors.
 */
TEST (tensorQueryValidate, configFromNonTensorCaps_n)
{
  GstTensorsConfig config;
  gboolean is_tensor = TRUE;

  _config_from_caps_string (
      "video/x-raw,format=RGB,width=4,height=4,framerate=(fraction)0/1", &is_tensor);
  EXPECT_FALSE (is_tensor);

  is_tensor = TRUE;
  _config_from_caps_string (
      "other/tensor,dimension=(string)4,type=(string){ uint8, int8 }", &is_tensor);
  EXPECT_FALSE (is_tensor);

  EXPECT_FALSE (gst_tensor_query_config_from_caps (nullptr, &config));
  gst_tensors_config_free (&config);
}

/**
 * @brief State shared with the raw nnstreamer-edge peer of the element tests.
 */
typedef struct {
  GMutex lock;
  gchar *client_id; /**< client id of the last data the raw server received */
  guint requests; /**< number of data the raw server received */
} RawPeer;

/**
 * @brief Event callback of the raw nnstreamer-edge peer; accepts any caps and records the client id.
 */
static int
_raw_peer_event_cb (nns_edge_event_h event_h, void *user_data)
{
  RawPeer *peer = (RawPeer *) user_data;
  nns_edge_event_e type;
  nns_edge_data_h data_h;
  char *val = nullptr;

  if (nns_edge_event_get_type (event_h, &type) != NNS_EDGE_ERROR_NONE)
    return NNS_EDGE_ERROR_NONE;

  if (type == NNS_EDGE_EVENT_NEW_DATA_RECEIVED && peer) {
    if (nns_edge_event_parse_new_data (event_h, &data_h) == NNS_EDGE_ERROR_NONE) {
      nns_edge_data_get_info (data_h, "client_id", &val);
      g_mutex_lock (&peer->lock);
      g_free (peer->client_id);
      peer->client_id = val;
      peer->requests++;
      g_mutex_unlock (&peer->lock);
      nns_edge_data_destroy (data_h);
    }
  }

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Count the buffers a tensor_sink receives.
 */
static void
_count_new_data (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  g_atomic_int_inc ((gint *) user_data);
}

/**
 * @brief Count the buffers a fakesink receives.
 */
static void
_count_handoff (GstElement *element, GstBuffer *buffer, GstPad *pad, gpointer user_data)
{
  g_atomic_int_inc ((gint *) user_data);
}

/**
 * @brief Wait until the sink receives a buffer or the pipeline posts an error.
 * @return TRUE if the pipeline posted an error.
 */
static gboolean
_wait_data_or_error (GstElement *pipeline, guint *count, guint timeout_ms)
{
  GstBus *bus = gst_element_get_bus (pipeline);
  GstMessage *msg = nullptr;
  guint waited;

  for (waited = 0; waited < timeout_ms && !msg && g_atomic_int_get ((gint *) count) == 0;
       waited += 10)
    msg = gst_bus_timed_pop_filtered (bus, 10 * GST_MSECOND, GST_MESSAGE_ERROR);

  if (msg)
    gst_message_unref (msg);
  gst_object_unref (bus);

  return (msg != nullptr);
}

#define QUERY_TEST_CAPS \
  "other/tensor,dimension=(string)4,type=(string)uint8,framerate=(fraction)0/1"
#define QUERY_TEST_CAPS_8 \
  "other/tensor,dimension=(string)8,type=(string)uint8,framerate=(fraction)0/1"
#define QUERY_TEST_CAPS_NO_RATE \
  "other/tensor,dimension=(string)4,type=(string)uint8"
#define QUERY_TEST_CAPS_FLEX \
  "other/tensors,format=flexible,framerate=(fraction)0/1"

/**
 * @brief Create edge data holding one flexible uint8 tensor of @a dim0 bytes.
 * @param extra The number of bytes to add to (or, if negative, cut from) the data described by the header.
 */
static nns_edge_data_h
_make_flex_edge_data (guint dim0, gint extra)
{
  nns_edge_data_h data_h = nullptr;

  if (nns_edge_data_create (&data_h) != NNS_EDGE_ERROR_NONE)
    return nullptr;

  _add_meta_memory (data_h, _NNS_TENSOR_FORMAT_FLEXIBLE, dim0, 0, extra);
  return data_h;
}

/**
 * @brief Create a tensor_query_serversrc pipeline whose capsfilter "cf" has the given caps.
 */
static GstElement *
_make_serversrc_pipeline (const gchar *caps, guint port, guint *count)
{
  gchar *pipeline;
  GstElement *gstpipe, *sink;

  pipeline = g_strdup_printf (
      "tensor_query_serversrc host=127.0.0.1 port=%u ! capsfilter name=cf caps=\"%s\" ! "
      "tee name=t t. ! queue ! tensor_query_serversink async=false "
      "t. ! queue ! fakesink name=sinkx signal-handoffs=true sync=false async=false",
      port, caps);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  g_free (pipeline);
  if (!gstpipe)
    return nullptr;

  sink = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  g_signal_connect (sink, "handoff", (GCallback) _count_handoff, count);
  gst_object_unref (sink);

  return gstpipe;
}

/**
 * @brief Connect a raw nnstreamer-edge query client to tensor_query_serversrc and send the edge data in order.
 * @param data The edge data to send (transfer full).
 * @return The client handle, to be released after the server handled the data.
 */
static nns_edge_h
_raw_client_send (guint port, nns_edge_data_h *data, guint num)
{
  nns_edge_h client_h = nullptr;
  char *client_id = nullptr;
  gchar *port_str;
  guint i;

  EXPECT_EQ (nns_edge_create_handle ("rawclient", NNS_EDGE_CONNECT_TYPE_TCP,
                 NNS_EDGE_NODE_TYPE_QUERY_CLIENT, &client_h),
      NNS_EDGE_ERROR_NONE);
  nns_edge_set_event_callback (client_h, _raw_peer_event_cb, nullptr);
  nns_edge_set_info (client_h, "HOST", "127.0.0.1");
  port_str = g_strdup_printf ("%u", get_available_port ());
  nns_edge_set_info (client_h, "PORT", port_str);
  g_free (port_str);
  EXPECT_EQ (nns_edge_start (client_h), NNS_EDGE_ERROR_NONE);
  EXPECT_EQ (nns_edge_connect (client_h, "127.0.0.1", port), NNS_EDGE_ERROR_NONE);

  nns_edge_get_info (client_h, "client_id", &client_id);
  for (i = 0; i < num; i++) {
    nns_edge_data_set_info (data[i], "client_id", client_id);
    EXPECT_EQ (nns_edge_send (client_h, data[i]), NNS_EDGE_ERROR_NONE);
    nns_edge_data_destroy (data[i]);
  }
  g_free (client_id);

  return client_h;
}

/**
 * @brief Send requests from a raw nnstreamer-edge client to tensor_query_serversrc.
 * @param caps The caps of the serversrc.
 * @param data The requests to send in order (transfer full).
 * @param[out] received The number of buffers the serversrc pushed downstream.
 * @return TRUE if the server pipeline posted an error.
 */
static gboolean
_send_to_serversrc (const gchar *caps, nns_edge_data_h *data, guint num, guint *received)
{
  GstElement *gstpipe;
  nns_edge_h client_h;
  guint port, count = 0;
  gboolean error;

  port = get_available_port ();
  gstpipe = _make_serversrc_pipeline (caps, port, &count);
  EXPECT_NE (gstpipe, nullptr);
  if (!gstpipe)
    return FALSE;

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  client_h = _raw_client_send (port, data, num);

  error = _wait_data_or_error (gstpipe, &count, 3000);
  if (!error)
    wait_pipeline_process_buffers (&count, num, 3000);
  *received = (guint) g_atomic_int_get (&count);

  nns_edge_release_handle (client_h);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (gstpipe);

  return error;
}

/**
 * @brief tensor_query_serversrc pushes every request that matches its caps.
 */
TEST (tensorQuery, serverSrcAcceptsMatchingData)
{
  const gsize sizes[] = { 4 };
  nns_edge_data_h data[3];
  guint received = 0;

  data[0] = _make_edge_data (sizes, 1);
  data[1] = _make_edge_data (sizes, 1);
  data[2] = _make_edge_data (sizes, 1);
  EXPECT_FALSE (_send_to_serversrc (QUERY_TEST_CAPS, data, 3, &received));
  EXPECT_EQ (received, 3U);
}

/**
 * @brief tensor_query_serversrc refuses a request whose memory size does not match its caps.
 */
TEST (tensorQuery, serverSrcRefusesWrongSize_n)
{
  const gsize sizes[] = { 8 };
  nns_edge_data_h data = _make_edge_data (sizes, 1);
  guint received = 0;

  EXPECT_TRUE (_send_to_serversrc (QUERY_TEST_CAPS, &data, 1, &received));
  EXPECT_EQ (received, 0U);
}

/**
 * @brief tensor_query_serversrc refuses a request whose memory count does not match its caps.
 */
TEST (tensorQuery, serverSrcRefusesWrongCount_n)
{
  const gsize sizes[] = { 2, 2 };
  nns_edge_data_h data = _make_edge_data (sizes, 2);
  guint received = 0;

  EXPECT_TRUE (_send_to_serversrc (QUERY_TEST_CAPS, &data, 1, &received));
  EXPECT_EQ (received, 0U);
}

/**
 * @brief tensor_query_serversrc pushes a matching request also when its tensor caps have no framerate.
 */
TEST (tensorQuery, serverSrcAcceptsMatchingDataWithoutFramerate)
{
  const gsize sizes[] = { 4 };
  nns_edge_data_h data = _make_edge_data (sizes, 1);
  guint received = 0;

  EXPECT_FALSE (_send_to_serversrc (QUERY_TEST_CAPS_NO_RATE, &data, 1, &received));
  EXPECT_EQ (received, 1U);
}

/**
 * @brief tensor_query_serversrc refuses a wrong-size request also when its tensor caps have no framerate.
 */
TEST (tensorQuery, serverSrcRefusesWrongSizeWithoutFramerate_n)
{
  const gsize sizes[] = { 8 };
  nns_edge_data_h data = _make_edge_data (sizes, 1);
  guint received = 0;

  EXPECT_TRUE (_send_to_serversrc (QUERY_TEST_CAPS_NO_RATE, &data, 1, &received));
  EXPECT_EQ (received, 0U);
}

/**
 * @brief tensor_query_serversrc pushes a flexible request whose header describes the data it carries.
 */
TEST (tensorQuery, serverSrcAcceptsFlexibleData)
{
  nns_edge_data_h data = _make_flex_edge_data (10, 0);
  guint received = 0;

  EXPECT_FALSE (_send_to_serversrc (QUERY_TEST_CAPS_FLEX, &data, 1, &received));
  EXPECT_EQ (received, 1U);
}

/**
 * @brief tensor_query_serversrc refuses a flexible request shorter than the data its header describes.
 */
TEST (tensorQuery, serverSrcRefusesTruncatedFlexibleData_n)
{
  nns_edge_data_h data = _make_flex_edge_data (10, -1);
  guint received = 0;

  EXPECT_TRUE (_send_to_serversrc (QUERY_TEST_CAPS_FLEX, &data, 1, &received));
  EXPECT_EQ (received, 0U);
}

/**
 * @brief tensor_query_serversrc checks requests against the caps it negotiated last.
 */
TEST (tensorQuery, serverSrcFollowsNewCaps)
{
  const gsize four[] = { 4 };
  const gsize eight[] = { 8 };
  GstElement *gstpipe, *cf;
  GstCaps *caps;
  nns_edge_data_h data;
  nns_edge_h client_h;
  guint port, count = 0;

  port = get_available_port ();
  gstpipe = _make_serversrc_pipeline (QUERY_TEST_CAPS, port, &count);
  ASSERT_NE (gstpipe, nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  data = _make_edge_data (four, 1);
  client_h = _raw_client_send (port, &data, 1);
  EXPECT_FALSE (_wait_data_or_error (gstpipe, &count, 3000));
  EXPECT_EQ ((guint) g_atomic_int_get ((gint *) &count), 1U);
  nns_edge_release_handle (client_h);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  /* The same element now negotiates 8-byte tensors. */
  cf = gst_bin_get_by_name (GST_BIN (gstpipe), "cf");
  caps = gst_caps_from_string (QUERY_TEST_CAPS_8);
  g_object_set (cf, "caps", caps, NULL);
  gst_caps_unref (caps);
  gst_object_unref (cf);

  g_atomic_int_set ((gint *) &count, 0);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  data = _make_edge_data (eight, 1);
  client_h = _raw_client_send (port, &data, 1);
  EXPECT_FALSE (_wait_data_or_error (gstpipe, &count, 3000));
  EXPECT_EQ ((guint) g_atomic_int_get ((gint *) &count), 1U);
  nns_edge_release_handle (client_h);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (gstpipe);
}

/**
 * @brief Answer one request of tensor_query_client from a raw nnstreamer-edge server.
 * @param caps The caps of the client input and of the server answers.
 * @param input The request buffer the client sends (transfer full).
 * @param answer The answer to send back (transfer full).
 * @param[out] received The number of buffers the client pushed downstream.
 * @return TRUE if the client pipeline posted an error.
 */
static gboolean
_answer_client (const gchar *caps, GstBuffer *input, nns_edge_data_h answer, guint *received)
{
  gchar *pipeline, *caps_str, *port_str;
  GstElement *gstpipe, *sink, *appsrc;
  nns_edge_h server_h = nullptr;
  RawPeer peer;
  guint port, count = 0, waited = 0;
  gchar *client_id = nullptr;
  gboolean error = FALSE;

  g_mutex_init (&peer.lock);
  peer.client_id = nullptr;
  peer.requests = 0;

  port = get_available_port ();
  EXPECT_EQ (nns_edge_create_handle ("rawserver", NNS_EDGE_CONNECT_TYPE_TCP,
                 NNS_EDGE_NODE_TYPE_QUERY_SERVER, &server_h),
      NNS_EDGE_ERROR_NONE);
  nns_edge_set_event_callback (server_h, _raw_peer_event_cb, &peer);
  nns_edge_set_info (server_h, "HOST", "127.0.0.1");
  port_str = g_strdup_printf ("%u", port);
  nns_edge_set_info (server_h, "PORT", port_str);
  g_free (port_str);
  caps_str = g_strdup_printf (
      "@query_server_src_caps@%s@query_server_sink_caps@%s", caps, caps);
  nns_edge_set_info (server_h, "CAPS", caps_str);
  g_free (caps_str);
  EXPECT_EQ (nns_edge_start (server_h), NNS_EDGE_ERROR_NONE);

  pipeline = g_strdup_printf ("appsrc name=appsrc caps=\"%s\" ! "
                              "tensor_query_client dest-host=127.0.0.1 dest-port=%u timeout=10000 ! "
                              "tensor_sink name=sinkx async=false",
      caps, port);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  g_free (pipeline);
  EXPECT_NE (gstpipe, nullptr);

  sink = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  g_signal_connect (sink, "new-data", (GCallback) _count_new_data, &count);
  gst_object_unref (sink);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  appsrc = gst_bin_get_by_name (GST_BIN (gstpipe), "appsrc");
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc), input), GST_FLOW_OK);
  gst_object_unref (appsrc);

  /* Wait for the request, then answer it. */
  while (waited < 5000) {
    g_mutex_lock (&peer.lock);
    client_id = g_strdup (peer.client_id);
    g_mutex_unlock (&peer.lock);
    if (client_id)
      break;
    g_usleep (10000);
    waited += 10;
  }
  EXPECT_NE (client_id, nullptr);

  nns_edge_data_set_info (answer, "client_id", client_id);
  g_free (client_id);
  EXPECT_EQ (nns_edge_send (server_h, answer), NNS_EDGE_ERROR_NONE);
  nns_edge_data_destroy (answer);

  error = _wait_data_or_error (gstpipe, &count, 3000);
  *received = (guint) g_atomic_int_get (&count);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (gstpipe);
  nns_edge_release_handle (server_h);

  g_free (peer.client_id);
  g_mutex_clear (&peer.lock);

  return error;
}

/**
 * @brief Create a request buffer of one static uint8 tensor of 4 bytes.
 */
static GstBuffer *
_static_input (void)
{
  return gst_buffer_new_wrapped (g_malloc0 (4), 4);
}

/**
 * @brief Create a request buffer of one flexible uint8 tensor of 4 bytes.
 */
static GstBuffer *
_flex_input (void)
{
  GstTensorMetaInfo meta;
  gsize hsize;
  gpointer data;

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_UINT8;
  meta.dimension[0] = 4;
  meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  hsize = gst_tensor_meta_info_get_header_size (&meta);

  data = g_malloc0 (hsize + 4);
  gst_tensor_meta_info_update_header (&meta, data);
  return gst_buffer_new_wrapped (data, hsize + 4);
}

/**
 * @brief tensor_query_client pushes an answer that matches the server caps.
 */
TEST (tensorQuery, clientAcceptsMatchingData)
{
  const gsize sizes[] = { 4 };
  guint received = 0;

  EXPECT_FALSE (_answer_client (
      QUERY_TEST_CAPS, _static_input (), _make_edge_data (sizes, 1), &received));
  EXPECT_EQ (received, 1U);
}

/**
 * @brief tensor_query_client refuses an answer whose memory size does not match the server caps.
 */
TEST (tensorQuery, clientRefusesWrongSize_n)
{
  const gsize sizes[] = { 8 };
  guint received = 0;

  EXPECT_TRUE (_answer_client (
      QUERY_TEST_CAPS, _static_input (), _make_edge_data (sizes, 1), &received));
  EXPECT_EQ (received, 0U);
}

/**
 * @brief tensor_query_client refuses an answer whose memory count does not match the server caps.
 */
TEST (tensorQuery, clientRefusesWrongCount_n)
{
  const gsize sizes[] = { 2, 2 };
  guint received = 0;

  EXPECT_TRUE (_answer_client (
      QUERY_TEST_CAPS, _static_input (), _make_edge_data (sizes, 2), &received));
  EXPECT_EQ (received, 0U);
}

/**
 * @brief tensor_query_client pushes a flexible answer whose header describes the data it carries.
 */
TEST (tensorQuery, clientAcceptsFlexibleData)
{
  guint received = 0;

  EXPECT_FALSE (_answer_client (QUERY_TEST_CAPS_FLEX, _flex_input (),
      _make_flex_edge_data (10, 0), &received));
  EXPECT_EQ (received, 1U);
}

/**
 * @brief tensor_query_client refuses a flexible answer shorter than the data its header describes.
 */
TEST (tensorQuery, clientRefusesTruncatedFlexibleData_n)
{
  guint received = 0;

  EXPECT_TRUE (_answer_client (QUERY_TEST_CAPS_FLEX, _flex_input (),
      _make_flex_edge_data (10, -1), &received));
  EXPECT_EQ (received, 0U);
}

/**
 * @brief Create edge data holding one 4-byte memory whose first byte is @a id.
 */
static nns_edge_data_h
_make_id_data (guint8 id)
{
  const gsize sizes[] = { 4 };
  nns_edge_data_h data_h = _make_edge_data (sizes, 1);
  void *data = nullptr;
  nns_size_t len = 0;

  if (data_h && nns_edge_data_get (data_h, 0, &data, &len) == NNS_EDGE_ERROR_NONE)
    *((guint8 *) data) = id;

  return data_h;
}

/**
 * @brief Pop the edge data of a receive queue and check that they are @a expected, in order.
 */
static void
_expect_queue (GAsyncQueue *queue, nns_edge_data_h *expected, guint num)
{
  nns_edge_data_h data_h;
  guint i;

  EXPECT_EQ (g_async_queue_length (queue), (gint) num);
  for (i = 0; i < num; i++) {
    data_h = g_async_queue_try_pop (queue);
    EXPECT_EQ (data_h, expected[i]);
    if (data_h)
      nns_edge_data_destroy (data_h);
  }
  EXPECT_EQ (g_async_queue_length (queue), 0);
}

/**
 * @brief Without a limit, the receive queue keeps every data in order.
 */
TEST (tensorQueryQueue, pushNoLimit)
{
  GAsyncQueue *queue = g_async_queue_new ();
  nns_edge_data_h data[10];
  guint i;

  for (i = 0; i < 10; i++) {
    data[i] = _make_id_data (i);
    ASSERT_NE (data[i], nullptr);
    EXPECT_EQ (gst_tensor_query_push_edge_data (queue, data[i], 0), 0U);
  }

  _expect_queue (queue, data, 10);
  g_async_queue_unref (queue);
}

/**
 * @brief A limit the queue does not exceed drops nothing, the largest limit included.
 */
TEST (tensorQueryQueue, pushWithinLimit)
{
  GAsyncQueue *queue = g_async_queue_new ();
  nns_edge_data_h data[3];
  guint i;

  for (i = 0; i < 3; i++) {
    data[i] = _make_id_data (i);
    ASSERT_NE (data[i], nullptr);
    EXPECT_EQ (gst_tensor_query_push_edge_data (queue, data[i], (i == 2) ? G_MAXUINT : 3U), 0U);
  }

  _expect_queue (queue, data, 3);
  g_async_queue_unref (queue);
}

/**
 * @brief A full receive queue drops its oldest data and keeps the latest ones.
 */
TEST (tensorQueryQueue, pushDropsOldest_n)
{
  GAsyncQueue *queue = g_async_queue_new ();
  nns_edge_data_h data[10];
  guint i, dropped = 0;

  for (i = 0; i < 10; i++) {
    data[i] = _make_id_data (i);
    ASSERT_NE (data[i], nullptr);
    dropped += gst_tensor_query_push_edge_data (queue, data[i], 3);
    EXPECT_EQ (g_async_queue_length (queue), (gint) MIN (i + 1, 3U));
  }
  EXPECT_EQ (dropped, 7U);

  _expect_queue (queue, &data[7], 3);
  g_async_queue_unref (queue);
}

/**
 * @brief A limit lowered while the queue holds more data drops all the excess oldest data at the next push.
 */
TEST (tensorQueryQueue, pushLoweredLimit_n)
{
  GAsyncQueue *queue = g_async_queue_new ();
  nns_edge_data_h data[6];
  guint i;

  for (i = 0; i < 5; i++) {
    data[i] = _make_id_data (i);
    ASSERT_NE (data[i], nullptr);
    EXPECT_EQ (gst_tensor_query_push_edge_data (queue, data[i], 0), 0U);
  }

  data[5] = _make_id_data (5);
  ASSERT_NE (data[5], nullptr);
  EXPECT_EQ (gst_tensor_query_push_edge_data (queue, data[5], 1), 5U);

  _expect_queue (queue, &data[5], 1);
  g_async_queue_unref (queue);
}

/**
 * @brief Invalid parameters push nothing.
 */
TEST (tensorQueryQueue, pushInvalidParam_n)
{
  GAsyncQueue *queue = g_async_queue_new ();
  nns_edge_data_h data_h = _make_id_data (0);

  ASSERT_NE (data_h, nullptr);
  EXPECT_EQ (gst_tensor_query_push_edge_data (nullptr, data_h, 1), 0U);
  EXPECT_EQ (gst_tensor_query_push_edge_data (queue, nullptr, 1), 0U);
  EXPECT_EQ (g_async_queue_length (queue), 0);

  nns_edge_data_destroy (data_h);
  g_async_queue_unref (queue);
}

/**
 * @brief max-buffers of the query elements is 0 (no limit) by default and can be set.
 */
TEST (tensorQuery, maxBuffersProperty)
{
  const gchar *names[] = { "tensor_query_serversrc", "tensor_query_client" };
  GstElement *element;
  guint i, val;

  for (i = 0; i < G_N_ELEMENTS (names); i++) {
    element = gst_element_factory_make (names[i], nullptr);
    ASSERT_NE (element, nullptr);

    val = 1U;
    g_object_get (element, "max-buffers", &val, NULL);
    EXPECT_EQ (val, 0U);

    g_object_set (element, "max-buffers", 5U, NULL);
    g_object_get (element, "max-buffers", &val, NULL);
    EXPECT_EQ (val, 5U);

    gst_object_unref (element);
  }
}

/**
 * @brief The first bytes of the buffers a fakesink received, in order.
 */
typedef struct {
  GMutex lock;
  GArray *ids;
} IdLog;

/**
 * @brief Record the first byte of a buffer a fakesink receives.
 */
static void
_log_id (GstElement *element, GstBuffer *buffer, GstPad *pad, gpointer user_data)
{
  IdLog *log = (IdLog *) user_data;
  guint8 id = 0xff;

  gst_buffer_extract (buffer, 0, &id, 1);
  g_mutex_lock (&log->lock);
  g_array_append_val (log->ids, id);
  g_mutex_unlock (&log->lock);
}

/**
 * @brief Wait until the id log has @a num entries, then a little longer to catch extra buffers.
 */
static void
_wait_ids (IdLog *log, guint num, guint timeout_ms)
{
  guint len, waited;

  for (waited = 0; waited <= timeout_ms; waited += 10) {
    g_mutex_lock (&log->lock);
    len = log->ids->len;
    g_mutex_unlock (&log->lock);
    if (len >= num)
      break;
    g_usleep (10000);
  }

  g_usleep (200000);
}

/**
 * @brief Adds up the data the elements of a debug category report as dropped from their receive queue.
 */
typedef struct {
  const gchar *category;
  gint dropped;
} DropCounter;

static GMutex drop_counter_lock; /**< guards drop_counter */
static DropCounter *drop_counter; /**< the counter in use, guarded by drop_counter_lock */

/**
 * @brief Debug log function adding up the "Dropped N ..." messages of the category being counted.
 */
static void
_count_dropped (GstDebugCategory *category, GstDebugLevel level,
    const gchar *file, const gchar *function, gint line, GObject *object,
    GstDebugMessage *message, gpointer user_data)
{
  const gchar *text;
  gint num;

  g_mutex_lock (&drop_counter_lock);
  if (drop_counter
      && g_strcmp0 (gst_debug_category_get_name (category), drop_counter->category) == 0) {
    text = gst_debug_message_get (message);
    if (text && g_str_has_prefix (text, "Dropped ")) {
      num = (gint) g_ascii_strtoull (text + 8, NULL, 10);
      g_atomic_int_add (&drop_counter->dropped, num);
    }
  }
  g_mutex_unlock (&drop_counter_lock);
}

/**
 * @brief Start counting the data dropped by the elements of a debug category.
 * The log function is added once and never removed, since removing one leaks the old list in GStreamer.
 */
static void
_drop_counter_start (DropCounter *counter, const gchar *category)
{
  static gsize added = 0;

  if (g_once_init_enter (&added)) {
    gst_debug_add_log_function (_count_dropped, nullptr, nullptr);
    g_once_init_leave (&added, 1);
  }

  counter->category = category;
  counter->dropped = 0;
  gst_debug_set_threshold_for_name (category, GST_LEVEL_DEBUG);

  g_mutex_lock (&drop_counter_lock);
  drop_counter = counter;
  g_mutex_unlock (&drop_counter_lock);
}

/**
 * @brief Stop counting the dropped data.
 */
static void
_drop_counter_stop (DropCounter *counter)
{
  g_mutex_lock (&drop_counter_lock);
  drop_counter = nullptr;
  g_mutex_unlock (&drop_counter_lock);

  gst_debug_unset_threshold_for_name (counter->category);
}

/**
 * @brief Wait until the counter reaches @a num, then a little longer to catch extra drops.
 * @return The number of dropped data.
 */
static guint
_wait_dropped (DropCounter *counter, guint num, guint timeout_ms)
{
  guint waited;

  for (waited = 0; waited <= timeout_ms; waited += 10) {
    if ((guint) g_atomic_int_get (&counter->dropped) >= num)
      break;
    g_usleep (10000);
  }

  g_usleep (200000);
  return (guint) g_atomic_int_get (&counter->dropped);
}

/**
 * @brief Pad probe callback that keeps buffers blocked until the probe is removed.
 */
static GstPadProbeReturn
_block_buffers (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  return GST_PAD_PROBE_OK;
}

/**
 * @brief Send ten requests (ids 0 to 9) to tensor_query_serversrc while its src pad is blocked, then unblock it.
 * @param max_buffers The max-buffers property of the serversrc.
 * @param[out] log The ids of the requests the serversrc pushed, in order.
 * @param expected The number of requests the serversrc is expected to push.
 * @return The number of requests the serversrc reported as dropped.
 */
static guint
_flood_serversrc (guint max_buffers, IdLog *log, guint expected)
{
  gchar *pipeline;
  GstElement *gstpipe, *element;
  GstPad *pad;
  nns_edge_data_h data[10];
  nns_edge_h client_h;
  DropCounter counter;
  gulong probe_id;
  guint i, port, dropped;

  _drop_counter_start (&counter, "tensor_query_serversrc");

  port = get_available_port ();
  pipeline = g_strdup_printf (
      "tensor_query_serversrc name=srcx host=127.0.0.1 port=%u max-buffers=%u ! "
      "capsfilter caps=\"%s\" ! tee name=t t. ! queue ! tensor_query_serversink async=false "
      "t. ! queue ! fakesink name=sinkx signal-handoffs=true sync=false async=false",
      port, max_buffers, QUERY_TEST_CAPS);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  g_free (pipeline);
  EXPECT_NE (gstpipe, nullptr);
  if (!gstpipe) {
    _drop_counter_stop (&counter);
    return 0;
  }

  element = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  g_signal_connect (element, "handoff", (GCallback) _log_id, log);
  gst_object_unref (element);

  element = gst_bin_get_by_name (GST_BIN (gstpipe), "srcx");
  pad = gst_element_get_static_pad (element, "src");
  gst_object_unref (element);
  probe_id = gst_pad_add_probe (pad,
      (GstPadProbeType) (GST_PAD_PROBE_TYPE_BLOCK | GST_PAD_PROBE_TYPE_BUFFER),
      _block_buffers, nullptr, nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  /* The serversrc takes one request and blocks pushing it; the others wait in its queue. */
  for (i = 0; i < 10; i++)
    data[i] = _make_id_data (i);
  client_h = _raw_client_send (port, data, 10);

  dropped = _wait_dropped (&counter, 10 - 1 - max_buffers, (max_buffers > 0) ? 5000 : 500);
  gst_pad_remove_probe (pad, probe_id);
  gst_object_unref (pad);
  _wait_ids (log, expected, 5000);

  nns_edge_release_handle (client_h);
  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (gstpipe);
  _drop_counter_stop (&counter);

  return dropped;
}

/**
 * @brief Without max-buffers, tensor_query_serversrc keeps every request that arrives while the pipeline is busy.
 */
TEST (tensorQuery, serverSrcKeepsAllRequestsByDefault)
{
  IdLog log;
  guint i;

  g_mutex_init (&log.lock);
  log.ids = g_array_new (FALSE, FALSE, sizeof (guint8));

  EXPECT_EQ (_flood_serversrc (0, &log, 10), 0U);

  EXPECT_EQ (log.ids->len, 10U);
  for (i = 0; i < log.ids->len; i++)
    EXPECT_EQ (g_array_index (log.ids, guint8, i), i);

  g_array_free (log.ids, TRUE);
  g_mutex_clear (&log.lock);
}

/**
 * @brief tensor_query_serversrc with max-buffers keeps the latest requests of a client out-running the pipeline.
 */
TEST (tensorQuery, serverSrcMaxBuffersDropsOldest_n)
{
  IdLog log;
  guint dropped, len;

#ifdef GST_DISABLE_GST_DEBUG
  GTEST_SKIP () << "The dropped requests are counted with the GStreamer debug log.";
#endif

  g_mutex_init (&log.lock);
  log.ids = g_array_new (FALSE, FALSE, sizeof (guint8));

  /**
   * Usually one request is taken before the pad blocks, 3 stay queued and 6 are dropped.
   * If all ten arrive before the element waits on its queue, 7 are dropped and 3 pushed.
   * Either way nothing is lost unaccounted and the last three pushed are the latest.
   */
  dropped = _flood_serversrc (3, &log, 3);
  EXPECT_GE (dropped, 6U);
  EXPECT_LE (dropped, 7U);

  len = log.ids->len;
  EXPECT_EQ (len + dropped, 10U);
  ASSERT_GE (len, 3U);
  EXPECT_EQ (g_array_index (log.ids, guint8, len - 3), 7U);
  EXPECT_EQ (g_array_index (log.ids, guint8, len - 2), 8U);
  EXPECT_EQ (g_array_index (log.ids, guint8, len - 1), 9U);

  g_array_free (log.ids, TRUE);
  g_mutex_clear (&log.lock);
}

/**
 * @brief Wait until the raw peer received @a num data.
 * @return TRUE if it did before the timeout.
 */
static gboolean
_wait_requests (RawPeer *peer, guint num, guint timeout_ms)
{
  guint waited, requests = 0;

  for (waited = 0; waited <= timeout_ms; waited += 10) {
    g_mutex_lock (&peer->lock);
    requests = peer->requests;
    g_mutex_unlock (&peer->lock);
    if (requests >= num)
      break;
    g_usleep (10000);
  }

  return (requests >= num);
}

/**
 * @brief Let two answers wait in the receive queue of tensor_query_client (max-request=1, no timeout).
 * The client sends two requests before any answer, the raw server answers both (ids 1 and 2),
 * then the client gets a third and a fourth input.
 * @param max_buffers The max-buffers property of the client.
 * @param[out] log The ids of the answers the client pushed, in order.
 * @param[out] dropped The number of answers the client reported as dropped.
 * @return TRUE if the client sent the third input to the server.
 */
static gboolean
_answer_client_twice (guint max_buffers, IdLog *log, guint *dropped)
{
  gchar *pipeline, *caps_str, *port_str, *client_id;
  GstElement *gstpipe, *element, *appsrc;
  nns_edge_h server_h = nullptr;
  nns_edge_data_h answer;
  DropCounter counter;
  RawPeer peer;
  guint i, port;
  gboolean third_request = FALSE;

  g_mutex_init (&peer.lock);
  peer.client_id = nullptr;
  peer.requests = 0;
  _drop_counter_start (&counter, "tensor_query_client");

  port = get_available_port ();
  EXPECT_EQ (nns_edge_create_handle ("rawserver", NNS_EDGE_CONNECT_TYPE_TCP,
                 NNS_EDGE_NODE_TYPE_QUERY_SERVER, &server_h),
      NNS_EDGE_ERROR_NONE);
  nns_edge_set_event_callback (server_h, _raw_peer_event_cb, &peer);
  nns_edge_set_info (server_h, "HOST", "127.0.0.1");
  port_str = g_strdup_printf ("%u", port);
  nns_edge_set_info (server_h, "PORT", port_str);
  g_free (port_str);
  caps_str = g_strdup_printf ("@query_server_src_caps@%s@query_server_sink_caps@%s",
      QUERY_TEST_CAPS, QUERY_TEST_CAPS);
  nns_edge_set_info (server_h, "CAPS", caps_str);
  g_free (caps_str);
  EXPECT_EQ (nns_edge_start (server_h), NNS_EDGE_ERROR_NONE);

  pipeline = g_strdup_printf ("appsrc name=appsrc caps=\"%s\" ! "
                              "tensor_query_client dest-host=127.0.0.1 dest-port=%u max-request=1 max-buffers=%u ! "
                              "fakesink name=sinkx signal-handoffs=true sync=false async=false",
      QUERY_TEST_CAPS, port, max_buffers);
  gstpipe = gst_parse_launch (pipeline, nullptr);
  g_free (pipeline);
  EXPECT_NE (gstpipe, nullptr);
  if (!gstpipe)
    goto done;

  element = gst_bin_get_by_name (GST_BIN (gstpipe), "sinkx");
  g_signal_connect (element, "handoff", (GCallback) _log_id, log);
  gst_object_unref (element);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  appsrc = gst_bin_get_by_name (GST_BIN (gstpipe), "appsrc");

  /* Two requests are pending before any answer arrives. */
  for (i = 1; i <= 2; i++) {
    EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc), _static_input ()), GST_FLOW_OK);
    EXPECT_TRUE (_wait_requests (&peer, i, 5000));
  }

  g_mutex_lock (&peer.lock);
  client_id = g_strdup (peer.client_id);
  g_mutex_unlock (&peer.lock);
  for (i = 1; i <= 2; i++) {
    answer = _make_id_data (i);
    nns_edge_data_set_info (answer, "client_id", client_id);
    EXPECT_EQ (nns_edge_send (server_h, answer), NNS_EDGE_ERROR_NONE);
    nns_edge_data_destroy (answer);
  }
  g_free (client_id);

  *dropped = _wait_dropped (&counter, 1, (max_buffers > 0) ? 5000 : 500);

  /* The third input goes to the server only if the client counts no more than one pending request. */
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc), _static_input ()), GST_FLOW_OK);
  third_request = _wait_requests (&peer, 3, (max_buffers > 0) ? 5000 : 500);
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc), _static_input ()), GST_FLOW_OK);
  _wait_ids (log, 2, 1000);
  gst_object_unref (appsrc);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  gst_object_unref (gstpipe);

done:
  nns_edge_release_handle (server_h);
  _drop_counter_stop (&counter);
  g_free (peer.client_id);
  g_mutex_clear (&peer.lock);

  return third_request;
}

/**
 * @brief Without max-buffers, tensor_query_client keeps every answer and pushes them in order.
 */
TEST (tensorQuery, clientKeepsAllAnswersByDefault)
{
  IdLog log;
  guint dropped = 1;

  g_mutex_init (&log.lock);
  log.ids = g_array_new (FALSE, FALSE, sizeof (guint8));

  /* Both answers are pending, so max-request=1 holds the third input back. */
  EXPECT_FALSE (_answer_client_twice (0, &log, &dropped));
  EXPECT_EQ (dropped, 0U);

  ASSERT_EQ (log.ids->len, 2U);
  EXPECT_EQ (g_array_index (log.ids, guint8, 0), 1U);
  EXPECT_EQ (g_array_index (log.ids, guint8, 1), 2U);

  g_array_free (log.ids, TRUE);
  g_mutex_clear (&log.lock);
}

/**
 * @brief tensor_query_client with max-buffers keeps the latest answer and no longer counts a dropped one as pending.
 */
TEST (tensorQuery, clientMaxBuffersDropsOldest_n)
{
  IdLog log;
  guint dropped = 0;

#ifdef GST_DISABLE_GST_DEBUG
  GTEST_SKIP () << "The dropped answers are counted with the GStreamer debug log.";
#endif

  g_mutex_init (&log.lock);
  log.ids = g_array_new (FALSE, FALSE, sizeof (guint8));

  /* The dropped answer is not pending anymore, so max-request=1 lets the third input go. */
  EXPECT_TRUE (_answer_client_twice (1, &log, &dropped));
  EXPECT_EQ (dropped, 1U);

  ASSERT_GE (log.ids->len, 1U);
  EXPECT_EQ (g_array_index (log.ids, guint8, 0), 2U);

  g_array_free (log.ids, TRUE);
  g_mutex_clear (&log.lock);
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
