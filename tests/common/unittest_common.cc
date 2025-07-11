/**
 * Copyright (C) 2018 Samsung Electronics Co., Ltd.
 *
 * @file	unittest_common.cc
 * @date	31 May 2018
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @brief	Unit test module for NNStreamer common library
 * @see		https://github.com/nnstreamer/nnstreamer
 * @bug		No known bugs except for NYI items.
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <nnstreamer_conf.h>
#include <nnstreamer_plugin_api.h>
#include <tensor_common.h>
#include <unistd.h>
#include <unittest_util.h>

/**
 * @brief Internal util function to get the number of key in the string.
 */
static guint
count_key_string (const gchar *string, const gchar *key)
{
  const gchar *pos = string;
  guint count = 0;

  g_assert (string != NULL && key != NULL);

  while ((pos = g_strstr_len (pos, -1, key)) != NULL) {
    pos += strlen (key);
    count++;
  }

  return count;
}

/**
 * @brief Internal function to update tensors info.
 */
static void
fill_tensors_info_for_test (GstTensorsInfo *info1, GstTensorsInfo *info2)
{
  GstTensorInfo *_info1, *_info2;

  g_assert (info1 != NULL && info2 != NULL);

  gst_tensors_info_init (info1);
  gst_tensors_info_init (info2);

  info1->num_tensors = info2->num_tensors = 2;

  _info1 = gst_tensors_info_get_nth_info (info1, 0);
  _info2 = gst_tensors_info_get_nth_info (info2, 0);

  _info1->type = _info2->type = _NNS_INT64;
  _info1->dimension[0] = _info2->dimension[0] = 2;
  _info1->dimension[1] = _info2->dimension[1] = 3;
  _info1->dimension[2] = _info2->dimension[2] = 1;
  _info1->dimension[3] = _info2->dimension[3] = 1;
  _info1->dimension[4] = _info2->dimension[4] = 2;
  _info1->dimension[5] = _info2->dimension[5] = 3;
  _info1->dimension[6] = _info2->dimension[6] = 1;
  _info1->dimension[7] = _info2->dimension[7] = 1;

  _info1 = gst_tensors_info_get_nth_info (info1, 1);
  _info2 = gst_tensors_info_get_nth_info (info2, 1);

  _info1->type = _info2->type = _NNS_FLOAT64;
  _info1->dimension[0] = _info2->dimension[0] = 5;
  _info1->dimension[1] = _info2->dimension[1] = 5;
  _info1->dimension[2] = _info2->dimension[2] = 1;
  _info1->dimension[3] = _info2->dimension[3] = 1;
  _info1->dimension[4] = _info2->dimension[4] = 5;
  _info1->dimension[5] = _info2->dimension[5] = 5;
  _info1->dimension[6] = _info2->dimension[6] = 1;
  _info1->dimension[7] = _info2->dimension[7] = 1;
}

/**
 * @brief Internal function to update tensors info.
 */
static void
fill_tensors_config_for_test (GstTensorsConfig *conf1, GstTensorsConfig *conf2)
{
  g_assert (conf1 != NULL && conf2 != NULL);

  gst_tensors_config_init (conf1);
  gst_tensors_config_init (conf2);

  conf1->rate_n = conf2->rate_n = 1;
  conf1->rate_d = conf2->rate_d = 2;

  fill_tensors_info_for_test (&conf1->info, &conf2->info);
}

/**
 * @brief Test for int32 type string.
 */
TEST (commonGetTensorType, failure_n)
{
  EXPECT_EQ (gst_tensor_get_type (""), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type (NULL), _NNS_END);
}

/**
 * @brief Test for int32 type string.
 */
TEST (commonGetTensorType, int32)
{
  EXPECT_EQ (gst_tensor_get_type ("int32"), _NNS_INT32);
  EXPECT_EQ (gst_tensor_get_type ("INT32"), _NNS_INT32);
  EXPECT_EQ (gst_tensor_get_type ("iNt32"), _NNS_INT32);
  EXPECT_EQ (gst_tensor_get_type ("InT32"), _NNS_INT32);
}

/**
 * @brief Test for int32 type string.
 */
TEST (commonGetTensorType, int32_n)
{
  EXPECT_EQ (gst_tensor_get_type ("InT322"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("int3"), _NNS_END);
}

/**
 * @brief Test for int16 type string.
 */
TEST (commonGetTensorType, int16)
{
  EXPECT_EQ (gst_tensor_get_type ("int16"), _NNS_INT16);
  EXPECT_EQ (gst_tensor_get_type ("INT16"), _NNS_INT16);
  EXPECT_EQ (gst_tensor_get_type ("iNt16"), _NNS_INT16);
  EXPECT_EQ (gst_tensor_get_type ("InT16"), _NNS_INT16);
}

/**
 * @brief Test for int16 type string.
 */
TEST (commonGetTensorType, int16_n)
{
  EXPECT_EQ (gst_tensor_get_type ("InT162"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("int1"), _NNS_END);
}

/**
 * @brief Test for int8 type string.
 */
TEST (commonGetTensorType, int8)
{
  EXPECT_EQ (gst_tensor_get_type ("int8"), _NNS_INT8);
  EXPECT_EQ (gst_tensor_get_type ("INT8"), _NNS_INT8);
  EXPECT_EQ (gst_tensor_get_type ("iNt8"), _NNS_INT8);
  EXPECT_EQ (gst_tensor_get_type ("InT8"), _NNS_INT8);
}

/**
 * @brief Test for int8 type string.
 */
TEST (commonGetTensorType, int8_n)
{
  EXPECT_EQ (gst_tensor_get_type ("InT82"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("int3"), _NNS_END);
}

/**
 * @brief Test for uint32 type string.
 */
TEST (commonGetTensorType, uint32)
{
  EXPECT_EQ (gst_tensor_get_type ("uint32"), _NNS_UINT32);
  EXPECT_EQ (gst_tensor_get_type ("UINT32"), _NNS_UINT32);
  EXPECT_EQ (gst_tensor_get_type ("uiNt32"), _NNS_UINT32);
  EXPECT_EQ (gst_tensor_get_type ("UInT32"), _NNS_UINT32);
}

/**
 * @brief Test for uint32 type string.
 */
TEST (commonGetTensorType, uint32_n)
{
  EXPECT_EQ (gst_tensor_get_type ("UInT322"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("uint3"), _NNS_END);
}

/**
 * @brief Test for uint16 type string.
 */
TEST (commonGetTensorType, uint16)
{
  EXPECT_EQ (gst_tensor_get_type ("uint16"), _NNS_UINT16);
  EXPECT_EQ (gst_tensor_get_type ("UINT16"), _NNS_UINT16);
  EXPECT_EQ (gst_tensor_get_type ("uiNt16"), _NNS_UINT16);
  EXPECT_EQ (gst_tensor_get_type ("UInT16"), _NNS_UINT16);
}

/**
 * @brief Test for uint16 type string.
 */
TEST (commonGetTensorType, uint16_n)
{
  EXPECT_EQ (gst_tensor_get_type ("UInT162"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("uint1"), _NNS_END);
}

/**
 * @brief Test for uint8 type string.
 */
TEST (commonGetTensorType, uint8)
{
  EXPECT_EQ (gst_tensor_get_type ("uint8"), _NNS_UINT8);
  EXPECT_EQ (gst_tensor_get_type ("UINT8"), _NNS_UINT8);
  EXPECT_EQ (gst_tensor_get_type ("uiNt8"), _NNS_UINT8);
  EXPECT_EQ (gst_tensor_get_type ("UInT8"), _NNS_UINT8);
}

/**
 * @brief Test for uint8 type string.
 */
TEST (commonGetTensorType, uint8_n)
{
  EXPECT_EQ (gst_tensor_get_type ("UInT82"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("uint3"), _NNS_END);
}

/**
 * @brief Test for float32 type string.
 */
TEST (commonGetTensorType, float32)
{
  EXPECT_EQ (gst_tensor_get_type ("float32"), _NNS_FLOAT32);
  EXPECT_EQ (gst_tensor_get_type ("FLOAT32"), _NNS_FLOAT32);
  EXPECT_EQ (gst_tensor_get_type ("float32"), _NNS_FLOAT32);
  EXPECT_EQ (gst_tensor_get_type ("FloaT32"), _NNS_FLOAT32);
}

/**
 * @brief Test for float32 type string.
 */
TEST (commonGetTensorType, float32_n)
{
  EXPECT_EQ (gst_tensor_get_type ("FloaT322"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("float3"), _NNS_END);
}

/**
 * @brief Test for float64 type string.
 */
TEST (commonGetTensorType, float64)
{
  EXPECT_EQ (gst_tensor_get_type ("float64"), _NNS_FLOAT64);
  EXPECT_EQ (gst_tensor_get_type ("FLOAT64"), _NNS_FLOAT64);
  EXPECT_EQ (gst_tensor_get_type ("float64"), _NNS_FLOAT64);
  EXPECT_EQ (gst_tensor_get_type ("FloaT64"), _NNS_FLOAT64);
}

/**
 * @brief Test for float64 type string.
 */
TEST (commonGetTensorType, float64_n)
{
  EXPECT_EQ (gst_tensor_get_type ("FloaT642"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("float6"), _NNS_END);
}

/**
 * @brief Test for int64 type string.
 */
TEST (commonGetTensorType, int64)
{
  EXPECT_EQ (gst_tensor_get_type ("int64"), _NNS_INT64);
  EXPECT_EQ (gst_tensor_get_type ("INT64"), _NNS_INT64);
  EXPECT_EQ (gst_tensor_get_type ("iNt64"), _NNS_INT64);
  EXPECT_EQ (gst_tensor_get_type ("InT64"), _NNS_INT64);
}

/**
 * @brief Test for int64 type string.
 */
TEST (commonGetTensorType, int64_n)
{
  EXPECT_EQ (gst_tensor_get_type ("InT642"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("int6"), _NNS_END);
}

/**
 * @brief Test for uint64 type string.
 */
TEST (commonGetTensorType, uint64)
{
  EXPECT_EQ (gst_tensor_get_type ("uint64"), _NNS_UINT64);
  EXPECT_EQ (gst_tensor_get_type ("UINT64"), _NNS_UINT64);
  EXPECT_EQ (gst_tensor_get_type ("uiNt64"), _NNS_UINT64);
  EXPECT_EQ (gst_tensor_get_type ("UInT64"), _NNS_UINT64);
}

/**
 * @brief Test for uint64 type string.
 */
TEST (commonGetTensorType, uint64_n)
{
  EXPECT_EQ (gst_tensor_get_type ("UInT642"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("uint6"), _NNS_END);
}

/**
 * @brief Test for tensor format (invalid param).
 */
TEST (commonGetTensorFormat, invalidParam01_n)
{
  EXPECT_EQ (gst_tensor_get_format (""), _NNS_TENSOR_FORMAT_END);
  EXPECT_EQ (gst_tensor_get_format (NULL), _NNS_TENSOR_FORMAT_END);
  EXPECT_EQ (gst_tensor_get_format ("invalid-fmt"), _NNS_TENSOR_FORMAT_END);
}

/**
 * @brief Test for tensor format string (invalid param).
 */
TEST (commonGetTensorFormat, invalidParam02_n)
{
  EXPECT_STREQ (gst_tensor_get_format_string (_NNS_TENSOR_FORMAT_END), NULL);
  EXPECT_STREQ (gst_tensor_get_format_string ((tensor_format) -1), NULL);
}

/**
 * @brief Test to find index of the key.
 */
TEST (commonFindKeyStrv, keyIndex)
{
  const gchar *teststrv[] = { "abcde", "ABCDEF", "1234", "abcabc", "tester", NULL };

  EXPECT_EQ (find_key_strv (teststrv, "abcde"), 0);
  EXPECT_EQ (find_key_strv (teststrv, "ABCDE"), 0);
  EXPECT_EQ (find_key_strv (teststrv, "ABCDEF"), 1);
  EXPECT_EQ (find_key_strv (teststrv, "1234"), 2);
  EXPECT_EQ (find_key_strv (teststrv, "tester"), 4);
  EXPECT_EQ (find_key_strv (teststrv, "abcabcd"), -1);
}

/**
 * @brief Test for tensor dimension.
 */
TEST (commonGetTensorDimension, case1)
{
  tensor_dim dim;
  gchar *dim_str;
  guint rank;

  rank = gst_tensor_parse_dimension ("345:123:433:177", dim);
  EXPECT_EQ (rank, 4U);
  EXPECT_EQ (dim[0], 345U);
  EXPECT_EQ (dim[1], 123U);
  EXPECT_EQ (dim[2], 433U);
  EXPECT_EQ (dim[3], 177U);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal (dim_str, "345:123:433:177"));
  g_free (dim_str);
}

/**
 * @brief Test for tensor dimension.
 */
TEST (commonGetTensorDimension, case2)
{
  tensor_dim dim;
  gchar *dim_str;
  guint rank;

  rank = gst_tensor_parse_dimension ("345:123:433", dim);
  EXPECT_EQ (rank, 3U);
  EXPECT_EQ (dim[0], 345U);
  EXPECT_EQ (dim[1], 123U);
  EXPECT_EQ (dim[2], 433U);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal (dim_str, "345:123:433:1"));
  g_free (dim_str);
}

/**
 * @brief Test for tensor dimension.
 */
TEST (commonGetTensorDimension, case3)
{
  tensor_dim dim;
  gchar *dim_str;
  guint rank;

  rank = gst_tensor_parse_dimension ("345:123", dim);
  EXPECT_EQ (rank, 2U);
  EXPECT_EQ (dim[0], 345U);
  EXPECT_EQ (dim[1], 123U);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal (dim_str, "345:123:1:1"));
  g_free (dim_str);
}

/**
 * @brief Test for tensor dimension.
 */
TEST (commonGetTensorDimension, case4)
{
  tensor_dim dim;
  gchar *dim_str;
  guint rank;

  rank = gst_tensor_parse_dimension ("345", dim);
  EXPECT_EQ (rank, 1U);
  EXPECT_EQ (dim[0], 345U);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal (dim_str, "345:1:1:1"));
  g_free (dim_str);
}

/**
 * @brief Test for tensor dimension.
 */
TEST (commonGetTensorDimension, case5)
{
  tensor_dim dim;
  gchar *dim_str;
  guint rank;

  rank = gst_tensor_parse_dimension ("345:123:433:177:851", dim);
  EXPECT_EQ (rank, 5U);
  EXPECT_EQ (dim[0], 345U);
  EXPECT_EQ (dim[1], 123U);
  EXPECT_EQ (dim[2], 433U);
  EXPECT_EQ (dim[3], 177U);
  EXPECT_EQ (dim[4], 851U);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal (dim_str, "345:123:433:177:851"));
  g_free (dim_str);
}

/**
 * @brief Test for tensor dimension.
 */
TEST (commonGetTensorDimension, case6)
{
  tensor_dim dim;
  gchar *dim_str;
  guint rank;

  rank = gst_tensor_parse_dimension ("345:123:433:177:851:369", dim);
  EXPECT_EQ (rank, 6U);
  EXPECT_EQ (dim[0], 345U);
  EXPECT_EQ (dim[1], 123U);
  EXPECT_EQ (dim[2], 433U);
  EXPECT_EQ (dim[3], 177U);
  EXPECT_EQ (dim[4], 851U);
  EXPECT_EQ (dim[5], 369U);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal (dim_str, "345:123:433:177:851:369"));
  g_free (dim_str);
}

/**
 * @brief Test for tensor dimension.
 */
TEST (commonGetTensorDimension, case7)
{
  tensor_dim dim;
  gchar *dim_str;
  guint rank;

  rank = gst_tensor_parse_dimension ("345:123:433:177:851:369:456", dim);
  EXPECT_EQ (rank, 7U);
  EXPECT_EQ (dim[0], 345U);
  EXPECT_EQ (dim[1], 123U);
  EXPECT_EQ (dim[2], 433U);
  EXPECT_EQ (dim[3], 177U);
  EXPECT_EQ (dim[4], 851U);
  EXPECT_EQ (dim[5], 369U);
  EXPECT_EQ (dim[6], 456U);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal (dim_str, "345:123:433:177:851:369:456"));
  g_free (dim_str);
}

/**
 * @brief Test for tensor dimension.
 */
TEST (commonGetTensorDimension, case8)
{
  tensor_dim dim;
  gchar *dim_str;
  guint rank;

  rank = gst_tensor_parse_dimension ("345:123:433:177:851:369:456:91", dim);
  EXPECT_EQ (rank, 8U);
  EXPECT_EQ (dim[0], 345U);
  EXPECT_EQ (dim[1], 123U);
  EXPECT_EQ (dim[2], 433U);
  EXPECT_EQ (dim[3], 177U);
  EXPECT_EQ (dim[4], 851U);
  EXPECT_EQ (dim[5], 369U);
  EXPECT_EQ (dim[6], 456U);
  EXPECT_EQ (dim[7], 91U);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal (dim_str, "345:123:433:177:851:369:456:91"));
  g_free (dim_str);
}

/**
 * @brief Test to copy tensor info.
 */
TEST (commonTensorInfo, copyTensor)
{
  GstTensorInfo src, dest;
  gchar *test_name = g_strdup ("test-tensor");

  gst_tensor_info_init (&src);
  gst_tensor_info_init (&dest);

  src = { test_name, _NNS_FLOAT32, { 1, 2, 3, 4 } };
  gst_tensor_info_copy (&dest, &src);

  EXPECT_TRUE (dest.name != src.name);
  EXPECT_STREQ (dest.name, test_name);
  EXPECT_EQ (dest.type, src.type);
  EXPECT_EQ (dest.dimension[0], src.dimension[0]);
  EXPECT_EQ (dest.dimension[1], src.dimension[1]);
  EXPECT_EQ (dest.dimension[2], src.dimension[2]);
  EXPECT_EQ (dest.dimension[3], src.dimension[3]);
  gst_tensor_info_free (&dest);

  src = { NULL, _NNS_INT32, { 5, 6, 7, 8 } };
  gst_tensor_info_copy (&dest, &src);

  EXPECT_TRUE (dest.name == NULL);
  EXPECT_EQ (dest.type, src.type);
  EXPECT_EQ (dest.dimension[0], src.dimension[0]);
  EXPECT_EQ (dest.dimension[1], src.dimension[1]);
  EXPECT_EQ (dest.dimension[2], src.dimension[2]);
  EXPECT_EQ (dest.dimension[3], src.dimension[3]);

  gst_tensor_info_free (&dest);
  g_free (test_name);
}

/**
 * @brief Test to copy tensor info.
 */
TEST (commonTensorInfo, copyTensors)
{
  GstTensorsInfo src, dest;
  const gchar test_name[] = "test-tensors";
  guint i;

  gst_tensors_info_init (&src);
  gst_tensors_info_init (&dest);

  src.num_tensors = 2;
  src.info[0] = { g_strdup (test_name), _NNS_INT32, { 1, 2, 3, 4 } };
  src.info[1] = { g_strdup (test_name), _NNS_FLOAT32, { 5, 6, 7, 8 } };
  gst_tensors_info_copy (&dest, &src);

  EXPECT_EQ (dest.num_tensors, src.num_tensors);

  for (i = 0; i < src.num_tensors; i++) {
    EXPECT_TRUE (dest.info[i].name != src.info[i].name);
    EXPECT_STREQ (dest.info[i].name, test_name);
    EXPECT_EQ (dest.info[i].type, src.info[i].type);
    EXPECT_EQ (dest.info[i].dimension[0], src.info[i].dimension[0]);
    EXPECT_EQ (dest.info[i].dimension[1], src.info[i].dimension[1]);
    EXPECT_EQ (dest.info[i].dimension[2], src.info[i].dimension[2]);
    EXPECT_EQ (dest.info[i].dimension[3], src.info[i].dimension[3]);
  }

  gst_tensors_info_free (&src);
  gst_tensors_info_free (&dest);
}

/**
 * @brief Test for data size.
 */
TEST (commonTensorInfo, size01_p)
{
  GstTensorsInfo info1, info2;
  GstTensorInfo *_info;
  gsize size1, size2;
  guint i;

  fill_tensors_info_for_test (&info1, &info2);

  _info = gst_tensors_info_get_nth_info (&info1, 0);
  size1 = gst_tensor_info_get_size (_info);
  size2 = gst_tensors_info_get_size (&info1, 0);

  EXPECT_TRUE (size1 == size2);

  size1 = 0;
  for (i = 0; i < info2.num_tensors; i++) {
    _info = gst_tensors_info_get_nth_info (&info2, i);
    size1 += gst_tensor_info_get_size (_info);
  }

  size2 = gst_tensors_info_get_size (&info2, -1);

  EXPECT_TRUE (size1 == size2);

  gst_tensors_info_free (&info1);
  gst_tensors_info_free (&info2);
}

/**
 * @brief Test for data size.
 */
TEST (commonTensorInfo, size02_n)
{
  GstTensorsInfo info1, info2;
  gsize size1;
  gint index;

  fill_tensors_info_for_test (&info1, &info2);

  /* get size with null param */
  index = (gint) info1.num_tensors - 1;
  size1 = gst_tensors_info_get_size (NULL, index);

  EXPECT_TRUE (size1 == 0);

  gst_tensors_info_free (&info1);
  gst_tensors_info_free (&info2);
}

/**
 * @brief Test for data size.
 */
TEST (commonTensorInfo, size03_n)
{
  GstTensorsInfo info1, info2;
  gsize size1;
  gint index;

  fill_tensors_info_for_test (&info1, &info2);

  /* get size with invalid index */
  index = (gint) info1.num_tensors;
  size1 = gst_tensors_info_get_size (&info1, index);

  EXPECT_TRUE (size1 == 0);

  gst_tensors_info_free (&info1);
  gst_tensors_info_free (&info2);
}

/**
 * @brief Test for same tensors info.
 */
TEST (commonTensorInfo, equal01_p)
{
  GstTensorsInfo info1, info2;

  fill_tensors_info_for_test (&info1, &info2);

  EXPECT_TRUE (gst_tensors_info_is_equal (&info1, &info2));

  gst_tensors_info_free (&info1);
  gst_tensors_info_free (&info2);
}

/**
 * @brief Test for same tensors info.
 */
TEST (commonTensorInfo, equal02_n)
{
  GstTensorsInfo info1, info2;

  gst_tensors_info_init (&info1);
  gst_tensors_info_init (&info2);

  /* test with invalid info */
  EXPECT_FALSE (gst_tensors_info_is_equal (&info1, &info2));
}

/**
 * @brief Test for same tensors info.
 */
TEST (commonTensorInfo, equal03_n)
{
  GstTensorsInfo info1, info2;

  fill_tensors_info_for_test (&info1, &info2);

  /* change info, this should not be compatible */
  info1.num_tensors = 1;

  EXPECT_FALSE (gst_tensors_info_is_equal (&info1, &info2));

  gst_tensors_info_free (&info1);
  gst_tensors_info_free (&info2);
}

/**
 * @brief Test for same tensors info.
 */
TEST (commonTensorInfo, equal04_n)
{
  GstTensorsInfo info1, info2;

  fill_tensors_info_for_test (&info1, &info2);

  /* change info, this should not be compatible */
  info1.info[0].type = _NNS_UINT64;

  EXPECT_FALSE (gst_tensors_info_is_equal (&info1, &info2));

  gst_tensors_info_free (&info1);
  gst_tensors_info_free (&info2);
}

/**
 * @brief Test for same tensors info.
 */
TEST (commonTensorInfo, equal05_n)
{
  GstTensorsInfo info1, info2;

  fill_tensors_info_for_test (&info1, &info2);

  /* change info, this should not be compatible */
  info2.info[1].dimension[0] = 10;

  EXPECT_FALSE (gst_tensors_info_is_equal (&info1, &info2));

  gst_tensors_info_free (&info1);
  gst_tensors_info_free (&info2);
}

/**
 * @brief Test for getting size of the tensor info with invalid param.
 */
TEST (commonTensorInfo, sizeInvalidParam_n)
{
  EXPECT_EQ (0U, gst_tensor_info_get_size (NULL));
}

/**
 * @brief Test for validating of the tensor info with invalid param.
 */
TEST (commonTensorInfo, validateInvalidParam_n)
{
  EXPECT_FALSE (gst_tensor_info_validate (NULL));
}

/**
 * @brief Test for validating of the tensors info with invalid param.
 */
TEST (commonTensorsInfo, validateInvalidParam_n)
{
  EXPECT_FALSE (gst_tensors_info_validate (NULL));
}

/**
 * @brief Test for validating of the tensors info with invalid format.
 */
TEST (commonTensorsInfo, validateInvalidFormat_n)
{
  GstTensorsInfo info;

  gst_tensors_info_init (&info);

  info.format = _NNS_TENSOR_FORMAT_END;
  EXPECT_FALSE (gst_tensors_info_validate (&info));
}

/**
 * @brief Test for getting nth info with invalid index.
 */
TEST (commonTensorsInfo, getNthInfoInvalidIndex_n)
{
  GstTensorsInfo info;

  gst_tensors_info_init (&info);

  EXPECT_EQ (NULL, gst_tensors_info_get_nth_info (&info, NNS_TENSOR_SIZE_LIMIT));

  gst_tensors_info_free (&info);
}

/**
 * @brief Test for comparing two tensor info with invalid param.
 */
TEST (commonTensorInfo, equalInvalidParam0_n)
{
  GstTensorInfo info;
  gst_tensor_info_init (&info);
  EXPECT_FALSE (gst_tensor_info_is_equal (NULL, &info));
}

/**
 * @brief Test for comparing two tensor info with invalid param.
 */
TEST (commonTensorInfo, equalInvalidParam1_n)
{
  GstTensorInfo info;
  gst_tensor_info_init (&info);
  EXPECT_FALSE (gst_tensor_info_is_equal (&info, NULL));
}

/**
 * @brief Test for comparing two tensors info with invalid param.
 */
TEST (commonTensorsInfo, equalInvalidParam0_n)
{
  GstTensorsInfo info;
  gst_tensors_info_init (&info);
  EXPECT_FALSE (gst_tensors_info_is_equal (NULL, &info));
}

/**
 * @brief Test for comparing two tensors info with invalid param.
 */
TEST (commonTensorsInfo, equalInvalidParam1_n)
{
  GstTensorsInfo info;
  gst_tensors_info_init (&info);
  EXPECT_FALSE (gst_tensors_info_is_equal (&info, NULL));
}

/**
 * @brief Test for getting rank with invalid param.
 */
TEST (commonTensorInfo, getrankInvalidParam0_n)
{
  EXPECT_EQ (0U, gst_tensor_info_get_rank (NULL));
}

/**
 * @brief Test for printing tensor info.
 */
TEST (commonTensorInfo, printInvalidParam_n)
{
  EXPECT_EQ (NULL, gst_tensor_info_to_string (NULL));
}

/**
 * @brief Test for parsing dimension with invalid param.
 */
TEST (commonTensorsInfo, parsingDimInvalidParam0_n)
{
  const gchar *dim_str = "1:2:3:4";
  EXPECT_EQ (0U, gst_tensors_info_parse_dimensions_string (NULL, dim_str));
}

/**
 * @brief Test for parsing dimension with invalid param.
 */
TEST (commonTensorsInfo, parsingDimInvalidParam1_n)
{
  GstTensorsInfo info;
  gst_tensors_info_init (&info);
  EXPECT_EQ (0U, gst_tensors_info_parse_dimensions_string (&info, NULL));
}

/**
 * @brief Test for parsing type with invalid param.
 */
TEST (commonTensorsInfo, parsingTypeInvalidParam0_n)
{
  const gchar *dim_str = "uint8";
  EXPECT_EQ (0U, gst_tensors_info_parse_types_string (NULL, dim_str));
}

/**
 * @brief Test for parsing type with invalid param.
 */
TEST (commonTensorsInfo, parsingTypeInvalidParam1_n)
{
  GstTensorsInfo info;
  gst_tensors_info_init (&info);
  EXPECT_EQ (0U, gst_tensors_info_parse_types_string (&info, NULL));
}

/**
 * @brief Test for parsing name with invalid param.
 */
TEST (commonTensorsInfo, parsingNameInvalidParam0_n)
{
  const gchar *dim_str = "tname";
  EXPECT_EQ (0U, gst_tensors_info_parse_names_string (NULL, dim_str));
}

/**
 * @brief Test for parsing name with invalid param.
 */
TEST (commonTensorsInfo, parsingNameInvalidParam1_n)
{
  GstTensorsInfo info;
  gst_tensors_info_init (&info);
  EXPECT_EQ (0U, gst_tensors_info_parse_names_string (&info, NULL));
}

/**
 * @brief Test for getting dimension with invalid param.
 */
TEST (commonTensorsInfo, getDimInvalidParam0_n)
{
  EXPECT_EQ (NULL, gst_tensors_info_get_dimensions_string (NULL));
}

/**
 * @brief Test for getting dimension with invalid param.
 */
TEST (commonTensorsInfo, getDimInvalidParam1_n)
{
  GstTensorsInfo info;
  gst_tensors_info_init (&info);
  info.num_tensors = 0;
  EXPECT_EQ (0U, gst_tensors_info_get_dimensions_string (&info));
}

/**
 * @brief Test for getting type with invalid param.
 */
TEST (commonTensorsInfo, getTypeInvalidParam0_n)
{
  EXPECT_EQ (NULL, gst_tensors_info_get_types_string (NULL));
}

/**
 * @brief Test for getting type with invalid param.
 */
TEST (commonTensorsInfo, getTypeInvalidParam1_n)
{
  GstTensorsInfo info;
  gst_tensors_info_init (&info);
  info.num_tensors = 0;
  EXPECT_EQ (0U, gst_tensors_info_get_types_string (&info));
}

/**
 * @brief Test for getting name with invalid param.
 */
TEST (commonTensorsInfo, getNameInvalidParam0_n)
{
  EXPECT_EQ (NULL, gst_tensors_info_get_names_string (NULL));
}

/**
 * @brief Test for getting name with invalid param.
 */
TEST (commonTensorsInfo, getNameInvalidParam1_n)
{
  GstTensorsInfo info;
  gst_tensors_info_init (&info);
  info.num_tensors = 0;
  EXPECT_EQ (0U, gst_tensors_info_get_names_string (&info));
}

/**
 * @brief Test for printing tensors info with invalid param.
 */
TEST (commonTensorsInfo, printInvalidParam_n)
{
  EXPECT_EQ (NULL, gst_tensors_info_to_string (NULL));
}

/**
 * @brief Test for printing tensors info with invalid index.
 */
TEST (commonTensorsInfo, printInvalidIndex_n)
{
  GstTensorsInfo info;
  gchar *str;

  gst_tensors_info_init (&info);
  info.num_tensors = NNS_TENSOR_SIZE_LIMIT + 1;

  str = gst_tensors_info_to_string (&info);

  EXPECT_EQ (1U, count_key_string (str, "out of bound"));

  g_free (str);
}

/**
 * @brief Test for printing tensors info comparison.
 */
TEST (commonTensorsInfo, compareString)
{
  GstTensorsInfo info1, info2;
  GstTensorInfo *_info1, *_info2;
  gchar *str;

  fill_tensors_info_for_test (&info1, &info2);

  /* same info */
  str = gst_tensors_info_compare_to_string (&info1, &info2);

  EXPECT_EQ (0U, count_key_string (str, "Not equal"));

  g_free (str);

  /* change number of tensors */
  _info1 = gst_tensors_info_get_nth_info (&info1, info1.num_tensors);
  _info1->type = _NNS_INT32;
  _info1->dimension[0] = 1;
  info1.num_tensors++;

  str = gst_tensors_info_compare_to_string (&info1, &info2);

  EXPECT_EQ (1U, count_key_string (str, "Not equal"));

  info1.num_tensors--;
  g_free (str);

  _info1 = gst_tensors_info_get_nth_info (&info2, 0);
  _info2 = gst_tensors_info_get_nth_info (&info2, 1);

  /* change tensor name */
  _info1->name = g_strdup ("test-tensor1");
  _info2->name = g_strdup ("test-tensor2");

  str = gst_tensors_info_compare_to_string (&info1, &info2);

  EXPECT_EQ (2U, count_key_string (str, "Not equal"));
  EXPECT_EQ (1U, count_key_string (str, "test-tensor1"));
  EXPECT_EQ (1U, count_key_string (str, "test-tensor2"));

  g_clear_pointer (&_info1->name, g_free);
  g_clear_pointer (&_info2->name, g_free);
  g_free (str);

  /* change tensor type */
  _info1->type = _NNS_UINT8;
  _info2->type = _NNS_INT16;

  str = gst_tensors_info_compare_to_string (&info1, &info2);

  EXPECT_EQ (2U, count_key_string (str, "Not equal"));
  EXPECT_EQ (1U, count_key_string (str, "uint8"));
  EXPECT_EQ (1U, count_key_string (str, "int16"));

  g_free (str);

  gst_tensors_info_free (&info1);
  gst_tensors_info_free (&info2);
}

/**
 * @brief Test for printing tensors info comparison.
 */
TEST (commonTensorsInfo, compareInvalidParam_n)
{
  EXPECT_EQ (NULL, gst_tensors_info_compare_to_string (NULL, NULL));
}

/**
 * @brief Test for same tensors config.
 */
TEST (commonTensorsConfig, equal01_p)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);

  EXPECT_TRUE (gst_tensors_config_is_equal (&conf1, &conf2));

  gst_tensors_config_free (&conf1);
  gst_tensors_config_free (&conf2);
}

/**
 * @brief Test for same tensors config.
 */
TEST (commonTensorsConfig, equal02_p)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);
  conf1.rate_n *= 2;
  conf1.rate_d *= 2;

  EXPECT_TRUE (gst_tensors_config_is_equal (&conf1, &conf2));

  gst_tensors_config_free (&conf1);
  gst_tensors_config_free (&conf2);
}

/**
 * @brief Test for same tensors config.
 */
TEST (commonTensorsConfig, equal03_p)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);
  conf1.rate_n *= 0;
  conf2.rate_n *= 0;

  EXPECT_TRUE (gst_tensors_config_is_equal (&conf1, &conf2));

  gst_tensors_config_free (&conf1);
  gst_tensors_config_free (&conf2);
}

/**
 * @brief Test for same tensors config.
 */
TEST (commonTensorsConfig, equal04_n)
{
  GstTensorsConfig conf1, conf2;

  gst_tensors_config_init (&conf1);
  gst_tensors_config_init (&conf2);

  EXPECT_FALSE (gst_tensors_config_is_equal (&conf1, &conf2));

  gst_tensors_config_free (&conf1);
  gst_tensors_config_free (&conf2);
}

/**
 * @brief Test for same tensors config.
 */
TEST (commonTensorsConfig, equal05_n)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);
  conf1.rate_n *= 2;
  conf1.rate_d *= 4;

  EXPECT_FALSE (gst_tensors_config_is_equal (&conf1, &conf2));

  gst_tensors_config_free (&conf1);
  gst_tensors_config_free (&conf2);
}

/**
 * @brief Test for same tensors config.
 */
TEST (commonTensorsConfig, equal06_n)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);
  conf1.rate_d *= 0;

  EXPECT_FALSE (gst_tensors_config_is_equal (&conf1, &conf2));

  gst_tensors_config_free (&conf1);
  gst_tensors_config_free (&conf2);
}

/**
 * @brief Test for same tensors config.
 */
TEST (commonTensorsConfig, equal07_n)
{
  GstTensorsConfig conf;
  gst_tensors_config_init (&conf);
  EXPECT_FALSE (gst_tensors_config_is_equal (NULL, &conf));
}

/**
 * @brief Test for same tensors config.
 */
TEST (commonTensorsConfig, equal08_n)
{
  GstTensorsConfig conf;
  gst_tensors_config_init (&conf);
  EXPECT_FALSE (gst_tensors_config_is_equal (&conf, NULL));
}

/**
 * @brief Test for same tensors config.
 */
TEST (commonTensorsConfig, equal09_p)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);

  /* compare flexible tensor */
  conf1.info.format = conf2.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  EXPECT_TRUE (gst_tensors_config_is_equal (&conf1, &conf2));

  gst_tensors_config_free (&conf1);
  gst_tensors_config_free (&conf2);
}

/**
 * @brief Test for same tensors config.
 */
TEST (commonTensorsConfig, equal10_n)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);

  /* change format, this should not be compatible */
  conf2.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  EXPECT_FALSE (gst_tensors_config_is_equal (&conf1, &conf2));

  gst_tensors_config_free (&conf1);
  gst_tensors_config_free (&conf2);
}

/**
 * @brief Test for same tensors config.
 */
TEST (commonTensorsConfig, equal11_n)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);

  /* change framerate, this should not be compatible */
  conf1.rate_n = 20;
  conf2.rate_n = 10;
  conf1.rate_d = conf2.rate_d = 1;

  EXPECT_FALSE (gst_tensors_config_is_equal (&conf1, &conf2));

  gst_tensors_config_free (&conf1);
  gst_tensors_config_free (&conf2);
}

/**
 * @brief Test for validating of the tensors config with invalid param.
 */
TEST (commonTensorsConfig, validateInvalidParam0_n)
{
  EXPECT_FALSE (gst_tensors_config_validate (NULL));
}

/**
 * @brief Test for validating of the tensors config with invalid param.
 */
TEST (commonTensorsConfig, validateInvalidParam1_n)
{
  GstTensorsConfig conf;

  gst_tensors_config_init (&conf);
  conf.rate_n = 1;

  EXPECT_FALSE (gst_tensors_config_validate (&conf));
}

/**
 * @brief Test for validating of the tensors config with invalid param.
 */
TEST (commonTensorsConfig, validateInvalidParam2_n)
{
  GstTensorsConfig conf;

  gst_tensors_config_init (&conf);
  conf.rate_d = 1;

  EXPECT_FALSE (gst_tensors_config_validate (&conf));
}

/**
 * @brief Test for getting config from structure with invalid param.
 */
TEST (commonTensorsConfig, fromStructureInvalidParam0_n)
{
  GstStructure structure = { 0 };

  EXPECT_FALSE (gst_tensors_config_from_structure (NULL, &structure));
}

/**
 * @brief Test for getting config from structure with invalid param.
 */
TEST (commonTensorsConfig, fromStructureInvalidParam1_n)
{
  GstTensorsConfig conf;
  gst_tensors_config_init (&conf);
  EXPECT_FALSE (gst_tensors_config_from_structure (&conf, NULL));
}

/**
 * @brief Test for getting tensor cap with invalid param.
 */
TEST (commonTensorsConfig, capsInvalidParam0_n)
{
  EXPECT_FALSE (gst_tensor_caps_from_config (NULL));
}

/**
 * @brief Test for getting tensor cap with invalid param.
 */
TEST (commonTensorsConfig, capsInvalidParam1_n)
{
  EXPECT_FALSE (gst_tensors_caps_from_config (NULL));
}

/**
 * @brief Test for parsing tensor cap.
 */
TEST (commonTensorsConfig, parseCapsFlexible_p)
{
  GstTensorsConfig config;
  GstCaps *caps = gst_caps_new_simple ("other/tensors", "format", G_TYPE_STRING,
      "flexible", "framerate", GST_TYPE_FRACTION, 30, 1, NULL);

  EXPECT_TRUE (gst_tensors_config_from_caps (&config, caps, TRUE));

  gst_tensors_config_free (&config);
  gst_caps_unref (caps);
}

/**
 * @brief Test for parsing tensor cap with invalid param.
 */
TEST (commonTensorsConfig, parseCapsInvalidParam0_n)
{
  GstCaps *caps = gst_caps_new_simple ("other/tensor", "format", G_TYPE_STRING,
      "flexible", "framerate", GST_TYPE_FRACTION, 30, 1, NULL);

  EXPECT_FALSE (gst_tensors_config_from_caps (NULL, caps, TRUE));

  gst_caps_unref (caps);
}

/**
 * @brief Test for parsing tensor cap with invalid param.
 */
TEST (commonTensorsConfig, parseCapsInvalidParam1_n)
{
  GstTensorsConfig config;

  EXPECT_FALSE (gst_tensors_config_from_caps (&config, NULL, TRUE));

  gst_tensors_config_free (&config);
}

/**
 * @brief Test for parsing tensor cap with unfixed caps.
 */
TEST (commonTensorsConfig, parseUnfixedCaps_n)
{
  GstTensorsConfig config;
  GstCaps *caps = gst_caps_new_simple ("other/tensor", "format", G_TYPE_STRING,
      "static", "framerate", GST_TYPE_FRACTION_RANGE, 0, 1, G_MAXINT, 1, NULL);

  EXPECT_TRUE (gst_tensors_config_from_caps (&config, caps, FALSE));
  gst_tensors_config_free (&config);

  EXPECT_FALSE (gst_tensors_config_from_caps (&config, caps, TRUE));
  gst_tensors_config_free (&config);

  gst_caps_unref (caps);
}

/**
 * @brief Test for printing tensors config.
 */
TEST (commonTensorsConfig, printConfig)
{
  GstTensorsConfig config;
  gchar *str;

  gst_tensors_config_init (&config);

  config.info.num_tensors = 2;
  config.info.info[0] = { g_strdup ("test-tensor1"), _NNS_INT32, { 1, 2, 3, 4 } };
  config.info.info[1] = { g_strdup ("test-tensor2"), _NNS_FLOAT32, { 5, 6, 7, 8 } };
  config.rate_n = 30;
  config.rate_d = 1;

  str = gst_tensors_config_to_string (&config);

  /* tensor info */
  EXPECT_EQ (1U, count_key_string (str, "test-tensor1"));
  EXPECT_EQ (1U, count_key_string (str, "test-tensor2"));
  EXPECT_EQ (1U, count_key_string (str, "int32"));
  EXPECT_EQ (1U, count_key_string (str, "float32"));
  EXPECT_EQ (1U, count_key_string (str, "1:2:3:4"));
  EXPECT_EQ (1U, count_key_string (str, "5:6:7:8"));

  /* framerate */
  EXPECT_EQ (1U, count_key_string (str, "30/1"));

  g_free (str);
  gst_tensors_config_free (&config);
}

/**
 * @brief Test for printing tensors config.
 */
TEST (commonTensorsConfig, printInvalidParam_n)
{
  EXPECT_EQ (NULL, gst_tensors_config_to_string (NULL));
}

/**
 * @brief Test for dimensions string in tensors info.
 */
TEST (commonTensorsInfoString, dimensions)
{
  GstTensorsInfo info;
  guint num_dims;
  gchar *str_dims;

  gst_tensors_info_init (&info);

  /* 1 tensor info */
  num_dims = gst_tensors_info_parse_dimensions_string (&info, "1:2:3:4");
  EXPECT_EQ (num_dims, 1U);

  info.num_tensors = num_dims;

  str_dims = gst_tensors_info_get_dimensions_string (&info);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal (str_dims, "1:2:3:4"));
  g_free (str_dims);

  /* 4 tensors info */
  num_dims = gst_tensors_info_parse_dimensions_string (&info, "1, 2:2, 3:3:3, 4:4:4:4");
  EXPECT_EQ (num_dims, 4U);

  info.num_tensors = num_dims;

  str_dims = gst_tensors_info_get_dimensions_string (&info);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal (str_dims, "1:1:1:1,2:2:1:1,3:3:3:1,4:4:4:4"));
  g_free (str_dims);

  /* extra */
  num_dims = gst_tensors_info_parse_dimensions_string (&info,
      "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20");
  EXPECT_EQ (num_dims, 20U);
  info.num_tensors = num_dims;

  str_dims = gst_tensors_info_get_dimensions_string (&info);
  EXPECT_TRUE (gst_tensor_dimension_string_is_equal (str_dims,
      "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20"));

  g_free (str_dims);

  /* max (NNS_TENSOR_SIZE_LIMIT) */
  GString *max_dims = g_string_new (NULL);
  guint exceed_lim = NNS_TENSOR_SIZE_LIMIT + 11;
  for (guint i = 0; i < exceed_lim; i++) {
    g_string_append_printf (max_dims, "%d", i);
    if (i < exceed_lim - 1)
      g_string_append (max_dims, ",");
  }

  str_dims = g_string_free (max_dims, FALSE);

  num_dims = gst_tensors_info_parse_dimensions_string (&info, str_dims);
  EXPECT_NE (num_dims, exceed_lim);
  EXPECT_EQ (num_dims, (guint) (NNS_TENSOR_SIZE_LIMIT));

  g_free (str_dims);
  gst_tensors_info_free (&info);
}

/**
 * @brief Test for types string in tensors info.
 */
TEST (commonTensorsInfoString, types)
{
  GstTensorsInfo info;
  guint num_types;
  gchar *str_types;

  gst_tensors_info_init (&info);

  /* 1 tensor info */
  num_types = gst_tensors_info_parse_types_string (&info, "uint16");
  EXPECT_EQ (num_types, 1U);

  info.num_tensors = num_types;

  str_types = gst_tensors_info_get_types_string (&info);
  EXPECT_STREQ (str_types, "uint16");
  g_free (str_types);

  /* 4 tensors info */
  num_types = gst_tensors_info_parse_types_string (&info, "int8, int16, int32, int64");
  EXPECT_EQ (num_types, 4U);

  info.num_tensors = num_types;

  str_types = gst_tensors_info_get_types_string (&info);
  EXPECT_STREQ (str_types, "int8,int16,int32,int64");
  g_free (str_types);

  /* extra */
  num_types = gst_tensors_info_parse_types_string (&info,
      "int8, int8, int8, int8, int8, int8, int8, int8, int8, int8, int8, "
      "int8, int8, int8, int8, int8, int8, int8, int8, int8, int8, int8");
  EXPECT_EQ (num_types, 22U);

  info.num_tensors = num_types;
  str_types = gst_tensors_info_get_types_string (&info);
  EXPECT_STREQ (str_types, "int8,int8,int8,int8,int8,int8,int8,int8,int8,int8,int8,"
                           "int8,int8,int8,int8,int8,int8,int8,int8,int8,int8,int8");
  g_free (str_types);

  /* max (NNS_TENSOR_SIZE_LIMIT) */
  GString *max_types = g_string_new (NULL);
  guint exceed_lim = NNS_TENSOR_SIZE_LIMIT + 13;
  for (guint i = 0; i < exceed_lim; i++) {
    g_string_append_printf (max_types, "%s", "uint8");
    if (i < exceed_lim - 1)
      g_string_append (max_types, ",");
  }

  str_types = g_string_free (max_types, FALSE);

  num_types = gst_tensors_info_parse_types_string (&info, str_types);
  EXPECT_NE (num_types, exceed_lim);
  EXPECT_EQ (num_types, (guint) (NNS_TENSOR_SIZE_LIMIT));

  g_free (str_types);
  gst_tensors_info_free (&info);
}

/**
 * @brief Test for names string in tensors info.
 */
TEST (commonTensorsInfoString, names)
{
  GstTensorsInfo info;
  guint i, num_names;
  gchar *str_names;

  gst_tensors_info_init (&info);

  /* 1 tensor info */
  num_names = gst_tensors_info_parse_names_string (&info, "t1");
  EXPECT_EQ (num_names, 1U);

  info.num_tensors = num_names;

  str_names = gst_tensors_info_get_names_string (&info);
  EXPECT_STREQ (str_names, "t1");
  g_free (str_names);

  /* 4 tensors info */
  num_names = gst_tensors_info_parse_names_string (&info, "tensor1, tensor2, tensor3, tensor4");
  EXPECT_EQ (num_names, 4U);

  info.num_tensors = num_names;

  str_names = gst_tensors_info_get_names_string (&info);
  EXPECT_STREQ (str_names, "tensor1,tensor2,tensor3,tensor4");
  g_free (str_names);
  gst_tensors_info_free (&info);

  /* empty name string */
  num_names = gst_tensors_info_parse_names_string (&info, ",,");
  EXPECT_EQ (num_names, 3U);

  info.num_tensors = num_names;
  for (i = 0; i < num_names; ++i) {
    EXPECT_TRUE (info.info[i].name == NULL);
  }

  str_names = gst_tensors_info_get_names_string (&info);
  EXPECT_STREQ (str_names, ",,");
  g_free (str_names);
  gst_tensors_info_free (&info);

  /* extra */
  num_names = gst_tensors_info_parse_names_string (&info,
      "t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, "
      "t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28");
  EXPECT_EQ (num_names, 28U);
  info.num_tensors = num_names;

  str_names = gst_tensors_info_get_names_string (&info);
  EXPECT_STREQ (str_names, "t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,"
                           "t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28");
  g_free (str_names);

  /* max (NNS_TENSOR_SIZE_LIMIT) */
  GString *max_names = g_string_new (NULL);
  guint exceed_lim = NNS_TENSOR_SIZE_LIMIT + 17;
  for (i = 0; i < exceed_lim; i++) {
    g_string_append_printf (max_names, "t%d", i);
    if (i < exceed_lim - 1)
      g_string_append (max_names, ",");
  }

  str_names = g_string_free (max_names, FALSE);

  num_names = gst_tensors_info_parse_names_string (&info, str_names);
  EXPECT_NE (num_names, exceed_lim);
  EXPECT_EQ (num_names, (guint) (NNS_TENSOR_SIZE_LIMIT));

  g_free (str_names);
  gst_tensors_info_free (&info);
}

/**
 * @brief Test for tensor meta info (default value after init).
 */
TEST (commonMetaInfo, initDefaultValue)
{
  GstTensorMetaInfo meta;
  guint i, major, minor;

  major = minor = 0;
  gst_tensor_meta_info_init (&meta);

  EXPECT_EQ (meta.type, _NNS_END);
  EXPECT_EQ (meta.format, _NNS_TENSOR_FORMAT_STATIC);
  EXPECT_EQ ((media_type) meta.media_type, _NNS_TENSOR);
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    EXPECT_EQ (meta.dimension[i], 0U);

  /* current version after init */
  EXPECT_TRUE (gst_tensor_meta_info_get_version (&meta, &major, &minor));
  EXPECT_TRUE (major > 0 || minor > 0);
}

/**
 * @brief Test for tensor meta info (version with invalid param).
 */
TEST (commonMetaInfo, versionInvalidParam01_n)
{
  guint major, minor;

  major = minor = 0;

  EXPECT_FALSE (gst_tensor_meta_info_get_version (NULL, &major, &minor));
}

/**
 * @brief Test for tensor meta info (version with invalid param).
 */
TEST (commonMetaInfo, versionInvalidParam02_n)
{
  GstTensorMetaInfo meta;
  guint major, minor;

  major = minor = 0;
  gst_tensor_meta_info_init (&meta);

  /* invalid magic */
  meta.magic = 0;
  EXPECT_FALSE (gst_tensor_meta_info_get_version (&meta, &major, &minor));
}

/**
 * @brief Test for tensor meta info (header size with invalid param).
 */
TEST (commonMetaInfo, headerSizeInvalidParam01_n)
{
  gsize hsize;

  hsize = gst_tensor_meta_info_get_header_size (NULL);
  EXPECT_EQ (hsize, 0U);
}

/**
 * @brief Test for tensor meta info (header size with invalid magic).
 */
TEST (commonMetaInfo, headerSizeInvalidParam02_n)
{
  GstTensorMetaInfo meta;
  gsize hsize;

  gst_tensor_meta_info_init (&meta);
  meta.magic = 0U;
  hsize = gst_tensor_meta_info_get_header_size (&meta);
  EXPECT_EQ (hsize, 0U);
}

/**
 * @brief Test for tensor meta info (header size with invalid version).
 */
TEST (commonMetaInfo, headerSizeInvalidParam03_n)
{
  GstTensorMetaInfo meta;
  gsize hsize;

  gst_tensor_meta_info_init (&meta);
  meta.version = 0U;
  hsize = gst_tensor_meta_info_get_header_size (&meta);
  EXPECT_EQ (hsize, 0U);
}

/**
 * @brief Test for tensor meta info (data size with invalid param).
 */
TEST (commonMetaInfo, dataSizeInvalidParam01_n)
{
  gsize dsize;

  dsize = gst_tensor_meta_info_get_data_size (NULL);
  EXPECT_EQ (dsize, 0U);
}

/**
 * @brief Test for tensor meta info (data size with invalid magic).
 */
TEST (commonMetaInfo, dataSizeInvalidParam02_n)
{
  GstTensorMetaInfo meta;
  gsize dsize;

  gst_tensor_meta_info_init (&meta);
  meta.magic = 0U;
  dsize = gst_tensor_meta_info_get_data_size (&meta);
  EXPECT_EQ (dsize, 0U);
}

/**
 * @brief Test for tensor meta info (data size with invalid version).
 */
TEST (commonMetaInfo, dataSizeInvalidParam03_n)
{
  GstTensorMetaInfo meta;
  gsize dsize;

  gst_tensor_meta_info_init (&meta);
  meta.version = 0U;
  dsize = gst_tensor_meta_info_get_data_size (&meta);
  EXPECT_EQ (dsize, 0U);
}

/**
 * @brief Test for tensor meta info (validate meta with invalid param).
 */
TEST (commonMetaInfo, validateInvalidParam01_n)
{
  gboolean valid;

  valid = gst_tensor_meta_info_validate (NULL);
  EXPECT_FALSE (valid);
}

/**
 * @brief Test for tensor meta info (validate meta with invalid magic).
 */
TEST (commonMetaInfo, validateInvalidParam02_n)
{
  GstTensorMetaInfo meta;
  gboolean valid;

  gst_tensor_meta_info_init (&meta);
  meta.magic = 0U;
  valid = gst_tensor_meta_info_validate (&meta);
  EXPECT_FALSE (valid);
}

/**
 * @brief Test for tensor meta info (validate meta with invalid version).
 */
TEST (commonMetaInfo, validateInvalidParam03_n)
{
  GstTensorMetaInfo meta;
  gboolean valid;

  gst_tensor_meta_info_init (&meta);
  meta.version = 0U;
  valid = gst_tensor_meta_info_validate (&meta);
  EXPECT_FALSE (valid);
}

/**
 * @brief Test for tensor meta info (validate meta with invalid meta).
 */
TEST (commonMetaInfo, validateInvalidParam04_n)
{
  GstTensorMetaInfo meta;
  gboolean valid;

  /* set valid meta */
  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_UINT8;
  meta.dimension[0] = 10;
  meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  meta.media_type = _NNS_VIDEO;
  valid = gst_tensor_meta_info_validate (&meta);
  EXPECT_TRUE (valid);

  /* invalid type */
  meta.type = _NNS_END;
  valid = gst_tensor_meta_info_validate (&meta);
  EXPECT_FALSE (valid);
  meta.type = _NNS_UINT8;

  /* invalid dimension */
  meta.dimension[0] = 0;
  valid = gst_tensor_meta_info_validate (&meta);
  EXPECT_FALSE (valid);
  meta.dimension[0] = 10;

  /* invalid format */
  meta.format = 100;
  valid = gst_tensor_meta_info_validate (&meta);
  EXPECT_FALSE (valid);
  meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  /* invalid media type */
  meta.media_type = _NNS_MEDIA_ANY;
  valid = gst_tensor_meta_info_validate (&meta);
  EXPECT_FALSE (valid);
}

/**
 * @brief Test for tensor meta info (update header with invalid param).
 */
TEST (commonMetaInfo, updateHeaderInvalidParam_n)
{
  GstTensorMetaInfo meta;
  gpointer header;
  gsize hsize;
  gboolean ret;

  gst_tensor_meta_info_init (&meta);
  hsize = gst_tensor_meta_info_get_header_size (&meta);
  header = g_malloc0 (hsize);

  ret = gst_tensor_meta_info_update_header (NULL, header);
  EXPECT_FALSE (ret);

  ret = gst_tensor_meta_info_update_header (&meta, NULL);
  EXPECT_FALSE (ret);

  g_free (header);
}

/**
 * @brief Test for tensor meta info (parse header with invalid param).
 */
TEST (commonMetaInfo, parseHeaderInvalidParam_n)
{
  GstTensorMetaInfo meta;
  gpointer header;
  gsize hsize;
  gboolean ret;

  gst_tensor_meta_info_init (&meta);
  hsize = gst_tensor_meta_info_get_header_size (&meta);
  header = g_malloc0 (hsize);

  ret = gst_tensor_meta_info_parse_header (NULL, header);
  EXPECT_FALSE (ret);

  ret = gst_tensor_meta_info_parse_header (&meta, NULL);
  EXPECT_FALSE (ret);

  g_free (header);
}

/**
 * @brief Test for tensor meta info (parse memory with invalid param).
 */
TEST (commonMetaInfo, parseMemInvalidParam_n)
{
  GstTensorMetaInfo meta;
  GstMemory *mem;
  gsize hsize;
  gboolean ret;

  gst_tensor_meta_info_init (&meta);
  hsize = gst_tensor_meta_info_get_header_size (&meta);
  mem = gst_allocator_alloc (NULL, hsize, NULL);

  ret = gst_tensor_meta_info_parse_memory (NULL, mem);
  EXPECT_FALSE (ret);

  ret = gst_tensor_meta_info_parse_memory (&meta, NULL);
  EXPECT_FALSE (ret);

  gst_memory_unref (mem);
}

/**
 * @brief Test for tensor meta info (parse memory with invalid memory size).
 */
TEST (commonMetaInfo, parseMemInvalidSize_n)
{
  GstTensorMetaInfo meta;
  GstMemory *mem;
  gsize hsize;
  gboolean ret;

  gst_tensor_meta_info_init (&meta);
  hsize = gst_tensor_meta_info_get_header_size (&meta) / 2;
  mem = gst_allocator_alloc (NULL, hsize, NULL);

  ret = gst_tensor_meta_info_parse_memory (&meta, mem);
  EXPECT_FALSE (ret);

  gst_memory_unref (mem);
}

/**
 * @brief Test for tensor meta info (append header to memory).
 */
TEST (commonMetaInfo, appendHeader)
{
  GstTensorMetaInfo meta1, meta2;
  GstMemory *result, *data;
  gsize hsize, msize;
  gboolean ret;

  gst_tensor_meta_info_init (&meta1);
  meta1.type = _NNS_INT16;
  meta1.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  meta1.media_type = _NNS_OCTET;
  meta1.dimension[0] = 300U;
  meta1.dimension[1] = 1U;

  hsize = gst_tensor_meta_info_get_header_size (&meta1);
  data = gst_allocator_alloc (NULL, 300, NULL);

  result = gst_tensor_meta_info_append_header (&meta1, data);
  EXPECT_TRUE (result != NULL);

  msize = gst_memory_get_sizes (result, NULL, NULL);
  EXPECT_EQ (msize, hsize + 300U);

  ret = gst_tensor_meta_info_parse_memory (&meta2, result);
  EXPECT_TRUE (ret);

  EXPECT_EQ (meta1.version, meta2.version);
  EXPECT_EQ (meta2.type, _NNS_INT16);
  EXPECT_EQ (meta2.format, _NNS_TENSOR_FORMAT_FLEXIBLE);
  EXPECT_EQ ((media_type) meta2.media_type, _NNS_OCTET);
  EXPECT_EQ (meta2.dimension[0], 300U);
  EXPECT_EQ (meta2.dimension[1], 1U);

  gst_memory_unref (data);
  gst_memory_unref (result);
}

/**
 * @brief Test for tensor meta info (append header to memory with invalid param).
 */
TEST (commonMetaInfo, appendHeaderInvalidParam_n)
{
  GstTensorMetaInfo meta;
  GstMemory *result, *data;

  gst_tensor_meta_info_init (&meta);
  data = gst_allocator_alloc (NULL, 300, NULL);

  result = gst_tensor_meta_info_append_header (NULL, data);
  EXPECT_FALSE (result != NULL);

  result = gst_tensor_meta_info_append_header (&meta, NULL);
  EXPECT_FALSE (result != NULL);

  gst_memory_unref (data);
}

/**
 * @brief Test for tensor meta info (convert meta).
 */
TEST (commonMetaInfo, convertMeta)
{
  GstTensorMetaInfo meta;
  GstTensorInfo info1, info2;
  gboolean ret;

  gst_tensor_info_init (&info1);
  info1.type = _NNS_INT16;
  gst_tensor_parse_dimension ("300:1", info1.dimension);

  ret = gst_tensor_info_convert_to_meta (&info1, &meta);
  EXPECT_TRUE (ret);

  gst_tensor_info_init (&info2);
  ret = gst_tensor_meta_info_convert (&meta, &info2);
  EXPECT_TRUE (ret);

  EXPECT_EQ (info2.type, _NNS_INT16);
  EXPECT_EQ (info2.dimension[0], 300U);
  EXPECT_EQ (info2.dimension[1], 1U);
}

/**
 * @brief Test for tensor meta info (convert gst-info with invalid param).
 */
TEST (commonMetaInfo, convertMetaInvalidParam01_n)
{
  GstTensorMetaInfo meta;
  GstTensorInfo info;
  gboolean ret;

  gst_tensor_info_init (&info);
  info.type = _NNS_UINT16;
  gst_tensor_parse_dimension ("100:1", info.dimension);

  ret = gst_tensor_info_convert_to_meta (NULL, &meta);
  EXPECT_FALSE (ret);

  ret = gst_tensor_info_convert_to_meta (&info, NULL);
  EXPECT_FALSE (ret);
}

/**
 * @brief Test for tensor meta info (convert meta with invalid param).
 */
TEST (commonMetaInfo, convertMetaInvalidParam02_n)
{
  GstTensorMetaInfo meta;
  GstTensorInfo info;
  gboolean ret;

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_UINT16;
  meta.format = _NNS_TENSOR_FORMAT_STATIC;
  gst_tensor_parse_dimension ("100:1", meta.dimension);

  ret = gst_tensor_meta_info_convert (NULL, &info);
  EXPECT_FALSE (ret);

  ret = gst_tensor_meta_info_convert (&meta, NULL);
  EXPECT_FALSE (ret);
}

/**
 * @brief Test for aggregation utils (clear data).
 */
TEST (commonAggregationUtil, clearData)
{
  const gint64 key1 = 0;
  const gint64 key2 = 100;
  GHashTable *table;

  table = gst_tensor_aggregation_init ();

  EXPECT_TRUE (gst_tensor_aggregation_get_adapter (table, key1) != NULL);
  EXPECT_TRUE (gst_tensor_aggregation_get_adapter (table, key2) != NULL);

  gst_adapter_push (gst_tensor_aggregation_get_adapter (table, key1),
      gst_buffer_new_allocate (NULL, 1024U, 0));
  gst_adapter_push (gst_tensor_aggregation_get_adapter (table, key2),
      gst_buffer_new_allocate (NULL, 512U, 0));
  gst_adapter_push (gst_tensor_aggregation_get_adapter (table, key1),
      gst_buffer_new_allocate (NULL, 100U, 0));

  EXPECT_EQ (gst_adapter_available (gst_tensor_aggregation_get_adapter (table, key1)), 1124U);
  EXPECT_EQ (gst_adapter_available (gst_tensor_aggregation_get_adapter (table, key2)), 512U);

  gst_tensor_aggregation_clear (table, key2);
  EXPECT_EQ (gst_adapter_available (gst_tensor_aggregation_get_adapter (table, key2)), 0U);
  gst_adapter_push (gst_tensor_aggregation_get_adapter (table, key2),
      gst_buffer_new_allocate (NULL, 200U, 0));
  EXPECT_EQ (gst_adapter_available (gst_tensor_aggregation_get_adapter (table, key2)), 200U);

  gst_tensor_aggregation_clear_all (table);
  EXPECT_EQ (gst_adapter_available (gst_tensor_aggregation_get_adapter (table, key1)), 0U);
  EXPECT_EQ (gst_adapter_available (gst_tensor_aggregation_get_adapter (table, key2)), 0U);

  g_hash_table_destroy (table);
}

/**
 * @brief Test for aggregation utils (null param).
 */
TEST (commonAggregationUtil, nullParam_n)
{
  GstAdapter *adapter;

  adapter = gst_tensor_aggregation_get_adapter (NULL, 0);
  EXPECT_FALSE (adapter != NULL);
}

/**
 * @brief Create null files
 */
static gchar *
create_null_file (const gchar *dir, const gchar *file)
{
  gchar *fullpath = g_build_path ("/", dir, file, NULL);
  FILE *fp = g_fopen (fullpath, "w");

  if (fp) {
    fclose (fp);
  } else {
    g_clear_pointer (&fullpath, g_free);
  }

  return fullpath;
}

/**
 * @brief Check string custom conf
 */
static gboolean
check_custom_conf (const gchar *group, const gchar *key, const gchar *expected)
{
  gchar *str = nnsconf_get_custom_value_string (group, key);
  gboolean ret = (0 == g_strcmp0 (str, expected));

  g_free (str);
  return ret;
}

/**
 * @brief Test custom configurations
 */
TEST (confCustom, envStr01)
{
  gchar *fullpath = g_build_path ("/", g_get_tmp_dir (), "nns-tizen-XXXXXX", NULL);
  gchar *dir = g_mkdtemp (fullpath);
  gchar *filename = g_build_path ("/", dir, "nnstreamer.ini", NULL);
  gchar *dirf = g_build_path ("/", dir, "filters", NULL);
  gchar *dircf = g_build_path ("/", dir, "custom", NULL);
  gchar *dird = g_build_path ("/", dir, "decoders", NULL);

  EXPECT_EQ (g_mkdir (dirf, 0755), 0);
  EXPECT_EQ (g_mkdir (dircf, 0755), 0);
  EXPECT_EQ (g_mkdir (dird, 0755), 0);

  FILE *fp = g_fopen (filename, "w");
  const gchar *fn;
  const gchar *base_confenv;
  gchar *confenv;

  ASSERT_TRUE (fp != NULL);

  base_confenv = g_getenv ("NNSTREAMER_CONF");
  confenv = (base_confenv != NULL) ? g_strdup (base_confenv) : NULL;

  g_fprintf (fp, "[common]\n");
  g_fprintf (fp, "enable_envvar=True\n");
  g_fprintf (fp, "[filter]\n");
  g_fprintf (fp, "filters=%s\n", dirf);
  g_fprintf (fp, "customfilters=%s\n", dircf);
  g_fprintf (fp, "[decoder]\n");
  g_fprintf (fp, "decoders=%s\n", dird);
  g_fprintf (fp, "[customX]\n");
  g_fprintf (fp, "abc=OFF\n");
  g_fprintf (fp, "def=on\n");
  g_fprintf (fp, "ghi=TRUE\n");
  g_fprintf (fp, "n01=fAlSe\n");
  g_fprintf (fp, "n02=yeah\n");
  g_fprintf (fp, "n03=NAH\n");
  g_fprintf (fp, "n04=1\n");
  g_fprintf (fp, "n05=0\n");
  g_fprintf (fp, "mzx=whatsoever\n");
  g_fprintf (fp, "[customY]\n");
  g_fprintf (fp, "mzx=dunno\n");
  g_fprintf (fp, "[customZ]\n");
  g_fprintf (fp, "mzx=wth\n");
  g_fprintf (fp, "n05=1\n");

  fclose (fp);

  gchar *f1 = create_null_file (
      dirf, "libnnstreamer_filter_fantastic" NNSTREAMER_SO_FILE_EXTENSION);
  gchar *f2 = create_null_file (
      dirf, "libnnstreamer_filter_neuralnetwork" NNSTREAMER_SO_FILE_EXTENSION);
  gchar *f3 = create_null_file (dird, "libnnstreamer_decoder_omg" NNSTREAMER_SO_FILE_EXTENSION);
  gchar *f4 = create_null_file (
      dird, "libnnstreamer_decoder_wthisgoingon" NNSTREAMER_SO_FILE_EXTENSION);
  gchar *f5 = create_null_file (dircf, "custom_mechanism" NNSTREAMER_SO_FILE_EXTENSION);
  gchar *f6 = create_null_file (dircf, "fastfaster" NNSTREAMER_SO_FILE_EXTENSION);

  EXPECT_TRUE (g_setenv ("NNSTREAMER_CONF", filename, TRUE));
  EXPECT_TRUE (nnsconf_loadconf (TRUE));

  fn = nnsconf_get_fullpath ("fantastic", NNSCONF_PATH_FILTERS);
  EXPECT_STREQ (fn, f1);
  fn = nnsconf_get_fullpath ("neuralnetwork", NNSCONF_PATH_FILTERS);
  EXPECT_STREQ (fn, f2);
  fn = nnsconf_get_fullpath ("notfound", NNSCONF_PATH_FILTERS);
  EXPECT_STREQ (fn, NULL);
  fn = nnsconf_get_fullpath ("omg", NNSCONF_PATH_DECODERS);
  EXPECT_STREQ (fn, f3);
  fn = nnsconf_get_fullpath ("wthisgoingon", NNSCONF_PATH_DECODERS);
  EXPECT_STREQ (fn, f4);
  fn = nnsconf_get_fullpath ("notfound", NNSCONF_PATH_DECODERS);
  EXPECT_STREQ (fn, NULL);
  fn = nnsconf_get_fullpath ("custom_mechanism", NNSCONF_PATH_CUSTOM_FILTERS);
  EXPECT_STREQ (fn, f5);
  fn = nnsconf_get_fullpath ("fastfaster", NNSCONF_PATH_CUSTOM_FILTERS);
  EXPECT_STREQ (fn, f6);
  fn = nnsconf_get_fullpath ("notfound", NNSCONF_PATH_CUSTOM_FILTERS);
  EXPECT_STREQ (fn, NULL);

  EXPECT_TRUE (check_custom_conf ("customX", "abc", "OFF"));
  EXPECT_FALSE (nnsconf_get_custom_value_bool ("customX", "abc", TRUE));
  EXPECT_FALSE (nnsconf_get_custom_value_bool ("customX", "abc", FALSE));
  EXPECT_TRUE (check_custom_conf ("customX", "def", "on"));
  EXPECT_TRUE (nnsconf_get_custom_value_bool ("customX", "def", FALSE));
  EXPECT_TRUE (nnsconf_get_custom_value_bool ("customX", "def", TRUE));
  EXPECT_TRUE (check_custom_conf ("customX", "ghi", "TRUE"));
  EXPECT_TRUE (nnsconf_get_custom_value_bool ("customX", "ghi", FALSE));
  EXPECT_TRUE (nnsconf_get_custom_value_bool ("customX", "ghi", TRUE));
  EXPECT_TRUE (check_custom_conf ("customX", "n02", "yeah"));
  EXPECT_TRUE (nnsconf_get_custom_value_bool ("customX", "n02", FALSE));
  EXPECT_TRUE (nnsconf_get_custom_value_bool ("customX", "n02", TRUE));
  EXPECT_TRUE (check_custom_conf ("customX", "n03", "NAH"));
  EXPECT_FALSE (nnsconf_get_custom_value_bool ("customX", "n03", FALSE));
  EXPECT_FALSE (nnsconf_get_custom_value_bool ("customX", "n03", TRUE));
  EXPECT_TRUE (check_custom_conf ("customX", "n04", "1"));
  EXPECT_TRUE (nnsconf_get_custom_value_bool ("customX", "n04", FALSE));
  EXPECT_TRUE (nnsconf_get_custom_value_bool ("customX", "n04", TRUE));
  EXPECT_TRUE (check_custom_conf ("customX", "n05", "0"));
  EXPECT_FALSE (nnsconf_get_custom_value_bool ("customX", "n05", FALSE));
  EXPECT_FALSE (nnsconf_get_custom_value_bool ("customX", "n05", TRUE));
  EXPECT_TRUE (check_custom_conf ("customX", "mzx", "whatsoever"));
  EXPECT_FALSE (nnsconf_get_custom_value_bool ("customX", "mzx", FALSE));
  EXPECT_TRUE (nnsconf_get_custom_value_bool ("customX", "mzx", TRUE));
  EXPECT_TRUE (check_custom_conf ("customY", "mzx", "dunno"));
  EXPECT_FALSE (nnsconf_get_custom_value_bool ("customY", "mzx", FALSE));
  EXPECT_TRUE (nnsconf_get_custom_value_bool ("customY", "mzx", TRUE));
  EXPECT_TRUE (check_custom_conf ("customZ", "mzx", "wth"));
  EXPECT_FALSE (nnsconf_get_custom_value_bool ("customZ", "mzx", FALSE));
  EXPECT_TRUE (nnsconf_get_custom_value_bool ("customZ", "mzx", TRUE));
  EXPECT_TRUE (check_custom_conf ("customZ", "n05", "1"));
  EXPECT_TRUE (nnsconf_get_custom_value_bool ("customZ", "n05", FALSE));
  EXPECT_TRUE (nnsconf_get_custom_value_bool ("customZ", "n05", TRUE));
  EXPECT_TRUE (check_custom_conf ("customW", "n05", NULL));
  EXPECT_FALSE (nnsconf_get_custom_value_bool ("customW", "n05", FALSE));
  EXPECT_TRUE (nnsconf_get_custom_value_bool ("customW", "n05", TRUE));

  g_free (f1);
  g_free (f2);
  g_free (f3);
  g_free (f4);
  g_free (f5);
  g_free (f6);
  g_free (fullpath);
  g_free (filename);
  g_free (dirf);
  g_free (dircf);
  g_free (dird);

  if (confenv) {
    EXPECT_TRUE (g_setenv ("NNSTREAMER_CONF", confenv, TRUE));
    g_free (confenv);
  } else {
    g_unsetenv ("NNSTREAMER_CONF");
  }
}

/**
 * @brief Test for extra configuration path
 */
TEST (confCustom, checkExtraConfPath_p)
{
  gchar *fullpath = g_build_path ("/", g_get_tmp_dir (), "nns-tizen-XXXXXX", NULL);
  gchar *dir = g_mkdtemp (fullpath);
  gchar *filename = g_build_path ("/", dir, "nnstreamer.ini", NULL);
  const gchar *extra_conf = "/opt/usr/vd/product.ini";
  gchar *confenv = g_strdup (g_getenv ("NNSTREAMER_CONF"));

  FILE *fp = g_fopen (filename, "w");
  ASSERT_TRUE (fp != NULL);
  g_fprintf (fp, "[common]\n");
  g_fprintf (fp, "extra_config_path=%s\n", extra_conf);
  fclose (fp);

  EXPECT_TRUE (g_setenv ("NNSTREAMER_CONF", filename, TRUE));
  EXPECT_TRUE (nnsconf_loadconf (TRUE));

  EXPECT_TRUE (check_custom_conf ("common", "extra_config_path", extra_conf));
  removeTempFile (&filename);
  g_free (fullpath);

  if (confenv) {
    EXPECT_TRUE (g_setenv ("NNSTREAMER_CONF", confenv, TRUE));
    g_free (confenv);
  } else {
    g_unsetenv ("NNSTREAMER_CONF");
  }
}

/**
 * @brief Test nnstreamer conf util (name prefix with invalid param).
 */
TEST (confCustom, subpluginPrefix_n)
{
  EXPECT_STREQ (nnsconf_get_subplugin_name_prefix (NNSCONF_PATH_END), NULL);
  EXPECT_STREQ (nnsconf_get_subplugin_name_prefix ((nnsconf_type_path) -1), NULL);
}

/**
 * @brief Test version control (positive)
 */
TEST (versionControl, getVer01)
{
  gchar *verstr = nnstreamer_version_string ();
  guint major, minor, micro;
  gchar *verstr2, *verstr3;
  nnstreamer_version_fetch (&major, &minor, &micro);

  verstr2 = g_strdup_printf ("NNStreamer %u.%u.%u", major, minor, micro);
  verstr3 = g_strdup_printf ("%u.%u.%u", major, minor, micro);

  EXPECT_STRCASEEQ (verstr, verstr2);

  EXPECT_STRCASEEQ (VERSION, verstr3);

  EXPECT_EQ ((int) major, NNSTREAMER_VERSION_MAJOR);
  EXPECT_EQ ((int) minor, NNSTREAMER_VERSION_MINOR);
  EXPECT_EQ ((int) micro, NNSTREAMER_VERSION_MICRO);

  g_free (verstr);
  g_free (verstr2);
  g_free (verstr3);
}

/**
 * @brief Test tensor buffer util (create static tensor buffer)
 */
TEST (commonUtil, createStaticTensorBuffer)
{
  GstTensorsConfig config;
  GstBuffer *in, *out;
  GstMemory *mem;
  GstMapInfo map;
  guint i, j, num, expected;
  guint *input, *output;
  gsize data_size;

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_STATIC;
  config.rate_n = config.rate_d = 1;
  config.info.num_tensors = 3U;
  gst_tensors_info_parse_dimensions_string (&config.info, "20,40,50");
  gst_tensors_info_parse_types_string (&config.info, "uint32,uint32,uint32");

  data_size = gst_tensors_info_get_size (&config.info, -1);
  input = (guint *) g_malloc (data_size);

  for (i = 0; i < data_size / sizeof (guint); i++)
    input[i] = i;

  in = gst_buffer_new_wrapped (input, data_size);
  out = gst_tensor_buffer_from_config (in, &config);
  ASSERT_TRUE (out != NULL);

  num = gst_buffer_n_memory (out);
  EXPECT_EQ (num, config.info.num_tensors);

  expected = 0U;
  for (i = 0; i < num; i++) {
    mem = gst_buffer_peek_memory (out, i);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
    EXPECT_EQ (map.size, gst_tensors_info_get_size (&config.info, i));

    output = (guint *) map.data;
    for (j = 0; j < map.size / sizeof (guint); j++)
      EXPECT_EQ (output[j], expected++);

    gst_memory_unmap (mem, &map);
  }

  gst_buffer_unref (out);
  gst_tensors_config_free (&config);
}

/**
 * @brief Test tensor buffer util (create flexible tensor buffer)
 */
TEST (commonUtil, createFlexTensorBuffer)
{
  GstTensorsConfig config;
  GstBuffer *in, *out;
  GstMemory *mem;
  GstMapInfo map;
  guint i, j, num;
  guint *input, *output;
  guint8 *data;
  GstTensorMetaInfo meta[3];
  gsize data_size, offset, hsize[3], dsize[3];

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  config.rate_n = config.rate_d = 1;
  config.info.num_tensors = 3U;
  gst_tensors_info_parse_dimensions_string (&config.info, "20,30,40");
  gst_tensors_info_parse_types_string (&config.info, "uint32,uint32,uint32");

  data_size = 0;
  for (i = 0; i < 3U; i++) {
    gst_tensor_info_convert_to_meta (&config.info.info[i], &meta[i]);
    hsize[i] = gst_tensor_meta_info_get_header_size (&meta[i]);
    dsize[i] = gst_tensor_meta_info_get_data_size (&meta[i]);
    data_size += (hsize[i] + dsize[i]);
  }

  data = (guint8 *) g_malloc (data_size);

  offset = 0;
  for (i = 0; i < 3U; i++) {
    gst_tensor_meta_info_update_header (&meta[i], data + offset);

    input = (guint *) (data + offset + hsize[i]);
    for (j = 0; j < dsize[i] / sizeof (guint); j++)
      input[j] = i * 10 + j;

    offset += (hsize[i] + dsize[i]);
  }

  in = gst_buffer_new_wrapped (data, data_size);
  out = gst_tensor_buffer_from_config (in, &config);
  ASSERT_TRUE (out != NULL);

  num = gst_buffer_n_memory (out);
  EXPECT_EQ (num, config.info.num_tensors);

  for (i = 0; i < num; i++) {
    mem = gst_buffer_peek_memory (out, i);
    ASSERT_TRUE (gst_memory_map (mem, &map, GST_MAP_READ));
    EXPECT_EQ (map.size, hsize[i] + dsize[i]);

    output = (guint *) (map.data + hsize[i]);
    for (j = 0; j < dsize[i] / sizeof (guint); j++)
      EXPECT_EQ (output[j], i * 10 + j);

    gst_memory_unmap (mem, &map);
  }

  gst_buffer_unref (out);
  gst_tensors_config_free (&config);
}

/**
 * @brief Test tensor buffer util (invalid config)
 */
TEST (commonUtil, createTensorBufferInvalidConfig_n)
{
  GstTensorsConfig config;
  GstBuffer *in, *out;

  gst_tensors_config_init (&config);

  in = gst_buffer_new_allocate (NULL, 200, NULL);
  out = gst_tensor_buffer_from_config (in, &config);
  EXPECT_FALSE (out != NULL);

  gst_buffer_unref (out);
  gst_tensors_config_free (&config);
}

/**
 * @brief Test tensor buffer util (null param)
 */
TEST (commonUtil, createTensorBufferNullParam_n)
{
  GstTensorsConfig config;
  GstBuffer *in, *out;
  gsize data_size;

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_STATIC;
  config.rate_n = config.rate_d = 1;
  config.info.num_tensors = 3U;
  gst_tensors_info_parse_dimensions_string (&config.info, "20,40,50");
  gst_tensors_info_parse_types_string (&config.info, "uint32,uint32,uint32");

  data_size = gst_tensors_info_get_size (&config.info, -1);

  in = gst_buffer_new_allocate (NULL, data_size, NULL);
  out = gst_tensor_buffer_from_config (NULL, &config);
  EXPECT_FALSE (out != NULL);
  gst_buffer_unref (in);
  gst_buffer_unref (out);

  in = gst_buffer_new_allocate (NULL, data_size, NULL);
  out = gst_tensor_buffer_from_config (in, NULL);
  EXPECT_FALSE (out != NULL);

  gst_buffer_unref (out);
  gst_tensors_config_free (&config);
}

/**
 * @brief Test tensor buffer util (invalid buffer size)
 */
TEST (commonUtil, createTensorBufferInvalidSize_n)
{
  GstTensorsConfig config;
  GstBuffer *in, *out;
  gsize data_size;

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_STATIC;
  config.rate_n = config.rate_d = 1;
  config.info.num_tensors = 3U;
  gst_tensors_info_parse_dimensions_string (&config.info, "20,40,50");
  gst_tensors_info_parse_types_string (&config.info, "uint32,uint32,uint32");

  data_size = gst_tensors_info_get_size (&config.info, -1) / 2;

  in = gst_buffer_new_allocate (NULL, data_size, NULL);
  out = gst_tensor_buffer_from_config (in, &config);
  EXPECT_FALSE (out != NULL);

  gst_buffer_unref (out);
  gst_tensors_config_free (&config);
}

/**
 * @brief Test tensor dimension validation check util
 */
TEST (commonUtil, tensorDimensionIsValid)
{
  tensor_dim dim;
  gint i;

  dim[0] = 3;
  dim[1] = 280;
  dim[2] = 40;
  dim[3] = 1;

  for (i = 4; i < NNS_TENSOR_RANK_LIMIT; i++) {
    dim[i] = 0;
  }

  EXPECT_TRUE (gst_tensor_dimension_is_valid (dim));
}

/**
 * @brief Test tensor dimension validation check util
 */
TEST (commonUtil, tensorDimensionIsValid_n)
{
  tensor_dim dim;
  gint i;

  dim[0] = 3;
  dim[1] = 280;
  dim[2] = 40;
  dim[3] = 0;

  for (i = 4; i < NNS_TENSOR_RANK_LIMIT; i++) {
    dim[i] = 1;
  }

  EXPECT_FALSE (gst_tensor_dimension_is_valid (dim));
}

/**
 * @brief Test tensor dimension compare util
 */
TEST (commonUtil, tensorDimensionIsEqual)
{
  tensor_dim dim1, dim2;
  gint i;

  dim1[0] = dim2[0] = 3;
  dim1[1] = dim2[1] = 280;
  dim1[2] = dim2[2] = 40;
  dim1[3] = dim2[3] = 1;

  for (i = 4; i < NNS_TENSOR_RANK_LIMIT; i++) {
    dim1[i] = 0;
    dim2[i] = 1;
  }

  EXPECT_TRUE (gst_tensor_dimension_is_equal (dim1, dim2));
}

/**
 * @brief Test tensor dimension compare util
 */
TEST (commonUtil, tensorDimensionIsEqual_n)
{
  tensor_dim dim1, dim2;
  gint i;

  dim1[0] = dim2[0] = 3;
  dim1[1] = dim2[1] = 280;
  dim1[2] = dim2[2] = 40;
  dim1[3] = dim2[3] = 1;
  dim1[4] = 0;
  dim2[4] = 2;

  for (i = 5; i < NNS_TENSOR_RANK_LIMIT; i++) {
    dim1[i] = 0;
    dim2[i] = 1;
  }

  EXPECT_FALSE (gst_tensor_dimension_is_equal (dim1, dim2));
}

/**
 * @brief Test tensor dimension compare util
 */
TEST (commonUtil, tensorDimensionIsEqualInvalid_n)
{
  tensor_dim dim1, dim2;
  gint i;

  dim1[0] = dim2[0] = 3;
  dim1[1] = dim2[1] = 4;
  dim1[2] = dim2[2] = 0;
  dim1[3] = dim2[3] = 1;

  for (i = 5; i < NNS_TENSOR_RANK_LIMIT; i++) {
    dim1[i] = dim2[i] = 0;
  }

  EXPECT_FALSE (gst_tensor_dimension_is_equal (dim1, dim2));
}

/**
 * @brief Test for parsing tensor dimension.
 */
TEST (commonUtil, tensorDimensionParseInvalidParam01_n)
{
  tensor_dim dim = { 0 };

  EXPECT_EQ (0U, gst_tensor_parse_dimension (NULL, dim));
}

/**
 * @brief Test for dimension rank utils.
 */
TEST (commonUtil, tensorDimensionGetRank)
{
  tensor_dim dim = { 0 };

  EXPECT_EQ (0U, gst_tensor_dimension_get_min_rank (dim));

  dim[0] = dim[1] = dim[2] = 3;
  dim[3] = 1;

  EXPECT_EQ (4U, gst_tensor_dimension_get_rank (dim));
  EXPECT_EQ (3U, gst_tensor_dimension_get_min_rank (dim));
}

/**
 * @brief Test for error util.
 */
TEST (commonUtil, errorMessage)
{
  const gchar err_message[] = "Adding error message for test";

  /* error message */
  _nnstreamer_error_clean ();
  _nnstreamer_error_write (err_message);

  EXPECT_STREQ (err_message, _nnstreamer_error ());

  _nnstreamer_error_clean ();
  EXPECT_EQ (NULL, _nnstreamer_error ());
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
