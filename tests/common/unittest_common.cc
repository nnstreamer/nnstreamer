/**
 * @file unittest_common.cc
 * @date 31 May 2018
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @brief Unit test module for NNStreamer common library
 * @see		https://github.com/nnstreamer/nnstreamer
 * @bug		No known bugs.
 *
 *  @brief Unit test module for NNStreamer common library
 *  @bug	No known bugs except for NYI items
 *
 *  Copyright 2018 Samsung Electronics
 *
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
 * @brief Test for int32 type string.
 */
TEST (common_get_tensor_type, failure_n)
{
  EXPECT_EQ (gst_tensor_get_type (""), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type (NULL), _NNS_END);
}

/**
 * @brief Test for int32 type string.
 */
TEST (common_get_tensor_type, int32)
{
  EXPECT_EQ (gst_tensor_get_type ("int32"), _NNS_INT32);
  EXPECT_EQ (gst_tensor_get_type ("INT32"), _NNS_INT32);
  EXPECT_EQ (gst_tensor_get_type ("iNt32"), _NNS_INT32);
  EXPECT_EQ (gst_tensor_get_type ("InT32"), _NNS_INT32);
}

/**
 * @brief Test for int32 type string.
 */
TEST (common_get_tensor_type, int32_n)
{
  EXPECT_EQ (gst_tensor_get_type ("InT322"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("int3"), _NNS_END);
}

/**
 * @brief Test for int16 type string.
 */
TEST (common_get_tensor_type, int16)
{
  EXPECT_EQ (gst_tensor_get_type ("int16"), _NNS_INT16);
  EXPECT_EQ (gst_tensor_get_type ("INT16"), _NNS_INT16);
  EXPECT_EQ (gst_tensor_get_type ("iNt16"), _NNS_INT16);
  EXPECT_EQ (gst_tensor_get_type ("InT16"), _NNS_INT16);
}

/**
 * @brief Test for int16 type string.
 */
TEST (common_get_tensor_type, int16_n)
{
  EXPECT_EQ (gst_tensor_get_type ("InT162"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("int1"), _NNS_END);
}

/**
 * @brief Test for int8 type string.
 */
TEST (common_get_tensor_type, int8)
{
  EXPECT_EQ (gst_tensor_get_type ("int8"), _NNS_INT8);
  EXPECT_EQ (gst_tensor_get_type ("INT8"), _NNS_INT8);
  EXPECT_EQ (gst_tensor_get_type ("iNt8"), _NNS_INT8);
  EXPECT_EQ (gst_tensor_get_type ("InT8"), _NNS_INT8);
}

/**
 * @brief Test for int8 type string.
 */
TEST (common_get_tensor_type, int8_n)
{
  EXPECT_EQ (gst_tensor_get_type ("InT82"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("int3"), _NNS_END);
}

/**
 * @brief Test for uint32 type string.
 */
TEST (common_get_tensor_type, uint32)
{
  EXPECT_EQ (gst_tensor_get_type ("uint32"), _NNS_UINT32);
  EXPECT_EQ (gst_tensor_get_type ("UINT32"), _NNS_UINT32);
  EXPECT_EQ (gst_tensor_get_type ("uiNt32"), _NNS_UINT32);
  EXPECT_EQ (gst_tensor_get_type ("UInT32"), _NNS_UINT32);
}

/**
 * @brief Test for uint32 type string.
 */
TEST (common_get_tensor_type, uint32_n)
{
  EXPECT_EQ (gst_tensor_get_type ("UInT322"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("uint3"), _NNS_END);
}

/**
 * @brief Test for uint16 type string.
 */
TEST (common_get_tensor_type, uint16)
{
  EXPECT_EQ (gst_tensor_get_type ("uint16"), _NNS_UINT16);
  EXPECT_EQ (gst_tensor_get_type ("UINT16"), _NNS_UINT16);
  EXPECT_EQ (gst_tensor_get_type ("uiNt16"), _NNS_UINT16);
  EXPECT_EQ (gst_tensor_get_type ("UInT16"), _NNS_UINT16);
}

/**
 * @brief Test for uint16 type string.
 */
TEST (common_get_tensor_type, uint16_n)
{
  EXPECT_EQ (gst_tensor_get_type ("UInT162"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("uint1"), _NNS_END);
}

/**
 * @brief Test for uint8 type string.
 */
TEST (common_get_tensor_type, uint8)
{
  EXPECT_EQ (gst_tensor_get_type ("uint8"), _NNS_UINT8);
  EXPECT_EQ (gst_tensor_get_type ("UINT8"), _NNS_UINT8);
  EXPECT_EQ (gst_tensor_get_type ("uiNt8"), _NNS_UINT8);
  EXPECT_EQ (gst_tensor_get_type ("UInT8"), _NNS_UINT8);
}

/**
 * @brief Test for uint8 type string.
 */
TEST (common_get_tensor_type, uint8_n)
{
  EXPECT_EQ (gst_tensor_get_type ("UInT82"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("uint3"), _NNS_END);
}

/**
 * @brief Test for float32 type string.
 */
TEST (common_get_tensor_type, float32)
{
  EXPECT_EQ (gst_tensor_get_type ("float32"), _NNS_FLOAT32);
  EXPECT_EQ (gst_tensor_get_type ("FLOAT32"), _NNS_FLOAT32);
  EXPECT_EQ (gst_tensor_get_type ("float32"), _NNS_FLOAT32);
  EXPECT_EQ (gst_tensor_get_type ("FloaT32"), _NNS_FLOAT32);
}

/**
 * @brief Test for float32 type string.
 */
TEST (common_get_tensor_type, float32_n)
{
  EXPECT_EQ (gst_tensor_get_type ("FloaT322"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("float3"), _NNS_END);
}

/**
 * @brief Test for float64 type string.
 */
TEST (common_get_tensor_type, float64)
{
  EXPECT_EQ (gst_tensor_get_type ("float64"), _NNS_FLOAT64);
  EXPECT_EQ (gst_tensor_get_type ("FLOAT64"), _NNS_FLOAT64);
  EXPECT_EQ (gst_tensor_get_type ("float64"), _NNS_FLOAT64);
  EXPECT_EQ (gst_tensor_get_type ("FloaT64"), _NNS_FLOAT64);
}

/**
 * @brief Test for float64 type string.
 */
TEST (common_get_tensor_type, float64_n)
{
  EXPECT_EQ (gst_tensor_get_type ("FloaT642"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("float6"), _NNS_END);
}

/**
 * @brief Test for int64 type string.
 */
TEST (common_get_tensor_type, int64)
{
  EXPECT_EQ (gst_tensor_get_type ("int64"), _NNS_INT64);
  EXPECT_EQ (gst_tensor_get_type ("INT64"), _NNS_INT64);
  EXPECT_EQ (gst_tensor_get_type ("iNt64"), _NNS_INT64);
  EXPECT_EQ (gst_tensor_get_type ("InT64"), _NNS_INT64);
}

/**
 * @brief Test for int64 type string.
 */
TEST (common_get_tensor_type, int64_n)
{
  EXPECT_EQ (gst_tensor_get_type ("InT642"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("int6"), _NNS_END);
}

/**
 * @brief Test for uint64 type string.
 */
TEST (common_get_tensor_type, uint64)
{
  EXPECT_EQ (gst_tensor_get_type ("uint64"), _NNS_UINT64);
  EXPECT_EQ (gst_tensor_get_type ("UINT64"), _NNS_UINT64);
  EXPECT_EQ (gst_tensor_get_type ("uiNt64"), _NNS_UINT64);
  EXPECT_EQ (gst_tensor_get_type ("UInT64"), _NNS_UINT64);
}

/**
 * @brief Test for uint64 type string.
 */
TEST (common_get_tensor_type, uint64_n)
{
  EXPECT_EQ (gst_tensor_get_type ("UInT642"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("uint6"), _NNS_END);
}

/**
 * @brief Test to find index of the key.
 */
TEST (common_find_key_strv, key_index)
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
TEST (common_get_tensor_dimension, case1)
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
  EXPECT_STREQ (dim_str, "345:123:433:177");
  g_free (dim_str);
}

/**
 * @brief Test for tensor dimension.
 */
TEST (common_get_tensor_dimension, case2)
{
  tensor_dim dim;
  gchar *dim_str;
  guint rank;

  rank = gst_tensor_parse_dimension ("345:123:433", dim);
  EXPECT_EQ (rank, 3U);
  EXPECT_EQ (dim[0], 345U);
  EXPECT_EQ (dim[1], 123U);
  EXPECT_EQ (dim[2], 433U);
  EXPECT_EQ (dim[3], 1U);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_STREQ (dim_str, "345:123:433:1");
  g_free (dim_str);
}

/**
 * @brief Test for tensor dimension.
 */
TEST (common_get_tensor_dimension, case3)
{
  tensor_dim dim;
  gchar *dim_str;
  guint rank;

  rank = gst_tensor_parse_dimension ("345:123", dim);
  EXPECT_EQ (rank, 2U);
  EXPECT_EQ (dim[0], 345U);
  EXPECT_EQ (dim[1], 123U);
  EXPECT_EQ (dim[2], 1U);
  EXPECT_EQ (dim[3], 1U);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_STREQ (dim_str, "345:123:1:1");
  g_free (dim_str);
}

/**
 * @brief Test for tensor dimension.
 */
TEST (common_get_tensor_dimension, case4)
{
  tensor_dim dim;
  gchar *dim_str;
  guint rank;

  rank = gst_tensor_parse_dimension ("345", dim);
  EXPECT_EQ (rank, 1U);
  EXPECT_EQ (dim[0], 345U);
  EXPECT_EQ (dim[1], 1U);
  EXPECT_EQ (dim[2], 1U);
  EXPECT_EQ (dim[3], 1U);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_STREQ (dim_str, "345:1:1:1");
  g_free (dim_str);
}

/**
 * @brief Test to copy tensor info.
 */
TEST (common_tensor_info, copy_tensor)
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

  src = { NULL, _NNS_INT32, { 5, 6, 7, 8 } };
  gst_tensor_info_copy (&dest, &src);

  EXPECT_TRUE (dest.name == NULL);
  EXPECT_EQ (dest.type, src.type);
  EXPECT_EQ (dest.dimension[0], src.dimension[0]);
  EXPECT_EQ (dest.dimension[1], src.dimension[1]);
  EXPECT_EQ (dest.dimension[2], src.dimension[2]);
  EXPECT_EQ (dest.dimension[3], src.dimension[3]);

  g_free (test_name);
}

/**
 * @brief Test to copy tensor info.
 */
TEST (common_tensor_info, copy_tensors)
{
  GstTensorsInfo src, dest;
  gchar *test_name = g_strdup ("test-tensors");
  guint i;

  gst_tensors_info_init (&src);
  gst_tensors_info_init (&dest);

  src.num_tensors = 2;
  src.info[0] = { test_name, _NNS_INT32, { 1, 2, 3, 4 } };
  src.info[1] = { test_name, _NNS_FLOAT32, { 5, 6, 7, 8 } };
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

  g_free (test_name);
}

/**
 * @brief Internal function to update tensors info.
 */
static void
fill_tensors_info_for_test (GstTensorsInfo *info1, GstTensorsInfo *info2)
{
  g_assert (info1 != NULL && info2 != NULL);

  gst_tensors_info_init (info1);
  gst_tensors_info_init (info2);

  info1->num_tensors = info2->num_tensors = 2;

  info1->info[0].type = info2->info[0].type = _NNS_INT64;
  info1->info[1].type = info2->info[1].type = _NNS_FLOAT64;

  info1->info[0].dimension[0] = info2->info[0].dimension[0] = 2;
  info1->info[0].dimension[1] = info2->info[0].dimension[1] = 3;
  info1->info[0].dimension[2] = info2->info[0].dimension[2] = 1;
  info1->info[0].dimension[3] = info2->info[0].dimension[3] = 1;

  info1->info[1].dimension[0] = info2->info[1].dimension[0] = 5;
  info1->info[1].dimension[1] = info2->info[1].dimension[1] = 5;
  info1->info[1].dimension[2] = info2->info[1].dimension[2] = 1;
  info1->info[1].dimension[3] = info2->info[1].dimension[3] = 1;
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
 * @brief Test for data size.
 */
TEST (common_tensor_info, size_01_p)
{
  GstTensorsInfo info1, info2;
  gsize size1, size2;
  guint i;

  fill_tensors_info_for_test (&info1, &info2);

  size1 = gst_tensor_info_get_size (&info1.info[0]);
  size2 = gst_tensors_info_get_size (&info1, 0);

  EXPECT_TRUE (size1 == size2);

  size1 = 0;
  for (i = 0; i < info2.num_tensors; i++) {
    size1 += gst_tensor_info_get_size (&info2.info[i]);
  }

  size2 = gst_tensors_info_get_size (&info2, -1);

  EXPECT_TRUE (size1 == size2);
}

/**
 * @brief Test for data size.
 */
TEST (common_tensor_info, size_02_n)
{
  GstTensorsInfo info1, info2;
  gsize size1;
  gint index;

  fill_tensors_info_for_test (&info1, &info2);

  /* get size with null param */
  index = (gint)info1.num_tensors - 1;
  size1 = gst_tensors_info_get_size (NULL, index);

  EXPECT_TRUE (size1 == 0);
}

/**
 * @brief Test for data size.
 */
TEST (common_tensor_info, size_03_n)
{
  GstTensorsInfo info1, info2;
  gsize size1;
  gint index;

  fill_tensors_info_for_test (&info1, &info2);

  /* get size with invalid index */
  index = (gint)info1.num_tensors;
  size1 = gst_tensors_info_get_size (&info1, index);

  EXPECT_TRUE (size1 == 0);
}

/**
 * @brief Test for same tensors info.
 */
TEST (common_tensor_info, equal_01_p)
{
  GstTensorsInfo info1, info2;

  fill_tensors_info_for_test (&info1, &info2);

  EXPECT_TRUE (gst_tensors_info_is_equal (&info1, &info2));
}

/**
 * @brief Test for same tensors info.
 */
TEST (common_tensor_info, equal_02_n)
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
TEST (common_tensor_info, equal_03_n)
{
  GstTensorsInfo info1, info2;

  fill_tensors_info_for_test (&info1, &info2);

  /* change info, this should not be compatible */
  info1.num_tensors = 1;

  EXPECT_FALSE (gst_tensors_info_is_equal (&info1, &info2));
}

/**
 * @brief Test for same tensors info.
 */
TEST (common_tensor_info, equal_04_n)
{
  GstTensorsInfo info1, info2;

  fill_tensors_info_for_test (&info1, &info2);

  /* change info, this should not be compatible */
  info1.info[0].type = _NNS_UINT64;

  EXPECT_FALSE (gst_tensors_info_is_equal (&info1, &info2));
}

/**
 * @brief Test for same tensors info.
 */
TEST (common_tensor_info, equal_05_n)
{
  GstTensorsInfo info1, info2;

  fill_tensors_info_for_test (&info1, &info2);

  /* change info, this should not be compatible */
  info2.info[1].dimension[0] = 10;

  EXPECT_FALSE (gst_tensors_info_is_equal (&info1, &info2));
}

/**
 * @brief Test for same tensors config.
 */
TEST (common_tensors_config, equal_01_p)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);

  EXPECT_TRUE (gst_tensors_config_is_equal (&conf1, &conf2));
}

/**
 * @brief Test for same tensors config.
 */
TEST (common_tensors_config, equal_02_p)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);
  conf1.rate_n *= 2;
  conf1.rate_d *= 2;

  EXPECT_TRUE (gst_tensors_config_is_equal (&conf1, &conf2));
}

/**
 * @brief Test for same tensors config.
 */
TEST (common_tensors_config, equal_03_p)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);
  conf1.rate_n *= 0;
  conf2.rate_n *= 0;

  EXPECT_TRUE (gst_tensors_config_is_equal (&conf1, &conf2));
}

/**
 * @brief Test for same tensors config.
 */
TEST (common_tensors_config, equal_04_n)
{
  GstTensorsConfig conf1, conf2;

  gst_tensors_config_init (&conf1);
  gst_tensors_config_init (&conf2);

  EXPECT_FALSE (gst_tensors_config_is_equal (&conf1, &conf2));
}

/**
 * @brief Test for same tensors config.
 */
TEST (common_tensors_config, equal_05_n)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);
  conf1.rate_n *= 2;
  conf1.rate_d *= 4;

  EXPECT_FALSE (gst_tensors_config_is_equal (&conf1, &conf2));
}

/**
 * @brief Test for same tensors config.
 */
TEST (common_tensors_config, equal_06_n)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);
  conf1.rate_d *= 0;

  EXPECT_FALSE (gst_tensors_config_is_equal (&conf1, &conf2));
}

/**
 * @brief Test for dimensions string in tensors info.
 */
TEST (common_tensors_info_string, dimensions)
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
  EXPECT_STREQ (str_dims, "1:2:3:4");
  g_free (str_dims);

  /* 4 tensors info */
  num_dims = gst_tensors_info_parse_dimensions_string (&info, "1, 2:2, 3:3:3, 4:4:4:4");
  EXPECT_EQ (num_dims, 4U);

  info.num_tensors = num_dims;

  str_dims = gst_tensors_info_get_dimensions_string (&info);
  EXPECT_STREQ (str_dims, "1:1:1:1,2:2:1:1,3:3:3:1,4:4:4:4");
  g_free (str_dims);

  /* max */
  num_dims = gst_tensors_info_parse_dimensions_string (&info,
      "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20");
  EXPECT_EQ (num_dims, (guint)NNS_TENSOR_SIZE_LIMIT);
}

/**
 * @brief Test for types string in tensors info.
 */
TEST (common_tensors_info_string, types)
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

  /* max */
  num_types = gst_tensors_info_parse_types_string (&info,
      "int8, int8, int8, int8, int8, int8, int8, int8, int8, int8, int8, "
      "int8, int8, int8, int8, int8, int8, int8, int8, int8, int8, int8");
  EXPECT_EQ (num_types, (guint)NNS_TENSOR_SIZE_LIMIT);
}

/**
 * @brief Test for names string in tensors info.
 */
TEST (common_tensors_info_string, names)
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

  /* max */
  num_names = gst_tensors_info_parse_names_string (&info,
      "t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, "
      "t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28");
  EXPECT_EQ (num_names, (guint)NNS_TENSOR_SIZE_LIMIT);
}

/**
 * @brief Test to replace string.
 */
TEST (common_string_util, replace_str_01)
{
  gchar *result;
  guint changed;

  result = g_strdup ("sourceelement ! parser ! converter ! format ! converter ! format ! converter ! sink");

  result = replace_string (result, "sourceelement", "src", NULL, &changed);
  EXPECT_EQ (changed, 1U);
  EXPECT_STREQ (result, "src ! parser ! converter ! format ! converter ! format ! converter ! sink");

  result = replace_string (result, "format", "fmt", NULL, &changed);
  EXPECT_EQ (changed, 2U);
  EXPECT_STREQ (result, "src ! parser ! converter ! fmt ! converter ! fmt ! converter ! sink");

  result = replace_string (result, "converter", "conv", NULL, &changed);
  EXPECT_EQ (changed, 3U);
  EXPECT_STREQ (result, "src ! parser ! conv ! fmt ! conv ! fmt ! conv ! sink");

  result = replace_string (result, "invalidname", "invalid", NULL, &changed);
  EXPECT_EQ (changed, 0U);
  EXPECT_STREQ (result, "src ! parser ! conv ! fmt ! conv ! fmt ! conv ! sink");

  g_free (result);
}

/**
 * @brief Test to replace string.
 */
TEST (common_string_util, replace_str_02)
{
  gchar *result;
  guint changed;

  result = g_strdup ("source! parser ! sources ! mysource ! source ! format !source! conv source");

  result = replace_string (result, "source", "src", " !", &changed);
  EXPECT_EQ (changed, 4U);
  EXPECT_STREQ (result, "src! parser ! sources ! mysource ! src ! format !src! conv src");

  result = replace_string (result, "src", "mysource", "! ", &changed);
  EXPECT_EQ (changed, 4U);
  EXPECT_STREQ (result, "mysource! parser ! sources ! mysource ! mysource ! format !mysource! conv mysource");

  result = replace_string (result, "source", "src", NULL, &changed);
  EXPECT_EQ (changed, 6U);
  EXPECT_STREQ (result, "mysrc! parser ! srcs ! mysrc ! mysrc ! format !mysrc! conv mysrc");

  result = replace_string (result, "mysrc", "src", ";", &changed);
  EXPECT_EQ (changed, 0U);
  EXPECT_STREQ (result, "mysrc! parser ! srcs ! mysrc ! mysrc ! format !mysrc! conv mysrc");

  g_free (result);
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
    g_free (fullpath);
    return NULL;
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
TEST (conf_custom, env_str_01)
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

  EXPECT_TRUE (FALSE != g_setenv ("NNSTREAMER_CONF", filename, TRUE));
  EXPECT_TRUE (nnsconf_loadconf (TRUE) == TRUE);

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

  if (confenv) {
    EXPECT_TRUE (g_setenv ("NNSTREAMER_CONF", confenv, TRUE));
    g_free (confenv);
  } else {
    g_unsetenv ("NNSTREAMER_CONF");
  }
}

/**
 * @brief Test version control (positive)
 */
TEST (version_control, get_ver_01)
{
  gchar *verstr = nnstreamer_version_string ();
  guint major, minor, micro;
  gchar *verstr2, *verstr3;
  nnstreamer_version_fetch (&major, &minor, &micro);

  verstr2 = g_strdup_printf ("NNStreamer %u.%u.%u", major, minor, micro);
  verstr3 = g_strdup_printf ("%u.%u.%u", major, minor, micro);

  EXPECT_STRCASEEQ (verstr, verstr2);

  EXPECT_STRCASEEQ (VERSION, verstr3);

  EXPECT_EQ ((int)major, NNSTREAMER_VERSION_MAJOR);
  EXPECT_EQ ((int)minor, NNSTREAMER_VERSION_MINOR);
  EXPECT_EQ ((int)micro, NNSTREAMER_VERSION_MICRO);

  g_free (verstr);
  g_free (verstr2);
  g_free (verstr3);
}

/**
 * @brief Test pad has tensor cap
 */
TEST (common_pad_cap, tensor_0)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GstElement *src_handle;
  GstPad *pad;
  gboolean is_tensor;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "fakesrc name=fsrc ! fakesink name=fsink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  src_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "fsrc");
  EXPECT_NE (src_handle, nullptr);
  pad = gst_element_get_static_pad (src_handle, "src");
  EXPECT_NE (pad, nullptr);

  /* "any" cap returns tensor cap */
  is_tensor = gst_pad_has_tensor_caps (pad);
  EXPECT_EQ (TRUE, is_tensor);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  gst_object_unref (pad);
  gst_object_unref (src_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
}

/**
 * @brief Test pad has tensor cap
 */
TEST (common_pad_cap, tensors_0)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GstElement *src_handle;
  GstPad *pad;
  gboolean is_tensor;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "videotestsrc ! videoconvert ! video/x-raw,width=160,height=120,format=RGB,framerate=(fraction)30/1 ! "
      "tensor_converter name=tsrc ! other/tensors,num_tensors=1,dimensions=(string)3:160:120:1, types=(string)uint8, framerate=(fraction)30/1 ! fakesink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  src_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "tsrc");
  EXPECT_NE (src_handle, nullptr);
  pad = gst_element_get_static_pad (src_handle, "src");
  EXPECT_NE (pad, nullptr);

  is_tensor = gst_pad_has_tensor_caps (pad);
  EXPECT_EQ (FALSE, is_tensor);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  gst_object_unref (pad);
  gst_object_unref (src_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
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
