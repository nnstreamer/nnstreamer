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
  EXPECT_STREQ (dim_str, "345:123:433:177");
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
  EXPECT_EQ (dim[3], 1U);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_STREQ (dim_str, "345:123:433:1");
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
  EXPECT_EQ (dim[2], 1U);
  EXPECT_EQ (dim[3], 1U);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_STREQ (dim_str, "345:123:1:1");
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

  gst_tensors_info_free (&dest);
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
TEST (commonTensorInfo, size01_p)
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
TEST (commonTensorInfo, size02_n)
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
TEST (commonTensorInfo, size03_n)
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
TEST (commonTensorInfo, equal01_p)
{
  GstTensorsInfo info1, info2;

  fill_tensors_info_for_test (&info1, &info2);

  EXPECT_TRUE (gst_tensors_info_is_equal (&info1, &info2));
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
  EXPECT_EQ (0, gst_tensor_info_get_rank (NULL));
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
 * @brief Test for same tensors config.
 */
TEST (commonTensorsConfig, equal01_p)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);

  EXPECT_TRUE (gst_tensors_config_is_equal (&conf1, &conf2));
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
  conf1.format = conf2.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  EXPECT_TRUE (gst_tensors_config_is_equal (&conf1, &conf2));
}

/**
 * @brief Test for same tensors config.
 */
TEST (commonTensorsConfig, equal10_n)
{
  GstTensorsConfig conf1, conf2;

  fill_tensors_config_for_test (&conf1, &conf2);

  /* change format, this should not be compatible */
  conf2.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

  EXPECT_FALSE (gst_tensors_config_is_equal (&conf1, &conf2));
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

  EXPECT_FALSE (gst_tensors_config_validate (NULL));
}

/**
 * @brief Test for validating of the tensors config with invalid param.
 */
TEST (commonTensorsConfig, validateInvalidParam2_n)
{
  GstTensorsConfig conf;
  gst_tensors_config_init (&conf);
  conf.rate_d = 1;
  EXPECT_FALSE (gst_tensors_config_validate (NULL));
}

/**
 * @brief Test for getting config from strucrure with invalid param.
 */
TEST (commonTensorsConfig, fromStructreInvalidParam0_n)
{
  GstStructure structure;

  EXPECT_FALSE (gst_tensors_config_from_structure (NULL, &structure));
}

/**
 * @brief Test for getting config from strucrure with invalid param.
 */
TEST (commonTensorsConfig, fromStructreInvalidParam1_n)
{
  GstTensorsConfig conf;
  gst_tensors_config_init (&conf);
  EXPECT_FALSE (gst_tensors_config_from_structure (&conf, NULL));
}

/**
 * @brief Test for getting tensor cap with invalid param.
 */
TEST (commonTensorConfig, capInvalidParam0_n)
{
  EXPECT_FALSE (gst_tensor_caps_from_config (NULL));
}

/**
 * @brief Test for getting tensor cap with invalid param.
 */
TEST (commonTensorsConfig, capInvalidParam1_n)
{
  EXPECT_FALSE (gst_tensors_caps_from_config (NULL));
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

  /* max */
  num_types = gst_tensors_info_parse_types_string (&info,
      "int8, int8, int8, int8, int8, int8, int8, int8, int8, int8, int8, "
      "int8, int8, int8, int8, int8, int8, int8, int8, int8, int8, int8");
  EXPECT_EQ (num_types, (guint)NNS_TENSOR_SIZE_LIMIT);
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

  /* max */
  num_names = gst_tensors_info_parse_names_string (&info,
      "t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, "
      "t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28");
  EXPECT_EQ (num_names, (guint)NNS_TENSOR_SIZE_LIMIT);
  info.num_tensors = num_names;
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
  for (i = 0; i < NNS_TENSOR_META_RANK_LIMIT; i++)
    EXPECT_EQ (meta.dimension[i], 0U);

  /* current version after init */
  gst_tensor_meta_info_get_version (&meta, &major, &minor);
  EXPECT_TRUE (major > 0 || minor > 0);
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
 * @brief Test for tensor meta info (header size with invalid meta).
 */
TEST (commonMetaInfo, headerSizeInvalidParam02_n)
{
  GstTensorMetaInfo meta = {0, };
  gsize hsize;

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
 * @brief Test for tensor meta info (data size with invalid meta).
 */
TEST (commonMetaInfo, dataSizeInvalidParam02_n)
{
  GstTensorMetaInfo meta = {0, };
  gsize dsize;

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
 * @brief Test for tensor meta info (validate meta with invalid meta).
 */
TEST (commonMetaInfo, validateInvalidParam02_n)
{
  GstTensorMetaInfo meta = { 0, };
  gboolean valid;

  /* invalid version */
  valid = gst_tensor_meta_info_validate (&meta);
  EXPECT_FALSE (valid);

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
 * @brief Test for tensor meta info (dimension rank mismatched).
 */
TEST (commonMetaInfo, convertMetaInvalidParam03_n)
{
  GstTensorMetaInfo meta;
  GstTensorInfo info;
  guint i;
  gboolean ret;

  gst_tensor_meta_info_init (&meta);
  meta.type = _NNS_UINT16;
  meta.format = _NNS_TENSOR_FORMAT_STATIC;
  /* rank > NNS_TENSOR_RANK_LIMIT */
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT + 3; i++)
    meta.dimension[i] = 2;

  ret = gst_tensor_meta_info_convert (&meta, &info);
  EXPECT_FALSE (ret);
}

/**
 * @brief Test to replace string.
 */
TEST (commonStringUtil, replaceStr01)
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
TEST (commonStringUtil, replaceStr02)
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

  EXPECT_EQ ((int)major, NNSTREAMER_VERSION_MAJOR);
  EXPECT_EQ ((int)minor, NNSTREAMER_VERSION_MINOR);
  EXPECT_EQ ((int)micro, NNSTREAMER_VERSION_MICRO);

  g_free (verstr);
  g_free (verstr2);
  g_free (verstr3);
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
