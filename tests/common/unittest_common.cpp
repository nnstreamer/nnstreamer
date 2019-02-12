/**
 * @file unittest_common.cpp
 * @date 31 May 2018
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @brief Unit test module for NNStreamer common library
 * @see		https://github.com/nnsuite/nnstreamer
 * @bug		No known bugs.
 *
 *  @brief Unit test module for NNStreamer common library
 *  @bug	No known bugs except for NYI items
 *
 *  Copyright 2018 Samsung Electronics
 *
 */

#include <gtest/gtest.h>
#include <unistd.h>
#include <tensor_common.h>

/**
 * @brief Test for int32 type string.
 */
TEST (common_get_tensor_type, int32)
{
  EXPECT_EQ (gst_tensor_get_type ("int32"), _NNS_INT32);
  EXPECT_EQ (gst_tensor_get_type ("INT32"), _NNS_INT32);
  EXPECT_EQ (gst_tensor_get_type ("iNt32"), _NNS_INT32);
  EXPECT_EQ (gst_tensor_get_type ("InT32"), _NNS_INT32);
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
  EXPECT_EQ (gst_tensor_get_type ("UInT642"), _NNS_END);
  EXPECT_EQ (gst_tensor_get_type ("uint6"), _NNS_END);
}

/**
 * @brief Test to find index of the key.
 */
TEST (common_find_key_strv, key_index)
{
  const gchar *teststrv[] = {
    "abcde",
    "ABCDEF",
    "1234",
    "abcabc",
    "tester",
    NULL
  };

  EXPECT_EQ (find_key_strv (teststrv, "abcde"), 0);
  EXPECT_EQ (find_key_strv (teststrv, "ABCDE"), 0);
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
  EXPECT_EQ (rank, 4);
  EXPECT_EQ (dim[0], 345);
  EXPECT_EQ (dim[1], 123);
  EXPECT_EQ (dim[2], 433);
  EXPECT_EQ (dim[3], 177);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_TRUE (g_str_equal (dim_str, "345:123:433:177"));
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
  EXPECT_EQ (rank, 3);
  EXPECT_EQ (dim[0], 345);
  EXPECT_EQ (dim[1], 123);
  EXPECT_EQ (dim[2], 433);
  EXPECT_EQ (dim[3], 1);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_TRUE (g_str_equal (dim_str, "345:123:433:1"));
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
  EXPECT_EQ (rank, 2);
  EXPECT_EQ (dim[0], 345);
  EXPECT_EQ (dim[1], 123);
  EXPECT_EQ (dim[2], 1);
  EXPECT_EQ (dim[3], 1);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_TRUE (g_str_equal (dim_str, "345:123:1:1"));
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
  EXPECT_EQ (rank, 1);
  EXPECT_EQ (dim[0], 345);
  EXPECT_EQ (dim[1], 1);
  EXPECT_EQ (dim[2], 1);
  EXPECT_EQ (dim[3], 1);

  dim_str = gst_tensor_get_dimension_string (dim);
  EXPECT_TRUE (g_str_equal (dim_str, "345:1:1:1"));
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
  EXPECT_TRUE (g_str_equal (dest.name, test_name));
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
    EXPECT_TRUE (g_str_equal (dest.info[i].name, test_name));
    EXPECT_EQ (dest.info[i].type, src.info[i].type);
    EXPECT_EQ (dest.info[i].dimension[0], src.info[i].dimension[0]);
    EXPECT_EQ (dest.info[i].dimension[1], src.info[i].dimension[1]);
    EXPECT_EQ (dest.info[i].dimension[2], src.info[i].dimension[2]);
    EXPECT_EQ (dest.info[i].dimension[3], src.info[i].dimension[3]);
  }

  g_free (test_name);
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
  EXPECT_EQ (num_dims, 1);

  info.num_tensors = num_dims;

  str_dims = gst_tensors_info_get_dimensions_string (&info);
  EXPECT_TRUE (g_str_equal (str_dims, "1:2:3:4"));
  g_free (str_dims);

  /* 4 tensors info */
  num_dims = gst_tensors_info_parse_dimensions_string (&info, "1, 2, 3, 4");
  EXPECT_EQ (num_dims, 4);

  info.num_tensors = num_dims;

  str_dims = gst_tensors_info_get_dimensions_string (&info);
  EXPECT_TRUE (g_str_equal (str_dims, "1:1:1:1,2:1:1:1,3:1:1:1,4:1:1:1"));
  g_free (str_dims);

  /* max */
  num_dims = gst_tensors_info_parse_dimensions_string (&info,
      "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20");
  EXPECT_EQ (num_dims, NNS_TENSOR_SIZE_LIMIT);
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
  EXPECT_EQ (num_types, 1);

  info.num_tensors = num_types;

  str_types = gst_tensors_info_get_types_string (&info);
  EXPECT_TRUE (g_str_equal (str_types, "uint16"));
  g_free (str_types);

  /* 4 tensors info */
  num_types = gst_tensors_info_parse_types_string (&info,
      "int8, int16, int32, int64");
  EXPECT_EQ (num_types, 4);

  info.num_tensors = num_types;

  str_types = gst_tensors_info_get_types_string (&info);
  EXPECT_TRUE (g_str_equal (str_types, "int8,int16,int32,int64"));
  g_free (str_types);

  /* max */
  num_types = gst_tensors_info_parse_types_string (&info,
      "int8, int8, int8, int8, int8, int8, int8, int8, int8, int8, int8, "
      "int8, int8, int8, int8, int8, int8, int8, int8, int8, int8, int8");
  EXPECT_EQ (num_types, NNS_TENSOR_SIZE_LIMIT);
}

/**
 * @brief Test for names string in tensors info.
 */
TEST (common_tensors_info_string, names)
{
  GstTensorsInfo info;
  guint num_names;
  gchar *str_names;

  gst_tensors_info_init (&info);

  /* 1 tensor info */
  num_names = gst_tensors_info_parse_names_string (&info, "t1");
  EXPECT_EQ (num_names, 1);

  info.num_tensors = num_names;

  str_names = gst_tensors_info_get_names_string (&info);
  EXPECT_TRUE (g_str_equal (str_names, "t1"));
  g_free (str_names);

  /* 4 tensors info */
  num_names = gst_tensors_info_parse_names_string (&info,
      "tensor1, tensor2, tensor3, tensor4");
  EXPECT_EQ (num_names, 4);

  info.num_tensors = num_names;

  str_names = gst_tensors_info_get_names_string (&info);
  EXPECT_TRUE (g_str_equal (str_names, "tensor1,tensor2,tensor3,tensor4"));
  g_free (str_names);

  /* max */
  num_names = gst_tensors_info_parse_names_string (&info,
      "t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, "
      "t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28");
  EXPECT_EQ (num_names, NNS_TENSOR_SIZE_LIMIT);
}

/**
 * @brief Main function for unit test.
 */
int
main (int argc, char **argv)
{
  testing::InitGoogleTest (&argc, argv);
  return RUN_ALL_TESTS ();
}
