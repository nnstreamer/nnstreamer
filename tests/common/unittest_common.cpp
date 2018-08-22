/**
 * @file unittest_common.cpp
 * @date 31 May 2018
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @brief Unit test module for NNStreamer common library
 * @see		https://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
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
  EXPECT_EQ (get_tensor_type ("int32"), _NNS_INT32);
  EXPECT_EQ (get_tensor_type ("INT32"), _NNS_INT32);
  EXPECT_EQ (get_tensor_type ("iNt32"), _NNS_INT32);
  EXPECT_EQ (get_tensor_type ("InT32"), _NNS_INT32);
  EXPECT_EQ (get_tensor_type ("InT322"), _NNS_END);
  EXPECT_EQ (get_tensor_type ("int3"), _NNS_END);
}

/**
 * @brief Test for int16 type string.
 */
TEST (common_get_tensor_type, int16)
{
  EXPECT_EQ (get_tensor_type ("int16"), _NNS_INT16);
  EXPECT_EQ (get_tensor_type ("INT16"), _NNS_INT16);
  EXPECT_EQ (get_tensor_type ("iNt16"), _NNS_INT16);
  EXPECT_EQ (get_tensor_type ("InT16"), _NNS_INT16);
  EXPECT_EQ (get_tensor_type ("InT162"), _NNS_END);
  EXPECT_EQ (get_tensor_type ("int1"), _NNS_END);
}

/**
 * @brief Test for int8 type string.
 */
TEST (common_get_tensor_type, int8)
{
  EXPECT_EQ (get_tensor_type ("int8"), _NNS_INT8);
  EXPECT_EQ (get_tensor_type ("INT8"), _NNS_INT8);
  EXPECT_EQ (get_tensor_type ("iNt8"), _NNS_INT8);
  EXPECT_EQ (get_tensor_type ("InT8"), _NNS_INT8);
  EXPECT_EQ (get_tensor_type ("InT82"), _NNS_END);
  EXPECT_EQ (get_tensor_type ("int3"), _NNS_END);
}

/**
 * @brief Test for uint32 type string.
 */
TEST (common_get_tensor_type, uint32)
{
  EXPECT_EQ (get_tensor_type ("uint32"), _NNS_UINT32);
  EXPECT_EQ (get_tensor_type ("UINT32"), _NNS_UINT32);
  EXPECT_EQ (get_tensor_type ("uiNt32"), _NNS_UINT32);
  EXPECT_EQ (get_tensor_type ("UInT32"), _NNS_UINT32);
  EXPECT_EQ (get_tensor_type ("UInT322"), _NNS_END);
  EXPECT_EQ (get_tensor_type ("uint3"), _NNS_END);
}

/**
 * @brief Test for uint16 type string.
 */
TEST (common_get_tensor_type, uint16)
{
  EXPECT_EQ (get_tensor_type ("uint16"), _NNS_UINT16);
  EXPECT_EQ (get_tensor_type ("UINT16"), _NNS_UINT16);
  EXPECT_EQ (get_tensor_type ("uiNt16"), _NNS_UINT16);
  EXPECT_EQ (get_tensor_type ("UInT16"), _NNS_UINT16);
  EXPECT_EQ (get_tensor_type ("UInT162"), _NNS_END);
  EXPECT_EQ (get_tensor_type ("uint1"), _NNS_END);
}

/**
 * @brief Test for uint8 type string.
 */
TEST (common_get_tensor_type, uint8)
{
  EXPECT_EQ (get_tensor_type ("uint8"), _NNS_UINT8);
  EXPECT_EQ (get_tensor_type ("UINT8"), _NNS_UINT8);
  EXPECT_EQ (get_tensor_type ("uiNt8"), _NNS_UINT8);
  EXPECT_EQ (get_tensor_type ("UInT8"), _NNS_UINT8);
  EXPECT_EQ (get_tensor_type ("UInT82"), _NNS_END);
  EXPECT_EQ (get_tensor_type ("uint3"), _NNS_END);
}

/**
 * @brief Test for float32 type string.
 */
TEST (common_get_tensor_type, float32)
{
  EXPECT_EQ (get_tensor_type ("float32"), _NNS_FLOAT32);
  EXPECT_EQ (get_tensor_type ("FLOAT32"), _NNS_FLOAT32);
  EXPECT_EQ (get_tensor_type ("float32"), _NNS_FLOAT32);
  EXPECT_EQ (get_tensor_type ("FloaT32"), _NNS_FLOAT32);
  EXPECT_EQ (get_tensor_type ("FloaT322"), _NNS_END);
  EXPECT_EQ (get_tensor_type ("float3"), _NNS_END);
}

/**
 * @brief Test for float64 type string.
 */
TEST (common_get_tensor_type, float64)
{
  EXPECT_EQ (get_tensor_type ("float64"), _NNS_FLOAT64);
  EXPECT_EQ (get_tensor_type ("FLOAT64"), _NNS_FLOAT64);
  EXPECT_EQ (get_tensor_type ("float64"), _NNS_FLOAT64);
  EXPECT_EQ (get_tensor_type ("FloaT64"), _NNS_FLOAT64);
  EXPECT_EQ (get_tensor_type ("FloaT642"), _NNS_END);
  EXPECT_EQ (get_tensor_type ("float6"), _NNS_END);
}

/**
 * @brief Test for int64 type string.
 */
TEST (common_get_tensor_type, int64)
{
  EXPECT_EQ (get_tensor_type ("int64"), _NNS_INT64);
  EXPECT_EQ (get_tensor_type ("INT64"), _NNS_INT64);
  EXPECT_EQ (get_tensor_type ("iNt64"), _NNS_INT64);
  EXPECT_EQ (get_tensor_type ("InT64"), _NNS_INT64);
  EXPECT_EQ (get_tensor_type ("InT642"), _NNS_END);
  EXPECT_EQ (get_tensor_type ("int6"), _NNS_END);
}

/**
 * @brief Test for uint64 type string.
 */
TEST (common_get_tensor_type, uint64)
{
  EXPECT_EQ (get_tensor_type ("uint64"), _NNS_UINT64);
  EXPECT_EQ (get_tensor_type ("UINT64"), _NNS_UINT64);
  EXPECT_EQ (get_tensor_type ("uiNt64"), _NNS_UINT64);
  EXPECT_EQ (get_tensor_type ("UInT64"), _NNS_UINT64);
  EXPECT_EQ (get_tensor_type ("UInT642"), _NNS_END);
  EXPECT_EQ (get_tensor_type ("uint6"), _NNS_END);
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
  uint32_t dim[NNS_TENSOR_SIZE_LIMIT][NNS_TENSOR_RANK_LIMIT];
  int rank[NNS_TENSOR_RANK_LIMIT];
  int num_tensors = get_tensor_dimension ("345:123:433:177", dim, rank);
  EXPECT_EQ (num_tensors, 1);
  EXPECT_EQ (rank[0], 4);
  EXPECT_EQ (dim[0][0], 345);
  EXPECT_EQ (dim[0][1], 123);
  EXPECT_EQ (dim[0][2], 433);
  EXPECT_EQ (dim[0][3], 177);
}

/**
 * @brief Test for tensor dimension.
 */
TEST (common_get_tensor_dimension, case2)
{
  uint32_t dim[NNS_TENSOR_SIZE_LIMIT][NNS_TENSOR_RANK_LIMIT];
  int rank[NNS_TENSOR_RANK_LIMIT];
  int num_tensors = get_tensor_dimension ("345:123:433", dim, rank);
  EXPECT_EQ (num_tensors, 1);
  EXPECT_EQ (rank[0], 3);
  EXPECT_EQ (dim[0][0], 345);
  EXPECT_EQ (dim[0][1], 123);
  EXPECT_EQ (dim[0][2], 433);
  EXPECT_EQ (dim[0][3], 1);
}

/**
 * @brief Test for tensor dimension.
 */
TEST (common_get_tensor_dimension, case3)
{
  uint32_t dim[NNS_TENSOR_SIZE_LIMIT][NNS_TENSOR_RANK_LIMIT];
  int rank[NNS_TENSOR_RANK_LIMIT];
  int num_tensors = get_tensor_dimension ("345:123", dim, rank);
  EXPECT_EQ (num_tensors, 1);
  EXPECT_EQ (rank[0], 2);
  EXPECT_EQ (dim[0][0], 345);
  EXPECT_EQ (dim[0][1], 123);
  EXPECT_EQ (dim[0][2], 1);
  EXPECT_EQ (dim[0][3], 1);
}

/**
 * @brief Test for tensor dimension.
 */
TEST (common_get_tensor_dimension, case4)
{
  uint32_t dim[NNS_TENSOR_SIZE_LIMIT][NNS_TENSOR_RANK_LIMIT];
  int rank[NNS_TENSOR_RANK_LIMIT];
  int num_tensors = get_tensor_dimension ("345", dim, rank);
  EXPECT_EQ (num_tensors, 1);
  EXPECT_EQ (rank[0], 1);
  EXPECT_EQ (dim[0][0], 345);
  EXPECT_EQ (dim[0][1], 1);
  EXPECT_EQ (dim[0][2], 1);
  EXPECT_EQ (dim[0][3], 1);
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
