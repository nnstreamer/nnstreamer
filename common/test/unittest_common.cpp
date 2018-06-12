/**
 *  @file unittest_common.cpp
 *  @date 31 May 2018
 *  @author MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 *  @brief Unit test module for NNStreamer common library
 *
 *  Copyright 2018 Samsung Electronics
 *
 */

#include <gtest/gtest.h>
#include <unistd.h>
#include <tensor_common.h>

TEST (common_get_tensor_type, int32_1)
{
  EXPECT_EQ (get_tensor_type ("int32"), _NNS_INT32);
}

TEST (common_get_tensor_type, int32_2)
{
  EXPECT_EQ (get_tensor_type ("INT32"), _NNS_INT32);
}

TEST (common_get_tensor_type, int32_3)
{
  EXPECT_EQ (get_tensor_type ("iNt32"), _NNS_INT32);
}

TEST (common_get_tensor_type, int32_4)
{
  EXPECT_EQ (get_tensor_type ("InT32"), _NNS_INT32);
}

TEST (common_get_tensor_type, int32_5)
{
  EXPECT_EQ (get_tensor_type ("InT322"), _NNS_END);
}

TEST (common_get_tensor_type, int32_6)
{
  EXPECT_EQ (get_tensor_type ("int3"), _NNS_END);
}


TEST (common_get_tensor_type, int16_1)
{
  EXPECT_EQ (get_tensor_type ("int16"), _NNS_INT16);
}

TEST (common_get_tensor_type, int16_2)
{
  EXPECT_EQ (get_tensor_type ("INT16"), _NNS_INT16);
}

TEST (common_get_tensor_type, int16_3)
{
  EXPECT_EQ (get_tensor_type ("iNt16"), _NNS_INT16);
}

TEST (common_get_tensor_type, int16_4)
{
  EXPECT_EQ (get_tensor_type ("InT16"), _NNS_INT16);
}

TEST (common_get_tensor_type, int16_5)
{
  EXPECT_EQ (get_tensor_type ("InT162"), _NNS_END);
}

TEST (common_get_tensor_type, int16_6)
{
  EXPECT_EQ (get_tensor_type ("int1"), _NNS_END);
}


TEST (common_get_tensor_type, int8_1)
{
  EXPECT_EQ (get_tensor_type ("int8"), _NNS_INT8);
}

TEST (common_get_tensor_type, int8_2)
{
  EXPECT_EQ (get_tensor_type ("INT8"), _NNS_INT8);
}

TEST (common_get_tensor_type, int8_3)
{
  EXPECT_EQ (get_tensor_type ("iNt8"), _NNS_INT8);
}

TEST (common_get_tensor_type, int8_4)
{
  EXPECT_EQ (get_tensor_type ("InT8"), _NNS_INT8);
}

TEST (common_get_tensor_type, int8_5)
{
  EXPECT_EQ (get_tensor_type ("InT82"), _NNS_END);
}

TEST (common_get_tensor_type, int8_6)
{
  EXPECT_EQ (get_tensor_type ("int3"), _NNS_END);
}


TEST (common_get_tensor_type, uint32_1)
{
  EXPECT_EQ (get_tensor_type ("uint32"), _NNS_UINT32);
}

TEST (common_get_tensor_type, uint32_2)
{
  EXPECT_EQ (get_tensor_type ("UINT32"), _NNS_UINT32);
}

TEST (common_get_tensor_type, uint32_3)
{
  EXPECT_EQ (get_tensor_type ("uiNt32"), _NNS_UINT32);
}

TEST (common_get_tensor_type, uint32_4)
{
  EXPECT_EQ (get_tensor_type ("UInT32"), _NNS_UINT32);
}

TEST (common_get_tensor_type, uint32_5)
{
  EXPECT_EQ (get_tensor_type ("UInT322"), _NNS_END);
}

TEST (common_get_tensor_type, uint32_6)
{
  EXPECT_EQ (get_tensor_type ("uint3"), _NNS_END);
}


TEST (common_get_tensor_type, uint16_1)
{
  EXPECT_EQ (get_tensor_type ("uint16"), _NNS_UINT16);
}

TEST (common_get_tensor_type, uint16_2)
{
  EXPECT_EQ (get_tensor_type ("UINT16"), _NNS_UINT16);
}

TEST (common_get_tensor_type, uint16_3)
{
  EXPECT_EQ (get_tensor_type ("uiNt16"), _NNS_UINT16);
}

TEST (common_get_tensor_type, uint16_4)
{
  EXPECT_EQ (get_tensor_type ("UInT16"), _NNS_UINT16);
}

TEST (common_get_tensor_type, uint16_5)
{
  EXPECT_EQ (get_tensor_type ("UInT162"), _NNS_END);
}

TEST (common_get_tensor_type, uint16_6)
{
  EXPECT_EQ (get_tensor_type ("uint1"), _NNS_END);
}


TEST (common_get_tensor_type, uint8_1)
{
  EXPECT_EQ (get_tensor_type ("uint8"), _NNS_UINT8);
}

TEST (common_get_tensor_type, uint8_2)
{
  EXPECT_EQ (get_tensor_type ("UINT8"), _NNS_UINT8);
}

TEST (common_get_tensor_type, uint8_3)
{
  EXPECT_EQ (get_tensor_type ("uiNt8"), _NNS_UINT8);
}

TEST (common_get_tensor_type, uint8_4)
{
  EXPECT_EQ (get_tensor_type ("UInT8"), _NNS_UINT8);
}

TEST (common_get_tensor_type, uint8_5)
{
  EXPECT_EQ (get_tensor_type ("UInT82"), _NNS_END);
}

TEST (common_get_tensor_type, uint8_6)
{
  EXPECT_EQ (get_tensor_type ("uint3"), _NNS_END);
}


TEST (common_get_tensor_type, float32_1)
{
  EXPECT_EQ (get_tensor_type ("float32"), _NNS_FLOAT32);
}

TEST (common_get_tensor_type, float32_2)
{
  EXPECT_EQ (get_tensor_type ("FLOAT32"), _NNS_FLOAT32);
}

TEST (common_get_tensor_type, float32_3)
{
  EXPECT_EQ (get_tensor_type ("float32"), _NNS_FLOAT32);
}

TEST (common_get_tensor_type, float32_4)
{
  EXPECT_EQ (get_tensor_type ("FloaT32"), _NNS_FLOAT32);
}

TEST (common_get_tensor_type, float32_5)
{
  EXPECT_EQ (get_tensor_type ("FloaT322"), _NNS_END);
}

TEST (common_get_tensor_type, float32_6)
{
  EXPECT_EQ (get_tensor_type ("float3"), _NNS_END);
}



TEST (common_get_tensor_type, float64_1)
{
  EXPECT_EQ (get_tensor_type ("float64"), _NNS_FLOAT64);
}

TEST (common_get_tensor_type, float64_2)
{
  EXPECT_EQ (get_tensor_type ("FLOAT64"), _NNS_FLOAT64);
}

TEST (common_get_tensor_type, float64_3)
{
  EXPECT_EQ (get_tensor_type ("float64"), _NNS_FLOAT64);
}

TEST (common_get_tensor_type, float64_4)
{
  EXPECT_EQ (get_tensor_type ("FloaT64"), _NNS_FLOAT64);
}

TEST (common_get_tensor_type, float64_5)
{
  EXPECT_EQ (get_tensor_type ("FloaT642"), _NNS_END);
}

TEST (common_get_tensor_type, float64_6)
{
  EXPECT_EQ (get_tensor_type ("float6"), _NNS_END);
}


static const gchar *teststrv[] = {
  "abcde",
  "ABCDEF",
  "1234",
  "abcabc",
  "tester",
  NULL
};

TEST (common_find_key_strv, case1)
{
  EXPECT_EQ (find_key_strv (teststrv, "abcde"), 0);
}

TEST (common_find_key_strv, case2)
{
  EXPECT_EQ (find_key_strv (teststrv, "ABCDE"), 0);
}

TEST (common_find_key_strv, case3)
{
  EXPECT_EQ (find_key_strv (teststrv, "1234"), 2);
}

TEST (common_find_key_strv, case4)
{
  EXPECT_EQ (find_key_strv (teststrv, "tester"), 4);
}

TEST (common_find_key_strv, case5)
{
  EXPECT_EQ (find_key_strv (teststrv, "abcabcd"), -1);
}


TEST (common_get_tensor_dimension, case1)
{
  uint32_t dim[NNS_TENSOR_RANK_LIMIT];
  int rank = get_tensor_dimension ("345:123:433:177", dim);
  EXPECT_EQ (rank, 4);
  EXPECT_EQ (dim[0], 345);
  EXPECT_EQ (dim[1], 123);
  EXPECT_EQ (dim[2], 433);
  EXPECT_EQ (dim[3], 177);
}


TEST (common_get_tensor_dimension, case2)
{
  uint32_t dim[NNS_TENSOR_RANK_LIMIT];
  int rank = get_tensor_dimension ("345:123:433", dim);
  EXPECT_EQ (rank, 3);
  EXPECT_EQ (dim[0], 345);
  EXPECT_EQ (dim[1], 123);
  EXPECT_EQ (dim[2], 433);
  EXPECT_EQ (dim[3], 1);
}

TEST (common_get_tensor_dimension, case3)
{
  uint32_t dim[NNS_TENSOR_RANK_LIMIT];
  int rank = get_tensor_dimension ("345:123", dim);
  EXPECT_EQ (rank, 2);
  EXPECT_EQ (dim[0], 345);
  EXPECT_EQ (dim[1], 123);
  EXPECT_EQ (dim[2], 1);
  EXPECT_EQ (dim[3], 1);
}

TEST (common_get_tensor_dimension, case4)
{
  uint32_t dim[NNS_TENSOR_RANK_LIMIT];
  int rank = get_tensor_dimension ("345", dim);
  EXPECT_EQ (rank, 1);
  EXPECT_EQ (dim[0], 345);
  EXPECT_EQ (dim[1], 1);
  EXPECT_EQ (dim[2], 1);
  EXPECT_EQ (dim[3], 1);
}

int
main (int argc, char **argv)
{
  testing::InitGoogleTest (&argc, argv);
  return RUN_ALL_TESTS ();
}
