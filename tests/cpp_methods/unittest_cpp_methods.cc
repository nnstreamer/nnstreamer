/**
 * @file        unittest_cpp_methods.cc
 * @date        15 Jan 2019
 * @brief       Unit test cases for tensor_filter::cpp
 * @see         https://github.com/nnsuite/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <tensor_common.h>
#include <nnstreamer.h>
#include <nnstreamer-single.h>
#include <nnstreamer-capi-private.h>

#include <tensor_filter_cpp.h>

class filter_basic: public tensor_filter_cpp {
  public:
    filter_basic(const char *str): tensor_filter_cpp(str) {}
    ~filter_basic() { }
    int getInputDim(GstTensorsInfo *info) {
      info->num_tensors = 1;
      info->info[0].type = _NNS_UINT8;
      info->info[0].dimension[0] = 4;
      info->info[0].dimension[1] = 1;
      info->info[0].dimension[2] = 1;
      info->info[0].dimension[3] = 1;
      return 0;
    }
    int getOutputDim(GstTensorsInfo *info) {
      info->num_tensors = 1;
      info->info[0].type = _NNS_UINT8;
      info->info[0].dimension[0] = 4;
      info->info[0].dimension[1] = 2;
      info->info[0].dimension[2] = 1;
      info->info[0].dimension[3] = 1;
      return 0;
    }
    int setInputDim(const GstTensorsInfo *in, GstTensorsInfo *out) {
      return -EINVAL;
    }
    bool isAllocatedBeforeInvoke() {
      return true;
    }
    int invoke(const GstTensorMemory *in, GstTensorMemory *out) {
      EXPECT_TRUE (in);
      EXPECT_TRUE (out);

      EXPECT_EQ (prop->input_meta.info[0].dimension[0], 4U);
      EXPECT_EQ (prop->input_meta.info[0].dimension[1], 1U);
      EXPECT_EQ (prop->input_meta.info[0].dimension[2], 1U);
      EXPECT_EQ (prop->input_meta.info[0].dimension[3], 1U);
      EXPECT_EQ (prop->output_meta.info[0].dimension[0], 4U);
      EXPECT_EQ (prop->output_meta.info[0].dimension[1], 2U);
      EXPECT_EQ (prop->output_meta.info[0].dimension[2], 1U);
      EXPECT_EQ (prop->output_meta.info[0].dimension[3], 1U);
      EXPECT_EQ (prop->input_meta.info[0].type, _NNS_UINT8);
      EXPECT_EQ (prop->output_meta.info[0].type, _NNS_UINT8);

      for (int i = 0; i < 4; i++) {
        *((uint8_t *) out[0].data + i) = *((uint8_t *) in[0].data + i) * 2;
        *((uint8_t *) out[0].data + i + 4) = *((uint8_t *) in[0].data + i) + 1;
      }
      return 0;
    }
};


/** @brief Positive case for the simpliest execution path */
TEST (cpp_filter_on_demand, basic_01)
{
  filter_basic basic("basic_01");
  basic._register();
}

/**
 * @brief Main GTest
 */
int main (int argc, char **argv)
{
  int result;

  testing::InitGoogleTest (&argc, argv);

  /* ignore tizen feature status while running the testcases */
  set_feature_state (1);

  result = RUN_ALL_TESTS ();

  set_feature_state (-1);

  return result;
}
