/**
 * @file        cppfilter_test.hh
 * @date        15 Jan 2019
 * @brief       Unit test cases for tensor_filter::cpp
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include <glib.h>
#include <tensor_filter_cpp.hh>

/** @brief Test C++ filter: uint8 3:4:4:1 in, uint8 3:4:4:2 out */
class filter_basic : public tensor_filter_cpp
{
  public:
  /** @brief Construct the test filter with the given filter name */
  filter_basic (const char *str);
  /** @brief Destructor of the test filter */
  ~filter_basic ();

  /** @brief Report the fixed input dimension, uint8 3:4:4:1 */
  int getInputDim (GstTensorsInfo *info);
  /** @brief Report the fixed output dimension, uint8 3:4:4:2 */
  int getOutputDim (GstTensorsInfo *info);
  /** @brief Reject dimension negotiation; the dimensions are fixed */
  int setInputDim (const GstTensorsInfo *in, GstTensorsInfo *out);
  /** @brief Tell the framework the output buffer is preallocated */
  bool isAllocatedBeforeInvoke ();
  /** @brief Write in*2 to the first output frame and in+1 to the second */
  int invoke (const GstTensorMemory *in, GstTensorMemory *out);

  /** @brief Verify the output file holds in*2 and in+1 for each frame */
  static int resultCompare (const char *inputFile, const char *outputFile,
      unsigned int nDropAllowed = 0);
};

/** @brief Test C++ filter: uint8 3:16:16:1 in, uint8 3:16:16:2 out */
class filter_basic2 : public tensor_filter_cpp
{
  public:
  /** @brief Construct the test filter with the given filter name */
  filter_basic2 (const char *str);
  /** @brief Destructor of the test filter */
  ~filter_basic2 ();

  /** @brief Report the fixed input dimension, uint8 3:16:16:1 */
  int getInputDim (GstTensorsInfo *info);
  /** @brief Report the fixed output dimension, uint8 3:16:16:2 */
  int getOutputDim (GstTensorsInfo *info);
  /** @brief Reject dimension negotiation; the dimensions are fixed */
  int setInputDim (const GstTensorsInfo *in, GstTensorsInfo *out);
  /** @brief Tell the framework the output buffer is preallocated */
  bool isAllocatedBeforeInvoke ();
  /** @brief Write in*3 to the first output frame and in+2 to the second */
  int invoke (const GstTensorMemory *in, GstTensorMemory *out);

  /** @brief Verify the output file holds in*3 and in+2 for each frame */
  static int resultCompare (const char *inputFile, const char *outputFile,
      unsigned int nDropAllowed = 0);
};
