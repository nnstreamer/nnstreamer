/**
 * @file        cppfilter_test.hh
 * @date        15 Jan 2019
 * @brief       Unit test cases for tensor_filter::cpp
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */
#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>

#include <glib.h>
#include <tensor_filter_cpp.hh>

class filter_basic: public tensor_filter_cpp {
  public:
    filter_basic(const char *str);
    ~filter_basic();

    int getInputDim(GstTensorsInfo *info);
    int getOutputDim(GstTensorsInfo *info);
    int setInputDim(const GstTensorsInfo *in, GstTensorsInfo *out);
    bool isAllocatedBeforeInvoke();
    int invoke(const GstTensorMemory *in, GstTensorMemory *out);

    static int resultCompare(const char *inputFile, const char *outputFile, unsigned int nDropAllowed=0);
};

class filter_basic2: public tensor_filter_cpp {
  public:
    filter_basic2(const char *str);
    ~filter_basic2();

    int getInputDim(GstTensorsInfo *info);
    int getOutputDim(GstTensorsInfo *info);
    int setInputDim(const GstTensorsInfo *in, GstTensorsInfo *out);
    bool isAllocatedBeforeInvoke();
    int invoke(const GstTensorMemory *in, GstTensorMemory *out);

    static int resultCompare(const char *inputFile, const char *outputFile, unsigned int nDropAllowed=0);
};
