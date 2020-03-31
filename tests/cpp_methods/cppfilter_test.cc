/**
 * @file        cppfilter_test.cc
 * @date        15 Jan 2019
 * @brief       Unit test cases for tensor_filter::cpp
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */
#include "cppfilter_test.hh"

filter_basic::filter_basic(const char *str): tensor_filter_cpp(str) {}

filter_basic::~filter_basic() {}

int filter_basic::getInputDim(GstTensorsInfo *info) {
  info->num_tensors = 1;
  info->info[0].type = _NNS_UINT8;
  info->info[0].dimension[0] = 3;
  info->info[0].dimension[1] = 4;
  info->info[0].dimension[2] = 4;
  info->info[0].dimension[3] = 1;
  return 0;
}

int filter_basic::getOutputDim(GstTensorsInfo *info) {
  info->num_tensors = 1;
  info->info[0].type = _NNS_UINT8;
  info->info[0].dimension[0] = 3;
  info->info[0].dimension[1] = 4;
  info->info[0].dimension[2] = 4;
  info->info[0].dimension[3] = 2;
  return 0;
}

int filter_basic::setInputDim(const GstTensorsInfo *in, GstTensorsInfo *out) {
  return -EINVAL;
}

bool filter_basic::isAllocatedBeforeInvoke() {
  return true;
}

int filter_basic::invoke(const GstTensorMemory *in, GstTensorMemory *out) {
  g_assert (in);
  g_assert (out);

  g_assert (prop->input_meta.info[0].dimension[0] == 3U);
  g_assert (prop->input_meta.info[0].dimension[1] == 4U);
  g_assert (prop->input_meta.info[0].dimension[2] == 4U);
  g_assert (prop->input_meta.info[0].dimension[3] == 1U);
  g_assert (prop->output_meta.info[0].dimension[0] == 3U);
  g_assert (prop->output_meta.info[0].dimension[1] == 4U);
  g_assert (prop->output_meta.info[0].dimension[2] == 4U);
  g_assert (prop->output_meta.info[0].dimension[3] == 2U);
  g_assert (prop->input_meta.info[0].type == _NNS_UINT8);
  g_assert (prop->output_meta.info[0].type == _NNS_UINT8);

  for (int i = 0; i < 4 * 4 * 3; i++) {
    *((uint8_t *) out[0].data + i) = *((uint8_t *) in[0].data + i) * 2;
    *((uint8_t *) out[0].data + i + 4 * 4 * 3) = *((uint8_t *) in[0].data + i) + 1;
  }
  return 0;
}

int filter_basic::resultCompare(const char *inputFile, const char *outputFile, unsigned int nDropAllowed) {
  std::ifstream is(inputFile);
  if (is.fail()) {
    g_printerr("File not found: (%s : %d)\n", inputFile, is.fail());
    return -255;
  }
  std::istream_iterator<uint8_t> istart(is), iend;
  std::vector<uint8_t> input(istart, iend);
  is >> std::noskipws;
  std::ifstream os(outputFile);

  if (os.fail()) {
    g_printerr("File not found: (%s : %d)\n", outputFile, os.fail());
    return -254;
  }
  std::istream_iterator<uint8_t> ostart(os), oend;
  std::vector<uint8_t> output(ostart, oend);
  os >> std::noskipws;

  unsigned int iframes = (input.size() / (3 * 4 * 4));
  unsigned int oframes = (output.size() / (3 * 4 * 4 * 2));

  if (input.size() % (3 * 4 * 4) != 0) {
    g_printerr("%zu, %zu\n", input.size(), output.size());
    return -1;
  }
  if (output.size() % (3 * 4 * 4 * 2) != 0) {
    g_printerr("%zu, %zu\n", input.size(), output.size());
    return -2;
  }
  if (oframes > iframes)
    return -3;
  if ((oframes + nDropAllowed) < iframes)
    return -4;

  for (unsigned int frame = 0; frame < oframes; frame++) {
    unsigned pos = frame * 3 * 4 * 4;
    for (unsigned int i = 0; i < (3 * 4 * 4); i++) {
      uint8_t o1 = output[pos * 2 + i];
      uint8_t o2 = output[pos * 2 + (3 * 4 * 4) + i];
      uint8_t in = input[pos + i];
      uint8_t in1 = in * 2;
      uint8_t in2 = in + 1;

      if (o1 != in1)
        return -5;
      if (o2 != in2)
        return -6;
    }
  }

  return 0;
}


filter_basic2::filter_basic2(const char *str): tensor_filter_cpp(str) {}

filter_basic2::~filter_basic2() {}

int filter_basic2::getInputDim(GstTensorsInfo *info) {
  info->num_tensors = 1;
  info->info[0].type = _NNS_UINT8;
  info->info[0].dimension[0] = 3;
  info->info[0].dimension[1] = 16;
  info->info[0].dimension[2] = 16;
  info->info[0].dimension[3] = 1;
  return 0;
}

int filter_basic2::getOutputDim(GstTensorsInfo *info) {
  info->num_tensors = 1;
  info->info[0].type = _NNS_UINT8;
  info->info[0].dimension[0] = 3;
  info->info[0].dimension[1] = 16;
  info->info[0].dimension[2] = 16;
  info->info[0].dimension[3] = 2;
  return 0;
}

int filter_basic2::setInputDim(const GstTensorsInfo *in, GstTensorsInfo *out) {
  return -EINVAL;
}

bool filter_basic2::isAllocatedBeforeInvoke() {
  return true;
}

int filter_basic2::invoke(const GstTensorMemory *in, GstTensorMemory *out) {
  g_assert (in);
  g_assert (out);

  g_assert (prop->input_meta.info[0].dimension[0] == 3U);
  g_assert (prop->input_meta.info[0].dimension[1] == 16U);
  g_assert (prop->input_meta.info[0].dimension[2] == 16U);
  g_assert (prop->input_meta.info[0].dimension[3] == 1U);
  g_assert (prop->output_meta.info[0].dimension[0] == 3U);
  g_assert (prop->output_meta.info[0].dimension[1] == 16U);
  g_assert (prop->output_meta.info[0].dimension[2] == 16U);
  g_assert (prop->output_meta.info[0].dimension[3] == 2U);
  g_assert (prop->input_meta.info[0].type == _NNS_UINT8);
  g_assert (prop->output_meta.info[0].type == _NNS_UINT8);

  for (int i = 0; i < 16 * 16 * 3; i++) {
    *((uint8_t *) out[0].data + i) = *((uint8_t *) in[0].data + i) * 3;
    *((uint8_t *) out[0].data + i + 16 * 16 * 3) = *((uint8_t *) in[0].data + i) + 2;
  }
  return 0;
}

int filter_basic2::resultCompare(const char *inputFile, const char *outputFile, unsigned int nDropAllowed) {
  std::ifstream is(inputFile);
  if (is.fail()) {
    g_printerr("File not found: (%s : %d)\n", inputFile, is.fail());
    return -255;
  }
  is >> std::noskipws;
  std::istream_iterator<uint8_t> istart(is), iend;
  std::vector<uint8_t> input(istart, iend);

  std::ifstream os(outputFile);
  if (os.fail()) {
    g_printerr("File not found: (%s : %d)\n", outputFile, os.fail());
    return -254;
  }
  os >> std::noskipws;
  std::istream_iterator<uint8_t> ostart(os), oend;
  std::vector<uint8_t> output(ostart, oend);

  unsigned int iframes = (input.size() / (3 * 16 * 16));
  unsigned int oframes = (output.size() / (3 * 16 * 16 * 2));

  if (input.size() % (3 * 16 * 16) != 0) {
    g_printerr("%zu, %zu\n", input.size(), output.size());
    return -1;
  }
  if (output.size() % (3 * 16 * 16 * 2) != 0) {
    g_printerr("%zu, %zu\n", input.size(), output.size());
    return -2;
  }
  if (oframes > iframes)
    return -3;
  if ((oframes + nDropAllowed) < iframes)
    return -4;

  for (unsigned int frame = 0; frame < oframes; frame++) {
    unsigned pos = frame * 3 * 16 * 16;
    for (unsigned int i = 0; i < (3 * 16 * 16); i++) {
      uint8_t o1 = output[pos * 2 + i];
      uint8_t o2 = output[pos * 2 + (3 * 16 * 16) + i];
      uint8_t in = input[pos + i];
      uint8_t in1 = in * 3;
      uint8_t in2 = in + 2;

      if (o1 != in1)
        return -5;
      if (o2 != in2)
        return -6;
    }
  }

  return 0;
}


class tensor_filter_cpp *reg1, *reg2, *reg3;

void init_shared_lib (void) __attribute__ ((constructor));
void fini_shared_lib (void) __attribute__ ((destructor));

void init_shared_lib (void)
{
  reg1 = new filter_basic("basic_so_01");
  reg2 = new filter_basic("basic_so_02");
  reg3 = new filter_basic2("basic_so2");
  reg1->_register();
  filter_basic::__register(reg2);
  reg3->_register();
}

void fini_shared_lib (void)
{
  filter_basic::__unregister("basic_so_01");
  reg2->_unregister();
  reg3->_unregister();

  delete reg1;
  delete reg2;
  delete reg3;
}
