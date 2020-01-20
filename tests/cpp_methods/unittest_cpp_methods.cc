/**
 * @file        unittest_cpp_methods.cc
 * @date        15 Jan 2019
 * @brief       Unit test cases for tensor_filter::cpp
 * @see         https://github.com/nnsuite/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */
#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <tensor_common.h>
#include <nnstreamer.h>
#include <nnstreamer-single.h>
#include <nnstreamer-capi-private.h>

#include <tensor_filter_cpp.h>
#include <unittest_util.h>

char *path_to_lib;

class filter_basic: public tensor_filter_cpp {
  public:
    filter_basic(const char *str): tensor_filter_cpp(str) {}
    ~filter_basic() { }
    int getInputDim(GstTensorsInfo *info) {
      info->num_tensors = 1;
      info->info[0].type = _NNS_UINT8;
      info->info[0].dimension[0] = 3;
      info->info[0].dimension[1] = 4;
      info->info[0].dimension[2] = 4;
      info->info[0].dimension[3] = 1;
      return 0;
    }
    int getOutputDim(GstTensorsInfo *info) {
      info->num_tensors = 1;
      info->info[0].type = _NNS_UINT8;
      info->info[0].dimension[0] = 3;
      info->info[0].dimension[1] = 4;
      info->info[0].dimension[2] = 4;
      info->info[0].dimension[3] = 2;
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

      EXPECT_EQ (prop->input_meta.info[0].dimension[0], 3U);
      EXPECT_EQ (prop->input_meta.info[0].dimension[1], 4U);
      EXPECT_EQ (prop->input_meta.info[0].dimension[2], 4U);
      EXPECT_EQ (prop->input_meta.info[0].dimension[3], 1U);
      EXPECT_EQ (prop->output_meta.info[0].dimension[0], 3U);
      EXPECT_EQ (prop->output_meta.info[0].dimension[1], 4U);
      EXPECT_EQ (prop->output_meta.info[0].dimension[2], 4U);
      EXPECT_EQ (prop->output_meta.info[0].dimension[3], 2U);
      EXPECT_EQ (prop->input_meta.info[0].type, _NNS_UINT8);
      EXPECT_EQ (prop->output_meta.info[0].type, _NNS_UINT8);

      for (int i = 0; i < 4 * 4 * 3; i++) {
        *((uint8_t *) out[0].data + i) = *((uint8_t *) in[0].data + i) * 2;
        *((uint8_t *) out[0].data + i + 4 * 4 * 3) = *((uint8_t *) in[0].data + i) + 1;
      }
      return 0;
    }

    static int resultCompare (const char *inputFile, const char *outputFile, unsigned int nDropAllowed=0) {
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
};

class filter_basic2: public tensor_filter_cpp {
  public:
    filter_basic2(const char *str): tensor_filter_cpp(str) {}
    ~filter_basic2() { }
    int getInputDim(GstTensorsInfo *info) {
      info->num_tensors = 1;
      info->info[0].type = _NNS_UINT8;
      info->info[0].dimension[0] = 3;
      info->info[0].dimension[1] = 16;
      info->info[0].dimension[2] = 16;
      info->info[0].dimension[3] = 1;
      return 0;
    }
    int getOutputDim(GstTensorsInfo *info) {
      info->num_tensors = 1;
      info->info[0].type = _NNS_UINT8;
      info->info[0].dimension[0] = 3;
      info->info[0].dimension[1] = 16;
      info->info[0].dimension[2] = 16;
      info->info[0].dimension[3] = 2;
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

      EXPECT_EQ (prop->input_meta.info[0].dimension[0], 3U);
      EXPECT_EQ (prop->input_meta.info[0].dimension[1], 16U);
      EXPECT_EQ (prop->input_meta.info[0].dimension[2], 16U);
      EXPECT_EQ (prop->input_meta.info[0].dimension[3], 1U);
      EXPECT_EQ (prop->output_meta.info[0].dimension[0], 3U);
      EXPECT_EQ (prop->output_meta.info[0].dimension[1], 16U);
      EXPECT_EQ (prop->output_meta.info[0].dimension[2], 16U);
      EXPECT_EQ (prop->output_meta.info[0].dimension[3], 2U);
      EXPECT_EQ (prop->input_meta.info[0].type, _NNS_UINT8);
      EXPECT_EQ (prop->output_meta.info[0].type, _NNS_UINT8);

      for (int i = 0; i < 16 * 16 * 3; i++) {
        *((uint8_t *) out[0].data + i) = *((uint8_t *) in[0].data + i) * 3;
        *((uint8_t *) out[0].data + i + 16 * 16 * 3) = *((uint8_t *) in[0].data + i) + 2;
      }
      return 0;
    }

    static int resultCompare (const char *inputFile, const char *outputFile, unsigned int nDropAllowed=0) {
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
};

/** @brief Positive case for the simpliest execution path */
TEST (cpp_filter_on_demand, basic_01)
{
  filter_basic basic("basic_01");
  EXPECT_EQ (basic._register(), 0);
  EXPECT_EQ (basic._unregister(), 0);
}

/** @brief Negative case for the simpliest execution path */
TEST (cpp_filter_on_demand, basic_02_n)
{
  filter_basic basic("basic_02");
  EXPECT_NE (basic._unregister(), 0);
  EXPECT_EQ (basic._register(), 0);
  EXPECT_NE (basic._register(), 0);
  EXPECT_EQ (basic._unregister(), 0);
  EXPECT_NE (basic._unregister(), 0);
}

/** @brief Negative case for the simpliest execution path w/ static calls */
TEST (cpp_filter_on_demand, basic_03_n)
{
  filter_basic basic("basic_03");
  EXPECT_NE (filter_basic::__unregister("basic_03"), 0);
  EXPECT_EQ (filter_basic::__register(&basic), 0);
  EXPECT_NE (filter_basic::__register(&basic), 0);
  EXPECT_EQ (filter_basic::__unregister("basic_03"), 0);
  EXPECT_NE (filter_basic::__unregister("basic_03"), 0);
}

/** @brief Negative case for the simpliest execution path w/ static calls */
TEST (cpp_filter_on_demand, basic_04_n)
{
  filter_basic basic("basic_04");
  EXPECT_NE (filter_basic::__unregister("basic_xx"), 0);
  EXPECT_NE (filter_basic::__unregister("basic_03"), 0);
  EXPECT_NE (filter_basic::__unregister("basic_04"), 0);
  EXPECT_EQ (filter_basic::__register(&basic), 0);
  EXPECT_NE (filter_basic::__register(&basic), 0);
  EXPECT_NE (filter_basic::__unregister("basic_xx"), 0);
  EXPECT_NE (filter_basic::__unregister("basic_03"), 0);
  EXPECT_EQ (filter_basic::__unregister("basic_04"), 0);
  EXPECT_NE (filter_basic::__unregister("basic_03"), 0);
  EXPECT_NE (filter_basic::__unregister("basic_04"), 0);
  EXPECT_NE (filter_basic::__unregister("basic_xx"), 0);
}

/** @brief Actual GST Pipeline with cpp on demand */
TEST (cpp_filter_on_demand, pipeline_01)
{
  filter_basic basic("pl01");
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);
  EXPECT_EQ (basic._register(), 0);
  gchar *str_pipeline = g_strdup_printf
      ("videotestsrc num-buffers=5 ! videoconvert ! videoscale ! video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tee name=t t. ! queue name=q1 ! tensor_filter framework=cpp model=pl01 ! filesink location=%s t. ! queue name=q2 ! filesink location=%s ", tmp1, tmp2);

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  if (pipeline) {
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, 500000U), 0);

    g_usleep(100000);
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, 500000U), 0);
    g_usleep(100000);

    gst_object_unref (pipeline);

    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp1), 0);
  }
  g_free (str_pipeline);

  g_free (tmp1);
  g_free (tmp2);
  EXPECT_EQ (basic._unregister(), 0);
}

/** @brief Negative case for the simpliest execution path */
TEST (cpp_filter_on_demand, unregistered_01_n)
{
  filter_basic basic("basic_01");
  gchar *str_pipeline = g_strdup_printf
      ("videotestsrc num-buffers=3 ! videoconvert ! videoscale ! video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tensor_filter framework=cpp model=XXbasic_01 ! fakesink");
  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  EXPECT_EQ (basic._register(), 0);
  gst_object_unref(pipeline);

  pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);
  if (pipeline)
    gst_object_unref(pipeline);
  EXPECT_EQ (basic._unregister(), 0);

  basic._unregister();
  g_free (str_pipeline);
  EXPECT_NE (basic._unregister(), 0);
}

TEST (cpp_filter_obj, base_01_n)
{
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();
  char *tmp3 = getTempFilename ();

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);
  EXPECT_NE (tmp3, nullptr);
  gchar *str_pipeline = g_strdup_printf
      ("videotestsrc num-buffers=5 ! videoconvert ! videoscale ! video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tee name=t t. ! queue name=q1 ! tensor_filter framework=cpp model=basic_so_01,%slibcppfilter_test.so ! filesink location=%s t. ! queue name=q2 ! filesink location=%s t. ! queue ! tensor_filter framework=cpp model=basic_so_03,%slibcppfilter_test.so ! filesink location=%s", path_to_lib, tmp1, tmp2, path_to_lib, tmp3);
GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  if (pipeline) {
    EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, 500000U), 0);

    gst_object_unref (pipeline);

  }
  g_free (str_pipeline);

  g_free (tmp1);
  g_free (tmp2);
  g_free (tmp3);
}

TEST (cpp_filter_obj, base_02_n)
{
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();
  char *tmp3 = getTempFilename ();

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);
  EXPECT_NE (tmp3, nullptr);
  gchar *str_pipeline = g_strdup_printf
      ("videotestsrc num-buffers=5 ! videoconvert ! videoscale ! video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tee name=t t. ! queue name=q1 ! tensor_filter framework=cpp model=basic_so_01,%slibcppfilter_test.so ! filesink location=%s t. ! queue name=q2 ! filesink location=%s t. ! queue ! tensor_filter framework=cpp model=basic_so_03,%slibcppfilter_test.so ! filesink location=%s", path_to_lib, tmp1, tmp2, path_to_lib, tmp3);

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  if (pipeline) {
    EXPECT_NE (setPipelineStateSync (pipeline, GST_STATE_PLAYING, 500000U), 0);

    gst_object_unref (pipeline);
  }
  g_free (str_pipeline);

  g_free (tmp1);
  g_free (tmp2);
  g_free (tmp3);
}

TEST (cpp_filter_obj, base_03)
{
  char *tmp1 = getTempFilename ();
  char *tmp2 = getTempFilename ();
  char *tmp3 = getTempFilename ();
  char *tmp4 = getTempFilename ();
  char *tmp5 = getTempFilename ();

  EXPECT_NE (tmp1, nullptr);
  EXPECT_NE (tmp2, nullptr);
  EXPECT_NE (tmp3, nullptr);
  EXPECT_NE (tmp4, nullptr);
  EXPECT_NE (tmp5, nullptr);
  gchar *str_pipeline = g_strdup_printf
      ("videotestsrc num-buffers=5 ! videoconvert ! videoscale ! video/x-raw,width=4,height=4,format=RGB ! tensor_converter ! tee name=t t. ! queue name=q1 ! tensor_filter framework=cpp model=basic_so_01,%slibcppfilter_test.so ! filesink location=%s sync=true t. ! queue name=q2 ! filesink location=%s sync=true t. ! queue ! tensor_filter framework=cpp model=basic_so_02,%slibcppfilter_test.so ! filesink location=%s sync=true videotestsrc num-buffers=5 ! videoconvert ! videoscale ! video/x-raw,width=16,height=16,format=RGB ! tensor_converter ! tee name=t2 t2. ! queue ! tensor_filter framework=cpp model=basic_so2,%slibcppfilter_test.so ! filesink location=%s sync=true t2. ! queue ! filesink location=%s sync=true ", path_to_lib, tmp1, tmp2, path_to_lib, tmp3, path_to_lib, tmp4, tmp5);

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch (str_pipeline, &err);

  EXPECT_NE (pipeline, nullptr);
  EXPECT_EQ (err, nullptr);

  if (err) {
    g_printerr ("Cannot construct pipeline: %s\n", err->message);
    g_clear_error (&err);
  }

  if (pipeline) {
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, 500000U), 0);

    g_usleep(300000);
    EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, 500000U), 0);

    gst_object_unref (pipeline);
    g_usleep(300000);

    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp1), 0);
    EXPECT_EQ (filter_basic::resultCompare (tmp2, tmp3), 0);
    EXPECT_EQ (filter_basic2::resultCompare (tmp5, tmp4), 0);
  }
  g_free (str_pipeline);

  g_free (tmp1);
  g_free (tmp2);
  g_free (tmp3);
  g_free (tmp4);
  g_free (tmp5);
}

/**
 * @brief Main GTest
 */
int main (int argc, char **argv)
{
  int result;
  int delete_path = 0;

  if (argc > 3 && !g_strcmp0(argv[1], "-libpath")) {
    path_to_lib = argv[2];
  } else {
    gchar *dir = g_path_get_dirname (argv[0]);
    path_to_lib = g_strdup_printf("%s/", dir);
    delete_path = 1;
    g_free(dir);
    g_printerr("LIBPATH = %s\n", path_to_lib);
  }

  testing::InitGoogleTest (&argc, argv);
  gst_init (&argc, &argv);

  /* ignore tizen feature status while running the testcases */
  set_feature_state (1);

  result = RUN_ALL_TESTS ();

  set_feature_state (-1);

  if (delete_path)
    g_free(path_to_lib);
  return result;
}

class tensor_filter_cpp *reg1, *reg2, *reg3;

#ifdef THIS_IS_SHLIB
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
#endif
