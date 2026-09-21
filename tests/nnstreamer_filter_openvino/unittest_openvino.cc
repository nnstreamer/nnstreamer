/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_openvino.cc
 * @author      Wook Song <wook16.song@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib/gstdio.h>
#include <gst/check/gstcheck.h>
#include <gst/check/gstharness.h>
#include <gst/check/gsttestclock.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api_filter.h>
#include <string.h>
#include <tensor_common.h>

#include <tensor_filter_openvino.hh>

const static gchar MODEL_BASE_NAME_MOBINET_V2[] = "openvino_mobilenetv2-int8-tf-0001";

const static uint32_t MOBINET_V2_IN_NUM_TENSOR = 1;
const static uint32_t MOBINET_V2_IN_DIMS[NNS_TENSOR_RANK_LIMIT] = {
  224,
  224,
  3,
  1,
  0,
};
const static uint32_t MOBINET_V2_OUT_NUM_TENSOR = 1;
const static uint32_t MOBINET_V2_OUT_DIMS[NNS_TENSOR_RANK_LIMIT] = {
  1001,
  1,
  1,
  1,
  0,
};

/** @brief wooksong: please fill in */
class TensorFilterOpenvinoTest : public TensorFilterOpenvino
{
  public:
  typedef TensorFilterOpenvino super;
  TensorFilterOpenvinoTest (std::string path_model_xml, std::string path_model_bin);
  ~TensorFilterOpenvinoTest ();

  InferenceEngine::InputsDataMap &getInputsDataMap ();
  void setInputsDataMap (InferenceEngine::InputsDataMap &map);
  InferenceEngine::OutputsDataMap &getOutputsDataMap ();
  void setOutputsDataMap (InferenceEngine::OutputsDataMap &map);
  const std::vector<std::string> &getInputTensorNames ();
  void setInputTensorNames (const std::vector<std::string> &names);
  const std::vector<std::string> &getOutputTensorNames ();
  void setOutputTensorNames (const std::vector<std::string> &names);
  const InferenceEngine::TensorDesc &getInputTensorDesc (guint nth);
  const InferenceEngine::TensorDesc &getOutputTensorDesc (guint nth);

  private:
  TensorFilterOpenvinoTest ();
};

/** @brief wooksong: please fill in */
TensorFilterOpenvinoTest::TensorFilterOpenvinoTest (
    std::string path_model_xml, std::string path_model_bin)
    : super (path_model_xml, path_model_bin)
{
  /* Nothing to do */
  ;
}

/** @brief wooksong: please fill in */
TensorFilterOpenvinoTest::~TensorFilterOpenvinoTest ()
{
  /* Nothing to do */
  ;
}

/** @brief wooksong: please fill in */
InferenceEngine::InputsDataMap &
TensorFilterOpenvinoTest::getInputsDataMap ()
{
  return this->_inputsDataMap;
}

/** @brief wooksong: please fill in */
void
TensorFilterOpenvinoTest::setInputsDataMap (InferenceEngine::InputsDataMap &map)
{
  this->_inputsDataMap = map;
}


/** @brief wooksong: please fill in */
InferenceEngine::OutputsDataMap &
TensorFilterOpenvinoTest::getOutputsDataMap ()
{
  return this->_outputsDataMap;
}

/** @brief wooksong: please fill in */
void
TensorFilterOpenvinoTest::setOutputsDataMap (InferenceEngine::OutputsDataMap &map)
{
  this->_outputsDataMap = map;
}

/** @brief Get the names of the input tensors that the given model declares */
const std::vector<std::string> &
TensorFilterOpenvinoTest::getInputTensorNames ()
{
  return this->_inputTensorNames;
}

/** @brief Replace the names of the input tensors that the given model declares */
void
TensorFilterOpenvinoTest::setInputTensorNames (const std::vector<std::string> &names)
{
  this->_inputTensorNames = names;
}

/** @brief Get the names of the output tensors that the given model declares */
const std::vector<std::string> &
TensorFilterOpenvinoTest::getOutputTensorNames ()
{
  return this->_outputTensorNames;
}

/** @brief Replace the names of the output tensors that the given model declares */
void
TensorFilterOpenvinoTest::setOutputTensorNames (const std::vector<std::string> &names)
{
  this->_outputTensorNames = names;
}

/** @brief Get the descriptor of the nth input tensor of the given model */
const InferenceEngine::TensorDesc &
TensorFilterOpenvinoTest::getInputTensorDesc (guint nth)
{
  return this->_inputTensorDescs[nth];
}

/** @brief Get the descriptor of the nth output tensor of the given model */
const InferenceEngine::TensorDesc &
TensorFilterOpenvinoTest::getOutputTensorDesc (guint nth)
{
  return this->_outputTensorDescs[nth];
}

/**
 * @brief Test cases for open and close callbacks varying the model files
 */
TEST (tensorFilterOpenvino, openAndClose0)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  std::string str_test_model;
  gchar *test_model;
  gint ret;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      MODEL_BASE_NAME_MOBINET_V2, NULL);
  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);
  prop->fwname = fw_name;
  prop->num_models = 1;
  prop->accl_str = "true:cpu";
  {
    const gchar *model_files[] = {
      test_model,
      NULL,
    };

    prop->model_files = model_files;

    ret = fw->open (prop, &private_data);
#ifdef __OPENVINO_CPU_EXT__
    EXPECT_EQ (ret, 0);
#else
    EXPECT_NE (ret, 0);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetENoDev);
#endif
  }

  fw->close (prop, &private_data);
  g_free (test_model);

  {
    gchar *test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
        str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
            .append (TensorFilterOpenvino::extXml)
            .c_str (),
        NULL);
    gchar *test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
        str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
            .append (TensorFilterOpenvino::extBin)
            .c_str (),
        NULL);
    const gchar *model_files[] = {
      test_model_xml,
      test_model_bin,
    };

    prop->num_models = 2;
    prop->model_files = model_files;

    ret = fw->open (prop, &private_data);
#ifdef __OPENVINO_CPU_EXT__
    EXPECT_EQ (ret, 0);
#else
    EXPECT_NE (ret, 0);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetENoDev);
#endif

    fw->close (prop, &private_data);
    g_free (test_model_xml);
    g_free (test_model_bin);
  }

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);
  {
    const gchar *model_files[] = {
      test_model,
      NULL,
    };

    prop->num_models = 1;
    prop->model_files = model_files;

    ret = fw->open (prop, &private_data);

#ifdef __OPENVINO_CPU_EXT__
    EXPECT_EQ (ret, 0);
#else
    EXPECT_NE (ret, 0);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetENoDev);
#endif
  }

  fw->close (prop, &private_data);
  g_free (test_model);

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  {
    const gchar *model_files[] = {
      test_model,
      NULL,
    };

    prop->num_models = 1;
    prop->model_files = model_files;

    ret = fw->open (prop, &private_data);

#ifdef __OPENVINO_CPU_EXT__
    EXPECT_EQ (ret, 0);
#else
    EXPECT_NE (ret, 0);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetENoDev);
#endif
  }

  fw->close (prop, &private_data);
  g_free (test_model);

  g_free (prop);
}

/**
 * @brief A test case for open and close callbacks with the private_data, which has the models already loaded
 */
TEST (tensorFilterOpenvino, openAndClose1)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  std::string str_test_model;
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  TensorFilterOpenvino *tfOv;
  gchar *test_model_xml;
  gchar *test_model_bin;
  gint ret;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);

  tfOv = new TensorFilterOpenvino (str_test_model.assign (test_model_xml),
      str_test_model.assign (test_model_bin));
  ret = tfOv->loadModel (ACCL_CPU);
#ifdef __OPENVINO_CPU_EXT__
  EXPECT_EQ (ret, 0);
#else
  EXPECT_NE (ret, 0);
  EXPECT_EQ (ret, TensorFilterOpenvino::RetENoDev);
#endif
  private_data = (gpointer) tfOv;

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);
  prop->fwname = fw_name;
  prop->num_models = 2;
  prop->accl_str = "true:cpu";
  {
    const gchar *model_files[] = {
      test_model_xml,
      test_model_bin,
    };

    prop->model_files = model_files;

    ret = fw->open (prop, &private_data);
#ifdef __OPENVINO_CPU_EXT__
    EXPECT_EQ (ret, 0);
#else
    EXPECT_NE (ret, 0);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetENoDev);
#endif
  }

  fw->close (prop, &private_data);

  g_free (test_model_xml);
  g_free (test_model_bin);
  g_free (prop);
}

/**
 * @brief A test case for open and close callbacks with the private_data, which has the models are not loaded
 */
TEST (tensorFilterOpenvino, openAndClose2)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  std::string str_test_model;
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  TensorFilterOpenvino *tfOv;
  gchar *test_model_xml;
  gchar *test_model_bin;
  gint ret;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);

  tfOv = new TensorFilterOpenvino (str_test_model.assign (test_model_xml),
      str_test_model.assign (test_model_bin));
  private_data = (gpointer) tfOv;

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);
  prop->fwname = fw_name;
  prop->num_models = 2;
  prop->accl_str = "true:cpu";
  {
    const gchar *model_files[] = {
      test_model_xml,
      test_model_bin,
    };

    prop->model_files = model_files;

    ret = fw->open (prop, &private_data);
#ifdef __OPENVINO_CPU_EXT__
    EXPECT_EQ (ret, 0);
#else
    EXPECT_NE (ret, 0);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetENoDev);
#endif
  }

  fw->close (prop, &private_data);

  g_free (test_model_xml);
  g_free (test_model_bin);
  g_free (prop);
}

/**
 * @brief Negative test cases for open and close callbacks with wrong model files
 */
TEST (tensorFilterOpenvino, openAndClose0_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  gchar *test_model;
  gint ret;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (
      root_path, "tests", "test_models", "models", "NOT_EXIST", NULL);
  const gchar *model_files[] = {
    test_model,
    NULL,
  };

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);

  prop->fwname = fw_name;
  prop->model_files = model_files;
  prop->num_models = 1;
  prop->accl_str = "true:cpu";

  ret = fw->open (prop, &private_data);
  EXPECT_NE (ret, TensorFilterOpenvino::RetSuccess);
  EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);

  g_free (test_model);
  fw->close (prop, &private_data);

  {
    std::string str_test_model;
    gchar *test_model_xml1 = g_build_filename (root_path, "tests", "test_models", "models",
        str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
            .append (TensorFilterOpenvino::extXml)
            .c_str (),
        NULL);
    gchar *test_model_xml2 = g_build_filename (root_path, "tests", "test_models", "models",
        str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
            .append (TensorFilterOpenvino::extXml)
            .c_str (),
        NULL);
    const gchar *model_files[] = {
      test_model_xml1,
      test_model_xml2,
    };

    prop->num_models = 2;
    prop->model_files = model_files;

    ret = fw->open (prop, &private_data);
    EXPECT_NE (ret, TensorFilterOpenvino::RetSuccess);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);

    fw->close (prop, &private_data);
    g_free (test_model_xml1);
    g_free (test_model_xml2);
  }

  {
    std::string str_test_model;
    gchar *test_model_bin1 = g_build_filename (root_path, "tests", "test_models", "models",
        str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
            .append (TensorFilterOpenvino::extBin)
            .c_str (),
        NULL);
    gchar *test_model_bin2 = g_build_filename (root_path, "tests", "test_models", "models",
        str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
            .append (TensorFilterOpenvino::extBin)
            .c_str (),
        NULL);
    const gchar *model_files[] = {
      test_model_bin1,
      test_model_bin2,
    };

    prop->num_models = 2;
    prop->model_files = model_files;

    ret = fw->open (prop, &private_data);
    EXPECT_NE (ret, TensorFilterOpenvino::RetSuccess);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);

    fw->close (prop, &private_data);
    g_free (test_model_bin1);
    g_free (test_model_bin2);
  }

  g_free (prop);
}

/**
 * @brief Negative test cases for open and close callbacks with accelerator
 *        property values, which are not supported
 */
TEST (tensorFilterOpenvino, openAndClose1_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  gchar *test_model;
  gint ret;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      MODEL_BASE_NAME_MOBINET_V2, NULL);
  const gchar *model_files[] = {
    test_model,
    NULL,
  };

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);

  prop->fwname = fw_name;
  prop->model_files = model_files;
  prop->num_models = 1;

  ret = fw->open (prop, &private_data);
  EXPECT_NE (ret, TensorFilterOpenvino::RetSuccess);
  EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);
  fw->close (prop, &private_data);

  prop->accl_str = "true:auto";
  ret = fw->open (prop, &private_data);
  EXPECT_NE (ret, TensorFilterOpenvino::RetSuccess);
  EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);
  fw->close (prop, &private_data);

  prop->accl_str = "true:gpu";
  ret = fw->open (prop, &private_data);
  EXPECT_NE (ret, TensorFilterOpenvino::RetSuccess);
  EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);
  fw->close (prop, &private_data);

#ifdef __OPENVINO_CPU_EXT__
  prop->accl_str = "true:npu.movidius";
  ret = fw->open (prop, &private_data);
  EXPECT_NE (ret, TensorFilterOpenvino::RetSuccess);
  EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);
#else
  prop->accl_str = "true:cpu";
  ret = fw->open (prop, &private_data);
  EXPECT_NE (ret, TensorFilterOpenvino::RetSuccess);
  EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);
#endif
  fw->close (prop, &private_data);

  g_free (prop);
  g_free (test_model);
}

/**
 * @brief Negative test cases for open and close callbacks with accelerator
 *        property values, which are wrong
 */
TEST (tensorFilterOpenvino, openAndClose2_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  gchar *test_model;
  gint ret;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      MODEL_BASE_NAME_MOBINET_V2, NULL);
  const gchar *model_files[] = {
    test_model,
    NULL,
  };

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);

  prop->fwname = fw_name;
  prop->model_files = model_files;
  prop->num_models = 1;

  ret = fw->open (prop, &private_data);
  EXPECT_NE (ret, TensorFilterOpenvino::RetSuccess);
  EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);
  fw->close (prop, &private_data);

#ifdef __OPENVINO_CPU_EXT__
  prop->accl_str = "true:npu.movidius";
  ret = fw->open (prop, &private_data);
  EXPECT_NE (ret, TensorFilterOpenvino::RetSuccess);
  EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);
#else
  prop->accl_str = "true:cpu";
  ret = fw->open (prop, &private_data);
  EXPECT_NE (ret, TensorFilterOpenvino::RetSuccess);
  EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);
#endif
  fw->close (prop, &private_data);

  g_free (prop);
  g_free (test_model);
}

/**
 * @brief A negative test case checking that a failed open leaves no private data
 *
 * The framework does not call the close callback for an open that has failed,
 * so the sub-plugin is the one that has to release what it has allocated.
 */
TEST (tensorFilterOpenvino, openAndClose3_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  gchar *test_model;
  gint ret;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      MODEL_BASE_NAME_MOBINET_V2, NULL);
  const gchar *model_files[] = {
    test_model,
    NULL,
  };

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);

  prop->fwname = fw_name;
  prop->model_files = model_files;
  prop->num_models = 1;
  /* No Movidius device is attached to the machine running this test */
  prop->accl_str = "true:npu.movidius";

  ret = fw->open (prop, &private_data);
  EXPECT_NE (ret, TensorFilterOpenvino::RetSuccess);
  EXPECT_TRUE (private_data == NULL);

  /** Closing is what the framework skips for an open that has failed, so the
   *  failed path is left alone and valgrind still sees what it leaks. */
  if (ret == TensorFilterOpenvino::RetSuccess)
    fw->close (prop, &private_data);

  g_free (prop);
  g_free (test_model);
}

/**
 * @brief A negative test case for the open callback with an unreadable model
 *
 * Reading the model is what throws here, and an exception that leaves a C
 * callback of the framework terminates the process instead of failing the open.
 */
TEST (tensorFilterOpenvino, openAndClose4_n)
{
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  gchar *tmp_dir;
  gchar *test_model_xml;
  gchar *test_model_bin;
  gint ret = TensorFilterOpenvino::RetSuccess;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close);

  tmp_dir = g_dir_make_tmp ("nnstreamer_openvino_XXXXXX", NULL);
  ASSERT_TRUE (tmp_dir != NULL);
  test_model_xml = g_build_filename (tmp_dir, "invalid.xml", NULL);
  test_model_bin = g_build_filename (tmp_dir, "invalid.bin", NULL);
  ASSERT_TRUE (g_file_set_contents (test_model_xml, "not a model at all", -1, NULL));
  ASSERT_TRUE (g_file_set_contents (test_model_bin, "", 0, NULL));

  const gchar *model_files[] = {
    test_model_xml,
    test_model_bin,
  };

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);

  prop->fwname = fw_name;
  prop->model_files = model_files;
  prop->num_models = 2;
  prop->accl_str = "true:npu.movidius";

  /** Nothing the Inference Engine throws may leave the callback, and catching
   *  it here is also what leaves the rest of the case to run */
  EXPECT_NO_THROW (ret = fw->open (prop, &private_data));
  EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);
  EXPECT_TRUE (private_data == NULL);

  g_free (prop);
  g_remove (test_model_xml);
  g_remove (test_model_bin);
  g_rmdir (tmp_dir);
  g_free (test_model_xml);
  g_free (test_model_bin);
  g_free (tmp_dir);
}

/**
 * @brief Test cases for getInputTensorDim and getOutputTensorDim callbacks
 */
TEST (tensorFilterOpenvino, getTensorDim0)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  GstTensorsInfo nns_tensors_info;
  GstTensorInfo *_info;
  gpointer private_data = NULL;
  std::string str_test_model;
  gchar *test_model;
  gint ret;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      MODEL_BASE_NAME_MOBINET_V2, NULL);
  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);
  prop->fwname = fw_name;
  prop->num_models = 1;
  prop->accl_str = "true:cpu";
  {
    const gchar *model_files[] = {
      test_model,
      NULL,
    };

    prop->model_files = model_files;

    ret = fw->open (prop, &private_data);
#ifdef __OPENVINO_CPU_EXT__
    EXPECT_EQ (ret, 0);
#else
    EXPECT_NE (ret, 0);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetENoDev);
#endif
  }

  /* Test getInputDimension () */
  ASSERT_TRUE (fw->getInputDimension);
  ret = fw->getInputDimension (prop, &private_data, &nns_tensors_info);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (nns_tensors_info.num_tensors, MOBINET_V2_IN_NUM_TENSOR);
  for (uint32_t i = 0; i < MOBINET_V2_IN_NUM_TENSOR; ++i) {
    _info = gst_tensors_info_get_nth_info (&nns_tensors_info, i);
    EXPECT_TRUE (gst_tensor_dimension_is_equal (_info->dimension, MOBINET_V2_IN_DIMS));
  }

  /* Test getOutputDimension () */
  ASSERT_TRUE (fw->getOutputDimension);
  ret = fw->getOutputDimension (prop, &private_data, &nns_tensors_info);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (nns_tensors_info.num_tensors, MOBINET_V2_OUT_NUM_TENSOR);
  for (uint32_t i = 0; i < MOBINET_V2_OUT_NUM_TENSOR; ++i) {
    _info = gst_tensors_info_get_nth_info (&nns_tensors_info, i);
    EXPECT_TRUE (gst_tensor_dimension_is_equal (_info->dimension, MOBINET_V2_OUT_DIMS));
  }

  fw->close (prop, &private_data);

  g_free (test_model);

  g_free (prop);
}

/**
 * @brief A negative test case for getInputTensorDim callbacks (The number of tensors is exceeded NNS_TENSOR_SIZE_LIMIT)
 */
TEST (tensorFilterOpenvino, getTensorDim0_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  std::string str_test_model;
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  GstTensorsInfo nns_tensors_info;
  gchar *test_model_xml;
  gchar *test_model_bin;
  gint ret;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);

  {
    TensorFilterOpenvinoTest tfOvTest (str_test_model.assign (test_model_xml),
        str_test_model.assign (test_model_bin));
    /** A test case when the number of tensors in input exceed is exceeded
     * NNS_TENSOR_SIZE_LIMIT */
    std::string name_input = std::string ("input");
    InferenceEngine::InputsDataMap inDataMap;
    InferenceEngine::SizeVector dims = InferenceEngine::SizeVector ();
    InferenceEngine::Data *data = new InferenceEngine::Data (
        name_input, dims, InferenceEngine::Precision::FP32);
    InferenceEngine::InputInfo *info = new InferenceEngine::InputInfo ();
    info->setInputData (InferenceEngine::DataPtr (data));
    inDataMap[name_input] = InferenceEngine::InputInfo::Ptr (info);

    for (int i = 1; i < NNS_TENSOR_SIZE_LIMIT + 1; ++i) {
      InferenceEngine::InputInfo *info = new InferenceEngine::InputInfo ();
      std::string name_input_n = std::string ((char *) &i);
      inDataMap[name_input_n] = InferenceEngine::InputInfo::Ptr (info);
    }

    tfOvTest.setInputsDataMap (inDataMap);
    ret = tfOvTest.loadModel (ACCL_CPU);
    private_data = (gpointer) &tfOvTest;

#ifdef __OPENVINO_CPU_EXT__
    EXPECT_EQ (ret, 0);
#else
    EXPECT_NE (ret, 0);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetENoDev);
#endif

    /* prepare properties */
    prop = g_new0 (GstTensorFilterProperties, 1);
    ASSERT_TRUE (prop != NULL);
    prop->fwname = fw_name;

    /* Test getInputDimension () */
    ASSERT_TRUE (fw->getInputDimension);
    ret = fw->getInputDimension (prop, &private_data, &nns_tensors_info);
    EXPECT_NE (ret, 0);
    g_free (prop);
  }

  g_free (test_model_xml);
  g_free (test_model_bin);
}

/**
 * @brief A negative test case for the getInputTensorDim callback (A wrong rank)
 */
TEST (tensorFilterOpenvino, getTensorDim1_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  std::string str_test_model;
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  GstTensorsInfo nns_tensors_info;
  gchar *test_model_xml;
  gchar *test_model_bin;
  gint ret;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);

  {
    TensorFilterOpenvinoTest tfOvTest (str_test_model.assign (test_model_xml),
        str_test_model.assign (test_model_bin));
    /** A test case when the number of ranks of a tensor in the input exceed is
     * exceeded NNS_TENSOR_RANK_LIMIT */
    std::string name_input = std::string ("input");
    InferenceEngine::SizeVector dims;
    InferenceEngine::InputsDataMap inDataMap;
    InferenceEngine::Data *data;
    InferenceEngine::InputInfo *info;

    dims = InferenceEngine::SizeVector (NNS_TENSOR_RANK_LIMIT + 1, 1);
    data = new InferenceEngine::Data (name_input, dims,
        InferenceEngine::Precision::FP32, InferenceEngine::ANY);
    info = new InferenceEngine::InputInfo ();
    info->setInputData (InferenceEngine::DataPtr (data));
    inDataMap[name_input] = InferenceEngine::InputInfo::Ptr (info);

    tfOvTest.setInputsDataMap (inDataMap);
    ret = tfOvTest.loadModel (ACCL_CPU);
    private_data = (gpointer) &tfOvTest;

#ifdef __OPENVINO_CPU_EXT__
    EXPECT_EQ (ret, 0);
#else
    EXPECT_NE (ret, 0);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetENoDev);
#endif

    /* prepare properties */
    prop = g_new0 (GstTensorFilterProperties, 1);
    ASSERT_TRUE (prop != NULL);
    prop->fwname = fw_name;

    /* Test getInputDimension () */
    ASSERT_TRUE (fw->getInputDimension);
    ret = fw->getInputDimension (prop, &private_data, &nns_tensors_info);
    EXPECT_NE (ret, 0);
    g_free (prop);
  }

  g_free (test_model_xml);
  g_free (test_model_bin);
}

/**
 * @brief A negative test case for getOutputTensorDim callbacks (The number of tensors is exceeded NNS_TENSOR_SIZE_LIMIT)
 */
TEST (tensorFilterOpenvino, getTensorDim2_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  std::string str_test_model;
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  GstTensorsInfo nns_tensors_info;
  gchar *test_model_xml;
  gchar *test_model_bin;
  gint ret;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);

  {
    TensorFilterOpenvinoTest tfOvTest (str_test_model.assign (test_model_xml),
        str_test_model.assign (test_model_bin));
    /** A test case when the number of tensors in input exceed is exceeded
     * NNS_TENSOR_SIZE_LIMIT */
    InferenceEngine::OutputsDataMap outDataMap;
    InferenceEngine::SizeVector dims = InferenceEngine::SizeVector ();

    for (int i = 0; i < NNS_TENSOR_SIZE_LIMIT + 1; ++i) {
      std::string name_output_n = std::to_string (i);
      InferenceEngine::Data *data = new InferenceEngine::Data (
          name_output_n, dims, InferenceEngine::Precision::FP32);
      InferenceEngine::DataPtr outputDataPtr (data);

      outDataMap[name_output_n] = outputDataPtr;
    }

    tfOvTest.setOutputsDataMap (outDataMap);
    ret = tfOvTest.loadModel (ACCL_CPU);
    private_data = (gpointer) &tfOvTest;

#ifdef __OPENVINO_CPU_EXT__
    EXPECT_EQ (ret, 0);
#else
    EXPECT_NE (ret, 0);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetENoDev);
#endif

    /* prepare properties */
    prop = g_new0 (GstTensorFilterProperties, 1);
    ASSERT_TRUE (prop != NULL);
    prop->fwname = fw_name;

    /* Test getOutputDimension () */
    ASSERT_TRUE (fw->getOutputDimension);
    ret = fw->getOutputDimension (prop, &private_data, &nns_tensors_info);
    EXPECT_NE (ret, 0);
    g_free (prop);
  }

  g_free (test_model_xml);
  g_free (test_model_bin);
}

/**
 * @brief A negative test case for getOutputTensorDim callbacks (The number of tensors is exceeded NNS_TENSOR_SIZE_LIMIT)
 */
TEST (tensorFilterOpenvino, getTensorDim3_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  std::string str_test_model;
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  GstTensorsInfo nns_tensors_info;
  gchar *test_model_xml;
  gchar *test_model_bin;
  gint ret;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);

  {
    TensorFilterOpenvinoTest tfOvTest (str_test_model.assign (test_model_xml),
        str_test_model.assign (test_model_bin));
    /** A test case when the number of ranks of a tensor in the input exceed is
     * exceeded NNS_TENSOR_RANK_LIMIT */
    std::string name_output = std::string ("output");
    InferenceEngine::SizeVector dims;
    InferenceEngine::OutputsDataMap outDataMap;
    InferenceEngine::Data *data;

    dims = InferenceEngine::SizeVector (NNS_TENSOR_RANK_LIMIT + 1, 1);
    data = new InferenceEngine::Data (name_output, dims,
        InferenceEngine::Precision::FP32, InferenceEngine::ANY);
    outDataMap[name_output] = InferenceEngine::DataPtr (data);

    tfOvTest.setOutputsDataMap (outDataMap);
    ret = tfOvTest.loadModel (ACCL_CPU);
    private_data = (gpointer) &tfOvTest;

#ifdef __OPENVINO_CPU_EXT__
    EXPECT_EQ (ret, 0);
#else
    EXPECT_NE (ret, 0);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetENoDev);
#endif

    /* prepare properties */
    prop = g_new0 (GstTensorFilterProperties, 1);
    ASSERT_TRUE (prop != NULL);
    prop->fwname = fw_name;

    /* Test getOutputDimension () */
    ASSERT_TRUE (fw->getOutputDimension);
    ret = fw->getOutputDimension (prop, &private_data, &nns_tensors_info);
    EXPECT_NE (ret, 0);
    g_free (prop);
  }

  g_free (test_model_xml);
  g_free (test_model_bin);
}

/**
 * @brief A test case for the helper function, convertFromIETypeStr ()
 */
TEST (tensorFilterOpenvino, convertFromIETypeStr0)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const std::vector<std::string> ie_support_type_strs = {
    "I8",
    "I16",
    "I32",
    "U8",
    "U16",
    "FP32",
  };
  const std::vector<tensor_type> nns_support_types = {
    _NNS_INT8,
    _NNS_INT16,
    _NNS_INT32,
    _NNS_UINT8,
    _NNS_UINT16,
    _NNS_FLOAT32,
  };
  std::string str_test_model;
  gchar *test_model_xml;
  gchar *test_model_bin;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);

  {
    TensorFilterOpenvinoTest tfOvTest (str_test_model.assign (test_model_xml),
        str_test_model.assign (test_model_bin));
    for (size_t i = 0; i < ie_support_type_strs.size (); ++i) {
      tensor_type ret_type;

      ret_type = tfOvTest.convertFromIETypeStr (ie_support_type_strs[i]);
      EXPECT_EQ (ret_type, nns_support_types[i]);
    }
  }

  g_free (test_model_xml);
  g_free (test_model_bin);
}

/**
 * @brief A negative test case for the helper function, convertFromIETypeStr ()
 */
TEST (tensorFilterOpenvino, convertFromIETypeStr0_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const std::vector<std::string> ie_not_support_type_strs = {
    "F64",
  };
  const std::vector<tensor_type> nns_support_types = {
    _NNS_FLOAT64,
  };
  std::string str_test_model;
  gchar *test_model_xml;
  gchar *test_model_bin;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);

  {
    TensorFilterOpenvinoTest tfOvTest (str_test_model.assign (test_model_xml),
        str_test_model.assign (test_model_bin));
    for (size_t i = 0; i < ie_not_support_type_strs.size (); ++i) {
      tensor_type ret_type;
      ret_type = tfOvTest.convertFromIETypeStr (ie_not_support_type_strs[i]);
      EXPECT_NE (ret_type, nns_support_types[i]);
    }
  }

  g_free (test_model_xml);
  g_free (test_model_bin);
}

/**
 * @brief A negative test case for the helper function, convertFromIETypeStr ()
 */
TEST (tensorFilterOpenvino, convertFromIETypeStr1_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const std::string ie_support_type_str ("Q78");
  std::string str_test_model;
  gchar *test_model_xml;
  gchar *test_model_bin;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);

  {
    TensorFilterOpenvinoTest tfOvTest (str_test_model.assign (test_model_xml),
        str_test_model.assign (test_model_bin));
    tensor_type ret_type;

    ret_type = tfOvTest.convertFromIETypeStr (ie_support_type_str);
    EXPECT_EQ (_NNS_END, ret_type);
  }

  g_free (test_model_xml);
  g_free (test_model_bin);
}

#define TEST_BLOB(prec, nns_type)                                                    \
  do {                                                                               \
    const InferenceEngine::Precision _prc (prec);                                    \
    InferenceEngine::TensorDesc tensorTestDesc (_prc, InferenceEngine::ANY);         \
    InferenceEngine::SizeVector dims (NNS_TENSOR_RANK_LIMIT);                        \
    TensorFilterOpenvinoTest tfOvTest (str_test_model.assign (test_model_xml),       \
        str_test_model.assign (test_model_bin));                                     \
    InferenceEngine::Blob::Ptr ret;                                                  \
    GstTensorMemory mem;                                                             \
                                                                                     \
    mem.size = gst_tensor_get_element_size (nns_type);                               \
    for (int i = 0; i < NNS_TENSOR_RANK_LIMIT; ++i) {                                \
      dims[i] = MOBINET_V2_IN_DIMS[i];                                               \
      mem.size *= MOBINET_V2_IN_DIMS[i];                                             \
    }                                                                                \
    tensorTestDesc.setDims (dims);                                                   \
    mem.data = (void *) g_malloc0 (mem.size);                                        \
                                                                                     \
    ret = tfOvTest.convertGstTensorMemoryToBlobPtr (tensorTestDesc, &mem, nns_type); \
    EXPECT_EQ (mem.size, ret->byteSize ());                                          \
    EXPECT_EQ (gst_tensor_get_element_size (nns_type), ret->element_size ());        \
    g_free (mem.data);                                                               \
  } while (0);

/**
 * @brief A test case for the helper function, convertFromIETypeStr ()
 */
TEST (tensorFilterOpenvino, convertGstTensorMemoryToBlobPtr0)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  std::string str_test_model;
  gchar *test_model_xml = NULL;
  gchar *test_model_bin = NULL;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  EXPECT_EQ (g_file_test (test_model_xml, G_FILE_TEST_IS_REGULAR), TRUE);

  test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);
  EXPECT_EQ (g_file_test (test_model_bin, G_FILE_TEST_IS_REGULAR), TRUE);

  TEST_BLOB (InferenceEngine::Precision::FP32, _NNS_FLOAT32);
  TEST_BLOB (InferenceEngine::Precision::U8, _NNS_UINT8);
  TEST_BLOB (InferenceEngine::Precision::U16, _NNS_UINT16);
  TEST_BLOB (InferenceEngine::Precision::I8, _NNS_INT8);
  TEST_BLOB (InferenceEngine::Precision::I16, _NNS_INT16);
  TEST_BLOB (InferenceEngine::Precision::I32, _NNS_INT32);

  g_free (test_model_xml);
  g_free (test_model_bin);
}

/**
 * @brief A test case checking that the tensor names of the model are kept
 */
TEST (tensorFilterOpenvino, getTensorNames0)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorsInfo nns_tensors_info;
  GstTensorInfo *_info;
  std::string str_test_model;
  gchar *test_model_xml;
  gchar *test_model_bin;
  guint i;
  gint ret;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);

  {
    TensorFilterOpenvinoTest tfOvTest (str_test_model.assign (test_model_xml),
        str_test_model.assign (test_model_bin));

    /** The names are known from the construction on, so that a run that does
     *  not call the dimension callbacks again still has them. */
    ASSERT_EQ (tfOvTest.getInputTensorNames ().size (), MOBINET_V2_IN_NUM_TENSOR);
    ASSERT_EQ (tfOvTest.getOutputTensorNames ().size (), MOBINET_V2_OUT_NUM_TENSOR);

    ret = tfOvTest.getInputTensorDim (&nns_tensors_info);
    ASSERT_EQ (ret, 0);
    ASSERT_EQ (tfOvTest.getInputTensorNames ().size (), nns_tensors_info.num_tensors);
    for (i = 0; i < nns_tensors_info.num_tensors; ++i) {
      _info = gst_tensors_info_get_nth_info (&nns_tensors_info, i);
      EXPECT_STREQ (tfOvTest.getInputTensorNames ()[i].c_str (), _info->name);
    }
    gst_tensors_info_free (&nns_tensors_info);

    ret = tfOvTest.getOutputTensorDim (&nns_tensors_info);
    ASSERT_EQ (ret, 0);
    ASSERT_EQ (tfOvTest.getOutputTensorNames ().size (), nns_tensors_info.num_tensors);
    for (i = 0; i < nns_tensors_info.num_tensors; ++i) {
      _info = gst_tensors_info_get_nth_info (&nns_tensors_info, i);
      EXPECT_STREQ (tfOvTest.getOutputTensorNames ()[i].c_str (), _info->name);
    }
    gst_tensors_info_free (&nns_tensors_info);
  }

  g_free (test_model_xml);
  g_free (test_model_bin);
}

/**
 * @brief A test case checking that the tensor descriptors of the model are kept
 */
TEST (tensorFilterOpenvino, getTensorDescs0)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  InferenceEngine::SizeVector dims;
  std::string str_test_model;
  gchar *test_model_xml;
  gchar *test_model_bin;
  size_t i;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);

  {
    TensorFilterOpenvinoTest tfOvTest (str_test_model.assign (test_model_xml),
        str_test_model.assign (test_model_bin));

    /** invoke () builds its blobs from these, so they are to be known from
     *  the construction on, as the names are. The Inference Engine orders
     *  the dimensions the other way around than NNStreamer does. */
    dims = tfOvTest.getInputTensorDesc (0).getDims ();
    ASSERT_GT (dims.size (), 0U);
    for (i = 0; i < dims.size (); ++i)
      EXPECT_EQ (dims[dims.size () - 1 - i], (size_t) MOBINET_V2_IN_DIMS[i]);

    dims = tfOvTest.getOutputTensorDesc (0).getDims ();
    ASSERT_GT (dims.size (), 0U);
    for (i = 0; i < dims.size (); ++i)
      EXPECT_EQ (dims[dims.size () - 1 - i], (size_t) MOBINET_V2_OUT_DIMS[i]);
  }

  g_free (test_model_xml);
  g_free (test_model_bin);
}

/**
 * @brief A test case for the helper function, getBlobName ()
 */
TEST (tensorFilterOpenvino, getBlobName0)
{
  const std::vector<std::string> model_names = { "model_0", "model_1" };
  GstTensorInfo info;
  std::string name;

  gst_tensor_info_init (&info);

  info.name = g_strdup ("user_given");
  EXPECT_TRUE (TensorFilterOpenvino::getBlobName (model_names, &info, 0, name));
  EXPECT_STREQ (name.c_str (), "user_given");
  g_free (info.name);

  /** The name a user does not give is the one the model declares */
  info.name = NULL;
  EXPECT_TRUE (TensorFilterOpenvino::getBlobName (model_names, &info, 0, name));
  EXPECT_STREQ (name.c_str (), "model_0");
  EXPECT_TRUE (TensorFilterOpenvino::getBlobName (model_names, &info, 1, name));
  EXPECT_STREQ (name.c_str (), "model_1");
}

/**
 * @brief A negative test case for the helper function, getBlobName ()
 */
TEST (tensorFilterOpenvino, getBlobName0_n)
{
  const std::vector<std::string> no_model_names;
  const std::vector<std::string> model_names = { "model_0" };
  GstTensorInfo info;
  std::string name;

  gst_tensor_info_init (&info);

  EXPECT_FALSE (TensorFilterOpenvino::getBlobName (no_model_names, &info, 0, name));
  EXPECT_FALSE (TensorFilterOpenvino::getBlobName (model_names, &info, 1, name));
}

/**
 * @brief A negative test case for the invoke callback with an unknown tensor name
 */
TEST (tensorFilterOpenvino, invoke0_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties *prop = NULL;
  GstTensorMemory input[MOBINET_V2_IN_NUM_TENSOR];
  GstTensorMemory output[MOBINET_V2_OUT_NUM_TENSOR] = {};
  GstTensorsInfo nns_tensors_info;
  GstTensorInfo *_info;
  std::string str_test_model;
  gchar *test_model_xml;
  gchar *test_model_bin;
  guint i;
  gint ret;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);

  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);

  {
    TensorFilterOpenvinoTest tfOvTest (str_test_model.assign (test_model_xml),
        str_test_model.assign (test_model_bin));

    ret = tfOvTest.getInputTensorDim (&nns_tensors_info);
    ASSERT_EQ (ret, 0);
    ASSERT_EQ (nns_tensors_info.num_tensors, MOBINET_V2_IN_NUM_TENSOR);
    gst_tensors_info_copy (&prop->input_meta, &nns_tensors_info);
    gst_tensors_info_free (&nns_tensors_info);

    /** A user who gives the dimensions and the types only leaves no names */
    for (i = 0; i < prop->input_meta.num_tensors; ++i) {
      _info = gst_tensors_info_get_nth_info (&prop->input_meta, i);
      g_free (_info->name);
      _info->name = NULL;
      input[i].size = gst_tensor_info_get_size (_info);
      input[i].data = g_malloc0 (input[i].size);
    }

    tfOvTest.setInputTensorNames (std::vector<std::string> ());

    ret = tfOvTest.invoke (prop, input, output);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);

    for (i = 0; i < prop->input_meta.num_tensors; ++i)
      g_free (input[i].data);
  }

  gst_tensors_info_free (&prop->input_meta);
  g_free (prop);
  g_free (test_model_xml);
  g_free (test_model_bin);
}

/**
 * @brief A negative test case for the invoke callback with an unknown output name
 */
TEST (tensorFilterOpenvino, invoke1_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties *prop = NULL;
  GstTensorMemory input[MOBINET_V2_IN_NUM_TENSOR] = {};
  GstTensorMemory output[MOBINET_V2_OUT_NUM_TENSOR];
  GstTensorsInfo nns_tensors_info;
  GstTensorInfo *_info;
  std::string str_test_model;
  gchar *test_model_xml;
  gchar *test_model_bin;
  guint i;
  gint ret;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extXml)
          .c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models", "models",
      str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
          .append (TensorFilterOpenvino::extBin)
          .c_str (),
      NULL);

  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);

  {
    TensorFilterOpenvinoTest tfOvTest (str_test_model.assign (test_model_xml),
        str_test_model.assign (test_model_bin));

    ret = tfOvTest.getOutputTensorDim (&nns_tensors_info);
    ASSERT_EQ (ret, 0);
    ASSERT_EQ (nns_tensors_info.num_tensors, MOBINET_V2_OUT_NUM_TENSOR);
    gst_tensors_info_copy (&prop->output_meta, &nns_tensors_info);
    gst_tensors_info_free (&nns_tensors_info);

    /** No input tensor is asked for, so that the output loop is the one run */
    for (i = 0; i < prop->output_meta.num_tensors; ++i) {
      _info = gst_tensors_info_get_nth_info (&prop->output_meta, i);
      g_free (_info->name);
      _info->name = NULL;
      output[i].size = gst_tensor_info_get_size (_info);
      output[i].data = g_malloc0 (output[i].size);
    }

    tfOvTest.setOutputTensorNames (std::vector<std::string> ());

    ret = tfOvTest.invoke (prop, input, output);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);

    for (i = 0; i < prop->output_meta.num_tensors; ++i)
      g_free (output[i].data);
  }

  gst_tensors_info_free (&prop->output_meta);
  g_free (prop);
  g_free (test_model_xml);
  g_free (test_model_bin);
}

#ifdef __OPENVINO_CPU_EXT__
/**
 * @brief A test case for the invoke callback when a user gives no tensor names
 */
TEST (tensorFilterOpenvino, invoke0)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  GstTensorMemory input[MOBINET_V2_IN_NUM_TENSOR];
  GstTensorMemory output[MOBINET_V2_OUT_NUM_TENSOR];
  GstTensorsInfo nns_tensors_info;
  GstTensorInfo *_info;
  gchar *test_model;
  guint i;
  gint ret;

  /* Check if mandatory methods are contained */
  ASSERT_TRUE (fw && fw->open && fw->close && fw->invoke_NN);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      MODEL_BASE_NAME_MOBINET_V2, NULL);
  const gchar *model_files[] = {
    test_model,
    NULL,
  };

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);
  prop->fwname = fw_name;
  prop->model_files = model_files;
  prop->num_models = 1;
  prop->accl_str = "true:cpu";

  ret = fw->open (prop, &private_data);
  ASSERT_EQ (ret, 0);

  ret = fw->getInputDimension (prop, &private_data, &nns_tensors_info);
  ASSERT_EQ (ret, 0);
  ASSERT_EQ (nns_tensors_info.num_tensors, MOBINET_V2_IN_NUM_TENSOR);
  gst_tensors_info_copy (&prop->input_meta, &nns_tensors_info);
  gst_tensors_info_free (&nns_tensors_info);

  ret = fw->getOutputDimension (prop, &private_data, &nns_tensors_info);
  ASSERT_EQ (ret, 0);
  ASSERT_EQ (nns_tensors_info.num_tensors, MOBINET_V2_OUT_NUM_TENSOR);
  gst_tensors_info_copy (&prop->output_meta, &nns_tensors_info);
  gst_tensors_info_free (&nns_tensors_info);

  /**
   * The framework keeps the tensor info given by a user as it is when the info
   * matches the model, and the comparison it uses ignores the names. A user who
   * gives the dimensions and the types only, therefore, leaves the names unset.
   */
  for (i = 0; i < prop->input_meta.num_tensors; ++i) {
    _info = gst_tensors_info_get_nth_info (&prop->input_meta, i);
    g_free (_info->name);
    _info->name = NULL;
    input[i].size = gst_tensor_info_get_size (_info);
    input[i].data = g_malloc0 (input[i].size);
  }
  for (i = 0; i < prop->output_meta.num_tensors; ++i) {
    _info = gst_tensors_info_get_nth_info (&prop->output_meta, i);
    g_free (_info->name);
    _info->name = NULL;
    output[i].size = gst_tensor_info_get_size (_info);
    output[i].data = g_malloc0 (output[i].size);
  }

  ret = fw->invoke_NN (prop, &private_data, input, output);
  EXPECT_EQ (ret, 0);

  /** A stop and a start of a pipeline replaces the instance, and the framework
   *  does not ask for the dimensions again once it has them configured. */
  fw->close (prop, &private_data);
  ret = fw->open (prop, &private_data);
  ASSERT_EQ (ret, 0);

  ret = fw->invoke_NN (prop, &private_data, input, output);
  EXPECT_EQ (ret, 0);

  for (i = 0; i < prop->input_meta.num_tensors; ++i)
    g_free (input[i].data);
  for (i = 0; i < prop->output_meta.num_tensors; ++i)
    g_free (output[i].data);

  fw->close (prop, &private_data);

  gst_tensors_info_free (&prop->input_meta);
  gst_tensors_info_free (&prop->output_meta);
  g_free (prop);
  g_free (test_model);
}
#endif /* __OPENVINO_CPU_EXT__ */

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
