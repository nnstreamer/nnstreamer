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
const static uint32_t MOBINET_V2_IN_DIMS[NNS_TENSOR_SIZE_LIMIT] = {
  224, 224, 3, 1,
};
const static uint32_t MOBINET_V2_OUT_NUM_TENSOR = 1;
const static uint32_t MOBINET_V2_OUT_DIMS[NNS_TENSOR_SIZE_LIMIT] = {
  1001, 1, 1, 1,
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

/**
 * @brief Test cases for open and close callbacks varying the model files
 */
TEST (tensor_filter_openvino, open_and_close_0)
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
      test_model, NULL,
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
      test_model_xml, test_model_bin,
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
      test_model, NULL,
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
      test_model, NULL,
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
TEST (tensor_filter_openvino, open_and_close_1)
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
  private_data = (gpointer)tfOv;

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);
  prop->fwname = fw_name;
  prop->num_models = 2;
  prop->accl_str = "true:cpu";
  {
    const gchar *model_files[] = {
      test_model_xml, test_model_bin,
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
TEST (tensor_filter_openvino, open_and_close_2)
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
  private_data = (gpointer)tfOv;

  /* prepare properties */
  prop = g_new0 (GstTensorFilterProperties, 1);
  ASSERT_TRUE (prop != NULL);
  prop->fwname = fw_name;
  prop->num_models = 2;
  prop->accl_str = "true:cpu";
  {
    const gchar *model_files[] = {
      test_model_xml, test_model_bin,
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
TEST (tensor_filter_openvino, open_and_close_0_n)
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
    test_model, NULL,
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
      test_model_xml1, test_model_xml2,
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
      test_model_bin1, test_model_bin2,
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
TEST (tensor_filter_openvino, open_and_close_1_n)
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
    test_model, NULL,
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
TEST (tensor_filter_openvino, open_and_close_2_n)
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
    test_model, NULL,
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
 * @brief Test cases for getInputTensorDim and getOutputTensorDim callbacks
 */
TEST (tensor_filter_openvino, getTensorDim_0)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  GstTensorsInfo nns_tensors_info;
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
      test_model, NULL,
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
    for (uint32_t j = 0; j < NNS_TENSOR_RANK_LIMIT; ++j) {
      EXPECT_EQ (nns_tensors_info.info[i].dimension[j], MOBINET_V2_IN_DIMS[j]);
    }
  }

  /* Test getOutputDimension () */
  ASSERT_TRUE (fw->getOutputDimension);
  ret = fw->getOutputDimension (prop, &private_data, &nns_tensors_info);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (nns_tensors_info.num_tensors, MOBINET_V2_OUT_NUM_TENSOR);
  for (uint32_t i = 0; i < MOBINET_V2_OUT_NUM_TENSOR; ++i) {
    for (uint32_t j = 0; j < NNS_TENSOR_RANK_LIMIT; ++j) {
      EXPECT_EQ (nns_tensors_info.info[i].dimension[j], MOBINET_V2_OUT_DIMS[j]);
    }
  }

  fw->close (prop, &private_data);

  g_free (test_model);

  g_free (prop);
}

/**
 * @brief A negative test case for getInputTensorDim callbacks (The number of tensors is exceeded NNS_TENSOR_SIZE_LIMIT)
 */
TEST (tensor_filter_openvino, getTensorDim_0_n)
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
      std::string name_input_n = std::string ((char *)&i);
      inDataMap[name_input_n] = InferenceEngine::InputInfo::Ptr (info);
    }

    tfOvTest.setInputsDataMap (inDataMap);
    ret = tfOvTest.loadModel (ACCL_CPU);
    private_data = (gpointer)&tfOvTest;

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
TEST (tensor_filter_openvino, getTensorDim_1_n)
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
    private_data = (gpointer)&tfOvTest;

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
TEST (tensor_filter_openvino, getTensorDim_2_n)
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
      std::string name_output_n = std::string ((char *)&i);
      InferenceEngine::Data *data = new InferenceEngine::Data (
          name_output_n, dims, InferenceEngine::Precision::FP32);
      InferenceEngine::DataPtr outputDataPtr (data);

      outDataMap[name_output_n] = outputDataPtr;
    }

    tfOvTest.setOutputsDataMap (outDataMap);
    ret = tfOvTest.loadModel (ACCL_CPU);
    private_data = (gpointer)&tfOvTest;

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
TEST (tensor_filter_openvino, getTensorDim_3_n)
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
    private_data = (gpointer)&tfOvTest;

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
TEST (tensor_filter_openvino, convertFromIETypeStr_0)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const std::vector<std::string> ie_suport_type_strs = {
    "I8", "I16", "I32", "U8", "U16", "FP32",
  };
  const std::vector<tensor_type> nns_support_types = {
    _NNS_INT8, _NNS_INT16, _NNS_INT32, _NNS_UINT8, _NNS_UINT16, _NNS_FLOAT32,
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
    for (size_t i = 0; i < ie_suport_type_strs.size (); ++i) {
      tensor_type ret_type;

      ret_type = tfOvTest.convertFromIETypeStr (ie_suport_type_strs[i]);
      EXPECT_EQ (ret_type, nns_support_types[i]);
    }
  }

  g_free (test_model_xml);
  g_free (test_model_bin);
}

/**
 * @brief A negative test case for the helper function, convertFromIETypeStr ()
 */
TEST (tensor_filter_openvino, convertFromIETypeStr_0_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const std::vector<std::string> ie_not_suport_type_strs = {
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
    for (size_t i = 0; i < ie_not_suport_type_strs.size (); ++i) {
      tensor_type ret_type;
      ret_type = tfOvTest.convertFromIETypeStr (ie_not_suport_type_strs[i]);
      EXPECT_NE (ret_type, nns_support_types[i]);
    }
  }

  g_free (test_model_xml);
  g_free (test_model_bin);
}

/**
 * @brief A negative test case for the helper function, convertFromIETypeStr ()
 */
TEST (tensor_filter_openvino, convertFromIETypeStr_1_n)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  const std::string ie_suport_type_str ("Q78");
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

    ret_type = tfOvTest.convertFromIETypeStr (ie_suport_type_str);
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
    mem.data = (void *)g_malloc0 (mem.size);                                         \
                                                                                     \
    ret = tfOvTest.convertGstTensorMemoryToBlobPtr (tensorTestDesc, &mem, nns_type); \
    EXPECT_EQ (mem.size, ret->byteSize ());                                          \
    EXPECT_EQ (gst_tensor_get_element_size (nns_type), ret->element_size ());        \
    g_free (mem.data);                                                               \
  } while (0);

/**
 * @brief A test case for the helper function, convertFromIETypeStr ()
 */
TEST (tensor_filter_openvino, convertGstTensorMemoryToBlobPtr_0)
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
