#include <gtest/gtest.h>
#include <gst/gst.h>
#include <gst/check/gstcheck.h>
#include <gst/check/gsttestclock.h>
#include <gst/check/gstharness.h>
#include <glib/gstdio.h>
#include <nnstreamer_plugin_api_filter.h>
#include <string.h>
#include <tensor_common.h>

#include <tensor_filter_openvino.hh>

const static gchar MODEL_BASE_NAME_MOBINET_V2[] =
    "openvino_mobilenetv2-int8-tf-0001";

/**
 * @brief Test cases for open and close callbacks varying the model files
 */
TEST (tensor_filter_openvino, open_and_close_0)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  std::string str_test_model;
  gchar *test_model;
  gint ret;

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

    ASSERT_TRUE (fw && fw->open && fw->close);

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
    gchar *test_model_xml = g_build_filename (root_path, "tests", "test_models",
        "models", str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
        .append (TensorFilterOpenvino::extXml).c_str (),
        NULL);
    gchar *test_model_bin = g_build_filename (root_path, "tests", "test_models",
        "models", str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
        .append (TensorFilterOpenvino::extBin).c_str (),
        NULL);
    const gchar *model_files[] = {
      test_model_xml, test_model_bin,
    };

    prop->num_models = 2;
    prop->model_files = model_files;

    ASSERT_TRUE (fw && fw->open && fw->close);

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
      .append (TensorFilterOpenvino::extBin).c_str (),
      NULL);
  {
    const gchar *model_files[] = {
      test_model, NULL,
    };

    prop->num_models = 1;
    prop->model_files = model_files;

    ASSERT_TRUE (fw && fw->open && fw->close);

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
      .append (TensorFilterOpenvino::extXml).c_str (),
      NULL);
  {
    const gchar *model_files[] = {
      test_model, NULL,
    };

    prop->num_models = 1;
    prop->model_files = model_files;

    ASSERT_TRUE (fw && fw->open && fw->close);

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
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  std::string str_test_model;
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  TensorFilterOpenvino *tfOv;
  gchar *test_model_xml;
  gchar *test_model_bin;
  gint ret;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models",
      "models", str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
      .append (TensorFilterOpenvino::extXml).c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models",
      "models", str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
      .append (TensorFilterOpenvino::extBin).c_str (),
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
      test_model_xml, test_model_bin,
    };

    prop->model_files = model_files;

    ASSERT_TRUE (fw && fw->open && fw->close);

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
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  std::string str_test_model;
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  TensorFilterOpenvino *tfOv;
  gchar *test_model_xml;
  gchar *test_model_bin;
  gint ret;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model_xml = g_build_filename (root_path, "tests", "test_models",
      "models", str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
      .append (TensorFilterOpenvino::extXml).c_str (),
      NULL);
  test_model_bin = g_build_filename (root_path, "tests", "test_models",
      "models", str_test_model.assign (MODEL_BASE_NAME_MOBINET_V2)
      .append (TensorFilterOpenvino::extBin).c_str (),
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
      test_model_xml, test_model_bin,
    };

    prop->model_files = model_files;

    ASSERT_TRUE (fw && fw->open && fw->close);

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
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  gchar *test_model;
  gint ret;

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  test_model = g_build_filename (root_path, "tests", "test_models", "models",
      "NOT_EXIST", NULL);
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

  ASSERT_TRUE (fw && fw->open && fw->close);

  ret = fw->open (prop, &private_data);
  EXPECT_NE (ret, TensorFilterOpenvino::RetSuccess);
  EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);

  g_free (test_model);
  fw->close (prop, &private_data);

  {
    std::string str_test_model;
    gchar *test_model_xml1 = g_build_filename (root_path, "tests",
        "test_models", "models", str_test_model
        .assign (MODEL_BASE_NAME_MOBINET_V2)
        .append (TensorFilterOpenvino::extXml).c_str (),
        NULL);
    gchar *test_model_xml2 = g_build_filename (root_path, "tests",
        "test_models", "models", str_test_model
        .assign (MODEL_BASE_NAME_MOBINET_V2)
        .append (TensorFilterOpenvino::extXml).c_str (),
        NULL);
    const gchar *model_files[] = {
      test_model_xml1, test_model_xml2,
    };

    prop->num_models = 2;
    prop->model_files = model_files;

    ASSERT_TRUE (fw && fw->open && fw->close);

    ret = fw->open (prop, &private_data);
    EXPECT_NE (ret, TensorFilterOpenvino::RetSuccess);
    EXPECT_EQ (ret, TensorFilterOpenvino::RetEInval);

    fw->close (prop, &private_data);
    g_free (test_model_xml1);
    g_free (test_model_xml2);
  }

  {
    std::string str_test_model;
    gchar *test_model_bin1 = g_build_filename (root_path, "tests",
        "test_models", "models", str_test_model
        .assign (MODEL_BASE_NAME_MOBINET_V2)
        .append (TensorFilterOpenvino::extBin).c_str (),
        NULL);
    gchar *test_model_bin2 = g_build_filename (root_path, "tests",
      "test_models", "models", str_test_model
      .assign (MODEL_BASE_NAME_MOBINET_V2)
      .append (TensorFilterOpenvino::extBin).c_str (),
      NULL);
    const gchar *model_files[] = {
      test_model_bin1, test_model_bin2,
    };

    prop->num_models = 2;
    prop->model_files = model_files;

    ASSERT_TRUE (fw && fw->open && fw->close);

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
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  gchar *test_model;
  gint ret;

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

  ASSERT_TRUE (fw && fw->open && fw->close);

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
  const gchar *root_path = g_getenv ("NNSTREAMER_BUILD_ROOT_PATH");
  const gchar fw_name[] = "openvino";
  const GstTensorFilterFramework *fw = nnstreamer_filter_find (fw_name);
  GstTensorFilterProperties *prop = NULL;
  gpointer private_data = NULL;
  gchar *test_model;
  gint ret;

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

  ASSERT_TRUE (fw && fw->open && fw->close);

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
