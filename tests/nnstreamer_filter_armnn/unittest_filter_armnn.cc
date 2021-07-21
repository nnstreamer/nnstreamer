/**
 * @file        unittest_filter_armnn.cc
 * @date        13 Dec 2019
 * @brief       Unit test for armnn tensor filter plugin.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <fcntl.h>
#include <glib.h>
#include <glib/gstdio.h> /* GStatBuf */
#include <sys/stat.h>

#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>

/**
 * @brief Set tensor filter properties
 */
static void
_SetFilterProp (GstTensorFilterProperties *prop, const gchar *name, const gchar **models)
{
  memset (prop, 0, sizeof (GstTensorFilterProperties));
  prop->fwname = name;
  prop->fw_opened = 0;
  prop->model_files = models;
  prop->num_models = g_strv_length ((gchar **)models);
}


/**
 * @brief Test armnn subplugin existence.
 */
TEST (nnstreamerFilterArmnn, checkExistence)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *)NULL);
}

/**
 * @brief Test armnn subplugin with failing open/close (no model file)
 */
TEST (nnstreamerFilterArmnn, openClose00_n)
{
  int ret;
  const gchar *model_files[] = {
    "temp.armnn", NULL,
  };
  GstTensorFilterProperties prop;
  void *data = NULL;

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *)NULL);

  _SetFilterProp (&prop, "armnn", model_files);
  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);
}

/**
 * @brief Test armnn subplugin with successful open/close
 */
TEST (nnstreamerFilterArmnn, openClose01_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file, NULL,
  };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *)NULL);
  _SetFilterProp (&prop, "armnn", model_files);

  /** close without open, should not crash */
  sp->close (&prop, &data);

  /** open and close successfully */
  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  sp->close (&prop, &data);

  /** close twice, should not crash */
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Get input/output dimensions with armnn subplugin
 */
TEST (nnstreamerFilterArmnn, getDimension)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorsInfo info, res;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file, NULL,
  };

  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *)NULL);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  info.num_tensors = 1;
  info.info[0].type = _NNS_FLOAT32;
  info.info[0].dimension[0] = 1;
  info.info[0].dimension[1] = 1;
  info.info[0].dimension[2] = 1;
  info.info[0].dimension[3] = 1;

  /** get input/output dimension successfully */
  ret = sp->getInputDimension (&prop, &data, NULL);
  EXPECT_NE (ret, 0);
  ret = sp->getInputDimension (&prop, &data, &res);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (res.num_tensors, info.num_tensors);
  EXPECT_EQ (res.info[0].type, info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], info.info[0].dimension[3]);

  ret = sp->getOutputDimension (&prop, &data, NULL);
  EXPECT_NE (ret, 0);
  ret = sp->getOutputDimension (&prop, &data, &res);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (res.num_tensors, info.num_tensors);
  EXPECT_EQ (res.info[0].type, info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], info.info[0].dimension[3]);

  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Get input dimensions before open
 */
TEST (nnstreamerFilterArmnn, getDimension1_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorsInfo res;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file, NULL,
  };

  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *)NULL);

  /** get input dimension without open */
  ret = sp->getInputDimension (&prop, &data, &res);
  EXPECT_NE (ret, 0);

  g_free (model_file);
}

/**
 * @brief Get output dimensions before open
 */
TEST (nnstreamerFilterArmnn, getDimension2_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorsInfo res;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file, NULL,
  };
  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *)NULL);

  /** get output dimension without open */
  ret = sp->getOutputDimension (&prop, &data, &res);
  EXPECT_NE (ret, 0);

  g_free (model_file);
}

/**
 * @brief Test armnn subplugin with successful invoke for tflite
 */
TEST (nnstreamerFilterArmnn, invoke00)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file, NULL,
  };

  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *)NULL);

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke successful */
  *((float *)input.data) = 10.0;
  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (*((float *)output.data), 12.0);

  *((float *)input.data) = 1.0;
  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (*((float *)output.data), 3.0);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Test armnn invoke before open
 */
TEST (nnstreamerFilterArmnn, invoke01_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file, NULL,
  };

  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *)NULL);

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  /** invoke before open */
  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_NE (ret, 0);

  g_free (model_file);
}

/**
 * @brief Test armnn invoke with invalid param
 */
TEST (nnstreamerFilterArmnn, invoke02_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorMemory input, output;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file, NULL,
  };

  _SetFilterProp (&prop, "armnn", model_files);
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *)NULL);

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke without subplugin data */
  ret = sp->invoke_NN (&prop, NULL, &input, &output);
  EXPECT_NE (ret, 0);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief Test armnn invoke with invalid param
 */
TEST (nnstreamerFilterArmnn, invoke03_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorMemory output;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file, NULL,
  };
  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *)NULL);

  output.size = sizeof (float) * 1;

  output.data = g_malloc (output.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke without input */
  ret = sp->invoke_NN (&prop, &data, NULL, &output);
  EXPECT_NE (ret, 0);

  g_free (output.data);
  sp->close (&prop, &data);

  g_free (model_file);
}

/**
 * @brief Test armnn invoke with invalid param
 */
TEST (nnstreamerFilterArmnn, invoke04_n)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorMemory input;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (
      root_path, "tests", "test_models", "models", "add.tflite", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file, NULL,
  };
  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *)NULL);

  input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke without output */
  ret = sp->invoke_NN (&prop, &data, &input, NULL);
  EXPECT_NE (ret, 0);

  g_free (input.data);
  sp->close (&prop, &data);
  g_free (model_file);
}

/**
 * @brief get argmax from the array
 */
template <typename T>
size_t
get_argmax (T *array, size_t size)
{
  size_t idx, max_idx = 0;
  T max_value = 0;
  for (idx = 0; idx < size; idx++) {
    if (max_value < array[idx]) {
      max_idx = idx;
      max_value = array[idx];
    }
  }

  return max_idx;
}

/**
 * @brief Test armnn subplugin with successful invoke for tflite advanced model
 */
TEST (nnstreamerFilterArmnn, invokeAdvanced)
{
  int ret, fd;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  GstTensorMemory input, output;
  GstTensorsInfo info, res;
  char *data_file;
  ssize_t data_read;
  size_t max_idx;
  gboolean status;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  status = g_file_test (model_file, G_FILE_TEST_EXISTS);
  if (!status) {
    g_free (model_file);
    ASSERT_EQ (status, TRUE);
    return;
  }

  const gchar *model_files[] = {
    model_file, NULL,
  };
  _SetFilterProp (&prop, "armnn", model_files);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *)NULL);

  info.num_tensors = 1;
  info.info[0].type = _NNS_UINT8;
  info.info[0].dimension[0] = 3;
  info.info[0].dimension[1] = 224;
  info.info[0].dimension[2] = 224;
  info.info[0].dimension[3] = 1;

  ret = sp->open (&prop, &data);
  if (ret == -EPERM) { /** TFLite Parser is not included in this instance */
    g_free (model_file);
    return;
  }
  EXPECT_EQ (ret, 0);

  /** get input/output dimension successfully */
  ret = sp->getInputDimension (&prop, &data, &res);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (res.num_tensors, info.num_tensors);
  EXPECT_EQ (res.info[0].type, info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], info.info[0].dimension[3]);

  input.size = gst_tensor_info_get_size (&res.info[0]);

  ret = sp->getOutputDimension (&prop, &data, &res);
  EXPECT_EQ (ret, 0);

  info.num_tensors = 1;
  info.info[0].type = _NNS_UINT8;
  info.info[0].dimension[0] = 1001;
  info.info[0].dimension[1] = 1;
  info.info[0].dimension[2] = 1;
  info.info[0].dimension[3] = 1;

  EXPECT_EQ (res.num_tensors, info.num_tensors);
  EXPECT_EQ (res.info[0].type, info.info[0].type);
  EXPECT_EQ (res.info[0].dimension[0], info.info[0].dimension[0]);
  EXPECT_EQ (res.info[0].dimension[1], info.info[0].dimension[1]);
  EXPECT_EQ (res.info[0].dimension[2], info.info[0].dimension[2]);
  EXPECT_EQ (res.info[0].dimension[3], info.info[0].dimension[3]);

  output.size = gst_tensor_info_get_size (&res.info[0]);

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  data_file = g_build_filename (
      root_path, "tests", "test_models", "data", "orange.raw", NULL);
  fd = open (data_file, O_RDONLY);

  EXPECT_TRUE (fd >= 0);
  /** Invoke output will not match with fd < 0 - input will be random data */
  if (fd >= 0) {
    data_read = read (fd, input.data, input.size);
    EXPECT_EQ (data_read, input.size);
    close (fd);
  }

  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_EQ (ret, 0);

  /** entry 952 (idx 951) is orange as per tests/test_models/labels/labels.txt
   */
  max_idx = get_argmax<guint8> ((guint8 *)output.data, output.size);
  EXPECT_EQ (max_idx, 951);

  g_free (data_file);
  g_free (output.data);
  g_free (input.data);
  g_free (model_file);
  sp->close (&prop, &data);
}

/**
 * @brief Test armnn subplugin with successful invoke for caffe
 */
TEST (nnstreamerFilterArmnn, invoke01)
{
  int ret;
  void *data = NULL;
  gchar *model_file, *data_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *input_uint8_data = NULL;
  gsize input_uint8_size = 0;
  GstTensorMemory output, input;
  ssize_t max_idx;
  const unsigned int num_labels = 10;
  GstTensorFilterProperties prop;
  ASSERT_NE (root_path, nullptr);

  /** armnn needs a directory with model file and metadata in that directory */
  model_file = g_build_filename (root_path, "tests", "test_models", "models",
      "lenet_iter_9000.caffemodel", NULL);
  ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

  data_file = g_build_filename (root_path, "tests", "test_models", "data", "9.raw", NULL);
  ASSERT_TRUE (g_file_test (data_file, G_FILE_TEST_EXISTS));

  const gchar *model_files[] = {
    model_file, NULL,
  };
  _SetFilterProp (&prop, "armnn", model_files);

  /** Manually configure the input for test */
  gst_tensors_info_init (&prop.input_meta);
  gst_tensors_info_init (&prop.output_meta);
  prop.output_meta.num_tensors = 1;
  prop.output_meta.info[0].type = _NNS_FLOAT32;
  prop.output_meta.info[0].name = g_strdup ("prob");
  prop.input_meta.num_tensors = 1;
  prop.input_meta.info[0].name = g_strdup ("data");

  EXPECT_TRUE (g_file_get_contents (data_file, &input_uint8_data, &input_uint8_size, NULL));

  /** Convert the data from uint8 to float */
  input.size = input_uint8_size * gst_tensor_get_element_size (_NNS_FLOAT32);
  input.data = g_malloc (input.size);
  for (gsize idx = 0; idx < input_uint8_size; idx++) {
    ((float *)input.data)[idx] = static_cast<float> (((guint8 *)input_uint8_data)[idx]);
    ((float *)input.data)[idx] -= 127.5;
    ((float *)input.data)[idx] /= 127.5;
  }

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("armnn");
  EXPECT_NE (sp, (void *)NULL);

  output.size = gst_tensor_get_element_size (_NNS_FLOAT32) * num_labels;
  output.data = g_malloc (output.size);

  ret = sp->open (&prop, &data);
  if (ret == -EPERM) { /** Caffe Parser Not Included in This System */
    g_free (data_file);
    g_free (input.data);
    g_free (input_uint8_data);
    g_free (output.data);
    g_free (prop.output_meta.info[0].name);
    g_free (prop.input_meta.info[0].name);
    g_free (model_file);
    return;
  }
  EXPECT_EQ (ret, 0);

  /** invoke successful */
  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_EQ (ret, 0);

  max_idx = get_argmax<float> ((float *)output.data, num_labels);
  EXPECT_EQ (max_idx, 9);

  sp->close (&prop, &data);

  /** Run the test again but with setting input meta as well */
  gst_tensors_info_init (&prop.input_meta);
  gst_tensors_info_init (&prop.output_meta);
  prop.output_meta.num_tensors = 1;
  prop.output_meta.info[0].type = _NNS_FLOAT32;
  prop.output_meta.info[0].name = g_strdup ("prob");

  prop.input_meta.num_tensors = 1;
  prop.input_meta.info[0].type = _NNS_FLOAT32;
  prop.input_meta.info[0].name = g_strdup ("data");
  prop.input_meta.info[0].dimension[0] = 28;
  prop.input_meta.info[0].dimension[1] = 28;
  prop.input_meta.info[0].dimension[2] = 1;
  prop.input_meta.info[0].dimension[3] = 1;

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /** invoke successful */
  ret = sp->invoke_NN (&prop, &data, &input, &output);
  EXPECT_EQ (ret, 0);

  max_idx = get_argmax<float> ((float *)output.data, num_labels);
  EXPECT_EQ (max_idx, 9);

  sp->close (&prop, &data);

  g_free (data_file);
  g_free (input.data);
  g_free (input_uint8_data);
  g_free (output.data);
  g_free (prop.output_meta.info[0].name);
  g_free (prop.input_meta.info[0].name);
  g_free (model_file);
}

/**
 * @brief Main gtest
 */
int
main (int argc, char **argv)
{
  int result = -1;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
