/**
 * @file        unittest_filter_single.cc
 * @date        30 Aug 2021
 * @brief       Unit test for tensor_filter_single
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <nnstreamer_plugin_api_util.h>

#include "../gst/nnstreamer/tensor_filter/tensor_filter_single.h"

#if defined(ENABLE_TENSORFLOW_LITE) || defined(ENABLE_TENSORFLOW2_LITE)
/**
 * @brief Test Fixture class for a tensor-filter single functionality.
 */
class NNSFilterSingleTest : public ::testing::Test
{
  protected:
  GTensorFilterSingle *single;
  GTensorFilterSingleClass *klass;
  GstTensorMemory input;
  GstTensorMemory output;
  gboolean loaded;

  public:
  /**
   * @brief Construct a new NNSFilterSingleTest object
   */
  NNSFilterSingleTest () : single (nullptr), klass (nullptr), loaded (FALSE)
  {
    input.data = output.data = nullptr;
    input.size = output.size = 0;
  }

  /**
   * @brief SetUp method for each test case
   */
  void SetUp () override
  {
    g_autofree gchar *model_file = nullptr;
    g_autofree gchar *data_file = nullptr;
    gsize length;
    const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

    input.size = 3U * 224 * 224;
    output.size = 1001U;

    single = (GTensorFilterSingle *) g_object_new (G_TYPE_TENSOR_FILTER_SINGLE, NULL);
    klass = (GTensorFilterSingleClass *) g_type_class_ref (G_TYPE_TENSOR_FILTER_SINGLE);

    /* supposed to run test in build directory */
    if (root_path == NULL)
      root_path = "..";

    model_file = g_build_filename (root_path, "tests", "test_models", "models",
        "mobilenet_v1_1.0_224_quant.tflite", NULL);
    ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

    data_file = g_build_filename (
        root_path, "tests", "test_models", "data", "orange.raw", NULL);
    ASSERT_TRUE (g_file_get_contents (data_file, (gchar **) &input.data, &length, NULL));
    ASSERT_TRUE (length == input.size);

    g_object_set (G_OBJECT (single), "framework", "tensorflow-lite", "model",
        model_file, NULL);
    loaded = TRUE;
  }

  /**
   * @brief TearDown method for each test case
   */
  void TearDown () override
  {
    g_type_class_unref (klass);
    g_object_unref (single);
    g_free (input.data);
    g_free (output.data);
  }

  /**
   * @brief Get the max score in the given array.
   * @param tensorMemory tensor memory which contains the target array.
   * @return gsize max value in the given array
   */
  gsize get_max_score (GstTensorMemory *tensorMemory)
  {
    guint8 *array;
    gsize idx, max_idx = 0;
    guint8 max_value = 0;

    array = (guint8 *) tensorMemory->data;

    for (idx = 0; idx < tensorMemory->size; idx++) {
      if (max_value < array[idx]) {
        max_idx = idx;
        max_value = array[idx];
      }
    }

    return max_idx;
  }
};

/**
 * @brief Test to invoke tf-lite model.
 */
TEST_F (NNSFilterSingleTest, invoke_p)
{
  ASSERT_TRUE (this->loaded);

  /* invoke the model and check label 'orange' (index 951) */
  EXPECT_TRUE (klass->invoke (single, &input, &output, TRUE));
  EXPECT_EQ (951U, get_max_score (&output));

  EXPECT_TRUE (klass->invoke (single, &input, &output, FALSE));
  EXPECT_EQ (951U, get_max_score (&output));

  EXPECT_TRUE (klass->input_configured (single));
  EXPECT_TRUE (klass->output_configured (single));
  EXPECT_FALSE (klass->allocate_in_invoke (single));

  klass->destroy_notify (single, &output);
}

/**
 * @brief Test to invoke tf-lite model with invalid param.
 */
TEST_F (NNSFilterSingleTest, invokeInvalidParam_n)
{
  ASSERT_TRUE (this->loaded);
  EXPECT_TRUE (klass->start (single));

  EXPECT_FALSE (klass->invoke (single, NULL, &output, FALSE));
  EXPECT_FALSE (klass->invoke (single, &input, NULL, FALSE));

  EXPECT_TRUE (klass->stop (single));
}

/**
 * @brief Test to invoke tf-lite model with invalid data.
 */
TEST_F (NNSFilterSingleTest, invokeInvalidSize_n)
{
  ASSERT_TRUE (this->loaded);
  EXPECT_TRUE (klass->start (single));

  /* request allocation with invalid size */
  output.size = 0;
  EXPECT_FALSE (klass->invoke (single, &input, &output, TRUE));

  EXPECT_TRUE (klass->stop (single));
}

/**
 * @brief Test to set invalid info.
 */
TEST_F (NNSFilterSingleTest, setInvalidInfo_n)
{
  GstTensorsInfo in_info, out_info;
  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  ASSERT_TRUE (this->loaded);
  EXPECT_TRUE (klass->start (single));

  /* valid tensor info */
  in_info.num_tensors = 1U;
  in_info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:224:224:1", in_info.info[0].dimension);
  EXPECT_TRUE (klass->set_input_info (single, &in_info, &out_info) == 0);

  /* request to set invalid tensor info */
  gst_tensor_parse_dimension ("1:1:1:1", in_info.info[0].dimension);
  EXPECT_FALSE (klass->set_input_info (single, &in_info, &out_info) == 0);

  EXPECT_TRUE (klass->stop (single));
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Test for filter property.
 */
TEST_F (NNSFilterSingleTest, invalidProperty_n)
{
  g_autofree gchar *string_val = NULL;

  ASSERT_TRUE (this->loaded);

  /* invalid property name */
  g_object_set (G_OBJECT (single), "framework-invalid", "invalid", NULL);
  g_object_get (G_OBJECT (single), "framework-invalid", &string_val, NULL);
  EXPECT_FALSE (string_val != NULL);

  g_object_get (G_OBJECT (single), "framework", &string_val, NULL);
  EXPECT_TRUE (string_val && g_str_equal (string_val, "tensorflow-lite"));
}
#endif /* ENABLE_TENSORFLOW_LITE */

#ifdef ENABLE_TENSORFLOW2_LITE
/**
 * @brief Test Fixture class for a tensor-filter single functionality with high rank.
 */
class NNSFilterSingleTestExtended : public ::testing::Test
{
  protected:
  GTensorFilterSingle *single;
  GTensorFilterSingleClass *klass;
  GstTensorMemory input[2];
  GstTensorMemory output;
  gboolean loaded;

  public:
  /**
   * @brief Construct a new NNSFilterSingleTestExtended object
   */
  NNSFilterSingleTestExtended ()
      : single (nullptr), klass (nullptr), loaded (FALSE)
  {
    input[0].data = input[1].data = output.data = nullptr;
    input[0].size = input[1].size = output.size = 0;
  }

  /**
   * @brief SetUp method for each test case
   */
  void SetUp () override
  {
    g_autofree gchar *model_file = nullptr;
    gsize length = 4 * 4 * 4 * 4 * 4;
    guint i;
    const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

    input[0].size = length * 4;
    input[1].size = length * 4;
    output.size = length * 4;

    single = (GTensorFilterSingle *) g_object_new (G_TYPE_TENSOR_FILTER_SINGLE, NULL);
    klass = (GTensorFilterSingleClass *) g_type_class_ref (G_TYPE_TENSOR_FILTER_SINGLE);

    /* supposed to run test in build directory */
    if (root_path == NULL)
      root_path = "..";

    model_file = g_build_filename (root_path, "tests", "test_models", "models",
        "sample_4x4x4x4x4_two_input_one_output.tflite", NULL);
    ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));

    input[0].data = g_malloc0 (sizeof (gfloat) * length);
    input[1].data = g_malloc0 (sizeof (gfloat) * length);

    for (i = 0; i < length; i++) {
      ((gfloat *) input[0].data)[i] = i;
      ((gfloat *) input[1].data)[i] = i + 1;
    }

    g_object_set (G_OBJECT (single), "framework", "tensorflow-lite", "model",
        model_file, NULL);
    loaded = TRUE;
  }

  /**
   * @brief TearDown method for each test case
   */
  void TearDown () override
  {
    g_type_class_unref (klass);
    g_object_unref (single);
    g_free (input[0].data);
    g_free (input[1].data);
    g_free (output.data);
  }
};

/**
 * @brief Test to invoke tf-lite model.
 */
TEST_F (NNSFilterSingleTestExtended, invoke_p)
{
  guint i, length = 4 * 4 * 4 * 4 * 4;
  ASSERT_TRUE (this->loaded);

  /* invoke the model and check output result */
  EXPECT_TRUE (klass->invoke (single, input, &output, TRUE));

  for (i = 0; i < length; i++)
    EXPECT_EQ (((gfloat *) output.data)[i],
        (((gfloat *) input[0].data)[i] + ((gfloat *) input[1].data)[i]));

  EXPECT_TRUE (klass->input_configured (single));
  EXPECT_TRUE (klass->output_configured (single));
  EXPECT_FALSE (klass->allocate_in_invoke (single));

  klass->destroy_notify (single, &output);
}

/**
 * @brief Test to set invalid info.
 */
TEST_F (NNSFilterSingleTestExtended, setInvalidInfo_n)
{
  GstTensorsInfo in_info, out_info;
  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  ASSERT_TRUE (this->loaded);
  EXPECT_TRUE (klass->start (single));

  /* valid tensor info */
  in_info.num_tensors = 2U;
  in_info.info[0].type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("4:4:4:4:4", in_info.info[0].dimension);
  in_info.info[1].type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("4:4:4:4:4", in_info.info[1].dimension);
  EXPECT_TRUE (klass->set_input_info (single, &in_info, &out_info) == 0);

  /* request to set invalid tensor info */
  in_info.num_tensors = 1U;
  EXPECT_FALSE (klass->set_input_info (single, &in_info, &out_info) == 0);

  EXPECT_TRUE (klass->stop (single));
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Test Fixture class for a tensor-filter-single with 32 in/out model.
 */
class NNSFilterSingleTestManyInOut : public ::testing::Test
{
  protected:
  GTensorFilterSingle *single;
  GTensorFilterSingleClass *klass;
  GstTensorMemory input[32];
  GstTensorMemory output[32];
  gboolean loaded;

  public:
  /**
   * @brief Constructor
   */
  NNSFilterSingleTestManyInOut ()
      : single (nullptr), klass (nullptr), loaded (FALSE)
  {
    for (guint i = 0; i < 32; i++) {
      input[i].data = output[i].data = nullptr;
      input[i].size = output[i].size = 0;
    }
  }

  /**
   * @brief SetUp method for each test case
   */
  void SetUp () override
  {
    g_autofree gchar *model_file = nullptr;
    guint num_tensors = 32;
    const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");


    single = (GTensorFilterSingle *) g_object_new (G_TYPE_TENSOR_FILTER_SINGLE, NULL);
    klass = (GTensorFilterSingleClass *) g_type_class_ref (G_TYPE_TENSOR_FILTER_SINGLE);

    /* supposed to run test in build directory */
    if (root_path == NULL)
      root_path = "..";

    model_file = g_build_filename (root_path, "tests", "test_models", "models",
        "simple_32_in_32_out.tflite", NULL);
    ASSERT_TRUE (g_file_test (model_file, G_FILE_TEST_EXISTS));


    for (guint i = 0; i < num_tensors; i++) {
      input[i].size = 1 * sizeof (float);
      input[i].data = g_malloc0 (1 * sizeof (float));

      ((gfloat *) input[i].data)[0] = 16.0f;

      output[i].size = 1 * sizeof (float);
    }

    g_object_set (G_OBJECT (single), "framework", "tensorflow-lite", "model",
        model_file, NULL);
    loaded = TRUE;
  }

  /**
   * @brief TearDown method for each test case
   */
  void TearDown () override
  {
    g_type_class_unref (klass);
    g_object_unref (single);

    for (guint i = 0; i < 32; i++) {
      g_free (input[i].data);
      g_free (output[i].data);
    }
  }
};

/**
 * @brief Test invoke of the tf-lite model.
 */
TEST_F (NNSFilterSingleTestManyInOut, invoke_p)
{
  ASSERT_TRUE (this->loaded);

  EXPECT_TRUE (klass->invoke (single, input, output, TRUE));
  for (guint i = 0; i < 32; i++)
    EXPECT_EQ (((float *) output[i].data)[0], 17.0f);

  EXPECT_TRUE (klass->invoke (single, input, output, FALSE));
  for (guint i = 0; i < 32; i++)
    EXPECT_EQ (((float *) output[i].data)[0], 17.0f);
}

/**
 * @brief Test to set invalid info.
 */
TEST_F (NNSFilterSingleTestManyInOut, setInvalidInfo_n)
{
  GstTensorsInfo in_info, out_info;
  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  ASSERT_TRUE (this->loaded);
  EXPECT_TRUE (klass->start (single));

  /* valid tensor info */
  in_info.num_tensors = 32U;
  for (guint i = 0; i < in_info.num_tensors; i++) {
    GstTensorInfo *info_i = gst_tensors_info_get_nth_info (&in_info, i);
    info_i->type = _NNS_FLOAT32;
    gst_tensor_parse_dimension ("1", info_i->dimension);
  }

  EXPECT_TRUE (klass->set_input_info (single, &in_info, &out_info) == 0);

  EXPECT_EQ (out_info.num_tensors, 32U);
  for (guint i = 0; i < out_info.num_tensors; i++) {
    GstTensorInfo *info_i = gst_tensors_info_get_nth_info (&out_info, i);
    EXPECT_EQ (info_i->type, _NNS_FLOAT32);
    EXPECT_EQ (info_i->dimension[0], 1U);
    EXPECT_EQ (gst_tensor_info_get_rank (info_i), 1U);
  }

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  /* request to set invalid tensor info */
  in_info.num_tensors = 16U;
  EXPECT_FALSE (klass->set_input_info (single, &in_info, &out_info) == 0);

  EXPECT_TRUE (klass->stop (single));

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

#endif /* ENABLE_TENSORFLOW2_LITE */

/**
 * @brief Test to start before initializing.
 */
TEST (testTensorFilterSingle, startUninitialized_n)
{
  GTensorFilterSingle *single;
  GTensorFilterSingleClass *klass;

  single = (GTensorFilterSingle *) g_object_new (G_TYPE_TENSOR_FILTER_SINGLE, NULL);
  klass = (GTensorFilterSingleClass *) g_type_class_ref (G_TYPE_TENSOR_FILTER_SINGLE);

  EXPECT_FALSE (klass->start (single));
  EXPECT_FALSE (klass->input_configured (single));
  EXPECT_FALSE (klass->output_configured (single));

  g_type_class_unref (klass);
  g_object_unref (single);
}

/**
 * @brief Test to invoke with unknown framework.
 */
TEST (testTensorFilterSingle, invokeUnknownFW_n)
{
  GTensorFilterSingle *single;
  GTensorFilterSingleClass *klass;
  GstTensorMemory in, out;

  in.size = out.size = 200U;
  in.data = g_malloc0 (in.size);
  out.data = g_malloc0 (out.size);

  single = (GTensorFilterSingle *) g_object_new (G_TYPE_TENSOR_FILTER_SINGLE, NULL);
  klass = (GTensorFilterSingleClass *) g_type_class_ref (G_TYPE_TENSOR_FILTER_SINGLE);

  /* set invalid fw and invoke */
  g_object_set (G_OBJECT (single), "framework", "unknown-fw", NULL);

  EXPECT_FALSE (klass->invoke (single, &in, &out, FALSE));

  g_type_class_unref (klass);
  g_object_unref (single);
  g_free (in.data);
  g_free (out.data);
}

/**
 * @brief Main GTest.
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
