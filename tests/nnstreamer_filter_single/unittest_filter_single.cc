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
#include <nnstreamer_plugin_api_single.h>

#include "../gst/nnstreamer/tensor_filter/tensor_filter_single.h"

/**
 * @brief Internal data structure to run tensor-filter single.
 */
typedef struct
{
  GTensorFilterSingle *single;
  GTensorFilterSingleClass *klass;
  GstTensorMemory input;
  GstTensorMemory output;
} single_test_data_s;

#ifdef ENABLE_TENSORFLOW_LITE
/**
 * @brief Get max score from output.
 */
static gsize
get_max_score (GstTensorMemory * output)
{
  guint8 *array;
  gsize idx, max_idx = 0;
  guint8 max_value = 0;

  array = (guint8 *) output->data;

  for (idx = 0; idx < output->size; idx++) {
    if (max_value < array[idx]) {
      max_idx = idx;
      max_value = array[idx];
    }
  }

  return max_idx;
}

/**
 * @brief Free test data.
 */
static void
free_test_data (single_test_data_s * test_data)
{
  g_type_class_unref (test_data->klass);
  g_object_unref (test_data->single);
  g_free (test_data->input.data);
  g_free (test_data->output.data);
}

/**
 * @brief Initialize test data, load tf-lite model and raw data.
 */
static gboolean
init_test_data (single_test_data_s * test_data)
{
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *model_file, *data_file;
  gsize length;
  gboolean loaded = FALSE;

  model_file = data_file = NULL;
  memset (test_data, 0, sizeof (single_test_data_s));
  test_data->input.size = 3U * 224 * 224;
  test_data->output.size = 1001U;

  test_data->single = (GTensorFilterSingle *) g_object_new (G_TYPE_TENSOR_FILTER_SINGLE, NULL);
  test_data->klass = (GTensorFilterSingleClass *) g_type_class_ref (G_TYPE_TENSOR_FILTER_SINGLE);

  /* supposed to run test in build directory */
  if (root_path == NULL)
    root_path = "..";

  model_file = g_build_filename (root_path, "tests", "test_models", "models",
      "mobilenet_v1_1.0_224_quant.tflite", NULL);
  if (!g_file_test (model_file, G_FILE_TEST_EXISTS)) {
    goto error;
  }

  data_file = g_build_filename (root_path, "tests", "test_models",
      "data", "orange.raw", NULL);
  if (!g_file_get_contents (data_file, (gchar **) &test_data->input.data, &length, NULL) ||
      length != test_data->input.size) {
    goto error;
  }

  g_object_set (G_OBJECT (test_data->single), "framework", "tensorflow-lite",
      "model", model_file, NULL);
  loaded = TRUE;

error:
  if (!loaded)
    free_test_data (test_data);

  g_free (model_file);
  g_free (data_file);

  return loaded;
}

/**
 * @brief Test to invoke tf-lite model.
 */
TEST (testTensorFilterSingle, invoke)
{
  single_test_data_s tdata = { 0, };

  ASSERT_TRUE (init_test_data (&tdata));

  /* invoke the model and check label 'orange' (index 951) */
  EXPECT_TRUE (tdata.klass->invoke (tdata.single, &tdata.input, &tdata.output, TRUE));
  EXPECT_EQ (get_max_score (&tdata.output), 951U);

  EXPECT_TRUE (tdata.klass->invoke (tdata.single, &tdata.input, &tdata.output, FALSE));
  EXPECT_EQ (get_max_score (&tdata.output), 951U);

  /* check status (tf-lite does not need extra info and data allocation) */
  EXPECT_TRUE (tdata.klass->input_configured (tdata.single));
  EXPECT_TRUE (tdata.klass->output_configured (tdata.single));
  EXPECT_FALSE (tdata.klass->allocate_in_invoke (tdata.single));
  tdata.klass->destroy_notify (tdata.single, &tdata.output);

  free_test_data (&tdata);
}

/**
 * @brief Test to invoke tf-lite model with invalid param.
 */
TEST (testTensorFilterSingle, invokeInvalidParam_n)
{
  single_test_data_s tdata = { 0, };

  ASSERT_TRUE (init_test_data (&tdata));
  EXPECT_TRUE (tdata.klass->start (tdata.single));

  EXPECT_FALSE (tdata.klass->invoke (tdata.single, NULL, &tdata.output, FALSE));
  EXPECT_FALSE (tdata.klass->invoke (tdata.single, &tdata.input, NULL, FALSE));

  EXPECT_TRUE (tdata.klass->stop (tdata.single));
  free_test_data (&tdata);
}

/**
 * @brief Test to invoke tf-lite model with invalid data.
 */
TEST (testTensorFilterSingle, invokeInvalidSize_n)
{
  single_test_data_s tdata = { 0, };

  ASSERT_TRUE (init_test_data (&tdata));
  EXPECT_TRUE (tdata.klass->start (tdata.single));

  /* request allocation with invalid size */
  tdata.output.size = 0;
  EXPECT_FALSE (tdata.klass->invoke (tdata.single, &tdata.input, &tdata.output, TRUE));

  EXPECT_TRUE (tdata.klass->stop (tdata.single));
  free_test_data (&tdata);
}

/**
 * @brief Test to set invalid info.
 */
TEST (testTensorFilterSingle, setInvalidInfo_n)
{
  single_test_data_s tdata = { 0, };
  GstTensorsInfo in_info, out_info;

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  ASSERT_TRUE (init_test_data (&tdata));
  EXPECT_TRUE (tdata.klass->start (tdata.single));

  /* valid tensor info */
  in_info.num_tensors = 1U;
  in_info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("3:224:224:1", in_info.info[0].dimension);
  EXPECT_TRUE (tdata.klass->set_input_info (tdata.single, &in_info, &out_info) == 0);

  /* request to set invalid tensor info */
  gst_tensor_parse_dimension ("1:1:1:1", in_info.info[0].dimension);
  EXPECT_FALSE (tdata.klass->set_input_info (tdata.single, &in_info, &out_info) == 0);

  EXPECT_TRUE (tdata.klass->stop (tdata.single));
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  free_test_data (&tdata);
}

/**
 * @brief Test for filter property.
 */
TEST (testTensorFilterSingle, invalidProperty_n)
{
  single_test_data_s tdata = { 0, };
  gchar *string_val = NULL;

  ASSERT_TRUE (init_test_data (&tdata));

  /* invalid property name */
  g_object_set (G_OBJECT (tdata.single), "framework-invalid", "invalid", NULL);
  g_object_get (G_OBJECT (tdata.single), "framework-invalid", &string_val, NULL);
  EXPECT_FALSE (string_val != NULL);

  g_object_get (G_OBJECT (tdata.single), "framework", &string_val, NULL);
  EXPECT_TRUE (string_val && g_str_equal (string_val, "tensorflow-lite"));
  g_free (string_val);

  free_test_data (&tdata);
}
#endif /* ENABLE_TENSORFLOW_LITE */

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
