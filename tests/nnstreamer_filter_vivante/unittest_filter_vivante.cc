/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_filter_vivante.cc
 * @date        22 Sep 2026
 * @brief       Unit test for the vivante tensor-filter sub-plugin
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 *
 * @details
 * The Vivante NPU and its SDK are not available to the CI machines, so this
 * test builds the sub-plugin against a mock of ovxlib and runs it against a
 * mock model library. It therefore covers the logic of the sub-plugin only:
 * loading a network binary, driving an NPU and the numerical results of an
 * inference are out of its reach.
 */

#include <gtest/gtest.h>
#include <dlfcn.h>
#include <glib.h>
#include <gst/gst.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api_util.h>
#include <tensor_typedef.h>

#include "mock_ovxlib.h"

#define MOCK_MODEL_NB "mock_vivante_model.nb"

static guint freed_tensor_name;

/**
 * @brief Count the tensor names that are released, and release them.
 * @details
 * Whether the sub-plugin leaks the names it gives to its tensors cannot be told
 * from its interface, because it releases its private data along with them. The
 * sub-plugin is linked into this binary, so this definition takes over the one
 * of the nnstreamer library and makes the release of a name observable.
 *
 * Symbol interposition is not limited to the sub-plugin: the calls of the
 * nnstreamer library and of the test itself land here as well. A test that
 * reads the counter therefore has to reset it right before the call it is
 * about to measure.
 */
extern "C" void
gst_tensors_info_free (GstTensorsInfo *info)
{
  static void (*real_free) (GstTensorsInfo *) = NULL;
  guint i;

  if (real_free == NULL) {
    real_free = (void (*) (GstTensorsInfo *)) dlsym (RTLD_NEXT, "gst_tensors_info_free");
    g_assert (real_free != NULL);
  }

  if (info != NULL) {
    for (i = 0; i < NNS_TENSOR_MEMORY_MAX; i++) {
      if (info->info[i].name != NULL)
        freed_tensor_name++;
    }

    if (info->extra != NULL) {
      for (i = 0; i < NNS_TENSOR_SIZE_EXTRA_LIMIT; i++) {
        if (info->extra[i].name != NULL)
          freed_tensor_name++;
      }
    }
  }

  real_free (info);
}

/**
 * @brief Test fixture for the vivante tensor-filter sub-plugin.
 */
class NNStreamerFilterVivanteTest : public ::testing::Test
{
  protected:
  const GstTensorFilterFramework *sp;
  const gchar *model_files[3];
  GstTensorFilterProperties prop;
  void *private_data;

  public:
  /**
   * @brief Construct a new test object
   */
  NNStreamerFilterVivanteTest () : sp (nullptr), private_data (nullptr)
  {
    model_files[0] = model_files[1] = model_files[2] = nullptr;
  }

  /**
   * @brief Set the properties for the given pair of model files
   */
  void SetFilterProperty (const gchar *nb_path, const gchar *so_path)
  {
    model_files[0] = nb_path;
    model_files[1] = so_path;
    model_files[2] = nullptr;

    memset (&prop, 0, sizeof (GstTensorFilterProperties));
    prop.fwname = "vivante";
    prop.fw_opened = 0;
    prop.model_files = model_files;
    prop.num_models = 2;
  }

  /**
   * @brief Get the byte size that the mock tensor of the given index has
   */
  static gsize GetMockTensorSize (guint index, gsize type_size)
  {
    return (2U + index) * 3U * 2U * 1U * type_size;
  }

  /**
   * @brief Check the dimension that the mock tensor of the given index reports
   */
  static void ExpectMockDimension (const GstTensorsInfo *info, guint index)
  {
    const GstTensorInfo *_info
        = gst_tensors_info_get_nth_info ((GstTensorsInfo *) info, index);

    ASSERT_NE (_info, nullptr);
    EXPECT_EQ (_info->dimension[0], 2U + index);
    EXPECT_EQ (_info->dimension[1], 3U);
    EXPECT_EQ (_info->dimension[2], 2U);
    for (guint i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
      EXPECT_EQ (_info->dimension[i], 1U);
  }

  /**
   * @brief Check the dimension of a tensor of as many dimensions as the SDK has
   */
  static void ExpectMaxRankDimension (const GstTensorsInfo *info)
  {
    const GstTensorInfo *_info
        = gst_tensors_info_get_nth_info ((GstTensorsInfo *) info, 0);

    ASSERT_NE (_info, nullptr);
    EXPECT_EQ (_info->dimension[0], 2U);
    EXPECT_EQ (_info->dimension[1], 3U);
    EXPECT_EQ (_info->dimension[2], 2U);
    EXPECT_EQ (_info->dimension[3], 1U);
    for (guint i = 4; i < VSI_NN_MAX_DIM_NUM; i++)
      EXPECT_EQ (_info->dimension[i], 1U);
  }

  /**
   * @brief TearDown method for each test case
   */
  void TearDown () override
  {
    EXPECT_EQ (mock_ovxlib_get_live_graph (), 0);
  }

  /**
   * @brief SetUp method for each test case
   */
  void SetUp () override
  {
    mock_ovxlib_reset ();
    freed_tensor_name = 0;
    private_data = nullptr;
    SetFilterProperty (MOCK_MODEL_NB, MOCK_VIVANTE_MODEL_PATH);

    sp = nnstreamer_filter_find ("vivante");
    ASSERT_NE (sp, nullptr);
  }
};

/**
 * @brief Open and close the sub-plugin with a model the mock accepts
 */
TEST_F (NNStreamerFilterVivanteTest, openClose)
{
  EXPECT_EQ (sp->open (&prop, &private_data), 0);
  ASSERT_NE (private_data, nullptr);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 1);

  sp->close (&prop, &private_data);
  EXPECT_EQ (private_data, nullptr);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 0);
}

/**
 * @brief Report the dimension and the type of the tensors of the model
 */
TEST_F (NNStreamerFilterVivanteTest, getDimension)
{
  GstTensorsInfo info;

  gst_tensors_info_init (&info);
  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  EXPECT_EQ (sp->getInputDimension (&prop, &private_data, &info), 0);
  EXPECT_EQ (info.num_tensors, 1U);
  EXPECT_EQ (info.info[0].type, _NNS_UINT8);
  ExpectMockDimension (&info, 0);
  EXPECT_TRUE (gst_tensors_info_validate (&info));
  gst_tensors_info_free (&info);

  EXPECT_EQ (sp->getOutputDimension (&prop, &private_data, &info), 0);
  EXPECT_EQ (info.num_tensors, 1U);
  EXPECT_EQ (info.info[0].type, _NNS_UINT8);
  ExpectMockDimension (&info, 0);
  EXPECT_TRUE (gst_tensors_info_validate (&info));
  gst_tensors_info_free (&info);

  sp->close (&prop, &private_data);
}

/**
 * @brief Report the dimension of every tensor of a model of several tensors
 */
TEST_F (NNStreamerFilterVivanteTest, getDimensionMultiple)
{
  GstTensorsInfo info;
  guint i;

  mock_ovxlib_set_tensor_num (3U, 2U);
  gst_tensors_info_init (&info);
  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  EXPECT_EQ (sp->getInputDimension (&prop, &private_data, &info), 0);
  EXPECT_EQ (info.num_tensors, 3U);
  for (i = 0; i < 3U; i++)
    ExpectMockDimension (&info, i);
  EXPECT_TRUE (gst_tensors_info_validate (&info));
  gst_tensors_info_free (&info);

  EXPECT_EQ (sp->getOutputDimension (&prop, &private_data, &info), 0);
  EXPECT_EQ (info.num_tensors, 2U);
  for (i = 0; i < 2U; i++)
    ExpectMockDimension (&info, i);
  EXPECT_TRUE (gst_tensors_info_validate (&info));
  gst_tensors_info_free (&info);

  sp->close (&prop, &private_data);
}

/**
 * @brief Convert every tensor type the sub-plugin knows about
 */
TEST_F (NNStreamerFilterVivanteTest, getDimensionType)
{
  const vsi_nn_type_e vsi_types[] = { VSI_NN_TYPE_INT8, VSI_NN_TYPE_UINT8,
    VSI_NN_TYPE_INT16, VSI_NN_TYPE_UINT16, VSI_NN_TYPE_FLOAT32, VSI_NN_TYPE_FLOAT64 };
  const tensor_type nns_types[]
      = { _NNS_INT8, _NNS_UINT8, _NNS_INT16, _NNS_UINT16, _NNS_FLOAT32, _NNS_END };
  GstTensorsInfo info;
  guint i;

  gst_tensors_info_init (&info);

  /* An unknown type becomes _NNS_END here; refusing it is left to the caller. */
  for (i = 0; i < G_N_ELEMENTS (vsi_types); i++) {
    mock_ovxlib_set_tensor_type (vsi_types[i]);
    ASSERT_EQ (sp->open (&prop, &private_data), 0);
    ASSERT_EQ (sp->getInputDimension (&prop, &private_data, &info), 0);
    EXPECT_EQ (info.info[0].type, nns_types[i]);
    gst_tensors_info_free (&info);
    sp->close (&prop, &private_data);
  }
}

/**
 * @brief Describe a tensor that uses as many dimensions as the SDK allows
 * @details
 * The mock reports the rank of the SDK, which is smaller than the rank limit of
 * nnstreamer, so this is the widest tensor a sound model can describe. A rank
 * above that maximum is covered by getDimensionOverstatedRank_n.
 */
TEST_F (NNStreamerFilterVivanteTest, getDimensionMaxRank)
{
  GstTensorsInfo info;

  mock_ovxlib_set_tensor_rank (VSI_NN_MAX_DIM_NUM);
  gst_tensors_info_init (&info);
  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  EXPECT_EQ (sp->getInputDimension (&prop, &private_data, &info), 0);
  ExpectMaxRankDimension (&info);
  gst_tensors_info_free (&info);

  EXPECT_EQ (sp->getOutputDimension (&prop, &private_data, &info), 0);
  ExpectMaxRankDimension (&info);
  gst_tensors_info_free (&info);

  sp->close (&prop, &private_data);
}

/**
 * @brief Convert the half precision type, which nnstreamer may not support
 */
TEST_F (NNStreamerFilterVivanteTest, getDimensionTypeFloat16)
{
  GstTensorsInfo info;

  gst_tensors_info_init (&info);
  mock_ovxlib_set_tensor_type (VSI_NN_TYPE_FLOAT16);
  ASSERT_EQ (sp->open (&prop, &private_data), 0);
  ASSERT_EQ (sp->getInputDimension (&prop, &private_data, &info), 0);

  EXPECT_TRUE (info.info[0].type == _NNS_FLOAT16 || info.info[0].type == _NNS_UINT16);

  gst_tensors_info_free (&info);
  sp->close (&prop, &private_data);
}

/**
 * @brief Invoke the model and check that the output comes from the mock
 */
TEST_F (NNStreamerFilterVivanteTest, invoke)
{
  GstTensorMemory input, output;
  guint8 *in_data, *out_data;
  gsize size = GetMockTensorSize (0, 1);
  guint i;

  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  in_data = (guint8 *) g_malloc0 (size);
  out_data = (guint8 *) g_malloc0 (size);
  for (i = 0; i < size; i++)
    in_data[i] = (guint8) i;

  input.data = in_data;
  input.size = size;
  output.data = out_data;
  output.size = size;

  EXPECT_EQ (sp->invoke_NN (&prop, &private_data, &input, &output), 0);
  for (i = 0; i < size; i++)
    EXPECT_EQ (out_data[i], (guint8) (i + 1));

  g_free (in_data);
  g_free (out_data);
  sp->close (&prop, &private_data);
}

/**
 * @brief Run the post-process of the model when the custom property asks for it
 */
TEST_F (NNStreamerFilterVivanteTest, invokePostProcess)
{
  GstTensorMemory input, output;
  gsize size = GetMockTensorSize (0, 1);

  prop.custom_properties = "postprocess";
  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  input.data = g_malloc0 (size);
  input.size = size;
  output.data = g_malloc0 (size);
  output.size = size;

  EXPECT_EQ (sp->invoke_NN (&prop, &private_data, &input, &output), 0);
  EXPECT_EQ (mock_ovxlib_get_post_process (), 1);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &private_data);
}

/**
 * @brief Invoke a model of several tensors and check every one of them
 */
TEST_F (NNStreamerFilterVivanteTest, invokeMultiple)
{
  const guint num = 3U;
  GstTensorMemory input[num], output[num];
  guint i, j;

  mock_ovxlib_set_tensor_num (num, num);
  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  for (i = 0; i < num; i++) {
    gsize size = GetMockTensorSize (i, 1);
    guint8 *in = (guint8 *) g_malloc0 (size);

    for (j = 0; j < size; j++)
      in[j] = (guint8) (i * 16U + j);

    input[i].data = in;
    input[i].size = size;
    output[i].data = g_malloc0 (size);
    output[i].size = size;
  }

  EXPECT_EQ (sp->invoke_NN (&prop, &private_data, input, output), 0);

  for (i = 0; i < num; i++) {
    const guint8 *out = (const guint8 *) output[i].data;

    for (j = 0; j < output[i].size; j++)
      EXPECT_EQ (out[j], (guint8) (i * 16U + j + 1U));

    g_free (input[i].data);
    g_free (output[i].data);
  }

  sp->close (&prop, &private_data);
}

/**
 * @brief Fail to invoke the model when its post-process reports a failure
 */
TEST_F (NNStreamerFilterVivanteTest, invokePostProcessFail_n)
{
  GstTensorMemory input, output;
  gsize size = GetMockTensorSize (0, 1);

  prop.custom_properties = "pp";
  ASSERT_EQ (sp->open (&prop, &private_data), 0);
  mock_ovxlib_set_post_process_fail (1);

  input.data = g_malloc0 (size);
  input.size = size;
  output.data = g_malloc0 (size);
  output.size = size;

  EXPECT_NE (sp->invoke_NN (&prop, &private_data, &input, &output), 0);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &private_data);
}

/**
 * @brief Close the opened model when another pair of model files is given
 */
TEST_F (NNStreamerFilterVivanteTest, reopenAnotherModel)
{
  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  SetFilterProperty ("another_" MOCK_MODEL_NB, MOCK_VIVANTE_MODEL_ALT_PATH);
  EXPECT_EQ (sp->open (&prop, &private_data), 0);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 1);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 1);

  sp->close (&prop, &private_data);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 2);
}

/**
 * @brief Fail to invoke the model when the mock reports a failure
 */
TEST_F (NNStreamerFilterVivanteTest, invokeFail_n)
{
  GstTensorMemory input, output;
  gsize size = GetMockTensorSize (0, 1);

  ASSERT_EQ (sp->open (&prop, &private_data), 0);
  mock_ovxlib_set_run_fail (1);

  input.data = g_malloc0 (size);
  input.size = size;
  output.data = g_malloc0 (size);
  output.size = size;

  EXPECT_NE (sp->invoke_NN (&prop, &private_data, &input, &output), 0);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &private_data);
}

/**
 * @brief Refuse a model library that cannot be loaded
 */
TEST_F (NNStreamerFilterVivanteTest, openUnknownLibrary_n)
{
  SetFilterProperty (MOCK_MODEL_NB, "there_is_no_such_library.so");

  EXPECT_NE (sp->open (&prop, &private_data), 0);
}

/**
 * @brief Refuse a model library that does not provide every required symbol
 */
TEST_F (NNStreamerFilterVivanteTest, openIncompleteLibrary_n)
{
  SetFilterProperty (MOCK_MODEL_NB, MOCK_VIVANTE_MODEL_PARTIAL_PATH);

  EXPECT_NE (sp->open (&prop, &private_data), 0);
}

/**
 * @brief Refuse to reopen without the path of the model library
 */
TEST_F (NNStreamerFilterVivanteTest, reopenWithoutLibrary_n)
{
  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  /* A refusal before anything is taken keeps the opened model alive. */
  SetFilterProperty (MOCK_MODEL_NB, "");
  EXPECT_NE (sp->open (&prop, &private_data), 0);
  EXPECT_NE (private_data, nullptr);

  SetFilterProperty ("", MOCK_VIVANTE_MODEL_PATH);
  EXPECT_NE (sp->open (&prop, &private_data), 0);
  EXPECT_NE (private_data, nullptr);

  SetFilterProperty (MOCK_MODEL_NB, MOCK_VIVANTE_MODEL_PATH);
  sp->close (&prop, &private_data);
}

/**
 * @brief Refuse to report a dimension without an opened model
 */
TEST_F (NNStreamerFilterVivanteTest, getDimensionWithoutOpen_n)
{
  GstTensorsInfo info;

  gst_tensors_info_init (&info);
  EXPECT_NE (sp->getInputDimension (&prop, &private_data, &info), 0);
  EXPECT_NE (sp->getOutputDimension (&prop, &private_data, &info), 0);
  gst_tensors_info_free (&info);
}

/**
 * @brief Main GTest
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

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
