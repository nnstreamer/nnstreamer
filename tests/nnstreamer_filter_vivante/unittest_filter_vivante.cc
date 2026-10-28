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
#include <gst/check/gstharness.h>
#include <gst/gst.h>

#include <nnstreamer_internal.h>
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
   * @brief Get the input information of the opened model
   */
  int GetInputInfo (GstTensorsInfo *info)
  {
    GstTensorsInfo out_info;
    int ret;

    gst_tensors_info_init (&out_info);
    ret = sp->getModelInfo (sp, &prop, private_data, GET_IN_OUT_INFO, info, &out_info);
    gst_tensors_info_free (&out_info);
    return ret;
  }

  /**
   * @brief Get the output information of the opened model
   */
  int GetOutputInfo (GstTensorsInfo *info)
  {
    GstTensorsInfo in_info;
    int ret;

    gst_tensors_info_init (&in_info);
    ret = sp->getModelInfo (sp, &prop, private_data, GET_IN_OUT_INFO, &in_info, info);
    gst_tensors_info_free (&in_info);
    return ret;
  }

  /**
   * @brief Make a tensor_filter that runs the mock model
   */
  static GstElement *MakeFilter (void)
  {
    GstElement *filter = gst_element_factory_make ("tensor_filter", NULL);
    gchar *model;

    if (filter == NULL)
      return NULL;

    model = g_strdup_printf ("%s,%s", MOCK_MODEL_NB, MOCK_VIVANTE_MODEL_PATH);
    g_object_set (filter, "framework", "vivante", "model", model, NULL);
    g_free (model);
    return filter;
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

  EXPECT_EQ (GetInputInfo (&info), 0);
  EXPECT_EQ (info.num_tensors, 1U);
  EXPECT_EQ (info.info[0].type, _NNS_UINT8);
  ExpectMockDimension (&info, 0);
  EXPECT_TRUE (gst_tensors_info_validate (&info));
  gst_tensors_info_free (&info);

  EXPECT_EQ (GetOutputInfo (&info), 0);
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

  EXPECT_EQ (GetInputInfo (&info), 0);
  EXPECT_EQ (info.num_tensors, 3U);
  for (i = 0; i < 3U; i++)
    ExpectMockDimension (&info, i);
  EXPECT_TRUE (gst_tensors_info_validate (&info));
  gst_tensors_info_free (&info);

  EXPECT_EQ (GetOutputInfo (&info), 0);
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
    ASSERT_EQ (GetInputInfo (&info), 0);
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

  EXPECT_EQ (GetInputInfo (&info), 0);
  ExpectMaxRankDimension (&info);
  gst_tensors_info_free (&info);

  EXPECT_EQ (GetOutputInfo (&info), 0);
  ExpectMaxRankDimension (&info);
  gst_tensors_info_free (&info);

  sp->close (&prop, &private_data);
}

/**
 * @brief Keep a model that overstates its rank inside the dimension array
 * @details
 * The attributes of a tensor hold VSI_NN_MAX_DIM_NUM sizes, so a rank above
 * that describes sizes the model does not have. For the input and the output
 * tensors alike, the sub-plugin takes only the dimensions the tensor can hold
 * and leaves the rest of the array at 1, rather than copying whatever follows
 * the size array.
 */
TEST_F (NNStreamerFilterVivanteTest, getDimensionOverstatedRank_n)
{
  GstTensorsInfo info;

  mock_ovxlib_set_tensor_rank (VSI_NN_MAX_DIM_NUM);
  mock_ovxlib_set_tensor_rank_overflow (VSI_NN_MAX_DIM_NUM + 4U);
  gst_tensors_info_init (&info);
  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  EXPECT_EQ (GetInputInfo (&info), 0);
  ExpectMockDimension (&info, 0);
  gst_tensors_info_free (&info);

  EXPECT_EQ (GetOutputInfo (&info), 0);
  ExpectMockDimension (&info, 0);
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
  ASSERT_EQ (GetInputInfo (&info), 0);

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

  EXPECT_EQ (sp->invoke (sp, &prop, private_data, &input, &output), 0);
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

  EXPECT_EQ (sp->invoke (sp, &prop, private_data, &input, &output), 0);
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

  EXPECT_EQ (sp->invoke (sp, &prop, private_data, input, output), 0);

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

  EXPECT_NE (sp->invoke (sp, &prop, private_data, &input, &output), 0);

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
 * @brief Keep the opened model when the same pair of model files is given
 */
TEST_F (NNStreamerFilterVivanteTest, reopenSameModel)
{
  GstTensorsInfo info;
  void *opened;

  ASSERT_EQ (sp->open (&prop, &private_data), 0);
  opened = private_data;

  /* A positive value tells that the model is already loaded. */
  EXPECT_EQ (sp->open (&prop, &private_data), 1);
  EXPECT_EQ (private_data, opened);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 1);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 0);

  gst_tensors_info_init (&info);
  EXPECT_EQ (GetInputInfo (&info), 0);
  EXPECT_EQ (info.num_tensors, 1U);
  ExpectMockDimension (&info, 0);
  gst_tensors_info_free (&info);

  freed_tensor_name = 0;
  sp->close (&prop, &private_data);
  EXPECT_EQ (private_data, nullptr);
  EXPECT_EQ (freed_tensor_name, 2U);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 0);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 1);
}

/**
 * @brief Load the model again when only the network binary is another one
 */
TEST_F (NNStreamerFilterVivanteTest, reopenAnotherNetwork)
{
  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  SetFilterProperty ("another_" MOCK_MODEL_NB, MOCK_VIVANTE_MODEL_PATH);
  EXPECT_EQ (sp->open (&prop, &private_data), 0);
  ASSERT_NE (private_data, nullptr);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 1);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 1);

  sp->close (&prop, &private_data);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 2);
}

/**
 * @brief Load the model again when only the model library is another one
 */
TEST_F (NNStreamerFilterVivanteTest, reopenAnotherLibrary)
{
  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  SetFilterProperty (MOCK_MODEL_NB, MOCK_VIVANTE_MODEL_ALT_PATH);
  EXPECT_EQ (sp->open (&prop, &private_data), 0);
  ASSERT_NE (private_data, nullptr);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 1);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 1);

  sp->close (&prop, &private_data);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 2);
}

/**
 * @brief Release the opened model when the model library given next is unusable
 */
TEST_F (NNStreamerFilterVivanteTest, reopenUnknownLibrary_n)
{
  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  /* The network binary is the same, yet this is not the opened model. */
  SetFilterProperty (MOCK_MODEL_NB, "there_is_no_such_library.so");
  EXPECT_LT (sp->open (&prop, &private_data), 0);
  EXPECT_EQ (private_data, nullptr);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 0);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 1);
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

  EXPECT_NE (sp->invoke (sp, &prop, private_data, &input, &output), 0);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &private_data);
}

/**
 * @brief Release the tensor names of the model when the filter is closed
 */
TEST_F (NNStreamerFilterVivanteTest, closeReleasesTensorName)
{
  mock_ovxlib_set_tensor_num (3U, 2U);
  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  freed_tensor_name = 0;
  sp->close (&prop, &private_data);
  EXPECT_EQ (freed_tensor_name, 5U);
}

/**
 * @brief Refuse a model library that cannot be loaded
 */
TEST_F (NNStreamerFilterVivanteTest, openUnknownLibrary_n)
{
  SetFilterProperty (MOCK_MODEL_NB, "there_is_no_such_library.so");

  EXPECT_NE (sp->open (&prop, &private_data), 0);
  EXPECT_EQ (private_data, nullptr);
}

/**
 * @brief Refuse a model library that does not provide every required symbol
 */
TEST_F (NNStreamerFilterVivanteTest, openIncompleteLibrary_n)
{
  SetFilterProperty (MOCK_MODEL_NB, MOCK_VIVANTE_MODEL_PARTIAL_PATH);

  EXPECT_NE (sp->open (&prop, &private_data), 0);
  EXPECT_EQ (private_data, nullptr);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 1);
}

/**
 * @brief Refuse a model whose network binary cannot be loaded
 */
TEST_F (NNStreamerFilterVivanteTest, openBrokenNetwork_n)
{
  mock_ovxlib_set_create_fail (1);

  EXPECT_NE (sp->open (&prop, &private_data), 0);
  EXPECT_EQ (private_data, nullptr);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 0);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 1);
}

/**
 * @brief Refuse a model whose input tensor cannot be read
 */
TEST_F (NNStreamerFilterVivanteTest, openBrokenInputTensor_n)
{
  mock_ovxlib_set_tensor_num (2U, 2U);
  mock_ovxlib_set_get_tensor_fail (1);

  EXPECT_NE (sp->open (&prop, &private_data), 0);
  EXPECT_EQ (private_data, nullptr);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 0);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 1);
}

/**
 * @brief Refuse a model whose output tensor cannot be read
 */
TEST_F (NNStreamerFilterVivanteTest, openBrokenOutputTensor_n)
{
  mock_ovxlib_set_tensor_num (2U, 2U);
  mock_ovxlib_set_get_tensor_fail (3);

  freed_tensor_name = 0;
  EXPECT_NE (sp->open (&prop, &private_data), 0);
  EXPECT_EQ (private_data, nullptr);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 0);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 1);
  /* Two inputs and the first output were named before the second failed. */
  EXPECT_EQ (freed_tensor_name, 3U);
}

/**
 * @brief Report every tensor of a model of more tensors than a buffer holds
 */
TEST_F (NNStreamerFilterVivanteTest, openManyTensors)
{
  const guint num = NNS_TENSOR_MEMORY_MAX + 1U;
  GstTensorsInfo info;
  guint i;

  mock_ovxlib_set_tensor_num (num, num);
  gst_tensors_info_init (&info);
  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  EXPECT_EQ (GetInputInfo (&info), 0);
  EXPECT_EQ (info.num_tensors, num);
  for (i = 0; i < num; i++)
    ExpectMockDimension (&info, i);
  gst_tensors_info_free (&info);

  EXPECT_EQ (GetOutputInfo (&info), 0);
  EXPECT_EQ (info.num_tensors, num);
  for (i = 0; i < num; i++)
    ExpectMockDimension (&info, i);
  gst_tensors_info_free (&info);

  sp->close (&prop, &private_data);
}

/**
 * @brief Accept a model of as many tensors as nnstreamer can describe
 */
TEST_F (NNStreamerFilterVivanteTest, openMaxTensors)
{
  const guint num = NNS_TENSOR_SIZE_LIMIT;
  GstTensorsInfo info;

  mock_ovxlib_set_tensor_num (num, num);
  gst_tensors_info_init (&info);
  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  EXPECT_EQ (GetInputInfo (&info), 0);
  EXPECT_EQ (info.num_tensors, num);
  ExpectMockDimension (&info, num - 1U);
  EXPECT_TRUE (gst_tensors_info_validate (&info));
  gst_tensors_info_free (&info);

  EXPECT_EQ (GetOutputInfo (&info), 0);
  EXPECT_EQ (info.num_tensors, num);
  ExpectMockDimension (&info, num - 1U);
  EXPECT_TRUE (gst_tensors_info_validate (&info));
  gst_tensors_info_free (&info);

  sp->close (&prop, &private_data);
}

/**
 * @brief Refuse a model that reports no input tensor
 */
TEST_F (NNStreamerFilterVivanteTest, openNoInputTensor_n)
{
  mock_ovxlib_set_tensor_num (0U, 1U);

  EXPECT_NE (sp->open (&prop, &private_data), 0);
  EXPECT_EQ (private_data, nullptr);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 1);
}

/**
 * @brief Refuse a model that reports no output tensor
 */
TEST_F (NNStreamerFilterVivanteTest, openNoOutputTensor_n)
{
  mock_ovxlib_set_tensor_num (1U, 0U);

  EXPECT_NE (sp->open (&prop, &private_data), 0);
  EXPECT_EQ (private_data, nullptr);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 1);
}

/**
 * @brief Refuse a model of more tensors than nnstreamer can describe
 */
TEST_F (NNStreamerFilterVivanteTest, openTooManyTensors_n)
{
  mock_ovxlib_set_tensor_num (NNS_TENSOR_SIZE_LIMIT + 1U, 1U);

  EXPECT_NE (sp->open (&prop, &private_data), 0);
  EXPECT_EQ (private_data, nullptr);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 0);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 1);
}

/**
 * @brief Refuse a model of more output tensors than nnstreamer can describe
 */
TEST_F (NNStreamerFilterVivanteTest, openTooManyOutputTensors_n)
{
  mock_ovxlib_set_tensor_num (1U, NNS_TENSOR_SIZE_LIMIT + 1U);

  EXPECT_NE (sp->open (&prop, &private_data), 0);
  EXPECT_EQ (private_data, nullptr);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 0);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 1);
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
 * @brief Tolerate a close of a filter that holds no model
 */
TEST_F (NNStreamerFilterVivanteTest, closeWithoutOpen_n)
{
  sp->close (&prop, &private_data);
  EXPECT_EQ (private_data, nullptr);
}

/**
 * @brief Refuse to report a dimension without an opened model
 */
TEST_F (NNStreamerFilterVivanteTest, getDimensionWithoutOpen_n)
{
  GstTensorsInfo info;

  gst_tensors_info_init (&info);
  EXPECT_NE (GetInputInfo (&info), 0);
  EXPECT_NE (GetOutputInfo (&info), 0);
  gst_tensors_info_free (&info);
}

/**
 * @brief Describe the sub-plugin through the v1 interface
 */
TEST_F (NNStreamerFilterVivanteTest, getFrameworkInfo)
{
  GstTensorFilterFrameworkInfo info;

  EXPECT_EQ (sp->version, GST_TENSOR_FILTER_FRAMEWORK_V1);

  memset (&info, 0xff, sizeof (info));
  EXPECT_EQ (sp->getFrameworkInfo (sp, &prop, private_data, &info), 0);
  EXPECT_STREQ (info.name, "vivante");
  EXPECT_EQ (info.allow_in_place, 0);
  EXPECT_EQ (info.allocate_in_invoke, 0);
  EXPECT_EQ (info.run_without_model, 0);
  EXPECT_EQ (info.verify_model_path, 0);
  EXPECT_EQ (info.hw_list, nullptr);
  EXPECT_EQ (info.num_hw, 0);
  EXPECT_EQ (info.accl_auto, ACCL_NONE);
  EXPECT_EQ (info.accl_default, ACCL_NONE);
  EXPECT_EQ (info.statistics, nullptr);
}

/**
 * @brief Take the automatic and the default accelerator
 * @details tensor_filter accepts these two for any registered sub-plugin, so
 * this checks only that the sub-plugin is registered under its name.
 */
TEST_F (NNStreamerFilterVivanteTest, checkHwAvailability)
{
  EXPECT_TRUE (gst_tensor_filter_check_hw_availability ("vivante", ACCL_AUTO, NULL));
  EXPECT_TRUE (gst_tensor_filter_check_hw_availability ("vivante", ACCL_DEFAULT, NULL));
}

/**
 * @brief Refuse a specific accelerator, as the sub-plugin did with the v0 interface
 */
TEST_F (NNStreamerFilterVivanteTest, checkHwAvailability_n)
{
  EXPECT_FALSE (gst_tensor_filter_check_hw_availability ("vivante", ACCL_NPU_VIVANTE, NULL));
  EXPECT_FALSE (gst_tensor_filter_check_hw_availability ("vivante", ACCL_CPU, NULL));
}

/**
 * @brief Refuse to set the input information of a model of static dimensions
 */
TEST_F (NNStreamerFilterVivanteTest, setInputInfo_n)
{
  GstTensorsInfo in_info, out_info;

  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  ASSERT_EQ (GetInputInfo (&in_info), 0);
  EXPECT_EQ (sp->getModelInfo (sp, &prop, private_data, SET_INPUT_INFO, &in_info, &out_info),
      -ENOENT);
  EXPECT_EQ (out_info.num_tensors, 0U);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);

  sp->close (&prop, &private_data);
}

/**
 * @brief Refuse every event, the model reload included, and keep the model
 */
TEST_F (NNStreamerFilterVivanteTest, eventHandler_n)
{
  const event_ops ops[] = { DESTROY_NOTIFY, RELOAD_MODEL, CUSTOM_PROP, SET_ACCELERATOR };
  GstTensorFilterFrameworkEventData data;
  guint i;

  ASSERT_EQ (sp->open (&prop, &private_data), 0);

  memset (&data, 0, sizeof (data));
  data.model_files = model_files;
  data.num_models = 2;
  for (i = 0; i < G_N_ELEMENTS (ops); i++)
    EXPECT_EQ (sp->eventHandler (sp, &prop, private_data, ops[i], &data), -ENOENT);

  EXPECT_EQ (mock_ovxlib_get_live_graph (), 1);
  EXPECT_EQ (mock_ovxlib_get_model_unload (), 0);

  sp->close (&prop, &private_data);
}

/**
 * @brief Run the sub-plugin through tensor_filter
 * @details
 * tensor_filter takes the information of the model and invokes it through the
 * v1 interface, and refuses to turn on the runtime update of the model, which
 * the sub-plugin does not support.
 */
TEST_F (NNStreamerFilterVivanteTest, pipeline)
{
  const gsize size = GetMockTensorSize (0, 1);
  GstHarness *h;
  GstElement *filter;
  GstBuffer *in_buf, *out_buf;
  GstMapInfo map;
  gboolean updatable = TRUE;
  guint i;

  filter = MakeFilter ();
  ASSERT_NE (filter, nullptr);
  gst_object_ref_sink (filter);
  g_object_set (filter, "is-updatable", TRUE, NULL);
  g_object_get (filter, "is-updatable", &updatable, NULL);
  EXPECT_FALSE (updatable);

  h = gst_harness_new_with_element (filter, "sink", "src");
  gst_harness_set_src_caps_str (h, "other/tensors,num_tensors=1,format=static,"
                                   "types=uint8,dimensions=2:3:2:1,framerate=(fraction)0/1");

  in_buf = gst_buffer_new_allocate (NULL, size, NULL);
  ASSERT_TRUE (gst_buffer_map (in_buf, &map, GST_MAP_WRITE));
  for (i = 0; i < size; i++)
    map.data[i] = (guint8) i;
  gst_buffer_unmap (in_buf, &map);

  EXPECT_EQ (gst_harness_push (h, in_buf), GST_FLOW_OK);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 1);

  out_buf = gst_harness_try_pull (h);
  ASSERT_NE (out_buf, nullptr);
  ASSERT_TRUE (gst_buffer_map (out_buf, &map, GST_MAP_READ));
  EXPECT_EQ (map.size, size);
  for (i = 0; i < MIN (map.size, size); i++)
    EXPECT_EQ (map.data[i], (guint8) (i + 1));
  gst_buffer_unmap (out_buf, &map);
  gst_buffer_unref (out_buf);

  g_object_set (filter, "is-updatable", TRUE, NULL);
  g_object_get (filter, "is-updatable", &updatable, NULL);
  EXPECT_FALSE (updatable);

  gst_harness_teardown (h);
  gst_object_unref (filter);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 0);
}

/**
 * @brief Refuse a stream that the model of static dimensions does not take
 */
TEST_F (NNStreamerFilterVivanteTest, pipelineOtherDimension_n)
{
  GstHarness *h;
  GstElement *filter;
  GstBuffer *out_buf;

  filter = MakeFilter ();
  ASSERT_NE (filter, nullptr);
  gst_object_ref_sink (filter);
  h = gst_harness_new_with_element (filter, "sink", "src");
  gst_harness_set_src_caps_str (h, "other/tensors,num_tensors=1,format=static,"
                                   "types=uint8,dimensions=4:3:2:1,framerate=(fraction)0/1");

  EXPECT_NE (gst_harness_push (h, gst_buffer_new_allocate (NULL, 24, NULL)), GST_FLOW_OK);
  /* The model is opened, so the stream is refused for its dimension. */
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 1);
  out_buf = gst_harness_try_pull (h);
  EXPECT_EQ (out_buf, nullptr);
  if (out_buf)
    gst_buffer_unref (out_buf);

  gst_harness_teardown (h);
  gst_object_unref (filter);
  EXPECT_EQ (mock_ovxlib_get_live_graph (), 0);
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
