/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_qnn.cc
 * @date    22 Nov 2026
 * @brief   Unit tests of the QNN tensor_filter sub-plugin on a mock QNN
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * The sub-plugin source is linked into this binary against the reduced QNN
 * header in mock/, and opens the mock model and backend libraries of
 * mock/qnn_mock_model.cc and mock/qnn_mock_backend.cc. See mock/qnn_mock.h.
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api_util.h>

#include <qnn_mock.h>

/**
 * @brief Fixture that opens the qnn sub-plugin on the mock libraries.
 */
class nnstreamerFilterQnn : public ::testing::Test
{
  protected:
  const GstTensorFilterFramework *sp;
  void *data;
  GstTensorFilterProperties prop;
  const gchar *model_files[3];

  /** @brief Find the sub-plugin and reset the mocks. */
  void SetUp () override
  {
    qnn_mock_backend_reset ();
    qnn_mock_model_reset ();
    sp = nnstreamer_filter_find ("qnn");
    ASSERT_TRUE (sp != nullptr);
    data = nullptr;
    setModel (QNN_MOCK_MODEL_PATH, QNN_MOCK_BACKEND_PATH);
  }

  /** @brief Close the filter and check that the mocks hold nothing. */
  void TearDown () override
  {
    if (data)
      sp->close (&prop, &data);
    EXPECT_EQ (qnn_mock_backend_live_objects (), 0);
    EXPECT_EQ (qnn_mock_model_live_graphs (), 0);
    qnn_mock_backend_reset ();
    qnn_mock_model_reset ();
  }

  /** @brief Set the model files; a null @a backend gives one model file. */
  void setModel (const gchar *model, const gchar *backend)
  {
    memset (&prop, 0, sizeof (prop));
    prop.fwname = "qnn";
    model_files[0] = model;
    model_files[1] = backend;
    model_files[2] = nullptr;
    prop.model_files = model_files;
    prop.num_models = backend ? 2 : 1;
  }

  /** @brief Open the filter with the current properties. */
  int open ()
  {
    return sp->open (&prop, &data);
  }

  /** @brief Get the model information of the opened filter. */
  int modelInfo (GstTensorsInfo *in, GstTensorsInfo *out)
  {
    return sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, in, out);
  }

  /** @brief Invoke the opened filter. */
  int invoke (const GstTensorMemory *in, GstTensorMemory *out)
  {
    return sp->invoke (sp, &prop, data, in, out);
  }

  /** @brief Check that an open fails and leaves nothing behind. */
  void expectOpenFails ()
  {
    EXPECT_NE (open (), 0);
    EXPECT_TRUE (data == nullptr);
    EXPECT_EQ (qnn_mock_backend_live_objects (), 0);
    EXPECT_EQ (qnn_mock_model_live_graphs (), 0);
  }

  /**
   * @brief Invoke with every input byte set to @a v, and check the outputs.
   *        The mock backend refuses (and the invoke fails) when the tensors
   *        it gets do not describe the model, e.g., lost quantization.
   */
  void expectInvokeWorks (guint8 v)
  {
    guint8 in_data[48], out_data[48];
    gint32 sum_data = -1;
    GstTensorMemory in = { in_data, sizeof (in_data) };
    GstTensorMemory out[2]
        = { { out_data, sizeof (out_data) }, { &sum_data, sizeof (sum_data) } };

    memset (in_data, v, sizeof (in_data));
    memset (out_data, 0, sizeof (out_data));
    ASSERT_EQ (invoke (&in, out), 0);
    for (guint i = 0; i < sizeof (out_data); i++)
      EXPECT_EQ (out_data[i], (guint8) (v + 1U));
    EXPECT_EQ (sum_data, 48 * (gint32) v);
  }

  /** @brief Open the default model and return the type of its input. */
  tensor_type inputType ()
  {
    GstTensorsInfo in, out;
    tensor_type type = _NNS_END;

    if (open () != 0)
      return _NNS_END;

    gst_tensors_info_init (&in);
    gst_tensors_info_init (&out);
    if (modelInfo (&in, &out) == 0 && in.num_tensors == 1U)
      type = in.info[0].type;
    gst_tensors_info_free (&in);
    gst_tensors_info_free (&out);

    sp->close (&prop, &data);
    data = nullptr;
    return type;
  }
};

/**
 * @brief Positive case: the framework information.
 */
TEST_F (nnstreamerFilterQnn, frameworkInfo)
{
  GstTensorFilterFrameworkInfo info;

  memset (&info, 0, sizeof (info));
  EXPECT_EQ (sp->getFrameworkInfo (sp, NULL, NULL, &info), 0);
  EXPECT_STREQ (info.name, "qnn");
  EXPECT_FALSE (info.run_without_model);
  EXPECT_TRUE (info.verify_model_path);
  EXPECT_FALSE (info.allocate_in_invoke);
  ASSERT_EQ (info.num_hw, 1);
  EXPECT_EQ (info.hw_list[0], ACCL_CPU);
}

/**
 * @brief Positive case: open the model, describe it and release every object.
 */
TEST_F (nnstreamerFilterQnn, openClose)
{
  GstTensorsInfo in, out;
  gchar *dim;

  ASSERT_EQ (open (), 0);
  EXPECT_EQ (qnn_mock_backend_calls ("backendCreate"), 1);
  EXPECT_EQ (qnn_mock_backend_calls ("contextCreate"), 1);
  EXPECT_EQ (qnn_mock_backend_calls ("graphCreate"), 1);
  EXPECT_EQ (qnn_mock_backend_calls ("graphFinalize"), 1);
  /* a backend, a context and a graph */
  EXPECT_EQ (qnn_mock_backend_live_objects (), 3);
  EXPECT_EQ (qnn_mock_model_live_graphs (), 1);

  gst_tensors_info_init (&in);
  gst_tensors_info_init (&out);
  ASSERT_EQ (modelInfo (&in, &out), 0);

  /* QNN lists dimensions outermost first, nnstreamer innermost first. */
  ASSERT_EQ (in.num_tensors, 1U);
  EXPECT_STREQ (in.info[0].name, "in0");
  EXPECT_EQ (in.info[0].type, _NNS_UINT8);
  dim = gst_tensor_get_dimension_string (in.info[0].dimension);
  EXPECT_STREQ (dim, "3:4:4:1");
  g_free (dim);

  ASSERT_EQ (out.num_tensors, 2U);
  EXPECT_STREQ (out.info[0].name, "out0");
  EXPECT_EQ (out.info[0].type, _NNS_UINT8);
  dim = gst_tensor_get_dimension_string (out.info[0].dimension);
  EXPECT_STREQ (dim, "3:4:4:1");
  g_free (dim);
  EXPECT_STREQ (out.info[1].name, "sum");
  EXPECT_EQ (out.info[1].type, _NNS_INT32);
  dim = gst_tensor_get_dimension_string (out.info[1].dimension);
  EXPECT_STREQ (dim, "1");
  g_free (dim);

  gst_tensors_info_free (&in);
  gst_tensors_info_free (&out);

  sp->close (&prop, &data);
  data = nullptr;
  EXPECT_EQ (qnn_mock_backend_live_objects (), 0);
  EXPECT_EQ (qnn_mock_model_live_graphs (), 0);
  EXPECT_EQ (qnn_mock_backend_calls ("contextFree"), 1);
  EXPECT_EQ (qnn_mock_backend_calls ("backendFree"), 1);
}

/**
 * @brief Positive case: invoke the graph on the buffers of the filter.
 */
TEST_F (nnstreamerFilterQnn, invoke)
{
  guint8 in_data[48], out_data[48];
  gint32 sum_data;
  GstTensorMemory in = { in_data, sizeof (in_data) };
  GstTensorMemory out[2]
      = { { out_data, sizeof (out_data) }, { &sum_data, sizeof (sum_data) } };

  ASSERT_EQ (open (), 0);

  for (int frame = 0; frame < 3; frame++) {
    gint32 expected_sum = 0;

    for (guint i = 0; i < sizeof (in_data); i++) {
      in_data[i] = (guint8) (i * 37 + frame * 101);
      expected_sum += in_data[i];
    }
    memset (out_data, 0, sizeof (out_data));
    sum_data = -1;

    ASSERT_EQ (invoke (&in, out), 0);
    for (guint i = 0; i < sizeof (in_data); i++)
      EXPECT_EQ (out_data[i], (guint8) (in_data[i] + 1U));
    EXPECT_EQ (sum_data, expected_sum);
  }

  EXPECT_EQ (qnn_mock_backend_calls ("graphExecute"), 3);
}

/**
 * @brief Positive case: every QNN data type the sub-plugin maps.
 */
TEST_F (nnstreamerFilterQnn, dataTypes)
{
  static const struct {
    Qnn_DataType_t qnn;
    tensor_type nns;
  } types[] = {
    { QNN_DATATYPE_INT_8, _NNS_INT8 },
    { QNN_DATATYPE_INT_16, _NNS_INT16 },
    { QNN_DATATYPE_INT_32, _NNS_INT32 },
    { QNN_DATATYPE_INT_64, _NNS_INT64 },
    { QNN_DATATYPE_UINT_8, _NNS_UINT8 },
    { QNN_DATATYPE_UINT_16, _NNS_UINT16 },
    { QNN_DATATYPE_UINT_32, _NNS_UINT32 },
    { QNN_DATATYPE_UINT_64, _NNS_UINT64 },
    { QNN_DATATYPE_FLOAT_16, _NNS_FLOAT16 },
    { QNN_DATATYPE_FLOAT_32, _NNS_FLOAT32 },
    { QNN_DATATYPE_FLOAT_64, _NNS_FLOAT64 },
    { QNN_DATATYPE_SFIXED_POINT_8, _NNS_INT8 },
    { QNN_DATATYPE_SFIXED_POINT_16, _NNS_INT16 },
    { QNN_DATATYPE_SFIXED_POINT_32, _NNS_INT32 },
    { QNN_DATATYPE_UFIXED_POINT_8, _NNS_UINT8 },
    { QNN_DATATYPE_UFIXED_POINT_16, _NNS_UINT16 },
    { QNN_DATATYPE_UFIXED_POINT_32, _NNS_UINT32 },
    { QNN_DATATYPE_BOOL_8, _NNS_UINT8 },
  };

  for (const auto &t : types) {
    qnn_mock_model_set_input_type (t.qnn);
    EXPECT_EQ (inputType (), t.nns) << "QNN data type 0x" << std::hex << (int) t.qnn;
  }
}

/**
 * @brief Positive case: version 2 tensors with dynamic-dimension flags, which
 *        reach the backend as the model set them.
 */
TEST_F (nnstreamerFilterQnn, tensorV2)
{
  qnn_mock_model_use_tensor_v2 (true);
  ASSERT_EQ (open (), 0);
  expectInvokeWorks (2);
}

/**
 * @brief Positive case: version 2 tensors without dynamic-dimension flags.
 */
TEST_F (nnstreamerFilterQnn, tensorV2WithoutFlags)
{
  qnn_mock_model_use_tensor_v2 (false);
  ASSERT_EQ (open (), 0);
  expectInvokeWorks (5);
}

/**
 * @brief Positive case: scale-offset quantization reaches the backend.
 */
TEST_F (nnstreamerFilterQnn, scaleOffsetQuantization)
{
  qnn_mock_model_set_quantization (QNN_QUANTIZATION_ENCODING_SCALE_OFFSET, 0);
  ASSERT_EQ (open (), 0);
  expectInvokeWorks (9);
}

/**
 * @brief Positive case: more than 16 tensors, up to the limit of 256.
 */
TEST_F (nnstreamerFilterQnn, manyTensors)
{
  static const guint counts[] = { 17, NNS_TENSOR_SIZE_LIMIT };

  for (const guint n : counts) {
    GstTensorsInfo in, out;
    GstTensorInfo *last;
    gchar *name;

    qnn_mock_model_set_num_tensors (n, n);
    ASSERT_EQ (open (), 0) << n << " tensors";

    gst_tensors_info_init (&in);
    gst_tensors_info_init (&out);
    ASSERT_EQ (modelInfo (&in, &out), 0);
    EXPECT_EQ (in.num_tensors, n);
    EXPECT_EQ (out.num_tensors, n);

    last = gst_tensors_info_get_nth_info (&in, n - 1);
    name = g_strdup_printf ("extra_in%u", n - 1);
    EXPECT_STREQ (last->name, name);
    EXPECT_EQ (last->type, _NNS_INT32);
    g_free (name);

    last = gst_tensors_info_get_nth_info (&out, n - 1);
    name = g_strdup_printf ("extra%u", n - 1);
    EXPECT_STREQ (last->name, name);
    EXPECT_EQ (last->type, _NNS_INT32);
    g_free (name);

    gst_tensors_info_free (&in);
    gst_tensors_info_free (&out);
    sp->close (&prop, &data);
    data = nullptr;
  }
}

/**
 * @brief Positive case: tensors of the largest rank nnstreamer takes.
 */
TEST_F (nnstreamerFilterQnn, maxRank)
{
  GstTensorsInfo in, out;
  gchar *dim;

  qnn_mock_model_set_ranks (NNS_TENSOR_RANK_LIMIT, NNS_TENSOR_RANK_LIMIT);
  ASSERT_EQ (open (), 0);

  gst_tensors_info_init (&in);
  gst_tensors_info_init (&out);
  ASSERT_EQ (modelInfo (&in, &out), 0);
  dim = gst_tensor_get_dimension_string (in.info[0].dimension);
  EXPECT_STREQ (dim, "3:4:4:1:1:1:1:1:1:1:1:1:1:1:1:1");
  g_free (dim);
  dim = gst_tensor_get_dimension_string (out.info[0].dimension);
  EXPECT_STREQ (dim, "3:4:4:1:1:1:1:1:1:1:1:1:1:1:1:1");
  g_free (dim);
  gst_tensors_info_free (&in);
  gst_tensors_info_free (&out);
}

/**
 * @brief Negative case: model information other than GET_IN_OUT_INFO, and events.
 */
TEST_F (nnstreamerFilterQnn, unsupportedOps_n)
{
  GstTensorsInfo in, out;
  GstTensorFilterFrameworkEventData event;

  ASSERT_EQ (open (), 0);
  gst_tensors_info_init (&in);
  gst_tensors_info_init (&out);
  EXPECT_EQ (sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, &in, &out), -ENOENT);
  gst_tensors_info_free (&in);
  gst_tensors_info_free (&out);

  memset (&event, 0, sizeof (event));
  EXPECT_EQ (sp->eventHandler (sp, &prop, data, RELOAD_MODEL, &event), -ENOENT);
}

/**
 * @brief Negative case: one model file instead of a model and a backend.
 */
TEST_F (nnstreamerFilterQnn, oneModelFile_n)
{
  setModel (QNN_MOCK_MODEL_PATH, nullptr);
  expectOpenFails ();
  EXPECT_EQ (qnn_mock_backend_calls ("getProviders"), 0);
}

/**
 * @brief Negative case: three model files instead of a model and a backend.
 */
TEST_F (nnstreamerFilterQnn, threeModelFiles_n)
{
  const gchar *files[4]
      = { QNN_MOCK_MODEL_PATH, QNN_MOCK_BACKEND_PATH, QNN_MOCK_BACKEND_PATH, nullptr };

  prop.model_files = files;
  prop.num_models = 3;
  expectOpenFails ();
  EXPECT_EQ (qnn_mock_backend_calls ("getProviders"), 0);
}

/**
 * @brief Negative case: the backend library does not exist.
 */
TEST_F (nnstreamerFilterQnn, backendMissing_n)
{
  setModel (QNN_MOCK_MODEL_PATH, "/nonexistent/libQnnMissing.so");
  expectOpenFails ();
  EXPECT_EQ (qnn_mock_backend_calls ("getProviders"), 0);
}

/**
 * @brief Negative case: the backend library has no QnnInterface_getProviders.
 */
TEST_F (nnstreamerFilterQnn, backendWithoutProviders_n)
{
  setModel (QNN_MOCK_MODEL_PATH, QNN_MOCK_MODEL_PATH);
  expectOpenFails ();
  EXPECT_EQ (qnn_mock_backend_calls ("getProviders"), 0);
}

/**
 * @brief Negative case: the backend lists no provider.
 */
TEST_F (nnstreamerFilterQnn, noProvider_n)
{
  qnn_mock_backend_set_num_providers (0);
  expectOpenFails ();
  EXPECT_EQ (qnn_mock_backend_calls ("getProviders"), 1);
  EXPECT_EQ (qnn_mock_backend_calls ("backendCreate"), 0);
}

/**
 * @brief Negative case: the backend cannot be created.
 */
TEST_F (nnstreamerFilterQnn, backendCreateFails_n)
{
  qnn_mock_backend_fail ("backendCreate");
  expectOpenFails ();
  EXPECT_EQ (qnn_mock_backend_calls ("backendCreate"), 1);
  EXPECT_EQ (qnn_mock_backend_calls ("contextCreate"), 0);
}

/**
 * @brief Negative case: the context cannot be created.
 */
TEST_F (nnstreamerFilterQnn, contextCreateFails_n)
{
  qnn_mock_backend_fail ("contextCreate");
  expectOpenFails ();
  EXPECT_EQ (qnn_mock_backend_calls ("contextCreate"), 1);
  EXPECT_EQ (qnn_mock_backend_calls ("backendFree"), 1);
  EXPECT_EQ (qnn_mock_backend_calls ("graphCreate"), 0);
}

/**
 * @brief Negative case: the model library does not exist.
 */
TEST_F (nnstreamerFilterQnn, modelMissing_n)
{
  setModel ("/nonexistent/libQnnMissingModel.so", QNN_MOCK_BACKEND_PATH);
  expectOpenFails ();
  EXPECT_EQ (qnn_mock_backend_calls ("contextCreate"), 1);
  EXPECT_EQ (qnn_mock_backend_calls ("graphCreate"), 0);
}

/**
 * @brief Negative case: the model fails to compose its graphs.
 */
TEST_F (nnstreamerFilterQnn, composeFails_n)
{
  qnn_mock_model_fail_compose ();
  expectOpenFails ();
  EXPECT_EQ (qnn_mock_backend_calls ("graphFinalize"), 0);
}

/**
 * @brief Negative case: the graph cannot be finalized.
 */
TEST_F (nnstreamerFilterQnn, finalizeFails_n)
{
  qnn_mock_backend_fail ("graphFinalize");
  expectOpenFails ();
  EXPECT_EQ (qnn_mock_backend_calls ("graphFinalize"), 1);
}

/**
 * @brief Negative case: QNN data types with no tensor type of nnstreamer.
 */
TEST_F (nnstreamerFilterQnn, unsupportedDataType_n)
{
  static const Qnn_DataType_t types[] = { QNN_DATATYPE_STRING,
    QNN_DATATYPE_SFIXED_POINT_4, QNN_DATATYPE_UFIXED_POINT_4, QNN_DATATYPE_UNDEFINED };

  for (const auto t : types) {
    qnn_mock_model_set_input_type (t);
    EXPECT_NE (open (), 0) << "QNN data type 0x" << std::hex << (int) t;
    EXPECT_TRUE (data == nullptr);
    /* refused after the graph is finalized, not before */
    EXPECT_EQ (qnn_mock_backend_calls ("graphFinalize"), 1);
    EXPECT_EQ (qnn_mock_backend_live_objects (), 0);
    EXPECT_EQ (qnn_mock_model_live_graphs (), 0);
    qnn_mock_backend_reset ();
  }
}

/**
 * @brief Negative case: the graph fails to execute.
 */
TEST_F (nnstreamerFilterQnn, executeFails_n)
{
  guint8 in_data[48] = { 0 }, out_data[48];
  gint32 sum_data;
  GstTensorMemory in = { in_data, sizeof (in_data) };
  GstTensorMemory out[2]
      = { { out_data, sizeof (out_data) }, { &sum_data, sizeof (sum_data) } };

  ASSERT_EQ (open (), 0);
  qnn_mock_backend_fail ("graphExecute");
  EXPECT_NE (invoke (&in, out), 0);
  EXPECT_EQ (qnn_mock_backend_calls ("graphExecute"), 1);
  EXPECT_EQ (qnn_mock_backend_calls ("errorGetMessage"), 1);
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

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
