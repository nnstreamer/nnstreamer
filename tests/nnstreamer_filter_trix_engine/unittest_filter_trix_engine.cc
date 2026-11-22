/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_trix_engine.cc
 * @date    22 Nov 2026
 * @brief   Unit tests of the trix-engine tensor_filter sub-plugin on a mock npu-engine
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * The sub-plugin source is linked into this binary against the mock
 * npu-engine of mock/, which reads a text model and needs no NPU. See
 * mock/trix_mock.h.
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api_util.h>

#include <trix_mock.h>

/** @brief A model of one 1x4x4x3 input and outputs of 1x4x4x3 and 1x1x1x4. */
static const gchar *MODEL = "trix-mock-model\n"
                            "in 1 4 4 3\n"
                            "out 1 4 4 3\n"
                            "out 1 1 1 4\n";

/**
 * @brief Fixture that writes a mock model into a private directory.
 */
class nnstreamerFilterTRIxEngine : public ::testing::Test
{
  protected:
  const GstTensorFilterFramework *sp;
  gchar *dir;
  gchar *model;
  void *data;
  GstTensorFilterProperties prop;
  const gchar *model_files[2];
  accl_hw hw[1];

  /** @brief Find the sub-plugin, write the model and set the properties. */
  void SetUp () override
  {
    dir = nullptr;
    model = nullptr;
    data = nullptr;
    trix_mock_reset ();
    sp = nnstreamer_filter_find ("trix-engine");
    ASSERT_TRUE (sp != nullptr);
    dir = g_dir_make_tmp ("nns-trix-XXXXXX", NULL);
    ASSERT_TRUE (dir != nullptr);
    model = writeModel ("model.tvn", MODEL);

    memset (&prop, 0, sizeof (prop));
    prop.fwname = "trix-engine";
    model_files[0] = model;
    model_files[1] = nullptr;
    prop.model_files = model_files;
    prop.num_models = 1;
    hw[0] = ACCL_NPU_SR;
    prop.hw_list = hw;
    prop.num_hw = 1;
    gst_tensors_info_init (&prop.input_meta);
    gst_tensors_info_init (&prop.output_meta);
  }

  /** @brief Close the filter, check that nothing is left and remove the files. */
  void TearDown () override
  {
    if (data)
      sp->close (&prop, &data);
    gst_tensors_info_free (&prop.input_meta);
    gst_tensors_info_free (&prop.output_meta);
    EXPECT_EQ (trix_mock_live_objects (), 0);
    if (model) {
      g_remove (model);
      g_free (model);
    }
    if (dir)
      g_rmdir (dir);
    g_free (dir);
    trix_mock_reset ();
  }

  /** @brief Write a model file into the directory and return its path. */
  gchar *writeModel (const gchar *name, const gchar *content)
  {
    gchar *path = g_build_filename (dir, name, NULL);

    EXPECT_TRUE (g_file_set_contents (path, content, -1, NULL));
    return path;
  }

  /** @brief Open the filter with the current properties. */
  int open ()
  {
    return sp->open (&prop, &data);
  }

  /** @brief Invoke the opened filter. */
  int invoke (const GstTensorMemory *in, GstTensorMemory *out)
  {
    return sp->invoke (sp, &prop, data, in, out);
  }

  /**
   * @brief Set the given type for the input and the outputs, and invoke with
   *        input buffers of the size of that type.
   */
  int invokeAs (tensor_type type)
  {
    GstTensorInfo *info;
    gsize esize = gst_tensor_get_element_size (type);
    GstTensorMemory in, out[2] = { { nullptr, 0 }, { nullptr, 0 } };
    int ret;

    prop.input_meta.num_tensors = 1;
    info = gst_tensors_info_get_nth_info (&prop.input_meta, 0);
    info->type = type;
    gst_tensor_parse_dimension ("3:4:4:1", info->dimension);
    prop.output_meta.num_tensors = 2;
    info = gst_tensors_info_get_nth_info (&prop.output_meta, 0);
    info->type = type;
    gst_tensor_parse_dimension ("3:4:4:1", info->dimension);
    info = gst_tensors_info_get_nth_info (&prop.output_meta, 1);
    info->type = type;
    gst_tensor_parse_dimension ("4:1:1:1", info->dimension);

    if (open () != 0)
      return -1;

    in.size = 48 * esize;
    in.data = g_malloc0 (in.size);
    ret = invoke (&in, out);
    if (ret == 0) {
      EXPECT_EQ (out[0].size, 48 * esize);
      EXPECT_EQ (out[1].size, 4 * esize);
    }
    g_free (in.data);
    free (out[0].data);
    free (out[1].data);

    sp->close (&prop, &data);
    data = nullptr;
    gst_tensors_info_free (&prop.input_meta);
    gst_tensors_info_free (&prop.output_meta);
    gst_tensors_info_init (&prop.input_meta);
    gst_tensors_info_init (&prop.output_meta);
    return ret;
  }
};

/**
 * @brief Positive case: the framework information.
 */
TEST_F (nnstreamerFilterTRIxEngine, frameworkInfo)
{
  GstTensorFilterFrameworkInfo info;

  memset (&info, 0, sizeof (info));
  EXPECT_EQ (sp->getFrameworkInfo (sp, NULL, NULL, &info), 0);
  EXPECT_STREQ (info.name, "trix-engine");
  EXPECT_FALSE (info.run_without_model);
  EXPECT_TRUE (info.verify_model_path);
  EXPECT_TRUE (info.allocate_in_invoke);
  EXPECT_FALSE (info.allow_in_place);
  ASSERT_EQ (info.num_hw, 1);
  EXPECT_EQ (info.hw_list[0], ACCL_NPU_SR);
}

/**
 * @brief Positive case: open a model, describe it, and release everything.
 */
TEST_F (nnstreamerFilterTRIxEngine, openModelInfo)
{
  GstTensorsInfo in, out;
  GstTensorFilterFrameworkEventData event;
  gchar *dim;

  ASSERT_EQ (open (), 0);
  EXPECT_EQ (trix_mock_last_device_type (), NPUCOND_TRIV2_CONN_SOCIP);
  /* a device and a model */
  EXPECT_EQ (trix_mock_live_objects (), 2);

  gst_tensors_info_init (&in);
  gst_tensors_info_init (&out);
  ASSERT_EQ (sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in, &out), 0);

  /* the model lists dimensions outermost first, nnstreamer innermost first */
  ASSERT_EQ (in.num_tensors, 1U);
  EXPECT_EQ (in.info[0].type, _NNS_UINT8);
  dim = gst_tensor_get_dimension_string (in.info[0].dimension);
  EXPECT_STREQ (dim, "3:4:4:1");
  g_free (dim);
  ASSERT_EQ (out.num_tensors, 2U);
  EXPECT_EQ (out.info[0].type, _NNS_UINT8);
  dim = gst_tensor_get_dimension_string (out.info[0].dimension);
  EXPECT_STREQ (dim, "3:4:4:1");
  g_free (dim);
  EXPECT_EQ (out.info[1].type, _NNS_UINT8);
  dim = gst_tensor_get_dimension_string (out.info[1].dimension);
  EXPECT_STREQ (dim, "4:1:1:1");
  g_free (dim);

  EXPECT_EQ (sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, &in, &out), -ENOENT);
  gst_tensors_info_free (&in);
  gst_tensors_info_free (&out);

  memset (&event, 0, sizeof (event));
  EXPECT_EQ (sp->eventHandler (sp, &prop, data, RELOAD_MODEL, &event), -ENOENT);

  sp->close (&prop, &data);
  data = nullptr;
  EXPECT_EQ (trix_mock_live_objects (), 0);
  EXPECT_EQ (trix_mock_calls ("unregisterNPUmodel_all"), 1);
  EXPECT_EQ (trix_mock_calls ("putNPUdevice"), 1);
}

/**
 * @brief Positive case: invoke the model on the buffers of the filter.
 */
TEST_F (nnstreamerFilterTRIxEngine, invoke)
{
  guint8 in_data[48];
  GstTensorMemory in = { in_data, sizeof (in_data) };
  data_layout layout;
  data_type type;

  ASSERT_EQ (open (), 0);

  for (int frame = 0; frame < 3; frame++) {
    GstTensorMemory out[2] = { { nullptr, 0 }, { nullptr, 0 } };

    for (guint i = 0; i < sizeof (in_data); i++)
      in_data[i] = (guint8) (i * 7 + frame);

    ASSERT_EQ (invoke (&in, out), 0);
    ASSERT_EQ (out[0].size, 48U);
    ASSERT_EQ (out[1].size, 4U);
    for (guint i = 0; i < 48; i++)
      EXPECT_EQ (((guint8 *) out[0].data)[i], (guint8) (in_data[i] + 1U));
    for (guint i = 0; i < 4; i++)
      EXPECT_EQ (((guint8 *) out[1].data)[i], (guint8) (in_data[i] + 2U));
    free (out[0].data);
    free (out[1].data);
  }

  EXPECT_EQ (trix_mock_calls ("submitNPU_request"), 3);
  EXPECT_EQ (trix_mock_calls ("removeNPU_request"), 3);
  /* with no type given and _NNS_LAYOUT_ANY (the default), the model decides */
  ASSERT_EQ (trix_mock_last_data_info (true, 0, &layout, &type), 0);
  EXPECT_EQ (layout, DATA_LAYOUT_MODEL);
  EXPECT_EQ (type, DATA_TYPE_UINT8);
  /* no request is left */
  EXPECT_EQ (trix_mock_live_objects (), 2);
}

/**
 * @brief Positive case: the layouts given to the filter reach the device.
 */
TEST_F (nnstreamerFilterTRIxEngine, layouts)
{
  guint8 in_data[48] = { 0 };
  GstTensorMemory in = { in_data, sizeof (in_data) };
  GstTensorMemory out[2] = { { nullptr, 0 }, { nullptr, 0 } };
  data_layout layout;
  data_type type;

  prop.input_layout[0] = _NNS_LAYOUT_NHWC;
  prop.output_layout[0] = _NNS_LAYOUT_NCHW;
  prop.output_layout[1] = _NNS_LAYOUT_NONE;
  ASSERT_EQ (open (), 0);
  ASSERT_EQ (invoke (&in, out), 0);
  free (out[0].data);
  free (out[1].data);

  ASSERT_EQ (trix_mock_last_data_info (true, 0, &layout, &type), 0);
  EXPECT_EQ (layout, DATA_LAYOUT_NHWC);
  ASSERT_EQ (trix_mock_last_data_info (false, 0, &layout, &type), 0);
  EXPECT_EQ (layout, DATA_LAYOUT_NCHW);
  ASSERT_EQ (trix_mock_last_data_info (false, 1, &layout, &type), 0);
  EXPECT_EQ (layout, DATA_LAYOUT_MODEL);
}

/**
 * @brief Positive case: the tensor types given to the filter reach the device.
 */
TEST_F (nnstreamerFilterTRIxEngine, dataTypes)
{
  static const struct {
    tensor_type nns;
    data_type trix;
  } types[] = {
    { _NNS_INT8, DATA_TYPE_INT8 },
    { _NNS_UINT8, DATA_TYPE_UINT8 },
    { _NNS_INT16, DATA_TYPE_INT16 },
    { _NNS_UINT16, DATA_TYPE_UINT16 },
    { _NNS_INT32, DATA_TYPE_INT32 },
    { _NNS_UINT32, DATA_TYPE_UINT32 },
    { _NNS_INT64, DATA_TYPE_INT64 },
    { _NNS_UINT64, DATA_TYPE_UINT64 },
    { _NNS_FLOAT32, DATA_TYPE_FLOAT32 },
    { _NNS_FLOAT64, DATA_TYPE_FLOAT64 },
  };

  for (const auto &t : types) {
    data_layout layout;
    data_type type;

    EXPECT_EQ (invokeAs (t.nns), 0) << gst_tensor_get_type_string (t.nns);
    ASSERT_EQ (trix_mock_last_data_info (true, 0, &layout, &type), 0);
    EXPECT_EQ (type, t.trix) << gst_tensor_get_type_string (t.nns);
    ASSERT_EQ (trix_mock_last_data_info (false, 1, &layout, &type), 0);
    EXPECT_EQ (type, t.trix) << gst_tensor_get_type_string (t.nns);
  }
}

/**
 * @brief Positive case: a tensor type npu-engine has no type for is left to
 *        the model, whose (mock) type is uint8.
 */
TEST_F (nnstreamerFilterTRIxEngine, dataTypeOfModel)
{
  data_layout layout;
  data_type type;
  GstTensorInfo *info;
  guint8 in_data[48] = { 0 };
  GstTensorMemory in = { in_data, sizeof (in_data) };
  GstTensorMemory out[2] = { { nullptr, 0 }, { nullptr, 0 } };

  prop.input_meta.num_tensors = 1;
  info = gst_tensors_info_get_nth_info (&prop.input_meta, 0);
  info->type = _NNS_FLOAT16;
  gst_tensor_parse_dimension ("3:4:4:1", info->dimension);
  prop.output_meta.num_tensors = 2;
  info = gst_tensors_info_get_nth_info (&prop.output_meta, 0);
  info->type = _NNS_FLOAT16;
  gst_tensor_parse_dimension ("3:4:4:1", info->dimension);
  info = gst_tensors_info_get_nth_info (&prop.output_meta, 1);
  info->type = _NNS_FLOAT16;
  gst_tensor_parse_dimension ("4:1:1:1", info->dimension);
  ASSERT_EQ (open (), 0);

  EXPECT_EQ (invoke (&in, out), 0);
  EXPECT_EQ (out[0].size, 48U);
  EXPECT_EQ (out[1].size, 4U);
  free (out[0].data);
  free (out[1].data);
  ASSERT_EQ (trix_mock_last_data_info (true, 0, &layout, &type), 0);
  EXPECT_EQ (type, DATA_TYPE_MODEL);
  ASSERT_EQ (trix_mock_last_data_info (false, 0, &layout, &type), 0);
  EXPECT_EQ (type, DATA_TYPE_MODEL);
}

/**
 * @brief Positive case: the input info given to the filter is used, and the
 *        output info, not given, comes from the model.
 */
TEST_F (nnstreamerFilterTRIxEngine, mixedInfo)
{
  GstTensorsInfo in, out;
  GstTensorInfo *info;
  gchar *dim;

  prop.input_meta.num_tensors = 1;
  info = gst_tensors_info_get_nth_info (&prop.input_meta, 0);
  info->type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("3:4:4:1", info->dimension);
  ASSERT_EQ (open (), 0);

  gst_tensors_info_init (&in);
  gst_tensors_info_init (&out);
  ASSERT_EQ (sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in, &out), 0);
  ASSERT_EQ (in.num_tensors, 1U);
  EXPECT_EQ (in.info[0].type, _NNS_FLOAT32);
  ASSERT_EQ (out.num_tensors, 2U);
  EXPECT_EQ (out.info[0].type, _NNS_UINT8);
  EXPECT_EQ (out.info[1].type, _NNS_UINT8);
  dim = gst_tensor_get_dimension_string (out.info[1].dimension);
  EXPECT_STREQ (dim, "4:1:1:1");
  g_free (dim);
  gst_tensors_info_free (&in);
  gst_tensors_info_free (&out);
}

/**
 * @brief Positive case: accelerators other than npu.sr in the list are skipped.
 */
TEST_F (nnstreamerFilterTRIxEngine, acceleratorList)
{
  accl_hw list[2] = { ACCL_CPU, ACCL_NPU_SR };

  prop.hw_list = list;
  prop.num_hw = 2;
  ASSERT_EQ (open (), 0);
  EXPECT_EQ (trix_mock_calls ("getNPUdeviceByTypeAny"), 1);
  EXPECT_EQ (trix_mock_last_device_type (), NPUCOND_TRIV2_CONN_SOCIP);
  sp->close (&prop, &data);
  data = nullptr;
  prop.hw_list = hw;
  prop.num_hw = 1;
}

/**
 * @brief Positive case: when a device of an npu.sr entry cannot be opened,
 *        the next npu.sr entry is tried.
 */
TEST_F (nnstreamerFilterTRIxEngine, acceleratorRetry)
{
  accl_hw list[2] = { ACCL_NPU_SR, ACCL_NPU_SR };

  prop.hw_list = list;
  prop.num_hw = 2;
  trix_mock_fail_next ("getNPUdeviceByTypeAny");
  ASSERT_EQ (open (), 0);
  EXPECT_EQ (trix_mock_calls ("getNPUdeviceByTypeAny"), 2);
  sp->close (&prop, &data);
  data = nullptr;
  prop.hw_list = hw;
  prop.num_hw = 1;
}

/**
 * @brief Positive case: a request that cannot be removed after its outputs
 *        are taken does not fail the invoke; closing the device drops it.
 */
TEST_F (nnstreamerFilterTRIxEngine, removeFails)
{
  guint8 in_data[48] = { 0 };
  GstTensorMemory in = { in_data, sizeof (in_data) };
  GstTensorMemory out[2] = { { nullptr, 0 }, { nullptr, 0 } };

  ASSERT_EQ (open (), 0);
  trix_mock_fail ("removeNPU_request");
  EXPECT_EQ (invoke (&in, out), 0);
  ASSERT_TRUE (out[0].data != nullptr);
  EXPECT_EQ (((guint8 *) out[0].data)[0], 1U);
  free (out[0].data);
  free (out[1].data);
  EXPECT_EQ (trix_mock_calls ("removeNPU_request"), 1);
  /* a device, a model and the request left behind */
  EXPECT_EQ (trix_mock_live_objects (), 3);
}

/**
 * @brief Negative case: no model path.
 */
TEST_F (nnstreamerFilterTRIxEngine, noModelPath_n)
{
  model_files[0] = "";
  EXPECT_NE (open (), 0);
  EXPECT_TRUE (data == nullptr);
  EXPECT_EQ (trix_mock_calls ("getNPUmodel_metadata"), 0);
}

/**
 * @brief Negative case: a file that is not a model.
 */
TEST_F (nnstreamerFilterTRIxEngine, badModel_n)
{
  gchar *bad = writeModel ("bad.tvn", "not a model\n");

  model_files[0] = bad;
  EXPECT_NE (open (), 0);
  EXPECT_TRUE (data == nullptr);
  EXPECT_EQ (trix_mock_calls ("getNPUmodel_metadata"), 1);
  EXPECT_EQ (trix_mock_calls ("getNPUdeviceByTypeAny"), 0);
  g_remove (bad);
  g_free (bad);
}

/**
 * @brief Negative case: no accelerator the sub-plugin supports.
 */
TEST_F (nnstreamerFilterTRIxEngine, noAccelerator_n)
{
  hw[0] = ACCL_CPU;
  EXPECT_NE (open (), 0);
  EXPECT_TRUE (data == nullptr);
  EXPECT_EQ (trix_mock_calls ("getNPUdeviceByTypeAny"), 0);
}

/**
 * @brief Negative case: no device is available.
 */
TEST_F (nnstreamerFilterTRIxEngine, noDevice_n)
{
  trix_mock_fail ("getNPUdeviceByTypeAny");
  EXPECT_NE (open (), 0);
  EXPECT_TRUE (data == nullptr);
  EXPECT_EQ (trix_mock_calls ("getNPUdeviceByTypeAny"), 1);
  EXPECT_EQ (trix_mock_calls ("registerNPUmodel"), 0);
}

/**
 * @brief Negative case: the device refuses the model.
 */
TEST_F (nnstreamerFilterTRIxEngine, registerFails_n)
{
  trix_mock_fail ("registerNPUmodel");
  EXPECT_NE (open (), 0);
  EXPECT_TRUE (data == nullptr);
  EXPECT_EQ (trix_mock_calls ("putNPUdevice"), 1);
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
