/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_filter_pytorch_mock.cc
 * @date        27 Oct 2026
 * @brief       Unit test for the pytorch tensor-filter sub-plugin on a LibTorch mock
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 *
 * @details
 * LibTorch is too large to install in every CI job, so this test builds the
 * sub-plugin against a reduced stand-in of the LibTorch API (mock/) and covers
 * the logic of the sub-plugin: model loading, the accelerator choice, the
 * type and shape conversion of the inputs and the serialization and the
 * validation of the outputs. Whether the sub-plugin still builds and runs
 * against a real LibTorch is checked by the weekly workflow pytorch.yml.
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api_util.h>
#include <tensor_typedef.h>

#include "torch_mock.h"

/**
 * @brief Whether the configuration asks for the GPU (--gpu-by-conf)
 * @details nnstreamer caches a configuration value at its first lookup, so
 * the two choices of the configuration are tested by two runs of this binary.
 */
static bool gpu_by_conf = false;

/**
 * @brief Test fixture for the pytorch tensor-filter sub-plugin.
 */
class NNStreamerFilterPytorchMockTest : public ::testing::Test
{
  protected:
  const GstTensorFilterFramework *sp;
  GstTensorFilterProperties prop;
  const gchar *model_files[2];
  void *private_data;
  gchar *tmpdir;
  GPtrArray *paths;

  public:
  /**
   * @brief Construct a new test object
   */
  NNStreamerFilterPytorchMockTest ()
      : sp (nullptr), private_data (nullptr), tmpdir (nullptr), paths (nullptr)
  {
    model_files[0] = model_files[1] = nullptr;
  }

  /**
   * @brief Find the sub-plugin and prepare a directory for mock models
   */
  void SetUp () override
  {
    torch_mock::reset ();

    sp = nnstreamer_filter_find ("pytorch");
    ASSERT_NE (sp, nullptr);

    tmpdir = g_dir_make_tmp ("nns-pytorch-mock-XXXXXX", NULL);
    ASSERT_NE (tmpdir, nullptr);
    paths = g_ptr_array_new_with_free_func (g_free);

    memset (&prop, 0, sizeof (GstTensorFilterProperties));
    gst_tensors_info_init (&prop.input_meta);
    gst_tensors_info_init (&prop.output_meta);
  }

  /**
   * @brief Close the sub-plugin and remove the mock models
   */
  void TearDown () override
  {
    if (sp && private_data)
      sp->close (&prop, &private_data);

    gst_tensors_info_free (&prop.input_meta);
    gst_tensors_info_free (&prop.output_meta);

    if (paths) {
      for (guint i = 0; i < paths->len; i++)
        g_remove ((const gchar *) g_ptr_array_index (paths, i));
      g_ptr_array_free (paths, TRUE);
    }
    if (tmpdir) {
      g_rmdir (tmpdir);
      g_free (tmpdir);
    }
  }

  /**
   * @brief Write a mock model of the given behaviour and return its path
   */
  const gchar *MakeModel (const gchar *behaviour)
  {
    gchar *name = g_strdup_printf ("%s.pt", behaviour);
    gchar *path = g_build_filename (tmpdir, name, NULL);

    g_free (name);
    EXPECT_TRUE (g_file_set_contents (path, behaviour, -1, NULL));
    g_ptr_array_add (paths, path);
    return path;
  }

  /**
   * @brief Set the model and the accelerator of the filter
   */
  void SetModel (const gchar *path, const gchar *accl = "cpu")
  {
    model_files[0] = path;
    prop.fwname = "pytorch";
    prop.model_files = model_files;
    prop.num_models = 1;
    prop.accl_str = accl;
  }

  /**
   * @brief Append a tensor of the given type and dimension to the tensors info
   */
  static guint AddTensor (GstTensorsInfo *info, tensor_type type, const gchar *dim)
  {
    GstTensorInfo *_info = gst_tensors_info_get_nth_info (info, info->num_tensors);
    guint rank;

    _info->type = type;
    rank = gst_tensor_parse_dimension (dim, _info->dimension);
    info->num_tensors++;
    return rank;
  }

  /**
   * @brief Append an input tensor
   */
  void AddInput (tensor_type type, const gchar *dim)
  {
    guint idx = prop.input_meta.num_tensors;

    prop.input_ranks[idx] = AddTensor (&prop.input_meta, type, dim);
  }

  /**
   * @brief Append an output tensor
   */
  void AddOutput (tensor_type type, const gchar *dim)
  {
    guint idx = prop.output_meta.num_tensors;

    prop.output_ranks[idx] = AddTensor (&prop.output_meta, type, dim);
  }

  /**
   * @brief Open the filter with the configured properties
   */
  int Open ()
  {
    return sp->open (&prop, &private_data);
  }

  /**
   * @brief Invoke the filter with tensors of the given byte sizes
   * @details Input i is filled with the byte value (i + 1) * 0x11 and every
   * output is cleared to 0xEE before the call.
   */
  int Invoke (GstTensorMemory *in, GstTensorMemory *out, const gsize *in_size,
      guint num_in, const gsize *out_size, guint num_out)
  {
    for (guint i = 0; i < num_in; i++) {
      in[i].size = in_size[i];
      in[i].data = g_malloc (in_size[i]);
      memset (in[i].data, (int) ((i + 1) * 0x11), in_size[i]);
    }
    for (guint i = 0; i < num_out; i++) {
      out[i].size = out_size[i];
      out[i].data = g_malloc (out_size[i]);
      memset (out[i].data, 0xEE, out_size[i]);
    }

    return sp->invoke_NN (&prop, &private_data, in, out);
  }

  /**
   * @brief Free the tensors of an Invoke () call
   */
  static void FreeTensors (GstTensorMemory *mem, guint num)
  {
    for (guint i = 0; i < num; i++) {
      g_free (mem[i].data);
      mem[i].data = nullptr;
    }
  }

  /**
   * @brief Whether every byte of the memory has the given value
   */
  static bool AllBytes (const GstTensorMemory *mem, guint8 value)
  {
    const guint8 *p = (const guint8 *) mem->data;

    for (gsize i = 0; i < mem->size; i++) {
      if (p[i] != value)
        return false;
    }
    return true;
  }
};

/**
 * @brief The sub-plugin registers itself as a v0 framework.
 */
TEST_F (NNStreamerFilterPytorchMockTest, registered)
{
  EXPECT_EQ (sp->version, GST_TENSOR_FILTER_FRAMEWORK_V0);
  EXPECT_STREQ (sp->name, "pytorch");
  EXPECT_TRUE (sp->verify_model_path);
  EXPECT_FALSE (sp->run_without_model);
}

/**
 * @brief Open loads the model on the CPU and puts it in evaluation mode.
 */
TEST_F (NNStreamerFilterPytorchMockTest, openClose)
{
  SetModel (MakeModel ("identity"));

  EXPECT_EQ (Open (), 0);
  EXPECT_NE (private_data, nullptr);
  EXPECT_EQ (torch_mock::loadCount (), 1);
  EXPECT_EQ (torch_mock::evalCount (), 1);
  EXPECT_EQ (torch_mock::lastLoadDevice (), c10::DeviceType::CPU);

  sp->close (&prop, &private_data);
  EXPECT_EQ (private_data, nullptr);
}

/**
 * @brief Closing a filter that is not open does nothing.
 */
TEST_F (NNStreamerFilterPytorchMockTest, closeNotOpened)
{
  SetModel (MakeModel ("identity"));

  sp->close (&prop, &private_data);
  EXPECT_EQ (private_data, nullptr);
}

/**
 * @brief Opening the same model again is skipped.
 */
TEST_F (NNStreamerFilterPytorchMockTest, openSameModel)
{
  SetModel (MakeModel ("identity"));

  ASSERT_EQ (Open (), 0);
  void *first = private_data;

  EXPECT_EQ (Open (), 1);
  EXPECT_EQ (private_data, first);
  EXPECT_EQ (torch_mock::loadCount (), 1);
}

/**
 * @brief Opening another model replaces the loaded one.
 */
TEST_F (NNStreamerFilterPytorchMockTest, openAnotherModel)
{
  SetModel (MakeModel ("identity"));
  ASSERT_EQ (Open (), 0);

  SetModel (MakeModel ("tuple"));
  EXPECT_EQ (Open (), 0);
  EXPECT_NE (private_data, nullptr);
  EXPECT_EQ (torch_mock::loadCount (), 2);
}

/**
 * @brief The sub-plugin takes exactly one model file.
 */
TEST_F (NNStreamerFilterPytorchMockTest, openTwoModels_n)
{
  SetModel (MakeModel ("identity"));
  prop.num_models = 2;

  EXPECT_EQ (Open (), -1);
  EXPECT_EQ (private_data, nullptr);
}

/**
 * @brief A model file that does not exist cannot be opened.
 */
TEST_F (NNStreamerFilterPytorchMockTest, openMissingModel_n)
{
  gchar *path = g_build_filename (tmpdir, "missing.pt", NULL);

  SetModel (path);
  EXPECT_EQ (Open (), -2);
  EXPECT_EQ (private_data, nullptr);
  EXPECT_EQ (torch_mock::loadCount (), 0);
  g_free (path);
}

/**
 * @brief A file that LibTorch cannot read is refused.
 */
TEST_F (NNStreamerFilterPytorchMockTest, openNotAModel_n)
{
  SetModel (MakeModel ("garbage"));

  EXPECT_EQ (Open (), -2);
  EXPECT_EQ (private_data, nullptr);
}

/**
 * @brief An invalid_argument from LibTorch while loading is caught.
 */
TEST_F (NNStreamerFilterPytorchMockTest, openInvalidArgument_n)
{
  SetModel (MakeModel ("load_invalid_argument"));

  EXPECT_EQ (Open (), -2);
  EXPECT_EQ (private_data, nullptr);
}

/**
 * @brief An exception of an unknown type while loading is caught.
 */
TEST_F (NNStreamerFilterPytorchMockTest, openUnknownException_n)
{
  SetModel (MakeModel ("load_unknown"));

  EXPECT_EQ (Open (), -2);
  EXPECT_EQ (private_data, nullptr);
}

/**
 * @brief Asking for the GPU loads the model on CUDA and moves the tensors.
 */
TEST_F (NNStreamerFilterPytorchMockTest, openGpu)
{
  GstTensorMemory in[1], out[1];
  const gsize size[1] = { 24 };

  torch_mock::setCudaAvailable (true);
  SetModel (MakeModel ("identity"), "true:gpu");
  AddInput (_NNS_UINT8, "4:3:2:1");
  AddOutput (_NNS_UINT8, "4:3:2:1");

  ASSERT_EQ (Open (), 0);
  EXPECT_EQ (torch_mock::lastLoadDevice (), c10::DeviceType::CUDA);

  EXPECT_EQ (Invoke (in, out, size, 1, size, 1), 0);
  ASSERT_EQ (torch_mock::lastInputs ().size (), 1U);
  EXPECT_EQ (torch_mock::lastInputs ()[0].device_type (), c10::DeviceType::CUDA);
  EXPECT_TRUE (AllBytes (&out[0], 0x11));

  FreeTensors (in, 1);
  FreeTensors (out, 1);
}

/**
 * @brief Without CUDA, a model asked for the GPU cannot be loaded.
 */
TEST_F (NNStreamerFilterPytorchMockTest, openGpuWithoutCuda_n)
{
  SetModel (MakeModel ("identity"), "true:gpu");

  EXPECT_EQ (Open (), -2);
  EXPECT_EQ (private_data, nullptr);
}

/**
 * @brief Asking for the CPU keeps the model on the CPU even with CUDA.
 */
TEST_F (NNStreamerFilterPytorchMockTest, openCpuWithCuda)
{
  torch_mock::setCudaAvailable (true);
  SetModel (MakeModel ("identity"), "true:cpu");

  ASSERT_EQ (Open (), 0);
  EXPECT_EQ (torch_mock::lastLoadDevice (), c10::DeviceType::CPU);
}

/**
 * @brief Without an accelerator, the model is loaded on the CPU.
 */
TEST_F (NNStreamerFilterPytorchMockTest, openDefaultAccelerator)
{
  torch_mock::setCudaAvailable (true);
  SetModel (MakeModel ("identity"), NULL);

  ASSERT_EQ (Open (), 0);
  EXPECT_EQ (torch_mock::lastLoadDevice (), c10::DeviceType::CPU);
}

/**
 * @brief With acceleration turned off, the configuration decides on the device.
 */
TEST_F (NNStreamerFilterPytorchMockTest, openDeviceByConfiguration)
{
  torch_mock::setCudaAvailable (true);
  SetModel (MakeModel ("identity"), "false");

  ASSERT_EQ (Open (), 0);
  EXPECT_EQ (torch_mock::lastLoadDevice (),
      gpu_by_conf ? c10::DeviceType::CUDA : c10::DeviceType::CPU);
}

/**
 * @brief The dimensions given to the filter are reported back.
 */
TEST_F (NNStreamerFilterPytorchMockTest, getDimension)
{
  GstTensorsInfo info;

  SetModel (MakeModel ("identity"));
  AddInput (_NNS_UINT8, "4:3:2:1");
  AddOutput (_NNS_FLOAT32, "10:1");
  ASSERT_EQ (Open (), 0);

  gst_tensors_info_init (&info);
  EXPECT_EQ (sp->getInputDimension (&prop, &private_data, &info), 0);
  EXPECT_TRUE (gst_tensors_info_is_equal (&info, &prop.input_meta));
  gst_tensors_info_free (&info);

  gst_tensors_info_init (&info);
  EXPECT_EQ (sp->getOutputDimension (&prop, &private_data, &info), 0);
  EXPECT_TRUE (gst_tensors_info_is_equal (&info, &prop.output_meta));
  gst_tensors_info_free (&info);
}

/**
 * @brief The callbacks refuse a filter that is not open.
 */
TEST_F (NNStreamerFilterPytorchMockTest, notOpened_n)
{
  GstTensorsInfo info;
  GstTensorMemory mem;

  SetModel (MakeModel ("identity"));
  gst_tensors_info_init (&info);
  memset (&mem, 0, sizeof (mem));

  EXPECT_EQ (sp->getInputDimension (&prop, &private_data, &info), -EINVAL);
  EXPECT_EQ (sp->getOutputDimension (&prop, &private_data, &info), -EINVAL);
  EXPECT_EQ (sp->invoke_NN (&prop, &private_data, &mem, &mem), -EINVAL);
  gst_tensors_info_free (&info);
}

/**
 * @brief The CPU and the GPU are the accelerators of the sub-plugin.
 */
TEST_F (NNStreamerFilterPytorchMockTest, checkAvailability)
{
  EXPECT_EQ (sp->checkAvailability (ACCL_CPU), 0);
  EXPECT_EQ (sp->checkAvailability (ACCL_GPU), 0);
  EXPECT_EQ (sp->checkAvailability (ACCL_NPU), -ENOENT);
}

/**
 * @brief The input is passed in its rank, with the dimensions reversed.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeShape)
{
  GstTensorMemory in[1], out[1];
  const gsize size[1] = { 24 };

  SetModel (MakeModel ("identity"));
  AddInput (_NNS_UINT8, "4:3:2");
  AddOutput (_NNS_UINT8, "4:3:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, size, 1, size, 1), 0);
  ASSERT_EQ (torch_mock::lastInputs ().size (), 1U);

  const at::Tensor &input = torch_mock::lastInputs ()[0];
  ASSERT_EQ (input.ndimension (), 3);
  EXPECT_EQ (input.sizes ()[0], 2);
  EXPECT_EQ (input.sizes ()[1], 3);
  EXPECT_EQ (input.sizes ()[2], 4);
  EXPECT_EQ (input.data_ptr (), in[0].data);
  EXPECT_EQ (input.device_type (), c10::DeviceType::CPU);
  EXPECT_TRUE (AllBytes (&out[0], 0x11));

  FreeTensors (in, 1);
  FreeTensors (out, 1);
}

/**
 * @brief Every element type of nnstreamer maps to the LibTorch one and back.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeTypes)
{
  const struct {
    tensor_type nns;
    c10::ScalarType torch;
    gsize size;
  } types[] = {
    { _NNS_UINT8, c10::ScalarType::Byte, 1 },
    { _NNS_INT8, c10::ScalarType::Char, 1 },
    { _NNS_INT16, c10::ScalarType::Short, 2 },
    { _NNS_INT32, c10::ScalarType::Int, 4 },
    { _NNS_INT64, c10::ScalarType::Long, 8 },
    { _NNS_FLOAT32, c10::ScalarType::Float, 4 },
    { _NNS_FLOAT64, c10::ScalarType::Double, 8 },
#ifdef FLOAT16_SUPPORT
    { _NNS_FLOAT16, c10::ScalarType::Half, 2 },
#endif
  };

  for (const auto &t : types) {
    GstTensorMemory in[1], out[1];
    const gsize size[1] = { 6 * t.size };

    SCOPED_TRACE (gst_tensor_get_type_string (t.nns));
    if (private_data)
      sp->close (&prop, &private_data);
    gst_tensors_info_free (&prop.input_meta);
    gst_tensors_info_free (&prop.output_meta);
    gst_tensors_info_init (&prop.input_meta);
    gst_tensors_info_init (&prop.output_meta);

    SetModel (MakeModel ("identity"));
    AddInput (t.nns, "3:2");
    AddOutput (t.nns, "3:2");
    ASSERT_EQ (Open (), 0);

    EXPECT_EQ (Invoke (in, out, size, 1, size, 1), 0);
    ASSERT_EQ (torch_mock::lastInputs ().size (), 1U);
    EXPECT_EQ (torch_mock::lastInputs ()[0].scalar_type (), t.torch);
    EXPECT_TRUE (AllBytes (&out[0], 0x11));

    FreeTensors (in, 1);
    FreeTensors (out, 1);
  }
}

/**
 * @brief An input of a type LibTorch does not take is refused.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeUnsupportedInputType_n)
{
  GstTensorMemory in[1], out[1];
  const gsize size[1] = { 6 };

  SetModel (MakeModel ("identity"));
  AddInput (_NNS_UINT32, "3:2");
  AddOutput (_NNS_UINT8, "3:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, size, 1, size, 1), -1);
  EXPECT_EQ (torch_mock::forwardCount (), 0);
  EXPECT_TRUE (AllBytes (&out[0], 0xEE));

  FreeTensors (in, 1);
  FreeTensors (out, 1);
}

#ifndef FLOAT16_SUPPORT
/**
 * @brief A float16 input is refused when nnstreamer is built without float16.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeFloat16Unsupported_n)
{
  GstTensorMemory in[1], out[1];
  const gsize size[1] = { 12 };

  SetModel (MakeModel ("identity"));
  AddInput (_NNS_FLOAT16, "3:2");
  AddOutput (_NNS_FLOAT16, "3:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, size, 1, size, 1), -1);
  EXPECT_EQ (torch_mock::forwardCount (), 0);
  EXPECT_TRUE (AllBytes (&out[0], 0xEE));

  FreeTensors (in, 1);
  FreeTensors (out, 1);
}
#endif

/**
 * @brief A model is run again without validating its outputs again.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeTwice)
{
  GstTensorMemory in[1], out[1];
  const gsize size[1] = { 4 };

  SetModel (MakeModel ("identity"));
  AddInput (_NNS_FLOAT32, "1");
  AddOutput (_NNS_FLOAT32, "1");
  ASSERT_EQ (Open (), 0);

  for (int i = 0; i < 2; i++) {
    EXPECT_EQ (Invoke (in, out, size, 1, size, 1), 0);
    EXPECT_TRUE (AllBytes (&out[0], 0x11));
    FreeTensors (in, 1);
    FreeTensors (out, 1);
  }
  EXPECT_EQ (torch_mock::forwardCount (), 2);
}

/**
 * @brief The tensors of a returned tuple fill the outputs in order.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeTuple)
{
  GstTensorMemory in[2], out[2];
  const gsize size[2] = { 6, 8 };

  SetModel (MakeModel ("tuple"));
  AddInput (_NNS_UINT8, "3:2");
  AddInput (_NNS_UINT8, "4:2");
  AddOutput (_NNS_UINT8, "3:2");
  AddOutput (_NNS_UINT8, "4:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, size, 2, size, 2), 0);
  EXPECT_TRUE (AllBytes (&out[0], 0x11));
  EXPECT_TRUE (AllBytes (&out[1], 0x22));

  FreeTensors (in, 2);
  FreeTensors (out, 2);
}

/**
 * @brief The tensors of a returned list fill the outputs in order.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeList)
{
  GstTensorMemory in[2], out[2];
  const gsize size[2] = { 6, 8 };

  SetModel (MakeModel ("list"));
  AddInput (_NNS_UINT8, "3:2");
  AddInput (_NNS_UINT8, "4:2");
  AddOutput (_NNS_UINT8, "3:2");
  AddOutput (_NNS_UINT8, "4:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, size, 2, size, 2), 0);
  EXPECT_TRUE (AllBytes (&out[0], 0x11));
  EXPECT_TRUE (AllBytes (&out[1], 0x22));

  FreeTensors (in, 2);
  FreeTensors (out, 2);
}

/**
 * @brief A list nested in a tuple is flattened into the outputs.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeNested)
{
  GstTensorMemory in[3], out[3];
  const gsize size[3] = { 2, 3, 4 };

  SetModel (MakeModel ("nested"));
  AddInput (_NNS_UINT8, "2");
  AddInput (_NNS_UINT8, "3");
  AddInput (_NNS_UINT8, "4");
  AddOutput (_NNS_UINT8, "2");
  AddOutput (_NNS_UINT8, "3");
  AddOutput (_NNS_UINT8, "4");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, size, 3, size, 3), 0);
  EXPECT_TRUE (AllBytes (&out[0], 0x11));
  EXPECT_TRUE (AllBytes (&out[1], 0x22));
  EXPECT_TRUE (AllBytes (&out[2], 0x33));

  FreeTensors (in, 3);
  FreeTensors (out, 3);
}

/**
 * @brief Returned tensors beyond the configured outputs are ignored.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeExtraOutputs)
{
  GstTensorMemory in[2], out[1];
  const gsize size[2] = { 6, 8 };

  SetModel (MakeModel ("tuple"));
  AddInput (_NNS_UINT8, "3:2");
  AddInput (_NNS_UINT8, "4:2");
  AddOutput (_NNS_UINT8, "3:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, size, 2, size, 1), 0);
  EXPECT_TRUE (AllBytes (&out[0], 0x11));

  FreeTensors (in, 2);
  FreeTensors (out, 1);
}

/**
 * @brief An output whose first dimension is 0 is accepted when its type matches.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeEmptyOutput)
{
  GstTensorMemory in[1], out[1];
  const gsize size[1] = { 6 };

  SetModel (MakeModel ("empty"));
  AddInput (_NNS_UINT8, "3:2");
  AddOutput (_NNS_UINT8, "3:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, size, 1, size, 1), 0);
  EXPECT_TRUE (AllBytes (&out[0], 0xEE));

  FreeTensors (in, 1);
  FreeTensors (out, 1);
}

/**
 * @brief A 0-dim (scalar) output is copied out after checking its type.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeScalarOutput)
{
  GstTensorMemory in[1], out[1];
  const gsize in_size[1] = { 6 };
  const gsize out_size[1] = { 1 };

  SetModel (MakeModel ("scalar"));
  AddInput (_NNS_UINT8, "3:2");
  AddOutput (_NNS_UINT8, "1");
  ASSERT_EQ (Open (), 0);

  for (int i = 0; i < 2; i++) {
    EXPECT_EQ (Invoke (in, out, in_size, 1, out_size, 1), 0);
    EXPECT_TRUE (AllBytes (&out[0], 0x00));
    FreeTensors (in, 1);
    FreeTensors (out, 1);
  }
}

/**
 * @brief A 0-dim (scalar) output is refused when its type differs.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeScalarOutputType_n)
{
  GstTensorMemory in[1], out[1];
  const gsize in_size[1] = { 6 };
  const gsize out_size[1] = { 4 };

  SetModel (MakeModel ("scalar"));
  AddInput (_NNS_UINT8, "3:2");
  AddOutput (_NNS_FLOAT32, "1");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, in_size, 1, out_size, 1), -2);
  EXPECT_TRUE (AllBytes (&out[0], 0xEE));

  FreeTensors (in, 1);
  FreeTensors (out, 1);
}

/**
 * @brief A 0-dim (scalar) output is refused when more elements are expected.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeScalarOutputSize_n)
{
  GstTensorMemory in[1], out[1];
  const gsize size[1] = { 6 };

  SetModel (MakeModel ("scalar"));
  AddInput (_NNS_UINT8, "3:2");
  AddOutput (_NNS_UINT8, "3:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, size, 1, size, 1), -2);
  EXPECT_TRUE (AllBytes (&out[0], 0xEE));

  FreeTensors (in, 1);
  FreeTensors (out, 1);
}

/**
 * @brief An output whose first dimension is 0 is refused when its type differs.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeEmptyOutputType_n)
{
  GstTensorMemory in[1], out[1];
  const gsize in_size[1] = { 6 };
  const gsize out_size[1] = { 24 };

  SetModel (MakeModel ("empty"));
  AddInput (_NNS_UINT8, "3:2");
  AddOutput (_NNS_FLOAT32, "3:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, in_size, 1, out_size, 1), -2);
  EXPECT_TRUE (AllBytes (&out[0], 0xEE));

  FreeTensors (in, 1);
  FreeTensors (out, 1);
}

/**
 * @brief An output of another type than the configured one is refused.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeOutputType_n)
{
  GstTensorMemory in[1], out[1];
  const gsize size[1] = { 6 };

  SetModel (MakeModel ("bool"));
  AddInput (_NNS_UINT8, "3:2");
  AddOutput (_NNS_UINT8, "3:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, size, 1, size, 1), -2);
  EXPECT_TRUE (AllBytes (&out[0], 0xEE));

  FreeTensors (in, 1);
  FreeTensors (out, 1);
}

/**
 * @brief An output larger than the configured one is refused before the copy.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeOutputSize_n)
{
  GstTensorMemory in[1], out[1];
  const gsize size[1] = { 6 };

  SetModel (MakeModel ("grow"));
  AddInput (_NNS_UINT8, "3:2");
  AddOutput (_NNS_UINT8, "3:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, size, 1, size, 1), -2);
  EXPECT_TRUE (AllBytes (&out[0], 0xEE));

  FreeTensors (in, 1);
  FreeTensors (out, 1);
}

/**
 * @brief A wrong tensor inside a tuple fails the whole output.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeTupleOutputType_n)
{
  GstTensorMemory in[2], out[2];
  const gsize in_size[2] = { 6, 8 };
  const gsize out_size[2] = { 6, 32 };

  SetModel (MakeModel ("tuple"));
  AddInput (_NNS_UINT8, "3:2");
  AddInput (_NNS_UINT8, "4:2");
  AddOutput (_NNS_UINT8, "3:2");
  AddOutput (_NNS_FLOAT32, "4:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, in_size, 2, out_size, 2), -2);

  FreeTensors (in, 2);
  FreeTensors (out, 2);
}

/**
 * @brief A wrong tensor inside a list fails the whole output.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeListOutputType_n)
{
  GstTensorMemory in[2], out[2];
  const gsize in_size[2] = { 6, 8 };
  const gsize out_size[2] = { 6, 32 };

  SetModel (MakeModel ("list"));
  AddInput (_NNS_UINT8, "3:2");
  AddInput (_NNS_UINT8, "4:2");
  AddOutput (_NNS_UINT8, "3:2");
  AddOutput (_NNS_FLOAT32, "4:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, in_size, 2, out_size, 2), -2);

  FreeTensors (in, 2);
  FreeTensors (out, 2);
}

/**
 * @brief An output that is not a tensor, a tuple or a list is refused.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeUnsupportedOutput_n)
{
  GstTensorMemory in[1], out[1];
  const gsize size[1] = { 6 };

  SetModel (MakeModel ("none"));
  AddInput (_NNS_UINT8, "3:2");
  AddOutput (_NNS_UINT8, "3:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (in, out, size, 1, size, 1), -3);

  FreeTensors (in, 1);
  FreeTensors (out, 1);
}

/**
 * @brief Errors of the first run are caught, and the next run is caught too.
 */
TEST_F (NNStreamerFilterPytorchMockTest, invokeFirstRunThrows_n)
{
  const gchar *behaviours[] = { "throw_runtime", "throw_exception", "throw_unknown" };

  for (const gchar *behaviour : behaviours) {
    GstTensorMemory in[1], out[1];
    const gsize size[1] = { 6 };

    SCOPED_TRACE (behaviour);
    if (private_data)
      sp->close (&prop, &private_data);
    gst_tensors_info_free (&prop.input_meta);
    gst_tensors_info_free (&prop.output_meta);
    gst_tensors_info_init (&prop.input_meta);
    gst_tensors_info_init (&prop.output_meta);
    torch_mock::reset ();

    SetModel (MakeModel (behaviour));
    AddInput (_NNS_UINT8, "3:2");
    AddOutput (_NNS_UINT8, "3:2");
    ASSERT_EQ (Open (), 0);

    EXPECT_EQ (Invoke (in, out, size, 1, size, 1), -4);
    FreeTensors (in, 1);
    FreeTensors (out, 1);

    EXPECT_EQ (Invoke (in, out, size, 1, size, 1), -4);
    FreeTensors (in, 1);
    FreeTensors (out, 1);
    EXPECT_EQ (torch_mock::forwardCount (), 2);
  }
}

/**
 * @brief The sub-plugin runs in a pipeline through tensor_filter.
 */
TEST_F (NNStreamerFilterPytorchMockTest, pipeline)
{
  const gchar *model = MakeModel ("identity");
  gchar *desc = g_strdup_printf (
      "videotestsrc num-buffers=3 pattern=black ! "
      "video/x-raw,format=RGB,width=4,height=4,framerate=(fraction)30/1 ! "
      "tensor_converter ! tensor_filter framework=pytorch model=%s "
      "input=3:4:4:1 inputtype=uint8 output=3:4:4:1 outputtype=uint8 ! "
      "fakesink",
      model);
  GstElement *pipeline = gst_parse_launch (desc, NULL);
  GstBus *bus;
  GstMessage *msg;

  g_free (desc);
  ASSERT_NE (pipeline, nullptr);

  ASSERT_NE (gst_element_set_state (pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
  bus = gst_element_get_bus (pipeline);
  msg = gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  ASSERT_NE (msg, nullptr);
  EXPECT_EQ (GST_MESSAGE_TYPE (msg), GST_MESSAGE_EOS);
  gst_message_unref (msg);
  gst_object_unref (bus);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (pipeline);

  EXPECT_EQ (torch_mock::loadCount (), 1);
  EXPECT_EQ (torch_mock::forwardCount (), 3);
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

  for (int i = 1; i < argc; i++) {
    if (g_strcmp0 (argv[i], "--gpu-by-conf") == 0)
      gpu_by_conf = true;
  }
  if (gpu_by_conf)
    g_setenv ("NNSTREAMER_pytorch_enable_use_gpu", "true", TRUE);

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  /* A filter that matches nothing would turn a run into a no-op. */
  if (testing::UnitTest::GetInstance ()->test_to_run_count () == 0) {
    g_printerr ("No test case ran.\n");
    result = 1;
  }

  return result;
}
