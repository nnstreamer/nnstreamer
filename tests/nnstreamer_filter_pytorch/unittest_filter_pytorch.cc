/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_filter_pytorch.cc
 * @date        12 Nov 2026
 * @brief       Unit test for the pytorch tensor-filter sub-plugin on a real LibTorch
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 *
 * @details
 * unittest_filter_pytorch_mock covers the logic of the sub-plugin on a
 * stand-in of the LibTorch API, which cannot tell where it drifts from
 * LibTorch. This test runs the built sub-plugin on the LibTorch it was built
 * against, with TorchScript models that it scripts and saves itself, so that
 * the models always match that LibTorch.
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>
#include <torch/script.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api_util.h>
#include <tensor_typedef.h>

/**
 * @brief Model returning the sum of its input, a 0-dim (scalar) tensor
 */
static const char scalar_model[] = "def forward(self, x):\n"
                                   "  return x.sum()\n";

/**
 * @brief Model returning a tuple of two tensors of different shapes
 */
static const char tuple_model[] = "def forward(self, x):\n"
                                  "  return x + 1, x.sum(0)\n";

/**
 * @brief Model returning a tuple of three tensors
 */
static const char tuple3_model[] = "def forward(self, x):\n"
                                   "  return x + 1, x.sum(0), x * 2\n";

/**
 * @brief Model returning a list of two tensors of different shapes
 */
static const char list_model[] = "def forward(self, x):\n"
                                 "  return [x + 1, x.sum(0)]\n";

/**
 * @brief Model returning a transposed, so non-contiguous, tensor of 0 to 5
 */
static const char transposed_model[]
    = "def forward(self, x):\n"
      "  y = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=x.dtype)\n"
      "  return y.t()\n";

/**
 * @brief Model returning an integer, which is not a tensor
 */
static const char int_model[] = "def forward(self, x):\n"
                                "  return 3\n";

/**
 * @brief Model raising an exception on a negative input
 */
static const char raise_model[] = "def forward(self, x):\n"
                                  "  assert not bool((x < 0).any()), 'negative input'\n"
                                  "  return x\n";

/**
 * @brief Test fixture for the pytorch tensor-filter sub-plugin.
 */
class NNStreamerFilterPytorchTest : public ::testing::Test
{
  protected:
  const GstTensorFilterFramework *sp;
  GstTensorFilterProperties prop;
  const gchar *model_files[2];
  void *private_data;
  gchar *tmpdir;
  gchar *model_path;

  public:
  /**
   * @brief Construct a new test object
   */
  NNStreamerFilterPytorchTest ()
      : sp (nullptr), private_data (nullptr), tmpdir (nullptr), model_path (nullptr)
  {
    model_files[0] = model_files[1] = nullptr;
  }

  /**
   * @brief Find the sub-plugin and prepare a directory for the models
   */
  void SetUp () override
  {
    memset (&prop, 0, sizeof (GstTensorFilterProperties));
    gst_tensors_info_init (&prop.input_meta);
    gst_tensors_info_init (&prop.output_meta);

    sp = nnstreamer_filter_find ("pytorch");
    ASSERT_NE (sp, nullptr);

    tmpdir = g_dir_make_tmp ("nns-pytorch-XXXXXX", NULL);
    ASSERT_NE (tmpdir, nullptr);
  }

  /**
   * @brief Close the sub-plugin and remove the model
   */
  void TearDown () override
  {
    if (sp && private_data)
      sp->close (&prop, &private_data);

    gst_tensors_info_free (&prop.input_meta);
    gst_tensors_info_free (&prop.output_meta);

    if (model_path) {
      g_remove (model_path);
      g_free (model_path);
    }
    if (tmpdir) {
      g_rmdir (tmpdir);
      g_free (tmpdir);
    }
  }

  /**
   * @brief Script a module whose methods are given in TorchScript and save it
   * @return The path of the saved model, nullptr on a LibTorch error
   */
  const gchar *SaveModel (const char *src)
  {
    gchar *path = g_build_filename (tmpdir, "model.pt", NULL);

    try {
      torch::jit::Module module ("nns_test_model");

      module.define (src);
      module.save (path);
    } catch (const std::exception &e) {
      ADD_FAILURE () << "Cannot script the model: " << e.what ();
      g_free (path);
      return nullptr;
    }

    g_free (model_path);
    model_path = path;
    return path;
  }

  /**
   * @brief Script and save the model, and give it to the filter
   */
  bool SetModel (const char *src)
  {
    model_files[0] = SaveModel (src);
    prop.fwname = "pytorch";
    prop.model_files = model_files;
    prop.num_models = 1;
    prop.accl_str = "cpu";
    return model_files[0] != nullptr;
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
   * @brief Invoke the filter on one float32 input filled with the given value
   * @details Every output is filled with -100.0 before the call.
   */
  int Invoke (float value, guint in_count, float **out, const guint *out_count, guint num_out)
  {
    GstTensorMemory in_mem, out_mem[NNS_TENSOR_SIZE_LIMIT];
    float *in = g_new (float, in_count);
    int ret;

    for (guint i = 0; i < in_count; i++)
      in[i] = value;
    in_mem.data = in;
    in_mem.size = in_count * sizeof (float);

    for (guint i = 0; i < num_out; i++) {
      out[i] = g_new (float, out_count[i]);
      for (guint j = 0; j < out_count[i]; j++)
        out[i][j] = -100.0f;
      out_mem[i].data = out[i];
      out_mem[i].size = out_count[i] * sizeof (float);
    }

    ret = sp->invoke (sp, &prop, private_data, &in_mem, out_mem);
    g_free (in);
    return ret;
  }

  /**
   * @brief Free the outputs of an Invoke () call
   */
  static void FreeOutputs (float **out, guint num_out)
  {
    for (guint i = 0; i < num_out; i++) {
      g_free (out[i]);
      out[i] = nullptr;
    }
  }

  /**
   * @brief Whether every element of the output has the given value
   */
  static bool AllValues (const float *out, guint count, float value)
  {
    for (guint i = 0; i < count; i++) {
      if (out[i] != value)
        return false;
    }
    return true;
  }
};

/**
 * @brief A 0-dim (scalar) output is copied out, on every invoke.
 */
TEST_F (NNStreamerFilterPytorchTest, invokeScalarOutput)
{
  float *out[1];
  const guint out_count[1] = { 1 };

  ASSERT_TRUE (SetModel (scalar_model));
  AddInput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "1");
  ASSERT_EQ (Open (), 0);

  for (int i = 0; i < 2; i++) {
    EXPECT_EQ (Invoke (1.5f, 6, out, out_count, 1), 0);
    EXPECT_FLOAT_EQ (out[0][0], 9.0f);
    FreeOutputs (out, 1);
  }
}

/**
 * @brief A 0-dim (scalar) output is refused when its type differs.
 */
TEST_F (NNStreamerFilterPytorchTest, invokeScalarOutputType_n)
{
  float *out[1];
  const guint out_count[1] = { 1 };

  ASSERT_TRUE (SetModel (scalar_model));
  AddInput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_INT32, "1");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (1.5f, 6, out, out_count, 1), -EINVAL);
  EXPECT_TRUE (AllValues (out[0], out_count[0], -100.0f));
  FreeOutputs (out, 1);
}

/**
 * @brief A 0-dim (scalar) output is refused for an output of two elements.
 */
TEST_F (NNStreamerFilterPytorchTest, invokeScalarOutputSize_n)
{
  float *out[1];
  const guint out_count[1] = { 2 };

  ASSERT_TRUE (SetModel (scalar_model));
  AddInput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (1.5f, 6, out, out_count, 1), -EINVAL);
  EXPECT_TRUE (AllValues (out[0], out_count[0], -100.0f));
  FreeOutputs (out, 1);
}

/**
 * @brief The tensors of a returned tuple fill the outputs in order.
 */
TEST_F (NNStreamerFilterPytorchTest, invokeTuple)
{
  float *out[2];
  const guint out_count[2] = { 6, 3 };

  ASSERT_TRUE (SetModel (tuple_model));
  AddInput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "3");
  ASSERT_EQ (Open (), 0);

  for (int i = 0; i < 2; i++) {
    EXPECT_EQ (Invoke (1.5f, 6, out, out_count, 2), 0);
    EXPECT_TRUE (AllValues (out[0], out_count[0], 2.5f));
    EXPECT_TRUE (AllValues (out[1], out_count[1], 3.0f));
    FreeOutputs (out, 2);
  }
}

/**
 * @brief A tuple is refused when one of its tensors does not fit its output.
 */
TEST_F (NNStreamerFilterPytorchTest, invokeTupleOutputSize_n)
{
  float *out[2];
  const guint out_count[2] = { 6, 6 };

  ASSERT_TRUE (SetModel (tuple_model));
  AddInput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "3:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (1.5f, 6, out, out_count, 2), -EINVAL);
  EXPECT_TRUE (AllValues (out[1], out_count[1], -100.0f));
  FreeOutputs (out, 2);
}

/**
 * @brief The tensors of a tuple beyond the number of outputs are dropped.
 */
TEST_F (NNStreamerFilterPytorchTest, invokeTupleExtraTensors)
{
  float *out[2];
  const guint out_count[2] = { 6, 3 };

  ASSERT_TRUE (SetModel (tuple3_model));
  AddInput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "3");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (1.5f, 6, out, out_count, 2), 0);
  EXPECT_TRUE (AllValues (out[0], out_count[0], 2.5f));
  EXPECT_TRUE (AllValues (out[1], out_count[1], 3.0f));
  FreeOutputs (out, 2);
}

/**
 * @brief The tensors of a returned list fill the outputs in order.
 */
TEST_F (NNStreamerFilterPytorchTest, invokeList)
{
  float *out[2];
  const guint out_count[2] = { 6, 3 };

  ASSERT_TRUE (SetModel (list_model));
  AddInput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "3");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (1.5f, 6, out, out_count, 2), 0);
  EXPECT_TRUE (AllValues (out[0], out_count[0], 2.5f));
  EXPECT_TRUE (AllValues (out[1], out_count[1], 3.0f));
  FreeOutputs (out, 2);
}

/**
 * @brief A list is refused when one of its tensors does not fit its output.
 */
TEST_F (NNStreamerFilterPytorchTest, invokeListOutputSize_n)
{
  float *out[2];
  const guint out_count[2] = { 6, 6 };

  ASSERT_TRUE (SetModel (list_model));
  AddInput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "3:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (1.5f, 6, out, out_count, 2), -EINVAL);
  EXPECT_TRUE (AllValues (out[1], out_count[1], -100.0f));
  FreeOutputs (out, 2);
}

/**
 * @brief A non-contiguous output is copied out in its logical order.
 */
TEST_F (NNStreamerFilterPytorchTest, invokeNonContiguousOutput)
{
  float *out[1];
  const guint out_count[1] = { 6 };
  const float expected[6] = { 0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f };

  ASSERT_TRUE (SetModel (transposed_model));
  AddInput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "2:3");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (1.5f, 6, out, out_count, 1), 0);
  for (guint i = 0; i < out_count[0]; i++)
    EXPECT_FLOAT_EQ (out[0][i], expected[i]);
  FreeOutputs (out, 1);
}

/**
 * @brief An output that is neither a tensor, a tuple nor a list is refused.
 */
TEST_F (NNStreamerFilterPytorchTest, invokeUnsupportedOutput_n)
{
  float *out[1];
  const guint out_count[1] = { 6 };

  ASSERT_TRUE (SetModel (int_model));
  AddInput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "3:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (1.5f, 6, out, out_count, 1), -EINVAL);
  EXPECT_TRUE (AllValues (out[0], out_count[0], -100.0f));
  FreeOutputs (out, 1);
}

/**
 * @brief An exception raised by the model fails that invoke only.
 */
TEST_F (NNStreamerFilterPytorchTest, invokeRaises_n)
{
  float *out[1];
  const guint out_count[1] = { 6 };

  ASSERT_TRUE (SetModel (raise_model));
  AddInput (_NNS_FLOAT32, "3:2");
  AddOutput (_NNS_FLOAT32, "3:2");
  ASSERT_EQ (Open (), 0);

  EXPECT_EQ (Invoke (1.5f, 6, out, out_count, 1), 0);
  EXPECT_TRUE (AllValues (out[0], out_count[0], 1.5f));
  FreeOutputs (out, 1);

  EXPECT_EQ (Invoke (-1.0f, 6, out, out_count, 1), -EINVAL);
  EXPECT_TRUE (AllValues (out[0], out_count[0], -100.0f));
  FreeOutputs (out, 1);

  EXPECT_EQ (Invoke (2.0f, 6, out, out_count, 1), 0);
  EXPECT_TRUE (AllValues (out[0], out_count[0], 2.0f));
  FreeOutputs (out, 1);
}

/**
 * @brief Run a float32 pipeline of tensor_filter until EOS or an error
 * @details Only how the run ends is checked; the invoke cases check the values.
 * @param model The path of the model
 * @param add The value added to the input, which is 0 (black) before
 * @param output The output dimension of the filter
 * @return The type of the message that ended the run, GST_MESSAGE_UNKNOWN on a timeout
 */
static GstMessageType
runPipeline (const gchar *model, int add, const gchar *output)
{
  gchar *desc = g_strdup_printf (
      "videotestsrc num-buffers=3 pattern=black ! "
      "video/x-raw,format=RGB,width=4,height=4,framerate=(fraction)30/1 ! "
      "tensor_converter ! "
      "tensor_transform mode=arithmetic option=typecast:float32,add:%d ! "
      "tensor_filter framework=pytorch model=%s "
      "input=3:4:4:1 inputtype=float32 output=%s outputtype=float32 ! "
      "fakesink",
      add, model, output);
  GstElement *pipeline = gst_parse_launch (desc, NULL);
  GstMessageType type = GST_MESSAGE_UNKNOWN;
  GstBus *bus;
  GstMessage *msg;

  g_free (desc);
  if (pipeline == nullptr)
    return type;

  if (gst_element_set_state (pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE) {
    bus = gst_element_get_bus (pipeline);
    msg = gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
        (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
    if (msg) {
      type = GST_MESSAGE_TYPE (msg);
      gst_message_unref (msg);
    }
    gst_object_unref (bus);
  }

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (pipeline);
  return type;
}

/**
 * @brief A pipeline runs a model with a scalar output to EOS.
 */
TEST_F (NNStreamerFilterPytorchTest, pipelineScalarOutput)
{
  const gchar *model = SaveModel (scalar_model);

  ASSERT_NE (model, nullptr);
  EXPECT_EQ (runPipeline (model, 0, "1"), GST_MESSAGE_EOS);
}

/**
 * @brief A pipeline posts an error, rather than aborting, when the model raises.
 */
TEST_F (NNStreamerFilterPytorchTest, pipelineRaises_n)
{
  const gchar *model = SaveModel (raise_model);

  ASSERT_NE (model, nullptr);
  EXPECT_EQ (runPipeline (model, 0, "3:4:4:1"), GST_MESSAGE_EOS);
  /* black frames plus -1 are negative, which the model refuses */
  EXPECT_EQ (runPipeline (model, -1, "3:4:4:1"), GST_MESSAGE_ERROR);
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

  /* A filter that matches nothing would turn a run into a no-op. */
  if (testing::UnitTest::GetInstance ()->test_to_run_count () == 0) {
    g_printerr ("No test case ran.\n");
    result = 1;
  }

  return result;
}
