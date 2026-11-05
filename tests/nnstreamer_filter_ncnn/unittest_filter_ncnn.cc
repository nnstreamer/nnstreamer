/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_ncnn.cc
 * @date    05 Nov 2026
 * @brief   Unit tests for the ncnn tensor filter sub-plugin
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * The cases run on passthrough.param, a model of a single Input layer that
 * hands its input straight back, so every output byte is known in advance.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <errno.h>
#include <glib.h>
#include <string.h>
#include <vector>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

#include <ncnn/cpu.h>

/**
 * @brief Whether ncnn can load a model on this host.
 * @details This is the very check the sub-plugin refuses on: ncnn counts no
 *          physical CPU where it cannot read cpuN/topology under
 *          /sys/devices/system/cpu, a GBS build chroot for instance.
 */
static bool
ncnnCanLoadModels (void)
{
  return ncnn::get_physical_cpu_count () > 0;
}

/**
 * @brief Get the path of a model in tests/test_models/models/ncnn_models.
 */
static gchar *
getNcnnModelPath (const gchar *name)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  g_autofree gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();

  return g_build_filename (
      root_path, "tests", "test_models", "models", "ncnn_models", name, NULL);
}

/**
 * @brief Fill tensors metadata from dimension and type strings, as the
 *        tensor_filter properties do. NULL leaves that part unset.
 */
static void
setTensorsInfo (GstTensorsInfo *info, const gchar *dims, const gchar *types)
{
  gst_tensors_info_free (info);
  gst_tensors_info_init (info);

  if (dims)
    info->num_tensors = gst_tensors_info_parse_dimensions_string (info, dims);
  if (types)
    gst_tensors_info_parse_types_string (info, types);
}

/**
 * @brief Fixture opening the ncnn sub-plugin on the passthrough model.
 */
class NNStreamerFilterNcnn : public ::testing::Test
{
  protected:
  const GstTensorFilterFramework *sp;
  GstTensorFilterProperties prop;
  gchar *param;
  gchar *bin;
  const gchar *models[3];
  std::vector<void *> instances; /**< opened and not closed yet */

  /**
   * @brief Find the sub-plugin and set the properties to the passthrough model.
   */
  void SetUp () override
  {
    sp = nnstreamer_filter_find ("ncnn");
    param = getNcnnModelPath ("passthrough.param");
    bin = getNcnnModelPath ("passthrough.bin");

    memset (&prop, 0, sizeof (prop));
    gst_tensors_info_init (&prop.input_meta);
    gst_tensors_info_init (&prop.output_meta);
    prop.fwname = "ncnn";
    models[0] = param;
    models[1] = bin;
    models[2] = NULL;
    prop.model_files = models;
    prop.num_models = 2;

    ASSERT_NE (sp, nullptr);
    ASSERT_TRUE (g_file_test (param, G_FILE_TEST_EXISTS));
    ASSERT_TRUE (g_file_test (bin, G_FILE_TEST_EXISTS));
    if (!ncnnCanLoadModels ())
      GTEST_SKIP () << "ncnn cannot read the CPU topology here, so it loads no model.";
  }

  /**
   * @brief Close what a failed case left open and release what SetUp allocated.
   */
  void TearDown () override
  {
    for (void *data : instances)
      sp->close (&prop, &data);
    instances.clear ();
    gst_tensors_info_free (&prop.input_meta);
    gst_tensors_info_free (&prop.output_meta);
    g_free (param);
    g_free (bin);
  }

  /**
   * @brief Open an instance with the given properties; NULL leaves one unset.
   */
  int openInstance (void **data, const gchar *in_dims, const gchar *in_types,
      const gchar *out_dims, const gchar *out_types, const gchar *custom)
  {
    setTensorsInfo (&prop.input_meta, in_dims, in_types);
    setTensorsInfo (&prop.output_meta, out_dims, out_types);
    prop.custom_properties = custom;
    *data = NULL;
    int ret = sp->open (&prop, data);
    if (ret == 0 && *data)
      instances.push_back (*data);
    return ret;
  }

  /**
   * @brief Close an instance that openInstance() created, if any.
   */
  void closeInstance (void **data)
  {
    if (*data) {
      instances.erase (std::remove (instances.begin (), instances.end (), *data),
          instances.end ());
      sp->close (&prop, data);
    }
    *data = NULL;
  }

  /**
   * @brief Run one inference of @a num_in floats into @a num_out floats.
   */
  int invokeOnce (void *data, const float *in, size_t num_in, float *out, size_t num_out)
  {
    GstTensorMemory input, output;

    input.data = (void *) in;
    input.size = num_in * sizeof (float);
    output.data = out;
    output.size = num_out * sizeof (float);
    return sp->invoke (NULL, &prop, data, &input, &output);
  }
};

/**
 * @brief Fill @a buf with values that differ at every position.
 */
static void
fillPattern (float *buf, size_t num)
{
  for (size_t i = 0; i < num; i++)
    buf[i] = 0.5f * (float) i - 3.0f;
}

/**
 * @brief The 3:2:4 shape: 6 floats a plane, which ncnn pads up to a cstep of 8.
 */
#define PLANE_PADDED_DIMS "3:2:4"
#define PLANE_PADDED_SIZE (3U * 2U * 4U)

/**
 * @brief A dimension padded with trailing 1s is the same tensor to ncnn.
 * @details Both instances report the same metadata, and both hand the input
 *          back byte for byte although every channel plane is padded in the
 *          ncnn matrix: a flat copy would shift each plane after the first.
 *          The buffers are on the heap, so memcheck sees a copy that overruns.
 */
TEST_F (NNStreamerFilterNcnn, trailingOnesKeepInfoAndBytes)
{
  void *plain = NULL, *padded = NULL;
  GstTensorsInfo in_plain, out_plain, in_padded, out_padded;
  const gsize bytes = PLANE_PADDED_SIZE * sizeof (float);
  g_autofree float *input = g_new0 (float, PLANE_PADDED_SIZE);
  g_autofree float *out_a = g_new0 (float, PLANE_PADDED_SIZE);
  g_autofree float *out_b = g_new0 (float, PLANE_PADDED_SIZE);

  gst_tensors_info_init (&in_plain);
  gst_tensors_info_init (&out_plain);
  gst_tensors_info_init (&in_padded);
  gst_tensors_info_init (&out_padded);

  ASSERT_EQ (openInstance (&plain, PLANE_PADDED_DIMS, "float32", NULL, NULL, NULL), 0);
  EXPECT_EQ (sp->getModelInfo (NULL, &prop, plain, GET_IN_OUT_INFO, &in_plain, &out_plain), 0);
  ASSERT_EQ (openInstance (&padded, PLANE_PADDED_DIMS ":1", "float32", NULL, NULL, NULL), 0);
  EXPECT_EQ (sp->getModelInfo (NULL, &prop, padded, GET_IN_OUT_INFO, &in_padded, &out_padded), 0);

  EXPECT_TRUE (gst_tensors_info_is_equal (&in_plain, &in_padded));
  EXPECT_TRUE (gst_tensors_info_is_equal (&out_plain, &out_padded));
  EXPECT_EQ (out_plain.num_tensors, 1U);
  EXPECT_EQ (out_plain.info[0].type, _NNS_FLOAT32);
  EXPECT_EQ (out_plain.info[0].dimension[0], 3U);
  EXPECT_EQ (out_plain.info[0].dimension[1], 2U);
  EXPECT_EQ (out_plain.info[0].dimension[2], 4U);
  EXPECT_EQ (gst_tensors_info_get_size (&out_plain, 0), bytes);

  fillPattern (input, PLANE_PADDED_SIZE);
  EXPECT_EQ (invokeOnce (plain, input, PLANE_PADDED_SIZE, out_a, PLANE_PADDED_SIZE), 0);
  EXPECT_EQ (invokeOnce (padded, input, PLANE_PADDED_SIZE, out_b, PLANE_PADDED_SIZE), 0);
  EXPECT_EQ (memcmp (out_a, input, bytes), 0);
  EXPECT_EQ (memcmp (out_b, input, bytes), 0);

  closeInstance (&plain);
  closeInstance (&padded);
  gst_tensors_info_free (&in_plain);
  gst_tensors_info_free (&out_plain);
  gst_tensors_info_free (&in_padded);
  gst_tensors_info_free (&out_padded);
}

/**
 * @brief A fifth axis of 1 is a trailing 1, which leaves 4 axes to ncnn.
 */
TEST_F (NNStreamerFilterNcnn, rankFiveTrailingOne)
{
  void *data = NULL;
  float input[3 * 2 * 2 * 2], output[3 * 2 * 2 * 2];

  ASSERT_EQ (openInstance (&data, "3:2:2:2:1", "float32", NULL, NULL, NULL), 0);
  fillPattern (input, G_N_ELEMENTS (input));
  EXPECT_EQ (invokeOnce (data, input, G_N_ELEMENTS (input), output, G_N_ELEMENTS (output)), 0);
  EXPECT_EQ (memcmp (output, input, sizeof (input)), 0);
  closeInstance (&data);
}

/**
 * @brief The output inferred from the model equals the one given explicitly.
 */
TEST_F (NNStreamerFilterNcnn, inferredOutputEqualsExplicit)
{
  void *inferred = NULL, *given = NULL;
  GstTensorsInfo in_a, out_a, in_b, out_b;
  float input[PLANE_PADDED_SIZE], res_a[PLANE_PADDED_SIZE], res_b[PLANE_PADDED_SIZE];

  gst_tensors_info_init (&in_a);
  gst_tensors_info_init (&out_a);
  gst_tensors_info_init (&in_b);
  gst_tensors_info_init (&out_b);

  ASSERT_EQ (openInstance (&inferred, PLANE_PADDED_DIMS, NULL, NULL, NULL, NULL), 0);
  EXPECT_EQ (sp->getModelInfo (NULL, &prop, inferred, GET_IN_OUT_INFO, &in_a, &out_a), 0);
  ASSERT_EQ (openInstance (&given, PLANE_PADDED_DIMS, "float32",
                 PLANE_PADDED_DIMS, "float32", NULL),
      0);
  EXPECT_EQ (sp->getModelInfo (NULL, &prop, given, GET_IN_OUT_INFO, &in_b, &out_b), 0);

  EXPECT_TRUE (gst_tensors_info_is_equal (&in_a, &in_b));
  EXPECT_TRUE (gst_tensors_info_is_equal (&out_a, &out_b));
  EXPECT_EQ (in_a.info[0].type, _NNS_FLOAT32);

  fillPattern (input, PLANE_PADDED_SIZE);
  EXPECT_EQ (invokeOnce (inferred, input, PLANE_PADDED_SIZE, res_a, PLANE_PADDED_SIZE), 0);
  EXPECT_EQ (invokeOnce (given, input, PLANE_PADDED_SIZE, res_b, PLANE_PADDED_SIZE), 0);
  EXPECT_EQ (memcmp (res_a, res_b, sizeof (res_a)), 0);
  EXPECT_EQ (memcmp (res_a, input, sizeof (input)), 0);

  closeInstance (&inferred);
  closeInstance (&given);
  gst_tensors_info_free (&in_a);
  gst_tensors_info_free (&out_a);
  gst_tensors_info_free (&in_b);
  gst_tensors_info_free (&out_b);
}

/**
 * @brief Without "input", the shape comes from SET_INPUT_INFO and the output
 *        follows it; a rejected candidate keeps the shape set before.
 */
TEST_F (NNStreamerFilterNcnn, inputInfoFromCaps)
{
  void *data = NULL;
  GstTensorsInfo in_info, out_info, bad_info, ignored;
  float input[PLANE_PADDED_SIZE], output[PLANE_PADDED_SIZE];

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);
  gst_tensors_info_init (&bad_info);
  gst_tensors_info_init (&ignored);

  ASSERT_EQ (openInstance (&data, NULL, NULL, NULL, NULL, NULL), 0);
  EXPECT_EQ (sp->getModelInfo (NULL, &prop, data, GET_IN_OUT_INFO, &in_info, &out_info),
      -ENOENT);

  setTensorsInfo (&in_info, PLANE_PADDED_DIMS, "float32");
  EXPECT_EQ (sp->getModelInfo (NULL, &prop, data, SET_INPUT_INFO, &in_info, &out_info), 0);
  EXPECT_EQ (out_info.num_tensors, 1U);
  EXPECT_EQ (out_info.info[0].type, _NNS_FLOAT32);
  EXPECT_EQ (gst_tensors_info_get_size (&out_info, 0), sizeof (output));

  setTensorsInfo (&bad_info, "5:2", "uint8");
  EXPECT_EQ (sp->getModelInfo (NULL, &prop, data, SET_INPUT_INFO, &bad_info, &ignored),
      -EINVAL);

  fillPattern (input, PLANE_PADDED_SIZE);
  EXPECT_EQ (invokeOnce (data, input, PLANE_PADDED_SIZE, output, PLANE_PADDED_SIZE), 0);
  EXPECT_EQ (memcmp (output, input, sizeof (input)), 0);

  closeInstance (&data);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  gst_tensors_info_free (&bad_info);
  gst_tensors_info_free (&ignored);
}

/**
 * @brief An input buffer of another size than configured is refused.
 */
TEST_F (NNStreamerFilterNcnn, invokeSizeMismatch_n)
{
  void *data = NULL;
  float input[PLANE_PADDED_SIZE + 1] = { 0 }, output[PLANE_PADDED_SIZE];

  ASSERT_EQ (openInstance (&data, PLANE_PADDED_DIMS, "float32", NULL, NULL, NULL), 0);
  EXPECT_NE (invokeOnce (data, input, PLANE_PADDED_SIZE + 1, output, PLANE_PADDED_SIZE), 0);
  EXPECT_NE (invokeOnce (data, input, PLANE_PADDED_SIZE, output, PLANE_PADDED_SIZE - 1), 0);
  closeInstance (&data);
}

/**
 * @brief ncnn handles float32 only, so another input type is refused.
 */
TEST_F (NNStreamerFilterNcnn, inputTypeUint8_n)
{
  void *data = NULL;

  EXPECT_NE (openInstance (&data, PLANE_PADDED_DIMS, "uint8", NULL, NULL, NULL), 0);
  EXPECT_EQ (data, nullptr);
}

/**
 * @brief ncnn handles float32 only, so another output type is refused.
 */
TEST_F (NNStreamerFilterNcnn, outputTypeInt32_n)
{
  void *data = NULL;

  EXPECT_NE (openInstance (&data, PLANE_PADDED_DIMS, "float32", PLANE_PADDED_DIMS, "int32", NULL),
      0);
  EXPECT_EQ (data, nullptr);
}

/**
 * @brief More input tensors than the model has inputs are refused.
 */
TEST_F (NNStreamerFilterNcnn, inputCountMismatch_n)
{
  void *data = NULL;

  EXPECT_NE (openInstance (&data, "3:2,3:2", "float32,float32", NULL, NULL, NULL), 0);
  EXPECT_EQ (data, nullptr);
}

/**
 * @brief An ncnn matrix has at most 4 axes.
 */
TEST_F (NNStreamerFilterNcnn, inputRankFive_n)
{
  void *data = NULL;

  EXPECT_NE (openInstance (&data, "3:2:2:2:2", "float32", NULL, NULL, NULL), 0);
  EXPECT_EQ (data, nullptr);
}

/**
 * @brief More than a param and bin pair is refused.
 */
TEST_F (NNStreamerFilterNcnn, modelCount_n)
{
  void *data = NULL;
  const gchar *three[] = { param, bin, param, NULL };

  prop.model_files = three;
  prop.num_models = 3;
  EXPECT_NE (openInstance (&data, PLANE_PADDED_DIMS, "float32", NULL, NULL, NULL), 0);
  EXPECT_EQ (data, nullptr);
  prop.model_files = models;
  prop.num_models = 2;
}

/**
 * @brief A single model file is read as a binary param, which a text one is not.
 * @details The very same properties open with the text param and its bin, so
 *          the refusal comes from the single-file path alone.
 */
TEST_F (NNStreamerFilterNcnn, singleTextParam_n)
{
  void *data = NULL;

  ASSERT_EQ (openInstance (&data, PLANE_PADDED_DIMS, "float32", NULL, NULL, NULL), 0);
  closeInstance (&data);

  prop.num_models = 1;
  EXPECT_NE (openInstance (&data, PLANE_PADDED_DIMS, "float32", NULL, NULL, NULL), 0);
  EXPECT_EQ (data, nullptr);
  prop.num_models = 2;
}

/**
 * @brief A param file that ncnn cannot parse is refused.
 */
TEST_F (NNStreamerFilterNcnn, invalidParamFile_n)
{
  void *data = NULL;

  /* the bin is empty, which is no valid param file */
  models[0] = bin;
  EXPECT_NE (openInstance (&data, PLANE_PADDED_DIMS, "float32", NULL, NULL, NULL), 0);
  EXPECT_EQ (data, nullptr);
  models[0] = param;
}

/**
 * @brief An unknown custom property is refused.
 */
TEST_F (NNStreamerFilterNcnn, customUnknown_n)
{
  void *data = NULL;

  EXPECT_NE (openInstance (&data, PLANE_PADDED_DIMS, "float32", NULL, NULL, "nakluv_esu:true"), 0);
  EXPECT_EQ (data, nullptr);
}

/**
 * @brief use_yolo_decoder takes true or false only.
 */
TEST_F (NNStreamerFilterNcnn, customYoloValue_n)
{
  void *data = NULL;

  EXPECT_NE (openInstance (&data, "6:3", "float32", "8:2", "float32", "use_yolo_decoder:fail"), 0);
  EXPECT_EQ (data, nullptr);
}

/**
 * @brief The yolo decoder output cannot be inferred from the model.
 */
TEST_F (NNStreamerFilterNcnn, yoloWithoutOutput_n)
{
  void *data = NULL;

  EXPECT_NE (openInstance (&data, "6:3", "float32", NULL, NULL, "use_yolo_decoder:true"), 0);
  EXPECT_EQ (data, nullptr);
}

/**
 * @brief A yolo output row with no room for the 5 box fields is refused.
 */
TEST_F (NNStreamerFilterNcnn, yoloOutputNarrow_n)
{
  void *data = NULL;

  EXPECT_NE (openInstance (&data, "6:3", "float32", "4:2", "float32", "use_yolo_decoder:true"), 0);
  EXPECT_EQ (data, nullptr);
}

/**
 * @brief A model row narrower than label, score and 4 corners is refused.
 */
TEST_F (NNStreamerFilterNcnn, yoloModelRowNarrow_n)
{
  void *data = NULL;
  float input[5 * 2] = { 0 }, output[10 * 2];

  ASSERT_EQ (openInstance (&data, "5:2", "float32", "10:2", "float32", "use_yolo_decoder:true"),
      0);
  EXPECT_NE (invokeOnce (data, input, 5 * 2, output, 10 * 2), 0);
  closeInstance (&data);
}

/**
 * @brief Whether ml_logw reaches GLib, where a log handler can count it.
 * @details On Tizen and Android it goes to dlog and the system log instead.
 */
#if !defined(__TIZEN__) && !defined(__ANDROID__)
#define SUBPLUGIN_WARNINGS_COUNTED (true)
#else
#define SUBPLUGIN_WARNINGS_COUNTED (false)
#endif

/**
 * @brief Count the warnings logged in the default domain.
 */
static void
countWarning (const gchar *domain, GLogLevelFlags level, const gchar *message, gpointer user_data)
{
  UNUSED (domain);
  UNUSED (level);
  UNUSED (message);
  (*(guint *) user_data)++;
}

/**
 * @brief Count the warnings logged in the default domain from now on.
 * @return The handler, which the caller removes before @a warnings goes away.
 */
static guint
startWarningCount (guint *warnings)
{
  guint handler = g_log_set_handler (NULL, G_LOG_LEVEL_WARNING, countWarning, warnings);

  g_warning ("[probe] ncnn warning handler self-check");
  EXPECT_EQ (*warnings, 1U) << "no live handler for warnings";
  *warnings = 0;
  return handler;
}

/**
 * @brief Decode detections the model hands over as rows of 6 values.
 * @details Each row is a label, a score and the corners x1, y1, x2, y2. The
 *          output holds 2 detections of 5 box fields and 3 label slots each,
 *          so the third row is dropped, and that is reported once only. The
 *          float past the output tensor catches a label slot written past it.
 */
TEST_F (NNStreamerFilterNcnn, yoloDecode)
{
  void *data = NULL;
  /* corners to clamp into 0..1, a label past the last slot, a row too many */
  const float rows[3][6] = {
    { 1.0f, 0.9f, -0.5f, 0.2f, 0.6f, 1.5f },
    { 3.0f, 0.4f, 0.1f, 0.1f, 0.3f, 0.5f },
    { 0.0f, 0.7f, 0.0f, 0.0f, 1.0f, 1.0f },
  };
  float output[8 * 2 + 1];
  guint warnings = 0;

  ASSERT_EQ (openInstance (&data, "6:3", "float32", "8:2", "float32", "use_yolo_decoder:true"), 0);

  guint handler = startWarningCount (&warnings);

  for (int round = 0; round < 2; round++) {
    for (size_t i = 0; i < G_N_ELEMENTS (output); i++)
      output[i] = 7.0f;

    EXPECT_EQ (invokeOnce (data, (const float *) rows, 3 * 6, output, 8 * 2), 0);

    EXPECT_FLOAT_EQ (output[0], 0.3f);
    EXPECT_FLOAT_EQ (output[1], 0.6f);
    EXPECT_FLOAT_EQ (output[2], 0.6f);
    EXPECT_FLOAT_EQ (output[3], 0.8f);
    EXPECT_FLOAT_EQ (output[4], 0.9f);
    EXPECT_FLOAT_EQ (output[5], 0.0f);
    EXPECT_FLOAT_EQ (output[6], 1.0f);
    EXPECT_FLOAT_EQ (output[7], 0.0f);

    EXPECT_FLOAT_EQ (output[8 + 0], 0.2f);
    EXPECT_FLOAT_EQ (output[8 + 1], 0.3f);
    EXPECT_FLOAT_EQ (output[8 + 2], 0.2f);
    EXPECT_FLOAT_EQ (output[8 + 3], 0.4f);
    EXPECT_FLOAT_EQ (output[8 + 4], 0.4f);
    EXPECT_FLOAT_EQ (output[8 + 5], 0.0f);
    EXPECT_FLOAT_EQ (output[8 + 6], 0.0f);
    EXPECT_FLOAT_EQ (output[8 + 7], 0.0f);
    EXPECT_FLOAT_EQ (output[8 * 2], 7.0f);
  }

  g_log_remove_handler (NULL, handler);
  if (SUBPLUGIN_WARNINGS_COUNTED) {
    EXPECT_EQ (warnings, 1U);
  }

  closeInstance (&data);
}

/**
 * @brief Detections that fit are all decoded, a negative label sets no slot,
 *        and nothing is reported as dropped.
 */
TEST_F (NNStreamerFilterNcnn, yoloDecodeFits)
{
  void *data = NULL;
  const float rows[2][6] = {
    { -1.0f, 0.5f, 0.0f, 0.0f, 0.5f, 0.5f },
    { 2.0f, 0.8f, 0.5f, 0.5f, 1.0f, 1.0f },
  };
  float output[8 * 2];
  guint warnings = 0;

  ASSERT_EQ (openInstance (&data, "6:2", "float32", "8:2", "float32", "use_yolo_decoder:true"), 0);

  guint handler = startWarningCount (&warnings);

  EXPECT_EQ (invokeOnce (data, (const float *) rows, 2 * 6, output, G_N_ELEMENTS (output)), 0);

  g_log_remove_handler (NULL, handler);
  EXPECT_EQ (warnings, 0U);

  EXPECT_FLOAT_EQ (output[0], 0.25f);
  EXPECT_FLOAT_EQ (output[1], 0.25f);
  EXPECT_FLOAT_EQ (output[2], 0.5f);
  EXPECT_FLOAT_EQ (output[3], 0.5f);
  EXPECT_FLOAT_EQ (output[4], 0.5f);
  EXPECT_FLOAT_EQ (output[5], 0.0f);
  EXPECT_FLOAT_EQ (output[6], 0.0f);
  EXPECT_FLOAT_EQ (output[7], 0.0f);

  EXPECT_FLOAT_EQ (output[8 + 0], 0.75f);
  EXPECT_FLOAT_EQ (output[8 + 1], 0.75f);
  EXPECT_FLOAT_EQ (output[8 + 2], 0.5f);
  EXPECT_FLOAT_EQ (output[8 + 3], 0.5f);
  EXPECT_FLOAT_EQ (output[8 + 4], 0.8f);
  EXPECT_FLOAT_EQ (output[8 + 5], 0.0f);
  EXPECT_FLOAT_EQ (output[8 + 6], 0.0f);
  EXPECT_FLOAT_EQ (output[8 + 7], 1.0f);

  closeInstance (&data);
}

/**
 * @brief Where ncnn cannot read the CPU topology, the sub-plugin refuses to
 *        open instead of letting ncnn divide by zero while loading.
 */
TEST (NNStreamerFilterNcnnHost, refusedWithoutCpuTopology_n)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("ncnn");
  g_autofree gchar *param = getNcnnModelPath ("passthrough.param");
  g_autofree gchar *bin = getNcnnModelPath ("passthrough.bin");
  const gchar *models[] = { param, bin, NULL };
  GstTensorFilterProperties prop;
  void *data = NULL;

  ASSERT_NE (sp, nullptr);
  if (ncnnCanLoadModels ())
    GTEST_SKIP () << "ncnn reads the CPU topology here; the refusal does not apply.";

  memset (&prop, 0, sizeof (prop));
  gst_tensors_info_init (&prop.input_meta);
  gst_tensors_info_init (&prop.output_meta);
  prop.fwname = "ncnn";
  prop.model_files = models;
  prop.num_models = 2;

  EXPECT_NE (sp->open (&prop, &data), 0);
  EXPECT_EQ (data, nullptr);
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
