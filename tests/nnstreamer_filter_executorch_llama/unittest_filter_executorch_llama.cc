/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_executorch_llama.cc
 * @date    20 Oct 2026
 * @brief   Unit test for the ExecuTorch Llama tensor filter sub-plugin
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 * llm_next_letter.pte is not trained; generateModel.py builds it to predict
 * the next token from the last one alone, so the generated text is exact: a
 * lowercase letter is followed by the next letter and 'z' by <|end_of_text|>,
 * a digit by the next digit (forever), anything else by 'a'. The prompt is
 * echoed in front of the generated text; the text of <|end_of_text|> is left
 * out of what is generated.
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>

#include <cstring>
#include <string>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api_util.h>

#define FW_NAME "executorch-llama"
#define LLM_MODEL "llm_next_letter.pte"
#define LLM_TOKENIZER "llm_next_letter_tokenizer.model"
#define LLM_EOS "<|end_of_text|>"

/**
 * @brief Build the path of a fixture under tests/test_models/models.
 */
static gchar *
_GetModelFilePath (const gchar *model_name)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  g_autofree gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();

  return g_build_filename (root_path, "tests", "test_models", "models", model_name, NULL);
}

/**
 * @brief Fill in the filter properties the framework callbacks expect.
 */
static void
_SetFilterProp (GstTensorFilterProperties *prop, const gchar **models)
{
  memset (prop, 0, sizeof (GstTensorFilterProperties));
  prop->fwname = FW_NAME;
  prop->fw_opened = 0;
  prop->model_files = models;
  prop->num_models = models ? g_strv_length ((gchar **) models) : 0;
  prop->invoke_dynamic = TRUE;
}

/**
 * @brief Test fixture holding the model and tokenizer paths.
 */
class nnstreamerFilterExecutorchLlama : public ::testing::Test
{
  protected:
  gchar *model;
  gchar *tokenizer;
  const GstTensorFilterFramework *sp;

  /**
   * @brief Find the fixtures and the sub-plugin.
   */
  void SetUp () override
  {
    model = _GetModelFilePath (LLM_MODEL);
    tokenizer = _GetModelFilePath (LLM_TOKENIZER);
    ASSERT_TRUE (g_file_test (model, G_FILE_TEST_IS_REGULAR));
    ASSERT_TRUE (g_file_test (tokenizer, G_FILE_TEST_IS_REGULAR));
    sp = nnstreamer_filter_find (FW_NAME);
    ASSERT_TRUE (sp != nullptr);
  }

  /**
   * @brief Free the fixture paths.
   */
  void TearDown () override
  {
    g_free (model);
    g_free (tokenizer);
  }

  /**
   * @brief Run one prompt through an opened instance and return the output text.
   */
  int generate (GstTensorFilterProperties *prop, void **data,
      const std::string &prompt, std::string &result)
  {
    GstTensorMemory input, output;
    int ret;

    input.data = (gpointer) prompt.data ();
    input.size = prompt.size ();
    output.data = NULL;
    output.size = 0;

    ret = sp->invoke (sp, prop, *data, &input, &output);
    if (ret == 0) {
      GstTensorInfo *info = gst_tensors_info_get_nth_info (&prop->output_meta, 0);

      result.assign ((const char *) output.data, output.size);
      EXPECT_EQ (prop->output_meta.num_tensors, 1U);
      EXPECT_EQ (prop->output_meta.format, _NNS_TENSOR_FORMAT_FLEXIBLE);
      EXPECT_EQ (info->type, _NNS_UINT8);
      EXPECT_EQ (info->dimension[0], output.size);
    }
    g_free (output.data);
    return ret;
  }
};

/**
 * @brief Check the executorch-llama sub-plugin is registered.
 */
TEST_F (nnstreamerFilterExecutorchLlama, checkExistence)
{
  EXPECT_EQ (sp->version, GST_TENSOR_FILTER_FRAMEWORK_V1);
}

/**
 * @brief Positive case with open/close.
 */
TEST_F (nnstreamerFilterExecutorchLlama, openClose)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;

  _SetFilterProp (&prop, model_files);
  EXPECT_EQ (sp->open (&prop, &data), 0);
  sp->close (&prop, &data);
}

/**
 * @brief The prompt is echoed and generation ends at the end-of-text token,
 *        whose text is not part of the output.
 */
TEST_F (nnstreamerFilterExecutorchLlama, generateUntilEos)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  std::string result;

  _SetFilterProp (&prop, model_files);
  ASSERT_EQ (sp->open (&prop, &data), 0);

  EXPECT_EQ (generate (&prop, &data, "Hi, w", result), 0);
  EXPECT_EQ (result, "Hi, wxyz");

  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief The echoed prompt is kept as it is, even when it is the text of the
 *        end-of-text token; only the generated one is left out.
 */
TEST_F (nnstreamerFilterExecutorchLlama, generateEosInPrompt)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  std::string result;

  _SetFilterProp (&prop, model_files);
  ASSERT_EQ (sp->open (&prop, &data), 0);

  EXPECT_EQ (generate (&prop, &data, LLM_EOS, result), 0);
  EXPECT_EQ (result, std::string (LLM_EOS) + "abcdefghijklmnopqrstuvwxyz");

  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief Generation stops at 128 tokens in all, the prompt included.
 */
TEST_F (nnstreamerFilterExecutorchLlama, generateUpToSeqLen)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  std::string result, expected;
  guint i;

  for (i = 0; i < 128; i++)
    expected += (char) ('0' + (3 + i) % 10);

  _SetFilterProp (&prop, model_files);
  ASSERT_EQ (sp->open (&prop, &data), 0);

  EXPECT_EQ (generate (&prop, &data, "345", result), 0);
  EXPECT_EQ (result, expected);

  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief Each buffer is a new prompt, not a continuation of the previous one.
 *
 * The runner keeps its position across calls. Without a reset the second
 * prompt would start past position 128 and have no room left to generate.
 */
TEST_F (nnstreamerFilterExecutorchLlama, generateRepeatedly)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  std::string first, second, third;

  _SetFilterProp (&prop, model_files);
  ASSERT_EQ (sp->open (&prop, &data), 0);

  EXPECT_EQ (generate (&prop, &data, "0", first), 0);
  EXPECT_EQ (first.size (), 128U);
  EXPECT_EQ (generate (&prop, &data, "0", second), 0);
  EXPECT_EQ (second, first);
  EXPECT_EQ (generate (&prop, &data, "x", third), 0);
  EXPECT_EQ (third, "xyz");

  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case: only invoke-dynamic is supported.
 */
TEST_F (nnstreamerFilterExecutorchLlama, invokeStatic_n)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  const char prompt[] = "abc";
  GstTensorMemory input, output;

  _SetFilterProp (&prop, model_files);
  ASSERT_EQ (sp->open (&prop, &data), 0);

  prop.invoke_dynamic = FALSE;
  input.data = (gpointer) prompt;
  input.size = sizeof (prompt) - 1;
  output.data = NULL;
  output.size = 0;
  EXPECT_NE (sp->invoke (sp, &prop, data, &input, &output), 0);
  EXPECT_TRUE (output.data == NULL);

  sp->close (&prop, &data);
}

/**
 * @brief Negative case: an empty prompt has nothing to start from.
 */
TEST_F (nnstreamerFilterExecutorchLlama, invokeEmptyPrompt_n)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  std::string result;

  _SetFilterProp (&prop, model_files);
  ASSERT_EQ (sp->open (&prop, &data), 0);

  EXPECT_NE (generate (&prop, &data, "", result), 0);

  sp->close (&prop, &data);
}

/**
 * @brief Negative case: a prompt that leaves no room to generate is an error.
 *
 * 128 prompt tokens fill the filter's limit, 256 the model's context. Both
 * aborted the process before. The instance stays usable afterwards.
 */
TEST_F (nnstreamerFilterExecutorchLlama, invokeLongPrompt_n)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  std::string result;

  _SetFilterProp (&prop, model_files);
  ASSERT_EQ (sp->open (&prop, &data), 0);

  EXPECT_NE (generate (&prop, &data, std::string (128, '0'), result), 0);
  EXPECT_NE (generate (&prop, &data, std::string (256, '0'), result), 0);
  EXPECT_NE (generate (&prop, &data, std::string (300, '0'), result), 0);

  EXPECT_EQ (generate (&prop, &data, std::string (127, '0'), result), 0);
  EXPECT_EQ (result, std::string (127, '0') + "1");

  gst_tensors_info_free (&prop.output_meta);
  sp->close (&prop, &data);
}

/**
 * @brief Negative case: the tokenizer is not given.
 */
TEST_F (nnstreamerFilterExecutorchLlama, openNoTokenizer_n)
{
  void *data = NULL;
  const gchar *model_files[] = { model, NULL };
  const gchar *two_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;

  _SetFilterProp (&prop, model_files);
  EXPECT_NE (sp->open (&prop, &data), 0);

  /* The count decides, not what lies past the end of the list. */
  _SetFilterProp (&prop, two_files);
  prop.num_models = 1;
  EXPECT_NE (sp->open (&prop, &data), 0);
}

/**
 * @brief Negative case: a file past the model and the tokenizer is not ignored.
 */
TEST_F (nnstreamerFilterExecutorchLlama, openExtraModelFile_n)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, tokenizer, NULL };
  GstTensorFilterProperties prop;

  _SetFilterProp (&prop, model_files);
  EXPECT_NE (sp->open (&prop, &data), 0);
}

/**
 * @brief Negative case: the model path does not exist.
 */
TEST_F (nnstreamerFilterExecutorchLlama, openNoModelFile_n)
{
  void *data = NULL;
  const gchar *model_files[] = { "some/invalid/model/path.pte", tokenizer, NULL };
  GstTensorFilterProperties prop;

  _SetFilterProp (&prop, model_files);
  EXPECT_NE (sp->open (&prop, &data), 0);
}

/**
 * @brief Negative case: the tokenizer path does not exist.
 */
TEST_F (nnstreamerFilterExecutorchLlama, openNoTokenizerFile_n)
{
  void *data = NULL;
  const gchar *model_files[] = { model, "some/invalid/tokenizer.model", NULL };
  GstTensorFilterProperties prop;

  _SetFilterProp (&prop, model_files);
  EXPECT_NE (sp->open (&prop, &data), 0);
}

/**
 * @brief Negative case: the model file is not an ExecuTorch program.
 */
TEST_F (nnstreamerFilterExecutorchLlama, openInvalidModel_n)
{
  void *data = NULL;
  const gchar *model_files[] = { tokenizer, tokenizer, NULL };
  GstTensorFilterProperties prop;

  _SetFilterProp (&prop, model_files);
  EXPECT_NE (sp->open (&prop, &data), 0);
}

/**
 * @brief Negative case: the tokenizer file is not in any format the runner reads.
 */
TEST_F (nnstreamerFilterExecutorchLlama, openInvalidTokenizer_n)
{
  void *data = NULL;
  g_autofree gchar *garbage_file = NULL;
  gint fd;

  fd = g_file_open_tmp ("executorch_llama_garbage_XXXXXX.model", &garbage_file, NULL);
  ASSERT_GE (fd, 0);
  ASSERT_TRUE (g_file_set_contents (garbage_file, "This is not a tokenizer.", -1, NULL));
  g_close (fd, NULL);

  const gchar *model_files[] = { model, garbage_file, NULL };
  GstTensorFilterProperties prop;

  _SetFilterProp (&prop, model_files);
  EXPECT_NE (sp->open (&prop, &data), 0);

  g_unlink (garbage_file);
}

/**
 * @brief The documented pipeline: text in a file, generated text out.
 */
TEST_F (nnstreamerFilterExecutorchLlama, pipeline)
{
  g_autofree gchar *tmp_dir = g_dir_make_tmp ("executorch_llama_XXXXXX", NULL);
  ASSERT_TRUE (tmp_dir != NULL);
  g_autofree gchar *in_file = g_build_filename (tmp_dir, "input.txt", NULL);
  g_autofree gchar *out_file = g_build_filename (tmp_dir, "result.txt", NULL);
  g_autofree gchar *contents = NULL;
  gsize len = 0;

  ASSERT_TRUE (g_file_set_contents (in_file, "Once upon a time, u", -1, NULL));

  g_autofree gchar *desc = g_strdup_printf (
      "filesrc location=%s ! application/octet-stream ! tensor_converter ! "
      "other/tensors,format=flexible ! "
      "tensor_filter framework=" FW_NAME " model=%s,%s invoke-dynamic=true ! "
      "other/tensors,format=flexible ! tensor_decoder mode=octet_stream ! "
      "application/octet-stream ! filesink location=%s",
      in_file, model, tokenizer, out_file);
  GstElement *pipeline = gst_parse_launch (desc, NULL);
  ASSERT_TRUE (pipeline != NULL);

  EXPECT_NE (gst_element_set_state (pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
  GstBus *bus = gst_element_get_bus (pipeline);
  GstMessage *msg = gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  gst_object_unref (bus);
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (pipeline);
  ASSERT_TRUE (msg != NULL);
  EXPECT_EQ (GST_MESSAGE_TYPE (msg), GST_MESSAGE_EOS);
  gst_message_unref (msg);

  ASSERT_TRUE (g_file_get_contents (out_file, &contents, &len, NULL));
  EXPECT_EQ (std::string (contents, len), "Once upon a time, uvwxyz");

  g_unlink (in_file);
  g_unlink (out_file);
  g_rmdir (tmp_dir);
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
