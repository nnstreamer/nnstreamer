/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_filter_llama2c.cc
 * @date    22 Nov 2026
 * @brief   Unit tests of the llama2c tensor_filter sub-plugin on a mock llama2c
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * The sub-plugin source is linked into this binary against the mock llama2c
 * of mock/, which reads only the Config header of a checkpoint. See
 * mock/llama2c_mock.h.
 */
#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gst/gst.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_plugin_api_util.h>

#include <llama2c_mock.h>

/**
 * @brief Fixture that writes checkpoints and a tokenizer into a private directory.
 */
class nnstreamerFilterLlama2c : public ::testing::Test
{
  protected:
  const GstTensorFilterFramework *sp;
  gchar *dir;
  gchar *tokenizer;
  GSList *files;

  /** @brief Find the sub-plugin, create the directory and reset the mock. */
  void SetUp () override
  {
    dir = nullptr;
    files = nullptr;
    tokenizer = nullptr;
    llama2c_mock_reset ();
    sp = nnstreamer_filter_find ("llama2c");
    ASSERT_TRUE (sp != nullptr);
    dir = g_dir_make_tmp ("nns-llama2c-XXXXXX", NULL);
    ASSERT_TRUE (dir != nullptr);
    tokenizer = writeFile ("tokenizer.bin", "tokenizer", 9);
  }

  /** @brief Remove the files and the directory. */
  void TearDown () override
  {
    for (GSList *l = files; l; l = l->next)
      g_remove ((const gchar *) l->data);
    g_slist_free_full (files, g_free);
    if (dir)
      g_rmdir (dir);
    g_free (dir);
    llama2c_mock_reset ();
  }

  /** @brief Write a file into the directory and return its path. */
  gchar *writeFile (const gchar *name, const gchar *content, gssize len)
  {
    gchar *path = g_build_filename (dir, name, NULL);

    EXPECT_TRUE (g_file_set_contents (path, content, len, NULL));
    files = g_slist_prepend (files, path);
    return path;
  }

  /** @brief Write a checkpoint whose Config header has @a vocab_size and @a seq_len. */
  gchar *writeCheckpoint (const gchar *name, gint32 vocab_size, gint32 seq_len)
  {
    /* dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len */
    const gint32 config[7] = { 8, 16, 1, 1, 1, vocab_size, seq_len };

    return writeFile (name, (const gchar *) config, sizeof (config));
  }

  /** @brief Fill the properties of a filter of the given model files. */
  void setProp (GstTensorFilterProperties *prop, const gchar **model_files,
      const gchar *model, const gchar *tok)
  {
    memset (prop, 0, sizeof (*prop));
    prop->fwname = "llama2c";
    model_files[0] = model;
    model_files[1] = tok;
    model_files[2] = nullptr;
    prop->model_files = model_files;
    prop->num_models = tok ? 2 : 1;
    prop->invoke_dynamic = TRUE;
    gst_tensors_info_init (&prop->input_meta);
    gst_tensors_info_init (&prop->output_meta);
  }

  /**
   * @brief Invoke a filter with @a text and return the text it generated, or
   *        NULL if the invoke fails.
   */
  gchar *generate (GstTensorFilterProperties *prop, void *data, const gchar *text)
  {
    GstTensorMemory in = { (gpointer) text, strlen (text) };
    GstTensorMemory out = { nullptr, 0 };
    gchar *result = nullptr;

    if (sp->invoke (sp, prop, data, &in, &out) != 0)
      return nullptr;

    if (out.data) {
      /* the output is the text and its terminating NUL */
      EXPECT_EQ (out.size, strlen ((gchar *) out.data) + 1);
      EXPECT_EQ (prop->output_meta.num_tensors, 1U);
      EXPECT_EQ (prop->output_meta.format, _NNS_TENSOR_FORMAT_FLEXIBLE);
      EXPECT_EQ (prop->output_meta.info[0].type, _NNS_UINT8);
      EXPECT_EQ (prop->output_meta.info[0].dimension[0], out.size);
      result = g_strdup ((gchar *) out.data);
      g_free (out.data);
    }
    return result;
  }
};

/**
 * @brief Positive case: the framework information.
 */
TEST_F (nnstreamerFilterLlama2c, frameworkInfo)
{
  GstTensorFilterFrameworkInfo info;

  memset (&info, 0, sizeof (info));
  EXPECT_EQ (sp->getFrameworkInfo (sp, NULL, NULL, &info), 0);
  EXPECT_STREQ (info.name, "llama2c");
  EXPECT_FALSE (info.run_without_model);
  EXPECT_TRUE (info.verify_model_path);
  EXPECT_TRUE (info.allocate_in_invoke);
  EXPECT_FALSE (info.allow_in_place);
}

/**
 * @brief Positive case: open a model and describe it; no other model-info
 *        operation and no event is handled.
 */
TEST_F (nnstreamerFilterLlama2c, openModelInfo)
{
  GstTensorFilterProperties prop;
  const gchar *model_files[3];
  void *data = nullptr;
  GstTensorsInfo in, out;
  GstTensorFilterFrameworkEventData event;
  gchar *model = writeCheckpoint ("model.bin", 512, 32);

  setProp (&prop, model_files, model, tokenizer);
  ASSERT_EQ (sp->open (&prop, &data), 0);
  EXPECT_EQ (llama2c_mock_calls ("build_transformer"), 1);
  EXPECT_EQ (llama2c_mock_calls ("build_tokenizer"), 1);
  EXPECT_EQ (llama2c_mock_calls ("build_sampler"), 1);
  /* the tokenizer and the sampler take the vocabulary size of the model */
  EXPECT_EQ (llama2c_mock_last_vocab_size (), 512);

  gst_tensors_info_init (&in);
  gst_tensors_info_init (&out);
  EXPECT_EQ (sp->getModelInfo (sp, &prop, data, GET_IN_OUT_INFO, &in, &out), 0);
  EXPECT_EQ (in.num_tensors, 1U);
  EXPECT_EQ (in.format, _NNS_TENSOR_FORMAT_FLEXIBLE);
  EXPECT_EQ (out.num_tensors, 1U);
  EXPECT_EQ (out.format, _NNS_TENSOR_FORMAT_FLEXIBLE);
  EXPECT_EQ (sp->getModelInfo (sp, &prop, data, SET_INPUT_INFO, &in, &out), -ENOENT);
  gst_tensors_info_free (&in);
  gst_tensors_info_free (&out);

  memset (&event, 0, sizeof (event));
  EXPECT_EQ (sp->eventHandler (sp, &prop, data, RELOAD_MODEL, &event), -ENOENT);

  sp->close (&prop, &data);
}

/**
 * @brief Positive case: generate from a prompt, with the steps of the model.
 */
TEST_F (nnstreamerFilterLlama2c, generate)
{
  GstTensorFilterProperties prop;
  const gchar *model_files[3];
  void *data = nullptr;
  gchar *model = writeCheckpoint ("model.bin", 512, 32);
  gchar *text, *prompt;

  setProp (&prop, model_files, model, tokenizer);
  ASSERT_EQ (sp->open (&prop, &data), 0);

  text = generate (&prop, data, "Once upon a time");
  EXPECT_STREQ (text, "Once upon a time -> 32 of 32");
  g_free (text);

  /* a trailing newline of a text file is not part of the prompt */
  text = generate (&prop, data, "Hello\n");
  EXPECT_STREQ (text, "Hello -> 32 of 32");
  g_free (text);
  prompt = llama2c_mock_last_prompt ();
  EXPECT_STREQ (prompt, "Hello");
  free (prompt);

  text = generate (&prop, data, "Hi\r");
  EXPECT_STREQ (text, "Hi -> 32 of 32");
  g_free (text);

  /* an empty input is an empty prompt */
  text = generate (&prop, data, "");
  EXPECT_STREQ (text, " -> 32 of 32");
  g_free (text);
  prompt = llama2c_mock_last_prompt ();
  EXPECT_STREQ (prompt, "");
  free (prompt);

  EXPECT_EQ (llama2c_mock_calls ("get_tokens"), 4);
  sp->close (&prop, &data);
}

/**
 * @brief Positive case: a negative vocab_size (unshared weights) is used as its absolute value.
 */
TEST_F (nnstreamerFilterLlama2c, unsharedWeights)
{
  GstTensorFilterProperties prop;
  const gchar *model_files[3];
  void *data = nullptr;
  gchar *model = writeCheckpoint ("model.bin", -300, 16);
  gchar *text;

  setProp (&prop, model_files, model, tokenizer);
  ASSERT_EQ (sp->open (&prop, &data), 0);
  EXPECT_EQ (llama2c_mock_last_vocab_size (), 300);

  text = generate (&prop, data, "a");
  EXPECT_STREQ (text, "a -> 16 of 16");
  g_free (text);

  sp->close (&prop, &data);
}

/**
 * @brief Negative case: the sub-plugin runs only with invoke-dynamic.
 */
TEST_F (nnstreamerFilterLlama2c, invokeStatic_n)
{
  GstTensorFilterProperties prop;
  const gchar *model_files[3];
  void *data = nullptr;
  gchar *model = writeCheckpoint ("model.bin", 512, 32);
  gchar text[] = "prompt";
  GstTensorMemory in = { text, sizeof (text) };
  GstTensorMemory out = { nullptr, 0 };

  setProp (&prop, model_files, model, tokenizer);
  ASSERT_EQ (sp->open (&prop, &data), 0);

  prop.invoke_dynamic = FALSE;
  EXPECT_NE (sp->invoke (sp, &prop, data, &in, &out), 0);
  EXPECT_TRUE (out.data == nullptr);
  EXPECT_EQ (llama2c_mock_calls ("get_tokens"), 0);

  sp->close (&prop, &data);
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
