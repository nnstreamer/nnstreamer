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
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
 * @brief The text pieces an invoke-async instance has dispatched, in order.
 */
struct AsyncPieces {
  std::mutex lock; /**< Guards the members below */
  std::condition_variable cond; /**< Signals a new piece, a hold or a release */
  std::vector<std::string> pieces; /**< The pieces dispatched so far */
  bool info_ok = true; /**< Every output info described its piece */
  size_t hold_at = 0; /**< Hold the dispatching thread at this many pieces; 0 not to */
  bool held = false; /**< The dispatching thread is being held */
  bool released = false; /**< Let a held dispatching thread go on */

  /**
   * @brief Return the pieces joined, waiting up to 10 seconds for them to be @a text.
   */
  std::string waitFor (const std::string &text)
  {
    std::unique_lock<std::mutex> l (lock);

    cond.wait_for (l, std::chrono::seconds (10),
        [this, &text] { return joined () == text; });
    return joined ();
  }

  /**
   * @brief Return the pieces joined. Call with the lock held.
   */
  std::string joined ()
  {
    std::string text;

    for (const std::string &piece : pieces)
      text += piece;
    return text;
  }
};

/**
 * @brief Async output callback: record the piece and release its data.
 */
static int
_AsyncOutputCb (GstTensorMemory *data, GstTensorsInfo *info, void *user_data)
{
  AsyncPieces *p = (AsyncPieces *) user_data;
  GstTensorInfo *_info = gst_tensors_info_get_nth_info (info, 0);
  std::unique_lock<std::mutex> l (p->lock);

  if (info->num_tensors != 1 || info->format != _NNS_TENSOR_FORMAT_FLEXIBLE
      || _info->type != _NNS_UINT8 || _info->dimension[0] != data[0].size)
    p->info_ok = false;
  p->pieces.emplace_back ((const char *) data[0].data, data[0].size);
  g_free (data[0].data);
  p->cond.notify_all ();

  if (p->pieces.size () == p->hold_at) {
    p->held = true;
    p->cond.notify_all ();
    p->cond.wait (l, [p] { return p->released; });
  }
  return 0;
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

  /**
   * @brief Open an instance with the custom property, run one prompt and close it.
   */
  int generateOnce (const gchar *custom, const std::string &prompt, std::string &result)
  {
    void *data = NULL;
    const gchar *model_files[] = { model, tokenizer, NULL };
    GstTensorFilterProperties prop;
    int ret;

    result.clear ();
    _SetFilterProp (&prop, model_files);
    prop.custom_properties = custom;
    ret = sp->open (&prop, &data);
    EXPECT_EQ (ret, 0) << custom;
    if (ret != 0)
      return ret;

    ret = generate (&prop, &data, prompt, result);
    gst_tensors_info_free (&prop.output_meta);
    sp->close (&prop, &data);
    return ret;
  }

  /**
   * @brief Open an invoke-async instance that hands its pieces to @a pieces.
   */
  int openAsync (GstTensorFilterProperties *prop, void **data,
      const gchar **model_files, const gchar *custom, AsyncPieces *pieces)
  {
    _SetFilterProp (prop, model_files);
    prop->custom_properties = custom;
    prop->invoke_async = TRUE;
    prop->async_callback = _AsyncOutputCb;
    prop->async_user_data = pieces;
    return sp->open (prop, data);
  }

  /**
   * @brief Queue a prompt to an invoke-async instance, which outputs nothing itself.
   */
  int invokeAsync (GstTensorFilterProperties *prop, void *data, const std::string &prompt)
  {
    GstTensorMemory input, output;
    int ret;

    input.data = (gpointer) prompt.data ();
    input.size = prompt.size ();
    output.data = NULL;
    output.size = 0;

    ret = sp->invoke (sp, prop, data, &input, &output);
    EXPECT_TRUE (output.data == NULL);
    EXPECT_EQ (output.size, 0U);
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
 * @brief The custom options are case-insensitive keys, spaces and empty
 *        entries are ignored, and options left out keep their defaults.
 */
TEST_F (nnstreamerFilterExecutorchLlama, customFormat)
{
  std::string result;

  EXPECT_EQ (generateOnce ("", "Hi, w", result), 0);
  EXPECT_EQ (result, "Hi, wxyz");
  EXPECT_EQ (generateOnce (" ECHO : False ,, Temperature:0 ,", "Hi, w", result), 0);
  EXPECT_EQ (result, "xyz");
  EXPECT_EQ (generateOnce ("echo:false,echo:true", "Hi, w", result), 0);
  EXPECT_EQ (result, "Hi, wxyz");
}

/**
 * @brief echo:false leaves the prompt out of the output.
 */
TEST_F (nnstreamerFilterExecutorchLlama, customEcho)
{
  std::string result;

  EXPECT_EQ (generateOnce ("echo:false", "Hi, w", result), 0);
  EXPECT_EQ (result, "xyz");
  EXPECT_EQ (generateOnce ("echo:true", "Hi, w", result), 0);
  EXPECT_EQ (result, "Hi, wxyz");
}

/**
 * @brief keep_eos:true keeps the text of the end-of-sequence token.
 */
TEST_F (nnstreamerFilterExecutorchLlama, customKeepEos)
{
  std::string result;

  EXPECT_EQ (generateOnce ("keep_eos:true", "Hi, w", result), 0);
  EXPECT_EQ (result, std::string ("Hi, wxyz") + LLM_EOS);
  EXPECT_EQ (generateOnce ("keep_eos:false", "Hi, w", result), 0);
  EXPECT_EQ (result, "Hi, wxyz");
}

/**
 * @brief max_new_tokens bounds the generated tokens, seq_len the prompt and
 *        the generated tokens together; -1 leaves either to the model.
 */
TEST_F (nnstreamerFilterExecutorchLlama, customTokenLimits)
{
  std::string result, digits;
  guint i;

  for (i = 0; i < 256; i++)
    digits += (char) ('0' + i % 10);

  EXPECT_EQ (generateOnce ("max_new_tokens:2", "Hi, w", result), 0);
  EXPECT_EQ (result, "Hi, wxy");
  EXPECT_EQ (generateOnce ("max_new_tokens:5", "0", result), 0);
  EXPECT_EQ (result, "012345");
  /* The smaller of the two wins. */
  EXPECT_EQ (generateOnce ("max_new_tokens:200", "0", result), 0);
  EXPECT_EQ (result, digits.substr (0, 128));
  EXPECT_EQ (generateOnce ("seq_len:200", "0", result), 0);
  EXPECT_EQ (result, digits.substr (0, 200));
  /* The model's context is 256 tokens. */
  EXPECT_EQ (generateOnce ("seq_len:-1", "0", result), 0);
  EXPECT_EQ (result, digits);
  EXPECT_EQ (generateOnce ("seq_len:-1,max_new_tokens:-1", "0", result), 0);
  EXPECT_EQ (result, digits);
  EXPECT_EQ (generateOnce ("seq_len:-1,max_new_tokens:3", "0", result), 0);
  EXPECT_EQ (result, "0123");
}

/**
 * @brief ignore_eos:true goes on past the end-of-sequence token.
 */
TEST_F (nnstreamerFilterExecutorchLlama, customIgnoreEos)
{
  std::string result;

  EXPECT_EQ (generateOnce ("ignore_eos:true,max_new_tokens:6", "x", result), 0);
  EXPECT_EQ (result, "xyzabc");
  EXPECT_EQ (generateOnce ("ignore_eos:true,max_new_tokens:6,keep_eos:true", "x", result), 0);
  EXPECT_EQ (result, std::string ("xyz") + LLM_EOS + "abc");
  EXPECT_EQ (generateOnce ("ignore_eos:false,max_new_tokens:6", "x", result), 0);
  EXPECT_EQ (result, "xyz");
}

/**
 * @brief num_bos and num_eos add tokens around the prompt, which count in
 *        seq_len and change what the model sees last.
 */
TEST_F (nnstreamerFilterExecutorchLlama, customBosEos)
{
  std::string result, expected;
  guint i;

  for (i = 0; i < 127; i++)
    expected += (char) ('0' + (3 + i) % 10);

  EXPECT_EQ (generateOnce ("num_bos:1", "345", result), 0);
  EXPECT_EQ (result, expected);
  EXPECT_EQ (generateOnce ("num_eos:1", "Hi", result), 0);
  EXPECT_EQ (result, "Hiabcdefghijklmnopqrstuvwxyz");
  EXPECT_EQ (generateOnce ("num_bos:0,num_eos:0", "Hi, w", result), 0);
  EXPECT_EQ (result, "Hi, wxyz");
}

/**
 * @brief temperature is taken as given, from 0 to 1; the fixture's margin
 *        makes the result the same at any of these.
 */
TEST_F (nnstreamerFilterExecutorchLlama, customTemperature)
{
  std::string result;

  EXPECT_EQ (generateOnce ("temperature:0", "Hi, w", result), 0);
  EXPECT_EQ (result, "Hi, wxyz");
  EXPECT_EQ (generateOnce ("temperature:0.5", "Hi, w", result), 0);
  EXPECT_EQ (result, "Hi, wxyz");
  EXPECT_EQ (generateOnce ("temperature:1", "Hi, w", result), 0);
  EXPECT_EQ (result, "Hi, wxyz");
}

/**
 * @brief Negative case: an invalid custom option fails to open.
 */
TEST_F (nnstreamerFilterExecutorchLlama, openInvalidCustom_n)
{
  const gchar *model_files[] = { model, tokenizer, NULL };
  const gchar *invalid[] = { "unknown:1", ":1", "echo", "echo:", "echo:yes",
    "echo:1", "keep_eos:on", "ignore_eos:no", "max_new_tokens:0",
    "max_new_tokens:-2", "max_new_tokens:12a", "max_new_tokens:2147483648",
    "seq_len:0", "seq_len:-2", "seq_len:1.5", "temperature:-0.1",
    "temperature:nan", "temperature:inf", "temperature:", "temperature:1e40",
    "temperature:0.5x", "temperature:1.5", "temperature:2", "num_bos:-1",
    "num_bos:128", "num_eos:x", "num_eos:128", "echo:false,unknown:1", NULL };
  GstTensorFilterProperties prop;
  guint i;

  for (i = 0; invalid[i]; i++) {
    void *data = NULL;

    _SetFilterProp (&prop, model_files);
    prop.custom_properties = invalid[i];
    EXPECT_NE (sp->open (&prop, &data), 0) << invalid[i];
    EXPECT_TRUE (data == NULL) << invalid[i];
  }
}

/**
 * @brief Negative case: a prompt for which nothing is left to output is an
 *        error, since a flexible tensor cannot be empty.
 */
TEST_F (nnstreamerFilterExecutorchLlama, generateNoText_n)
{
  std::string result;

  /* 'z' is followed by the end-of-sequence token alone. */
  EXPECT_NE (generateOnce ("echo:false,max_new_tokens:1", "z", result), 0);

  EXPECT_EQ (generateOnce ("echo:false,max_new_tokens:1,keep_eos:true", "z", result), 0);
  EXPECT_EQ (result, LLM_EOS);
  EXPECT_EQ (generateOnce ("echo:false,max_new_tokens:1", "y", result), 0);
  EXPECT_EQ (result, "z");
}

/**
 * @brief With invoke-async the prompt and each generated token go out as
 *        outputs of their own, the end-of-sequence text left out.
 */
TEST_F (nnstreamerFilterExecutorchLlama, asyncStream)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  AsyncPieces pieces;

  ASSERT_EQ (openAsync (&prop, &data, model_files, NULL, &pieces), 0);

  EXPECT_EQ (invokeAsync (&prop, data, "Hi, w"), 0);
  EXPECT_EQ (pieces.waitFor ("Hi, wxyz"), "Hi, wxyz");

  sp->close (&prop, &data);
  EXPECT_EQ (pieces.pieces, std::vector<std::string> ({ "Hi, w", "x", "y", "z" }));
  EXPECT_TRUE (pieces.info_ok);
  gst_tensors_info_free (&prop.output_meta);
}

/**
 * @brief With invoke-async the custom options apply as they do without it.
 */
TEST_F (nnstreamerFilterExecutorchLlama, asyncCustom)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  AsyncPieces pieces;
  const std::string expected = std::string ("xyz") + LLM_EOS + "a";

  ASSERT_EQ (openAsync (&prop, &data, model_files,
                 "echo:false,keep_eos:true,ignore_eos:true,max_new_tokens:5", &pieces),
      0);

  EXPECT_EQ (invokeAsync (&prop, data, "w"), 0);
  EXPECT_EQ (pieces.waitFor (expected), expected);

  sp->close (&prop, &data);
  EXPECT_EQ (pieces.pieces, std::vector<std::string> ({ "x", "y", "z", LLM_EOS, "a" }));
  gst_tensors_info_free (&prop.output_meta);
}

/**
 * @brief Prompts queued with invoke-async are generated in order, each from
 *        the beginning.
 */
TEST_F (nnstreamerFilterExecutorchLlama, asyncQueue)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  AsyncPieces pieces;

  ASSERT_EQ (openAsync (&prop, &data, model_files, NULL, &pieces), 0);

  EXPECT_EQ (invokeAsync (&prop, data, "x"), 0);
  EXPECT_EQ (invokeAsync (&prop, data, "Hi, w"), 0);
  EXPECT_EQ (invokeAsync (&prop, data, "x"), 0);
  EXPECT_EQ (pieces.waitFor ("xyzHi, wxyzxyz"), "xyzHi, wxyzxyz");

  sp->close (&prop, &data);
  EXPECT_EQ (pieces.pieces.size (), 10U);
  gst_tensors_info_free (&prop.output_meta);
}

/**
 * @brief With invoke-async the echoed prompt goes out even when no token
 *        text follows it.
 */
TEST_F (nnstreamerFilterExecutorchLlama, asyncPromptOnly)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  AsyncPieces pieces;

  ASSERT_EQ (openAsync (&prop, &data, model_files, "max_new_tokens:1", &pieces), 0);

  /* 'z' is followed by the end-of-sequence token alone. */
  EXPECT_EQ (invokeAsync (&prop, data, "z"), 0);
  EXPECT_EQ (invokeAsync (&prop, data, "y"), 0);
  EXPECT_EQ (pieces.waitFor ("zyz"), "zyz");

  sp->close (&prop, &data);
  EXPECT_EQ (pieces.pieces, std::vector<std::string> ({ "z", "y", "z" }));
  gst_tensors_info_free (&prop.output_meta);
}

/**
 * @brief Negative case: with invoke-async an empty prompt fails the invoke,
 *        and a prompt that the runner refuses outputs nothing, not even its
 *        echo, while the next one is still generated.
 */
TEST_F (nnstreamerFilterExecutorchLlama, asyncInvalidPrompt_n)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  AsyncPieces pieces;

  ASSERT_EQ (openAsync (&prop, &data, model_files, NULL, &pieces), 0);

  EXPECT_NE (invokeAsync (&prop, data, ""), 0);
  /* Too long for the model's context, found only by the generation itself. */
  EXPECT_EQ (invokeAsync (&prop, data, std::string (300, '0')), 0);
  /* Refused after the runner has echoed it: no room left within seq_len. */
  EXPECT_EQ (invokeAsync (&prop, data, std::string (128, '0')), 0);
  EXPECT_EQ (invokeAsync (&prop, data, "x"), 0);
  EXPECT_EQ (pieces.waitFor ("xyz"), "xyz");

  sp->close (&prop, &data);
  EXPECT_EQ (pieces.pieces, std::vector<std::string> ({ "x", "y", "z" }));
  gst_tensors_info_free (&prop.output_meta);
}

/**
 * @brief Closing an invoke-async instance stops the generation in progress
 *        and drops the prompts still queued.
 *
 * The callback holds the generation at its second piece, so the close starts
 * while 254 more tokens are still to come for the first prompt.
 */
TEST_F (nnstreamerFilterExecutorchLlama, asyncCloseStops)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  AsyncPieces pieces;

  pieces.hold_at = 2;
  ASSERT_EQ (openAsync (&prop, &data, model_files, "seq_len:-1", &pieces), 0);

  EXPECT_EQ (invokeAsync (&prop, data, "0"), 0);
  EXPECT_EQ (invokeAsync (&prop, data, "x"), 0);
  {
    std::unique_lock<std::mutex> l (pieces.lock);
    ASSERT_TRUE (pieces.cond.wait_for (
        l, std::chrono::seconds (10), [&pieces] { return pieces.held; }));
  }

  bool closing = false;
  std::thread closer ([this, &prop, &data, &pieces, &closing] {
    {
      std::lock_guard<std::mutex> l (pieces.lock);
      closing = true;
    }
    pieces.cond.notify_all ();
    sp->close (&prop, &data);
  });
  {
    std::unique_lock<std::mutex> l (pieces.lock);
    pieces.cond.wait (l, [&closing] { return closing; });
  }
  /* The close only has to begin before 10 pieces, of the 256, have gone out. */
  std::this_thread::sleep_for (std::chrono::milliseconds (200));
  {
    std::lock_guard<std::mutex> l (pieces.lock);
    pieces.released = true;
  }
  pieces.cond.notify_all ();
  closer.join ();

  EXPECT_TRUE (data == NULL);
  EXPECT_LT (pieces.pieces.size (), 10U);
  EXPECT_EQ (pieces.joined ().find ('x'), std::string::npos);
  gst_tensors_info_free (&prop.output_meta);
}

/**
 * @brief Closing an invoke-async instance stops the generation at a token
 *        whose text is left out, too.
 *
 * The callback holds the generation at "z", which is followed by the
 * end-of-sequence token and then 'a' with ignore_eos. The runner prints each
 * token it generates on the standard output, so 'a' after the end-of-sequence
 * text there means that the left-out token did not stop the generation.
 */
TEST_F (nnstreamerFilterExecutorchLlama, asyncCloseStopsAtLeftOutEos)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  AsyncPieces pieces;
  std::string printed;

  pieces.hold_at = 2;
  ASSERT_EQ (openAsync (&prop, &data, model_files, "seq_len:-1,ignore_eos:true", &pieces), 0);

  testing::internal::CaptureStdout ();
  EXPECT_EQ (invokeAsync (&prop, data, "y"), 0);
  {
    std::unique_lock<std::mutex> l (pieces.lock);
    EXPECT_TRUE (pieces.cond.wait_for (
        l, std::chrono::seconds (10), [&pieces] { return pieces.held; }));
  }

  bool closing = false;
  std::thread closer ([this, &prop, &data, &pieces, &closing] {
    {
      std::lock_guard<std::mutex> l (pieces.lock);
      closing = true;
    }
    pieces.cond.notify_all ();
    sp->close (&prop, &data);
  });
  {
    std::unique_lock<std::mutex> l (pieces.lock);
    pieces.cond.wait (l, [&closing] { return closing; });
  }
  /* The close has to begin before the end-of-sequence token comes. */
  std::this_thread::sleep_for (std::chrono::milliseconds (200));
  {
    std::lock_guard<std::mutex> l (pieces.lock);
    pieces.released = true;
  }
  pieces.cond.notify_all ();
  closer.join ();
  printed = testing::internal::GetCapturedStdout ();

  EXPECT_TRUE (data == NULL);
  EXPECT_EQ (pieces.joined (), "yz");
  EXPECT_NE (printed.find ("yz" LLM_EOS), std::string::npos);
  EXPECT_EQ (printed.find (LLM_EOS "a"), std::string::npos);
  gst_tensors_info_free (&prop.output_meta);
}

/**
 * @brief Closing an invoke-async instance with nothing queued, or one that
 *        has never been invoked, returns at once.
 */
TEST_F (nnstreamerFilterExecutorchLlama, asyncCloseIdle)
{
  void *data = NULL;
  const gchar *model_files[] = { model, tokenizer, NULL };
  GstTensorFilterProperties prop;
  AsyncPieces pieces;

  ASSERT_EQ (openAsync (&prop, &data, model_files, NULL, &pieces), 0);
  sp->close (&prop, &data);
  EXPECT_TRUE (data == NULL);

  ASSERT_EQ (openAsync (&prop, &data, model_files, NULL, &pieces), 0);
  EXPECT_EQ (invokeAsync (&prop, data, "y"), 0);
  EXPECT_EQ (pieces.waitFor ("yz"), "yz");
  sp->close (&prop, &data);
  EXPECT_TRUE (data == NULL);
  EXPECT_EQ (pieces.pieces.size (), 2U);
  gst_tensors_info_free (&prop.output_meta);
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
 * @brief Pull the non-empty buffers of the appsink until they make up @a text.
 *
 * tensor_filter also pushes an empty buffer for each invoke-async invoke.
 */
static std::vector<std::string>
_PullPieces (GstElement *sink, const std::string &text)
{
  std::vector<std::string> pieces;
  std::string joined;

  while (joined != text) {
    GstSample *sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), 10 * GST_SECOND);
    GstMapInfo map;

    if (!sample)
      break;
    if (gst_buffer_map (gst_sample_get_buffer (sample), &map, GST_MAP_READ)) {
      if (map.size > 0) {
        pieces.emplace_back ((const char *) map.data, map.size);
        joined += pieces.back ();
      }
      gst_buffer_unmap (gst_sample_get_buffer (sample), &map);
    }
    gst_sample_unref (sample);
  }

  return pieces;
}

/**
 * @brief The pipeline with invoke-async: the prompt and each generated token
 *        arrive as buffers of their own, in order, for each input buffer.
 */
TEST_F (nnstreamerFilterExecutorchLlama, pipelineAsync)
{
  const std::string prompt = "Once upon a time, u";

  g_autofree gchar *desc = g_strdup_printf (
      "appsrc name=src caps=application/octet-stream ! tensor_converter ! "
      "other/tensors,format=flexible ! "
      "tensor_filter framework=" FW_NAME " model=%s,%s invoke-dynamic=true invoke-async=true ! "
      "other/tensors,format=flexible ! tensor_decoder mode=octet_stream ! "
      "application/octet-stream ! appsink name=sink sync=false",
      model, tokenizer);
  GstElement *pipeline = gst_parse_launch (desc, NULL);
  ASSERT_TRUE (pipeline != NULL);
  GstElement *src = gst_bin_get_by_name (GST_BIN (pipeline), "src");
  GstElement *sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");

  EXPECT_NE (gst_element_set_state (pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (src),
                 gst_buffer_new_wrapped (g_strdup (prompt.c_str ()), prompt.size ())),
      GST_FLOW_OK);
  EXPECT_EQ (_PullPieces (sink, prompt + "vwxyz"),
      std::vector<std::string> ({ prompt, "v", "w", "x", "y", "z" }));

  EXPECT_EQ (gst_app_src_push_buffer (
                 GST_APP_SRC (src), gst_buffer_new_wrapped (g_strdup ("x"), 1)),
      GST_FLOW_OK);
  EXPECT_EQ (_PullPieces (sink, "xyz"), std::vector<std::string> ({ "x", "y", "z" }));

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (src);
  gst_object_unref (sink);
  gst_object_unref (pipeline);
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
