/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    llama2c_mock.cc
 * @date    22 Nov 2026
 * @brief   A mock llama2c library for the llama2c sub-plugin tests
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * See llama2c_mock.h for what it does.
 */
#include <map>
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "api.h"
}
#include "llama2c_mock.h"

namespace
{
std::map<std::string, int> calls;
llama2c_mock_result result_mode = LLAMA2C_MOCK_RESULT_TEXT;
int live = 0;
int rebuilds = 0;
std::string last_prompt;
bool has_last_prompt = false;
int last_steps = 0;
int last_vocab_size = 0;
} /* namespace */

extern "C" {

/**
 * @brief Read the Config header of a checkpoint, as llama2c does. Unlike
 *        llama2c, it neither maps nor checks the weights after the header.
 */
void
build_transformer (Transformer *t, char *checkpoint_path)
{
  calls["build_transformer"]++;

  FILE *file = fopen (checkpoint_path, "rb");
  if (!file) {
    fprintf (stderr, "Couldn't open file %s\n", checkpoint_path);
    exit (EXIT_FAILURE);
  }
  if (fread (&t->config, sizeof (Config), 1, file) != 1)
    exit (EXIT_FAILURE);
  fclose (file);
  t->config.vocab_size = abs (t->config.vocab_size);

  if (t->data) {
    rebuilds++;
  } else {
    live++;
  }
  t->data = (float *) calloc (1, sizeof (float));
  t->fd = 3;
}

/** @brief Free a transformer. */
void
free_transformer (Transformer *t)
{
  calls["free_transformer"]++;
  if (t->data) {
    free (t->data);
    t->data = nullptr;
    live--;
  }
  t->fd = -1;
}

/** @brief Build a tokenizer; the file is not read. */
void
build_tokenizer (Tokenizer *t, char *tokenizer_path, int vocab_size)
{
  calls["build_tokenizer"]++;
  if (!tokenizer_path) {
    fprintf (stderr, "Couldn't load (null)\n");
    exit (EXIT_FAILURE);
  }

  if (!t->vocab)
    live++;
  free (t->vocab);
  t->vocab = (char **) calloc (1, sizeof (char *));
  t->vocab_size = vocab_size;
  t->max_token_length = 16;
  last_vocab_size = vocab_size;
}

/** @brief Free a tokenizer. */
void
free_tokenizer (Tokenizer *t)
{
  calls["free_tokenizer"]++;
  if (t->vocab) {
    free (t->vocab);
    t->vocab = nullptr;
    live--;
  }
}

/** @brief Build a sampler. */
void
build_sampler (Sampler *sampler, int vocab_size, float temperature, float topp,
    unsigned long long rng_seed)
{
  calls["build_sampler"]++;
  if (!sampler->probindex)
    live++;
  free (sampler->probindex);
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  sampler->probindex = (ProbIndex *) calloc (1, sizeof (ProbIndex));
  last_vocab_size = vocab_size;
}

/** @brief Free a sampler. */
void
free_sampler (Sampler *sampler)
{
  calls["free_sampler"]++;
  if (sampler->probindex) {
    free (sampler->probindex);
    sampler->probindex = nullptr;
    live--;
  }
}

/** @brief Return "<prompt> -> <steps> of <seq_len>", NULL or "" (see llama2c_mock.h). */
char *
get_tokens (Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
    char *prompt, int steps)
{
  calls["get_tokens"]++;
  has_last_prompt = (prompt != nullptr);
  last_prompt = prompt ? prompt : "";
  last_steps = steps;

  if (!transformer || !transformer->data || !tokenizer || !tokenizer->vocab
      || !sampler || !sampler->probindex)
    return nullptr;

  switch (result_mode) {
    case LLAMA2C_MOCK_RESULT_NULL:
      return nullptr;
    case LLAMA2C_MOCK_RESULT_EMPTY:
      return strdup ("");
    default:
      break;
  }

  std::string text = last_prompt + " -> " + std::to_string (steps) + " of "
                     + std::to_string (transformer->config.seq_len);
  return strdup (text.c_str ());
}

/** @brief Restore the defaults. */
void
llama2c_mock_reset (void)
{
  calls.clear ();
  result_mode = LLAMA2C_MOCK_RESULT_TEXT;
  live = 0;
  rebuilds = 0;
  last_prompt.clear ();
  has_last_prompt = false;
  last_steps = 0;
  last_vocab_size = 0;
}

/** @brief Set what get_tokens () returns. */
void
llama2c_mock_set_result (llama2c_mock_result result)
{
  result_mode = result;
}

/** @brief The number of objects built and not freed. */
int
llama2c_mock_live_objects (void)
{
  return live;
}

/** @brief The number of rebuilds of a built transformer. */
int
llama2c_mock_rebuilds (void)
{
  return rebuilds;
}

/** @brief The number of calls of the named function. */
int
llama2c_mock_calls (const char *function)
{
  auto it = calls.find (function);
  return (it == calls.end ()) ? 0 : it->second;
}

/** @brief The prompt of the last get_tokens () call. */
char *
llama2c_mock_last_prompt (void)
{
  return has_last_prompt ? strdup (last_prompt.c_str ()) : nullptr;
}

/** @brief The steps of the last get_tokens () call. */
int
llama2c_mock_last_steps (void)
{
  return last_steps;
}

/** @brief The vocabulary size of the last tokenizer and sampler. */
int
llama2c_mock_last_vocab_size (void)
{
  return last_vocab_size;
}

} /* extern "C" */
