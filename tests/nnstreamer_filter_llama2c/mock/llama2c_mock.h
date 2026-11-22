/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    llama2c_mock.h
 * @date    22 Nov 2026
 * @brief   Controls of the mock llama2c library for the sub-plugin tests
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * The mock reads the Config header of a checkpoint as llama2c does and
 * ignores the rest of both files. get_tokens () returns
 * "<prompt> -> <steps> of <seq_len>" unless told otherwise, so a test sees
 * the prompt and the step count the sub-plugin passed and the seq_len of the
 * checkpoint the transformer it passed was built from. Like llama2c,
 * build_transformer () exits the process when the checkpoint cannot be
 * read, and so does build_tokenizer () for a NULL path.
 */
#ifndef __LLAMA2C_MOCK_H__
#define __LLAMA2C_MOCK_H__

#ifdef __cplusplus
extern "C" {
#endif

/** @brief What get_tokens () returns. */
typedef enum {
  LLAMA2C_MOCK_RESULT_TEXT = 0, /**< "<prompt> -> <steps> of <seq_len>" */
  LLAMA2C_MOCK_RESULT_NULL, /**< NULL */
  LLAMA2C_MOCK_RESULT_EMPTY, /**< an empty string */
} llama2c_mock_result;

/**
 * @brief Restore the default result and clear the counters (live objects
 *        included) and the last call.
 */
void llama2c_mock_reset (void);

/**
 * @brief Set what later get_tokens () calls return.
 */
void llama2c_mock_set_result (llama2c_mock_result result);

/**
 * @brief The number of transformers, tokenizers and samplers built and not freed.
 */
int llama2c_mock_live_objects (void);

/**
 * @brief The number of build_transformer () calls on a transformer that was
 *        built and not freed (llama2c leaks the old one then).
 */
int llama2c_mock_rebuilds (void);

/**
 * @brief The number of calls of the named function since the last reset.
 */
int llama2c_mock_calls (const char *function);

/**
 * @brief The prompt of the last get_tokens () call.
 * @return A newly allocated string, or NULL; free it with free().
 */
char *llama2c_mock_last_prompt (void);

/**
 * @brief The steps of the last get_tokens () call.
 */
int llama2c_mock_last_steps (void);

/**
 * @brief The vocabulary size the last tokenizer and sampler were built with.
 */
int llama2c_mock_last_vocab_size (void);

#ifdef __cplusplus
}
#endif

#endif /* __LLAMA2C_MOCK_H__ */
