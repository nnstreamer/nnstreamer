/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    api.h
 * @date    22 Nov 2026
 * @brief   The llama2c API the llama2c sub-plugin uses, for the mock tests
 * @see     http://github.com/nnstreamer/nnstreamer
 *          https://github.com/nnsuite/llama2.c
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * This header stands in for include/api.h of nnsuite/llama2.c (MIT) when the
 * sub-plugin is built for the tests in this directory. The structures keep
 * the members and the order of that library at commit 0804e8a1, so that the
 * sub-plugin compiles here as it does against the library. Only the
 * functions the sub-plugin calls are declared; mock/llama2c_mock.cc
 * implements them.
 */
#ifndef __LLAMA2C_MOCK_API_H__
#define __LLAMA2C_MOCK_API_H__

#include <stdint.h>
#include <sys/types.h>

/** @brief The hyperparameters of a model, as stored at the head of a checkpoint. */
typedef struct {
  int dim;
  int hidden_dim;
  int n_layers;
  int n_heads;
  int n_kv_heads;
  int vocab_size;
  int seq_len;
} Config;

/** @brief The weights of a model. */
typedef struct {
  float *token_embedding_table;
  float *rms_att_weight;
  float *rms_ffn_weight;
  float *wq;
  float *wk;
  float *wv;
  float *wo;
  float *w1;
  float *w2;
  float *w3;
  float *rms_final_weight;
  float *wcls;
} TransformerWeights;

/** @brief The activation buffers of a model. */
typedef struct {
  float *x;
  float *xb;
  float *xb2;
  float *hb;
  float *hb2;
  float *q;
  float *k;
  float *v;
  float *att;
  float *logits;
  float *key_cache;
  float *value_cache;
} RunState;

/** @brief A model. */
typedef struct {
  Config config;
  TransformerWeights weights;
  RunState state;
  int fd;
  float *data;
  ssize_t file_size;
} Transformer;

/** @brief A vocabulary entry. */
typedef struct {
  char *str;
  int id;
} TokenIndex;

/** @brief A BPE tokenizer. */
typedef struct {
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512];
} Tokenizer;

/** @brief A probability and its token, for top-p sampling. */
typedef struct {
  float prob;
  int index;
} ProbIndex;

/** @brief A sampler of the next token. */
typedef struct {
  int vocab_size;
  ProbIndex *probindex;
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

/** @brief Load a checkpoint into @a t; exits the process if it cannot be read. */
void build_transformer (Transformer *t, char *checkpoint_path);

/** @brief Free what build_transformer () loaded. */
void free_transformer (Transformer *t);

/** @brief Load a tokenizer of @a vocab_size entries into @a t. */
void build_tokenizer (Tokenizer *t, char *tokenizer_path, int vocab_size);

/** @brief Free what build_tokenizer () loaded. */
void free_tokenizer (Tokenizer *t);

/** @brief Set up a sampler. */
void build_sampler (Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);

/** @brief Free what build_sampler () allocated. */
void free_sampler (Sampler *sampler);

/** @brief Generate at most @a steps tokens from @a prompt; free the result with free (). */
char *get_tokens (Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps);

#endif /* __LLAMA2C_MOCK_API_H__ */
