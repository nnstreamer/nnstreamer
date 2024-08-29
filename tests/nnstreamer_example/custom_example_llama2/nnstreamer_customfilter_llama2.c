/**
 * NNStreamer Custom Filter Example llama2.c
 * Copyright (C) 2024 Yelin Jeong <yelini.jeong@samsung.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file	nnstreamer_customfilter_llama2.c
 * @date	29 Aug 2024
 * @brief	Custom NNStreamer Filter Example llama2.c
 * @author	Yelin Jeong <yelini.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 * @see     https://github.com/karpathy/llama2.c
 * @note    The output is bundled into one and not by token unit.
 */

#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#include <nnstreamer_plugin_api.h>
#include <nnstreamer_util.h>
#include <tensor_filter_custom.h>

#define CHECKPOINT_PATH "model.bin"
#define TOKENIZER_PATH "tokenizer.bin"

#define TEMPERATURE \
  1.0f // 0.0 = greedy deterministic. 1.0 = original. don't set higher
#define TOPP \
  0.9f // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
#define RNG_SEED 0ULL // seed rng with time by default

// TODO: Find the appropriate length
#define PROMPT_LENGTH 2048

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
  int dim; // transformer dimension
  int hidden_dim; // for ffn layers
  int n_layers; // number of layers
  int n_heads; // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
  int vocab_size; // vocabulary size, usually 256 (byte-level)
  int seq_len; // max sequence length
} Config;

typedef struct {
  // token embedding table
  float *token_embedding_table; // (vocab_size, dim)
  // weights for rmsnorms
  float *rms_att_weight; // (layer, dim) rmsnorm weights
  float *rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  float *wq; // (layer, dim, n_heads * head_size)
  float *wk; // (layer, dim, n_kv_heads * head_size)
  float *wv; // (layer, dim, n_kv_heads * head_size)
  float *wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  float *w1; // (layer, hidden_dim, dim)
  float *w2; // (layer, dim, hidden_dim)
  float *w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  float *rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  float *wcls;
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x; // activation at current time stamp (dim,)
  float *xb; // same, but inside a residual branch (dim,)
  float *xb2; // an additional buffer just for convenience (dim,)
  float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
  float *q; // query (dim,)
  float *k; // key (dim,)
  float *v; // value (dim,)
  float *att; // buffer for scores/attention values (n_heads, seq_len)
  float *logits; // output logits
  // kv cache
  float *key_cache; // (layer, seq_len, dim)
  float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
  Config config; // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state; // buffers for the "wave" of activations in the forward pass
  // some more state needed to properly clean up the memory mapping (sigh)
  int fd; // file descriptor for memory mapping
  float *data; // memory mapped data pointer
  ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

static void
malloc_run_state (RunState *s, Config *p)
{
  // we calloc instead of malloc to keep valgrind happy
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s->x = calloc (p->dim, sizeof (float));
  s->xb = calloc (p->dim, sizeof (float));
  s->xb2 = calloc (p->dim, sizeof (float));
  s->hb = calloc (p->hidden_dim, sizeof (float));
  s->hb2 = calloc (p->hidden_dim, sizeof (float));
  s->q = calloc (p->dim, sizeof (float));
  s->key_cache = calloc (p->n_layers * p->seq_len * kv_dim, sizeof (float));
  s->value_cache = calloc (p->n_layers * p->seq_len * kv_dim, sizeof (float));
  s->att = calloc (p->n_heads * p->seq_len, sizeof (float));
  s->logits = calloc (p->vocab_size, sizeof (float));
  // ensure all mallocs went fine
  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->key_cache
      || !s->value_cache || !s->att || !s->logits) {
    fprintf (stderr, "malloc failed!\n");
    exit (EXIT_FAILURE);
  }
}

static void
free_run_state (RunState *s)
{
  free (s->x);
  free (s->xb);
  free (s->xb2);
  free (s->hb);
  free (s->hb2);
  free (s->q);
  free (s->att);
  free (s->logits);
  free (s->key_cache);
  free (s->value_cache);
}

static void
memory_map_weights (TransformerWeights *w, Config *p, float *ptr, int shared_weights)
{
  int head_size = p->dim / p->n_heads;
  // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
  unsigned long long n_layers = p->n_layers;
  w->token_embedding_table = ptr;
  ptr += p->vocab_size * p->dim;
  w->rms_att_weight = ptr;
  ptr += n_layers * p->dim;
  w->wq = ptr;
  ptr += n_layers * p->dim * (p->n_heads * head_size);
  w->wk = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wv = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wo = ptr;
  ptr += n_layers * (p->n_heads * head_size) * p->dim;
  w->rms_ffn_weight = ptr;
  ptr += n_layers * p->dim;
  w->w1 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->w2 = ptr;
  ptr += n_layers * p->hidden_dim * p->dim;
  w->w3 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->rms_final_weight = ptr;
  ptr += p->dim;
  ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
  ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
  w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

static void
read_checkpoint (char *checkpoint, Config *config, TransformerWeights *weights,
    int *fd, float **data, ssize_t *file_size)
{
  int shared_weights;
  float *weights_ptr;
  FILE *file = fopen (checkpoint, "rb");
  if (!file) {
    fprintf (stderr, "Couldn't open file %s\n", checkpoint);
    exit (EXIT_FAILURE);
  }
  // read in the config header
  if (fread (config, sizeof (Config), 1, file) != 1) {
    exit (EXIT_FAILURE);
  }
  // negative vocab size is hacky way of signaling unshared weights. bit yikes.
  shared_weights = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs (config->vocab_size);
  // figure out the file size
  fseek (file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell (file); // get the file size, in bytes
  fclose (file);
  // memory map the Transformer weights into the data pointer
  *fd = open (checkpoint, O_RDONLY); // open in read only mode
  if (*fd == -1) {
    fprintf (stderr, "open failed!\n");
    exit (EXIT_FAILURE);
  }
  *data = mmap (NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) {
    fprintf (stderr, "mmap failed!\n");
    exit (EXIT_FAILURE);
  }
  weights_ptr = *data + sizeof (Config) / sizeof (float);
  memory_map_weights (weights, config, weights_ptr, shared_weights);
}

static void
build_transformer (Transformer *t, char *checkpoint_path)
{
  // read in the Config and the Weights from the checkpoint
  read_checkpoint (checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
  // allocate the RunState buffers
  malloc_run_state (&t->state, &t->config);
}

static void
free_transformer (Transformer *t)
{
  // close the memory mapping
  if (t->data != MAP_FAILED) {
    munmap (t->data, t->file_size);
  }
  if (t->fd != -1) {
    close (t->fd);
  }
  // free the RunState buffers
  free_run_state (&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

static void
rmsnorm (float *o, float *x, float *weight, int size)
{
  int j;
  // calculate sum of squares
  float ss = 0.0f;
  for (j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf (ss);
  // normalize and scale
  for (j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

static void
softmax (float *x, int size)
{
  float sum;
  int i;
  // find max value (for numerical stability)
  float max_val = x[0];
  for (i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  sum = 0.0f;
  for (i = 0; i < size; i++) {
    x[i] = expf (x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

static void
matmul (float *xout, float *x, float *w, int n, int d)
{
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int i, j;

  // TODO: ignoring unknown pragmas
  //#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}

static float *
forward (Transformer *transformer, int token, int pos)
{
  unsigned long long l;

  // a few convenience variables
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim = p->hidden_dim;
  int head_size = dim / p->n_heads;

  // copy the token embedding into x
  float *content_row = w->token_embedding_table + token * dim;
  memcpy (x, content_row, dim * sizeof (*x));

  // forward all the layers
  for (l = 0; l < (unsigned long long) p->n_layers; l++) {
    int loff, i, h;
    // attention rmsnorm
    rmsnorm (s->xb, x, w->rms_att_weight + l * dim, dim);

    // key and value point to the kv cache
    loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    // qkv matmuls for this position
    matmul (s->q, s->xb, w->wq + l * dim * dim, dim, dim);
    matmul (s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
    matmul (s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    for (i = 0; i < dim; i += 2) {
      int head_dim = i % head_size, v;
      float freq = 1.0f / powf (10000.0f, head_dim / (float) head_size);
      float val = pos * freq;
      float fcr = cosf (val);
      float fci = sinf (val);
      int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
      for (v = 0; v < rotn; v++) {
        float *vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }

    // multihead attention. iterate over all heads
    // TODO: ignoring unknown pragmas
    // #pragma omp parallel for private(h)
    for (h = 0; h < p->n_heads; h++) {
      int t;
      float *xb;
      // get the query vector for this head
      float *q = s->q + h * head_size;
      // attention scores for this head
      float *att = s->att + h * p->seq_len;
      // iterate over all timesteps, including the current one
      for (t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (i = 0; i < head_size; i++) {
          score += q[i] * k[i];
        }
        score /= sqrtf (head_size);
        // save the score to the attention buffer
        att[t] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax (att, pos + 1);

      // weighted sum of the values, store back into xb
      xb = s->xb + h * head_size;
      memset (xb, 0, head_size * sizeof (float));
      for (t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // get the attention weight for this timestep
        float a = att[t];
        // accumulate the weighted value into xb
        for (i = 0; i < head_size; i++) {
          xb[i] += a * v[i];
        }
      }
    }

    // final matmul to get the output of the attention
    matmul (s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

    // residual connection back into x
    for (i = 0; i < dim; i++) {
      x[i] += s->xb2[i];
    }

    // ffn rmsnorm
    rmsnorm (s->xb, x, w->rms_ffn_weight + l * dim, dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul (s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
    matmul (s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

    // SwiGLU non-linearity
    for (i = 0; i < hidden_dim; i++) {
      float val = s->hb[i];
      // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
      val *= (1.0f / (1.0f + expf (-val)));
      // elementwise multiply with w3(x)
      val *= s->hb2[i];
      s->hb[i] = val;
    }

    // final matmul to get the output of the ffn
    matmul (s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

    // residual connection
    for (i = 0; i < dim; i++) {
      x[i] += s->xb[i];
    }
  }

  // final rmsnorm
  rmsnorm (x, x, w->rms_final_weight, dim);

  // classifier into logits
  matmul (s->logits, x, w->wcls, p->dim, p->vocab_size);
  return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
  char *str;
  int id;
} TokenIndex;

typedef struct {
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

static int
compare_tokens (const void *a, const void *b)
{
  return strcmp (((TokenIndex *) a)->str, ((TokenIndex *) b)->str);
}

static void
build_tokenizer (Tokenizer *t, char *tokenizer_path, int vocab_size)
{
  int i, len;
  FILE *file;
  // i should have written the vocab_size into the tokenizer file... sigh
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (char **) malloc (vocab_size * sizeof (char *));
  t->vocab_scores = (float *) malloc (vocab_size * sizeof (float));
  t->sorted_vocab = NULL; // initialized lazily
  for (i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char) i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }
  // read in the file
  file = fopen (tokenizer_path, "rb");
  if (!file) {
    fprintf (stderr, "couldn't load %s\n", tokenizer_path);
    exit (EXIT_FAILURE);
  }
  if (fread (&t->max_token_length, sizeof (int), 1, file) != 1) {
    fprintf (stderr, "failed read\n");
    exit (EXIT_FAILURE);
  }

  for (i = 0; i < vocab_size; i++) {
    if (fread (t->vocab_scores + i, sizeof (float), 1, file) != 1) {
      fprintf (stderr, "failed read\n");
      exit (EXIT_FAILURE);
    }
    if (fread (&len, sizeof (int), 1, file) != 1) {
      fprintf (stderr, "failed read\n");
      exit (EXIT_FAILURE);
    }
    t->vocab[i] = (char *) malloc (len + 1);
    if (fread (t->vocab[i], len, 1, file) != 1) {
      fprintf (stderr, "failed read\n");
      exit (EXIT_FAILURE);
    }
    t->vocab[i][len] = '\0'; // add the string terminating token
  }
  fclose (file);
}

static void
free_tokenizer (Tokenizer *t)
{
  int i;
  for (i = 0; i < t->vocab_size; i++) {
    free (t->vocab[i]);
  }
  free (t->vocab);
  free (t->vocab_scores);
  free (t->sorted_vocab);
}

static char *
decode (Tokenizer *t, int prev_token, int token)
{
  unsigned char byte_val;
  char *piece = t->vocab[token];
  // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
  if (prev_token == 1 && piece[0] == ' ') {
    piece++;
  }
  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  if (sscanf (piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char *) t->byte_pieces + byte_val * 2;
  }
  return piece;
}

static void
safe_save (char *piece, char *result)
{
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  if (piece == NULL) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint (byte_val) || isspace (byte_val))) {
      return; // bad byte, don't print it
    }
  }

  // TODO: reduce memcpy
  strncat (result, piece, PROMPT_LENGTH);
}

static int
str_lookup (char *str, TokenIndex *sorted_vocab, int vocab_size)
{
  // efficiently find the perfect match for str in vocab, return its index or -1 if not found
  TokenIndex tok = { .str = str }; // acts as the key to search for
  TokenIndex *res = bsearch (&tok, sorted_vocab, vocab_size, sizeof (TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

static void
encode (Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens)
{
  int i;
  char *str_buffer, *c;
  size_t str_len;
  // encode the string text (input) into an upper-bound preallocated tokens[] array
  // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
  if (text == NULL) {
    fprintf (stderr, "cannot encode NULL text\n");
    exit (EXIT_FAILURE);
  }

  if (t->sorted_vocab == NULL) {
    // lazily malloc and sort the vocabulary
    t->sorted_vocab = malloc (t->vocab_size * sizeof (TokenIndex));
    for (i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort (t->sorted_vocab, t->vocab_size, sizeof (TokenIndex), compare_tokens);
  }

  // create a temporary buffer that will store merge candidates of always two consecutive tokens
  // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
  str_buffer = malloc ((t->max_token_length * 2 + 1 + 2) * sizeof (char));
  str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=1) token, if desired
  if (bos)
    tokens[(*n_tokens)++] = 1;

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have the
  // energy to read more of the sentencepiece code to figure out what it's doing
  if (text[0] != '\0') {
    int dummy_prefix = str_lookup ((char *) " ", t->sorted_vocab, t->vocab_size);
    tokens[(*n_tokens)++] = dummy_prefix;
  }

  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ↔ UTF-8 conversion
  // First code point Last code point Byte 1  Byte 2  Byte 3  Byte 4
  // U+0000   U+007F      0xxxxxxx
  // U+0080   U+07FF      110xxxxx    10xxxxxx
  // U+0800   U+FFFF      1110xxxx    10xxxxxx        10xxxxxx
  // U+10000  U+10FFFF    11110xxx    10xxxxxx        10xxxxxx        10xxxxxx

  // process the raw (UTF-8) byte sequence of the input string
  for (c = text; *c != '\0'; c++) {
    int id;
    // reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
    // 0x80 is 10000000
    // in UTF-8, all continuation bytes start with "10" in first two bits
    // so in English this is: "if this byte is not a continuation byte"
    if ((*c & 0xC0) != 0x80) {
      // this byte must be either a leading byte (11...) or an ASCII char (0x...)
      // => reset our location, as we're starting a new UTF-8 codepoint
      str_len = 0;
    }

    // append the current byte to the buffer
    str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
    str_buffer[str_len] = '\0';

    // while the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning str_buffer size.
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // ok c+1 is not a continuation byte, so we've read in a full codepoint
    id = str_lookup (str_buffer, t->sorted_vocab, t->vocab_size);

    if (id != -1) {
      // we found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      for (i = 0; i < (int) str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char) str_buffer[i] + 3;
      }
    }
    str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair each iteration, according the scores in vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (i = 0; i < (*n_tokens - 1); i++) {
      int id;
      // check if we can merge the pair (tokens[i], tokens[i+1])
      sprintf (str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
      id = str_lookup (str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1) {
      break; // we couldn't find any more pairs to merge, so we're done
    }

    // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    tokens[best_idx] = best_id;
    // delete token at position best_idx+1, shift the entire sequence back 1
    for (i = best_idx + 1; i < (*n_tokens - 1); i++) {
      tokens[i] = tokens[i + 1];
    }
    (*n_tokens)--; // token length decreased
  }

  // add optional EOS (=2) token, if desired
  if (eos)
    tokens[(*n_tokens)++] = 2;

  free (str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
  int vocab_size;
  ProbIndex *probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

static int
sample_argmax (float *probabilities, int n)
{
  // return the index that has the highest probability
  int max_i = 0, i;
  float max_p = probabilities[0];
  for (i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

static int
sample_mult (float *probabilities, int n, float coin)
{
  int i;
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

static int
compare (const void *a, const void *b)
{
  ProbIndex *a_ = (ProbIndex *) a;
  ProbIndex *b_ = (ProbIndex *) b;
  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;
  return 0;
}

static int
sample_topp (float *probabilities, int n, float topp, ProbIndex *probindex, float coin)
{
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int i, last_idx, n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  float cumulative_prob, r, cdf;
  for (i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort (probindex, n0, sizeof (ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  cumulative_prob = 0.0f;
  last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  r = coin * cumulative_prob;
  cdf = 0.0f;
  for (i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

static void
build_sampler (Sampler *sampler, int vocab_size, float temperature, float topp,
    unsigned long long rng_seed)
{
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = malloc (sampler->vocab_size * sizeof (ProbIndex));
}

static void
free_sampler (Sampler *sampler)
{
  free (sampler->probindex);
}

static unsigned int
random_u32 (unsigned long long *state)
{
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

static float
random_f32 (unsigned long long *state)
{ // random float32 in [0,1)
  return (random_u32 (state) >> 8) / 16777216.0f;
}

static int
sample (Sampler *sampler, float *logits)
{
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax (logits, sampler->vocab_size);
  } else {
    int q;
    float coin;
    // apply the temperature to the logits
    for (q = 0; q < sampler->vocab_size; q++) {
      logits[q] /= sampler->temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax (logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    coin = random_f32 (&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult (logits, sampler->vocab_size, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp (
          logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

// ----------------------------------------------------------------------------
// utilities: time

static long
time_in_ms (void)
{
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  clock_gettime (CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

static void
generate (Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
    char *prompt, int steps, char *result)
{
  int next, token, pos, num_prompt_tokens = 0, *prompt_tokens;
  long start;
  const char *empty_prompt = "";
  if (prompt == NULL) {
    prompt = (char *) empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  prompt_tokens = (int *) malloc ((strlen (prompt) + 3) * sizeof (int)); // +3 for '\0', ?BOS, ?EOS
  encode (tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    fprintf (stderr, "something is wrong, expected at least 1 prompt token\n");
    exit (EXIT_FAILURE);
  }

  // start the main loop
  start = 0; // used to time our code, only initialized after first iteration
  token = prompt_tokens[0]; // kick off with the first token in the prompt
  pos = 0; // position in the sequence
  while (pos < steps) {
    char *piece;
    // forward the transformer to get logits for the next token
    float *logits = forward (transformer, token, pos);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = sample (sampler, logits);
    }
    pos++;

    // data-dependent terminating condition: the BOS (=1) token delimits sequences
    if (next == 1) {
      break;
    }

    // print the token as string, decode it with the Tokenizer object
    piece = decode (tokenizer, token, next);
    safe_save (piece, result); // skips "unsafe" bytes
    fflush (stdout);
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) {
      start = time_in_ms ();
    }
  }
  printf ("\n");

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  if (pos > 1) {
    long end = time_in_ms ();
    fprintf (stderr, "achieved tok/s: %f\n", (pos - 1) / (double) (end - start) * 1000);
  }

  free (prompt_tokens);
}

/**
 * @brief _pt_data Internal data structure
 */
typedef struct _pt_data {
  Transformer *transformer;
  Tokenizer *tokenizer;
  Sampler *sampler;
  char *prompt; // prompt string
  int steps; // number of steps to run for

  GstTensorsInfo info;
} pt_data;

/**
 * @brief init callback of tensor_filter custom
 */
static void *
llama_init (const GstTensorFilterProperties *prop)
{
  pt_data *data = (pt_data *) malloc (sizeof (pt_data));
  Config *config;
  UNUSED (prop);

  gst_tensors_info_init (&data->info);

  data->info.num_tensors = 1;
  data->info.info[0].type = _NNS_UINT8;
  data->info.info[0].dimension[0] = PROMPT_LENGTH;

  data->transformer = (Transformer *) malloc (sizeof (Transformer));

  // build the Transformer via the model .bin file
  build_transformer (data->transformer, (char *) CHECKPOINT_PATH);
  config = &data->transformer->config;

  if (data->steps == 0 || data->steps > config->seq_len)
    data->steps = config->seq_len; // override to ~max length

  data->tokenizer = (Tokenizer *) malloc (sizeof (Tokenizer));

  // build the Tokenizer via the tokenizer .bin file
  build_tokenizer (data->tokenizer, (char *) TOKENIZER_PATH, config->vocab_size);

  data->sampler = (Sampler *) malloc (sizeof (Sampler));

  // build the Sampler
  build_sampler (data->sampler, config->vocab_size, TEMPERATURE, TOPP, RNG_SEED);

  return data;
}

/**
 * @brief exit callback of tensor_filter custom
 */
static void
llama_exit (void *private_data, const GstTensorFilterProperties *prop)
{
  pt_data *data = private_data;
  UNUSED (prop);

  gst_tensors_info_free (&data->info);
  free_transformer (data->transformer);
  free_tokenizer (data->tokenizer);
  free_sampler (data->sampler);

  free (data->transformer);
  free (data->tokenizer);
  free (data->sampler);
  free (data);
}

/**
 * @brief setInputDimension callback of tensor_filter custom
 */
static int
get_inputDim (void *private_data, const GstTensorFilterProperties *prop, GstTensorsInfo *info)
{
  pt_data *data = private_data;
  UNUSED (prop);

  gst_tensors_info_copy (info, &data->info);
  return 0;
}

/**
 * @brief getOutputDimension callback of tensor_filter custom
 */
static int
get_outputDim (void *private_data, const GstTensorFilterProperties *prop, GstTensorsInfo *info)
{
  pt_data *data = private_data;
  UNUSED (prop);

  gst_tensors_info_copy (info, &data->info);
  return 0;
}

/**
 * @brief invoke callback of tensor_filter custom
 */
static int
llama_invoke (void *private_data, const GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  pt_data *data = private_data;
  char *result = (char *) malloc (PROMPT_LENGTH);
  char *prompt = (char *) input[0].data;
  int len;
  UNUSED (prop);

  len = strlen (prompt);
  if (len > 1) {
    prompt[strlen (prompt) - 1] = ' '; // Remove new line from text file
  } else {
    prompt = NULL;
  }

  output->size = PROMPT_LENGTH;

  generate (data->transformer, data->tokenizer, data->sampler, prompt, data->steps, result);
  memcpy (output[0].data, result, PROMPT_LENGTH);

  free (result);
  return 0;
}

static NNStreamer_custom_class NNStreamer_custom_body = {
  .initfunc = llama_init,
  .exitfunc = llama_exit,
  .getInputDim = get_inputDim,
  .getOutputDim = get_outputDim,
  .invoke = llama_invoke,
};

/* The dyn-loaded object */
NNStreamer_custom_class *NNStreamer_custom = &NNStreamer_custom_body;
