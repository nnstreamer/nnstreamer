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
 */

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <tensor_filter_custom.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_util.h>

#define CHECKPOINT_PATH "model.bin"

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
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
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
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

static void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

static void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

static void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
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

static void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    int shared_weights;
    float* weights_ptr;
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

static void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

static void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

/**
 * @brief _pt_data Internal data structure
 */
typedef struct _pt_data {
  Transformer* transformer;
} pt_data;

/**
 * @brief init callback of tensor_filter custom
 */
static void *
llama_init (const GstTensorFilterProperties * prop)
{
  pt_data *data = (pt_data *) malloc (sizeof (pt_data));
  UNUSED (prop);

  data->transformer = (Transformer *) malloc (sizeof (Transformer));

  build_transformer(data->transformer, (char*) CHECKPOINT_PATH);
  return data;
}

/**
 * @brief exit callback of tensor_filter custom
 */
static void
llama_exit (void *private_data, const GstTensorFilterProperties * prop)
{
  pt_data *data = private_data;
  UNUSED (prop);

  free_transformer(data->transformer);
  free (data->transformer);
  free (data);
}

/**
 * @brief setInputDimension callback of tensor_filter custom
 */
static int
get_inputDim (void *private_data, const GstTensorFilterProperties * prop,
    GstTensorsInfo * info)
{
  UNUSED (private_data);
  UNUSED (prop);
  UNUSED (info);
  return 0;
}

/**
 * @brief getOutputDimension callback of tensor_filter custom
 */
static int
get_outputDim (void *private_data, const GstTensorFilterProperties * prop,
    GstTensorsInfo * info)
{
  UNUSED (private_data);
  UNUSED (prop);
  UNUSED (info);
  return 0;
}

/**
 * @brief invoke callback of tensor_filter custom
 */
static int
llama_invoke (void *private_data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  UNUSED (private_data);
  UNUSED (prop);
  UNUSED (input);
  UNUSED (output);

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
