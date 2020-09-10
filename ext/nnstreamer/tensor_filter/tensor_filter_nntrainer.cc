/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Tensor_Filter, nntrainer Module
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 */
/**
 * @file	tensor_filter_nntrainer.cc
 * @date	09 Sept 2020
 * @brief	nntrainer inference module for tensor_filter gstreamer plugin
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (tensorflow-lite) for tensor_filter.
 * Fill in "GstTensorFilterFramework" for tensor_filter.h/c
 *
 */

#include <algorithm>
#include <limits.h>
#include <unistd.h>

#include <nnstreamer_conf.h>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>

#include <neuralnet.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

#define NUM_DIM 4

static const gchar *nntrainer_accl_support[] = { NULL };

/**
 * @brief	Internal data structure for nntrainer
 */
typedef struct {
  int rank;
  std::vector<std::int64_t> dims;
} nntrainer_tensor_info_s;

class NNTrainer
{
  public:
  /**
   * member functions.
   */
  NNTrainer (const char *model_config_);
  ~NNTrainer ();

  int init (const GstTensorFilterProperties *prop);
  int loadModel ();
  const char *getModelConfig ();

  int getInputTensorDim (GstTensorsInfo *info);
  int getOutputTensorDim (GstTensorsInfo *info);
  int run (const GstTensorMemory *input, GstTensorMemory *output);
  void freeOutputTensor (void *data);
  int validateTensor (const GstTensorsInfo *tensorInfo, int is_input);

  private:
  char *model_config;
  nntrainer::NeuralNetwork *model;

  GstTensorsInfo inputTensorMeta;
  GstTensorsInfo outputTensorMeta;

  std::vector<nntrainer_tensor_info_s> input_tensor_info;
  std::map<void *, nntrainer::Tensor *> outputTensorMap;
};

void init_filter_nntrainer (void) __attribute__ ((constructor));
void fini_filter_nntrainer (void) __attribute__ ((destructor));

NNTrainer::NNTrainer (const char *model_config_)
{
  g_assert (model_config_ != NULL);
  model_config = g_strdup (model_config_);
  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);
}

NNTrainer::~NNTrainer ()
{
  if (model != nullptr) {
    model->finalize ();
    delete model;
  }

  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);
  g_free (model_config);
}

int
NNTrainer::init (const GstTensorFilterProperties *prop)
{
  if (loadModel ()) {
    ml_loge ("Failed to load model");
    return -1;
  }

  if (validateTensor (&prop->input_meta, 1)) {
    ml_loge ("Failed to validate input tensor");
    return -2;
  }

  if (validateTensor (&prop->output_meta, 0)) {
    ml_loge ("Failed to validate output tensor");
    return -3;
  }

  try {
    model->init ();
    model->readModel ();
  } catch (...) {
    ml_loge ("Failed to initialize model");
    model->finalize ();
    return -1;
  }

  gst_tensors_info_copy (&inputTensorMeta, &prop->input_meta);
  gst_tensors_info_copy (&outputTensorMeta, &prop->output_meta);

  return 0;
}

const char *
NNTrainer::getModelConfig ()
{
  return model_config;
}

int
NNTrainer::validateTensor (const GstTensorsInfo *tensorInfo, int is_input)
{

  nntrainer::TensorDim dim;
  nntrainer_tensor_info_s info_s;
  unsigned int order[3] = { 1, 3, 2 };
  // unsigned int order[3] = {1, 2, 3};

  if (is_input)
    dim = model->getInputDimension ();
  else
    dim = model->getOutputDimension ();

  g_assert (tensorInfo->info[0].type == _NNS_FLOAT32);
  info_s.rank = NUM_DIM;

  for (unsigned int i = 0; i < NUM_DIM - 1; ++i) {
    g_assert (tensorInfo->info[0].dimension[i] == dim.getDim ()[order[i]]);
    info_s.dims.push_back (dim.getDim ()[order[i]]);
  }

  g_assert (tensorInfo->info[0].dimension[NUM_DIM - 1] == dim.getDim ()[0]);
  info_s.dims.push_back (dim.getDim ()[0]);

  if (is_input) {
    input_tensor_info.push_back (info_s);
  }

  return 0;
}

int
NNTrainer::loadModel ()
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif
  gsize file_size;
  gchar *content = nullptr;
  GError *file_error = nullptr;

  g_assert (model_config != nullptr);
  if (!g_file_test (model_config, G_FILE_TEST_IS_REGULAR)) {
    ml_loge ("the file of model_config (%s) is not valid (not regular)\n", model_config);
    return -1;
  }

  if (!g_file_get_contents (model_config, &content, &file_size, &file_error)) {
    ml_loge ("Error reading model config!! - %s", file_error->message);
    g_clear_error (&file_error);
    return -2;
  }

  model = new nntrainer::NeuralNetwork (false);

  try {
    model->loadFromConfig (model_config);
  } catch (...) {
    ml_loge ("Cannot load model from config\n");
    model->finalize ();
    return -1;
  }

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Model is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  return 0;
}

int
NNTrainer::getInputTensorDim (GstTensorsInfo *info)
{
  gst_tensors_info_copy (info, &inputTensorMeta);
  return 0;
}

int
NNTrainer::getOutputTensorDim (GstTensorsInfo *info)
{
  gst_tensors_info_copy (info, &outputTensorMeta);
  return 0;
}

int
NNTrainer::run (const GstTensorMemory *input, GstTensorMemory *output)
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif
  std::vector<nntrainer::Tensor *> output_tensors;
  nntrainer::Tensor *out = new nntrainer::Tensor ();

  std::vector<std::int64_t> d = input_tensor_info[0].dims;
  nntrainer::Tensor X
      = nntrainer::Tensor (nntrainer::TensorDim (d[3], d[0], d[2], d[1]),
          static_cast<float *> (input[0].data));

  output_tensors.push_back (out);
  std::shared_ptr<const nntrainer::Tensor> o;

  o = model->inference (X);
  if (o == nullptr) {
    model->finalize ();
    delete out;
    return -1;
  }

  *out = *o;

  output[0].data = out->getData ();

  outputTensorMap.insert (std::make_pair (output[0].data, output_tensors[0]));

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Run() is finished: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  return 0;
}

void
NNTrainer::freeOutputTensor (void *data)
{
  if (data != nullptr) {
    std::map<void *, nntrainer::Tensor *>::iterator it = outputTensorMap.find (data);
    if (it != outputTensorMap.end ()) {
      std::cout << "asdfasdfasdfasdf" << std::endl;
      delete it->second;
      outputTensorMap.erase (data);
    }
  }
}

static void
nntrainer_close (const GstTensorFilterProperties *prop, void **private_data)
{
  NNTrainer *nntrainer = static_cast<NNTrainer *> (*private_data);

  if (!nntrainer)
    return;
  delete nntrainer;
  *private_data = NULL;
}

static int
nntrainer_loadModelFile (const GstTensorFilterProperties *prop, void **private_data)
{
  NNTrainer *nntrainer;
  const gchar *model_file;
  if (prop->num_models != 1)
    return -1;

  nntrainer = static_cast<NNTrainer *> (*private_data);
  model_file = prop->model_files[0];

  if (nntrainer != NULL) {
    if (g_strcmp0 (model_file, nntrainer->getModelConfig ()) == 0)
      return 1; /* skipped */

    nntrainer_close (prop, private_data);
  }

  nntrainer = new NNTrainer (model_file);
  if (nntrainer == NULL) {
    g_printerr ("Failed to allocate memory for filter subplugin: nntrainer\n");
    return -1;
  }

  if (nntrainer->init (prop) != 0) {
    *private_data = NULL;
    delete nntrainer;

    g_printerr ("failed to initailize the object: nntrainer\n");
    return -2;
  }

  *private_data = nntrainer;
  return 0;
}

static int
nntrainer_open (const GstTensorFilterProperties *prop, void **private_data)
{
  int status = nntrainer_loadModelFile (prop, private_data);
  return status;
}

static int
nntrainer_run (const GstTensorFilterProperties *prop, void **private_data,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  NNTrainer *nntrainer = static_cast<NNTrainer *> (*private_data);
  g_return_val_if_fail (nntrainer && input && output, -EINVAL);

  return nntrainer->run (input, output);
}

static int
nntrainer_getInputDim (const GstTensorFilterProperties *prop,
    void **private_data, GstTensorsInfo *info)
{
  NNTrainer *nntrainer = static_cast<NNTrainer *> (*private_data);
  g_return_val_if_fail (nntrainer && info, -EINVAL);
  return nntrainer->getInputTensorDim (info);
}

static int
nntrainer_getOutputDim (const GstTensorFilterProperties *prop,
    void **private_data, GstTensorsInfo *info)
{
  NNTrainer *nntrainer = static_cast<NNTrainer *> (*private_data);
  g_return_val_if_fail (nntrainer && info, -EINVAL);
  return nntrainer->getOutputTensorDim (info);
}

static void
nntrainer_destroyNotify (void **private_data, void *data)
{
  NNTrainer *nntrainer = static_cast<NNTrainer *> (*private_data);
  std::cout << "nnstrainer_dest" << std::endl;
  if (nntrainer) {
    nntrainer->freeOutputTensor (data);
  }
}

static int
nntrainer_checkAvailability (accl_hw hw)
{
  if (g_strv_contains (nntrainer_accl_support, get_accl_hw_str (hw)))
    return 0;

  return -ENOENT;
}


static gchar filter_subplugin_nntrainer[] = "nntrainer";

static GstTensorFilterFramework NNS_support_nntrainer = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = nntrainer_open,
  .close = nntrainer_close,
};


void
init_filter_nntrainer (void)
{
  NNS_support_nntrainer.name = filter_subplugin_nntrainer;
  NNS_support_nntrainer.allow_in_place = FALSE;
  NNS_support_nntrainer.allocate_in_invoke = TRUE;
  NNS_support_nntrainer.run_without_model = FALSE;
  NNS_support_nntrainer.verify_model_path = FALSE;
  NNS_support_nntrainer.invoke_NN = nntrainer_run;
  NNS_support_nntrainer.getInputDimension = nntrainer_getInputDim;
  NNS_support_nntrainer.getOutputDimension = nntrainer_getOutputDim;
  NNS_support_nntrainer.destroyNotify = nntrainer_destroyNotify;
  NNS_support_nntrainer.checkAvailability = nntrainer_checkAvailability;

  nnstreamer_filter_probe (&NNS_support_nntrainer);
}

void
fini_filter_nntrainer (void)
{
  nnstreamer_filter_exit (NNS_support_nntrainer.name);
}
