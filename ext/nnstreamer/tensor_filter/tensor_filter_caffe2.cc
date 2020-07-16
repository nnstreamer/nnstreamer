/**
 * GStreamer Tensor_Filter, caffe2 Module
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All rights reserved.
 * Copyright (C) 2019 Hyoung Joo Ahn <hello.ahn@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file    tensor_filter_caffe2.cc
 * @date    27 May 2019
 * @brief   Caffe2 module for tensor_filter gstreamer plugin
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  HyoungJoo Ahn <hello.ahn@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (caffe2) for tensor_filter.
 */

#include <iostream>
#include <unistd.h>
#include <algorithm>

#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>

#include <caffe2/core/workspace.h>
#include <caffe2/core/init.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

static const gchar *caffe2_accl_support[] = { NULL };

using namespace caffe2;

/**
 * @brief	ring cache structure
 */
class Caffe2Core
{
public:
  Caffe2Core (const char * _model_path, const char * _model_path_sub);
  ~Caffe2Core ();

  int init (const GstTensorFilterProperties * prop);
  int loadModels ();
  const char* getPredModelPath ();
  const char* getInitModelPath ();
  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int run (const GstTensorMemory * input, GstTensorMemory * output);

private:

  char *init_model_path;
  char *pred_model_path;
  bool first_run;

  GstTensorsInfo inputTensorMeta;  /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta;  /**< The tensor info of output tensors */

  Workspace workSpace;
  NetDef initNet, predictNet;
  std::map <char*, Tensor*> inputTensorMap;

  int initInputTensor ();
};

void init_filter_caffe2 (void) __attribute__ ((constructor));
void fini_filter_caffe2 (void) __attribute__ ((destructor));

/**
 * @brief	Caffe2Core creator
 * @param	_model_path	: the logical path to '{model_name}.tffile' file
 * @note	the model of _model_path will be loaded simultaneously
 * @return	Nothing
 */
Caffe2Core::Caffe2Core (const char * _model_path, const char *_model_path_sub)
{
  init_model_path = g_strdup (_model_path);
  pred_model_path = g_strdup (_model_path_sub);
  first_run = true;

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);

  if (!GlobalInitAlreadyRun () && !GlobalInit ()) {
    throw std::runtime_error ("Failed to initialize caffe2.");
  }
}

/**
 * @brief	Caffe2Core Destructor
 * @return	Nothing
 */
Caffe2Core::~Caffe2Core ()
{
  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);
  g_free (pred_model_path);
  g_free (init_model_path);
}

/**
 * @brief	initialize the object with caffe2 model
 * @return 0 if OK. non-zero if error.
 *        -1 if the model is not loaded.
 *        -2 if the initialization of input tensor is failed.
 *        -3 if the initialization of output tensor is failed.
 */
int
Caffe2Core::init (const GstTensorFilterProperties * prop)
{
  if (loadModels ()) {
    ml_loge ("Failed to load model\n");
    return -1;
  }

  gst_tensors_info_copy (&inputTensorMeta, &prop->input_meta);
  gst_tensors_info_copy (&outputTensorMeta, &prop->output_meta);

  if (initInputTensor ()) {
    ml_loge ("Failed to initialize input tensor\n");
    return -2;
  }

  first_run = true;
  return 0;
}

#define initializeTensor(type)\
do {\
  ReinitializeTensor (\
      inputTensor,\
      {\
        inputTensorMeta.info[i].dimension[3],\
        inputTensorMeta.info[i].dimension[2],\
        inputTensorMeta.info[i].dimension[1],\
        inputTensorMeta.info[i].dimension[0]\
      },\
      at::dtype<type> ().device (CPU)\
  );\
} while (0);

/**
 * @brief initialize the input tensor
 */
int
Caffe2Core::initInputTensor ()
{
  guint i;

  inputTensorMap.clear ();
  for (i = 0; i < inputTensorMeta.num_tensors; i++) {
    Tensor *inputTensor = workSpace.CreateBlob (inputTensorMeta.info[i].name)
      ->GetMutable<Tensor> ();

    switch (inputTensorMeta.info[i].type) {
      case _NNS_INT32:
        initializeTensor (int32_t);
        break;
      case _NNS_UINT32:
        ml_loge ("invalid data type is used");
        return -1;
      case _NNS_INT16:
        initializeTensor (int16_t);
        break;
      case _NNS_UINT16:
        initializeTensor (uint16_t);
        break;
      case _NNS_INT8:
        initializeTensor (int8_t);
        break;
      case _NNS_UINT8:
        initializeTensor (uint8_t);
        break;
      case _NNS_FLOAT64:
        initializeTensor (double);
        break;
      case _NNS_FLOAT32:
        initializeTensor (float);
        break;
      case _NNS_INT64:
        initializeTensor (int64_t);
        break;
      case _NNS_UINT64:
        ml_loge ("invalid data type is used");
        return -1;
      default:
        ml_loge ("invalid data type is used");
        return -1;
    }

    inputTensorMap.insert (
      std::make_pair (inputTensorMeta.info[i].name, inputTensor)
    );
  }
  return 0;
}

/**
 * @brief	get the model path
 * @return the model path.
 */
const char *
Caffe2Core::getPredModelPath ()
{
  return pred_model_path;
}

/**
 * @brief	get the model path
 * @return the model path.
 */
const char *
Caffe2Core::getInitModelPath ()
{
  return init_model_path;
}

/**
 * @brief	load the caffe2 model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 */
int
Caffe2Core::loadModels ()
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif
  if (!g_file_test (init_model_path, G_FILE_TEST_IS_REGULAR)) {
    ml_loge ("the file of init_model_path is not valid: %s\n", init_model_path);
    return -1;
  }
  if (!g_file_test (pred_model_path, G_FILE_TEST_IS_REGULAR)) {
    ml_loge ("the file of pred_model_path is not valid: %s\n", pred_model_path);
    return -1;
  }
  CAFFE_ENFORCE (ReadProtoFromFile (init_model_path, &initNet));
  CAFFE_ENFORCE (ReadProtoFromFile (pred_model_path, &predictNet));

  /* set device type as CPU. If it is required, GPU/CUDA will be added as an option */
  predictNet.mutable_device_option()->set_device_type(PROTO_CPU);
  initNet.mutable_device_option()->set_device_type(PROTO_CPU);

  for (int i = 0; i < predictNet.op_size(); ++i) {
    predictNet.mutable_op(i)->mutable_device_option()->set_device_type(PROTO_CPU);
  }
  for (int i = 0; i < initNet.op_size(); ++i) {
    initNet.mutable_op(i)->mutable_device_option()->set_device_type(PROTO_CPU);
  }

  CAFFE_ENFORCE (workSpace.RunNetOnce (initNet));
  CAFFE_ENFORCE (workSpace.CreateNet (predictNet));
#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Model is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  return 0;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
Caffe2Core::getInputTensorDim (GstTensorsInfo * info)
{
  gst_tensors_info_copy (info, &inputTensorMeta);
  return 0;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
Caffe2Core::getOutputTensorDim (GstTensorsInfo * info)
{
  gst_tensors_info_copy (info, &outputTensorMeta);
  return 0;
}

/**
 * @brief	run the model with the input.
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
Caffe2Core::run (const GstTensorMemory * input, GstTensorMemory * output)
{
  unsigned int i;
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  for (i = 0; i < inputTensorMeta.num_tensors; i++) {
    Tensor *inputTensor = inputTensorMap.
                            find(inputTensorMeta.info[i].name)->second;

    switch (inputTensorMeta.info[i].type) {
      case _NNS_INT32:
        inputTensor->ShareExternalPointer ((int32_t*) input[i].data);
        break;
      case _NNS_UINT32:
        ml_loge ("invalid data type is used");
        return -1;
      case _NNS_INT16:
        inputTensor->ShareExternalPointer ((int16_t*) input[i].data);
        break;
      case _NNS_UINT16:
        inputTensor->ShareExternalPointer ((uint16_t*) input[i].data);
        break;
      case _NNS_INT8:
        inputTensor->ShareExternalPointer ((int8_t*) input[i].data);
        break;
      case _NNS_UINT8:
        inputTensor->ShareExternalPointer ((uint8_t*) input[i].data);
        break;
      case _NNS_FLOAT64:
        inputTensor->ShareExternalPointer ((double*) input[i].data);
        break;
      case _NNS_FLOAT32:
        inputTensor->ShareExternalPointer ((float*) input[i].data);
        break;
      case _NNS_INT64:
        inputTensor->ShareExternalPointer ((int64_t*) input[i].data);
        break;
      case _NNS_UINT64:
        ml_loge ("invalid data type is used");
        return -1;
      default:
        ml_loge ("invalid data type is used");
        return -1;
    }
  }

  /**
   * As the input information has not been verified, the first run for the model
   * is encapsulated in a try-catch block
   */
  if (first_run) {
    try {
      workSpace.RunNet (predictNet.name ());
      first_run = false;
    } catch(const std::runtime_error& re) {
      ml_loge ("Runtime error while running the model: %s", re.what());
      return -4;
    } catch(const std::exception& ex)	{
      ml_loge ("Exception while running the model : %s", ex.what());
      return -4;
    } catch (...) {
      ml_loge ("Unknown exception while running the model");
      return -4;
    }
  } else {
    workSpace.RunNet (predictNet.name ());
  }

  for (i = 0; i < outputTensorMeta.num_tensors; i++) {
    const auto& out = workSpace.GetBlob (outputTensorMeta.info[i].name)
      ->Get<Tensor> ();

    switch (outputTensorMeta.info[i].type) {
      case _NNS_INT32:
        output[i].data = out.data<int32_t>();
        break;
      case _NNS_UINT32:
        ml_loge ("invalid data type (uint32) is used");
        return -1;
      case _NNS_INT16:
        output[i].data = out.data<int16_t>();
        break;
      case _NNS_UINT16:
        output[i].data = out.data<uint16_t>();
        break;
      case _NNS_INT8:
        output[i].data = out.data<int8_t>();
        break;
      case _NNS_UINT8:
        output[i].data = out.data<uint8_t>();
        break;
      case _NNS_FLOAT64:
        output[i].data = out.data<double>();
        break;
      case _NNS_FLOAT32:
        output[i].data = out.data<float>();
        break;
      case _NNS_INT64:
        output[i].data = out.data<int64_t>();
        break;
      case _NNS_UINT64:
        ml_loge ("invalid data type (uint64) is used");
        return -1;
      default:
        ml_loge ("invalid data type is used");
        return -1;
    }
  }

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Run() is finished: %" G_GINT64_FORMAT,
      (stop_time - start_time));
#endif

  return 0;
}

/**
 * @brief Free privateData and move on.
 */
static void
caffe2_close (const GstTensorFilterProperties * prop, void **private_data)
{
  Caffe2Core *core = static_cast<Caffe2Core *>(*private_data);

  if (!core)
    return;

  delete core;
  *private_data = NULL;
}

/**
 * @brief Load caffe2 modelfile
 * @param prop property of tensor_filter instance
 * @param private_data : caffe2 plugin's private data
 * @return 0 if successfully loaded. 1 if skipped (already loaded).
 *        -1 if the object construction is failed.
 *        -2 if the object initialization if failed
 */
static int
caffe2_loadModelFile (const GstTensorFilterProperties * prop,
    void **private_data)
{
  Caffe2Core *core;
  const gchar *init_model;
  const gchar *pred_model;

  if (prop->num_models != 2) {
    ml_loge ("Caffe2 requires two model files\n");
    return -1;
  }

  /* In caffe2, model_files[0] is a init model, and model_files[1] is a pred model */
  core = static_cast<Caffe2Core *>(*private_data);
  init_model = prop->model_files[0];
  pred_model = prop->model_files[1];
  g_return_val_if_fail (init_model && pred_model, -1);

  if (core != NULL) {
    if (g_strcmp0 (init_model, core->getInitModelPath ()) == 0 &&
        g_strcmp0 (pred_model, core->getPredModelPath ()) == 0)
      return 1; /* skipped */

    caffe2_close (prop, private_data);
  }

  try {
    core = new Caffe2Core (init_model, pred_model);
  } catch (std::bad_alloc &e) {
    ml_loge ("Failed to allocate memory for filter subplugin: Caffe2\n");
    return -1;
  } catch (std::runtime_error &e) {
    ml_loge ("Error for subplugin Caffe2: %s.", e.what ());
    return -1;
  } catch (...) {
    ml_loge ("Unknown error thrown for subplugin Caffe2.");
    return -1;
  }

  if (core->init (prop) != 0) {
    *private_data = NULL;
    delete core;

    ml_loge ("failed to initialize the object: Caffe2");
    return -2;
  }

  *private_data = core;

  return 0;
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop property of tensor_filter instance
 * @param private_data : caffe2 plugin's private data
 */
static int
caffe2_open (const GstTensorFilterProperties * prop, void **private_data)
{
  int status = caffe2_loadModelFile (prop, private_data);

  return status;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : caffe2 plugin's private data
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
static int
caffe2_run (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  Caffe2Core *core = static_cast<Caffe2Core *>(*private_data);
  g_return_val_if_fail (core && input && output, -EINVAL);

  return core->run (input, output);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : caffe2 plugin's private data
 * @param[out] info The dimesions and types of input tensors
 */
static int
caffe2_getInputDim (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * info)
{
  Caffe2Core *core = static_cast<Caffe2Core *>(*private_data);
  g_return_val_if_fail (core && info, -EINVAL);

  return core->getInputTensorDim (info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : caffe2 plugin's private data
 * @param[out] info The dimesions and types of output tensors
 */
static int
caffe2_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  Caffe2Core *core = static_cast<Caffe2Core *>(*private_data);
  g_return_val_if_fail (core && info, -EINVAL);

  return core->getOutputTensorDim (info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param[in] private_data caffe2 plugin's private data
 * @param[in] data The data element.
 */
static void
caffe2_destroyNotify (void **private_data, void *data)
{
  /* do nothing */
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param[in] hw backend accelerator hardware
 * @return 0 if supported. -errno if not supported.
 */
static int
caffe2_checkAvailability (accl_hw hw)
{
  if (g_strv_contains (caffe2_accl_support, get_accl_hw_str (hw)))
    return 0;

  return -ENOENT;
}

static gchar filter_subplugin_caffe2[] = "caffe2";

static GstTensorFilterFramework NNS_support_caffe2 = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = caffe2_open,
  .close = caffe2_close,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_caffe2 (void)
{
  NNS_support_caffe2.name = filter_subplugin_caffe2;
  NNS_support_caffe2.allow_in_place = FALSE;      /** @todo: support this to optimize performance later. */
  NNS_support_caffe2.allocate_in_invoke = TRUE;
  NNS_support_caffe2.run_without_model = FALSE;
  NNS_support_caffe2.verify_model_path = FALSE;
  NNS_support_caffe2.invoke_NN = caffe2_run;
  NNS_support_caffe2.getInputDimension = caffe2_getInputDim;
  NNS_support_caffe2.getOutputDimension = caffe2_getOutputDim;
  NNS_support_caffe2.destroyNotify = caffe2_destroyNotify;
  NNS_support_caffe2.checkAvailability = caffe2_checkAvailability;

  nnstreamer_filter_probe (&NNS_support_caffe2);
}

/** @brief Destruct the subplugin */
void
fini_filter_caffe2 (void)
{
  nnstreamer_filter_exit (NNS_support_caffe2.name);
}
