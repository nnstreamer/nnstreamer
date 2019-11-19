/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file   tensor_filter_caffe2_core.cc
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @date   31/5/2019
 * @brief  connection with caffe2 libraries.
 *
 * @bug     No known bugs.
 */

#include <unistd.h>
#include <algorithm>

#include <nnstreamer_plugin_api.h>
#include "tensor_filter_caffe2_core.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

std::map <char*, Tensor*> Caffe2Core::inputTensorMap;

/**
 * @brief	Caffe2Core creator
 * @param	_model_path	: the logical path to '{model_name}.tffile' file
 * @note	the model of _model_path will be loaded simultaneously
 * @return	Nothing
 */
Caffe2Core::Caffe2Core (const char * _model_path, const char *_model_path_sub)
{
  g_assert (_model_path != NULL && _model_path_sub != NULL);
  pred_model_path = g_strdup (_model_path);
  init_model_path = g_strdup (_model_path_sub);
  first_run = true;

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);
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
    g_critical ("Failed to load model\n");
    return -1;
  }

  gst_tensors_info_copy (&inputTensorMeta, &prop->input_meta);
  gst_tensors_info_copy (&outputTensorMeta, &prop->output_meta);

  if (initInputTensor ()) {
    g_critical ("Failed to initialize input tensor\n");
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
        g_critical ("invalid data type is used");
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
        g_critical ("invalid data type is used");
        return -1;
      default:
        g_critical ("invalid data type is used");
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
    g_critical ("the file of init_model_path is not valid: %s\n", init_model_path);
    return -1;
  }
  if (!g_file_test (pred_model_path, G_FILE_TEST_IS_REGULAR)) {
    g_critical ("the file of pred_model_path is not valid: %s\n", pred_model_path);
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
  int i;
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
        g_critical ("invalid data type is used");
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
        g_critical ("invalid data type is used");
        return -1;
      default:
        g_critical ("invalid data type is used");
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
      g_critical ("Runtime error while running the model: %s", re.what());
      return -4;
    } catch(const std::exception& ex)	{
      g_critical ("Exception while running the model : %s", ex.what());
      return -4;
    } catch (...) {
      g_critical ("Unknown exception while running the model");
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
        g_critical ("invalid data type (uint32) is used");
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
        g_critical ("invalid data type (uint64) is used");
        return -1;
      default:
        g_critical ("invalid data type is used");
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
 * @brief	call the creator of Caffe2Core class.
 * @param	_model_path	: the logical path to '{model_name}.tffile' file
 * @return	Caffe2Core class
 */
void *
caffe2_core_new (const char *_model_path, const char *_model_path_sub)
{
  return new Caffe2Core (_model_path, _model_path_sub);
}

/**
 * @brief	delete the Caffe2Core class.
 * @param	caffe2	: the class object
 * @return	Nothing
 */
void
caffe2_core_delete (void * caffe2)
{
  Caffe2Core *c = (Caffe2Core *) caffe2;
  delete c;
}

/**
 * @brief	initialize the object with caffe2 model
 * @param	caffe2	: the class object
 * @return 0 if OK. non-zero if error.
 */
int
caffe2_core_init (void * caffe2, const GstTensorFilterProperties * prop)
{
  Caffe2Core *c = (Caffe2Core *) caffe2;
  return c->init (prop);
}

/**
 * @brief	get the model path
 * @param	caffe2	: the class object
 * @return the model path.
 */
const char *
caffe2_core_getInitModelPath (void * caffe2)
{
  Caffe2Core *c = (Caffe2Core *) caffe2;
  return c->getInitModelPath ();
}

/**
 * @brief	get the model path
 * @param	caffe2	: the class object
 * @return the model path.
 */
const char *
caffe2_core_getPredModelPath (void * caffe2)
{
  Caffe2Core *c = (Caffe2Core *) caffe2;
  return c->getPredModelPath ();
}

/**
 * @brief	get the Dimension of Input Tensor of model
 * @param	caffe2	: the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
caffe2_core_getInputDim (void * caffe2, GstTensorsInfo * info)
{
  Caffe2Core *c = (Caffe2Core *) caffe2;
  return c->getInputTensorDim (info);
}

/**
 * @brief	get the Dimension of Output Tensor of model
 * @param	caffe2	: the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
caffe2_core_getOutputDim (void * caffe2, GstTensorsInfo * info)
{
  Caffe2Core *c = (Caffe2Core *) caffe2;
  return c->getOutputTensorDim (info);
}

/**
 * @brief	run the model
 * @param	caffe2	: the class object
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
caffe2_core_run (void * caffe2, const GstTensorMemory * input,
    GstTensorMemory * output)
{
  Caffe2Core *c = (Caffe2Core *) caffe2;
  return c->run (input, output);
}

/**
 * @brief	the destroy notify method for caffe2. it will free the output tensor
 * @param[in] data : the data element destroyed at the pipeline
 */
void
caffe2_core_destroyNotify (void * data)
{
  /* do nothing */
}
