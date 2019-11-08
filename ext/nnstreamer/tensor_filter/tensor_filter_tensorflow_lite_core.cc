/**
 * Copyright (C) 2018 Samsung Electronics Co., Ltd. All rights reserved.
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
 * @file   tensor_filter_tensorflow_lite_core.cc
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @date   7/5/2018
 * @brief  connection with tflite libraries.
 *
 * @bug     No known bugs.
 */

#include <unistd.h>
#include <limits.h>
#include <algorithm>

#include <nnstreamer_plugin_api.h>
#include <nnstreamer_conf.h>
#include "tensor_filter_tensorflow_lite_core.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif


/**
 * @brief	TFLiteCore creator
 * @param	_model_path	: the logical path to '{model_name}.tffile' file
 * @note	the model of _model_path will be loaded simultaneously
 * @return	Nothing
 */
TFLiteCore::TFLiteCore (const char * _model_path, nnapi_hw hw)
{
  g_assert (_model_path != NULL);
  model_path = g_strdup (_model_path);
  interpreter = nullptr;
  model = nullptr;

  if (hw == NNAPI_UNKNOWN) {
    use_nnapi = nnsconf_get_custom_value_bool ("tensorflowlite", "enable_nnapi", FALSE);
  } else {
    use_nnapi = TRUE;
  }
  accel = hw;

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);
}

/**
 * @brief	TFLiteCore Destructor
 * @return	Nothing
 */
TFLiteCore::~TFLiteCore ()
{
  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);
}

/**
 * @brief	initialize the object with tflite model
 * @return 0 if OK. non-zero if error.
 *        -1 if the model is not loaded.
 *        -2 if the initialization of input tensor is failed.
 *        -3 if the initialization of output tensor is failed.
 */
int
TFLiteCore::init ()
{
  if (loadModel ()) {
    g_critical ("Failed to load model\n");
    return -1;
  }
  if (setInputTensorProp ()) {
    g_critical ("Failed to initialize input tensor\n");
    return -2;
  }
  if (setOutputTensorProp ()) {
    g_critical ("Failed to initialize output tensor\n");
    return -3;
  }
  return 0;
}

/**
 * @brief	get the model path
 * @return the model path.
 */
const char *
TFLiteCore::getModelPath ()
{
  return model_path;
}

/**
 * @brief	load the tflite model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::loadModel ()
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  if (!interpreter) {
    if (!g_file_test (model_path, G_FILE_TEST_IS_REGULAR)) {
      g_critical ("the file of model_path (%s) is not valid (not regular)\n", model_path);
      return -1;
    }
    model =
        std::unique_ptr <tflite::FlatBufferModel>
        (tflite::FlatBufferModel::BuildFromFile (model_path));
    if (!model) {
      g_critical ("Failed to mmap model\n");
      return -1;
    }
    /* If got any trouble at model, active below code. It'll be help to analyze. */
    /* model->error_reporter (); */

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder (*model, resolver) (&interpreter);
    if (!interpreter) {
      g_critical ("Failed to construct interpreter\n");
      return -2;
    }

    interpreter->UseNNAPI(use_nnapi);

#ifdef ENABLE_TFLITE_NNAPI_DELEGATE
    if (use_nnapi) {
      nnfw_delegate.reset (new ::nnfw::tflite::NNAPIDelegate);
      if (nnfw_delegate->BuildGraph (interpreter.get()) != kTfLiteOk) {
        g_critical ("Fail to BuildGraph");
        return -3;
      }
    }
#endif

    /** set allocation type to dynamic for in/out tensors */
    int tensor_idx;

    int tensorSize = interpreter->inputs ().size ();
    for (int i = 0; i < tensorSize; ++i) {
      tensor_idx = interpreter->inputs ()[i];
      interpreter->tensor (tensor_idx)->allocation_type = kTfLiteDynamic;
    }

    tensorSize = interpreter->outputs ().size ();
    for (int i = 0; i < tensorSize; ++i) {
      tensor_idx = interpreter->outputs ()[i];
      interpreter->tensor (tensor_idx)->allocation_type = kTfLiteDynamic;
    }

    if (interpreter->AllocateTensors () != kTfLiteOk) {
      g_critical ("Failed to allocate tensors\n");
      return -2;
    }
  }
#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Model is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  return 0;
}

/**
 * @brief	return the data type of the tensor
 * @param tfType	: the defined type of Tensorflow Lite
 * @return the enum of defined _NNS_TYPE
 */
tensor_type
TFLiteCore::getTensorType (TfLiteType tfType)
{
  switch (tfType) {
    case kTfLiteFloat32:
      return _NNS_FLOAT32;
    case kTfLiteUInt8:
      return _NNS_UINT8;
    case kTfLiteInt32:
      return _NNS_INT32;
    case kTfLiteBool:
      return _NNS_INT8;
    case kTfLiteInt64:
      return _NNS_INT64;
    case kTfLiteString:
    default:
      /** @todo Support other types */
      break;
  }

  return _NNS_END;
}

/**
 * @brief extract and store the information of given tensor list
 * @param tensor_idx_list list of index of tensors in tflite interpreter
 * @param[out] tensorMeta tensors to set the info into
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::setTensorProp (const std::vector<int> &tensor_idx_list,
    GstTensorsInfo * tensorMeta)
{
  tensorMeta->num_tensors = tensor_idx_list.size ();

  for (int i = 0; i < tensorMeta->num_tensors; ++i) {
    if (getTensorDim (tensor_idx_list[i], tensorMeta->info[i].dimension)) {
      g_critical ("failed to get the dimension of input tensors");
      return -1;
    }
    tensorMeta->info[i].type =
        getTensorType (interpreter->tensor (tensor_idx_list[i])->type);

#if (DBG)
    gchar *dim_str =
        gst_tensor_get_dimension_string (tensorMeta->info[i].dimension);
    g_message ("tensorMeta[%d] >> type:%d, dim[%s]",
        i, tensorMeta->info[i].type, dim_str);
    g_free (dim_str);
#endif
  }
  return 0;
}

/**
 * @brief extract and store the information of input tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::setInputTensorProp ()
{
  return setTensorProp (interpreter->inputs (), &inputTensorMeta);
}

/**
 * @brief extract and store the information of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::setOutputTensorProp ()
{
  return setTensorProp (interpreter->outputs (), &outputTensorMeta);
}

/**
 * @brief	return the Dimension of Tensor.
 * @param tensor_idx	: the real index of model of the tensor
 * @param[out] dim	: the array of the tensor
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getTensorDim (int tensor_idx, tensor_dim dim)
{
  TfLiteIntArray *tensor_dims = interpreter->tensor (tensor_idx)->dims;
  int len = tensor_dims->size;
  g_assert (len <= NNS_TENSOR_RANK_LIMIT);

  /* the order of dimension is reversed at CAPS negotiation */
  std::reverse_copy (tensor_dims->data, tensor_dims->data + len, dim);

  /* fill the remnants with 1 */
  for (int i = len; i < NNS_TENSOR_RANK_LIMIT; ++i) {
    dim[i] = 1;
  }

  return 0;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getInputTensorDim (GstTensorsInfo * info)
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
TFLiteCore::getOutputTensorDim (GstTensorsInfo * info)
{
  gst_tensors_info_copy (info, &outputTensorMeta);
  return 0;
}

/**
 * @brief set the Dimension for Input Tensor.
 * @param info Structure for input tensor info.
 * @return 0 if OK. non-zero if error.
 * @note rank can be changed dependent on the model
 */
int
TFLiteCore::setInputTensorDim (const GstTensorsInfo * info)
{
  TfLiteStatus status;
  const std::vector<int> &input_idx_list = interpreter->inputs ();

  /** Cannot change the number of inputs */
  if (info->num_tensors != input_idx_list.size ())
    return -EINVAL;

  for (int tensor_idx = 0; tensor_idx < info->num_tensors; ++tensor_idx) {
    tensor_type tf_type;
    const GstTensorInfo *tensor_info;
    int input_rank;

    tensor_info = &info->info[tensor_idx];

    /** cannot change the type of input */
    tf_type = getTensorType (
        interpreter->tensor (input_idx_list[tensor_idx])->type);
    if (tf_type != tensor_info->type)
      return -EINVAL;

    /**
     * Given that the rank intended by the user cannot be exactly determined,
     * iterate over all possible ranks starting from MAX rank to the actual rank
     * of the dimension array. In case of none of these ranks work, return error
     */
    input_rank = gst_tensor_info_get_rank (&info->info[tensor_idx]);
    for (int rank = NNS_TENSOR_RANK_LIMIT; rank >= input_rank; rank--) {
			std::vector<int> dims(rank);
      /* the order of dimension is reversed at CAPS negotiation */
      for (int idx = 0; idx < rank; ++idx) {
        /** check overflow when storing uint32_t in int container */
        if (tensor_info->dimension[rank - idx - 1] > INT_MAX)
          return -ERANGE;
        dims[idx] = tensor_info->dimension[rank - idx - 1];
      }

      status = interpreter->ResizeInputTensor(input_idx_list[tensor_idx], dims);
      if (status != kTfLiteOk)
        continue;

      break;
    }

    /** return error when none of the ranks worked */
    if (status != kTfLiteOk)
      return -EINVAL;

  }

  status = interpreter->AllocateTensors();
  if (status != kTfLiteOk)
    return -EINVAL;

  return 0;
}

/**
 * @brief	run the model with the input.
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::invoke (const GstTensorMemory * input, GstTensorMemory * output)
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  std::vector <int> tensors_idx;
  int tensor_idx;
  TfLiteTensor *tensor_ptr;
  TfLiteStatus status;

  for (int i = 0; i < outputTensorMeta.num_tensors; ++i) {
    tensor_idx = interpreter->outputs ()[i];
    tensor_ptr = interpreter->tensor (tensor_idx);

    g_assert (tensor_ptr->bytes == output[i].size);
    tensor_ptr->data.raw = (char *) output[i].data;
    tensors_idx.push_back (tensor_idx);
  }

  for (int i = 0; i < inputTensorMeta.num_tensors; ++i) {
    tensor_idx = interpreter->inputs ()[i];
    tensor_ptr = interpreter->tensor (tensor_idx);

    g_assert (tensor_ptr->bytes == input[i].size);
    tensor_ptr->data.raw = (char *) input[i].data;
    tensors_idx.push_back (tensor_idx);
  }

#ifdef ENABLE_TFLITE_NNAPI_DELEGATE
  if (use_nnapi)
    status = nnfw_delegate->Invoke (interpreter.get());
  else
#endif
    status = interpreter->Invoke ();

  /** if it is not `nullptr`, tensorflow makes `free()` the memory itself. */
  int tensorSize = tensors_idx.size ();
  for (int i = 0; i < tensorSize; ++i) {
    interpreter->tensor (tensors_idx[i])->data.raw = nullptr;
  }

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Invoke() is finished: %" G_GINT64_FORMAT,
      (stop_time - start_time));
#endif

  if (status != kTfLiteOk) {
    g_critical ("Failed to invoke");
    return -1;
  }

  return 0;
}

/**
 * @brief	call the creator of TFLiteCore class.
 * @param	_model_path	: the logical path to '{model_name}.tffile' file
 * @return	TFLiteCore class
 */
void *
tflite_core_new (const char * _model_path, nnapi_hw hw)
{
  return new TFLiteCore (_model_path, hw);
}

/**
 * @brief	delete the TFLiteCore class.
 * @param	tflite	: the class object
 * @return	Nothing
 */
void
tflite_core_delete (void * tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  delete c;
}

/**
 * @brief	initialize the object with tflite model
 * @param	tflite	: the class object
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_init (void * tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->init ();
}

/**
 * @brief	get the model path
 * @param	tflite	: the class object
 * @return the model path.
 */
const char *
tflite_core_getModelPath (void * tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getModelPath ();
}

/**
 * @brief	get the Dimension of Input Tensor of model
 * @param	tflite	: the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_getInputDim (void * tflite, GstTensorsInfo * info)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getInputTensorDim (info);
}

/**
 * @brief	get the Dimension of Output Tensor of model
 * @param	tflite	: the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_getOutputDim (void * tflite, GstTensorsInfo * info)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getOutputTensorDim (info);
}

/**
 * @brief set the Dimension of Input Tensor of model
 * @param tflite the class object
 * @param in_info Structure for Input tensor info.
 * @param[out] out_info Structure for Output tensor info.
 * @return 0 if OK. non-zero if error.
 * @detail Output Tensor info is recalculated based on the set Input Tensor Info
 */
int
tflite_core_setInputDim (void * tflite, const GstTensorsInfo * in_info,
    GstTensorsInfo * out_info)
{
  int status;
  TFLiteCore *c = (TFLiteCore *) tflite;
  GstTensorsInfo cur_in_info;

  /** get current input tensor info for resetting */
  status = c->getInputTensorDim (&cur_in_info);
  if (status != 0)
    return status;

  /** set new input tensor info */
  status = c->setInputTensorDim (in_info);
  if (status != 0) {
    g_assert (c->setInputTensorDim (&cur_in_info) == 0);
    return status;
  }

  /** update input tensor info */
  if ((status = c->setInputTensorProp ()) != 0) {
    g_assert (c->setInputTensorDim (&cur_in_info) == 0);
    g_assert (c->setInputTensorProp () == 0);
    return status;
  }

  /** update output tensor info */
  if ((status = c->setOutputTensorProp ()) != 0) {
    g_assert (c->setInputTensorDim (&cur_in_info) == 0);
    g_assert (c->setInputTensorProp () == 0);
    g_assert (c->setOutputTensorProp () == 0);
    return status;
  }

  /** get output tensor info to be returned */
  status = c->getOutputTensorDim (out_info);
  if (status != 0) {
    g_assert (c->setInputTensorDim (&cur_in_info) == 0);
    g_assert (c->setInputTensorProp () == 0);
    g_assert (c->setOutputTensorProp () == 0);
    return status;
  }

  return 0;
}

/**
 * @brief	invoke the model
 * @param	tflite	: the class object
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_invoke (void * tflite, const GstTensorMemory * input,
    GstTensorMemory * output)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->invoke (input, output);
}
