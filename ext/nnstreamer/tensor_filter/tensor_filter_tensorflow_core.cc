/**
 * Copyright (C) 2018 Samsung Electronics Co., Ltd. All rights reserved.
 * Copyright (C) 2018 HyoungJoo Ahn <hello.ahn@samsung.com>
 * Copyright (C) 2018 Jijoong Moon <jjioong.moon@samsung.com>
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
 * @file   tensor_filter_tensorflow_core.cc
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @date   08/02/2018
 * @brief  connection with tensorflow libraries.
 *
 * @bug     No known bugs.
 */

#include <nnstreamer_plugin_api.h>
#include "tensor_filter_tensorflow_core.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

std::map <void*, Tensor> TFCore::outputTensorMap;

/**
 * @brief	TFCore creator
 * @param	_model_path	: the logical path to '{model_name}.pb' file
 * @note	the model of _model_path will be loaded simultaneously
 * @return	Nothing
 */
TFCore::TFCore (const char * _model_path)
{
  model_path = _model_path;
  configured = false;

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);
}

/**
 * @brief	TFCore Destructor
 * @return	Nothing
 */
TFCore::~TFCore ()
{
  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);
}

/**
 * @brief	initialize the object with tensorflow model
 * @return 0 if OK. non-zero if error.
 */
int
TFCore::init (const GstTensorFilterProperties * prop,
  const gboolean tf_mem_optmz)
{
  gst_tensors_info_copy (&inputTensorMeta, &prop->input_meta);
  gst_tensors_info_copy (&outputTensorMeta, &prop->output_meta);

  mem_optmz = tf_mem_optmz;

  return loadModel ();
}

/**
 * @brief	get the model path
 * @return the model path.
 */
const char *
TFCore::getModelPath ()
{
  return model_path;
}

/**
 * @brief	load the tf model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 *        -1 if the modelfile is not valid(or not exist).
 *        -2 if the pb file is not loaded.
 *        -3 if the input properties are different with model.
 *        -4 if the Tensorflow session is not initialized.
 *        -5 if the Tensorflow session is not created.
 */
int
TFCore::loadModel ()
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  Status status;
  GraphDef graph_def;

  if (!g_file_test (model_path, G_FILE_TEST_IS_REGULAR)) {
    g_critical ("the file of model_path (%s) is not valid (not regular)\n", model_path);
    return -1;
  }
  status = ReadBinaryProto (Env::Default (), model_path, &graph_def);
  if (!status.ok()) {
    g_critical ("Failed to read graph.\n%s", status.ToString().c_str());
    return -2;
  }

  /* validate input tensor */
  if (validateInputTensor (graph_def)) {
    g_critical ("Input Tensor Information is not valid");
    return -3;
  }

  /* get session */
  status = NewSession (SessionOptions (), &session);
  if (!status.ok()) {
    g_critical ("Failed to init new session.\n%s", status.ToString().c_str());
    return -4;
  }

  status = session->Create (graph_def);
  if (!status.ok()) {
    g_critical ("Failed to create session.\n%s", status.ToString().c_str());
    return -5;
  }

  /* prepare output tensor */
  for (int i = 0; i < outputTensorMeta.num_tensors; ++i) {
    output_tensor_names.push_back (outputTensorMeta.info[i].name);
  }

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Model is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  return 0;
}

/**
 * @brief	return the data type of the tensor
 * @param tfType	: the defined type of Tensorflow
 * @return the enum of defined _NNS_TYPE
 */
tensor_type
TFCore::getTensorTypeFromTF (DataType tfType)
{
  switch (tfType) {
    case DT_INT32:
      return _NNS_INT32;
    case DT_UINT32:
      return _NNS_UINT32;
    case DT_INT16:
      return _NNS_INT16;
    case DT_UINT16:
      return _NNS_UINT16;
    case DT_INT8:
      return _NNS_INT8;
    case DT_UINT8:
      return _NNS_UINT8;
    case DT_INT64:
      return _NNS_INT64;
    case DT_UINT64:
      return _NNS_UINT64;
    case DT_FLOAT:
      return _NNS_FLOAT32;
    case DT_DOUBLE:
      return _NNS_FLOAT64;
    default:
      /** @todo Support other types */
      break;
  }

  return _NNS_END;
}

/**
 * @brief	return the data type of the tensor for Tensorflow
 * @param[in] tType	: the defined type of NNStreamer
 * @param[out] tf_type : the result type in TF_DataType
 * @return the result of type converting.
 */
gboolean
TFCore::getTensorTypeToTF_Capi (tensor_type tType, TF_DataType * tf_type)
{
  switch (tType) {
    case _NNS_INT32:
      *tf_type = TF_INT32;
      break;
    case _NNS_UINT32:
      *tf_type = TF_UINT32;
      break;
    case _NNS_INT16:
      *tf_type = TF_INT16;
      break;
    case _NNS_UINT16:
      *tf_type = TF_UINT16;
      break;
    case _NNS_INT8:
      *tf_type = TF_INT8;
      break;
    case _NNS_UINT8:
      *tf_type = TF_UINT8;
      break;
    case _NNS_INT64:
      *tf_type = TF_INT64;
      break;
    case _NNS_UINT64:
      *tf_type = TF_UINT64;
      break;
    case _NNS_FLOAT32:
      *tf_type = TF_FLOAT;
      break;
    case _NNS_FLOAT64:
      *tf_type = TF_DOUBLE;
      break;
    default:
      return FALSE;
  }
  return TRUE;
}

/**
 * @brief	check the inserted information about input tensor with model
 * @return 0 if OK. non-zero if error.
 *        -1 if the number of input tensors is not matched.
 *        -2 if the name of input tensors is not matched.
 *        -3 if the type of input tensors is not matched.
 *        -4 if the dimension of input tensors is not matched.
 *        -5 if the rank of input tensors exceeds our capacity NNS_TENSOR_RANK_LIMIT.
 */
int
TFCore::validateInputTensor (const GraphDef &graph_def)
{
  std::vector <const NodeDef*> placeholders;
  int length;

  for (const NodeDef& node : graph_def.node ()) {
    if (node.op () == "Placeholder") {
      placeholders.push_back (&node);
    }
  }

  if (placeholders.empty ()) {
    GST_WARNING ("No inputs spotted.");
    /* do nothing? */
    return 0;
  }

  length = placeholders.size ();
  GST_INFO ("Found possible inputs: %d", length);

  if (inputTensorMeta.num_tensors != length) {
    GST_ERROR ("Input Tensor is not valid: the number of input tensor is different\n");
    return -1;
  }

  for (int i = 0; i < length; ++i) {
    const NodeDef* node = placeholders[i];
    string shape_description = "None";
    if (node->attr ().count ("shape")) {
      TensorShapeProto shape_proto = node->attr ().at ("shape").shape ();
      Status shape_status = PartialTensorShape::IsValidShape (shape_proto);
      if (shape_status.ok ()) {
        shape_description = PartialTensorShape (shape_proto).DebugString ();
      } else {
        shape_description = shape_status.error_message ();
      }
    }
    char chars[] = "[]";
    for (unsigned int j = 0; j < strlen (chars); ++j)
    {
      shape_description.erase (
        std::remove (
          shape_description.begin (),
          shape_description.end (),
          chars[j]
        ),
        shape_description.end ()
      );
    }

    DataType dtype = DT_INVALID;
    char *tensor_name = inputTensorMeta.info[i].name;

    if (node->attr ().count ("dtype")) {
      dtype = node->attr ().at ("dtype").type ();
    }

    if (!tensor_name || !g_str_equal (tensor_name, node->name ().c_str ())) {
      GST_ERROR ("Input Tensor is not valid: the name of input tensor is different\n");
      return -2;
    }

    if (inputTensorMeta.info[i].type != getTensorTypeFromTF (dtype)) {
      /* consider the input data as bytes if tensor type is string */
      if (dtype == DT_STRING) {
        GST_WARNING ("Input type is string, ignore type comparision.");
      } else {
        GST_ERROR ("Input Tensor is not valid: the type of input tensor is different\n");
        return -3;
      }
    }

    gchar **str_dims;
    guint rank, dim;
    TensorShape ts = TensorShape ({});

    str_dims = g_strsplit (shape_description.c_str (), ",", -1);
    rank = g_strv_length (str_dims);

    if (rank > NNS_TENSOR_RANK_LIMIT) {
      GST_ERROR ("The Rank of Input Tensor is not affordable. It's over our capacity.\n");
      g_strfreev (str_dims);
      return -5;
    }

    for (int j = 0; j < rank; ++j) {
      dim = inputTensorMeta.info[i].dimension[rank - j - 1];
      ts.AddDim (dim);

      if (g_str_equal (str_dims[j], "?"))
        continue;

      if (dim != (guint) g_ascii_strtoull (str_dims[j], NULL, 10)) {
        GST_ERROR ("Input Tensor is not valid: the dim of input tensor is different\n");
        g_strfreev (str_dims);
        return -4;
      }
    }
    g_strfreev (str_dims);

    /* add input tensor info */
    tf_tensor_info_s info_s = { dtype, ts };
    input_tensor_info.push_back (info_s);
  }
  return 0;
}

/**
 * @brief	check the inserted information about output tensor with model
 * @return 0 if OK. non-zero if error.
 *        -1 if the number of output tensors is not matched.
 *        -2 if the dimension of output tensors is not matched.
 *        -3 if the type of output tensors is not matched.
 */
int
TFCore::validateOutputTensor (const std::vector <Tensor> &outputs)
{
  if (outputTensorMeta.num_tensors != outputs.size()) {
    GST_ERROR ("Invalid output meta: different size");
    return -1;
  }

  for (int i = 0; i < outputTensorMeta.num_tensors; ++i) {
    tensor_type otype;
    gsize num;

    otype = getTensorTypeFromTF (outputs[i].dtype());
    num = gst_tensor_get_element_count (outputTensorMeta.info[i].dimension);

    if (num != outputs[i].NumElements()) {
      GST_ERROR ("Invalid output meta: different element count");
      return -2;
    }

    if (outputTensorMeta.info[i].type != otype) {
      GST_ERROR ("Invalid output meta: different type");
      return -3;
    }
  }

  configured = true;
  return 0;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
TFCore::getInputTensorDim (GstTensorsInfo * info)
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
TFCore::getOutputTensorDim (GstTensorsInfo * info)
{
  gst_tensors_info_copy (info, &outputTensorMeta);
  return 0;
}

/**
 * @brief	ring cache structure
 */
class TFBuffer : public TensorBuffer {
 public:
  void* data_;
  size_t len_;

#if (TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION < 13)
  explicit TFBuffer (void* data_ptr) : data_(data_ptr) {}
  void* data () const override { return data_; }
#elif (TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 13)
  explicit TFBuffer (void* data_ptr) : TensorBuffer (data_ptr) {}
#else
#error This supports Tensorflow 1.x only.
#endif

  size_t size () const override { return len_; }
  TensorBuffer* root_buffer () override { return this; }
  void FillAllocationDescription (AllocationDescription* proto) const override {
    int64 rb = size ();
    proto->set_requested_bytes (rb);
    proto->set_allocator_name (cpu_allocator ()->Name ());
  }

  /* Prevents input forwarding from mutating this buffer. */
  bool OwnsMemory () const override { return false; }
};

/**
 * @brief	run the model with the input.
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 *         -1 if the model does not work properly.
 *         -2 if the output properties are different with model.
 */
int
TFCore::run (const GstTensorMemory * input, GstTensorMemory * output)
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  std::vector <std::pair <string, Tensor>> input_feeds;
  std::vector <Tensor> outputs;

  for (int i = 0; i < inputTensorMeta.num_tensors; ++i) {
    Tensor in;

    /* If the datatype is STRING, it should be handled in specific process */
    if (input_tensor_info[i].type == DT_STRING) {
      in = Tensor (input_tensor_info[i].type, input_tensor_info[i].shape);
      in.scalar<string>()() = string ((char *) input[i].data, input[i].size);
    } else {
      if (mem_optmz) {
        TFBuffer *buf;
        TF_DataType dataType;

        if (!getTensorTypeToTF_Capi (input[i].type, &dataType)){
          g_critical ("This data type is not valid: %d", input[i].type);
          return -1;
        }

        /* this input tensor should be UNREF */
        buf = new TFBuffer (input[i].data);
        buf->len_ = input[i].size;

        in = TensorCApi::MakeTensor (
          dataType,
          input_tensor_info[i].shape,
          buf
        );

        buf->Unref();

        if (!in.IsAligned ()) {
          g_critical ("the input tensor %s is not aligned", inputTensorMeta.info[i].name);
          return -2;
        }
      } else {
        in = Tensor (input_tensor_info[i].type, input_tensor_info[i].shape);
        /* copy data */
        std::copy_n ((char *) input[i].data, input[i].size,
            const_cast<char *>(in.tensor_data().data()));
      }
    }
    input_feeds.push_back ({inputTensorMeta.info[i].name, in});
  }

  Status run_status =
      session->Run (input_feeds, output_tensor_names, {}, &outputs);

  if (!run_status.ok()) {
    g_critical ("Failed to run model: %s\n", run_status.ToString().c_str());
    return -1;
  }

  /* validate output tensor once */
  if (!configured && validateOutputTensor (outputs)) {
    g_critical ("Output Tensor Information is not valid");
    return -2;
  }

  for (int i = 0; i < outputTensorMeta.num_tensors; ++i) {
    /**
     * @todo support DT_STRING output tensor
     */
    output[i].data = const_cast<char *>(outputs[i].tensor_data().data());
    outputTensorMap.insert (std::make_pair (output[i].data, outputs[i]));
  }

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Invoke() is finished: %" G_GINT64_FORMAT,
      (stop_time - start_time));
#endif

  return 0;
}

void *
tf_core_new (const char * _model_path)
{
  return new TFCore (_model_path);
}

/**
 * @brief	delete the TFCore class.
 * @param	tf	: the class object
 * @return	Nothing
 */
void
tf_core_delete (void * tf)
{
  TFCore *c = (TFCore *) tf;
  delete c;
}

/**
 * @brief	initialize the object with tf model
 * @param	tf	: the class object
 * @return 0 if OK. non-zero if error.
 */
int
tf_core_init (void * tf, const GstTensorFilterProperties * prop,
  const gboolean tf_mem_optmz)
{
  TFCore *c = (TFCore *) tf;
  return c->init (prop, tf_mem_optmz);
}

/**
 * @brief	get model path
 * @param	tf	: the class object
 * @return	model path
 */
const char *
tf_core_getModelPath (void * tf)
{
  TFCore *c = (TFCore *) tf;
  return c->getModelPath ();
}

/**
 * @brief	get the Dimension of Input Tensor of model
 * @param	tf	the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
tf_core_getInputDim (void * tf, GstTensorsInfo * info)
{
  TFCore *c = (TFCore *) tf;
  return c->getInputTensorDim (info);
}

/**
 * @brief	get the Dimension of Output Tensor of model
 * @param	tf	the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
tf_core_getOutputDim (void * tf, GstTensorsInfo * info)
{
  TFCore *c = (TFCore *) tf;
  return c->getOutputTensorDim (info);
}

/**
 * @brief	run the model
 * @param	tf	: the class object
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
tf_core_run (void * tf, const GstTensorMemory * input, GstTensorMemory * output)
{
  TFCore *c = (TFCore *) tf;
  return c->run (input, output);
}

/**
 * @brief	the destroy notify method for tensorflow. it will free the output tensor
 * @param[in] data : the data element destroyed at the pipeline
 */
void
tf_core_destroyNotify (void * data)
{
  TFCore::outputTensorMap.erase (data);
}
