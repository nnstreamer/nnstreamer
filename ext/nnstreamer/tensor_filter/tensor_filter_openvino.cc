/**
 * GStreamer Tensor_Filter, OpenVino (DLDT) Module
 * Copyright (C) 2019 Wook Song <wook16.song@samsung.com>
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
 * @file    tensor_filter_openvino.cc
 * @date    23 Dec 2019
 * @brief   Tensor_filter subplugin for OpenVino (DLDT).
 * @see     http://github.com/nnsuite/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (OpenVino) for tensor_filter.
 */

#include <glib.h>
#include <nnstreamer_plugin_api_filter.h>
#include <tensor_common.h>
#ifdef __OPENVINO_CPU_EXT__
#include <ext_list.hpp>
#endif /* __OPENVINO_CPU_EXT__ */
#include <inference_engine.hpp>
#include <iostream>
#include <string>
#include <vector>

const gchar *openvino_accl_support[] = {
  ACCL_CPU_STR,
  NULL
};

void init_filter_openvino (void) __attribute__ ((constructor));
void fini_filter_openvino (void) __attribute__ ((destructor));

class TensorFilterOpenvino
{
public:
  enum RetVal
  {
    RetSuccess = 0,
    RetEBusy = -EBUSY,
    RetEInval = -EINVAL,
    RetENoDev = -ENODEV,
    RetEOverFlow = -EOVERFLOW,
  };

  static tensor_type convertFromIETypeStr (std::string type);
  static InferenceEngine::Blob::Ptr convertGstTensorMemoryToBlobPtr (
      const InferenceEngine::TensorDesc tensorDesc,
      const GstTensorMemory * gstTensor);

  TensorFilterOpenvino (std::string path_model_xml, std::string path_model_bin);
  ~TensorFilterOpenvino ();

  // TODO: Need to support other acceleration devices
  int loadModel ();

  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int invoke (const GstTensorFilterProperties * prop,
      const GstTensorMemory * input, GstTensorMemory * output);
  std::string getPathModelXml ();
  std::string getPathModelBin ();

  static const std::string extBin;
  static const std::string extXml;

private:
  TensorFilterOpenvino ();

  InferenceEngine::Core _ieCore;
  InferenceEngine::CNNNetReader _networkReaderCNN;
  InferenceEngine::CNNNetwork _networkCNN;
  InferenceEngine::InputsDataMap _inputsDataMap;
  InferenceEngine::TensorDesc _inputTensorDescs[NNS_TENSOR_SIZE_LIMIT];
  InferenceEngine::OutputsDataMap _outputsDataMap;
  InferenceEngine::TensorDesc _outputTensorDescs[NNS_TENSOR_SIZE_LIMIT];
  InferenceEngine::ExecutableNetwork _executableNet;
  InferenceEngine::InferRequest _inferRequest;

  std::string pathModelXml;
  std::string pathModelBin;
  bool isLoaded;
  accl_hw hw;
};

const std::string TensorFilterOpenvino::extBin = ".bin";
const std::string TensorFilterOpenvino::extXml = ".xml";

/**
 * @brief Convert the string representing the tensor data type to _nns_tensor_type
 * @param type a std::string representing the tensor data type in InferenceEngine
 * @return _nns_tensor_type corresponding to the tensor data type in InferenceEngine if OK, otherwise _NNS_END
 */
tensor_type
TensorFilterOpenvino::convertFromIETypeStr (std::string type)
{
  if (type[0] == 'U') {
    if (type[1] == '8')
      return _NNS_UINT8;
    else
      return _NNS_UINT16;
  } else if (type[0] == 'I') {
    if (type[1] == '1')
      return _NNS_INT16;
    else if (type[1] == '3')
      return _NNS_INT32;
    else
      return _NNS_INT8;
  } else if (type[0] == 'F') {
    if (type[2] == '3')
      return _NNS_FLOAT32;
    else
      return _NNS_END;
  } else {
    return _NNS_END;
  }
}

/**
 * @brief Convert a tensor container in NNS to a tensor container in IE
 * @param tensorDesc the class that defines a Tensor description to be converted from a GstTensorMemory
 * @param gstTensor the container of a tensor in NNS to be coverted to a tensor container in IE
 * @return a pointer to the Blob which is a container of a tensor in IE if OK, otherwise nullptr
 */
InferenceEngine::Blob::Ptr
TensorFilterOpenvino::convertGstTensorMemoryToBlobPtr (
    const InferenceEngine::TensorDesc tensorDesc,
    const GstTensorMemory * gstTensor)
{
  switch (gstTensor->type) {
    case _NNS_UINT8:
      return InferenceEngine::Blob::Ptr (
          new InferenceEngine::TBlob<uint8_t>(
              tensorDesc,
              (uint8_t *) gstTensor->data, gstTensor->size
              )
          );
    case _NNS_UINT16:
      return InferenceEngine::Blob::Ptr (
          new InferenceEngine::TBlob<uint16_t>(
              tensorDesc,
              (uint16_t *) gstTensor->data, gstTensor->size
              )
          );
    case _NNS_INT8:
      return InferenceEngine::Blob::Ptr (
          new InferenceEngine::TBlob<int8_t>(
              tensorDesc,
              (int8_t *) gstTensor->data, gstTensor->size
              )
          );
    case _NNS_INT16:
      return InferenceEngine::Blob::Ptr (
          new InferenceEngine::TBlob<int16_t>(
              tensorDesc,
              (int16_t *) gstTensor->data, gstTensor->size
              )
          );
    case _NNS_INT32:
      return InferenceEngine::Blob::Ptr (
          new InferenceEngine::TBlob<int32_t>(
              tensorDesc,
              (int32_t *) gstTensor->data, gstTensor->size
              )
          );
    case _NNS_FLOAT32:
      return InferenceEngine::Blob::Ptr (
          new InferenceEngine::TBlob<float>(
              tensorDesc,
              (float *) gstTensor->data, gstTensor->size
              )
          );
    default:
      return nullptr;
  }
}


/**
 * @brief Get a path where the model file in XML format is located
 * @return a std::string of the path
 */
std::string
TensorFilterOpenvino::getPathModelXml ()
{
  return this->pathModelXml;
}

/**
 * @brief Get a path where the model file in bin format is located
 * @return a std::string of the path
 */
std::string
TensorFilterOpenvino::getPathModelBin ()
{
  return this->pathModelBin;
}

/**
 * @brief TensorFilterOpenvino constructor
 * @param pathModelXml the path of the given model in a XML format
 * @param pathModelBin the path of the given model in a Bin format
 * @return  Nothing
 */
TensorFilterOpenvino::TensorFilterOpenvino (std::string pathModelXml,
    std::string pathModelBin)
{
  this->pathModelXml = pathModelXml;
  this->pathModelBin = pathModelBin;
  (this->_networkReaderCNN).ReadNetwork (this->pathModelXml);
  (this->_networkReaderCNN).ReadWeights (this->pathModelBin);
  this->_networkCNN = (this->_networkReaderCNN).getNetwork ();
  this->_inputsDataMap = (this->_networkCNN).getInputsInfo ();
  this->_outputsDataMap = (this->_networkCNN).getOutputsInfo ();
  this->isLoaded = false;
  this->hw = hw;
}

/**
 * @brief TensorFilterOpenvino Destructor
 * @return  Nothing
 */
TensorFilterOpenvino::~TensorFilterOpenvino ()
{
  ;
}

/**
 * @brief Load the given neural network into the target device
 * @return 0 (TensorFilterOpenvino::RetSuccess) if OK, negative values if error
 */
int
TensorFilterOpenvino::loadModel ()
{
  std::string targetDevice;
  std::vector<std::string> strVector;
  std::vector<std::string>::iterator strVectorIter;

  if (this->isLoaded) {
    // TODO: Can OpenVino support to replace the loaded model with a new one?
    g_critical ("The model file is already loaded onto the device.");
    return RetEBusy;
  }

  strVector = this->_ieCore.GetAvailableDevices ();
  if (strVector.size () == 0) {
    g_critical ("No devices found for the OpenVino toolkit");
    return RetENoDev;
  }

  switch (this->hw)
  {
  /** TODO: Currently, the CPU (amd64) is the only acceleration device.
   *        Need to check the 'accelerator' property.
   */
  case ACCL_CPU:
#ifdef __OPENVINO_CPU_EXT__
    strVectorIter = std::find(strVector.begin (), strVector.end (),
        "CPU");
    if (strVectorIter == strVector.end ()) {
      g_critical ("Failed to find the CPU plugin of the OpenVino toolkit");
      return RetEInval;
    }
    this->_ieCore.AddExtension(
        std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>(),
        "CPU");
#endif /* __OPENVINO_CPU_EXT__ */
    break;
  default:
    goto err;
  }

  this->_executableNet = this->_ieCore.LoadNetwork (this->_networkCNN,
        *strVectorIter);
  this->isLoaded = true;
  this->_inferRequest = this->_executableNet.CreateInferRequest ();

  return RetSuccess;
err:
  g_critical ("Failed to load the modes onto the device.");

  return RetENoDev;
}

/**
 * @brief	Get the information about the dimensions of input tensors from the given model
 * @param[out] info metadata containing the dimesions and types information of the input tensors
 * @return 0 (TensorFilterOpenvino::RetSuccess) if OK, negative values if error
 */
int
TensorFilterOpenvino::getInputTensorDim (GstTensorsInfo * info)
{
  InferenceEngine::InputsDataMap *inputsDataMap = &(this->_inputsDataMap);
  InferenceEngine::InputsDataMap::iterator inputDataMapIter;
  int ret, i, j;

  gst_tensors_info_init (info);

  info->num_tensors = (uint32_t) inputsDataMap->size ();
  if (info->num_tensors > NNS_TENSOR_SIZE_LIMIT) {
    g_critical ("The number of input tenosrs in the model "
        "exceeds more than NNS_TENSOR_SIZE_LIMIT, %s",
        NNS_TENSOR_SIZE_LIMIT_STR);
    ret = RetEOverFlow;
    goto failed;
  }

  for (inputDataMapIter = inputsDataMap->begin (), i = 0;
      inputDataMapIter != inputsDataMap->end (); ++inputDataMapIter, ++i) {
    InferenceEngine::SizeVector::reverse_iterator sizeVecRIter;
    InferenceEngine::TensorDesc eachInputTensorDesc;
    InferenceEngine::InputInfo::Ptr eachInputInfo;
    InferenceEngine::SizeVector dimsSizeVec;
    std::string ieTensorTypeStr;
    tensor_type nnsTensorType;

    eachInputInfo = inputDataMapIter->second;
    eachInputTensorDesc = eachInputInfo->getTensorDesc ();
    dimsSizeVec = eachInputTensorDesc.getDims ();
    if (dimsSizeVec.size () > NNS_TENSOR_RANK_LIMIT) {
      g_critical ("The ranks of dimensions of InputTensor[%d] in the model "
          "exceeds NNS_TENSOR_RANK_LIMIT, %u", i, NNS_TENSOR_RANK_LIMIT);
      ret = RetEOverFlow;
      goto failed;
    }

    for (sizeVecRIter = dimsSizeVec.rbegin (), j = 0;
        sizeVecRIter != dimsSizeVec.rend (); ++sizeVecRIter, ++j) {
      info->info[i].dimension[j] = (*sizeVecRIter != 0 ? *sizeVecRIter : 1);
    }
    for (int k = j; k < NNS_TENSOR_RANK_LIMIT; ++k) {
      info->info[i].dimension[k] = 1;
    }

    ieTensorTypeStr = eachInputInfo->getPrecision ().name ();
    nnsTensorType = TensorFilterOpenvino::convertFromIETypeStr (ieTensorTypeStr);
    if (nnsTensorType == _NNS_END) {
      g_critical ("The type of tensor elements, %s, "
          "in the model is not supported", ieTensorTypeStr.c_str ());
      ret = RetEInval;
      goto failed;
    }

    info->info[i].type = nnsTensorType;
    info->info[i].name = g_strdup (eachInputInfo->name ().c_str ());
    this->_inputTensorDescs[i] = eachInputTensorDesc;
  }

  return TensorFilterOpenvino::RetSuccess;

failed:
  g_critical ("Failed to get dimension information about input tensor");

  return ret;
}

/**
 * @brief	Get the information about the dimensions of output tensors from the given model
 * @param[out] info metadata containing the dimesions and types information of the output tensors
 * @return 0 (TensorFilterOpenvino::RetSuccess) if OK, negative values if error
 */
int
TensorFilterOpenvino::getOutputTensorDim (GstTensorsInfo * info)
{
  InferenceEngine::OutputsDataMap *outputsDataMap = &(this->_outputsDataMap);
  InferenceEngine::OutputsDataMap::iterator outputDataMapIter;
  int ret, i, j;

  gst_tensors_info_init (info);

  info->num_tensors = (uint32_t) outputsDataMap->size ();
  if (info->num_tensors > NNS_TENSOR_SIZE_LIMIT) {
    g_critical ("The number of output tenosrs in the model "
        "exceeds more than NNS_TENSOR_SIZE_LIMIT, %s",
        NNS_TENSOR_SIZE_LIMIT_STR);
    ret = RetEOverFlow;
    goto failed;
  }

  for (outputDataMapIter = outputsDataMap->begin (), i = 0;
      outputDataMapIter != outputsDataMap->end (); ++outputDataMapIter, ++i) {
    InferenceEngine::SizeVector::reverse_iterator sizeVecRIter;
    InferenceEngine::TensorDesc eachOutputTensorDesc;
    InferenceEngine::SizeVector dimsSizeVec;
    InferenceEngine::DataPtr eachOutputInfo;
    std::string ieTensorTypeStr;
    tensor_type nnsTensorType;

    eachOutputInfo = outputDataMapIter->second;
    eachOutputTensorDesc = eachOutputInfo->getTensorDesc ();
    dimsSizeVec = eachOutputTensorDesc.getDims ();
    if (dimsSizeVec.size () > NNS_TENSOR_RANK_LIMIT) {
      g_critical ("The ranks of dimensions of OutputTensor[%d] in the model "
          "exceeds NNS_TENSOR_RANK_LIMIT, %u", i, NNS_TENSOR_RANK_LIMIT);
      ret = RetEOverFlow;
      goto failed;
    }

    for (sizeVecRIter = dimsSizeVec.rbegin (), j = 0;
        sizeVecRIter != dimsSizeVec.rend (); ++sizeVecRIter, ++j) {
      info->info[i].dimension[j] = (*sizeVecRIter != 0 ? *sizeVecRIter : 1);
    }
    for (int k = j; k < NNS_TENSOR_RANK_LIMIT; ++k) {
      info->info[i].dimension[k] = 1;
    }

    ieTensorTypeStr = eachOutputInfo->getPrecision ().name ();
    nnsTensorType = TensorFilterOpenvino::convertFromIETypeStr (ieTensorTypeStr);
    if (nnsTensorType == _NNS_END) {
      g_critical ("The type of tensor elements, %s, "
          "in the model is not supported", ieTensorTypeStr.c_str ());
      ret = RetEInval;
      goto failed;
    }

    info->info[i].type = nnsTensorType;
    info->info[i].name = g_strdup (eachOutputInfo->getName ().c_str ());
    this->_outputTensorDescs[i] = eachOutputTensorDesc;
  }

  return TensorFilterOpenvino::RetSuccess;

failed:
  g_critical ("Failed to get dimension information about output tensor");

  return ret;
}

/**
 * @brief Do inference using Inference Engine of the OpenVino framework
 * @param prop property of tensor_filter instance
 * @param[in] input the array of input tensors
 * @param[out] output the array of output tensors
 * @return RetSuccess if OK. non-zero if error
 */
int
TensorFilterOpenvino::invoke (const GstTensorFilterProperties * prop,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  InferenceEngine::BlobMap inBlobMap;
  InferenceEngine::BlobMap outBlobMap;
  guint num_tensors;
  guint i;

  num_tensors = (prop->input_meta).num_tensors;
  for (i = 0; i < num_tensors; ++i) {
    const GstTensorInfo *info = &((prop->input_meta).info[i]);
    InferenceEngine::Blob::Ptr blob = convertGstTensorMemoryToBlobPtr (
        this->_inputTensorDescs[i], &(input[i]));
    if (blob == nullptr) {
      g_critical ("Failed to create a blob for the input tensor: %u", i);
      return RetEInval;
    }
    inBlobMap.insert (make_pair (std::string(info->name), blob));
  }
  this->_inferRequest.SetInput (inBlobMap);

  num_tensors = (prop->output_meta).num_tensors;
  for (i = 0; i < num_tensors; ++i) {
    const GstTensorInfo *info = &((prop->output_meta).info[i]);
    InferenceEngine::Blob::Ptr blob = convertGstTensorMemoryToBlobPtr (
        this->_outputTensorDescs[i], &(output[i]));
    outBlobMap.insert (make_pair (std::string(info->name), blob));
    if (blob == nullptr) {
      g_critical ("Failed to create a blob for the output tensor: %u", i);
      return RetEInval;
    }
  }
  this->_inferRequest.SetOutput (outBlobMap);

  this->_inferRequest.Infer();

  return RetSuccess;
}


/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data TensorFilterOpenvino plugin's private data
 * @param[in] input the array of input tensors
 * @param[out] output the array of output tensors
 * @return 0 if OK. non-zero if error
 */
static int
ov_invoke (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  TensorFilterOpenvino *tfOv =
      static_cast < TensorFilterOpenvino * >(*private_data);

  return tfOv->invoke (prop, input, output);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data TensorFilterOpenvino plugin's private data
 * @param[out] info the dimesions and types of input tensors
 * @return 0 (TensorFilterOpenvino::RetSuccess) if OK, negative values if error
 */
static int
ov_getInputDim (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * info)
{
  TensorFilterOpenvino *tfOv =
      static_cast < TensorFilterOpenvino * >(*private_data);

  g_return_val_if_fail (tfOv != nullptr, TensorFilterOpenvino::RetEInval);

  return tfOv->getInputTensorDim (info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data TensorFilterOpenvino plugin's private data
 * @param[out] info The dimesions and types of output tensors
 * @return 0 (TensorFilterOpenvino::RetSuccess) if OK, negative values if error
 */
static int
ov_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  TensorFilterOpenvino *tfOv =
      static_cast < TensorFilterOpenvino * >(*private_data);

  g_return_val_if_fail (tfOv != nullptr, TensorFilterOpenvino::RetEInval);

  return tfOv->getOutputTensorDim (info);
}

/**
 * @brief Standard tensor_filter callback to close sub-plugin
 */
static void
ov_close (const GstTensorFilterProperties * prop, void **private_data)
{
  TensorFilterOpenvino *tfOv =
      static_cast < TensorFilterOpenvino * >(*private_data);

  delete tfOv;
  *private_data = NULL;
}

/**
 * @brief Standard tensor_filter callback to open sub-plugin
 * @return 0 (TensorFilterOpenvino::RetSuccess) if OK, negative values if error
 */
static int
ov_open (const GstTensorFilterProperties * prop, void **private_data)
{
  std::string path_model_prefix;
  std::string model_path_xml;
  std::string model_path_bin;
  guint num_models_xml = 0;
  guint num_models_bin = 0;
  TensorFilterOpenvino *tfOv;
  accl_hw accelerator;

  accelerator = parse_accl_hw (prop->accl_str, openvino_accl_support);
#ifndef __OPENVINO_CPU_EXT__
  if (accelerator == ACCL_CPU) {
    g_critical ("Accelerating via CPU is not supported on the current platform");
    return TensorFilterOpenvino::RetEInval;
  }
#endif
  if (accelerator == ACCL_NONE
      || accelerator == ACCL_AUTO
      || accelerator == ACCL_DEFAULT) {
    g_critical ("Setting a specific accelerating device is required");
    return TensorFilterOpenvino::RetEInval;
  }

  if (prop->num_models == 1) {
    if (g_str_has_suffix (prop->model_files[0],
        TensorFilterOpenvino::extBin.c_str ())
        || g_str_has_suffix (prop->model_files[0],
        TensorFilterOpenvino::extXml.c_str ())) {
      std::string model_path = std::string (prop->model_files[0]);

      path_model_prefix = model_path.substr (0, model_path.length () - 4);
    } else {
      path_model_prefix = std::string (prop->model_files[0]);
    }

    model_path_xml = path_model_prefix + TensorFilterOpenvino::extXml;
    model_path_bin = path_model_prefix + TensorFilterOpenvino::extBin;
  } else{
    for (gint i = 0; i < prop->num_models; ++i) {
      if (g_str_has_suffix (prop->model_files[i],
          TensorFilterOpenvino::extXml.c_str ())) {
        num_models_xml++;
        model_path_xml = std::string (prop->model_files[i]);
      } else if (g_str_has_suffix (prop->model_files[i],
          TensorFilterOpenvino::extBin.c_str ())) {
        num_models_bin++;
        model_path_bin = std::string (prop->model_files[i]);
      }

      if (num_models_xml > 1) {
        g_critical ("Too many model files in a XML format are provided.");
        return TensorFilterOpenvino::RetEInval;
      } else if (num_models_bin > 1) {
        g_critical ("Too many model files in a BIN format are provided.");
        return TensorFilterOpenvino::RetEInval;
      }
    }
  }

  if (!g_file_test (model_path_xml.c_str (), G_FILE_TEST_IS_REGULAR)) {
    g_critical ("Failed to open the XML model file, %s",
        model_path_xml.c_str ());
    return TensorFilterOpenvino::RetEInval;
  }
  if (!g_file_test (model_path_bin.c_str (), G_FILE_TEST_IS_REGULAR)) {
    g_critical ("Failed to open the BIN model file, %s",
        model_path_bin.c_str ());
    return TensorFilterOpenvino::RetEInval;
  }

  tfOv = static_cast<TensorFilterOpenvino *>(*private_data);
  if (tfOv != nullptr) {
    if ((tfOv->getPathModelBin () == model_path_bin) &&
        (tfOv->getPathModelXml () == model_path_xml)) {
      return TensorFilterOpenvino::RetSuccess;
    }

    ov_close (prop, private_data);
    tfOv = nullptr;
  }

  tfOv = new TensorFilterOpenvino (model_path_xml, model_path_bin);
  *private_data = tfOv;

  return tfOv->loadModel ();
}

static gchar filter_subplugin_openvino[] = "openvino";

static GstTensorFilterFramework NNS_support_openvino = {
  .name = filter_subplugin_openvino,
  .allow_in_place = FALSE,
  .allocate_in_invoke = FALSE,
  .run_without_model = FALSE,
  .invoke_NN = ov_invoke,
  .getInputDimension = ov_getInputDim,
  .getOutputDimension = ov_getOutputDim,
  .setInputDimension = NULL,
  .open = ov_open,
  .close = ov_close,
};

/**
 * @brief Initialize this object for tensor_filter sub-plugin runtime register
 */
void
init_filter_openvino (void)
{
  nnstreamer_filter_probe (&NNS_support_openvino);
}

/**
 * @brief Finalize the subplugin
 */
void
fini_filter_openvino (void)
{
  nnstreamer_filter_exit (NNS_support_openvino.name);
}
