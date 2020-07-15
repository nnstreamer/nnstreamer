/**
 * GStreamer Tensor_Filter, OpenVino (DLDT) Module
 * Copyright (C) 2019 Wook Song <wook16.song@samsung.com>
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
 * @file    tensor_filter_openvino.hh
 * @date    23 Dec 2019
 * @brief   Tensor_filter subplugin for OpenVino (DLDT).
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (OpenVino) for tensor_filter.
 *
 * @note This header file is only for internal use.
 *
 * To Packagers:
 *
 * This should not to be exposed with the development packages to the application developers.
 */

#ifndef __TENSOR_FILTER_OPENVINO_H__
#define __TENSOR_FILTER_OPENVINO_H__

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
  static bool isAcclDevSupported (std::vector<std::string> &devsVector,
      accl_hw hw);

  TensorFilterOpenvino (std::string path_model_xml, std::string path_model_bin);
  ~TensorFilterOpenvino ();

  /** @todo Need to support other acceleration devices */
  int loadModel (accl_hw hw);
  bool isModelLoaded () {
    return _isLoaded;
  }

  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int invoke (const GstTensorFilterProperties * prop,
      const GstTensorMemory * input, GstTensorMemory * output);
  std::string getPathModelXml ();
  void setPathModelXml (std::string pathXml);
  std::string getPathModelBin ();
  void setPathModelBin (std::string pathBin);

  static const std::string extBin;
  static const std::string extXml;

protected:
  InferenceEngine::InputsDataMap _inputsDataMap;
  InferenceEngine::OutputsDataMap _outputsDataMap;

private:
  TensorFilterOpenvino ();

  InferenceEngine::Core _ieCore;
  InferenceEngine::CNNNetReader _networkReaderCNN;
  InferenceEngine::CNNNetwork _networkCNN;
  InferenceEngine::TensorDesc _inputTensorDescs[NNS_TENSOR_SIZE_LIMIT];
  InferenceEngine::TensorDesc _outputTensorDescs[NNS_TENSOR_SIZE_LIMIT];
  InferenceEngine::ExecutableNetwork _executableNet;
  InferenceEngine::InferRequest _inferRequest;
  static std::map<accl_hw, std::string> _nnsAcclHwToOVDevMap;

  std::string _pathModelXml;
  std::string _pathModelBin;
  bool _isLoaded;
  accl_hw _hw;
};

#endif /* __TENSOR_FILTER_OPENVINO_H__ */
