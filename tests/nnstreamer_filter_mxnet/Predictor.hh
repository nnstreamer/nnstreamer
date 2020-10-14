/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * This example demonstrates image classification workflow with pre-trained models using MXNet C++ API.
 * The example performs following tasks.
 * 1. Load the pre-trained model.
 * 2. Load the parameters of pre-trained model.
 * 3. Load the inference dataset and create a new ImageRecordIter.
 * 4. Run the forward pass and obtain throughput & accuracy.
 */
#include <stdexcept>
#ifndef _WIN32
#include <sys/time.h>
#endif
#include <fstream>
#include <iostream>
#include <map>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <type_traits>
#include <opencv2/opencv.hpp>
#include "mxnet/c_api.h"
#include "mxnet/tuple.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "mxnet-cpp/initializer.h"

using namespace mxnet::cpp;

double ms_now() {
  double ret;
#ifdef _WIN32
  auto timePoint = std::chrono::high_resolution_clock::now().time_since_epoch();
  ret = std::chrono::duration<double, std::milli>(timePoint).count();
#else
  struct timeval time;
  gettimeofday(&time, nullptr);
  ret = 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
#endif
  return ret;
}

inline bool file_exists (const std::string &name)
{
  std::ifstream fhandle(name.c_str());
  return fhandle.good();
}

// define the data type for NDArray, aliged with the definition in mshadow/base.h
enum TypeFlag {
  kFloat32 = 0,
  kFloat64 = 1,
  kFloat16 = 2,
  kUint8 = 3,
  kInt32 = 4,
  kInt8  = 5,
  kInt64 = 6,
};

/*
 * class Predictor
 *
 * This class encapsulates the functionality to load the model, prepare dataset and run the forward pass.
 */

class Predictor {
 public:
    Predictor() {}
    Predictor(const std::string& model_json_file,
              const std::string& model_params_file,
              const Shape& input_shape,
              const std::string& data_layer_type,
              bool use_gpu,
              bool enable_tensorrt,
              MXDataIter * val_iter);
    ~Predictor();

    static MXDataIter *CreateImageRecordIter(
              const std::string& dataset,
              const Shape& input_shape,
              const std::string& data_layer_type,
              const std::vector<float>& rgb_mean,
              const std::vector<float>& rgb_std,
              int shuffle_chunk_seed,
              int seed,
              const int data_nthreads,
              bool use_gpu);
    void LogInferenceResult(std::vector<mx_float> &log_vector, int num_inference_batches);
 private:
    bool AdvanceDataIter(int skipped_batches);
    void LoadModel(const std::string& model_json_file);
    void LoadParameters(const std::string& model_parameters_file);
    void SplitParamMap(const std::map<std::string, NDArray> &paramMap,
        std::map<std::string, NDArray> *argParamInTargetContext,
        std::map<std::string, NDArray> *auxParamInTargetContext,
        Context targetContext);
    void ConvertParamMapToTargetContext(const std::map<std::string, NDArray> &paramMap,
        std::map<std::string, NDArray> *paramMapInTargetContext,
        Context targetContext);

    int GetDataLayerType();

    MXDataIter * val_iter_;
    std::map<std::string, NDArray> args_map_;
    std::map<std::string, NDArray> aux_map_;
    Symbol net_;
    Shape input_shape_;
    Executor *executor_;
    Context global_ctx_ = Context::cpu();

    bool use_gpu_;
    bool enable_tensorrt_;
    std::string data_layer_type_;
};


/*
 * The constructor takes following parameters as input:
 *  1. model_json_file:  The model in json formatted file.
 *  2. model_params_file: File containing model parameters
 *  3. input_shape: Shape of input data to the model. Since this class will be running one inference at a time,
 *                  the input shape is required to be in format Shape(1, number_of_channels, height, width)
 *                  The input image will be resized to (height x width) size before running the inference.
 *  4. data_layer_type: data type for data layer
 *  5. use_gpu: determine if run inference on GPU
 *  6. enable_tensorrt: determine if enable TensorRT
 *  7. val_iter: validation dataset iterator
 *
 * The constructor will:
 *  1. Load the model and parameter files.
 *  2. Infer and construct NDArrays according to the input argument and create an executor.
 */
Predictor::Predictor(const std::string& model_json_file,
              const std::string& model_params_file,
              const Shape& input_shape,
              const std::string& data_layer_type,
              bool use_gpu,
              bool enable_tensorrt,
              MXDataIter * val_iter)
    : val_iter_(val_iter),
      input_shape_(input_shape),
      use_gpu_(use_gpu),
      enable_tensorrt_(enable_tensorrt),
      data_layer_type_(data_layer_type){
  if (use_gpu) {
    global_ctx_ = Context::gpu();
  }
  // Load the model
  LoadModel(model_json_file);
  // Initilize the parameters
  LoadParameters(model_params_file);

  int dtype = GetDataLayerType();
  if (dtype == -1) {
    throw std::runtime_error("Unsupported data layer type...");
  }
  args_map_["data"] = NDArray(input_shape_, global_ctx_, false, dtype);
  Shape label_shape(input_shape_[0]);

  std::vector<NDArray> arg_arrays;
  std::vector<NDArray> grad_arrays;
  std::vector<OpReqType> grad_reqs;
  std::vector<NDArray> aux_arrays;

  // infer and create ndarrays according to the given input ndarrays.
  net_.InferExecutorArrays(global_ctx_, &arg_arrays, &grad_arrays, &grad_reqs,
                           &aux_arrays, args_map_, std::map<std::string, NDArray>(),
                           std::map<std::string, OpReqType>(), aux_map_);

  // Create an executor after binding the model to input parameters.
  executor_ = new Executor(net_, global_ctx_, arg_arrays, grad_arrays, grad_reqs, aux_arrays);
}

/*
 * The following function is used to get the data layer type for input data
 */
int Predictor::GetDataLayerType() {
  int ret_type = -1;
  if (data_layer_type_ == "float32") {
    ret_type = kFloat32;
  } else if (data_layer_type_ == "int8") {
    ret_type = kInt8;
  } else if (data_layer_type_ == "uint8") {
    ret_type = kUint8;
  } else {
    LG << "Unsupported data layer type " << data_layer_type_ << "..."
       << "Please use one of {float32, int8, uint8}";
  }
  return ret_type;
}

/*
 * create a new ImageRecordIter according to the given parameters:
 *  1. dataset: data file (.rec) to be used for inference
 *  2. input_shape: Shape of input data to the model. Since this class will be running one inference at a time,
 *                  the input shape is required to be in format Shape(1, number_of_channels, height, width)
 *                  The input image will be resized to (height x width) size before running the inference.
 *  3. data_layer_type: data type for data layer
 *  4. rgb_mean: mean value to be subtracted on R/G/B channel
 *  5. rgb_std: standard deviation on R/G/B channel
 *  6. shuffle_chunk_seed: shuffling chunk seed
 *  7. seed: shuffling seed
 *  8. data_nthreads: number of threads for data loading
 *  9. use_gpu: determine if run inference on GPU
 */
MXDataIter * Predictor::CreateImageRecordIter(
              const std::string& dataset,
              const Shape& input_shape,
              const std::string& data_layer_type,
              const std::vector<float>& rgb_mean,
              const std::vector<float>& rgb_std,
              int shuffle_chunk_seed,
              int seed,
              const int data_nthreads,
              bool use_gpu) {
  MXDataIter * val_iter = new MXDataIter("ImageRecordIter");
  if (!file_exists (dataset)) {
    throw std::runtime_error("Error: " + dataset + " must be provided");
  }

  std::vector<index_t> shape_vec;
  for (index_t i = 1; i < input_shape.ndim(); i++)
    shape_vec.push_back(input_shape[i]);
  mxnet::TShape data_shape(shape_vec.begin(), shape_vec.end());

  // set image record parser parameters
  val_iter->SetParam("path_imgrec", dataset);
  val_iter->SetParam("label_width", 1);
  val_iter->SetParam("data_shape", data_shape);
  val_iter->SetParam("preprocess_threads", data_nthreads);
  val_iter->SetParam("shuffle_chunk_seed", shuffle_chunk_seed);

  // set Batch parameters
  val_iter->SetParam("batch_size", input_shape[0]);

  // image record parameters
  val_iter->SetParam("shuffle", true);
  val_iter->SetParam("seed", seed);

  // set normalize parameters
  val_iter->SetParam("mean_r", rgb_mean[0]);
  val_iter->SetParam("mean_g", rgb_mean[1]);
  val_iter->SetParam("mean_b", rgb_mean[2]);
  val_iter->SetParam("std_r", rgb_std[0]);
  val_iter->SetParam("std_g", rgb_std[1]);
  val_iter->SetParam("std_b", rgb_std[2]);

  // set prefetcher parameters
  if (use_gpu) {
    val_iter->SetParam("ctx", "gpu");
  } else {
    val_iter->SetParam("ctx", "cpu");
  }
  val_iter->SetParam("dtype", data_layer_type);

  val_iter->CreateDataIter();
  return val_iter;
}

/*
 * The following function loads the model from json file.
 */
void Predictor::LoadModel(const std::string& model_json_file) {
  if (!file_exists (model_json_file)) {
    LG << "Model file " << model_json_file << " does not exist";
    throw std::runtime_error("Model file does not exist");
  }
  LG << "Loading the model from " << model_json_file << std::endl;
  net_ = Symbol::Load(model_json_file);
  if (enable_tensorrt_) {
    net_ = net_.GetBackendSymbol("TensorRT");
  }
}

/*
 * The following function loads the model parameters.
 */
void Predictor::LoadParameters(const std::string& model_parameters_file) {
  if (!file_exists (model_parameters_file)) {
    LG << "Parameter file " << model_parameters_file << " does not exist";
    throw std::runtime_error("Model parameters does not exist");
  }
  LG << "Loading the model parameters from " << model_parameters_file << std::endl;
  std::map<std::string, NDArray> parameters;
  NDArray::Load(model_parameters_file, 0, &parameters);
  if (enable_tensorrt_) {
    std::map<std::string, NDArray> intermediate_args_map;
    std::map<std::string, NDArray> intermediate_aux_map;
    SplitParamMap(parameters, &intermediate_args_map, &intermediate_aux_map, Context::cpu());
    contrib::InitTensorRTParams(net_, &intermediate_args_map, &intermediate_aux_map);
    ConvertParamMapToTargetContext(intermediate_args_map, &args_map_, global_ctx_);
    ConvertParamMapToTargetContext(intermediate_aux_map, &aux_map_, global_ctx_);
  } else {
    SplitParamMap(parameters, &args_map_, &aux_map_, global_ctx_);
  }
  /*WaitAll is need when we copy data between GPU and the main memory*/
  NDArray::WaitAll();
}

/*
 * The following function split loaded param map into arg parm
 *   and aux param with target context
 */
void Predictor::SplitParamMap(const std::map<std::string, NDArray> &paramMap,
    std::map<std::string, NDArray> *argParamInTargetContext,
    std::map<std::string, NDArray> *auxParamInTargetContext,
    Context targetContext) {
  for (const auto& pair : paramMap) {
    std::string type = pair.first.substr(0, 4);
    std::string name = pair.first.substr(4);
    if (type == "arg:") {
      (*argParamInTargetContext)[name] = pair.second.Copy(targetContext);
    } else if (type == "aux:") {
      (*auxParamInTargetContext)[name] = pair.second.Copy(targetContext);
    }
  }
}

/*
 * The following function copy the param map into the target context
 */
void Predictor::ConvertParamMapToTargetContext(const std::map<std::string, NDArray> &paramMap,
    std::map<std::string, NDArray> *paramMapInTargetContext,
    Context targetContext) {
  for (const auto& pair : paramMap) {
    (*paramMapInTargetContext)[pair.first] = pair.second.Copy(targetContext);
  }
}

/*
 * The following function runs the forward pass on the model
 * and use real data for testing accuracy and performance.
 */
void Predictor::LogInferenceResult(std::vector<mx_float> &log_vector, int num_inference_batches) {
  log_vector.reserve(num_inference_batches);
  // Create metrics
  Accuracy val_acc;

  val_iter_->Reset();
  val_acc.Reset();
  int nBatch = 0;

  double ms = ms_now();
  while (val_iter_->Next()) {
    auto data_batch = val_iter_->GetDataBatch();
    data_batch.data.CopyTo(&args_map_["data"]);
    NDArray::WaitAll();

    // running on forward pass
    executor_->Forward(false);
    NDArray::WaitAll();
    NDArray result;
    Operator("argmax_channel")(executor_->outputs[0]).Invoke(result);

    // Write to the log file
    mx_uint len = result.GetShape()[0];
    std::vector<mx_float> data_vector(len);
    result.SyncCopyToCPU(&data_vector, len);
    log_vector.push_back(data_vector[0]);

    // Update score
    val_acc.Update(data_batch.label, executor_->outputs[0]);
    if (++nBatch >= num_inference_batches) {
      break;
    }
  }

  ms = ms_now() - ms;
  auto args_name = net_.ListArguments();
  LG << "INFO:" << "Finished inference with: " << nBatch * input_shape_[0]
     << " images ";
  LG << "INFO:" << "Batch size = " << input_shape_[0] << " for inference";
  LG << "INFO:" << "Accuracy: " << val_acc.Get();
  LG << "INFO:" << "Throughput: " << (1000.0 * nBatch * input_shape_[0] / ms)
     << " images per second";
}

Predictor::~Predictor() {
  if (executor_) {
    delete executor_;
  }
}
