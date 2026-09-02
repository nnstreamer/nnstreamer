/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2020 Samsung Electronics
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file    tensor_filter_subplugin_trix_engine.hh
 * @date    20 Jan 2020
 * @brief   NNStreamer tensor-filter subplugin trix_engine header
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs
 */

#ifndef __TENSOR_FILTER_SUBPLUGIN_TRIxEngine_H__
#define __TENSOR_FILTER_SUBPLUGIN_TRIxEngine_H__

/* npu-engine headers */
#include <libnpuhost.h>
#include <npubinfmt.h>

/* nnstreamer plugin api headers */
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

namespace nnstreamer
{

/**
 * @brief Class for TRIx-Engine subplugin
 */
class TensorFilterTRIxEngine : public tensor_filter_subplugin
{
  public:
  TensorFilterTRIxEngine ();
  /** @brief Destruct the TRIx-Engine subplugin instance */
  ~TensorFilterTRIxEngine ();

  /* mandatory methods */
  /** @brief Method to get an empty object */
  tensor_filter_subplugin &getEmptyInstance ();
  /** @brief Configure TRIx-Engine instance */
  void configure_instance (const GstTensorFilterProperties *prop);
  /** @brief Invoke TRIxEngine using input tensors */
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  /** @brief Get TRIxEngine framework info */
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  /** @brief Get TRIxEngine model info */
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  /** @brief Method to handle the event */
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);

  /* static methods */
  /** @brief Register the subplugin */
  static void init_filter_trix_engine ();
  /** @brief Destruct the subplugin */
  static void fini_filter_trix_engine ();

  private:
  /** @brief Convert data layout (from NNStreamer to TRIx-Engine) */
  static data_layout convert_data_layout (const tensor_layout &layout);
  /** @brief Convert data type (from NNStreamer to TRIx-Engine) */
  static data_type convert_data_type (const tensor_type &type);

  static TensorFilterTRIxEngine *registered;
  static const char *name;
  static const accl_hw hw_list[];
  static const int num_hw;

  /** @brief Set data info of input/output tensors using metadata */
  void set_data_info (const GstTensorFilterProperties *prop);
  /** @brief Feed the tensor data to input buffers before invoke() */
  void feed_input_data (const GstTensorMemory *input, input_buffers *input_buf);
  /** @brief Extract the tensor data from output buffers after invoke() */
  void extract_output_data (const output_buffers *output_buf, GstTensorMemory *output);

  /* trix-engine vars */
  dev_type dev_type_;
  npudev_h dev_;
  gchar *model_path_;
  npubin_meta *model_meta_;
  uint32_t model_id_;
  tensors_data_info trix_in_info_;
  tensors_data_info trix_out_info_;

  /* nnstreamer vars */
  GstTensorsInfo nns_in_info_;
  GstTensorsInfo nns_out_info_;
};

} /* namespace nnstreamer */

#endif /* __TENSOR_FILTER_SUBPLUGIN_TRIxEngine_H__ */
