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
#include <npubinfmt.h>
#include <libnpuhost.h>

/* nnstreamer plugin api headers */
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_util.h>
#include <nnstreamer_log.h>

namespace nnstreamer {

/**
 * @brief Class for TRIx-Engine subplugin
 */
class TensorFilterTRIxEngine : public tensor_filter_subplugin {
 public:
  TensorFilterTRIxEngine ();
  ~TensorFilterTRIxEngine ();

  /* mandatory methods */
  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);

  /* static methods */
  static void init_filter_trix_engine ();
  static void fini_filter_trix_engine ();

 private:
  static data_layout convert_data_layout (const tensor_layout &layout);
  static data_type convert_data_type (const tensor_type &type);

  static TensorFilterTRIxEngine *registered;
  static const char *name;
  static const accl_hw hw_list[];
  static const int num_hw;

  void set_data_info (const GstTensorFilterProperties *prop);
  void feed_input_data (const GstTensorMemory *input, input_buffers *input_buf);
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
