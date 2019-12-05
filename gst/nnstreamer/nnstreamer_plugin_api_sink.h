/**
 * NNStreamer API for Tensor_Filter Sub-Plugins
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file  nnstreamer_plugin_api_sink.h
 * @date  05 Dec 2019
 * @brief Mandatory APIs for NNStreamer Sink sub-plugins (No External Dependencies)
 * @see https://github.com/nnsuite/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __NNS_PLUGIN_API_SINK_H__
#define __NNS_PLUGIN_API_SINK_H__

#include "tensor_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Property values for the sink framework.
 */
typedef struct _GstTensorSinkProperties
{
  const char *fwname;
  int fw_opened;
  const char *option;
  int input_configured;
  GstTensorsInfo input_meta;
  int output_configured;
} GstTensorSinkProperties;

/**
 * @brief The sink subplugin class
 */
typedef struct _GstTensorSinkFramework
{
  char *name; /**< fwname of the subplugin */
  int is_TensorsInfo_Updatable; /**< non-zero if it supports dim/type change */
  int (*send) (const GstTensorSinkProperties * prop, void **private_data,
      const GstTensorMemory * input);
      /**< Note that input_meta (dim/type) may be updated in run-time if
       * is_TensorInfo_Updatable holds. If it holds, send should check
       * if it has been updated for every frame. Otherwise, changing the
       * dimension is not allowed.
       * @todo We are not so familiar with dynamic dimension in a pipeline.
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data data structure of the subplugin.
       * @param[in] input The input frame buffer. The subplugin should
       *                  check the integrity by looking at its size.
       */
  int (*open) (const GstTensorSinkProperties * prop, void **private_data);
      /**< Prepare the given sink element
       * @param[in] prop read-only property values
       * @param[in/out] private_data data structure of the subplugin.
       */
  int (*close) (const GstTensorSinkProperties * prop, void **private_data);
      /**< Stop the given sink element
       * @param[in] prop read-only property values
       * @param[in/out] private_data data structure of the subplugin.
       */
} GstTensorSinkFramework;

/**
 * @brief Register the subplugin.
 * @param[in] tfsink Tensor-Sink subplugin to be registered.
 */
extern int
nnstreamer_sink_probe (GstTensorSinkFramework *tfsink);

/**
 * @brief Unregister the subplugin.
 * @param[in] name Name of the Tensor-Sink subplugin to be removed.
 */
extern void
nnstreamer_sink_exit (const char *name);

/**
 * @brief Find the subplugin.
 * @param[in] name Name of the Tensor-Sink subplugin to be found.
 */
extern const GstTensorSinkFramework *
nnstreamer_sink_find (const char *name);

#ifdef __cplusplus
}
#endif

#endif /* __NNS_PLUGIN_API_SINK_H__ */
