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
 * @file  nnstreamer_plugin_api_filters.h
 * @date  30 Jan 2019
 * @brief Mandatory APIs for NNStreamer Filter sub-plugins (No External Dependencies)
 * @see https://github.com/nnsuite/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __NNS_PLUGIN_API_FILTER_H__
#define __NNS_PLUGIN_API_FILTER_H__

#include "tensor_typedef.h"

/* extern functions for subplugin management, exist in tensor_filter.c */
/**
 * @brief Filter subplugin should call this to register itself
 * @param[in] tfsp Tensor-Filter Sub-Plugin to be registered
 * @return TRUE if registered. FALSE is failed or duplicated.
 */
extern int tensor_filter_probe (GstTensorFilterFramework *tfsp);
/**
 * @brief filter sub-plugin may call this to unregister itself
 * @param[in] name the name of filter sub-plugin
 */
extern void tensor_filter_exit (const char *name);

#endif /* __NNS_PLUGIN_API_FILTER_H__ */
