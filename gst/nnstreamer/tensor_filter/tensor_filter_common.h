/**
 * Copyright (C) 2019 Parichay kapoor <pk.kapoor@samsung.com>
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
 * @file	tensor_filter_common.h
 * @date	28 Aug 2019
 * @brief	Common functions for various tensor_filters
 * @see	  http://github.com/nnsuite/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug	  No known bugs except for NYI items
 */

#ifndef __G_TENSOR_FILTER_COMMON_H__
#define __G_TENSOR_FILTER_COMMON_H__

#include <nnstreamer_subplugin.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>

/**
 * @brief Structure definition for common tensor-filter properties.
 */
typedef struct _GstTensorFilterPrivate
{
  void *privateData; /**< NNFW plugin's private data is stored here */
  GstTensorFilterProperties prop; /**< NNFW plugin's properties */
  const GstTensorFilterFramework *fw; /**< The implementation core of the NNFW. NULL if not configured */

  /* internal properties for tensor-filter */
  gboolean silent; /**< Verbose mode if FALSE. int instead of gboolean for non-glib custom plugins */
  gboolean configured; /**< True if already successfully configured tensor metadata */
  gboolean is_updatable; /**<  a given model to the filter is updatable if TRUE */
  GstTensorsConfig in_config; /**< input tensor info */
  GstTensorsConfig out_config; /**< output tensor info */
} GstTensorFilterPrivate;

/**
 * @brief Printout the comparison results of two tensors.
 * @param[in] info1 The tensors to be shown on the left hand side
 * @param[in] info2 The tensors to be shown on the right hand side
 * @todo If this is going to be used by other elements, move this to nnstreamer/tensor_common.
 */
extern void
gst_tensor_filter_compare_tensors (GstTensorsInfo * info1,
    GstTensorsInfo * info2);

/**
 * @brief Installs all the properties for tensor_filter
 * @param[in] gobject_class Glib object class whose properties will be set
 */
extern void
gst_tensor_filter_install_properties (GObjectClass * gobject_class);

/**
 * @brief Initialize the properties for tensor-filter.
 */
extern void
gst_tensor_filter_common_init_property (GstTensorFilterPrivate * priv);

/**
 * @brief Free the properties for tensor-filter.
 */
extern void
gst_tensor_filter_common_free_property (GstTensorFilterPrivate * priv);

/**
 * @brief Set the properties for tensor_filter
 * @param[in] priv Struct containing the properties of the object
 * @param[in] prop_id Id for the property
 * @param[in] value Container to return the asked property
 * @param[in] pspec Metadata to specify the parameter
 * @return TRUE if prop_id is value, else FALSE
 */
extern gboolean
gst_tensor_filter_common_set_property (GstTensorFilterPrivate * priv,
    guint prop_id, const GValue * value, GParamSpec * pspec);

/**
 * @brief Get the properties for tensor_filter
 * @param[in] priv Struct containing the properties of the object
 * @param[in] prop_id Id for the property
 * @param[in] value Container to return the asked property
 * @param[in] pspec Metadata to specify the parameter
 * @return TRUE if prop_id is value, else FALSE
 */
extern gboolean
gst_tensor_filter_common_get_property (GstTensorFilterPrivate * priv,
    guint prop_id, GValue * value, GParamSpec * pspec);

/**
 * @brief Open NN framework.
 */
extern void
gst_tensor_filter_common_open_fw (GstTensorFilterPrivate * priv);

/**
 * @brief Close NN framework.
 */
extern void
gst_tensor_filter_common_close_fw (GstTensorFilterPrivate * priv);

#endif /* __G_TENSOR_FILTER_COMMON_H__ */
