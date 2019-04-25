/**
 * NNStreamer Common API Header for Plug-Ins
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file  nnstreamer_plugin_api.h
 * @date  24 Jan 2019
 * @brief Optional/Addtional NNStreamer APIs for sub-plugin writers. (Need Glib)
 * @see https://github.com/nnsuite/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com> and Wook Song <wook16.song@samsung.com>
 * @bug No known bugs except for NYI items
 *
 */
#ifndef __NNS_PLUGIN_API_H__
#define __NNS_PLUGIN_API_H__

#include <glib.h>
#include <gst/gst.h>
#include <tensor_typedef.h>

G_BEGIN_DECLS

/**
 * @brief String representations for each tensor element type.
 */
extern const gchar *tensor_element_typename[];

/**
 * @brief Initialize the tensor info structure
 * @param info tensor info structure to be initialized
 */
extern void
gst_tensor_info_init (GstTensorInfo * info);

/**
 * @brief Free allocated data in tensor info structure
 * @param info tensor info structure
 */
extern void
gst_tensor_info_free (GstTensorInfo * info);

/**
 * @brief Get data size of single tensor
 * @param info tensor info structure
 * @return data size
 */
extern gsize
gst_tensor_info_get_size (const GstTensorInfo *info);

/**
 * @brief Check the tensor info is valid
 * @param info tensor info structure
 * @return TRUE if info is valid
 */
extern gboolean
gst_tensor_info_validate (const GstTensorInfo *info);

/**
 * @brief Compare tensor info
 * @param TRUE if equal
 */
extern gboolean
gst_tensor_info_is_equal (const GstTensorInfo * i1, const GstTensorInfo * i2);

/**
 * @brief Copy tensor info up to n elements
 * @note Copied info should be freed with gst_tensor_info_free()
 */
extern void
gst_tensor_info_copy_n (GstTensorInfo * dest, const GstTensorInfo * src,
    const guint n);

/**
 * @brief Copy tensor info
 * @note Copied info should be freed with gst_tensor_info_free()
 */
extern void
gst_tensor_info_copy (GstTensorInfo * dest, const GstTensorInfo * src);

/**
 * @brief Initialize the tensors info structure
 * @param info tensors info structure to be initialized
 */
extern void
gst_tensors_info_init (GstTensorsInfo *info);

/**
 * @brief Free allocated data in tensors info structure
 * @param info tensors info structure
 */
extern void
gst_tensors_info_free (GstTensorsInfo * info);

/**
 * @brief Parse the string of dimensions
 * @param info tensors info structure
 * @param dim_string string of dimensions
 * @return number of parsed dimensions
 */
extern guint
gst_tensors_info_parse_dimensions_string (GstTensorsInfo *info,
    const gchar *dim_string);

/**
 * @brief Parse the string of types
 * @param info tensors info structure
 * @param type_string string of types
 * @return number of parsed types
 */
extern guint
gst_tensors_info_parse_types_string (GstTensorsInfo * info,
    const gchar * type_string);

/**
 * @brief Parse the string of names
 * @param info tensors info structure
 * @param name_string string of names
 * @return number of parsed names
 */
extern guint
gst_tensors_info_parse_names_string (GstTensorsInfo * info,
    const gchar * name_string);

/**
 * @brief Get the string of dimensions in tensors info
 * @param info tensors info structure
 * @return string of dimensions in tensors info (NULL if the number of tensors is 0)
 * @note The returned value should be freed with g_free()
 */
extern gchar *
gst_tensors_info_get_dimensions_string (const GstTensorsInfo * info);

/**
 * @brief Get the string of types in tensors info
 * @param info tensors info structure
 * @return string of types in tensors info (NULL if the number of tensors is 0)
 * @note The returned value should be freed with g_free()
 */
extern gchar *
gst_tensors_info_get_types_string (const GstTensorsInfo * info);

/**
 * @brief Get the string of tensor names in tensors info
 * @param info tensors info structure
 * @return string of names in tensors info (NULL if the number of tensors is 0)
 * @note The returned value should be freed with g_free()
 */
extern gchar *
gst_tensors_info_get_names_string (const GstTensorsInfo * info);

/**
 * @brief Check the tensors info is valid
 * @param info tensors info structure
 * @return TRUE if info is valid
 */
extern gboolean
gst_tensors_info_validate (const GstTensorsInfo *info);

/**
 * @brief Compare tensors info
 * @param TRUE if equal
 */
extern gboolean
gst_tensors_info_is_equal (const GstTensorsInfo * i1, const GstTensorsInfo * i2);

/**
 * @brief Copy tensor info
 * @note Copied info should be freed with gst_tensors_info_free()
 */
extern void
gst_tensors_info_copy (GstTensorsInfo * dest, const GstTensorsInfo * src);

/**
 * @brief Initialize the tensor config info structure
 * @param config tensor config structure to be initialized
 */
extern void
gst_tensor_config_init (GstTensorConfig * config);

/**
 * @brief Check the tensor is all configured
 * @param config tensor config structure
 * @return TRUE if configured
 */
extern gboolean
gst_tensor_config_validate (const GstTensorConfig * config);

/**
 * @brief Compare tensor config info
 * @param TRUE if equal
 */
extern gboolean
gst_tensor_config_is_equal (const GstTensorConfig * c1,
    const GstTensorConfig * c2);

/**
 * @brief Get media type from structure
 * @param structure structure to be interpreted
 * @return corresponding media type (returns _NNS_MEDIA_END for unsupported type)
 */
extern media_type
gst_tensor_media_type_from_structure (const GstStructure * structure);

/**
 * @brief Parse structure and set tensor config info (for other/tensor)
 * @param config tensor config structure to be filled
 * @param structure structure to be interpreted
 * @note Change dimention if tensor contains N frames.
 * @return TRUE if no error
 */
extern gboolean
gst_tensor_config_from_structure (GstTensorConfig *config,
    const GstStructure *structure);

/**
 * @brief Parse structure and set tensors config (for other/tensors)
 * @param config tensors config structure to be filled
 * @param structure structure to be interpreted
 * @return TRUE if no error
 */
extern gboolean
gst_tensors_config_from_structure (GstTensorsConfig *config,
    const GstStructure *structure);

/**
 * @brief Initialize the tensors config info structure (for other/tensors)
 * @param config tensors config structure to be initialized
 */
extern void
gst_tensors_config_init (GstTensorsConfig * config);

/**
 * @brief Check the tensors are all configured (for other/tensors)
 * @param config tensor config structure
 * @return TRUE if configured
 */
extern gboolean
gst_tensors_config_validate (const GstTensorsConfig * config);

/**
 * @brief Compare tensor config info (for other/tensors)
 * @param TRUE if equal
 */
extern gboolean
gst_tensors_config_is_equal (const GstTensorsConfig * c1,
    const GstTensorsConfig * c2);

/**
 * @brief Get tensor caps from tensor config (for other/tensor)
 * @param config tensor config info
 * @return caps for given config
 */
extern GstCaps *
gst_tensor_caps_from_config (const GstTensorConfig * config);

/**
 * @brief Get caps from tensors config (for other/tensors)
 * @param config tensors config info
 * @return caps for given config
 */
extern GstCaps *
gst_tensors_caps_from_config (const GstTensorsConfig * config);

/**
 * @brief Check the tensor dimension is valid
 * @param dim tensor dimension
 * @return TRUE if dimension is valid
 */
extern gboolean
gst_tensor_dimension_is_valid (const tensor_dim dim);

/**
 * @brief Parse tensor dimension parameter string
 * @return The Rank. 0 if error.
 * @param dimstr The dimension string in the format of d1:d2:d3:d4, d1:d2:d3, d1:d2, or d1, where dN is a positive integer and d1 is the innermost dimension; i.e., dim[d4][d3][d2][d1];
 * @param dim dimension to be filled.
 */
extern guint
gst_tensor_parse_dimension (const gchar * dimstr, tensor_dim dim);

/**
 * @brief Get dimension string from given tensor dimension.
 * @param dim tensor dimension
 * @return Formatted string of given dimension (d1:d2:d3:d4).
 * @note The returned value should be freed with g_free()
 */
extern gchar *
gst_tensor_get_dimension_string (const tensor_dim dim);

/**
 * @brief Count the number of elemnts of a tensor
 * @return The number of elements. 0 if error.
 * @param dim The tensor dimension
 */
extern gsize
gst_tensor_get_element_count (const tensor_dim dim);

/**
 * @brief Get tensor_type from string tensor_type input
 * @return Corresponding tensor_type. _NNS_END if unrecognized value is there.
 * @param typestr The string type name, supposed to be one of tensor_element_typename[]
 */
extern tensor_type
gst_tensor_get_type (const gchar * typestr);

/**
 * @brief Find the index value of the given key string array
 * @return Corresponding index
 * @param strv Null terminated array of gchar *
 * @param key The key string value
 */
extern gint
find_key_strv (const gchar ** strv, const gchar * key);

G_END_DECLS
#endif /* __NNS_PLUGIN_API_H__ */
