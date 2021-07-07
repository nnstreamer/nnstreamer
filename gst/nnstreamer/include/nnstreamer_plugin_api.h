/**
 * NNStreamer Common API Header for Plug-Ins
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file  nnstreamer_plugin_api.h
 * @date  24 Jan 2019
 * @brief Optional/Addtional NNStreamer APIs for sub-plugin writers. (Need Glib)
 * @see https://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com> and Wook Song <wook16.song@samsung.com>
 * @bug No known bugs except for NYI items
 *
 */
#ifndef __NNS_PLUGIN_API_H__
#define __NNS_PLUGIN_API_H__

#include <glib.h>
#include <gst/gst.h>
#include <tensor_typedef.h>
#include <nnstreamer_version.h>

G_BEGIN_DECLS

/**
 * @brief Check given mimetype is tensor stream.
 * @param structure structure to be interpreted
 * @return TRUE if mimetype is tensor stream
 */
extern gboolean
gst_structure_is_tensor_stream (const GstStructure * structure);

/**
 * @brief Get media type from structure
 * @param structure structure to be interpreted
 * @return corresponding media type (returns _NNS_MEDIA_INVALID for unsupported type)
 */
extern media_type
gst_structure_get_media_type (const GstStructure * structure);

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
 * @return TRUE if equal, FALSE if given tensor infos are invalid or not equal.
 */
extern gboolean
gst_tensor_info_is_equal (const GstTensorInfo * i1, const GstTensorInfo * i2);

/**
 * @brief Check given info is flexible tensor.
 * @return TRUE if it is flexible.
 */
extern gboolean
gst_tensor_info_is_flexible (const GstTensorInfo * info);

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
 * @brief Convert GstTensorInfo structure to GstTensorMetaInfo.
 * @param[in] info GstTensorInfo to be converted
 * @param[out] meta tensor meta structure to be filled
 * @return TRUE if successfully set the meta
 */
extern gboolean
gst_tensor_info_convert_to_meta (GstTensorInfo * info, GstTensorMetaInfo * meta);

/**
 * @brief Get tensor rank
 * @param info tensor info structure
 * @return tensor rank (Minimum rank is 1 if given info is valid)
 */
extern gint
gst_tensor_info_get_rank (const GstTensorInfo * info);

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
 * @brief Get data size of single tensor
 * @param info tensors info structure
 * @param index the index of tensor (-1 to get total size of tensors)
 * @return data size
 */
gsize
gst_tensors_info_get_size (const GstTensorsInfo * info, gint index);

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
 * @return TRUE if equal, FALSE if given tensor infos are invalid or not equal.
 */
extern gboolean
gst_tensors_info_is_equal (const GstTensorsInfo * i1, const GstTensorsInfo * i2);

/**
 * @brief Check given info contains flexible tensor.
 * @return TRUE if it is flexible.
 */
extern gboolean
gst_tensors_info_is_flexible (const GstTensorsInfo * info);

/**
 * @brief Copy tensor info
 * @note Copied info should be freed with gst_tensors_info_free()
 */
extern void
gst_tensors_info_copy (GstTensorsInfo * dest, const GstTensorsInfo * src);

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
 * @brief Parse caps from peer pad and set tensors config.
 * @param pad GstPad to get the capabilities
 * @param config tensors config structure to be filled
 * @param is_fixed flag to be updated when peer caps is fixed (not mandatory, do nothing when the param is null)
 * @return TRUE if successfully configured from peer
 */
extern gboolean
gst_tensors_config_from_peer (GstPad * pad, GstTensorsConfig * config,
    gboolean * is_fixed);

/**
 * @brief Initialize the tensors config info structure (for other/tensors)
 * @param config tensors config structure to be initialized
 */
extern void
gst_tensors_config_init (GstTensorsConfig * config);

/**
 * @brief Free allocated data in tensors config structure
 * @param config tensors config structure
 */
extern void
gst_tensors_config_free (GstTensorsConfig * config);

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
 * @brief Copy tensors config
 */
extern void
gst_tensors_config_copy (GstTensorsConfig * dest, const GstTensorsConfig * src);

/**
 * @brief Get tensor caps from tensors config (for other/tensor)
 * @param config tensors config info
 * @return caps for given config
 */
extern GstCaps *
gst_tensor_caps_from_config (const GstTensorsConfig * config);

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
 * @brief Get dimension string from given tensor dimension and rank count.
 * @param dim tensor dimension
 * @param rank rank count of given tensor dimension
 * @return Formatted string of given dimension
 * @note If rank count is 3, then returned string is 'd1:d2:d3`.
 * The returned value should be freed with g_free().
 */
extern gchar *
gst_tensor_get_rank_dimension_string (const tensor_dim dim, const unsigned int rank);

/**
 * @brief Count the number of elements of a tensor
 * @return The number of elements. 0 if error.
 * @param dim The tensor dimension
 */
extern gulong
gst_tensor_get_element_count (const tensor_dim dim);

/**
 * @brief Get element size of tensor type (byte per element)
 */
extern gsize
gst_tensor_get_element_size (tensor_type type);

/**
 * @brief Get tensor type from string input.
 * @return Corresponding tensor_type. _NNS_END if unrecognized value is there.
 * @param typestr The string type name, supposed to be one of tensor_element_typename[]
 */
extern tensor_type
gst_tensor_get_type (const gchar * typestr);

/**
 * @brief Get type string of tensor type.
 */
extern const gchar *
gst_tensor_get_type_string (tensor_type type);

/**
 * @brief Get tensor format from string input.
 * @param format_str The string format name, supposed to be one of tensor_format_name[].
 * @return Corresponding tensor_format. _NNS_TENSOR_FORMAT_END if unrecognized value is there.
 */
extern tensor_format
gst_tensor_get_format (const gchar * format_str);

/**
 * @brief Get tensor format string.
 */
extern const gchar *
gst_tensor_get_format_string (tensor_format format);

/**
 * @brief set alignment that default allocator would align to
 * @param alignment bytes of alignment
 */
extern void gst_tensor_alloc_init (gsize alignment);

/**
 * @brief Find the index value of the given key string array
 * @return Corresponding index
 * @param strv Null terminated array of gchar *
 * @param key The key string value
 */
extern gint
find_key_strv (const gchar ** strv, const gchar * key);

/**
 * @brief Replaces string.
 * This function deallocates the input source string.
 * @param[in] source The input string. This will be freed when returning the replaced string.
 * @param[in] what The string to search for.
 * @param[in] to The string to be replaced.
 * @param[in] delimiters The characters which specify the place to split the string. Set NULL to replace all matched string.
 * @param[out] count The count of replaced. Set NULL if it is unnecessary.
 * @return Newly allocated string. The returned string should be freed with g_free().
 */
extern gchar *
replace_string (gchar * source, const gchar * what, const gchar * to, const gchar * delimiters, guint * count);

/**
 * @brief Initialize the tensor meta info structure.
 * @param[in,out] meta tensor meta structure to be initialized
 */
extern void
gst_tensor_meta_info_init (GstTensorMetaInfo * meta);

/**
 * @brief Get the version of tensor meta.
 * @param[in] meta tensor meta structure
 * @param[out] major pointer to get the major version number
 * @param[out] minor pointer to get the minor version number
 */
extern void
gst_tensor_meta_info_get_version (GstTensorMetaInfo * meta, guint * major, guint * minor);

/**
 * @brief Check the meta info is valid.
 * @param[in] meta tensor meta structure
 * @return TRUE if given meta is valid
 */
extern gboolean
gst_tensor_meta_info_validate (GstTensorMetaInfo * meta);

/**
 * @brief Get the header size to handle a tensor meta.
 * @param[in] meta tensor meta structure
 * @return Header size for meta info (0 if meta is invalid)
 */
extern gsize
gst_tensor_meta_info_get_header_size (GstTensorMetaInfo * meta);

/**
 * @brief Get the data size calculated from tensor meta.
 * @param[in] meta tensor meta structure
 * @return The data size for meta info (0 if meta is invalid)
 */
extern gsize
gst_tensor_meta_info_get_data_size (GstTensorMetaInfo * meta);

/**
 * @brief Update header from tensor meta.
 * @param[in] meta tensor meta structure
 * @param[out] header pointer to header to be updated
 * @return TRUE if successfully set the header
 * @note User should allocate enough memory for header (see gst_tensor_meta_info_get_header_size()).
 */
extern gboolean
gst_tensor_meta_info_update_header (GstTensorMetaInfo * meta, gpointer header);

/**
 * @brief Parse header and fill the tensor meta.
 * @param[out] meta tensor meta structure to be filled
 * @param[in] header pointer to header to be parsed
 * @return TRUE if successfully set the meta
 */
extern gboolean
gst_tensor_meta_info_parse_header (GstTensorMetaInfo * meta, gpointer header);

/**
 * @brief Parse memory and fill the tensor meta.
 * @param[out] meta tensor meta structure to be filled
 * @param[in] mem pointer to GstMemory to be parsed
 * @return TRUE if successfully set the meta
 */
extern gboolean
gst_tensor_meta_info_parse_memory (GstTensorMetaInfo * meta, GstMemory * mem);

/**
 * @brief Append header to memory.
 * @param[in] meta tensor meta structure
 * @param[in] mem pointer to GstMemory
 * @return Newly allocated GstMemory (Caller should free returned memory using gst_memory_unref())
 */
extern GstMemory *
gst_tensor_meta_info_append_header (GstTensorMetaInfo * meta, GstMemory * mem);

/**
 * @brief Convert GstTensorMetaInfo structure to GstTensorInfo.
 * @param[in] meta tensor meta structure to be converted
 * @param[out] info GstTensorInfo to be filled
 * @return TRUE if successfully set the info
 */
gboolean
gst_tensor_meta_info_convert (GstTensorMetaInfo * meta, GstTensorInfo * info);

/**
 * @brief Get the version of NNStreamer.
 * @return Newly allocated string. The returned string should be freed with g_free().
 */
extern gchar *
nnstreamer_version_string (void);

/**
 * @brief Get the version of NNStreamer (int, divided).
 * @param[out] major MAJOR.minor.micro, won't set if it's null.
 * @param[out] minor major.MINOR.micro, won't set if it's null.
 * @param[out] micro major.minor.MICRO, won't set if it's null.
 */
extern void
nnstreamer_version_fetch (guint * major, guint * minor, guint * micro);

G_END_DECLS
#endif /* __NNS_PLUGIN_API_H__ */
