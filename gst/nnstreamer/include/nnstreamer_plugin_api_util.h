/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer Common API Header for sub-plugin writers
 * Copyright (C) 2022 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file  nnstreamer_plugin_api_util.h
 * @date  28 Jan 2022
 * @brief Optional/Additional NNStreamer APIs for sub-plugin writers. (No GStreamer dependency)
 * @see https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __NNS_PLUGIN_API_UTIL_H__
#define __NNS_PLUGIN_API_UTIL_H__

#include <glib.h>
#include <tensor_typedef.h>
#include <nnstreamer_version.h>

G_BEGIN_DECLS

/**
 * @brief If the given string is NULL, print "(NULL)". Copied from `GST_STR_NULL`
 */
#define _STR_NULL(str) ((str) ? (str) : "(NULL)")

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
extern guint
gst_tensor_info_get_rank (const GstTensorInfo * info);

/**
 * @brief GstTensorInfo represented as a string.
 * @param info GstTensorInfo structure.
 * @return The newly allocated string representing the tensor info. Caller should free the value using g_free().
 */
extern gchar *
gst_tensor_info_to_string (const GstTensorInfo * info);

/**
 * @brief Get the pointer of nth tensor information.
 * @param info tensors info structure
 * @param index the index of tensor to be fetched
 * @return The pointer to tensor info structure
 */
extern GstTensorInfo *
gst_tensors_info_get_nth_info (GstTensorsInfo * info, guint index);

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
 * @brief Get the string of dimensions in tensors info and rank count
 * @param info tensors info structure
 * @param rank rank count of given tensor dimension
 * @param padding fill 1 if actual rank is smaller than rank
 * @return Formatted string of given dimension
 * @note If rank count is 3, then returned string is 'd1:d2:d3`.
 * The returned value should be freed with g_free()
 */
extern gchar *
gst_tensors_info_get_rank_dimensions_string (const GstTensorsInfo * info,
    const unsigned int rank, const gboolean padding);

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
 * @brief Copy tensor info
 * @note Copied info should be freed with gst_tensors_info_free()
 */
extern void
gst_tensors_info_copy (GstTensorsInfo * dest, const GstTensorsInfo * src);

/**
 * @brief GstTensorsInfo represented as a string.
 * @param info GstTensorsInfo structure.
 * @return The newly allocated string representing the tensors info. Caller should free the value using g_free().
 */
extern gchar *
gst_tensors_info_to_string (const GstTensorsInfo * info);

/**
 * @brief Printout the comparison results of two tensors as a string.
 * @param[in] info1 The tensors to be shown on the left hand side.
 * @param[in] info2 The tensors to be shown on the right hand side.
 * @return The printout string allocated. Caller should free the value using g_free().
 */
extern gchar *
gst_tensors_info_compare_to_string (const GstTensorsInfo * info1, const GstTensorsInfo * info2);

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
 * @brief Tensor config represented as a string.
 * @param config tensor config structure.
 * @return The newly allocated string representing the config. Caller should free the value using g_free().
 */
extern gchar *
gst_tensors_config_to_string (const GstTensorsConfig * config);

/**
 * @brief Macro to check stream format (static tensors for caps negotiation)
 */
#define gst_tensors_config_is_static(c) ((c)->info.format == _NNS_TENSOR_FORMAT_STATIC)

/**
 * @brief Macro to check stream format (flexible tensors for caps negotiation)
 */
#define gst_tensors_config_is_flexible(c) ((c)->info.format == _NNS_TENSOR_FORMAT_FLEXIBLE)

/**
 * @brief Macro to check stream format (sparse tensors for caps negotiation)
 */
#define gst_tensors_config_is_sparse(c) ((c)->info.format == _NNS_TENSOR_FORMAT_SPARSE)

/**
 * @brief Check the tensor dimension is valid
 * @param dim tensor dimension
 * @return TRUE if dimension is valid
 */
extern gboolean
gst_tensor_dimension_is_valid (const tensor_dim dim);

/**
 * @brief Compare the tensor dimension.
 * @return TRUE if given tensors have same dimension.
 */
extern gboolean
gst_tensor_dimension_is_equal (const tensor_dim dim1, const tensor_dim dim2);

/**
 * @brief Get the rank of tensor dimension.
 * @param dim tensor dimension.
 * @return tensor rank (Minimum rank is 1 if given info is valid)
 */
extern guint
gst_tensor_dimension_get_rank (const tensor_dim dim);

/**
 * @brief Get the minimum rank of tensor dimension.
 * @details The C-arrays with dim 4:4:4 and 4:4:4:1 have same data. In this case, this function returns min rank 3.
 * @param dim tensor dimension.
 * @return tensor rank (Minimum rank is 1 if given dimension is valid)
 */
extern guint
gst_tensor_dimension_get_min_rank (const tensor_dim dim);

/**
 * @brief Parse tensor dimension parameter string
 * @return The Rank. 0 if error.
 * @param dimstr The dimension string in the format of d1:...:d16, d1:d2:d3, d1:d2, or d1, where dN is a positive integer and d1 is the innermost dimension; i.e., dim[d16]...[d1];
 * @param dim dimension to be filled.
 */
extern guint
gst_tensor_parse_dimension (const gchar * dimstr, tensor_dim dim);

/**
 * @brief Get dimension string from given tensor dimension.
 * @param dim tensor dimension
 * @return Formatted string of given dimension (d1:d2:d3:...:d15:d16).
 * @note The returned value should be freed with g_free()
 */
extern gchar *
gst_tensor_get_dimension_string (const tensor_dim dim);

/**
 * @brief Get dimension string from given tensor dimension and rank count.
 * @param dim tensor dimension
 * @param rank rank count of given tensor dimension
 * @param padding fill 1 if actual rank is smaller than rank
 * @return Formatted string of given dimension
 * @note If rank count is 3, then returned string is 'd1:d2:d3`.
 * The returned value should be freed with g_free().
 */
extern gchar *
gst_tensor_get_rank_dimension_string (const tensor_dim dim,
    const unsigned int rank, const gboolean padding);

/**
 * @brief Compare dimension strings
 * @return TRUE if equal, FALSE if given dimension strings are invalid or not equal.
 */
extern gboolean
gst_tensor_dimension_string_is_equal (const gchar * dimstr1, const gchar * dimstr2);

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
 * @brief Find the index value of the given key string array
 * @return Corresponding index
 * @param strv Null terminated array of gchar *
 * @param key The key string value
 */
extern gint
find_key_strv (const gchar ** strv, const gchar * key);

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
 * @return TRUE if successfully get the version
 */
extern gboolean
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
#endif /* __NNS_PLUGIN_API_UTIL_H__ */
