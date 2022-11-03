/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer Common API Header for Plug-Ins
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
 * Copyright (C) 2019 Wook Song <wook16.song@samsung.com>
 */
/**
 * @file  nnstreamer_plugin_api.h
 * @date  24 Jan 2019
 * @brief Optional/Additional NNStreamer APIs for sub-plugin writers. (Need Gst devel)
 * @see https://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com> and Wook Song <wook16.song@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __NNS_PLUGIN_API_H__
#define __NNS_PLUGIN_API_H__

#include <glib.h>
#include <gst/gst.h>
#include <tensor_typedef.h>
#include <nnstreamer_version.h>
#include <nnstreamer_plugin_api_util.h>

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
 * @brief set alignment that default allocator would align to
 * @param alignment bytes of alignment
 */
extern void gst_tensor_alloc_init (gsize alignment);

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
 * @brief Update caps dimension for negotiation
 * @param caps caps to compare and update
 * @param peer_caps caps to compare
 */
extern void
gst_tensor_caps_update_dimension (GstCaps *caps, GstCaps *peer_caps);

/**
 * @brief  Try intersecting @caps1 and @caps2 for tensor stream
 * @param caps1 a GstCaps to intersect
 * @param caps2 a GstCaps to intersect
 * @return TRUE if intersection would be not empty.
 */
extern gboolean
gst_tensor_caps_can_intersect (GstCaps *caps1, GstCaps *caps2);

G_END_DECLS
#endif /* __NNS_PLUGIN_API_H__ */
