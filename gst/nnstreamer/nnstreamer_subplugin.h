/**
 * NNStreamer Subplugin Manager
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	nnstreamer_subplugin.h
 * @date	27 Nov 2018
 * @brief	Subplugin Manager for NNStreamer
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * To Packagers:
 *
 * This fils it to be packaged as "devel" package for NN developers. (subplugin writers)
 *
 * @note        Any independent subplugin (existing as an independent .so)
 *              should have an exported variable "nnstreamer_subplugin",
 *              whose type is nnstreamer_subplugin_data.
 *              A built-in subplugin (existing in libnnstreamer.so) does not
 *              need this.
 */
#ifndef __GST_NNSTREAMER_SUBPLUGIN_H__
#define __GST_NNSTREAMER_SUBPLUGIN_H__

#include <stdint.h>
#include "nnstreamer_conf.h"

typedef enum {
  NNS_SUBPLUGIN_FILTER = NNSCONF_FILTERS,
  NNS_SUBPLUGIN_DECODER = NNSCONF_DECODERS,

  NNS_SUBPLUGIN_END,
} subpluginType;

#define NNS_SUBPLUGIN_CHECKER (0xdeadbeef)

/**
 * @brief Any independent shared object subplugin must export this as
 * "nnstreamer_subplugin"
 */
typedef struct {
  uint32_t checker; /**< This MUST be 0xdeadbeef (NNS_SUBPLUGIN_CHECKER) */
  subpluginType type; /**< The subplugin type */
  const char *name; /**< The name of subplugin. The resulting .so file should be ${prefix}${name}.so */
  void *data; /**< Subplugin specific data (usually funcptr callbacks), forwarded to subplugin's corresponding handler */
} nnstreamer_subplugin_data;

/**
 * @brief Retrieve the resigered data with the subplugin name.
 * @param[in] type Subplugin Type
 * @param[in] name Subplugin Name. The filename should be libnnstreamer_${type}_${name}.so
 * @return The registered data
 */
extern const void *
get_subplugin (subpluginType type, const char *name);

/**
 * @brief Register the subplugin. If duplicated name exists, it is rejected.
 * @param[in] type Subplugin Type
 * @param[in] name Subplugin Name. The filename should be subplugin_prefixes[type]${name}.so
 * @param[in] data The registered data
 * @return TRUE if regitered as new. FALSE if duplicated (overwritten/updated).
 */
extern gboolean
register_subplugin (subpluginType type, const char *name, const void *data);

/**
 * @brief Unregister the subplugin.
 * @param[in] type Subplugin type
 * @param[in] name Subplugin Name. The filename should be subplugin_prefixes[type]${name}.so
 * @return TRUE if unregitered. FALSE if rejected or error.
 *
 * @warning Subplugins checked out with get_subplugins can still be used after unregister.
 */
extern gboolean
unregister_subplugin (subpluginType type, const char *name);

/**
 * @brief Call this at subplugin probe to avoid duplicated registration
 */
extern void hold_register_subplugin (subpluginType type, const char *name, void *data);

#endif /* __GST_NNSTREAMER_SUBPLUGIN_H__ */
