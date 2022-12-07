/**
 * NNStreamer Subplugin Manager
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	nnstreamer_subplugin.h
 * @date	27 Nov 2018
 * @brief	Subplugin Manager for NNStreamer
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * To Packagers:
 *
 * This file is to be packaged as "devel" package for NN developers. (subplugin writers)
 *
 * @note        Any independent subplugin (existing as an independent .so)
 *              should call register_subplugin () (or its wrapper) with the
 *              subplugin's constructor function.
 */
#ifndef __GST_NNSTREAMER_SUBPLUGIN_H__
#define __GST_NNSTREAMER_SUBPLUGIN_H__

#include <stdint.h>
#include "nnstreamer_conf.h"

G_BEGIN_DECLS

typedef enum {
  NNS_SUBPLUGIN_FILTER = NNSCONF_PATH_FILTERS,
  NNS_SUBPLUGIN_DECODER = NNSCONF_PATH_DECODERS,
  NNS_EASY_CUSTOM_FILTER = NNSCONF_PATH_EASY_CUSTOM_FILTERS,
  NNS_SUBPLUGIN_CONVERTER = NNSCONF_PATH_CONVERTERS,
  NNS_SUBPLUGIN_TRAINER = NNSCONF_PATH_TRAINERS,
  NNS_CUSTOM_CONVERTER,
  NNS_CUSTOM_DECODER,
  NNS_IF_CUSTOM,

  NNS_SUBPLUGIN_END,
} subpluginType;

#define NNS_SUBPLUGIN_CHECKER (0xdeadbeef)

/**
 * @brief Retrieve the registered data with the subplugin name.
 * @param[in] type Subplugin Type
 * @param[in] name Subplugin Name. The filename should be libnnstreamer_${type}_${name}.so
 * @return The registered data
 */
extern const void *
get_subplugin (subpluginType type, const char *name);

/**
 * @brief Get the list of registered subplugins.
 * @param[in] type Subplugin Type
 * @return The list of subplugin name
 * @note Caller should free the returned value using g_strfreev()
 */
extern gchar **
get_all_subplugins (subpluginType type);

/**
 * @brief Register the subplugin. If duplicated name exists, it is rejected.
 * @param[in] type Subplugin Type
 * @param[in] name Subplugin Name. The filename should be subplugin_prefixes[type]${name}.so
 * @param[in] data The registered data
 * @return TRUE if registered as new. FALSE if duplicated (overwritten/updated).
 */
extern gboolean
register_subplugin (subpluginType type, const char *name, const void *data);

/**
 * @brief Unregister the subplugin.
 * @param[in] type Subplugin type
 * @param[in] name Subplugin Name. The filename should be subplugin_prefixes[type]${name}.so
 * @return TRUE if unregistered. FALSE if rejected or error.
 *
 * @warning Subplugins checked out with get_subplugins can still be used after unregister.
 */
extern gboolean
unregister_subplugin (subpluginType type, const char *name);

extern void
subplugin_set_custom_property_desc (subpluginType type, const char *name,
    const gchar * prop, va_list varargs);

extern GData *
subplugin_get_custom_property_desc (subpluginType type, const char *name);

G_END_DECLS
#endif /* __GST_NNSTREAMER_SUBPLUGIN_H__ */
