/**
 * NNStreamer Configurations / Environmental Variable Manager.
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
 * @file	nnstreamer_conf.h
 * @date	26 Nov 2018
 * @brief	Internal header for conf/env-var management.
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * If there are duplicated configurations for the same element,
 * the one with higher priority may override (if it cannot be stacked up),
 *
 * - (Highest) Environmental variables
 * - The configuration file (default: /etc/nnstreamer.ini)
 * - (Lowest) Internal hardcoded values.
 *
 * Do not export this to devel pacakge. This is an internal header.
 */
#ifndef __GST_NNSTREAMER_CONF_H__
#define __GST_NNSTREAMER_CONF_H__

#include <glib.h>
G_BEGIN_DECLS

/* Env-var names */
#define NNSTREAMER_ENVVAR_CONF_FILE     "NNSTREAMER_CONF"
#define NNSTREAMER_ENVVAR_FILTERS       "NNSTREAMER_FILTERS"
#define NNSTREAMER_ENVVAR_DECODERS      "NNSTREAMER_DECODERS"
#define NNSTREAMER_ENVVAR_CUSTOMFILTERS "NNSTREAMER_CUSTOMFILTERS"
#define NNSTREAMER_ENVVAR_TF_MEM_OPTMZ  "NNSTREAMER_TF_MEM_OPTMZ"

/* Internal Hardcoded Values */
#define NNSTREAMER_DEFAULT_CONF_FILE    "/etc/nnstreamer.ini"
#define NNSTREAMER_FILTERS              "/usr/lib/nnstreamer/filters/"
#define NNSTREAMER_DECODERS             "/usr/lib/nnstreamer/decoders/"
#define NNSTREAMER_CUSTOM_FILTERS       "/usr/lib/nnstreamer/customfilters/"
#define NNSTREAMER_TF_MEM_OPTMZ         "false"
/**
 *  Note that users still can place their custom filters anywhere if they
 * designate them with the full path.
 */

/* Subplugin Naming Rules */
#define NNSTREAMER_PREFIX_DECODER	"libnnstreamer_decoder_"
#define NNSTREAMER_PREFIX_FILTER	"libnnstreamer_filter_"
#define NNSTREAMER_PREFIX_CUSTOMFILTERS	""
/* Custom filter does not have prefix */

typedef enum {
  NNSCONF_PATH_FILTERS = 0,
  NNSCONF_PATH_DECODERS,
  NNSCONF_PATH_CUSTOM_FILTERS,
  NNSCONF_PATH_CONFFILE,

  NNSCONF_PATH_END,
} nnsconf_type_path;

typedef enum {
  NNSCONF_VAL_TF_MEM_OPTMZ = 0,

  NNSCONF_VAL_END,
} nnsconf_type_value;

extern const gchar *subplugin_prefixes[NNSCONF_PATH_END];

/**
 * @brief Load the .ini file
 * @param[in] force_reload TRUE if you want to clean up and load conf again.
 * @return TRUE if successful or skipped. FALSE if error reading something.
 */
extern gboolean
nnsconf_loadconf (gboolean force_reload);

/**
 * @brief Search for "file2find" file in the configured paths for the type
 * @param[in] file2find The filename including extensions (e.g., .so) to find.
 * @param[in] type The type (FILTERS/DECODERS/CUSTOM_FILTERS)
 * @return The full path to the file. Caller MUST NOT modify this.
 *         Returns NULL if we cannot find the file.
 */
extern const gchar *
nnsconf_get_fullpath_fromfile (const gchar *file2find, nnsconf_type_path type);

/**
 * @brief Get the configured paths for the type with sub-plugin name.
 * @param[in] The subplugin name except for the prefix and postfix (.so) to find
 * @param[in] type The type (FILTERS/DECODERS/CUSTOM_FILTERS)
 * @return The full path to the file. Caller MUST NOT modify this.
 *         Returns NULL if we cannot find the file.
 *
 * This is mainly supposed to be used by CUSTOM_FILTERS
 */
extern const gchar *
nnsconf_get_fullpath (const gchar *subpluginname, nnsconf_type_path type);

/**
 * @brief Get the configured boolean values for the type
 * @param[in] type The type (TF_MEM_OPTMZ)
 * @return The boolean value to the file. Caller MUST NOT modify this.
 *         Returns FALSE if we cannot find the file or the value as a DEFAULT.
 */
extern gboolean
nnsconf_get_value_bool (nnsconf_type_value type);

/**
 * @brief Get the custom configuration value from .ini and envvar.
 * @detail For predefined configurations defined in this header,
 *         use the given enum for faster configuration processing.
 *         For custom configurations not defined in this header,
 *         you may use this API to access your own custom configurations.
 *         Configuration values may be loaded only once during runtime,
 *         thus, if the values are changed in run-time, the changes are
 *         not guaranteed to be reflected.
 *         The ENVVAR is supposed to be NNSTREAMER_${group}_${key}, which
 *         has higher priority than the .ini configuration.
 *         Be careful not to use special characters in group name ([, ], _).
 * @param[in] group The group name, [group], in .ini file.
 * @param[in] key The key name, key = value, in .ini file.
 * @return The newly allocated string. A caller must free it. NULL if it's not available.
 */
extern gchar *
nnsconf_get_custom_value_string (const gchar *group, const gchar *key);

/**
 * @brief Get the custom configuration value from .ini and envvar.
 * @detail For predefined configurations defined in this header,
 *         use the given enum for faster configuration processing.
 *         For custom configurations not defined in this header,
 *         you may use this API to access your own custom configurations.
 *         Configuration values may be loaded only once during runtime,
 *         thus, if the values are changed in run-time, the changes are
 *         not guaranteed to be reflected.
 *         The ENVVAR is supposed to be NNSTREAMER_${group}_${key}, which
 *         has higher priority than the .ini configuration.
 *         Be careful not to use special characters in group name ([, ], _).
 * @param[in] group The group name, [group], in .ini file.
 * @param[in] key The key name, key = value, in .ini file.
 * @param[in] def The default return value in case there is no value available.
 * @return The value interpreted as TRUE/FALSE.
 */
extern gboolean
nnsconf_get_custom_value_bool (const gchar *group, const gchar *key, gboolean def);

G_END_DECLS
#endif /* __GST_NNSTREAMER_CONF_H__ */
