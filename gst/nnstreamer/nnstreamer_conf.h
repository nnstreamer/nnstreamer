/**
 * NNStreamer Configurations / Environmental Variable Manager.
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
 * @file	nnstreamer_conf.h
 * @date	26 Nov 2018
 * @brief	Internal header for conf/env-var management.
 * @see		https://github.com/nnstreamer/nnstreamer
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
 * Do not export this to devel package. This is an internal header.
 */
#ifndef __GST_NNSTREAMER_CONF_H__
#define __GST_NNSTREAMER_CONF_H__

#include <glib.h>
G_BEGIN_DECLS

/* Hard-coded system-dependent root path prefix */
#ifdef G_OS_WIN32
#define NNSTREAMER_SYS_ROOT_PATH_PREFIX "c:\\"
#else
#define NNSTREAMER_SYS_ROOT_PATH_PREFIX "/"
#endif /* G_OS_WIN32 */

/**
 * Hard-coded system-dependent file extension string of shared
 * (dynamic loadable) object
 */
#ifdef __MACOS__
#define NNSTREAMER_SO_FILE_EXTENSION	".dylib"
#else
#define NNSTREAMER_SO_FILE_EXTENSION	".so"
#endif

/* Internal Hardcoded Values */
#define NNSTREAMER_DEFAULT_CONF_FILE    "/etc/nnstreamer.ini"
#ifndef NNSTREAMER_CONF_FILE
#define NNSTREAMER_CONF_FILE NNSTREAMER_DEFAULT_CONF_FILE
#endif
#define NNSTREAMER_ENVVAR_CONF_FILE     "NNSTREAMER_CONF"

typedef enum {
  NNSCONF_PATH_FILTERS = 0,
  NNSCONF_PATH_DECODERS,
  NNSCONF_PATH_CUSTOM_FILTERS,
  NNSCONF_PATH_EASY_CUSTOM_FILTERS,
  NNSCONF_PATH_CONVERTERS,
  NNSCONF_PATH_TRAINERS,

  NNSCONF_PATH_END,
} nnsconf_type_path;

/* struct for sub-plugins info (name and full path) */
typedef struct
{
  gchar **names;
  gchar **paths;
} subplugin_info_s;

/**
 * @brief Load the .ini file
 * @param[in] force_reload TRUE if you want to clean up and load conf again.
 * @return TRUE if successful or skipped. FALSE if error reading something.
 */
extern gboolean
nnsconf_loadconf (gboolean force_reload);

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
nnsconf_get_fullpath (const gchar * subpluginname, nnsconf_type_path type);

/**
 * @brief Public function to validate sub-plugin library is available.
 * @param[in] type The type (FILTERS/DECODERS/CUSTOM_FILTERS)
 * @param[in] fullpath The full path to the file.
 * @return True if the file is regular and can be added to the list.
 */
extern gboolean
nnsconf_validate_file (nnsconf_type_path type, const gchar * fullpath);

/**
 * @brief Get sub-plugin's name prefix.
 * @param[in] type The type (FILTERS/DECODERS/CUSTOM_FILTERS)
 * @return Predefined prefix string for given type.
 */
extern const gchar *
nnsconf_get_subplugin_name_prefix (nnsconf_type_path type);

/**
 * @brief Public function to get the list of sub-plugins name and path
 * @param[in] type The type (FILTERS/DECODERS/CUSTOM_FILTERS)
 * @param[out] info The data structure which contains the name and full path of sub-plugins
 * @return total number of sub-plugins for given type
 * @note DO NOT free sub-plugins info
 */
extern guint
nnsconf_get_subplugin_info (nnsconf_type_path type, subplugin_info_s * info);

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
nnsconf_get_custom_value_string (const gchar * group, const gchar * key);

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
nnsconf_get_custom_value_bool (const gchar * group, const gchar * key, gboolean def);

/**
 * @brief NNStreamer configuration dump as string.
 * @param[out] str Preallocated string for the output (dump).
 * @param[in] size The size of given str.
 */
extern void
nnsconf_dump (gchar * str, gulong size);

extern void
nnsconf_subplugin_dump (gchar * str, gulong size);

G_END_DECLS
#endif /* __GST_NNSTREAMER_CONF_H__ */
