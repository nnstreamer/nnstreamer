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
 * @file	nnstreamer_subplugin.c
 * @date	27 Nov 2018
 * @brief	Subplugin Manager for NNStreamer
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <glib.h>
#include <gmodule.h>

#include "nnstreamer_log.h"
#include "nnstreamer_subplugin.h"
#include "nnstreamer_conf.h"
#include <nnstreamer_util.h>

/** @brief Array of dynamic loaded handles */
static GPtrArray *handles = NULL;

static void init_subplugin (void) __attribute__((constructor));
static void fini_subplugin (void) __attribute__((destructor));

typedef struct
{
  char *name; /**< The name of subplugin */
  const void *data; /**< subplugin specific data forwarded from the subplugin */
  GData *custom_dlist; /**< [OPTIONAL] subplugin specific custom property desc list */
} subpluginData;

static GHashTable *subplugins[NNS_SUBPLUGIN_END] = { 0 };

/** @brief Protects handles and subplugins */
G_LOCK_DEFINE_STATIC (splock);

/** @brief Private function for g_hash_table data destructor, GDestroyNotify */
static void
_spdata_destroy (gpointer _data)
{
  subpluginData *data = _data;

  g_datalist_clear (&data->custom_dlist);

  g_free (data->name);
  g_free (data);
}

typedef enum
{
  NNS_SEARCH_FILENAME,
  NNS_SEARCH_GETALL,
  NNS_SEARCH_NO_OP,
} subpluginSearchLogic;

static subpluginSearchLogic searchAlgorithm[] = {
  [NNS_SUBPLUGIN_FILTER] = NNS_SEARCH_FILENAME,
  [NNS_SUBPLUGIN_DECODER] = NNS_SEARCH_FILENAME,
  [NNS_EASY_CUSTOM_FILTER] = NNS_SEARCH_FILENAME,
  [NNS_SUBPLUGIN_CONVERTER] = NNS_SEARCH_GETALL,
  [NNS_SUBPLUGIN_TRAINER] = NNS_SEARCH_FILENAME,
  [NNS_CUSTOM_CONVERTER] = NNS_SEARCH_NO_OP,
  [NNS_CUSTOM_DECODER] = NNS_SEARCH_NO_OP,
  [NNS_IF_CUSTOM] = NNS_SEARCH_NO_OP,
  [NNS_SUBPLUGIN_END] = NNS_SEARCH_NO_OP,
};

/**
 * @brief Internal function to get sub-plugin data.
 */
static subpluginData *
_get_subplugin_data (subpluginType type, const gchar * name)
{
  subpluginData *spdata = NULL;

  G_LOCK (splock);
  if (subplugins[type] == NULL) {
    subplugins[type] =
        g_hash_table_new_full (g_str_hash, g_str_equal, g_free,
        _spdata_destroy);
  } else {
    spdata = g_hash_table_lookup (subplugins[type], name);
  }
  G_UNLOCK (splock);

  return spdata;
}

/**
 * @brief Internal function to scan sub-plugin.
 */
static subpluginData *
_search_subplugin (subpluginType type, const gchar * name, const gchar * path)
{
  subpluginData *spdata = NULL;
  GModule *module;

  g_return_val_if_fail (name != NULL, NULL);
  g_return_val_if_fail (path != NULL, NULL);

  module = g_module_open (path, G_MODULE_BIND_LOCAL);
  /* If this is a correct subplugin, it will register itself */
  if (module == NULL) {
    ml_loge ("Cannot open %s(%s) with error %s.", name, path,
        g_module_error ());
    return NULL;
  }

  spdata = _get_subplugin_data (type, name);
  if (spdata) {
    g_ptr_array_add (handles, (gpointer) module);
  } else {
    ml_loge
        ("nnstreamer_subplugin of %s(%s) is broken. It does not call register_subplugin with its init function.",
        name, path);
    g_module_close (module);
  }

  return spdata;
}

/** @brief Public function defined in the header */
const void *
get_subplugin (subpluginType type, const char *name)
{
  subpluginData *spdata = NULL;

  g_return_val_if_fail (name, NULL);

  if (searchAlgorithm[type] == NNS_SEARCH_GETALL) {
    nnsconf_type_path conf_type = (nnsconf_type_path) type;
    subplugin_info_s info;
    guint i;
    guint ret = nnsconf_get_subplugin_info (conf_type, &info);

    for (i = 0; i < ret; i++) {
      _search_subplugin (type, info.names[i], info.paths[i]);
    }

    searchAlgorithm[type] = NNS_SEARCH_NO_OP;
  }

  spdata = _get_subplugin_data (type, name);
  if (spdata == NULL && searchAlgorithm[type] == NNS_SEARCH_FILENAME) {
    /** Search and register if found with the conf */
    nnsconf_type_path conf_type = (nnsconf_type_path) type;
    const gchar *fullpath = nnsconf_get_fullpath (name, conf_type);

    if (nnsconf_validate_file (conf_type, fullpath)) {
      spdata = _search_subplugin (type, name, fullpath);
    }
  }

  return (spdata != NULL) ? spdata->data : NULL;
}

/** @brief Public function defined in the header */
gchar **
get_all_subplugins (subpluginType type)
{
  GString *names;
  subplugin_info_s info;
  gchar **list = NULL;
  gchar *name;
  guint i, total;

  names = g_string_new (NULL);

  /* get registered subplugins */
  G_LOCK (splock);
  if (subplugins[type]) {
    list = (gchar **) g_hash_table_get_keys_as_array (subplugins[type], NULL);
  }
  G_UNLOCK (splock);

  if (list) {
    name = g_strjoinv (",", list);
    g_string_append (names, name);
    g_free (name);
  }

  /* get subplugins from configuration */
  total = nnsconf_get_subplugin_info ((nnsconf_type_path) type, &info);

  for (i = 0; i < total; i++) {
    name = info.names[i];

    if (!list || !g_strv_contains ((const gchar * const *) list, name)) {
      if (list || i > 0)
        g_string_append (names, ",");

      g_string_append (names, name);
    }
  }

  g_free (list);

  /* finally get the list of subplugins */
  name = g_string_free (names, FALSE);
  list = g_strsplit (name, ",", -1);
  g_free (name);

  return list;
}

/** @brief Public function defined in the header */
gboolean
register_subplugin (subpluginType type, const char *name, const void *data)
{
  /** @todo data out of scope at add */
  subpluginData *spdata = NULL;
  gboolean ret;

  g_return_val_if_fail (name, FALSE);
  g_return_val_if_fail (data, FALSE);

  switch (type) {
    case NNS_SUBPLUGIN_FILTER:
    case NNS_SUBPLUGIN_DECODER:
    case NNS_EASY_CUSTOM_FILTER:
    case NNS_SUBPLUGIN_CONVERTER:
    case NNS_SUBPLUGIN_TRAINER:
    case NNS_CUSTOM_DECODER:
    case NNS_IF_CUSTOM:
    case NNS_CUSTOM_CONVERTER:
      break;
    default:
      /* unknown sub-plugin type */
      return FALSE;
  }

  /* check the sub-pugin name */
  if (g_ascii_strcasecmp (name, "any") == 0 ||
      g_ascii_strcasecmp (name, "auto") == 0) {
    ml_loge ("Failed, the name %s is not allowed.", name);
    return FALSE;
  }

  spdata = _get_subplugin_data (type, name);
  if (spdata) {
    /* already exists */
    ml_logw ("Subplugin %s is already registered.", name);
    return FALSE;
  }

  spdata = g_new0 (subpluginData, 1);
  if (spdata == NULL) {
    ml_loge ("Failed to allocate memory for subplugin registration.");
    return FALSE;
  }

  spdata->name = g_strdup (name);
  spdata->data = data;
  g_datalist_init (&spdata->custom_dlist);

  G_LOCK (splock);
  ret = g_hash_table_insert (subplugins[type], g_strdup (name), spdata);
  G_UNLOCK (splock);

  return ret;
}

/** @brief Public function defined in the header */
gboolean
unregister_subplugin (subpluginType type, const char *name)
{
  gboolean ret;

  g_return_val_if_fail (name, FALSE);
  g_return_val_if_fail (subplugins[type], FALSE);

  G_LOCK (splock);
  ret = g_hash_table_remove (subplugins[type], name);
  G_UNLOCK (splock);

  return ret;
}

/** @brief dealloc function for handles */
static void
_close_handle (gpointer data)
{
/**
 * Ubuntu 16.04 / GLIBC 2.23 Workaround
 * If we do dlclose at exit() function, it may incur
 * https://bugzilla.redhat.com/show_bug.cgi?id=1264556#c42
 * , which is a GLIBC bug at 2.23.
 * The corresponding error message is:
 * Inconsistency detected by ld.so: dl-close.c: 811:
 * _dl_close: Assertion `map->l_init_called' failed!
 * Note that Tizen 5.5 / GLIBC 2.24 has the same bug!
 */
#if defined(__GLIBC__) && (__GLIBC__ == 2) && (__GLIBC_MINOR__ <= 24)
  UNUSED (data);
  return;                       /* Do not call close and return */
#else
  g_module_close ((GModule *) data);
#endif
}

/**
 * @brief common interface to set custom property description of a sub-plugin.
 */
void
subplugin_set_custom_property_desc (subpluginType type, const char *name,
    const gchar * prop, va_list varargs)
{
  subpluginData *spdata;

  g_return_if_fail (name != NULL);
  g_return_if_fail (subplugins[type] != NULL);

  spdata = _get_subplugin_data (type, name);
  g_return_if_fail (spdata != NULL);

  g_datalist_clear (&spdata->custom_dlist);

  while (prop) {
    gchar *desc = va_arg (varargs, gchar *);

    if (G_UNLIKELY (desc == NULL)) {
      ml_logw ("No description for %s", prop);
      return;
    }

    g_datalist_set_data (&spdata->custom_dlist, prop, desc);
    prop = va_arg (varargs, gchar *);
  }
}

/**
 * @brief common interface to get custom property description of a sub-plugin.
 */
GData *
subplugin_get_custom_property_desc (subpluginType type, const char *name)
{
  subpluginData *spdata;

  g_return_val_if_fail (name != NULL, NULL);
  g_return_val_if_fail (subplugins[type] != NULL, NULL);

  spdata = _get_subplugin_data (type, name);
  if (spdata)
    return spdata->custom_dlist;

  return NULL;
}

/** @brief Create handles at the start of library */
static void
init_subplugin (void)
{
  G_LOCK (splock);
  g_assert (NULL == handles); /** Internal error (duplicated init call?) */
  handles = g_ptr_array_new_full (16, _close_handle);
  G_UNLOCK (splock);
}

/** @brief Free handles at the start of library */
static void
fini_subplugin (void)
{
  G_LOCK (splock);
  g_assert (handles); /** Internal error (init not called?) */

  /* iterate and call close by calling g_array_clear */
  g_ptr_array_free (handles, TRUE);
  handles = NULL;
  G_UNLOCK (splock);
}
