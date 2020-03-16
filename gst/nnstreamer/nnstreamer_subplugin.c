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
 * @file	nnstreamer_subplugin.c
 * @date	27 Nov 2018
 * @brief	Subplugin Manager for NNStreamer
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <dlfcn.h>
#if !defined(__clang__) && (defined(__GNUC__) || defined(__GNUG__))
#include <features.h>           /* Check libc version for dlclose-related workaround */
#endif
#include <glib.h>

#include "nnstreamer_subplugin.h"
#include "nnstreamer_conf.h"

/** @brief Array of dlopen-ed handles */
static GPtrArray *handles = NULL;

static void init_subplugin (void) __attribute__ ((constructor));
static void fini_subplugin (void) __attribute__ ((destructor));

typedef struct
{
  char *name; /**< The name of subplugin */
  const void *data; /**< subplugin specific data forwarded from the subplugin */
} subpluginData;

static GHashTable *subplugins[NNS_SUBPLUGIN_END] = { 0 };

/** @brief Protects handles and subplugins */
G_LOCK_DEFINE_STATIC (splock);

/** @brief Private function for g_hash_table data destructor, GDestroyNotify */
static void
_spdata_destroy (gpointer _data)
{
  subpluginData *data = _data;
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
  void *handle;

  g_return_val_if_fail (name != NULL, NULL);
  g_return_val_if_fail (path != NULL, NULL);

  dlerror ();
  handle = dlopen (path, RTLD_NOW);
  /* If this is a correct subplugin, it will register itself */
  if (handle == NULL) {
    g_critical ("Cannot dlopen %s(%s) with error %s.", name, path, dlerror ());
    return NULL;
  }

  spdata = _get_subplugin_data (type, name);
  if (spdata) {
    g_ptr_array_add (handles, (gpointer) handle);
  } else {
    g_critical
        ("nnstreamer_subplugin of %s(%s) is broken. It does not call register_subplugin with its init function.",
        name, path);
    dlclose (handle);
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
      break;
    default:
      /* unknown sub-plugin type */
      return FALSE;
  }

  /* check the sub-pugin name */
  if (g_ascii_strcasecmp (name, "auto") == 0) {
    g_critical ("Failed, the name %s is not allowed.", name);
    return FALSE;
  }

  spdata = _get_subplugin_data (type, name);
  if (spdata) {
    /* already exists */
    g_warning ("Subplugin %s is already registered.", name);
    return FALSE;
  }

  spdata = g_new0 (subpluginData, 1);
  if (spdata == NULL) {
    g_critical ("Failed to allocate memory for subplugin registration.");
    return FALSE;
  }

  spdata->name = g_strdup (name);
  spdata->data = data;

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

/** @brief dlclose as dealloc function for handles */
static void
_dlclose (gpointer data)
{
/**
 * Ubuntu 16.04 / GLIBC 2.23 Workaround
 * If we do dlclose at exit() function, it may incur
 * https://bugzilla.redhat.com/show_bug.cgi?id=1264556#c42
 * , which is a GLIBC bug at 2.23.
 * The corresponding error message is:
 * Inconsistency detected by ld.so: dl-close.c: 811:
 * _dl_close: Assertion `map->l_init_called' failed!
 */
#if defined(__GLIBC__) && (__GLIBC__ == 2) && (__GLIBC_MINOR__ == 23)
  return;                       /* Do not call dlclose and return */
#else
  void *ptr = data;
  dlclose (ptr);
#endif
}

/** @brief Create handles at the start of library */
static void
init_subplugin (void)
{
  G_LOCK (splock);
  g_assert (NULL == handles);
  handles = g_ptr_array_new_full (16, _dlclose);
  G_UNLOCK (splock);
}

/** @brief Free handles at the start of library */
static void
fini_subplugin (void)
{
  G_LOCK (splock);
  g_assert (handles);

  /* iterate and call dlclose by calling g_array_clear */
  g_ptr_array_free (handles, TRUE);
  handles = NULL;
  G_UNLOCK (splock);
}
