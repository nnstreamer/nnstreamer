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
#include <glib.h>
#include <gmodule.h>
#include <gst/gstinfo.h>
#include "nnstreamer_subplugin.h"
#include "nnstreamer_conf.h"

typedef struct
{
  char *name; /**< The name of subplugin */
  const void *data; /**< subplugin specific data forwarded from the subplugin */
  void *handle; /**< dlopen'ed handle */
} subpluginData;

static GHashTable *subplugins[NNS_SUBPLUGIN_END] = { 0 };

G_LOCK_DEFINE_STATIC (splock);

/** @brief Private function for g_hash_table data destructor, GDestroyNotify */
static void
_spdata_destroy (gpointer _data)
{
  subpluginData *data = _data;
  g_free (data->name);
  if (data->handle)
    dlclose (data->handle);
  g_free (data);
}

/** @brief Public function defined in the header */
const void *
get_subplugin (subpluginType type, const char *name)
{
  GHashTable *table;
  subpluginData *data;
  void *handle;

  g_return_val_if_fail (name, NULL);

  G_LOCK (splock);

  if (subplugins[type] == NULL)
    subplugins[type] =
        g_hash_table_new_full (g_str_hash, g_str_equal, g_free,
        _spdata_destroy);

  table = subplugins[type];
  data = g_hash_table_lookup (table, name);

  if (data == NULL) {
    /** Search and register if found with the conf */
    nnsconf_type_path conf_type = (nnsconf_type_path) type;
    const gchar *fullpath = nnsconf_get_fullpath (name, conf_type);

    if (!nnsconf_validate_file (conf_type, fullpath))
      goto error;               /* No Such Thing !!! */

    G_UNLOCK (splock);

    /** clear any existing errors */
    dlerror ();
    handle = dlopen (fullpath, RTLD_NOW);
    if (NULL == handle) {
      GST_ERROR ("Cannot dlopen %s (%s) with error %s.", name, fullpath,
          dlerror ());
      return NULL;
    }

    G_LOCK (splock);

    /* If a subplugin's constructor has called register_subplugin, skip the rest */
    data = g_hash_table_lookup (table, name);
    if (data == NULL) {
      GST_ERROR
          ("nnstreamer_subplugin of %s (%s) is broken. It does not call register_subplugin with its init function.",
          name, fullpath);
      goto error_handle;
    }
  }

  G_UNLOCK (splock);
  return data->data;

error_handle:
  dlclose (handle);
error:
  G_UNLOCK (splock);
  return NULL;
}

/** @brief Public function defined in the header */
gboolean
register_subplugin (subpluginType type, const char *name, const void *data)
{
  /** @todo data out of scope at add */
  subpluginData *spdata;
  gboolean ret;

  g_return_val_if_fail (name, FALSE);
  g_return_val_if_fail (data, FALSE);

  switch (type) {
    case NNS_SUBPLUGIN_FILTER:
    case NNS_SUBPLUGIN_DECODER:
      break;
    default:
      /* unknown sub-plugin type */
      return FALSE;
  }

  if (subplugins[type] == NULL) {
    subplugins[type] =
        g_hash_table_new_full (g_str_hash, g_str_equal, g_free,
        _spdata_destroy);
  } else {
    subpluginData *data;

    G_LOCK (splock);
    data = g_hash_table_lookup (subplugins[type], name);
    G_UNLOCK (splock);

    if (data) {
      /* already exists */
      GST_ERROR ("Subplugin %s is already registered.", name);
      return FALSE;
    }
  }

  spdata = g_new (subpluginData, 1);
  g_assert (spdata);

  spdata->name = g_strdup (name);
  spdata->data = data;
  spdata->handle = NULL;

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
