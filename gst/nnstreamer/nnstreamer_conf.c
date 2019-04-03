/** * NNStreamer Configurations / Environmental Variable Manager.
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
 * @file	nnstreamer_conf.c
 * @date	26 Nov 2018
 * @brief	NNStreamer Configuration (conf file, env-var) Management.
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <glib.h>
#include <glib/gprintf.h>
#include <glib/gstdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <gmodule.h>
#include "nnstreamer_conf.h"

#define CONF_SOURCES (3)
typedef struct
{
  gboolean loaded;            /**< TRUE if loaded at least once */

  gchar *conffile;            /**< Location of conf file. */

  /*************************************************
   * Cached Raw Values                             *
   * 0: ENVVAR. 1: CONFFILE, 2: Hardcoded          *
   *************************************************/
  gchar *pathFILTERS[CONF_SOURCES];        /**< directory paths for FILTERS */
  gchar *pathDECODERS[CONF_SOURCES];       /**< directory paths for DECODERS */
  gchar *pathCUSTOM_FILTERS[CONF_SOURCES]; /**< directory paths for CUSTOM FILTERS */
  gchar *valueTF_MEM_OPTMZ[CONF_SOURCES];  /**< value of TF_MEM_OPTMZ */

  /*************************************************
   * Processed Values                              *
   *************************************************/
  gchar **filesFILTERS;        /**< Null terminated list of full filepaths for FILTERS */
  gchar **filesDECODERS;;      /**< Null terminated list of full filepaths for DECODERS */
  gchar **filesCUSTOM_FILTERS; /**< Null terminated list of full filepaths for CUSTOM FILTERS */

  gchar **basenameFILTERS;        /**< Null terminated list of basenames for FILTERS */
  gchar **basenameDECODERS;;      /**< Null terminated list of basenames for DECODERS */
  gchar **basenameCUSTOM_FILTERS; /**< Null terminated list of basenames for CUSTOM FILTERS */

  gboolean boolTF_MEM_OPTMZ; /**< The flag to decide using memcpy or not at Tensorflow in boolean type */
} confdata;

static confdata conf = { 0 };

/**
 * @brief Private function to get strdup-ed env-var if it's valid.
 *        Otherwise, NULL
 *
 * @retval strdup-ed env-var value
 * @param[in] name Environmetal variable name
 */
static gchar *
_strdup_getenv (const char *name)
{
  /**
   * @todo Evaluate if we need to use secure_getenv() here
   *  (and compatible with other OS
   */
  const gchar *tmp = g_getenv (name);

  if (tmp == NULL)
    return NULL;
  return g_strdup (tmp);
}

/**
 * @brief Private function to fill in ".so list" with fullpath-filenames in a directory.
 * @param[in] dir Directory to be searched
 * @param[in] list The filename list to be updated
 * @param[in/out] counter increased by the number of appended elements.
 * @return The list updated.
 * @todo This assumes .so for all sub plugins. Support Windows/Mac/iOS!
 */
static GSList *
_get_fullpath_filenames (const gchar * dir, GSList * list, uint32_t * counter)
{
  GDir *gdir = g_dir_open (dir, 0U, NULL);
  const gchar *name;
  gchar *dirfullpath;

  if (gdir == NULL)
    return list;

  if (dir[strlen (dir) - 1] != '/')
    dirfullpath = g_strconcat (dir, "/", NULL);
  else
    dirfullpath = g_strdup (dir);

  while (NULL != (name = g_dir_read_name (gdir))) {
    list = g_slist_prepend (list, g_strconcat (dirfullpath, name, NULL));
    *counter = *counter + 1;
  }
  g_free (dirfullpath);
  g_dir_close (gdir);
  return list;
}

/**
 * @brief Private function to fill in ".so list" with basenames in a directory.
 * @param[in] dir Directory to be searched
 * @param[in] list The filename list to be updated
 * @param[in/out] counter increased by the number of appended elements.
 * @return The updated list
 * @todo This assumes .so for all sub plugins. Support Windows/Mac/iOS!
 */
static GSList *
_get_basenames (const gchar * dir, GSList * list, uint32_t * counter)
{
  GDir *gdir = g_dir_open (dir, 0U, NULL);
  const gchar *name;

  if (gdir == NULL)
    return list;

  while (NULL != (name = g_dir_read_name (gdir))) {
    list = g_slist_prepend (list, g_path_get_basename (name));
    *counter = *counter + 1;
  }
  g_dir_close (gdir);
  return list;
}

/**
 * @brief Data structure for _g_list_foreach_vstr_helper
 */
typedef struct
{
  gchar **vstr;    /**< The vstr data (string array) */
  uint32_t cursor;  /**< The first "empty" element in vstr */
  uint32_t size;    /**< The number of "g_char *" in vstr, excluding the terminator */
} vstr_helper;

/**
 * @brief Private function to help convert linked-list to vstr with foreach
 * @data The element data of linked-list
 * @user_data The struct to fill in vstr
 */
static void
_g_list_foreach_vstr_helper (gpointer data, gpointer user_data)
{
  vstr_helper *helper = (vstr_helper *) user_data;
  g_assert (helper->cursor < helper->size);
  helper->vstr[helper->cursor] = data;
  helper->cursor++;
}

/**
 * @brief Private function to fill in vstr
 */
static void
_fill_in_vstr (gchar *** fullpath_vstr, gchar *** basename_vstr,
    gchar * searchpath[CONF_SOURCES])
{
  GSList *lstF = NULL, *lstB = NULL;
  vstr_helper vstrF, vstrB;
  uint32_t counterF = 0, counterB = 0;
  int i;

  counterF = 0;
  counterB = 0;
  for (i = 0; i < CONF_SOURCES; i++) {
    if (searchpath[i]) {
      lstF = _get_fullpath_filenames (searchpath[i], lstF, &counterF);
      lstB = _get_basenames (searchpath[i], lstB, &counterB);
    } else {
      lstF = _get_fullpath_filenames ("./", lstF, &counterF);
      lstB = _get_basenames ("./", lstB, &counterB);
    }
  }
  g_assert (counterF == counterB);

  /* Because _get_* does "prepend", reverse them to have the correct order. */
  lstF = g_slist_reverse (lstF);
  lstB = g_slist_reverse (lstB);

  *fullpath_vstr = g_malloc_n (counterF + 1, sizeof (gchar *));
  *basename_vstr = g_malloc_n (counterB + 1, sizeof (gchar *));

  (*fullpath_vstr)[counterF] = NULL;
  (*basename_vstr)[counterB] = NULL;

  vstrF.vstr = *fullpath_vstr;
  vstrB.vstr = *basename_vstr;
  vstrF.size = counterF;
  vstrB.size = counterB;
  vstrF.cursor = 0;
  vstrB.cursor = 0;
  g_slist_foreach (lstF, _g_list_foreach_vstr_helper, (gpointer) & vstrF);
  g_slist_foreach (lstB, _g_list_foreach_vstr_helper, (gpointer) & vstrB);

  /* Do not free elements. They are now at vstr */
  g_slist_free (lstF);
  g_slist_free (lstB);
}

/** @brief Public function defined in the header */
gboolean
nnsconf_loadconf (gboolean force_reload)
{
  g_autoptr (GError) error = NULL;
  g_autoptr (GKeyFile) key_file = NULL;
  GStatBuf gsbuf;
  int stt, i;

  if (FALSE == force_reload && TRUE == conf.loaded)
    return TRUE;

  key_file = g_key_file_new ();

  if (TRUE == force_reload && TRUE == conf.loaded) {
    /* Do Clean Up */
    g_free (conf.conffile);

    for (i = 0; i < CONF_SOURCES; i++) {
      g_free (conf.pathFILTERS[i]);
      g_free (conf.pathDECODERS[i]);
      g_free (conf.pathCUSTOM_FILTERS[i]);
      g_free (conf.valueTF_MEM_OPTMZ[i]);
    }

    g_strfreev (conf.filesFILTERS);
    g_strfreev (conf.filesDECODERS);
    g_strfreev (conf.filesCUSTOM_FILTERS);
    g_strfreev (conf.basenameFILTERS);
    g_strfreev (conf.basenameDECODERS);
    g_strfreev (conf.basenameCUSTOM_FILTERS);
  }

  /* Read from Environmental Variables */
  conf.conffile = _strdup_getenv (NNSTREAMER_ENVVAR_CONF_FILE);
  stt = 0;                      /* File not found or not configured */
  if (conf.conffile != NULL) {
    if (0 == g_stat (conf.conffile, &gsbuf)) {
      /** @todo Check g_stat results (OK if it's regular file) */
      stt = 1;
    }
  }
  if (0 == stt) {
    if (NULL != conf.conffile)
      g_free (conf.conffile);
    conf.conffile = g_strdup (NNSTREAMER_DEFAULT_CONF_FILE);
  }

  conf.pathFILTERS[0] = _strdup_getenv (NNSTREAMER_ENVVAR_FILTERS);
  conf.pathDECODERS[0] = _strdup_getenv (NNSTREAMER_ENVVAR_DECODERS);
  conf.pathCUSTOM_FILTERS[0] = _strdup_getenv (NNSTREAMER_ENVVAR_CUSTOMFILTERS);
  conf.valueTF_MEM_OPTMZ[0] = _strdup_getenv (NNSTREAMER_ENVVAR_TF_MEM_OPTMZ);

  /* Read the conf file. It's ok even if we cannot load it. */
  if (conf.conffile &&
      g_key_file_load_from_file (key_file, conf.conffile, G_KEY_FILE_NONE,
          &error)) {

    conf.pathFILTERS[1] =
        g_key_file_get_string (key_file, "filter", "filters", &error);
    conf.pathDECODERS[1] =
        g_key_file_get_string (key_file, "decoder", "decoders", &error);
    conf.pathCUSTOM_FILTERS[1] =
        g_key_file_get_string (key_file, "filter", "customfilters", &error);
    conf.valueTF_MEM_OPTMZ[1] =
        g_key_file_get_string (key_file, "tensorflow", "mem_optmz", &error);
  }

  /* Strdup the hardcoded */
  conf.pathFILTERS[2] = g_strdup (NNSTREAMER_FILTERS);
  conf.pathDECODERS[2] = g_strdup (NNSTREAMER_DECODERS);
  conf.pathCUSTOM_FILTERS[2] = g_strdup (NNSTREAMER_CUSTOM_FILTERS);
  conf.valueTF_MEM_OPTMZ[2] = g_strdup (NNSTREAMER_TF_MEM_OPTMZ);

  /* Fill in conf.files* */
  _fill_in_vstr (&conf.filesFILTERS, &conf.basenameFILTERS, conf.pathFILTERS);
  _fill_in_vstr (&conf.filesDECODERS, &conf.basenameDECODERS,
      conf.pathDECODERS);
  _fill_in_vstr (&conf.filesCUSTOM_FILTERS, &conf.basenameCUSTOM_FILTERS,
      conf.pathCUSTOM_FILTERS);

  /* Fill in conf.bool values */
  for (i = 0; i < CONF_SOURCES; i++) {
    if (conf.valueTF_MEM_OPTMZ[i] && conf.valueTF_MEM_OPTMZ[i][0]) {
      if (!g_ascii_strncasecmp ("1", conf.valueTF_MEM_OPTMZ[i], 1) ||   /* 1 */
          !g_ascii_strncasecmp ("t", conf.valueTF_MEM_OPTMZ[i], 1) ||   /* True */
          !g_ascii_strncasecmp ("on", conf.valueTF_MEM_OPTMZ[i], 2) ||  /* On */
          !g_ascii_strncasecmp ("y", conf.valueTF_MEM_OPTMZ[i], 1))     /* Yes */
        conf.boolTF_MEM_OPTMZ = TRUE;
      else
        conf.boolTF_MEM_OPTMZ = FALSE;
      break;
    }
  }

  conf.loaded = TRUE;
  g_key_file_free (key_file);

  return TRUE;
}

/** @brief Public function defined in the header */
const gchar *
nnsconf_get_fullpath_fromfile (const gchar * file2find, nnsconf_type_path type)
{
  gchar **vstr, **vstrFull;
  int i;

  switch (type) {
    case NNSCONF_PATH_FILTERS:
      vstr = conf.basenameFILTERS;
      vstrFull = conf.filesFILTERS;
      break;
    case NNSCONF_PATH_DECODERS:
      vstr = conf.basenameDECODERS;
      vstrFull = conf.filesDECODERS;
      break;
    case NNSCONF_PATH_CUSTOM_FILTERS:
      vstr = conf.basenameCUSTOM_FILTERS;
      vstrFull = conf.filesCUSTOM_FILTERS;
      break;
    default:
      return NULL;
  }

  if (vstr == NULL)
    return NULL;

  i = 0;
  while (vstr[i] != NULL) {
    /** @todo To support Windows, use case insensitive if it's Windows */
    if (!g_strcmp0 (file2find, vstr[i]))
      return vstrFull[i];
    i++;
  };

  return NULL;
}

const gchar *subplugin_prefixes[NNSCONF_PATH_END] = {
  [NNSCONF_PATH_FILTERS] = NNSTREAMER_PREFIX_FILTER,
  [NNSCONF_PATH_DECODERS] = NNSTREAMER_PREFIX_DECODER,
  [NNSCONF_PATH_CUSTOM_FILTERS] = NNSTREAMER_PREFIX_CUSTOMFILTERS,
  NULL,
};

/** @brief Public function defined in the header */
const gchar *
nnsconf_get_fullpath (const gchar * subpluginname, nnsconf_type_path type)
{
  nnsconf_loadconf (FALSE);

  gchar *filename =
      g_strconcat (subplugin_prefixes[type], subpluginname, ".so", NULL);
  const gchar *ret = nnsconf_get_fullpath_fromfile (filename, type);

  g_free (filename);
  return ret;
}

/** @brief Public function defined in the header */
gboolean
nnsconf_get_value_bool (nnsconf_type_value type)
{
  gboolean ret;

  nnsconf_loadconf (FALSE);

  switch (type) {
    case NNSCONF_VAL_TF_MEM_OPTMZ:
      ret = conf.boolTF_MEM_OPTMZ;
      break;

    default:
      ret = FALSE;
  }

  return ret;
}

/**
 * @brief Internal cache for the custom key-values
 */
static GHashTable *custom_table = NULL;

/** @brief Public function defined in the header */
gchar *
nnsconf_get_custom_value_string (const gchar * group, const gchar * key)
{
  gchar *hashkey = g_strdup_printf ("[%s]%s", group, key);
  gchar *value = NULL;

  nnsconf_loadconf (FALSE);     /* Load .ini file path */

  if (NULL == custom_table)
    custom_table =
        g_hash_table_new_full (g_str_hash, g_str_equal, g_free, g_free);

  value = g_hash_table_lookup (custom_table, hashkey);

  if (NULL == value) {
    gchar *envkey = g_strdup_printf ("NNSTREAMER_%s_%s", group, key);

    /* 1. Read envvar */
    value = _strdup_getenv (envkey);
    g_free (envkey);

    /* 2. Read ini */
    if (NULL == value && conf.conffile) {
      g_autoptr (GError) error = NULL;
      g_autoptr (GKeyFile) key_file = g_key_file_new ();

      if (g_key_file_load_from_file (key_file, conf.conffile, G_KEY_FILE_NONE,
              &error)) {
        value = g_key_file_get_string (key_file, group, key, &error);
      }

      g_key_file_free (key_file);

    }

    if (value)
      g_hash_table_insert (custom_table, hashkey, value);
  }

  if (NULL == value)
    return NULL;

  return g_strdup (value);
}

/** @brief Public function defined in the header */
gboolean
nnsconf_get_custom_value_bool (const gchar * group, const gchar * key,
    gboolean def)
{
  gchar *strval = nnsconf_get_custom_value_string (group, key);
  gboolean ret = def;

  if (NULL == strval)
    return ret;

  /** 1/0, true/false, t/f, yes/no, on/off. case incensitive. */
  if (strval[0] == '1' || strval[0] == 't' || strval[0] == 'T' ||
      strval[0] == 'y' || strval[0] == 'Y' ||
      !g_ascii_strncasecmp ("on", strval, 2))
    ret = TRUE;
  if (strval[0] == '0' || strval[0] == 'f' || strval[0] == 'F' ||
      strval[0] == 'n' || strval[0] == 'N' ||
      !g_ascii_strncasecmp ("of", strval, 2))
    ret = FALSE;

  g_free (strval);
  return ret;
}
