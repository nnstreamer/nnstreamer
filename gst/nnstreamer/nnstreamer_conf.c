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

#include <string.h>
#include <glib.h>

#include "nnstreamer_conf.h"

/**
 * Note that users still can place their custom filters anywhere if they
 * designate them with the full path.
 */

/* Subplugin Naming Rules */
#define NNSTREAMER_PREFIX_DECODER	"libnnstreamer_decoder_"
#define NNSTREAMER_PREFIX_FILTER	"libnnstreamer_filter_"
#define NNSTREAMER_PREFIX_CUSTOMFILTERS	""
#define NNSTREAMER_PREFIX_CONVERTER	""
/* Custom filter does not have prefix */

/* Env-var names */
static const gchar *NNSTREAMER_ENVVAR[NNSCONF_PATH_END] = {
  [NNSCONF_PATH_FILTERS] = "NNSTREAMER_FILTERS",
  [NNSCONF_PATH_DECODERS] = "NNSTREAMER_DECODERS",
  [NNSCONF_PATH_CUSTOM_FILTERS] = "NNSTREAMER_CUSTOMFILTERS",
  [NNSCONF_PATH_CONVERTERS] = "NNSTREAMER_CONVERTERS"
};

static const gchar *NNSTREAMER_PATH[NNSCONF_PATH_END] = {
  [NNSCONF_PATH_FILTERS] = "/usr/lib/nnstreamer/filters/",
  [NNSCONF_PATH_DECODERS] = "/usr/lib/nnstreamer/decoders/",
  [NNSCONF_PATH_CUSTOM_FILTERS] = "/usr/lib/nnstreamer/customfilters/",
  [NNSCONF_PATH_CONVERTERS] = "/usr/lib/nnstreamer/converters/",
};

static const gchar *subplugin_prefixes[] = {
  [NNSCONF_PATH_FILTERS] = NNSTREAMER_PREFIX_FILTER,
  [NNSCONF_PATH_DECODERS] = NNSTREAMER_PREFIX_DECODER,
  [NNSCONF_PATH_CUSTOM_FILTERS] = NNSTREAMER_PREFIX_CUSTOMFILTERS,
  [NNSCONF_PATH_EASY_CUSTOM_FILTERS] = NNSTREAMER_PREFIX_CUSTOMFILTERS, /**< Same as Custom Filters */
  [NNSCONF_PATH_CONVERTERS] = NNSTREAMER_PREFIX_CONVERTER,
  [NNSCONF_PATH_END] = NULL
};

typedef enum
{
  CONF_SOURCE_ENVVAR = 0,
  CONF_SOURCE_INI = 1,
  CONF_SOURCE_HARDCODE = 2,
  CONF_SOURCE_END,
} conf_sources;

typedef struct
{
  /*************************************************
   * Cached Raw Values                             *
   * 0: ENVVAR. 1: CONFFILE, 2: Hardcoded          *
   *************************************************/
  gchar *path[CONF_SOURCE_END]; /**< directory paths */

  /*************************************************
   * Processed Values                              *
   *************************************************/
  gchar **files; /**< Null terminated list of full filepaths */
  gchar **basename; /**< Null terminated list of basenames */
} subplugin_conf;

typedef struct
{
  gboolean loaded;            /**< TRUE if loaded at least once */
  gboolean enable_envvar;     /**< TRUE to parse env variables */
  gboolean enable_symlink;    /**< TRUE to allow symbolic link file */

  gchar *conffile;            /**< Location of conf file. */

  subplugin_conf conf[NNSCONF_PATH_END];
} confdata;

static confdata conf = { 0 };

/**
 * @brief Parse string to get boolean value.
 */
static gboolean
_parse_bool_string (const gchar * strval, gboolean def)
{
  gboolean res = def;

  if (strval) {
    /* 1/0, true/false, t/f, yes/no, on/off. case incensitive. */
    if (strval[0] == '1' || strval[0] == 't' || strval[0] == 'T' ||
        strval[0] == 'y' || strval[0] == 'Y' ||
        g_ascii_strncasecmp ("on", strval, 2) == 0) {
      res = TRUE;
    } else if (strval[0] == '0' || strval[0] == 'f' || strval[0] == 'F' ||
        strval[0] == 'n' || strval[0] == 'N' ||
        g_ascii_strncasecmp ("of", strval, 2) == 0) {
      res = FALSE;
    }
  }

  return res;
}

/**
 * @brief Private function to get strdup-ed env-var if it's valid.
 *        Otherwise, NULL
 *
 * @retval strdup-ed env-var value
 * @param[in] name Environmetal variable name
 */
static gchar *
_strdup_getenv (const gchar * name)
{
  /**
   * @todo Evaluate if we need to use secure_getenv() here
   *  (and compatible with other OS
   */
  const gchar *tmp = g_getenv (name);

  return (tmp != NULL) ? g_strdup (tmp) : NULL;
}

/**
 * @brief Private function to validate .so file can be added to the list.
 */
static gboolean
_validate_file (nnsconf_type_path type, const gchar * fullpath)
{
  /* ignore directory */
  if (!fullpath || !g_file_test (fullpath, G_FILE_TEST_IS_REGULAR))
    return FALSE;
  /* ignore symbol link file */
  if (!conf.enable_symlink && g_file_test (fullpath, G_FILE_TEST_IS_SYMLINK))
    return FALSE;
  /** @todo how to validate with nnsconf type. */
  return TRUE;
}

/**
 * @brief Private function to fill in ".so/.dylib list" with fullpath-filenames in a directory.
 * @param[in] type conf type to scan.
 * @param[in] dir Directory to be searched.
 * @param[in/out] listF The fullpath list to be updated.
 * @param[in/out] listB The basename list to be updated.
 * @param[in/out] counter increased by the number of appended elements.
 * @return True if successfully updated.
 * @todo This assumes .so/.dylib for all sub plugins. Support Windows!
 */
static gboolean
_get_filenames (nnsconf_type_path type, const gchar * dir, GSList ** listF,
    GSList ** listB, guint * counter)
{
  GDir *gdir;
  const gchar *name;
  gchar *fullpath;

  if ((gdir = g_dir_open (dir, 0U, NULL)) == NULL)
    return FALSE;

  while (NULL != (name = g_dir_read_name (gdir))) {
    /* check file prefix for given type, currently handle .so and .dylib. */
    if (g_str_has_prefix (name, subplugin_prefixes[type]) &&
        g_str_has_suffix (name, NNSTREAMER_SO_FILE_EXTENSION)) {
      fullpath = g_build_filename (dir, name, NULL);

      if (_validate_file (type, fullpath)) {
        *listF = g_slist_prepend (*listF, fullpath);
        *listB = g_slist_prepend (*listB, g_path_get_basename (name));
        *counter = *counter + 1;
      } else {
        g_free (fullpath);
      }
    }
  }

  g_dir_close (gdir);
  return TRUE;
}

/**
 * @brief Private function to get sub-plugins list with type.
 */
static gboolean
_get_subplugin_with_type (nnsconf_type_path type, gchar *** basename,
    gchar *** filepath)
{
  if (type >= NNSCONF_PATH_END) {
    /* unknown type */
    g_critical ("Failed to get sub-plugins, unknown sub-plugin type.");
    return FALSE;
  }

  /* Easy custom uses the configuration of custom */
  if (type == NNSCONF_PATH_EASY_CUSTOM_FILTERS)
    type = NNSCONF_PATH_CUSTOM_FILTERS;

  *basename = conf.conf[type].basename;
  *filepath = conf.conf[type].files;
  return TRUE;
}

/**
 * @brief Data structure for _g_list_foreach_vstr_helper
 */
typedef struct
{
  gchar **vstr;    /**< The vstr data (string array) */
  guint cursor;  /**< The first "empty" element in vstr */
  guint size;    /**< The number of "g_char *" in vstr, excluding the terminator */
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
    gchar * searchpath[CONF_SOURCE_END], nnsconf_type_path type)
{
  GSList *lstF = NULL, *lstB = NULL;
  vstr_helper vstrF, vstrB;
  guint i, counter;

  counter = 0;
  for (i = 0; i < CONF_SOURCE_END; i++) {
    if (searchpath[i]) {
      _get_filenames (type, searchpath[i], &lstF, &lstB, &counter);
    }
  }

  /* Because _get_* does "prepend", reverse them to have the correct order. */
  lstF = g_slist_reverse (lstF);
  lstB = g_slist_reverse (lstB);

  *fullpath_vstr = g_malloc0_n (counter + 1, sizeof (gchar *));
  g_assert (*fullpath_vstr != NULL);
  *basename_vstr = g_malloc0_n (counter + 1, sizeof (gchar *));
  g_assert (*basename_vstr != NULL);

  vstrF.vstr = *fullpath_vstr;
  vstrB.vstr = *basename_vstr;
  vstrF.size = counter;
  vstrB.size = counter;
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
  const gchar root_path_prefix[] = NNSTREAMER_SYS_ROOT_PATH_PREFIX;
  g_autoptr (GKeyFile) key_file = NULL;
  guint i, t;

  if (FALSE == force_reload && TRUE == conf.loaded)
    return TRUE;

  if (TRUE == force_reload && TRUE == conf.loaded) {
    /* Do Clean Up */
    g_free (conf.conffile);

    for (t = 0; t < NNSCONF_PATH_END; t++) {

      for (i = 0; i < CONF_SOURCE_END; i++) {
        g_free (conf.conf[t].path[i]);
      }

      g_strfreev (conf.conf[t].files);
      g_strfreev (conf.conf[t].basename);
    }

    /* init with 0 */
    memset (&conf, 0, sizeof (confdata));
  }
#ifndef __TIZEN__
  /** if it's not Tizen, configuration from env-var has a higher priority */
  conf.conffile = _strdup_getenv (NNSTREAMER_ENVVAR_CONF_FILE);
  if (conf.conffile && !g_file_test (conf.conffile, G_FILE_TEST_IS_REGULAR)) {
    g_free (conf.conffile);
    conf.conffile = NULL;
  }
#endif
  if (conf.conffile == NULL) {
    /**
     * Priority of reading a conf file
     * 1) read from NNSTREAMER_CONF_FILE
     * 2) read from NNSTREAMER_DEFAULT_CONF_FILE
     * 3) read from env-var
     */
    if (g_path_is_absolute (NNSTREAMER_CONF_FILE)) {
      conf.conffile = g_strdup (NNSTREAMER_CONF_FILE);
    } else {
      /** default value of 'sysconfdir' in meson is 'etc' */
      conf.conffile = g_build_path (G_DIR_SEPARATOR_S, root_path_prefix,
          NNSTREAMER_CONF_FILE, NULL);
    }

    if (!g_file_test (conf.conffile, G_FILE_TEST_IS_REGULAR)) {
      /* File not found or not configured */
      g_free (conf.conffile);

      if (g_file_test (NNSTREAMER_DEFAULT_CONF_FILE, G_FILE_TEST_IS_REGULAR)) {
        conf.conffile = g_strdup (NNSTREAMER_DEFAULT_CONF_FILE);
      } else {
        /* Try to read from Environmental Variables */
        conf.conffile = _strdup_getenv (NNSTREAMER_ENVVAR_CONF_FILE);
      }
    }
  }

  if (conf.conffile) {
    key_file = g_key_file_new ();
    g_assert (key_file != NULL);

    /* Read the conf file. It's ok even if we cannot load it. */
    if (g_key_file_load_from_file (key_file, conf.conffile, G_KEY_FILE_NONE,
            NULL)) {
      gchar *value;

      value = g_key_file_get_string (key_file, "common", "enable_envvar", NULL);
      conf.enable_envvar = _parse_bool_string (value, FALSE);
      g_free (value);

      value =
          g_key_file_get_string (key_file, "common", "enable_symlink", NULL);
      conf.enable_symlink = _parse_bool_string (value, FALSE);
      g_free (value);

      conf.conf[NNSCONF_PATH_FILTERS].path[CONF_SOURCE_INI] =
          g_key_file_get_string (key_file, "filter", "filters", NULL);
      conf.conf[NNSCONF_PATH_DECODERS].path[CONF_SOURCE_INI] =
          g_key_file_get_string (key_file, "decoder", "decoders", NULL);
      conf.conf[NNSCONF_PATH_CUSTOM_FILTERS].path[CONF_SOURCE_INI] =
          g_key_file_get_string (key_file, "filter", "customfilters", NULL);
      conf.conf[NNSCONF_PATH_CONVERTERS].path[CONF_SOURCE_INI] =
          g_key_file_get_string (key_file, "converter", "converters", NULL);
    }

    g_key_file_free (key_file);
  } else {
    /**
     * Failed to get the configuration.
     * Note that Android API does not use the configuration.
     */
    g_warning ("Failed to load the configuration, no config file found.");
  }

  for (t = 0; t < NNSCONF_PATH_END; t++) {
    if (t == NNSCONF_PATH_EASY_CUSTOM_FILTERS)
      continue;                 /* It does not have its own configuration */

    /* Read from env variables. */
    if (conf.enable_envvar)
      conf.conf[t].path[CONF_SOURCE_ENVVAR] =
          _strdup_getenv (NNSTREAMER_ENVVAR[t]);

    /* Strdup the hardcoded */
    conf.conf[t].path[CONF_SOURCE_HARDCODE] = g_strdup (NNSTREAMER_PATH[t]);

    /* Fill in conf.files* */
    _fill_in_vstr (&conf.conf[t].files, &conf.conf[t].basename,
        conf.conf[t].path, t);
  }

  conf.loaded = TRUE;
  return TRUE;
}

/** @brief Public function defined in the header */
const gchar *
nnsconf_get_fullpath_from_file (const gchar * file2find, nnsconf_type_path type)
{
  gchar **vstr, **vstrFull;
  guint i;

  if (!_get_subplugin_with_type (type, &vstr, &vstrFull))
    return NULL;

  if (vstr == NULL)
    return NULL;

  i = 0;
  while (vstr[i] != NULL) {
    /** @todo To support Windows, use case insensitive if it's Windows */
    if (g_strcmp0 (file2find, vstr[i]) == 0)
      return vstrFull[i];
    i++;
  };

  return NULL;
}

/** @brief Public function defined in the header */
const gchar *
nnsconf_get_fullpath (const gchar * subpluginname, nnsconf_type_path type)
{
  gchar *filename;
  const gchar *ret;

  nnsconf_loadconf (FALSE);

  filename = g_strconcat (subplugin_prefixes[type], subpluginname,
      NNSTREAMER_SO_FILE_EXTENSION, NULL);
  ret = nnsconf_get_fullpath_from_file (filename, type);

  g_free (filename);
  return ret;
}

/**
 * @brief Public function to validate sub-plugin library is available.
 */
gboolean
nnsconf_validate_file (nnsconf_type_path type, const gchar * fullpath)
{
  nnsconf_loadconf (FALSE);

  return _validate_file (type, fullpath);
}

/**
 * @brief Get sub-plugin's name prefix.
 * @param[in] type The type (FILTERS/DECODERS/CUSTOM_FILTERS)
 * @return Predefined prefix string for given type.
 */
const gchar *
nnsconf_get_subplugin_name_prefix (nnsconf_type_path type)
{
  return subplugin_prefixes[type];
}

/**
 * @brief Public function to get the list of sub-plugins basename and path
 * @return total number of sub-plugins for given type
 * @note DO NOT free sub-plugins info
 */
guint
nnsconf_get_subplugin_info (nnsconf_type_path type, subplugin_info_s * info)
{
  gchar **vstr, **vstrFull;

  g_return_val_if_fail (info != NULL, 0);
  info->names = info->paths = NULL;

  nnsconf_loadconf (FALSE);

  if (!_get_subplugin_with_type (type, &vstr, &vstrFull))
    return 0;

  info->names = vstr;
  info->paths = vstrFull;

  return g_strv_length (vstr);
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
    /* 1. Read envvar */
    if (conf.enable_envvar) {
      gchar *envkey = g_strdup_printf ("NNSTREAMER_%s_%s", group, key);

      value = _strdup_getenv (envkey);
      g_free (envkey);
    }

    /* 2. Read ini */
    if (NULL == value && conf.conffile) {
      g_autoptr (GKeyFile) key_file = g_key_file_new ();

      g_assert (key_file != NULL);

      if (g_key_file_load_from_file (key_file, conf.conffile, G_KEY_FILE_NONE,
              NULL)) {
        value = g_key_file_get_string (key_file, group, key, NULL);
      }
    }

    if (value)
      g_hash_table_insert (custom_table, hashkey, value);
  }

  return (value != NULL) ? g_strdup (value) : NULL;
}

/** @brief Public function defined in the header */
gboolean
nnsconf_get_custom_value_bool (const gchar * group, const gchar * key,
    gboolean def)
{
  gchar *strval;
  gboolean ret;

  strval = nnsconf_get_custom_value_string (group, key);
  ret = _parse_bool_string (strval, def);

  g_free (strval);
  return ret;
}
