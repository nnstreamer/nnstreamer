/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer Configuration Checker Utility
 * Copyright (C) 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file	confchk.c
 * @date	13 Aug 2020
 * @brief	NNStreamer configuration checker utility
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is a utility for nnstreamer developers.
 * This shows the effective nnstreamer configurations.
 *
 * Internal mechanism:
 *   Load up libnnstreamer.so with gstreamer
 */
#include <glib.h>
#include <gst/gst.h>
#include <dlfcn.h>

#define STR_BOOL(x) ((x) ? "TRUE" : "FALSE")

/**
 * @brief Dump NNStreamer configurations
 * @param[in] path the filepath of nnstreamer library
 */
static int
get_nnsconf_dump (const gchar * path)
{
  void *handle;
  void (*nnsconf_dump) (gchar * str, gulong size);
  void (*nnsconf_subplugin_dump) (gchar * str, gulong size);
  gchar dump[8192];

  handle = dlopen (path, RTLD_LAZY);
  if (!handle) {
    g_printerr ("Error opening %s: %s\n", path, dlerror ());
    return -1;
  }

  nnsconf_dump = dlsym (handle, "nnsconf_dump");
  if (!nnsconf_dump) {
    g_printerr ("Error loading nnsconf_dump: %s\n", dlerror ());
    dlclose (handle);
    return -2;
  }

  nnsconf_subplugin_dump = dlsym (handle, "nnsconf_subplugin_dump");
  if (!nnsconf_subplugin_dump) {
    g_printerr ("Error loading nnsconf_subplugin_dump: %s\n", dlerror ());
    dlclose (handle);
    return -2;
  }

  nnsconf_dump (dump, 8192);

  g_print ("\n");
  g_print ("NNStreamer configuration:\n");
  g_print ("============================================================\n");
  g_print ("%s\n", dump);
  g_print ("============================================================\n");

  nnsconf_subplugin_dump (dump, 8192);

  g_print ("\n");
  g_print ("NNStreamer registered sub-plugins:\n");
  g_print ("============================================================\n");
  g_print ("%s\n", dump);
  g_print ("============================================================\n");

  dlclose (handle);

  return 0;
}

/**
 * @brief Main routine
 */
int
main (int argc, char *argv[])
{
  GstPlugin *nnstreamer;

  gst_init (&argc, &argv);

  nnstreamer = gst_plugin_load_by_name ("nnstreamer");

  if (!nnstreamer) {
    g_printerr
        ("Cannot load nnstreamer plugin. Please check if nnstreamer is installed and GST_PLUGIN_PATH is properly configured.\n");
    return -1;
  }

  g_print ("NNStreamer version: %s\n", gst_plugin_get_version (nnstreamer));
  g_print ("           loaded : %s\n",
      STR_BOOL (gst_plugin_is_loaded (nnstreamer)));
  g_print ("           path   : %s\n", gst_plugin_get_filename (nnstreamer));

  get_nnsconf_dump (gst_plugin_get_filename (nnstreamer));

  return 0;
}
