/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Pipeline from/to PBTxt Converter Parser
 * Copyright (C) 2021 MyungJoo Ham <myungjoo.ham@samsung.com>
 * Copyright (C) 2021 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file    toplevel.c
 * @date    27 Apr 2021
 * @brief   Top-level program to parse gst pipeline and pbtxt
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include <stdio.h>

#define G_LOG_USE_STRUCTURED 1
#include <glib.h>

#include "types.h"
#include "convert.h"

#define INPUT_MAXLEN (512)

static gboolean from_pbtxt = FALSE;
static gboolean verbose = FALSE;

static GOptionEntry entries[] = {
  {"from-pbtxt", 'p', 0, G_OPTION_ARG_NONE, &from_pbtxt,
        "From pbtxt to gst pipeline", NULL},
  {"verbose", 'v', 0, G_OPTION_ARG_NONE, &verbose, "Enable verbose messages",
        NULL},
  {NULL}
};

/** @brief Get input string for parsing */
static gboolean
get_input_string (char *str)
{
  GIOChannel *channel = g_io_channel_unix_new (fileno (stdin));
  GIOStatus status;
  gboolean ret = TRUE;
  GError *error = NULL;
  gsize length;

  status = g_io_channel_set_encoding (channel, "UTF-8", &error);
  if (status == G_IO_STATUS_ERROR) {
    g_printerr ("Error detected while setting encoding: %s\n", error->message);
    g_error_free (error);
    ret = FALSE;
    goto out;
  }

  status =
      g_io_channel_read_chars (channel, str, INPUT_MAXLEN, &length, &error);
  if (status == G_IO_STATUS_ERROR) {
    g_printerr ("Error detected while reading an input string: %s\n",
        error->message);
    g_error_free (error);
    ret = FALSE;
    goto out;
  }

  if (status != G_IO_STATUS_NORMAL) {
    ret = FALSE;
    goto out;
  }

  str[length - 1] = '\x00';

out:
  g_io_channel_shutdown (channel, TRUE, NULL);
  return ret;
}

/** @brief Log handler */
static void
log_handler (const gchar * log_domain, GLogLevelFlags log_level,
    const gchar * message, gpointer user_data)
{
  (void) log_domain;
  (void) log_level;
  (void) user_data;
  g_printerr ("%s", message);
}

/** @brief Main routine for this program */
int
main (int argc, char *argv[])
{
  GError *error = NULL;
  GOptionContext *context;
  char input_str[INPUT_MAXLEN] = { '\x00' };
  GLogLevelFlags log_flags;
  _Element *pipeline;

  context =
      g_option_context_new ("- Prototxt to/from GStreamer Pipeline Converver");
  g_option_context_add_main_entries (context, entries, NULL);
  if (!g_option_context_parse (context, &argc, &argv, &error)) {
    g_printerr ("Option parsing failed: %s\n", error->message);
    g_error_free (error);
    return -1;
  }
  g_option_context_free (context);

  if (from_pbtxt) {
    g_print ("NYI: pbtxt-to-gstpipe conversion\n");
    return 0;
  }

  if (!get_input_string (input_str)) {
    g_printerr ("Unable to get an input string for GStreamer pipeline\n");
    return -1;
  }

  /* all messages from glib */
  log_flags = G_LOG_LEVEL_MASK | G_LOG_FLAG_FATAL | G_LOG_FLAG_RECURSION;
  if (!verbose)
    log_flags &= ~G_LOG_LEVEL_DEBUG;

  g_log_set_handler (NULL, log_flags, log_handler, NULL);

  pipeline = priv_gst_parse_launch (input_str, NULL, NULL, __PARSE_FLAG_NONE);
  if (pipeline == NULL) {
    g_printerr ("Unable to parse the given pipeline string");
    return -1;
  }

  convert_to_pbtxt (pipeline);
  nnstparser_element_unref (pipeline);

  return 0;
}
