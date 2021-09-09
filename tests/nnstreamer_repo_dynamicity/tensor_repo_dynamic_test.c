/**
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file    tensor_repo_dynamic_test.c
 * @date    03 Dec 2018
 * @brief   test case to test tensor repo plugin dynamism
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except NYI.
 *
 *                                  repository
 * +------------+   +----------+        +-----------------+
 * | Multi-sink |<--| repo_src |<---+---|     slot 0      |<---+--+
 * +------------+   +----------+    |   +-----------------+    |  |
 *                                  +---|     slot 1      |<---+  |
 *                                      +-----------------+       |
 *                                                                |
 * +-----------+   +--------+   +-----------+   +-----------+     |
 * | Multi-src |-->| pngdec |-->| converter |-->| repo_sink |-----+
 * +-----------+   +--------+   +-----------+   +-----------+
 *
 */

#include <string.h>
#include <stdlib.h>
#include <gst/gst.h>
#include <tensor_common.h>
#include <nnstreamer_util.h>

static GMainLoop *loop = NULL;

#ifndef DBG
#define DBG FALSE
#endif

#define _print_log(...) if (DBG) g_message (__VA_ARGS__)

/**
 * @brief Bus Call Back Function
 */
static gboolean
my_bus_callback (GstBus * bus, GstMessage * message, gpointer data)
{
  UNUSED (bus);
  UNUSED (data);
  _print_log ("Got %s message\n", GST_MESSAGE_TYPE_NAME (message));

  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_ERROR:{
      GError *err;
      gchar *debug;

      gst_message_parse_error (message, &err, &debug);
      _print_log ("Error: %s\n", err->message);
      g_error_free (err);
      g_free (debug);

      g_main_loop_quit (loop);
      break;
    }
    case GST_MESSAGE_EOS:
      g_main_loop_quit (loop);
      break;
    default:
      break;
  }

  return TRUE;
}

/**
 * @brief Switch slot index
 */
static gboolean
switch_slot_index (GstElement * tensor_repo)
{
  guint active_slot, new_slot;

  _print_log ("switching");
  g_object_get (G_OBJECT (tensor_repo), "slot-index", &active_slot, NULL);

  new_slot = active_slot ? 0 : 1;

  g_object_set (G_OBJECT (tensor_repo), "slot-index", new_slot, NULL);

  _print_log ("current active memory slot is  : %d and it is changed to %d\n",
      active_slot, new_slot);

  return (GST_STATE (GST_ELEMENT (tensor_repo)) == GST_STATE_PLAYING);
}


/**
 * @brief Main function to evalute tensor_repo dynamicity
 */
int
main (int argc, char *argv[])
{
  GstElement *pipeline, *multifilesrc, *pngdec, *tensor_converter;
  GstElement *tensor_reposrc, *tensor_reposink, *multifilesink, *queue;
  GstBus *bus;
  GstCaps *msrc_cap, *reposrc_cap, *tmp;
  gboolean src_silent, sink_silent;
  guint src_slot, sink_slot, sink_rate;

  gst_init (&argc, &argv);
  msrc_cap = reposrc_cap = tmp = NULL;
  src_silent = sink_silent = TRUE;
  src_slot = sink_slot = sink_rate = G_MAXUINT;

  loop = g_main_loop_new (NULL, FALSE);

  pipeline = gst_pipeline_new ("pipeline");
  multifilesrc = gst_element_factory_make ("multifilesrc", "multifilesrc");
  g_object_set (G_OBJECT (multifilesrc), "location", "testsequence_%1d.png",
      NULL);
  msrc_cap =
      gst_caps_new_simple ("image/png", "framerate", GST_TYPE_FRACTION, 30, 1,
      NULL);
  g_object_set (G_OBJECT (multifilesrc), "caps", msrc_cap, NULL);
  gst_caps_unref (msrc_cap);

  pngdec = gst_element_factory_make ("pngdec", "pngdec");

  tensor_converter =
      gst_element_factory_make ("tensor_converter", "tensor_converter");

  queue = gst_element_factory_make ("queue", "queue");

  tensor_reposink =
      gst_element_factory_make ("tensor_reposink", "tensor_reposink");
  g_object_set (G_OBJECT (tensor_reposink), "silent", FALSE, NULL);
  g_object_set (G_OBJECT (tensor_reposink), "slot-index", 0, NULL);
  g_object_set (G_OBJECT (tensor_reposink), "signal-rate", 0, NULL);

  tensor_reposrc =
      gst_element_factory_make ("tensor_reposrc", "tensor_reposrc");
  g_object_set (G_OBJECT (tensor_reposrc), "silent", FALSE, NULL);
  g_object_set (G_OBJECT (tensor_reposrc), "slot-index", 0, NULL);

  reposrc_cap = gst_caps_new_simple ("other/tensor",
      "dimension", G_TYPE_STRING, "3:16:16:1",
      "type", G_TYPE_STRING, "uint8",
      "framerate", GST_TYPE_FRACTION, 30, 1, NULL);

  g_object_set (G_OBJECT (tensor_reposrc), "caps", reposrc_cap, NULL);

  /* check tensor-repo properties */
  g_object_get (G_OBJECT (tensor_reposink), "silent", &sink_silent,
      "slot-index", &sink_slot, "signal-rate", &sink_rate, NULL);
  if (sink_silent || sink_slot || sink_rate)
    goto error;

  g_object_get (G_OBJECT (tensor_reposrc), "silent", &src_silent,
      "slot-index", &src_slot, "caps", &tmp, NULL);
  if (src_silent || src_slot || !gst_caps_is_equal (reposrc_cap, tmp))
    goto error;

  multifilesink = gst_element_factory_make ("multifilesink", "multifilesink");
  g_object_set (G_OBJECT (multifilesink), "location",
      "tensorsequence01_%1d.log", NULL);

  gst_bin_add_many (GST_BIN (pipeline), multifilesrc, pngdec, tensor_converter,
      queue, tensor_reposink, tensor_reposrc, multifilesink, NULL);

  gst_element_link (multifilesrc, pngdec);
  gst_element_link (pngdec, tensor_converter);
  gst_element_link (tensor_converter, queue);
  gst_element_link (queue, tensor_reposink);
  gst_element_link (tensor_reposrc, multifilesink);

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));

  gst_bus_add_watch (bus, my_bus_callback, NULL);
  gst_object_unref (bus);
  gst_element_set_state (GST_ELEMENT (pipeline), GST_STATE_PLAYING);

  g_timeout_add (100, (GSourceFunc) switch_slot_index, tensor_reposink);
  g_timeout_add (100, (GSourceFunc) switch_slot_index, tensor_reposrc);

  g_timeout_add (100, (GSourceFunc) switch_slot_index, tensor_reposink);
  g_timeout_add (100, (GSourceFunc) switch_slot_index, tensor_reposrc);

  g_timeout_add (100, (GSourceFunc) switch_slot_index, tensor_reposink);
  g_timeout_add (100, (GSourceFunc) switch_slot_index, tensor_reposrc);


  g_main_loop_run (loop);

  gst_element_set_state (GST_ELEMENT (pipeline), GST_STATE_READY);

error:
  gst_object_unref (GST_OBJECT (pipeline));
  gst_caps_unref (reposrc_cap);
  gst_caps_unref (tmp);
  exit (0);
}
