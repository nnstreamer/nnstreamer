/**
 * @file    tensor_repo_dynamic_test.c
 * @date    03 Dec 2018
 * @brief   test case to test tensor repo plugin dynamism
 * @see     https://github.com/nnsuite/nnstreamer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except NYI.
 */

#include <string.h>
#include <stdlib.h>

#include <gst/gst.h>
#include <tensor_common.h>
#include <tensor_repo.h>

static GMainLoop *loop = NULL;

/**
 * @brief Bus Call Back Function
 */
static gboolean
my_bus_callback (GstBus * bus, GstMessage * message, gpointer data)
{
  g_print ("Got %s message\n", GST_MESSAGE_TYPE_NAME (message));

  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_ERROR:{
      GError *err;
      gchar *debug;

      gst_message_parse_error (message, &err, &debug);
      g_print ("Error: %s\n", err->message);
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
 * @brief Main function to evalute tensor_repo dynamicity
 */
int
main (int argc, char *argv[])
{
  GstElement *pipeline, *multifilesrc, *pngdec, *tensor_converter,
      *tensor_reposink;
  GstElement *tensor_reposrc, *multifilesink;
  GstBus *bus;
  GstCaps *msrc_cap, *reposrc_cap;

  gst_init (&argc, &argv);

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

  tensor_reposink =
      gst_element_factory_make ("tensor_reposink", "tensor_reposink");
  g_object_set (G_OBJECT (tensor_reposink), "silent", FALSE, NULL);
  g_object_set (G_OBJECT (tensor_reposink), "slot-index", 0, NULL);

  tensor_reposrc =
      gst_element_factory_make ("tensor_reposrc", "tensor_reposrc");
  g_object_set (G_OBJECT (tensor_reposrc), "silent", FALSE, NULL);
  g_object_set (G_OBJECT (tensor_reposrc), "slot-index", 0, NULL);

  reposrc_cap = gst_caps_new_simple ("other/tensor",
      "dim1", G_TYPE_INT, 3,
      "dim2", G_TYPE_INT, 100,
      "dim3", G_TYPE_INT, 100,
      "dim4", G_TYPE_INT, 1,
      "type", G_TYPE_STRING, "uint8",
      "framerate", GST_TYPE_FRACTION, 30, 1, NULL);

  g_object_set (G_OBJECT (tensor_reposrc), "caps", reposrc_cap, NULL);
  gst_caps_unref (reposrc_cap);

  multifilesink = gst_element_factory_make ("multifilesink", "multifilesink");
  g_object_set (G_OBJECT (multifilesink), "location",
      "tensorsequence01_%1d.log", NULL);

  gst_bin_add_many (GST_BIN (pipeline), multifilesrc, pngdec, tensor_converter,
      tensor_reposink, tensor_reposrc, multifilesink, NULL);

  gst_element_link (multifilesrc, pngdec);
  gst_element_link (pngdec, tensor_converter);
  gst_element_link (tensor_converter, tensor_reposink);
  gst_element_link (tensor_reposrc, multifilesink);

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));

  gst_bus_add_watch (bus, my_bus_callback, NULL);
  gst_object_unref (bus);
  gst_element_set_state (GST_ELEMENT (pipeline), GST_STATE_PLAYING);

  g_main_loop_run (loop);

  gst_element_set_state (GST_ELEMENT (pipeline), GST_STATE_READY);

  gst_object_unref (GST_OBJECT (pipeline));

  exit (0);
}
