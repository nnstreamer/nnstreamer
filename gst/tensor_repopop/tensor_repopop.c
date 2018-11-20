/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Samsung Electronics Co., Ltd.
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
 */

/**
 * SECTION: element-tensor_repopop
 *
 * Pop elemnt to handle tensor repo
 *
 * @file	tensor_repopop.c
 * @date	19 Nov 2018
 * @brief	GStreamer plugin to handle tensor repository
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "tensor_repopop.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

/**
 * @brief Macro for debug message.
 */
#define silent_debug(...) \
    debug_print (DBG, __VA_ARGS__)

GST_DEBUG_CATEGORY_STATIC (gst_tensor_repopop_debug);
#define GST_CAT_DEFAULT gst_tensor_repopop_debug

/**
 * @brief tensor_repopop properties
 */
enum
{
  PROP_0,
  PROP_CAPS,
  PROP_SILENT
};

#define DEFAULT_SILENT TRUE

extern GstTensorRepo _repo;

/**
 * @brief tensor_repopop src template
 */
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT));

static void gst_tensor_repopop_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_repopop_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_repopop_dispose (GObject * object);
static gboolean gst_tensor_repopop_query (GstBaseSrc * src, GstQuery * query);
static GstCaps *gst_tensor_repopop_getcaps (GstBaseSrc * src, GstCaps * filter);

#define gst_tensor_repopop_parent_class parent_class
G_DEFINE_TYPE (GstTensorRepoPop, gst_tensor_repopop, GST_TYPE_PUSH_SRC);

static void
gst_tensor_repopop_class_init (GstTensorRepoPopClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (kalss);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstPushSrcClass *pushsrc_class = GST_PUSH_SRC_CLASS (klass);
  GstBaseSrcClass *basesrc_class = GST_BASE_SRC_CLASS (kalss);

  gobject_class->set_property = gst_tensor_repopop_set_property;
  gobject_class->get_property = gst_tensor_repopop_get_property;

  g_object_class_install_property (gobject_class, PROP_CAPS,
      g_param_spec_boxed ("caps", "Caps",
          "Caps describing the format of the data.",
          GST_TYPE_CAPS, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));


  gobject_class->dispose = gst_tensor_repopop_dispose;

  basesrc_class->get_caps = gst_tensor_repopop_getcaps;
  basesrc_class->query = gst_tensor_repopop_query;

  pushsrc_class->create = gst_tensor_repopop_create;

  gst_element_class_set_static_metadata (element_class,
      "TensorRepoPop",
      "Pop/TensorRepo",
      "Pop element to handle tensor repository",
      "Samsung Electronics Co., Ltd.");

  gst_element_class_add_static_pad_template (element_class, &src_template);
}

static void
gst_tensor_repopop_init (GstTensorRepoPop * self)
{
  self->silent = TRUE;

}


























/**
 * @brief Function to initialize the plugin.
 *
 * See GstPluginInitFunc() for more details.
 */
NNSTREAMER_PLUGIN_INIT (tensor_repopop)
{
  GST_DEBUG_CATEGORY_INIT (gst_tensor_repopop_debug, "tensor_repopop",
      0, "tensor_repopop element");

  return gst_element_register (plugin, "tensor_repopop",
      GST_RANK_NONE, GST_TYPE_TENSOR_REPOPOP);
}

#ifndef SINGLE_BINARY
/**
 * @brief Definition for identifying tensor_repopop plugin.
 *
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "nnstreamer"
#endif

/**
 * @brief Macro to define the entry point of the plugin.
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensor_repopop,
    "Pop element to handle tensor repository",
    gst_tensor_repopop_plugin_init, VERSION, "LGPL", "nnstreamer",
    "https://github.com/nnsuite/nnstreamer");
#endif
