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
 * SECTION: element-tensor_reposrc
 *
 * Pop elemnt to handle tensor repo
 *
 * @file	tensor_reposrc.c
 * @date	19 Nov 2018
 * @brief	GStreamer plugin to handle tensor repository
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include "tensor_reposrc.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_reposrc_debug);
#define GST_CAT_DEFAULT gst_tensor_reposrc_debug

/**
 * @brief tensor_reposrc properties
 */
enum
{
  PROP_0,
  PROP_CAPS,
  PROP_SLOT_ID,
  PROP_SILENT
};

#define DEFAULT_SILENT TRUE
#define DEFAULT_INDEX 0

/**
 * @brief external repo
 */
extern GstTensorRepo _repo;

/**
 * @brief tensor_reposrc src template
 */
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT));

static void gst_tensor_reposrc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_reposrc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_reposrc_dispose (GObject * object);
static GstCaps *gst_tensor_reposrc_getcaps (GstBaseSrc * src, GstCaps * filter);
static GstFlowReturn gst_tensor_reposrc_create (GstPushSrc * src,
    GstBuffer ** buffer);

#define gst_tensor_reposrc_parent_class parent_class
G_DEFINE_TYPE (GstTensorRepoSrc, gst_tensor_reposrc, GST_TYPE_PUSH_SRC);

/**
 * @brief class initialization of tensor_reposrc
 */
static void
gst_tensor_reposrc_class_init (GstTensorRepoSrcClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstPushSrcClass *pushsrc_class = GST_PUSH_SRC_CLASS (klass);
  GstBaseSrcClass *basesrc_class = GST_BASE_SRC_CLASS (klass);

  gobject_class->set_property = gst_tensor_reposrc_set_property;
  gobject_class->get_property = gst_tensor_reposrc_get_property;

  g_object_class_install_property (gobject_class, PROP_CAPS,
      g_param_spec_boxed ("caps", "Caps",
          "Caps describing the format of the data.",
          GST_TYPE_CAPS, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SLOT_ID,
      g_param_spec_uint ("slot-index", "Slot Index", "repository slot index",
          0, UINT_MAX, DEFAULT_INDEX,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gobject_class->dispose = gst_tensor_reposrc_dispose;

  basesrc_class->get_caps = gst_tensor_reposrc_getcaps;
  pushsrc_class->create = gst_tensor_reposrc_create;

  gst_element_class_set_static_metadata (element_class,
      "TensorRepoSrc",
      "Pop/TensorRepo",
      "Pop element to handle tensor repository",
      "Samsung Electronics Co., Ltd.");

  gst_element_class_add_static_pad_template (element_class, &src_template);
}

/**
 * @brief object initialization of tensor_reposrc
 */
static void
gst_tensor_reposrc_init (GstTensorRepoSrc * self)
{
  self->silent = TRUE;
  self->ini = FALSE;
  gst_tensors_config_init (&self->config);
  self->caps = NULL;
}

/**
 * @brief object dispose of tensor_reposrc
 */
static void
gst_tensor_reposrc_dispose (GObject * object)
{
  gboolean ret;
  GstTensorRepoSrc *self = GST_TENSOR_REPOSRC (object);

  ret = gst_tensor_repo_remove_repodata (self->myid);
  if (!ret)
    GST_ELEMENT_ERROR (self, RESOURCE, WRITE,
        ("Cannot remove [key: %d] in repo", self->myid), NULL);

  if (self->caps)
    gst_caps_unref (self->caps);

  G_OBJECT_CLASS (parent_class)->dispose (object);
}

/**
 * @brief get cap of tensor_reposrc
 */
static GstCaps *
gst_tensor_reposrc_getcaps (GstBaseSrc * src, GstCaps * filter)
{
  GstCaps *cap;
  GstTensorRepoSrc *self = GST_TENSOR_REPOSRC (src);

  GST_DEBUG_OBJECT (self, "returning %" GST_PTR_FORMAT, self->caps);

  if (self->caps) {
    if (filter) {
      cap =
          gst_caps_intersect_full (filter, self->caps,
          GST_CAPS_INTERSECT_FIRST);
    } else
      cap = gst_caps_ref (self->caps);
  } else {
    if (filter) {
      cap = gst_caps_ref (filter);
    } else
      cap = gst_caps_new_any ();
  }

  return cap;
}

/**
 * @brief set property of tensor_reposrc
 */
static void
gst_tensor_reposrc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{

  GstTensorRepoSrc *self = GST_TENSOR_REPOSRC (object);

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      break;
    case PROP_SLOT_ID:
      self->myid = g_value_get_uint (value);
      break;
    case PROP_CAPS:
    {
      GstStructure *st = NULL;
      const GstCaps *caps = gst_value_get_caps (value);
      GstCaps *new_caps;

      if (caps == NULL) {
        new_caps = gst_caps_new_any ();
      } else {
        new_caps = gst_caps_copy (caps);
      }
      gst_caps_replace (&self->caps, new_caps);
      gst_pad_set_caps (GST_BASE_SRC_PAD (self), new_caps);
      st = gst_caps_get_structure (new_caps, 0);

      gst_tensors_config_from_structure (&self->config, st);

      if (new_caps && gst_caps_get_size (new_caps) == 1 && st
          && gst_structure_get_fraction (st, "framerate", &self->fps_n,
              &self->fps_d)) {
        GST_INFO_OBJECT (self, "Seting framerate to %d/%d", self->fps_n,
            self->fps_d);
      } else {
        self->fps_n = -1;
        self->fps_d = -1;
      }
    }
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief get property of tensor_reposrc
 */
static void
gst_tensor_reposrc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorRepoSrc *self = GST_TENSOR_REPOSRC (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_SLOT_ID:
      g_value_set_uint (value, self->myid);
      break;
    case PROP_CAPS:
      gst_value_set_caps (value, self->caps);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief create func of tensor_reposrc
 */
static GstFlowReturn
gst_tensor_reposrc_create (GstPushSrc * src, GstBuffer ** buffer)
{
  GstTensorRepoSrc *self;
  GstBuffer *buf = NULL;

  self = GST_TENSOR_REPOSRC (src);
  gst_tensor_repo_wait ();
  if (gst_tensor_repo_check_eos (self->myid)) {
    return GST_FLOW_EOS;
  }

  if (!self->ini) {
    int i;
    guint num_tensors = self->config.info.num_tensors;
    gsize size = 0;
    buf = gst_buffer_new ();
    for (i = 0; i < num_tensors; i++) {
      GstMemory *mem;
      GstMapInfo info;
      size = gst_tensor_info_get_size (&self->config.info.info[i]);
      mem = gst_allocator_alloc (NULL, size, NULL);
      gst_memory_map (mem, &info, GST_MAP_WRITE);
      memset (info.data, 0, size);
      gst_memory_unmap (mem, &info);
      gst_buffer_append_memory (buf, mem);
    }
    self->ini = TRUE;
  } else {
    buf = gst_tensor_repo_get_buffer (self->myid);
    if (buf == NULL)
      return GST_FLOW_EOS;
  }

  *buffer = buf;

  return GST_FLOW_OK;
}

/**
 * @brief Function to initialize the plugin.
 *
 * See GstPluginInitFunc() for more details.
 */
NNSTREAMER_PLUGIN_INIT (tensor_reposrc)
{
  GST_DEBUG_CATEGORY_INIT (gst_tensor_reposrc_debug, "tensor_reposrc",
      0, "tensor_reposrc element");

  return gst_element_register (plugin, "tensor_reposrc",
      GST_RANK_NONE, GST_TYPE_TENSOR_REPOSRC);
}
