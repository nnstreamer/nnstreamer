/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Samsung Electronics Co., Ltd.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
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
 * @file	gsttensor_reposrc.c
 * @date	19 Nov 2018
 * @brief	GStreamer plugin to handle tensor repository
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include "gsttensor_repo.h"
#include "gsttensor_reposrc.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_reposrc_debug);
#define GST_CAT_DEFAULT gst_tensor_reposrc_debug
#define CAPS_STRING GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT

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
#define INVALID_INDEX G_MAXUINT

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
  GstPadTemplate *pad_template;
  GstCaps *pad_caps;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_reposrc_debug, "tensor_reposrc", 0,
      "Source element to handle tensor repository");

  gobject_class->set_property = gst_tensor_reposrc_set_property;
  gobject_class->get_property = gst_tensor_reposrc_get_property;
  gobject_class->dispose = gst_tensor_reposrc_dispose;

  g_object_class_install_property (gobject_class, PROP_CAPS,
      g_param_spec_boxed ("caps", "Caps",
          "Caps describing the format of the data.",
          GST_TYPE_CAPS, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SLOT_ID,
      g_param_spec_uint ("slot-index", "Slot Index", "repository slot index",
          0, INVALID_INDEX - 1, DEFAULT_INDEX,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  basesrc_class->get_caps = gst_tensor_reposrc_getcaps;
  pushsrc_class->create = gst_tensor_reposrc_create;

  gst_element_class_set_static_metadata (element_class,
      "TensorRepoSrc",
      "Source/Tensor/Repository",
      "Pop element to handle tensor repository",
      "Samsung Electronics Co., Ltd.");

  /* pad template */
  pad_caps = gst_caps_from_string (GST_TENSOR_CAP_DEFAULT "; "
      GST_TENSORS_CAP_DEFAULT);
  pad_template = gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
      pad_caps);
  gst_element_class_add_pad_template (element_class, pad_template);
  gst_caps_unref (pad_caps);
}

/**
 * @brief object initialization of tensor_reposrc
 */
static void
gst_tensor_reposrc_init (GstTensorRepoSrc * self)
{
  self->silent = TRUE;
  self->ini = FALSE;
  self->negotiation = FALSE;
  gst_tensors_config_init (&self->config);
  self->caps = NULL;
  self->set_startid = FALSE;
  self->myid = INVALID_INDEX;
}

/**
 * @brief object dispose of tensor_reposrc
 */
static void
gst_tensor_reposrc_dispose (GObject * object)
{
  GstTensorRepoSrc *self = GST_TENSOR_REPOSRC (object);

  if (self->myid != INVALID_INDEX
      && !gst_tensor_repo_remove_repodata (self->myid))
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
  GstTensorRepoSrc *self = GST_TENSOR_REPOSRC (src);
  GstCaps *cap, *check, *result;
  GstStructure *st = NULL;

  GST_DEBUG_OBJECT (self, "returning %" GST_PTR_FORMAT, self->caps);

  if (self->caps) {
    if (filter) {
      cap = gst_caps_intersect_full (filter, self->caps,
          GST_CAPS_INTERSECT_FIRST);
    } else
      cap = gst_caps_ref (self->caps);
  } else {
    if (filter) {
      cap = gst_caps_ref (filter);
    } else
      cap = gst_caps_new_any ();
  }

  check = gst_caps_from_string (CAPS_STRING);
  result = gst_caps_intersect_full (cap, check, GST_CAPS_INTERSECT_FIRST);

  if (!result) {
    GST_ELEMENT_ERROR (GST_ELEMENT (self), STREAM, WRONG_TYPE,
        ("Only Tensor/Tensors MIME are supported for now"), (NULL));
  }
  gst_caps_unref (check);
  gst_caps_unref (cap);

  st = gst_caps_get_structure (result, 0);
  gst_tensors_config_from_structure (&self->config, st);

  return result;
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
      self->o_myid = self->myid;
      self->myid = g_value_get_uint (value);
      self->negotiation = FALSE;

      gst_tensor_repo_add_repodata (self->myid, FALSE);

      if (!self->set_startid) {
        self->o_myid = self->myid;
        self->set_startid = TRUE;
      }

      if (self->o_myid != self->myid)
        gst_tensor_repo_set_changed (self->o_myid, self->myid, FALSE);
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

      if (new_caps && gst_caps_get_size (new_caps) == 1 && st
          && gst_structure_get_fraction (st, "framerate", &self->fps_n,
              &self->fps_d)) {
        GST_INFO_OBJECT (self, "Seting framerate to %d/%d", self->fps_n,
            self->fps_d);
      } else {
        self->fps_n = -1;
        self->fps_d = -1;
      }

      if (new_caps)
        gst_caps_unref (new_caps);
      self->negotiation = FALSE;
      break;
    }
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
 * @brief create dummy buffer for initialization
 */
static GstBuffer *
gst_tensor_reposrc_gen_dummy_buffer (GstTensorRepoSrc * self)
{
  GstBuffer *buf = NULL;
  GstMemory *mem;
  GstMapInfo info;
  guint i, num_tensors;
  gsize size = 0;

  buf = gst_buffer_new ();
  num_tensors = self->config.info.num_tensors;

  for (i = 0; i < num_tensors; i++) {
    size = gst_tensor_info_get_size (&self->config.info.info[i]);
    mem = gst_allocator_alloc (NULL, size, NULL);

    if (!gst_memory_map (mem, &info, GST_MAP_WRITE)) {
      gst_allocator_free (NULL, mem);
      ml_logf ("Cannot mep gst memory (tensor-repo-src)\n");
      gst_buffer_unref (buf);
      return NULL;
    }
    memset (info.data, 0, size);
    gst_memory_unmap (mem, &info);

    gst_buffer_append_memory (buf, mem);
  }

  return buf;
}

/**
 * @brief create func of tensor_reposrc
 */
static GstFlowReturn
gst_tensor_reposrc_create (GstPushSrc * src, GstBuffer ** buffer)
{
  GstTensorRepoSrc *self;
  GstBuffer *buf = NULL;
  GstCaps *caps = NULL;
  gboolean eos = FALSE;
  guint newid;

  self = GST_TENSOR_REPOSRC (src);
  gst_tensor_repo_wait ();

  if (!self->ini) {
    buf = gst_tensor_reposrc_gen_dummy_buffer (self);
    self->ini = TRUE;
  } else {
    while (!buf && !eos) {
      buf = gst_tensor_repo_get_buffer (self->myid, &eos, &newid, &caps);
    }

    if (eos)
      goto handle_eos;

    if (!self->negotiation && buf != NULL) {
      if (!gst_tensor_caps_can_intersect (self->caps, caps)) {
        GST_ELEMENT_ERROR (GST_ELEMENT (self), CORE, NEGOTIATION,
            ("Negotiation Failed! : repo_sink & repos_src"), (NULL));

        gst_tensor_repo_set_eos (self->myid);
        goto handle_eos;
      }

      self->negotiation = TRUE;
    }

    if (caps)
      gst_caps_unref (caps);
  }

  *buffer = buf;
  return GST_FLOW_OK;

handle_eos:
  if (buf)
    gst_buffer_unref (buf);
  if (caps)
    gst_caps_unref (caps);
  return GST_FLOW_EOS;
}
