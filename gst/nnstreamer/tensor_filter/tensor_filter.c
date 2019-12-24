/**
 * GStreamer Tensor_Filter
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
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
 * @file	tensor_filter.c
 * @date	24 May 2018
 * @brief	GStreamer plugin to use general neural network frameworks as filters
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 * @todo  set priority among properties
 * @todo  logic for dynamic properties(like model change)
 *
 * This is the main plugin for per-NN-framework plugins.
 * Specific implementations for each NN framework must be written
 * in each framework specific files; e.g., tensor_filter_tensorflow_lite.c
 *
 */

/**
 * SECTION:element-tensor_filter
 *
 * A plugin that invokes neural network models and their framework or
 * an independent shared object implementing tensor_filter_custom.h.
 * The input and output are always in the format of other/tensor or
 * other/tensors.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! tensor_filter framework=tensorflow-lite, model=./inception_v3.pb, input=3:224:224, output=1000 ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 *
 * If input is other/tensor C array input[1][224][224][3] and
 * output is other/tensor C array output[1][1][1][1000]
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>

#include "tensor_filter.h"

/** @todo rename & move this to better location */
#define EVENT_NAME_UPDATE_MODEL "evt_update_model"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!priv->silent)
#endif

/**
 * @brief Macro for debug message.
 */
#define silent_debug(...) do { \
    if (DBG) { \
      GST_DEBUG_OBJECT (self, __VA_ARGS__); \
    } \
  } while (0)

#define silent_debug_caps(caps,msg) do { \
  if (DBG) { \
    if (caps) { \
      GstStructure *caps_s; \
      gchar *caps_s_string; \
      guint caps_size, caps_idx; \
      caps_size = gst_caps_get_size (caps);\
      for (caps_idx = 0; caps_idx < caps_size; caps_idx++) { \
        caps_s = gst_caps_get_structure (caps, caps_idx); \
        caps_s_string = gst_structure_to_string (caps_s); \
        GST_DEBUG_OBJECT (self, msg " = %s\n", caps_s_string); \
        g_free (caps_s_string); \
      } \
    } \
  } \
} while (0)

#define silent_debug_info(i,msg) do { \
  if (DBG) { \
    guint info_idx; \
    gchar *dim_str; \
    GST_DEBUG_OBJECT (self, msg " total %d", (i)->num_tensors); \
    for (info_idx = 0; info_idx < (i)->num_tensors; info_idx++) { \
      dim_str = gst_tensor_get_dimension_string ((i)->info[info_idx].dimension); \
      GST_DEBUG_OBJECT (self, "[%d] type=%d dim=%s", info_idx, (i)->info[info_idx].type, dim_str); \
      g_free (dim_str); \
    } \
  } \
} while (0)

GST_DEBUG_CATEGORY_STATIC (gst_tensor_filter_debug);
#define GST_CAT_DEFAULT gst_tensor_filter_debug

/**
 * @brief Default caps string for both sink and source pad.
 */
#define CAPS_STRING GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT

/**
 * @brief The capabilities of the inputs
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

/**
 * @brief The capabilities of the outputs
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

#define gst_tensor_filter_parent_class parent_class
G_DEFINE_TYPE (GstTensorFilter, gst_tensor_filter, GST_TYPE_BASE_TRANSFORM);

/* GObject vmethod implementations */
static void gst_tensor_filter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_filter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_filter_finalize (GObject * object);

/* GstBaseTransform vmethod implementations */
static GstFlowReturn gst_tensor_filter_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static GstCaps *gst_tensor_filter_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static GstCaps *gst_tensor_filter_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_tensor_filter_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_tensor_filter_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize);
static gboolean gst_tensor_filter_start (GstBaseTransform * trans);
static gboolean gst_tensor_filter_stop (GstBaseTransform * trans);
static gboolean gst_tensor_filter_sink_event (GstBaseTransform * trans,
    GstEvent * event);

/**
 * @brief Invoke callbacks of nn framework. Guarantees calling open for the first call.
 */
#define gst_tensor_filter_call(priv,ret,funcname,...) do { \
      gst_tensor_filter_common_open_fw (priv); \
      ret = -1; \
      if (priv->prop.fw_opened && priv->fw && priv->fw->funcname) { \
        ret = priv->fw->funcname (&priv->prop, &priv->privateData, __VA_ARGS__); \
      } \
    } while (0)

/**
 * @brief initialize the tensor_filter's class
 */
static void
gst_tensor_filter_class_init (GstTensorFilterClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_filter_debug, "tensor_filter", 0,
      "Tensor filter to invoke neural network model");

  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_filter_set_property;
  gobject_class->get_property = gst_tensor_filter_get_property;
  gobject_class->finalize = gst_tensor_filter_finalize;

  gst_tensor_filter_install_properties (gobject_class);

  gst_element_class_set_details_simple (gstelement_class,
      "Tensor_Filter",
      "Converter/Filter/Tensor",
      "Handles NN Frameworks (e.g., tensorflow) as Media Filters with other/tensor type stream",
      "MyungJoo Ham <myungjoo.ham@samsung.com>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));

  /* Refer: https://gstreamer.freedesktop.org/documentation/design/element-transform.html */
  trans_class->passthrough_on_same_caps = FALSE;

  /* Processing units */
  trans_class->transform = GST_DEBUG_FUNCPTR (gst_tensor_filter_transform);

  /* Negotiation units */
  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_filter_transform_caps);
  trans_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_tensor_filter_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_filter_set_caps);

  /* Allocation units */
  trans_class->transform_size =
      GST_DEBUG_FUNCPTR (gst_tensor_filter_transform_size);

  /* setup sink event */
  trans_class->sink_event = GST_DEBUG_FUNCPTR (gst_tensor_filter_sink_event);

  /* start/stop to call open/close */
  trans_class->start = GST_DEBUG_FUNCPTR (gst_tensor_filter_start);
  trans_class->stop = GST_DEBUG_FUNCPTR (gst_tensor_filter_stop);
}

/**
 * @brief initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_filter_init (GstTensorFilter * self)
{
  GstTensorFilterPrivate *priv;

  priv = &self->priv;

  gst_tensor_filter_common_init_property (priv);
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_tensor_filter_finalize (GObject * object)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;

  self = GST_TENSOR_FILTER (object);
  priv = &self->priv;

  gst_tensor_filter_common_close_fw (priv);
  gst_tensor_filter_common_free_property (priv);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Calculate output buffer size.
 * @param self "this" pointer
 * @param index index of output tensors (if index < 0, the size of all output tensors will be returned.)
 * @return output buffer size
 */
static gsize
gst_tensor_filter_get_output_size (GstTensorFilter * self, guint index)
{
  GstTensorFilterPrivate *priv;
  GstTensorsInfo *info;

  priv = &self->priv;
  info = &priv->prop.output_meta;
  g_assert (index < info->num_tensors);

  return gst_tensor_info_get_size (&info->info[index]);
}

/**
 * @brief Setter for tensor_filter properties.
 */
static void
gst_tensor_filter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;

  self = GST_TENSOR_FILTER (object);
  priv = &self->priv;

  silent_debug ("Setting property for prop %d.\n", prop_id);

  if (!gst_tensor_filter_common_set_property (priv, prop_id, value, pspec))
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
}

/**
 * @brief Getter for tensor_filter properties.
 */
static void
gst_tensor_filter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;

  self = GST_TENSOR_FILTER (object);
  priv = &self->priv;

  silent_debug ("Getting property for prop %d.\n", prop_id);

  if (!gst_tensor_filter_common_get_property (priv, prop_id, value, pspec))
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
}

/**
 * @brief Free the data allocated for tensor transform
 * @details default function for tensor filter framework if not provided by the
 *          framework. The data is in GPtrArray - first element is private data
 *          of framework and second element is the data to be freed.
 */
static void
gst_tensor_filter_destroy_notify (void *data)
{
  GPtrArray *array = (GPtrArray *) data;
  GstTensorFilter *self = (GstTensorFilter *) g_ptr_array_index (array, 0);
  void *tensor_data = (void *) g_ptr_array_index (array, 1);
  g_ptr_array_free (array, TRUE);

  if (self->priv.fw->destroyNotify) {
    self->priv.fw->destroyNotify (&self->priv.privateData, tensor_data);
  } else {
    g_free (tensor_data);
  }
}

/**
 * @brief non-ip transform. required vmethod of GstBaseTransform.
 */
static GstFlowReturn
gst_tensor_filter_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;
  GstTensorFilterProperties *prop;
  GstMemory *in_mem[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo in_info[NNS_TENSOR_SIZE_LIMIT];
  GstMemory *out_mem[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo out_info[NNS_TENSOR_SIZE_LIMIT];
  GstTensorMemory in_tensors[NNS_TENSOR_SIZE_LIMIT];
  GstTensorMemory out_tensors[NNS_TENSOR_SIZE_LIMIT];
  guint i;
  gint ret;
  gboolean allocate_in_invoke;

  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;
  prop = &priv->prop;

  if (G_UNLIKELY (!priv->configured))
    goto unknown_format;
  if (G_UNLIKELY (!priv->fw))
    goto unknown_framework;
  if (G_UNLIKELY (!priv->fw->run_without_model) &&
      G_UNLIKELY (!(prop->model_files &&
              prop->num_models > 0 && prop->model_files[0])))
    goto unknown_model;
  if (G_UNLIKELY (!priv->fw->invoke_NN))
    goto unknown_invoke;

  /* 0. Check all properties. */
  silent_debug ("Invoking %s with %s model\n", priv->fw->name,
      GST_STR_NULL (prop->model_files[0]));
  allocate_in_invoke = gst_tensor_filter_allocate_in_invoke (priv);

  /* 1. Set input tensors from inbuf. */
  g_assert (gst_buffer_n_memory (inbuf) == prop->input_meta.num_tensors);

  for (i = 0; i < prop->input_meta.num_tensors; i++) {
    in_mem[i] = gst_buffer_peek_memory (inbuf, i);
    g_assert (gst_memory_map (in_mem[i], &in_info[i], GST_MAP_READ));

    in_tensors[i].data = in_info[i].data;
    in_tensors[i].size = in_info[i].size;
    in_tensors[i].type = prop->input_meta.info[i].type;
  }

  /* 2. Prepare output tensors. */
  g_assert (outbuf);
  g_assert (gst_buffer_get_size (outbuf) == 0);

  for (i = 0; i < prop->output_meta.num_tensors; i++) {
    out_tensors[i].data = NULL;
    out_tensors[i].size = gst_tensor_filter_get_output_size (self, i);
    out_tensors[i].type = prop->output_meta.info[i].type;

    /* allocate memory if allocate_in_invoke is FALSE */
    if (allocate_in_invoke == FALSE) {
      out_mem[i] = gst_allocator_alloc (NULL, out_tensors[i].size, NULL);
      g_assert (gst_memory_map (out_mem[i], &out_info[i], GST_MAP_WRITE));

      out_tensors[i].data = out_info[i].data;
    }
  }

  /* 3. Call the filter-subplugin callback, "invoke" */
  gst_tensor_filter_call (priv, ret, invoke_NN, in_tensors, out_tensors);
  /** @todo define enum to indicate status code */
  g_assert (ret >= 0);

  /* 4. Update result and free map info. */
  for (i = 0; i < prop->output_meta.num_tensors; i++) {
    if (allocate_in_invoke) {
      GPtrArray *data_array = g_ptr_array_new ();
      g_ptr_array_add (data_array, (gpointer) self);
      g_ptr_array_add (data_array, (gpointer) out_tensors[i].data);

      /* filter-subplugin allocated new memory, update this */
      out_mem[i] =
          gst_memory_new_wrapped (0, out_tensors[i].data, out_tensors[i].size,
          0, out_tensors[i].size, (gpointer) data_array,
          gst_tensor_filter_destroy_notify);
    } else {
      gst_memory_unmap (out_mem[i], &out_info[i]);
    }

    /* append the memory block to outbuf */
    gst_buffer_append_memory (outbuf, out_mem[i]);
  }

  for (i = 0; i < prop->input_meta.num_tensors; i++) {
    gst_memory_unmap (in_mem[i], &in_info[i]);
  }

  /* 5. Return result! */
  if (ret > 0) {
    /** @todo define enum to indicate status code */
    /* drop this buffer */
    return GST_BASE_TRANSFORM_FLOW_DROPPED;
  }

  return GST_FLOW_OK;
unknown_format:
  GST_ELEMENT_ERROR (self, CORE, NOT_IMPLEMENTED, (NULL), ("unknown format"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_framework:
  /**
    * This is fatal; if framework is not configured until this stage,
    * it means that an extension is missing or not configured.
    * We need readable messages for non-developers
    */
  g_error
      ("\nA nnstreamer extension is not installed or framework property of tensor_filter is incorrect: [%s] is not found.\n\n",
      prop->fwname);
  GST_ELEMENT_ERROR (self, CORE, NOT_IMPLEMENTED, (NULL),
      ("framework not configured"));
  return GST_FLOW_ERROR;
unknown_model:
  GST_ELEMENT_ERROR (self, CORE, NOT_IMPLEMENTED, (NULL),
      ("model filepath not configured"));
  return GST_FLOW_ERROR;
unknown_invoke:
  GST_ELEMENT_ERROR (self, CORE, NOT_IMPLEMENTED, (NULL),
      ("invoke function is not defined"));
  return GST_FLOW_ERROR;
}

/**
 * @brief Load tensor info from NN model.
 * (both input and output tensor)
 */
static void
gst_tensor_filter_load_tensor_info (GstTensorFilter * self)
{
  GstTensorFilterPrivate *priv;
  GstTensorFilterProperties *prop;
  GstTensorsInfo in_info, out_info;
  int res;

  priv = &self->priv;
  prop = &priv->prop;

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  /* supposed fixed in-tensor info if getInputDimension is defined. */
  if (!prop->input_configured) {
    gst_tensor_filter_call (priv, res, getInputDimension, &in_info);

    if (res == 0) {
      g_assert (in_info.num_tensors > 0);

      /** if set-property called and already has info, verify it! */
      if (prop->input_meta.num_tensors > 0) {
        if (!gst_tensors_info_is_equal (&in_info, &prop->input_meta)) {
          GST_ERROR_OBJECT (self, "The input tensor is not compatible.");
          gst_tensor_filter_compare_tensors (&in_info, &prop->input_meta);
          goto done;
        }
      } else {
        gst_tensors_info_copy (&prop->input_meta, &in_info);
      }

      prop->input_configured = TRUE;
      silent_debug_info (&in_info, "input tensor");
    }
  }

  /* supposed fixed out-tensor info if getOutputDimension is defined. */
  if (!prop->output_configured) {
    gst_tensor_filter_call (priv, res, getOutputDimension, &out_info);

    if (res == 0) {
      g_assert (out_info.num_tensors > 0);

      /** if set-property called and already has info, verify it! */
      if (prop->output_meta.num_tensors > 0) {
        if (!gst_tensors_info_is_equal (&out_info, &prop->output_meta)) {
          GST_ERROR_OBJECT (self, "The output tensor is not compatible.");
          gst_tensor_filter_compare_tensors (&out_info, &prop->output_meta);
          goto done;
        }
      } else {
        gst_tensors_info_copy (&prop->output_meta, &out_info);
      }

      prop->output_configured = TRUE;
      silent_debug_info (&out_info, "output tensor");
    }
  }

done:
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Configure input and output tensor info from incaps.
 * @param self "this" pointer
 * @param incaps received caps for sink pad
 * @return TRUE if fully configured
 */
static gboolean
gst_tensor_filter_configure_tensor (GstTensorFilter * self,
    const GstCaps * incaps)
{
  GstTensorFilterPrivate *priv;
  GstTensorFilterProperties *prop;
  GstStructure *structure;
  GstTensorsConfig in_config, out_config;

  g_return_val_if_fail (incaps != NULL, FALSE);

  priv = &self->priv;
  prop = &priv->prop;
  gst_tensors_config_init (&in_config);
  gst_tensors_config_init (&out_config);

  /**
   * GstTensorFilter has to parse the tensor dimension and type from NN model.
   * 1. Call functions getInputDimension and getOutputDimension to get the dimension and type.
   * 2. If these functions are not defined, call setInputDimension with parsed info from caps.
   * 3. If set-prop configured dimension, verify the dimension with fw callbacks.
   */
  gst_tensor_filter_load_tensor_info (self);

  structure = gst_caps_get_structure (incaps, 0);
  gst_tensors_config_from_structure (&in_config, structure);

  /**
   * Check configuration from caps.
   * If true, fully configured tensor info from caps.
   */
  if (gst_tensors_config_validate (&in_config)) {
    /** if set-property called and already has info, verify it! */
    if (prop->input_meta.num_tensors > 0) {
      if (!gst_tensors_info_is_equal (&in_config.info, &prop->input_meta)) {
        GST_ERROR_OBJECT (self, "The input tensor is not compatible.");
        gst_tensor_filter_compare_tensors (&in_config.info, &prop->input_meta);
        goto done;
      }
    } else {
      gst_tensors_info_copy (&prop->input_meta, &in_config.info);
    }

    prop->input_configured = TRUE;

    /** call setInputDimension if output tensor is not configured */
    if (!prop->output_configured) {
      GstTensorsInfo out_info;
      int res;

      gst_tensors_info_init (&out_info);
      gst_tensor_filter_call (priv, res, setInputDimension, &in_config.info,
          &out_info);

      if (res == 0) {
        /** if set-property called and already has info, verify it! */
        if (prop->output_meta.num_tensors > 0) {
          if (!gst_tensors_info_is_equal (&out_info, &prop->output_meta)) {
            GST_ERROR_OBJECT (self, "The output tensor is not compatible.");
            gst_tensor_filter_compare_tensors (&out_info, &prop->output_meta);
            gst_tensors_info_free (&out_info);
            goto done;
          }
        } else {
          gst_tensors_info_copy (&prop->output_meta, &out_info);
        }

        prop->output_configured = TRUE;
        silent_debug_info (&out_info, "output tensor");
      }

      gst_tensors_info_free (&out_info);

      if (!prop->output_configured) {
        GST_ERROR_OBJECT (self, "Failed to get output tensor info.\n");
        goto done;
      }
    }

    /**
     * @todo framerate of output tensors
     * How can we update the framerate?
     * GstTensorFilter cannot assure the framerate.
     * Simply set the framerate of out-tensor from incaps.
     */
    gst_tensors_info_copy (&out_config.info, &prop->output_meta);
    out_config.rate_n = in_config.rate_n;
    out_config.rate_d = in_config.rate_d;

    if (priv->configured) {
      /** already configured, compare to old. */
      g_assert (gst_tensors_config_is_equal (&priv->in_config, &in_config));
      g_assert (gst_tensors_config_is_equal (&priv->out_config, &out_config));
    } else {
      gst_tensors_info_copy (&priv->in_config.info, &in_config.info);
      priv->in_config.rate_n = in_config.rate_n;
      priv->in_config.rate_d = in_config.rate_d;

      gst_tensors_info_copy (&priv->out_config.info, &out_config.info);
      priv->out_config.rate_n = out_config.rate_n;
      priv->out_config.rate_d = out_config.rate_d;

      priv->configured = TRUE;
    }
  }

done:
  gst_tensors_info_free (&in_config.info);
  gst_tensors_info_free (&out_config.info);
  return priv->configured;
}

/**
 * @brief Get caps for given config.
 * @param self "this" pointer
 * @param config tensor config info
 */
static GstCaps *
gst_tensor_filter_caps_from_config (GstTensorFilter * self,
    GstTensorsConfig * config)
{
  GstCaps *caps;

  g_return_val_if_fail (config != NULL, NULL);

  if (config->info.num_tensors < 2) {
    GstTensorConfig c;

    /**
     * supposed other/tensor if the number of tensor is less than 2.
     */
    c.info = config->info.info[0];
    c.rate_n = config->rate_n;
    c.rate_d = config->rate_d;

    caps = gst_tensor_caps_from_config (&c);
  } else {
    caps = gst_tensors_caps_from_config (config);
  }

  return caps;
}

/**
 * @brief configure tensor-srcpad cap from "proposed" cap.
 *
 * @trans ("this" pointer)
 * @direction (why do we need this?)
 * @caps sinkpad cap (if direction GST_PAD_SINK)
 * @filter this element's cap (don't know specifically.)
 *
 * Be careful not to fix/set caps at this stage. Negotiation not completed yet.
 */
static GstCaps *
gst_tensor_filter_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;
  GstTensorFilterProperties *prop;
  GstTensorsConfig config;
  GstCaps *result;
  GstStructure *structure;

  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;
  prop = &priv->prop;

  /* Not ready */
  if (priv->fw == NULL)
    return NULL;

  gst_tensors_config_init (&config);

  silent_debug ("Direction = %d\n", direction);
  silent_debug_caps (caps, "from");
  silent_debug_caps (filter, "filter");

  /**
   * GstTensorFilter has to parse the tensor dimension and type from NN model.
   * In this stage, in-caps may not be fixed yet.
   * To get the tensor info and generate pad-caps, call getInputDimension and getOutputDimension.
   * If these functions are not defined, we have to call setInputDimension, and then it will fully configure the tensor info.
   *
   * @todo how to set the framerate of output tensors
   */
  gst_tensor_filter_load_tensor_info (self);

  structure = gst_caps_get_structure (caps, 0);
  gst_tensors_config_from_structure (&config, structure);

  if (direction == GST_PAD_SINK) {
    /* caps: sink pad. get src pad info */
    if (prop->output_configured) {
      /* caps with sub-plugin's tensor info */
      config.info = prop->output_meta;
      result = gst_tensor_filter_caps_from_config (self, &config);
    } else {
      /* check in-tensor info to call setInputDimension */
      if (gst_tensors_info_validate (&config.info)) {
        GstTensorsInfo out_info;
        int res = -1;

        /* call setInputDimension with given input tensor */
        gst_tensors_info_init (&out_info);
        gst_tensor_filter_call (priv, res, setInputDimension, &config.info,
            &out_info);

        if (res == 0) {
          config.info = out_info;
          result = gst_tensor_filter_caps_from_config (self, &config);
        } else {
          GST_ERROR_OBJECT (self, "Cannot get the output tensor info.");
          result = gst_caps_from_string (CAPS_STRING);
        }

        gst_tensors_info_free (&out_info);
      } else {
        /* we don't know the exact tensor info yet */
        result = gst_caps_from_string (CAPS_STRING);
      }
    }
  } else {
    /* caps: src pad. get sink pad info */
    if (prop->input_configured) {
      /* caps with sub-plugin's tensor info */
      config.info = prop->input_meta;
      result = gst_tensor_filter_caps_from_config (self, &config);
    } else {
      /* we don't know the exact tensor info from src pad caps */
      result = gst_caps_from_string (CAPS_STRING);
    }
  }

  if (filter && gst_caps_get_size (filter) > 0) {
    GstCaps *intersection;

    /**
     * @todo We do not have a testcase hitting here. Thus, we do not ensure the validity here.
     * However, according to gstreamer doxygen entry, if filter is given, that's not to be ignored.
     * For now, we assume that if caps-size is 0, filter is "ANY".
     */

    intersection =
        gst_caps_intersect_full (filter, result, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (result);
    result = intersection;
  }

  silent_debug_caps (result, "to");
  return result;
}

/**
 * @brief fixate caps. required vmethod of GstBaseTransform.
 */
static GstCaps *
gst_tensor_filter_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;
  GstCaps *result;

  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;

  silent_debug ("fixate_caps, direction = %d\n", direction);
  silent_debug_caps (caps, "caps");
  silent_debug_caps (othercaps, "othercaps");

  /** Removes no-used-variable warning for priv in when DBG is set */
  if (priv->fw == NULL) {
    gst_caps_unref (othercaps);
    return NULL;
  }

  /**
   * To get the out-caps, GstTensorFilter has to parse tensor info from NN model.
   */

  result = gst_tensor_filter_transform_caps (trans, direction, caps, othercaps);
  gst_caps_unref (othercaps);
  result = gst_caps_make_writable (result);
  result = gst_caps_fixate (result);

  silent_debug_caps (result, "result");
  return result;
}

/**
 * @brief set caps. required vmethod of GstBaseTransform.
 */
static gboolean
gst_tensor_filter_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;
  GstStructure *structure;
  GstTensorsConfig config;

  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;

  silent_debug_caps (incaps, "incaps");
  silent_debug_caps (outcaps, "outcaps");

  if (!gst_tensor_filter_configure_tensor (self, incaps)) {
    GST_ERROR_OBJECT (self, "Failed to configure tensor.");
    return FALSE;
  }

  if (!gst_tensors_config_validate (&priv->in_config)) {
    GST_ERROR_OBJECT (self, "Failed to validate input tensor.");
    return FALSE;
  }

  if (!gst_tensors_config_validate (&priv->out_config)) {
    GST_ERROR_OBJECT (self, "Failed to validate output tensor.");
    return FALSE;
  }

  /** compare output tensor */
  structure = gst_caps_get_structure (outcaps, 0);
  gst_tensors_config_from_structure (&config, structure);

  if (!gst_tensors_config_is_equal (&priv->out_config, &config)) {
    GST_ERROR_OBJECT (self, "Invalid outcaps.");
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Tell the framework the required size of buffer based on the info of the other side pad. optional vmethod of BaseTransform
 *
 * We cannot directly get the value from size value, we need to review the pad-caps.
 * This is called when non-ip mode is used.
 */
static gboolean
gst_tensor_filter_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;

  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;

  g_assert (priv->configured);

  /**
   * Consider multi-tensors.
   * Set each memory block in transform()
   */
  *othersize = 0;
  return TRUE;
}

/**
 * @brief Event handler for sink pad of tensor filter.
 * @param trans "this" pointer
 * @param event a passed event object
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_filter_sink_event (GstBaseTransform * trans, GstEvent * event)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;

  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CUSTOM_DOWNSTREAM:
    {
      const GstStructure *structure = gst_event_get_structure (event);
      int ret = -1;

      if (structure == NULL ||
          !gst_structure_has_name (structure, EVENT_NAME_UPDATE_MODEL))
        break;

      if (priv->is_updatable) {
        const GValue *value =
            gst_structure_get_value (structure, "model_files");

        if (value != NULL) {
          g_object_set (self, "model", value, NULL);
          ret = 0;
        }
      }

      gst_event_unref (event);

      return (ret == 0);
    }
    default:
      break;
  }

  /** other events are handled in the default event handler */
  return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event (trans, event);
}

/**
 * @brief Called when the element starts processing. optional vmethod of BaseTransform
 * @param trans "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_filter_start (GstBaseTransform * trans)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;

  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;

  /* If it is not configured properly, don't allow to start! */
  if (priv->fw == NULL)
    return FALSE;

  gst_tensor_filter_common_open_fw (priv);
  return priv->prop.fw_opened;
}

/**
 * @brief Called when the element stops processing. optional vmethod of BaseTransform
 * @param trans "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_filter_stop (GstBaseTransform * trans)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;

  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;

  gst_tensor_filter_common_close_fw (priv);
  return TRUE;
}
