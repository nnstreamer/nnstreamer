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

#include <gst/gstinfo.h>
#include <gst/gst.h>
#include <glib.h>
#include <string.h>

#include "tensor_filter.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
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

#define g_free_const(x) g_free((void*)(long)(x))

/**
 * @brief Filter subplugin should call this to register itself
 * @param[in] tfsp Tensor-Filter Sub-Plugin to be registered
 * @return TRUE if registered. FALSE is failed or duplicated.
 */
int
tensor_filter_probe (GstTensorFilterFramework * tfsp)
{
  return register_subplugin (NNS_SUBPLUGIN_FILTER, tfsp->name, tfsp);
}

/**
 * @brief filter sub-plugin may call this to unregister itself
 * @param[in] name the name of filter sub-plugin
 */
void
tensor_filter_exit (const char *name)
{
  unregister_subplugin (NNS_SUBPLUGIN_FILTER, name);
}

/**
 * @brief Find filter sub-plugin with the name
 * @param[in] name The name of tensor_filter sub-plugin
 * @return NULL if not found or the sub-plugin object.
 */
static const GstTensorFilterFramework *
tensor_filter_find (const gchar * name)
{
  return get_subplugin (NNS_SUBPLUGIN_FILTER, name);
}

GST_DEBUG_CATEGORY_STATIC (gst_tensor_filter_debug);
#define GST_CAT_DEFAULT gst_tensor_filter_debug

/**
 * @brief GstTensorFilter properties.
 */
enum
{
  PROP_0,
  PROP_SILENT,
  PROP_FRAMEWORK,
  PROP_MODEL,
  PROP_INPUT,
  PROP_INPUTTYPE,
  PROP_INPUTNAME,
  PROP_OUTPUT,
  PROP_OUTPUTTYPE,
  PROP_OUTPUTNAME,
  PROP_CUSTOM,
};

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
static GstFlowReturn gst_tensor_filter_transform_ip (GstBaseTransform * trans,
    GstBuffer * buf);
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

static void gst_tensor_filter_compare_tensors (GstTensorsInfo * info1,
    GstTensorsInfo * info2);

/**
 * @brief Open nn framework.
 */
#define gst_tensor_filter_open_fw(filter) do { \
      if (filter->prop.fw_opened == FALSE && filter->fw) { \
        if (filter->fw->open != NULL) {\
          if (filter->fw->open (&filter->prop, &filter->privateData) == 0) \
            filter->prop.fw_opened = TRUE; \
        } else {\
          filter->prop.fw_opened = TRUE; \
        } \
      } \
    } while (0)

/**
 * @brief Close nn framework.
 */
#define gst_tensor_filter_close_fw(filter) do { \
      if (filter->prop.fw_opened) { \
        if (filter->fw && filter->fw->close) \
          filter->fw->close (&filter->prop, &filter->privateData); \
        filter->prop.fw_opened = FALSE; \
        g_free_const (filter->prop.fwname); \
        filter->prop.fwname = NULL; \
        filter->fw = NULL; \
        filter->privateData = NULL; \
      } \
    } while (0)

/**
 * @brief Invoke callbacks of nn framework. Guarantees calling open for the first call.
 */
#define gst_tensor_filter_call(filter,ret,funcname,...) do { \
      gst_tensor_filter_open_fw (filter); \
      ret = -1; \
      if (filter->prop.fw_opened && filter->fw && filter->fw->funcname) { \
        ret = filter->fw->funcname (&filter->prop, &filter->privateData, __VA_ARGS__); \
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

  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_filter_set_property;
  gobject_class->get_property = gst_tensor_filter_get_property;
  gobject_class->finalize = gst_tensor_filter_finalize;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          FALSE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_FRAMEWORK,
      g_param_spec_string ("framework", "Framework",
          "Neural network framework", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_MODEL,
      g_param_spec_string ("model", "Model filepath",
          "File path to the model file", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_INPUT,
      g_param_spec_string ("input", "Input dimension",
          "Input tensor dimension from inner array, upto 4 dimensions ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_INPUTNAME,
      g_param_spec_string ("inputname", "Name of Input Tensor",
          "The Name of Input Tensor", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_INPUTTYPE,
      g_param_spec_string ("inputtype", "Input tensor element type",
          "Type of each element of the input tensor ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_OUTPUTNAME,
      g_param_spec_string ("outputname", "Name of Output Tensor",
          "The Name of Output Tensor", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_OUTPUT,
      g_param_spec_string ("output", "Output dimension",
          "Output tensor dimension from inner array, upto 4 dimensions ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_OUTPUTTYPE,
      g_param_spec_string ("outputtype", "Output tensor element type",
          "Type of each element of the output tensor ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_CUSTOM,
      g_param_spec_string ("custom", "Custom properties for subplugins",
          "Custom properties for subplugins ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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
  trans_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_tensor_filter_transform_ip);

  /* Negotiation units */
  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_filter_transform_caps);
  trans_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_tensor_filter_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_filter_set_caps);

  /* Allocation units */
  trans_class->transform_size =
      GST_DEBUG_FUNCPTR (gst_tensor_filter_transform_size);

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
  GstTensorFilterProperties *prop;

  prop = &self->prop;

  /* init NNFW properties */
  prop->fwname = NULL;
  prop->fw_opened = FALSE;
  prop->input_configured = FALSE;
  prop->output_configured = FALSE;
  prop->model_file = NULL;
  prop->custom_properties = NULL;
  gst_tensors_info_init (&prop->input_meta);
  gst_tensors_info_init (&prop->output_meta);

  /* init internal properties */
  self->fw = NULL;
  self->privateData = NULL;
  self->silent = TRUE;
  self->configured = FALSE;
  gst_tensors_config_init (&self->in_config);
  gst_tensors_config_init (&self->out_config);
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_tensor_filter_finalize (GObject * object)
{
  GstTensorFilter *self;
  GstTensorFilterProperties *prop;

  self = GST_TENSOR_FILTER (object);
  prop = &self->prop;

  gst_tensor_filter_close_fw (self);

  g_free_const (prop->fwname);
  g_free_const (prop->model_file);
  g_free_const (prop->custom_properties);

  gst_tensors_info_free (&prop->input_meta);
  gst_tensors_info_free (&prop->output_meta);

  gst_tensors_info_free (&self->in_config.info);
  gst_tensors_info_free (&self->out_config.info);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Calculate output buffer size.
 * @param self "this" pointer
 * @param index index of output tensors (if index < 0, the size of all output tensors will be returned.)
 * @return output buffer size
 */
static gsize
gst_tensor_filter_out_size (GstTensorFilter * self, gint index)
{
  GstTensorsInfo *info;
  guint i;
  gsize out_size = 0;

  g_assert (self->configured);

  info = &self->prop.output_meta;

  if (index < 0) {
    /** calculate all output tensors */
    for (i = 0; i < info->num_tensors; i++) {
      out_size += gst_tensor_info_get_size (&info->info[i]);
    }
  } else {
    g_assert (index < info->num_tensors);

    out_size = gst_tensor_info_get_size (&info->info[index]);
  }

  return out_size;
}

/**
 * @brief Setter for tensor_filter properties.
 */
static void
gst_tensor_filter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorFilter *self;
  GstTensorFilterProperties *prop;

  self = GST_TENSOR_FILTER (object);
  prop = &self->prop;

  silent_debug ("Setting property for prop %d.\n", prop_id);

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      silent_debug ("Debug mode = %d", self->silent);
      break;
    case PROP_FRAMEWORK:
    {
      const gchar *fw_name = g_value_get_string (value);

      if (self->fw != NULL) {
        gst_tensor_filter_close_fw (self);
      }
      self->fw = tensor_filter_find (fw_name);

      silent_debug ("Framework = %s\n", fw_name);
      if (self->fw == NULL) {
        GST_WARNING_OBJECT (self,
            "Cannot identify the given neural network framework, %s\n",
            fw_name);
        break;
      }

      g_free_const (prop->fwname);
      prop->fwname = g_strdup (fw_name);

      /* See if mandatory methods are filled in */
      g_assert (self->fw->invoke_NN);
      g_assert ((self->fw->getInputDimension && self->fw->getOutputDimension)
          || self->fw->setInputDimension);
      break;
    }
    case PROP_MODEL:
    {
      gchar *model_file;

      if (prop->model_file) {
        gst_tensor_filter_close_fw (self);
        g_free_const (prop->model_file);
        prop->model_file = NULL;
      }

      /* Once configures, it cannot be changed in runtime */
      model_file = g_value_dup_string (value);
      g_assert (model_file);

      silent_debug ("Model = %s\n", model_file);
      if (!g_file_test (model_file, G_FILE_TEST_IS_REGULAR)) {
        GST_ERROR_OBJECT (self, "Cannot find the model file: %s\n", model_file);
        g_free (model_file);
      } else {
        prop->model_file = model_file;
      }
      break;
    }
    case PROP_INPUT:
      g_assert (!prop->input_configured && value);
      /* Once configures, it cannot be changed in runtime */
      {
        guint num_dims;

        num_dims = gst_tensors_info_parse_dimensions_string (&prop->input_meta,
            g_value_get_string (value));

        if (prop->input_meta.num_tensors > 0 &&
            prop->input_meta.num_tensors != num_dims) {
          GST_WARNING_OBJECT (self,
              "Invalid input-dim, given param does not match with old value.");
        }

        prop->input_meta.num_tensors = num_dims;
      }
      break;
    case PROP_OUTPUT:
      g_assert (!prop->output_configured && value);
      /* Once configures, it cannot be changed in runtime */
      {
        guint num_dims;

        num_dims = gst_tensors_info_parse_dimensions_string (&prop->output_meta,
            g_value_get_string (value));

        if (prop->output_meta.num_tensors > 0 &&
            prop->output_meta.num_tensors != num_dims) {
          GST_WARNING_OBJECT (self,
              "Invalid output-dim, given param does not match with old value.");
        }

        prop->output_meta.num_tensors = num_dims;
      }
      break;
    case PROP_INPUTTYPE:
      g_assert (!prop->input_configured && value);
      /* Once configures, it cannot be changed in runtime */
      {
        guint num_types;

        num_types = gst_tensors_info_parse_types_string (&prop->input_meta,
            g_value_get_string (value));

        if (prop->input_meta.num_tensors > 0 &&
            prop->input_meta.num_tensors != num_types) {
          GST_WARNING_OBJECT (self,
              "Invalid input-type, given param does not match with old value.");
        }

        prop->input_meta.num_tensors = num_types;
      }
      break;
    case PROP_OUTPUTTYPE:
      g_assert (!prop->output_configured && value);
      /* Once configures, it cannot be changed in runtime */
      {
        guint num_types;

        num_types = gst_tensors_info_parse_types_string (&prop->output_meta,
            g_value_get_string (value));

        if (prop->output_meta.num_tensors > 0 &&
            prop->output_meta.num_tensors != num_types) {
          GST_WARNING_OBJECT (self,
              "Invalid output-type, given param does not match with old value.");
        }

        prop->output_meta.num_tensors = num_types;
      }
      break;
    case PROP_INPUTNAME:
      /* INPUTNAME is required by tensorflow to designate the order of tensors */
      g_assert (!prop->input_configured && value);
      /* Once configures, it cannot be changed in runtime */
      {
        guint num_names;

        num_names = gst_tensors_info_parse_names_string (&prop->input_meta,
            g_value_get_string (value));

        if (prop->input_meta.num_tensors > 0 &&
            prop->input_meta.num_tensors != num_names) {
          GST_WARNING_OBJECT (self,
              "Invalid input-name, given param does not match with old value.");
        }

        prop->input_meta.num_tensors = num_names;
      }
      break;
    case PROP_OUTPUTNAME:
      /* OUTPUTNAME is required by tensorflow to designate the order of tensors */
      g_assert (!prop->output_configured && value);
      /* Once configures, it cannot be changed in runtime */
      {
        guint num_names;

        num_names = gst_tensors_info_parse_names_string (&prop->output_meta,
            g_value_get_string (value));

        if (prop->output_meta.num_tensors > 0 &&
            prop->output_meta.num_tensors != num_names) {
          GST_WARNING_OBJECT (self,
              "Invalid output-name, given param does not match with old value.");
        }

        prop->output_meta.num_tensors = num_names;
      }
      break;
    case PROP_CUSTOM:
      /* In case updated custom properties in runtime! */
      g_free_const (prop->custom_properties);
      prop->custom_properties = g_value_dup_string (value);
      silent_debug ("Custom Option = %s\n", prop->custom_properties);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter for tensor_filter properties.
 */
static void
gst_tensor_filter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorFilter *self;
  GstTensorFilterProperties *prop;

  self = GST_TENSOR_FILTER (object);
  prop = &self->prop;

  silent_debug ("Getting property for prop %d.\n", prop_id);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_FRAMEWORK:
      g_value_set_string (value, prop->fwname);
      break;
    case PROP_MODEL:
      g_value_set_string (value, prop->model_file);
      break;
    case PROP_INPUT:
      if (prop->input_meta.num_tensors > 0) {
        gchar *dim_str;

        dim_str = gst_tensors_info_get_dimensions_string (&prop->input_meta);

        g_value_set_string (value, dim_str);
        g_free (dim_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_OUTPUT:
      if (prop->output_meta.num_tensors > 0) {
        gchar *dim_str;

        dim_str = gst_tensors_info_get_dimensions_string (&prop->output_meta);

        g_value_set_string (value, dim_str);
        g_free (dim_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_INPUTTYPE:
      if (prop->input_meta.num_tensors > 0) {
        gchar *type_str;

        type_str = gst_tensors_info_get_types_string (&prop->input_meta);

        g_value_set_string (value, type_str);
        g_free (type_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_OUTPUTTYPE:
      if (prop->output_meta.num_tensors > 0) {
        gchar *type_str;

        type_str = gst_tensors_info_get_types_string (&prop->output_meta);

        g_value_set_string (value, type_str);
        g_free (type_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_INPUTNAME:
      if (prop->input_meta.num_tensors > 0) {
        gchar *name_str;

        name_str = gst_tensors_info_get_names_string (&prop->input_meta);

        g_value_set_string (value, name_str);
        g_free (name_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_OUTPUTNAME:
      if (prop->output_meta.num_tensors > 0) {
        gchar *name_str;

        name_str = gst_tensors_info_get_names_string (&prop->output_meta);

        g_value_set_string (value, name_str);
        g_free (name_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_CUSTOM:
      g_value_set_string (value, prop->custom_properties);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
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
  GstTensorFilterProperties *prop;
  GstMemory *in_mem[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo in_info[NNS_TENSOR_SIZE_LIMIT];
  GstMemory *out_mem[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo out_info[NNS_TENSOR_SIZE_LIMIT];
  GstTensorMemory in_tensors[NNS_TENSOR_SIZE_LIMIT];
  GstTensorMemory out_tensors[NNS_TENSOR_SIZE_LIMIT];
  gint i, ret;

  self = GST_TENSOR_FILTER_CAST (trans);
  prop = &self->prop;

  if (G_UNLIKELY (!self->configured))
    goto unknown_format;
  if (G_UNLIKELY (!self->fw))
    goto unknown_framework;
  if (G_UNLIKELY (!prop->model_file))
    goto unknown_model;
  if (G_UNLIKELY (!self->fw->invoke_NN))
    goto unknown_invoke;

  /* 0. Check all properties. */
  silent_debug ("Invoking %s with %s model\n", self->fw->name,
      prop->model_file);

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
    out_tensors[i].size = gst_tensor_filter_out_size (self, i);
    out_tensors[i].type = prop->output_meta.info[i].type;

    /* allocate memory if allocate_in_invoke is FALSE */
    if (self->fw->allocate_in_invoke == FALSE) {
      out_mem[i] = gst_allocator_alloc (NULL, out_tensors[i].size, NULL);
      g_assert (gst_memory_map (out_mem[i], &out_info[i], GST_MAP_WRITE));

      out_tensors[i].data = out_info[i].data;
    }
  }

  /* 3. Call the filter-subplugin callback, "invoke" */
  gst_tensor_filter_call (self, ret, invoke_NN, in_tensors, out_tensors);
  /** @todo define enum to indicate status code */
  g_assert (ret >= 0);

  /* 4. Update result and free map info. */
  for (i = 0; i < prop->output_meta.num_tensors; i++) {
    if (self->fw->allocate_in_invoke) {
      /* filter-subplugin allocated new memory, update this */
      out_mem[i] =
          gst_memory_new_wrapped (0, out_tensors[i].data, out_tensors[i].size,
          0, out_tensors[i].size, out_tensors[i].data,
          self->fw->destroyNotify ? self->fw->destroyNotify : g_free);
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
 * @brief in-place transform. required vmethod of GstBaseTransform.
 */
static GstFlowReturn
gst_tensor_filter_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  /** @todo 0. Check all properties and inbuf size. */
  /** @todo 0-1. This shouldn't reach here if in-place mode if OFF with the subplugin */
  /** @todo 0-1. , which could be done at *_caps with gst_base_transform_set_in_place() */
  /** @todo 1. Resize buf if output is larger than input */
  /** @todo 2. Call the filter-subplugin callback, "invoke" */
  /** @todo 3. Return result! */
  g_assert (0);
  return GST_FLOW_ERROR;
}

/**
 * @brief Load tensor info from NN model.
 * (both input and output tensor)
 */
static void
gst_tensor_filter_load_tensor_info (GstTensorFilter * self)
{
  GstTensorFilterProperties *prop;
  GstTensorsInfo in_info, out_info;
  int res;

  prop = &self->prop;

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  /* supposed fixed in-tensor info if getInputDimension is defined. */
  if (!prop->input_configured) {
    gst_tensor_filter_call (self, res, getInputDimension, &in_info);

    if (res == 0) {
      g_assert (in_info.num_tensors > 0);

      /** if set-property called and already has info, verify it! */
      if (prop->input_meta.num_tensors > 0) {
        if (!gst_tensors_info_is_equal (&in_info, &prop->input_meta)) {
          GST_ERROR_OBJECT (self, "The input tensor is not compatible.");
          gst_tensor_filter_compare_tensors (&in_info, &prop->input_meta);
          g_assert (0);
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
    gst_tensor_filter_call (self, res, getOutputDimension, &out_info);

    if (res == 0) {
      g_assert (out_info.num_tensors > 0);

      /** if set-property called and already has info, verify it! */
      if (prop->output_meta.num_tensors > 0) {
        if (!gst_tensors_info_is_equal (&out_info, &prop->output_meta)) {
          GST_ERROR_OBJECT (self, "The output tensor is not compatible.");
          gst_tensor_filter_compare_tensors (&out_info, &prop->output_meta);
          g_assert (0);
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
 * @brief Printout the comparison results of two tensors.
 * @param[in] info1 The tensors to be shown on the left hand side
 * @param[in] info2 The tensors to be shown on the right hand side
 * @todo If this is going to be used by other elements, move this to nnstreamer/tensor_common.
 */
static void
gst_tensor_filter_compare_tensors (GstTensorsInfo * info1,
    GstTensorsInfo * info2)
{
  gchar null[] = "";
  gchar *result = null;
  gchar *dimstr = NULL;
  int i;

  for (i = 0; i < NNS_TENSOR_SIZE_LIMIT; i++) {
    gchar *line, *tmp;
    gchar *left = NULL, *right = NULL;

    if (info1->num_tensors <= i && info2->num_tensors <= i)
      break;

    if (info1->num_tensors > i) {
      dimstr = gst_tensor_get_dimension_string (info1->info[i].dimension);
      left = g_strdup_printf ("%s [%s]",
          tensor_element_typename[info1->info[i].type], dimstr);
      g_free (dimstr);
    } else {
      left = null;
    }

    if (info2->num_tensors > i) {
      dimstr = gst_tensor_get_dimension_string (info2->info[i].dimension);
      right = g_strdup_printf ("%s [%s]",
          tensor_element_typename[info2->info[i].type], dimstr);
      g_free (dimstr);
    } else {
      right = null;
    }

    line =
        g_strdup_printf ("%2d : %s | %s %s\n", i, left, right,
        g_str_equal (left, right) ? "" : "FAILED");
    if (left[0] != '\0')
      g_free (left);
    if (right[0] != '\0')
      g_free (right);

    tmp = g_strdup_printf ("%s%s", result, line);

    if (result[0] != '\0')
      g_free (result);
    result = tmp;
    g_free (line);
  }

  if (result != null) {
    /* print warning message */
    g_warning ("Tensor info :\n%s", result);
    g_free (result);
  }
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
  GstTensorFilterProperties *prop;
  GstStructure *structure;
  GstTensorsConfig in_config, out_config;

  g_return_val_if_fail (incaps != NULL, FALSE);

  prop = &self->prop;
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
      gst_tensor_filter_call (self, res, setInputDimension, &in_config.info,
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
        g_assert (0);
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

    if (self->configured) {
      /** already configured, compare to old. */
      g_assert (gst_tensors_config_is_equal (&self->in_config, &in_config));
      g_assert (gst_tensors_config_is_equal (&self->out_config, &out_config));
    } else {
      gst_tensors_info_copy (&self->in_config.info, &in_config.info);
      self->in_config.rate_n = in_config.rate_n;
      self->in_config.rate_d = in_config.rate_d;

      gst_tensors_info_copy (&self->out_config.info, &out_config.info);
      self->out_config.rate_n = out_config.rate_n;
      self->out_config.rate_d = out_config.rate_d;

      self->configured = TRUE;
    }
  }

done:
  gst_tensors_info_free (&in_config.info);
  gst_tensors_info_free (&out_config.info);
  return self->configured;
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
  GstTensorsConfig config;
  GstCaps *result;
  GstStructure *structure;

  self = GST_TENSOR_FILTER_CAST (trans);
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
    if (self->prop.output_configured) {
      /* fixed tensor info */
      config.info = self->prop.output_meta;
      result = gst_tensor_filter_caps_from_config (self, &config);
    } else {
      /* check in-tensor info to call setInputDimension */
      if (gst_tensors_info_validate (&config.info)) {
        GstTensorsInfo out_info;
        int res = -1;

        /* call setInputDimension with given input tensor */
        gst_tensors_info_init (&out_info);
        gst_tensor_filter_call (self, res, setInputDimension, &config.info,
            &out_info);

        if (res == 0) {
          config.info = out_info;
          result = gst_tensor_filter_caps_from_config (self, &config);
        } else {
          GST_ERROR_OBJECT (self, "Cannot get the output tensor info.");
          g_assert (0);
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
    if (self->prop.input_configured) {
      /* fixed tensor info */
      config.info = self->prop.input_meta;
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
  GstCaps *result;

  self = GST_TENSOR_FILTER_CAST (trans);

  silent_debug ("fixate_caps, direction = %d\n", direction);
  silent_debug_caps (caps, "caps");
  silent_debug_caps (othercaps, "othercaps");

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
  GstStructure *structure;
  GstTensorsConfig config;

  self = GST_TENSOR_FILTER_CAST (trans);

  silent_debug_caps (incaps, "incaps");
  silent_debug_caps (outcaps, "outcaps");

  if (!gst_tensor_filter_configure_tensor (self, incaps)) {
    GST_ERROR_OBJECT (self, "Failed to configure tensor.");
    return FALSE;
  }

  if (!gst_tensors_config_validate (&self->in_config)) {
    GST_ERROR_OBJECT (self, "Failed to validate input tensor.");
    return FALSE;
  }

  if (!gst_tensors_config_validate (&self->out_config)) {
    GST_ERROR_OBJECT (self, "Failed to validate output tensor.");
    return FALSE;
  }

  /** compare output tensor */
  structure = gst_caps_get_structure (outcaps, 0);
  gst_tensors_config_from_structure (&config, structure);

  if (!gst_tensors_config_is_equal (&self->out_config, &config)) {
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

  self = GST_TENSOR_FILTER_CAST (trans);

  g_assert (self->configured);

  /**
   * Consider multi-tensors.
   * Set each memory block in transform()
   */
  *othersize = 0;
  return TRUE;
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

  self = GST_TENSOR_FILTER_CAST (trans);

  gst_tensor_filter_open_fw (self);
  return TRUE;
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

  self = GST_TENSOR_FILTER_CAST (trans);

  gst_tensor_filter_close_fw (self);
  return TRUE;
}

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
NNSTREAMER_PLUGIN_INIT (tensor_filter)
{
  /**
   * debug category for filtering log messages
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_filter_debug, "tensor_filter",
      0, "tensor_filter element");

  return gst_element_register (plugin, "tensor_filter", GST_RANK_NONE,
      GST_TYPE_TENSOR_FILTER);
}
