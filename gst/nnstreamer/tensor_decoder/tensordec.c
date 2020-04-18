/**
 * GStreamer / NNStreamer tensor_decoder main
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
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
 *
 */
/**
 * @file        tensordec.c
 * @date        26 Mar 2018
 * @brief       GStreamer plugin to convert tensors (as a filter for other general neural network filters) to other media types
 * @see    	https://github.com/nnstreamer/nnstreamer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         gst_tensordec_transform_size () may be incorrect if direction is SINK.
 *
 */

/**
 * SECTION:element-tensordec
 *
 * A filter that converts tensor stream for NN frameworks to media stream.
 * The input is always in the format of other/tensor
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesink ! tensor_decoder ! fakesrc silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include "tensordec.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

#define silent_debug(...) do { \
    if (DBG) { \
      GST_DEBUG_OBJECT (self, __VA_ARGS__); \
    } \
  } while (0)

#define silent_debug_caps(caps,msg) do {\
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

GST_DEBUG_CATEGORY_STATIC (gst_tensordec_debug);
#define GST_CAT_DEFAULT gst_tensordec_debug

/**
 * @brief Properties.
 */
enum
{
  PROP_0,
  PROP_SILENT,
  PROP_MODE,
  PROP_MODE_OPTION1,
  PROP_MODE_OPTION2,
  PROP_MODE_OPTION3,
  PROP_MODE_OPTION4,
  PROP_MODE_OPTION5,
  PROP_MODE_OPTION6,
  PROP_MODE_OPTION7,
  PROP_MODE_OPTION8,
  PROP_MODE_OPTION9,
  PROP_SUBPLUGINS
};

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE

/**
 * @brief Support multi-tensor along with single-tensor as the input
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
    GST_STATIC_CAPS ("ANY"));

#define gst_tensordec_parent_class parent_class
G_DEFINE_TYPE (GstTensorDec, gst_tensordec, GST_TYPE_BASE_TRANSFORM);

/** GObject vmethod implementations */
static void gst_tensordec_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensordec_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensordec_class_finalize (GObject * object);

/** GstBaseTransform vmethod implementations */
static GstFlowReturn gst_tensordec_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static GstCaps *gst_tensordec_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static GstCaps *gst_tensordec_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_tensordec_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_tensordec_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize);

/**
 * @brief Validate decoder sub-plugin's data.
 */
static gboolean
nnstreamer_decoder_validate (const GstTensorDecoderDef * decoder)
{
  if (!decoder || !decoder->modename) {
    /* invalid name */
    return FALSE;
  }

  if (!decoder->init || !decoder->getOutCaps || !decoder->decode) {
    /* invalid methods in decoder sub-plugin */
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Decoder's sub-plugin should call this function to register itself.
 * @param[in] decoder Decoder sub-plugin to be registered.
 * @return TRUE if registered. FALSE is failed or duplicated.
 */
int
nnstreamer_decoder_probe (GstTensorDecoderDef * decoder)
{
  g_return_val_if_fail (nnstreamer_decoder_validate (decoder), FALSE);
  return register_subplugin (NNS_SUBPLUGIN_DECODER, decoder->modename, decoder);
}

/**
 * @brief Decoder's sub-plugin may call this to unregister itself.
 * @param[in] name The name of decoder sub-plugin.
 */
void
nnstreamer_decoder_exit (const char *name)
{
  unregister_subplugin (NNS_SUBPLUGIN_DECODER, name);
}

/**
 * @brief Find decoder sub-plugin with the name.
 * @param[in] name The name of decoder sub-plugin.
 * @return NULL if not found or the sub-plugin object has an error.
 */
const GstTensorDecoderDef *
nnstreamer_decoder_find (const char *name)
{
  return get_subplugin (NNS_SUBPLUGIN_DECODER, name);
}

/**
 * @brief Macro to clean sub-plugin data
 */
#define gst_tensor_decoder_clean_plugin(self) do { \
    if (self->decoder) { \
      if (self->decoder->exit) \
        self->decoder->exit (&self->plugin_data); \
      else \
        g_free (self->plugin_data); \
      self->plugin_data = NULL; \
    } \
  } while (0)

/**
 * @brief Get media caps from tensor config
 * @param self "this" pointer
 * @param config tensor config info
 * @return caps for media type
 */
static GstCaps *
gst_tensordec_media_caps_from_tensor (GstTensorDec * self,
    const GstTensorsConfig * config)
{
  g_return_val_if_fail (config != NULL, NULL);

  if (self->decoder == NULL) {
    GST_ERROR_OBJECT (self, "Decoder plugin is not yet configured.");
    return NULL;
  }

  /* call sub-plugin vmethod */
  return self->decoder->getOutCaps (&self->plugin_data, config);
}

/**
 * @brief Parse structure and return media caps
 * @param self "this" pointer
 * @param structure structure to be interpreted
 */
static GstCaps *
gst_tensordec_media_caps_from_structure (GstTensorDec * self,
    const GstStructure * structure)
{
  GstTensorsConfig config;
  GstCaps *result = NULL;

  if (gst_tensors_config_from_structure (&config, structure)) {
    result = gst_tensordec_media_caps_from_tensor (self, &config);
  }

  if (result == NULL) {
    /* we cannot specify the media type */
    result = gst_caps_new_any ();
  }

  return result;
}

/**
 * @brief Check tensor config is consistent
 * @param self "this" pointer to check consistency
 * @param t_info newly configured tensor metadata
 */
static gboolean
gst_tensordec_check_consistency (GstTensorDec * self, GstTensorsConfig * config)
{
  g_return_val_if_fail (self != NULL, FALSE);
  g_return_val_if_fail (config != NULL, FALSE);

  if (self->configured) {
    return gst_tensors_config_is_equal (&self->tensor_config, config);
  }

  /** not configured yet */
  return FALSE;
}

/**
 * @brief initialize the tensordec's class
 */
static void
gst_tensordec_class_init (GstTensorDecClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensordec_debug, "tensor_decoder", 0,
      "Element to convert tensor to media stream");

  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensordec_set_property;
  gobject_class->get_property = gst_tensordec_get_property;
  gobject_class->finalize = gst_tensordec_class_finalize;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODE,
      g_param_spec_string ("mode", "Mode", "Decoder mode", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODE_OPTION1,
      g_param_spec_string ("option1", "Mode option 1",
          "option for specific decoder modes, 1st one.", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODE_OPTION2,
      g_param_spec_string ("option2", "Mode option 2",
          "option for specific decoder modes, 2nd one.", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODE_OPTION3,
      g_param_spec_string ("option3", "Mode option 3",
          "option for specific decoder modes, 3rd one.", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODE_OPTION4,
      g_param_spec_string ("option4", "Mode option 4",
          "option for specific decoder modes, 4th one.", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODE_OPTION5,
      g_param_spec_string ("option5", "Mode option 5",
          "option for specific decoder modes, 5th one.", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODE_OPTION6,
      g_param_spec_string ("option6", "Mode option 6",
          "option for specific decoder modes, 6th one.", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODE_OPTION7,
      g_param_spec_string ("option7", "Mode option 7",
          "option for specific decoder modes, 7th one.", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODE_OPTION8,
      g_param_spec_string ("option8", "Mode option 8",
          "option for specific decoder modes, 8th one.", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODE_OPTION9,
      g_param_spec_string ("option9", "Mode option 9",
          "option for specific decoder modes, 9th one.", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SUBPLUGINS,
      g_param_spec_string ("sub-plugins", "Sub-plugins",
          "Registrable sub-plugins list", "",
          G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_details_simple (gstelement_class,
      "TensorDecoder",
      "Converter/Tensor",
      "Converts tensor stream of C-Array for neural network framework filters to audio or video stream",
      "Jijoong Moon <jijoong.moon@samsung.com>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));

  /** Refer: https://gstreamer.freedesktop.org/documentation/design/element-transform.html */
  trans_class->passthrough_on_same_caps = FALSE;
  trans_class->transform_ip_on_passthrough = FALSE;

  /** Processing units */
  trans_class->transform = GST_DEBUG_FUNCPTR (gst_tensordec_transform);

  /** Negotiation units */
  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_tensordec_transform_caps);
  trans_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_tensordec_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensordec_set_caps);

  /** Allocation units */
  trans_class->transform_size =
      GST_DEBUG_FUNCPTR (gst_tensordec_transform_size);
}

/**
 * @brief initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensordec_init (GstTensorDec * self)
{
  guint i;

  self->silent = DEFAULT_SILENT;
  self->configured = FALSE;
  self->negotiated = FALSE;
  self->decoder = NULL;
  self->plugin_data = NULL;

  for (i = 0; i < TensorDecMaxOpNum; i++)
    self->option[i] = NULL;

  gst_tensors_config_init (&self->tensor_config);
}

/**
 * @brief Process plugin (self->decoder) with given options if available
 * @retval FALSE if error. TRUE if OK (or SKIP)
 */
static gboolean
gst_tensordec_process_plugin_options (GstTensorDec * self, guint opnum)
{
  g_assert (opnum < TensorDecMaxOpNum);
  if (self->decoder == NULL)
    return TRUE;                /* decoder plugin not available. */
  if (self->decoder->setOption == NULL)
    return TRUE;                /* This decoder cannot process options */
  if (self->option[opnum] == NULL)
    return TRUE;                /* No option to process */
  return self->decoder->setOption (&self->plugin_data, opnum,
      self->option[opnum]);
}

/**
 * @brief A macro to process incoming per-mode option
 * @param[in] opnum The option number (1 to TensorDecMaxOpNum)
 */
#define PROP_MODE_OPTION(opnum) \
    case PROP_MODE_OPTION ## opnum: \
      g_free (self->option[(opnum) - 1]); \
      self->option[(opnum) - 1] = g_value_dup_string (value); \
      if (gst_tensordec_process_plugin_options (self, (opnum) - 1) != TRUE) \
        GST_ERROR_OBJECT (self, "Configuring option for tensor-decoder failed (option %d = %s)", \
            (opnum), self->option[(opnum) - 1]); \
      break

/**
 * @brief Set property (GObject vmethod)
 */
static void
gst_tensordec_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorDec *self;

  self = GST_TENSOR_DECODER (object);

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      break;
    case PROP_MODE:
    {
      const GstTensorDecoderDef *decoder;
      const gchar *mode_string;
      guint i;
      int status;

      mode_string = g_value_get_string (value);
      decoder = nnstreamer_decoder_find (mode_string);

      /* See if we are using "plugin" */
      if (nnstreamer_decoder_validate (decoder)) {
        silent_debug ("tensor_decoder plugin mode (%s)\n", mode_string);

        if (decoder == self->decoder) {
          /* Already configured??? */
          GST_WARNING_OBJECT (self,
              "nnstreamer tensor_decoder %s is already confgured.\n",
              mode_string);
        } else {
          /* Changing decoder. Deallocate the previous */
          gst_tensor_decoder_clean_plugin (self);
          self->decoder = decoder;
        }

        status = self->decoder->init (&self->plugin_data);
        /** @todo Do proper error handling */
        g_assert (status);

        for (i = 0; i < TensorDecMaxOpNum; i++)
          if (!gst_tensordec_process_plugin_options (self, i))
            GST_WARNING_OBJECT (self,
                "Failed to configure while setting the option %d.", (i + 1));
      } else {
        GST_ERROR_OBJECT (self,
            "The given mode for tensor_decoder, %s, is unrecognized.\n",
            mode_string);
        gst_tensor_decoder_clean_plugin (self);
        self->decoder = NULL;
      }

      break;
    }
      PROP_MODE_OPTION (1);
      PROP_MODE_OPTION (2);
      PROP_MODE_OPTION (3);
      PROP_MODE_OPTION (4);
      PROP_MODE_OPTION (5);
      PROP_MODE_OPTION (6);
      PROP_MODE_OPTION (7);
      PROP_MODE_OPTION (8);
      PROP_MODE_OPTION (9);
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief A macro to read per-mode option
 * @param[in] opnum The option number (1 to TensorDecMaxOpNum)
 */
#define PROP_READ_OPTION(opnum) \
    case PROP_MODE_OPTION ## opnum: \
      g_value_set_string (value, self->option[opnum - 1]); \
      break

/**
 * @brief Get property (GObject vmethod)
 */
static void
gst_tensordec_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorDec *self;

  self = GST_TENSOR_DECODER (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_MODE:
      if (self->decoder)
        g_value_set_string (value, self->decoder->modename);
      else
        g_value_set_string (value, "");
      break;
      PROP_READ_OPTION (1);
      PROP_READ_OPTION (2);
      PROP_READ_OPTION (3);
      PROP_READ_OPTION (4);
      PROP_READ_OPTION (5);
      PROP_READ_OPTION (6);
      PROP_READ_OPTION (7);
      PROP_READ_OPTION (8);
      PROP_READ_OPTION (9);
    case PROP_SUBPLUGINS:
    {
      subplugin_info_s sinfo;
      guint i, total;

      total = nnsconf_get_subplugin_info (NNSCONF_PATH_DECODERS, &sinfo);

      if (total > 0) {
        GString *subplugins;
        const gchar *prefix_str;
        gsize prefix, extension, len;

        subplugins = g_string_new (NULL);

        prefix_str = nnsconf_get_subplugin_name_prefix (NNSCONF_PATH_DECODERS);
        prefix = strlen (prefix_str);
        extension = strlen (NNSTREAMER_SO_FILE_EXTENSION);

        for (i = 0; i < total; ++i) {
          /* remove file extension */
          len = strlen (sinfo.names[i]) - prefix - extension;
          g_string_append_len (subplugins, sinfo.names[i] + prefix, len);

          if (i < total - 1) {
            g_string_append (subplugins, ",");
          }
        }

        g_value_take_string (value, g_string_free (subplugins, FALSE));
      } else {
        g_value_set_string (value, "");
      }

      break;
    }
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Finalize instance (GObject vmethod)
 */
static void
gst_tensordec_class_finalize (GObject * object)
{
  GstTensorDec *self;
  guint i;

  self = GST_TENSOR_DECODER (object);

  gst_tensor_decoder_clean_plugin (self);

  for (i = 0; i < TensorDecMaxOpNum; ++i) {
    g_free (self->option[i]);
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Configure tensor metadata from sink caps
 */
static gboolean
gst_tensordec_configure (GstTensorDec * self, const GstCaps * caps)
{
  GstStructure *structure;
  GstTensorsConfig config;

  /** This caps is coming from tensor */
  structure = gst_caps_get_structure (caps, 0);

  if (!gst_tensors_config_from_structure (&config, structure)) {
    GST_ERROR_OBJECT (self, "Cannot configure tensor from structure");
    return FALSE;
  }

  if (!gst_tensors_config_validate (&config)) {
    GST_ERROR_OBJECT (self, "Not configured yet");
    return FALSE;
  }

  if (self->configured && !gst_tensordec_check_consistency (self, &config)) {
    GST_ERROR_OBJECT (self, "Mismatched to old metadata");
    return FALSE;
  }

  if (self->decoder == NULL) {
    GST_ERROR_OBJECT (self, "Decoder plugin is not yet configured.");
    return FALSE;
  }

  self->tensor_config = config;
  self->configured = TRUE;
  return TRUE;
}

/**
 * @brief non-ip transform. required vmethod for BaseTransform class.
 */
static GstFlowReturn
gst_tensordec_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstTensorDec *self;
  GstFlowReturn res;
  gboolean status;

  self = GST_TENSOR_DECODER_CAST (trans);

  if (G_UNLIKELY (!self->negotiated))
    goto unknown_tensor;
  if (G_UNLIKELY (!self->configured))
    goto unknown_format;

  if (self->decoder) {
    GstMemory *in_mem[NNS_TENSOR_SIZE_LIMIT];
    GstMapInfo in_info[NNS_TENSOR_SIZE_LIMIT];
    GstTensorMemory input[NNS_TENSOR_SIZE_LIMIT];
    guint i, num_tensors;

    num_tensors = self->tensor_config.info.num_tensors;
    g_assert (gst_buffer_n_memory (inbuf) == num_tensors);

    for (i = 0; i < num_tensors; i++) {
      in_mem[i] = gst_buffer_peek_memory (inbuf, i);
      status = gst_memory_map (in_mem[i], &in_info[i], GST_MAP_READ);
      g_assert (status);

      input[i].data = in_info[i].data;
      input[i].size = in_info[i].size;
      input[i].type = self->tensor_config.info.info[i].type;
    }

    res = self->decoder->decode (&self->plugin_data, &self->tensor_config,
        input, outbuf);

    for (i = 0; i < num_tensors; i++)
      gst_memory_unmap (in_mem[i], &in_info[i]);
  } else {
    GST_ERROR_OBJECT (self, "Decoder plugin not yet configured.");
    goto unknown_type;
  }

  return res;

unknown_format:
  GST_ERROR_OBJECT (self, "Hit unknown_format");
  GST_ELEMENT_ERROR (self, CORE, NOT_IMPLEMENTED, (NULL), ("unknown format"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_tensor:
  GST_ERROR_OBJECT (self, "Hit unknown_tensor");
  GST_ELEMENT_ERROR (self, CORE, NOT_IMPLEMENTED, (NULL),
      ("unknown format for tensor"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_type:
  GST_ELEMENT_ERROR (self, CORE, NOT_IMPLEMENTED, (NULL),
      ("not implemented decoder mode"));
  return GST_FLOW_NOT_SUPPORTED;
}

/**
 * @brief configure tensor-srcpad cap from "proposed" cap.
 *
 * @trans ("this" pointer)
 * @direction (why do we need this?)
 * @caps sinkpad cap
 * @filter this element's cap (don't know specifically.)
 */
static GstCaps *
gst_tensordec_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstTensorDec *self;
  GstCaps *result;

  self = GST_TENSOR_DECODER_CAST (trans);

  /* Not ready */
  if (self->decoder == NULL)
    return NULL;

  silent_debug ("Direction = %d\n", direction);
  silent_debug_caps (caps, "from");
  silent_debug_caps (filter, "filter");

  if (direction == GST_PAD_SINK) {
    /** caps = sinkpad (other/tensor) return = srcpad (media) */
    GstStructure *s = gst_caps_get_structure (caps, 0);
    result = gst_tensordec_media_caps_from_structure (self, s);
  } else if (direction == GST_PAD_SRC) {
    /** caps = srcpad (media) return = sinkpad (other/tensor) */
    /** @todo We may do more specific actions here */
    result = gst_caps_from_string (CAPS_STRING);
  } else {
    g_assert (0);
    return NULL;
  }

  if (filter && gst_caps_get_size (filter) > 0) {
    GstCaps *intersection;

    intersection =
        gst_caps_intersect_full (filter, result, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (result);
    result = intersection;
  }

  silent_debug_caps (result, "to");

  GST_DEBUG_OBJECT (self, "Direction[%d] transformed %" GST_PTR_FORMAT
      " into %" GST_PTR_FORMAT, direction, caps, result);
  return result;
}

/**
 * @brief fixate caps. required vmethod of BaseTransform
 */
static GstCaps *
gst_tensordec_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  GstTensorDec *self;
  GstCaps *supposed;
  GstCaps *result;

  self = GST_TENSOR_DECODER_CAST (trans);

  silent_debug_caps (caps, "from caps");
  silent_debug_caps (othercaps, "from othercaps");

  GST_DEBUG_OBJECT (self, "trying to fixate othercaps %" GST_PTR_FORMAT
      " based on caps %" GST_PTR_FORMAT, othercaps, caps);

  /** @todo The code below assumes that direction is GST_PAD_SINK */
  g_assert (direction == GST_PAD_SINK);

  if (gst_tensordec_configure (self, caps)) {
    supposed =
        gst_tensordec_media_caps_from_tensor (self, &self->tensor_config);
  } else {
    GstStructure *s = gst_caps_get_structure (caps, 0);
    supposed = gst_tensordec_media_caps_from_structure (self, s);
  }

  result = gst_caps_intersect (othercaps, supposed);
  gst_caps_unref (supposed);

  if (gst_caps_is_empty (result)) {
    gst_caps_unref (result);
    result = othercaps;
  } else {
    gst_caps_unref (othercaps);
  }

  GST_DEBUG_OBJECT (self, "now fixating %" GST_PTR_FORMAT, result);

  result = gst_caps_make_writable (result);
  result = gst_caps_fixate (result);

  if (direction == GST_PAD_SINK) {
    if (gst_caps_is_subset (caps, result)) {
      gst_caps_replace (&result, caps);
    }
  }
  return result;
}

/**
 * @brief set caps. required vmethod of BaseTransform
 */
static gboolean
gst_tensordec_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps)
{
  GstTensorDec *self;

  self = GST_TENSOR_DECODER_CAST (trans);

  silent_debug_caps (incaps, "from incaps");
  silent_debug_caps (outcaps, "from outcaps");

  if (gst_tensordec_configure (self, incaps)) {
    GstCaps *supposed = gst_tensordec_media_caps_from_tensor (self,
        &self->tensor_config);

    /** Check if outcaps ==equivalent== supposed */
    if (gst_caps_is_always_compatible (outcaps, supposed)) {
      self->negotiated = TRUE;
    } else {
      GST_ERROR_OBJECT (self,
          "This is not compatible with the supposed output pad cap");
    }

    gst_caps_unref (supposed);
  }

  return self->negotiated;
}

/**
 * @brief Tell the framework the required size of buffer based on the info of the other side pad. optional vmethod of BaseTransform
 *
 * This is called when non-ip mode is used.
 */
static gboolean
gst_tensordec_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize)
{
  GstTensorDec *self;

  if (direction == GST_PAD_SRC)
    return FALSE;
  /** @todo If direction = SRC, you may need different interpretation! */
  self = GST_TENSOR_DECODER_CAST (trans);

  g_assert (self->configured);
  g_assert (self->decoder);

  if (self->decoder->getTransformSize)
    *othersize = self->decoder->getTransformSize (&self->plugin_data,
        &self->tensor_config, caps, size, othercaps, direction);
  else
    *othersize = 0;

  return TRUE;
}
