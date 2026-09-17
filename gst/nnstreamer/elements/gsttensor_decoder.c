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
 * @file        gsttensor_decoder.c
 * @date        26 Mar 2018
 * @brief       GStreamer plugin to convert tensors (as a filter for other general neural network filters) to other media types
 * @see    	https://github.com/nnstreamer/nnstreamer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         gst_tensordec_transform_size () may be incorrect if direction is SINK.
 * @bug         If configured = TRUE, it holds TRUE until exit. What if configuration changes in run-time?
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
#include "gsttensor_decoder.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

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
  PROP_SUBPLUGINS,
  PROP_CONFIG
};

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE

/**
 * @brief Support multi-tensor along with single-tensor as the input
 */
#define CAPS_STRING GST_TENSOR_CAP_DEFAULT ";" GST_TENSORS_CAP_DEFAULT ";" GST_TENSORS_FLEX_CAP_DEFAULT

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
G_DEFINE_TYPE (GstTensorDecoder, gst_tensordec, GST_TYPE_BASE_TRANSFORM);

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
 * @brief set custom property description for tensor decoder sub-plugin
 */
void
nnstreamer_decoder_set_custom_property_desc (const char *name, const char *prop,
    ...)
{
  va_list varargs;

  va_start (varargs, prop);
  subplugin_set_custom_property_desc (NNS_SUBPLUGIN_DECODER, name, prop,
      varargs);
  va_end (varargs);
}

/**
 * @brief How often a property set rebuilds the private data for caps that changed under it
 */
#define TENSOR_DECODER_REBUILD_RETRY (4)

/**
 * @brief The private data of a decoder sub-plugin with the calls that use it.
 */
struct _GstTensorDecoderPlugin
{
  const GstTensorDecoderDef *decoder; /**< The sub-plugin the data belongs to */
  void *data; /**< The sub-plugin's private data */
  gint refcount; /**< The element's reference and one per call in progress */
};

/**
 * @brief Drop a reference to the sub-plugin's private data, releasing it with the last one.
 */
static void
gst_tensordec_plugin_unref (GstTensorDecoderPlugin * plugin)
{
  if (plugin == NULL || !g_atomic_int_dec_and_test (&plugin->refcount))
    return;

  if (plugin->decoder->exit)
    plugin->decoder->exit (&plugin->data);
  else
    g_free (plugin->data);
  g_free (plugin);
}

/**
 * @brief Take a reference to the current private data of the sub-plugin.
 * @return The private data to call the sub-plugin with, or NULL if there is none.
 */
static GstTensorDecoderPlugin *
gst_tensordec_plugin_ref (GstTensorDecoder * self)
{
  GstTensorDecoderPlugin *plugin;

  g_mutex_lock (&self->plugin_lock);
  plugin = self->plugin;
  if (plugin)
    g_atomic_int_inc (&plugin->refcount);
  g_mutex_unlock (&self->plugin_lock);

  return plugin;
}

/**
 * @brief Create the private data of the current sub-plugin with the current options and put it in place.
 * @param self "this" pointer
 * @param opnum The option (0-based) being set, whose failure is an error, or TensorDecMaxOpNum to start over from the numbered order of the options
 * @param config The negotiated tensor config to give the new private data with getOutCaps () before it is used, or NULL
 * @note The sub-plugin is called without the lock. The previous private data is released when the last call using it returns, and a rebuild overtaken by a newer one is dropped.
 */
static void
gst_tensordec_plugin_rebuild (GstTensorDecoder * self, guint opnum,
    const GstTensorsConfig * config)
{
  const GstTensorDecoderDef *decoder;
  GstTensorDecoderPlugin *plugin = NULL, *old;
  gchar *option[TensorDecMaxOpNum];
  guint order[TensorDecMaxOpNum];
  guint gen, i, n;

  g_mutex_lock (&self->plugin_lock);
  decoder = self->decoder;
  if (opnum < TensorDecMaxOpNum && (!decoder || !decoder->setOption)) {
    /* the sub-plugin takes no options */
    g_mutex_unlock (&self->plugin_lock);
    return;
  }
  gen = ++self->plugin_gen;
  for (i = 0; i < TensorDecMaxOpNum; i++) {
    if (opnum == TensorDecMaxOpNum)
      self->option_order[i] = i;
    order[i] = self->option_order[i];
    option[i] = g_strdup (self->option[i]);
  }
  g_mutex_unlock (&self->plugin_lock);

  if (decoder) {
    plugin = g_new0 (GstTensorDecoderPlugin, 1);
    plugin->decoder = decoder;
    plugin->refcount = 1;

    if (0 == decoder->init (&plugin->data)) {
      ml_loge ("Failed to initialize a decode subplugin, \"%s\".\n",
          decoder->modename);
      config = NULL;
    } else if (decoder->setOption) {
      for (i = 0; i < TensorDecMaxOpNum; i++) {
        n = order[i];
        if (option[n] == NULL || decoder->setOption (&plugin->data, n,
                option[n]))
          continue;

        if (n == opnum)
          GST_ERROR_OBJECT (self,
              "Configuring option for tensor-decoder failed (option %u = %s)",
              n + 1, option[n]);
        else
          GST_WARNING_OBJECT (self,
              "Failed to configure while setting the option %u.", n + 1);
      }
    }

    /* a sub-plugin may set up what decode () needs in getOutCaps () */
    if (config) {
      GstCaps *caps = decoder->getOutCaps (&plugin->data, config);

      if (caps)
        gst_caps_unref (caps);
    }
  }

  g_mutex_lock (&self->plugin_lock);
  if (gen == self->plugin_gen) {
    old = self->plugin;
    self->plugin = plugin;
  } else {
    old = plugin;
  }
  g_mutex_unlock (&self->plugin_lock);

  gst_tensordec_plugin_unref (old);
  for (i = 0; i < TensorDecMaxOpNum; i++)
    g_free (option[i]);
}

/**
 * @brief Rebuild the private data for a property set. If the stream is negotiated, the new data gets its config and the element renegotiates, since the property may change the output caps.
 * @note The incoming caps may change while the data is built. The rebuild is then repeated, so that the data which stays is the one the stream was described by, and not one built for caps the stream has left. A change that arrives after the rebuild took its generation is covered by the generation itself; what the repeat covers is the few instructions between reading the caps and taking it, which no test can hold a thread in.
 */
static void
gst_tensordec_plugin_rebuild_on_property (GstTensorDecoder * self, guint opnum)
{
  GstBaseTransform *trans = GST_BASE_TRANSFORM (self);
  GstPad *sinkpad = GST_BASE_TRANSFORM_SINK_PAD (trans);
  GstCaps *caps = gst_pad_get_current_caps (sinkpad);
  guint retry;

  for (retry = 0; retry < TENSOR_DECODER_REBUILD_RETRY; retry++) {
    GstTensorsConfig config;
    GstCaps *now;

    if (caps && gst_tensors_config_from_caps (&config, caps, TRUE)) {
      gst_tensordec_plugin_rebuild (self, opnum, &config);
      gst_tensors_config_free (&config);
      gst_base_transform_reconfigure_src (trans);
    } else {
      gst_tensordec_plugin_rebuild (self, opnum, NULL);
    }

    now = gst_pad_get_current_caps (sinkpad);
    if (now == caps || (now && caps && gst_caps_is_equal (now, caps))) {
      if (now)
        gst_caps_unref (now);
      break;
    }

    if (caps)
      gst_caps_unref (caps);
    caps = now;
  }

  if (retry == TENSOR_DECODER_REBUILD_RETRY)
    GST_WARNING_OBJECT (self,
        "The incoming caps kept changing while the decoder sub-plugin was given the property, so its data may still be built for the caps from before the last change.");

  if (caps)
    gst_caps_unref (caps);
}

/**
 * @brief Store an option and rebuild the private data, giving the options to the sub-plugin in the order they were set since the mode.
 */
static void
gst_tensordec_set_option (GstTensorDecoder * self, guint opnum,
    const GValue * value)
{
  guint i, j;

  g_mutex_lock (&self->plugin_lock);
  g_free (self->option[opnum]);
  self->option[opnum] = g_value_dup_string (value);
  for (i = 0, j = 0; i < TensorDecMaxOpNum; i++) {
    if (self->option_order[i] != opnum)
      self->option_order[j++] = self->option_order[i];
  }
  self->option_order[j] = opnum;
  g_mutex_unlock (&self->plugin_lock);

  gst_tensordec_plugin_rebuild_on_property (self, opnum);
}

/**
 * @brief Get media caps from tensor config
 * @param self "this" pointer
 * @param config tensor config info
 * @return caps for media type
 */
static GstCaps *
gst_tensordec_get_media_caps_from_config (GstTensorDecoder * self,
    const GstTensorsConfig * config)
{
  GstTensorDecoderPlugin *plugin;
  GstCaps *caps;

  g_return_val_if_fail (config != NULL, NULL);

  plugin = gst_tensordec_plugin_ref (self);
  if (plugin == NULL) {
    if (self->is_custom) {
      caps = gst_caps_from_string ("application/octet-stream");
      if (config->rate_n >= 0 && config->rate_d > 0)
        gst_caps_set_simple (caps, "framerate",
            GST_TYPE_FRACTION, config->rate_n, config->rate_d, NULL);
      return caps;
    }
    GST_ERROR_OBJECT (self, "Decoder plugin is not yet configured.");
    return NULL;
  }

  /* call sub-plugin vmethod */
  caps = plugin->decoder->getOutCaps (&plugin->data, config);
  gst_tensordec_plugin_unref (plugin);

  return caps;
}

/**
 * @brief Parse tensor caps and return media caps
 * @param self "this" pointer
 * @param caps tensor caps to be interpreted
 */
static GstCaps *
gst_tensordec_get_media_caps (GstTensorDecoder * self, const GstCaps * caps)
{
  GstTensorsConfig config;
  GstCaps *result = NULL;

  if (gst_tensors_config_from_caps (&config, caps, TRUE)) {
    result = gst_tensordec_get_media_caps_from_config (self, &config);
    gst_tensors_config_free (&config);

    if (result == NULL) {
      /* the sub-plugin cannot decode this config, so nothing may come out */
      GST_ERROR_OBJECT (self,
          "The decoder sub-plugin does not accept the coming tensor config.");
      return gst_caps_new_empty ();
    }
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
gst_tensordec_check_consistency (GstTensorDecoder * self,
    GstTensorsConfig * config)
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
gst_tensordec_class_init (GstTensorDecoderClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;
  gchar **subplugins = NULL;
  gchar *strbuf;
  static gchar *strprint = NULL;

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

  subplugins = get_all_subplugins (NNS_SUBPLUGIN_DECODER);
  strbuf = g_strjoinv (", ", subplugins);
  g_free (strprint);
  strprint = g_strdup_printf
      ("Decoder mode. Other options (option1 to optionN) depend on the specified model. For more detail on optionX for each mode, please refer to the documentation or nnstreamer-check utility. Available modes (decoder subplugins) are: {%s}.",
      strbuf);

  g_object_class_install_property (gobject_class, PROP_MODE,
      g_param_spec_string ("mode", "Mode", strprint, "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_free (strbuf);
  g_strfreev (subplugins);

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

  g_object_class_install_property (gobject_class, PROP_CONFIG,
      g_param_spec_string ("config-file", "Configuration-file",
          "Path to configuration file which contains plugins properties", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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
 * set pad callback functions
 * initialize instance structure
 */
static void
gst_tensordec_init (GstTensorDecoder * self)
{
  guint i;

  self->silent = DEFAULT_SILENT;
  self->configured = FALSE;
  self->negotiated = FALSE;
  self->decoder = NULL;
  self->plugin = NULL;
  self->plugin_gen = 0;
  g_mutex_init (&self->plugin_lock);
  self->is_custom = FALSE;
  self->custom.func = NULL;
  self->custom.data = NULL;
  self->config_path = NULL;
  for (i = 0; i < TensorDecMaxOpNum; i++) {
    self->option[i] = NULL;
    self->option_order[i] = i;
  }

  gst_tensors_config_init (&self->tensor_config);
}

/**
 * @brief A macro to process incoming per-mode option
 * @param[in] opnum The option number (1 to TensorDecMaxOpNum)
 */
#define PROP_MODE_OPTION(opnum) \
    case PROP_MODE_OPTION ## opnum: \
      gst_tensordec_set_option (self, (opnum) - 1, value); \
      break

/**
 * @brief Set property (GObject vmethod)
 */
static void
gst_tensordec_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorDecoder *self;

  self = GST_TENSOR_DECODER (object);

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      break;
    case PROP_MODE:
    {
      const GstTensorDecoderDef *decoder, *prev;
      const gchar *mode_string;

      mode_string = g_value_get_string (value);
      if (g_ascii_strcasecmp (mode_string, "custom-code") == 0) {
        self->is_custom = TRUE;
        break;
      }

      decoder = nnstreamer_decoder_find (mode_string);

      /* See if we are using "plugin" */
      if (!nnstreamer_decoder_validate (decoder)) {
        GST_ERROR_OBJECT (self,
            "The given mode for tensor_decoder, %s, is unrecognized.\n",
            mode_string);
        decoder = NULL;
      } else {
        silent_debug (self, "tensor_decoder plugin mode (%s)\n", mode_string);
      }

      g_mutex_lock (&self->plugin_lock);
      prev = self->decoder;
      self->decoder = decoder;
      g_mutex_unlock (&self->plugin_lock);

      if (decoder && decoder == prev) {
        /* Already configured??? */
        GST_WARNING_OBJECT (self,
            "nnstreamer tensor_decoder %s is already configured.\n",
            mode_string);
      }

      /* The new private data replaces the previous, which is released when no call uses it */
      gst_tensordec_plugin_rebuild_on_property (self, TensorDecMaxOpNum);
      break;
    }
    case PROP_CONFIG:
    {
      g_free (self->config_path);
      self->config_path = g_strdup (g_value_get_string (value));
      gst_tensor_parse_config_file (self->config_path, object);
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
      g_mutex_lock (&self->plugin_lock); \
      g_value_set_string (value, self->option[opnum - 1]); \
      g_mutex_unlock (&self->plugin_lock); \
      break

/**
 * @brief Get property (GObject vmethod)
 */
static void
gst_tensordec_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorDecoder *self;

  self = GST_TENSOR_DECODER (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_MODE:
      g_mutex_lock (&self->plugin_lock);
      if (self->is_custom)
        g_value_set_string (value, "custom-code");
      else if (self->decoder)
        g_value_set_string (value, self->decoder->modename);
      else
        g_value_set_string (value, "");
      g_mutex_unlock (&self->plugin_lock);
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
      gchar **str_array = get_all_subplugins (NNS_SUBPLUGIN_DECODER);

      if (str_array) {
        g_value_take_string (value, g_strjoinv (",", str_array));
        g_strfreev (str_array);
      } else {
        g_value_set_string (value, "");
      }
      break;
    }
    case PROP_CONFIG:
      g_value_set_string (value, self->config_path ? self->config_path : "");
      break;
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
  GstTensorDecoder *self;
  guint i;

  self = GST_TENSOR_DECODER (object);

  gst_tensordec_plugin_unref (self->plugin);
  self->plugin = NULL;
  g_mutex_clear (&self->plugin_lock);

  gst_tensors_config_free (&self->tensor_config);
  g_free (self->config_path);
  for (i = 0; i < TensorDecMaxOpNum; ++i) {
    g_free (self->option[i]);
  }
  self->custom.func = NULL;
  self->custom.data = NULL;

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Configure tensor metadata from sink caps
 */
static gboolean
gst_tensordec_configure (GstTensorDecoder * self, const GstCaps * in_caps,
    const GstCaps * out_caps)
{
  GstTensorsConfig config;

  if (self->decoder == NULL && !self->is_custom) {
    GST_ERROR_OBJECT (self, "Decoder plugin is not yet configured.");
    return FALSE;
  }

  if (!gst_tensors_config_from_caps (&config, in_caps, TRUE)) {
    GST_ERROR_OBJECT (self, "Cannot configure tensor from in-caps.");
    return FALSE;
  }

  /**
   * If previous input configuration is set and is not compatible with incoming caps,
   * get possible media caps from sub-plugin and change input configuration.
   */
  if (self->configured && !gst_tensordec_check_consistency (self, &config)) {
    GstCaps *supposed;
    gboolean compatible;

    supposed = gst_tensordec_get_media_caps_from_config (self, &config);
    if (supposed == NULL) {
      GST_ERROR_OBJECT (self,
          "The decoder sub-plugin does not accept the coming tensor config.");
      gst_tensors_config_free (&config);
      return FALSE;
    }

    compatible = gst_caps_is_always_compatible (out_caps, supposed);
    gst_caps_unref (supposed);

    /** Check if outcaps is compatible with new caps */
    if (!compatible) {
      GST_ERROR_OBJECT (self, "The coming tensor config is not valid.");
      gst_tensors_config_free (&config);
      return FALSE;
    }

    gst_tensordec_plugin_rebuild (self, TensorDecMaxOpNum, NULL);
  }

  gst_tensors_config_free (&self->tensor_config);
  self->tensor_config = config;
  self->configured = TRUE;
  return TRUE;
}

/**
 * @brief Release the mapped memories of the incoming buffer.
 * @param mem the memories acquired from the incoming buffer
 * @param info the map info of each memory
 * @param num the number of the memories to release
 */
static void
gst_tensordec_release_input (GstMemory * mem[], GstMapInfo info[], guint num)
{
  guint i;

  for (i = 0; i < num; i++) {
    gst_memory_unmap (mem[i], &info[i]);
    gst_memory_unref (mem[i]);
  }
}

/**
 * @brief Validate the memory of the incoming buffer with the negotiated config.
 * @param self this pointer of tensor_decoder
 * @param index the index of the tensor in the incoming buffer
 * @param data the mapped memory of the tensor
 * @param size the mapped size of @a data
 * @return TRUE if the memory holds the tensor the decoder sub-plugin will read
 */
static gboolean
gst_tensordec_check_input_size (GstTensorDecoder * self, guint index,
    gpointer data, gsize size)
{
  gsize expected;

  if (gst_tensors_config_is_flexible (&self->tensor_config)) {
    GstTensorMetaInfo meta;
    gsize hsize;

    gst_tensor_meta_info_init (&meta);

    if (size < gst_tensor_meta_info_get_header_size (&meta) ||
        !gst_tensor_meta_info_parse_header (&meta, data)) {
      GST_ERROR_OBJECT (self,
          "Failed to parse the meta info of the %u'th tensor in the incoming buffer.",
          index);
      return FALSE;
    }

    /* a meta version this build cannot size gives no offset to the data */
    hsize = gst_tensor_meta_info_get_header_size (&meta);
    if (hsize == 0) {
      GST_ERROR_OBJECT (self,
          "The meta info of the %u'th tensor in the incoming buffer declares a version of which the header layout is unknown.",
          index);
      return FALSE;
    }

    expected = hsize + gst_tensor_meta_info_get_data_size (&meta);

    if (size < expected) {
      GST_ERROR_OBJECT (self,
          "The %u'th tensor of the incoming buffer is %zd bytes, which is smaller than %zd bytes described by its own meta info.",
          index, size, expected);
      return FALSE;
    }
  } else {
    expected = gst_tensors_info_get_size (&self->tensor_config.info, index);

    if (size != expected) {
      GST_ERROR_OBJECT (self,
          "The %u'th tensor of the incoming buffer is %zd bytes, which is expected to be %zd bytes by the negotiated caps. Check whether the incoming stream is consistent with the caps of the sink pad.",
          index, size, expected);
      return FALSE;
    }
  }

  return TRUE;
}

/**
 * @brief non-ip transform. required vmethod for BaseTransform class.
 */
static GstFlowReturn
gst_tensordec_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstTensorDecoder *self;
  GstFlowReturn res;

  self = GST_TENSOR_DECODER_CAST (trans);

  if (G_UNLIKELY (!self->negotiated))
    goto unknown_tensor;
  if (G_UNLIKELY (!self->configured))
    goto unknown_format;

  if (self->decoder || self->is_custom) {
    GstMemory *in_mem[NNS_TENSOR_SIZE_LIMIT];
    GstMapInfo in_info[NNS_TENSOR_SIZE_LIMIT];
    GstTensorMemory input[NNS_TENSOR_SIZE_LIMIT];
    guint i, num_tensors, num_mems;

    num_mems = gst_tensor_buffer_get_count (inbuf);
    if (gst_tensors_config_is_flexible (&self->tensor_config)) {
      self->tensor_config.info.num_tensors = num_mems;
    }
    num_tensors = self->tensor_config.info.num_tensors;
    if (num_mems != num_tensors) {
      GST_ERROR_OBJECT (self,
          "The incoming buffer has %u memory chunks, which is expected to be %u, the number of tensors of the negotiated caps.",
          num_mems, num_tensors);
      return GST_FLOW_ERROR;
    }

    for (i = 0; i < num_tensors; i++) {
      in_mem[i] = gst_tensor_buffer_get_nth_memory (inbuf, i);
      if (!gst_memory_map (in_mem[i], &in_info[i], GST_MAP_READ)) {
        ml_logf ("Failed to map in_mem[%u].\n", i);

        gst_tensordec_release_input (in_mem, in_info, i);
        gst_memory_unref (in_mem[i]);
        return GST_FLOW_ERROR;
      }

      if (!gst_tensordec_check_input_size (self, i, in_info[i].data,
              in_info[i].size)) {
        gst_tensordec_release_input (in_mem, in_info, i + 1);
        return GST_FLOW_ERROR;
      }

      input[i].data = in_info[i].data;
      input[i].size = in_info[i].size;
    }
    if (!self->is_custom) {
      GstTensorDecoderPlugin *plugin = gst_tensordec_plugin_ref (self);

      if (plugin) {
        res = plugin->decoder->decode (&plugin->data, &self->tensor_config,
            input, outbuf);
        gst_tensordec_plugin_unref (plugin);
      } else {
        GST_ERROR_OBJECT (self, "Decoder plugin is not configured.");
        res = GST_FLOW_ERROR;
      }
    } else if (self->custom.func != NULL) {
      res = self->custom.func (input, &self->tensor_config, self->custom.data,
          outbuf);
    } else {
      GST_ERROR_OBJECT (self, "Custom decoder callback is not registered.");
      res = GST_FLOW_ERROR;
    }

    gst_tensordec_release_input (in_mem, in_info, num_tensors);
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
  GstTensorDecoder *self;
  GstCaps *result;

  self = GST_TENSOR_DECODER_CAST (trans);

  /* Not ready */
  if (self->decoder == NULL && !self->is_custom)
    return NULL;

  if (self->is_custom) {
    const decoder_custom_cb_s *ptr = NULL;

    g_mutex_lock (&self->plugin_lock);
    if (self->option[0] == NULL) {
      g_mutex_unlock (&self->plugin_lock);
      nns_logw ("Tensor decoder custom option is not given.");
      return NULL;
    }
    self->custom.func = NULL;
    ptr = get_subplugin (NNS_CUSTOM_DECODER, self->option[0]);
    g_mutex_unlock (&self->plugin_lock);
    if (!ptr) {
      nns_logw ("Failed to find custom subplugin of the tensor_decoder");
      return NULL;
    }
    self->custom.func = ptr->func;
    self->custom.data = ptr->data;
  }

  silent_debug (self, "Direction = %d\n", direction);
  silent_debug_caps (self, caps, "from");
  silent_debug_caps (self, filter, "filter");

  if (direction == GST_PAD_SINK) {
    /** caps = sinkpad (other/tensor) return = srcpad (media) */
    result = gst_tensordec_get_media_caps (self, caps);
  } else if (direction == GST_PAD_SRC) {
    /** caps = srcpad (media) return = sinkpad (other/tensor) */
    /** @todo We may do more specific actions here */
    result = gst_caps_from_string (CAPS_STRING);
  } else {
    g_assert (0);               /* Internal logic error! */
    return NULL;
  }

  if (filter && gst_caps_get_size (filter) > 0) {
    GstCaps *intersection;

    intersection =
        gst_caps_intersect_full (filter, result, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (result);
    result = intersection;
  }

  silent_debug_caps (self, result, "to");

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
  GstTensorDecoder *self;
  GstCaps *supposed;
  GstCaps *result;

  self = GST_TENSOR_DECODER_CAST (trans);

  silent_debug_caps (self, caps, "from caps");
  silent_debug_caps (self, othercaps, "from othercaps");

  GST_DEBUG_OBJECT (self, "trying to fixate othercaps %" GST_PTR_FORMAT
      " based on caps %" GST_PTR_FORMAT, othercaps, caps);

  /**
   * In gst_tensordec_transform_caps, we have refused to specify caps
   * if the direction is GST_PAD_SRC. Thus, gstreamer shouldn't fixate
   * it with GST_PAD_SRC. If this happens, it's either internal logic
   * error or GST bug.
   */
  if (direction != GST_PAD_SINK) {
    ml_logf_stacktrace ("Invalid direction for tensor-decoder fixate caps.\n");
    return NULL;
  }

  if (gst_tensordec_configure (self, caps, othercaps)) {
    supposed =
        gst_tensordec_get_media_caps_from_config (self, &self->tensor_config);
  } else {
    supposed = gst_tensordec_get_media_caps (self, caps);
  }

  /**
   * An empty supposed is the refusal gst_tensordec_get_media_caps () reports,
   * which must not reach the "keep othercaps" fallback below: that is there for
   * an intersection which came out empty, not for a sub-plugin saying no.
   * A refused config is normally turned away by transform_caps () before the
   * fixation is reached, so this stands guard over whatever does not go there.
   */
  if (supposed == NULL || gst_caps_is_empty (supposed)) {
    GST_ERROR_OBJECT (self,
        "The decoder sub-plugin does not accept the coming tensor config.");
    if (supposed)
      gst_caps_unref (supposed);
    gst_caps_unref (othercaps);
    return gst_caps_new_empty ();
  }

  result = gst_caps_intersect (othercaps, supposed);
  gst_caps_unref (supposed);

  if (gst_caps_is_empty (result)) {
    gst_caps_unref (result);
    result = othercaps;
  } else {
    gst_caps_unref (othercaps);
  }

  if (gst_caps_is_any (result)) {
    /* neither the config nor the peer says what to output, and ANY has no fixed form */
    GST_ERROR_OBJECT (self,
        "Cannot tell the output caps of the coming tensor config.");
    gst_caps_unref (result);
    return gst_caps_new_empty ();
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
  GstTensorDecoder *self = GST_TENSOR_DECODER_CAST (trans);

  silent_debug_caps (self, incaps, "from incaps");
  silent_debug_caps (self, outcaps, "from outcaps");

  /**
   * Answer for this negotiation only and leave the flag alone on a refusal: a
   * refused caps event is not stored on the pad, and when the earlier caps
   * come back, fixate_caps stores their config again while GstBaseTransform
   * skips set_caps for caps equal to the current ones.
   */
  if (gst_tensordec_configure (self, incaps, outcaps)) {
    GstCaps *supposed = gst_tensordec_get_media_caps_from_config (self,
        &self->tensor_config);
    gboolean compatible;

    if (supposed == NULL) {
      GST_ERROR_OBJECT (self,
          "The decoder sub-plugin does not accept the negotiated tensor config.");
      return FALSE;
    }

    /** Check if outcaps ==equivalent== supposed */
    compatible = gst_caps_is_always_compatible (outcaps, supposed);
    if (compatible) {
      self->negotiated = TRUE;
    } else {
      GST_ERROR_OBJECT (self,
          "This is not compatible with the supposed output pad cap");
    }

    gst_caps_unref (supposed);
    return compatible;
  }

  return FALSE;
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
  GstTensorDecoder *self;
  GstTensorDecoderPlugin *plugin;

  if (direction == GST_PAD_SRC)
    return FALSE;
  /** @todo If direction = SRC, you may need different interpretation! */
  self = GST_TENSOR_DECODER_CAST (trans);

  g_assert (self->configured);

  plugin = self->is_custom ? NULL : gst_tensordec_plugin_ref (self);
  if (plugin && plugin->decoder->getTransformSize)
    *othersize = plugin->decoder->getTransformSize (&plugin->data,
        &self->tensor_config, caps, size, othercaps, direction);
  else
    *othersize = 0;
  gst_tensordec_plugin_unref (plugin);

  return TRUE;
}

/**
 * @brief Registers a callback for tensor_decoder custom condition
 * @return 0 if success. -ERRNO if error.
 */
int
nnstreamer_decoder_custom_register (const gchar * name,
    tensor_decoder_custom func, void *data)
{
  decoder_custom_cb_s *ptr;

  g_return_val_if_fail (name && strlen (name), -EINVAL);
  g_return_val_if_fail (func, -EINVAL);

  if (!(ptr = g_try_new0 (decoder_custom_cb_s, 1)))
    return -ENOMEM;

  ptr->func = func;
  ptr->data = data;

  if (register_subplugin (NNS_CUSTOM_DECODER, name, ptr))
    return 0;

  g_free (ptr);
  return -EINVAL;
}

/**
 * @brief Unregisters a callback for tensor_decoder custom condition
 * @return 0 if success. -ERRNO if error.
 */
int
nnstreamer_decoder_custom_unregister (const gchar * name)
{
  decoder_custom_cb_s *ptr;

  ptr = (decoder_custom_cb_s *) get_subplugin (NNS_CUSTOM_DECODER, name);
  if (!unregister_subplugin (NNS_CUSTOM_DECODER, name)) {
    ml_loge ("Failed to unregister custom callback %s.", name);
    return -EINVAL;
  }
  g_free (ptr);

  return 0;
}
