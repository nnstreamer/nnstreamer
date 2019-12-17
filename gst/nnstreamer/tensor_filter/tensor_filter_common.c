/**
 * Copyright (C) 2019 Parichay Kapoor <pk.kapoor@samsung.com>
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
 * @file	tensor_filter_common.c
 * @date	28 Aug 2019
 * @brief	Common functions for various tensor_filters
 * @see	  http://github.com/nnsuite/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug	  No known bugs except for NYI items
 *
 */

#include <string.h>

#include <tensor_common.h>

#include "tensor_filter_common.h"

/**
 * Basic elements to form accelerator regex forming
 */
#define REGEX_ACCL_ELEM_START "("
#define REGEX_ACCL_ELEM_PREFIX "(?<!!)"
#define REGEX_ACCL_ELEM_SUFFIX ""
#define REGEX_ACCL_ELEM_DELIMITER "|"
#define REGEX_ACCL_ELEM_END ")?"

#define REGEX_ACCL_START "(^(true)[:]?([(]?("
#define REGEX_ACCL_PREFIX ""
#define REGEX_ACCL_SUFFIX ""
#define REGEX_ACCL_DELIMITER "|"
#define REGEX_ACCL_END ")*[)]?))"

const gchar *regex_accl_utils[] = {
  REGEX_ACCL_START,
  REGEX_ACCL_PREFIX,
  REGEX_ACCL_SUFFIX,
  REGEX_ACCL_DELIMITER,
  REGEX_ACCL_END,
  NULL
};

const gchar *regex_accl_elem_utils[] = {
  REGEX_ACCL_ELEM_START,
  REGEX_ACCL_ELEM_PREFIX,
  REGEX_ACCL_ELEM_SUFFIX,
  REGEX_ACCL_ELEM_DELIMITER,
  REGEX_ACCL_ELEM_END,
  NULL
};

/**
 * @brief Free memory
 */
#define g_free_const(x) g_free((void*)(long)(x))
#define g_strfreev_const(x) g_strfreev((void*)(long)(x))

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
  PROP_SUBPLUGINS,
  PROP_ACCELERATOR,
  PROP_IS_UPDATABLE,
};

/**
 * @brief to get and register hardware accelerator backend enum
 */
static GType
accl_hw_get_type (void)
    G_GNUC_CONST;

/**
 * @brief copy the string from src to destination
 * @param[in] dest destination string
 * @param[in] src source string
 * @return updated destination string
 */
     static gchar *strcpy2 (gchar * dest, const gchar * src)
{
  memcpy (dest, src, strlen (src));
  return dest + strlen (src);
}

/**
 * @brief create regex for the given string list and regex basic elements
 * @param[in] enum_list list of strings to form regex for
 * @param[in] regex_utils list of basic elements to form regex
 * @return the formed regex (to be freed by the caller), NULL on error
 */
static gchar *
create_regex (const gchar ** enum_list, const gchar ** regex_utils)
{
  gchar regex[4096];
  gchar *regex_ptr = regex;
  const gchar **strings = enum_list;
  const gchar *iterator = *strings;
  const gchar *escape_separator = "\\.";
  const gchar *escape_chars = ".";
  gchar **regex_split;
  gchar *regex_escaped;

  if (iterator == NULL)
    return NULL;

  /** create the regex string */
  regex_ptr = strcpy2 (regex_ptr, regex_utils[0]);
  regex_ptr = strcpy2 (regex_ptr, regex_utils[1]);
  regex_ptr = strcpy2 (regex_ptr, iterator);
  regex_ptr = strcpy2 (regex_ptr, regex_utils[2]);
  for (iterator = strings[1]; iterator != NULL; iterator = *++strings) {
    regex_ptr = strcpy2 (regex_ptr, regex_utils[3]);
    regex_ptr = strcpy2 (regex_ptr, regex_utils[1]);
    regex_ptr = strcpy2 (regex_ptr, iterator);
    regex_ptr = strcpy2 (regex_ptr, regex_utils[2]);
  }
  regex_ptr = strcpy2 (regex_ptr, regex_utils[4]);
  *regex_ptr = '\0';

  /** escape the special characters */
  regex_split = g_strsplit_set (regex, escape_chars, -1);
  regex_escaped = g_strjoinv (escape_separator, regex_split);
  g_strfreev (regex_split);

  return regex_escaped;
}

/**
 * @brief Validate filter sub-plugin's data.
 */
static gboolean
nnstreamer_filter_validate (const GstTensorFilterFramework * tfsp)
{
  if (!tfsp || !tfsp->name) {
    /* invalid fw name */
    return FALSE;
  }

  if (!tfsp->invoke_NN) {
    /* no invoke function */
    return FALSE;
  }

  if (!(tfsp->getInputDimension && tfsp->getOutputDimension) &&
      !tfsp->setInputDimension) {
    /* no method to get tensor info */
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Filter's sub-plugin should call this function to register itself.
 * @param[in] tfsp Tensor-Filter Sub-Plugin to be registered.
 * @return TRUE if registered. FALSE is failed or duplicated.
 */
int
nnstreamer_filter_probe (GstTensorFilterFramework * tfsp)
{
  g_return_val_if_fail (nnstreamer_filter_validate (tfsp), FALSE);
  return register_subplugin (NNS_SUBPLUGIN_FILTER, tfsp->name, tfsp);
}

/**
 * @brief Filter's sub-plugin may call this to unregister itself.
 * @param[in] name The name of filter sub-plugin.
 */
void
nnstreamer_filter_exit (const char *name)
{
  unregister_subplugin (NNS_SUBPLUGIN_FILTER, name);
}

/**
 * @brief Find filter sub-plugin with the name.
 * @param[in] name The name of filter sub-plugin.
 * @return NULL if not found or the sub-plugin object has an error.
 */
const GstTensorFilterFramework *
nnstreamer_filter_find (const char *name)
{
  return get_subplugin (NNS_SUBPLUGIN_FILTER, name);
}

/**
 * @brief Parse the string of model
 * @param[out] prop Struct containing the properties of the object
 * @param[in] model_files the prediction model paths
 */
static void
gst_tensor_filter_parse_modelpaths_string (GstTensorFilterProperties * prop,
    const gchar * model_files)
{
  if (prop == NULL)
    return;

  g_strfreev_const (prop->model_files);

  if (model_files) {
    prop->model_files = (const gchar **) g_strsplit_set (model_files, ",", -1);
    prop->num_models = g_strv_length ((gchar **) prop->model_files);
  } else {
    prop->model_files = NULL;
    prop->num_models = 0;
  }
}


/**
 * @brief Printout the comparison results of two tensors.
 * @param[in] info1 The tensors to be shown on the left hand side
 * @param[in] info2 The tensors to be shown on the right hand side
 */
void
gst_tensor_filter_compare_tensors (GstTensorsInfo * info1,
    GstTensorsInfo * info2)
{
  gchar *result = NULL;
  gchar *line, *tmp, *left, *right;
  guint i;

  for (i = 0; i < NNS_TENSOR_SIZE_LIMIT; i++) {
    if (info1->num_tensors <= i && info2->num_tensors <= i)
      break;

    if (info1->num_tensors > i) {
      tmp = gst_tensor_get_dimension_string (info1->info[i].dimension);
      left = g_strdup_printf ("%s [%s]",
          gst_tensor_get_type_string (info1->info[i].type), tmp);
      g_free (tmp);
    } else {
      left = g_strdup ("None");
    }

    if (info2->num_tensors > i) {
      tmp = gst_tensor_get_dimension_string (info2->info[i].dimension);
      right = g_strdup_printf ("%s [%s]",
          gst_tensor_get_type_string (info2->info[i].type), tmp);
      g_free (tmp);
    } else {
      right = g_strdup ("None");
    }

    line =
        g_strdup_printf ("%2d : %s | %s %s\n", i, left, right,
        g_str_equal (left, right) ? "" : "FAILED");

    g_free (left);
    g_free (right);

    if (result) {
      tmp = g_strdup_printf ("%s%s", result, line);
      g_free (result);
      g_free (line);

      result = tmp;
    } else {
      result = line;
    }
  }

  if (result) {
    /* print warning message */
    g_warning ("Tensor info :\n%s", result);
    g_free (result);
  }
}

/**
 * @brief Installs all the properties for tensor_filter
 * @param[in] gobject_class Glib object class whose properties will be set
 */
void
gst_tensor_filter_install_properties (GObjectClass * gobject_class)
{
  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          FALSE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_FRAMEWORK,
      g_param_spec_string ("framework", "Framework",
          "Neural network framework", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_MODEL,
      g_param_spec_string ("model", "Model filepath",
          "File path to the model file. Separated with ',' in case of multiple model files(like caffe2)",
          "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_INPUT,
      g_param_spec_string ("input", "Input dimension",
          "Input tensor dimension from inner array, up to 4 dimensions ?", "",
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
          "Output tensor dimension from inner array, up to 4 dimensions ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_OUTPUTTYPE,
      g_param_spec_string ("outputtype", "Output tensor element type",
          "Type of each element of the output tensor ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_CUSTOM,
      g_param_spec_string ("custom", "Custom properties for subplugins",
          "Custom properties for subplugins ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SUBPLUGINS,
      g_param_spec_string ("sub-plugins", "Sub-plugins",
          "Registrable sub-plugins list", "",
          G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_ACCELERATOR,
      g_param_spec_string ("accelerator", "ACCELERATOR",
          "Set accelerator for the subplugin with format "
          "(true/false):(comma separated ACCELERATOR(s)). "
          "true/false determines if accelerator is to be used. "
          "list of accelerators determines the backend (ignored with false). "
          "Example, if GPU, NPU can be used but not CPU - true:(GPU,NPU,!CPU). "
          "Note that only a few subplugins support this property.",
          "", G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_IS_UPDATABLE,
      g_param_spec_boolean ("is-updatable", "Updatable model",
          "Indicate whether a given model to this tensor filter is "
          "updatable in runtime. (e.g., with on-device training)",
          FALSE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}

/**
 * @brief Initialize the properties for tensor-filter.
 */
void
gst_tensor_filter_common_init_property (GstTensorFilterPrivate * priv)
{
  GstTensorFilterProperties *prop;

  prop = &priv->prop;

  /* init NNFW properties */
  prop->fwname = NULL;
  prop->fw_opened = FALSE;
  prop->input_configured = FALSE;
  prop->output_configured = FALSE;
  prop->model_files = NULL;
  prop->num_models = 0;
  prop->accl_str = NULL;
  prop->custom_properties = NULL;
  gst_tensors_info_init (&prop->input_meta);
  gst_tensors_info_init (&prop->output_meta);

  /* init internal properties */
  priv->fw = NULL;
  priv->privateData = NULL;
  priv->silent = TRUE;
  priv->configured = FALSE;
  gst_tensors_config_init (&priv->in_config);
  gst_tensors_config_init (&priv->out_config);
}

/**
 * @brief Free the properties for tensor-filter.
 */
void
gst_tensor_filter_common_free_property (GstTensorFilterPrivate * priv)
{
  GstTensorFilterProperties *prop;

  prop = &priv->prop;

  g_free_const (prop->fwname);
  g_free_const (prop->accl_str);
  g_free_const (prop->custom_properties);
  g_strfreev_const (prop->model_files);

  gst_tensors_info_free (&prop->input_meta);
  gst_tensors_info_free (&prop->output_meta);

  gst_tensors_info_free (&priv->in_config.info);
  gst_tensors_info_free (&priv->out_config.info);
}

/**
 * @brief Set the properties for tensor_filter
 * @param[in] priv Struct containing the properties of the object
 * @param[in] prop_id Id for the property
 * @param[in] value Container to return the asked property
 * @param[in] pspec Metadata to specify the parameter
 * @return TRUE if prop_id is value, else FALSE
 */
gboolean
gst_tensor_filter_common_set_property (GstTensorFilterPrivate * priv,
    guint prop_id, const GValue * value, GParamSpec * pspec)
{
  GstTensorFilterProperties *prop;

  prop = &priv->prop;

  switch (prop_id) {
    case PROP_SILENT:
      priv->silent = g_value_get_boolean (value);
      break;
    case PROP_FRAMEWORK:
    {
      const gchar *fw_name = g_value_get_string (value);
      const GstTensorFilterFramework *fw;

      if (priv->fw != NULL) {
        /* close old framework */
        gst_tensor_filter_common_close_fw (priv);
      }

      g_debug ("Framework = %s\n", fw_name);

      fw = nnstreamer_filter_find (fw_name);

      /* See if mandatory methods are filled in */
      if (nnstreamer_filter_validate (fw)) {
        priv->fw = fw;
        prop->fwname = g_strdup (fw_name);
      } else {
        g_warning ("Cannot identify the given neural network framework, %s\n",
            fw_name);
      }
      break;
    }
    case PROP_MODEL:
    {
      const gchar *model_files = g_value_get_string (value);
      int idx;

      g_assert (model_files);
      gst_tensor_filter_parse_modelpaths_string (prop, model_files);

      for (idx = 0; idx < prop->num_models; idx++) {
        if (!g_file_test (prop->model_files[idx], G_FILE_TEST_IS_REGULAR)) {
          g_critical ("Cannot find the model file [%d]: %s\n",
              idx, prop->model_files[idx]);
        }
      }

      /* reload model if FW has been already opened */
      if (priv->prop.fw_opened && priv->is_updatable) {
        if (priv->fw && priv->fw->reloadModel) {
          if (priv->fw->reloadModel (&priv->prop, &priv->privateData) != 0) {
            g_critical ("Fail to reload model\n");
          }
        }
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
          g_warning
              ("Invalid input-dim, given param does not match with old value.");
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
          g_warning
              ("Invalid output-dim, given param does not match with old value.");
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
          g_warning
              ("Invalid input-type, given param does not match with old value.");
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
          g_warning
              ("Invalid output-type, given param does not match with old value.");
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
          g_warning
              ("Invalid input-name, given param does not match with old value.");
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
          g_warning
              ("Invalid output-name, given param does not match with old value.");
        }

        prop->output_meta.num_tensors = num_names;
      }
      break;
    case PROP_CUSTOM:
      /* In case updated custom properties in runtime! */
      g_free_const (prop->custom_properties);
      prop->custom_properties = g_value_dup_string (value);
      g_debug ("Custom Option = %s\n", prop->custom_properties);
      break;
    case PROP_ACCELERATOR:
    {
      /**
       * TODO: allow updating the subplugin accelerator after it has been init
       * by reopening
       */
      if (priv->prop.fw_opened == TRUE) {
        break;
      }

      prop->accl_str = g_value_dup_string (value);
      break;
    }
    case PROP_IS_UPDATABLE:
    {
      if (priv->fw->reloadModel != NULL)
        priv->is_updatable = g_value_get_boolean (value);
      break;
    }
    default:
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Get the properties for tensor_filter
 * @param[in] priv Struct containing the properties of the object
 * @param[in] prop_id Id for the property
 * @param[in] value Container to return the asked property
 * @param[in] pspec Metadata to specify the parameter
 * @return TRUE if prop_id is value, else FALSE
 */
gboolean
gst_tensor_filter_common_get_property (GstTensorFilterPrivate * priv,
    guint prop_id, GValue * value, GParamSpec * pspec)
{
  GstTensorFilterProperties *prop;

  prop = &priv->prop;

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, priv->silent);
      break;
    case PROP_FRAMEWORK:
      g_value_set_string (value, prop->fwname);
      break;
    case PROP_MODEL:
    {
      GString *gstr_models = g_string_new (NULL);
      gchar *models;
      int idx;

      /* return a comma-separated string */
      for (idx = 0; idx < prop->num_models; ++idx) {
        if (idx != 0) {
          g_string_append (gstr_models, ",");
        }

        g_string_append (gstr_models, prop->model_files[idx]);
      }

      models = g_string_free (gstr_models, FALSE);
      g_value_take_string (value, models);
      break;
    }
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
    case PROP_SUBPLUGINS:
    {
      GString *subplugins;
      subplugin_info_s sinfo;
      guint i, total;

      subplugins = g_string_new (NULL);

      /* add custom */
      /** @todo Let's not hardcode default subplugins */
      g_string_append (subplugins, "custom,custom-easy");

      total = nnsconf_get_subplugin_info (NNSCONF_PATH_FILTERS, &sinfo);

      if (total > 0) {
        const gchar *prefix_str;
        gsize prefix, extension, len;

        prefix_str = nnsconf_get_subplugin_name_prefix (NNSCONF_PATH_FILTERS);
        prefix = strlen (prefix_str);
        extension = strlen (NNSTREAMER_SO_FILE_EXTENSION);

        for (i = 0; i < total; ++i) {
          g_string_append (subplugins, ",");

          /* remove file extension */
          len = strlen (sinfo.names[i]) - prefix - extension;
          g_string_append_len (subplugins, sinfo.names[i] + prefix, len);
        }
      }

      g_value_take_string (value, g_string_free (subplugins, FALSE));
      break;
    }
    case PROP_ACCELERATOR:
      if (prop->accl_str != NULL) {
        g_value_set_string (value, prop->accl_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_IS_UPDATABLE:
      g_value_set_boolean (value, priv->is_updatable);
      break;
    default:
      /* unknown property */
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Open NN framework.
 */
void
gst_tensor_filter_common_open_fw (GstTensorFilterPrivate * priv)
{
  if (!priv->prop.fw_opened && priv->fw) {
    if (priv->fw->open) {
      /* at least one model should be configured before opening fw */
      if (G_UNLIKELY (!priv->fw->run_without_model) &&
          G_UNLIKELY (!(priv->prop.model_files &&
                  priv->prop.num_models > 0 && priv->prop.model_files[0])))
        return;
      /* 0 if successfully loaded. 1 if skipped (already loaded). */
      if (priv->fw->open (&priv->prop, &priv->privateData) >= 0)
        priv->prop.fw_opened = TRUE;
    } else {
      priv->prop.fw_opened = TRUE;
    }
  }
}

/**
 * @brief Close NN framework.
 */
void
gst_tensor_filter_common_close_fw (GstTensorFilterPrivate * priv)
{
  if (priv->prop.fw_opened) {
    if (priv->fw && priv->fw->close) {
      priv->fw->close (&priv->prop, &priv->privateData);
    }
    priv->prop.fw_opened = FALSE;
    g_free_const (priv->prop.fwname);
    priv->prop.fwname = NULL;
    priv->fw = NULL;
    priv->privateData = NULL;
  }
}

/**
 * @brief return accl_hw type from string
 * @param key The key string value
 * @return Corresponding index. Returns ACCL_NONE if not found.
 */
accl_hw
get_accl_hw_type (const gchar * key)
{
  GEnumClass *enum_class;
  GEnumValue *enum_value;

  enum_class = g_type_class_ref (accl_hw_get_type ());
  enum_value = g_enum_get_value_by_name (enum_class, key);
  g_type_class_unref (enum_class);

  if (enum_value == NULL)
    return ACCL_NONE;
  return enum_value->value;
}

/**
 * @brief return string based on accl_hw type
 * @param key The key enum value
 * @return Corresponding string. Returns ACCL_NONE_STR if not found.
 * @note Do not free the returned char *
 */
const gchar *
get_accl_hw_str (const accl_hw key)
{
  GEnumClass *enum_class;
  GEnumValue *enum_value;

  enum_class = g_type_class_ref (accl_hw_get_type ());
  enum_value = g_enum_get_value (enum_class, key);
  g_type_class_unref (enum_class);

  if (enum_value == NULL)
    return ACCL_NONE_STR;
  return enum_value->value_name;
}

/**
 * @brief parse user given string to extract accelerator based on given regex
 * @param[in] accelerators user given input
 * @param[in] supported_accelerators list ofi supported accelerators
 * @return Corresponding string. Returns ACCL_NONE_STR if not found.
 */
accl_hw
parse_accl_hw (const gchar * accelerators,
    const gchar ** supported_accelerators)
{
  GRegex *nnapi_elem;
  GMatchInfo *match_info;
  gboolean use_accl;
  accl_hw accl;
  gchar *regex_accl = NULL;
  gchar *regex_accl_elem = NULL;

  if (accelerators == NULL)
    return ACCL_DEFAULT;

  /* If set by user, get the precise accelerator */
  regex_accl = create_regex (supported_accelerators, regex_accl_utils);
  use_accl = (gboolean) g_regex_match_simple (regex_accl, accelerators,
      G_REGEX_CASELESS, G_REGEX_MATCH_NOTEMPTY);
  g_free (regex_accl);
  if (use_accl == TRUE) {
    /** Default to auto mode */
    accl = ACCL_AUTO;
    regex_accl_elem =
        create_regex (supported_accelerators, regex_accl_elem_utils);
    nnapi_elem =
        g_regex_new (regex_accl_elem, G_REGEX_CASELESS, G_REGEX_MATCH_NOTEMPTY,
        NULL);
    g_free (regex_accl_elem);

    /** Now match each provided element and get specific accelerator */
    if (g_regex_match (nnapi_elem, accelerators, G_REGEX_MATCH_NOTEMPTY,
            &match_info)) {

      while (g_match_info_matches (match_info)) {
        gchar *word = g_match_info_fetch (match_info, 0);
        accl = get_accl_hw_type (word);
        g_free (word);
        break;
      }
    }
    g_match_info_free (match_info);
    g_regex_unref (nnapi_elem);
  } else {
    return ACCL_NONE;
  }

  return accl;
}

/**
 * @brief to get and register hardware accelerator backend enum
 */
static GType
accl_hw_get_type (void)
{
  static volatile gsize g_accl_hw_type_id__volatile = 0;

  if (g_once_init_enter (&g_accl_hw_type_id__volatile)) {
    static const GEnumValue values[] = {
      {ACCL_NONE, ACCL_NONE_STR, ACCL_NONE_STR},
      {ACCL_DEFAULT, ACCL_DEFAULT_STR, ACCL_DEFAULT_STR},
      {ACCL_AUTO, ACCL_AUTO_STR, ACCL_AUTO_STR},
      {ACCL_CPU, ACCL_CPU_STR, ACCL_CPU_STR},
      {ACCL_CPU_NEON, ACCL_CPU_NEON_STR, ACCL_CPU_NEON_STR},
      {ACCL_GPU, ACCL_GPU_STR, ACCL_GPU_STR},
      {ACCL_NPU, ACCL_NPU_STR, ACCL_NPU_STR},
      {ACCL_NPU_MOVIDIUS, ACCL_NPU_MOVIDIUS_STR, ACCL_NPU_MOVIDIUS_STR},
      {ACCL_NPU_EDGE_TPU, ACCL_NPU_EDGE_TPU_STR, ACCL_NPU_EDGE_TPU_STR},
      {ACCL_NPU_VIVANTE, ACCL_NPU_VIVANTE_STR, ACCL_NPU_VIVANTE_STR},
      {ACCL_NPU_SRCN, ACCL_NPU_SRCN_STR, ACCL_NPU_SRCN_STR},
      {ACCL_NPU_SR, ACCL_NPU_SR_STR, ACCL_NPU_SR_STR},
      {0, NULL, NULL}
    };

    GType g_accl_hw_type_id =
        g_enum_register_static (g_intern_static_string ("accl_hw"), values);
    g_once_init_leave (&g_accl_hw_type_id__volatile, g_accl_hw_type_id);
  }

  return g_accl_hw_type_id__volatile;
}
