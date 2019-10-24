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

#define REGEX_NNAPI_OPTION "(^(true|false)$|^(true|false)([:]?(cpu|gpu|npu)))"

/**
 * @brief Free memory
 */
#define g_free_const(x) g_free((void*)(long)(x))

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
  PROP_NNAPI
};

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
 * @return number of parsed model path
 */
static guint
gst_tensor_filter_parse_modelpaths_string (GstTensorFilterProperties * prop,
    const gchar * model_files)
{
  gchar **models;
  gchar **model_0;
  gchar **model_1;
  guint num_models = 0;
  guint num_model_0 = 0;
  guint num_model_1 = 0;

  g_return_val_if_fail (prop != NULL, 0);

  if (model_files) {
    models = g_strsplit_set (model_files, ",", -1);
    num_models = g_strv_length (models);

    if (num_models == 1) {
      prop->model_file = g_strdup (models[0]);
    } else if (num_models == 2) {
      model_0 = g_strsplit_set (models[0], "=", -1);
      model_1 = g_strsplit_set (models[1], "=", -1);

      num_model_0 = g_strv_length (model_0);
      num_model_1 = g_strv_length (model_1);

      if (num_model_0 == 1 && num_model_1 == 1) {
        prop->model_file_sub = g_strdup (model_0[0]);
        prop->model_file = g_strdup (model_1[0]);
      } else if (g_ascii_strncasecmp (model_0[0], "init", 4) == 0 ||
          g_ascii_strncasecmp (model_0[0], "Init", 4) == 0) {
        prop->model_file_sub = g_strdup (model_0[1]);

        if (num_model_1 == 2)
          prop->model_file = g_strdup (model_1[1]);
        else
          prop->model_file = g_strdup (model_1[0]);
      } else if (g_ascii_strncasecmp (model_0[0], "pred", 4) == 0 ||
          g_ascii_strncasecmp (model_0[0], "Pred", 4) == 0) {
        prop->model_file = g_strdup (model_0[1]);

        if (num_model_1 == 2)
          prop->model_file_sub = g_strdup (model_1[1]);
        else
          prop->model_file_sub = g_strdup (model_1[0]);
      } else if (g_ascii_strncasecmp (model_1[0], "init", 4) == 0 ||
          g_ascii_strncasecmp (model_1[0], "Init", 4) == 0) {
        prop->model_file_sub = g_strdup (model_1[1]);

        if (num_model_0 == 2)
          prop->model_file = g_strdup (model_0[1]);
        else
          prop->model_file = g_strdup (model_0[0]);
      } else if (g_ascii_strncasecmp (model_1[0], "pred", 4) == 0 ||
          g_ascii_strncasecmp (model_1[0], "Pred", 4) == 0) {
        prop->model_file = g_strdup (model_1[1]);

        if (num_model_0 == 2)
          prop->model_file_sub = g_strdup (model_0[1]);
        else
          prop->model_file_sub = g_strdup (model_0[0]);
      }
      g_strfreev (model_0);
      g_strfreev (model_1);
    }
    g_strfreev (models);
  }
  return num_models;
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
          "File path to the model file. Separated with \
          ',' in case of multiple model files(like caffe2)", "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
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
  g_object_class_install_property (gobject_class, PROP_NNAPI,
      g_param_spec_string ("nnapi", "NNAPI",
          "Enable Neural Newtorks API ?", "", G_PARAM_READWRITE));
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
  prop->model_file = NULL;
  prop->nnapi = NULL;
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
  g_free_const (prop->model_file);
  g_free_const (prop->custom_properties);

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
      guint model_num;

      if (prop->model_file) {
        gst_tensor_filter_common_close_fw (priv);
        g_free_const (prop->model_file);
        prop->model_file = NULL;
      }

      if (prop->model_file_sub) {
        gst_tensor_filter_common_close_fw (priv);
        g_free_const (prop->model_file_sub);
        prop->model_file_sub = NULL;
      }

      /* Once configures, it cannot be changed in runtime */
      g_assert (model_files);
      model_num = gst_tensor_filter_parse_modelpaths_string (prop, model_files);
      if (model_num == 1) {
        if (!g_file_test (prop->model_file, G_FILE_TEST_IS_REGULAR))
          g_critical ("Cannot find the model file: %s\n"
              "Ignore this if you have specified name and the model is preloaded.\n"
              "Note that this should be refactored.\n", prop->model_file);
          /** @todo This mechanism should be refactored. */
      } else if (model_num == 2) {
        if (!g_file_test (prop->model_file_sub, G_FILE_TEST_IS_REGULAR))
          g_critical ("Cannot find the init model file: %s\n",
              prop->model_file_sub);
        if (!g_file_test (prop->model_file, G_FILE_TEST_IS_REGULAR))
          g_critical ("Cannot find the pred model file: %s\n",
              prop->model_file);
      } else if (model_num > 2) {
        /** @todo if the new NN framework requires more than 2 model files, this area will be implemented */
        g_critical
            ("There is no NN framework that requires model files more than 2. Current Input model files are :%d\n",
            model_num);
      } else {
        g_critical ("Set model file path first\n");
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
    case PROP_NNAPI:
      prop->nnapi = g_value_dup_string (value);
      if (!g_regex_match_simple (REGEX_NNAPI_OPTION, prop->nnapi, 0, 0)) {
        g_critical
            ("nnapi: \'%s\' is not valid string: it should be in the form of BOOL:ACCELLERATOR or BOOL with a regex, "
            REGEX_NNAPI_OPTION "\n", prop->nnapi);
        break;
      }
      break;
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
    case PROP_NNAPI:
      g_value_set_string (value, prop->nnapi);
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
