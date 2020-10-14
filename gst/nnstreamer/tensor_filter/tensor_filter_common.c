/**
 * Copyright (C) 2019 Parichay Kapoor <pk.kapoor@samsung.com>
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
 * @file	tensor_filter_common.c
 * @date	28 Aug 2019
 * @brief	Common functions for various tensor_filters
 * @see	  http://github.com/nnstreamer/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug	  No known bugs except for NYI items
 *
 */

#include <string.h>

#include <hw_accel.h>
#include <nnstreamer_log.h>
#include <tensor_common.h>

#include "tensor_filter_common.h"

#define silent_debug_info(i,msg) do { \
  if (DBG) { \
    guint info_idx; \
    gchar *dim_str; \
    g_debug (msg " total %d", (i)->num_tensors); \
    for (info_idx = 0; info_idx < (i)->num_tensors; info_idx++) { \
      dim_str = gst_tensor_get_dimension_string ((i)->info[info_idx].dimension); \
      g_debug ("[%d] type=%d dim=%s", info_idx, (i)->info[info_idx].type, dim_str); \
      g_free (dim_str); \
    } \
  } \
} while (0)

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

static const gchar *regex_accl_utils[] = {
  REGEX_ACCL_START,
  REGEX_ACCL_PREFIX,
  REGEX_ACCL_SUFFIX,
  REGEX_ACCL_DELIMITER,
  REGEX_ACCL_END,
  NULL
};

static const gchar *regex_accl_elem_utils[] = {
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
 * @brief parse user given string to extract list of accelerators based on given regex
 */
static GList *parse_accl_hw_all (const gchar * accelerators,
    const gchar ** supported_accelerators);
static gint _gtfc_setprop_IS_UPDATABLE (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value);
static gint _gtfc_setprop_ACCELERATOR (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value);

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
  PROP_INPUTLAYOUT,
  PROP_INPUTRANKS,
  PROP_OUTPUT,
  PROP_OUTPUTTYPE,
  PROP_OUTPUTNAME,
  PROP_OUTPUTLAYOUT,
  PROP_OUTPUTRANKS,
  PROP_CUSTOM,
  PROP_SUBPLUGINS,
  PROP_ACCELERATOR,
  PROP_IS_UPDATABLE,
  PROP_LATENCY,
  PROP_THROUGHPUT,
};

/**
 * @brief Initialize the tensors layout.
 */
static void
gst_tensors_layout_init (tensors_layout layout)
{
  int i;

  for (i = 0; i < NNS_TENSOR_SIZE_LIMIT; i++) {
    layout[i] = _NNS_LAYOUT_ANY;
  }
}

/**
 * @brief Initialize the tensors ranks
 */
static void
gst_tensors_rank_init (unsigned int ranks[])
{
  int i;
  for (i = 0; i < NNS_TENSOR_SIZE_LIMIT; ++i) {
    ranks[i] = 0;
  }
}

/**
 * @brief Get tensor layout from string input.
 * @return Corresponding tensor_layout.
 */
static tensor_layout
gst_tensor_parse_layout_string (const gchar * layoutstr)
{
  gchar *layout_string;
  tensor_layout layout = _NNS_LAYOUT_ANY;

  if (layoutstr == NULL)
    return layout;

  /* remove spaces */
  layout_string = g_strdup (layoutstr);
  g_strstrip (layout_string);

  /* empty string */
  if (0 == strlen (layout_string))
    goto done;

  if (g_ascii_strcasecmp (layoutstr, "NCHW") == 0) {
    layout = _NNS_LAYOUT_NCHW;
  } else if (g_ascii_strcasecmp (layoutstr, "NHWC") == 0) {
    layout = _NNS_LAYOUT_NHWC;
  } else if (g_ascii_strcasecmp (layoutstr, "ANY") == 0) {
    layout = _NNS_LAYOUT_ANY;
  } else {
    GST_WARNING ("Invalid layout, defaulting to none layout.");
    layout = _NNS_LAYOUT_NONE;
  }

done:
  g_free (layout_string);
  return layout;
}

/**
 * @brief Parse the string of tensor layouts
 * @param layout layout of the tensors
 * @param layout_string string of layout
 * @return number of parsed layouts
 */
static guint
gst_tensors_parse_layouts_string (tensors_layout layout,
    const gchar * layout_string)
{
  guint num_layouts = 0;

  g_return_val_if_fail (layout != NULL, 0);

  if (layout_string) {
    guint i;
    gchar **str_layouts;

    str_layouts = g_strsplit_set (layout_string, ",.", -1);
    num_layouts = g_strv_length (str_layouts);

    if (num_layouts > NNS_TENSOR_SIZE_LIMIT) {
      GST_WARNING ("Invalid param, layouts (%d) max (%d)\n",
          num_layouts, NNS_TENSOR_SIZE_LIMIT);

      num_layouts = NNS_TENSOR_SIZE_LIMIT;
    }

    for (i = 0; i < num_layouts; i++) {
      layout[i] = gst_tensor_parse_layout_string (str_layouts[i]);
    }

    g_strfreev (str_layouts);
  }

  return num_layouts;
}

/**
 * @brief Get the rank string of the tensors
 * @param prop GstTensorFilterProperties object
 * @param isInput TRUE if target is input tensors
 * @return rank string of the tensors
 */
static gchar *
gst_tensors_get_rank_string (const GstTensorFilterProperties * prop,
    gboolean isInput)
{
  gchar *rank_str = NULL;
  guint i;
  guint num_tenosrs;

  g_return_val_if_fail (prop != NULL, NULL);

  num_tenosrs =
      isInput ? prop->input_meta.num_tensors : prop->output_meta.num_tensors;

  if (num_tenosrs > 0) {
    GString *rank = g_string_new (NULL);

    for (i = 0; i < num_tenosrs; ++i) {
      if (isInput) {
        if (prop->input_ranks[i] != 0)
          g_string_append_printf (rank, "%u", prop->input_ranks[i]);
        else
          g_string_append_printf (rank, "%d",
              gst_tensor_info_get_rank (&prop->input_meta.info[i]));
      } else {
        if (prop->output_ranks[i] != 0)
          g_string_append_printf (rank, "%u", prop->output_ranks[i]);
        else
          g_string_append_printf (rank, "%d",
              gst_tensor_info_get_rank (&prop->output_meta.info[i]));
      }

      if (i < num_tenosrs - 1)
        g_string_append_printf (rank, ",");
    }
    rank_str = g_string_free (rank, FALSE);
  }

  return rank_str;
}

/**
 * @brief Get layout string of tensor layout.
 * @param layout layout of the tensor
 * @return string of layout in tensor
 */
static const gchar *
gst_tensor_get_layout_string (tensor_layout layout)
{
  switch (layout) {
    case _NNS_LAYOUT_NCHW:
      return "NCHW";
    case _NNS_LAYOUT_NHWC:
      return "NHWC";
    case _NNS_LAYOUT_NONE:
      return "NONE";
    case _NNS_LAYOUT_ANY:
      return "ANY";
    default:
      return NULL;
  }
}

/**
 * @brief Get the string of layout of tensors
 * @param layout layout of the tensors
 * @return string of layouts in tensors
 * @note The returned value should be freed with g_free()
 */
static gchar *
gst_tensors_get_layout_string (const GstTensorsInfo * info,
    tensors_layout layout)
{
  gchar *layout_str = NULL;

  g_return_val_if_fail (info != NULL, NULL);

  if (info->num_tensors > 0) {
    guint i;
    GString *layouts = g_string_new (NULL);

    for (i = 0; i < info->num_tensors; i++) {
      g_string_append (layouts, gst_tensor_get_layout_string (layout[i]));

      if (i < info->num_tensors - 1) {
        g_string_append (layouts, ",");
      }
    }

    layout_str = g_string_free (layouts, FALSE);
  }

  return layout_str;
}

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
 * @brief Verify validity of path for given model file if verify_model_path is set
 * @param[in] priv Struct containing the common tensor-filter properties of the object
 * @return TRUE if there is no error
 */
static inline gboolean
verify_model_path (const GstTensorFilterPrivate * priv)
{
  const GstTensorFilterProperties *prop;
  gboolean ret = TRUE;
  int verify_model_path = 0, i;

  if (priv == NULL)
    return FALSE;

  prop = &(priv->prop);

  if (g_strcmp0 (prop->fwname, "custom-easy") == 0)
    return TRUE;

  if (GST_TF_FW_V0 (priv->fw)) {
    verify_model_path = priv->fw->verify_model_path;
  } else if (GST_TF_FW_V1 (priv->fw)) {
    verify_model_path = priv->info.verify_model_path;
  }

  if ((prop->model_files != NULL) && (verify_model_path == TRUE)) {
    for (i = 0; i < prop->num_models; i++) {
      if (!g_file_test (prop->model_files[i], G_FILE_TEST_IS_REGULAR)) {
        ml_loge ("Cannot find the model file [%d]: %s\n",
            i, prop->model_files[i]);
        ret = FALSE;
      }
    }
  }

  return ret;
}

/**
 * @brief Initialize the GstTensorFilterProperties object
 */
static void
gst_tensor_filter_properties_init (GstTensorFilterProperties * prop)
{
  prop->fwname = NULL;
  prop->fw_opened = FALSE;
  prop->model_files = NULL;
  prop->num_models = 0;

  prop->input_configured = FALSE;
  gst_tensors_info_init (&prop->input_meta);
  gst_tensors_layout_init (prop->input_layout);
  gst_tensors_rank_init (prop->input_ranks);

  prop->output_configured = FALSE;
  gst_tensors_info_init (&prop->output_meta);
  gst_tensors_layout_init (prop->output_layout);
  gst_tensors_rank_init (prop->output_ranks);

  prop->custom_properties = NULL;
  prop->accl_str = NULL;
}

/**
 * @brief Initialize the GstTensorFilterFrameworkInfo object
 */
static void
gst_tensor_filter_framework_info_init (GstTensorFilterFrameworkInfo * info)
{
  info->name = NULL;
  info->allow_in_place = 0;
  info->allocate_in_invoke = 0;
  info->run_without_model = 0;
  info->verify_model_path = 0;
  info->hw_list = NULL;
  info->accl_auto = -1;
  info->accl_default = -1;
  info->statistics = NULL;
}

/**
 * @brief Initialize the GstTensorFilterFrameworkInfo object
 */
static void
gst_tensor_filter_statistics_init (GstTensorFilterStatistics * stat)
{
  stat->total_invoke_num = 0;
  stat->total_invoke_latency = 0;
  stat->old_total_invoke_num = 0;
  stat->old_total_invoke_latency = 0;
  stat->latest_invoke_time = 0;
  stat->recent_latencies = g_queue_new ();
}

/**
 * @brief Validate filter sub-plugin's data.
 */
static gboolean
nnstreamer_filter_validate (const GstTensorFilterFramework * tfsp)
{
  if (GST_TF_FW_V0 (tfsp)) {
    if (!tfsp->name) {
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
  } else if (GST_TF_FW_V1 (tfsp)) {
    GstTensorFilterFrameworkInfo info;
    GstTensorFilterProperties prop;

    if (!tfsp->invoke || !tfsp->getFrameworkInfo || !tfsp->getModelInfo ||
        !tfsp->eventHandler) {
      /** Mandatory callbacks are not defined */
      return FALSE;
    }

    gst_tensor_filter_properties_init (&prop);
    if (tfsp->getFrameworkInfo (tfsp, &prop, NULL, &info) != 0) {
      /* unable to get framework info */
      return FALSE;
    }

    if (!info.name) {
      /* invalid fw name */
      return FALSE;
    }
  } else {
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
  GstTensorFilterFrameworkInfo info;
  GstTensorFilterProperties prop;
  const char *name = NULL;

  g_return_val_if_fail (nnstreamer_filter_validate (tfsp), FALSE);

  if (GST_TF_FW_V0 (tfsp)) {
    name = tfsp->name;
  } else if (GST_TF_FW_V1 (tfsp)) {
    gst_tensor_filter_properties_init (&prop);
    if (0 != tfsp->getFrameworkInfo (tfsp, &prop, NULL, &info)) {
      ml_loge ("getFrameworkInfo() failed.\n");
      return FALSE;
    }
    name = info.name;
  }

  return register_subplugin (NNS_SUBPLUGIN_FILTER, name, tfsp);
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
 * @brief check if the allocate_in_invoke is valid for the framework
 * @param[in] priv Struct containing the properties of the object
 * @return TRUE if valid, FALSE on error
 */
gboolean
gst_tensor_filter_allocate_in_invoke (GstTensorFilterPrivate * priv)
{
  int allocate_in_invoke = 0;

  if (GST_TF_FW_V0 (priv->fw)) {
    allocate_in_invoke = priv->fw->allocate_in_invoke;
    if (allocate_in_invoke == TRUE && priv->fw->allocateInInvoke) {
      if (priv->fw->allocateInInvoke (&priv->privateData) == 0) {
        allocate_in_invoke = TRUE;
      } else {
        allocate_in_invoke = FALSE;
      }
    }
  } else if (GST_TF_FW_V1 (priv->fw)) {
    allocate_in_invoke = priv->info.allocate_in_invoke;
  }

  return allocate_in_invoke;
}

/**
 * @brief Free the data allocated for tensor filter output
 * @param[in] priv Struct containing the properties of the object
 * @param[in] data Data to be freed
 */
void
gst_tensor_filter_destroy_notify_util (GstTensorFilterPrivate * priv,
    void *data)
{
  GstTensorFilterFrameworkEventData event_data;

  if (GST_TF_FW_V0 (priv->fw) && priv->fw->destroyNotify) {
    priv->fw->destroyNotify (&priv->privateData, data);
  } else if (GST_TF_FW_V1 (priv->fw)) {
    event_data.data = data;
    if (priv->fw->eventHandler (priv->fw, &priv->prop, priv->privateData,
            DESTROY_NOTIFY, &event_data) == -ENOENT) {
      g_free (data);
    }
  } else {
    g_free (data);
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
    nns_logw ("Tensor info :\n%s", result);
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
  g_object_class_install_property (gobject_class, PROP_INPUTLAYOUT,
      g_param_spec_string ("inputlayout", "Input Data Layout",
          "Set channel first (NCHW) or channel last layout (NHWC) or None for input data. "
          "Layout of the data can be any or NHWC or NCHW or none for now. ",
          "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_INPUTRANKS,
      g_param_spec_string ("inputranks", "Rank of Input Tensor",
          "The Rank of the Input Tensor, which is separated with ',' in case of multiple Tensors",
          "", G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
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
  g_object_class_install_property (gobject_class, PROP_OUTPUTLAYOUT,
      g_param_spec_string ("outputlayout", "Output Data Layout",
          "Set channel first (NCHW) or channel last layout (NHWC) or None for output data. "
          "Layout of the data can be any or NHWC or NCHW or none for now. ",
          "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_OUTPUTRANKS,
      g_param_spec_string ("outputranks", "Rank of Out Tensor",
          "The Rank of the Out Tensor, which is separated with ',' in case of multiple Tensors",
          "", G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));
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
          "Example, if GPU, NPU can be used but not CPU - true:npu,gpu,!cpu. "
          "The full list of accelerators can be found in nnstreamer_plugin_api_filter.h. "
          "Note that only a few subplugins support this property.",
          "", G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_IS_UPDATABLE,
      g_param_spec_boolean ("is-updatable", "Updatable model",
          "Indicate whether a given model to this tensor filter is "
          "updatable in runtime. (e.g., with on-device training)",
          FALSE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_LATENCY,
      g_param_spec_int ("latency", "The average latency",
          "Turn on performance profiling for the average latency "
          "over the recent 10 inferences in microseconds. "
          "Currently, this accepts either 0 (OFF) or 1 (ON).",
          0 /** min */ , 1 /** max */ , 0 /** default: off */ ,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_THROUGHPUT,
      g_param_spec_int ("throughput", "The average throughput (FPS)",
          "Turn on performance profiling for the average throughput "
          "in the number of outputs per seconds (i.e., FPS), multiplied by 1000 "
          "to represent a floating point using an integer. "
          "Currently, this accepts either 0 (OFF) or 1 (ON).",
          0 /** min */ , 1 /** max */ , 0 /** default: off */ ,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}

/**
 * @brief Initialize the properties for tensor-filter.
 */
void
gst_tensor_filter_common_init_property (GstTensorFilterPrivate * priv)
{
  /* init NNFW properties */
  gst_tensor_filter_properties_init (&priv->prop);
  gst_tensor_filter_framework_info_init (&priv->info);
  gst_tensor_filter_statistics_init (&priv->stat);

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
  if (GST_TF_FW_V0 (priv->fw)) {
    g_free_const (prop->accl_str);
  } else if (GST_TF_FW_V1 (priv->fw)) {
    g_free (prop->hw_list);
  }
  g_free_const (prop->custom_properties);
  g_strfreev_const (prop->model_files);

  gst_tensors_info_free (&prop->input_meta);
  gst_tensors_info_free (&prop->output_meta);

  gst_tensors_info_free (&priv->in_config.info);
  gst_tensors_info_free (&priv->out_config.info);

  if (priv->stat.recent_latencies != NULL) {
    GQueue *queue = priv->stat.recent_latencies;
    gint64 *latency;
    while ((latency = g_queue_pop_tail (queue)) != NULL)
      g_free (latency);
    g_queue_free (queue);
  }
}

/**
 * @brief Parse the accelerator hardwares to be used for this framework
 * @param[in] priv Struct containing the properties of the object
 * @param[in] prop Struct containing the properties of the framework
 * @param[in] accelerators user given input for hardare accelerators
 * @note The order of preference set by the user is maintained
 */
static void
gst_tensor_filter_parse_accelerator (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const char *accelerators)
{
  gint status, idx;
  GstTensorFilterFrameworkInfo *info;
  const gchar **accl_support;
  GList *match_accl, *iter;

  info = &priv->info;
  gst_tensor_filter_framework_info_init (info);

  prop->num_hw = 0;
  g_free (prop->hw_list);
  prop->hw_list = NULL;

  /** Get h/w accelerators supported by framework */
  if (info->name == NULL) {
    status = priv->fw->getFrameworkInfo (priv->fw, prop, priv->privateData,
        info);
    if (status != 0 || info->hw_list == NULL) {
      ml_logw ("Unable to fetch accelerators supported by the framework.");
      return;
    }
  }

  if (info->num_hw == 0)
    return;

  /**
   * Convert the list to string format
   * Extra 2 entries for basic accelerators : auto and default
   */
  accl_support = g_malloc (sizeof (gchar *) * (info->num_hw + 2 + 1));

  for (idx = 0; idx < info->num_hw; idx++) {
    accl_support[idx] = get_accl_hw_str (info->hw_list[idx]);
  }
  accl_support[info->num_hw] = ACCL_AUTO_STR;
  accl_support[info->num_hw + 1] = ACCL_DEFAULT_STR;
  accl_support[info->num_hw + 2] = NULL;

  /** Parse the user given h/w accelerators intersected with supported h/w */
  match_accl = parse_accl_hw_all (accelerators, accl_support);
  g_free (accl_support);

  /** Convert the GList to regular array */
  prop->num_hw = g_list_length (match_accl);
  prop->hw_list = g_malloc (sizeof (accl_hw) * prop->num_hw);
  for (iter = match_accl, idx = 0; iter != NULL; iter = iter->next, idx++) {
    prop->hw_list[idx] = GPOINTER_TO_INT (iter->data);
    if (prop->hw_list[idx] == ACCL_AUTO) {
      prop->hw_list[idx] = info->accl_auto;
      if (info->accl_auto < ACCL_NONE)
        prop->hw_list[idx] = info->hw_list[0];
    } else if (prop->hw_list[idx] == ACCL_DEFAULT) {
      prop->hw_list[idx] = info->accl_default;
      if (info->accl_default < ACCL_NONE)
        prop->hw_list[idx] = info->hw_list[0];
    }
  }
  g_list_free (match_accl);
}

/**
 * @brief Get available framework from given file when user selects auto option
 * @param[in] model_files the prediction model paths
 * @param[in] num_models the number of model files
 * @return Detected framework name (NULL if it fails to detect automatically)
 */
gchar *
gst_tensor_filter_framework_auto_detection (const gchar ** model_files,
    unsigned int num_models)
{
  const GstTensorFilterFramework *fw = NULL;
  gchar *file_extension, *file_extension_2;
  gchar *fw_priority_str = NULL, *detected_fw = NULL;
  gchar **fw_priority_str_arr;
  guint i;

  /* supposed it is ONE if given model is directory */
  if (g_file_test (model_files[0], G_FILE_TEST_IS_DIR)) {
    detected_fw = g_strdup ("nnfw");
    goto done;
  }

  file_extension = strrchr (model_files[0], '.');

  /* Detect framework based on file extension */
  if (num_models == 1) {
    if (g_ascii_strcasecmp (file_extension, ".tflite") == 0)
      fw_priority_str =
          nnsconf_get_custom_value_string ("filter",
          "framework_priority_tflite");
    else if (g_ascii_strcasecmp (file_extension, ".pb") == 0)
      detected_fw = g_strdup ("tensorflow");
    else if (g_ascii_strcasecmp (file_extension, ".pt") == 0)
      detected_fw = g_strdup ("pytorch");
    else if (g_ascii_strcasecmp (file_extension, ".py") == 0)
      detected_fw = g_strdup ("python");
    else if (g_ascii_strcasecmp (file_extension, ".graph") == 0)
      detected_fw = g_strdup ("movidius-ncsdk2");
    else if (g_ascii_strcasecmp (file_extension, ".circle") == 0)
      detected_fw = g_strdup ("nnfw");
    else if (g_ascii_strcasecmp (file_extension,
            NNSTREAMER_SO_FILE_EXTENSION) == 0)
      detected_fw = g_strdup ("custom");
    else if (g_ascii_strcasecmp (file_extension, ".bin") == 0
        || g_ascii_strcasecmp (file_extension, ".xml") == 0)
      detected_fw = g_strdup ("openvino");
  } else if (num_models == 2) {
    file_extension_2 = strrchr (model_files[1], '.');
    if (g_ascii_strcasecmp (file_extension, ".pb") == 0
        && g_ascii_strcasecmp (file_extension_2, ".pb") == 0
        && g_ascii_strcasecmp (model_files[0], model_files[1]))
      detected_fw = g_strdup ("caffe2");
    else if ((g_ascii_strcasecmp (file_extension, ".so") == 0
            && g_ascii_strcasecmp (file_extension_2, ".nb") == 0)
        || (g_ascii_strcasecmp (file_extension_2, ".so") == 0
            && g_ascii_strcasecmp (file_extension, ".nb") == 0))
      detected_fw = g_strdup ("vivante");
  } else {
    nns_logw ("Invalid the number of models");
    return detected_fw;
  }

  /* .tflite have multiple framework. */
  if (fw_priority_str != NULL) {
    fw_priority_str_arr = g_strsplit (fw_priority_str, ",", -1);

    for (i = 0; i < g_strv_length (fw_priority_str_arr); i++) {
      fw = nnstreamer_filter_find (fw_priority_str_arr[i]);

      if (fw) {
        detected_fw = g_strdup (fw_priority_str_arr[i]);
        break;
      } else {
        nns_logd ("Cannot identify the neural network framework. %s\n",
            fw_priority_str_arr[i]);
      }
    }

    g_free (fw_priority_str);
    g_strfreev (fw_priority_str_arr);
  }

done:
  if (detected_fw) {
    nns_logi ("Automatically detected framework is %s.", detected_fw);
    nns_logd
        ("If you want to change priority of framework for auto detection, Please edit meson_option.txt. You can find the file at root path of nnstreamer. \n");
  } else {
    nns_logw ("Cannot get any neural network framework for given model");
  }

  return detected_fw;
}

/**
 * @brief automatically selecting framework for tensor filter
 * @param[in] priv Struct containing the properties of the object
 * @param[in] fw_name Framework name
 */
void
gst_tensor_filter_get_available_framework (GstTensorFilterPrivate * priv,
    const char *fw_name)
{
  GstTensorFilterProperties *prop;
  const GstTensorFilterFramework *fw;
  gchar *detected_fw = NULL;

  if (fw_name == NULL)
    return;

  prop = &priv->prop;

  if (g_ascii_strcasecmp (fw_name, "auto") == 0) {
    if (prop->model_files == NULL) {
      /* If model file is not loaded, get framework after loading the model */
      prop->fwname = g_strdup (fw_name);
      return;
    }

    detected_fw =
        gst_tensor_filter_framework_auto_detection (prop->model_files,
        prop->num_models);
  } else {
    detected_fw = g_strdup (fw_name);
  }

  /* init fw-name (case if fw-name is auto) */
  if (prop->fwname) {
    g_free_const (prop->fwname);
    prop->fwname = NULL;
  }

  nns_logd ("Framework = %s\n", fw_name);

  fw = nnstreamer_filter_find (detected_fw);

  if (fw) {
    /** Get framework info for v1 */
    gst_tensor_filter_framework_info_init (&priv->info);
    if (GST_TF_FW_V1 (fw) && fw->getFrameworkInfo (fw, prop, NULL, &priv->info)
        < 0) {
      nns_logw ("Cannot get the given framework info, %s\n", fw_name);
      g_free (detected_fw);
      return;
    }
    priv->fw = fw;
    prop->fwname = detected_fw;

    /** update the accelerator if already set based on v0 or v1 */
    if (GST_TF_FW_V1 (priv->fw)) {
      if (prop->accl_str) {
        const gchar *accl_str = prop->accl_str;
        gst_tensor_filter_parse_accelerator (priv, &priv->prop, accl_str);
        g_free_const (accl_str);
      } else {
        prop->hw_list = NULL;
      }
    }
  } else {
    nns_logw ("Cannot identify the given neural network framework, %s\n",
        fw_name);
    g_free (detected_fw);
  }
}

/** @brief Handle "PROP_FRAMEWORK" for set-property */
static gint
_gtfc_setprop_FRAMEWORK (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  gint status;
  const gchar *fw_name = g_value_get_string (value);
  GValue val = G_VALUE_INIT;

  if (priv->fw != NULL) {
    if (g_strcmp0 (priv->prop.fwname, fw_name) != 0) {
      /* close old framework, if different */
      gst_tensor_filter_common_close_fw (priv);
      priv->fw = NULL;
    } else {
      ml_logd ("Framework = %s\n", fw_name);
      return 0;
    }
  }

  gst_tensor_filter_get_available_framework (priv, fw_name);

  /** set PROP_IS_UPDATABLE in case it was set before framework */
  g_value_init (&val, G_TYPE_BOOLEAN);
  g_value_set_boolean (&val, priv->is_updatable);
  status = _gtfc_setprop_IS_UPDATABLE (priv, prop, &val);
  g_value_unset (&val);
  if (status != 0) {
    g_warning ("Set propery is-updatable failed with error: %d", status);
    return status;
  }

  /** set PROP_ACCELERATOR in case it was set before framework */
  if (prop->accl_str) {
    g_value_init (&val, G_TYPE_STRING);
    g_value_set_string (&val, prop->accl_str);
    status = _gtfc_setprop_ACCELERATOR (priv, prop, &val);
    g_value_unset (&val);
    if (status != 0) {
      g_warning ("Set propery accelerator failed with error: %d", status);
      return status;
    }
  }

  return 0;
}

/** @brief Handle "PROP_MODEL" for set-property */
static gint
_gtfc_setprop_MODEL (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  gint status = 0;
  const gchar *model_files = g_value_get_string (value);
  GstTensorFilterProperties _prop;

  if (!model_files) {
    ml_loge ("Invalid model provided to the tensor-filter.");
    return 0;
  }
  _prop.model_files = NULL;

  if (prop->fw_opened) {
    /** Store a copy of the original prop in case the reload fails */
    memcpy (&_prop, prop, sizeof (GstTensorFilterProperties));
    _prop.model_files =
        (const gchar **) g_strdupv ((gchar **) prop->model_files);
  }

  gst_tensor_filter_parse_modelpaths_string (prop, model_files);

  if (prop->fwname != NULL && g_ascii_strcasecmp (prop->fwname, "auto") == 0)
    gst_tensor_filter_get_available_framework (priv, "auto");

  /**
   * Reload model if FW has been already opened;
   * In the case of reloading model files, each priv->fw (tensor filter for each nnfw)
   * has responsibility for the verification of the path regardless of priv->fw->verify_model_path.
   */
  if (prop->fw_opened) {
    if (GST_TF_FW_V0 (priv->fw) && priv->is_updatable) {
      if (priv->fw->reloadModel &&
          priv->fw->reloadModel (prop, &priv->privateData) != 0) {
        status = -1;
      }
    } else if (GST_TF_FW_V1 (priv->fw) && priv->is_updatable) {
      GstTensorFilterFrameworkEventData data;
      data.model_files = prop->model_files;
      data.num_models = prop->num_models;
      /** original prop is sent and not the updated prop */
      if (priv->fw->eventHandler (priv->fw, &_prop, priv->privateData,
              RELOAD_MODEL, &data) != 0) {
        status = -1;
      }
    }

    if (status == 0) {
      g_strfreev_const (_prop.model_files);
    } else {
      ml_loge ("Fail to reload model\n");
      g_strfreev_const (prop->model_files);
      prop->model_files = _prop.model_files;
      prop->num_models = _prop.num_models;
    }
  }

  return 0;
}

/** @brief Handle "PROP_INPUT" for set-property */
static gint
_gtfc_setprop_INPUT (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  if (!prop->input_configured && value) {
    guint num_dims;
    gchar **str_dims;
    guint i;

    str_dims = g_strsplit_set (g_value_get_string (value), ",.", -1);
    num_dims = g_strv_length (str_dims);

    if (num_dims > NNS_TENSOR_SIZE_LIMIT) {
      GST_WARNING ("Invalid param, dimensions (%d) max (%d)\n",
          num_dims, NNS_TENSOR_SIZE_LIMIT);

      num_dims = NNS_TENSOR_SIZE_LIMIT;
    }

    for (i = 0; i < num_dims; ++i) {
      prop->input_ranks[i] = gst_tensor_parse_dimension (str_dims[i],
          prop->input_meta.info[i].dimension);
    }
    g_strfreev (str_dims);

    if (num_dims > 0) {
      if (prop->input_meta.num_tensors > 0 &&
          prop->input_meta.num_tensors != num_dims) {
        ml_logw
            ("Invalid input-dim, given param does not match with old value.");
      }

      prop->input_meta.num_tensors = num_dims;
    }
  } else if (value) {
    /** Once configured, it cannot be changed in runtime for now */
    ml_loge
        ("Cannot change input-dim once the element/pipeline is configured.");
  }
  return 0;
}

/** @brief Handle "PROP_OUTPUT" for set-property */
static gint
_gtfc_setprop_OUTPUT (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  if (!prop->output_configured && value) {
    guint num_dims;
    gchar **str_dims;
    guint i;

    str_dims = g_strsplit_set (g_value_get_string (value), ",.", -1);
    num_dims = g_strv_length (str_dims);

    if (num_dims > NNS_TENSOR_SIZE_LIMIT) {
      GST_WARNING ("Invalid param, dimensions (%d) max (%d)\n",
          num_dims, NNS_TENSOR_SIZE_LIMIT);

      num_dims = NNS_TENSOR_SIZE_LIMIT;
    }

    for (i = 0; i < num_dims; ++i) {
      prop->output_ranks[i] = gst_tensor_parse_dimension (str_dims[i],
          prop->output_meta.info[i].dimension);
    }
    g_strfreev (str_dims);

    if (num_dims > 0) {
      if (prop->output_meta.num_tensors > 0 &&
          prop->output_meta.num_tensors != num_dims) {
        ml_logw
            ("Invalid output-dim, given param does not match with old value.");
      }

      prop->output_meta.num_tensors = num_dims;
    }
  } else if (value) {
    /** Once configured, it cannot be changed in runtime for now */
    ml_loge
        ("Cannot change output-dim once the element/pipeline is configured.");
  }
  return 0;
}

/** @brief Handle "PROP_INPUTTYPE" for set-property */
static gint
_gtfc_setprop_INPUTTYPE (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  if (!prop->input_configured && value) {
    guint num_types;

    num_types = gst_tensors_info_parse_types_string (&prop->input_meta,
        g_value_get_string (value));

    if (num_types > 0) {
      if (prop->input_meta.num_tensors > 0 &&
          prop->input_meta.num_tensors != num_types) {
        ml_logw
            ("Invalid input-type, given param does not match with old value.");
      }

      prop->input_meta.num_tensors = num_types;
    }
  } else if (value) {
    /** Once configured, it cannot be changed in runtime for now */
    ml_loge
        ("Cannot change input-type once the element/pipeline is configured.");
  }
  return 0;
}

/** @brief Handle "PROP_OUTPUTTYPE" for set-property */
static gint
_gtfc_setprop_OUTPUTTYPE (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  if (!prop->output_configured && value) {
    guint num_types;

    num_types = gst_tensors_info_parse_types_string (&prop->output_meta,
        g_value_get_string (value));

    if (num_types > 0) {
      if (prop->output_meta.num_tensors > 0 &&
          prop->output_meta.num_tensors != num_types) {
        ml_logw
            ("Invalid output-type, given param does not match with old value.");
      }

      prop->output_meta.num_tensors = num_types;
    }
  } else if (value) {
    /** Once configured, it cannot be changed in runtime for now */
    ml_loge
        ("Cannot change output-type once the element/pipeline is configured.");
  }
  return 0;
}

/** @brief Handle "PROP_INPUTNAME" for set-property */
static gint
_gtfc_setprop_INPUTNAME (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  if (!prop->input_configured && value) {
    guint num_names;

    num_names = gst_tensors_info_parse_names_string (&prop->input_meta,
        g_value_get_string (value));

    if (num_names > 0) {
      if (prop->input_meta.num_tensors > 0 &&
          prop->input_meta.num_tensors != num_names) {
        ml_logw
            ("Invalid input-name, given param does not match with old value.");
      }

      prop->input_meta.num_tensors = num_names;
    }
  } else if (value) {
    /** Once configured, it cannot be changed in runtime for now */
    ml_loge
        ("Cannot change input-name once the element/pipeline is configured.");
  }
  return 0;
}

/** @brief Handle "PROP_OUTPUTNAME" for set-property */
static gint
_gtfc_setprop_OUTPUTNAME (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  if (!prop->output_configured && value) {
    guint num_names;

    num_names = gst_tensors_info_parse_names_string (&prop->output_meta,
        g_value_get_string (value));

    if (num_names > 0) {
      if (prop->output_meta.num_tensors > 0 &&
          prop->output_meta.num_tensors != num_names) {
        ml_logw
            ("Invalid output-name, given param does not match with old value.");
      }

      prop->output_meta.num_tensors = num_names;
    }
  } else if (value) {
    /** Once configured, it cannot be changed in runtime for now */
    ml_loge
        ("Cannot change output-name once the element/pipeline is configured.");
  }
  return 0;
}

/** @brief Handle "PROP_CUSTOM" for set-property */
static gint
_gtfc_setprop_CUSTOM (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  gint status = 0;
  if (!priv->prop.fw_opened) {
    g_free_const (prop->custom_properties);
    prop->custom_properties = g_value_dup_string (value);
  } else {
    if (GST_TF_FW_V0 (priv->fw)) {
      ml_loge
          ("Cannot change custom-prop once the element/pipeline is configured.");
    } else if (GST_TF_FW_V1 (priv->fw)) {
      GstTensorFilterFrameworkEventData data;

      data.custom_properties = g_value_dup_string (value);
      status = priv->fw->eventHandler
          (priv->fw, prop, priv->privateData, CUSTOM_PROP, &data);
      if (status == 0) {
        g_free_const (prop->custom_properties);
        prop->custom_properties = g_value_dup_string (value);
      }

      g_free_const (data.custom_properties);
    }
  }

  return 0;
}

/** @brief Handle "PROP_ACCELERATOR" for set-property */
static gint
_gtfc_setprop_ACCELERATOR (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  gint status = 0;
  const gchar *accelerators = g_value_get_string (value);

  if (priv->prop.fw_opened == TRUE) {
    if (GST_TF_FW_V0 (priv->fw)) {
      ml_loge
          ("Cannot change accelerator once the element/pipeline is configured.");
    } else if (GST_TF_FW_V1 (priv->fw)) {
      GstTensorFilterProperties _prop;
      GstTensorFilterFrameworkEventData data;
      memcpy (&_prop, prop, sizeof (GstTensorFilterProperties));

      prop->hw_list = NULL;
      gst_tensor_filter_parse_accelerator (priv, prop, accelerators);
      data.num_hw = prop->num_hw;
      data.hw_list = prop->hw_list;

      status = priv->fw->eventHandler
          (priv->fw, &_prop, priv->privateData, SET_ACCELERATOR, &data);
      if (status == 0) {
        g_free (_prop.hw_list);
      } else {
        prop->num_hw = _prop.num_hw;
        g_free (prop->hw_list);
        prop->hw_list = _prop.hw_list;
      }
    }
    return 0;
  }

  if (priv->fw) {
    if (GST_TF_FW_V0 (priv->fw)) {
      g_free_const (prop->accl_str);
      prop->accl_str = g_strdup (accelerators);
    } else if (GST_TF_FW_V1 (priv->fw)) {
      gst_tensor_filter_parse_accelerator (priv, &priv->prop, accelerators);
    }
  } else {
    prop->accl_str = g_strdup (accelerators);
  }
  return 0;
}

/** @brief Handle "PROP_IS_UPDATABLE" for set-property */
static gint
_gtfc_setprop_IS_UPDATABLE (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  if (priv->fw) {
    if (GST_TF_FW_V0 (priv->fw) && priv->fw->reloadModel == NULL) {
      priv->is_updatable = FALSE;
      return 0;
    } else if (GST_TF_FW_V1 (priv->fw) &&
        priv->fw->eventHandler (priv->fw, prop, priv->privateData, RELOAD_MODEL,
            NULL)
        == -ENOENT) {
      priv->is_updatable = FALSE;
      return 0;
    }
  }

  priv->is_updatable = g_value_get_boolean (value);
  return 0;
}

/** @brief Handle "PROP_INPUTLAYOUT" for set-property */
static gint
_gtfc_setprop_INPUTLAYOUT (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  guint num_layouts;

  if (!prop->output_configured && value) {
    num_layouts = gst_tensors_parse_layouts_string (prop->input_layout,
        g_value_get_string (value));

    if (num_layouts > 0) {
      if (prop->input_meta.num_tensors > 0 &&
          prop->input_meta.num_tensors != num_layouts) {
        ml_logw ("Invalid input-layout, given param does not fit.");
      }

      prop->input_meta.num_tensors = num_layouts;
    }
  } else if (value) {
    /** Update the properties */
    if (GST_TF_FW_V0 (priv->fw)) {
      /* Once configured, it cannot be changed in runtime */
      ml_loge
          ("Cannot change input-layout once the element/pipeline is configured.");
    } else if (GST_TF_FW_V1 (priv->fw)) {
      GstTensorFilterFrameworkEventData data;

      data.info = NULL;
      num_layouts = gst_tensors_parse_layouts_string (data.layout,
          g_value_get_string (value));

      if (num_layouts > 0) {
        if (prop->input_meta.num_tensors > 0 &&
            prop->input_meta.num_tensors != num_layouts) {
          ml_logw ("Invalid input-layout, given param does not fit.");
        }

        if (priv->fw->eventHandler
            (priv->fw, prop, priv->privateData, SET_INPUT_PROP, &data) == 0) {
          memcpy (priv->prop.input_layout, data.layout,
              sizeof (tensor_layout) * NNS_TENSOR_SIZE_LIMIT);
        } else {
          ml_logw ("Unable to update input layout.");
        }
      }
    }
  }
  return 0;
}

/** @brief Handle "PROP_OUTPUTLAYOUT" for set-property */
static gint
_gtfc_setprop_OUTPUTLAYOUT (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  guint num_layouts;

  if (!prop->output_configured && value) {
    num_layouts = gst_tensors_parse_layouts_string (prop->output_layout,
        g_value_get_string (value));

    if (num_layouts > 0) {
      if (prop->output_meta.num_tensors > 0 &&
          prop->output_meta.num_tensors != num_layouts) {
        ml_logw ("Invalid output-layout, given param does not fit.");
      }

      prop->output_meta.num_tensors = num_layouts;
    }
  } else if (value) {
    /** Update the properties */
    if (GST_TF_FW_V0 (priv->fw)) {
      /* Once configured, it cannot be changed in runtime */
      ml_loge
          ("Cannot change output-layout once the element/pipeline is configured.");
    } else if (GST_TF_FW_V1 (priv->fw)) {
      GstTensorFilterFrameworkEventData data;

      data.info = NULL;
      num_layouts = gst_tensors_parse_layouts_string (data.layout,
          g_value_get_string (value));

      if (num_layouts > 0) {
        if (prop->output_meta.num_tensors > 0 &&
            prop->output_meta.num_tensors != num_layouts) {
          ml_logw ("Invalid output-layout, given param does not fit.");
        }

        if (priv->fw->eventHandler
            (priv->fw, prop, priv->privateData, SET_OUTPUT_PROP, &data) == 0) {
          memcpy (priv->prop.output_layout, data.layout,
              sizeof (tensor_layout) * NNS_TENSOR_SIZE_LIMIT);
        } else {
          ml_logw ("Unable to update output layout.");
        }
      }
    }
  }
  return 0;
}

/** @brief Handle "PROP_LATENCY" for set-property */
static gint
_gtfc_setprop_LATENCY (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  gint latency_mode;

  if (!value)
    return 0;

  latency_mode = g_value_get_int (value);
  if (latency_mode != 0 && latency_mode != 1) {
    ml_logw ("Invalid argument, nither 0 (OFF) nor 1 (ON).");
    return 0;
  }

  priv->latency_mode = latency_mode;

  return 0;
}

/** @brief Handle "PROP_THROUGHPUT" for set-property */
static gint
_gtfc_setprop_THROUGHPUT (GstTensorFilterPrivate * priv,
    GstTensorFilterProperties * prop, const GValue * value)
{
  gint throughput_mode;

  if (!value)
    return 0;

  throughput_mode = g_value_get_int (value);
  if (throughput_mode != 0 && throughput_mode != 1) {
    ml_logw ("Invalid argument, nither 0 (OFF) nor 1 (ON).");
    return 0;
  }

  priv->throughput_mode = throughput_mode;

  return 0;
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
  gint status = 0;
  GstTensorFilterProperties *prop;

  prop = &priv->prop;

  switch (prop_id) {
    case PROP_SILENT:
      priv->silent = g_value_get_boolean (value);
      break;
    case PROP_FRAMEWORK:
      status = _gtfc_setprop_FRAMEWORK (priv, prop, value);
      break;
    case PROP_MODEL:
      status = _gtfc_setprop_MODEL (priv, prop, value);
      break;
    case PROP_INPUT:
      status = _gtfc_setprop_INPUT (priv, prop, value);
      break;
    case PROP_OUTPUT:
      status = _gtfc_setprop_OUTPUT (priv, prop, value);
      break;
    case PROP_INPUTTYPE:
      status = _gtfc_setprop_INPUTTYPE (priv, prop, value);
      break;
    case PROP_OUTPUTTYPE:
      status = _gtfc_setprop_OUTPUTTYPE (priv, prop, value);
      break;
    case PROP_INPUTNAME:
      /* INPUTNAME is required by tensorflow to designate the order of tensors */
      status = _gtfc_setprop_INPUTNAME (priv, prop, value);
      break;
    case PROP_OUTPUTNAME:
      /* OUTPUTNAME is required by tensorflow to designate the order of tensors */
      status = _gtfc_setprop_OUTPUTNAME (priv, prop, value);
      break;
    case PROP_CUSTOM:
      status = _gtfc_setprop_CUSTOM (priv, prop, value);
      break;
    case PROP_ACCELERATOR:
      status = _gtfc_setprop_ACCELERATOR (priv, prop, value);
      break;
    case PROP_IS_UPDATABLE:
      status = _gtfc_setprop_IS_UPDATABLE (priv, prop, value);
      break;
    case PROP_INPUTLAYOUT:
      status = _gtfc_setprop_INPUTLAYOUT (priv, prop, value);
      break;
    case PROP_OUTPUTLAYOUT:
      status = _gtfc_setprop_OUTPUTLAYOUT (priv, prop, value);
      break;
    case PROP_LATENCY:
      status = _gtfc_setprop_LATENCY (priv, prop, value);
      break;
    case PROP_THROUGHPUT:
      status = _gtfc_setprop_THROUGHPUT (priv, prop, value);
      break;
    default:
      return FALSE;
  }

  /** Although no one return !0 at this point, let's enable error handling. */
  if (0 != status)
    return FALSE;

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
      g_value_set_string (value, (prop->fwname != NULL) ? prop->fwname : "");
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

        g_value_take_string (value, dim_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_OUTPUT:
      if (prop->output_meta.num_tensors > 0) {
        gchar *dim_str;

        dim_str = gst_tensors_info_get_dimensions_string (&prop->output_meta);

        g_value_take_string (value, dim_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_INPUTRANKS:
      if (prop->input_meta.num_tensors > 0) {
        gchar *rank_str;

        rank_str = gst_tensors_get_rank_string (prop, TRUE);

        g_value_set_string (value, rank_str);
        g_free (rank_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_OUTPUTRANKS:
      if (prop->output_meta.num_tensors > 0) {
        gchar *rank_str;

        rank_str = gst_tensors_get_rank_string (prop, FALSE);

        g_value_set_string (value, rank_str);
        g_free (rank_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_INPUTTYPE:
      if (prop->input_meta.num_tensors > 0) {
        gchar *type_str;

        type_str = gst_tensors_info_get_types_string (&prop->input_meta);

        g_value_take_string (value, type_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_OUTPUTTYPE:
      if (prop->output_meta.num_tensors > 0) {
        gchar *type_str;

        type_str = gst_tensors_info_get_types_string (&prop->output_meta);

        g_value_take_string (value, type_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_INPUTNAME:
      if (prop->input_meta.num_tensors > 0) {
        gchar *name_str;

        name_str = gst_tensors_info_get_names_string (&prop->input_meta);

        g_value_take_string (value, name_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_OUTPUTNAME:
      if (prop->output_meta.num_tensors > 0) {
        gchar *name_str;

        name_str = gst_tensors_info_get_names_string (&prop->output_meta);

        g_value_take_string (value, name_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_CUSTOM:
      g_value_set_string (value,
          (prop->custom_properties != NULL) ? prop->custom_properties : "");
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

      for (i = 0; i < total; ++i) {
        g_string_append (subplugins, ",");
        g_string_append (subplugins, sinfo.names[i]);
      }

      g_value_take_string (value, g_string_free (subplugins, FALSE));
      break;
    }
    case PROP_ACCELERATOR:
    {
      gint idx;
      GString *accl;

      if (priv->fw == NULL || GST_TF_FW_V0 (priv->fw)) {
        if (prop->accl_str != NULL) {
          g_value_set_string (value, prop->accl_str);
        } else {
          g_value_set_string (value, "");
        }
      } else if (GST_TF_FW_V1 (priv->fw)) {
        if (prop->num_hw == 0) {
          g_value_set_string (value, "");
        } else {
          accl = g_string_new (NULL);

          for (idx = 0; idx < prop->num_hw; idx++) {
            g_string_append (accl, get_accl_hw_str (prop->hw_list[idx]));
          }
          g_value_take_string (value, g_string_free (accl, FALSE));
        }
      }
      break;
    }
    case PROP_IS_UPDATABLE:
      g_value_set_boolean (value, priv->is_updatable);
      break;
    case PROP_INPUTLAYOUT:
      if (prop->input_meta.num_tensors > 0) {
        gchar *layout_str;

        layout_str = gst_tensors_get_layout_string (&prop->input_meta,
            prop->input_layout);

        g_value_take_string (value, layout_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_OUTPUTLAYOUT:
      if (prop->output_meta.num_tensors > 0) {
        gchar *layout_str;

        layout_str = gst_tensors_get_layout_string (&prop->output_meta,
            prop->output_layout);

        g_value_take_string (value, layout_str);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_LATENCY:
      if (priv->latency_mode == 1) {
        g_value_set_int (value, prop->latency);
      } else {
        /* invalid */
        g_value_set_int (value, -1);
      }
      break;
    case PROP_THROUGHPUT:
      if (priv->throughput_mode == 1) {
        g_value_set_int (value, prop->throughput);
      } else {
        /* invalid */
        g_value_set_int (value, -1);
      }
      break;
    default:
      /* unknown property */
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Load tensor info from NN model.
 * (both input and output tensor)
 */
void
gst_tensor_filter_load_tensor_info (GstTensorFilterPrivate * priv)
{
  GstTensorFilterProperties *prop;
  GstTensorsInfo in_info, out_info;
  int res_in = -1, res_out = -1;

  prop = &priv->prop;

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  if (GST_TF_FW_V1 (priv->fw)) {
    if (!prop->input_configured || !prop->output_configured) {
      gst_tensor_filter_v1_call (priv, res_in, getModelInfo, GET_IN_OUT_INFO,
          &in_info, &out_info);
      res_out = res_in;
    }
  } else {
    if (!prop->input_configured)
      gst_tensor_filter_v0_call (priv, res_in, getInputDimension, &in_info);
    if (!prop->output_configured)
      gst_tensor_filter_v0_call (priv, res_out, getOutputDimension, &out_info);
  }

  /* supposed fixed in-tensor info if getInputDimension was success. */
  if (!prop->input_configured && res_in == 0) {
    g_assert (in_info.num_tensors > 0);

    /** if set-property called and already has info, verify it! */
    if (prop->input_meta.num_tensors > 0) {
      if (!gst_tensors_info_is_equal (&in_info, &prop->input_meta)) {
        g_critical ("The input tensor is not compatible.");
        gst_tensor_filter_compare_tensors (&in_info, &prop->input_meta);
        goto done;
      }
    } else {
      gst_tensors_info_copy (&prop->input_meta, &in_info);
    }

    prop->input_configured = TRUE;
    silent_debug_info (&in_info, "input tensor");
  }

  /* supposed fixed out-tensor info if getOutputDimension was success. */
  if (!prop->output_configured && res_out == 0) {
    g_assert (out_info.num_tensors > 0);

    /** if set-property called and already has info, verify it! */
    if (prop->output_meta.num_tensors > 0) {
      if (!gst_tensors_info_is_equal (&out_info, &prop->output_meta)) {
        g_critical ("The output tensor is not compatible.");
        gst_tensor_filter_compare_tensors (&out_info, &prop->output_meta);
        goto done;
      }
    } else {
      gst_tensors_info_copy (&prop->output_meta, &out_info);
    }

    prop->output_configured = TRUE;
    silent_debug_info (&out_info, "output tensor");
  }

done:
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
}

/**
 * @brief Open NN framework.
 */
void
gst_tensor_filter_common_open_fw (GstTensorFilterPrivate * priv)
{
  int run_without_model = 0;

  if (!priv->prop.fw_opened && priv->fw) {
    if (priv->fw->open) {
      /* at least one model should be configured before opening fw */
      if (GST_TF_FW_V0 (priv->fw)) {
        run_without_model = priv->fw->run_without_model;
      } else if (GST_TF_FW_V1 (priv->fw)) {
        run_without_model = priv->info.run_without_model;
      }

      if (G_UNLIKELY (!run_without_model) &&
          G_UNLIKELY (!(priv->prop.model_files &&
                  priv->prop.num_models > 0 && priv->prop.model_files[0]))) {
        return;
      }
      /* 0 if successfully loaded. 1 if skipped (already loaded). */
      if (verify_model_path (priv)) {
        if (priv->fw->open (&priv->prop, &priv->privateData) >= 0) {
          /* Update the framework info once it has been opened */
          if (GST_TF_FW_V1 (priv->fw) &&
              priv->fw->getFrameworkInfo (priv->fw, &priv->prop,
                  priv->privateData, &priv->info) != 0) {
            priv->fw->close (&priv->prop, &priv->privateData);
          } else {
            priv->prop.fw_opened = TRUE;
          }
        }
      }
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
    priv->prop.input_configured = priv->prop.output_configured = FALSE;
    priv->prop.fw_opened = FALSE;
    g_free_const (priv->prop.fwname);
    priv->prop.fwname = NULL;
    priv->fw = NULL;
    priv->privateData = NULL;
    priv->configured = FALSE;
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
 * @brief parse user given string to extract list of accelerators based on given regex
 * @param[in] accelerators user given input
 * @param[in] supported_accelerators list of supported accelerators
 * @return Corresponding list of accelerators maintaining given order
 * @note Returned list must be freed by the caller
 */
static GList *
parse_accl_hw_all (const gchar * accelerators,
    const gchar ** supported_accelerators)
{
  GRegex *nnapi_elem;
  GMatchInfo *match_info;
  gboolean use_accl;
  accl_hw accl;
  gchar *regex_accl = NULL;
  gchar *regex_accl_elem = NULL;
  GList *match_accl = NULL;

  if (accelerators == NULL) {
    match_accl = g_list_append (match_accl, GINT_TO_POINTER (ACCL_DEFAULT));
    return match_accl;
  }

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
        if (accl > 0 || (accl == 0 && g_strcmp0 (word, ACCL_NONE_STR) == 0)) {
          match_accl = g_list_append (match_accl, GINT_TO_POINTER (accl));
        }
        g_free (word);
        g_match_info_next (match_info, NULL);
      }
    } else {
      ml_logw
          ("Using AUTO accelerator config, User provided accelerator(s) do not intersect with framework's supported accelerators.");
    }
    g_match_info_free (match_info);
    g_regex_unref (nnapi_elem);

    if (g_list_length (match_accl) == 0) {
      match_accl = g_list_append (match_accl, GINT_TO_POINTER (ACCL_AUTO));
    }
  } else {
    match_accl = g_list_append (match_accl, GINT_TO_POINTER (ACCL_NONE));
  }

  return match_accl;
}

/**
 * @brief Added basic accelerators (auto, default) to supported accelerators
 * @note returned array must be freed by the caller
 */
static const gchar **
add_basic_supported_accelerators (const gchar ** supported_accelerators)
{
  gint num_hw = 0, idx = 0;
  const gchar **accl_support;

  /** Count number of elements for the array */
  while (supported_accelerators[num_hw] != NULL)
    num_hw += 1;
  num_hw += 2;

  /** Allocate the array */
  accl_support = g_try_malloc0 (sizeof (gchar *) * (num_hw + 1));
  if (accl_support == NULL) {
    ml_loge ("Failed to allocate memory for accelerators");
    return NULL;
  }

  /** Fill the array */
  while (supported_accelerators[idx] != NULL) {
    accl_support[idx] = supported_accelerators[idx];
    idx += 1;
  }
  accl_support[idx++] = ACCL_AUTO_STR;
  accl_support[idx++] = ACCL_DEFAULT_STR;
  accl_support[idx] = NULL;

  return accl_support;
}

/**
 * @brief Filter accelerators based on the runtime system
 * @note returned array must be freed by the caller
 * @details This filters out NEON accelerator if the system running the
 * tensor_filter does not support NEON instructions
 */
static const gchar **
filter_supported_accelerators (const gchar ** supported_accelerators)
{
  gint num_hw = 0, idx = 0;
  const gchar **accl_support;
  gint neon_available = cpu_neon_accel_available ();

  /** Count number of elements for the array */
  while (supported_accelerators[num_hw] != NULL) {
    num_hw += 1;
  }

  /** Allocate the array */
  accl_support = g_malloc (sizeof (gchar *) * (num_hw + 1));

  /** Fill the array */
  idx = 0;
  num_hw = 0;
  while (supported_accelerators[idx] != NULL) {
    if (g_ascii_strncasecmp (supported_accelerators[idx], ACCL_CPU_NEON_STR,
            strlen (ACCL_CPU_NEON_STR)) == 0 && neon_available != 0) {
      g_critical ("Neon instructions are not available on this device.");
    } else {
      accl_support[num_hw] = supported_accelerators[idx];
      num_hw += 1;
    }
    idx += 1;
  }
  accl_support[num_hw] = NULL;

  return accl_support;
}

/**
 * @brief parse user given string to extract accelerator based on given regex
 * @param[in] accelerators user given input
 * @param[in] supported_accelerators list of supported accelerators
 * @param[in] auto_accelerator accelerator to use in auto case (when acceleration is enabled but specific accelerator is not provided or not matching)
 * @param[in] default_accelerator accelerator to use by default
 * @return Corresponding accelerator. Returns ACCL_NONE if not found.
 */
static accl_hw
parse_accl_hw_util (const gchar * accelerators,
    const gchar ** supported_accelerators, const gchar * auto_accelerator,
    const gchar * default_accelerator)
{
  GList *match_accl;
  accl_hw hw;
  const gchar **all_supported_accelerators;

  /** add auto and default accelerator into list of supported accelerators */
  all_supported_accelerators =
      add_basic_supported_accelerators (supported_accelerators);
  if (all_supported_accelerators) {
    match_accl = parse_accl_hw_all (accelerators, all_supported_accelerators);
    g_free (all_supported_accelerators);
  } else {
    return ACCL_NONE;
  }

  if (NULL == match_accl) {
    ml_loge ("There is no match hardware accelerators from {%s}.\n",
        accelerators);
    return ACCL_NONE;
  }

  hw = GPOINTER_TO_INT (match_accl->data);
  g_list_free (match_accl);

  if (hw == ACCL_AUTO)
    return get_accl_hw_type (auto_accelerator);
  else if (hw == ACCL_DEFAULT)
    return get_accl_hw_type (default_accelerator);

  /** This can be ACCL_NONE (no acceleration) or a specific accelerator */
  return hw;
}

/**
 * @brief Check if this accelerator can be used based on the runtime system
 * @retval 0 if filter can be used, -errno otherwise
 */
static gint
runtime_check_supported_accelerator (const gchar * accl)
{
  const gchar **accl_support, **filtered_accl_support;
  gint ret = 0;

  /** Allocate the array */
  accl_support = g_malloc (sizeof (gchar *) * (2));

  /** Fill the array */
  accl_support[0] = accl;
  accl_support[1] = NULL;

  filtered_accl_support = filter_supported_accelerators (accl_support);
  if (!filtered_accl_support || filtered_accl_support[0] == NULL) {
    ret = -ENOENT;
  } else {
    ret = 0;
  }

  g_free (filtered_accl_support);
  g_free (accl_support);

  return ret;
}

/**
 * @brief parse user given string to extract accelerator based on given regex filling in optional arguments
 */
accl_hw
parse_accl_hw_fill (parse_accl_args accl_args)
{
  const gchar *in_accl = accl_args.in_accl;
  const gchar **sup_accl = accl_args.sup_accl;
  const gchar *def_accl, *auto_accl;
  const gchar **filtered_accl;
  accl_hw ret = ACCL_NONE;

  if (accl_args.sup_accl == NULL || accl_args.sup_accl[0] == NULL)
    return ret;

  /** remove unsupported accelerators from this list based on runtime system */
  filtered_accl = filter_supported_accelerators (accl_args.sup_accl);
  if (!filtered_accl) {
    return ret;
  }

  /** filtered supported accelerators can be empty */
  sup_accl = filtered_accl;
  if (sup_accl[0] == NULL) {
    g_free (filtered_accl);
    return ret;
  }

  /** update default accelerator if it is not available at runtime */
  if (accl_args.def_accl &&
      runtime_check_supported_accelerator (accl_args.def_accl) == 0) {
    def_accl = accl_args.def_accl;
  } else {
    def_accl = sup_accl[0];
  }

  /** update auto accelerator if it is not available at runtime */
  if (accl_args.auto_accl &&
      runtime_check_supported_accelerator (accl_args.auto_accl) == 0) {
    auto_accl = accl_args.auto_accl;
  } else {
    auto_accl = sup_accl[0];
  }

  ret = parse_accl_hw_util (in_accl, sup_accl, auto_accl, def_accl);
  g_free (filtered_accl);

  return ret;
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
#if defined(__aarch64__) || defined(__arm__)
      /** Retreive NEON_STR when searching for SIMD/NEON on arm architectures */
      {ACCL_CPU_NEON, ACCL_CPU_NEON_STR, ACCL_CPU_NEON_STR},
#endif
      {ACCL_CPU_SIMD, ACCL_CPU_SIMD_STR, ACCL_CPU_SIMD_STR},
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

/**
 * @brief check if the given hw is supported by the framework
 */
gboolean
gst_tensor_filter_check_hw_availability (const GstTensorFilterFramework * fw,
    accl_hw hw)
{
  gint idx = 0;
  gboolean available = FALSE;
  GstTensorFilterFrameworkInfo info;
  GstTensorFilterProperties prop;

  /** Only check for specific HW, DEFAULT/AUTO are always supported */
  if (hw == ACCL_AUTO || hw == ACCL_DEFAULT) {
    available = TRUE;
  } else if (GST_TF_FW_V0 (fw) &&
      (!fw->checkAvailability || fw->checkAvailability (hw) == 0)) {
    available = TRUE;
  } else if (GST_TF_FW_V1 (fw)) {
    gst_tensor_filter_properties_init (&prop);
    if (!fw->getFrameworkInfo (fw, &prop, NULL, &info)) {
      for (idx = 0; idx < info.num_hw; idx++) {
        if (info.hw_list[idx] == hw) {
          available = TRUE;
          break;
        }
      }
    }
  }

  return available;
}
