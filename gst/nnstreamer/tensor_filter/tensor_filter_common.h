/**
 * Copyright (C) 2019 Parichay kapoor <pk.kapoor@samsung.com>
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
 * @file	tensor_filter_common.h
 * @date	28 Aug 2019
 * @brief	Common functions for various tensor_filters
 * @see	  http://github.com/nnstreamer/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug	  No known bugs except for NYI items
 */

#ifndef __G_TENSOR_FILTER_COMMON_H__
#define __G_TENSOR_FILTER_COMMON_H__

#include <nnstreamer_subplugin.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!priv->silent)
#endif

/** Check tensor_filter framework version */
#define GST_TF_FW_VN(fw, vn) \
    (fw && checkGstTensorFilterFrameworkVersion (fw->version, vn))
#define GST_TF_FW_V0(fw) GST_TF_FW_VN (fw, 0)
#define GST_TF_FW_V1(fw) GST_TF_FW_VN (fw, 1)

/**
 * @brief Invoke callbacks of nn framework. Guarantees calling open for the first call.
 */
#define gst_tensor_filter_v0_call(priv,ret,funcname,...) do { \
      gst_tensor_filter_common_open_fw (priv); \
      ret = -1; \
      if ((priv)->prop.fw_opened && (priv)->fw && (priv)->fw->funcname) { \
        ret = (priv)->fw->funcname (&(priv)->prop, &(priv)->privateData, __VA_ARGS__); \
      } \
    } while (0)

#define gst_tensor_filter_v1_call(priv,ret,funcname,...) do { \
      gst_tensor_filter_common_open_fw (priv); \
      ret = -1; \
      if ((priv)->prop.fw_opened && (priv)->fw && (priv)->fw->funcname) { \
        ret = (priv)->fw->funcname ((priv)->fw, &(priv)->prop, (priv)->privateData, __VA_ARGS__); \
      } \
    } while (0)

#define GST_TF_FW_INVOKE_COMPAT(priv,ret,in,out) do { \
      if (GST_TF_FW_V0 ((priv)->fw)) { \
        ret = priv->fw->invoke_NN (&(priv)->prop, &(priv)->privateData, (in), (out)); \
      } else if (GST_TF_FW_V1 ((priv)->fw)) { \
        ret = priv->fw->invoke ((priv)->fw, &(priv)->prop, (priv)->privateData, (in), (out)); \
      } else { \
        g_assert(FALSE); \
      } \
    } while (0)

#define GST_TF_STAT_MAX_RECENT (10)

/**
 * @brief Structure definition for tensor-filter statistics
 */
typedef struct _GstTensorFilterStatistics
{
  gint64 total_invoke_num;      /**< number of total invokes */
  gint64 total_invoke_latency;  /**< accumulated invoke latency (usec) */
  gint64 old_total_invoke_num;      /**< cached value. number of total invokes */
  gint64 old_total_invoke_latency;  /**< cached value. accumulated invoke latency (usec) */
  gint64 latest_invoke_time;    /**< the latest invoke time (usec) */
  void *recent_latencies;       /**< data structure (e.g., queue) to hold recent latencies */
} GstTensorFilterStatistics;

/**
 * @brief Structure definition for tensor-filter in/out combination
 */
typedef struct _GstTensorFilterCombination
{
  GList *in_combi; /**< Select the input tensor(s) to invoke the models */
  GList *out_combi_i; /**< Select the output tensor(s) from the input tensor(s) */
  GList *out_combi_o; /**< Select the output tensor(s) from the model output */
  gboolean in_combi_defined; /**< True if input combination is defined */
  gboolean out_combi_i_defined;/**< True if output combination from input is defined */
  gboolean out_combi_o_defined;/**< True if output combination from model output is defined */
} GstTensorFilterCombination;

/**
 * @brief Structure definition for common tensor-filter properties.
 */
typedef struct _GstTensorFilterPrivate
{
  void *privateData; /**< NNFW plugin's private data is stored here */
  GstTensorFilterProperties prop; /**< NNFW plugin's properties */
  GstTensorFilterFrameworkInfo info; /**< NNFW framework info */
  GstTensorFilterStatistics stat; /**< NNFW plugin's statistics */
  const GstTensorFilterFramework *fw; /**< The implementation core of the NNFW. NULL if not configured */

  /* internal properties for tensor-filter */
  gboolean silent; /**< Verbose mode if FALSE. int instead of gboolean for non-glib custom plugins */
  gboolean configured; /**< True if already successfully configured tensor metadata */
  gboolean is_updatable; /**<  a given model to the filter is updatable if TRUE */
  GstTensorsConfig in_config; /**< input tensor info */
  GstTensorsConfig out_config; /**< output tensor info */

  gint latency_mode;     /**< latency profiling mode (0: off, 1: on, ...) */
  gint throughput_mode;  /**< throughput profiling mode (0: off, 1: on, ...) */

  GstClockTime prev_ts;  /**< previous timestamp */
  GstClockTimeDiff throttling_delay;  /**< throttling delay from tensor rate */
  GstClockTimeDiff throttling_accum;  /**< accumulated frame durations for throttling */

  GstTensorFilterCombination combi;
} GstTensorFilterPrivate;

/**
 * @brief Printout the comparison results of two tensors.
 * @param[in] info1 The tensors to be shown on the left hand side
 * @param[in] info2 The tensors to be shown on the right hand side
 * @todo If this is going to be used by other elements, move this to nnstreamer/tensor_common.
 */
extern void
gst_tensor_filter_compare_tensors (GstTensorsInfo * info1,
    GstTensorsInfo * info2);

/**
 * @brief check if the allocate_in_invoke is valid for the framework
 * @param[in] priv Struct containing the properties of the object
 * @return TRUE if valid, FALSE on error
 */
extern gboolean
gst_tensor_filter_allocate_in_invoke (GstTensorFilterPrivate * priv);

/**
 * @brief Installs all the properties for tensor_filter
 * @param[in] gobject_class Glib object class whose properties will be set
 */
extern void gst_tensor_filter_install_properties (GObjectClass * gobject_class);

/**
 * @brief Initialize the properties for tensor-filter.
 */
extern void
gst_tensor_filter_common_init_property (GstTensorFilterPrivate * priv);

/**
 * @brief Free the properties for tensor-filter.
 */
extern void
gst_tensor_filter_common_free_property (GstTensorFilterPrivate * priv);

/**
 * @brief Get available framework from given file when user selects auto option
 * @param[in] model_files the prediction model paths
 * @param[in] num_models the number of model files
 * @return Detected framework name (NULL if it fails to detect automatically)
 */
gchar *gst_tensor_filter_framework_auto_detection (const gchar ** model_files,
    unsigned int num_models);

/**
 * @brief automatically selecting framework for tensor filter
 * @param[in] priv Struct containing the properties of the object
 * @param[in] fw_name Framework name
 */
void
gst_tensor_filter_get_available_framework (GstTensorFilterPrivate * priv,
    const char *fw_name);

/**
 * @brief Set the properties for tensor_filter
 * @param[in] priv Struct containing the properties of the object
 * @param[in] prop_id Id for the property
 * @param[in] value Container to return the asked property
 * @param[in] pspec Metadata to specify the parameter
 * @return TRUE if prop_id is value, else FALSE
 */
extern gboolean
gst_tensor_filter_common_set_property (GstTensorFilterPrivate * priv,
    guint prop_id, const GValue * value, GParamSpec * pspec);

/**
 * @brief Get the properties for tensor_filter
 * @param[in] priv Struct containing the properties of the object
 * @param[in] prop_id Id for the property
 * @param[in] value Container to return the asked property
 * @param[in] pspec Metadata to specify the parameter
 * @return TRUE if prop_id is value, else FALSE
 */
extern gboolean
gst_tensor_filter_common_get_property (GstTensorFilterPrivate * priv,
    guint prop_id, GValue * value, GParamSpec * pspec);

/**
 * @brief Load tensor info from NN model.
 * (both input and output tensor)
 */
extern void
gst_tensor_filter_load_tensor_info (GstTensorFilterPrivate * priv);

/**
 * @brief Open NN framework.
 */
extern void gst_tensor_filter_common_open_fw (GstTensorFilterPrivate * priv);

/**
 * @brief Close NN framework.
 */
extern void gst_tensor_filter_common_close_fw (GstTensorFilterPrivate * priv);

/**
 * @brief check if the given hw is supported by the framework
 */
extern gboolean
gst_tensor_filter_check_hw_availability (const GstTensorFilterFramework *fw,
    accl_hw hw);

/**
 * @brief Free the data allocated for tensor filter output
 */
extern void
gst_tensor_filter_destroy_notify_util (GstTensorFilterPrivate *priv, void *data);

#endif /* __G_TENSOR_FILTER_COMMON_H__ */
