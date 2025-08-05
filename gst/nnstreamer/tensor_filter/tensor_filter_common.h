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

#include <glib-object.h>
#include <errno.h>
#include <nnstreamer_subplugin.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_plugin_api_filter.h>
#include "nnstreamer_watchdog.h"

G_BEGIN_DECLS

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!priv->silent)
#endif

#define TF_MODELNAME(prop) \
    ((prop)->model_files ? ((prop)->model_files[0]) : "[No Model File]")

/** Check tensor_filter framework version */
#define GST_TF_FW_VN(fw, vn) \
    (fw && checkGstTensorFilterFrameworkVersion (fw->version, vn))
#define GST_TF_FW_V0(fw) GST_TF_FW_VN (fw, 0)
#define GST_TF_FW_V1(fw) GST_TF_FW_VN (fw, 1)

/**
 * @brief Invoke callbacks of nn framework. Guarantees calling open for the first call.
 */
#define gst_tensor_filter_v0_call(priv,ret,funcname,...) do { \
      if (!gst_tensor_filter_common_open_fw (priv) || !(priv)->fw->funcname) { \
        ret = -1; \
      } else { \
        ret = (priv)->fw->funcname (&(priv)->prop, &(priv)->privateData, __VA_ARGS__); \
      } \
    } while (0)

#define gst_tensor_filter_v1_call(priv,ret,funcname,...) do { \
      if (!gst_tensor_filter_common_open_fw (priv) || !(priv)->fw->funcname) { \
        ret = -1; \
      } else { \
        ret = (priv)->fw->funcname ((priv)->fw, &(priv)->prop, (priv)->privateData, __VA_ARGS__); \
      } \
    } while (0)

#define GST_TF_FW_INVOKE_COMPAT(priv,ret,in,out) do { \
      ret = -1; \
      if (GST_TF_FW_V0 ((priv)->fw)) { \
        ret = (priv)->fw->invoke_NN (&(priv)->prop, &(priv)->privateData, (in), (out)); \
      } else if (GST_TF_FW_V1 ((priv)->fw)) { \
        ret = (priv)->fw->invoke ((priv)->fw, &(priv)->prop, (priv)->privateData, (in), (out)); \
      } \
    } while (0)

#define GST_TF_STAT_MAX_RECENT (10)

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
  PROP_INPUTCOMBINATION,
  PROP_OUTPUTCOMBINATION,
  PROP_SHARED_TENSOR_FILTER_KEY,
  PROP_LATENCY_REPORT,
  PROP_INVOKE_DYNAMIC,
  PROP_INVOKE_ASYNC,
  PROP_CONFIG,
  PROP_SUSPEND
};

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
  guint latency_ignore_count;   /* number of initial latency measurements to ignore in averaging */
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
 * @brief Data Structure to store shared table
 */
typedef struct {
  void *shared_interpreter; /**< the model representation for each sub-plugins */
  GList *referred_list; /**< the referred list about the instances sharing the same key */
} GstTensorFilterSharedModelRepresentation;

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
  gchar *config_path; /**< Path to configuration file */
  GstTensorsConfig in_config; /**< input tensor info */
  GstTensorsConfig out_config; /**< output tensor info */

  gint latency_mode;     /**< latency profiling mode (0: off, 1: on, ...) */
  gint throughput_mode;  /**< throughput profiling mode (0: off, 1: on, ...) */
  gboolean latency_reporting; /**< reporting of estimated filter latency is enabled */
  gint64 latency_reported; /**< latency value reported (ns) in last LATENCY query */

  GstTensorFilterCombination combi;
  nns_watchdog_h watchdog_h; /**< Watchdog to monitor occasional inputs */;
} GstTensorFilterPrivate;

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
extern void
gst_tensor_filter_install_properties (GObjectClass * gobject_class);

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
 * @brief Configure input tensor info with combi option.
 */
extern gboolean
gst_tensor_filter_common_get_combined_in_info (GstTensorFilterPrivate * priv,
    const GstTensorsInfo * in, GstTensorsInfo * combined);

/**
 * @brief Configure output tensor info with combi option.
 */
extern gboolean
gst_tensor_filter_common_get_combined_out_info (GstTensorFilterPrivate * priv,
    const GstTensorsInfo * in, const GstTensorsInfo * out, GstTensorsInfo * combined);

/**
 * @brief Get output tensor info from NN model with given input info.
 */
extern gboolean
gst_tensor_filter_common_get_out_info (GstTensorFilterPrivate * priv,
    GstTensorsInfo * in, GstTensorsInfo * out);

/**
 * @brief Load tensor info from NN model.
 * (both input and output tensor)
 */
extern void
gst_tensor_filter_load_tensor_info (GstTensorFilterPrivate * priv);

/**
 * @brief Open NN framework.
 */
extern gboolean
gst_tensor_filter_common_open_fw (GstTensorFilterPrivate * priv);

/**
 * @brief Unload NN framework.
 */
extern void
gst_tensor_filter_common_unload_fw (GstTensorFilterPrivate * priv);

/**
 * @brief Close NN framework.
 */
extern void
gst_tensor_filter_common_close_fw (GstTensorFilterPrivate * priv);

/**
 * @brief Get neural network framework name from given model file. This does not guarantee the framework is available on the target device.
 * @param[in] model_files the prediction model paths
 * @param[in] num_models the number of model files
 * @param[in] load_conf flag to load configuration for the priority of framework
 * @return Possible framework name (NULL if it fails to detect automatically). Caller should free returned value using g_free().
 */
extern gchar *
gst_tensor_filter_detect_framework (const gchar * const *model_files, const guint num_models, const gboolean load_conf);

/**
 * @brief Check if the given hw is supported by the framework.
 * @param[in] name The name of filter sub-plugin.
 * @param[in] hw Backend accelerator hardware.
 * @param[in] custom User-defined string to handle detailed hardware option.
 * @return TRUE if given hw is available.
 */
extern gboolean
gst_tensor_filter_check_hw_availability (const gchar * name, const accl_hw hw, const gchar * custom);

/**
 * @brief Free the data allocated for tensor filter output
 */
extern void
gst_tensor_filter_destroy_notify_util (GstTensorFilterPrivate *priv, void *data);

/**
 * @brief Enables tensor-filter to use asynchronous invoke.
 *        Sets callback and the private data for asynchronous operations.
 *
 * This function is used when the sub-plugin receives an input and generates multiple outputs asynchronously.
 *
 * @param[in] prop GstTensorFilterProperties object.
 * @param[in] callback The callback function to be invoked during async operations.
 * @param[in] user_data The private data to be passed to the callback function.
 */
extern void
gst_tensor_filter_enable_invoke_async (GstTensorFilterProperties * prop, GstTensorDataCallback callback, void *user_data);

/**
 * @brief Disables the asynchronous invoke for tensor_filter.
 *        Resets callback and handle to disable asynchronous operations.
 *
 * @param[in] prop GstTensorFilterProperties object.
 */
extern void
gst_tensor_filter_disable_invoke_async (GstTensorFilterProperties * prop);

G_END_DECLS
#endif /* __G_TENSOR_FILTER_COMMON_H__ */
