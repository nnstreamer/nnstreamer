/**
 * NNStreamer API for Tensor_Filter Sub-Plugins
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file  nnstreamer_plugin_api_filter.h
 * @date  30 Jan 2019
 * @brief Mandatory APIs for NNStreamer Filter sub-plugins (No External Dependencies)
 * @see https://github.com/nnsuite/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __NNS_PLUGIN_API_FILTER_H__
#define __NNS_PLUGIN_API_FILTER_H__

#include "tensor_typedef.h"

/** Macros for accelerator types */
#define ACCL_NONE_STR "none"
#define ACCL_DEFAULT_STR  "default"
#define ACCL_AUTO_STR "auto"
#define ACCL_CPU_STR  "cpu"
#define ACCL_CPU_NEON_STR  "cpu.neon"
#define ACCL_GPU_STR  "gpu"
#define ACCL_NPU_STR  "npu"
#define ACCL_NPU_MOVIDIUS_STR  "npu.movidius"
#define ACCL_NPU_EDGE_TPU_STR  "npu.edgetpu"
#define ACCL_NPU_VIVANTE_STR  "npu.vivante"
#define ACCL_NPU_SRCN_STR  "npu.srcn" /** srcn hardware supported by nnfw */
#define ACCL_NPU_SR_STR  "npu.sr"

#define GST_TENSOR_FILTER_FRAMEWORK_BASE (0xDEAFDEAD00000000ULL)
#define GST_TENSOR_FILTER_FRAMEWORK_V0 (GST_TENSOR_FILTER_FRAMEWORK_BASE)
#define GST_TENSOR_FILTER_FRAMEWORK_V1 (GST_TENSOR_FILTER_FRAMEWORK_BASE | 0x10000ULL)

/** TODO: update this to 1 after supporting version 1 GstTensorFilterFramework in tensor_filter.c */
#define GST_TENSOR_FILTER_API_VERSION_DEFINED (0)
#define GST_TENSOR_FILTER_API_VERSION_MIN (0)	/* The minimum API version supported (could be obsolete) */
#define GST_TENSOR_FILTER_API_VERSION_MAX (0)	/* The maximum API version supported (recommended) */

/**
 * @brief Check the value of the version field of GstTensorFilterFramework
 */
#define checkGstTensorFilterFrameworkVersion(value, version) \
  ((GST_TENSOR_FILTER_FRAMEWORK_BASE | ((version) << 16)) == (value & 0xFFFFFFFFFFFF0000ULL))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief acceleration hw properties.
 * @details Enabling acceleration (choosing any accelerator hardware other
 *          ACCL_NONE) will enable acceleration for the framework dependent on
 *          the acceleration supported by the framework.
 *
 *          For example, enabling acceleration with tflite will enable NNAPI.
 *          However, with NNFW will enable GPU/NEON etc.
 *
 *          Appropriate acceleration should be used with each framework. For
 *          example, ACCL_CPU_NEON is supported with NNFW tensor filter. Using
 *          ACCL_NEON with pytorch would result in a warning message, and
 *          the accelerator would fallback on ACCL_AUTO.
 *
 *          ACCL_AUTO automatically chooses the accelerator based on the ones
 *          supported by the subplugin framework. However, ACCL_DEFAULT would
 *          use the accelerator set by the subplugin framework, if any.
 *
 * @note Acceleration for a framework can be enabled/disabled in the build. If
 *       the acceleration for a framework was disabled in the build, setting the
 *       accelerator while use a tensor filter with that framework will have no
 *       effect.
 *
 * @note Add definition of new accelerators to accl_hw_get_type() in
 *       tensor_filter_common.c as well.
 */
typedef enum
{
  /**< no explicit acceleration */
  ACCL_NONE    = 0,          /**< no acceleration (defaults to CPU) */

  /** If there is no default config, and device needs to be specified, fallback to ACCL_AUTO */
  ACCL_AUTO    = 0x1,        /**< choose optimized device automatically */
  ACCL_DEFAULT = 0x2,     /**< use default device configuration by the framework */

  /** Enables acceleration, 0xn000 any version of that device, 0xnxxx: device # xxx-1 */
  ACCL_CPU          = 0x1000,     /**< specify device as CPU, if possible */
  ACCL_CPU_NEON     = 0x1100,     /**< specify device as NEON in cpu, if possible */
  ACCL_GPU          = 0x2000,     /**< specify device as GPU, if possible */
  ACCL_NPU          = 0x4000,     /**< specify device as any NPU, if possible */
  ACCL_NPU_MOVIDIUS = 0x4001,     /**< specify device as movidius, if possible */
  ACCL_NPU_EDGE_TPU = 0x4002,     /**< specify device as edge tpu, if possible */
  ACCL_NPU_VIVANTE  = 0x4003,     /**< specify device as vivante, if possible */
  ACCL_NPU_SRCN     = 0x4004,     /**< specify device as srcn, if possible */
  ACCL_NPU_SR       = 0x4100,     /**< specify device as any SR npu, if possible */
} accl_hw;

/**
 * @brief Internal tensor layout format for other/tensor
 *
 * The layout is needed by some of the subplugins to appropriately  process the
 * data based on the axis of the channel in the data. Layout information will be
 * currently utilized by only some of the sublpugins (SNAP, NNFW), and should be
 * provided to use some of the subplugins (SNAP).
 *
 * Tensor layout is stored locally in the tensor filter element, and not shared
 * with other elements in the pipeline. Thus, tensor layout is not part of the
 * capabilities of the element, and does not take part in the caps negotiation.
 *
 * NONE layout implies that the layout of the data is neither NHWC nor NCHW. '
 * However, ANY layout implies that the layout of the provided data is not
 * relevant.
 *
 * @note Providing tensor layout can also decide acceleration to be supported
 * as not all the accelerators might support all the layouts (NYI).
 */
typedef enum _nns_tensor_layout
{
  _NNS_LAYOUT_ANY = 0,     /**< does not care about the data layout */
  _NNS_LAYOUT_NHWC,        /**< NHWC: channel last layout */
  _NNS_LAYOUT_NCHW,        /**< NCHW: channel first layout */
  _NNS_LAYOUT_NONE,        /**< NONE: none of the above defined layouts */
} tensor_layout;

typedef tensor_layout tensors_layout[NNS_TENSOR_SIZE_LIMIT];

/**
 * @brief GstTensorFilter's properties for NN framework (internal data structure)
 *
 * Because custom filters of tensor_filter may need to access internal data
 * of GstTensorFilter, we define this data structure here.
 */
typedef struct _GstTensorFilterProperties
{
  const char *fwname; /**< The name of NN Framework */
  int fw_opened; /**< TRUE IF open() is called or tried. Use int instead of gboolean because this is refered by custom plugins. */
  const char **model_files; /**< Filepath to the model file (as an argument for NNFW). char instead of gchar for non-glib custom plugins */
  int num_models; /**< number of model files. Some frameworks need multiple model files to initialize the graph (caffe, caffe2) */

  int input_configured; /**< TRUE if input tensor is configured. Use int instead of gboolean because this is refered by custom plugins. */
  GstTensorsInfo input_meta; /**< configured input tensor info */
  tensors_layout input_layout; /**< data layout info provided as a property to tensor_filter for the input, defaults to _NNS_LAYOUT_ANY for all the tensors */

  int output_configured; /**< TRUE if output tensor is configured. Use int instead of gboolean because this is refered by custom plugins. */
  GstTensorsInfo output_meta; /**< configured output tensor info */
  tensors_layout output_layout; /**< data layout info provided as a property to tensor_filter for the output, defaults to _NNS_LAYOUT_ANY for all the tensors */

  const char *custom_properties; /**< sub-plugin specific custom property values in string */
  union {
    struct {
      accl_hw *hw_list; /**< accelerators supported by framework intersected with user provided accelerator preference, use in GstTensorFilterFramework V1 only */
      int num_hw;       /**< number of hardare accelerators in the hw_list supported by the framework */
    };
    const char *accl_str; /**< accelerator configuration passed in as parameter, use in GstTensorFilterFramework V0 only */
  };

} GstTensorFilterProperties;

/**
 * @brief Tensor_Filter Subplugin framework related information
 *
 * All the information except the supported acclerator is provided statically.
 * Accelerators can be provided based on static or dynamic check dependent on framework support.
 */
typedef struct _GstTensorFilterFrameworkInfo
{
  char *name; /**< Name of the neural network framework, searchable by FRAMEWORK property */
  int allow_in_place; /**< TRUE(nonzero) if InPlace transfer of input-to-output is allowed. Not supported in main, yet */
  int allocate_in_invoke; /**< TRUE(nonzero) if invoke_NN is going to allocate outputptr by itself and return the address via outputptr. Do not change this value after cap negotiation is complete (or the stream has been started). */
  int run_without_model; /**< TRUE(nonzero) when the neural network framework does not need a model file. Tensor-filter will run invoke_NN without model. */
  int verify_model_path; /**< TRUE(nonzero) when the NNS framework, not the sub-plugin, should verify the path of model files. */
  accl_hw *hw_list; /**< List of supported hardwares by the framework.  Positive response of this check does not guarantee successful running of model with this accelerator. */
  int num_hw; /**< number of hardware accelerators in the hw_list supported by the framework */
} GstTensorFilterFrameworkInfo;

/**
 * @brief Tensor_Filter Subplugin related events
 *
 * These are possible set of events that can be supported by the tensor filter subplugin.
 */
typedef enum
{
  DESTROY_NOTIFY,   /**< Free the data element allocated in the invoke callback */
  RELOAD_MODEL,     /**< Reloads the subplugin with newely provided model */
  CUSTOM_PROP,      /**< Update the custom properties for the framework */
  SET_INPUT_PROP,   /**< Update input tensor info and layout */
  SET_OUTPUT_PROP,  /**< Update output tensor info and layout */
  SET_ACCELERATOR,  /**< Update accelerator of the subplugin to be used as backend */
} event_ops;

/**
 * @brief Tensor_Filter Subplugin's model related info gathering operations
 */
typedef enum
{
  GET_IN_OUT_INFO,   /**< Gets the input and output tensor info */
  SET_INPUT_INFO,   /**< Sets the provided input tensor info, and get updated output tensor info */
} model_info_ops;

/**
 * @brief User data for the tensor_tilter subplugin related events
 */
typedef struct _GstTensorFilterFrameworkEventData
{
  /** Union of the user data for each event supported by eventHandler */
  union {
    /** for DESTROY_NOTIFY event */
    struct {
      void * data;  /**< The data element to be destroyed */
    };

    /** for RELOAD_MODEL event */
    struct {
      const char **model_files;  /**< Filepath to the new list of model files */
      int num_models;   /**< Updated number of the model files */
    };

    /** for CUSTOM_PROP event */
    struct {
      const char *custom_properties; /**< sub-plugin specific custom property values in string */
    };

    /** for SET_INPUT_PROP/SET_OUTPUT_PROP event */
    struct {
      const GstTensorsInfo * info;  /**< The tensor info to be updated to */
      tensors_layout layout;        /**< The layout of the tensor to be updated to */
    };

    /** for SET_ACCELERATOR event */
    struct {
      accl_hw *hw_list;   /**< accelerators supported by framework intersected with the new user provided accelerator preference */
      int num_hw;         /**< number of hardare accelerators in the hw_list supported by the framework */
    };

  };
} GstTensorFilterFrameworkEventData;

/**
 * @brief Tensor_Filter Subplugin definition
 *
 * Common callback parameters:
 * prop Filter properties. Read Only.
 * private_data Subplugin's private data. Set this (*private_data = XXX) if you want to change filter->private_data.
 */
typedef struct _GstTensorFilterFramework
{
  uint64_t version;
  /**< Version of the struct
   * | 32bit (validity check) | 16bit (API version) | 16bit (Subplugin's internal version. Tensor_filter does not case) |
   * API version will be 0x0 (earlier version (_GstTensorFilterFramework_v0)) or 0x1 (newer version (_GstTensorFilterFramework_v1))
   */

  int (*open) (const GstTensorFilterProperties * prop, void **private_data);
  /**< Optional. tensor_filter.c will call this before any of other callbacks and will call once before calling close.
   *
   * Note: If 'open' callback is not defined, then the private_data passed in other callbacks will be NULL.
   *
   * @param[in] prop read-only property values
   * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer. Normally, open() allocates memory for private_data.
   * @return 0 if ok. < 0 if error.
   */

  void (*close) (const GstTensorFilterProperties * prop, void **private_data);
  /**< Optional. tensor_filter.c will not call other callbacks after calling close. Free-ing private_data is this function's responsibility. Set NULL after that.
   *
   * @param[in] prop read-only property values
   * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer. Normally, close() frees private_data and set NULL.
   */

  /**
   * @brief Distinct elements between two versions of the subplugin interfaces
   */
  union
  {
    /**
     * @brief Tensor_Filter Subplugin definition Version 0
     */
    struct /** _GstTensorFilterFramework_v0 */
    {
      char *name; /**< Name of the neural network framework, searchable by FRAMEWORK property */
      int allow_in_place; /**< TRUE(nonzero) if InPlace transfer of input-to-output is allowed. Not supported in main, yet */
      int allocate_in_invoke; /**< TRUE(nonzero) if invoke_NN is going to allocate outputptr by itself and return the address via outputptr. Do not change this value after cap negotiation is complete (or the stream has been started). */
      int run_without_model; /**< TRUE(nonzero) when the neural network framework does not need a model file. Tensor-filter will run invoke_NN without model. */
      int verify_model_path; /**< TRUE(nonzero) when the NNS framework, not the sub-plugin, should verify the path of model files. */

      int (*invoke_NN) (const GstTensorFilterProperties * prop, void **private_data,
          const GstTensorMemory * input, GstTensorMemory * output);
      /**< Mandatory callback. Invoke the given network model.
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[in] input The array of input tensors. Allocated and filled by tensor_filter/main
       * @param[out] output The array of output tensors. Allocated by tensor_filter/main and to be filled by invoke_NN. If allocate_in_invoke is TRUE, sub-plugin should allocate the memory block for output tensor. (data in GstTensorMemory)
       * @return 0 if OK. non-zero if error.
       */

      int (*getInputDimension) (const GstTensorFilterProperties * prop,
          void **private_data, GstTensorsInfo * info);
      /**< Optional. Set NULL if not supported. Get dimension of input tensor
       * If getInputDimension is NULL, setInputDimension must be defined.
       * If getInputDimension is defined, it is recommended to define getOutputDimension
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[out] info structure of tensor info (return value)
       * @return 0 if OK. non-zero if error.
       */

      int (*getOutputDimension) (const GstTensorFilterProperties * prop,
          void **private_data, GstTensorsInfo * info);
      /**< Optional. Set NULL if not supported. Get dimension of output tensor
       * If getInputDimension is NULL, setInputDimension must be defined.
       * If getInputDimension is defined, it is recommended to define getOutputDimension
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[out] info structure of tensor info (return value)
       * @return 0 if OK. non-zero if error.
       */

      int (*setInputDimension) (const GstTensorFilterProperties * prop,
          void **private_data, const GstTensorsInfo * in_info,
          GstTensorsInfo * out_info);
      /**< Optional. Set Null if not supported.Tensor_Filter::main will
       * configure input dimension from pad-cap in run-time for the sub-plugin.
       * Then, the sub-plugin is required to return corresponding output dimension
       * If this is NULL, both getInput/OutputDimension must be non-NULL.
       *
       * When you use this, do NOT allocate or fix internal data structure based on it
       * until invoke is called. Gstreamer may try different dimensions before
       * settling down.
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[in] in_info structure of input tensor info
       * @param[out] out_info structure of output tensor info (return value)
       * @return 0 if OK. non-zero if error.
       */

      void (*destroyNotify) (void **private_data, void * data);
      /**< Optional. tensor_filter.c will call it when 'allocate_in_invoke' flag of the framework is TRUE and allocateInInvoke also return enabled. Basically, it is called when the data element is destroyed. If it's set as NULL, g_free() will be used as a default. It will be helpful when the data pointer is included as an object of a nnfw. For instance, if the data pointer is removed when the object is gone, it occurs error. In this case, the objects should be maintained for a while first and destroyed when the data pointer is destroyed. Those kinds of logic could be defined at this method.
       *
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[in] data the data element.
       */

      int (*reloadModel) (const GstTensorFilterProperties * prop, void **private_data);
      /**< Optional. tensor_filter.c will call it when a model property is newly configured. Also, 'is-updatable' property of the framework should be TRUE. This function reloads a new model specified in the 'prop' argument. Note that it requires extra memory size enough to temporarily hold both old and new models during this function to hide the reload overhead.
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer. Normally, close() frees private_data and set NULL.
       * @return 0 if ok. < 0 if error.
       */

      int (*checkAvailability) (accl_hw hw);
      /**< Optional. Check if the provided hardware accelerator is supported. This check is static or dynamic based on framework support. Positive response of this check does not guarantee successful running of model with this accelerator. The static check can be performed without opening the framework.
       *
       * @param[in] hw backend accelerator hardware
       * @return 0 if supported. -errno if not supported.
       */

       int (*allocateInInvoke) (void **private_data);
       /**< Optional. tensor_filter.c will call it when allocate_in_invoke is set to TRUE. This check if the provided model for the framework supports allocation at invoke or not. If this is not defined, then the value of allocate_in_invoke is assumed to be final for all models.
        *
        * @param[in] private_data A subplugin may save its internal private data here.
        * @return 0 if supported. -errno if not supported.
        */
    }
#ifdef __NO_ANONYMOUS_NESTED_STRUCT
        v0
#endif
    ;

    /**
     * @brief Tensor_Filter Subplugin definition Version 1
     */
    struct /** _GstTensorFilterFramework_v1 */
    {
      int (*invoke) (const GstTensorFilterProperties * prop, void *private_data,
          const GstTensorMemory * input, GstTensorMemory * output);
      /**< Mandatory callback. Invoke the given network model.
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[in] input The array of input tensors. Allocated and filled by tensor_filter/main
       * @param[out] output The array of output tensors. Allocated by tensor_filter/main and to be filled by invoke_NN. If allocate_in_invoke is TRUE, sub-plugin should allocate the memory block for output tensor. (data in GstTensorMemory)
       * @return 0 if OK. non-zero if error.
       */

      int (*getFrameworkInfo) (const GstTensorFilterProperties * prop,
          void *private_data, GstTensorFilterFrameworkInfo *fw_info);
      /**< Mandatory callback. Get the frameworks statically determined info. Argument 'private_data' can be NULL. If provided 'private_data' is not NULL, then some info, such as 'allocate_in_invoke', can be updated based on the model being used (inferred from the 'private_data' provided). This updated info is useful for custom filter, as some custom filter's ability to support 'allocate_in_invoke' depends on the opened model.
       *
       * @param[in] prop read-only property values
       * @param[in] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer. This parameter can be NULL.
       * @param[out] fw_info struct to hold frameworks info. Must be allocated by the caller (return value).
       * @return 0 if OK. non-zero if error.
       *
       * @note CAUTION: private_data can be NULL if the framework is not yet opened by the caller.
       */

      int (*getModelInfo) (const GstTensorFilterProperties * prop,
          void *private_data, model_info_ops ops,
          GstTensorsInfo *in_info, GstTensorsInfo *out_info);
      /**< Mandatory callback. Gets the model related tensor info.
       * If ops == GET_IN_OUT_INFO, in_info would contain the input tensor info, and out_info would contain the output tensor info.
       * If ops == SET_INPUT_INFO, in_info would contain the provided input tensor info, and out_info would contain the updated output tensor info.
       * At least one of SET_INPUT_INFO and GET_IN_OUT_INFO operations must be supported.
       *
       * Note: Tensor_Filter::main will configure input dimension from pad-cap in
       * run-time for the sub-plugin. Then, the sub-plugin is required to return
       * corresponding output dimension, and will call SET_INPUT_INFO operation.
       *
       * Note: With SET_INPUT_INFO operation, the caller must NOT allocate or fix
       * internal data structure based on the return value until invoke is called.
       * Gstreamer may try different dimensions before settling down.
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[in] ops operation to be performed
       * @param[in/out] in_info structure of input tensor info
       * @param[out] out_info structure of output tensor info (return value)
       * @return 0 if OK. non-zero if error. -ENOENT is operation is not supported.
       */

      int (*eventHandler) (const GstTensorFilterProperties * prop,
          void *private_data, event_ops ops, GstTensorFilterFrameworkEventData * data);
      /**< Mandatory callback. Runs the event corresponding to the passed operation.
       * If ops == DESTROY_NOTIFY: tensor_filter.c will call it when 'allocate_in_invoke' property of the framework is TRUE. Basically, it is called when the data element is destroyed. If it's set as NULL, g_free() will be used as a default. It will be helpful when the data pointer is included as an object of a nnfw. For instance, if the data pointer is removed when the object is gone, it occurs error. In this case, the objects should be maintained for a while first and destroyed when the data pointer is destroyed. Those kinds of logic could be defined at this method.
       * If ops == RELOAD_MODEL: tensor_filter.c will call it when a model property is newly configured. Also, 'is-updatable' property of the framework should be TRUE. This function reloads a new model passed in as argument via data. Note that it requires extra memory size enough to temporarily hold both old and new models during this function to hide the reload overhead.
       * If ops == CUSTOM_PROP: tensor_filter will call to update the custom properties of the subplugin.
       * If ops == SET_INPUT_PROP: tensor_filter will call to update the property of the subplugin. This function will take tensor info and layout as the argument. This operation can update input tensor shape, type, name and layout.
       * If ops == SET_OUTPUT_PROP: tensor_filter will call to update the property of the subplugin. This function will take tensor info and layout as the argument. This operation can update output tensor shape, type, name and layout.
       * If ops == SET_ACCELERATOR: tensor_filter will call to update the property of the subplugin. This function will take accelerator list as the argument. This operation will update the backend to be used by the corresponding subplugin.
       * List of operations to be supported are optional.
       * Note: In these operations, the argument 'prop' will not contain the updated information, but will be updated after the corresponding operation is succeeded.
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[in] ops operation to be performed
       * @param[in/out] data user sata for the supported handlers (can be NULL)
       * @return 0 if OK. non-zero if error. -ENOENT if operation is not supported. -EINVAL if operation is supported but provided arguments are invalid.
       */
      void *subplugin_data; /**< A subplugin may store its own private global data here as well. Do not store data of each instance here, it is shared across all instances of a subplugin. */
    }
#ifdef __NO_ANONYMOUS_NESTED_STRUCT
        v1
#endif
    ;
  };
} GstTensorFilterFramework;

/* extern functions for subplugin management, exist in tensor_filter.c */
/**
 * @brief Filter's sub-plugin should call this function to register itself.
 * @param[in] tfsp Tensor-Filter Sub-Plugin to be registered.
 * @return TRUE if registered. FALSE is failed or duplicated.
 *
 * @note Do not change the subplugins callbacks after probing the filter.
 */
extern int
nnstreamer_filter_probe (GstTensorFilterFramework * tfsp);

/**
 * @brief Filter's sub-plugin may call this to unregister itself.
 * @param[in] name The name of filter sub-plugin.
 */
extern void
nnstreamer_filter_exit (const char *name);

/**
 * @brief Find filter sub-plugin with the name.
 * @param[in] name The name of filter sub-plugin.
 * @return NULL if not found or the sub-plugin object has an error.
 */
extern const GstTensorFilterFramework *
nnstreamer_filter_find (const char *name);

/**
 * @brief return accl_hw type from string
 */
extern accl_hw
get_accl_hw_type (const char * str);

/**
 * @brief return string based on accl_hw type
 */
extern const char *
get_accl_hw_str (const accl_hw key);

/**
 * @brief parse user given string to extract accelerator based on given regex
 */
extern accl_hw
parse_accl_hw (const char * accelerators, const char ** supported_accelerators);

#ifdef __cplusplus
}
#endif

#endif /* __NNS_PLUGIN_API_FILTER_H__ */
