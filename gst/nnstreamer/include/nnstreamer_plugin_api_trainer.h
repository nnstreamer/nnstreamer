/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer API for Tensor_Trainer Sub-Plugins
 * Copyright (C) 2022 Hyunil Park <hyunil46.park@samsung.com>
 */
/**
 * @file  nnstreamer_plugin_api_trainer.h
 * @date  1 Dec 2022
 * @brief Mandatory APIs for NNStreamer Trainer sub-plugins (No External Dependencies)
 * @see https://github.com/nnstreamer/nnstreamer
 * @author  Hyunil Park <hyunil46.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __NNS_PLUGIN_API_TRAINER_H__
#define __NNS_PLUGIN_API_TRAINER_H__

#include "tensor_typedef.h"

#define GST_TENSOR_TRAINER_FRAMEWORK_BASE (0xDEAFDEAD00000000ULL)
#define GST_TENSOR_TRAINER_FRAMEWORK_V1 (GST_TENSOR_TRAINER_FRAMEWORK_BASE | 0x10000ULL)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief GstTensorTrainer's properties for neural network framework (internal data structure)
 *
 * Internal data of GstTensorTrainer required by tensor_trainer's custom subplugin.
 */
typedef struct _GstTensorTrainerProperties
{
  GstTensorsInfo input_meta;    /**< configured input tensor info */
  const char *model_config;    /**< The configuration file path for creating model */
  const char *model_save_path;    /**< The file path to save the model */
  int64_t num_inputs;    /**< The number of input lists, the input is where framework receive the features to train the model, num_inputs indicates how many inputs there are. */
  int64_t num_labels;    /**< The number of label lists, the label is where framework receive the class to train the model, num_labels indicates how many labels there are. */
  int64_t num_train_samples;    /**< The number of train sample used to train the model. */
  int64_t num_valid_samples;    /**< The number of valid sample used to train the model. */
  int64_t num_epochs;    /**< The number of repetition of total train and valid sample. subplugin must receive total samples((num_train_samples + num_valid_samples) * num_epochs) */

  GCond *train_complete_cond;    /**< Tensor trainer wait when receive EOS before model training is complete, subplugin should send signal when model train is complete. */
} GstTensorTrainerProperties;

/**
 * @brief Tensor_Trainer Subplugin framework related information
 *
 * All the information is provided statically.
 */
typedef struct _GstTensorTrainerFrameworkInfo
{
  const char *name;    /**< Name of the neural network framework, searchable by FRAMEWORK property. */
<<<<<<< HEAD
  int train_complete;  /**< Check if train is complete, Use int instead of gboolean because this is refered by custom plugins. */
  int64_t epoch_cnt;    /**< Number of currently completed epochs */
=======
  gboolean  train_complete;  /**< Check if train is complete */
<<<<<<< HEAD
>>>>>>> 56ee2c43 ([API][trainer] Add train_complete to GstTensorTrainerFrameworkInfo)
=======
  int64_t epoch_cnt;    /**< Number of currently completed epochs */
>>>>>>> 5ea682a9 ([API][trainer] Add number of epoch and epoch count)
} GstTensorTrainerFrameworkInfo;

typedef struct _GstTensorTrainerFramework GstTensorTrainerFramework;

/**
 * @brief tensor_trainer Subplugin definition
 *
 * Common callback parameters:
 * prop Trainer properties. Read Only.
 * private_data Subplugin's private data. Set this (*private_data = XXX) if you want to change trainer->private_data.
 */
struct _GstTensorTrainerFramework
{
  uint64_t version;
  /**< Version of the struct
   * | 32bit (validity check) | 16bit (API version) | 16bit (Subplugin's internal version) |
   */

  int (*create) (const GstTensorTrainerFramework * self,
      const GstTensorTrainerProperties * prop, void **private_data);
  /**< tensor_trainer call this to create the model
   * @param[in] prop read-only property values
   * @param[in/out] private_data, a subplugin may save its internal private data here.
   * @return 0 if ok. < 0 if error.
   */

  int (*destroy) (const GstTensorTrainerFramework * self,
      const GstTensorTrainerProperties * prop, void **private_data);
  /**< tensor_trainer call this to destroy the model, Set NULL after that.
   * @param[in] prop read-only property values
   * @param[in/out] private_data, a subplugin may save its internal private data here.
   * @return 0 if ok. < 0 if error.
   */

  int (*start) (const GstTensorTrainerFramework * self,
      const GstTensorTrainerProperties * prop, void *private_data);
  /**< tensor_trainer call this to start training the model
   * @param[in] prop read-only property values
   * @param[in] private_data, a subplugin may save its internal private data here.
   * @return 0 if ok. < 0 if error.
   */

  int (*push_data) (const GstTensorTrainerFramework * self,
      const GstTensorTrainerProperties * prop,
      void *private_data, const GstTensorMemory * input);
  /**< tensor_trainer call this to push tensor data to subplugin, subplugin constructs a data set using input.
   * @param[in] prop read-only property values
   * @param[in] private_data, a subplugin may save its internal private data here.
   * @param[in] input The array of input tensors. Allocated and filled by tensor_trainer
   * @return 0 if ok. < 0 if error.
   */

  int (*getFrameworkInfo) (const GstTensorTrainerFramework * self,
      const GstTensorTrainerProperties * prop, void *private_data,
      GstTensorTrainerFrameworkInfo *fw_info);
  /**< Mandatory callback. Get the frameworks statically determined info.
   * @param[in] prop read-only property values
   * @param[in] private_data A subplugin may save its internal private data here.
   * @param[out] fw_info struct to hold frameworks info. Must be allocated by the caller (return value).
   * @return 0 if OK. non-zero if error.
   *
   * @note CAUTION: private_data can be NULL if the framework is not yet opened by the caller.
   */

  /* Need to make (*eventHandler)*/

};

/* extern functions for subplugin management, exist in tensor_trainer.c */
/**
 * @brief Trainer's sub-plugin should call this function to register itself.
 * @param[in] ttsp tensor_trainer sub-plugin to be registered.
 * @return TRUE if registered. FALSE is failed.
 *
 * @note Do not change the subplugins callbacks after probing the filter.
 */
extern int
nnstreamer_trainer_probe (GstTensorTrainerFramework * ttsp);

/**
 * @brief Trainer's sub-plugin may call this to unregister itself.
 * @param[in] ttsp tensor_trainer sub-plugin to be unregistered.
 * @return TRUE if unregistered. FALSE is failed.
 */

extern int
nnstreamer_trainer_exit (GstTensorTrainerFramework * ttsp);

#ifdef __cplusplus
}
#endif

#endif /* __NNS_PLUGIN_API_TRAINER_H__ */
