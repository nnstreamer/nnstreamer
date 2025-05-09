/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file	gsttensor_trainer.h
 * @date	20 October 2022
 * @brief	GStreamer plugin to train tensor data using NN Frameworks
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyunil Park <hyunil46.park@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_TRAINER_H__
#define __GST_TENSOR_TRAINER_H__


#include <gst/gst.h>
#include <tensor_typedef.h>
#include <tensor_common.h>

#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_plugin_api_trainer.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_TRAINER \
  (gst_tensor_trainer_get_type())
#define GST_TENSOR_TRAINER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_TRAINER,GstTensorTrainer))
#define GST_TENSOR_TRAINER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_TRAINER,GstTensorTrainerClass))
#define GST_IS_TENSOR_TRAINER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_TRAINER))
#define GST_IS_TENSOR_TRAINER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_TRAINER))

typedef struct _GstTensorTrainer GstTensorTrainer;
typedef struct _GstTensorTrainerClass GstTensorTrainerClass;

/**
 * @brief GstTensorTrainer data structure
 */
struct _GstTensorTrainer
{
  GstElement element; /**< parent object */

  GstPad *sinkpad;
  GstPad *srcpad;

  gchar *fw_name;

  gboolean fw_created;
  gboolean is_training_complete;
  gboolean is_epoch_complete;

  GstTensorMemory input_tensors[NNS_TENSOR_SIZE_LIMIT];
  GstTensorsInfo output_meta;
  GstTensorsConfig out_config;
  GstTensorsConfig in_config;

  guint required_sample;
  guint cur_epoch_data_cnt;      /**< number of total push data in one eposh */

  void *privateData; /**< NNFW plugin's private data is stored here */
  const GstTensorTrainerFramework *fw; /**< Subplugin definition */
  GstTensorTrainerProperties prop; /**< NNFW plugin's properties */
  GstTensorTrainerEventNotifier notifier; /**< Event notifier */

  GMutex training_completion_lock;
  GCond training_completion_cond;
  GMutex epoch_completion_lock;
  GCond epoch_completion_cond;

  GThread *dummy_data_thread; /**< Dummy data generation thread */
};

/**
 * @brief GstTensorTrainerClass data structure.
 */
struct _GstTensorTrainerClass
{
  GstElementClass parent_class; /**< parent class */
};

/**
 * @brief Function to get type of tensor_trainer.
 */
GType gst_tensor_trainer_get_type (void);

G_END_DECLS

#endif /* __GST_TENSOR_TRAINER_H__ */
