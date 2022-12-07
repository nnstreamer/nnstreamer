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
#include <gst/base/gstbasetransform.h>
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
#define GST_TENSOR_TRAINER_CAST(obj)  ((GstTensorTrainer *)(obj))
typedef struct _GstTensorTrainer GstTensorTrainer;
typedef struct _GstTensorTrainerClass GstTensorTrainerClass;

/**
 * @brief GstTensorTrainer data structure
 */
struct _GstTensorTrainer
{
  GstBaseTransform element;

  gchar *fw_name;
  gchar *model_config;
  gchar *model_save_path;
  gchar *input_dimensions;
  gchar *output_dimensions;
  gchar *input_type;
  gchar *output_type;
  gboolean push_output;

  gboolean configured;
  int input_configured;
  int output_configured;
  int inputtype_configured;
  int outputtype_configured;
  unsigned int input_ranks[NNS_TENSOR_SIZE_LIMIT];
  unsigned int output_ranks[NNS_TENSOR_SIZE_LIMIT];
  GstTensorsInfo output_meta;

  /* draft */
  int fw_created;
  int fw_stop;

  void *privateData; /**< NNFW plugin's private data is stored here */
  const GstTensorTrainerFramework *fw;  /* for test, need to make */
  GstTensorTrainerProperties prop; /**< NNFW plugin's properties */
};

/**
 * @brief GstTensorTrainerClass data structure.
 */
struct _GstTensorTrainerClass
{
  GstBaseTransformClass parent_class;
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_trainer_get_type (void);

G_END_DECLS
#endif /* __GST_TENSOR_TRAINER_H__ */
