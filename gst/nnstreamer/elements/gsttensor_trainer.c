/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file	gsttensor_trainer.c
 * @date	20 October 2022
 * @brief	GStreamer plugin to train tensor data using NN Frameworks
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyunil Park <hyunil46.park@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * ## Example launch line
 * |[
 * gst-launch-1.0 datareposrc location=mnist_trainingSet.dat json=mnist.json start-sample-index=3 stop-sample-index=202 epochs=5 ! \
 * tensor_trainer framework=nntrainer model-config=mnist.ini model-save-path=model.bin \
 * num-inputs=1 num-labels=1 num-training-samples=100 num-validation-samples=100 epochs=5 ! \
 * tensor_sink
 * ]|
 *
 * Total number of data to be received is 1000((num-training-samples + num-validation-samples) * epochs)
 * 
 * output tensors : dimensions=1:1:4, types=float64.
 * values are training loss, training accuracy, validation loss and validation accuracy.
 * -INFINITY value is stored if the value fetched from the sub-plugin is not greater than 0.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <stdlib.h>
#include <nnstreamer_subplugin.h>
#include <nnstreamer_util.h>
#include "gsttensor_trainer.h"
#include <unistd.h>
#include <math.h>

/**
 * @brief Default caps string for both sink and source pad.
 */
#define CAPS_STRING GST_TENSORS_CAP_MAKE ("{ static, flexible }")

/**
 * @brief The capabilities of the sink pad
 */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

/**
 * @brief The capabilities of the src pad
 */
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

GST_DEBUG_CATEGORY_STATIC (gst_tensor_trainer_debug);
#define GST_CAT_DEFAULT gst_tensor_trainer_debug
#define gst_tensor_trainer_parent_class parent_class
G_DEFINE_TYPE (GstTensorTrainer, gst_tensor_trainer, GST_TYPE_ELEMENT);

/**
 * @brief Statistical from the model being trained
 * An enum value indicates the value stored at the index of the output tensor.
 */
enum
{
  TRAINING_LOSS,
  TRAINING_ACCURACY,
  VALIDATION_LOSS,
  VALIDATION_ACCURACY
};
#define MODEL_STATS_SIZE 4

/**
 * @brief Default framework property value
 */
#define DEFAULT_PROP_INPUT_LIST 1
#define DEFAULT_PROP_LABEL_LIST 1
#define DEFAULT_PROP_TRAIN_SAMPLES 0
#define DEFAULT_PROP_VALID_SAMPLES 0
#define DEFAULT_PROP_EPOCHS 1
/**
 * @brief Default string property value
 */
#define DEFAULT_STR_PROP_VALUE ""

/**
 * @brief tensor_trainer properties
 */
enum
{
  PROP_0,
  PROP_FRAMEWORK,
  PROP_MODEL_CONFIG,
  PROP_MODEL_SAVE_PATH,
  PROP_NUM_INPUTS,              /* number of input list */
  PROP_NUM_LABELS,              /* number of label list */
  PROP_NUM_TRAINING_SAMPLES,    /* number of training data */
  PROP_NUM_VALIDATION_SAMPLES,  /* number of validation data */
  PROP_EPOCHS,                  /* Repetitions of training */
};

#define TRAINER_TENSOR_SIZE_LIMT ((NNS_TENSOR_SIZE_LIMIT) + (NNS_TENSOR_SIZE_EXTRA_LIMIT))

static void gst_tensor_trainer_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_trainer_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_trainer_finalize (GObject * object);
static gboolean gst_tensor_trainer_sink_event (GstPad * sinkpad,
    GstObject * parent, GstEvent * event);
static gboolean gst_tensor_trainer_sink_query (GstPad * sinkpad,
    GstObject * parent, GstQuery * query);
static gboolean gst_tensor_trainer_src_query (GstPad * srcpad,
    GstObject * parent, GstQuery * query);
static GstFlowReturn gst_tensor_trainer_chain (GstPad * sinkpad,
    GstObject * parent, GstBuffer * inbuf);
static GstCaps *gst_tensor_trainer_query_caps (GstTensorTrainer * trainer,
    GstPad * pad, GstCaps * filter);
static GstStateChangeReturn gst_tensor_trainer_change_state (GstElement *
    element, GstStateChange transition);

static void gst_tensor_trainer_set_prop_framework (GstTensorTrainer * trainer,
    const GValue * value);
static void gst_tensor_trainer_set_prop_model_config_file_path (GstTensorTrainer
    * trainer, const GValue * value);
static void gst_tensor_trainer_set_model_save_path (GstTensorTrainer * trainer,
    const GValue * value);
static gboolean gst_tensor_trainer_find_framework (GstTensorTrainer * trainer,
    const char *name);
static gboolean gst_tensor_trainer_create_framework (GstTensorTrainer *
    trainer);
static gsize gst_tensor_trainer_get_tensor_size (GstTensorTrainer * trainer,
    guint index, gboolean is_input);
static gboolean gst_tensor_trainer_create_model (GstTensorTrainer * trainer);
static void gst_tensor_trainer_create_event_notifier (GstTensorTrainer *
    trainer);
static void gst_tensor_trainer_train_model (GstTensorTrainer * trainer);
static void gst_tensor_trainer_output_dimension (GstTensorTrainer * trainer);
static void gst_tensor_trainer_output_type (GstTensorTrainer * trainer);

/**
 * @brief initialize the tensor_trainer's class
 */
static void
gst_tensor_trainer_class_init (GstTensorTrainerClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, "tensor_trainer", 0,
      "Tensor trainer to train neural network model");

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_tensor_trainer_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_tensor_trainer_get_property);
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_tensor_trainer_finalize);

  /* Called when the element's state changes */
  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_trainer_change_state);

  /* Install properties for tensor_trainer */
  g_object_class_install_property (gobject_class, PROP_FRAMEWORK,
      g_param_spec_string ("framework", "Framework",
          "Neural network framework to be used for model training",
          DEFAULT_STR_PROP_VALUE,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODEL_CONFIG,
      g_param_spec_string ("model-config", "Model configuration file path",
          "Model configuration file is used to configure the model "
          "to be trained in neural network framework, set the file path",
          DEFAULT_STR_PROP_VALUE,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODEL_SAVE_PATH,
      g_param_spec_string ("model-save-path", "Model save path",
          "Path to save the trained model in framework, if model-config "
          "contains information about the save file, it is ignored",
          DEFAULT_STR_PROP_VALUE,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_NUM_INPUTS,
      g_param_spec_uint ("num-inputs", "Number of inputs",
          "An input in a tensor can have one or more features data,"
          "set how many inputs are received", 0, NNS_TENSOR_SIZE_LIMIT, 1,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_NUM_LABELS,
      g_param_spec_uint ("num-labels", "Number of labels",
          "A label in a tensor can have one or more classes data,"
          "set how many labels are received", 0, NNS_TENSOR_SIZE_LIMIT, 1,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_NUM_TRAINING_SAMPLES,
      g_param_spec_uint ("num-training-samples", "Number of training samples",
          "A sample can consist of multiple inputs and labels in tensors of a gstbuffer"
          ", set how many samples are taken for training model",
          0, G_MAXINT, 0,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_NUM_VALIDATION_SAMPLES,
      g_param_spec_uint ("num-validation-samples",
          "Number of validation samples",
          "A sample can consist of multiple inputs and labels in tensors of a gstbuffer"
          ", set how many samples are taken for validation model",
          0, G_MAXINT, 0,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_EPOCHS,
      g_param_spec_uint ("epochs", "Number of epoch",
          "Epochs are repetitions of training samples and validation smaples, "
          "number of samples received for model training is "
          "(num-training-samples+num-validation-samples)*epochs", 0, G_MAXINT,
          DEFAULT_PROP_EPOCHS,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  gst_element_class_set_details_simple (gstelement_class, "TensorTrainer",
      "Trainer/Tensor", "Train tensor data using NN Frameworks",
      "Samsung Electronics Co., Ltd.");

  /* Add pad template */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_template));
}

/**
 * @brief Initialize tensor_trainer.
 */
static void
gst_tensor_trainer_init (GstTensorTrainer * trainer)
{
  GST_DEBUG ("<ENTER>");
  /** setup sink pad */
  trainer->sinkpad = gst_pad_new_from_static_template (&sink_template, "sink");
  gst_pad_set_event_function (trainer->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_trainer_sink_event));
  gst_pad_set_query_function (trainer->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_trainer_sink_query));
  gst_pad_set_chain_function (trainer->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_trainer_chain));
  GST_PAD_SET_PROXY_CAPS (trainer->sinkpad);
  gst_element_add_pad (GST_ELEMENT (trainer), trainer->sinkpad);

  /** setup src pad */
  trainer->srcpad = gst_pad_new_from_static_template (&src_template, "src");
  gst_pad_set_query_function (trainer->srcpad,
      GST_DEBUG_FUNCPTR (gst_tensor_trainer_src_query));
  GST_PAD_SET_PROXY_CAPS (trainer->srcpad);
  gst_element_add_pad (GST_ELEMENT (trainer), trainer->srcpad);

  /** init properties */
  trainer->fw_name = g_strdup (DEFAULT_STR_PROP_VALUE);
  trainer->model_config = g_strdup (DEFAULT_STR_PROP_VALUE);
  trainer->model_save_path = g_strdup (DEFAULT_STR_PROP_VALUE);
  trainer->output_dimensions = g_strdup (DEFAULT_STR_PROP_VALUE);
  trainer->output_type = g_strdup (DEFAULT_STR_PROP_VALUE);
  trainer->prop.num_inputs = DEFAULT_PROP_INPUT_LIST;
  trainer->prop.num_labels = DEFAULT_PROP_LABEL_LIST;
  trainer->prop.num_training_samples = DEFAULT_PROP_TRAIN_SAMPLES;
  trainer->prop.num_validation_samples = DEFAULT_PROP_VALID_SAMPLES;
  trainer->prop.num_epochs = DEFAULT_PROP_EPOCHS;

  trainer->fw = NULL;
  trainer->fw_created = FALSE;
  trainer->input_configured = FALSE;
  trainer->output_configured = FALSE;
  trainer->inputtype_configured = FALSE;
  trainer->is_training_complete = FALSE;
  trainer->is_epoch_complete = FALSE;
  trainer->total_push_data_cnt = 0;

  gst_tensors_config_init (&trainer->in_config);
  gst_tensors_config_init (&trainer->out_config);

  g_cond_init (&trainer->training_completion_cond);
  g_mutex_init (&trainer->training_completion_lock);
  g_cond_init (&trainer->epoch_completion_cond);
  g_mutex_init (&trainer->epoch_completion_lock);

  gst_tensor_trainer_output_dimension (trainer);
  gst_tensor_trainer_output_type (trainer);
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_tensor_trainer_finalize (GObject * object)
{
  GstTensorTrainer *trainer;

  trainer = GST_TENSOR_TRAINER (object);

  g_free (trainer->fw_name);
  g_free (trainer->model_config);
  g_free (trainer->model_save_path);
  g_free (trainer->output_dimensions);
  g_free (trainer->output_type);

  gst_tensors_config_free (&trainer->in_config);
  gst_tensors_config_free (&trainer->out_config);

  g_cond_clear (&trainer->training_completion_cond);
  g_mutex_clear (&trainer->training_completion_lock);
  g_cond_clear (&trainer->epoch_completion_cond);
  g_mutex_clear (&trainer->epoch_completion_lock);

  if (trainer->fw_created && trainer->fw) {
    trainer->fw->destroy (trainer->fw, &trainer->prop, &trainer->privateData);
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_trainsink properties.
 */
static void
gst_tensor_trainer_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorTrainer *trainer;

  trainer = GST_TENSOR_TRAINER (object);

  switch (prop_id) {
    case PROP_FRAMEWORK:
      gst_tensor_trainer_set_prop_framework (trainer, value);
      break;
    case PROP_MODEL_CONFIG:
      gst_tensor_trainer_set_prop_model_config_file_path (trainer, value);
      break;
    case PROP_MODEL_SAVE_PATH:
      gst_tensor_trainer_set_model_save_path (trainer, value);
      break;
    case PROP_NUM_INPUTS:
      trainer->prop.num_inputs = g_value_get_uint (value);
      break;
    case PROP_NUM_LABELS:
      trainer->prop.num_labels = g_value_get_uint (value);
      break;
    case PROP_NUM_TRAINING_SAMPLES:
      trainer->prop.num_training_samples = g_value_get_uint (value);
      break;
    case PROP_NUM_VALIDATION_SAMPLES:
      trainer->prop.num_validation_samples = g_value_get_uint (value);
      break;
    case PROP_EPOCHS:
      trainer->prop.num_epochs = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter tensor_trainsink properties.
 */
static void
gst_tensor_trainer_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorTrainer *trainer;

  trainer = GST_TENSOR_TRAINER (object);

  switch (prop_id) {
    case PROP_FRAMEWORK:
      g_value_set_string (value, trainer->fw_name);
      break;
    case PROP_MODEL_CONFIG:
      g_value_set_string (value, trainer->model_config);
      break;
    case PROP_MODEL_SAVE_PATH:
      g_value_set_string (value, trainer->model_save_path);
      break;
    case PROP_NUM_INPUTS:
      g_value_set_uint (value, trainer->prop.num_inputs);
      break;
    case PROP_NUM_LABELS:
      g_value_set_uint (value, trainer->prop.num_labels);
      break;
    case PROP_NUM_TRAINING_SAMPLES:
      g_value_set_uint (value, trainer->prop.num_training_samples);
      break;
    case PROP_NUM_VALIDATION_SAMPLES:
      g_value_set_uint (value, trainer->prop.num_validation_samples);
      break;
    case PROP_EPOCHS:
      g_value_set_uint (value, trainer->prop.num_epochs);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Check invalid param
 */
static gboolean
gst_tensor_trainer_check_invalid_param (GstTensorTrainer * trainer)
{
  g_return_val_if_fail (trainer != NULL, FALSE);

  /* Parameters that can be retrieved from caps will be removed */
  if (!trainer->fw_name || !trainer->model_config || !trainer->model_save_path
      || trainer->prop.num_epochs <= 0 || trainer->prop.num_inputs <= 0
      || trainer->prop.num_labels <= 0) {
    GST_ERROR_OBJECT (trainer, "Check for invalid param value");

    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Change state of tensor_trainsink.
 */
static GstStateChangeReturn
gst_tensor_trainer_change_state (GstElement * element,
    GstStateChange transition)
{
  GstTensorTrainer *trainer = GST_TENSOR_TRAINER (element);
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;

  switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
      GST_INFO_OBJECT (trainer, "NULL_TO_READY");

      if (!gst_tensor_trainer_check_invalid_param (trainer))
        goto state_change_failed;
      break;

    case GST_STATE_CHANGE_READY_TO_PAUSED:
      GST_INFO_OBJECT (trainer, "READY_TO_PAUSED");
      break;

    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      GST_INFO_OBJECT (trainer, "PAUSED_TO_PLAYING");
      if (!gst_tensor_trainer_create_model (trainer))
        goto state_change_failed;

      gst_tensor_trainer_create_event_notifier (trainer);
      gst_tensor_trainer_train_model (trainer);
      break;

    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      GST_INFO_OBJECT (trainer, "PLAYING_TO_PAUSED");
      break;

    case GST_STATE_CHANGE_PAUSED_TO_READY:
      GST_INFO_OBJECT (trainer, "PAUSED_TO_READY");
      /* stop model train ? */
      break;

    case GST_STATE_CHANGE_READY_TO_NULL:
      GST_INFO_OBJECT (trainer, "READY_TO_NULL");
      /* destroy or reset model ? */
      break;

    default:
      break;
  }

  return ret;

state_change_failed:
  GST_ERROR_OBJECT (trainer, "state change failed");

  return GST_STATE_CHANGE_FAILURE;
}

/**
 * @brief Wait for epoch eompletion
 */
static void
gst_tensor_trainer_wait_for_epoch_completion (GstTensorTrainer * trainer)
{
  g_return_if_fail (trainer != NULL);

  g_mutex_lock (&trainer->epoch_completion_lock);
  if (trainer->is_epoch_complete) {
    /* It's already completed */
    trainer->is_epoch_complete = FALSE;
    g_mutex_unlock (&trainer->epoch_completion_lock);
    return;
  }

  GST_INFO_OBJECT (trainer, "wait for epoch_completion_cond signal");
  g_cond_wait (&trainer->epoch_completion_cond,
      &trainer->epoch_completion_lock);
  trainer->is_epoch_complete = FALSE;
  g_mutex_unlock (&trainer->epoch_completion_lock);
}

/**
 * @brief Check if current epochs is complete, 
 * tensor_trainer wait for one of epochs to complete before getting the results from the subplugin
 */
static gboolean
gst_tensor_trainer_epochs_is_complete (GstTensorTrainer * trainer)
{
  int required_sample;

  g_return_val_if_fail (trainer != NULL, FALSE);
  g_return_val_if_fail (trainer->fw != NULL, FALSE);
  g_return_val_if_fail (&trainer->prop != NULL, FALSE);

  required_sample =
      trainer->prop.num_training_samples + trainer->prop.num_validation_samples;
  if (trainer->total_push_data_cnt % required_sample != 0)
    return FALSE;

  gst_tensor_trainer_wait_for_epoch_completion (trainer);

  return TRUE;
}

/**
 * @brief Chain function, this function does the actual processing.
 */
static GstFlowReturn
gst_tensor_trainer_chain (GstPad * sinkpad, GstObject * parent,
    GstBuffer * inbuf)
{
  GstTensorTrainer *trainer;
  GstBuffer *outbuf = NULL;
  gint ret = -1;
  guint num_tensors, i;
  gsize header_size, expected;
  gboolean in_flexible, out_flexible;
  GstMemory *in_mem[TRAINER_TENSOR_SIZE_LIMT] = { 0, };
  GstMapInfo in_info[TRAINER_TENSOR_SIZE_LIMT];
  GstMemory *out_mem[TRAINER_TENSOR_SIZE_LIMT] = { 0, };
  GstMapInfo out_info[TRAINER_TENSOR_SIZE_LIMT];
  GstTensorMemory in_tensors[TRAINER_TENSOR_SIZE_LIMT];
  GstTensorMemory push_tensors[TRAINER_TENSOR_SIZE_LIMT];
  GstTensorMemory out_tensors[TRAINER_TENSOR_SIZE_LIMT];
  GstTensorMetaInfo in_meta[TRAINER_TENSOR_SIZE_LIMT];
  GstTensorMetaInfo out_meta[TRAINER_TENSOR_SIZE_LIMT];
  GstTensorInfo *info = NULL;

  double model_stats[MODEL_STATS_SIZE] =
      { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
  void *ptr;

  trainer = GST_TENSOR_TRAINER (parent);
  in_flexible = gst_tensor_pad_caps_is_flexible (sinkpad);
  num_tensors = gst_tensor_buffer_get_count (inbuf);

  if (num_tensors >= TRAINER_TENSOR_SIZE_LIMT)
    return GST_FLOW_ERROR;

  for (i = 0; i < num_tensors; i++) {
    in_mem[i] = gst_tensor_buffer_get_nth_memory (inbuf, i);
    if (!gst_memory_map (in_mem[i], &in_info[i], GST_MAP_READ)) {
      GST_ERROR_OBJECT (trainer, "Could not map in_mem[%u] GstMemory", i);
      goto error;
    }

    /* Get header size */
    header_size = 0;
    if (in_flexible) {
      if (!gst_tensor_meta_info_parse_header (&in_meta[i], in_info[i].data)) {
        GST_ERROR_OBJECT (trainer, "Invalid Flexible tensors");
        goto error;
      }

      header_size = gst_tensor_meta_info_get_header_size (&in_meta[i]);
      info = gst_tensors_info_get_nth_info (&trainer->prop.input_meta, i);
      if (info == NULL)
        goto error;
      gst_tensor_meta_info_convert (&in_meta[i], info);
      GST_INFO ("flexible header size:%zd", header_size);
    } else {
      GST_INFO ("not flexible header, size:%zd", header_size);
    }

    in_tensors[i].data = in_info[i].data + header_size;
    in_tensors[i].size = in_info[i].size - header_size;
  }

  /* Prepare tensor to push */
  if (in_flexible)
    trainer->prop.input_meta.num_tensors = num_tensors;
  else {
    /* Check number of input tensors */
    GST_DEBUG_OBJECT (trainer, "num_tensors: %u",
        trainer->prop.input_meta.num_tensors);
    if (num_tensors != trainer->prop.input_meta.num_tensors) {
      GST_ERROR_OBJECT (trainer, "Invalid memory blocks(%u),"
          "number of input tensors may be (%u)", num_tensors,
          trainer->prop.input_meta.num_tensors);
      goto error;
    }
  }

  /* Check size of input tensors */
  for (i = 0; i < trainer->prop.input_meta.num_tensors; i++) {
    expected = gst_tensor_trainer_get_tensor_size (trainer, i, TRUE);
    if (expected != in_tensors[i].size) {
      GST_ERROR_OBJECT (trainer,
          "Invalid tensor size (%u'th memory chunk: %zd)"
          ", expected size (%zd)", i, in_tensors[i].size, expected);
      goto error;
    }
    /* Copy to data pointer */
    push_tensors[i] = in_tensors[i];
    GST_INFO ("in_tensors[%u].size= %zd", i, in_tensors[i].size);
    GST_INFO ("in_tensors[%u].data: %p", i, in_tensors[i].data);
    GST_INFO ("push_tensors[%u].size= %zd", i, push_tensors[i].size);
    GST_INFO ("push_tensors[%u].data: %p", i, push_tensors[i].data);
  }

  ret =
      trainer->fw->push_data (trainer->fw, &trainer->prop, trainer->privateData,
      push_tensors);
  trainer->total_push_data_cnt++;

  /* Free in info */
  for (i = 0; i < num_tensors; i++) {
    gst_memory_unmap (in_mem[i], &in_info[i]);
    gst_memory_unref (in_mem[i]);
  }

  if (ret < 0) {
    GST_ERROR_OBJECT (trainer, "push error");
    return GST_FLOW_ERROR;
  }

  /** Update result if one of epochs is complete,
      push one outbuf is necessary to change pipeline state.
      Scheduling with subplugin does not work.
   */
  if (trainer->total_push_data_cnt == 1
      || gst_tensor_trainer_epochs_is_complete (trainer)) {

    /* Prepare output tensor */
    for (i = 0; i < trainer->output_meta.num_tensors; i++) {
      out_tensors[i].data = NULL;
      out_tensors[i].size =
          gst_tensor_trainer_get_tensor_size (trainer, i, FALSE);

      /* Get header size */
      header_size = 0;
      out_flexible = gst_tensor_pad_caps_is_flexible (trainer->srcpad);
      if (out_flexible) {
        gst_tensor_info_convert_to_meta (&trainer->output_meta.info[i],
            &out_meta[i]);
        header_size = gst_tensor_meta_info_get_header_size (&out_meta[i]);
        GST_INFO ("flexible header size:%zd", header_size);
      } else {
        GST_INFO ("not flexible header size:%zd", header_size);
      }

      out_mem[i] =
          gst_allocator_alloc (NULL, out_tensors[i].size + header_size, NULL);
      if (!out_mem[i]) {
        GST_ERROR_OBJECT (trainer, "Failed to allocate memory");
        goto error;
      }

      if (!gst_memory_map (out_mem[i], &out_info[i], GST_MAP_WRITE)) {
        GST_ERROR_OBJECT (trainer, "Could not map in_mem[%u] GstMemory", i);
        goto error;
      }

      out_tensors[i].data = out_info[i].data + header_size;

      /* Append header */
      if (out_flexible) {
        if (!gst_tensor_meta_info_update_header (&out_meta[i],
                out_info[i].data)) {
          GST_ERROR_OBJECT (trainer, "Failed to update header ");
          goto error;
        }
      }

      ret =
          trainer->fw->getStatus (trainer->fw, &trainer->prop,
          trainer->privateData);
      if (ret < 0) {
        GST_ERROR_OBJECT (trainer, "Failed to Get status from sub-plugin.(%s).",
            trainer->fw_name);
        return GST_FLOW_ERROR;
      }
      /* If the value is invalid, it is already set by -INFINITY. */
      if (trainer->prop.training_loss > 0)
        model_stats[TRAINING_LOSS] = trainer->prop.training_loss;
      if (trainer->prop.training_accuracy > 0)
        model_stats[TRAINING_ACCURACY] = trainer->prop.training_accuracy;
      if (trainer->prop.validation_loss > 0)
        model_stats[VALIDATION_LOSS] = trainer->prop.validation_loss;
      if (trainer->prop.validation_accuracy > 0)
        model_stats[VALIDATION_ACCURACY] = trainer->prop.validation_accuracy;

      GST_DEBUG_OBJECT (trainer,
          "#%u/%u epochs [training_loss: %f, training_accuracy: %f, validation_loss: %f, validation_accuracy: %f]",
          trainer->prop.epoch_count, trainer->prop.num_epochs,
          model_stats[TRAINING_LOSS], model_stats[TRAINING_ACCURACY],
          model_stats[VALIDATION_LOSS], model_stats[VALIDATION_ACCURACY]);

      /* updatd out_tensors */
      /* write training loss, training accuracy, validation loss, validation accuracy */
      ptr = out_info[i].data;
      memcpy (ptr, model_stats, sizeof (model_stats));
    }

    /* Free out info */
    for (i = 0; i < trainer->output_meta.num_tensors; i++) {
      if (out_mem[i])
        gst_memory_unmap (out_mem[i], &out_info[i]);
    }

    outbuf = gst_buffer_new ();
    for (i = 0; i < trainer->output_meta.num_tensors; i++) {
      /* append the memory block to outbuf */
      gst_buffer_append_memory (outbuf, out_mem[i]);
    }
    GST_INFO ("out_buffer size : %zd", gst_buffer_get_size (outbuf));

    gst_pad_push (trainer->srcpad, outbuf);
  }

  gst_buffer_unref (inbuf);

  return GST_FLOW_OK;

error:
  for (i = 0; i < num_tensors; i++) {
    if (in_mem[i]) {
      gst_memory_unmap (in_mem[i], &in_info[i]);
      gst_memory_unref (in_mem[i]);
    }
  }

  for (i = 0; i < trainer->output_meta.num_tensors; i++) {
    if (out_mem[i]) {
      gst_memory_unmap (out_mem[i], &out_info[i]);
      gst_allocator_free (out_mem[i]->allocator, out_mem[i]);
    }
  }

  return GST_FLOW_ERROR;
}

/**
 * @brief Get pad caps for caps negotiation.
 */
static GstCaps *
gst_tensor_trainer_query_caps (GstTensorTrainer * trainer,
    GstPad * pad, GstCaps * filter)
{
  GstCaps *caps;
  GstTensorsConfig *config;

  g_return_val_if_fail (trainer != NULL, NULL);
  g_return_val_if_fail (pad != NULL, NULL);

  /* tensor config info for given pad */
  if (pad == trainer->sinkpad) {
    config = &trainer->in_config;
  } else {
    config = &trainer->out_config;
  }

  caps = gst_tensor_pad_possible_caps_from_config (pad, config);
  GST_DEBUG_OBJECT (trainer, "caps %" GST_PTR_FORMAT, caps);
  GST_DEBUG_OBJECT (trainer, "filter %" GST_PTR_FORMAT, filter);

  if (caps && filter) {
    GstCaps *result;
    result = gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (caps);
    caps = result;
  }

  GST_DEBUG_OBJECT (trainer, "result caps %" GST_PTR_FORMAT, caps);

  return caps;
}

/**
 * @brief Wait for training completion
 */
static gboolean
gst_tensor_trainer_wait_for_training_completion (GstTensorTrainer * trainer)
{
  g_return_val_if_fail (trainer != NULL, FALSE);

  g_mutex_lock (&trainer->training_completion_lock);
  if (trainer->is_training_complete) {
    /* It's already completed */
    trainer->is_training_complete = FALSE;
    g_mutex_unlock (&trainer->training_completion_lock);
    return TRUE;
  }

  GST_INFO_OBJECT (trainer,
      "got GST_EVENT_EOS event but training is not completed, state is %d",
      GST_STATE (trainer));

  GST_INFO_OBJECT (trainer, "wait for training_completion_cond signal");
  g_cond_wait (&trainer->training_completion_cond,
      &trainer->training_completion_lock);
  trainer->is_training_complete = FALSE;
  g_mutex_unlock (&trainer->training_completion_lock);

  return TRUE;
}

/**
 * @brief Event handler for sink pad of tensor_trainer
 */
static gboolean
gst_tensor_trainer_sink_event (GstPad * sinkpad, GstObject * parent,
    GstEvent * event)
{
  GstTensorTrainer *trainer;
  trainer = GST_TENSOR_TRAINER (parent);

  GST_DEBUG_OBJECT (trainer, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_EOS:
      if (gst_tensor_trainer_wait_for_training_completion (trainer))
        GST_DEBUG_OBJECT (trainer, "training is completed in sub-plugin[%s]",
            trainer->fw_name);
      break;
    case GST_EVENT_FLUSH_START:
      GST_INFO_OBJECT (trainer, "get GST_EVENT_FLUSH_START event");
      break;
    case GST_EVENT_FLUSH_STOP:
      GST_INFO_OBJECT (trainer, "get GST_EVENT_FLUSH_STOP event");
      break;
    case GST_EVENT_CAPS:
    {
      GstCaps *in_caps;
      GstCaps *out_caps;
      GstStructure *structure;
      GstTensorsConfig config;
      gboolean ret = FALSE;

      gst_event_parse_caps (event, &in_caps);
      GST_INFO_OBJECT (trainer, "[in-caps] : %" GST_PTR_FORMAT, in_caps);

      structure = gst_caps_get_structure (in_caps, 0);
      if (!gst_tensors_config_from_structure (&config, structure) ||
          !gst_tensors_config_validate (&config)) {
        gst_tensors_config_free (&config);
        gst_event_unref (event);
        return FALSE;
      }

      /* copy TensorsInfo from negotiated caps to GstTensorTrainerProperties's input_meta */
      gst_tensors_info_copy (&trainer->prop.input_meta, &config.info);

      /* set tensor-config and out caps */
      trainer->in_config = config;
      trainer->out_config.rate_n = config.rate_n;
      trainer->out_config.rate_d = config.rate_d;
      gst_tensors_info_copy (&trainer->out_config.info, &trainer->output_meta);

      out_caps =
          gst_tensor_pad_caps_from_config (trainer->srcpad,
          &trainer->out_config);
      GST_INFO_OBJECT (trainer, "[out-caps] : %" GST_PTR_FORMAT, out_caps);

      ret = gst_pad_set_caps (trainer->srcpad, out_caps);

      gst_event_unref (event);
      gst_caps_unref (out_caps);
      return ret;
    }
    default:
      break;
  }

  return gst_pad_event_default (sinkpad, parent, event);
}

/**
 * @brief This function handles sink pad query.
 */
static gboolean
gst_tensor_trainer_sink_query (GstPad * sinkpad, GstObject * parent,
    GstQuery * query)
{
  GstTensorTrainer *trainer;
  trainer = GST_TENSOR_TRAINER (parent);

  GST_DEBUG_OBJECT (trainer, "Received '%s' query: %" GST_PTR_FORMAT,
      GST_QUERY_TYPE_NAME (query), query);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CAPS:
    {
      GstCaps *caps;
      GstCaps *filter;

      GST_DEBUG_OBJECT (trainer, "[GST_QUERY_CAPS]");
      gst_query_parse_caps (query, &filter);
      GST_DEBUG_OBJECT (trainer, "Caps from query : %" GST_PTR_FORMAT, filter);

      caps = gst_tensor_trainer_query_caps (trainer, sinkpad, filter);

      GST_INFO_OBJECT (trainer, "[GST_QUERY_CAPS] : %" GST_PTR_FORMAT, caps);
      gst_query_set_caps_result (query, caps);
      gst_caps_unref (caps);

      return TRUE;
    }
    case GST_QUERY_ACCEPT_CAPS:
    {
      GstCaps *caps;
      GstCaps *template_caps;
      gboolean result = FALSE;

      GST_DEBUG_OBJECT (trainer, "[GST_QUERY_ACCEPT_CAPS]");
      gst_query_parse_accept_caps (query, &caps);
      GST_INFO_OBJECT (trainer, "Accept caps from query : %" GST_PTR_FORMAT,
          caps);

      if (gst_caps_is_fixed (caps)) {
        template_caps = gst_pad_get_pad_template_caps (sinkpad);
        GST_DEBUG_OBJECT (trainer, "sinkpad template_caps : %" GST_PTR_FORMAT,
            template_caps);

        result = gst_caps_can_intersect (template_caps, caps);
        gst_caps_unref (template_caps);

        GST_DEBUG_OBJECT (trainer, "intersect caps : %" GST_PTR_FORMAT, caps);
      }

      gst_query_set_accept_caps_result (query, result);
      return TRUE;
    }
    default:
      break;
  }

  return gst_pad_query_default (sinkpad, parent, query);
}

/**
 * @brief This function handles src pad query.
 */
static gboolean
gst_tensor_trainer_src_query (GstPad * srcpad, GstObject * parent,
    GstQuery * query)
{
  GstTensorTrainer *trainer;
  trainer = GST_TENSOR_TRAINER (parent);

  GST_DEBUG_OBJECT (trainer, "Received %s query: %" GST_PTR_FORMAT,
      GST_QUERY_TYPE_NAME (query), query);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CAPS:
    {
      GstCaps *caps;
      GstCaps *filter;
      GST_DEBUG_OBJECT (trainer, "[GST_QUERY_CAPS]");
      gst_query_parse_caps (query, &filter);
      GST_DEBUG_OBJECT (trainer, "Caps from query : %" GST_PTR_FORMAT, filter);
      caps = gst_tensor_trainer_query_caps (trainer, srcpad, filter);

      GST_INFO_OBJECT (trainer, "[GST_QUERY_CAPS] : %" GST_PTR_FORMAT, caps);
      gst_query_set_caps_result (query, caps);
      gst_caps_unref (caps);
      return TRUE;
    }
    default:
      break;
  }
  return gst_pad_query_default (srcpad, parent, query);
}

/**
 * @brief Handle "PROP_FRAMEWORK" for set-property
 */
static void
gst_tensor_trainer_set_prop_framework (GstTensorTrainer * trainer,
    const GValue * value)
{
  g_free (trainer->fw_name);
  trainer->fw_name = g_value_dup_string (value);
  GST_INFO_OBJECT (trainer, "Framework: %s", trainer->fw_name);

  /** @todo Check valid framework */
}

/**
 * @brief Handle "PROP_MODEL_CONFIG" for set-property
 */
static void
gst_tensor_trainer_set_prop_model_config_file_path (GstTensorTrainer *
    trainer, const GValue * value)
{
  g_free (trainer->model_config);
  trainer->model_config = g_value_dup_string (value);
  trainer->prop.model_config = trainer->model_config;
  GST_INFO_OBJECT (trainer, "Model configuration file path: %s",
      trainer->prop.model_config);
}

/**
 * @brief Handle "PROP_MODEL_SAVE_PATH" for set-property
 */
static void
gst_tensor_trainer_set_model_save_path (GstTensorTrainer * trainer,
    const GValue * value)
{
  g_free (trainer->model_save_path);
  trainer->model_save_path = g_value_dup_string (value);
  trainer->prop.model_save_path = trainer->model_save_path;
  GST_INFO_OBJECT (trainer, "File path to save the model: %s",
      trainer->prop.model_save_path);
}

/**
 * @brief Find Trainer sub-plugin with the name.
 */
static gboolean
gst_tensor_trainer_find_framework (GstTensorTrainer * trainer, const char *name)
{
  const GstTensorTrainerFramework *fw = NULL;

  g_return_val_if_fail (name != NULL, FALSE);
  g_return_val_if_fail (trainer != NULL, FALSE);

  GST_INFO_OBJECT (trainer, "Try to find framework: %s", name);

  fw = get_subplugin (NNS_SUBPLUGIN_TRAINER, name);
  if (!fw) {
    GST_ERROR_OBJECT (trainer, "Can not find framework(%s)", trainer->fw_name);
    return FALSE;
  }

  GST_INFO_OBJECT (trainer, "Find framework %s:%p", trainer->fw_name, fw);
  trainer->fw = fw;

  return TRUE;
}

/**
 * @brief Create NN framework.
 */
static gboolean
gst_tensor_trainer_create_framework (GstTensorTrainer * trainer)
{
  g_return_val_if_fail (trainer != NULL, FALSE);

  if (!trainer->fw || trainer->fw_created) {
    GST_ERROR_OBJECT (trainer, "fw is not opened(%d) or fw is not null(%p)",
        trainer->fw_created, trainer->fw);
    return FALSE;
  }

  if (!trainer->fw->create) {
    GST_ERROR_OBJECT (trainer, "Could not create framework");
    return FALSE;
  }

  GST_DEBUG_OBJECT (trainer, "%p", trainer->privateData);
  if (trainer->fw->create (trainer->fw, &trainer->prop,
          &trainer->privateData) >= 0)
    trainer->fw_created = TRUE;
  GST_DEBUG_OBJECT (trainer, "Success, Framework: %p", trainer->privateData);

  return TRUE;
}

/**
 * @brief Calculate tensor buffer size
 */
gsize
gst_tensor_trainer_get_tensor_size (GstTensorTrainer * trainer,
    guint index, gboolean is_input)
{
  GstTensorsInfo *info;

  if (is_input)
    info = &trainer->prop.input_meta;
  else
    info = &trainer->output_meta;

  /* Internal Logic Error: out of bound */
  if (index >= info->num_tensors) {
    GST_ERROR_OBJECT (trainer, "has inconsistent data");
    return 0;
  }

  return gst_tensors_info_get_size (info, index);
}

/**
 * @brief Create model
 */
static gboolean
gst_tensor_trainer_create_model (GstTensorTrainer * trainer)
{
  gboolean ret = TRUE;

  g_return_val_if_fail (trainer != NULL, FALSE);
  g_return_val_if_fail (trainer->fw_name != NULL, FALSE);

  ret = gst_tensor_trainer_find_framework (trainer, trainer->fw_name);
  if (!ret)
    return ret;

  if (trainer->fw) {
    /* model create and compile */
    ret = gst_tensor_trainer_create_framework (trainer);
  }

  return ret;
}

/**
 * @brief Create a event notifier
 */
static void
gst_tensor_trainer_create_event_notifier (GstTensorTrainer * trainer)
{
  g_return_if_fail (trainer != NULL);
  g_return_if_fail (trainer->fw != NULL);

  trainer->notifier.notifier = (void *) trainer;
}

/**
 * @brief Train model
 */
static void
gst_tensor_trainer_train_model (GstTensorTrainer * trainer)
{
  gint ret = -1;
  g_return_if_fail (trainer != NULL);
  g_return_if_fail (trainer->fw != NULL);
  g_return_if_fail (trainer->fw->start != NULL);

  GST_DEBUG_OBJECT (trainer, "Start training model");
  ret =
      trainer->fw->start (trainer->fw, &trainer->prop, &trainer->notifier,
      trainer->privateData);
  if (ret != 0) {
    GST_ERROR_OBJECT (trainer, "model training is failed");
  }
}

/**
 * @brief initialize the output tensor dimension
 */
static void
gst_tensor_trainer_output_dimension (GstTensorTrainer * trainer)
{
  GstTensorsInfo *info;
  int i = 0;
  int value[8] = { 1, 1, 4, 1, 1, 1, 1, 1 };  /** loss, accuracy, val_loss, val_accuracy */
  g_return_if_fail (trainer != NULL);

  info = &trainer->output_meta;
  trainer->output_ranks[0] = 4;
  trainer->output_configured = TRUE;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    info->info[0].dimension[i] = value[i];

  info->num_tensors = 1;
}

/**
 * @brief initialize the output tensor type
 */
static void
gst_tensor_trainer_output_type (GstTensorTrainer * trainer)
{
  GstTensorsInfo *info;
  g_return_if_fail (trainer != NULL);

  info = &trainer->output_meta;
  info->info[0].type = _NNS_FLOAT64;
}

/**
 * @brief Trainer's sub-plugin should call this function to register itself.
 * @param[in] ttsp tensor_trainer sub-plugin to be registered.
 * @return TRUE if registered. FALSE is failed or duplicated.
 */
int
nnstreamer_trainer_probe (GstTensorTrainerFramework * ttsp)
{
  GstTensorTrainerFrameworkInfo info;
  GstTensorTrainerProperties prop;
  const char *name = NULL;
  int ret = 0;

  g_return_val_if_fail (ttsp != NULL, 0);

  memset (&prop, 0, sizeof (GstTensorTrainerProperties));
  gst_tensors_info_init (&prop.input_meta);

  if (ret != ttsp->getFrameworkInfo (ttsp, &prop, NULL, &info)) {
    GST_ERROR ("getFrameworkInfo() failed");
    return FALSE;
  }
  name = info.name;

  return register_subplugin (NNS_SUBPLUGIN_TRAINER, name, ttsp);
}

/**
 * @brief Trainer's sub-plugin may call this to unregister itself.
 * @param[in] ttsp tensor_trainer sub-plugin to be unregistered.
 * @return TRUE if unregistered. FALSE is failed.
 */
int
nnstreamer_trainer_exit (GstTensorTrainerFramework * ttsp)
{
  GstTensorTrainerFrameworkInfo info;
  GstTensorTrainerProperties prop;
  const char *name = NULL;
  int ret = 0;

  g_return_val_if_fail (ttsp != NULL, 0);

  memset (&prop, 0, sizeof (GstTensorTrainerProperties));
  gst_tensors_info_init (&prop.input_meta);

  if (ret != ttsp->getFrameworkInfo (ttsp, &prop, NULL, &info)) {
    GST_ERROR ("getFrameworkInfo() failed");
    return FALSE;
  }
  name = info.name;

  return unregister_subplugin (NNS_SUBPLUGIN_TRAINER, name);
}

/**
 * @brief Trainer's sub-plugin may call this to send event.
 * @param[in] notifier event notifier, sub-plugin must send events with this.
 * @param[in] type event type
 */
void
nnstreamer_trainer_notify_event (GstTensorTrainerEventNotifier * notifier,
    GstTensorTrainerEventType type, void *data)
{
  GstTensorTrainer *trainer;
  g_return_if_fail (notifier != NULL);
  g_return_if_fail (type < TRAINER_EVENT_UNKNOWN || type > 0);
  UNUSED (data);

  trainer = (GstTensorTrainer *) notifier->notifier;
  g_return_if_fail (GST_IS_TENSOR_TRAINER (trainer));

  GST_DEBUG ("Received GstTensorTrainerEvent(%d)", type);

  switch (type) {
    case TRAINER_EVENT_EPOCH_COMPLETION:
      g_mutex_lock (&trainer->epoch_completion_lock);
      trainer->is_epoch_complete = TRUE;
      GST_DEBUG ("send epoch_completion_cond signal");
      g_cond_signal (&trainer->epoch_completion_cond);
      g_mutex_unlock (&trainer->epoch_completion_lock);
      break;
    case TRAINER_EVENT_TRAINING_COMPLETION:
      g_mutex_lock (&trainer->training_completion_lock);
      trainer->is_training_complete = TRUE;
      GST_DEBUG ("send training_completion_cond signal");
      g_cond_signal (&trainer->training_completion_cond);
      g_mutex_unlock (&trainer->training_completion_lock);
      break;
    default:
      break;
  }
}
