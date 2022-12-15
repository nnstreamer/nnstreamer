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
 * gst-launch-1.0 videotestsrc ! video/x-raw, format=RGB, width=640, height=480 !
 * tensor_converter ! tensor_transform mode=typecast option=float32 !
 * tensor_trainer framework=nntrainer model-config=/usr/bin/model.ini
 * push-output=false input=3:640:480:1 inputtype=float32
 * output=1:1:1:1 outputtype=float32 ! tensor_sink
 * ]|
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <stdlib.h>
#include <nnstreamer_subplugin.h>
#include <nnstreamer_util.h>
#include "gsttensor_trainer.h"
#include <unistd.h>
#include <sys/syscall.h>

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
 * @brief Default framework property value
 */
#define DEFAULT_PROP_FRAMEWORK "nntrainer"

/**
 * @brief Default string property value 
 */
#define DEFAULT_STR_PROP_VALUE ""

/**
 * @brief Default string property value 
 */
enum
{
  PROP_0,
  PROP_FRAMEWORK,
  PROP_MODEL_CONFIG,
  PROP_MODEL_SAVE_PATH,
  PROP_INPUT,
  PROP_OUTPUT,
  PROP_INPUTTYPE,
  PROP_OUTPUTTYPE,
  PROP_PUSH_OUTPUT,
  PROP_INPUT_LIST,              /* number of input list */
  PROP_LABEL_LIST,              /* number of label list */
  PROP_TRAIN_SAMPLES,           /*number of train data */
  PROP_VALID_SAMPLES,           /*number of vaild data */
};

#define NNS_TENSOR_TYPES 11

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
static void gst_tensor_trainer_set_prop_dimension (GstTensorTrainer * trainer,
    const GValue * value, const gboolean is_input);
static void gst_tensor_trainer_set_prop_type (GstTensorTrainer * trainer,
    const GValue * value, const gboolean is_input);
static void gst_tensor_trainer_find_framework (GstTensorTrainer * trainer,
    const char *name);
static void gst_tensor_trainer_create_framework (GstTensorTrainer * trainer);
static gsize gst_tensor_trainer_get_tensor_size (GstTensorTrainer * trainer,
    guint index, gboolean is_input);
static void gst_tensor_trainer_create_model (GstTensorTrainer * trainer);
static void gst_tensor_trainer_train_model (GstTensorTrainer * trainer);

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
      g_param_spec_string ("framework", "Framework", "Neural network framework",
          DEFAULT_PROP_FRAMEWORK,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODEL_CONFIG,
      g_param_spec_string ("model-config", "Model configuration file path",
          "Configuration file path for creating model",
          DEFAULT_STR_PROP_VALUE,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODEL_SAVE_PATH,
      g_param_spec_string ("model-save-path", "Model save path",
          "File path to save the model", DEFAULT_STR_PROP_VALUE,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_INPUT,
      g_param_spec_string ("input", "Input dimension",
          "Input tensors dimension from inner array, up to 4 dimensions ?",
          DEFAULT_STR_PROP_VALUE,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_OUTPUT,
      g_param_spec_string ("output", "Output dimension",
          "Output tensors dimension from inner array, up to 4 dimensions ?",
          DEFAULT_STR_PROP_VALUE,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_INPUTTYPE,
      g_param_spec_string ("inputtype", "Input tensor element type",
          "Type of each element of the input tensor ?",
          DEFAULT_STR_PROP_VALUE,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_OUTPUTTYPE,
      g_param_spec_string ("outputtype", "Output tensor element type",
          "Type of each element of the input tensor ?",
          DEFAULT_STR_PROP_VALUE,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_PUSH_OUTPUT,
      g_param_spec_boolean ("push-output", "Push output tensor",
          "Add output tensors to output GstBuffer", FALSE,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_INPUT_LIST,
      g_param_spec_uint ("input-lists", "Number of input lists",
          "Number of input list", 0, NNS_TENSOR_SIZE_LIMIT, 1,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LABEL_LIST,
      g_param_spec_uint ("label-lists", "Number of label lists",
          "Number of label list", 0, NNS_TENSOR_SIZE_LIMIT, 1,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_TRAIN_SAMPLES,
      g_param_spec_uint ("train-samples", "Number of train samples",
          "Number of train samples", 0, G_MAXUINT, 1,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_VALID_SAMPLES,
      g_param_spec_uint ("valid-samples", "Number of valid samples",
          "Number of valid samples", 0, G_MAXUINT, 1,
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
  trainer->fw_name = g_strdup (DEFAULT_PROP_FRAMEWORK);
  trainer->model_config = g_strdup (DEFAULT_STR_PROP_VALUE);
  trainer->model_save_path = g_strdup (DEFAULT_STR_PROP_VALUE);
  trainer->input_dimensions = g_strdup (DEFAULT_STR_PROP_VALUE);
  trainer->output_dimensions = g_strdup (DEFAULT_STR_PROP_VALUE);
  trainer->input_type = g_strdup (DEFAULT_STR_PROP_VALUE);
  trainer->output_type = g_strdup (DEFAULT_STR_PROP_VALUE);
  trainer->push_output = FALSE;

  trainer->fw = NULL;
  trainer->fw_created = 0;      /* for test */
  trainer->configured = 0;
  trainer->input_configured = 0;
  trainer->output_configured = FALSE;
  trainer->inputtype_configured = 0;
  trainer->outputtype_configured = 0;

  trainer->total_invoke_num = 0;

  g_cond_init (&trainer->train_complete_cond);
  g_mutex_init (&trainer->trainer_lock);
  trainer->prop.train_complete_cond = &trainer->train_complete_cond;
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
  g_free (trainer->input_dimensions);
  g_free (trainer->output_dimensions);
  g_free (trainer->input_type);
  g_free (trainer->output_type);

  g_cond_clear (&trainer->train_complete_cond);
  g_mutex_clear (&trainer->trainer_lock);

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
    case PROP_INPUT:
      gst_tensor_trainer_set_prop_dimension (trainer, value, TRUE);
      break;
    case PROP_OUTPUT:
      gst_tensor_trainer_set_prop_dimension (trainer, value, FALSE);
      break;
    case PROP_INPUTTYPE:
      gst_tensor_trainer_set_prop_type (trainer, value, TRUE);
      break;
    case PROP_OUTPUTTYPE:
      gst_tensor_trainer_set_prop_type (trainer, value, FALSE);
      break;
    case PROP_PUSH_OUTPUT:
      trainer->push_output = g_value_get_boolean (value);
      GST_INFO_OBJECT (trainer, "push output: %d", trainer->push_output);
      break;
    case PROP_INPUT_LIST:
      trainer->prop.num_inputs = g_value_get_uint (value);
      break;
    case PROP_LABEL_LIST:
      trainer->prop.num_labels = g_value_get_uint (value);
      break;
    case PROP_TRAIN_SAMPLES:
      trainer->prop.num_train_samples = g_value_get_uint (value);
      break;
    case PROP_VALID_SAMPLES:
      trainer->prop.num_valid_samples = g_value_get_uint (value);
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
    case PROP_INPUT:
      g_value_set_string (value, trainer->input_dimensions);
      break;
    case PROP_OUTPUT:
      g_value_set_string (value, trainer->output_dimensions);
      break;
    case PROP_INPUTTYPE:
      g_value_set_string (value, trainer->input_type);
      break;
    case PROP_OUTPUTTYPE:
      g_value_set_string (value, trainer->output_type);
      break;
    case PROP_PUSH_OUTPUT:
      g_value_set_boolean (value, trainer->push_output);
      break;
    case PROP_INPUT_LIST:
      g_value_set_uint (value, trainer->prop.num_inputs);
      break;
    case PROP_LABEL_LIST:
      g_value_set_uint (value, trainer->prop.num_labels);
      break;
    case PROP_TRAIN_SAMPLES:
      g_value_set_uint (value, trainer->prop.num_train_samples);
      break;
    case PROP_VALID_SAMPLES:
      g_value_set_uint (value, trainer->prop.num_valid_samples);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
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
      break;

    case GST_STATE_CHANGE_READY_TO_PAUSED:
      GST_INFO_OBJECT (trainer, "READY_TO_PAUSED");
      gst_tensor_trainer_create_model (trainer);
      break;

    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      GST_INFO_OBJECT (trainer, "PAUSED_TO_PLAYING");
      gst_tensor_trainer_train_model (trainer);
      break;

    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      GST_INFO_OBJECT (trainer, "PLAYING_TO_PAUSED");
      /* pause model train */
      break;

    case GST_STATE_CHANGE_PAUSED_TO_READY:
      GST_INFO_OBJECT (trainer, "PAUSED_TO_READY");
      /* stop model train */
      break;

    case GST_STATE_CHANGE_READY_TO_NULL:
      GST_INFO_OBJECT (trainer, "READY_TO_NULL");
      /* destroy or reset model ? */
      break;

    default:
      break;
  }

  return ret;
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
  guint mem_blocks, i;
  gsize header_size, expected;
  gboolean in_flexible, out_flexible;
  GstMemory *in_mem[NNS_TENSOR_SIZE_LIMIT] = { 0, };
  GstMapInfo in_info[NNS_TENSOR_SIZE_LIMIT];
  GstMemory *out_mem[NNS_TENSOR_SIZE_LIMIT] = { 0, };
  GstMapInfo out_info[NNS_TENSOR_SIZE_LIMIT];
  GstTensorMemory in_tensors[NNS_TENSOR_SIZE_LIMIT];
  GstTensorMemory invoke_tensors[NNS_TENSOR_SIZE_LIMIT];
  GstTensorMemory out_tensors[NNS_TENSOR_SIZE_LIMIT];
  GstTensorMetaInfo in_meta[NNS_TENSOR_SIZE_LIMIT];
  GstTensorMetaInfo out_meta[NNS_TENSOR_SIZE_LIMIT];

  pid_t pid = getpid ();
  pid_t tid = syscall (SYS_gettid);

  trainer = GST_TENSOR_TRAINER (parent);

  GST_DEBUG_OBJECT (trainer, "pid: %d, tid: %d", pid, tid);

  mem_blocks = gst_buffer_n_memory (inbuf);
  for (i = 0; i < mem_blocks; i++) {
    in_mem[i] = gst_buffer_peek_memory (inbuf, i);
    if (!gst_memory_map (in_mem[i], &in_info[i], GST_MAP_READ)) {
      GST_ERROR_OBJECT (trainer, "Could not map in_mem[%d] GstMemory", i);
      goto error;
    }
    in_flexible = gst_tensor_pad_caps_is_flexible (sinkpad);
    /* Get header size */
    header_size = 0;
    if (in_flexible) {
      gst_tensor_meta_info_parse_header (&in_meta[i], &in_info[i].data);
      header_size = gst_tensor_meta_info_get_header_size (&in_meta[i]);
      GST_INFO ("flexible header size:%zd", header_size);
    } else {
      GST_INFO ("not flexible header size:%zd", header_size);
    }

    in_tensors[i].data = in_info[i].data + header_size;
    in_tensors[i].size = in_info[i].size - header_size;
    GST_INFO ("tensor size: %zd", in_tensors[i].size);
  }

  /* Prepare tensor to invoke */
  /* Check number of input tensors */
  GST_DEBUG_OBJECT (trainer, "num_tensors: %d",
      trainer->prop.input_meta.num_tensors);
  if (mem_blocks != trainer->prop.input_meta.num_tensors) {
    GST_ERROR_OBJECT (trainer, "Invalid memory blocks(%d),"
        "number of input tensors may be (%d)", mem_blocks,
        trainer->prop.input_meta.num_tensors);
    goto error;
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
    invoke_tensors[i] = in_tensors[i];
    GST_INFO ("in_tensors[%d].size= %zd", i, in_tensors[i].size);
    GST_INFO ("in_tensors[%d].data: %p", i, in_tensors[i].data);
    GST_INFO ("invoke_tensors[%d].size= %zd", i, invoke_tensors[i].size);
    GST_INFO ("invoke_tensors[%d].data: %p", i, invoke_tensors[i].data);
  }

  /* Prepare output tensor */
  if (trainer->push_output) {
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
        GST_ERROR_OBJECT (trainer, "Could not map in_mem[%d] GstMemory", i);
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
    }
    /* Call the trainer-subplugin callback, invoke */
    ret =
        trainer->fw->invoke (trainer->fw, &trainer->prop,
        trainer->privateData, invoke_tensors);

    /* Free out info */
    for (i = 0; i < trainer->output_meta.num_tensors; i++) {
      if (out_mem[i])
        gst_memory_unmap (out_mem[i], &out_info[i]);
      //if (ret != 0) {
      //  gst_allocator_free (out_mem[i]->allocator, out_mem[i]);
      //}
    }
  } else {
    ret =
        trainer->fw->invoke (trainer->fw, &trainer->prop,
        trainer->privateData, invoke_tensors);
  }

  trainer->total_invoke_num++;

  /* Free in info */
  for (i = 0; i < mem_blocks; i++)
    gst_memory_unmap (in_mem[i], &in_info[i]);

  if (ret < 0) {
    GST_ERROR_OBJECT (trainer, "Invoke error");
    return GST_FLOW_ERROR;
  }

  /* Update result if one of epochs is complete,
     push one outbuf is necessary to change pipeline state.
     Scheduling with subplugin does not work.
   */
  if (trainer->total_invoke_num == 1
      || trainer->total_invoke_num ==
      trainer->prop.num_train_samples + trainer->prop.num_valid_samples) {
    outbuf = gst_buffer_new ();
    for (i = 0; i < trainer->output_meta.num_tensors; i++) {
      /* append the memory block to outbuf */
      gst_buffer_append_memory (outbuf, out_mem[i]);
    }
    GST_INFO ("after out buffer size : %zd", gst_buffer_get_size (outbuf));

    gst_pad_push (trainer->srcpad, outbuf);
    return GST_FLOW_OK;
  }

  return GST_FLOW_OK;
error:
  mem_blocks = gst_buffer_n_memory (inbuf);
  for (i = 0; i < mem_blocks; i++) {
    if (in_mem[i])
      gst_memory_unmap (in_mem[i], &in_info[i]);
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
  GstCaps *result;
  GstCaps *caps;
  GstStructure *structure;
  gboolean configured = FALSE;
  GstTensorsConfig in_config;

  gst_tensors_config_init (&in_config);
  gst_tensors_config_init (&trainer->out_config);

  caps = gst_tensor_pad_possible_caps_from_config (pad, &in_config);
  structure = gst_caps_get_structure (caps, 0);

  if (!gst_tensors_config_from_structure (&in_config, structure))
    return NULL;
  gst_caps_unref (caps);

  /* Need to set property (input, inputtype, output, outputtype)
     for trainer->input_meta and trainer->output_meta */
  if (pad == trainer->srcpad) {
    if (trainer->input_configured) {
      gst_tensors_info_copy (&trainer->out_config.info,
          &trainer->prop.input_meta);
      configured = TRUE;
    }
  } else {
    if (trainer->output_configured) {
      configured = TRUE;
      gst_tensors_info_copy (&trainer->out_config.info, &trainer->output_meta);
    }
  }

  if (configured)
    /* Output info may be configured */
    result =
        gst_tensor_pad_possible_caps_from_config (pad, &trainer->out_config);
  else
    result = gst_caps_from_string (CAPS_STRING);

  GST_DEBUG_OBJECT (trainer, "caps intersect without filter %" GST_PTR_FORMAT,
      result);

  if (filter) {
    GstCaps *intersection;
    intersection =
        gst_caps_intersect_full (result, filter, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (result);
    result = intersection;
    GST_DEBUG_OBJECT (trainer, "result caps %" GST_PTR_FORMAT, result);
  }
  gst_tensors_config_free (&in_config);

  return result;
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
      GST_INFO_OBJECT (trainer, "get GST_EVENT_EOS event..state is %d",
          GST_STATE (trainer));
      g_mutex_lock (&trainer->trainer_lock);
      GST_INFO_OBJECT (trainer, "wait for train_complete_cond signal");
      g_cond_wait (&trainer->train_complete_cond, &trainer->trainer_lock);
      g_mutex_unlock (&trainer->trainer_lock);
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
      GstTensorsConfig in_config;
      gboolean ret = FALSE;

      gst_event_parse_caps (event, &in_caps);
      GST_INFO_OBJECT (trainer, "[in-caps] : %" GST_PTR_FORMAT, in_caps);
      /* set out caps */

      structure = gst_caps_get_structure (in_caps, 0);
      if (!gst_tensors_config_from_structure (&in_config, structure))
        return ret;

      if (!gst_tensors_info_is_equal (&in_config.info,
              &trainer->prop.input_meta)) {
        GST_ERROR_OBJECT (trainer,
            "The input tensors info is different between incaps and set property value. "
            "Please check pipeline's input tensor info and tensor_trainer's set property values"
            "(input, inputtype, output and outputtype)");
        return ret;
      }

      trainer->out_config.rate_n = in_config.rate_n;
      trainer->out_config.rate_d = in_config.rate_d;

      out_caps =
          gst_tensor_pad_caps_from_config (trainer->srcpad,
          &trainer->out_config);
      GST_INFO_OBJECT (trainer, "[out-caps] : %" GST_PTR_FORMAT, out_caps);

      ret = gst_pad_set_caps (trainer->srcpad, out_caps);

      gst_event_unref (event);
      gst_caps_unref (out_caps);

      gst_tensors_config_free (&in_config);
      gst_tensors_config_free (&trainer->out_config);

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
  GST_INFO_OBJECT (trainer, "framework: %s", trainer->fw_name);

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
  GST_INFO_OBJECT (trainer, "model configuration file path: %s",
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
  GST_INFO_OBJECT (trainer, "file path to save the model: %s",
      trainer->prop.model_save_path);
}

/**
 * @brief Handle "PROP_INPUT" and "PROP_OUTPUT" for set-property
 */
static void
gst_tensor_trainer_set_prop_dimension (GstTensorTrainer * trainer,
    const GValue * value, const gboolean is_input)
{
  GstTensorsInfo *info;
  unsigned int *rank;
  guint num_dims;
  gchar **str_dims;
  guint i;

  if ((is_input && trainer->input_configured) || (!is_input
          && trainer->output_configured)) {
    GST_ERROR_OBJECT (trainer,
        "Cannot change %s dimension" "the element/pipeline is configured.",
        (is_input) ? "input" : "output");
    return;
  }

  if (is_input) {
    info = &trainer->prop.input_meta;
    rank = trainer->input_ranks;
    trainer->input_configured = TRUE;
    g_free (trainer->input_dimensions);
    trainer->input_dimensions = g_value_dup_string (value);
    GST_INFO_OBJECT (trainer, "input: %s", trainer->input_dimensions);
  } else {
    info = &trainer->output_meta;
    rank = trainer->output_ranks;
    trainer->output_configured = TRUE;
    g_free (trainer->output_dimensions);
    trainer->output_dimensions = g_value_dup_string (value);
    GST_INFO_OBJECT (trainer, "output: %s", trainer->output_dimensions);
  }

  str_dims = g_strsplit_set (g_value_get_string (value), ",.", -1);
  num_dims = g_strv_length (str_dims);

  if (num_dims > NNS_TENSOR_SIZE_LIMIT) {
    GST_WARNING_OBJECT (trainer, "Invalid param, dimensions(%d) max(%d)",
        num_dims, NNS_TENSOR_SIZE_LIMIT);
    num_dims = NNS_TENSOR_SIZE_LIMIT;
  }

  for (i = 0; i < num_dims; i++)
    rank[i] = gst_tensor_parse_dimension (str_dims[i], info->info[i].dimension);

  info->num_tensors = num_dims;

  g_strfreev (str_dims);
}

/**
 * @brief Handle "PROP_INPUTTYPE" and "PROP_OUTPUTTYPE" for set-property
 */
static void
gst_tensor_trainer_set_prop_type (GstTensorTrainer * trainer,
    const GValue * value, const gboolean is_input)
{
  GstTensorsInfo *info;
  guint num_types;

  if ((is_input && trainer->inputtype_configured) || (!is_input
          && trainer->outputtype_configured)) {
    GST_ERROR_OBJECT (trainer,
        "Cannot change %stype" "the element/pipeline is configured.",
        (is_input) ? "input" : "output");
    return;
  }

  if (is_input) {
    info = &trainer->prop.input_meta;
    trainer->inputtype_configured = TRUE;
    g_free (trainer->input_type);
    trainer->input_type = g_value_dup_string (value);
    GST_INFO_OBJECT (trainer, "inputtype : %s", trainer->input_type);
  } else {
    info = &trainer->output_meta;
    trainer->outputtype_configured = TRUE;
    g_free (trainer->output_type);
    trainer->output_type = g_value_dup_string (value);
    GST_INFO_OBJECT (trainer, "outputtype : %s", trainer->output_type);
  }

  num_types =
      gst_tensors_info_parse_types_string (info, g_value_get_string (value));

  info->num_tensors = num_types;
}

/**
 * @brief Find Trainer sub-plugin with the name.
 */
static void
gst_tensor_trainer_find_framework (GstTensorTrainer * trainer, const char *name)
{
  const GstTensorTrainerFramework *fw = NULL;

  g_return_if_fail (name != NULL);
  g_return_if_fail (trainer != NULL);

  GST_INFO ("find framework: %s", name);

  fw = get_subplugin (NNS_SUBPLUGIN_TRAINER, name);
  if (fw) {
    GST_INFO_OBJECT (trainer, "find framework %s:%p", trainer->fw_name, fw);
    trainer->fw = fw;
  } else {
    GST_ERROR_OBJECT (trainer, "Can not find framework(%s)", trainer->fw_name);
  }
}

/**
 * @brief Create NN framework.
 */
static void
gst_tensor_trainer_create_framework (GstTensorTrainer * trainer)
{
  pid_t pid = getpid ();
  pid_t tid = syscall (SYS_gettid);
  g_return_if_fail (trainer != NULL);
  GST_ERROR_OBJECT (trainer, "pid: %d, tid: %d", pid, tid);
  if (!trainer->fw || trainer->fw_created) {
    GST_ERROR_OBJECT (trainer, "fw is not opened(%d) or fw is null(%p)",
        trainer->fw_created, trainer->fw);
    return;
  }

  /* For test */
  if (!trainer->fw->create) {   /* fw->create, create model with configuration file */
    GST_ERROR_OBJECT (trainer, "Could not find fw->create");
    return;
  }
  /* Test code, need to create with load ini file */
  GST_ERROR ("%p", trainer->privateData);
  if (trainer->fw->create (trainer->fw, &trainer->prop,
          &trainer->privateData) >= 0)
    trainer->fw_created = TRUE;
  GST_ERROR ("%p", trainer->privateData);
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

  return gst_tensor_info_get_size (&info->info[index]);
}

/**
 * @brief Create model
 */
static void
gst_tensor_trainer_create_model (GstTensorTrainer * trainer)
{
  g_return_if_fail (trainer != NULL);
  GST_DEBUG_OBJECT (trainer, "called");

  if (trainer->fw_name)
    gst_tensor_trainer_find_framework (trainer, trainer->fw_name);
  if (trainer->fw) {
    /* model create and compile */
    gst_tensor_trainer_create_framework (trainer);
  }
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
  g_return_if_fail (trainer->fw->train != NULL);

  GST_DEBUG_OBJECT (trainer, "called");
  ret = trainer->fw->train (trainer->fw, &trainer->prop, trainer->privateData);
  if (ret != 0) {
    GST_ERROR_OBJECT (trainer, "model train is failed");
  }
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
