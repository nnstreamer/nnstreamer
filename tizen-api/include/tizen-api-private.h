/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd All Rights Reserved
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
 */
/**
 * @file tizen-api-private.h
 * @date 07 March 2019
 * @brief Tizen NNStreamer/Pipeline(main) C-API Private Header.
 *        This file should NOT be exported to SDK or devel package.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __TIZEN_NNSTREAMER_API_PRIVATE_H__
#define __TIZEN_NNSTREAMER_API_PRIVATE_H__

#include <glib.h>
#include <gmodule.h>
#include <gst/gst.h>
#include "nnstreamer.h"
#include <tizen_error.h>
#include <dlog.h>
#include <nnstreamer/tensor_typedef.h>
#include <gst/app/gstappsrc.h>

#define DLOG_TAG "nnstreamer-capi-pipeline"

#define dlogi(...) \
    dlog_print (DLOG_INFO, DLOG_TAG, __VA_ARGS__)

#define dlogw(...) \
    dlog_print (DLOG_WARN, DLOG_TAG, __VA_ARGS__)

#define dloge(...) \
    dlog_print (DLOG_ERROR, DLOG_TAG, __VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Possible controls on elements of a pipeline.
 */
typedef enum {
  ML_PIPELINE_ELEMENT_UNKNOWN = 0x0,
  ML_PIPELINE_ELEMENT_SINK = 0x1,
  ML_PIPELINE_ELEMENT_APP_SRC = 0x2,
  ML_PIPELINE_ELEMENT_APP_SINK = 0x3,
  ML_PIPELINE_ELEMENT_VALVE = 0x4,
  ML_PIPELINE_ELEMENT_SWITCH_INPUT = 0x8,
  ML_PIPELINE_ELEMENT_SWITCH_OUTPUT = 0x9,
} ml_pipeline_element_type_e;

/**
 * @brief Internal private representation of pipeline handle.
 */
typedef struct _ml_pipeline ml_pipeline;

/**
 * @brief An element that may be controlled individually in a pipeline.
 */
typedef struct _ml_pipeline_element {
  GstElement *element; /**< The Sink/Src/Valve/Switch element */
  ml_pipeline *pipe; /**< The main pipeline */
  char *name;
  ml_pipeline_element_type_e type;
  GstPad *src;
  GstPad *sink; /**< Unref this at destroy */
  GstTensorsInfo tensorsinfo;
  size_t size;

  GList *handles;
  int maxid; /**< to allocate id for each handle */
  gulong handle_id;

  GMutex lock; /**< Lock for internal values */
} ml_pipeline_element;

/**
 * @brief Internal private representation of pipeline handle.
 * @detail This should not be exposed to applications
 */
struct _ml_pipeline {
  GstElement *element;    /**< The pipeline itself (GstPipeline) */
  GMutex lock;            /**< Lock for pipeline operations */
  GHashTable *namednodes; /**< hash table of "element"s. */
};

/**
 * @brief Internal private representation of sink handle of GstTensorSink
 * @detail This represents a single instance of callback registration. This should not be exposed to applications.
 */
typedef struct _ml_pipeline_sink {
  ml_pipeline *pipe; /**< The pipeline, which is the owner of this ml_pipeline_sink */
  ml_pipeline_element *element;
  guint32 id;
  ml_pipeline_sink_cb cb;
  void *pdata;
} ml_pipeline_sink;

/**
 * @brief Internal private representation of src handle of GstAppSrc
 * @detail This represents a single instance of registration. This should not be exposed to applications.
 */
typedef struct _ml_pipeline_src {
  ml_pipeline *pipe;
  ml_pipeline_element *element;
  guint32 id;
} ml_pipeline_src;

/**
 * @brief Internal private representation of switch handle (GstInputSelector, GstOutputSelector)
 * @detail This represents a single instance of registration. This should not be exposed to applications.
 */
typedef struct _ml_pipeline_switch {
  ml_pipeline *pipe;
  ml_pipeline_element *element;
  guint32 id;
} ml_pipeline_switch;

/**
 * @brief Internal private representation of valve handle (GstValve)
 * @detail This represents a single instance of registration. This should not be exposed to applications.
 */
typedef struct _ml_pipeline_valve {
  ml_pipeline *pipe;
  ml_pipeline_element *element;
  guint32 id;
} ml_pipeline_valve;

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /*__TIZEN_NNSTREAMER_API_PRIVATE_H__*/
