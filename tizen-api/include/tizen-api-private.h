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
#include <nnstreamer/tensor_typedef.h>
#include <gst/app/gstappsrc.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Possible controls on elements of a pipeline.
 */
typedef enum {
  NNSAPI_UNKNOWN = 0x0,
  NNSAPI_SINK = 0x1,
  NNSAPI_SRC = 0x2,
  NNSAPI_VALVE = 0x3,
  NNSAPI_SWITCH_INPUT = 0x8,
  NNSAPI_SWITCH_OUTPUT = 0x9,
} elementType;

/**
 * @brief Internal private representation of pipeline handle.
 */
typedef struct _nns_pipeline nns_pipeline;

/**
 * @brief An element that may be controlled individually in a pipeline.
 */
typedef struct _element {
  GstElement *element; /**< The Sink/Src/Valve/Switch element */
  nns_pipeline *pipe; /**< The main pipeline */
  char *name;
  elementType type;
  GstPad *src;
  GstPad *sink; /**< Unref this at destroy */
  GstTensorsInfo tensorsinfo;
  size_t size;

  GList *handles;
  int maxid; /**< to allocate id for each handle */

  GMutex lock; /**< Lock for internal values */
} element;

/**
 * @brief Internal private representation of pipeline handle.
 * @detail This should not be exposed to applications
 */
struct _nns_pipeline {
  GstElement *element;    /**< The pipeline itself (GstPipeline) */
  GMutex lock;            /**< Lock for pipeline operations */
  GHashTable *namednodes; /**< hash table of "element"s. */
};

/**
 * @brief Internal private representation of sink handle of GstTensorSink
 * @detail This represents a single instance of callback registration. This should not be exposed to applications.
 */
typedef struct _nns_sink {
  nns_pipeline *pipe; /**< The pipeline, which is the owner of this nns_sink */
  element *element;
  guint32 id;
  nns_sink_cb cb;
  void *pdata;
} nns_sink;

/**
 * @brief Internal private representation of src handle of GstAppSrc
 * @detail This represents a single instance of registration. This should not be exposed to applications.
 */
typedef struct _nns_src {
  nns_pipeline *pipe;
  element *element;
  guint32 id;
} nns_src;

/**
 * @brief Internal private representation of switch handle (GstInputSelector, GstOutputSelector)
 * @detail This represents a single instance of registration. This should not be exposed to applications.
 */
typedef struct _nns_switch {
  nns_pipeline *pipe;
  element *element;
  guint32 id;
} nns_switch;

/**
 * @brief Internal private representation of valve handle (GstValve)
 * @detail This represents a single instance of registration. This should not be exposed to applications.
 */
typedef struct _nns_valve {
  nns_pipeline *pipe;
  element *element;
  guint32 id;
} nns_valve;

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /*__TIZEN_NNSTREAMER_API_PRIVATE_H__*/
