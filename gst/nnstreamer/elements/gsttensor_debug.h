/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer tensor_debug
 * Copyright (C) 2022 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file	gsttensor_debug.h
 * @date	23 Sep 2022
 * @brief	GStreamer plugin to help debug tensor streams.
 *
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_DEBUG_H__
#define __GST_TENSOR_DEBUG_H__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <tensor_common.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_DEBUG \
  (gst_tensor_debug_get_type())
#define GST_TENSOR_DEBUG(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_DEBUG,GstTensorDebug))
#define GST_TENSOR_DEBUG_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_DEBUG,GstTensorDebugClass))
#define GST_IS_TENSOR_DEBUG(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_DEBUG))
#define GST_IS_TENSOR_DEBUG_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_DEBUG))
#define GST_TENSOR_DEBUG_CAST(obj)  ((GstTensorDebug *)(obj))

typedef struct _GstTensorDebug GstTensorDebug;
typedef struct _GstTensorDebugClass GstTensorDebugClass;

/**
 * @brief Property "OUTPUT" specification
 */
typedef enum
{
  TDBG_OUTPUT_DISABLED = 0x00, /**< disable output. */
  TDBG_OUTPUT_CONSOLE_I = 0x11, /**< console/debug info */
  TDBG_OUTPUT_CONSOLE_W = 0x12, /**< console/debug warn */
  TDBG_OUTPUT_CONSOLE_E = 0x13, /**< console/debug error (non fatal) */
  TDBG_OUTPUT_GSTDBG_I = 0x24, /**< gst-debug info */
  TDBG_OUTPUT_GSTDBG_W = 0x28, /**< gst-debug warn */
  TDBG_OUTPUT_GSTDBG_E = 0x2C, /**< gst-debug error (non fatal) */
  TDBG_OUTPUT_CIRCULARBUF = 0x100, /**< in circular buffer for later retrievals. (@todo NYI NOT_SUPPORTED) */
  TDBG_OUTPUT_FILEWRITE = 0x200, /**< Write to a file. (@todo NYI NOT_SUPPORTED) */
} tdbg_output_mode;

/**
 * @brief Property "CAP" specification
 */
typedef enum
{
  TDBG_CAP_DISABLED = 0, /**< No output for tensor-capability info */
  TDBG_CAP_SHOW_UPDATE = 1, /**< Output tensor-capability if there is a change in gstcap */
  TDBG_CAP_SHOW_UPDATE_F = 2, /**< Output tensor-capability if there is a change in dimensions even if gstcap has no changes. This happens with format=flexible or sparse */
  TDBG_CAP_SHOW_ALWAYS = 3,
} tdbg_cap_mode;

/**
 * @brief Property "META" specification
 */
typedef enum
{
  TDBG_META_DISABLED = 0x0, /**< Don't bring up metadata of a stream */
  TDBG_META_TIMESTAMP = 0x1, /**< Enable timestamp info */
  TDBG_META_QUERYSERVER = 0x2, /**< Enable tensor-query-server related info */
} tdbg_meta_mode;

/**
 * @brief Internal data structure for tensor_debug instances.
 */
struct _GstTensorDebug
{
  GstBaseTransform element;	/**< This is the parent object */

  GstPad *sinkpad; /**< sink pad */
  GstPad *srcpad; /**< src pad */

  gboolean silent; /**< true to print minimized log */

  tdbg_output_mode output_mode;
  tdbg_cap_mode cap_mode;
  tdbg_meta_mode meta_mode;
};

/**
 * @brief GstTensorDebugClass data structure.
 */
struct _GstTensorDebugClass
{
  GstBaseTransformClass parent_class; /**< parent class = transform */
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_debug_get_type (void);


G_END_DECLS

#endif /** __GST_TENSOR_DEBUG_H__ */
