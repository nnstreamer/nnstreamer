/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2023 Samsung Electronics Co., Ltd.
 *
 * @file	gstdatareposink.c
 * @date	30 March 2023
 * @brief	GStreamer plugin that writes data from buffers to files in in MLOps Data repository
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyunil Park <hyunil46.park@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_DATA_REPO_SINK_H__
#define __GST_DATA_REPO_SINK_H__

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <json-glib/json-glib.h>
#include "gstdatarepo.h"

G_BEGIN_DECLS
#define GST_TYPE_DATA_REPO_SINK \
  (gst_data_repo_sink_get_type())
#define GST_DATA_REPO_SINK(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DATA_REPO_SINK,GstDataRepoSink))
#define GST_DATA_REPO_SINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DATA_REPO_SINK,GstDataRepoSinkClass))
#define GST_IS_DATA_REPO_SINK(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DATA_REPO_SINK))
#define GST_IS_DATA_REPO_SINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DATA_REPO_SINK))
#define GST_DATA_REPO_SINK_CAST(obj) ((GstDataRepoSink *)obj)

typedef struct _GstDataRepoSink GstDataRepoSink;
typedef struct _GstDataRepoSinkClass GstDataRepoSinkClass;

/**
 * @brief GstDataRepoSink data structure
 */
struct _GstDataRepoSink
{
  GstBaseSink element;

  GstCaps *fixed_caps;            /**< to get meta info */
  JsonObject *json_object;        /**< JSON object */
  JsonArray *sample_offset_array;   /**< offset array of sample */
  JsonArray *tensor_size_array;    /**< size array of flexible tensor */
  JsonArray *tensor_count_array;  /**< array for the number of cumulative tensors */
  guint flexible_tensor_count;

  gboolean is_flexible_tensors;
  gint fd;                        /**< open file descriptor*/
  GstDataRepoDataType data_type;  /**< data type */
  gint total_samples;             /**< The number of total samples, in the case of multi-files, it is used as an index. */
  guint64 fd_offset;              /**< offset of fd */
  guint sample_size;             /**< size of one sample */

  /* property */
  gchar *filename;    /**< filename */
  gchar *json_filename; /**< "JSON file path to store the meta information */
};

/**
 * @brief GstDataRepoSinkClass data structure.
 */
struct _GstDataRepoSinkClass
{
  GstBaseSinkClass parent_class;
};

GType gst_data_repo_sink_get_type (void);

G_END_DECLS
#endif /* __GST_DATA_REPO_SINK_H__ */
