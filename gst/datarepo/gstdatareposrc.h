/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file	gstdatareposrc.h
 * @date	31 January 2023
 * @brief	GStreamer plugin to read file in MLOps Data repository into buffers
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyunil Park <hyunil46.park@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_DATA_REPO_SRC_H__
#define __GST_DATA_REPO_SRC_H__

#include <sys/types.h>
#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>
#include <json-glib/json-glib.h>
#include <tensor_typedef.h>
#include "gstdatarepo.h"

G_BEGIN_DECLS
#define GST_TYPE_DATA_REPO_SRC \
  (gst_data_repo_src_get_type())
#define GST_DATA_REPO_SRC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DATA_REPO_SRC,GstDataRepoSrc))
#define GST_DATA_REPO_SRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DATA_REPO_SRC,GstDataRepoSrcClass))
#define GST_IS_DATA_REPO_SRC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DATA_REPO_SRC))
#define GST_IS_DATA_REPO_SRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DATA_REPO_SRC))

typedef struct _GstDataRepoSrc GstDataRepoSrc;
typedef struct _GstDataRepoSrcClass GstDataRepoSrcClass;

/**
 * @brief GstDataRepoSrc data structure
 */
struct _GstDataRepoSrc {

  GstPushSrc parent;            /**< parent object */

  gboolean is_start;            /**< check if datareposrc is started */
  gboolean successful_read;     /**< used for checking EOS when reading more than one images(multi-files) from a path */
  gint fd;                      /**< open file descriptor */
  gint file_size;               /**< file size, in bytes */
  guint64 read_position;        /**< position of fd */
  guint64 fd_offset;            /**< offset of fd */
  guint64 start_offset;         /**< start offset to read */
  guint64 last_offset;          /**< last offset to read */
  gsize tensors_size[NNS_TENSOR_SIZE_LIMIT];   /**< each tensors size in a sample */
  gsize tensors_offset[NNS_TENSOR_SIZE_LIMIT]; /**< each tensors offset in a sample */
  gint current_sample_index;    /**< current index of sample or file to read */
  gboolean first_epoch_is_done;
  guint total_samples;           /**< The number of total samples */
  guint num_samples;             /**< The number of samples to be used out of the total samples in the file */
  gsize sample_size;             /**< size of one sample */
  GstDataRepoDataType data_type; /**< media type */

  /* property */
  gchar *filename;              /**< filename */
  gchar *json_filename;         /**< json filename containing meta information of the filename */
  gchar *tensors_seq_str;       /**< tensors in a sample are read into gstBuffer according to tensors_sequence */
  guint start_sample_index;     /**< start index of sample to read, in case of image, the starting index of the numbered files */
  guint stop_sample_index;      /**< stop index of sample to read, in case of image, the stoppting index of the numbered files */
  guint epochs;                 /**< repetition of range of files or samples to read */
  gboolean is_shuffle;          /**< shuffle the sample index */

  GArray *shuffled_index_array; /**< shuffled sample index array */
  guint array_index;            /**< element index of shuffled_index_array */

  guint tensors_seq[NNS_TENSOR_SIZE_LIMIT];  /**< tensors sequence in a sample that will be read into gstbuffer */
  guint tensors_seq_cnt;
  gboolean need_changed_caps;   /**< When tensors-sequence changes, caps need to be changed */
  GstCaps *caps;                /**< optional property, datareposrc should get data format from JSON file caps field */

  /* flexible tensors */
  GstTensorsConfig config;          /**< tensors information from current caps */
  JsonArray *sample_offset_array;   /**< offset array of sample */
  JsonArray *tensor_size_array;     /**< size array of flexible tensor to be stored in a Gstbuffer */
  JsonArray *tensor_count_array;    /**< array for the number of cumulative tensors */
  JsonParser *parser;               /**< Keep JSON data after parsing JSON file */
  guint sample_offset_array_len;
  guint tensor_size_array_len;
  guint tensor_count_array_len;

  GstClockTime running_time;    /**< one frame running time */
  gint rate_n, rate_d;
  guint64 n_frame;
};

/**
 * @brief GstDataRepoSrcClass data structure.
 */
struct _GstDataRepoSrcClass {
  GstPushSrcClass parent_class;
};

GType gst_data_repo_src_get_type (void);

G_END_DECLS
#endif /* __GST_DATA_REPO_SRC_H__ */
