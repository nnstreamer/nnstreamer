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
#include <tensor_typedef.h>

G_BEGIN_DECLS
#define GST_TYPE_DATA_REPO_SRC \
  (gst_data_repo_src_get_type())
#define GST_DATA_REPO_SRC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DATA_REPO_SRC,GstDataRepoSrc))
#define GST_DATA_REPO_SRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DATA_REPO_SRC,GstRepoSrcClass))
#define GST_IS_DATA_REPO_SRC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DATA_REPO_SRC))
#define GST_IS_DATA_REPO_SRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DATA_REPO_SRC))

/* media_type has not IMAGE type */
#define _NNS_IMAGE 5  /**<supposedly image/png, image/jpeg and etc */

#define MAX_ITEM NNS_TENSOR_SIZE_LIMIT

typedef struct _GstDataRepoSrc GstDataRepoSrc;
typedef struct _GstDataRepoSrcClass GstDataRepoSrcClass;

/**
 * @brief GstDataRepoSrc data structure
 */
struct _GstDataRepoSrc {

  GstPushSrc parent; /**< parent object */

  gboolean successful_read; /**< Used for checking EOS when reading more than one images(multi-files) from a path */
  gint fd;	        			  /**< open file descriptor */
  guint64 read_position;		/**< position of fd */
  guint64 offset;
  guint tensors_size[MAX_ITEM];   /**< For other/tensors media type, each tensor size is stored */

  gint start_frame_index;   /**< Start index of sample to read, in case of image, the starting index of the numbered files */
  gint stop_frame_index;    /**< End index of sample to read, in case of image, the endig index of the numbered files */
  gint frame_index;         /**< Current index of sample or file to read */

  /* property */
  gchar *filename;          /**< filename */
  guint media_size;         /**< media size */
  guint media_type;         /**< media type */

};

/**
 * @brief GstDataRepoSrcClass data structure.
 */
struct _GstDataRepoSrcClass {
  GstPushSrcClass parent_calss;
};

GType gst_data_repo_src_get_type (void);

G_END_DECLS
#endif /* __GST_DATA_REPO_SRC_H__ */
