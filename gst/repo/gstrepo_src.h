/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file	gstrepo_src.h
 * @date	31 January 2023
 * @brief	GStreamer plugin to read data using NN Frameworks
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyunil Park <hyunil46.park@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_REPO_SRC_H__
#define __GST_REPO_SRC_H__

#include <sys/types.h>
#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>
#include "tensor_typedef.h"

G_BEGIN_DECLS
#define GST_TYPE_REPO_SRC \
  (gst_repo_src_get_type())
#define GST_REPO_SRC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_REPO_SRC,GstRepoSrc))
#define GST_REPO_SRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_REPO_SRC,GstRepoSrcClass))
#define GST_IS_REPO_SRC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_REPO_SRC))
#define GST_IS_REPO_SRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_REPO_SRC))

#define MAX_ITEM NNS_TENSOR_SIZE_LIMIT

typedef struct _GstRepoSrc GstRepoSrc;
typedef struct _GstRepoSrcClass GstRepoSrcClass;

/**
 * @brief GstRepoSrc data structure
 */
struct _GstRepoSrc {

  GstPushSrc parent; /**< parent object */

  gint fd;	        			  /**< open file descriptor */
  guint64 read_position;		/**< position of fd */
  guint64 offset;
  guint item_size[MAX_ITEM];

  /* property */
  gchar *filename;          /**< filename */
  guint length;             /**< buffer size */

};

struct _GstRepoSrcClass {
  GstPushSrcClass parent_calss;
};

GType gst_repo_src_get_type (void);

G_END_DECLS
#endif /* __GST_REPO_SRC_H__ */
