/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2020 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file	gstjoin.h
 * @date	10 Nov 2020
 * @brief	Select the out that arrived first among the input streams
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Gichan Jang <gichan2.jang@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_JOIN_H__
#define __GST_JOIN_H__

#include <gst/gst.h>

G_BEGIN_DECLS
#define GST_TYPE_JOIN \
  (gst_join_get_type())
#define GST_JOIN(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_JOIN, GstJoin))
#define GST_JOIN_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_JOIN, GstJoinClass))
#define GST_IS_JOIN(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_JOIN))
#define GST_IS_JOIN_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_JOIN))
typedef struct _GstJoin GstJoin;
typedef struct _GstJoinClass GstJoinClass;

#define GST_JOIN_GET_LOCK(sel) (&((GstJoin*)(sel))->lock)
#define GST_JOIN_GET_COND(sel) (&((GstJoin*)(sel))->cond)
#define GST_JOIN_LOCK(sel) (g_mutex_lock (GST_JOIN_GET_LOCK(sel)))
#define GST_JOIN_UNLOCK(sel) (g_mutex_unlock (GST_JOIN_GET_LOCK(sel)))
#define GST_JOIN_WAIT(sel) (g_cond_wait (GST_JOIN_GET_COND(sel), \
			GST_JOIN_GET_LOCK(sel)))
#define GST_JOIN_BROADCAST(sel) (g_cond_broadcast (GST_JOIN_GET_COND(sel)))

/**
 * @brief Internal data structure for join instances.
 */
struct _GstJoin
{
  GstElement element;

  GstPad *srcpad;

  GstPad *active_sinkpad;
  guint n_pads;                 /* number of pads */
  guint padcount;               /* sequence number for pads */

  gboolean have_group_id;

  GMutex lock;
  GCond cond;
  gboolean eos;
  gboolean eos_sent;
};

/**
 * @brief GstJoinClass inherits GstElementClass.
 */
struct _GstJoinClass
{
  GstElementClass parent_class;
};

/**
 * @brief Get Type function required for gst elements
 */
G_GNUC_INTERNAL GType gst_join_get_type (void);

G_END_DECLS
#endif /* __GST_JOIN_H__ */
