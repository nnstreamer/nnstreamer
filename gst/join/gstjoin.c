/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2020 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file	gstjoin.c
 * @date	10 Nov 2020
 * @brief	Select the out that arrived first among the input streams
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Gichan Jang <gichan2.jang@samsung.com>
 * @bug		No known bugs except for NYI items
 */
/**
 * SECTION:element-join
 * @see_also #GstInputSelector
 * @note A join has reduced and changed input-selector's function.
 *
 * Connect recently arrived buffer from N input streams to the output pad.
 * N streams should not operate at the same time.
 * All capabilities (input stream i and output stream) should be the same.
 * For example, If one sinkpad is receiving buffer, the others should be stopped.
 * <refsect2>
 * <title>Example launch line</title>
 * gst-launch-1.0 ... (input stream 0) ! join.sink_0 \
 *                ... (input stream 1) ! join.sink_1 \
 *                ... \
                  ... (input stream N) ! join.sink_n \
                  join name=join ! (arrived input stream) ...
 * </refsect2>
 */

#include "gstjoin.h"

GST_DEBUG_CATEGORY_STATIC (join_debug);
#define GST_CAT_DEFAULT join_debug

#define GST_JOIN_GET_LOCK(sel) (&((GstJoin*)(sel))->lock)
#define GST_JOIN_GET_COND(sel) (&((GstJoin*)(sel))->cond)
#define GST_JOIN_LOCK(sel) (g_mutex_lock (GST_JOIN_GET_LOCK(sel)))
#define GST_JOIN_UNLOCK(sel) (g_mutex_unlock (GST_JOIN_GET_LOCK(sel)))
#define GST_JOIN_WAIT(sel) (g_cond_wait (GST_JOIN_GET_COND(sel), \
			GST_JOIN_GET_LOCK(sel)))

/**
 * @brief The capabilities of the inputs
 */
static GstStaticPadTemplate gst_join_sink_factory =
GST_STATIC_PAD_TEMPLATE ("sink_%u",
    GST_PAD_SINK,
    GST_PAD_REQUEST,
    GST_STATIC_CAPS_ANY);

/**
 * @brief The capabilities of the outputs
 */
static GstStaticPadTemplate gst_join_src_factory =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

enum
{
  PROP_0,
  PROP_N_PADS,
  PROP_ACTIVE_PAD,
};

static GstPad *gst_join_get_active_sinkpad (GstJoin * sel);
static GstPad *gst_join_get_linked_pad (GstJoin * sel,
    GstPad * pad, gboolean strict);
static gboolean gst_join_set_active_pad (GstJoin * self, GstPad * pad);

#define GST_TYPE_JOIN_PAD \
  (gst_join_pad_get_type())
#define GST_JOIN_PAD(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_JOIN_PAD, GstJoinPad))
#define GST_JOIN_PAD_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_JOIN_PAD, GstJoinPadClass))
#define GST_IS_JOIN_PAD(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_JOIN_PAD))
#define GST_IS_JOIN_PAD_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_JOIN_PAD))
#define GST_JOIN_PAD_CAST(obj) \
  ((GstJoinPad *)(obj))

typedef struct _GstJoinPad GstJoinPad;
typedef struct _GstJoinPadClass GstJoinPadClass;

/**
 * @brief GstJoinPad data structure.
 */
struct _GstJoinPad
{
  GstPad parent;

  guint group_id;               /* Group ID from the last stream-start */

  GstSegment segment;           /* the current segment on the pad */
  guint32 segment_seqnum;       /* sequence number of the current segment */
};

/**
 * @brief _GstJoinPadClass data structure
 */
struct _GstJoinPadClass
{
  GstPadClass parent;
};

GType gst_join_pad_get_type (void);
static void gst_join_pad_finalize (GObject * object);
static void gst_join_pad_reset (GstJoinPad * pad);
static gboolean gst_join_pad_event (GstPad * pad, GstObject * parent,
    GstEvent * event);
static gboolean gst_join_pad_query (GstPad * pad, GstObject * parent,
    GstQuery * query);
static GstIterator *gst_join_pad_iterate_linked_pads (GstPad * pad,
    GstObject * parent);
static GstFlowReturn gst_join_pad_chain (GstPad * pad, GstObject * parent,
    GstBuffer * buf);

G_DEFINE_TYPE (GstJoinPad, gst_join_pad, GST_TYPE_PAD);

/**
 * @brief initialize the join's pad class
 */
static void
gst_join_pad_class_init (GstJoinPadClass * klass)
{
  GObjectClass *gobject_class;

  gobject_class = (GObjectClass *) klass;

  gobject_class->finalize = gst_join_pad_finalize;
}

/**
 * @brief initialize the join pad
 */
static void
gst_join_pad_init (GstJoinPad * pad)
{
  gst_join_pad_reset (pad);
}

/**
 * @brief finalize the join pad
 */
static void
gst_join_pad_finalize (GObject * object)
{
  G_OBJECT_CLASS (gst_join_pad_parent_class)->finalize (object);
}

/**
 * @brief Clear and reset join pad.
 * @note must be called with the JOIN_LOCK
 */
static void
gst_join_pad_reset (GstJoinPad * pad)
{
  GST_OBJECT_LOCK (pad);
  gst_segment_init (&pad->segment, GST_FORMAT_UNDEFINED);
  GST_OBJECT_UNLOCK (pad);
}

/**
 * @brief strictly get the linked pad from the sinkpad.
 * @return If the pad is active, return the srcpad else return NULL.
 */
static GstIterator *
gst_join_pad_iterate_linked_pads (GstPad * pad, GstObject * parent)
{
  GstJoin *sel;
  GstPad *otherpad;
  GstIterator *it = NULL;
  GValue val = { 0, };

  sel = GST_JOIN (parent);

  otherpad = gst_join_get_linked_pad (sel, pad, TRUE);
  if (otherpad) {
    g_value_init (&val, GST_TYPE_PAD);
    g_value_set_object (&val, otherpad);
    it = gst_iterator_new_single (GST_TYPE_PAD, &val);
    g_value_unset (&val);
    gst_object_unref (otherpad);
  }

  return it;
}

/**
 * @brief forward sticky event
 */
static gboolean
forward_sticky_events (GstPad * sinkpad, GstEvent ** event, gpointer user_data)
{
  GstJoin *sel = GST_JOIN (user_data);

  GST_DEBUG_OBJECT (sinkpad, "forward sticky event %" GST_PTR_FORMAT, *event);

  if (GST_EVENT_TYPE (*event) == GST_EVENT_SEGMENT) {
    GstSegment *seg = &GST_JOIN_PAD (sinkpad)->segment;
    GstEvent *e;

    e = gst_event_new_segment (seg);
    gst_event_set_seqnum (e, GST_JOIN_PAD_CAST (sinkpad)->segment_seqnum);

    gst_pad_push_event (sel->srcpad, e);
  } else if (GST_EVENT_TYPE (*event) == GST_EVENT_STREAM_START
      && !sel->have_group_id) {
    GstEvent *tmp =
        gst_pad_get_sticky_event (sel->srcpad, GST_EVENT_STREAM_START, 0);

    /* Only push stream-start once if not all our streams have a stream-id */
    if (!tmp) {
      gst_pad_push_event (sel->srcpad, gst_event_ref (*event));
    } else {
      gst_event_unref (tmp);
    }
  } else {
    gst_pad_push_event (sel->srcpad, gst_event_ref (*event));
  }
  return TRUE;
}

/**
 * @brief event function for sink pad
 */
static gboolean
gst_join_pad_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  gboolean res = TRUE;
  gboolean forward;
  GstJoin *sel;
  GstJoinPad *selpad;
  GstPad *active_sinkpad;

  sel = GST_JOIN (parent);
  selpad = GST_JOIN_PAD_CAST (pad);
  GST_DEBUG_OBJECT (selpad, "received event %" GST_PTR_FORMAT, event);

  GST_JOIN_LOCK (sel);

  active_sinkpad = gst_join_get_active_sinkpad (sel);

  /* only forward if we are dealing with the active sinkpad */
  forward = (pad == active_sinkpad);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *prev_caps, *new_caps;

      if (!(prev_caps = gst_pad_get_current_caps (active_sinkpad)))
        break;

      gst_event_parse_caps (event, &new_caps);

      if (!gst_caps_is_equal (prev_caps, new_caps)) {
        GST_ERROR_OBJECT (sel, "Capabilities of the sinks should be the same.");
        res = FALSE;
      }

      gst_caps_unref (prev_caps);

      break;
    }
    case GST_EVENT_STREAM_START:{
      if (!gst_event_parse_group_id (event, &selpad->group_id)) {
        sel->have_group_id = FALSE;
        selpad->group_id = 0;
      }
      break;
    }
    case GST_EVENT_SEGMENT:
    {
      gst_event_copy_segment (event, &selpad->segment);
      selpad->segment_seqnum = gst_event_get_seqnum (event);

      GST_DEBUG_OBJECT (pad, "configured SEGMENT %" GST_SEGMENT_FORMAT,
          &selpad->segment);
      break;
    }
    default:
      break;
  }
  GST_JOIN_UNLOCK (sel);

  if (forward) {
    GST_DEBUG_OBJECT (pad, "forwarding event");
    res = gst_pad_push_event (sel->srcpad, event);
  } else {
    gst_event_unref (event);
  }

  return res;
}

/**
 * @brief handlesink sink pad query
 * @return TRUE if the query was performed successfully.
 */
static gboolean
gst_join_pad_query (GstPad * pad, GstObject * parent, GstQuery * query)
{
  gboolean res = FALSE;
  GstJoin *self = (GstJoin *) parent;

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CAPS:
    case GST_QUERY_POSITION:
    case GST_QUERY_DURATION:
    case GST_QUERY_CONTEXT:
      /**
       * always proxy caps/position/duration/context queries, regardless of active pad or not
       * See https://bugzilla.gnome.org/show_bug.cgi?id=775445
       */
      res = gst_pad_peer_query (self->srcpad, query);
      break;
    case GST_QUERY_ALLOCATION:{
      GstPad *active_sinkpad;
      GstJoin *sel = GST_JOIN (parent);

      /**
       * Only do the allocation query for the active sinkpad,
       * after switching a reconfigure event is sent and upstream
       * should reconfigure and do a new allocation query
       */
      if (GST_PAD_DIRECTION (pad) == GST_PAD_SINK) {
        GST_JOIN_LOCK (sel);
        active_sinkpad = gst_join_get_active_sinkpad (sel);
        GST_JOIN_UNLOCK (sel);

        if (pad != active_sinkpad) {
          res = FALSE;
          goto done;
        }
      }
    }
      /* fall through */
    default:
      res = gst_pad_query_default (pad, parent, query);
      break;
  }

done:
  return res;
}

/**
 * @brief Chain function, this function does the actual processing.
 */
static GstFlowReturn
gst_join_pad_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  GstJoin *sel;
  GstFlowReturn res;
  GstPad *active_sinkpad;
  GstPad *prev_active_sinkpad = NULL;
  GstJoinPad *selpad;

  sel = GST_JOIN (parent);
  selpad = GST_JOIN_PAD_CAST (pad);
  GST_DEBUG_OBJECT (selpad,
      "entering chain for buf %p with timestamp %" GST_TIME_FORMAT, buf,
      GST_TIME_ARGS (GST_BUFFER_PTS (buf)));

  GST_JOIN_LOCK (sel);

  GST_LOG_OBJECT (pad, "getting active pad");

  prev_active_sinkpad =
      sel->active_sinkpad ? gst_object_ref (sel->active_sinkpad) : NULL;

  if (sel->active_sinkpad != pad) {
    gst_join_set_active_pad (sel, pad);
  }
  active_sinkpad = pad;

  /* update the segment on the srcpad */
  if (GST_BUFFER_PTS_IS_VALID (buf)) {
    GstClockTime start_time = GST_BUFFER_PTS (buf);

    GST_LOG_OBJECT (pad, "received start time %" GST_TIME_FORMAT,
        GST_TIME_ARGS (start_time));
    if (GST_BUFFER_DURATION_IS_VALID (buf))
      GST_LOG_OBJECT (pad, "received end time %" GST_TIME_FORMAT,
          GST_TIME_ARGS (start_time + GST_BUFFER_DURATION (buf)));

    GST_OBJECT_LOCK (pad);
    selpad->segment.position = start_time;
    GST_OBJECT_UNLOCK (pad);
  }

  GST_JOIN_UNLOCK (sel);

  /* if we have a pending events, push them now */
  if (G_UNLIKELY (prev_active_sinkpad != active_sinkpad)) {
    gst_pad_sticky_events_foreach (GST_PAD_CAST (selpad), forward_sticky_events,
        sel);
  }

  /* forward */
  GST_LOG_OBJECT (pad, "Forwarding buffer %p with timestamp %" GST_TIME_FORMAT,
      buf, GST_TIME_ARGS (GST_BUFFER_PTS (buf)));

  res = gst_pad_push (sel->srcpad, buf);
  GST_LOG_OBJECT (pad, "Buffer %p forwarded result=%d", buf, res);

  if (prev_active_sinkpad)
    gst_object_unref (prev_active_sinkpad);
  prev_active_sinkpad = NULL;

  return res;
}

static void gst_join_dispose (GObject * object);
static void gst_join_finalize (GObject * object);

static void gst_join_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static GstPad *gst_join_request_new_pad (GstElement * element,
    GstPadTemplate * templ, const gchar * unused, const GstCaps * caps);

#define gst_join_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstJoin, gst_join, GST_TYPE_ELEMENT,
    GST_DEBUG_CATEGORY_INIT (join_debug,
        "join", 0, "An input stream join element"));

/**
 * @brief initialize the join's class
 */
static void
gst_join_class_init (GstJoinClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->dispose = gst_join_dispose;
  gobject_class->finalize = gst_join_finalize;

  gobject_class->get_property = gst_join_get_property;

  g_object_class_install_property (gobject_class, PROP_N_PADS,
      g_param_spec_uint ("n-pads", "Number of Pads",
          "The number of sink pads", 0, G_MAXUINT, 0,
          G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ACTIVE_PAD,
      g_param_spec_object ("active-pad", "Active pad",
          "The currently active sink pad", GST_TYPE_PAD,
          G_PARAM_READABLE | GST_PARAM_MUTABLE_PLAYING |
          G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata (gstelement_class, "Input join",
      "Generic", "N-to-1 input stream join",
      "Gichan Jang <gichan2.jang@samsung.com>, ");

  gst_element_class_add_static_pad_template (gstelement_class,
      &gst_join_sink_factory);
  gst_element_class_add_static_pad_template (gstelement_class,
      &gst_join_src_factory);

  gstelement_class->request_new_pad = gst_join_request_new_pad;
}

/**
 * @brief initialize the join element
 */
static void
gst_join_init (GstJoin * sel)
{
  sel->srcpad = gst_pad_new ("src", GST_PAD_SRC);
  gst_pad_set_iterate_internal_links_function (sel->srcpad,
      GST_DEBUG_FUNCPTR (gst_join_pad_iterate_linked_pads));
  GST_OBJECT_FLAG_SET (sel->srcpad, GST_PAD_FLAG_PROXY_CAPS);
  gst_element_add_pad (GST_ELEMENT (sel), sel->srcpad);
  /* sinkpad management */
  sel->active_sinkpad = NULL;
  sel->padcount = 0;
  sel->have_group_id = TRUE;

  g_mutex_init (&sel->lock);
  g_cond_init (&sel->cond);
}

/**
 * @brief dispose function for join element
 */
static void
gst_join_dispose (GObject * object)
{
  GstJoin *sel = GST_JOIN (object);

  if (sel->active_sinkpad) {
    gst_object_unref (sel->active_sinkpad);
    sel->active_sinkpad = NULL;
  }
  G_OBJECT_CLASS (parent_class)->dispose (object);
}

/**
 * @brief finalize join element.
 */
static void
gst_join_finalize (GObject * object)
{
  GstJoin *sel = GST_JOIN (object);

  g_mutex_clear (&sel->lock);
  g_cond_clear (&sel->cond);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief set active sink pad.
 * @return TRUE when the active pad changed.
 * @note this function must be called with the JOIN_LOCK.
 */
static gboolean
gst_join_set_active_pad (GstJoin * self, GstPad * pad)
{
  GstJoinPad *old, *new;
  GstPad **active_pad_p;

  if (pad == self->active_sinkpad)
    return FALSE;

  /* guard against users setting a src pad or foreign pad as active pad */
  if (pad != NULL) {
    g_return_val_if_fail (GST_PAD_IS_SINK (pad), FALSE);
    g_return_val_if_fail (GST_IS_JOIN_PAD (pad), FALSE);
    g_return_val_if_fail (GST_PAD_PARENT (pad) == GST_ELEMENT_CAST (self),
        FALSE);
  }

  old = GST_JOIN_PAD_CAST (self->active_sinkpad);
  new = GST_JOIN_PAD_CAST (pad);

  GST_DEBUG_OBJECT (self, "setting active pad to %s:%s",
      GST_DEBUG_PAD_NAME (new));

  active_pad_p = &self->active_sinkpad;
  gst_object_replace ((GstObject **) active_pad_p, GST_OBJECT_CAST (pad));

  if (old && old != new)
    gst_pad_push_event (GST_PAD_CAST (old), gst_event_new_reconfigure ());
  if (new)
    gst_pad_push_event (GST_PAD_CAST (new), gst_event_new_reconfigure ());

  GST_DEBUG_OBJECT (self, "New active pad is %" GST_PTR_FORMAT,
      self->active_sinkpad);

  return TRUE;
}

/**
 * @brief Getter for join properties.
 */
static void
gst_join_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstJoin *sel = GST_JOIN (object);

  switch (prop_id) {
    case PROP_N_PADS:
      GST_JOIN_LOCK (object);
      g_value_set_uint (value, sel->n_pads);
      GST_JOIN_UNLOCK (object);
      break;
    case PROP_ACTIVE_PAD:
      GST_JOIN_LOCK (object);
      g_value_set_object (value, sel->active_sinkpad);
      GST_JOIN_UNLOCK (object);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Get linked pad
 */
static GstPad *
gst_join_get_linked_pad (GstJoin * sel, GstPad * pad, gboolean strict)
{
  GstPad *otherpad = NULL;

  GST_JOIN_LOCK (sel);
  if (pad == sel->srcpad)
    otherpad = sel->active_sinkpad;
  else if (pad == sel->active_sinkpad || !strict)
    otherpad = sel->srcpad;
  if (otherpad)
    gst_object_ref (otherpad);
  GST_JOIN_UNLOCK (sel);

  return otherpad;
}

/**
 * @brief Get or create the active sinkpad.
 * @note  must be called with JOIN_LOCK.
 */
static GstPad *
gst_join_get_active_sinkpad (GstJoin * sel)
{
  GstPad *active_sinkpad;

  active_sinkpad = sel->active_sinkpad;
  if (active_sinkpad == NULL) {
    GValue item = G_VALUE_INIT;
    GstIterator *iter = gst_element_iterate_sink_pads (GST_ELEMENT_CAST (sel));
    GstIteratorResult ires;

    while ((ires = gst_iterator_next (iter, &item)) == GST_ITERATOR_RESYNC)
      gst_iterator_resync (iter);
    if (ires == GST_ITERATOR_OK) {
      /**
       * If no pad is currently selected, we return the first usable pad to
       * guarantee consistency
       */

      active_sinkpad = sel->active_sinkpad = g_value_dup_object (&item);
      g_value_reset (&item);
      GST_DEBUG_OBJECT (sel, "Activating pad %s:%s",
          GST_DEBUG_PAD_NAME (active_sinkpad));
    } else
      GST_WARNING_OBJECT (sel, "Couldn't find a default sink pad");
    gst_iterator_free (iter);
  }

  return active_sinkpad;
}

/**
 * @brief request new sink pad
 */
static GstPad *
gst_join_request_new_pad (GstElement * element, GstPadTemplate * templ,
    const gchar * unused, const GstCaps * caps)
{
  GstJoin *sel;
  gchar *name = NULL;
  GstPad *sinkpad = NULL;
  (void) unused;
  (void) caps;

  g_return_val_if_fail (templ->direction == GST_PAD_SINK, NULL);

  sel = GST_JOIN (element);

  GST_JOIN_LOCK (sel);

  GST_LOG_OBJECT (sel, "Creating new pad sink_%u", sel->padcount);
  name = g_strdup_printf ("sink_%u", sel->padcount++);
  sinkpad = g_object_new (GST_TYPE_JOIN_PAD,
      "name", name, "direction", templ->direction, "template", templ, NULL);
  g_free (name);

  sel->n_pads++;

  gst_pad_set_event_function (sinkpad, GST_DEBUG_FUNCPTR (gst_join_pad_event));
  gst_pad_set_query_function (sinkpad, GST_DEBUG_FUNCPTR (gst_join_pad_query));
  gst_pad_set_chain_function (sinkpad, GST_DEBUG_FUNCPTR (gst_join_pad_chain));
  gst_pad_set_iterate_internal_links_function (sinkpad,
      GST_DEBUG_FUNCPTR (gst_join_pad_iterate_linked_pads));

  GST_OBJECT_FLAG_SET (sinkpad, GST_PAD_FLAG_PROXY_CAPS);
  GST_OBJECT_FLAG_SET (sinkpad, GST_PAD_FLAG_PROXY_ALLOCATION);
  gst_pad_set_active (sinkpad, TRUE);
  gst_element_add_pad (GST_ELEMENT (sel), sinkpad);
  GST_JOIN_UNLOCK (sel);

  return sinkpad;
}

/**
 * @brief register this element
 */
static gboolean
plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, "join", 0,
      "gstreamer join element");

  if (!gst_element_register (plugin, "join", GST_RANK_NONE, GST_TYPE_JOIN)) {
    return FALSE;
  }

  return TRUE;
}

#ifndef PACKAGE
#define PACKAGE "join"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    join,
    "Select the out that arrived first among the input streams",
    plugin_init, VERSION, "LGPL", PACKAGE,
    "https://github.com/nnstreamer/nnstreamer")
