/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file tizen-api-pipeline.c
 * @date 11 March 2019
 * @brief Tizen NNStreamer/Pipeline(main) C-API Wrapper.
 *        This allows to construct and control NNStreamer pipelines.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <glib.h>
#include <glib-object.h>        /* Get GType from GObject Instances */
#include <gmodule.h>
#include <tizen-api-private.h>
#include <dlog.h>
#include <nnstreamer/tensor_typedef.h>
#include <nnstreamer/nnstreamer_plugin_api.h>

#include <gst/gstbuffer.h>
#include <gst/app/app.h>        /* To push data to pipeline */

#define DLOG_TAG "nnstreamer-capi-pipeline"

#define handle_init(type, name, h) \
  nns_##type *name= (h); \
  nns_pipeline *p; \
  element *elem; \
  int ret = NNS_ERROR_NONE; \
  if (h == NULL) { \
    dlog_print (DLOG_ERROR, DLOG_TAG, \
        "The given handle is invalid"); \
    return NNS_ERROR_INVALID_PARAMETER; \
  } \
\
  p = name->pipe; \
  elem = name->element; \
  if (p == NULL || elem == NULL || p != elem->pipe) { \
    dlog_print (DLOG_ERROR, DLOG_TAG, \
        "The handle appears to be broken."); \
    return NNS_ERROR_INVALID_PARAMETER; \
  } \
\
  g_mutex_lock (&p->lock); \
  g_mutex_lock (&elem->lock); \
\
  if (NULL == g_list_find (elem->handles, name)) { \
    dlog_print (DLOG_ERROR, DLOG_TAG, \
        "The handle does not exists."); \
    ret = NNS_ERROR_INVALID_PARAMETER; \
    goto unlock_return; \
  }

#define handle_exit(h) \
unlock_return: \
  g_mutex_unlock (&elem->lock); \
  g_mutex_unlock (&p->lock); \
  return ret;

/**
 * @brief Internal function to create a refereable element in a pipeline
 */
static element *
construct_element (GstElement * e, nns_pipeline * p, const char *name,
    elementType t)
{
  element *ret = g_new0 (element, 1);
  ret->element = e;
  ret->pipe = p;
  ret->name = g_strdup (name);
  ret->type = t;
  ret->handles = NULL;
  ret->src = NULL;
  ret->sink = NULL;
  gst_tensors_info_init (&ret->tensorsinfo);
  ret->size = 0;
  ret->maxid = 0;
  g_mutex_init (&ret->lock);
  return ret;
}

/**
 * @brief Internal function to convert GstTensorsInfo into nns_tensors_info_s structure.
 */
static int
get_tensors_info_from_GstTensorsInfo (GstTensorsInfo * gst_tensorsinfo,
    nns_tensors_info_s * tensors_info)
{
  if (!gst_tensorsinfo) {
    dlog_print (DLOG_ERROR, DLOG_TAG, "GstTensorsInfo should not be NULL!");
    return NNS_ERROR_INVALID_PARAMETER;
  }

  /** Currently, the data structures of GstTensorsInfo are
   * completely same as that of nns_tensors_info_s. */
  memcpy (tensors_info, gst_tensorsinfo, sizeof (GstTensorsInfo));

  return NNS_ERROR_NONE;
}

/**
 * @brief Handle a sink element for registered nns_sink_cb
 */
static void
cb_sink_event (GstElement * e, GstBuffer * b, gpointer data)
{
  element *elem = data;

  /** @todo CRITICAL if the pipeline is being killed, don't proceed! */

  GstMemory *mem[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo info[NNS_TENSOR_SIZE_LIMIT];
  guint i;
  guint num_mems;
  GList *l;
  const char *buf[NNS_TENSOR_SIZE_LIMIT];
  size_t size[NNS_TENSOR_SIZE_LIMIT];
  size_t total_size = 0;

  num_mems = gst_buffer_n_memory (b);

  if (num_mems > NNS_TENSOR_SIZE_LIMIT) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "Number of memory chunks in a GstBuffer exceed the limit: %u > %u",
        num_mems, NNS_TENSOR_SIZE_LIMIT);
    return;
  }

  for (i = 0; i < num_mems; i++) {
    mem[i] = gst_buffer_peek_memory (b, i);
    gst_memory_map (mem[i], &info[i], GST_MAP_READ);
    buf[i] = (const char *) info[i].data;
    size[i] = info[i].size;
    total_size += size[i];
  }

  g_mutex_lock (&elem->lock);

  /** @todo This assumes that padcap is static */
  if (elem->sink == NULL) {
    /* Get the sink-pad-cap */
    elem->sink = gst_element_get_static_pad (elem->element, "sink");

    if (elem->sink) {
      /* sinkpadcap available (negotiated) */
      GstCaps *caps = gst_pad_get_current_caps (elem->sink);

      if (caps) {
        guint n_caps = gst_caps_get_size (caps);
        GstTensorsConfig tconfig;
        gboolean found = FALSE;

        for (i = 0; i < n_caps; i++) {
          GstStructure *s = gst_caps_get_structure (caps, i);

          found = gst_tensors_config_from_structure (&tconfig, s);
          if (found)
            break;
        }

        if (found) {
          memcpy (&elem->tensorsinfo, &tconfig.info, sizeof (GstTensorsInfo));
          elem->size = 0;

          if (elem->tensorsinfo.num_tensors != num_mems) {
            dlog_print (DLOG_ERROR, DLOG_TAG,
                "The sink event of [%s] cannot be handled because the tensor type mismatches.",
                elem->name);
            gst_caps_unref (caps);
            elem->sink = NULL;
            g_mutex_unlock (&elem->lock);

            return;
          }

          for (i = 0; i < elem->tensorsinfo.num_tensors; i++) {
            size_t sz = gst_tensor_info_get_size (&elem->tensorsinfo.info[i]);

            if (sz != size[i]) {
              dlog_print (DLOG_ERROR, DLOG_TAG,
                  "The sink event of [%s] cannot be handled because the tensor dimension mismatches.",
                  elem->name);
              gst_caps_unref (caps);
              elem->sink = NULL;
              g_mutex_unlock (&elem->lock);

              return;
            }

            elem->size += sz;
          }
        } else {
          elem->sink = NULL;    /* It is not valid */
          /** @todo What if it keeps being "NULL"? Exception handling at 2nd frame? */
        }

        gst_caps_unref (caps);
      }
    }
  }

  /* Get the data! */
  if (gst_buffer_get_size (b) != total_size ||
      (elem->size > 0 && total_size != elem->size)) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "The buffersize mismatches. All the three values must be the same: %zu, %zu, %zu",
        total_size, elem->size, gst_buffer_get_size (b));
    g_mutex_unlock (&elem->lock);
    return;
  }

  /* Iterate e->handles, pass the data to them */
  for (l = elem->handles; l != NULL; l = l->next) {
    nns_tensors_info_s tensors_info;
    nns_sink *sink = l->data;
    nns_sink_cb callback = sink->cb;

    get_tensors_info_from_GstTensorsInfo (&elem->tensorsinfo, &tensors_info);

    callback (buf, size, &tensors_info, sink->pdata);

    /** @todo Measure time. Warn if it takes long. Kill if it takes too long. */
  }

  g_mutex_unlock (&elem->lock);

  for (i = 0; i < num_mems; i++) {
    gst_memory_unmap (mem[i], &info[i]);
  }

  return;
}

/**
 * @brief Private function for nns_pipeline_destroy, cleaning up nodes in namednodes
 */
static void
cleanup_node (gpointer data)
{
  element *e = data;
  g_mutex_lock (&e->lock);
  g_free (e->name);
  if (e->src)
    gst_object_unref (e->src);
  if (e->sink)
    gst_object_unref (e->sink);

  /** @todo CRITICAL. Stop the handle callbacks if they are running/ready */
  if (e->handles)
    g_list_free_full (e->handles, g_free);
  e->handles = NULL;

  g_mutex_unlock (&e->lock);
  g_mutex_clear (&e->lock);

  g_free (e);
}

/**
 * @brief Construct the pipeline (more info in nnstreamer.h)
 */
int
nns_pipeline_construct (const char *pipeline_description, nns_pipeline_h * pipe)
{
  GError *err = NULL;
  GstElement *pipeline;
  GstIterator *it = NULL;
  int ret = NNS_ERROR_NONE;

  nns_pipeline *pipe_h;

  if (FALSE == gst_init_check (NULL, NULL, &err)) {
    if (err) {
      dlog_print (DLOG_ERROR, DLOG_TAG,
          "Gstreamer has the following error: %s", err->message);
      g_error_free (err);
    } else {
      dlog_print (DLOG_ERROR, DLOG_TAG,
          "Cannot initialize gstreamer. Unknown reason.");
    }
    return NNS_ERROR_STREAMS_PIPE;
  }

  pipeline = gst_parse_launch (pipeline_description, &err);
  if (pipeline == NULL || err) {
    if (err) {
      dlog_print (DLOG_ERROR, DLOG_TAG,
          "Cannot parse and launch the given pipeline = [%s], %s",
          pipeline_description, err->message);
      g_error_free (err);
    } else {
      dlog_print (DLOG_ERROR, DLOG_TAG,
          "Cannot parse and launch the given pipeline = [%s], unknown reason",
          pipeline_description);
    }
    return NNS_ERROR_STREAMS_PIPE;
  }

  pipe_h = g_new0 (nns_pipeline, 1);
  *pipe = pipe_h;
  g_assert (GST_IS_PIPELINE (pipeline));
  pipe_h->element = pipeline;
  g_mutex_init (&pipe_h->lock);
  g_mutex_lock (&pipe_h->lock);

  pipe_h->namednodes =
      g_hash_table_new_full (g_str_hash, g_str_equal, g_free, cleanup_node);

  it = gst_bin_iterate_elements (GST_BIN (pipeline));
  if (it != NULL) {
    gboolean done = FALSE;
    GValue item = G_VALUE_INIT;
    GObject *obj;
    gchar *name;
    GstElementFactory *tensor_sink = gst_element_factory_find ("tensor_sink");
    GstElementFactory *valve = gst_element_factory_find ("valve");
    GstElementFactory *inputs = gst_element_factory_find ("input-selector");
    GstElementFactory *outputs = gst_element_factory_find ("output-selector");

    /* Fill in the hashtable, "namednodes" with named Elements */
    while (!done) {
      switch (gst_iterator_next (it, &item)) {
        case GST_ITERATOR_OK:
          obj = g_value_get_object (&item);

          if (GST_IS_ELEMENT (obj)) {
            GstElement *elem = GST_ELEMENT (obj);
            name = gst_element_get_name (elem);
            if (name != NULL) {
              element *e = NULL;

              if (G_TYPE_CHECK_INSTANCE_TYPE (elem,
                      gst_element_factory_get_element_type (tensor_sink))) {
                e = construct_element (elem, pipe_h, name, NNSAPI_SINK);
                g_object_set (elem, "emit-signal", (gboolean) TRUE, NULL);
                g_signal_connect (elem, "new-data", (GCallback) cb_sink_event,
                    e);
              } else if (G_TYPE_CHECK_INSTANCE_TYPE (elem, GST_TYPE_APP_SRC)) {
                e = construct_element (elem, pipe_h, name, NNSAPI_SRC);
              } else if (G_TYPE_CHECK_INSTANCE_TYPE (elem,
                      gst_element_factory_get_element_type (valve))) {
                e = construct_element (elem, pipe_h, name, NNSAPI_VALVE);
              } else if (G_TYPE_CHECK_INSTANCE_TYPE (elem,
                      gst_element_factory_get_element_type (inputs))) {
                e = construct_element (elem, pipe_h, name, NNSAPI_SWITCH_INPUT);
              } else if (G_TYPE_CHECK_INSTANCE_TYPE (elem,
                      gst_element_factory_get_element_type (outputs))) {
                e = construct_element (elem, pipe_h, name,
                    NNSAPI_SWITCH_OUTPUT);
              } else {
                /** @todo CRITICAL HANDLE THIS! */
              }

              if (e != NULL)
                g_hash_table_insert (pipe_h->namednodes, g_strdup (name), e);
            }
            g_free (name);
          }
          g_value_reset (&item);

          break;
        case GST_ITERATOR_RESYNC:
        case GST_ITERATOR_ERROR:
          dlog_print (DLOG_WARN, DLOG_TAG,
              "There is an error or a resync-event while inspecting a pipeline. However, we can still execute the pipeline.");
        case GST_ITERATOR_DONE:
          done = TRUE;
      }
    }
    g_value_unset (&item);
    /** @todo CRITICAL check the validity of elem=item registered in e */
    gst_iterator_free (it);

    g_object_unref (tensor_sink);
    g_object_unref (valve);
    g_object_unref (inputs);
    g_object_unref (outputs);
  }

  g_mutex_unlock (&pipe_h->lock);
  return ret;
}

/**
 * @brief Destroy the pipeline (more info in nnstreamer.h)
 */
int
nns_pipeline_destroy (nns_pipeline_h pipe)
{
  nns_pipeline *p = pipe;
  GstStateChangeReturn scret;
  GstState state, pending;

  if (p == NULL)
    return NNS_ERROR_INVALID_PARAMETER;

  g_mutex_lock (&p->lock);

  /* if it's PLAYING, PAUSE it. */
  scret = gst_element_get_state (p->element, &state, &pending, 10000000UL);     /* 10ms */
  if (scret != GST_STATE_CHANGE_FAILURE && state == GST_STATE_PLAYING) {
    /* Pause the pipeline if it's Playing */
    scret = gst_element_set_state (p->element, GST_STATE_PAUSED);
    if (scret == GST_STATE_CHANGE_FAILURE) {
      g_mutex_unlock (&p->lock);
      return NNS_ERROR_STREAMS_PIPE;
    }
  }

  /** @todo Ensure all callbacks are gone. (kill'em all!) THIS IS CRITICAL! */
  g_mutex_unlock (&p->lock);
  g_usleep (50000);             /* do 50ms sleep until we have it implemented. Let them complete. And hope they don't call start(). */
  g_mutex_lock (&p->lock);

  /** Destroy registered callback handles */
  g_hash_table_remove_all (p->namednodes);

  /** Stop (NULL State) the pipeline */
  scret = gst_element_set_state (p->element, GST_STATE_NULL);
  if (scret != GST_STATE_CHANGE_SUCCESS) {
    g_mutex_unlock (&p->lock);
    return NNS_ERROR_STREAMS_PIPE;
  }

  gst_object_unref (p->element);

  g_mutex_unlock (&p->lock);
  g_mutex_clear (&p->lock);
  return NNS_ERROR_NONE;
}

/**
 * @brief Get the pipeline state (more info in nnstreamer.h)
 */
int
nns_pipeline_getstate (nns_pipeline_h pipe, nns_pipeline_state_e * state)
{
  nns_pipeline *p = pipe;
  GstState _state;
  GstState pending;
  GstStateChangeReturn scret;
  *state = NNS_PIPELINE_UNKNOWN;

  if (p == NULL)
    return NNS_ERROR_INVALID_PARAMETER;

  g_mutex_lock (&p->lock);
  scret = gst_element_get_state (p->element, &_state, &pending, 100000UL);      /* Do it within 100us! */
  g_mutex_unlock (&p->lock);

  if (scret == GST_STATE_CHANGE_FAILURE)
    return NNS_ERROR_STREAMS_PIPE;

  *state = _state;
  return NNS_ERROR_NONE;
}

/****************************************************
 ** NNStreamer Pipeline Start/Stop Control         **
 ****************************************************/
/**
 * @brief Start/Resume the pipeline! (more info in nnstreamer.h)
 */
int
nns_pipeline_start (nns_pipeline_h pipe)
{
  nns_pipeline *p = pipe;
  GstStateChangeReturn scret;

  g_mutex_lock (&p->lock);
  scret = gst_element_set_state (p->element, GST_STATE_PLAYING);
  g_mutex_unlock (&p->lock);

  if (scret == GST_STATE_CHANGE_FAILURE)
    return NNS_ERROR_STREAMS_PIPE;

  return NNS_ERROR_NONE;
}

/**
 * @brief Pause the pipeline! (more info in nnstreamer.h)
 */
int
nns_pipeline_stop (nns_pipeline_h pipe)
{
  nns_pipeline *p = pipe;
  GstStateChangeReturn scret;

  g_mutex_lock (&p->lock);
  scret = gst_element_set_state (p->element, GST_STATE_PAUSED);
  g_mutex_unlock (&p->lock);

  if (scret == GST_STATE_CHANGE_FAILURE)
    return NNS_ERROR_STREAMS_PIPE;

  return NNS_ERROR_NONE;
}

/****************************************************
 ** NNStreamer Pipeline Sink/Src Control           **
 ****************************************************/
/**
 * @brief Register a callback for sink (more info in nnstreamer.h)
 */
int
nns_pipeline_sink_register (nns_pipeline_h pipe, const char *sinkname,
    nns_sink_cb cb, nns_sink_h * h, void *pdata)
{
  element *elem;
  nns_pipeline *p = pipe;
  nns_sink *sink;
  int ret = NNS_ERROR_NONE;

  if (pipe == NULL) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "The first argument, pipeline handle is not valid.");
    return NNS_ERROR_INVALID_PARAMETER;
  }

  if (cb == NULL) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "The callback argument, cb, is not valid.");
    return NNS_ERROR_INVALID_PARAMETER;
  }

  g_mutex_lock (&p->lock);
  elem = g_hash_table_lookup (p->namednodes, sinkname);

  if (elem == NULL) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "There is no element named [%s] in the pipeline.", sinkname);
    ret = NNS_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  if (elem->type != NNSAPI_SINK) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "The element [%s] in the pipeline is not a tensor_sink.", sinkname);
    ret = NNS_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  *h = g_new0 (nns_sink, 1);
  sink = *h;

  sink->pipe = p;
  sink->element = elem;
  sink->cb = cb;
  sink->pdata = pdata;

  g_mutex_lock (&elem->lock);

  elem->maxid++;
  sink->id = elem->maxid;
  elem->handles = g_list_append (elem->handles, sink);

  g_mutex_unlock (&elem->lock);

unlock_return:
  g_mutex_unlock (&p->lock);
  return ret;
}

/**
 * @brief Unregister a callback for sink (more info in nnstreamer.h)
 */
int
nns_pipeline_sink_unregister (nns_sink_h h)
{
  handle_init (sink, sink, h);

  elem->handles = g_list_remove (elem->handles, sink);

  handle_exit (h);
}

/**
 * @brief Implementation of policies decalred by nns_buf_policy_e in nnstreamer.h,
 *        "Free"
 */
static void
nnsbufpolicy_free (gpointer data)
{
  g_free (data);
}

/**
 * @brief Implementation of policies decalred by nns_buf_policy_e in nnstreamer.h.
 *        "Do Nothing"
 */
static void
nnsbufpolicy_nop (gpointer data)
{
  /* DO NOTHING! */
}

/**
 * @brief Implementation of policies decalred by nns_buf_policy_e in nnstreamer.h.
 */
static const GDestroyNotify bufpolicy[NNS_BUF_POLICY_MAX] = {
  [NNS_BUF_FREE_BY_NNSTREAMER] = nnsbufpolicy_free,
  [NNS_BUF_DO_NOT_FREE1] = nnsbufpolicy_nop,
};

/**
 * @brief Get a handle to operate a src (more info in nnstreamer.h)
 */
int nns_pipeline_src_gethandle
    (nns_pipeline_h pipe, const char *srcname,
    nns_tensors_info_s * tensors_info, nns_src_h * h)
{
  nns_pipeline *p = pipe;
  element *elem;
  nns_src *src;
  int ret = NNS_ERROR_NONE, i;

  if (pipe == NULL) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "The first argument, pipeline handle is not valid.");
    return NNS_ERROR_INVALID_PARAMETER;
  }

  g_mutex_lock (&p->lock);

  elem = g_hash_table_lookup (p->namednodes, srcname);

  if (elem == NULL) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "There is no element named [%s] in the pipeline.", srcname);
    ret = NNS_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  if (elem->type != NNSAPI_SRC) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "The element [%s] in the pipeline is not a tensor_src.", srcname);
    ret = NNS_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  if (elem->src == NULL)
    elem->src = gst_element_get_static_pad (elem->element, "src");

  if (elem->src != NULL) {
    /** @todo : refactor this along with nns_pipeline_src_inputdata */
    GstCaps *caps = gst_pad_get_allowed_caps (elem->src);

    /** @todo caps may be NULL for prerolling */
    if (caps == NULL) {
      dlog_print (DLOG_WARN, DLOG_TAG,
          "Cannot find caps. The pipeline is not yet negotiated for tensor_src, [%s].",
          srcname);
    } else {
      guint n_caps = gst_caps_get_size (caps);
      GstTensorsConfig tconfig;
      gboolean found = FALSE;

      for (i = 0; i < n_caps; i++) {
        GstStructure *s = gst_caps_get_structure (caps, i);

        found = gst_tensors_config_from_structure (&tconfig, s);
        if (found)
          break;
      }

      if (found) {
        memcpy (&elem->tensorsinfo, &tconfig.info, sizeof (GstTensorsInfo));
        elem->size = 0;
        for (i = 0; i < elem->tensorsinfo.num_tensors; i++) {
          size_t sz = gst_tensor_info_get_size (&elem->tensorsinfo.info[i]);
          elem->size += sz;
        }
      }
    }
  } else {
    ret = NNS_ERROR_STREAMS_PIPE;
    goto unlock_return;
  }

  ret = get_tensors_info_from_GstTensorsInfo (&elem->tensorsinfo, tensors_info);
  if (ret != NNS_ERROR_NONE)
    goto unlock_return;

  *h = g_new (nns_src, 1);
  src = *h;

  src->pipe = p;
  src->element = elem;

  g_mutex_lock (&elem->lock);

  elem->maxid++;
  src->id = elem->maxid;
  elem->handles = g_list_append (elem->handles, src);

  g_mutex_unlock (&elem->lock);

unlock_return:
  g_mutex_unlock (&p->lock);

  return ret;
}

/**
 * @brief Close a src node (more info in nnstreamer.h)
 */
int
nns_pipeline_src_puthandle (nns_src_h h)
{
  handle_init (src, src, h);

  elem->handles = g_list_remove (elem->handles, src);

  handle_exit (h);
}

/**
 * @brief Push a data frame to a src (more info in nnstreamer.h)
 */
int
nns_pipeline_src_inputdata (nns_src_h h,
    nns_buf_policy_e policy, char *buf[], const size_t size[],
    unsigned int num_tensors)
{
  /** @todo NYI */
  GstBuffer *buffer;
  GstFlowReturn gret;
  unsigned int i;

  handle_init (src, src, h);

  if (num_tensors < 1 || num_tensors > NNS_TENSOR_SIZE_LIMIT) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "The tensor size if invalid. It should be 1 ~ %u; where it is %u",
        NNS_TENSOR_SIZE_LIMIT, num_tensors);
    ret = NNS_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  /** @todo This assumes that padcap is static */
  if (elem->src == NULL) {
    /* Get the src-pad-cap */
    elem->src = gst_element_get_static_pad (elem->element, "src");
  }

  if (elem->src != NULL && elem->size == 0) {
    /* srcpadcap available (negoticated) */
    GstCaps *caps = gst_pad_get_allowed_caps (elem->src);

    if (caps) {
      guint n_caps = gst_caps_get_size (caps);
      GstTensorsConfig tconfig;
      gboolean found = FALSE;

      for (i = 0; i < n_caps; i++) {
        GstStructure *s = gst_caps_get_structure (caps, i);

        found = gst_tensors_config_from_structure (&tconfig, s);
        if (found)
          break;
      }

      if (found) {
        memcpy (&elem->tensorsinfo, &tconfig.info, sizeof (GstTensorsInfo));
        elem->size = 0;

        if (elem->tensorsinfo.num_tensors != num_tensors) {
          dlog_print (DLOG_ERROR, DLOG_TAG,
              "The src push of [%s] cannot be handled because the number of tensors in a frame mismatches. %u != %u",
              elem->name, elem->tensorsinfo.num_tensors, num_tensors);
          gst_caps_unref (caps);
          elem->sink = NULL;
          ret = NNS_ERROR_STREAMS_PIPE;
          goto unlock_return;
        }

        for (i = 0; i < elem->tensorsinfo.num_tensors; i++) {
          size_t sz = gst_tensor_info_get_size (&elem->tensorsinfo.info[i]);

          if (sz != size[i]) {
            dlog_print (DLOG_ERROR, DLOG_TAG,
                "The sink event of [%s] cannot be handled because the tensor dimension mismatches.",
                elem->name);
            gst_caps_unref (caps);
            elem->sink = NULL;
            ret = NNS_ERROR_INVALID_PARAMETER;
            goto unlock_return;
          }

          elem->size += sz;

          if (sz != size[i]) {
            dlog_print (DLOG_ERROR, DLOG_TAG,
                "The given input tensor size (%d'th, %zu bytes) mismatches the source pad (%zu bytes)",
                i, size[i], sz);
            gst_caps_unref (caps);
            elem->sink = NULL;
            ret = NNS_ERROR_INVALID_PARAMETER;
            goto unlock_return;
          }
        }
      } else {
        elem->src = NULL;       /* invalid! */
        /** @todo What if it keeps being "NULL"? */
      }
      gst_caps_unref (caps);
    }
  }

  if (elem->size == 0) {
    dlog_print (DLOG_WARN, DLOG_TAG,
        "The pipeline is not ready to accept inputs. The input is ignored.");
    ret = NNS_ERROR_TRY_AGAIN;
    goto unlock_return;
  }

  /* Create buffer to be pushed from buf[] */
  buffer = gst_buffer_new_wrapped_full (GST_MEMORY_FLAG_READONLY,
      buf[0], size[0], 0, size[0], buf[0], bufpolicy[policy]);
  for (i = 1; i < num_tensors; i++) {
    GstBuffer *addbuffer =
        gst_buffer_new_wrapped_full (GST_MEMORY_FLAG_READONLY, buf[i], size[i],
        0, size[i], buf[i], bufpolicy[policy]);
    buffer = gst_buffer_append (buffer, addbuffer);

    /** @todo Verify that gst_buffer_append lists tensors/gstmem in the correct order */
  }

  /* Push the data! */
  gret = gst_app_src_push_buffer (GST_APP_SRC (elem->element), buffer);

  if (gret == GST_FLOW_FLUSHING) {
    dlog_print (DLOG_WARN, DLOG_TAG,
        "The pipeline is not in PAUSED/PLAYING. The input may be ignored.");
    ret = NNS_ERROR_TRY_AGAIN;
  } else if (gret == GST_FLOW_EOS) {
    dlog_print (DLOG_WARN, DLOG_TAG,
        "THe pipeline is in EOS state. The input is ignored.");
    ret = NNS_ERROR_STREAMS_PIPE;
  }

  handle_exit (h);
}

/****************************************************
 ** NNStreamer Pipeline Switch/Valve Control       **
 ****************************************************/

/**
 * @brief Get a handle to operate a selector (more info in nnstreamer.h)
 */
int
nns_pipeline_switch_gethandle (nns_pipeline_h pipe, const char *switchname,
    nns_switch_type_e * type, nns_switch_h * h)
{
  element *elem;
  nns_pipeline *p = pipe;
  nns_switch *swtc;
  int ret = NNS_ERROR_NONE;

  if (pipe == NULL) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "The first argument, pipeline handle, is not valid.");
    return NNS_ERROR_INVALID_PARAMETER;
  }

  if (switchname == NULL) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "The second argument, switchname, is not valid.");
    return NNS_ERROR_INVALID_PARAMETER;
  }

  g_mutex_lock (&p->lock);
  elem = g_hash_table_lookup (p->namednodes, switchname);

  if (elem == NULL) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "There is no switch element named [%s] in the pipeline.", switchname);
    ret = NNS_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  if (elem->type != NNSAPI_SWITCH_INPUT && elem->type != NNSAPI_SWITCH_OUTPUT) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "There is an element named [%s] in the pipeline, but it is not an input/output switch",
        switchname);
    ret = NNS_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  *h = g_new0 (nns_switch, 1);
  swtc = *h;

  swtc->pipe = p;
  swtc->element = elem;

  if (type) {
    if (elem->type == NNSAPI_SWITCH_INPUT)
      *type = NNS_SWITCH_INPUTSELECTOR;
    else if (elem->type == NNSAPI_SWITCH_OUTPUT)
      *type = NNS_SWITCH_OUTPUTSELECTOR;
    else {
      dlog_print (DLOG_ERROR, DLOG_TAG,
          "Internal data of switch-handle [%s] is broken. It is fatal.",
          elem->name);
      ret = NNS_ERROR_INVALID_PARAMETER;
      goto unlock_return;
    }
  }

  g_mutex_lock (&elem->lock);

  elem->maxid++;
  swtc->id = elem->maxid;
  elem->handles = g_list_append (elem->handles, swtc);

  g_mutex_unlock (&elem->lock);

unlock_return:
  g_mutex_unlock (&p->lock);
  return ret;
}

/**
 * @brief Close the given switch handle (more info in nnstreamer.h)
 */
int
nns_pipeline_switch_puthandle (nns_switch_h h)
{
  handle_init (switch, swtc, h);

  elem->handles = g_list_remove (elem->handles, swtc);

  handle_exit (h);
}

/**
 * @brief Control the switch (more info in nnstreamer.h)
 */
int
nns_pipeline_switch_select (nns_switch_h h, const char *padname)
{
  GstPad *active_pad, *new_pad;
  gchar *active_name;

  handle_init (switch, swtc, h);

  g_object_get (G_OBJECT (elem->element), "active-pad", &active_pad, NULL);
  active_name = gst_pad_get_name (active_pad);

  if (!g_strcmp0 (padname, active_name)) {
    dlog_print (DLOG_INFO, DLOG_TAG,
        "Switch is called, but there is no effective changes: %s->%s.",
        active_name, padname);
    g_free (active_name);
    gst_object_unref (active_pad);

    goto unlock_return;
  }

  g_free (active_name);
  gst_object_unref (active_pad);

  new_pad = gst_element_get_static_pad (elem->element, padname);
  if (new_pad == NULL) {
    /* Not Found! */
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "Cannot find the pad, [%s], from the switch, [%s].",
        padname, elem->name);
    ret = NNS_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  g_object_set (G_OBJECT (elem->element), "active-pad", new_pad, NULL);
  gst_object_unref (new_pad);

  dlog_print (DLOG_INFO, DLOG_TAG,
      "Switched to [%s] successfully at switch [%s].", padname, elem->name);

  handle_exit (h);
}

/**
 * @brief List nodes of a switch (more info in nnstreamer.h)
 */
int
nns_pipeline_switch_nodelist (nns_switch_h h, char ***list)
{
  GstIterator *it;
  GValue item = G_VALUE_INIT;
  gboolean done = FALSE;
  GList *dllist = NULL;
  GstPad *pad;
  int counter = 0;

  handle_init (switch, swtc, h);

  if (elem->type == NNSAPI_SWITCH_INPUT)
    it = gst_element_iterate_sink_pads (elem->element);
  else if (elem->type == NNSAPI_SWITCH_OUTPUT)
    it = gst_element_iterate_src_pads (elem->element);
  else {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "The element, [%s], is supposed to be input/output switch, but it is not. Internal data structure is broken.",
        elem->name);
    ret = NNS_ERROR_STREAMS_PIPE;
    goto unlock_return;
  }

  while (!done) {
    switch (gst_iterator_next (it, &item)) {
      case GST_ITERATOR_OK:
        pad = GST_PAD (g_value_get_object (&item));
        dllist = g_list_append (dllist, gst_pad_get_name (pad));
        counter++;
        g_value_reset (&item);
        break;
      case GST_ITERATOR_RESYNC:
        g_list_free_full (dllist, g_free);      /* This frees all strings as well */
        dllist = NULL;
        counter = 0;
        gst_iterator_resync (it);
        break;
      case GST_ITERATOR_ERROR:
        dlog_print (DLOG_ERROR, DLOG_TAG,
            "Cannot access the list of pad properly of a switch, [%s].",
            elem->name);
        ret = NNS_ERROR_STREAMS_PIPE;
        break;
      case GST_ITERATOR_DONE:
        done = TRUE;
        break;
    }
  }

  /* There has been no error with that "while" loop. */
  if (ret == NNS_ERROR_NONE) {
    int i = 0;
    GList *l;

    *list = g_malloc0 (sizeof (char *) * (counter + 1));

    for (l = dllist; l != NULL; l = l->next) {
      (*list)[i] = l->data;     /* Allocated by gst_pad_get_name(). Caller has to free it */
      i++;

      if (i > counter) {
        g_list_free_full (dllist, g_free);      /* This frees all strings as well */
        g_free (list);

        dlog_print (DLOG_ERROR, DLOG_TAG,
            "Internal data inconsistency. This could be a bug in nnstreamer. Switch [%s].",
            elem->name);
        ret = NNS_ERROR_STREAMS_PIPE;
        goto unlock_return;
      }
    }
  }
  g_list_free (dllist);         /* This does not free the strings.. fortunately. */

  handle_exit (h);
}

/**
 * @brief Get a handle to operate a Valve (more info in nnstreamer.h)
 */
int nns_pipeline_valve_gethandle
    (nns_pipeline_h pipe, const char *valvename, nns_valve_h * h)
{
  element *elem;
  nns_pipeline *p = pipe;
  nns_valve *valve;
  int ret = NNS_ERROR_NONE;

  if (pipe == NULL) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "The first argument, pipeline handle, is not valid.");
    return NNS_ERROR_INVALID_PARAMETER;
  }

  if (valvename == NULL) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "The second argument, valvename, is not valid.");
    return NNS_ERROR_INVALID_PARAMETER;
  }

  g_mutex_lock (&p->lock);
  elem = g_hash_table_lookup (p->namednodes, valvename);

  if (elem == NULL) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "There is no valve element named [%s] in the pipeline.", valvename);
    ret = NNS_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  if (elem->type != NNSAPI_VALVE) {
    dlog_print (DLOG_ERROR, DLOG_TAG,
        "There is an element named [%s] in the pipeline, but it is not a valve",
        valvename);
    ret = NNS_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  *h = g_new0 (nns_valve, 1);
  valve = *h;

  valve->pipe = p;
  valve->element = elem;

  g_mutex_lock (&elem->lock);

  elem->maxid++;
  valve->id = elem->maxid;
  elem->handles = g_list_append (elem->handles, valve);

  g_mutex_unlock (&elem->lock);

unlock_return:
  g_mutex_unlock (&p->lock);
  return ret;
}

/**
 * @brief Close the given valve handle (more info in nnstreamer.h)
 */
int
nns_pipeline_valve_puthandle (nns_valve_h h)
{
  handle_init (valve, valve, h);

  elem->handles = g_list_remove (elem->handles, valve);

  handle_exit (h);
}

/**
 * @brief Control the valve with the given handle (more info in nnstreamer.h)
 */
int
nns_pipeline_valve_control (nns_valve_h h, int valve_drop)
{
  gboolean drop;
  handle_init (valve, valve, h);

  g_object_get (G_OBJECT (elem->element), "drop", &drop, NULL);

  if ((valve_drop != 0) == (drop != FALSE)) {
    /* Nothing to do */
    dlog_print (DLOG_INFO, DLOG_TAG,
        "Valve is called, but there is no effective changes: %d->%d",
        ! !drop, ! !valve_drop);
    goto unlock_return;
  }

  g_object_set (G_OBJECT (elem->element), "drop", ! !valve_drop, NULL);
  dlog_print (DLOG_INFO, DLOG_TAG,
      "Valve is changed: %d->%d", ! !drop, ! !valve_drop);

  handle_exit (h);
}
