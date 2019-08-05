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
 * @file nnstreamer-capi-pipeline.c
 * @date 11 March 2019
 * @brief NNStreamer/Pipeline(main) C-API Wrapper.
 *        This allows to construct and control NNStreamer pipelines.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <string.h>
#include <glib.h>
#include <glib-object.h>        /* Get GType from GObject Instances */
#include <gmodule.h>

#include <nnstreamer/tensor_typedef.h>
#include <nnstreamer/nnstreamer_plugin_api.h>

#include <gst/gstbuffer.h>
#include <gst/app/app.h>        /* To push data to pipeline */

#include "nnstreamer-capi-private.h"

#define handle_init(type, name, h) \
  ml_pipeline_##type *name= (h); \
  ml_pipeline *p; \
  ml_pipeline_element *elem; \
  int ret = ML_ERROR_NONE; \
  check_feature_state (); \
  if ((h) == NULL) { \
    ml_loge ("The given handle is invalid"); \
    return ML_ERROR_INVALID_PARAMETER; \
  } \
\
  p = name->pipe; \
  elem = name->element; \
  if (p == NULL || elem == NULL || p != elem->pipe) { \
    ml_loge ("The handle appears to be broken."); \
    return ML_ERROR_INVALID_PARAMETER; \
  } \
\
  g_mutex_lock (&p->lock); \
  g_mutex_lock (&elem->lock); \
\
  if (NULL == g_list_find (elem->handles, name)) { \
    ml_loge ("The handle does not exists."); \
    ret = ML_ERROR_INVALID_PARAMETER; \
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
static ml_pipeline_element *
construct_element (GstElement * e, ml_pipeline * p, const char *name,
    ml_pipeline_element_e t)
{
  ml_pipeline_element *ret = g_new0 (ml_pipeline_element, 1);
  ret->element = e;
  ret->pipe = p;
  ret->name = g_strdup (name);
  ret->type = t;
  ret->handles = NULL;
  ret->src = NULL;
  ret->sink = NULL;
  ml_tensors_info_initialize (&ret->tensors_info);
  ret->size = 0;
  ret->maxid = 0;
  ret->handle_id = 0;
  g_mutex_init (&ret->lock);
  return ret;
}

/**
 * @brief Internal function to get the tensors info from the element caps.
 */
static gboolean
get_tensors_info_from_caps (GstCaps * caps, ml_tensors_info_s * info)
{
  GstStructure *s;
  GstTensorsConfig config;
  guint i, n_caps;
  gboolean found = FALSE;

  ml_tensors_info_initialize (info);
  n_caps = gst_caps_get_size (caps);

  for (i = 0; i < n_caps; i++) {
    s = gst_caps_get_structure (caps, i);
    found = gst_tensors_config_from_structure (&config, s);

    if (found) {
      ml_tensors_info_copy_from_gst (info, &config.info);
      break;
    }
  }

  return found;
}

/**
 * @brief Handle a sink element for registered ml_pipeline_sink_cb
 */
static void
cb_sink_event (GstElement * e, GstBuffer * b, gpointer user_data)
{
  ml_pipeline_element *elem = user_data;

  /** @todo CRITICAL if the pipeline is being killed, don't proceed! */

  GstMemory *mem[ML_TENSOR_SIZE_LIMIT];
  GstMapInfo info[ML_TENSOR_SIZE_LIMIT];
  guint i;
  guint num_mems;
  GList *l;
  ml_tensors_data_s *data = NULL;
  size_t total_size = 0;

  num_mems = gst_buffer_n_memory (b);

  if (num_mems > ML_TENSOR_SIZE_LIMIT) {
    ml_loge ("Number of memory chunks in a GstBuffer exceed the limit: %u > %u",
        num_mems, ML_TENSOR_SIZE_LIMIT);
    return;
  }

  /* set tensor data */
  data = g_new0 (ml_tensors_data_s, 1);
  g_assert (data);

  data->num_tensors = num_mems;
  for (i = 0; i < num_mems; i++) {
    mem[i] = gst_buffer_peek_memory (b, i);
    gst_memory_map (mem[i], &info[i], GST_MAP_READ);

    data->tensors[i].tensor = info[i].data;
    data->tensors[i].size = info[i].size;

    total_size += info[i].size;
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
        gboolean found;

        found = get_tensors_info_from_caps (caps, &elem->tensors_info);
        gst_caps_unref (caps);

        if (found) {
          elem->size = 0;

          if (elem->tensors_info.num_tensors != num_mems) {
            ml_loge
                ("The sink event of [%s] cannot be handled because the number of tensors mismatches.",
                elem->name);

            gst_object_unref (elem->sink);
            elem->sink = NULL;
            goto error;
          }

          for (i = 0; i < elem->tensors_info.num_tensors; i++) {
            size_t sz = ml_tensor_info_get_size (&elem->tensors_info.info[i]);

            if (sz != data->tensors[i].size) {
              ml_loge
                  ("The sink event of [%s] cannot be handled because the tensor dimension mismatches.",
                  elem->name);

              gst_object_unref (elem->sink);
              elem->sink = NULL;
              goto error;
            }

            elem->size += sz;
          }
        } else {
          gst_object_unref (elem->sink);
          elem->sink = NULL;    /* It is not valid */
          goto error;
          /** @todo What if it keeps being "NULL"? Exception handling at 2nd frame? */
        }
      }
    }
  }

  /* Get the data! */
  if (gst_buffer_get_size (b) != total_size ||
      (elem->size > 0 && total_size != elem->size)) {
    ml_loge
        ("The buffersize mismatches. All the three values must be the same: %zu, %zu, %zu",
        total_size, elem->size, gst_buffer_get_size (b));
    goto error;
  }

  /* Iterate e->handles, pass the data to them */
  for (l = elem->handles; l != NULL; l = l->next) {
    ml_pipeline_sink *sink = l->data;
    ml_pipeline_sink_cb callback = sink->cb;

    callback (data, &elem->tensors_info, sink->pdata);

    /** @todo Measure time. Warn if it takes long. Kill if it takes too long. */
  }

error:
  g_mutex_unlock (&elem->lock);

  for (i = 0; i < num_mems; i++) {
    gst_memory_unmap (mem[i], &info[i]);
  }

  if (data) {
    g_free (data);
    data = NULL;
  }
  return;
}

/**
 * @brief Handle a appsink element for registered ml_pipeline_sink_cb
 */
static GstFlowReturn
cb_appsink_new_sample (GstElement * e, gpointer user_data)
{
  GstSample *sample;
  GstBuffer *buffer;

  /* get the sample from appsink */
  sample = gst_app_sink_pull_sample (GST_APP_SINK (e));
  buffer = gst_sample_get_buffer (sample);

  cb_sink_event (e, buffer, user_data);

  gst_sample_unref (sample);
  return GST_FLOW_OK;
}

/**
 * @brief Callback for bus message.
 */
static void
cb_bus_sync_message (GstBus * bus, GstMessage * message, gpointer user_data)
{
  ml_pipeline *pipe_h;

  pipe_h = (ml_pipeline *) user_data;

  if (pipe_h == NULL)
    return;

  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_STATE_CHANGED:
      if (GST_MESSAGE_SRC (message) == GST_OBJECT_CAST (pipe_h->element)) {
        GstState old_state, new_state;

        gst_message_parse_state_changed (message, &old_state, &new_state, NULL);

        ml_logd ("The pipeline state changed from %s to %s.",
            gst_element_state_get_name (old_state),
            gst_element_state_get_name (new_state));

        if (pipe_h->cb) {
          ml_pipeline_state_e ml_state = (ml_pipeline_state_e) new_state;
          pipe_h->cb (ml_state, pipe_h->pdata);
        }
      }
      break;
    default:
      break;
  }
}

/**
 * @brief Private function for ml_pipeline_destroy, cleaning up nodes in namednodes
 */
static void
cleanup_node (gpointer data)
{
  ml_pipeline_element *e = data;

  g_mutex_lock (&e->lock);
  g_free (e->name);
  if (e->src)
    gst_object_unref (e->src);
  if (e->sink)
    gst_object_unref (e->sink);

  /** @todo CRITICAL. Stop the handle callbacks if they are running/ready */
  if (e->handle_id > 0) {
    g_signal_handler_disconnect (e->element, e->handle_id);
    e->handle_id = 0;
  }

  if (e->handles)
    g_list_free_full (e->handles, g_free);
  e->handles = NULL;

  ml_tensors_info_free (&e->tensors_info);

  g_mutex_unlock (&e->lock);
  g_mutex_clear (&e->lock);

  g_free (e);
}

/**
 * @brief Construct the pipeline (more info in nnstreamer.h)
 */
int
ml_pipeline_construct (const char *pipeline_description,
    ml_pipeline_state_cb cb, void *user_data, ml_pipeline_h * pipe)
{
  GError *err = NULL;
  GstElement *pipeline;
  GstIterator *it = NULL;
  int ret = ML_ERROR_NONE;

  ml_pipeline *pipe_h;

  check_feature_state ();

  if (pipe == NULL)
    return ML_ERROR_INVALID_PARAMETER;

  /* init null */
  *pipe = NULL;

  if (FALSE == gst_init_check (NULL, NULL, &err)) {
    if (err) {
      ml_loge ("GStreamer has the following error: %s", err->message);
      g_error_free (err);
    } else {
      ml_loge ("Cannot initialize GStreamer. Unknown reason.");
    }
    return ML_ERROR_STREAMS_PIPE;
  }

  pipeline = gst_parse_launch (pipeline_description, &err);
  if (pipeline == NULL || err) {
    if (err) {
      ml_loge ("Cannot parse and launch the given pipeline = [%s], %s",
          pipeline_description, err->message);
      g_error_free (err);
    } else {
      ml_loge
          ("Cannot parse and launch the given pipeline = [%s], unknown reason",
          pipeline_description);
    }
    return ML_ERROR_STREAMS_PIPE;
  }

  pipe_h = g_new0 (ml_pipeline, 1);
  *pipe = pipe_h;
  g_assert (GST_IS_PIPELINE (pipeline));
  pipe_h->element = pipeline;
  g_mutex_init (&pipe_h->lock);

  /* bus and message callback */
  pipe_h->bus = gst_element_get_bus (pipeline);
  g_assert (pipe_h->bus);

  gst_bus_enable_sync_message_emission (pipe_h->bus);
  pipe_h->signal_msg = g_signal_connect (pipe_h->bus, "sync-message",
      G_CALLBACK (cb_bus_sync_message), pipe_h);

  pipe_h->cb = cb;
  pipe_h->pdata = user_data;

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
              ml_pipeline_element_e element_type = ML_PIPELINE_ELEMENT_UNKNOWN;

              if (G_TYPE_CHECK_INSTANCE_TYPE (elem,
                      gst_element_factory_get_element_type (tensor_sink))) {
                element_type = ML_PIPELINE_ELEMENT_SINK;
              } else if (G_TYPE_CHECK_INSTANCE_TYPE (elem, GST_TYPE_APP_SRC)) {
                element_type = ML_PIPELINE_ELEMENT_APP_SRC;
              } else if (G_TYPE_CHECK_INSTANCE_TYPE (elem, GST_TYPE_APP_SINK)) {
                element_type = ML_PIPELINE_ELEMENT_APP_SINK;
              } else if (G_TYPE_CHECK_INSTANCE_TYPE (elem,
                      gst_element_factory_get_element_type (valve))) {
                element_type = ML_PIPELINE_ELEMENT_VALVE;
              } else if (G_TYPE_CHECK_INSTANCE_TYPE (elem,
                      gst_element_factory_get_element_type (inputs))) {
                element_type = ML_PIPELINE_ELEMENT_SWITCH_INPUT;
              } else if (G_TYPE_CHECK_INSTANCE_TYPE (elem,
                      gst_element_factory_get_element_type (outputs))) {
                element_type = ML_PIPELINE_ELEMENT_SWITCH_OUTPUT;
              } else {
                /** @todo CRITICAL HANDLE THIS! */
              }

              if (element_type != ML_PIPELINE_ELEMENT_UNKNOWN) {
                ml_pipeline_element *e;

                e = construct_element (elem, pipe_h, name, element_type);
                g_hash_table_insert (pipe_h->namednodes, g_strdup (name), e);
              }

              g_free (name);
            }
          }

          g_value_reset (&item);
          break;
        case GST_ITERATOR_RESYNC:
        case GST_ITERATOR_ERROR:
          ml_logw
              ("There is an error or a resync-event while inspecting a pipeline. However, we can still execute the pipeline.");
          /* fallthrough */
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
ml_pipeline_destroy (ml_pipeline_h pipe)
{
  ml_pipeline *p = pipe;
  GstStateChangeReturn scret;
  GstState state;

  check_feature_state ();

  if (p == NULL)
    return ML_ERROR_INVALID_PARAMETER;

  g_mutex_lock (&p->lock);

  /* Before changing the state, remove all callbacks. */
  p->cb = NULL;
  g_signal_handler_disconnect (p->bus, p->signal_msg);
  gst_object_unref (p->bus);

  g_hash_table_remove_all (p->namednodes);

  /* if it's PLAYING, PAUSE it. */
  scret = gst_element_get_state (p->element, &state, NULL, 10 * GST_MSECOND);     /* 10ms */
  if (scret != GST_STATE_CHANGE_FAILURE && state == GST_STATE_PLAYING) {
    /* Pause the pipeline if it's Playing */
    scret = gst_element_set_state (p->element, GST_STATE_PAUSED);
    if (scret == GST_STATE_CHANGE_FAILURE) {
      g_mutex_unlock (&p->lock);
      return ML_ERROR_STREAMS_PIPE;
    }
  }

  /** @todo Ensure all callbacks are gone. (kill'em all!) THIS IS CRITICAL! */
  g_mutex_unlock (&p->lock);
  g_usleep (50000);             /* do 50ms sleep until we have it implemented. Let them complete. And hope they don't call start(). */
  g_mutex_lock (&p->lock);

  /** Destroy registered callback handles */
  g_hash_table_destroy (p->namednodes);
  p->namednodes = NULL;

  /** Stop (NULL State) the pipeline */
  scret = gst_element_set_state (p->element, GST_STATE_NULL);
  if (scret != GST_STATE_CHANGE_SUCCESS) {
    g_mutex_unlock (&p->lock);
    return ML_ERROR_STREAMS_PIPE;
  }

  gst_object_unref (p->element);

  g_mutex_unlock (&p->lock);
  g_mutex_clear (&p->lock);

  g_free (p);
  return ML_ERROR_NONE;
}

/**
 * @brief Get the pipeline state (more info in nnstreamer.h)
 */
int
ml_pipeline_get_state (ml_pipeline_h pipe, ml_pipeline_state_e * state)
{
  ml_pipeline *p = pipe;
  GstState _state;
  GstStateChangeReturn scret;

  check_feature_state ();

  if (p == NULL || state == NULL)
    return ML_ERROR_INVALID_PARAMETER;

  *state = ML_PIPELINE_STATE_UNKNOWN;

  g_mutex_lock (&p->lock);
  scret = gst_element_get_state (p->element, &_state, NULL, GST_MSECOND);      /* Do it within 1ms! */
  g_mutex_unlock (&p->lock);

  if (scret == GST_STATE_CHANGE_FAILURE)
    return ML_ERROR_STREAMS_PIPE;

  *state = (ml_pipeline_state_e) _state;
  return ML_ERROR_NONE;
}

/****************************************************
 ** NNStreamer Pipeline Start/Stop Control         **
 ****************************************************/
/**
 * @brief Start/Resume the pipeline! (more info in nnstreamer.h)
 */
int
ml_pipeline_start (ml_pipeline_h pipe)
{
  ml_pipeline *p = pipe;
  GstStateChangeReturn scret;

  check_feature_state ();

  if (p == NULL)
    return ML_ERROR_INVALID_PARAMETER;

  g_mutex_lock (&p->lock);
  scret = gst_element_set_state (p->element, GST_STATE_PLAYING);
  g_mutex_unlock (&p->lock);

  if (scret == GST_STATE_CHANGE_FAILURE)
    return ML_ERROR_STREAMS_PIPE;

  return ML_ERROR_NONE;
}

/**
 * @brief Pause the pipeline! (more info in nnstreamer.h)
 */
int
ml_pipeline_stop (ml_pipeline_h pipe)
{
  ml_pipeline *p = pipe;
  GstStateChangeReturn scret;

  check_feature_state ();

  if (p == NULL)
    return ML_ERROR_INVALID_PARAMETER;

  g_mutex_lock (&p->lock);
  scret = gst_element_set_state (p->element, GST_STATE_PAUSED);
  g_mutex_unlock (&p->lock);

  if (scret == GST_STATE_CHANGE_FAILURE)
    return ML_ERROR_STREAMS_PIPE;

  return ML_ERROR_NONE;
}

/****************************************************
 ** NNStreamer Pipeline Sink/Src Control           **
 ****************************************************/
/**
 * @brief Register a callback for sink (more info in nnstreamer.h)
 */
int
ml_pipeline_sink_register (ml_pipeline_h pipe, const char *sink_name,
    ml_pipeline_sink_cb cb, void *user_data, ml_pipeline_sink_h * h)
{
  ml_pipeline_element *elem;
  ml_pipeline *p = pipe;
  ml_pipeline_sink *sink;
  int ret = ML_ERROR_NONE;

  check_feature_state ();

  if (h == NULL) {
    ml_loge ("The argument sink handle is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /* init null */
  *h = NULL;

  if (pipe == NULL) {
    ml_loge ("The first argument, pipeline handle is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (sink_name == NULL) {
    ml_loge ("The second argument, sink name is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (cb == NULL) {
    ml_loge ("The callback argument, cb, is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  g_mutex_lock (&p->lock);
  elem = g_hash_table_lookup (p->namednodes, sink_name);

  if (elem == NULL) {
    ml_loge ("There is no element named [%s] in the pipeline.", sink_name);
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  if (elem->type != ML_PIPELINE_ELEMENT_SINK &&
      elem->type != ML_PIPELINE_ELEMENT_APP_SINK) {
    ml_loge ("The element [%s] in the pipeline is not a sink element.",
        sink_name);
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  if (elem->handle_id > 0) {
    ml_logw ("Sink callback is already registered.");
    ret = ML_ERROR_NONE;
    goto unlock_return;
  }

  /* set callback for new data */
  if (elem->type == ML_PIPELINE_ELEMENT_SINK) {
    /* tensor_sink */
    g_object_set (G_OBJECT (elem->element), "emit-signal", (gboolean) TRUE,
        NULL);
    elem->handle_id =
        g_signal_connect (elem->element, "new-data", G_CALLBACK (cb_sink_event),
        elem);
  } else {
    /* appsink */
    g_object_set (G_OBJECT (elem->element), "emit-signals", (gboolean) TRUE,
        NULL);
    elem->handle_id =
        g_signal_connect (elem->element, "new-sample",
        G_CALLBACK (cb_appsink_new_sample), elem);
  }

  if (elem->handle_id == 0) {
    ml_loge ("Failed to connect a signal to the element [%s].", sink_name);
    ret = ML_ERROR_STREAMS_PIPE;
    goto unlock_return;
  }

  *h = g_new0 (ml_pipeline_sink, 1);
  sink = *h;

  sink->pipe = p;
  sink->element = elem;
  sink->cb = cb;
  sink->pdata = user_data;

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
ml_pipeline_sink_unregister (ml_pipeline_sink_h h)
{
  handle_init (sink, sink, h);

  if (elem->handle_id > 0) {
    g_signal_handler_disconnect (elem->element, elem->handle_id);
    elem->handle_id = 0;
  }

  elem->handles = g_list_remove (elem->handles, sink);
  g_free (sink);

  handle_exit (h);
}

/**
 * @brief Implementation of policies decalred by ml_pipeline_buf_policy_e in nnstreamer.h,
 *        "Free"
 */
static void
ml_buf_policy_cb_free (gpointer data)
{
  g_free (data);
}

/**
 * @brief Implementation of policies decalred by ml_pipeline_buf_policy_e in nnstreamer.h.
 *        "Do Nothing"
 */
static void
ml_buf_policy_cb_nop (gpointer data)
{
  /* DO NOTHING! */
}

/**
 * @brief Implementation of policies decalred by ml_pipeline_buf_policy_e in nnstreamer.h.
 */
static const GDestroyNotify ml_buf_policy[ML_PIPELINE_BUF_POLICY_MAX] = {
  [ML_PIPELINE_BUF_POLICY_AUTO_FREE] = ml_buf_policy_cb_free,
  [ML_PIPELINE_BUF_POLICY_DO_NOT_FREE] = ml_buf_policy_cb_nop,
};

/**
 * @brief Parse tensors info of src element.
 */
static int
ml_pipeline_src_parse_tensors_info (ml_pipeline_element * elem)
{
  int ret = ML_ERROR_NONE;

  if (elem->src == NULL) {
    elem->src = gst_element_get_static_pad (elem->element, "src");
    elem->size = 0;

    if (elem->src == NULL) {
      ret = ML_ERROR_STREAMS_PIPE;
    } else {
      GstCaps *caps = gst_pad_get_allowed_caps (elem->src);
      guint i;
      gboolean found = FALSE;
      size_t sz;

      if (caps) {
        found = get_tensors_info_from_caps (caps, &elem->tensors_info);
        gst_caps_unref (caps);
      }

      if (found) {
        for (i = 0; i < elem->tensors_info.num_tensors; i++) {
          sz = ml_tensor_info_get_size (&elem->tensors_info.info[i]);
          elem->size += sz;
        }
      } else {
        ml_logw
            ("Cannot find caps. The pipeline is not yet negotiated for src element [%s].",
            elem->name);
        gst_object_unref (elem->src);
        elem->src = NULL;

        ret = ML_ERROR_TRY_AGAIN;
      }
    }
  }

  return ret;
}

/**
 * @brief Get a handle to operate a src (more info in nnstreamer.h)
 */
int
ml_pipeline_src_get_handle (ml_pipeline_h pipe, const char *src_name,
    ml_pipeline_src_h * h)
{
  ml_pipeline *p = pipe;
  ml_pipeline_element *elem;
  ml_pipeline_src *src;
  int ret = ML_ERROR_NONE;

  check_feature_state ();

  if (h == NULL) {
    ml_loge ("The argument source handle is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /* init null */
  *h = NULL;

  if (pipe == NULL) {
    ml_loge ("The first argument, pipeline handle is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (src_name == NULL) {
    ml_loge ("The second argument, source name is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  g_mutex_lock (&p->lock);

  elem = g_hash_table_lookup (p->namednodes, src_name);

  if (elem == NULL) {
    ml_loge ("There is no element named [%s] in the pipeline.", src_name);
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  if (elem->type != ML_PIPELINE_ELEMENT_APP_SRC) {
    ml_loge ("The element [%s] in the pipeline is not a source element.",
        src_name);
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  *h = g_new (ml_pipeline_src, 1);
  src = *h;

  src->pipe = p;
  src->element = elem;

  g_mutex_lock (&elem->lock);

  elem->maxid++;
  src->id = elem->maxid;
  elem->handles = g_list_append (elem->handles, src);

  ml_pipeline_src_parse_tensors_info (elem);
  g_mutex_unlock (&elem->lock);

unlock_return:
  g_mutex_unlock (&p->lock);

  return ret;
}

/**
 * @brief Close a src node (more info in nnstreamer.h)
 */
int
ml_pipeline_src_release_handle (ml_pipeline_src_h h)
{
  handle_init (src, src, h);

  elem->handles = g_list_remove (elem->handles, src);
  g_free (src);

  handle_exit (h);
}

/**
 * @brief Push a data frame to a src (more info in nnstreamer.h)
 */
int
ml_pipeline_src_input_data (ml_pipeline_src_h h, ml_tensors_data_h data,
    ml_pipeline_buf_policy_e policy)
{
  /** @todo NYI */
  GstBuffer *buffer;
  GstMemory *mem;
  GstFlowReturn gret;
  ml_tensors_data_s *_data;
  unsigned int i;

  handle_init (src, src, h);

  _data = (ml_tensors_data_s *) data;
  if (!_data) {
    ml_loge ("The given param data is invalid.");
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  if (_data->num_tensors < 1 || _data->num_tensors > ML_TENSOR_SIZE_LIMIT) {
    ml_loge ("The tensor size is invalid. It should be 1 ~ %u; where it is %u",
        ML_TENSOR_SIZE_LIMIT, _data->num_tensors);
    ret = ML_ERROR_INVALID_PARAMETER;
    goto destroy_data;
  }

  ret = ml_pipeline_src_parse_tensors_info (elem);

  if (ret != ML_ERROR_NONE) {
    ml_logw ("The pipeline is not ready to accept inputs. The input is ignored.");
    goto destroy_data;
  }

  if (elem->tensors_info.num_tensors != _data->num_tensors) {
    ml_loge
        ("The src push of [%s] cannot be handled because the number of tensors in a frame mismatches. %u != %u",
        elem->name, elem->tensors_info.num_tensors, _data->num_tensors);

    ret = ML_ERROR_INVALID_PARAMETER;
    goto destroy_data;
  }

  for (i = 0; i < elem->tensors_info.num_tensors; i++) {
    size_t sz = ml_tensor_info_get_size (&elem->tensors_info.info[i]);

    if (sz != _data->tensors[i].size) {
      ml_loge
          ("The given input tensor size (%d'th, %zu bytes) mismatches the source pad (%zu bytes)",
          i, _data->tensors[i].size, sz);

      ret = ML_ERROR_INVALID_PARAMETER;
      goto destroy_data;
    }
  }

  /* Create buffer to be pushed from buf[] */
  buffer = gst_buffer_new ();
  for (i = 0; i < _data->num_tensors; i++) {
    mem = gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY,
        _data->tensors[i].tensor, _data->tensors[i].size, 0,
        _data->tensors[i].size, _data->tensors[i].tensor, ml_buf_policy[policy]);
    gst_buffer_append_memory (buffer, mem);

    /** @todo Verify that gst_buffer_append lists tensors/gstmem in the correct order */
  }

  /* Push the data! */
  gret = gst_app_src_push_buffer (GST_APP_SRC (elem->element), buffer);

  /* Free data ptr if buffer policy is auto-free */
  if (policy == ML_PIPELINE_BUF_POLICY_AUTO_FREE) {
    g_free (_data);
    _data = NULL;
  }

  if (gret == GST_FLOW_FLUSHING) {
    ml_logw ("The pipeline is not in PAUSED/PLAYING. The input may be ignored.");
    ret = ML_ERROR_TRY_AGAIN;
  } else if (gret == GST_FLOW_EOS) {
    ml_logw ("THe pipeline is in EOS state. The input is ignored.");
    ret = ML_ERROR_STREAMS_PIPE;
  }

destroy_data:
  if (_data != NULL && policy == ML_PIPELINE_BUF_POLICY_AUTO_FREE) {
    /* Free data handle */
    ml_tensors_data_destroy (data);
  }

  handle_exit (h);
}

/**
 * @brief Gets a handle for the tensors metadata of given src node.
 */
int
ml_pipeline_src_get_tensors_info (ml_pipeline_src_h h,
    ml_tensors_info_h * info)
{
  handle_init (src, src, h);

  if (info == NULL) {
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  ret = ml_pipeline_src_parse_tensors_info (elem);

  if (ret == ML_ERROR_NONE) {
    ml_tensors_info_create (info);
    ml_tensors_info_clone (*info, &elem->tensors_info);
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
ml_pipeline_switch_get_handle (ml_pipeline_h pipe, const char *switch_name,
    ml_pipeline_switch_e * type, ml_pipeline_switch_h * h)
{
  ml_pipeline_element *elem;
  ml_pipeline *p = pipe;
  ml_pipeline_switch *swtc;
  int ret = ML_ERROR_NONE;

  check_feature_state ();

  if (h == NULL) {
    ml_loge ("The argument switch handle is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /* init null */
  *h = NULL;

  if (pipe == NULL) {
    ml_loge ("The first argument, pipeline handle, is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (switch_name == NULL) {
    ml_loge ("The second argument, switch name, is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  g_mutex_lock (&p->lock);
  elem = g_hash_table_lookup (p->namednodes, switch_name);

  if (elem == NULL) {
    ml_loge ("There is no switch element named [%s] in the pipeline.",
        switch_name);
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  if (elem->type == ML_PIPELINE_ELEMENT_SWITCH_INPUT) {
    if (type)
      *type = ML_PIPELINE_SWITCH_INPUT_SELECTOR;
  } else if (elem->type == ML_PIPELINE_ELEMENT_SWITCH_OUTPUT) {
    if (type)
      *type = ML_PIPELINE_SWITCH_OUTPUT_SELECTOR;
  } else {
    ml_loge
        ("There is an element named [%s] in the pipeline, but it is not an input/output switch",
        switch_name);

    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  *h = g_new0 (ml_pipeline_switch, 1);
  swtc = *h;

  swtc->pipe = p;
  swtc->element = elem;

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
ml_pipeline_switch_release_handle (ml_pipeline_switch_h h)
{
  handle_init (switch, swtc, h);

  elem->handles = g_list_remove (elem->handles, swtc);
  g_free (swtc);

  handle_exit (h);
}

/**
 * @brief Control the switch (more info in nnstreamer.h)
 */
int
ml_pipeline_switch_select (ml_pipeline_switch_h h, const char *pad_name)
{
  GstPad *active_pad, *new_pad;
  gchar *active_name;

  handle_init (switch, swtc, h);

  if (pad_name == NULL) {
    ml_loge ("The second argument, pad name, is not valid.");
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  g_object_get (G_OBJECT (elem->element), "active-pad", &active_pad, NULL);
  active_name = gst_pad_get_name (active_pad);

  if (!g_strcmp0 (pad_name, active_name)) {
    ml_logi ("Switch is called, but there is no effective changes: %s->%s.",
        active_name, pad_name);
    g_free (active_name);
    gst_object_unref (active_pad);

    goto unlock_return;
  }

  g_free (active_name);
  gst_object_unref (active_pad);

  new_pad = gst_element_get_static_pad (elem->element, pad_name);
  if (new_pad == NULL) {
    /* Not Found! */
    ml_loge ("Cannot find the pad, [%s], from the switch, [%s].",
        pad_name, elem->name);
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  g_object_set (G_OBJECT (elem->element), "active-pad", new_pad, NULL);
  gst_object_unref (new_pad);

  ml_logi ("Switched to [%s] successfully at switch [%s].", pad_name, elem->name);

  handle_exit (h);
}

/**
 * @brief Gets the pad names of a switch.
 */
int
ml_pipeline_switch_get_pad_list (ml_pipeline_switch_h h, char ***list)
{
  GstIterator *it;
  GValue item = G_VALUE_INIT;
  gboolean done = FALSE;
  GList *dllist = NULL;
  GstPad *pad;
  int counter = 0;

  handle_init (switch, swtc, h);

  if (list == NULL) {
    ml_loge ("The second argument, list, is not valid.");
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  /* init null */
  *list = NULL;

  if (elem->type == ML_PIPELINE_ELEMENT_SWITCH_INPUT)
    it = gst_element_iterate_sink_pads (elem->element);
  else if (elem->type == ML_PIPELINE_ELEMENT_SWITCH_OUTPUT)
    it = gst_element_iterate_src_pads (elem->element);
  else {
    ml_loge
        ("The element, [%s], is supposed to be input/output switch, but it is not. Internal data structure is broken.",
        elem->name);
    ret = ML_ERROR_STREAMS_PIPE;
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
        ml_loge ("Cannot access the list of pad properly of a switch, [%s].",
            elem->name);
        ret = ML_ERROR_STREAMS_PIPE;
        break;
      case GST_ITERATOR_DONE:
        done = TRUE;
        break;
    }
  }

  /* There has been no error with that "while" loop. */
  if (ret == ML_ERROR_NONE) {
    int i = 0;
    GList *l;

    *list = g_malloc0 (sizeof (char *) * (counter + 1));

    for (l = dllist; l != NULL; l = l->next) {
      (*list)[i] = l->data;     /* Allocated by gst_pad_get_name(). Caller has to free it */
      i++;

      if (i > counter) {
        g_list_free_full (dllist, g_free);      /* This frees all strings as well */
        g_free (*list);
        *list = NULL;

        ml_loge
            ("Internal data inconsistency. This could be a bug in nnstreamer. Switch [%s].",
            elem->name);
        ret = ML_ERROR_STREAMS_PIPE;
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
int
ml_pipeline_valve_get_handle (ml_pipeline_h pipe, const char *valve_name,
    ml_pipeline_valve_h * h)
{
  ml_pipeline_element *elem;
  ml_pipeline *p = pipe;
  ml_pipeline_valve *valve;
  int ret = ML_ERROR_NONE;

  check_feature_state ();

  if (h == NULL) {
    ml_loge ("The argument valve handle is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /* init null */
  *h = NULL;

  if (pipe == NULL) {
    ml_loge ("The first argument, pipeline handle, is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (valve_name == NULL) {
    ml_loge ("The second argument, valve name, is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  g_mutex_lock (&p->lock);
  elem = g_hash_table_lookup (p->namednodes, valve_name);

  if (elem == NULL) {
    ml_loge ("There is no valve element named [%s] in the pipeline.", valve_name);
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  if (elem->type != ML_PIPELINE_ELEMENT_VALVE) {
    ml_loge
        ("There is an element named [%s] in the pipeline, but it is not a valve",
        valve_name);
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  *h = g_new0 (ml_pipeline_valve, 1);
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
ml_pipeline_valve_release_handle (ml_pipeline_valve_h h)
{
  handle_init (valve, valve, h);

  elem->handles = g_list_remove (elem->handles, valve);
  g_free (valve);

  handle_exit (h);
}

/**
 * @brief Control the valve with the given handle (more info in nnstreamer.h)
 */
int
ml_pipeline_valve_set_open (ml_pipeline_valve_h h, bool open)
{
  gboolean drop = FALSE;
  handle_init (valve, valve, h);

  g_object_get (G_OBJECT (elem->element), "drop", &drop, NULL);

  if ((open != false) != (drop != FALSE)) {
    /* Nothing to do */
    ml_logi ("Valve is called, but there is no effective changes");
    goto unlock_return;
  }

  drop = (open) ? FALSE : TRUE;
  g_object_set (G_OBJECT (elem->element), "drop", drop, NULL);

  handle_exit (h);
}
