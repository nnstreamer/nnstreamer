/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
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
 * @see	https://github.com/nnstreamer/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <string.h>
#include <glib.h>
#include <gst/gstbuffer.h>
#include <gst/app/app.h>        /* To push data to pipeline */

#include "nnstreamer-capi-private.h"
#include "tensor_typedef.h"
#include "tensor_filter_custom_easy.h"
#include "nnstreamer_plugin_api.h"


#define handle_init(name, h) \
  ml_pipeline_common_elem *name= (h); \
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
 * @brief Internal function to create a referable element in a pipeline
 */
static ml_pipeline_element *
construct_element (GstElement * e, ml_pipeline * p, const char *name,
    ml_pipeline_element_e t)
{
  ml_pipeline_element *ret = g_new0 (ml_pipeline_element, 1);

  if (ret == NULL) {
    ml_loge ("Failed to allocate memory for the pipeline.");
    return NULL;
  }

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
  ret->is_media_stream = FALSE;
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
  if (data == NULL) {
    ml_loge ("Failed to allocate memory for tensors data in sink callback.");
    return;
  }

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

            /* Not configured, yet. */
            if (sz == 0)
              ml_loge ("The caps for sink(%s) is not configured.", elem->name);

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
    ml_pipeline_sink_cb callback;
    ml_pipeline_common_elem *sink = l->data;
    if (sink->callback_info == NULL)
      continue;

    callback = sink->callback_info->cb;
    if (callback)
      callback (data, &elem->tensors_info, sink->callback_info->pdata);

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
    case GST_MESSAGE_EOS:
      pipe_h->isEOS = TRUE;
      break;
    case GST_MESSAGE_STATE_CHANGED:
      if (GST_MESSAGE_SRC (message) == GST_OBJECT_CAST (pipe_h->element)) {
        GstState old_state, new_state;

        gst_message_parse_state_changed (message, &old_state, &new_state, NULL);
        pipe_h->pipe_state = (ml_pipeline_state_e) new_state;

        ml_logd ("The pipeline state changed from %s to %s.",
            gst_element_state_get_name (old_state),
            gst_element_state_get_name (new_state));

        if (pipe_h->state_cb.cb) {
          pipe_h->state_cb.cb (pipe_h->pipe_state, pipe_h->state_cb.user_data);
        }
      }
      break;
    default:
      break;
  }
}

/**
 * @brief Clean up each element of the pipeline.
 */
static void
free_element_handle (gpointer data)
{
  ml_pipeline_common_elem *item = (ml_pipeline_common_elem *) data;
  g_free (item->callback_info);
  g_free (item);
}

/**
 * @brief Private function for ml_pipeline_destroy, cleaning up nodes in namednodes
 */
static void
cleanup_node (gpointer data)
{
  ml_pipeline_element *e = data;

  g_mutex_lock (&e->lock);
  if (e->type == ML_PIPELINE_ELEMENT_APP_SRC && !e->pipe->isEOS) {
    int eos_check_cnt = 0;

    /** to push EOS event, the pipeline should be in PLAYING state */
    gst_element_set_state (e->pipe->element, GST_STATE_PLAYING);

    if (gst_app_src_end_of_stream (GST_APP_SRC (e->element)) != GST_FLOW_OK) {
      ml_logw ("Failed to set EOS in %s", e->name);
    }
    while (!e->pipe->isEOS) {
      eos_check_cnt++;
      /** check EOS every 1ms */
      g_usleep (1000);
      if (eos_check_cnt >= EOS_MESSAGE_TIME_LIMIT) {
        ml_loge ("Failed to get EOS message");
        break;
      }
    }
  }

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
    g_list_free_full (e->handles, free_element_handle);
  e->handles = NULL;

  ml_tensors_info_free (&e->tensors_info);

  g_mutex_unlock (&e->lock);
  g_mutex_clear (&e->lock);

  g_free (e);
}

/**
 * @brief Private function to release the pipeline resources
 */
static void
cleanup_resource (gpointer data)
{
  pipeline_resource_s *res = data;

  /* check resource type and free data */
  if (g_str_has_prefix (res->type, "tizen")) {
    release_tizen_resource (res->handle, res->type);
  }

  g_free (res->type);
  g_free (res);
}

/**
 * @brief Converts predefined element in pipeline description.
 */
static int
convert_element (ml_pipeline_h pipe, const gchar * description, gchar ** result,
    gboolean is_internal)
{
  gchar *converted;
  int status = ML_ERROR_NONE;

  g_return_val_if_fail (pipe, ML_ERROR_INVALID_PARAMETER);
  g_return_val_if_fail (description && result, ML_ERROR_INVALID_PARAMETER);

  /* init null */
  *result = NULL;

  converted = g_strdup (description);

  /* convert pre-defined element for Tizen */
  status = convert_tizen_element (pipe, &converted, is_internal);

  if (status == ML_ERROR_NONE) {
    ml_logd ("Converted pipeline: %s", converted);
    *result = converted;
  } else {
    g_free (converted);
  }

  return status;
}

/**
 * @brief Iterate elements and prepare element handle.
 */
static int
iterate_element (ml_pipeline * pipe_h, GstElement * pipeline)
{
  GstIterator *it = NULL;
  int status = ML_ERROR_NONE;

  g_return_val_if_fail (pipe_h && pipeline, ML_ERROR_INVALID_PARAMETER);

  g_mutex_lock (&pipe_h->lock);

  it = gst_bin_iterate_elements (GST_BIN (pipeline));
  if (it != NULL) {
    gboolean done = FALSE;
    GValue item = G_VALUE_INIT;
    GObject *obj;
    gchar *name;

    /* Fill in the hashtable, "namednodes" with named Elements */
    while (!done) {
      switch (gst_iterator_next (it, &item)) {
        case GST_ITERATOR_OK:
          obj = g_value_get_object (&item);

          if (GST_IS_ELEMENT (obj)) {
            GstElement *elem = GST_ELEMENT (obj);
            GstPluginFeature *feature =
                GST_PLUGIN_FEATURE (gst_element_get_factory (elem));
            const gchar *plugin_name =
                gst_plugin_feature_get_plugin_name (feature);
            const gchar *element_name = gst_plugin_feature_get_name (feature);

            /* validate the availability of the plugin */
            if (ml_check_plugin_availability (plugin_name,
                    element_name) != ML_ERROR_NONE) {
              status = ML_ERROR_NOT_SUPPORTED;
              done = TRUE;
              break;
            }

            name = gst_element_get_name (elem);
            if (name != NULL) {
              ml_pipeline_element_e element_type = ML_PIPELINE_ELEMENT_UNKNOWN;

              if (g_str_equal (element_name, "tensor_sink")) {
                element_type = ML_PIPELINE_ELEMENT_SINK;
              } else if (g_str_equal (element_name, "appsrc")) {
                element_type = ML_PIPELINE_ELEMENT_APP_SRC;
              } else if (g_str_equal (element_name, "appsink")) {
                element_type = ML_PIPELINE_ELEMENT_APP_SINK;
              } else if (g_str_equal (element_name, "valve")) {
                element_type = ML_PIPELINE_ELEMENT_VALVE;
              } else if (g_str_equal (element_name, "input-selector")) {
                element_type = ML_PIPELINE_ELEMENT_SWITCH_INPUT;
              } else if (g_str_equal (element_name, "output-selector")) {
                element_type = ML_PIPELINE_ELEMENT_SWITCH_OUTPUT;
              } else {
                /** @todo CRITICAL HANDLE THIS! */
              }

              /* check 'sync' property in sink element */
              if (element_type == ML_PIPELINE_ELEMENT_SINK ||
                  element_type == ML_PIPELINE_ELEMENT_APP_SINK) {
                gboolean sync = FALSE;

                g_object_get (G_OBJECT (elem), "sync", &sync, NULL);
                if (sync) {
                  ml_logw
                      ("It is recommended to apply 'sync=false' property to a sink element in most AI applications. Otherwise, inference results of large neural networks will be frequently dropped by the synchronization mechanism at the sink element.");
                }
              }

              if (element_type != ML_PIPELINE_ELEMENT_UNKNOWN) {
                ml_pipeline_element *e;

                e = construct_element (elem, pipe_h, name, element_type);
                if (e != NULL) {
                  g_hash_table_insert (pipe_h->namednodes, g_strdup (name), e);
                } else {
                  /* allocation failure */
                  status = ML_ERROR_OUT_OF_MEMORY;
                  done = TRUE;
                }
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
  }

  g_mutex_unlock (&pipe_h->lock);
  return status;
}

/**
 * @brief Internal function to construct the pipeline.
 * If is_internal is true, this will ignore the permission in Tizen.
 */
static int
construct_pipeline_internal (const char *pipeline_description,
    ml_pipeline_state_cb cb, void *user_data, ml_pipeline_h * pipe,
    gboolean is_internal)
{
  GError *err = NULL;
  GstElement *pipeline;
  gchar *description = NULL;
  int status = ML_ERROR_NONE;

  ml_pipeline *pipe_h;

  check_feature_state ();

  if (!pipe || !pipeline_description)
    return ML_ERROR_INVALID_PARAMETER;

  /* init null */
  *pipe = NULL;

  if ((status = ml_initialize_gstreamer ()) != ML_ERROR_NONE)
    return status;

  /* prepare pipeline handle */
  pipe_h = g_new0 (ml_pipeline, 1);
  if (pipe_h == NULL) {
    ml_loge ("Failed to allocate handle for pipeline.");
    return ML_ERROR_OUT_OF_MEMORY;
  }

  g_mutex_init (&pipe_h->lock);

  pipe_h->isEOS = FALSE;
  pipe_h->pipe_state = ML_PIPELINE_STATE_UNKNOWN;
  pipe_h->namednodes =
      g_hash_table_new_full (g_str_hash, g_str_equal, g_free, cleanup_node);

  pipe_h->resources =
      g_hash_table_new_full (g_str_hash, g_str_equal, g_free, cleanup_resource);

  /* convert predefined element and launch the pipeline */
  status =
      convert_element ((ml_pipeline_h) pipe_h, pipeline_description,
      &description, is_internal);
  if (status != ML_ERROR_NONE)
    goto failed;

  pipeline = gst_parse_launch (description, &err);
  g_free (description);

  if (pipeline == NULL || err) {
    ml_loge ("Cannot parse and launch the given pipeline = [%s], %s",
        pipeline_description, (err) ? err->message : "unknown reason");
    g_clear_error (&err);

    if (pipeline)
      gst_object_unref (pipeline);

    status = ML_ERROR_STREAMS_PIPE;
    goto failed;
  }

  g_assert (GST_IS_PIPELINE (pipeline));
  pipe_h->element = pipeline;

  /* bus and message callback */
  pipe_h->bus = gst_element_get_bus (pipeline);
  g_assert (pipe_h->bus);

  gst_bus_enable_sync_message_emission (pipe_h->bus);
  pipe_h->signal_msg = g_signal_connect (pipe_h->bus, "sync-message",
      G_CALLBACK (cb_bus_sync_message), pipe_h);

  /* state change callback */
  pipe_h->state_cb.cb = cb;
  pipe_h->state_cb.user_data = user_data;

  /* iterate elements and prepare element handle */
  status = iterate_element (pipe_h, pipeline);

  /* finally set pipeline state to PAUSED */
  if (status == ML_ERROR_NONE) {
    status = ml_pipeline_stop ((ml_pipeline_h) pipe_h);
  }

failed:
  if (status != ML_ERROR_NONE) {
    /* failed to construct the pipeline */
    ml_pipeline_destroy ((ml_pipeline_h) pipe_h);
  } else {
    *pipe = pipe_h;
  }

  return status;
}

/**
 * @brief Construct the pipeline (more info in nnstreamer.h)
 */
int
ml_pipeline_construct (const char *pipeline_description,
    ml_pipeline_state_cb cb, void *user_data, ml_pipeline_h * pipe)
{
  /* not an internal pipeline construction */
  return construct_pipeline_internal (pipeline_description, cb, user_data, pipe,
      FALSE);
}

#if defined (__TIZEN__)
/**
 * @brief Construct the pipeline (Tizen internal, see nnstreamer-tizen-internal.h)
 */
int
ml_pipeline_construct_internal (const char *pipeline_description,
    ml_pipeline_state_cb cb, void *user_data, ml_pipeline_h * pipe)
{
  /* Tizen internal pipeline construction */
  return construct_pipeline_internal (pipeline_description, cb, user_data, pipe,
      TRUE);
}
#endif /* __TIZEN__ */

/**
 * @brief Destroy the pipeline (more info in nnstreamer.h)
 */
int
ml_pipeline_destroy (ml_pipeline_h pipe)
{
  ml_pipeline *p = pipe;
  GstStateChangeReturn scret;
  GstState state;
  guint check_paused_cnt = 0;

  check_feature_state ();

  if (p == NULL)
    return ML_ERROR_INVALID_PARAMETER;

  g_mutex_lock (&p->lock);

  /* Before changing the state, remove all callbacks. */
  p->state_cb.cb = NULL;

  g_hash_table_remove_all (p->namednodes);
  g_hash_table_remove_all (p->resources);

  if (p->element) {
    /* Pause the pipeline if it's playing */
    scret = gst_element_get_state (p->element, &state, NULL, 10 * GST_MSECOND); /* 10ms */
    if (scret != GST_STATE_CHANGE_FAILURE && state == GST_STATE_PLAYING) {
      scret = gst_element_set_state (p->element, GST_STATE_PAUSED);
      if (scret == GST_STATE_CHANGE_FAILURE) {
        g_mutex_unlock (&p->lock);
        return ML_ERROR_STREAMS_PIPE;
      }
    }

    g_mutex_unlock (&p->lock);
    while (p->pipe_state == ML_PIPELINE_STATE_PLAYING) {
      check_paused_cnt++;
      /** check PAUSED every 1ms */
      g_usleep (1000);
      if (check_paused_cnt >= WAIT_PAUSED_TIME_LIMIT) {
        ml_loge ("Failed to wait until state changed to PAUSED");
        break;
      }
    }
    g_mutex_lock (&p->lock);

    /* Stop (NULL State) the pipeline */
    scret = gst_element_set_state (p->element, GST_STATE_NULL);
    if (scret != GST_STATE_CHANGE_SUCCESS) {
      g_mutex_unlock (&p->lock);
      return ML_ERROR_STREAMS_PIPE;
    }

    if (p->bus) {
      g_signal_handler_disconnect (p->bus, p->signal_msg);
      gst_object_unref (p->bus);
    }

    gst_object_unref (p->element);
    p->element = NULL;
  }

  /* Destroy registered callback handles and resources */
  g_hash_table_destroy (p->namednodes);
  g_hash_table_destroy (p->resources);
  p->namednodes = p->resources = NULL;

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
  scret = gst_element_get_state (p->element, &_state, NULL, GST_MSECOND);       /* Do it within 1ms! */
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
  int status = ML_ERROR_NONE;

  check_feature_state ();

  if (p == NULL)
    return ML_ERROR_INVALID_PARAMETER;

  g_mutex_lock (&p->lock);

  /* check the resources when starting the pipeline */
  if (g_hash_table_size (p->resources)) {
    GHashTableIter iter;
    gpointer key, value;

    /* iterate all handle and acquire res if released */
    g_hash_table_iter_init (&iter, p->resources);
    while (g_hash_table_iter_next (&iter, &key, &value)) {
      if (g_str_has_prefix (key, "tizen")) {
        status = get_tizen_resource (pipe, key);
        if (status != ML_ERROR_NONE)
          goto done;
      }
    }
  }

  scret = gst_element_set_state (p->element, GST_STATE_PLAYING);
  if (scret == GST_STATE_CHANGE_FAILURE)
    status = ML_ERROR_STREAMS_PIPE;

done:
  g_mutex_unlock (&p->lock);
  return status;
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
  ml_pipeline_common_elem *sink;
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
    /* no need to connect signal to sink element */
    ml_logw ("Sink callback is already registered.");
  } else {
    /* set callback for new data */
    if (elem->type == ML_PIPELINE_ELEMENT_SINK) {
      /* tensor_sink */
      g_object_set (G_OBJECT (elem->element), "emit-signal", (gboolean) TRUE,
          NULL);
      elem->handle_id =
          g_signal_connect (elem->element, "new-data",
          G_CALLBACK (cb_sink_event), elem);
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
  }

  sink = g_new0 (ml_pipeline_common_elem, 1);
  if (sink == NULL) {
    ml_loge ("Failed to allocate the sink handle for %s.", sink_name);
    ret = ML_ERROR_OUT_OF_MEMORY;
    goto unlock_return;
  }

  sink->callback_info = g_new (callback_info_s, 1);
  if (sink->callback_info == NULL) {
    g_free (sink);
    ml_loge ("Failed to allocate the sink handle for %s.", sink_name);
    ret = ML_ERROR_OUT_OF_MEMORY;
    goto unlock_return;
  }

  sink->pipe = p;
  sink->element = elem;
  sink->callback_info->cb = cb;
  sink->callback_info->pdata = user_data;
  *h = sink;

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
  handle_init (sink, h);

  if (elem->handle_id > 0) {
    g_signal_handler_disconnect (elem->element, elem->handle_id);
    elem->handle_id = 0;
  }

  elem->handles = g_list_remove (elem->handles, sink);
  g_free (sink->callback_info);
  g_free (sink);

  handle_exit (h);
}

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

        if (!found && gst_caps_is_fixed (caps)) {
          GstStructure *caps_s;
          const gchar *mimetype;

          caps_s = gst_caps_get_structure (caps, 0);
          mimetype = gst_structure_get_name (caps_s);

          if (!g_str_equal (mimetype, "other/tensor") &&
              !g_str_equal (mimetype, "other/tensors")) {
            elem->is_media_stream = TRUE;
          }
        }

        gst_caps_unref (caps);
      }

      if (found) {
        for (i = 0; i < elem->tensors_info.num_tensors; i++) {
          sz = ml_tensor_info_get_size (&elem->tensors_info.info[i]);
          elem->size += sz;
        }
      } else {
        if (!elem->is_media_stream) {
          ml_logw
              ("Cannot find caps. The pipeline is not yet negotiated for src element [%s].",
              elem->name);
          gst_object_unref (elem->src);
          elem->src = NULL;
          ret = ML_ERROR_TRY_AGAIN;
        }
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
  ml_pipeline_common_elem *src;
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

  src = *h = g_new0 (ml_pipeline_common_elem, 1);
  if (src == NULL) {
    ml_loge ("Failed to allocate the src handle for %s.", src_name);
    ret = ML_ERROR_OUT_OF_MEMORY;
    goto unlock_return;
  }

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
  handle_init (src, h);

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
  GstBuffer *buffer;
  GstMemory *mem;
  gpointer mem_data;
  gsize mem_size;
  GstFlowReturn gret;
  ml_tensors_data_s *_data;
  unsigned int i;

  handle_init (src, h);

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
    ml_logw
        ("The pipeline is not ready to accept inputs. The input is ignored.");
    goto destroy_data;
  }

  if (!elem->is_media_stream) {
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
  }

  /* Create buffer to be pushed from buf[] */
  buffer = gst_buffer_new ();
  for (i = 0; i < _data->num_tensors; i++) {
    mem_data = _data->tensors[i].tensor;
    mem_size = _data->tensors[i].size;

    mem = gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY,
        mem_data, mem_size, 0, mem_size, mem_data,
        (policy == ML_PIPELINE_BUF_POLICY_AUTO_FREE) ? g_free : NULL);

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
    ml_logw
        ("The pipeline is not in PAUSED/PLAYING. The input may be ignored.");
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
ml_pipeline_src_get_tensors_info (ml_pipeline_src_h h, ml_tensors_info_h * info)
{
  handle_init (src, h);

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
  ml_pipeline_common_elem *swtc;
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

  swtc = *h = g_new0 (ml_pipeline_common_elem, 1);
  if (swtc == NULL) {
    ml_loge ("Failed to allocate the switch handle for %s.", switch_name);
    ret = ML_ERROR_OUT_OF_MEMORY;
    goto unlock_return;
  }

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
  handle_init (swtc, h);

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

  handle_init (swtc, h);

  if (pad_name == NULL) {
    ml_loge ("The second argument, pad name, is not valid.");
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  g_object_get (G_OBJECT (elem->element), "active-pad", &active_pad, NULL);
  active_name = gst_pad_get_name (active_pad);

  if (g_strcmp0 (pad_name, active_name) == 0) {
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

  ml_logi ("Switched to [%s] successfully at switch [%s].", pad_name,
      elem->name);

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

  handle_init (swtc, h);

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

  gst_iterator_free (it);

  /* There has been no error with that "while" loop. */
  if (ret == ML_ERROR_NONE) {
    int i = 0;
    GList *l;

    *list = g_malloc0 (sizeof (char *) * (counter + 1));
    if (*list == NULL) {
      ml_loge ("Failed to allocate memory for pad list.");
      ret = ML_ERROR_OUT_OF_MEMORY;
      goto unlock_return;
    }

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
  ml_pipeline_common_elem *valve;
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
    ml_loge ("There is no valve element named [%s] in the pipeline.",
        valve_name);
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

  valve = *h = g_new0 (ml_pipeline_common_elem, 1);
  if (valve == NULL) {
    ml_loge ("Failed to allocate the valve handle for %s.", valve_name);
    ret = ML_ERROR_OUT_OF_MEMORY;
    goto unlock_return;
  }

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
  handle_init (valve, h);

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
  handle_init (valve, h);

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

/********************************************************
 ** NNStreamer Element Property Control in Pipeline    **
 ********************************************************/
/**
 * @brief Gets an element handle in NNStreamer pipelines to control its properties.
 */
int
ml_pipeline_element_get_handle (ml_pipeline_h pipe, const char *element_name,
    ml_pipeline_element_h * elem_h)
{
  int ret = ML_ERROR_NONE;
  ml_pipeline_element *elem;
  ml_pipeline_common_elem *common_elem;
  ml_pipeline *p = pipe;

  /* Check input parameter */
  if (pipe == NULL) {
    ml_loge ("The first argument, pipeline handle, is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (element_name == NULL) {
    ml_loge ("The second argument, element name, is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (elem_h == NULL) {
    ml_loge ("The argument element handle is not valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }
  *elem_h = NULL;

  g_mutex_lock (&p->lock);

  /* 1. Search element in lookup table first */
  elem = g_hash_table_lookup (p->namednodes, element_name);
  if (elem == NULL) {
    /* 2. Search element in pipeline itself */
    GstElement *gst_elem;

    gst_elem = gst_bin_get_by_name (GST_BIN (p->element), element_name);
    if (gst_elem == NULL) {
      ml_loge ("The element named [%s] is not found in the pipeline",
          element_name);
      ret = ML_ERROR_INVALID_PARAMETER;
      goto unlock_return;
    }

    /* Caching for next search */
    elem = construct_element (gst_elem, pipe, element_name,
        ML_PIPELINE_ELEMENT_COMMON);
    if (elem == NULL) {
      ml_loge ("Failed to allocate the internal memory");
      ret = ML_ERROR_OUT_OF_MEMORY;
      goto unlock_return;
    }
    g_hash_table_insert (p->namednodes, g_strdup (element_name), elem);
  }

  /* Type checking */
  if (elem->type == ML_PIPELINE_ELEMENT_UNKNOWN) {
    ml_loge
        ("There is an element named [%s] in the pipeline, but it is unknown type.",
        element_name);
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  common_elem = *elem_h = g_new0 (ml_pipeline_common_elem, 1);
  if (common_elem == NULL) {
    ml_loge ("Failed to allocate the internal handler for %s.", element_name);
    ret = ML_ERROR_OUT_OF_MEMORY;
    goto unlock_return;
  }

  common_elem->pipe = p;
  common_elem->element = elem;

  g_mutex_lock (&elem->lock);
  elem->maxid++;
  common_elem->id = elem->maxid;
  elem->handles = g_list_append (elem->handles, common_elem);
  g_mutex_unlock (&elem->lock);

unlock_return:
  g_mutex_unlock (&p->lock);
  return ret;
}

/**
 * @brief Releases the given element handle.
 */
int
ml_pipeline_element_release_handle (ml_pipeline_element_h elem_h)
{
  handle_init (common_elem, elem_h);

  elem->handles = g_list_remove (elem->handles, common_elem);
  g_free (common_elem);

  handle_exit (elem_h);
}

/**
 * @brief Check property existence and its type.
 */
static bool
ml_pipeline_element_check_property (GObjectClass * class,
    const char *property_name, const GType type)
{
  GParamSpec *pspec = NULL;

  /* Check property existence */
  pspec = g_object_class_find_property (class, property_name);
  if (pspec == NULL) {
    ml_loge ("The property name [%s] does not exist.", property_name);
    return FALSE;
  }

  /* Compare property's type with given type */
  if (!((pspec->value_type == type) ||
          (type == G_TYPE_ENUM && G_TYPE_IS_ENUM (pspec->value_type)) ||
          (type == G_TYPE_INT64 && pspec->value_type == G_TYPE_LONG) ||
          (type == G_TYPE_UINT64 && pspec->value_type == G_TYPE_ULONG) ||
          (type == G_TYPE_INT && G_TYPE_IS_ENUM (pspec->value_type)) ||
          (type == G_TYPE_UINT && G_TYPE_IS_ENUM (pspec->value_type)) ||
          (type == G_TYPE_DOUBLE && pspec->value_type == G_TYPE_FLOAT))) {
    ml_loge ("The type of property name [%s] is '%s'", property_name,
        g_type_name (pspec->value_type));
    return FALSE;
  }
  return TRUE;
}

/**
 * @brief Sets the value of given element's property in NNStreamer pipelines.
 */
static int
ml_pipeline_element_set_property (ml_pipeline_element_h elem_h,
    const char *property_name, gpointer value, GType type)
{
  handle_init (common_elem, elem_h);

  /* Check the input parameter */
  if (property_name == NULL) {
    ml_loge ("The second argument, property name is not valid.");
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  /* Check property existence & its type */
  if (!ml_pipeline_element_check_property (G_OBJECT_GET_CLASS (elem->element),
          property_name, type)) {
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  /* Set property */
  if (type == G_TYPE_DOUBLE || type == G_TYPE_FLOAT) {
    g_object_set (G_OBJECT (elem->element), property_name,
        *(double *) value, NULL);
  } else if (type == G_TYPE_INT64) {
    g_object_set (G_OBJECT (elem->element), property_name,
        *(int64_t *) value, NULL);
  } else if (type == G_TYPE_UINT64) {
    g_object_set (G_OBJECT (elem->element), property_name,
        *(uint64_t *) value, NULL);
  } else {
    g_object_set (G_OBJECT (elem->element), property_name, value, NULL);
  }

  handle_exit (elem_h);
}

/**
 * @brief Gets the value of given element's property in NNStreamer pipelines.
 */
static int
ml_pipeline_element_get_property (ml_pipeline_element_h elem_h,
    const char *property_name, GType type, gpointer pvalue)
{
  handle_init (common_elem, elem_h);

  /* Check the input parameter */
  if (property_name == NULL) {
    ml_loge ("The second argument, property_name is not valid.");
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  if (pvalue == NULL) {
    ml_loge ("The third argument, value is not valid.");
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  /* Check property existence & its type */
  if (!ml_pipeline_element_check_property (G_OBJECT_GET_CLASS (elem->element),
          property_name, type)) {
    ret = ML_ERROR_INVALID_PARAMETER;
    goto unlock_return;
  }

  /* Get property */
  g_object_get (G_OBJECT (elem->element), property_name, pvalue, NULL);

  handle_exit (elem_h);
}

/**
 * @brief Sets the boolean value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_set_property_bool (ml_pipeline_element_h elem_h,
    const char *property_name, const int32_t value)
{
  return ml_pipeline_element_set_property (elem_h, property_name,
      GINT_TO_POINTER (value), G_TYPE_BOOLEAN);
}

/**
 * @brief Sets the string value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_set_property_string (ml_pipeline_element_h elem_h,
    const char *property_name, const char *value)
{
  return ml_pipeline_element_set_property (elem_h, property_name,
      (gpointer) value, G_TYPE_STRING);
}

/**
 * @brief Sets the integer value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_set_property_int32 (ml_pipeline_element_h elem_h,
    const char *property_name, const int32_t value)
{
  return ml_pipeline_element_set_property (elem_h, property_name,
      GINT_TO_POINTER (value), G_TYPE_INT);
}

/**
 * @brief Sets the integer 64bit value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_set_property_int64 (ml_pipeline_element_h elem_h,
    const char *property_name, const int64_t value)
{
  return ml_pipeline_element_set_property (elem_h, property_name,
      (gpointer) (&value), G_TYPE_INT64);
}

/**
 * @brief Sets the unsigned integer value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_set_property_uint32 (ml_pipeline_element_h elem_h,
    const char *property_name, const uint32_t value)
{
  return ml_pipeline_element_set_property (elem_h, property_name,
      GUINT_TO_POINTER (value), G_TYPE_UINT);
}

/**
 * @brief Sets the unsigned integer 64bit value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_set_property_uint64 (ml_pipeline_element_h elem_h,
    const char *property_name, const uint64_t value)
{
  return ml_pipeline_element_set_property (elem_h, property_name,
      (gpointer) (&value), G_TYPE_UINT64);
}

/**
 * @brief Sets the floating point value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_set_property_double (ml_pipeline_element_h elem_h,
    const char *property_name, const double value)
{
  return ml_pipeline_element_set_property (elem_h, property_name,
      (gpointer) (&value), G_TYPE_DOUBLE);
}

/**
 * @brief Sets the enumeration value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_set_property_enum (ml_pipeline_element_h elem_h,
    const char *property_name, const uint32_t value)
{
  return ml_pipeline_element_set_property (elem_h, property_name,
      GUINT_TO_POINTER (value), G_TYPE_ENUM);
}

/**
 * @brief Gets the boolean value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_get_property_bool (ml_pipeline_element_h elem_h,
    const char *property_name, int32_t * value)
{
  return ml_pipeline_element_get_property (elem_h, property_name,
      G_TYPE_BOOLEAN, (gpointer) value);
}

/**
 * @brief Gets the string value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_get_property_string (ml_pipeline_element_h elem_h,
    const char *property_name, char **value)
{
  return ml_pipeline_element_get_property (elem_h, property_name,
      G_TYPE_STRING, (gpointer) value);
}

/**
 * @brief Gets the integer value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_get_property_int32 (ml_pipeline_element_h elem_h,
    const char *property_name, int32_t * value)
{
  return ml_pipeline_element_get_property (elem_h, property_name,
      G_TYPE_INT, (gpointer) value);
}

/**
 * @brief Gets the integer 64bit value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_get_property_int64 (ml_pipeline_element_h elem_h,
    const char *property_name, int64_t * value)
{
  return ml_pipeline_element_get_property (elem_h, property_name,
      G_TYPE_INT64, (gpointer) value);
}

/**
 * @brief Gets the unsigned integer value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_get_property_uint32 (ml_pipeline_element_h elem_h,
    const char *property_name, uint32_t * value)
{
  return ml_pipeline_element_get_property (elem_h, property_name,
      G_TYPE_UINT, (gpointer) value);
}

/**
 * @brief Gets the unsigned integer 64bit value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_get_property_uint64 (ml_pipeline_element_h elem_h,
    const char *property_name, uint64_t * value)
{
  return ml_pipeline_element_get_property (elem_h, property_name,
      G_TYPE_UINT64, (gpointer) value);
}

/**
 * @brief Gets the floating point value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_get_property_double (ml_pipeline_element_h elem_h,
    const char *property_name, double *value)
{
  return ml_pipeline_element_get_property (elem_h, property_name,
      G_TYPE_DOUBLE, (gpointer) value);
}

/**
 * @brief Gets the enumeration value of element's property in NNStreamer pipelines.
 */
int
ml_pipeline_element_get_property_enum (ml_pipeline_element_h elem_h,
    const char *property_name, uint32_t * value)
{
  return ml_pipeline_element_get_property (elem_h, property_name,
      G_TYPE_ENUM, (gpointer) value);
}

/**
 * @brief Gets the element of pipeline itself (GstElement).
 */
GstElement *
ml_pipeline_get_gst_element (ml_pipeline_h pipe)
{
  ml_pipeline *p = (ml_pipeline *) pipe;
  GstElement *element = NULL;

  if (p) {
    g_mutex_lock (&p->lock);

    element = p->element;
    if (element)
      gst_object_ref (element);

    g_mutex_unlock (&p->lock);
  }

  return element;
}

/**
 * @brief Releases custom filter handle.
 */
static void
ml_pipeline_custom_free_handle (ml_custom_filter_s * custom)
{
  if (custom) {
    g_free (custom->name);
    ml_tensors_info_destroy (custom->in_info);
    ml_tensors_info_destroy (custom->out_info);

    g_free (custom);
  }
}

/**
 * @brief Invoke callback for custom-easy filter.
 */
static int
ml_pipeline_custom_invoke (void *data, const GstTensorFilterProperties * prop,
    const GstTensorMemory * in, GstTensorMemory * out)
{
  int status;
  ml_custom_filter_s *c;
  ml_tensors_data_h in_data, out_data;
  ml_tensors_data_s *_data;
  guint i;

  c = (ml_custom_filter_s *) data;
  in_data = out_data = NULL;

  /* internal error? */
  if (!c || !c->cb)
    return -1;

  /* prepare invoke */
  status = ml_tensors_data_create_no_alloc (c->in_info, &in_data);
  if (status != ML_ERROR_NONE)
    goto done;

  _data = (ml_tensors_data_s *) in_data;
  for (i = 0; i < _data->num_tensors; i++)
    _data->tensors[i].tensor = in[i].data;

  status = ml_tensors_data_create_no_alloc (c->out_info, &out_data);
  if (status != ML_ERROR_NONE)
    goto done;

  _data = (ml_tensors_data_s *) out_data;
  for (i = 0; i < _data->num_tensors; i++)
    _data->tensors[i].tensor = out[i].data;

  /* call invoke callback */
  status = c->cb (in_data, out_data, c->pdata);

done:
  /* NOTE: DO NOT free tensor data */
  g_free (in_data);
  g_free (out_data);

  return status;
}

/**
 * @brief Registers a custom filter.
 */
int
ml_pipeline_custom_easy_filter_register (const char *name,
    const ml_tensors_info_h in, const ml_tensors_info_h out,
    ml_custom_easy_invoke_cb cb, void *user_data,
    ml_custom_easy_filter_h * custom)
{
  int status = ML_ERROR_NONE;
  ml_custom_filter_s *c;
  GstTensorsInfo in_info, out_info;

  check_feature_state ();

  if (!name || !cb || !custom)
    return ML_ERROR_INVALID_PARAMETER;

  /* init null */
  *custom = NULL;

  if (!ml_tensors_info_is_valid (in) || !ml_tensors_info_is_valid (out))
    return ML_ERROR_INVALID_PARAMETER;

  /* create and init custom handle */
  if ((c = g_new0 (ml_custom_filter_s, 1)) == NULL)
    return ML_ERROR_OUT_OF_MEMORY;

  c->name = g_strdup (name);
  c->cb = cb;
  c->pdata = user_data;
  ml_tensors_info_create (&c->in_info);
  ml_tensors_info_create (&c->out_info);

  ml_tensors_info_clone (c->in_info, in);
  ml_tensors_info_clone (c->out_info, out);

  /* register custom filter */
  ml_tensors_info_copy_from_ml (&in_info, in);
  ml_tensors_info_copy_from_ml (&out_info, out);

  if (NNS_custom_easy_register (name, ml_pipeline_custom_invoke, c,
          &in_info, &out_info) != 0) {
    nns_loge ("Failed to register custom filter %s.", name);
    status = ML_ERROR_INVALID_PARAMETER;
  }

  if (status == ML_ERROR_NONE) {
    *custom = c;
  } else {
    ml_pipeline_custom_free_handle (c);
  }

  return status;
}

/**
 * @brief Unregisters the custom filter.
 */
int
ml_pipeline_custom_easy_filter_unregister (ml_custom_easy_filter_h custom)
{
  ml_custom_filter_s *c;

  check_feature_state ();

  if (!custom)
    return ML_ERROR_INVALID_PARAMETER;

  c = (ml_custom_filter_s *) custom;

  if (NNS_custom_easy_unregister (c->name) != 0) {
    ml_loge ("Failed to unregister custom filter %s.", c->name);
    return ML_ERROR_INVALID_PARAMETER;
  }

  ml_pipeline_custom_free_handle (c);
  return ML_ERROR_NONE;
}
