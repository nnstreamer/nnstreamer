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
  g_mutex_init (&ret->lock);
  return ret;
}

/**
 * @brief Handle a sink element for registered nns_sink_cb
 */
static void
cb_sink_event (GstElement * e, GstBuffer * b, gpointer data)
{
  element *elem = data;

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
    dlog_print (DLOG_ERROR, "nnstreamer-capi-pipeline",
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

          g_assert (elem->tensorsinfo.num_tensors == num_mems);

          for (i = 0; i < elem->tensorsinfo.num_tensors; i++) {
            size_t sz = gst_tensor_info_get_size (&elem->tensorsinfo.info[i]);

            g_assert (sz == size[i]);
            elem->size += sz;
          }
        } else {
          elem->sink = NULL;    /* It is not valid */
          /** @todo What if it keeps being "NULL"? Exception handling at 2nd frame? */
        }
      }
    }
  }

  g_assert (gst_buffer_get_size (b) == total_size);
  if (elem->size > 0)
    g_assert (gst_buffer_get_size (b) == elem->size);

  /* Iterate e->handles, pass the data to them */
  for (l = elem->handles; l != NULL; l = l->next) {
    nns_sink *sink = l->data;
    nns_sink_cb callback = sink->cb;

    callback (buf, size, &elem->tensorsinfo, sink->pdata);

    /** @todo Measure time. Warn if it takes long. Kill if it takes too long. */
  }

  g_mutex_unlock (&elem->lock);

  for (i = 0; i < num_mems; i++) {
    gst_memory_unmap (mem[i], &info[i]);
  }

  return;
}

/**
 * @brief Construct the pipeline (more info in tizen-api.h)
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
      dlog_print (DLOG_ERROR, "nnstreamer-capi-pipeline",
          "Gstreamer has the following error: %s", err->message);
      g_error_free (err);
    } else {
      dlog_print (DLOG_ERROR, "nnstreamer-capi-pipeline",
          "Cannot initialize gstreamer. Unknown reason.");
    }
    return NNS_ERROR_PIPELINE_FAIL;
  }

  pipeline = gst_parse_launch (pipeline_description, &err);
  if (pipeline == NULL) {
    if (err) {
      dlog_print (DLOG_ERROR, "nnstreamer-capi-pipeline",
          "Cannot parse and launch the given pipeline = [%s], %s",
          pipeline_description, err->message);
      g_error_free (err);
    } else {
      dlog_print (DLOG_ERROR, "nnstreamer-capi-pipeline",
          "Cannot parse and launch the given pipeline = [%s], unknown reason",
          pipeline_description);
    }
    return NNS_ERROR_PIPELINE_FAIL;
  }

  pipe_h = g_new0 (nns_pipeline, 1);
  *pipe = pipe_h;
  g_assert (GST_IS_PIPELINE (pipeline));
  pipe_h->element = pipeline;
  g_mutex_init (&pipe_h->lock);
  g_mutex_lock (&pipe_h->lock);

  pipe_h->namednodes = g_hash_table_new (g_str_hash, g_str_equal);

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
              }

              if (e != NULL)
                g_hash_table_insert (pipe_h->namednodes, e, name);
            }
            g_free (name);
          }
          g_value_reset (&item);

          break;
        case GST_ITERATOR_RESYNC:
        case GST_ITERATOR_ERROR:
          dlog_print (DLOG_WARN, "nnstreamer-capi-pipeline",
              "There is an error or a resync-event while inspecting a pipeline. However, we can still execute the pipeline.");
        case GST_ITERATOR_DONE:
          done = TRUE;
      }
    }
    g_value_unset (&item);
    gst_iterator_free (it);

    g_object_unref (tensor_sink);
    g_object_unref (valve);
    g_object_unref (inputs);
    g_object_unref (outputs);
  }

  /** @todo CRITICAL: Prepare the pipeline. Maybe as a forked thread */


  g_mutex_unlock (&pipe_h->lock);
  return ret;
}

/**
 * @brief Destroy the pipeline (more info in tizen-api.h)
 */
int
nns_pipeline_destroy (nns_pipeline_h pipe)
{
  /* nns_pipeline *p = pipe; */

  /** @todo NYI */

  /** @todo Pause the pipeline if it's Playing */

  /** @todo Ensure all callbacks are gone. (kill'em all!) */

  /** @todo Stop (NULL State) the pipeline */

  /** @todo Destroy Everything */

  return NNS_ERROR_NONE;
}

/**
 * @brief Get the pipeline state (more info in tizen-api.h)
 */
int
nns_pipeline_getstate (nns_pipeline_h pipe, nns_pipeline_state * state)
{
  /* *state = NNSAPI_UNKNOWN; */

  /** @todo NYI */

  return NNS_ERROR_NONE;
}

/****************************************************
 ** NNStreamer Pipeline Start/Stop Control         **
 ****************************************************/
/**
 * @brief Start/Resume the pipeline! (more info in tizen-api.h)
 */
int
nns_pipeline_start (nns_pipeline_h pipe)
{
  /** @todo NYI */

  return NNS_ERROR_NONE;
}

/**
 * @brief Pause the pipeline! (more info in tizen-api.h)
 */
int
nns_pipeline_stop (nns_pipeline_h pipe)
{
  /** @todo NYI */

  return NNS_ERROR_NONE;
}

/****************************************************
 ** NNStreamer Pipeline Sink/Src Control           **
 ****************************************************/
/**
 * @brief Register a callback for sink (more info in tizen-api.h)
 */
int
nns_pipeline_sink_register (nns_pipeline_h pipe, const char *sinkname,
    nns_sink_cb cb, nns_sink_h * h, void *pdata)
{
  /** @todo NYI */

  return NNS_ERROR_NONE;
}

/**
 * @brief Unregister a callback for sink (more info in tizen-api.h)
 */
int
nns_pipeline_sink_unregister (nns_sink_h h)
{
  /** @todo NYI */

  return NNS_ERROR_NONE;
}

/**
 * @brief Get a handle to operate a src (more info in tizen-api.h)
 */
int nns_pipeline_src_gethandle
    (nns_pipeline_h pipe, const char *srcname, GstTensorsInfo * tensorsinfo,
    nns_src_h * h)
{
  /** @todo NYI */

  return NNS_ERROR_NONE;
}

/**
 * @brief Close a src node (more info in tizen-api.h)
 */
int
nns_pipeline_src_puthandle (nns_src_h h)
{
  /** @todo NYI */

  return NNS_ERROR_NONE;
}

/**
 * @brief Push a data frame to a src (more info in tizen-api.h)
 */
int
nns_pipeline_src_inputdata (nns_src_h h,
    nns_buf_policy policy, char *buf, size_t size)
{
  /** @todo NYI */

  return NNS_ERROR_NONE;
}

/****************************************************
 ** NNStreamer Pipeline Switch/Valve Control       **
 ****************************************************/

/**
 * @brief Get a handle to operate a selector (more info in tizen-api.h)
 */
int nns_pipeline_switch_gethandle
    (nns_pipeline_h pipe, const char *switchname, int *num_nodes,
    nns_switch_type type, nns_switch_h * h)
{
  /** @todo NYI */

  return NNS_ERROR_NONE;
}

/**
 * @brief Close the given switch handle (more info in tizen-api.h)
 */
int
nns_pipeline_switch_puthandle (nns_switch_h h)
{
  /** @todo NYI */

  return NNS_ERROR_NONE;
}

/**
 * @brief Control the switch (more info in tizen-api.h)
 */
int
nns_pipeline_switch_select (nns_switch_h h, int node)
{
  /** @todo NYI */

  return NNS_ERROR_NONE;
}

/**
 * @brief Get a handle to operate a Valve (more info in tizen-api.h)
 */
int nns_pipeline_valve_gethandle
    (nns_pipeline_h pipe, const char *valvename, nns_valve_h * h)
{
  /** @todo NYI */

  return NNS_ERROR_NONE;
}

/**
 * @brief Close the given valve handle (more info in tizen-api.h)
 */
int
nns_pipeline_valve_puthandle (nns_valve_h h)
{
  /** @todo NYI */

  return NNS_ERROR_NONE;
}

/**
 * @brief Control the valve with the given handle (more info in tizen-api.h)
 */
int
nns_pipeline_valve_control (nns_valve_h h, int valve_open)
{
  /** @todo NYI */

  return NNS_ERROR_NONE;
}
