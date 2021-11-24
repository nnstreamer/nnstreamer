/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer Common Header's Contents (pipeline extension)
 * Copyright (C) 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file	tensor_common_pipeline.c
 * @date	14 Apr 2020
 * @brief	Common data for NNStreamer (pipeline extension)
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <nnstreamer_util.h>
#include <string.h>
#include <tensor_common.h>

static const gchar *gst_tensor_time_sync_mode_string[] = {
  [SYNC_NOSYNC] = "nosync",
  [SYNC_SLOWEST] = "slowest",
  [SYNC_BASEPAD] = "basepad",
  [SYNC_REFRESH] = "refresh",
  [SYNC_END] = NULL
};

/**
 * @brief Get the corresponding mode from the string value.
 * @param[in] str The string value for the mode.
 * @return Corresponding mode for the string. SYNC_END for errors.
 */
tensor_time_sync_mode
gst_tensor_time_sync_get_mode (const gchar * str)
{
  gint index;

  index = find_key_strv (gst_tensor_time_sync_mode_string, str);

  return (index < 0) ? SYNC_END : index;
}

/**
 * @brief Get the time-sync mode string.
 * @return Corresponding mode string.
 */
const gchar *
gst_tensor_time_sync_get_mode_string (tensor_time_sync_mode mode)
{
  return gst_tensor_time_sync_mode_string[mode];
}

/**
 * @brief Setup time sync option.
 * @param[in/out] filter "this" pointer. Sync mode & option MUST BE set already.
 * @return True if successfully set the option.
 */
gboolean
gst_tensor_time_sync_set_option_data (tensor_time_sync_data * sync)
{
  g_return_val_if_fail (sync != NULL, FALSE);

  if (sync->mode == SYNC_END || sync->option == NULL)
    return FALSE;

  switch (sync->mode) {
    case SYNC_NOSYNC:
      break;
    case SYNC_SLOWEST:
      break;
    case SYNC_BASEPAD:
    {
      guint sink_id;
      guint duration;
      gchar **strv;

      strv = g_strsplit (sync->option, ":", 2);
      if (strv[0] != NULL)
        sink_id = (guint) g_ascii_strtoull (strv[0], NULL, 10);
      else
        sink_id = 0;

      if (strv[1] != NULL)
        duration = (guint) g_ascii_strtoull (strv[1], NULL, 10);
      else
        duration = G_MAXINT;

      sync->data_basepad.sink_id = sink_id;
      sync->data_basepad.duration = duration;
      g_strfreev (strv);
      break;
    }
    default:
      /* unknown mode */
      GST_WARNING ("Unknown mode = %d", sync->mode);
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Internal function to detect EOS using the number of empty pads.
 * @param[in] collect Collect pad.
 * @param[in] sync Synchronization option.
 * @param[in] empty The number of empty pads (pad has no buffer).
 * @return True if EOS.
 */
static gboolean
_gst_tensor_time_sync_is_eos (GstCollectPads * collect,
    tensor_time_sync_data * sync, guint empty)
{
  guint total;
  gboolean is_eos = FALSE;

  total = g_slist_length (collect->data);

  switch (sync->mode) {
    case SYNC_REFRESH:
      if (empty == total)
        is_eos = TRUE;
      break;
    default:
      if (empty > 0)
        is_eos = TRUE;
      break;
  }

  return is_eos;
}

/**
 * @brief A function call to decide current timestamp among collected pads based on PTS.
 * It will decide current timestamp according to sync option.
 * GstMeta is also copied with same sync mode.
 */
gboolean
gst_tensor_time_sync_get_current_time (GstCollectPads * collect,
    tensor_time_sync_data * sync, GstClockTime * current_time,
    GstBuffer * tensors_buf)
{
  GSList *walk = NULL;
  guint count, empty_pad;

  g_return_val_if_fail (collect != NULL, FALSE);
  g_return_val_if_fail (sync != NULL, FALSE);
  g_return_val_if_fail (current_time != NULL, FALSE);

  walk = collect->data;
  count = empty_pad = 0;

  while (walk) {
    GstCollectData *data;
    GstBuffer *buf;
    gboolean need_update = FALSE;

    data = (GstCollectData *) walk->data;
    buf = gst_collect_pads_peek (collect, data);
    walk = g_slist_next (walk);

    if (buf) {
      switch (sync->mode) {
        case SYNC_NOSYNC:
          /* fall-through */
        case SYNC_SLOWEST:
        case SYNC_REFRESH:
          if (*current_time < GST_BUFFER_PTS (buf))
            need_update = TRUE;
          break;
        case SYNC_BASEPAD:
          if (count == sync->data_basepad.sink_id)
            need_update = TRUE;
          break;
        default:
          break;
      }
      if (need_update) {
        *current_time = GST_BUFFER_PTS (buf);
        gst_buffer_copy_into (tensors_buf, buf, GST_BUFFER_COPY_METADATA,
            0, -1);
      }
      gst_buffer_unref (buf);
    } else {
      empty_pad++;
    }

    count++;
  }

  return _gst_tensor_time_sync_is_eos (collect, sync, empty_pad);
}

/**
 * @brief A function to be called while processing a flushing event.
 * It should clear old buffer and reset pad data.
 */
void
gst_tensor_time_sync_flush (GstCollectPads * collect)
{
  GSList *walk;
  GstTensorCollectPadData *pad;

  g_return_if_fail (collect != NULL);

  walk = collect->data;
  while (walk) {
    pad = (GstTensorCollectPadData *) walk->data;

    if (pad->buffer) {
      gst_buffer_unref (pad->buffer);
      pad->buffer = NULL;
    }

    walk = g_slist_next (walk);
  }
}

/**
 * @brief Common code for both
 *        gst_tensor_time_sync_buffer_from_collectpad_SYNC_*
 */
static gboolean
_gst_tensor_time_sync_buffer_update (GstBuffer ** buf,
    GstCollectPads * collect, GstCollectData * data,
    GstClockTime current, GstClockTime base, tensor_time_sync_data * sync)
{
  GstTensorCollectPadData *pad;

  pad = (GstTensorCollectPadData *) data;

  *buf = gst_collect_pads_peek (collect, data);
  if (*buf != NULL) {
    if (GST_BUFFER_PTS (*buf) < current) {
      gst_buffer_unref (*buf);
      if (pad->buffer != NULL)
        gst_buffer_unref (pad->buffer);
      pad->buffer = gst_collect_pads_pop (collect, data);
      return FALSE;
    }

    if ((sync->mode == SYNC_SLOWEST && pad->buffer != NULL &&
            (ABS (GST_CLOCK_DIFF (current, GST_BUFFER_PTS (pad->buffer))) <
                ABS (GST_CLOCK_DIFF (current, GST_BUFFER_PTS (*buf))))) ||
        (sync->mode == SYNC_BASEPAD && pad->buffer != NULL &&
            (((GstClockTime) ABS (GST_CLOCK_DIFF (current,
                            GST_BUFFER_PTS (*buf)))) > base))) {
      /* keep last buffer */
    } else {
      /* update last buffer */
      if (pad->buffer != NULL)
        gst_buffer_unref (pad->buffer);
      pad->buffer = gst_collect_pads_pop (collect, data);
    }

    gst_buffer_unref (*buf);
  }

  *buf = gst_buffer_ref (pad->buffer);
  return TRUE;
}

/**
 * @brief A function call to make tensors from collected pads.
 * It decide which buffer is going to be used according to sync option.
 * @return True to push buffer.
 */
gboolean
gst_tensor_time_sync_buffer_from_collectpad (GstCollectPads * collect,
    tensor_time_sync_data * sync, GstClockTime current_time,
    GstBuffer * tensors_buf, GstTensorsConfig * configs, gboolean * is_eos)
{
  GSList *walk = NULL;
  GstCollectData *data;
  GstTensorCollectPadData *pad;
  GstBuffer *buf = NULL;
  GstMemory *mem;
  gint old_numerator = G_MAXINT;
  gint old_denominator = G_MAXINT;
  guint counting, empty_pad;
  GstTensorsConfig in_configs;
  GstClockTime base_time = 0;
  guint i, n_mem;
  GstMemory *in_mem[NNS_TENSOR_SIZE_LIMIT];
  tensor_format in_formats[NNS_TENSOR_SIZE_LIMIT];

  g_return_val_if_fail (collect != NULL, FALSE);
  g_return_val_if_fail (sync != NULL, FALSE);
  g_return_val_if_fail (tensors_buf != NULL, FALSE);
  g_return_val_if_fail (configs != NULL, FALSE);
  g_return_val_if_fail (is_eos != NULL, FALSE);

  walk = collect->data;
  counting = empty_pad = 0;

  if (sync->mode == SYNC_BASEPAD) {
    walk = g_slist_nth (walk, sync->data_basepad.sink_id);
    if (walk == NULL) {
      GST_ERROR_OBJECT (collect, "Cannot get GstCollectData from GSList");
      return FALSE;
    }

    data = (GstCollectData *) walk->data;
    pad = (GstTensorCollectPadData *) data;

    buf = gst_collect_pads_peek (collect, data);
    if (buf != NULL) {
      if (pad->buffer != NULL)
        base_time =
            MIN ((GstClockTimeDiff) sync->data_basepad.duration,
            ABS (GST_CLOCK_DIFF (GST_BUFFER_PTS (buf),
                    GST_BUFFER_PTS (pad->buffer))) - 1);
      gst_buffer_unref (buf);
    }
  }

  walk = collect->data;

  while (walk) {
    gboolean configured = FALSE;
    gboolean is_empty = FALSE;

    data = (GstCollectData *) walk->data;
    pad = (GstTensorCollectPadData *) data;

    if (gst_pad_has_current_caps (pad->pad)) {
      GstCaps *caps = gst_pad_get_current_caps (pad->pad);
      GstStructure *s = gst_caps_get_structure (caps, 0);

      gst_tensors_config_from_structure (&in_configs, s);
      gst_caps_unref (caps);

      configured = gst_tensors_config_validate (&in_configs);
    }

    /**
     * This would be an internal logic error.
     * in_configs should be already confirmed valid at the negotiation phase
     * and this function should be called in a running pipeline.
     * If new sync mode is enabled (e.g., handle output when a pad gets new buffer),
     * this may cause unexpected exception.
     */
    if (!configured) {
      return FALSE;
    }

    if (in_configs.rate_d < old_denominator)
      old_denominator = in_configs.rate_d;
    if (in_configs.rate_n < old_numerator)
      old_numerator = in_configs.rate_n;

    walk = g_slist_next (walk);

    switch (sync->mode) {
      case SYNC_SLOWEST:
        /* fall-through */
      case SYNC_BASEPAD:
        if (!_gst_tensor_time_sync_buffer_update (&buf, collect, data,
                current_time, base_time, sync))
          return FALSE;
        is_empty = (buf == NULL);
        break;
      case SYNC_NOSYNC:
        buf = gst_collect_pads_pop (collect, data);
        is_empty = (buf == NULL);
        break;
      case SYNC_REFRESH:
        buf = gst_collect_pads_pop (collect, data);
        if (buf != NULL) {
          if (pad->buffer != NULL) {
            gst_buffer_unref (pad->buffer);
          }
          pad->buffer = gst_buffer_ref (buf);
        } else {
          if (pad->buffer == NULL) {
            *is_eos = FALSE;
            ml_logd ("Not the all buffers are arrived yet.");
            return FALSE;
          }
          is_empty = TRUE;
          buf = gst_buffer_ref (pad->buffer);
        }
        break;
      default:
        break;
    }

    if (GST_IS_BUFFER (buf)) {
      buf = gst_tensor_buffer_from_config (buf, &in_configs);
      n_mem = gst_buffer_n_memory (buf);

      /** These are internal logic error. If given inputs are incorrect,
          the negotiation should have been failed before this stage. */
      if (gst_tensors_config_is_static (&in_configs))
        g_assert (n_mem == in_configs.info.num_tensors);
      g_assert ((counting + n_mem) < NNS_TENSOR_SIZE_LIMIT);

      if (gst_tensors_config_is_flexible (&in_configs))
        configs->format = _NNS_TENSOR_FORMAT_FLEXIBLE;

      for (i = 0; i < n_mem; ++i) {
        in_mem[counting] = gst_buffer_get_memory (buf, i);

        configs->info.info[counting] = in_configs.info.info[i];
        in_formats[counting] = in_configs.format;
        counting++;
      }

      gst_buffer_unref (buf);
    }
    if (is_empty)
      empty_pad++;
  }

  /* append memories to output buffer */
  for (i = 0; i < counting; i++) {
    mem = in_mem[i];

    if (gst_tensors_config_is_flexible (configs)) {
      /* append header if input tensor is not flexible */
      if (in_formats[i] != _NNS_TENSOR_FORMAT_FLEXIBLE) {
        GstTensorMetaInfo meta;

        gst_tensor_info_convert_to_meta (&configs->info.info[i], &meta);
        mem = gst_tensor_meta_info_append_header (&meta, in_mem[i]);
        gst_memory_unref (in_mem[i]);
      }
    }

    gst_buffer_append_memory (tensors_buf, mem);
  }

  configs->info.num_tensors = counting;
  configs->rate_d = old_denominator;
  configs->rate_n = old_numerator;

  GST_BUFFER_PTS (tensors_buf) = current_time;

  /* check eos */
  *is_eos = _gst_tensor_time_sync_is_eos (collect, sync, empty_pad);
  return !(*is_eos);
}

/**
 * @brief Configure gst-buffer with tensors information.
 * NNStreamer handles single memory chunk as single tensor.
 * If incoming buffer has invalid memories, separate it and generate new gst-buffer using tensors information.
 * Note that this function always takes the ownership of input buffer.
 * @param in input buffer
 * @param config tensors config structure
 * @return Newly allocated buffer. Null if failed. Caller should unref the buffer using gst_buffer_unref().
 */
GstBuffer *
gst_tensor_buffer_from_config (GstBuffer * in, GstTensorsConfig * config)
{
  GstBuffer *out = NULL;
  GstMemory *all = NULL;
  GstMapInfo map;
  guint i, num;
  gsize total, offset;
  gsize mem_size[NNS_TENSOR_SIZE_LIMIT];
  gboolean configured = FALSE;

  if (!GST_IS_BUFFER (in)) {
    nns_loge ("Failed to get tensor buffer, invalid input buffer.");
    return NULL;
  }

  if (!gst_tensors_config_validate (config)) {
    nns_loge ("Failed to get tensor buffer, invalid tensor configuration.");
    goto error;
  }

  num = gst_buffer_n_memory (in);
  total = gst_buffer_get_size (in);

  /* get memory size */
  if (gst_tensors_config_is_static (config)) {
    if (num == config->info.num_tensors) {
      /* Do nothing, pass input buffer. */
      out = gst_buffer_ref (in);
      goto done;
    }

    num = config->info.num_tensors;
    for (i = 0; i < num; i++)
      mem_size[i] = gst_tensors_info_get_size (&config->info, i);
  } else {
    if (num > 1) {
      /* Suppose it is already configured. */
      out = gst_buffer_ref (in);
      goto done;
    }

    if (!gst_buffer_map (in, &map, GST_MAP_READ)) {
      nns_loge ("Failed to get tensor buffer, cannot get the memory info.");
      goto error;
    }

    num = 0;
    offset = 0;
    while (offset < total) {
      GstTensorMetaInfo meta;
      gpointer h = map.data + offset;

      gst_tensor_meta_info_parse_header (&meta, h);
      mem_size[num] = gst_tensor_meta_info_get_header_size (&meta);
      mem_size[num] += gst_tensor_meta_info_get_data_size (&meta);

      offset += mem_size[num];
      num++;
    }

    gst_buffer_unmap (in, &map);

    if (num == 1) {
      /* Do nothing, pass input buffer. */
      out = gst_buffer_ref (in);
      goto done;
    }
  }

  /* configure output buffer */
  out = gst_buffer_new ();
  all = gst_buffer_get_all_memory (in);
  offset = 0;

  for (i = 0; i < num; i++) {
    /* invalid memory size */
    if (offset + mem_size[i] > total) {
      nns_loge ("Failed to get tensor buffer, data size is mismatched.");
      goto error;
    }

    gst_buffer_append_memory (out, gst_memory_share (all, offset, mem_size[i]));
    offset += mem_size[i];
  }

  gst_buffer_copy_into (out, in, GST_BUFFER_COPY_METADATA, 0, -1);

done:
  configured = TRUE;
error:
  gst_buffer_unref (in);

  if (all)
    gst_memory_unref (all);

  if (!configured) {
    if (out) {
      gst_buffer_unref (out);
      out = NULL;
    }
  }

  return out;
}

/**
 * @brief Internal struct to handle aggregation data in hash table.
 */
typedef struct
{
  GstAdapter *adapter;
} gst_tensor_aggregation_data_s;

#define AGGREGATION_DEFAULT_KEY 0xC0FFEEU

/**
 * @brief Internal function to free aggregation data.
 */
static void
gst_tensor_aggregation_free_data (gpointer data)
{
  gst_tensor_aggregation_data_s *aggr;

  aggr = (gst_tensor_aggregation_data_s *) data;
  if (aggr) {
    gst_adapter_clear (aggr->adapter);
    g_object_unref (aggr->adapter);

    g_free (aggr);
  }
}

/**
 * @brief Internal function to add new aggregation data.
 */
static gst_tensor_aggregation_data_s *
gst_tensor_aggregation_add_data (GHashTable * table, const guint32 key)
{
  gst_tensor_aggregation_data_s *aggr;
  guint32 hashkey;

  g_return_val_if_fail (table != NULL, NULL);
  if (key == 0)
    hashkey = AGGREGATION_DEFAULT_KEY;
  else
    hashkey = key;
  aggr = g_new0 (gst_tensor_aggregation_data_s, 1);
  aggr->adapter = gst_adapter_new ();

  g_hash_table_insert (table, GINT_TO_POINTER (hashkey), aggr);
  return aggr;
}

/**
 * @brief Internal function to get aggregation data.
 */
static gst_tensor_aggregation_data_s *
gst_tensor_aggregation_get_data (GHashTable * table, const guint32 key)
{
  g_return_val_if_fail (table != NULL, NULL);

  return (gst_tensor_aggregation_data_s *) g_hash_table_lookup (table,
      GINT_TO_POINTER (key == 0 ? AGGREGATION_DEFAULT_KEY : key));
}

/**
 * @brief Internal function to remove all buffers from aggregation data.
 */
static void
gst_tensor_aggregation_clear_internal (gpointer key, gpointer value,
    gpointer user_data)
{
  gst_tensor_aggregation_data_s *aggr;

  UNUSED (key);
  UNUSED (user_data);

  aggr = (gst_tensor_aggregation_data_s *) value;
  if (aggr) {
    gst_adapter_clear (aggr->adapter);
  }
}

/**
 * @brief Gets new hash table for tensor aggregation.
 * @return Newly allocated hash table, caller should release this using g_hash_table_destroy().
 */
GHashTable *
gst_tensor_aggregation_init (void)
{
  GHashTable *table;

  table = g_hash_table_new_full (g_direct_hash, g_direct_equal, NULL,
      gst_tensor_aggregation_free_data);

  /**
   * Add default adapter (for the case if buffer has no specific id).
   * If gst-buffer has tensor-meta which includes client-id,
   * e.g., aggregation frames from multiple clients on query-server pipeline,
   * nnstreamer element should parse meta and request adapter with this id.
   * However, on normal pipeline, gst-buffer does not contain tensor-meta,
   * then the element may request adapter with null key string.
   */
  gst_tensor_aggregation_add_data (table, AGGREGATION_DEFAULT_KEY);

  return table;
}

/**
 * @brief Clears buffers from adapter.
 * @param table a hash table instance initialized with gst_tensor_aggregation_init()
 * @param key the key to look up (set null to get default adapter)
 */
void
gst_tensor_aggregation_clear (GHashTable * table, const guint32 key)
{
  gst_tensor_aggregation_data_s *aggr;

  g_return_if_fail (table != NULL);

  aggr = gst_tensor_aggregation_get_data (table, key);
  gst_tensor_aggregation_clear_internal (NULL, aggr, NULL);
}

/**
 * @brief Clears buffers from all adapters in hash table.
 * @param table a hash table instance initialized with gst_tensor_aggregation_init()
 */
void
gst_tensor_aggregation_clear_all (GHashTable * table)
{
  g_hash_table_foreach (table, gst_tensor_aggregation_clear_internal, NULL);
}

/**
 * @brief Gets adapter from hash table.
 * @param table a hash table instance initialized with gst_tensor_aggregation_init()
 * @param key the key to look up (set null to get default adapter)
 * @return gst-adapter instance. DO NOT release this instance.
 */
GstAdapter *
gst_tensor_aggregation_get_adapter (GHashTable * table, const guint32 key)
{
  gst_tensor_aggregation_data_s *aggr;

  g_return_val_if_fail (table != NULL, NULL);

  aggr = gst_tensor_aggregation_get_data (table, key);
  if (!aggr) {
    /*append new data */
    aggr = gst_tensor_aggregation_add_data (table, key);
  }

  return aggr->adapter;
}
