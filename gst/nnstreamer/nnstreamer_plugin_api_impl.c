/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer Common Header's Contents (pipeline extension)
 * Copyright (C) 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file	nnstreamer_plugin_api_impl.c
 * @date	14 Apr 2020
 * @brief	Common data for NNStreamer, the GStreamer plugin for neural networks
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

#define NNS_TENSOR_EXTRA_MAGIC 0xf00dc0de

/**
 * @brief Data structure to describe a "extra" tensor data.
 * This represents the information of the NNS_TENSOR_SIZE_LIMIT-th memory block for tensor stream.
 */
typedef struct
{
  uint32_t magic;
  uint32_t version;
  uint32_t num_extra_tensors;
  uint64_t reserved;
  GstTensorInfo infos[NNS_TENSOR_SIZE_EXTRA_LIMIT];
} GstTensorExtraInfo;

/**
 * @brief Check if given memory has extra tensors.
 * @param[in] map GstMapInfo of GstMemory to be checked.
 * @return TRUE if @map has extra tensors, otherwise FALSE.
 */
static gboolean
gst_memory_map_is_extra_tensor (GstMapInfo * map)
{
  GstTensorExtraInfo *extra_info;
  gboolean is_extra;

  g_return_val_if_fail (map != NULL, FALSE);

  if (map->size < sizeof (GstTensorExtraInfo))
    return FALSE;

  extra_info = (GstTensorExtraInfo *) map->data;

  /* check magic in header (extra info) of the memory */
  is_extra = (extra_info && extra_info->magic == NNS_TENSOR_EXTRA_MAGIC);

  return is_extra;
}

/**
 * @brief Initialize GstTensorExtraInfo structure with given @a memory.
 * @param[in/out] extra GstTensorExtraInfo to be initialized.
 * @param[in] reserved_size The memory size of extra memory block.
 */
static void
gst_tensor_extra_info_init (GstTensorExtraInfo * extra, gsize reserved_size)
{
  guint i;

  g_return_if_fail (extra != NULL);

  extra->magic = NNS_TENSOR_EXTRA_MAGIC;
  extra->version = 0;
  extra->num_extra_tensors = 0;

  /* set reserved size of NNS_TENSOR_SIZE_LIMIT-th memory */
  extra->reserved = reserved_size;
  for (i = 0; i < NNS_TENSOR_SIZE_EXTRA_LIMIT; ++i) {
    gst_tensor_info_init (&extra->infos[i]);
  }
}

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
 * @brief Internal function to update buffer in pad data based on the sync mode.
 */
static gboolean
_gst_tensor_time_sync_buffer_update (GstCollectPads * collect,
    GstCollectData * data, GstClockTime current, GstClockTime base,
    tensor_time_sync_data * sync)
{
  GstTensorCollectPadData *pad;
  GstBuffer *buf;

  pad = (GstTensorCollectPadData *) data;

  buf = gst_collect_pads_peek (collect, data);
  if (buf != NULL) {
    if (GST_BUFFER_PTS (buf) < current) {
      gst_buffer_unref (buf);
      if (pad->buffer != NULL)
        gst_buffer_unref (pad->buffer);
      pad->buffer = gst_collect_pads_pop (collect, data);
      return FALSE;
    }

    if ((sync->mode == SYNC_SLOWEST && pad->buffer != NULL &&
            (ABS (GST_CLOCK_DIFF (current, GST_BUFFER_PTS (pad->buffer))) <
                ABS (GST_CLOCK_DIFF (current, GST_BUFFER_PTS (buf))))) ||
        (sync->mode == SYNC_BASEPAD && pad->buffer != NULL &&
            (((GstClockTime) ABS (GST_CLOCK_DIFF (current,
                            GST_BUFFER_PTS (buf)))) > base))) {
      /* keep last buffer */
    } else {
      /* update last buffer */
      if (pad->buffer != NULL)
        gst_buffer_unref (pad->buffer);
      pad->buffer = gst_collect_pads_pop (collect, data);
    }

    gst_buffer_unref (buf);
  }

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
  guint i;
  GstMemory *in_mem[NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT];
  tensor_format in_formats[NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT];

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

  gst_tensors_config_init (&in_configs);

  while (walk) {
    gboolean configured = FALSE;
    gboolean is_empty = FALSE;

    data = (GstCollectData *) walk->data;
    pad = (GstTensorCollectPadData *) data;

    if (gst_pad_has_current_caps (pad->pad)) {
      GstCaps *caps = gst_pad_get_current_caps (pad->pad);
      GstStructure *s = gst_caps_get_structure (caps, 0);

      if (gst_tensors_config_validate (&in_configs))
        gst_tensors_config_free (&in_configs);

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
        if (!_gst_tensor_time_sync_buffer_update (collect, data,
                current_time, base_time, sync))
          return FALSE;
        buf = gst_buffer_ref (pad->buffer);
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
      guint32 n_tensor = gst_tensor_buffer_get_count (buf);
      buf = gst_tensor_buffer_from_config (buf, &in_configs);

      /** These are internal logic error. If given inputs are incorrect,
          the negotiation should have been failed before this stage. */
      if (gst_tensors_config_is_static (&in_configs))
        g_assert (n_tensor == in_configs.info.num_tensors);
      g_assert ((counting + n_tensor) <=
          NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT);

      if (gst_tensors_config_is_flexible (&in_configs))
        configs->info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

      for (i = 0; i < n_tensor; ++i) {
        in_mem[counting] = gst_tensor_buffer_get_nth_memory (buf, i);

        /* set info */
        gst_tensor_info_copy (gst_tensors_info_get_nth_info (&configs->info,
                counting), gst_tensors_info_get_nth_info (&in_configs.info, i));
        in_formats[counting] = in_configs.info.format;
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
      if (i >= NNS_TENSOR_SIZE_LIMIT) {
        /** @todo support extra tensors in case of flexible */
        nns_loge ("The extra tensor is not supported for flexible tensor.");
        return FALSE;
      }

      if (in_formats[i] != _NNS_TENSOR_FORMAT_FLEXIBLE) {
        GstTensorMetaInfo meta;

        gst_tensor_info_convert_to_meta (&configs->info.info[i], &meta);
        mem = gst_tensor_meta_info_append_header (&meta, in_mem[i]);
        gst_memory_unref (in_mem[i]);
      }
    }

    if (i < NNS_TENSOR_SIZE_LIMIT) {
      gst_buffer_append_memory (tensors_buf, mem);
    } else {
      GstTensorInfo *info = &configs->info.extra[i - NNS_TENSOR_SIZE_LIMIT];
      if (!gst_tensor_buffer_append_memory (tensors_buf, mem, info)) {
        nns_loge ("Failed to append memory to buffer.");
        return FALSE;
      }
    }
  }

  configs->info.num_tensors = counting;
  configs->rate_d = old_denominator;
  configs->rate_n = old_numerator;

  GST_BUFFER_PTS (tensors_buf) = current_time;

  gst_tensors_config_free (&in_configs);

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
  gboolean is_extra = FALSE;

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
    if ((is_extra = (num > NNS_TENSOR_SIZE_LIMIT)))
      num = NNS_TENSOR_SIZE_LIMIT;
    for (i = 0; i < num; i++)
      mem_size[i] = gst_tensors_info_get_size (&config->info, i);
    if (is_extra) {
      mem_size[num - 1] += sizeof (GstTensorExtraInfo);
      for (; i < config->info.num_tensors; i++)
        mem_size[num - 1] += gst_tensors_info_get_size (&config->info, i);
    }
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

      if (num >= NNS_TENSOR_SIZE_LIMIT - 1) {
        /* Suppose remained memory may include extra tensors. */
        mem_size[num++] = total - offset;
        break;
      }

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

/**
 * @brief Internal function to get caps for single tensor from config.
 */
static GstCaps *
_get_tensor_caps (const GstTensorsConfig * config)
{
  GstCaps *caps;
  GstStructure *structure;
  const GstTensorInfo *_info;

  g_return_val_if_fail (config != NULL, NULL);

  _info = &config->info.info[0];

  if (config->info.num_tensors > 1)
    return NULL;

  caps = gst_caps_from_string (GST_TENSOR_CAP_DEFAULT);

  /* structure for backward compatibility */
  structure = gst_structure_new_empty (NNS_MIMETYPE_TENSOR);

  if (gst_tensor_dimension_is_valid (_info->dimension)) {
    gchar *dim_str = gst_tensor_get_dimension_string (_info->dimension);

    gst_caps_set_simple (caps, "dimension", G_TYPE_STRING, dim_str, NULL);
    g_free (dim_str);

    dim_str =
        gst_tensor_get_rank_dimension_string (_info->dimension,
        NNS_TENSOR_RANK_LIMIT_PREV);
    gst_structure_set (structure, "dimension", G_TYPE_STRING, dim_str, NULL);
    g_free (dim_str);
  }

  if (_info->type != _NNS_END) {
    gst_caps_set_simple (caps, "type", G_TYPE_STRING,
        gst_tensor_get_type_string (_info->type), NULL);
    gst_structure_set (structure, "type", G_TYPE_STRING,
        gst_tensor_get_type_string (_info->type), NULL);
  }

  if (config->rate_n >= 0 && config->rate_d > 0) {
    gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION,
        config->rate_n, config->rate_d, NULL);
    gst_structure_set (structure, "framerate", GST_TYPE_FRACTION,
        config->rate_n, config->rate_d, NULL);
  }

  gst_caps_append_structure (caps, structure);
  return caps;
}

/**
 * @brief Internal function to get caps for multi tensors from config.
 */
static GstCaps *
_get_tensors_caps (const GstTensorsConfig * config)
{
  GstCaps *caps;
  GstStructure *structure;

  g_return_val_if_fail (config != NULL, NULL);

  caps = gst_caps_from_string (GST_TENSORS_CAP_DEFAULT);

  /* structure for backward compatibility */
  structure = gst_structure_new_empty (NNS_MIMETYPE_TENSORS);

  if (config->info.num_tensors > 0) {
    gchar *dim_str, *type_str;

    dim_str = gst_tensors_info_get_dimensions_string (&config->info);
    type_str = gst_tensors_info_get_types_string (&config->info);

    gst_caps_set_simple (caps, "num_tensors", G_TYPE_INT,
        config->info.num_tensors, NULL);
    gst_caps_set_simple (caps, "dimensions", G_TYPE_STRING, dim_str, NULL);
    gst_caps_set_simple (caps, "types", G_TYPE_STRING, type_str, NULL);
    g_free (dim_str);

    dim_str =
        gst_tensors_info_get_rank_dimensions_string (&config->info,
        NNS_TENSOR_RANK_LIMIT_PREV);

    gst_structure_set (structure, "num_tensors", G_TYPE_INT,
        config->info.num_tensors, NULL);
    gst_structure_set (structure, "dimensions", G_TYPE_STRING, dim_str, NULL);
    gst_structure_set (structure, "types", G_TYPE_STRING, type_str, NULL);

    g_free (dim_str);
    g_free (type_str);
  }

  if (config->rate_n >= 0 && config->rate_d > 0) {
    gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION,
        config->rate_n, config->rate_d, NULL);
    gst_structure_set (structure, "framerate", GST_TYPE_FRACTION,
        config->rate_n, config->rate_d, NULL);
  }

  gst_caps_append_structure (caps, structure);
  return caps;
}

/**
 * @brief Internal function to get caps for flexible tensor from config.
 */
static GstCaps *
_get_flexible_caps (const GstTensorsConfig * config)
{
  GstCaps *caps;

  caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);

  if (config->rate_n >= 0 && config->rate_d > 0) {
    gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION,
        config->rate_n, config->rate_d, NULL);
  }

  return caps;
}

/**
 * @brief Check given mimetype is tensor stream.
 * @param structure structure to be interpreted
 * @return TRUE if mimetype is tensor stream
 */
gboolean
gst_structure_is_tensor_stream (const GstStructure * structure)
{
  const gchar *name;

  name = gst_structure_get_name (structure);
  g_return_val_if_fail (name != NULL, FALSE);

  return (g_str_equal (name, NNS_MIMETYPE_TENSOR) ||
      g_str_equal (name, NNS_MIMETYPE_TENSORS));
}

/**
 * @brief Get media type from structure
 * @param structure structure to be interpreted
 * @return corresponding media type (returns _NNS_MEDIA_INVALID for unsupported type)
 */
media_type
gst_structure_get_media_type (const GstStructure * structure)
{
  const gchar *name;

  name = gst_structure_get_name (structure);

  g_return_val_if_fail (name != NULL, _NNS_MEDIA_INVALID);

  if (g_str_has_prefix (name, "video/")) {
    return _NNS_VIDEO;
  }

  if (g_str_has_prefix (name, "audio/")) {
    return _NNS_AUDIO;
  }

  if (g_str_has_prefix (name, "text/")) {
    return _NNS_TEXT;
  }

  if (g_str_equal (name, "application/octet-stream")) {
    return _NNS_OCTET;
  }

  if (gst_structure_is_tensor_stream (structure)) {
    return _NNS_TENSOR;
  }

  /* unknown or unsupported type */
  return _NNS_MEDIA_INVALID;
}

/**
 * @brief Parse caps from peer pad and set tensors config.
 * @param pad GstPad to get the capabilities
 * @param config tensors config structure to be filled
 * @param is_fixed flag to be updated when peer caps is fixed (not mandatory, do nothing when the param is null)
 * @return TRUE if successfully configured from peer
 */
gboolean
gst_tensors_config_from_peer (GstPad * pad, GstTensorsConfig * config,
    gboolean * is_fixed)
{
  GstCaps *peer_caps;
  GstStructure *structure;
  gboolean ret = FALSE;

  g_return_val_if_fail (GST_IS_PAD (pad), FALSE);
  g_return_val_if_fail (config != NULL, FALSE);

  gst_tensors_config_init (config);

  if ((peer_caps = gst_pad_peer_query_caps (pad, NULL))) {
    if (gst_caps_get_size (peer_caps) > 0) {
      structure = gst_caps_get_structure (peer_caps, 0);
      ret = gst_tensors_config_from_structure (config, structure);
    }

    if (ret && is_fixed)
      *is_fixed = gst_caps_is_fixed (peer_caps);

    gst_caps_unref (peer_caps);
  }

  return ret;
}

/**
 * @brief Check whether two structures have the same dimension
 */
static gboolean
_is_structure_dimension_same (GstStructure * st1, GstStructure * st2,
    const gchar * fieldname)
{
  const char *dim_str1;
  const char *dim_str2;

  g_return_val_if_fail (gst_structure_has_field (st1, fieldname), FALSE);
  g_return_val_if_fail (gst_structure_has_field (st2, fieldname), FALSE);

  dim_str1 = gst_structure_get_string (st1, fieldname);
  dim_str2 = gst_structure_get_string (st2, fieldname);

  return gst_tensor_dimension_string_is_equal (dim_str1, dim_str2);
}

/**
 * @brief Update caps dimensions for negotiation
 * @param caps caps to compare and update
 * @param peer_caps caps to compare
 */
void
gst_tensor_caps_update_dimension (GstCaps * caps, GstCaps * peer_caps)
{
  GstStructure *structure;
  GstStructure *structure_peer;
  guint i, j;

  g_return_if_fail (caps != NULL);
  g_return_if_fail (peer_caps != NULL);

  for (i = 0; i < gst_caps_get_size (caps); i++) {
    structure = gst_caps_get_structure (caps, i);

    for (j = 0; j < gst_caps_get_size (peer_caps); j++) {
      structure_peer = gst_caps_get_structure (peer_caps, j);

      /* other/tensor */
      if (gst_structure_has_field (structure, "dimension")
          && gst_structure_has_field (structure_peer, "dimension")) {
        /* update dimensions for negotiation */
        if (_is_structure_dimension_same (structure, structure_peer,
                "dimension")) {
          gst_structure_set (structure, "dimension", G_TYPE_STRING,
              gst_structure_get_string (structure_peer, "dimension"), NULL);
        }
      }
      /* other/tensors */
      else if (gst_structure_has_field (structure, "dimensions")
          && gst_structure_has_field (structure_peer, "dimensions")) {
        /* update dimensions for negotiation */
        if (_is_structure_dimension_same (structure, structure_peer,
                "dimensions")) {
          gst_structure_set (structure, "dimensions", G_TYPE_STRING,
              gst_structure_get_string (structure_peer, "dimensions"), NULL);
        }
      }
    }
  }
}

/**
 * @brief  Try intersecting @caps1 and @caps2 for tensor stream
 * @param caps1 a GstCaps to intersect
 * @param caps2 a GstCaps to intersect
 * @return TRUE if intersection would be not empty.
 */
gboolean
gst_tensor_caps_can_intersect (GstCaps * caps1, GstCaps * caps2)
{
  GstStructure *structure1;
  GstStructure *structure2;
  GstStructure *structure_copy1;
  GstStructure *structure_copy2;

  const gchar *name1;
  const gchar *name2;

  gboolean intersectable;

  if (gst_caps_can_intersect (caps1, caps2))
    return TRUE;

  structure1 = gst_caps_get_structure (caps1, 0);
  structure2 = gst_caps_get_structure (caps2, 0);

  if (!gst_structure_is_tensor_stream (structure1)
      || !gst_structure_is_tensor_stream (structure2))
    return FALSE;

  name1 = gst_structure_get_name (structure1);
  name2 = gst_structure_get_name (structure2);

  if (!g_str_equal (name1, name2))
    return FALSE;

  /* other/tensor */
  if (g_str_equal (name1, NNS_MIMETYPE_TENSOR)) {
    if (gst_structure_has_field (structure1, "dimension")
        && gst_structure_has_field (structure2, "dimension")) {
      if (!_is_structure_dimension_same (structure1, structure2, "dimension"))
        return FALSE;
    }
  }
  /* other/tensors */
  else if (gst_structure_has_field (structure1, "dimensions")
      && gst_structure_has_field (structure2, "dimensions")) {
    if (!_is_structure_dimension_same (structure1, structure2, "dimensions"))
      return FALSE;
  }

  structure_copy1 = gst_structure_copy (structure1);
  structure_copy2 = gst_structure_copy (structure2);

  gst_structure_remove_field (structure_copy1, "dimension");
  gst_structure_remove_field (structure_copy1, "dimensions");
  gst_structure_remove_field (structure_copy2, "dimension");
  gst_structure_remove_field (structure_copy2, "dimensions");

  intersectable =
      gst_structure_can_intersect (structure_copy1, structure_copy2);

  gst_structure_free (structure_copy1);
  gst_structure_free (structure_copy2);

  return intersectable;
}

/**
 * @brief Get pad caps from tensors config and caps of the peer connected to the pad.
 * @param pad GstPad to get possible caps
 * @param config tensors config structure
 * @return caps for given config. Caller is responsible for unreffing the returned caps.
 */
GstCaps *
gst_tensor_pad_caps_from_config (GstPad * pad, const GstTensorsConfig * config)
{
  GstCaps *caps = NULL;
  GstCaps *templ;
  gboolean is_flexible, peer_is_flexible, peer_has_tensor_caps;
  GstCaps *peer_caps;

  g_return_val_if_fail (GST_IS_PAD (pad), NULL);
  g_return_val_if_fail (config != NULL, NULL);

  templ = gst_pad_get_pad_template_caps (pad);

  /* check peer caps */
  peer_is_flexible = peer_has_tensor_caps = FALSE;

  peer_caps = gst_pad_peer_query_caps (pad, NULL);
  if (peer_caps && gst_caps_get_size (peer_caps) > 0) {
    GstCaps *tmp;
    GstStructure *st;
    GstTensorsConfig peer_config;

    tmp = gst_caps_from_string (GST_TENSOR_CAP_DEFAULT);
    peer_has_tensor_caps = gst_caps_can_intersect (tmp, peer_caps);
    gst_caps_unref (tmp);

    st = gst_caps_get_structure (peer_caps, 0);
    if (gst_tensors_config_from_structure (&peer_config, st))
      peer_is_flexible = gst_tensors_config_is_flexible (&peer_config);
    gst_tensors_config_free (&peer_config);
  }

  /* other/tensors (flexible) */
  is_flexible = gst_tensors_config_is_flexible (config);

  if (is_flexible || peer_is_flexible) {
    caps = _get_flexible_caps (config);
    goto intersectable;
  }

  /* other/tensor */
  if (config->info.num_tensors == 1 && peer_has_tensor_caps) {
    caps = _get_tensor_caps (config);
    if (peer_caps)
      gst_tensor_caps_update_dimension (caps, peer_caps);

    if (gst_caps_can_intersect (caps, templ))
      goto done;

    gst_caps_unref (caps);
  }

  /* other/tensors (static) */
  caps = _get_tensors_caps (config);
  if (peer_caps)
    gst_tensor_caps_update_dimension (caps, peer_caps);

intersectable:
  if (!gst_caps_can_intersect (caps, templ)) {
    gst_caps_unref (caps);
    caps = NULL;
  }

done:
  gst_caps_unref (templ);
  if (peer_caps)
    gst_caps_unref (peer_caps);
  caps = gst_caps_truncate (caps);
  return caps;
}

/**
 * @brief Get all possible caps from tensors config. Unlike gst_tensor_pad_caps_from_config(), this function does not check peer caps.
 * @param pad GstPad to get possible caps
 * @param config tensors config structure
 * @return caps for given config. Caller is responsible for unreffing the returned caps.
 */
GstCaps *
gst_tensor_pad_possible_caps_from_config (GstPad * pad,
    const GstTensorsConfig * config)
{
  GstCaps *caps, *tmp;
  GstCaps *templ;

  g_return_val_if_fail (GST_IS_PAD (pad), NULL);
  g_return_val_if_fail (config != NULL, NULL);

  caps = gst_caps_new_empty ();
  templ = gst_pad_get_pad_template_caps (pad);

  /* append caps for static tensor */
  if (gst_tensors_config_is_static (config)) {
    /* other/tensor */
    if ((tmp = _get_tensor_caps (config)) != NULL) {
      if (gst_caps_can_intersect (tmp, templ))
        gst_caps_append (caps, tmp);
      else
        gst_caps_unref (tmp);
    }

    /* other/tensors */
    if ((tmp = _get_tensors_caps (config)) != NULL) {
      if (gst_caps_can_intersect (tmp, templ))
        gst_caps_append (caps, tmp);
      else
        gst_caps_unref (tmp);
    }
  }

  /* caps for flexible tensor */
  if ((tmp = _get_flexible_caps (config)) != NULL) {
    if (gst_caps_can_intersect (tmp, templ))
      gst_caps_append (caps, tmp);
    else
      gst_caps_unref (tmp);
  }

  /* if no possible caps for given config, return null. */
  if (gst_caps_is_empty (caps)) {
    gst_caps_unref (caps);
    caps = NULL;
  }

  gst_caps_unref (templ);
  return caps;
}

/**
  * @brief Get tensor format of current pad caps.
  * @param pad GstPad to check current caps.
  * @return The tensor_format of current pad caps.
  *
  * If pad does not have tensor caps return _NNS_TENSOR_FORMAT_END
  */
tensor_format
gst_tensor_pad_get_format (GstPad * pad)
{
  GstCaps *caps;
  tensor_format ret = _NNS_TENSOR_FORMAT_END;

  g_return_val_if_fail (GST_IS_PAD (pad), _NNS_TENSOR_FORMAT_END);

  caps = gst_pad_get_current_caps (pad);
  if (caps) {
    GstStructure *structure;
    GstTensorsConfig config;

    structure = gst_caps_get_structure (caps, 0);
    if (gst_tensors_config_from_structure (&config, structure)) {
      ret = config.info.format;
    }
    gst_caps_unref (caps);
    gst_tensors_config_free (&config);
  }

  return ret;
}

/**
 * @brief Get caps from tensors config (for other/tensors)
 * @param config tensors config info
 * @return caps for given config
 */
GstCaps *
gst_tensors_caps_from_config (const GstTensorsConfig * config)
{
  GstCaps *caps;

  g_return_val_if_fail (config != NULL, NULL);

  if (gst_tensors_config_is_flexible (config)) {
    caps = _get_flexible_caps (config);
  } else {
    caps = _get_tensors_caps (config);
  }

  caps = gst_caps_truncate (caps);

  return caps;
}

/**
 * @brief Get tensor caps from tensors config
 * @param config tensors config info
 * @return caps for given config
 */
GstCaps *
gst_tensor_caps_from_config (const GstTensorsConfig * config)
{
  GstCaps *caps;
  g_return_val_if_fail (config != NULL, NULL);

  caps = _get_tensor_caps (config);
  caps = gst_caps_truncate (caps);

  return caps;
}

/**
 * @brief Parse structure and set tensors config (for other/tensors)
 * @param config tensors config structure to be filled
 * @param structure structure to be interpreted
 * @return TRUE if no error
 */
gboolean
gst_tensors_config_from_structure (GstTensorsConfig * config,
    const GstStructure * structure)
{
  const gchar *name;
  tensor_format format = _NNS_TENSOR_FORMAT_STATIC;

  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensors_config_init (config);

  g_return_val_if_fail (structure != NULL, FALSE);

  name = gst_structure_get_name (structure);

  if (g_str_equal (name, NNS_MIMETYPE_TENSOR)) {
    /* other/tensor is always static */
    config->info.num_tensors = 1;

    if (gst_structure_has_field (structure, "dimension")) {
      const gchar *dim_str = gst_structure_get_string (structure, "dimension");
      gst_tensor_parse_dimension (dim_str, config->info.info[0].dimension);
    }

    if (gst_structure_has_field (structure, "type")) {
      const gchar *type_str = gst_structure_get_string (structure, "type");
      config->info.info[0].type = gst_tensor_get_type (type_str);
    }
  } else if (g_str_equal (name, NNS_MIMETYPE_TENSORS)) {
    if (gst_structure_has_field (structure, "format")) {
      const gchar *format_str;

      format_str = gst_structure_get_string (structure, "format");
      format = gst_tensor_get_format (format_str);

      if (format == _NNS_TENSOR_FORMAT_END) {
        GST_INFO
            ("Invalid format %s, it should be one of %s. Suppose tensor format is static.",
            _STR_NULL (format_str), GST_TENSOR_FORMAT_ALL);
      } else {
        config->info.format = format;
      }
    }

    if (config->info.format == _NNS_TENSOR_FORMAT_STATIC) {
      gst_structure_get_int (structure, "num_tensors",
          (gint *) (&config->info.num_tensors));

      if (config->info.num_tensors > NNS_TENSOR_SIZE_LIMIT) {
        /* create extra info */
        gst_tensors_info_extra_create (&config->info);
      }

      /* parse dimensions */
      if (gst_structure_has_field (structure, "dimensions")) {
        const gchar *dims_str;
        guint num_dims;

        dims_str = gst_structure_get_string (structure, "dimensions");
        num_dims =
            gst_tensors_info_parse_dimensions_string (&config->info, dims_str);

        if (config->info.num_tensors != num_dims) {
          nns_logw ("Invalid param, dimensions (%d) tensors (%d)\n",
              num_dims, config->info.num_tensors);
        }
      }

      /* parse types */
      if (gst_structure_has_field (structure, "types")) {
        const gchar *types_str;
        guint num_types;

        types_str = gst_structure_get_string (structure, "types");
        num_types =
            gst_tensors_info_parse_types_string (&config->info, types_str);

        if (config->info.num_tensors != num_types) {
          nns_logw ("Invalid param, types (%d) tensors (%d)\n",
              num_types, config->info.num_tensors);
        }
      }
    }
  } else {
    nns_logw ("Unsupported type = %s\n", name ? name : "Unknown");
    return FALSE;
  }

  if (gst_structure_has_field (structure, "framerate")) {
    gst_structure_get_fraction (structure, "framerate", &config->rate_n,
        &config->rate_d);
  }

  return TRUE;
}

/**
 * @brief Parse memory and fill the tensor meta.
 * @param[out] meta tensor meta structure to be filled
 * @param[in] mem pointer to GstMemory to be parsed
 * @return TRUE if successfully set the meta
 */
gboolean
gst_tensor_meta_info_parse_memory (GstTensorMetaInfo * meta, GstMemory * mem)
{
  GstMapInfo map;
  gsize hsize, msize;
  gboolean ret;

  g_return_val_if_fail (mem != NULL, FALSE);
  g_return_val_if_fail (meta != NULL, FALSE);

  gst_tensor_meta_info_init (meta);

  /* Check header size of tensor-meta. */
  hsize = gst_tensor_meta_info_get_header_size (meta);
  msize = gst_memory_get_sizes (mem, NULL, NULL);
  if (msize < hsize)
    return FALSE;

  if (!gst_memory_map (mem, &map, GST_MAP_READ)) {
    nns_loge ("Failed to get the meta, cannot map the memory.");
    return FALSE;
  }

  ret = gst_tensor_meta_info_parse_header (meta, map.data);

  gst_memory_unmap (mem, &map);
  return ret;
}

/**
 * @brief Append header to memory.
 * @param[in] meta tensor meta structure
 * @param[in] mem pointer to GstMemory
 * @return Newly allocated GstMemory (Caller should free returned memory using gst_memory_unref())
 */
GstMemory *
gst_tensor_meta_info_append_header (GstTensorMetaInfo * meta, GstMemory * mem)
{
  GstMemory *new_mem = NULL;
  gsize msize, hsize;
  GstMapInfo old_map, new_map;

  g_return_val_if_fail (mem != NULL, NULL);
  g_return_val_if_fail (gst_tensor_meta_info_validate (meta), NULL);

  if (!gst_memory_map (mem, &old_map, GST_MAP_READ)) {
    nns_loge ("Failed to append header, cannot map the old memory.");
    return NULL;
  }

  /* memory size (header + old memory) */
  hsize = gst_tensor_meta_info_get_header_size (meta);
  msize = hsize + old_map.size;

  new_mem = gst_allocator_alloc (NULL, msize, NULL);
  if (!gst_memory_map (new_mem, &new_map, GST_MAP_WRITE)) {
    nns_loge ("Failed to append header, cannot map the new memory.");
    gst_memory_unmap (mem, &old_map);
    gst_memory_unref (new_mem);
    return NULL;
  }

  /* set header and copy old data */
  gst_tensor_meta_info_update_header (meta, new_map.data);
  memcpy (new_map.data + hsize, old_map.data, old_map.size);

  gst_memory_unmap (mem, &old_map);
  gst_memory_unmap (new_mem, &new_map);
  return new_mem;
}

/**
 * @brief Get the nth GstMemory from given @a buffer.
 * @param[in] buffer GstBuffer to be parsed.
 * @param[in] index Index of GstMemory to be returned.
 * @return GstMemory if found, otherwise NULL (Caller should free returned memory using gst_memory_unref()).
 */
GstMemory *
gst_tensor_buffer_get_nth_memory (GstBuffer * buffer, const guint index)
{
  guint i, num_tensors, offset = 0;
  GstMemory *extra_tensors_memory, *res_mem;
  GstMapInfo extra_tensors_map;
  GstTensorExtraInfo *extra_info;

  if (!GST_IS_BUFFER (buffer)) {
    nns_loge ("Failed to parse GstBuffer (invalid input buffer).");
    return NULL;
  }

  num_tensors = gst_tensor_buffer_get_count (buffer);
  if (num_tensors == 0U) {
    nns_loge ("num_tensors is 0. Please check the buffer.");
    return NULL;
  }

  /* If num_tensors is less than or equal to NNS_TENSOR_SIZE_LIMIT, it's trivial. */
  if (num_tensors <= NNS_TENSOR_SIZE_LIMIT || index < NNS_TENSOR_SIZE_LIMIT - 1) {
    return gst_buffer_get_memory (buffer, index);
  }

  /* Check the buffer contains NNS_TENSOR_SIZE_LIMIT memory */
  if (gst_buffer_n_memory (buffer) != NNS_TENSOR_SIZE_LIMIT) {
    nns_loge ("Failed to get %d-th memory from buffer (invalid buffer size).",
        index);
    return NULL;
  }

  /* If num_tensors is greater than NNS_TENSOR_SIZE_LIMIT, we need to parse extra info. */
  extra_tensors_memory =
      gst_buffer_get_memory (buffer, NNS_TENSOR_SIZE_LIMIT - 1);
  if (!extra_tensors_memory) {
    nns_loge ("Failed to get %d-th memory", NNS_TENSOR_SIZE_LIMIT);
    return NULL;
  }

  if (!gst_memory_map (extra_tensors_memory, &extra_tensors_map, GST_MAP_READ)) {
    nns_loge ("Failed to map %d-th memory", NNS_TENSOR_SIZE_LIMIT);
    gst_memory_unref (extra_tensors_memory);
    return NULL;
  }

  /* check header (extra info) of the memory */
  if (!gst_memory_map_is_extra_tensor (&extra_tensors_map)) {
    nns_loge ("Invalid extra header");
    gst_memory_unmap (extra_tensors_memory, &extra_tensors_map);
    gst_memory_unref (extra_tensors_memory);
    return NULL;
  }

  extra_info = (GstTensorExtraInfo *) extra_tensors_map.data;

  /* check index */
  if (index >= extra_info->num_extra_tensors + NNS_TENSOR_SIZE_LIMIT) {
    nns_loge ("Invalid index");
    gst_memory_unmap (extra_tensors_memory, &extra_tensors_map);
    gst_memory_unref (extra_tensors_memory);
    return NULL;
  }

  /* parse the memory */
  offset = sizeof (GstTensorExtraInfo);

  /* If index is NNS_TENSOR_SIZE_LIMIT - 1 */
  if (index == NNS_TENSOR_SIZE_LIMIT - 1) {
    res_mem =
        gst_memory_share (extra_tensors_memory, offset, extra_info->reserved);
    gst_memory_unmap (extra_tensors_memory, &extra_tensors_map);
    gst_memory_unref (extra_tensors_memory);

    return res_mem;
  }

  offset += extra_info->reserved;

  for (i = 1; i <= index - NNS_TENSOR_SIZE_LIMIT; ++i) {
    offset += gst_tensor_info_get_size (&extra_info->infos[i - 1]);
  }

  /* wrap it as GstMemory */
  res_mem =
      gst_memory_share (extra_tensors_memory, offset,
      gst_tensor_info_get_size (&extra_info->infos[index -
              NNS_TENSOR_SIZE_LIMIT]));

  /* cleanup and return */
  gst_memory_unmap (extra_tensors_memory, &extra_tensors_map);
  gst_memory_unref (extra_tensors_memory);

  return res_mem;
}

/**
 * @brief Append @a memory to given @a buffer.
 * @param[in/out] buffer GstBuffer to be appended.
 * @param[in] memory GstMemory to append. This function will take ownership of this.
 * @param[in] info GstTensorInfo of given @a memory.
 * @return TRUE if successfully appended, otherwise FALSE.
 */
gboolean
gst_tensor_buffer_append_memory (GstBuffer * buffer, GstMemory * memory,
    const GstTensorInfo * info)
{
  guint num_mems, offset, new_mem_index;
  GstMemory *new_memory, *last_memory;
  gsize new_mem_size, last_mem_size;
  GstMapInfo new_memory_map, last_memory_map, incoming_memory_map;
  GstTensorExtraInfo *extra_info;
  GstTensorMetaInfo meta;
  gboolean is_extra, is_static;

  if (!GST_IS_BUFFER (buffer)) {
    nns_loge ("Failed to append memory, given buffer is invalid.");
    return FALSE;
  }

  if (!memory) {
    nns_loge ("Failed to append memory, given memory is NULL.");
    return FALSE;
  }

  if (gst_tensor_meta_info_parse_memory (&meta, memory)) {
    is_static = (meta.format == _NNS_TENSOR_FORMAT_STATIC);
  } else {
    /* Suppose given memory is static tensor. */
    is_static = TRUE;

    /* Error case if given tensor-info is invalid. */
    if (!gst_tensor_info_validate (info)) {
      nns_loge ("Failed to get tensor info (invalid input info).");
      return FALSE;
    }
  }

  num_mems = gst_buffer_n_memory (buffer);

  /* trivial call to gst_buffer_append_memory */
  if (num_mems < NNS_TENSOR_SIZE_LIMIT) {
    gst_buffer_append_memory (buffer, memory);
    return TRUE;
  }

  /* given buffer has NNS_TENSOR_SIZE_LIMIT memory blocks */
  last_memory = gst_buffer_peek_memory (buffer, num_mems - 1);
  if (!last_memory) {
    nns_loge ("Failed to get last memory");
    return FALSE;
  }

  if (!gst_memory_map (last_memory, &last_memory_map, GST_MAP_READ)) {
    nns_loge ("Failed to map last memory");
    return FALSE;
  }

  new_mem_size = last_mem_size = gst_memory_get_sizes (last_memory, NULL, NULL);

  /* if the memory does not have proper header, append it */
  is_extra = gst_memory_map_is_extra_tensor (&last_memory_map);
  if (!is_extra) {
    new_mem_size += sizeof (GstTensorExtraInfo);
  }

  new_mem_size += gst_memory_get_sizes (memory, NULL, NULL);

  new_memory = gst_allocator_alloc (NULL, new_mem_size, NULL);
  if (!new_memory) {
    nns_loge ("Failed to allocate memory for extra tensors.");
    gst_memory_unmap (last_memory, &last_memory_map);
    return FALSE;
  }

  if (!gst_memory_map (new_memory, &new_memory_map, GST_MAP_WRITE)) {
    nns_loge ("Failed to map extra memory");
    gst_memory_unmap (last_memory, &last_memory_map);
    gst_memory_unref (new_memory);
    return FALSE;
  }

  if (!gst_memory_map (memory, &incoming_memory_map, GST_MAP_READ)) {
    nns_loge ("Failed to map incoming memory");
    gst_memory_unmap (new_memory, &new_memory_map);
    gst_memory_unmap (last_memory, &last_memory_map);
    gst_memory_unref (new_memory);
    return FALSE;
  }

  extra_info = (GstTensorExtraInfo *) new_memory_map.data;

  /* if the last_memory does not have proper header, append it */
  if (!is_extra) {
    gst_tensor_extra_info_init (extra_info, last_mem_size);
    offset = sizeof (GstTensorExtraInfo);
  } else {
    offset = 0;
  }

  /* copy last_memory into new_memory */
  memcpy (new_memory_map.data + offset, last_memory_map.data,
      last_memory_map.size);

  /* copy incoming_memory into new_memory */
  new_mem_index = extra_info->num_extra_tensors;
  extra_info->num_extra_tensors += 1;

  /* Copy tensor info into extra. */
  if (is_static) {
    gst_tensor_info_copy (&extra_info->infos[new_mem_index], info);
  } else {
    gst_tensor_meta_info_convert (&meta, &extra_info->infos[new_mem_index]);
  }

  memcpy (new_memory_map.data + offset + last_memory_map.size,
      incoming_memory_map.data, incoming_memory_map.size);

  gst_memory_unmap (new_memory, &new_memory_map);
  gst_memory_unmap (memory, &incoming_memory_map);
  gst_memory_unmap (last_memory, &last_memory_map);

  gst_memory_unref (memory);
  gst_buffer_replace_memory (buffer, num_mems - 1, new_memory);

  return TRUE;
}

/**
 * @brief Get the number of tensors in the buffer.
 */
guint
gst_tensor_buffer_get_count (GstBuffer * buffer)
{
  guint num_mems;
  GstMemory *mem;
  GstMapInfo map;
  GstTensorExtraInfo *extra_info;

  g_return_val_if_fail (buffer != NULL, 0);

  num_mems = gst_buffer_n_memory (buffer);
  if (num_mems < NNS_TENSOR_SIZE_LIMIT) {
    return num_mems;
  }

  /* num_mems == NNS_TENSOR_SIZE_LIMIT */
  mem = gst_buffer_peek_memory (buffer, num_mems - 1);
  if (!mem) {
    nns_loge ("Failed to get the last memory.");
    return 0;
  }

  if (!gst_memory_map (mem, &map, GST_MAP_READ)) {
    nns_loge ("Failed to map the last memory.");
    return 0;
  }

  if (gst_memory_map_is_extra_tensor (&map)) {
    extra_info = (GstTensorExtraInfo *) map.data;
    num_mems = extra_info->num_extra_tensors + NNS_TENSOR_SIZE_LIMIT;
  } else {
    nns_logi ("The last memory does not have extra tensors header. "
        "Assuming the number of tensors is %d.", num_mems);
  }

  gst_memory_unmap (mem, &map);

  return num_mems;
}

/**
 * @brief Sets the value of a property based on the specified property value and GParamSpec.
 *
 * @param prop_value A pointer to the GValue where the property value will be set.
 * @param param_spec A pointer to the GParamSpec that describes the property.
 * @param property_value A string representing the value to be set for the property.
 *
 * @note This API is intended to be used by gst_tensor_parse_config_file ()
 */

static void
set_property_value (GValue * prop_value, const GParamSpec * param_spec,
    const gchar * property_value)
{
  GType value_type = G_PARAM_SPEC_VALUE_TYPE (param_spec);
  g_value_init (prop_value, value_type);

  if (value_type == G_TYPE_BOOLEAN) {
    gboolean value = g_ascii_strcasecmp (property_value, "true") == 0;
    g_value_set_boolean (prop_value, value);
  } else if (value_type == G_TYPE_INT) {
    gint value = atoi (property_value);
    g_value_set_int (prop_value, value);
  } else if (value_type == G_TYPE_UINT) {
    guint value = atoi (property_value);
    g_value_set_uint (prop_value, value);
  } else if (value_type == G_TYPE_FLOAT) {
    gfloat value = atof (property_value);
    g_value_set_float (prop_value, value);
  } else if (value_type == G_TYPE_DOUBLE) {
    gdouble value = atof (property_value);
    g_value_set_double (prop_value, value);
  } else {
    g_value_set_string (prop_value, property_value); /** default is string */
  }
}

/**
 * @brief Parses a configuration file and sets the corresponding properties on a GObject.
 *
 * This function reads the contents of the configuration file located at the given path
 * and sets the properties of the specified GObject based on the configuration data.
 *
 * @param config_path The path to the configuration file.
 * @param object      The GObject on which to set the properties.
 *
 * @note The responsibility of managing the memory of the GObject passed as a parameter
 *       lies outside this function.
 */

void
gst_tensor_parse_config_file (const gchar * config_path, const GObject * object)
{
  GError *error = NULL;
  gchar *config_data = NULL;
  gchar **lines = NULL;
  gchar **line = NULL;
  GObjectClass *g_object_class = G_OBJECT_GET_CLASS (object);

  if (!g_file_get_contents (config_path, &config_data, NULL, &error)) {
    GST_DEBUG ("Failed to read config file: %s\n", error->message);
    g_error_free (error);
    return;
  }

  lines = g_strsplit (config_data, "\n", -1);
  g_free (config_data);

  /** Iterate over each line */
  for (line = lines; *line; ++line) {
    gchar **parts = g_strsplit (*line, "=", 2);
    if (g_strv_length (parts) == 2) {
      gchar *property_name = g_strstrip (g_strdup (parts[0]));
      gchar *property_value = g_strstrip (g_strdup (parts[1]));

      GParamSpec *pdata =
          g_object_class_find_property (g_object_class, property_name);

      if (pdata != NULL) {
        GValue prop_value = G_VALUE_INIT;
        set_property_value (&prop_value, pdata, property_value);
        g_object_set_property (G_OBJECT (object), pdata->name, &prop_value);
        g_value_unset (&prop_value);
      }

      g_free (property_name);
      g_free (property_value);
    }

    g_strfreev (parts);
  }

  g_strfreev (lines);
}
