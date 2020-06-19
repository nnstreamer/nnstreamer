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

#include <tensor_common.h>
#include <string.h>

static const gchar *gst_tensor_time_sync_mode_string[] = {
  [SYNC_NOSYNC] = "nosync",
  [SYNC_SLOWEST] = "slowest",
  [SYNC_BASEPAD] = "basepad",
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
 * @param[in/out] filter "this" pointer. sync_mode & option MUST BE set already.
 * @return True if successfully set the option.
 */
gboolean
gst_tensor_time_sync_set_option_data (tensor_time_sync_data * sync)
{
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

  /** @todo update below with each sync mode */
  if (empty > 0 || empty == total)
    is_eos = TRUE;

  return is_eos;
}

/**
 * @brief A function call to decide current timestamp among collected pads based on PTS.
 * It will decide current timestamp according to sync option.
 */
gboolean
gst_tensor_time_sync_get_current_time (GstCollectPads * collect,
    tensor_time_sync_data * sync, GstClockTime * current_time)
{
  GSList *walk = NULL;
  guint count, empty_pad;

  walk = collect->data;
  count = empty_pad = 0;

  while (walk) {
    GstCollectData *data;
    GstBuffer *buf;

    data = (GstCollectData *) walk->data;
    buf = gst_collect_pads_peek (collect, data);
    walk = g_slist_next (walk);

    if (buf) {
      switch (sync->mode) {
        case SYNC_NOSYNC:
          /* fall-through */
        case SYNC_SLOWEST:
          if (*current_time < GST_BUFFER_PTS (buf))
            *current_time = GST_BUFFER_PTS (buf);
          break;
        case SYNC_BASEPAD:
          if (count == sync->data_basepad.sink_id)
            *current_time = GST_BUFFER_PTS (buf);
          break;
        default:
          break;
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
            (ABS (GST_CLOCK_DIFF (current, GST_BUFFER_PTS (*buf))) > base))) {
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
            MIN (sync->data_basepad.duration,
            ABS (GST_CLOCK_DIFF (GST_BUFFER_PTS (buf),
                    GST_BUFFER_PTS (pad->buffer))) - 1);
      gst_buffer_unref (buf);
    }
  }

  walk = collect->data;

  while (walk) {
    gboolean configured = FALSE;

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
        if (FALSE == _gst_tensor_time_sync_buffer_update (&buf, collect, data,
                current_time, base_time, sync))
          return FALSE;
        break;
      case SYNC_NOSYNC:
        buf = gst_collect_pads_pop (collect, data);
        break;
      default:
        break;
    }

    if (GST_IS_BUFFER (buf)) {
      guint i, n_mem;

      n_mem = gst_buffer_n_memory (buf);

      /** These are internal logic error. If given inputs are incorrect,
          the negotiation should have been failed before this stage. */
      g_assert (n_mem == in_configs.info.num_tensors);
      g_assert ((counting + n_mem) < NNS_TENSOR_SIZE_LIMIT);

      for (i = 0; i < n_mem; ++i) {
        mem = gst_buffer_get_memory (buf, i);
        gst_buffer_append_memory (tensors_buf, mem);
        configs->info.info[counting] = in_configs.info.info[i];
        counting++;
      }

      gst_buffer_unref (buf);
    } else {
      empty_pad++;
    }
  }

  configs->info.num_tensors = counting;
  configs->rate_d = old_denominator;
  configs->rate_n = old_numerator;

  GST_BUFFER_PTS (tensors_buf) = current_time;

  /* check eos */
  *is_eos = _gst_tensor_time_sync_is_eos (collect, sync, empty_pad);
  return !(*is_eos);
}
