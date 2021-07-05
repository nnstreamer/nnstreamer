/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file	tensor_crop.c
 * @date	10 May 2021
 * @brief	GStreamer element to crop the regions of incoming tensor
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

/**
 * SECTION:element-tensor_crop
 *
 * tensor_crop is a GStreamer element to crop the regions of incoming tensor.
 *
 * tensor_crop has two always sink pads - raw and info.
 * The raw pad accepts tensor (other/tensor) which will be cropped with crop info.
 * The info pad has capability for flexible tensor stream (other/tensors-flexible), that can have a various buffer size for crop info.
 * Incoming buffer on info pad should be an array of crop info.
 * Note that NNStreamer supports maximum 16 (NNS_TENSOR_SIZE_LIMIT) memory blocks in a buffer.
 * So, when incoming buffer on info pad has more than 16 crop-info array, tensor_crop will ignore the data and output buffer will have 16 memory blocks.
 *
 * The output is always in the format of other/tensors-flexible.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 tensor_crop name=crop ! (cropped tensors) ... \
 *     videotestsrc ! videoconvert ! video/x-raw,format=RGB ! tensor_converter ! tee name=t \
 *       t. ! queue ! crop.raw \
 *       t. ! queue ! (process raw video tensor and push buffer which includes crop info) ! crop.info
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include "tensor_crop.h"
#include "tensor_data.h"

/**
 * @brief Internal data structure to describe tensor region.
 */
typedef struct
{
  guint x;
  guint y;
  guint w;
  guint h;
} tensor_region_s;

/**
 * @brief Internal data structure to describe cropping tensor data.
 * @todo Add various mode to crop tensor. Now tensor-crop handles NHWC data format only.
 */
typedef struct
{
  guint num;
  tensor_region_s region[NNS_TENSOR_SIZE_LIMIT];
} tensor_crop_info_s;

GST_DEBUG_CATEGORY_STATIC (gst_tensor_crop_debug);
#define GST_CAT_DEFAULT gst_tensor_crop_debug

/**
 * @brief tensor_crop properties
 */
enum
{
  PROP_0,
  PROP_LATENESS,
  PROP_SILENT
};

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE

/**
 * @brief Default value to compare timestamp of raw and info buffer, in milliseconds (-1 means no synchronization).
 */
#define DEFAULT_LATENESS (-1)

/**
 * @brief Template for sink pad (raw data).
 */
static GstStaticPadTemplate raw_template = GST_STATIC_PAD_TEMPLATE ("raw",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT ";"
        GST_TENSORS_CAP_MAKE ("{ static, flexible }")));

/**
 * @brief Template for sink pad (crop info).
 */
static GstStaticPadTemplate info_template = GST_STATIC_PAD_TEMPLATE ("info",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_FLEX_CAP_DEFAULT));

/**
 * @brief Template for src pad.
 */
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_FLEX_CAP_DEFAULT));

#define gst_tensor_crop_parent_class parent_class
G_DEFINE_TYPE (GstTensorCrop, gst_tensor_crop, GST_TYPE_ELEMENT);

static void gst_tensor_crop_finalize (GObject * object);
static void gst_tensor_crop_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_crop_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static GstStateChangeReturn gst_tensor_crop_change_state (GstElement * element,
    GstStateChange transition);
static gboolean gst_tensor_crop_src_event (GstPad * pad, GstObject * parent,
    GstEvent * event);
static gboolean gst_tensor_crop_sink_event (GstCollectPads * pads,
    GstCollectData * data, GstEvent * event, gpointer user_data);
static GstFlowReturn gst_tensor_crop_collected (GstCollectPads * pads,
    gpointer user_data);

/**
 * @brief Initialize the tensor_crop's class.
 */
static void
gst_tensor_crop_class_init (GstTensorCropClass * klass)
{
  GObjectClass *object_class;
  GstElementClass *element_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_crop_debug, "tensor_crop", 0,
      "Element to crop the regions of incoming tensor");

  object_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;

  object_class->set_property = gst_tensor_crop_set_property;
  object_class->get_property = gst_tensor_crop_get_property;
  object_class->finalize = gst_tensor_crop_finalize;

  /**
   * GstTensorCrop::lateness:
   *
   * The time difference between raw and info buffer, in milliseconds (-1 means no synchronization).
   * If raw and info buffers on the pads have different timestamp and time-diff is larger than 'lateness',
   * tensor-crop will drop old buffer and wait for next buffers.
   */
  g_object_class_install_property (object_class, PROP_LATENESS,
      g_param_spec_int ("lateness", "Lateness",
          "The time difference between raw and info buffer in milliseconds",
          -1, G_MAXINT, DEFAULT_LATENESS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorCrop::silent:
   *
   * The flag to enable/disable debugging messages.
   */
  g_object_class_install_property (object_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  element_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_crop_change_state);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&raw_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&info_template));

  gst_element_class_set_static_metadata (element_class,
      "TensorCrop",
      "Filter/Tensor",
      "Element to crop the regions of incoming tensor",
      "Samsung Electronics Co., Ltd.");
}

/**
 * @brief Clear and reset old pad data.
 */
static void
gst_tensor_crop_pad_reset (GstTensorCropPadData * cpad)
{
  gst_tensors_config_free (&cpad->config);
  gst_tensors_config_init (&cpad->config);
}

/**
 * @brief Clear and reset old data in tensor_crop.
 */
static void
gst_tensor_crop_reset (GstTensorCrop * self)
{
  GstTensorCropPadData *cpad;
  GSList *walk;

  if (self->collect) {
    walk = self->collect->data;

    while (walk) {
      cpad = (GstTensorCropPadData *) walk->data;

      gst_tensor_crop_pad_reset (cpad);
      walk = g_slist_next (walk);
    }
  }

  self->send_stream_start = TRUE;
}

/**
 * @brief Initialize tensor_crop element.
 */
static void
gst_tensor_crop_init (GstTensorCrop * self)
{
  /* setup sink pad */
  self->sinkpad_raw = gst_pad_new_from_static_template (&raw_template, "raw");
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad_raw);

  self->sinkpad_info =
      gst_pad_new_from_static_template (&info_template, "info");
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad_info);

  self->collect = gst_collect_pads_new ();
  gst_collect_pads_set_function (self->collect,
      GST_DEBUG_FUNCPTR (gst_tensor_crop_collected), self);
  gst_collect_pads_set_event_function (self->collect,
      GST_DEBUG_FUNCPTR (gst_tensor_crop_sink_event), self);

  gst_collect_pads_add_pad (self->collect, self->sinkpad_raw,
      sizeof (GstTensorCropPadData), NULL, TRUE);
  gst_collect_pads_add_pad (self->collect, self->sinkpad_info,
      sizeof (GstTensorCropPadData), NULL, TRUE);

  /* setup src pad */
  self->srcpad = gst_pad_new_from_static_template (&src_template, "src");
  gst_pad_set_event_function (self->srcpad,
      GST_DEBUG_FUNCPTR (gst_tensor_crop_src_event));
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);

  /* init properties */
  self->lateness = DEFAULT_LATENESS;
  self->silent = DEFAULT_SILENT;
  self->send_stream_start = TRUE;
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_tensor_crop_finalize (GObject * object)
{
  GstTensorCrop *self;

  self = GST_TENSOR_CROP (object);

  gst_tensor_crop_reset (self);

  if (self->collect) {
    gst_object_unref (self->collect);
    self->collect = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_crop properties.
 */
static void
gst_tensor_crop_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorCrop *self;

  self = GST_TENSOR_CROP (object);

  switch (prop_id) {
    case PROP_LATENESS:
      self->lateness = g_value_get_int (value);
      break;
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter for tensor_crop properties.
 */
static void
gst_tensor_crop_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorCrop *self;

  self = GST_TENSOR_CROP (object);

  switch (prop_id) {
    case PROP_LATENESS:
      g_value_set_int (value, self->lateness);
      break;
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Handle state transition.
 */
static GstStateChangeReturn
gst_tensor_crop_change_state (GstElement * element, GstStateChange transition)
{
  GstTensorCrop *self;
  GstStateChangeReturn ret;

  self = GST_TENSOR_CROP (element);

  switch (transition) {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      gst_collect_pads_start (self->collect);
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      gst_collect_pads_stop (self->collect);
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      gst_tensor_crop_reset (self);
      break;
    default:
      break;
  }

  return ret;
}

/**
 * @brief Handle event on src pad.
 */
static gboolean
gst_tensor_crop_src_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  g_return_val_if_fail (event != NULL, FALSE);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_SEEK:
      /* disable seeking */
      gst_event_unref (event);
      return FALSE;
    default:
      break;
  }

  return gst_pad_event_default (pad, parent, event);
}

/**
 * @brief Handle event on sink pad.
 */
static gboolean
gst_tensor_crop_sink_event (GstCollectPads * pads, GstCollectData * data,
    GstEvent * event, gpointer user_data)
{
  GstTensorCropPadData *cpad;

  g_return_val_if_fail (event != NULL, FALSE);

  cpad = (GstTensorCropPadData *) data;

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;
      GstStructure *structure;

      gst_event_parse_caps (event, &caps);
      structure = gst_caps_get_structure (caps, 0);

      gst_tensors_config_from_structure (&cpad->config, structure);

      gst_event_unref (event);
      return gst_tensors_config_validate (&cpad->config);
    }
    default:
      break;
  }

  return gst_collect_pads_event_default (pads, data, event, FALSE);
}

/**
 * @brief Set pad caps if not negotiated.
 */
static GstFlowReturn
gst_tensor_crop_negotiate (GstTensorCrop * self)
{
  if (!gst_pad_has_current_caps (self->sinkpad_raw)) {
    GST_ERROR_OBJECT (self,
        "The raw pad of tensor_crop '%s' does not have pad caps.",
        GST_ELEMENT_NAME (self));
    return GST_FLOW_NOT_NEGOTIATED;
  }

  if (!gst_pad_has_current_caps (self->sinkpad_info)) {
    GST_ERROR_OBJECT (self,
        "The info pad of tensor_crop '%s' does not have pad caps.",
        GST_ELEMENT_NAME (self));
    return GST_FLOW_NOT_NEGOTIATED;
  }

  if (!gst_pad_has_current_caps (self->srcpad)) {
    GstCaps *caps;
    GstSegment segment;
    GstTensorsConfig config;
    GstTensorCropPadData *cpad;
    GSList *walk;

    if (self->send_stream_start) {
      gchar *sid;

      sid = g_strdup_printf ("%s-%08x",
          GST_ELEMENT_NAME (self), g_random_int ());
      gst_pad_push_event (self->srcpad, gst_event_new_stream_start (sid));
      g_free (sid);

      self->send_stream_start = FALSE;
    }

    /**
     * Get config from collect-pads and set framerate.
     * Output is always flexible tensor.
     */
    gst_tensors_config_init (&config);
    config.info.info[0].format = _NNS_TENSOR_FORMAT_FLEXIBLE;

    walk = self->collect->data;
    while (walk) {
      cpad = (GstTensorCropPadData *) walk->data;

      if (config.rate_n < 0 ||
          gst_util_fraction_compare (cpad->config.rate_n, cpad->config.rate_d,
              config.rate_n, config.rate_d) < 0) {
        config.rate_n = cpad->config.rate_n;
        config.rate_d = cpad->config.rate_d;
      }

      walk = g_slist_next (walk);
    }

    caps = gst_tensors_caps_from_config (&config);
    gst_pad_set_caps (self->srcpad, caps);
    gst_caps_unref (caps);

    gst_segment_init (&segment, GST_FORMAT_TIME);
    gst_pad_push_event (self->srcpad, gst_event_new_segment (&segment));
  }

  return GST_FLOW_OK;
}

/**
 * @brief Internal function to prepare output meta info.
 */
static gboolean
gst_tensor_crop_prepare_out_meta (GstTensorCrop * self, gpointer buffer,
    GstTensorMetaInfo * meta, GstTensorInfo * info)
{
  GstCaps *caps;
  GstStructure *structure;
  GstTensorsConfig config;
  GstTensorInfo *_info;
  gboolean ret = FALSE;

  gst_tensor_meta_info_init (meta);
  gst_tensor_info_init (info);

  caps = gst_pad_get_current_caps (self->sinkpad_raw);
  structure = gst_caps_get_structure (caps, 0);

  if (!gst_tensors_config_from_structure (&config, structure)) {
    GST_ERROR_OBJECT (self, "Failed to get the config from caps.");
    goto done;
  }

  /**
   * @note tensor-crop handles single tensor. Parse first one.
   */
  _info = &config.info.info[0];

  if (gst_tensor_info_is_flexible (_info)) {
    /* meta from buffer */
    if (gst_tensor_meta_info_parse_header (meta, buffer)) {
      ret = gst_tensor_meta_info_convert (meta, info);
    }
  } else {
    /* meta from caps */
    ret = gst_tensor_info_convert_to_meta (_info, meta);
    gst_tensor_info_copy (info, _info);
  }

  /* output is flex tensor */
  meta->format = _NNS_TENSOR_FORMAT_FLEXIBLE;

done:
  gst_caps_unref (caps);
  gst_tensors_config_free (&config);
  return ret;
}

/**
 * @brief Internal function to parse buffer and fill crop info.
 */
static gboolean
gst_tensor_crop_get_crop_info (GstTensorCrop * self, GstBuffer * info,
    tensor_crop_info_s * cinfo)
{
  GstMemory *mem;
  GstMapInfo map;
  GstTensorMetaInfo meta;
  gsize hsize, dsize, esize;
  guint i, j;
  guint8 *pos, *src, *desc;
  gboolean ret = FALSE;

  i = gst_buffer_n_memory (info);
  g_assert (i > 0);
  if (i > 1) {
    GST_WARNING_OBJECT (self,
        "Info buffer has %u memories, parse first one.", i);
  }

  mem = gst_buffer_peek_memory (info, 0);
  if (!gst_memory_map (mem, &map, GST_MAP_READ)) {
    GST_ERROR_OBJECT (self, "Failed to map the info buffer.");
    return FALSE;
  }

  /* parse crop-info from flex tensor */
  if (!gst_tensor_meta_info_parse_header (&meta, map.data)) {
    GST_ERROR_OBJECT (self, "Failed to get the meta from info buffer.");
    goto done;
  }

  hsize = gst_tensor_meta_info_get_header_size (&meta);
  dsize = gst_tensor_meta_info_get_data_size (&meta);
  esize = gst_tensor_get_element_size (meta.type);

  if (hsize + dsize != map.size) {
    GST_ERROR_OBJECT (self,
        "Invalid meta info, info buffer size is incorrect (received %zd, expected %zd).",
        map.size, hsize + dsize);
    goto done;
  }

  /**
   * @todo Add various mode to crop tensor.
   * Now tensor-crop handles NHWC data format only.
   */
  g_assert ((dsize % (esize * 4)) == 0);

  memset (cinfo, 0, sizeof (tensor_crop_info_s));

  cinfo->num = dsize / (esize * 4);
  cinfo->num = MIN (cinfo->num, NNS_TENSOR_SIZE_LIMIT);

  for (i = 0; i < cinfo->num; i++) {
    pos = map.data + hsize + (esize * 4 * i);

    for (j = 0; j < 4; j++) {
      src = pos + (esize * j);
      desc = (guint8 *) (&cinfo->region[i]) + sizeof (guint) * j;

      gst_tensor_data_raw_typecast (src, meta.type, desc, _NNS_UINT32);
    }
  }

  ret = TRUE;

done:
  gst_memory_unmap (mem, &map);
  return ret;
}

/**
 * @brief Internal function to crop incoming buffer.
 */
static GstBuffer *
gst_tensor_crop_do_cropping (GstTensorCrop * self, GstBuffer * raw,
    tensor_crop_info_s * cinfo)
{
  GstBuffer *result = NULL;
  GstMemory *mem;
  GstMapInfo map;
  GstTensorMetaInfo meta;
  GstTensorInfo info;
  gboolean flexible;
  gsize hsize, esize, dsize;
  guint8 *cropped, *dpos, *desc, *src;
  guint i, j, ch, mw, mh, _x, _y, _w, _h;

  i = gst_buffer_n_memory (raw);
  g_assert (i > 0);
  if (i > 1) {
    GST_WARNING_OBJECT (self,
        "Raw data buffer has %u memories, parse first one.", i);
  }

  mem = gst_buffer_peek_memory (raw, 0);
  if (!gst_memory_map (mem, &map, GST_MAP_READ)) {
    GST_ERROR_OBJECT (self, "Failed to map the raw buffer.");
    return NULL;
  }

  if (!gst_tensor_crop_prepare_out_meta (self, map.data, &meta, &info)) {
    GST_ERROR_OBJECT (self, "Failed to get the output meta.");
    goto done;
  }

  flexible = (info.format == _NNS_TENSOR_FORMAT_FLEXIBLE);
  hsize = flexible ? gst_tensor_meta_info_get_header_size (&meta) : 0;
  dsize = gst_tensor_meta_info_get_data_size (&meta);
  dpos = map.data + hsize;
  if ((hsize + dsize) != map.size) {
    GST_ERROR_OBJECT (self,
        "Raw buffer has invalid data size (received %zd, expected %zd).",
        map.size, dsize);
    goto done;
  }

  result = gst_buffer_new ();

  /** @todo Add various mode to crop tensor. */
  ch = info.dimension[0];
  mw = info.dimension[1];
  mh = info.dimension[2];
  esize = gst_tensor_get_element_size (info.type);
  hsize = gst_tensor_meta_info_get_header_size (&meta);

  for (i = 0; i < cinfo->num; i++) {
    _x = (cinfo->region[i].x < mw) ? cinfo->region[i].x : mw;
    _y = (cinfo->region[i].y < mh) ? cinfo->region[i].y : mh;
    _w = (_x + cinfo->region[i].w - 1 < mw) ? cinfo->region[i].w : (mw - _x);
    _h = (_y + cinfo->region[i].h - 1 < mh) ? cinfo->region[i].h : (mh - _y);

    g_assert (_w > 0 && _h > 0);
    dsize = hsize + (esize * ch * _w * _h);
    cropped = (guint8 *) g_malloc0 (dsize);

    /* set header for flex tensor */
    meta.dimension[1] = _w;
    meta.dimension[2] = _h;
    meta.dimension[3] = 1;
    gst_tensor_meta_info_update_header (&meta, cropped);

    for (j = 0; j < _h; j++) {
      src = dpos + esize * ch * (_x + (j + _y) * mw);
      desc = cropped + hsize + (esize * ch * _w) * j;
      memcpy (desc, src, (esize * ch * _w));
    }

    gst_buffer_append_memory (result,
        gst_memory_new_wrapped (0, cropped, dsize, 0, dsize, cropped, g_free));
  }

  /* set timestamp from raw buffer */
  gst_buffer_copy_into (result, raw, GST_BUFFER_COPY_METADATA, 0, -1);

done:
  gst_memory_unmap (mem, &map);
  return result;
}

/**
 * @brief Internal function to transform the input buffer.
 */
static GstFlowReturn
gst_tensor_crop_chain (GstTensorCrop * self,
    GstCollectData * data_raw, GstCollectData * data_info)
{
  GstFlowReturn ret;
  GstBuffer *buf_raw, *buf_info, *result;
  tensor_crop_info_s cinfo;
  gboolean drop_raw, drop_info;

  g_return_val_if_fail (data_raw && data_info, GST_FLOW_ERROR);

  buf_raw = gst_collect_pads_peek (self->collect, data_raw);
  buf_info = gst_collect_pads_peek (self->collect, data_info);
  drop_raw = (buf_raw != NULL);
  drop_info = (buf_info != NULL);

  if (!buf_raw || !buf_info) {
    ret = GST_FLOW_EOS;
    goto done;
  }

  /**
   * The case when raw and info have different timestamp.
   * Compare timestamp and if time diff is less than lateness, crop raw buffer.
   */
  if (self->lateness >= 0) {
    GstClockTime ts_raw, ts_info, lateness;

    ts_raw = GST_BUFFER_TIMESTAMP (buf_raw);
    ts_info = GST_BUFFER_TIMESTAMP (buf_info);
    lateness = self->lateness * GST_MSECOND;

    if (GST_CLOCK_TIME_IS_VALID (ts_raw) && GST_CLOCK_TIME_IS_VALID (ts_info)) {
      if (ABS (GST_CLOCK_DIFF (ts_raw, ts_info)) > lateness) {
        GST_DEBUG_OBJECT (self, "Drop old buffer and wait for next.");
        GST_DEBUG_OBJECT (self, "Raw buffer ts: %" GST_TIME_FORMAT,
            GST_TIME_ARGS (ts_raw));
        GST_DEBUG_OBJECT (self, "Info buffer ts: %" GST_TIME_FORMAT,
            GST_TIME_ARGS (ts_info));

        /* clear old buffer and return ok to get next buffer */
        if (ts_raw > ts_info)
          drop_raw = FALSE;
        else
          drop_info = FALSE;

        ret = GST_FLOW_OK;
        goto done;
      }
    } else {
      GST_WARNING_OBJECT (self,
          "Incoming buffer has invalid timestamp, continue cropping data.");
    }
  }

  if (!gst_tensor_crop_get_crop_info (self, buf_info, &cinfo)) {
    ret = GST_FLOW_ERROR;
    goto done;
  }

  result = gst_tensor_crop_do_cropping (self, buf_raw, &cinfo);
  ret = gst_pad_push (self->srcpad, result);

done:
  if (buf_raw)
    gst_buffer_unref (buf_raw);
  if (buf_info)
    gst_buffer_unref (buf_info);

  /* clear buffer in collect pads */
  if (drop_raw)
    gst_buffer_unref (gst_collect_pads_pop (self->collect, data_raw));
  if (drop_info)
    gst_buffer_unref (gst_collect_pads_pop (self->collect, data_info));

  return ret;
}

/**
 * @brief Chain function called when the buffer is available on all of the collect pads.
 */
static GstFlowReturn
gst_tensor_crop_collected (GstCollectPads * pads, gpointer user_data)
{
  GstTensorCrop *self;
  GstCollectData *data_raw, *data_info;
  GSList *walk;
  GstFlowReturn ret;

  self = GST_TENSOR_CROP (user_data);
  data_raw = data_info = NULL;

  ret = gst_tensor_crop_negotiate (self);
  if (ret != GST_FLOW_OK)
    return ret;

  for (walk = pads->data; walk; walk = g_slist_next (walk)) {
    GstCollectData *data;

    data = (GstCollectData *) walk->data;

    if (data->pad == self->sinkpad_raw) {
      data_raw = data;
    } else if (data->pad == self->sinkpad_info) {
      data_info = data;
    }
  }

  return gst_tensor_crop_chain (self, data_raw, data_info);
}
