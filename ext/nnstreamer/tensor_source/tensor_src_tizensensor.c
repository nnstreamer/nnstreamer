/**
 * GStreamer Tensor_Src_TizenSensor
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 *
 */
/**
 * @file	tensor_src_tizensensor.c
 * @date	07 Nov 2019
 * @brief	GStreamer plugin to support Tizen sensor framework (sensord)
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */

/**
 * SECTION:element-tensor_src_tizensensor
 *
 * #tensor_src_tizensensor extends #gstbasesrc source element to handle Tizen
 * Sensor-Framework (sensord) as input.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m tensor_src_tizensensor type=ACCELEROMETER sequence=0 mode=POLLING ! fakesink
 * ]|
 * </refsect2>
 *
 * Available types can be retrieved with Tizen System/Sensor APIs:
 * https://docs.tizen.org/application/native/api/wearable/latest/group__CAPI__SYSTEM__SENSOR__MODULE.html#ga92804cd50337aef93d0e3a3807a9cf33 (Tizen 5.5 Mobile API)
 *
 * In case there are multiple sensors for a given sensor type,
 * you may designate the sequence to choose non-0 sensor instance.
 * which is equivalent to choosing list[sequence] from
 * Tizen-API / sensor_get_sensor_list(ACCELEROMETER, list, count);
 * When the sequence is not specified, the first (.0) is chosen.
 * You may specify the enum value of the sensor (sensor_type_e) defined
 * in sensor.h of Tizen with type.
 *
 * If sequence = -1 (default), we use "default sensor".
 *
 * @todo More manual entries coming.
 *
 * @todo Allow to use sensor URIs to designate a sensor
 * https://docs.tizen.org/application/native/api/mobile/latest/group__CAPI__SYSTEM__SENSOR__LISTENER__MODULE.html#CAPI_SYSTEM_SENSOR_LISTENER_MODULE_URI
 *
 * @todo Add "Listener" mode (creates data only if there are updates)
 *
 * @todo Every mode should handle timestamp/duration properly!
 *
 * @todo Add "power management" options (Tizen sensor f/w accepts such)
 *
 * @todo Some sensor types are privileged. We need privilege control.
 * Some sensor types are privileged. An application should have the privilege http://tizen.org/privilege/healthinfo to get handles for the following sensors: SENSOR_HRM, SENSOR_HRM_LED_GREEN, SENSOR_HRM_LED_IR, SENSOR_HRM_LED_RED, SENSOR_HUMAN_PEDOMETER, SENSOR_HUMAN_SLEEP_MONITOR, SENSOR_HUMAN_SLEEP_DETECTOR, and SENSOR_HUMAN_STRESS_MONITOR.
 *
 * @todo Some sensor types appear to have mixed types (float32 & int32).
 * @todo Add a property to set output tensor type (float32/int32/...) along
 *       with a few simple multiplications (e.g., x1000) and casting options.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <errno.h>

#include <gst/gst.h>
#include <glib.h>

/** @todo VALIDATE: Tizen's System/Sensor Public C-API */
#include <sensor.h>

#include <tensor_typedef.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_log.h>

#include "tensor_src_tizensensor.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

/**
 * @brief Macro for debug message.
 */
#define silent_debug(...) do { \
    if (DBG) { \
      GST_DEBUG_OBJECT (self, __VA_ARGS__); \
    } \
  } while (0)

GST_DEBUG_CATEGORY_STATIC (gst_tensor_src_tizensensor_debug);
#define GST_CAT_DEFAULT gst_tensor_src_tizensensor_debug

/**
 * @brief tensor_src_tizensensor properties.
 */
enum
{
  PROP_0,
  PROP_SILENT,
  PROP_TYPE,
  PROP_SEQUENCE,
  PROP_MODE,
  PROP_FREQ,
};

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_PROP_SILENT TRUE

/**
 * @brief Default Tizen sensor type
 */
#define DEFAULT_PROP_TYPE -1    /* Denotes "ALL" (any) */

/**
 * @brief Default sensor value retrieving mode
 */
#define DEFAULT_PROP_MODE "polling"

/**
 * @brief Default sensor retrieving frequency
 */
#define DEFAULT_PROP_FREQ_N 10
#define DEFAULT_PROP_FREQ_D 1

/**
 * @brief Default sequence number
 */
#define DEFAULT_PROP_SEQUENCE -1

#define _LOCK(obj) g_mutex_lock (&(obj)->lock)
#define _UNLOCK(obj) g_mutex_unlock (&(obj)->lock)

/** GObject method implementation */
static void gst_tensor_src_tizensensor_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_src_tizensensor_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_tensor_src_tizensensor_finalize (GObject * object);

/** GstBaseSrc method implementation */
static gboolean gst_tensor_src_tizensensor_start (GstBaseSrc * src);
static gboolean gst_tensor_src_tizensensor_stop (GstBaseSrc * src);
static gboolean gst_tensor_src_tizensensor_event (GstBaseSrc * src,
    GstEvent * event);
static gboolean gst_tensor_src_tizensensor_set_caps (GstBaseSrc * src,
    GstCaps * caps);
static GstCaps *gst_tensor_src_tizensensor_get_caps (GstBaseSrc * src,
    GstCaps * filter);
static GstCaps *gst_tensor_src_tizensensor_fixate (GstBaseSrc * src,
    GstCaps * caps);
static gboolean gst_tensor_src_tizensensor_is_seekable (GstBaseSrc * src);
static gboolean gst_tensor_src_tizensensor_query (GstBaseSrc * src, GstQuery * query);
static GstFlowReturn gst_tensor_src_tizensensor_create (GstBaseSrc * src,
    guint64 offset, guint size, GstBuffer ** buf);
static GstFlowReturn gst_tensor_src_tizensensor_fill (GstBaseSrc * src,
    guint64 offset, guint size, GstBuffer * buf);

/** internal functions */

#define gst_tensor_src_tizensensor_parent_class parent_class
G_DEFINE_TYPE (GstTensorSrcTIZENSENSOR, gst_tensor_src_tizensensor,
    GST_TYPE_BASE_SRC);

#define GST_TYPE_TIZEN_SENSOR_TYPE (tizen_sensor_get_type ())
/**
 * @brief Support GEnumValue array for Tizen sensor framework's sensor_type_e (sensor.h)
 * @todo We need an automated maintanence system for sensor.h's sensor_type_e, which makes a build error if it has been changed.
 */
static GType
tizen_sensor_get_type (void)
{
  static GType etype = 0;
  if (etype == 0) {
    static const GEnumValue values[] = {
      {SENSOR_ALL, "SENSOR_ALL", "all"},
      {SENSOR_ACCELEROMETER, "SENSOR_ACCELEROMETER", "accelerometer"},
      {SENSOR_GRAVITY, "SENSOR_GRAVITY", "gravity"},
      {SENSOR_LINEAR_ACCELERATION, "SENSOR_LINEAR_ACCELERATION",
          "linear_acceleration"},
      {SENSOR_MAGNETIC, "SENSOR_MAGNETIC", "magnetic"},
      {SENSOR_ROTATION_VECTOR, "SENSOR_ROTATION_VECTOR", "rotation_vector"},
      {SENSOR_ORIENTATION, "SENSOR_ORIENTATION", "orientation"},
      {SENSOR_GYROSCOPE, "SENSOR_GYROSCOPE", "gyroscope"},
      {SENSOR_LIGHT, "SENSOR_LIGHT", "light"},
      {SENSOR_PROXIMITY, "SENSOR_PROXIMITY", "proximity"},
      {SENSOR_PRESSURE, "SENSOR_PRESSURE", "pressure"},
      {SENSOR_ULTRAVIOLET, "SENSOR_ULTRAVIOLET", "ultraviolet"},
      {SENSOR_TEMPERATURE, "SENSOR_TEMPERATURE", "temperature"},
      {SENSOR_HUMIDITY, "SENSOR_HUMIDITY", "humidity"},
      {SENSOR_HRM, "SENSOR_HRM", "hrm"},
      {SENSOR_HRM_LED_GREEN, "SENSOR_HRM_LED_GREEN", "hrm_led_green"},
      {SENSOR_HRM_LED_IR, "SENSOR_HRM_LED_IR", "hrm_led_ir"},
      {SENSOR_HRM_LED_RED, "SENSOR_HRM_LED_RED", "hrm_led_red"},
      {SENSOR_GYROSCOPE_UNCALIBRATED, "SENSOR_GYROSCOPE_UNCALIBRATED",
          "gyroscope_uncalibrated"},
      {SENSOR_GEOMAGNETIC_UNCALIBRATED, "SENSOR_GEOMAGNETIC_UNCALIBRATED",
          "geomagnetic_uncalibrated"},
      {SENSOR_GYROSCOPE_ROTATION_VECTOR, "SENSOR_GYROSCOPE_ROTATION_VECTOR",
          "gyroscope_rotation_vector"},
      {SENSOR_GEOMAGNETIC_ROTATION_VECTOR, "SENSOR_GEOMAGNETIC_ROTATION_VECTOR",
          "geomagnetic_rotation_vector"},
      {SENSOR_SIGNIFICANT_MOTION, "SENSOR_SIGNIFICANT_MOTION",
          "significant_motion"},
      {SENSOR_HUMAN_PEDOMETER, "SENSOR_HUMAN_PEDOMETER", "human_pedometer"},
      {SENSOR_HUMAN_SLEEP_MONITOR, "SENSOR_HUMAN_SLEEP_MONITOR",
          "human_sleep_monitor"},
      {SENSOR_HUMAN_SLEEP_DETECTOR, "SENSOR_HUMAN_SLEEP_DETECTOR",
          "human_sleep_detector"},
      {SENSOR_HUMAN_STRESS_MONITOR, "SENSOR_HUMAN_STRESS_MONITOR",
          "human_stress_monitor"},
      {SENSOR_LAST, "SENSOR_LAST", "last"},
      {SENSOR_CUSTOM, "SENSOR_CUSTOM", "custom"},
      {0, NULL, NULL},
    };
    etype = g_enum_register_static ("sensor_type_e", values);
  }
  return etype;
}

static GHashTable *tizensensors = NULL;

/**
 * @brief Specification for each Tizen Sensor Type
 */
typedef struct
{
  sensor_type_e type;
  int value_count;
  GstTensorInfo tinfo;
} TizenSensorSpec;

/**
 * @brief Tizen sensor type specification
 * @details According to Tizen document,
 * https://developer.tizen.org/development/guides/native-application/location-and-sensors/device-sensors
 * Each sensor type has predetermined dimensions and types
 */
static TizenSensorSpec tizensensorspecs[] = {
  {.type = SENSOR_ACCELEROMETER,.value_count = 3,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {3, 1, 1, 1}}},
  {.type = SENSOR_GRAVITY,.value_count = 3,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {3, 1, 1, 1}}},
  {.type = SENSOR_LINEAR_ACCELERATION,.value_count = 3,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {3, 1, 1, 1}}},
  {.type = SENSOR_MAGNETIC,.value_count = 3,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {3, 1, 1, 1}}},
  {.type = SENSOR_ROTATION_VECTOR,.value_count = 4,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {4, 1, 1, 1}}},
  {.type = SENSOR_ORIENTATION,.value_count = 3,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {3, 1, 1, 1}}},
  {.type = SENSOR_GYROSCOPE,.value_count = 3,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {3, 1, 1, 1}}},
  {.type = SENSOR_LIGHT,.value_count = 1,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {1, 1, 1, 1}}},
  {.type = SENSOR_PROXIMITY,.value_count = 1,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {1, 1, 1, 1}}},
  {.type = SENSOR_PRESSURE,.value_count = 1,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {1, 1, 1, 1}}},
  {.type = SENSOR_ULTRAVIOLET,.value_count = 1,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {1, 1, 1, 1}}},
  {.type = SENSOR_TEMPERATURE,.value_count = 1,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {1, 1, 1, 1}}},
  {.type = SENSOR_HUMIDITY,.value_count = 1,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {1, 1, 1, 1}}},
  {.type = SENSOR_HRM,.value_count = 1,
        .tinfo = {.name = NULL,.type = _NNS_INT32,
          .dimension = {1, 1, 1, 1}}},
  {.type = SENSOR_HRM_LED_GREEN,.value_count = 1,
        .tinfo = {.name = NULL,.type = _NNS_INT32,
          .dimension = {1, 1, 1, 1}}},
  {.type = SENSOR_HRM_LED_IR,.value_count = 1,
        .tinfo = {.name = NULL,.type = _NNS_INT32,
          .dimension = {1, 1, 1, 1}}},
  {.type = SENSOR_HRM_LED_RED,.value_count = 1,
        .tinfo = {.name = NULL,.type = _NNS_INT32,
          .dimension = {1, 1, 1, 1}}},
  {.type = SENSOR_GYROSCOPE_UNCALIBRATED,.value_count = 6,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {6, 1, 1, 1}}},
  {.type = SENSOR_GEOMAGNETIC_UNCALIBRATED,.value_count = 6,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {6, 1, 1, 1}}},
  {.type = SENSOR_GYROSCOPE_ROTATION_VECTOR,.value_count = 4,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {4, 1, 1, 1}}},
  {.type = SENSOR_GEOMAGNETIC_ROTATION_VECTOR,.value_count = 4,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {4, 1, 1, 1}}},
  {.type = SENSOR_SIGNIFICANT_MOTION,.value_count = 1,
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {1, 1, 1, 1}}},
  {.type = SENSOR_HUMAN_PEDOMETER,.value_count = 8,     /* Last 5 values might be flost32..? */
        .tinfo = {.name = NULL,.type = _NNS_INT32,
          .dimension = {8, 1, 1, 1}}},
  {.type = SENSOR_HUMAN_SLEEP_MONITOR,.value_count = 1, /* STATE */
        .tinfo = {.name = NULL,.type = _NNS_INT32,
          .dimension = {1, 1, 1, 1}}},
  {.type = SENSOR_HUMAN_SLEEP_DETECTOR,.value_count = 1, /** @todo check! */
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {1, 1, 1, 1}}},
  {.type = SENSOR_HUMAN_STRESS_MONITOR,.value_count = 1, /** @todo check! */
        .tinfo = {.name = NULL,.type = _NNS_FLOAT32,
          .dimension = {1, 1, 1, 1}}},
  {.type = SENSOR_LAST,.value_count = 0,.tinfo = {0,}},
};

#define GST_TYPE_TIZEN_SENSOR_MODE (tizen_sensor_get_mode ())
/**
 * @brief Provide options of sensor operations
 */
static GType
tizen_sensor_get_mode (void)
{
  static GType etype = 0;
  if (etype == 0) {
    static const GEnumValue values[] = {
      {TZN_SENSOR_MODE_POLLING, "POLLING", "polling"},
      {0, NULL, NULL},
    };
    etype = g_enum_register_static ("sensor_op_modes", values);
  }
  return etype;
}

/**
 * @brief initialize the tensor_src_tizensensor class.
 */
static void
gst_tensor_src_tizensensor_class_init (GstTensorSrcTIZENSENSORClass * klass)
{
  GObjectClass *gobject_class = (GObjectClass *) klass;
  GstElementClass *gstelement_class = (GstElementClass *) klass;
  GstBaseSrcClass *gstbasesrc_class = (GstBaseSrcClass *) klass;
  GstPadTemplate *pad_template;
  GstCaps *pad_caps;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_src_tizensensor_debug,
      "tensor_src_tizensensor", 0,
      "src element to support Tizen sensor framework");

  gobject_class->set_property = gst_tensor_src_tizensensor_set_property;
  gobject_class->get_property = gst_tensor_src_tizensensor_get_property;
  gobject_class->finalize = gst_tensor_src_tizensensor_finalize;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent",
          "Produce verbose output", DEFAULT_PROP_SILENT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_TYPE,
      g_param_spec_enum ("type", "Tizen Sensor Type (enum)",
          "Tizen sensor type as a enum-name, defined in sensor.h of Tizen",
          GST_TYPE_TIZEN_SENSOR_TYPE, SENSOR_ALL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SEQUENCE,
      g_param_spec_int ("sequence", "Sequence number of a sensor type",
          "Select a sensor if there are multiple sensors of a type",
          -1, G_MAXINT, DEFAULT_PROP_SEQUENCE, G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_MODE,
      g_param_spec_enum ("mode", "Sensor data retrieval mode (enum)",
          "Determine how sensor data are retrieved (e.g. polling)",
          GST_TYPE_TIZEN_SENSOR_MODE, TZN_SENSOR_MODE_POLLING,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_FREQ,
      gst_param_spec_fraction ("framerate", "Framerate",
          "Rate of data retrievals from a sensor. Effective only when "
          "mode is ACTIVE_POLLING",
          0, 1, G_MAXINT, 1,
          DEFAULT_PROP_FREQ_N, DEFAULT_PROP_FREQ_D,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /* pad template */
  /** @todo Narrow down allowed tensors/tensor. */
  pad_caps = gst_caps_from_string (GST_TENSOR_CAP_DEFAULT "; "
      GST_TENSORS_CAP_WITH_NUM ("1"));
  pad_template = gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
      pad_caps);
  gst_element_class_add_pad_template (gstelement_class, pad_template);
  gst_caps_unref (pad_caps);

  gst_element_class_set_static_metadata (gstelement_class,
      "TensorSrcTizenSensor", "Source/Tizen-Sensor-FW/Tensor",
      "Creates tensor(s) stream from a given Tizen sensour framework node",
      "MyungJoo Ham <myungjoo.ham@samsung.com>");

  gstbasesrc_class->set_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_src_tizensensor_set_caps);
  gstbasesrc_class->get_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_src_tizensensor_get_caps);
  gstbasesrc_class->fixate =
      GST_DEBUG_FUNCPTR (gst_tensor_src_tizensensor_fixate);
  gstbasesrc_class->is_seekable =
      GST_DEBUG_FUNCPTR (gst_tensor_src_tizensensor_is_seekable);
  gstbasesrc_class->start =
      GST_DEBUG_FUNCPTR (gst_tensor_src_tizensensor_start);
  gstbasesrc_class->stop = GST_DEBUG_FUNCPTR (gst_tensor_src_tizensensor_stop);
  gstbasesrc_class->query = GST_DEBUG_FUNCPTR (gst_tensor_src_tizensensor_query);
  gstbasesrc_class->create =
      GST_DEBUG_FUNCPTR (gst_tensor_src_tizensensor_create);
  gstbasesrc_class->fill = GST_DEBUG_FUNCPTR (gst_tensor_src_tizensensor_fill);
  gstbasesrc_class->event =
      GST_DEBUG_FUNCPTR (gst_tensor_src_tizensensor_event);
}

/**
 * @brief initialize tensor_src_tizensensor element.
 */
static void
gst_tensor_src_tizensensor_init (GstTensorSrcTIZENSENSOR * self)
{
  /** init properties */
  self->configured = FALSE;
  self->silent = DEFAULT_PROP_SILENT;
  self->running = FALSE;
  self->freq_n = DEFAULT_PROP_FREQ_N;
  self->freq_d = DEFAULT_PROP_FREQ_D;

  g_mutex_init (&self->lock);

  /**
   * @todo TBD. Update This!
   * format of the source since IIO device as a source is live and operates
   * at a fixed frequency, GST_FORMAT_TIME is used
   */
  gst_base_src_set_format (GST_BASE_SRC (self), GST_FORMAT_TIME);
  /** set the source to be a live source */
  gst_base_src_set_live (GST_BASE_SRC (self), TRUE);
  /** set base_src to automatically timestamp outgoing buffers
   * based on the current running_time of the pipeline.
   */
  gst_base_src_set_do_timestamp (GST_BASE_SRC (self), TRUE);
  /**
   * set async is necessary to make state change async
   * sync state changes does not need calling _start_complete() from _start()
   */
  gst_base_src_set_async (GST_BASE_SRC (self), TRUE);

  /** @todo TBD. Let's assume each frame has a fixed size */
  gst_base_src_set_dynamic_size (GST_BASE_SRC (self), FALSE);

  if (NULL == tizensensors) {
    int i;
    tizensensors = g_hash_table_new (g_direct_hash, g_direct_equal);

    for (i = 0; tizensensorspecs[i].type != SENSOR_LAST; i++) {
      g_assert (g_hash_table_insert (tizensensors,
              GINT_TO_POINTER (tizensensorspecs[i].type),
              &tizensensorspecs[i].tinfo) == TRUE);
      g_assert (tizensensorspecs[i].value_count ==
          tizensensorspecs[i].tinfo.dimension[0]);
    }
  }
}

/**
 * @brief This cleans up.
 * @details This cleans up the Tizen sensor handle/context,
 *          ready for a new handle/context or exit.
 *          This does not alter saved properties.
 */
static int
_ts_clean_up_handle (GstTensorSrcTIZENSENSOR * self)
{
  if (TRUE == self->running) {
    sensor_listener_stop (self->listener);
    g_assert (self->configured == TRUE);
  }

  self->running = FALSE;

  if (TRUE == self->configured) {
    sensor_destroy_listener (self->listener);
  }

  self->src_spec = NULL;
  self->listener = NULL;
  self->sensor = NULL;

  self->configured = FALSE;
  return 0;
}

/**
 * @brief Sensor event (data retrieval) handler
 * @details This is for TZN_SENSOR_MODE_ACTIVE_POLLING
 */
static void __attribute__ ((unused))
_ts_tizen_sensor_callback (sensor_h sensor, sensor_event_s events[],
    int events_count, void *user_data)
{
  GstTensorSrcTIZENSENSOR *self = (GstTensorSrcTIZENSENSOR *) user_data;
  sensor_event_s *event;
  sensor_type_e type;
  int n_tensor_size = gst_tensor_get_element_count (self->src_spec->dimension);

  g_assert (self->configured);
  g_assert (self->running);
  g_assert (events_count > 0);

  /** @todo last or first sensor data? */
  event = &events[events_count - 1];

  sensor_get_type (sensor, &type);

  g_assert (type == self->type);
  g_assert (n_tensor_size == event->value_count);

  /** @todo Call some GST/BASESRC callback to fill things in from event */

  /** @todo Get proper timestamp from Tizen API, record it to metadata */

  g_assert (1 == 0);/** @todo NYI. Needed if we add more modes */
}

/**
 * @brief Calculate interval in ms from framerate
 * @details This is effective only for TZN_SENSOR_MODE_ACTIVE_POLLING.
 */
static unsigned int
_ts_get_interval_ms (GstTensorSrcTIZENSENSOR * self)
{
  if (self->freq_n == 0)
    return 100;                 /* If it's 0Hz, assume 100ms interval */

  g_assert (self->freq_d > 0 && self->freq_n > 0);

  return gst_util_uint64_scale_int ((guint64) self->freq_d, 1000, self->freq_n);
}

/**
 * @brief Get handle, setup context, make it ready!
 */
static int
_ts_configure_handle (GstTensorSrcTIZENSENSOR * self)
{
  int ret = 0;
  const GstTensorInfo *val = g_hash_table_lookup (tizensensors,
      GINT_TO_POINTER (self->type));
  bool supported = false;

  if (NULL == val) {
    nns_loge ("The given sensor type (%d) is not supported.\n", self->type);
    return -ENODEV;
  }
  self->src_spec = val;

  /* Based on Tizen Native App (Sensor) Guide */
  /* 1. Check if the sensor supported */
  ret = sensor_is_supported (self->type, &supported);
  if (ret != SENSOR_ERROR_NONE) {
    nns_loge ("Tizen sensor framework is not working (sensor_is_supported).\n");
    return -ENODEV;
  }

  if (false == supported) {
    GST_ERROR_OBJECT (self,
        "Tizen sensor framework API, sensor_is_supported(), says the sensor %d is not supported",
        self->type);
    return -EINVAL;
  }

  /* 2. Get sensor listener */
  if (self->sequence == -1) {
    /* Get the default sensor */
    ret = sensor_get_default_sensor (self->type, &self->sensor);
    if (ret != SENSOR_ERROR_NONE) {
      nns_loge ("Cannot get default sensor");
      return ret;
    }
  } else {
    sensor_h *list;
    int count;

    /* Use the sequence number to choose one */
    ret = sensor_get_sensor_list (self->type, &list, &count);
    if (ret != SENSOR_ERROR_NONE) {
      nns_loge ("Cannot get sensor list");
      return ret;
    }

    if (count <= self->sequence) {
      GST_ERROR_OBJECT (self,
          "The requested sensor sequence %d for sensor %d is not available. The max-sequence is used instead",
          self->sequence, self->type);
      self->sequence = 0;
      g_free (list);
      return -EINVAL;
    }

    self->sensor = list[self->sequence];
    g_free (list);
  }

  ret = sensor_create_listener (self->sensor, &self->listener);
  if (ret != SENSOR_ERROR_NONE) {
    nns_loge ("Cannot create sensor listener");
    return ret;
  }

  /* 3. Configure interval_ms */
  self->interval_ms = _ts_get_interval_ms (self);

  ret = sensor_listener_set_interval (self->listener, self->interval_ms);
  if (ret != SENSOR_ERROR_NONE) {
    nns_loge ("Cannot set the sensor interval");
    return ret;
  }

  nns_logi ("Set sensor_listener interval: %ums", self->interval_ms);

  /* 4. Register sensor event handler */
  switch (self->mode) {
    case TZN_SENSOR_MODE_POLLING:
      /* Nothing to do. Let Gst poll data */
      break;
#if 0 /** Use this if TZN_SENSOR_MODE_ACTIVE_POLLING is implemented */
    case TZN_SENSOR_MODE_ACTIVE_POLLING:
      ret = sensor_listener_set_events_cb (listener,
          _ts_tizen_sensor_callback, self);
      if (ret != SENSOR_ERROR_NONE)
        return ret;
      break;
#endif
    default:
      GST_ERROR_OBJECT (self,
          "The requested mode (%d) is invalid, use values defined in sensor_op_modes only.",
          self->mode);
  }

  self->configured = TRUE;
  return 0;
}

/**
 * @brief Keeping the handle/context, reconfigure a few parameters
 */
static int
_ts_reconfigure (GstTensorSrcTIZENSENSOR * self)
{
  _ts_clean_up_handle (self);
  return _ts_configure_handle (self);
}

/**
 * @brief set tensor_src_tizensensor properties
 */
static void
gst_tensor_src_tizensensor_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec)
{
  GstTensorSrcTIZENSENSOR *self = GST_TENSOR_SRC_TIZENSENSOR (object);
  int ret = 0;

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      silent_debug ("Set silent = %d", self->silent);
      break;
    case PROP_TYPE:
    {
      sensor_type_e new_type = g_value_get_enum (value);

      _LOCK (self);

      if (new_type != self->type) {
        /* Different sensor is being used. Clean it up! */
        if (self->configured)
          ret = _ts_clean_up_handle (self);

        if (ret) {
          GST_ERROR_OBJECT (self, "_ts_clean_up_handle() returns %d", ret);
        }

        silent_debug ("Set type from %d --> %d.", self->type, new_type);
        self->type = new_type;
      } else {
        silent_debug ("Set type ignored (%d --> %d).", self->type, new_type);
      }

      _UNLOCK (self);
    }
      break;
    case PROP_SEQUENCE:
    {
      gint new_sequence = g_value_get_int (value);

      _LOCK (self);

      if (self->sequence != new_sequence) {
        /* Different sensor is being used. Clean it up! */
        if (self->configured)
          ret = _ts_clean_up_handle (self);

        if (ret) {
          GST_ERROR_OBJECT (self, "_ts_clean_up_handle() returns %d", ret);
        }

        silent_debug ("Set sequence from %d --> %d.", self->sequence,
            new_sequence);
        self->sequence = new_sequence;
      } else {
        silent_debug ("Set sequence ignored (%d --> %d).", self->sequence,
            new_sequence);
      }

      _UNLOCK (self);
    }
      break;
    case PROP_MODE:
    {
      sensor_op_modes new_mode = g_value_get_enum (value);
      sensor_op_modes old_mode = self->mode;

      _LOCK (self);

      if (new_mode != self->mode) {
        silent_debug ("Set mode from %d --> %d.", self->mode, new_mode);
        self->mode = new_mode;

        /* Same sensor is kept. Only mode is changed */
        if (self->configured)
          ret = _ts_reconfigure (self);

        if (ret) {
          self->mode = old_mode;
          GST_ERROR_OBJECT (self, "_ts_reconfigure () returns %d", ret);
        }

      } else {
        silent_debug ("Set mode ignored (%d --> %d).", self->mode, new_mode);
      }

      _UNLOCK (self);
    }
      break;
    case PROP_FREQ:
    {
      gint n = self->freq_n;
      gint d = self->freq_d;

      _LOCK (self);

      self->freq_n = gst_value_get_fraction_numerator (value);
      self->freq_d = gst_value_get_fraction_denominator (value);

      if (self->freq_n < 0)
        self->freq_n = 0;
      if (self->freq_d < 1)
        self->freq_d = 1;

      silent_debug ("Set operating frequency %d/%d --> %d/%d",
          n, d, self->freq_n, self->freq_d);

      if (n != self->freq_n || d != self->freq_d) {
        /* Same sensor is kept. Only frequency is changed */
        if (self->configured)
          ret = _ts_reconfigure (self);

        if (ret) {
          self->freq_n = n;
          self->freq_d = d;
          GST_ERROR_OBJECT (self,
              "Calling _ts_reconfigure at set PROP_FREQ has failed. _ts_reconfigure () returns %d",
              ret);
        }
      }
      _UNLOCK (self);
    }
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief get tensor_src_tizensensor properties
 */
static void
gst_tensor_src_tizensensor_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec)
{
  GstTensorSrcTIZENSENSOR *self = GST_TENSOR_SRC_TIZENSENSOR (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_TYPE:
      g_value_set_enum (value, self->type);
      break;
    case PROP_SEQUENCE:
      g_value_set_int (value, self->sequence);
      break;
    case PROP_MODE:
      g_value_set_enum (value, self->mode);
      break;
    case PROP_FREQ:
      gst_value_set_fraction (value, self->freq_n, self->freq_d);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief finalize the instance
 */
static void
gst_tensor_src_tizensensor_finalize (GObject * object)
{
  GstTensorSrcTIZENSENSOR *self = GST_TENSOR_SRC_TIZENSENSOR (object);

  _LOCK (self);

  _ts_clean_up_handle (self);

  _UNLOCK (self);
  g_mutex_clear (&self->lock);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}


/**
 * @brief start function
 * @details This is called when state changed null to ready.
 *          load the device and init the device resources
 *          We won't configure before start is called.
 *          Postcondition: configured = TRUE. src = RUNNING.
 */
static gboolean
gst_tensor_src_tizensensor_start (GstBaseSrc * src)
{
  int ret = 0;
  GstTensorSrcTIZENSENSOR *self = GST_TENSOR_SRC_TIZENSENSOR_CAST (src);
  gboolean retval = TRUE;
  guint blocksize;

  _LOCK (self);

  /* 1. Clean it up if there is a previous session */
  if (TRUE == self->configured) {
    ret = _ts_clean_up_handle (self);
    if (ret) {
      GST_ERROR_OBJECT (self,
          "Start method failed, cleaning up previous context failed. _ts_clean_up_handle () returns %d",
          ret);
      retval = FALSE;           /* FAIL! */
      goto exit;
    }
  }

  /* 2. Configure handle / context */
  ret = _ts_configure_handle (self);
  if (ret) {
    retval = FALSE;
    goto exit;
  }
  g_assert (self->configured == TRUE);

  /* 3. Fire it up! */
  if (sensor_listener_start (self->listener) != 0) {
    /* Failed to start listener. Clean this up */
    ret = _ts_clean_up_handle (self);
    if (ret) {
      GST_ERROR_OBJECT (self, "_ts_clean_up_handle () returns %d", ret);
    }
    retval = FALSE;
    goto exit;
  }

  /* set data size */
  blocksize = gst_tensor_info_get_size (self->src_spec);
  gst_base_src_set_blocksize (src, blocksize);

  self->running = TRUE;

  /** complete the start of the base src */
  gst_base_src_start_complete (src, GST_FLOW_OK);

exit:
  _UNLOCK (self);
  return retval;
}

/**
 * @brief stop function.
 * @details This is called when state changed ready to null.
 *          Postcondition: configured = FALSE. src = STOPPED.
 */
static gboolean
gst_tensor_src_tizensensor_stop (GstBaseSrc * src)
{
  int ret = 0;
  GstTensorSrcTIZENSENSOR *self = GST_TENSOR_SRC_TIZENSENSOR_CAST (src);
  gboolean retval = TRUE;

  _LOCK (self);

  ret = _ts_clean_up_handle (self);
  if (ret) {
    GST_ERROR_OBJECT (self,
        "Stop method failed, cleaning up previous context failed. _ts_clean_up_handle () returns %d",
        ret);
    retval = FALSE;             /* FAIL! */
    goto exit;
  }

  g_assert (FALSE == self->configured);

exit:
  _UNLOCK (self);
  return retval;
}

/**
 * @brief handle events
 */
static gboolean
gst_tensor_src_tizensensor_event (GstBaseSrc * src, GstEvent * event)
{
  /** No events to be handled yet */
  return GST_BASE_SRC_CLASS (parent_class)->event (src, event);
}

/**
 * @brief Get possible GstCap from the configuration of self.
 */
static GstCaps *
_ts_get_gstcaps_from_conf (GstTensorSrcTIZENSENSOR * self)
{
  const GstTensorInfo *spec;
  GstCaps *retval;

  spec = self->src_spec;

  if (FALSE == self->configured || SENSOR_ALL == self->type || NULL == spec) {
    retval = gst_caps_from_string (GST_TENSOR_CAP_DEFAULT "; "
      GST_TENSORS_CAP_WITH_NUM ("1"));
  } else {
    GstTensorConfig tensor_config;
    GstTensorsConfig tensors_config;

    gst_tensor_config_init (&tensor_config);

    gst_tensors_config_init (&tensors_config);
    tensors_config.info.num_tensors = 1;

    gst_tensor_info_copy (&tensor_config.info, spec);
    tensor_config.rate_n = self->freq_n;
    tensor_config.rate_d = self->freq_d;

    retval = gst_tensor_caps_from_config (&tensor_config);

    gst_tensor_info_copy (&tensors_config.info.info[0], spec);
    tensors_config.rate_n = self->freq_n;
    tensors_config.rate_d = self->freq_d;

    gst_caps_append (retval, gst_tensors_caps_from_config (&tensors_config));
  }

  return retval;
}

/**
 * @brief set new caps
 * @retval TRUE if it's acceptable. FALSE if it's not acceptable.
 */
static gboolean
gst_tensor_src_tizensensor_set_caps (GstBaseSrc * src, GstCaps * caps)
{
  GstTensorSrcTIZENSENSOR *self = GST_TENSOR_SRC_TIZENSENSOR_CAST (src);
  GstCaps *cap_tensor;
  gboolean retval = FALSE;

  _LOCK (self);

  cap_tensor = _ts_get_gstcaps_from_conf (self);

  /* Check if it's compatible with either tensor or tensors */
  retval = gst_caps_can_intersect (caps, cap_tensor);
  gst_caps_unref (cap_tensor);

  _UNLOCK (self);
  return retval;
}

/**
 * @brief get caps of subclass
 * @note basesrc _get_caps returns the caps from the pad_template
 * however, we set the caps manually and needs to returned here
 */
static GstCaps *
gst_tensor_src_tizensensor_get_caps (GstBaseSrc * src, GstCaps * filter)
{
  GstCaps *caps;
  GstPad *pad;

  pad = src->srcpad;
  caps = gst_pad_get_current_caps (pad);

  if (caps == NULL)
    caps = gst_pad_get_pad_template_caps (pad);

  if (filter) {
    GstCaps *intersection =
        gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (caps);
    caps = intersection;
  }

  return caps;
}

/**
 * @brief fixate the caps when needed during negotiation
 */
static GstCaps *
gst_tensor_src_tizensensor_fixate (GstBaseSrc * src, GstCaps * caps)
{
  GstTensorSrcTIZENSENSOR *self = GST_TENSOR_SRC_TIZENSENSOR_CAST (src);
  GstCaps *cap_tensor;
  GstCaps *retval = NULL;

  _LOCK (self);

  cap_tensor = _ts_get_gstcaps_from_conf (self);

  if (TRUE == gst_caps_can_intersect (caps, cap_tensor))
    retval = gst_caps_intersect (caps, cap_tensor);
  gst_caps_unref (cap_tensor);

  _UNLOCK (self);
  return gst_caps_fixate (retval);
}

/**
 * @brief Sensor nodes are not seekable.
 */
static gboolean
gst_tensor_src_tizensensor_is_seekable (GstBaseSrc * src)
{
  nns_logd ("tensor_src_tizensensor is not seekable");
  return FALSE;
}

/**
 * @brief Handle queries.
 *
 * GstBaseSrc method implementation.
 */
static gboolean
gst_tensor_src_tizensensor_query (GstBaseSrc * src, GstQuery * query)
{
  gboolean res = FALSE;
  GstTensorSrcTIZENSENSOR *self = GST_TENSOR_SRC_TIZENSENSOR_CAST (src);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_LATENCY:
    {
      GstClockTime min_latency, max_latency = -1;
      gint freq_d, freq_n;

      freq_d = self->freq_d;
      freq_n = self->freq_n;

      /* we must have a framerate */
      if (freq_n <= 0 || freq_d <= 0) {
        GST_WARNING_OBJECT (self,
            "Can't give latency since framerate isn't fixated");
        goto done;
      }

      /* min latency is the time to capture one frame/field */
      min_latency = gst_util_uint64_scale_int (GST_SECOND, freq_d, freq_n);

      GST_DEBUG_OBJECT (self,
          "Reporting latency min %" GST_TIME_FORMAT " max %" GST_TIME_FORMAT,
          GST_TIME_ARGS (min_latency), GST_TIME_ARGS (max_latency));

      gst_query_set_latency (query, TRUE, min_latency, max_latency);

      res = TRUE;
      break;
    }
    default:
      res = GST_BASE_SRC_CLASS (parent_class)->query (src, query);
      break;
  }

done:
  return res;
}

/**
 * @brief create a buffer with requested size and offset
 * @note offset, size ignored as the tensor src tizensensor does not support pull mode
 */
static GstFlowReturn
gst_tensor_src_tizensensor_create (GstBaseSrc * src, guint64 offset,
    guint size, GstBuffer ** buffer)
{
  GstTensorSrcTIZENSENSOR *self = GST_TENSOR_SRC_TIZENSENSOR_CAST (src);
  GstBuffer *buf = gst_buffer_new ();
  GstMemory *mem;
  guint buffer_size;
  GstFlowReturn retval = GST_FLOW_OK;

  _LOCK (self);

  if (FALSE == self->configured) {
    GST_ERROR_OBJECT (self,
        "Buffer creation requested while the element is not configured. gst_tensor_src_tizensensor_create() cannot proceed if it is not configured.");
    retval = GST_FLOW_ERROR;
    goto exit;
  }

  g_assert (self->src_spec);    /* It should be valid if configured */

  /* We don't have multi-tensor (tensors with num-tensors > 1) */
  buffer_size = gst_tensor_info_get_size (self->src_spec);
  mem = gst_allocator_alloc (NULL, buffer_size, NULL);
  if (mem == NULL) {
    GST_ERROR_OBJECT (self,
        "Cannot allocate memory for gst buffer of %u bytes", buffer_size);
    retval = GST_FLOW_ERROR;
    goto exit;
  }
  gst_buffer_append_memory (buf, mem);

  _UNLOCK (self);
  retval = gst_tensor_src_tizensensor_fill (src, offset, buffer_size, buf);
  _LOCK (self);
  if (retval != GST_FLOW_OK)
    goto exit;

  *buffer = buf;

exit:
  _UNLOCK (self);
  return retval;
}

#define cast_loop(values, count, dest, desttype) \
do { \
  int i; \
  char *destptr = (char *) (dest); \
  for (i = 0; i < (count); i++) \
    *(destptr + (sizeof (desttype) * i)) = (desttype) (values)[i]; \
} while (0)

#define case_cast_loop(values, count, dest, desttype, desttypeenum) \
case desttypeenum: \
  cast_loop (values, count, dest, desttype); \
  break;

/**
 * @brief Copy sensor's values[] to Gst memory map.
 */
static void
_ts_assign_values (float values[], int count, GstMapInfo * map,
    const GstTensorInfo * spec)
{
  switch (spec->type) {
    case _NNS_FLOAT32:
      memcpy (map->data, values, sizeof (float) * count);
      break;
      case_cast_loop (values, count, map->data, int64_t, _NNS_INT64);
      case_cast_loop (values, count, map->data, int32_t, _NNS_INT32);
      case_cast_loop (values, count, map->data, int16_t, _NNS_INT16);
      case_cast_loop (values, count, map->data, int8_t, _NNS_INT8);
      case_cast_loop (values, count, map->data, uint64_t, _NNS_UINT64);
      case_cast_loop (values, count, map->data, uint32_t, _NNS_UINT32);
      case_cast_loop (values, count, map->data, uint16_t, _NNS_UINT16);
      case_cast_loop (values, count, map->data, uint8_t, _NNS_UINT8);
      case_cast_loop (values, count, map->data, double, _NNS_FLOAT64);
    default:
      g_assert (0);   /** Other types are not implemented! */
  }
}

/**
 * @brief fill the buffer with data
 * @note ignore offset,size as there is pull mode
 * @note buffer timestamp is already handled by gstreamer with gst clock
 * @note Get data from Tizen Sensor F/W. Get the timestamp as well!
 */
static GstFlowReturn
gst_tensor_src_tizensensor_fill (GstBaseSrc * src, guint64 offset,
    guint size, GstBuffer * buffer)
{
  GstTensorSrcTIZENSENSOR *self = GST_TENSOR_SRC_TIZENSENSOR_CAST (src);
  sensor_event_s *events = NULL;
  GstFlowReturn retval = GST_FLOW_OK;
  GstMemory *mem;
  GstMapInfo map;

  _LOCK (self);

  if (FALSE == self->configured) {
    GST_ERROR_OBJECT (self,
        "gst_tensor_src_tizensensor_fill() cannot proceed if it is not configured.");
    retval = GST_FLOW_ERROR;
    goto exit;
  }

  if (size != gst_tensor_info_get_size (self->src_spec)) {
    GST_ERROR_OBJECT (self,
        "gst_tensor_src_tizensensor_fill() requires size value (%u) to be matched with the configurations of sensors (%lu)",
        size, (unsigned long) gst_tensor_info_get_size (self->src_spec));
    retval = GST_FLOW_ERROR;
    goto exit;
  }

  mem = gst_buffer_peek_memory (buffer, 0);
  if (FALSE == gst_memory_map (mem, &map, GST_MAP_WRITE)) {
    GST_ERROR_OBJECT (self,
        "gst_tensor_src_tizensensor_fill() cannot map the given buffer for writing data.");
    retval = GST_FLOW_ERROR;
    goto exit;
  }

  if (self->mode == TZN_SENSOR_MODE_POLLING) {
    sensor_event_s *event;
    int count = 0;
    gint64 duration;
    int ret;

    /* 1. Read sensor data directly from Tizen API */
    ret = sensor_listener_read_data_list (self->listener, &events, &count);
    if (ret != SENSOR_ERROR_NONE || count == 0) {
      GST_ERROR_OBJECT (self,
          "Tizen sensor read failed: sensor_listener_read_data returned %d, count %d",
          ret, count);
      retval = GST_FLOW_ERROR;
      goto exit_unmap;
    }

    event = &events[count - 1];
    if (event->value_count != self->src_spec->dimension[0]) {
      GST_ERROR_OBJECT (self,
          "The number of values (%d) mismatches the metadata (%d)",
          event->value_count, self->src_spec->dimension[0]);
      retval = GST_FLOW_ERROR;
      goto exit_unmap;
    }

    /* 2. Do not timestamp. Let BaseSrc timestamp */

    nns_logd ("read sensor_data at %" GST_TIME_FORMAT,
        GST_TIME_ARGS (event->timestamp * 1000));

    /* 3. Set duration so that BaseSrc handles the frequency */
    if (self->freq_n == 0)
      /* 100ms */
      duration = 100 * 1000 * 1000;
    else
      duration = gst_util_uint64_scale_int (GST_SECOND,
          self->freq_d, self->freq_n);

    GST_BUFFER_DURATION (buffer) = duration;

    /* 4. Write values to buffer. Be careful on type casting */
    _ts_assign_values (event->values, event->value_count, &map, self->src_spec);
  } else {
    /** NYI! */
    GST_ERROR_OBJECT (self,
        "gst_tensor_src_tizensensor_fill reached unimplemented code.");
    retval = GST_FLOW_ERROR;
    goto exit_unmap;
  }

exit_unmap:
  g_free (events);
  gst_memory_unmap (mem, &map);
exit:
  _UNLOCK (self);
  return retval;
}
