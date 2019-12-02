/**
 * GStreamer Tensor_Src_TizenSensor
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 *
 */
/**
 * @file	tensor_src_tizensensor.c
 * @date	07 Nov 2019
 * @brief	GStreamer plugin to support Tizen sensor framework (sensord)
 * @see		http://github.com/nnsuite/nnstreamer
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
 * gst-launch -v -m tensor_src_tizensensor type=ACCELEROMETER sequence=0 mode=POLLING freq=1/1 ! fakesink
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
 * in sensor.h of Tizen with type, instead of typename.
 *
 * @todo More manual entries coming.
 * @todo Allow to use sensor URIs to designate a sensor
 * https://docs.tizen.org/application/native/api/mobile/latest/group__CAPI__SYSTEM__SENSOR__LISTENER__MODULE.html#CAPI_SYSTEM_SENSOR_LISTENER_MODULE_URI
 * @todo Some sensor tpes are privileged. We need privilege control.
 * Some sensor types are privileged. An application should have the privilege http://tizen.org/privilege/healthinfo to get handles for the following sensors: SENSOR_HRM, SENSOR_HRM_LED_GREEN, SENSOR_HRM_LED_IR, SENSOR_HRM_LED_RED, SENSOR_HUMAN_PEDOMETER, SENSOR_HUMAN_SLEEP_MONITOR, SENSOR_HUMAN_SLEEP_DETECTOR, and SENSOR_HUMAN_STRESS_MONITOR.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gst.h>
#include <glib.h>
#include <errno.h>

/** @todo VALIDATE: Tizen's System/Sensor Public C-API */
#include <sensor.h>

#include <tensor_typedef.h>

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
 * @brief tensor_src_iio properties.
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
#define DEFAULT_PROP_FREQ_N 1
#define DEFAULT_PROP_FREQ_D 1

/**
 * @brief Default sequence number
 */
#define DEFAULT_PROP_SEQUENCE 0

#define _LOCK(obj) g_mutex_lock (&(obj)->lock);
#define _UNLOCK(obj) g_mutex_unlock (&(obj)->lock);

/**
 * @brief Template for src pad.
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT));

/** GObject method implementation */
static void gst_tensor_src_tizensensor_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_src_tizensensor_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_tensor_src_tizensensor_finalize (GObject * object);

/** GstBaseSrc method implementation */
static gboolean gst_tensor_src_tizensensor_start (GstBaseSrc * src);
static gboolean gst_tensor_src_tizensensor_stop (GstBaseSrc * src);
static GstStateChangeReturn gst_tensor_src_tizensensor_change_state (GstElement
    * element, GstStateChange transition);
static gboolean gst_tensor_src_tizensensor_event (GstBaseSrc * src,
    GstEvent * event);
static gboolean gst_tensor_src_tizensensor_set_caps (GstBaseSrc * src,
    GstCaps * caps);
static GstCaps *gst_tensor_src_tizensensor_get_caps (GstBaseSrc * src,
    GstCaps * filter);
static GstCaps *gst_tensor_src_tizensensor_fixate (GstBaseSrc * src,
    GstCaps * caps);
static gboolean gst_tensor_src_tizensensor_is_seekable (GstBaseSrc * src);
static GstFlowReturn gst_tensor_src_tizensensor_create (GstBaseSrc * src,
    guint64 offset, guint size, GstBuffer ** buf);
static GstFlowReturn gst_tensor_src_tizensensor_fill (GstBaseSrc * src,
    guint64 offset, guint size, GstBuffer * buf);
static void gst_tensor_src_tizensensor_get_times (GstBaseSrc * basesrc,
    GstBuffer * buffer, GstClockTime * start, GstClockTime * end);

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

/**
 * @brief Sensor data retrieval modes
 * @details More entries coming soon!
 */
typedef enum
{
  TZN_SENSOR_MODE_POLLING = 0;
} sensor_op_modes;

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

  gobject_class->set_property = gst_tensor_src_tizensensor_set_property;
  gobject_class->get_property = gst_tensor_src_tizensensor_get_property;
  gobject_class->finalize = gst_tensor_src_tizensensor_finalize;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent",
          "Produce verbose output", DEFAULT_PROP_SILENT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_TYPE,
      g_param_spec_enum ("typename", "Tizen Sensor Type Name (enum)",
          "Tizen sensor type as a enum-name, defined in sensor.h of Tizen",
          GST_TYPE_TIZEN_SENSOR_TYPE, SENSOR_ALL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SEQUENCE,
      g_param_spec_uint ("sequence", "Sequence number of a sensor type",
          "Select a sensor if there are multiple sensors of a type",
          0, G_MAXUINT, DEFAULT_PROP_SEQUENCE,
          G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_MODE,
      g_param_spec_enum ("mode", "Sensor data retrieval mode (enum)",
          "Determine how sensor data are retrieved (e.g. polling)",
          GST_TYPE_TIZEN_SENSOR_MODE, TZN_SENSOR_MODE_POLLING,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_FREQ,
      gst_param_spec_fraction ("freq", "Frequency",
          "Rate of data retrievals from a sensor",
          0, 1, G_MAXINT, 1,
          DEFAULT_PROP_FREQ_N, DEFAULT_PROP_FREQ_D,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_static_pad_template (gstelement_class, &src_factory);
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
  gstbasesrc_class->get_times =
      GST_DEBUG_FUNCPTR (gst_tensor_src_tizensensor_get_times);
  gstbasesrc_class->start =
      GST_DEBUG_FUNCPTR (gst_tensor_src_tizensensor_start);
  gstbasesrc_class->stop = GST_DEBUG_FUNCPTR (gst_tensor_src_tizensensor_stop);
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

  g_mutex_init (&self->lock);

  /**
   * @todo TBD. Update This!
   * format of the source since IIO device as a source is live and operates
   * at a fixed frequency, GST_FORMAT_TIME is used
   */
  gst_base_src_set_format (GST_BASE_SRC (self), GST_FORMAT_TIME);
  /** set the source to be live */
  gst_base_src_set_live (GST_BASE_SRC (self), TRUE);
  /** set the timestamps on each buffer */
  gst_base_src_set_do_timestamp (GST_BASE_SRC (self), TRUE);
  /**
   * set async is necessary to make state change async
   * sync state changes does not need calling _start_complete() from _start()
   */
  gst_base_src_set_async (GST_BASE_SRC (self), TRUE);
}

/**
 * @brief This cleans up.
 * @details This cleans up the Tizen sensor handle/context,
 *          ready for a new handle/context or exit.
 *          This does not alter saved properties.
 */
static int
_ts_clean_up_handle (GstTensorSrcTIZENSENSOR *self)
{
  self->configured = FALSE;

  return -EINVAL; /** @todo NYI */
}

/**
 * @brief Get handle, setup context, make it ready!
 */
static int
_ts_configure_handle (GstTensorSrcTIZENSENSOR *self)
{
  self->configured = TRUE;

  return -EINVAL; /** @todo NYI */
}

/**
 * @brief Keeping the handle/context, reconfigure a few parameters
 */
static int
_ts_reconfigure (GstTensorSrcTIZENSENSOR *self)
{
  /* Nothing to do if it's not configured, yet */
  if (FALSE == self->configured)
    return 0;

  /** @todo If the stream is "flowing", we need some special treatments */

  self->configured = TRUE; /* Keep it configured! */
  return -EINVAL; /** @todo NYI */
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
        ret = _ts_clean_up_handle (self);

        if (ret) {
          GST_ELEMENT_ERROR (self, TIZEN_SENSOR, FAILED,
              ("Calling _ts_clean_up_handle at set PROP_TYPE has failed."),
              ("_ts_clean_up_handle() returns %d", ret));
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
      guint new_sequence = g_value_get_uint (value);

      _LOCK (self);

      if (self->sequence != new_sequence) {
        /* Different sensor is being used. Clean it up! */
        ret = _ts_clean_up_handle (self);

        if (ret) {
          GST_ELEMENT_ERROR (self, TIZEN_SENSOR, FAILED,
              ("Calling _ts_clean_up_handle at set PROP_SEQUENCE has failed."),
              ("_ts_clean_up_handle() returns %d", ret));
        }

        silent_debug ("Set sequence from %u --> %u.", self->sequence,
            new_sequence);
        self->sequence = new_sequence;
      } else {
        silent_debug ("Set sequence ignored (%u --> %u).", self->sequence,
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
          GST_ELEMENT_ERROR (self, TIZEN_SENSOR, FAILED,
              ("Calling _ts_reconfigure at set PROP_MODE has failed."),
              ("_ts_reconfigure () returns %d", ret));
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

      silent_debug ("Set operating frequency %d/%d --> %d/%d",
          n, d, self->freq_n, self->freq_d);

      if (n != self->freq_n || d != self->freq_d) {
        /* Same sensor is kept. Only mode is changed */
        if (self->configured)
          ret = _ts_reconfigure (self);

        if (ret) {
          self->freq_n = n;
          self->freq_d = d;
          GST_ELEMENT_ERROR (self, TIZEN_SENSOR, FAILED,
              ("Calling _ts_reconfigure at set PROP_FREQ has failed."),
              ("_ts_reconfigure () returns %d", ret));
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
    g_value_set_uint (value, self->sequence);
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

  _LOCK (self);

  /* 1. Clean it up if there is a previous session */
  if (TRUE == self->configured) {
    ret = _ts_clean_up_handle (self);
    if (ret) {
      GST_ELEMENT_ERROR (self, TIZEN_SENSOR, FAILED,
          ("Start method failed, cleaning up previous context failed."),
          ("_ts_clean_up_handle () returns %d", ret));
      retval = FALSE; /* FAIL! */
      goto exit;
    }
  }

  /* 2. Configure handle / context */
  ret = _ts_configure_handle (self);
  g_assert (self->configured == TRUE);

  /** @todo TBD. Let's assume each frame has a fixed size */
  gst_base_src_set_dynamic_size (src, FALSE);

  /* 3. Fire it up! */
  g_assert (1 == 0); /** @todo NYI */

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
    GST_ELEMENT_ERROR (self, TIZEN_SENSOR, FAILED,
        ("Stop method failed, cleaning up previous context failed."),
        ("_ts_clean_up_handle () returns %d", ret));
    retval = FALSE; /* FAIL! */
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
  /** @todo NYI */

  /** No events to be handled yet */
  return GST_BASE_SRC_CLASS (parent_class)->event (src, event);
}

/**
 * @brief set new caps
 */
static gboolean
gst_tensor_src_tizensensor_set_caps (GstBaseSrc * src, GstCaps * caps)
{
  /** @todo NYI */

  return FALSE;                 /* FAIL. NYI */
}

/**
 * @brief get caps of subclass
 * @note basesrc _get_caps returns the caps from the pad_template
 * however, we set the caps manually and needs to returned here
 */
static GstCaps *
gst_tensor_src_tizensensor_get_caps (GstBaseSrc * src, GstCaps * filter)
{
  /** @todo NYI */

  return NULL;                  /* FAIL. NYI */
}

/**
 * @brief fixate the caps when needed during negotiation
 */
static GstCaps *
gst_tensor_src_tizensensor_fixate (GstBaseSrc * src, GstCaps * caps)
{
  /** @todo NYI */

  return NULL;                  /* FAIL. NYI */
}

/**
 * @brief Perform state change.
 */
static GstStateChangeReturn
gst_tensor_src_tizensensor_change_state (GstElement * element,
    GstStateChange transition)
{
  /** @todo NYI */

  return NULL;                  /* FAIL. NYI */
}

/**
 * @brief Sensor nodes are not seekable.
 */
static gboolean
gst_tensor_src_tizensensor_is_seekable (GstBaseSrc * src)
{
  return FALSE;
}

/**
 * @brief returns the time for the buffers
 */
static void
gst_tensor_src_tizensensor_get_times (GstBaseSrc * basesrc,
    GstBuffer * buffer, GstClockTime * start, GstClockTime * end)
{
  /** @todo NYI */
  g_assert (FALSE);
}

/**
 * @brief create a buffer with requested size and offset
 * @note offset, size ignored as the tensor src tizensensor does not support pull mode
 */
static GstFlowReturn
gst_tensor_src_tizensensor_create (GstBaseSrc * src, guint64 offset,
    guint size, GstBuffer ** buffer)
{
  /** @todo NYI */
  return GST_FLOW_ERROR;
}

/**
 * @brief fill the buffer with data
 * @note ignore offset,size as there is pull mode
 * @note buffer timestamp is already handled by gstreamer with gst clock
 */
static GstFlowReturn
gst_tensor_src_tizensensor_fill (GstBaseSrc * src, guint64 offset,
    guint size, GstBuffer * buffer)
{
  /** @todo NYI */
  return GST_FLOW_ERROR;
}

/**
 * @brief Register the plugin for GStreamer. This is an independent plugin.
 */
static gboolean
gst_nnstreamer_tizen_sensor_init (GstPlugin * plugin)
{
  if (!gst_element_register (plugin, "tensor_src_tizensensor",
          GST_RANK_NONE, GST_TYPE_TENSOR_SRC_TIZENSENSOR)) {
    GST_ERROR
        ("Failed to register nnstreamer's tensor_src_tizensensor plugin.");
    return FALSE;
  }

  return TRUE;
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nnstreamer_tizensensor,
    "nnstreamer Tizen sensor framework extension",
    gst_nnstreamer_tizen_sensor_init, VERSION, "LGPL", "nnstreamer",
    "https://github.com/nnsuite/nnstreamer");
