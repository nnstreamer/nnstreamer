/**
 * @file	dummy_sensor.h
 * @date	28 Nov 2019
 * @brief	Dummy Tizen Sensor API support for unit tests.
 * @see		https://github.com/nnsuite/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 * @details	The sensor framework source plugin should be
 *              linked with dummy_sensor.
 *
 *              This will simply connect values from publish()
 *              to listener().
 *
 *              This has sensor-fw APIs that are used by
 *              nnstreamer only.
 */
#ifndef __DUMMY_SENSOR_H__
#define __DUMMY_SENSOR_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
#include <stdint.h>
#include <stdbool.h>
#include <glib.h>

#include <errno.h>
#include <tizen_error.h>


typedef enum
{
	SENSOR_ALL = -1,                        /**< All sensors. This can be used to retrieve #sensor_h for all available sensors. */
	SENSOR_ACCELEROMETER,                   /**< Accelerometer */
	SENSOR_GRAVITY,                         /**< Gravity sensor */
	SENSOR_LINEAR_ACCELERATION,             /**< Linear acceleration sensor */
	SENSOR_MAGNETIC,                        /**< Magnetic sensor */
	SENSOR_ROTATION_VECTOR,                 /**< Rotation vector sensor */
	SENSOR_ORIENTATION,                     /**< Orientation sensor */
	SENSOR_GYROSCOPE,                       /**< Gyroscope */
	SENSOR_LIGHT,                           /**< Light sensor */
	SENSOR_PROXIMITY,                       /**< Proximity sensor */
	SENSOR_PRESSURE,                        /**< Pressure sensor */
	SENSOR_ULTRAVIOLET,                     /**< Ultraviolet sensor */
	SENSOR_TEMPERATURE,                     /**< Temperature sensor */
	SENSOR_HUMIDITY,                        /**< Humidity sensor */
	SENSOR_HRM,                             /**< Heart-rate monitor @if MOBILE (Since 2.3.1) @endif
	                                             @n Privilege : http://tizen.org/privilege/healthinfo */
	SENSOR_HRM_LED_GREEN,                   /**< Green LED sensor of HRM @if MOBILE (Since 2.3.1) @endif
	                                             @n Privilege : http://tizen.org/privilege/healthinfo */
	SENSOR_HRM_LED_IR,                      /**< Infra-Red LED sensor of HRM @if MOBILE (Since 2.3.1) @endif
	                                             @n Privilege : http://tizen.org/privilege/healthinfo */
	SENSOR_HRM_LED_RED,                     /**< Red LED sensor of HRM @if MOBILE (Since 2.3.1) @endif
	                                             @n Privilege : http://tizen.org/privilege/healthinfo */
	SENSOR_GYROSCOPE_UNCALIBRATED,          /**< Uncalibrated Gyroscope sensor
	                                             @if MOBILE (Since 2.4) @elseif WEARABLE (Since 2.3.2) @endif */
	SENSOR_GEOMAGNETIC_UNCALIBRATED,        /**< Uncalibrated Geomagnetic sensor
	                                             @if MOBILE (Since 2.4) @elseif WEARABLE (Since 2.3.2) @endif */
	SENSOR_GYROSCOPE_ROTATION_VECTOR,       /**< Gyroscope-based rotation vector sensor
	                                             @if MOBILE (Since 2.4) @elseif WEARABLE (Since 2.3.2) @endif */
	SENSOR_GEOMAGNETIC_ROTATION_VECTOR,     /**< Geomagnetic-based rotation vector sensor
	                                             @if MOBILE (Since 2.4) @elseif WEARABLE (Since 2.3.2) @endif */
	SENSOR_SIGNIFICANT_MOTION = 0x100,      /**< Significant motion sensor (Since 4.0) */
	SENSOR_HUMAN_PEDOMETER = 0x300,         /**< Pedometer (Since 3.0)
	                                             @n Privilege : http://tizen.org/privilege/healthinfo */
	SENSOR_HUMAN_SLEEP_MONITOR,             /**< Sleep monitor (Since 3.0)
	                                             @n Privilege : http://tizen.org/privilege/healthinfo */
	SENSOR_HUMAN_SLEEP_DETECTOR,            /**< Sleep detector (Since 3.0)
	                                             @n Privilege : http://tizen.org/privilege/healthinfo */
	SENSOR_HUMAN_STRESS_MONITOR,            /**< Stress monitor (Since 3.0)
	                                             @n Privilege : http://tizen.org/privilege/healthinfo */
	SENSOR_LAST,                            /**< End of sensor enum values (Deprecated since 3.0) */
	SENSOR_CUSTOM = 0x2710,                 /**< Custom sensor (Deprecated since 3.0) */
} sensor_type_e;

/**
 * @brief   Enumeration for errors.
 * @since_tizen @if MOBILE 2.3 @elseif WEARABLE 2.3.1 @endif
 */
typedef enum {
	SENSOR_ERROR_NONE                  = TIZEN_ERROR_NONE,                 /**< Successful */
	SENSOR_ERROR_IO_ERROR              = TIZEN_ERROR_IO_ERROR,             /**< I/O error */
	SENSOR_ERROR_INVALID_PARAMETER     = TIZEN_ERROR_INVALID_PARAMETER,    /**< Invalid parameter */
	SENSOR_ERROR_NOT_SUPPORTED         = TIZEN_ERROR_NOT_SUPPORTED,        /**< Not supported */
	SENSOR_ERROR_PERMISSION_DENIED     = TIZEN_ERROR_PERMISSION_DENIED,    /**< Permission denied */
	SENSOR_ERROR_OUT_OF_MEMORY         = TIZEN_ERROR_OUT_OF_MEMORY,        /**< Out of memory */
	SENSOR_ERROR_NO_DATA               = TIZEN_ERROR_NO_DATA,              /**< No data available
                                                                                @if MOBILE (Since 3.0) @elseif WEARABLE (Since 2.3.2) @endif */
	SENSOR_ERROR_NOT_NEED_CALIBRATION  = TIZEN_ERROR_SENSOR | 0x03,        /**< Sensor doesn't need calibration */
	SENSOR_ERROR_OPERATION_FAILED      = TIZEN_ERROR_SENSOR | 0x06,        /**< Operation failed */
	SENSOR_ERROR_NOT_AVAILABLE         = TIZEN_ERROR_SENSOR | 0x07,        /**< The sensor is supported, but currently not available
                                                                                @if MOBILE (Since 3.0) @elseif WEARABLE (Since 2.3.2) @endif */
} sensor_error_e;


/* event should be exactly same with the original */
typedef struct
{
  int accuracy;                  /**< Accuracy of sensor data */
  unsigned long long timestamp;  /**< Time when the sensor data was observed */
  int value_count;               /**< Number of sensor data values stored in #sensor_event_s::values */
  float values[16];  /**< Sensor data values */
} sensor_event_s;

typedef struct {
  sensor_type_e type;
  uint32_t id;
  GHashTable *listeners;
  sensor_event_s last_recorded;
} sensor_s;
typedef void* sensor_h;

typedef struct {
  sensor_s *listening;
  int is_listening;
} sensor_listener_s;

typedef void* sensor_listener_h;





/* main */
extern int
sensor_is_supported (sensor_type_e type, bool * supported);

extern int
sensor_get_default_sensor (sensor_type_e type, sensor_h *sensor);

extern int
sensor_get_sensor_list (sensor_type_e type, sensor_h **list, int *sensor_count);

extern int
sensor_get_type (sensor_h sensor, sensor_type_e *type);



/* listener */
extern int
sensor_create_listener (sensor_h sensor, sensor_listener_h *listener);

extern int
sensor_destroy_listener (sensor_listener_h listener);

extern int
sensor_listener_start (sensor_listener_h listener);

extern int
sensor_listener_stop (sensor_listener_h listener);

extern int
sensor_listener_read_data (sensor_listener_h listener, sensor_event_s *event);


/* publish data */
extern int
dummy_publish (sensor_h sensor, sensor_event_s value);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __DUMMY_SENSOR_H__ */
