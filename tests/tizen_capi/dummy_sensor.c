/**
 * @file	dummy_sensor.h
 * @date	28 Nov 2019
 * @brief	Dummy Tizen Sensor API support for unit tests.
 * @see		https://github.com/nnsuite/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 * @details	The sensor framework source plugin should be
 *              linked with dummy_sensor.
 */

#include <time.h>

#include "dummy_sensor.h"
#include <errno.h>
#include <string.h>

static void init_timestamps (void) __attribute__ ((constructor));

static sensor_s sensors[][3] = {
  /* 0 = SENSOR_ACCELEROMETER */
  {{.type = SENSOR_ACCELEROMETER,.id = 0,.listeners = NULL, .last_recorded =
              {0}},
      {.type = SENSOR_ACCELEROMETER,.id = 1,.listeners = NULL, .last_recorded =
            {0}},
      {.type = SENSOR_ACCELEROMETER,.id = 2,.listeners = NULL, .last_recorded =
            {0}}},
  {{0}, {0}, {0}},                          /* 1 */
  {{0}, {0}, {0}},                          /* 1 */
  {{0}, {0}, {0}},                          /* 2 */
  {{0}, {0}, {0}},                          /* 3 */
  {{0}, {0}, {0}},                          /* 4 */
  {{0}, {0}, {0}},                          /* 5 */
  {{0}, {0}, {0}},                          /* 6 */
  {{.type = SENSOR_LIGHT,.id = 0,.listeners = NULL, .last_recorded = {0}},
      {.type = SENSOR_LIGHT,.id = 1,.listeners = NULL, .last_recorded = {0}},
      {.type = SENSOR_LIGHT,.id = 2,.listeners = NULL, .last_recorded = {0}}}
};

/**
 * @brief Dummy (simulation) Tizen Sensor Framework API
 */
int
sensor_is_supported (sensor_type_e type, bool * supported)
{
  if (type == SENSOR_ACCELEROMETER || type == SENSOR_LIGHT)
    *supported = true;
  else
    *supported = false;
  return 0;
}

/**
 * @brief Dummy (simulation) Tizen Sensor Framework API
 */
int
sensor_get_default_sensor (sensor_type_e type, sensor_h * sensor)
{
  bool supported;

  sensor_is_supported (type, &supported);
  if (supported == false) {
    return -EINVAL;
  }

  *sensor = &(sensors[type][0]);
  return 0;
}

/**
 * @brief Dummy (simulation) Tizen Sensor Framework API
 */
int
sensor_get_sensor_list (sensor_type_e type, sensor_h ** list, int *sensor_count)
{
  bool supported;
  int i;

  sensor_is_supported (type, &supported);
  if (supported == false) {
    *list = NULL;
    *sensor_count = 0;
    return 0;
  }

  *list = g_new0 (sensor_h, 3);
  for (i = 0; i < 3; i++)
    (*list)[i] = &(sensors[type][i]);
  *sensor_count = 3;

  return 0;
}

/**
 * @brief Dummy (simulation) Tizen Sensor Framework API
 */
int
sensor_get_type (sensor_h sensor, sensor_type_e * type)
{
  sensor_s *ptr = sensor;
  bool supported;

  sensor_is_supported (ptr->type, &supported);
  if (supported)
    *type = ptr->type;
  else
    return -EINVAL;

  return 0;
}

/**
 * @brief Dummy (simulation) Tizen Sensor Framework API
 */
int
sensor_create_listener (sensor_h sensor, sensor_listener_h * listener)
{
  sensor_listener_s *ptr = g_new0 (sensor_listener_s, 1);
  sensor_type_e type;
  GHashTable *table;

  ptr->is_listening = 0;
  ptr->listening = sensor;

  if (NULL == sensor || sensor_get_type (sensor, &type) < 0 ||
      ptr->listening->id > 3)
    return -EINVAL;

  if (NULL == ptr->listening->listeners) {
    ptr->listening->listeners = g_hash_table_new (NULL, NULL);
  }
  table = ptr->listening->listeners;

  g_hash_table_add (table, ptr);

  *listener = ptr;
  return 0;
}

/**
 * @brief Dummy (simulation) Tizen Sensor Framework API
 */
int
sensor_destroy_listener (sensor_listener_h listener)
{
  bool removed = false;
  sensor_listener_s *l = listener;
  sensor_s *s;
  GHashTable *table;

  if (l == NULL)
    return -EINVAL;

  s = l->listening;
  if (s == NULL)
    return -EINVAL;

  table = s->listeners;
  if (table == NULL)
    return -EINVAL;

  removed = g_hash_table_remove (table, l);
  if (removed == false)
    return -EINVAL;

  return 0;
}

/**
 * @brief Dummy (simulation) Tizen Sensor Framework API
 */
int
sensor_listener_start (sensor_listener_h listener)
{
  sensor_listener_s *ptr = listener;
  if (NULL == listener)
    return -EINVAL;

  ptr->is_listening = 1;
  return 0;
}

/**
 * @brief Dummy (simulation) Tizen Sensor Framework API
 */
int
sensor_listener_stop (sensor_listener_h listener)
{
  sensor_listener_s *ptr = listener;
  if (NULL == listener)
    return -EINVAL;

  ptr->is_listening = 0;
  return 0;
}

/**
 * @brief Dummy (simulation) Tizen Sensor Framework API
 */
int
sensor_listener_read_data (sensor_listener_h listener, sensor_event_s * event)
{
  sensor_listener_s *ptr = listener;
  sensor_s *s;

  if (NULL == listener || NULL == event)
    return -EINVAL;

  s = ptr->listening;
  if (NULL == s || !ptr->is_listening)
    return -EINVAL;

  memcpy (event, &(s->last_recorded), sizeof (sensor_event_s));

  return 0;
}

/**
 * @brief Dummy Tizen Sensor.
 */
int
dummy_publish (sensor_h sensor, sensor_event_s value)
{
  sensor_s *s;

  if (NULL == sensor)
    return -EINVAL;

  s = sensor;

  memcpy (&(s->last_recorded), &value, sizeof (sensor_event_s));

  if (s->last_recorded.timestamp == 0) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    s->last_recorded.timestamp = ((unsigned long long)(t.tv_sec)*1000000LL +
        (unsigned long long)(t.tv_nsec)/1000);
  }

  return 0;
}

/**
 * @brief Initialize default timestamps values to avoid runtime errors in Gst
 */
static void init_timestamps (void)
{
  struct timespec t;
  unsigned long long ts;
  clock_gettime(CLOCK_MONOTONIC, &t);
  ts = ((unsigned long long)(t.tv_sec)*1000000LL +
      (unsigned long long)(t.tv_nsec)/1000);
  sensors[0][0].last_recorded.timestamp = ts;
  sensors[0][1].last_recorded.timestamp = ts;
  sensors[0][2].last_recorded.timestamp = ts;
  sensors[7][0].last_recorded.timestamp = ts;
  sensors[7][1].last_recorded.timestamp = ts;
  sensors[7][2].last_recorded.timestamp = ts;
}
