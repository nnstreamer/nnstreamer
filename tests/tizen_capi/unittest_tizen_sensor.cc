/**
 * @file	unittest_tizen_sensor.cc
 * @date	25 Nov 2019
 * @brief	Unit test for NNStreamer's tensor-src-tizensensor.
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#ifndef __TIZEN__
/* These works only in Tizen */
#error This unit test works only in Tizen. This needs Tizen Sensor Framework.
#endif

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <nnstreamer-capi-private.h>
#include <nnstreamer.h>
#include <unittest_util.h>
#include "dummy_sensor.h" /* Dummy Tizen Sensor Framework */

static const unsigned int TEST_TIME_OUT_TIZEN_SENSOR_MS
    = 10000U; /* timeout occur after 10 seconds */

#define wait_for_start(handle, state, status)                     \
  do {                                                            \
    int counter = 0;                                              \
    while (state != ML_PIPELINE_STATE_PLAYING && counter < 100) { \
      g_usleep (100000);                                          \
      counter++;                                                  \
      status = ml_pipeline_get_state (handle, &state);            \
      EXPECT_EQ (status, ML_ERROR_NONE);                          \
    }                                                             \
  } while (0)

/**
 * @brief Test pipeline creation of it
 */
TEST (tizensensor_as_source, virtual_sensor_create_01)
{
  gchar *pipeline;
  sensor_event_s value;
  sensor_h sensor;
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;

  sensor_get_default_sensor (SENSOR_LIGHT, &sensor);

  value.accuracy = 1;
  value.timestamp = 0U;
  value.value_count = 1;
  value.values[0] = 0.01;
  EXPECT_EQ (dummy_publish (sensor, value), 0);

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "tensor_src_tizensensor type=SENSOR_LIGHT sequence=0 num-buffers=3 ! fakesink");
  gstpipe = gst_parse_launch (pipeline, &err);
  if (gstpipe) {
    status = 0;
    gst_object_unref (gstpipe);
  } else {
    status = -1;
    g_printerr ("GST PARSE LAUNCH FAILED: [%s], %s\n", pipeline,
        (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_free (pipeline);
}

/**
 * @brief Test pipeline creation.
 */
TEST (tizensensor_as_source, virtual_sensor_create_02)
{
  gchar *pipeline;
  sensor_event_s value;
  sensor_h sensor;
  sensor_h *sensor_list;
  GstElement *gstpipe;
  int status = 0;
  int count;

  sensor_get_sensor_list (SENSOR_LIGHT, &sensor_list, &count);
  EXPECT_EQ (count, 3);
  sensor = sensor_list[2];
  g_free (sensor_list);

  value.accuracy = 1;
  value.timestamp = 0U;
  value.value_count = 1;
  value.values[0] = 0.01;
  EXPECT_EQ (dummy_publish (sensor, value), 0);

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "tensor_src_tizensensor type=SENSOR_ACCELEROMETER sequence=-1 num-buffers=3 ! fakesink");
  gstpipe = gst_parse_launch (pipeline, NULL);
  if (gstpipe) {
    status = 0;
    gst_object_unref (gstpipe);
  } else {
    status = -1;
  }
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_free (pipeline);
}

#define MAX_VERIFY_DATA (256)
typedef struct {
  int cursor;
  int num_data;
  float golden[256][16];
  ml_tensor_type_e type;
  int dim0;
  unsigned int checked;
  int negative;
} verify_data;

static verify_data data; /* Too big for stack. Use this global var */

/**
 * @brief Test if the sensor-reading matches the golden values.
 */
static void
callback_nns (const ml_tensors_data_h data, const ml_tensors_info_h info, void *user_data)
{
  verify_data *vdata = (verify_data *)user_data;
  int status;
  unsigned int count;
  ml_tensor_type_e type;
  ml_tensor_dimension dimension;

  void *raw_data;
  size_t data_size;
  float *dataptr;

  status = ml_tensors_info_get_count (info, &count);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (count, 1);

  status = ml_tensors_info_get_tensor_type (info, 0, &type);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (type, ML_TENSOR_TYPE_FLOAT32);

  status = ml_tensors_info_get_tensor_dimension (info, 0, dimension);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (dimension[0], 1);

  status = ml_tensors_data_get_tensor_data (data, 0, &raw_data, &data_size);
  EXPECT_EQ (status, ML_ERROR_NONE);
  dataptr = (float *)raw_data;

  if (vdata->negative) {
    EXPECT_FALSE (dataptr[0] == vdata->golden[vdata->cursor][0]
                  || dataptr[0] == vdata->golden[vdata->cursor + 1][0]);
  } else {
    EXPECT_TRUE (dataptr[0] == vdata->golden[vdata->cursor][0]
                 || dataptr[0] == vdata->golden[vdata->cursor + 1][0]);
  }

  if (dataptr[0] == vdata->golden[vdata->cursor + 1][0])
    vdata->cursor += 1;

  vdata->checked += 1;
}

/**
 * @brief Test pipeline creation and sink
 */
TEST (tizensensor_as_source, virtual_sensor_flow_03)
{
  gchar *pipeline;
  sensor_event_s value;
  sensor_h sensor;
  int status = 0;
  ml_pipeline_h handle;
  ml_pipeline_sink_h s_handle;
  ml_pipeline_state_e state;

  status = sensor_get_default_sensor (SENSOR_LIGHT, &sensor);
  EXPECT_EQ (status, 0);

  value.accuracy = 1;
  value.timestamp = 0U;
  value.value_count = 1;
  value.values[0] = 0.01;
  EXPECT_EQ (dummy_publish (sensor, value), 0);

  data.checked = 0;
  data.dim0 = 1;
  data.type = ML_TENSOR_TYPE_FLOAT32;
  data.cursor = 0;
  data.num_data = 3;
  data.golden[0][0] = 0.01;
  data.golden[1][0] = 1.01;
  data.golden[2][0] = 3.31;
  data.negative = 0;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "tensor_src_tizensensor type=SENSOR_LIGHT sequence=-1 num-buffers=30 framerate=60/1 ! tensor_sink name=getv");
  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (handle, "getv", callback_nns, &data, &s_handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (s_handle != NULL);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  wait_for_start (handle, state, status);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PLAYING);

  g_usleep (10000); /* Let a frame or more flow */
  value.values[0] = 1.01;
  EXPECT_EQ (dummy_publish (sensor, value), 0);

  EXPECT_TRUE (wait_pipeline_process_buffers (&data.checked, 2, TEST_TIME_OUT_TIZEN_SENSOR_MS));

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PAUSED);

  status = ml_pipeline_sink_unregister (s_handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test pipeline creation and sink
 */
TEST (tizensensor_as_source, virtual_sensor_flow_04)
{
  gchar *pipeline;
  sensor_event_s value;
  sensor_h sensor;
  sensor_h *sensor_list;
  int status = 0;
  int count;
  ml_pipeline_h handle;
  ml_pipeline_sink_h s_handle;
  ml_pipeline_state_e state;

  sensor_get_sensor_list (SENSOR_LIGHT, &sensor_list, &count);
  EXPECT_EQ (count, 3);
  sensor = sensor_list[2];
  g_free (sensor_list);

  value.accuracy = 1;
  value.timestamp = 0U;
  value.value_count = 1;
  value.values[0] = 0.01;
  EXPECT_EQ (dummy_publish (sensor, value), 0);

  data.checked = 0;
  data.dim0 = 1;
  data.type = ML_TENSOR_TYPE_FLOAT32;
  data.cursor = 0;
  data.num_data = 3;
  data.golden[0][0] = 0.01;
  data.golden[1][0] = 1.01;
  data.golden[2][0] = 3.31;
  data.negative = 0;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "tensor_src_tizensensor type=SENSOR_LIGHT sequence=2 num-buffers=50 framerate=100/1 ! tensor_sink name=getv");
  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (handle, "getv", callback_nns, &data, &s_handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (s_handle != NULL);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  count = 0;
  wait_for_start (handle, state, status);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PLAYING);

  g_usleep (10000); /* Let a frame or more flow */
  value.values[0] = 1.01;
  EXPECT_EQ (dummy_publish (sensor, value), 0);

  EXPECT_TRUE (wait_pipeline_process_buffers (&data.checked, 2, TEST_TIME_OUT_TIZEN_SENSOR_MS));

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PAUSED);

  status = ml_pipeline_sink_unregister (s_handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test pipeline creation and sink (negative)
 */
TEST (tizensensor_as_source, virtual_sensor_flow_05_n)
{
  gchar *pipeline;
  sensor_event_s value;
  sensor_h sensor;
  sensor_h *sensor_list;
  int status = 0;
  int count;
  ml_pipeline_h handle;
  ml_pipeline_sink_h s_handle;
  ml_pipeline_state_e state;

  sensor_get_sensor_list (SENSOR_LIGHT, &sensor_list, &count);
  EXPECT_EQ (count, 3);
  sensor = sensor_list[2];

  value.accuracy = 1;
  value.timestamp = 0U;
  value.value_count = 1;
  value.values[0] = 0.00;
  EXPECT_EQ (dummy_publish (sensor_list[0], value), 0);
  EXPECT_EQ (dummy_publish (sensor_list[1], value), 0);
  EXPECT_EQ (dummy_publish (sensor_list[2], value), 0);
  g_free (sensor_list);

  value.accuracy = 1;
  value.timestamp = 0U;
  value.value_count = 1;
  value.values[0] = 0.01;
  EXPECT_EQ (dummy_publish (sensor, value), 0);

  data.checked = 0;
  data.dim0 = 1;
  data.type = ML_TENSOR_TYPE_FLOAT32;
  data.cursor = 0;
  data.num_data = 3;
  data.golden[0][0] = 0.01;
  data.golden[1][0] = 1.01;
  data.golden[2][0] = 3.31;
  data.negative = 1;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf (
      "tensor_src_tizensensor type=SENSOR_LIGHT sequence=1 num-buffers=50 framerate=100/1 ! tensor_sink name=getv");
  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_sink_register (handle, "getv", callback_nns, &data, &s_handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_TRUE (s_handle != NULL);

  status = ml_pipeline_start (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */
  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  /* At this moment, it can be READY, PAUSED, or PLAYING */
  EXPECT_NE (state, ML_PIPELINE_STATE_UNKNOWN);
  EXPECT_NE (state, ML_PIPELINE_STATE_NULL);

  wait_for_start (handle, state, status);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PLAYING);

  g_usleep (10000); /* Let a frame or more flow */
  value.values[0] = 1.01;
  EXPECT_EQ (dummy_publish (sensor, value), 0);

  EXPECT_TRUE (wait_pipeline_process_buffers (&data.checked, 2, TEST_TIME_OUT_TIZEN_SENSOR_MS));

  status = ml_pipeline_stop (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_usleep (10000); /* 10ms. Wait a bit. */

  status = ml_pipeline_get_state (handle, &state);
  EXPECT_EQ (status, ML_ERROR_NONE);
  EXPECT_EQ (state, ML_PIPELINE_STATE_PAUSED);

  status = ml_pipeline_sink_unregister (s_handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_NONE);

  g_free (pipeline);
}

/**
 * @brief Test pipeline creation with not supported sensor (negative)
 */
TEST (tizensensor_as_source, virtual_sensor_create_06_n)
{
  gchar *pipeline;
  int status = 0;
  ml_pipeline_h handle;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("tensor_src_tizensensor type=SENSOR_HRM_LED_GREEN ! tensor_sink");
  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_STREAMS_PIPE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (pipeline);
}

/**
 * @brief Test pipeline creation with invalid sensor (negative)
 */
TEST (tizensensor_as_source, virtual_sensor_create_07_n)
{
  gchar *pipeline;
  int status = 0;
  ml_pipeline_h handle;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("tensor_src_tizensensor type=invalid_sensor ! tensor_sink");
  status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
  EXPECT_EQ (status, ML_ERROR_STREAMS_PIPE);

  status = ml_pipeline_destroy (handle);
  EXPECT_EQ (status, ML_ERROR_INVALID_PARAMETER);

  g_free (pipeline);
}

/**
 * @brief Test for tizen sensor get property
 */
TEST (tizensensor_as_source, get_property_1)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("tensor_src_tizensensor name=srcx ! fakesink");
  gstpipe = gst_parse_launch (pipeline, &err);
  if (gstpipe) {
    gboolean silent;
    guint sequence, freq_n, freq_p, mode;
    GEnumValue sensor_type;
    GstElement *sensor_handle;

    sensor_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "srcx");
    EXPECT_NE (sensor_handle, nullptr);

    g_object_set (sensor_handle, "silent", TRUE, NULL);
    g_object_get (sensor_handle, "silent", &silent, NULL);
    EXPECT_TRUE (silent);

    g_object_set (sensor_handle, "type", SENSOR_LIGHT, NULL);
    g_object_get (sensor_handle, "type", &sensor_type, NULL);
    EXPECT_EQ (sensor_type.value, SENSOR_LIGHT);

    g_object_set (sensor_handle, "sequence", 0, NULL);
    g_object_get (sensor_handle, "sequence", &sequence, NULL);
    EXPECT_EQ (sequence, 0);

    g_object_set (sensor_handle, "mode", 0, NULL);
    g_object_get (sensor_handle, "mode", &mode, NULL);
    EXPECT_EQ (mode, 0);

    g_object_set (sensor_handle, "framerate", 10, 1, NULL);
    g_object_get (sensor_handle, "framerate", &freq_n, &freq_p, NULL);
    EXPECT_EQ (freq_n, 10);
    EXPECT_EQ (freq_p, 1);

    gst_object_unref (sensor_handle);
    gst_object_unref (gstpipe);
  } else {
    status = -1;
    g_printerr ("GST PARSE LAUNCH FAILED: [%s], %s\n", pipeline,
        (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_free (pipeline);
}

/**
 * @brief Test for tizen sensor get property
 */
TEST (tizensensor_as_source, get_property_2_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("tensor_src_tizensensor name=srcx ! fakesink");
  gstpipe = gst_parse_launch (pipeline, &err);
  if (gstpipe) {
    GstElement *sensor_handle;
    gchar *str = NULL;

    sensor_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "srcx");
    EXPECT_NE (sensor_handle, nullptr);

    g_object_get (sensor_handle, "invalid_prop", &str, NULL);
    /* getting unknown property, str should be null */
    EXPECT_TRUE (str == NULL);

    status = 0;
    gst_object_unref (sensor_handle);
    gst_object_unref (gstpipe);
  } else {
    status = -1;
    g_printerr ("GST PARSE LAUNCH FAILED: [%s], %s\n", pipeline,
        (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_free (pipeline);
}

/**
 * @brief Test for tizen sensor set and get property
 */
TEST (tizensensor_as_source, get_property_3_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("tensor_src_tizensensor name=srcx ! fakesink");
  gstpipe = gst_parse_launch (pipeline, &err);
  if (gstpipe) {
    guint freq_n, freq_d, mode;
    GEnumValue sensor_type;
    GstElement *sensor_handle;

    sensor_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "srcx");
    EXPECT_NE (sensor_handle, nullptr);

    g_object_set (sensor_handle, "type", SENSOR_HRM_LED_GREEN, NULL);
    g_object_get (sensor_handle, "type", &sensor_type, NULL);
    EXPECT_EQ (sensor_type.value, SENSOR_HRM_LED_GREEN);

    g_object_set (sensor_handle, "type", SENSOR_LIGHT, NULL);
    g_object_get (sensor_handle, "type", &sensor_type, NULL);
    EXPECT_EQ (sensor_type.value, SENSOR_LIGHT);

    g_object_set (sensor_handle, "mode", 0, NULL);
    g_object_get (sensor_handle, "mode", &mode, NULL);
    EXPECT_EQ (mode, 0);

    g_object_set (sensor_handle, "mode", 1, NULL);
    g_object_get (sensor_handle, "mode", &mode, NULL);
    EXPECT_NE (mode, 1);

    g_object_set (sensor_handle, "framerate", 0, 1, NULL);
    g_object_get (sensor_handle, "framerate", &freq_n, &freq_d, NULL);
    EXPECT_EQ (freq_n, 0);
    EXPECT_EQ (freq_d, 1);

    g_object_set (sensor_handle, "framerate", 30, 1, NULL);
    g_object_get (sensor_handle, "framerate", &freq_n, &freq_d, NULL);
    EXPECT_EQ (freq_n, 30);
    EXPECT_EQ (freq_d, 1);

    g_object_set (sensor_handle, "framerate", -1, -1, NULL);
    g_object_get (sensor_handle, "framerate", &freq_n, &freq_d, NULL);
    EXPECT_NE (freq_n, -1);
    EXPECT_NE (freq_d, -1);

    gst_object_unref (sensor_handle);
    gst_object_unref (gstpipe);
  } else {
    status = -1;
    g_printerr ("GST PARSE LAUNCH FAILED: [%s], %s\n", pipeline,
        (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, ML_ERROR_NONE);
  g_free (pipeline);
}

/**
 * @brief Main GTest
 */
int
main (int argc, char **argv)
{
  int result = -1;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  /* ignore tizen feature status while running the testcases */
  set_feature_state (SUPPORTED);

  gst_init (&argc, &argv);
  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  set_feature_state (NOT_CHECKED_YET);

  return result;
}
