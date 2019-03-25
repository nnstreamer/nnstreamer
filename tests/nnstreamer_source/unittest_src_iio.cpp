/**
 * @file	unittest_src_iio.cpp
 * @date	22 March 2019
 * @brief	Unit test for tensor_src_iio
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs.
 */
#include <gtest/gtest.h>
#include <gst/gst.h>
#include <gst/check/gstharness.h>

#define ELEMENT_NAME "tensor_src_iio"

const guint DEFAULT_BUFFER_CAPACITY = 1;
const gulong DEFAULT_FREQUENCY = 0;
const gboolean DEFAULT_SILENT = TRUE;
const gboolean DEFAULT_MERGE_CHANNELS = FALSE;

const gchar *mode[] = { "one-shot", "continuous" };
const gchar *channels[] = { "auto", "all" };

/**
 * @brief tests properties of tensor source IIO
 */
TEST (test_tensor_src_iio, properties)
{
  const gchar default_name[] = "tensorsrciio0";
  const gchar device[] = "test-device-1";
  const gchar trigger[] = "test-trigger-1";

  GstHarness *hrnss = NULL;
  GstElement *src_iio = NULL;
  gchar *name;
  gboolean silent;
  guint buffer_capacity;
  gulong frequency;
  gboolean merge_channels;

  gboolean ret_silent;
  gchar *ret_mode;
  gchar *ret_device;
  gchar *ret_trigger;
  gchar *ret_channels;
  guint ret_buffer_capacity;
  gulong ret_frequency;
  gboolean ret_merge_channels;

  /** setup */
  hrnss = gst_harness_new_empty ();
  ASSERT_TRUE (hrnss != NULL);
  gst_harness_add_parse (hrnss, ELEMENT_NAME);
  src_iio = gst_harness_find_element (hrnss, ELEMENT_NAME);
  ASSERT_TRUE (src_iio != NULL);

  /** check the default name */
  name = gst_element_get_name (src_iio);
  ASSERT_TRUE (name != NULL);
  EXPECT_STREQ (default_name, name);
  g_free (name);

  /** silent mode test */
  g_object_get (src_iio, "silent", &ret_silent, NULL);
  EXPECT_EQ (ret_silent, DEFAULT_SILENT);
  silent = FALSE;
  g_object_set (src_iio, "silent", silent, NULL);
  g_object_get (src_iio, "silent", &ret_silent, NULL);
  EXPECT_EQ (ret_silent, silent);

  /** operating mode test */
  g_object_get (src_iio, "mode", &ret_mode, NULL);
  EXPECT_STREQ (ret_mode, mode[1]);
  g_object_set (src_iio, "mode", mode[0], NULL);
  g_object_get (src_iio, "mode", &ret_mode, NULL);
  EXPECT_STREQ (ret_mode, mode[0]);
  g_object_set (src_iio, "mode", mode[1], NULL);
  g_object_get (src_iio, "mode", &ret_mode, NULL);
  EXPECT_STREQ (ret_mode, mode[1]);

  /** setting device test */
  g_object_set (src_iio, "device", device, NULL);
  g_object_get (src_iio, "device", &ret_device, NULL);
  EXPECT_STREQ (ret_device, device);

  /** setting trigger test */
  g_object_set (src_iio, "trigger", trigger, NULL);
  g_object_get (src_iio, "trigger", &ret_trigger, NULL);
  EXPECT_STREQ (ret_trigger, trigger);

  /** setting channels test */
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, channels[0]);
  g_object_set (src_iio, "channels", channels[1], NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, channels[1]);
  g_object_set (src_iio, "channels", channels[0], NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, channels[0]);

  /** buffer_capacity test */
  g_object_get (src_iio, "buffer-capacity", &ret_buffer_capacity, NULL);
  EXPECT_EQ (ret_buffer_capacity, DEFAULT_BUFFER_CAPACITY);
  buffer_capacity = 100;
  g_object_set (src_iio, "buffer-capacity", buffer_capacity, NULL);
  g_object_get (src_iio, "buffer-capacity", &ret_buffer_capacity, NULL);
  EXPECT_EQ (ret_buffer_capacity, buffer_capacity);

  /** frequency test */
  g_object_get (src_iio, "frequency", &ret_frequency, NULL);
  EXPECT_EQ (ret_frequency, DEFAULT_FREQUENCY);
  frequency = 100;
  g_object_set (src_iio, "frequency", frequency, NULL);
  g_object_get (src_iio, "frequency", &ret_frequency, NULL);
  EXPECT_EQ (ret_frequency, frequency);

  /** merge_channels mode test */
  g_object_get (src_iio, "merge-channels-data", &ret_merge_channels, NULL);
  EXPECT_EQ (ret_merge_channels, DEFAULT_MERGE_CHANNELS);
  merge_channels = TRUE;
  g_object_set (src_iio, "merge-channels-data", merge_channels, NULL);
  g_object_get (src_iio, "merge-channels-data", &ret_merge_channels, NULL);
  EXPECT_EQ (ret_merge_channels, merge_channels);

  /* teardown */
  gst_harness_teardown (hrnss);
}

/**
 * @brief Main function for unit test.
 */
int
main (int argc, char **argv)
{
  testing::InitGoogleTest (&argc, argv);

  gst_init (&argc, &argv);

  return RUN_ALL_TESTS ();
}
