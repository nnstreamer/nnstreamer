/**
 * @file        unittest_demux.cc
 * @date        18 Sep 2026
 * @brief       Unit test for tensor_demux
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <tensor_common.h>
#include <unittest_util.h>
#include "../gst/nnstreamer/elements/gsttensor_demux.h"

/**
 * @brief Number of tensors a stream carries to reach GstTensorsInfo::extra.
 */
#define EXTRA_NUM_TENSORS ((guint) (NNS_TENSOR_MEMORY_MAX + 4))

/**
 * @brief Start a standalone tensor_demux and hand out its sink pad.
 * @param demux the element to start, filled in by this function
 * @return the sink pad of the element, which the caller should unref
 */
static GstPad *
_start_tensor_demux (GstElement **demux)
{
  GstElement *element;
  GstPad *sinkpad;

  element = gst_element_factory_make ("tensor_demux", NULL);
  if (element == NULL)
    return NULL;

  gst_element_set_state (element, GST_STATE_PLAYING);
  sinkpad = gst_element_get_static_pad (element, "sink");
  gst_pad_send_event (sinkpad, gst_event_new_stream_start ("tensordemux-extra"));

  *demux = element;
  return sinkpad;
}

/**
 * @brief Renegotiate a stream of more tensors than NNS_TENSOR_MEMORY_MAX, which
 *        reparses the configuration of tensor_demux.
 */
TEST (tensorDemuxExtraTensors, capsRenegotiation)
{
  GstElement *demux = NULL;
  GstPad *sinkpad;
  GstCaps *caps;

  sinkpad = _start_tensor_demux (&demux);
  ASSERT_NE (sinkpad, nullptr);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_caps (caps)));
  gst_caps_unref (caps);

  EXPECT_EQ (GST_TENSOR_DEMUX (demux)->tensors_config.info.num_tensors, EXTRA_NUM_TENSORS);
  EXPECT_NE (GST_TENSOR_DEMUX (demux)->tensors_config.info.extra, nullptr);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS + 1, EXTRA_NUM_TENSORS + 1);
  EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_caps (caps)));
  gst_caps_unref (caps);

  EXPECT_EQ (GST_TENSOR_DEMUX (demux)->tensors_config.info.num_tensors,
      EXTRA_NUM_TENSORS + 1);

  gst_element_set_state (demux, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (demux);
}

/**
 * @brief Negotiate a stream whose tensors are not all described.
 */
TEST (tensorDemuxExtraTensors, capsRenegotiation_n)
{
  GstElement *demux = NULL;
  GstPad *sinkpad;
  GstCaps *caps;

  sinkpad = _start_tensor_demux (&demux);
  ASSERT_NE (sinkpad, nullptr);

  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS);
  EXPECT_TRUE (gst_pad_send_event (sinkpad, gst_event_new_caps (caps)));
  gst_caps_unref (caps);

  /* the element does not propagate a parse failure, so check its config */
  caps = caps_with_tensors (EXTRA_NUM_TENSORS, EXTRA_NUM_TENSORS - 1);
  gst_pad_send_event (sinkpad, gst_event_new_caps (caps));
  gst_caps_unref (caps);

  EXPECT_EQ (GST_TENSOR_DEMUX (demux)->tensors_config.info.num_tensors, 0U);

  gst_element_set_state (demux, GST_STATE_NULL);
  gst_object_unref (sinkpad);
  gst_object_unref (demux);
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

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
