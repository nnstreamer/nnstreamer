/**
 * @file        unittest_converter_timestamp.cc
 * @date        02 Sep 2026
 * @brief       Unit test for the timestamp handling of tensor_converter
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/check/gstharness.h>
#include <gst/check/gsttestclock.h>
#include <gst/gst.h>
#include <unittest_util.h>

#define FRAME_SIZE (8U)
#define OCTET_CAPS "application/octet-stream"
#define OCTET_FPS_CAPS "application/octet-stream,framerate=(fraction)5/1"
#define VIDEO_FPS_CAPS \
  "video/x-raw,format=GRAY8,width=8,height=1,framerate=(fraction)10/1"
#define CONVERTER_LAUNCH "tensor_converter input-dim=8 input-type=uint8"
#define FAKESRC_LAUNCH "fakesrc num-buffers=5 sizetype=fixed sizemax=8"
#define ONE_HOUR (3600 * GST_SECOND)
#define TEST_TIMEOUT_MS (10000U)
#define IDENTITY_PADDING (40U)

/**
 * @brief Create a harness around a bare tensor_converter with the given caps.
 */
static GstHarness *
harness_new_full (const gchar *caps, gboolean set_timestamp, guint frames_per_tensor)
{
  GstHarness *h = gst_harness_new ("tensor_converter");

  if (g_str_has_prefix (caps, OCTET_CAPS))
    g_object_set (h->element, "input-dim", "8", "input-type", "uint8", NULL);
  g_object_set (h->element, "set-timestamp", set_timestamp, "frames-per-tensor",
      frames_per_tensor, NULL);
  gst_harness_set_src_caps_str (h, caps);
  return h;
}

#define harness_new(caps) harness_new_full (caps, TRUE, 1)

/**
 * @brief Set the state of the harnessed element synchronously.
 */
static void
harness_set_state (GstHarness *h, GstState state)
{
  EXPECT_EQ (gst_element_set_state (h->element, state), GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (GST_STATE (h->element), state);
}

/**
 * @brief Push one frame through the harness and return the output buffer.
 */
static GstBuffer *
harness_push_frame (GstHarness *h, GstClockTime pts, GstClockTime duration)
{
  GstBuffer *in = gst_harness_create_buffer (h, FRAME_SIZE);

  GST_BUFFER_PTS (in) = pts;
  GST_BUFFER_DURATION (in) = duration;
  EXPECT_EQ (gst_harness_push (h, in), GST_FLOW_OK);

  return gst_harness_try_pull (h);
}

/**
 * @brief Push an untimestamped frame and return the timestamp assigned to it.
 */
static GstClockTime
harness_push_get_pts (GstHarness *h)
{
  GstBuffer *out = harness_push_frame (h, GST_CLOCK_TIME_NONE, GST_CLOCK_TIME_NONE);
  GstClockTime pts;

  if (out == NULL) {
    ADD_FAILURE () << "no output buffer";
    return GST_CLOCK_TIME_NONE;
  }

  pts = GST_BUFFER_PTS (out);
  gst_buffer_unref (out);
  return pts;
}

/**
 * @brief Positive: while PLAYING, an untimestamped frame gets the running time.
 */
TEST (tensorConverterTimestamp, playingRunningTime_p)
{
  GstHarness *h = harness_new (OCTET_CAPS);
  GstBuffer *out;

  gst_harness_set_time (h, 10 * GST_SECOND);
  gst_element_set_base_time (h->element, 10 * GST_SECOND);

  out = harness_push_frame (h, GST_CLOCK_TIME_NONE, GST_CLOCK_TIME_NONE);
  ASSERT_TRUE (out != NULL);
  EXPECT_EQ (GST_BUFFER_PTS (out), 0U);
  EXPECT_FALSE (GST_BUFFER_DURATION_IS_VALID (out));
  gst_buffer_unref (out);

  gst_harness_set_time (h, 10 * GST_SECOND + 500 * GST_MSECOND);
  EXPECT_EQ (harness_push_get_pts (h), 500 * GST_MSECOND);

  gst_harness_set_time (h, 10 * GST_SECOND + 700 * GST_MSECOND);
  EXPECT_EQ (harness_push_get_pts (h), 700 * GST_MSECOND);

  gst_harness_teardown (h);
}

/**
 * @brief Positive: a clock behind the base time yields a zero timestamp.
 */
TEST (tensorConverterTimestamp, playingClockBehindBaseTime_p)
{
  GstHarness *h = harness_new (OCTET_CAPS);

  gst_harness_set_time (h, 10 * GST_SECOND);
  gst_element_set_base_time (h->element, 20 * GST_SECOND);

  EXPECT_EQ (harness_push_get_pts (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Regression (#4898): a clock without a base time must not leak the
 * absolute clock value into the timestamp.
 */
TEST (tensorConverterTimestamp, pausedBeforeFirstPlaying_p)
{
  GstHarness *h = harness_new (OCTET_CAPS);

  harness_set_state (h, GST_STATE_PAUSED);
  ASSERT_EQ (gst_element_get_base_time (h->element), 0U);
  gst_harness_set_time (h, ONE_HOUR);

  EXPECT_EQ (harness_push_get_pts (h), 0U);
  EXPECT_EQ (harness_push_get_pts (h), 0U);

  harness_set_state (h, GST_STATE_PLAYING);
  gst_element_set_base_time (h->element, ONE_HOUR);
  EXPECT_EQ (harness_push_get_pts (h), 0U);

  gst_harness_set_time (h, ONE_HOUR + 50 * GST_MSECOND);
  EXPECT_EQ (harness_push_get_pts (h), 50 * GST_MSECOND);

  gst_harness_teardown (h);
}

/**
 * @brief Positive: frames pushed while PAUSED keep the last timestamp and the
 * running time continues after resuming.
 */
TEST (tensorConverterTimestamp, pausedAfterPlayingKeepsLast_p)
{
  GstHarness *h = harness_new (OCTET_CAPS);

  gst_element_set_base_time (h->element, 10 * GST_SECOND);
  gst_harness_set_time (h, 10 * GST_SECOND + 500 * GST_MSECOND);
  EXPECT_EQ (harness_push_get_pts (h), 500 * GST_MSECOND);

  harness_set_state (h, GST_STATE_PAUSED);
  gst_harness_set_time (h, ONE_HOUR);
  EXPECT_EQ (harness_push_get_pts (h), 500 * GST_MSECOND);
  EXPECT_EQ (harness_push_get_pts (h), 500 * GST_MSECOND);

  harness_set_state (h, GST_STATE_PLAYING);
  gst_element_set_base_time (h->element, ONE_HOUR - 500 * GST_MSECOND);
  EXPECT_EQ (harness_push_get_pts (h), 500 * GST_MSECOND);

  gst_harness_set_time (h, ONE_HOUR + 100 * GST_MSECOND);
  EXPECT_EQ (harness_push_get_pts (h), 600 * GST_MSECOND);

  gst_harness_teardown (h);
}

/**
 * @brief Positive: without a previous timestamp, the fallback is the segment start.
 */
TEST (tensorConverterTimestamp, pausedUsesSegmentStart_p)
{
  GstHarness *h = harness_new (OCTET_CAPS);
  GstSegment segment;

  gst_segment_init (&segment, GST_FORMAT_TIME);
  segment.start = segment.time = 2 * GST_SECOND;
  EXPECT_TRUE (gst_harness_push_event (h, gst_event_new_segment (&segment)));

  harness_set_state (h, GST_STATE_PAUSED);
  gst_harness_set_time (h, ONE_HOUR);
  EXPECT_EQ (harness_push_get_pts (h), 2 * GST_SECOND);
  EXPECT_EQ (harness_push_get_pts (h), 2 * GST_SECOND);

  harness_set_state (h, GST_STATE_PLAYING);
  gst_element_set_base_time (h->element, ONE_HOUR);
  EXPECT_LT (harness_push_get_pts (h), GST_SECOND);

  gst_harness_teardown (h);
}

/**
 * @brief Positive: a frame that already carries timestamps is left untouched.
 */
TEST (tensorConverterTimestamp, validInputTimestampUntouched_p)
{
  GstHarness *h = harness_new (OCTET_CAPS);
  GstBuffer *out;

  gst_harness_set_time (h, ONE_HOUR);

  out = harness_push_frame (h, 42 * GST_MSECOND, 7 * GST_MSECOND);
  ASSERT_TRUE (out != NULL);
  EXPECT_EQ (GST_BUFFER_PTS (out), 42 * GST_MSECOND);
  EXPECT_EQ (GST_BUFFER_DURATION (out), 7 * GST_MSECOND);
  gst_buffer_unref (out);

  harness_set_state (h, GST_STATE_PAUSED);
  out = harness_push_frame (h, 43 * GST_MSECOND, GST_CLOCK_TIME_NONE);
  ASSERT_TRUE (out != NULL);
  EXPECT_EQ (GST_BUFFER_PTS (out), 43 * GST_MSECOND);
  EXPECT_FALSE (GST_BUFFER_DURATION_IS_VALID (out));
  gst_buffer_unref (out);

  EXPECT_EQ (harness_push_get_pts (h), 43 * GST_MSECOND);

  gst_harness_teardown (h);
}

/**
 * @brief Negative: with set-timestamp=false nothing is stamped in any state.
 */
TEST (tensorConverterTimestamp, setTimestampFalse_n)
{
  GstHarness *h = harness_new_full (OCTET_CAPS, FALSE, 1);
  GstBuffer *out;

  gst_harness_set_time (h, ONE_HOUR);
  gst_element_set_base_time (h->element, 10 * GST_SECOND);
  EXPECT_FALSE (GST_CLOCK_TIME_IS_VALID (harness_push_get_pts (h)));

  harness_set_state (h, GST_STATE_PAUSED);
  EXPECT_FALSE (GST_CLOCK_TIME_IS_VALID (harness_push_get_pts (h)));

  out = harness_push_frame (h, 42 * GST_MSECOND, GST_CLOCK_TIME_NONE);
  ASSERT_TRUE (out != NULL);
  EXPECT_EQ (GST_BUFFER_PTS (out), 42 * GST_MSECOND);
  gst_buffer_unref (out);

  gst_harness_teardown (h);
}

/**
 * @brief Positive: without a clock the timestamp falls back to the history.
 */
TEST (tensorConverterTimestamp, playingWithoutClock_p)
{
  GstHarness *h = harness_new (OCTET_CAPS);

  EXPECT_TRUE (gst_element_set_clock (h->element, NULL));
  gst_element_set_base_time (h->element, 10 * GST_SECOND);

  EXPECT_EQ (harness_push_get_pts (h), 0U);
  EXPECT_EQ (harness_push_get_pts (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Positive: with a framerate the clock is never consulted.
 */
TEST (tensorConverterTimestamp, framerateIgnoresClock_p)
{
  GstHarness *h = harness_new (VIDEO_FPS_CAPS);
  GstBuffer *out;

  gst_harness_set_time (h, ONE_HOUR);

  out = harness_push_frame (h, GST_CLOCK_TIME_NONE, GST_CLOCK_TIME_NONE);
  ASSERT_TRUE (out != NULL);
  EXPECT_EQ (GST_BUFFER_PTS (out), 0U);
  EXPECT_EQ (GST_BUFFER_DURATION (out), 100 * GST_MSECOND);
  gst_buffer_unref (out);

  EXPECT_EQ (harness_push_get_pts (h), 100 * GST_MSECOND);

  harness_set_state (h, GST_STATE_PAUSED);
  gst_harness_set_time (h, 2 * ONE_HOUR);
  EXPECT_EQ (harness_push_get_pts (h), 200 * GST_MSECOND);
  EXPECT_EQ (harness_push_get_pts (h), 300 * GST_MSECOND);

  gst_harness_teardown (h);
}

/**
 * @brief Positive: an octet stream with a framerate takes the framerate branch.
 */
TEST (tensorConverterTimestamp, octetFramerate_p)
{
  GstHarness *h = harness_new (OCTET_FPS_CAPS);
  GstBuffer *out;

  gst_harness_set_time (h, ONE_HOUR);

  out = harness_push_frame (h, GST_CLOCK_TIME_NONE, GST_CLOCK_TIME_NONE);
  ASSERT_TRUE (out != NULL);
  EXPECT_EQ (GST_BUFFER_PTS (out), 0U);
  EXPECT_EQ (GST_BUFFER_DURATION (out), 200 * GST_MSECOND);
  gst_buffer_unref (out);

  EXPECT_EQ (harness_push_get_pts (h), 200 * GST_MSECOND);

  gst_harness_teardown (h);
}

/**
 * @brief Positive: a flush clears the timestamp history.
 */
TEST (tensorConverterTimestamp, flushResetsHistory_p)
{
  GstHarness *h = harness_new (OCTET_CAPS);
  GstSegment segment;

  gst_element_set_base_time (h->element, 10 * GST_SECOND);
  gst_harness_set_time (h, 10 * GST_SECOND + 500 * GST_MSECOND);
  EXPECT_EQ (harness_push_get_pts (h), 500 * GST_MSECOND);

  EXPECT_TRUE (gst_harness_push_event (h, gst_event_new_flush_start ()));
  EXPECT_TRUE (gst_harness_push_event (h, gst_event_new_flush_stop (TRUE)));
  gst_segment_init (&segment, GST_FORMAT_TIME);
  EXPECT_TRUE (gst_harness_push_event (h, gst_event_new_segment (&segment)));

  harness_set_state (h, GST_STATE_PAUSED);
  gst_harness_set_time (h, ONE_HOUR);
  EXPECT_EQ (harness_push_get_pts (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Positive: a BYTES segment is replaced by a TIME segment starting at 0.
 */
TEST (tensorConverterTimestamp, bytesSegmentConverted_p)
{
  GstHarness *h = harness_new (OCTET_CAPS);
  GstSegment segment;
  GstEvent *event;
  const GstSegment *out_segment = NULL;
  guint num_segments = 0;

  gst_segment_init (&segment, GST_FORMAT_BYTES);
  EXPECT_TRUE (gst_harness_push_event (h, gst_event_new_segment (&segment)));

  harness_set_state (h, GST_STATE_PAUSED);
  gst_harness_set_time (h, ONE_HOUR);
  EXPECT_EQ (harness_push_get_pts (h), 0U);

  while ((event = gst_harness_try_pull_event (h)) != NULL) {
    if (GST_EVENT_TYPE (event) == GST_EVENT_SEGMENT) {
      gst_event_parse_segment (event, &out_segment);
      EXPECT_EQ (out_segment->format, GST_FORMAT_TIME);
      num_segments++;
    }
    gst_event_unref (event);
  }
  EXPECT_EQ (num_segments, 2U);

  harness_set_state (h, GST_STATE_PLAYING);
  gst_element_set_base_time (h->element, ONE_HOUR);
  gst_harness_set_time (h, ONE_HOUR + 10 * GST_MSECOND);
  EXPECT_EQ (harness_push_get_pts (h), 10 * GST_MSECOND);

  gst_harness_teardown (h);
}

/**
 * @brief Positive: aggregated frames carry the timestamp of their first frame.
 */
TEST (tensorConverterTimestamp, framesPerTensorAggregation_p)
{
  GstHarness *h = harness_new_full (OCTET_CAPS, TRUE, 2);
  GstBuffer *out;

  gst_element_set_base_time (h->element, 10 * GST_SECOND);

  gst_harness_set_time (h, 10 * GST_SECOND + 100 * GST_MSECOND);
  out = harness_push_frame (h, GST_CLOCK_TIME_NONE, GST_CLOCK_TIME_NONE);
  EXPECT_TRUE (out == NULL);

  gst_harness_set_time (h, 10 * GST_SECOND + 200 * GST_MSECOND);
  out = harness_push_frame (h, GST_CLOCK_TIME_NONE, GST_CLOCK_TIME_NONE);
  ASSERT_TRUE (out != NULL);
  EXPECT_EQ (gst_buffer_get_size (out), 2 * FRAME_SIZE);
  EXPECT_EQ (GST_BUFFER_PTS (out), 100 * GST_MSECOND);
  gst_buffer_unref (out);

  gst_harness_teardown (h);
}

/**
 * @brief Positive: going through READY clears the timestamp history.
 */
TEST (tensorConverterTimestamp, readyResetsHistory_p)
{
  GstHarness *h = harness_new (OCTET_CAPS);

  gst_element_set_base_time (h->element, 10 * GST_SECOND);
  gst_harness_set_time (h, 10 * GST_SECOND + 500 * GST_MSECOND);
  EXPECT_EQ (harness_push_get_pts (h), 500 * GST_MSECOND);

  harness_set_state (h, GST_STATE_READY);
  harness_set_state (h, GST_STATE_PAUSED);
  EXPECT_TRUE (gst_harness_push_event (h, gst_event_new_stream_start ("ready")));
  gst_harness_set_src_caps_str (h, OCTET_CAPS);

  gst_harness_set_time (h, ONE_HOUR);
  EXPECT_EQ (harness_push_get_pts (h), 0U);

  gst_harness_teardown (h);
}

/**
 * @brief Wait until the pipeline reaches the given state.
 */
static gboolean
wait_for_state (GstElement *pipeline, GstState state)
{
  GstState current = GST_STATE_VOID_PENDING;
  GstStateChangeReturn ret;

  ret = gst_element_get_state (pipeline, &current, NULL, TEST_TIMEOUT_MS * GST_MSECOND);
  return (ret == GST_STATE_CHANGE_SUCCESS && current == state);
}

/**
 * @brief Push an untimestamped frame into the appsrc of a pipeline.
 */
static void
pipeline_push_frame (GstElement *appsrc)
{
  GstBuffer *buffer = gst_buffer_new_allocate (NULL, FRAME_SIZE, NULL);

  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc), buffer), GST_FLOW_OK);
}

/**
 * @brief Pull a sample from the appsink and return its timestamp.
 */
static GstClockTime
pipeline_pull_pts (GstElement *appsink, gboolean preroll)
{
  GstSample *sample;
  GstClockTime pts;

  if (preroll)
    sample = gst_app_sink_try_pull_preroll (
        GST_APP_SINK (appsink), TEST_TIMEOUT_MS * GST_MSECOND);
  else
    sample = gst_app_sink_try_pull_sample (
        GST_APP_SINK (appsink), TEST_TIMEOUT_MS * GST_MSECOND);

  if (sample == NULL) {
    ADD_FAILURE () << "no sample";
    return GST_CLOCK_TIME_NONE;
  }

  pts = GST_BUFFER_PTS (gst_sample_get_buffer (sample));
  gst_sample_unref (sample);
  return pts;
}

/**
 * @brief Build the appsrc-to-appsink pipeline used by the pipeline tests.
 */
static GstElement *
pipeline_new (GstElement **appsrc, GstElement **appsink, GstElement **converter)
{
  GstElement *pipeline
      = gst_parse_launch ("appsrc name=src format=bytes ! " OCTET_CAPS " ! " CONVERTER_LAUNCH
                          " name=conv ! queue ! appsink name=sink sync=false",
          NULL);

  if (pipeline == NULL)
    return NULL;

  *appsrc = gst_bin_get_by_name (GST_BIN (pipeline), "src");
  *appsink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");
  *converter = gst_bin_get_by_name (GST_BIN (pipeline), "conv");
  return pipeline;
}

/**
 * @brief Positive: in a pipeline driven by a test clock, timestamps are exact
 * running times and do not advance while PAUSED.
 */
TEST (tensorConverterTimestampPipeline, testClockRunningTime_p)
{
  GstElement *pipeline, *appsrc, *appsink, *converter;
  GstClock *clock = gst_test_clock_new_with_start_time (ONE_HOUR);

  pipeline = pipeline_new (&appsrc, &appsink, &converter);
  ASSERT_TRUE (pipeline != NULL);
  gst_pipeline_use_clock (GST_PIPELINE (pipeline), clock);

  EXPECT_EQ (gst_element_set_state (pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_ASYNC);
  pipeline_push_frame (appsrc);
  ASSERT_TRUE (wait_for_state (pipeline, GST_STATE_PLAYING));
  EXPECT_EQ (pipeline_pull_pts (appsink, FALSE), 0U);
  EXPECT_EQ (gst_element_get_base_time (converter), ONE_HOUR);

  gst_test_clock_advance_time (GST_TEST_CLOCK (clock), 300 * GST_MSECOND);
  pipeline_push_frame (appsrc);
  EXPECT_EQ (pipeline_pull_pts (appsink, FALSE), 300 * GST_MSECOND);

  gst_element_set_state (pipeline, GST_STATE_PAUSED);
  gst_test_clock_advance_time (GST_TEST_CLOCK (clock), 5 * GST_SECOND);
  pipeline_push_frame (appsrc);
  EXPECT_EQ (pipeline_pull_pts (appsink, TRUE), 300 * GST_MSECOND);
  ASSERT_TRUE (wait_for_state (pipeline, GST_STATE_PAUSED));

  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  ASSERT_TRUE (wait_for_state (pipeline, GST_STATE_PLAYING));
  EXPECT_EQ (pipeline_pull_pts (appsink, FALSE), 300 * GST_MSECOND);

  gst_test_clock_advance_time (GST_TEST_CLOCK (clock), 100 * GST_MSECOND);
  pipeline_push_frame (appsrc);
  EXPECT_EQ (pipeline_pull_pts (appsink, FALSE), 400 * GST_MSECOND);

  EXPECT_EQ (gst_element_set_state (pipeline, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);
  gst_object_unref (appsrc);
  gst_object_unref (appsink);
  gst_object_unref (converter);
  gst_object_unref (pipeline);
  gst_object_unref (clock);
}

/**
 * @brief Timestamp recorder attached to the converter source pad.
 */
typedef struct {
  gint count;
  GstClockTime pts;
} probe_data_s;

/**
 * @brief Record the timestamp of every buffer leaving the converter.
 */
static GstPadProbeReturn
probe_record_pts (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  probe_data_s *data = (probe_data_s *) user_data;

  data->pts = GST_BUFFER_PTS (GST_PAD_PROBE_INFO_BUFFER (info));
  g_atomic_int_inc (&data->count);
  return GST_PAD_PROBE_OK;
}

/**
 * @brief Wait until the probe has seen the expected number of buffers.
 */
static gboolean
wait_for_probe (probe_data_s *data, gint expected)
{
  guint waited = 0;

  while (g_atomic_int_get (&data->count) < expected && waited < TEST_TIMEOUT_MS) {
    g_usleep (10000);
    waited += 10;
  }
  return g_atomic_int_get (&data->count) >= expected;
}

/**
 * @brief Regression (#4898): emulate the state-change window in a real
 * pipeline. The converter already holds a clock but no base time yet.
 */
TEST (tensorConverterTimestampPipeline, clockWithoutBaseTime_p)
{
  GstElement *pipeline, *appsrc, *appsink, *converter;
  GstClock *clock = gst_system_clock_obtain ();
  GstPad *srcpad;
  probe_data_s data = { 0, GST_CLOCK_TIME_NONE };

  pipeline = pipeline_new (&appsrc, &appsink, &converter);
  ASSERT_TRUE (pipeline != NULL);

  srcpad = gst_element_get_static_pad (converter, "src");
  ASSERT_TRUE (srcpad != NULL);
  gst_pad_add_probe (srcpad, GST_PAD_PROBE_TYPE_BUFFER, probe_record_pts, &data, NULL);

  EXPECT_EQ (gst_element_set_state (pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_ASYNC);
  pipeline_push_frame (appsrc);
  ASSERT_TRUE (wait_for_state (pipeline, GST_STATE_PAUSED));
  EXPECT_EQ (pipeline_pull_pts (appsink, TRUE), 0U);
  ASSERT_TRUE (wait_for_probe (&data, 1));

  ASSERT_EQ (gst_element_get_base_time (converter), 0U);
  EXPECT_TRUE (gst_element_set_clock (converter, clock));
  EXPECT_GT (gst_clock_get_time (clock), GST_SECOND);

  pipeline_push_frame (appsrc);
  ASSERT_TRUE (wait_for_probe (&data, 2));
  EXPECT_EQ (data.pts, 0U);

  EXPECT_EQ (gst_element_set_state (pipeline, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);
  gst_object_unref (srcpad);
  gst_object_unref (appsrc);
  gst_object_unref (appsink);
  gst_object_unref (converter);
  gst_object_unref (pipeline);
  gst_object_unref (clock);
}

/**
 * @brief Run a gst-launch style pipeline to EOS and return the final message type.
 */
static GstMessageType
run_pipeline_to_end (const gchar *description)
{
  GstElement *pipeline = gst_parse_launch (description, NULL);
  GstBus *bus;
  GstMessage *message;
  GstMessageType type = GST_MESSAGE_UNKNOWN;

  if (pipeline == NULL)
    return GST_MESSAGE_UNKNOWN;

  bus = gst_element_get_bus (pipeline);
  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  message = gst_bus_timed_pop_filtered (bus, TEST_TIMEOUT_MS * GST_MSECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  if (message != NULL) {
    type = GST_MESSAGE_TYPE (message);
    gst_message_unref (message);
  }

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (bus);
  gst_object_unref (pipeline);
  return type;
}

/**
 * @brief Build a converter pipeline padded with identity elements before a
 * synchronizing sink, widening the state-change window of #4898.
 */
static gchar *
padded_sync_pipeline (const gchar *source, const gchar *converter, const gchar *sink)
{
  GString *desc = g_string_new (source);
  guint i;

  g_string_append_printf (desc, " ! %s ! ", converter);
  for (i = 0; i < IDENTITY_PADDING; i++)
    g_string_append (desc, "identity ! ");
  g_string_append (desc, sink);

  return g_string_free (desc, FALSE);
}

/**
 * @brief Positive: untimestamped octet frames reach a synchronizing sink.
 */
TEST (tensorConverterTimestampPipeline, syncSinkCompletes_p)
{
  gchar *desc = padded_sync_pipeline (
      FAKESRC_LAUNCH " ! " OCTET_CAPS, CONVERTER_LAUNCH, "fakesink sync=true");
  guint i;

  for (i = 0; i < 3; i++)
    EXPECT_EQ (run_pipeline_to_end (desc), GST_MESSAGE_EOS);

  g_free (desc);
}

/**
 * @brief Positive: the same holds with a queue thread before the sink.
 */
TEST (tensorConverterTimestampPipeline, syncSinkWithQueueCompletes_p)
{
  gchar *desc = padded_sync_pipeline (FAKESRC_LAUNCH " ! " OCTET_CAPS,
      CONVERTER_LAUNCH, "queue ! fakesink sync=true");

  EXPECT_EQ (run_pipeline_to_end (desc), GST_MESSAGE_EOS);

  g_free (desc);
}

/**
 * @brief Positive: with set-timestamp=false the sink does not wait either.
 */
TEST (tensorConverterTimestampPipeline, syncSinkNoTimestampCompletes_p)
{
  gchar *desc = padded_sync_pipeline (FAKESRC_LAUNCH " ! " OCTET_CAPS,
      CONVERTER_LAUNCH " set-timestamp=false", "fakesink sync=true");

  EXPECT_EQ (run_pipeline_to_end (desc), GST_MESSAGE_EOS);

  g_free (desc);
}

/**
 * @brief Positive: a live video source without a framerate still completes.
 */
TEST (tensorConverterTimestampPipeline, liveVideoNoFramerateCompletes_p)
{
  gchar *desc = padded_sync_pipeline ("videotestsrc is-live=true num-buffers=5 ! "
                                      "video/x-raw,format=GRAY8,width=16,height=16,framerate=0/1",
      "tensor_converter", "fakesink sync=true");

  EXPECT_EQ (run_pipeline_to_end (desc), GST_MESSAGE_EOS);

  g_free (desc);
}

/**
 * @brief Positive: tensor_mux keeps synchronizing two untimestamped branches.
 */
TEST (tensorConverterTimestampPipeline, muxSyncSinkCompletes_p)
{
  gchar *desc = g_strdup_printf ("%s ! %s ! %s ! mux.sink_0 %s ! %s ! %s ! mux.sink_1 "
                                 "tensor_mux name=mux ! fakesink sync=true",
      FAKESRC_LAUNCH, OCTET_CAPS, CONVERTER_LAUNCH, FAKESRC_LAUNCH, OCTET_CAPS,
      CONVERTER_LAUNCH);

  EXPECT_EQ (run_pipeline_to_end (desc), GST_MESSAGE_EOS);

  g_free (desc);
}

/**
 * @brief Positive: tensor_aggregator consumes converter-stamped frames.
 */
TEST (tensorConverterTimestampPipeline, aggregatorSyncSinkCompletes_p)
{
  gchar *desc = padded_sync_pipeline (FAKESRC_LAUNCH " ! " OCTET_CAPS, CONVERTER_LAUNCH,
      "tensor_aggregator frames-in=1 frames-out=5 frames-flush=5 ! fakesink sync=true");

  EXPECT_EQ (run_pipeline_to_end (desc), GST_MESSAGE_EOS);

  g_free (desc);
}

/**
 * @brief Push a one-byte tensor with an explicit timestamp into an appsrc.
 */
static void
pipeline_push_tensor (GstElement *appsrc, guint8 value, GstClockTime pts)
{
  GstBuffer *buffer = gst_buffer_new_allocate (NULL, 1, NULL);

  gst_buffer_fill (buffer, 0, &value, 1);
  GST_BUFFER_PTS (buffer) = pts;
  EXPECT_EQ (gst_app_src_push_buffer (GST_APP_SRC (appsrc), buffer), GST_FLOW_OK);
}

/**
 * @brief Feed a basepad-synchronized muxer two equal timestamps on its base
 * pad, the pattern the converter emits while PAUSED, and check that the far
 * frame on the other pad is held back every time.
 */
static void
run_basepad_duplicate_timestamps (const gchar *muxer, guint mem_index, guint byte_index)
{
  const gchar *caps = "other/tensor,dimension=(string)1:1:1:1,type=(string)uint8,"
                      "framerate=(fraction)0/1";
  gchar *desc = g_strdup_printf ("appsrc name=src0 format=time ! %s ! mux.sink_0 "
                                 "appsrc name=src1 format=time ! %s ! mux.sink_1 "
                                 "%s name=mux sync-mode=basepad sync-option=0:50000000 ! "
                                 "appsink name=sink sync=false",
      caps, caps, muxer);
  GstElement *pipeline = gst_parse_launch (desc, NULL);
  GstElement *src0, *src1, *sink;
  GstSample *sample;
  guint received = 0;

  g_free (desc);
  ASSERT_TRUE (pipeline != NULL);
  src0 = gst_bin_get_by_name (GST_BIN (pipeline), "src0");
  src1 = gst_bin_get_by_name (GST_BIN (pipeline), "src1");
  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  pipeline_push_tensor (src0, 0, 0);
  pipeline_push_tensor (src0, 1, 0);
  pipeline_push_tensor (src0, 2, 10 * GST_MSECOND);
  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (src0)), GST_FLOW_OK);
  pipeline_push_tensor (src1, 100, 0);
  pipeline_push_tensor (src1, 101, GST_SECOND);
  EXPECT_EQ (gst_app_src_end_of_stream (GST_APP_SRC (src1)), GST_FLOW_OK);

  while ((sample = gst_app_sink_try_pull_sample (GST_APP_SINK (sink), TEST_TIMEOUT_MS * GST_MSECOND))
         != NULL) {
    GstBuffer *buffer = gst_sample_get_buffer (sample);
    GstMapInfo map;

    EXPECT_GT (gst_buffer_n_memory (buffer), mem_index);
    if (gst_buffer_map (buffer, &map, GST_MAP_READ)) {
      EXPECT_GT (map.size, byte_index);
      EXPECT_EQ (map.data[byte_index], 100U);
      gst_buffer_unmap (buffer, &map);
    } else {
      ADD_FAILURE () << "cannot map the muxed buffer";
    }
    gst_sample_unref (sample);
    received++;
  }
  EXPECT_EQ (received, 3U);
  EXPECT_TRUE (gst_app_sink_is_eos (GST_APP_SINK (sink)));

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (src0);
  gst_object_unref (src1);
  gst_object_unref (sink);
  gst_object_unref (pipeline);
}

/**
 * @brief Positive: tensor_mux basepad sync keeps its tolerance across equal
 * base-pad timestamps.
 */
TEST (tensorConverterTimestampPipeline, muxBasepadDuplicateTimestamps_p)
{
  run_basepad_duplicate_timestamps ("tensor_mux", 1, 1);
}

/**
 * @brief Positive: tensor_merge basepad sync keeps its tolerance across equal
 * base-pad timestamps.
 */
TEST (tensorConverterTimestampPipeline, mergeBasepadDuplicateTimestamps_p)
{
  run_basepad_duplicate_timestamps ("tensor_merge mode=linear option=0", 0, 1);
}

/**
 * @brief Negative: an octet stream without dimensions fails to negotiate.
 */
TEST (tensorConverterTimestampPipeline, octetWithoutDimension_n)
{
  EXPECT_EQ (run_pipeline_to_end ("fakesrc num-buffers=1 sizetype=fixed sizemax=10 ! " OCTET_CAPS
                                  " ! tensor_converter ! fakesink sync=true"),
      GST_MESSAGE_ERROR);
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
