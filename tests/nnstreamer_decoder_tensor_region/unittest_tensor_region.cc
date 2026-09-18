/**
 * @file	unittest_tensor_region.cc
 * @date	20 June 2023
 * @brief	Unit test for tensor_decoder::tensor_region. (testcases to check data conversion or buffer transfer)
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Harsh Jain <hjain24in@gmail.com>
 * @bug		No known bugs.
 */
#include <gtest/gtest.h>
#include <cmath>
#include <glib/gstdio.h>
#include <gst/check/gstcheck.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_subplugin.h>
#include <string.h>
#include <tensor_common.h>
#include <tensor_meta.h>
#include <unistd.h>
#include <unittest_util.h>


/**
 * @brief Call back function for tensor_region to parse outbuf
 *
 * @param sink The sink element
 * @param user_data User data passed to the callback function
 */
static void
new_data_cb (GstElement *sink, const gpointer user_data)
{
  GstSample *sample = nullptr;

  g_signal_emit_by_name (sink, "pull-sample", &sample);

  /** Expected values of cropping info for orange.png */
  guint32 expected_values[] = { 58U, 62U, 219U, 211U };

  if (sample != nullptr) {
    GstBuffer *outbuf = gst_sample_get_buffer (sample);
    GstMemory *mem = gst_buffer_peek_memory (outbuf, 0);

    if (mem != nullptr) {
      GstMapInfo out_info;

      if (gst_memory_map (mem, &out_info, GST_MAP_READ)) {
        GstTensorMetaInfo map;
        guint32 *data_ptr = nullptr;

        gst_tensor_meta_info_parse_header (&map, out_info.data);

        gsize hsize = gst_tensor_meta_info_get_header_size (&map);
        gsize dsize = gst_tensor_meta_info_get_data_size (&map);
        ASSERT_EQ (_NNS_UINT32, map.type);

        gsize esize = sizeof (guint32);

        ASSERT_EQ (hsize + dsize, out_info.size);
        ASSERT_EQ (0U, (dsize % (esize * 4)));

        data_ptr = (guint32 *) (out_info.data + hsize);

        for (int i = 0; i < 4; i++) {
          EXPECT_EQ (expected_values[i], data_ptr[i]);
        }

        gst_memory_unmap (mem, &out_info);
      }
    }

    gst_sample_unref (sample);
  }
}

/**
 * @brief Structure to hold information related to TensorRegion.
 */
struct TensorRegion {
  GstElement *pipeline; /**< The pipeline element */
  GstElement *app_sink; /**< The app sink element */
};


/**
 * @brief Callback function to handle pipeline messages.
 *
 * @param bus The GStreamer bus.
 * @param message The GStreamer message.
 * @param data Pointer to the TensorRegion structure.
 * @return gboolean Returns TRUE to continue receiving messages.
 */
static gboolean
on_pipeline_message (GstBus *bus, GstMessage *message, TensorRegion *data)
{
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_EOS:
      break;
    case GST_MESSAGE_ERROR:
      {
        g_print ("Received error\n");

        GError *err = NULL;
        gchar *dbg_info = NULL;

        gst_message_parse_error (message, &err, &dbg_info);
        g_printerr ("ERROR from element %s: %s\n",
            GST_OBJECT_NAME (message->src), err->message);
        g_printerr ("Debugging info: %s\n", (dbg_info) ? dbg_info : "none");
        g_error_free (err);
        g_free (dbg_info);
      }

      break;
    case GST_MESSAGE_STATE_CHANGED:
      break;
    default:
      break;
  }

  /** Return FALSE to stop receiving messages after the callback function
   * has handled the current message. */
  return G_SOURCE_CONTINUE;
}


/**
 * @brief Test for tensor_decoder::tensor_region
 */
TEST (tensorDecoder, tensorRegion)
{
  GstBus *bus;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  if (root_path == nullptr)
    root_path = "..";

  g_autofree gchar *tensor_0 = g_build_filename (root_path, "tests",
      "nnstreamer_decoder_tensor_region", "mobilenet_ssd_tensor.0", nullptr);
  g_autofree gchar *tensor_1 = g_build_filename (root_path, "tests",
      "nnstreamer_decoder_tensor_region", "mobilenet_ssd_tensor.1", nullptr);
  g_autofree gchar *labels_path = g_build_filename (
      root_path, "tests", "test_models", "labels", "labels.txt", nullptr);
  g_autofree gchar *box_priors_path = g_build_filename (root_path, "tests",
      "nnstreamer_decoder_boundingbox", "box_priors.txt", nullptr);

  ASSERT_TRUE (g_file_test (tensor_0, G_FILE_TEST_EXISTS));
  ASSERT_TRUE (g_file_test (tensor_1, G_FILE_TEST_EXISTS));
  ASSERT_TRUE (g_file_test (labels_path, G_FILE_TEST_EXISTS));
  ASSERT_TRUE (g_file_test (box_priors_path, G_FILE_TEST_EXISTS));

  /** Create the GStreamer pipeline */
  gchar *pipeline_str = g_strdup_printf ("multifilesrc name=fs1 location=%s start-index=0 stop-index=1 caps=application/octet-stream ! tensor_converter name=el1 input-dim=4:1:1917:1 input-type=float32 ! mux.sink_0 \
       multifilesrc name=fs2 location=%s start-index=0 stop-index=1 caps=application/octet-stream ! tensor_converter name=el2 input-dim=91:1917:1 input-type=float32 ! mux.sink_1 \
       tensor_mux name=mux ! other/tensors,format=static ! tensor_decoder mode=tensor_region option1=1 option2=%s option3=%s ! appsink name=sinkx ",
      tensor_0, tensor_1, labels_path, box_priors_path);

  GstElement *pipeline = gst_parse_launch (pipeline_str, nullptr);
  g_free (pipeline_str);

  GstElement *app_sink = gst_bin_get_by_name (GST_BIN (pipeline), "sinkx");

  /** Create the TensorRegion structure and assign pipeline and app_sink */
  TensorRegion data;
  data.pipeline = pipeline;
  data.app_sink = app_sink;
  bus = gst_element_get_bus (data.pipeline);
  gst_bus_add_watch (bus, (GstBusFunc) on_pipeline_message, &data);
  gst_object_unref (bus);

  /** Enable signal emission from the app_sink */
  g_object_set (app_sink, "emit-signals", TRUE, NULL);

  /** Connect the new-sample callback to the app_sink */
  g_signal_connect (app_sink, "new-sample", G_CALLBACK (new_data_cb), nullptr);

  /** Start playing the pipeline */
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  g_usleep (1000000);

  /** Free resources */
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (pipeline);

  /** Unref app_sink */
  gst_object_unref (app_sink);
}

/** The number of detections in the input tensors of the direct decoder tests */
#define DETECTIONS (2U)
/** The number of labels in the input tensors of the direct decoder tests */
#define LABELS (2U)
/** A detection score far above the default threshold */
#define DETECTED (10.0f)
/** A detection score far below the default threshold */
#define NOT_DETECTED (-10.0f)

static guint glib_critical_count;
static gchar glib_critical_msg[256];

/**
 * @brief Count the critical messages GLib itself logs.
 */
static void
count_glib_critical (const gchar *domain, GLogLevelFlags level,
    const gchar *message, gpointer user_data)
{
  glib_critical_count++;
  if (message != NULL)
    g_strlcpy (glib_critical_msg, message, sizeof (glib_critical_msg));
}

/**
 * @brief Watch the critical messages of the GLib domain, with a counter self-check.
 * @return the handler id, to be released with g_log_remove_handler ()
 */
static guint
watch_glib_critical (void)
{
  guint handler = g_log_set_handler ("GLib",
      (GLogLevelFlags) (G_LOG_LEVEL_CRITICAL | G_LOG_FLAG_FATAL | G_LOG_FLAG_RECURSION),
      count_glib_critical, NULL);

  glib_critical_count = 0;
  g_log ("GLib", G_LOG_LEVEL_CRITICAL, "tensor_region test: counter self-check");
  EXPECT_EQ (glib_critical_count, 1U);

  glib_critical_count = 0;
  glib_critical_msg[0] = '\0';
  return handler;
}

/**
 * @brief The decoder allocates and releases its private data without complaint.
 */
TEST (tensorRegionLifecycle, initAndExit)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("tensor_region");
  void *pdata = NULL;
  guint handler;

  ASSERT_TRUE (decoder != NULL);

  handler = watch_glib_critical ();
  ASSERT_TRUE (decoder->init (&pdata));
  decoder->exit (&pdata);
  g_log_remove_handler ("GLib", handler);

  EXPECT_EQ (glib_critical_count, 0U) << glib_critical_msg;
  EXPECT_TRUE (pdata == NULL);
}

/**
 * @brief Test fixture holding an instance of the tensor_region decoder.
 *
 * Every box prior is 0.5, so with the default scales a box offset of
 * { 0, 0, 0, 0 } is the centered box of half the frame size, a center offset of
 * 10 moves the center by half the frame, and a size offset of 5 * ln (k)
 * multiplies the size by k.
 */
class tensorRegionDecode : public ::testing::Test
{
  protected:
  const GstTensorDecoderDef *decoder;
  void *pdata;
  gchar *label_file;
  gchar *prior_file;
  GstTensorsConfig config;

  /**
   * @brief Initialize the decoder with a label file, box priors and input.
   */
  void SetUp () override
  {
    GstCaps *caps;
    GstTensorInfo *info;

    pdata = NULL;
    label_file = getTempFilename ();
    prior_file = getTempFilename ();
    ASSERT_TRUE (label_file != NULL);
    ASSERT_TRUE (prior_file != NULL);
    ASSERT_TRUE (g_file_set_contents (label_file, "background\nobject\n", -1, NULL));
    ASSERT_TRUE (g_file_set_contents (
        prior_file, "0.5 0.5\n0.5 0.5\n0.5 0.5\n0.5 0.5\n", -1, NULL));

    decoder = nnstreamer_decoder_find ("tensor_region");
    ASSERT_TRUE (decoder != NULL);
    ASSERT_TRUE (decoder->init (&pdata));
    ASSERT_TRUE (decoder->setOption (&pdata, 1, label_file));
    ASSERT_TRUE (decoder->setOption (&pdata, 2, prior_file));

    gst_tensors_config_init (&config);
    config.rate_n = 0;
    config.rate_d = 1;
    config.info.num_tensors = 2;

    info = gst_tensors_info_get_nth_info (&config.info, 0);
    info->type = _NNS_FLOAT32;
    info->dimension[0] = 4;
    info->dimension[1] = 1;
    info->dimension[2] = DETECTIONS;
    info->dimension[3] = 1;

    info = gst_tensors_info_get_nth_info (&config.info, 1);
    info->type = _NNS_FLOAT32;
    info->dimension[0] = LABELS;
    info->dimension[1] = DETECTIONS;
    info->dimension[2] = 1;

    caps = decoder->getOutCaps (&pdata, &config);
    ASSERT_TRUE (caps != NULL);
    gst_caps_unref (caps);
  }

  /**
   * @brief Release the decoder and the files.
   */
  void TearDown () override
  {
    if (pdata != NULL)
      decoder->exit (&pdata);
    gst_tensors_config_free (&config);
    removeTempFile (&label_file);
    removeTempFile (&prior_file);
  }

  /**
   * @brief Decode the given boxes and scores into num regions.
   * @param[in] boxes DETECTIONS boxes of 4 offsets each
   * @param[in] scores The score of the only non-background label per detection
   * @param[in] num The number of regions to request (option1)
   * @param[out] regions num regions of 4 values each (x, y, w, h)
   * @return TRUE if the decoder produced exactly num regions.
   */
  gboolean decode (const float *boxes, const float *scores, guint num, guint32 *regions)
  {
    GstTensorMemory input[2];
    float detections[LABELS * DETECTIONS];
    g_autofree gchar *num_str = g_strdup_printf ("%u", num);
    GstBuffer *outbuf;
    GstMemory *mem;
    GstMapInfo map;
    GstTensorMetaInfo meta;
    gboolean ret = FALSE;
    guint d;

    for (d = 0; d < DETECTIONS; d++) {
      detections[d * LABELS] = NOT_DETECTED;
      detections[d * LABELS + 1] = scores[d];
    }

    input[0].data = (gpointer) boxes;
    input[0].size = 4 * DETECTIONS * sizeof (float);
    input[1].data = detections;
    input[1].size = sizeof (detections);

    if (!decoder->setOption (&pdata, 0, num_str))
      return FALSE;

    outbuf = gst_buffer_new ();
    if (decoder->decode (&pdata, &config, input, outbuf) != GST_FLOW_OK
        || gst_buffer_n_memory (outbuf) != 1U)
      goto done;

    mem = gst_buffer_peek_memory (outbuf, 0);
    if (!gst_memory_map (mem, &map, GST_MAP_READ))
      goto done;

    if (gst_tensor_meta_info_parse_header (&meta, map.data)
        && gst_tensor_meta_info_get_data_size (&meta) == 4 * num * sizeof (guint32)
        && gst_tensor_meta_info_get_header_size (&meta)
                   + gst_tensor_meta_info_get_data_size (&meta)
               == map.size) {
      memcpy (regions, map.data + gst_tensor_meta_info_get_header_size (&meta),
          4 * num * sizeof (guint32));
      ret = TRUE;
    }

    gst_memory_unmap (mem, &map);
  done:
    gst_buffer_unref (outbuf);
    return ret;
  }

  /**
   * @brief Decode a single detected box into one region.
   */
  gboolean decodeOne (float y, float x, float h, float w, guint32 *region)
  {
    const float boxes[4 * DETECTIONS] = { y, x, h, w, 0, 0, 0, 0 };
    const float scores[DETECTIONS] = { DETECTED, NOT_DETECTED };

    return decode (boxes, scores, 1, region);
  }
};

/**
 * @brief Expect a region to be the given values.
 */
static void
expectRegion (const guint32 *region, guint32 x, guint32 y, guint32 w, guint32 h)
{
  EXPECT_EQ (x, region[0]);
  EXPECT_EQ (y, region[1]);
  EXPECT_EQ (w, region[2]);
  EXPECT_EQ (h, region[3]);
}

/**
 * @brief A box inside the frame is emitted as it is.
 */
TEST_F (tensorRegionDecode, regionInFrame)
{
  guint32 region[4];

  ASSERT_TRUE (decodeOne (0, 0, 0, 0, region));
  expectRegion (region, 75U, 75U, 150U, 150U);
}

/**
 * @brief A box crossing the right edge is cut at the edge.
 */
TEST_F (tensorRegionDecode, regionPastRightEdge)
{
  guint32 region[4];

  ASSERT_TRUE (decodeOne (0, 10.0f, 0, 0, region));
  expectRegion (region, 225U, 75U, 75U, 150U);
}

/**
 * @brief A box crossing the left edge keeps only its part inside the frame.
 */
TEST_F (tensorRegionDecode, regionPastLeftEdge)
{
  guint32 region[4];

  ASSERT_TRUE (decodeOne (0, -10.0f, 0, 0, region));
  expectRegion (region, 0U, 75U, 75U, 150U);
}

/**
 * @brief A box crossing the top and bottom edges is cut at both.
 */
TEST_F (tensorRegionDecode, regionPastTopAndBottomEdges)
{
  guint32 region[4];

  /** 5 * ln (4): the height becomes twice the frame height */
  ASSERT_TRUE (decodeOne (0, 0, 5.0f * log (4.0f), 0, region));
  expectRegion (region, 75U, 0U, 150U, 300U);
}

/**
 * @brief A box entirely right of the frame gives no region.
 */
TEST_F (tensorRegionDecode, regionRightOfFrame_n)
{
  guint32 region[4];

  ASSERT_TRUE (decodeOne (0, 30.0f, 0, 0, region));
  expectRegion (region, 0U, 0U, 0U, 0U);
}

/**
 * @brief A box too far right for its start to fit in an int gives no region.
 */
TEST_F (tensorRegionDecode, regionFarRightOfFrame_n)
{
  guint32 region[4];

  ASSERT_TRUE (decodeOne (0, 1.0e10f, 0, 0, region));
  expectRegion (region, 0U, 0U, 0U, 0U);
}

/**
 * @brief A box entirely left of the frame gives no region.
 */
TEST_F (tensorRegionDecode, regionLeftOfFrame_n)
{
  guint32 region[4];

  ASSERT_TRUE (decodeOne (0, -30.0f, 0, 0, region));
  expectRegion (region, 0U, 0U, 0U, 0U);
}

/**
 * @brief A box entirely below the frame gives no region.
 */
TEST_F (tensorRegionDecode, regionBelowFrame_n)
{
  guint32 region[4];

  ASSERT_TRUE (decodeOne (30.0f, 0, 0, 0, region));
  expectRegion (region, 0U, 0U, 0U, 0U);
}

/**
 * @brief A box whose size overflows to infinity gives no region.
 */
TEST_F (tensorRegionDecode, regionInfiniteSize_n)
{
  guint32 region[4];

  ASSERT_TRUE (decodeOne (0, 0, 0, 1.0e6f, region));
  expectRegion (region, 0U, 0U, 0U, 0U);
}

/**
 * @brief A box with a NaN coordinate gives no region.
 */
TEST_F (tensorRegionDecode, regionNotANumber_n)
{
  guint32 region[4];

  ASSERT_TRUE (decodeOne (0, NAN, 0, 0, region));
  expectRegion (region, 0U, 0U, 0U, 0U);
}

/**
 * @brief A box narrower than a pixel gives no region, not a zero-width one.
 */
TEST_F (tensorRegionDecode, regionBelowOnePixel_n)
{
  guint32 region[4];

  /** the width becomes 0.5 * e^-6 of the frame, less than a pixel of 300 */
  ASSERT_TRUE (decodeOne (0, 0, 0, -30.0f, region));
  expectRegion (region, 0U, 0U, 0U, 0U);
}

/**
 * @brief Without a frame size (empty option4), no box gives a region.
 */
TEST_F (tensorRegionDecode, regionEmptyFrame_n)
{
  guint32 region[4];

  ASSERT_TRUE (decoder->setOption (&pdata, 3, ""));
  ASSERT_TRUE (decodeOne (0, 0, 0, 0, region));
  expectRegion (region, 0U, 0U, 0U, 0U);
}

/**
 * @brief A frame wider than an int keeps the region within an int.
 */
TEST_F (tensorRegionDecode, regionLargeFrame)
{
  guint32 region[4];

  ASSERT_TRUE (decoder->setOption (&pdata, 3, "4000000000:300"));
  ASSERT_TRUE (decodeOne (0, 0, 0, 0, region));
  expectRegion (region, 1000000000U, 75U, (guint32) G_MAXINT - 1000000000U, 150U);
}

/**
 * @brief The option4 frame size bounds the region, not the default one.
 */
TEST_F (tensorRegionDecode, regionFrameFromOption)
{
  guint32 region[4];

  ASSERT_TRUE (decoder->setOption (&pdata, 3, "640:480"));
  ASSERT_TRUE (decodeOne (10.0f, 10.0f, 0, 0, region));
  expectRegion (region, 480U, 360U, 160U, 120U);
}

/**
 * @brief A dropped box does not take a region away from a box in the frame.
 */
TEST_F (tensorRegionDecode, droppedBoxTakesNoRegion_n)
{
  /** the first box is outside the frame and has the higher score */
  const float boxes[4 * DETECTIONS] = { 0, 30.0f, 0, 0, 0, 0, 0, 0 };
  const float scores[DETECTIONS] = { DETECTED + 1.0f, DETECTED };
  guint32 regions[8];

  ASSERT_TRUE (decode (boxes, scores, 2, regions));
  expectRegion (regions, 75U, 75U, 150U, 150U);
  expectRegion (regions + 4, 0U, 0U, 0U, 0U);
}


/**
 * @brief Main function for unit test.
 */
int
main (int argc, char **argv)
{
  int ret = -1;
  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  try {
    ret = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return ret;
}
