/**
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file        unittest_decoder_boundingbox.cc
 * @date        04 Sep 2026
 * @brief       Unit test for the bounding_boxes mode of tensor_decoder
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <string.h>
#include <unittest_util.h>

#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api_util.h>

#define OV_DESC_SIZE (7U)
#define OV_DETECTION_MAX (200U)
#define OV_TENSOR_ELEMENTS (OV_DESC_SIZE * OV_DETECTION_MAX)
#define OV_TENSOR_SIZE (OV_TENSOR_ELEMENTS * sizeof (float))

#define MODEL_WIDTH (640U)
#define MODEL_HEIGHT (480U)
#define OUT_WIDTH (64U)
#define OUT_HEIGHT (48U)
#define OUT_PIXELS (OUT_WIDTH * OUT_HEIGHT)

#define LABEL_HEIGHT (12U)
#define LABEL_PIXELS (OUT_WIDTH * LABEL_HEIGHT)
#define LABEL_TEXT "X\n"
/* A two-byte UTF-8 character; both bytes are negative as a signed char */
#define LABEL_TEXT_NON_ASCII "\xc3\xa9\n"
/* What the sprite table holds for every byte that is not printable ASCII */
#define LABEL_TEXT_FALLBACK "**\n"

#define BOX_PIXEL (0xFF0000FFU)

/**
 * @brief Fill the first descriptor of an ov-person-detection output tensor.
 * @details The model emits coordinates normalized to its own input size, which
 *          the decoder scales to the output frame. The second descriptor gets a
 *          negative image id, which terminates the list of detections.
 */
static void
setDetection (float *tensor, float x_min, float y_min, float x_max, float y_max)
{
  tensor[0] = 0.0f;
  tensor[1] = 0.0f;
  tensor[2] = 1.0f;
  tensor[3] = x_min;
  tensor[4] = y_min;
  tensor[5] = x_max;
  tensor[6] = y_max;
  tensor[OV_DESC_SIZE] = -1.0f;
}

/**
 * @brief Write the given floats to a new temp file.
 * @return the file name, to be released with removeTempFile ()
 */
static gchar *
writeTensorFile (const float *data, guint elements)
{
  gchar *name = getTempFilename ();

  if (name != NULL
      && !g_file_set_contents (name, (const gchar *) data, elements * sizeof (float), NULL))
    removeTempFile (&name);

  return name;
}

/**
 * @brief Run the pipeline to EOS and read back the frame the decoder wrote.
 */
static gboolean
runAndReadFrame (const gchar *pipeline_str, const gchar *out_file, uint32_t *frame, guint pixels)
{
  GstElement *pipeline = gst_parse_launch (pipeline_str, NULL);
  gchar *content = NULL;
  gsize len = 0;
  gboolean ret = FALSE;

  if (pipeline == NULL)
    return FALSE;

  if (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT) == 0) {
    GstBus *bus = gst_element_get_bus (pipeline);
    GstMessage *msg = gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
        (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));

    if (msg != NULL) {
      ret = (GST_MESSAGE_TYPE (msg) == GST_MESSAGE_EOS);
      gst_message_unref (msg);
    }
    gst_object_unref (bus);
  }

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (pipeline);

  if (ret) {
    ret = g_file_get_contents (out_file, &content, &len, NULL)
          && len == pixels * sizeof (uint32_t);
    if (ret)
      memcpy (frame, content, len);
    g_free (content);
  }

  return ret;
}

/**
 * @brief Decode the given tensor with the bounding_boxes decoder.
 * @param[in] tensor OV_TENSOR_ELEMENTS floats of ov-person-detection output
 * @param[out] frame OUT_PIXELS RGBA pixels drawn by the decoder
 */
static gboolean
decodeBoundingBoxes (const float *tensor, uint32_t *frame)
{
  gchar *in_file = writeTensorFile (tensor, OV_TENSOR_ELEMENTS);
  gchar *out_file = getTempFilename ();
  gchar *pipeline_str;
  gboolean ret = FALSE;

  if (in_file != NULL && out_file != NULL) {
    pipeline_str = g_strdup_printf (
        "filesrc location=%s blocksize=%u ! application/octet-stream ! "
        "tensor_converter input-dim=%u:%u:1:1 input-type=float32 ! "
        "tensor_decoder mode=bounding_boxes option1=ov-person-detection "
        "option4=%u:%u option5=%u:%u ! "
        "filesink location=%s buffer-mode=unbuffered sync=false async=false",
        in_file, (guint) OV_TENSOR_SIZE, OV_DESC_SIZE, OV_DETECTION_MAX,
        OUT_WIDTH, OUT_HEIGHT, MODEL_WIDTH, MODEL_HEIGHT, out_file);

    ret = runAndReadFrame (pipeline_str, out_file, frame, OUT_PIXELS);
    g_free (pipeline_str);
  }

  removeTempFile (&in_file);
  removeTempFile (&out_file);

  return ret;
}

/**
 * @brief Decode one labelled detection into a frame shorter than a character cell.
 * @details mobilenet-ssd-postprocess is the simplest mode that draws labels: the box,
 *          its class and its score come straight from four small tensors. The box sits
 *          low enough that the label is written from row 0, and the output frame is
 *          shorter than the 13 rows of a character, which is what the sprite loop has
 *          to be clipped against.
 * @param[in] label_text the content of the label file, one label per line
 * @param[out] frame LABEL_PIXELS RGBA pixels drawn by the decoder
 */
static gboolean
decodeLabelledBox (const gchar *label_text, uint32_t *frame)
{
  const float num[] = { 1.0f };
  const float classes[] = { 0.0f };
  const float scores[] = { 0.9f };
  const float boxes[] = { 0.5f, 0.25f, 0.75f, 0.5f }; /* y_min, x_min, y_max, x_max */
  gchar *num_file = writeTensorFile (num, 1);
  gchar *class_file = writeTensorFile (classes, 1);
  gchar *score_file = writeTensorFile (scores, 1);
  gchar *box_file = writeTensorFile (boxes, 4);
  gchar *label_file = getTempFilename ();
  gchar *out_file = getTempFilename ();
  gchar *pipeline_str;
  gboolean ret = FALSE;

  if (num_file != NULL && class_file != NULL && score_file != NULL
      && box_file != NULL && label_file != NULL && out_file != NULL
      && g_file_set_contents (label_file, label_text, -1, NULL)) {
    pipeline_str = g_strdup_printf (
        "tensor_mux name=mux ! tensor_decoder mode=bounding_boxes "
        "option1=mobilenet-ssd-postprocess option2=%s option4=%u:%u option5=%u:%u ! "
        "filesink location=%s buffer-mode=unbuffered sync=false async=false "
        "filesrc location=%s blocksize=4 ! application/octet-stream ! "
        "tensor_converter input-dim=1 input-type=float32 ! mux.sink_0 "
        "filesrc location=%s blocksize=4 ! application/octet-stream ! "
        "tensor_converter input-dim=1:1 input-type=float32 ! mux.sink_1 "
        "filesrc location=%s blocksize=4 ! application/octet-stream ! "
        "tensor_converter input-dim=1:1 input-type=float32 ! mux.sink_2 "
        "filesrc location=%s blocksize=16 ! application/octet-stream ! "
        "tensor_converter input-dim=4:1 input-type=float32 ! mux.sink_3",
        label_file, OUT_WIDTH, LABEL_HEIGHT, MODEL_WIDTH, MODEL_HEIGHT,
        out_file, num_file, class_file, score_file, box_file);

    ret = runAndReadFrame (pipeline_str, out_file, frame, LABEL_PIXELS);
    g_free (pipeline_str);
  }

  removeTempFile (&num_file);
  removeTempFile (&class_file);
  removeTempFile (&score_file);
  removeTempFile (&box_file);
  removeTempFile (&label_file);
  removeTempFile (&out_file);

  return ret;
}

/**
 * @brief Count the pixels the decoder has drawn.
 */
static guint
countDrawnPixels (const uint32_t *frame, guint pixels)
{
  guint i, count = 0;

  for (i = 0; i < pixels; i++) {
    if (frame[i] != 0U)
      count++;
  }

  return count;
}

/**
 * @brief A box inside the frame is drawn on its four edges.
 */
TEST (tensorDecoderBoundingBox, drawBoxInFrame)
{
  float tensor[OV_TENSOR_ELEMENTS] = { 0.0f };
  uint32_t frame[OUT_PIXELS] = { 0U };

  setDetection (tensor, 0.25f, 0.25f, 0.75f, 0.75f);
  ASSERT_TRUE (decodeBoundingBoxes (tensor, frame));

  /* The box covers x 16 to 48 and y 12 to 36 of the 64x48 output frame */
  EXPECT_EQ (frame[12 * OUT_WIDTH + 16], BOX_PIXEL);
  EXPECT_EQ (frame[12 * OUT_WIDTH + 48], BOX_PIXEL);
  EXPECT_EQ (frame[36 * OUT_WIDTH + 16], BOX_PIXEL);
  EXPECT_EQ (frame[36 * OUT_WIDTH + 48], BOX_PIXEL);
  EXPECT_EQ (frame[24 * OUT_WIDTH + 16], BOX_PIXEL);
  EXPECT_EQ (frame[24 * OUT_WIDTH + 48], BOX_PIXEL);
  EXPECT_EQ (frame[24 * OUT_WIDTH + 32], 0U);
  EXPECT_EQ (frame[11 * OUT_WIDTH + 16], 0U);
}

/**
 * @brief A box starting left of the frame is clamped to the frame.
 * @details Without clamping the negative position is promoted to a huge
 *          unsigned offset, which the vertical edge loop writes through.
 */
TEST (tensorDecoderBoundingBox, drawBoxAcrossLeftEdge)
{
  float tensor[OV_TENSOR_ELEMENTS] = { 0.0f };
  uint32_t frame[OUT_PIXELS] = { 0U };

  setDetection (tensor, -0.25f, 0.25f, 0.5f, 0.75f);
  ASSERT_TRUE (decodeBoundingBoxes (tensor, frame));

  /* x -16 is clamped to 0, the right edge stays at 32 */
  EXPECT_EQ (frame[12 * OUT_WIDTH + 0], BOX_PIXEL);
  EXPECT_EQ (frame[12 * OUT_WIDTH + 32], BOX_PIXEL);
  EXPECT_EQ (frame[36 * OUT_WIDTH + 0], BOX_PIXEL);
  EXPECT_EQ (frame[36 * OUT_WIDTH + 32], BOX_PIXEL);
  EXPECT_EQ (frame[24 * OUT_WIDTH + 0], BOX_PIXEL);
  EXPECT_EQ (frame[24 * OUT_WIDTH + 32], BOX_PIXEL);
  EXPECT_EQ (frame[24 * OUT_WIDTH + 33], 0U);
}

/**
 * @brief A box that ends below the frame is clamped to the last row.
 */
TEST (tensorDecoderBoundingBox, drawBoxAcrossBottomEdge)
{
  float tensor[OV_TENSOR_ELEMENTS] = { 0.0f };
  uint32_t frame[OUT_PIXELS] = { 0U };

  setDetection (tensor, 0.25f, 0.5f, 0.75f, 1.5f);
  ASSERT_TRUE (decodeBoundingBoxes (tensor, frame));

  EXPECT_EQ (frame[24 * OUT_WIDTH + 16], BOX_PIXEL);
  EXPECT_EQ (frame[(OUT_HEIGHT - 1) * OUT_WIDTH + 16], BOX_PIXEL);
  EXPECT_EQ (frame[(OUT_HEIGHT - 1) * OUT_WIDTH + 48], BOX_PIXEL);
  EXPECT_EQ (frame[30 * OUT_WIDTH + 16], BOX_PIXEL);
}

/**
 * @brief A box entirely right of the frame draws nothing.
 * @details The unclamped left edge used to wrap into the following rows and
 *          paint a vertical line where no box is.
 */
TEST (tensorDecoderBoundingBox, skipBoxRightOfFrame)
{
  float tensor[OV_TENSOR_ELEMENTS] = { 0.0f };
  uint32_t frame[OUT_PIXELS] = { 0U };

  setDetection (tensor, 1.5f, 0.25f, 1.75f, 0.75f);
  ASSERT_TRUE (decodeBoundingBoxes (tensor, frame));

  EXPECT_EQ (countDrawnPixels (frame, OUT_PIXELS), 0U);
}

/**
 * @brief A box entirely left of the frame draws nothing.
 */
TEST (tensorDecoderBoundingBox, skipBoxLeftOfFrame)
{
  float tensor[OV_TENSOR_ELEMENTS] = { 0.0f };
  uint32_t frame[OUT_PIXELS] = { 0U };

  setDetection (tensor, -1.5f, 0.25f, -1.25f, 0.75f);
  ASSERT_TRUE (decodeBoundingBoxes (tensor, frame));

  EXPECT_EQ (countDrawnPixels (frame, OUT_PIXELS), 0U);
}

/**
 * @brief A box entirely above the frame draws nothing.
 */
TEST (tensorDecoderBoundingBox, skipBoxAboveFrame)
{
  float tensor[OV_TENSOR_ELEMENTS] = { 0.0f };
  uint32_t frame[OUT_PIXELS] = { 0U };

  setDetection (tensor, 0.25f, -1.5f, 0.75f, -1.25f);
  ASSERT_TRUE (decodeBoundingBoxes (tensor, frame));

  EXPECT_EQ (countDrawnPixels (frame, OUT_PIXELS), 0U);
}

/**
 * @brief A box entirely below the frame draws nothing.
 */
TEST (tensorDecoderBoundingBox, skipBoxBelowFrame)
{
  float tensor[OV_TENSOR_ELEMENTS] = { 0.0f };
  uint32_t frame[OUT_PIXELS] = { 0U };

  setDetection (tensor, 0.25f, 1.2f, 0.75f, 1.5f);
  ASSERT_TRUE (decodeBoundingBoxes (tensor, frame));

  EXPECT_EQ (countDrawnPixels (frame, OUT_PIXELS), 0U);
}

/**
 * @brief A box with a negative size draws nothing.
 */
TEST (tensorDecoderBoundingBox, skipInvertedBox_n)
{
  float tensor[OV_TENSOR_ELEMENTS] = { 0.0f };
  uint32_t frame[OUT_PIXELS] = { 0U };

  setDetection (tensor, 0.75f, 0.75f, 0.25f, 0.25f);
  ASSERT_TRUE (decodeBoundingBoxes (tensor, frame));

  EXPECT_EQ (countDrawnPixels (frame, OUT_PIXELS), 0U);
}

/**
 * @brief The label of a box is clipped by an output frame shorter than a character.
 * @details The character rows that do not fit are written past the end of the frame, so
 *          the overrun itself is only observable to a memory checker; this case pins the
 *          label being drawn and gives the sprite path coverage the memcheck runs reach.
 */
TEST (tensorDecoderBoundingBox, drawLabelInShortFrame)
{
  uint32_t frame[LABEL_PIXELS] = { 0U };
  guint i, label_pixels = 0;

  ASSERT_TRUE (decodeLabelledBox (LABEL_TEXT, frame));

  /* The box covers x 16 to 32 and y 6 to 9; its left columns are behind the label */
  EXPECT_EQ (frame[6 * OUT_WIDTH + 32], BOX_PIXEL);
  EXPECT_EQ (frame[9 * OUT_WIDTH + 32], BOX_PIXEL);

  /* The label is written from row 0, above the box */
  for (i = 0; i < 6 * OUT_WIDTH; i++) {
    if (frame[i] != 0U)
      label_pixels++;
  }
  EXPECT_GT (label_pixels, 0U);
}

/**
 * @brief A label outside ASCII is drawn as the fallback glyph.
 * @details The sprite table has an entry for every byte value, but a label byte
 *          above 0x7f is negative as a signed char and used to index that table
 *          several gigabytes past its end.
 */
TEST (tensorDecoderBoundingBox, drawNonAsciiLabel)
{
  uint32_t frame[LABEL_PIXELS] = { 0U };
  uint32_t fallback[LABEL_PIXELS] = { 0U };

  ASSERT_TRUE (decodeLabelledBox (LABEL_TEXT_NON_ASCII, frame));
  ASSERT_TRUE (decodeLabelledBox (LABEL_TEXT_FALLBACK, fallback));

  EXPECT_GT (countDrawnPixels (fallback, LABEL_PIXELS), 0U);
  EXPECT_EQ (memcmp (frame, fallback, sizeof (frame)), 0);
}

/**
 * @brief Swapping the mode of a configured decoder is refused, not divided by.
 * @details option1 takes the box properties from a table shared by the whole
 *          process and does not renegotiate, so a decoder that was configured
 *          with one mode's model input size reaches decode () holding another
 *          mode's, which may never have been given one. The mode swapped to
 *          here is one no other case in this binary configures, so its size is
 *          still the zero it was initialised with.
 */
TEST (tensorDecoderBoundingBox, swapModeOfConfiguredDecoder_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  GstTensorsConfig config;
  GstTensorMemory input;
  GstBuffer *outbuf;
  void *pdata = NULL;

  ASSERT_TRUE (decoder != NULL);
  ASSERT_TRUE (decoder->init (&pdata));

  EXPECT_TRUE (decoder->setOption (&pdata, 0, "ov-person-detection"));
  EXPECT_TRUE (decoder->setOption (&pdata, 3, "64:48"));
  EXPECT_TRUE (decoder->setOption (&pdata, 4, "640:480"));
  EXPECT_TRUE (decoder->setOption (&pdata, 0, "mobilenet-ssd"));

  gst_tensors_config_init (&config);
  memset (&input, 0, sizeof (input));
  outbuf = gst_buffer_new ();

  EXPECT_EQ (decoder->decode (&pdata, &config, &input, outbuf), GST_FLOW_ERROR);

  gst_buffer_unref (outbuf);
  gst_tensors_config_free (&config);
  decoder->exit (&pdata);
}

/**
 * @brief Build a tensors config the ov-person-detection mode accepts.
 */
static void
setOvDetectionConfig (GstTensorsConfig *config)
{
  gst_tensors_config_init (config);
  config->info.num_tensors = 1;
  config->info.info[0].type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("7:200:1:1", config->info.info[0].dimension);
  config->rate_n = 0;
  config->rate_d = 1;
}

/**
 * @brief The label path is refused while no decoding mode has been chosen.
 * @details Every option but the mode is handed to the box properties of the
 *          mode, which option1 has not looked up yet. The label file has to
 *          hold a label, because an empty one is refused before the box
 *          properties are reached and would pass for the wrong reason.
 */
TEST (tensorDecoderBoundingBox, setLabelPathBeforeMode_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  gchar *label_file = getTempFilename ();
  void *pdata = NULL;

  ASSERT_TRUE (decoder != NULL);
  ASSERT_TRUE (label_file != NULL);
  ASSERT_TRUE (g_file_set_contents (label_file, "person\n", -1, NULL));
  ASSERT_TRUE (decoder->init (&pdata));

  EXPECT_FALSE (decoder->setOption (&pdata, 1, label_file));

  decoder->exit (&pdata);
  removeTempFile (&label_file);
}

/**
 * @brief The per-mode option is refused while no decoding mode has been chosen.
 */
TEST (tensorDecoderBoundingBox, setOptionInternalBeforeMode_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  void *pdata = NULL;

  ASSERT_TRUE (decoder != NULL);
  ASSERT_TRUE (decoder->init (&pdata));

  EXPECT_FALSE (decoder->setOption (&pdata, 2, "0:0.25:0.45"));

  decoder->exit (&pdata);
}

/**
 * @brief The model input size is refused while no decoding mode has been chosen.
 */
TEST (tensorDecoderBoundingBox, setInputModelSizeBeforeMode_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  void *pdata = NULL;

  ASSERT_TRUE (decoder != NULL);
  ASSERT_TRUE (decoder->init (&pdata));

  EXPECT_FALSE (decoder->setOption (&pdata, 4, "300:300"));

  decoder->exit (&pdata);
}

/**
 * @brief A decoder without a mode describes no output caps.
 * @details This is the path a pipeline that never gives option1 takes, and the
 *          caps query runs before any option can still arrive.
 */
TEST (tensorDecoderBoundingBox, getOutCapsBeforeMode_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  GstTensorsConfig config;
  void *pdata = NULL;

  ASSERT_TRUE (decoder != NULL);
  ASSERT_TRUE (decoder->init (&pdata));

  setOvDetectionConfig (&config);
  EXPECT_TRUE (decoder->getOutCaps (&pdata, &config) == NULL);

  gst_tensors_config_free (&config);
  decoder->exit (&pdata);
}

/**
 * @brief A mode that cannot be looked up leaves the configured one in place.
 * @details option1 is writable while the pipeline runs, so a mode name that
 *          matches no box properties must not replace the working ones with
 *          the nothing the lookup returned.
 */
TEST (tensorDecoderBoundingBox, keepModeOnUnknownMode_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  GstTensorsConfig config;
  GstCaps *caps;
  void *pdata = NULL;

  ASSERT_TRUE (decoder != NULL);
  ASSERT_TRUE (decoder->init (&pdata));

  EXPECT_TRUE (decoder->setOption (&pdata, 0, "ov-person-detection"));
  EXPECT_TRUE (decoder->setOption (&pdata, 3, "64:48"));
  EXPECT_TRUE (decoder->setOption (&pdata, 4, "640:480"));

  EXPECT_FALSE (decoder->setOption (&pdata, 0, "no-such-decoding-mode"));

  setOvDetectionConfig (&config);
  caps = decoder->getOutCaps (&pdata, &config);
  EXPECT_TRUE (caps != NULL);
  if (caps)
    gst_caps_unref (caps);

  gst_tensors_config_free (&config);
  decoder->exit (&pdata);
}

/**
 * @brief Push one ov-person-detection tensor and pull the frame drawn for it.
 * @param[out] frame OUT_PIXELS RGBA pixels drawn by the decoder
 */
static gboolean
pushAndPullFrame (GstHarness *h, const float *tensor, uint32_t *frame)
{
  GstBuffer *in = gst_buffer_new_allocate (NULL, OV_TENSOR_SIZE, NULL);
  GstBuffer *out;
  GstMapInfo map;
  gboolean ret = FALSE;

  gst_buffer_fill (in, 0, tensor, OV_TENSOR_SIZE);
  if (gst_harness_push (h, in) != GST_FLOW_OK)
    return FALSE;

  out = gst_harness_try_pull (h);
  if (out == NULL)
    return FALSE;

  if (gst_buffer_map (out, &map, GST_MAP_READ)) {
    if (map.size == OUT_PIXELS * sizeof (uint32_t)) {
      memcpy (frame, map.data, map.size);
      ret = TRUE;
    }
    gst_buffer_unmap (out, &map);
  }
  gst_buffer_unref (out);

  return ret;
}

/**
 * @brief A renegotiated stream is decoded with the options it was given.
 * @details A new tensor config re-initialises the sub-plugin, and the fresh
 *          BoundingBox holds no decoding mode until the options are given
 *          again. A framerate change is the least a config can change by, so
 *          the frame drawn after it has to match the one drawn before.
 */
TEST (tensorDecoderBoundingBox, renegotiateKeepsOptions)
{
  float tensor[OV_TENSOR_ELEMENTS] = { 0.0f };
  uint32_t before[OUT_PIXELS] = { 0U };
  uint32_t after[OUT_PIXELS] = { 0U };
  GstTensorsConfig config;
  GstElement *dec;
  GstHarness *h;
  gchar *option4, *option5;

  dec = gst_element_factory_make ("tensor_decoder", NULL);
  ASSERT_TRUE (dec != NULL);
  gst_object_ref_sink (dec);

  option4 = g_strdup_printf ("%u:%u", OUT_WIDTH, OUT_HEIGHT);
  option5 = g_strdup_printf ("%u:%u", MODEL_WIDTH, MODEL_HEIGHT);
  g_object_set (dec, "mode", "bounding_boxes", "option1", "ov-person-detection",
      "option4", option4, "option5", option5, NULL);
  g_free (option4);
  g_free (option5);

  h = gst_harness_new_with_element (dec, "sink", "src");
  gst_object_unref (dec);
  ASSERT_TRUE (h != NULL);

  setOvDetectionConfig (&config);
  setDetection (tensor, 0.25f, 0.25f, 0.75f, 0.75f);

  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  EXPECT_TRUE (pushAndPullFrame (h, tensor, before));
  EXPECT_GT (countDrawnPixels (before, OUT_PIXELS), 0U);

  config.rate_n = 30;
  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  EXPECT_TRUE (pushAndPullFrame (h, tensor, after));
  EXPECT_EQ (memcmp (before, after, sizeof (before)), 0);

  gst_tensors_config_free (&config);
  gst_harness_teardown (h);
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
    g_warning ("catch testing::internal::ClassUniqueToAlwaysTrue");
  }

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch testing::internal::GoogleTestFailureException");
  }

  return result;
}
