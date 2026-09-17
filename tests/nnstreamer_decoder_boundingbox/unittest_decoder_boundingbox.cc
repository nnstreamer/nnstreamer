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
#include <cmath>
#include <glib.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <unittest_util.h>

#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

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
  EXPECT_TRUE (decoder->setOption (&pdata, 0, "yolov10"));

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

#define BOX_OUT_WIDTH (64U)
#define BOX_OUT_HEIGHT (48U)
#define BOX_OUT_PIXELS (BOX_OUT_WIDTH * BOX_OUT_HEIGHT)

/**
 * @brief Build a float32 tensors config with the given dimensions.
 */
static void
setFloatConfig (GstTensorsConfig *config, guint num_tensors, const gchar *const *dims)
{
  guint i;

  gst_tensors_config_init (config);
  config->info.num_tensors = num_tensors;
  for (i = 0; i < num_tensors; i++) {
    config->info.info[i].type = _NNS_FLOAT32;
    gst_tensor_parse_dimension (dims[i], config->info.info[i].dimension);
  }
  config->rate_n = 0;
  config->rate_d = 1;
}

/**
 * @brief Tell whether the decoder accepts the given config.
 */
static gboolean
acceptsConfig (const GstTensorDecoderDef *decoder, void **pdata, const GstTensorsConfig *config)
{
  GstCaps *caps = decoder->getOutCaps (pdata, config);

  if (caps == NULL)
    return FALSE;

  gst_caps_unref (caps);
  return TRUE;
}

/**
 * @brief Decode the given tensors and copy out the BOX_OUT_PIXELS frame drawn for them.
 */
static gboolean
decodeFrame (const GstTensorDecoderDef *decoder, void **pdata,
    const GstTensorsConfig *config, const GstTensorMemory *input, uint32_t *frame)
{
  GstBuffer *outbuf = gst_buffer_new ();
  gboolean ret = FALSE;

  if (decoder->decode (pdata, config, input, outbuf) == GST_FLOW_OK
      && gst_buffer_get_size (outbuf) == BOX_OUT_PIXELS * sizeof (uint32_t)) {
    gst_buffer_extract (outbuf, 0, frame, BOX_OUT_PIXELS * sizeof (uint32_t));
    ret = TRUE;
  }

  gst_buffer_unref (outbuf);
  return ret;
}

/**
 * @brief Floats placed at the end of a page that is followed by an unreadable one.
 */
typedef struct {
  guint8 *base; /**< start of the two mapped pages */
  gsize page; /**< page size */
} GuardedFloats;

/**
 * @brief Copy the floats right in front of an unreadable page.
 * @details Reading one element past them faults at once, so an over-read is
 *          observable without a memory checker.
 * @return pointer to the copied floats, or NULL if the pages cannot be mapped
 */
static float *
newGuardedFloats (GuardedFloats *g, const float *src, guint elements)
{
  gsize size = elements * sizeof (float);

  g->page = (gsize) sysconf (_SC_PAGESIZE);
  g->base = (guint8 *) mmap (NULL, 2 * g->page, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (g->base == MAP_FAILED) {
    g->base = NULL;
    return NULL;
  }

  if (mprotect (g->base + g->page, g->page, PROT_NONE) != 0) {
    munmap (g->base, 2 * g->page);
    g->base = NULL;
    return NULL;
  }

  memcpy (g->base + g->page - size, src, size);
  return (float *) (g->base + g->page - size);
}

/**
 * @brief Release the pages of newGuardedFloats ().
 */
static void
freeGuardedFloats (GuardedFloats *g)
{
  if (g->base != NULL)
    munmap (g->base, 2 * g->page);
  g->base = NULL;
}

/**
 * @brief Decode one mobilenet-ssd-postprocess detection whose count tensor says @a num.
 * @details Every tensor holds exactly one detection and ends right in front of an
 *          unreadable page.
 */
static gboolean
decodeSsdPpCount (float num, uint32_t *frame)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  const gchar *const dims[] = { "1", "1:1", "1:1", "4:1" };
  const float classes[] = { 0.0f };
  const float scores[] = { 0.9f };
  const float boxes[] = { 0.25f, 0.25f, 0.75f, 0.75f };
  GuardedFloats g[4] = { { NULL, 0 }, { NULL, 0 }, { NULL, 0 }, { NULL, 0 } };
  GstTensorMemory input[4];
  GstTensorsConfig config;
  void *pdata = NULL;
  gboolean ret = FALSE;
  guint i;

  if (decoder == NULL || !decoder->init (&pdata))
    return FALSE;

  input[0].data = newGuardedFloats (&g[0], &num, 1);
  input[1].data = newGuardedFloats (&g[1], classes, 1);
  input[2].data = newGuardedFloats (&g[2], scores, 1);
  input[3].data = newGuardedFloats (&g[3], boxes, 4);
  for (i = 0; i < 4; i++)
    input[i].size = (i == 3) ? 4 * sizeof (float) : sizeof (float);

  setFloatConfig (&config, 4, dims);

  if (input[0].data && input[1].data && input[2].data && input[3].data
      && decoder->setOption (&pdata, 0, "mobilenet-ssd-postprocess")
      && decoder->setOption (&pdata, 3, "64:48") && decoder->setOption (&pdata, 4, "640:480")
      && acceptsConfig (decoder, &pdata, &config))
    ret = decodeFrame (decoder, &pdata, &config, input, frame);

  gst_tensors_config_free (&config);
  for (i = 0; i < 4; i++)
    freeGuardedFloats (&g[i]);
  decoder->exit (&pdata);

  return ret;
}

/**
 * @brief A detection count within the tensors draws the detection.
 */
TEST (tensorDecoderBoundingBox, ssdPpDetectionCountWithinTensors)
{
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };
  uint32_t none[BOX_OUT_PIXELS] = { 0U };

  ASSERT_TRUE (decodeSsdPpCount (1.0f, frame));
  EXPECT_GT (countDrawnPixels (frame, BOX_OUT_PIXELS), 0U);

  ASSERT_TRUE (decodeSsdPpCount (0.0f, none));
  EXPECT_EQ (countDrawnPixels (none, BOX_OUT_PIXELS), 0U);
}

/**
 * @brief A detection count larger than the tensors is bounded by them.
 * @details The count is a value the model writes, and the decoder used to read
 *          that many classes, scores and boxes, past the end of the tensors.
 */
TEST (tensorDecoderBoundingBox, ssdPpDetectionCountAboveTensors_n)
{
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };
  uint32_t expected[BOX_OUT_PIXELS] = { 0U };

  ASSERT_TRUE (decodeSsdPpCount (1.0f, expected));
  ASSERT_TRUE (decodeSsdPpCount (5.0f, frame));
  EXPECT_EQ (memcmp (frame, expected, sizeof (frame)), 0);
}

/**
 * @brief A negative or NaN detection count decodes no detection.
 * @details The count used to be converted to int unchecked and handed to
 *          g_array_sized_new (), which takes it as a huge unsigned size.
 */
TEST (tensorDecoderBoundingBox, ssdPpDetectionCountNegative_n)
{
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };

  ASSERT_TRUE (decodeSsdPpCount (-1.0f, frame));
  EXPECT_EQ (countDrawnPixels (frame, BOX_OUT_PIXELS), 0U);

  ASSERT_TRUE (decodeSsdPpCount (NAN, frame));
  EXPECT_EQ (countDrawnPixels (frame, BOX_OUT_PIXELS), 0U);
}

#define PALM_DETECTIONS (72U)
#define PALM_INFO_SIZE (18U)
/* One layer of stride 32: 6x6 cells of 2 anchors, as many anchors as PALM_DETECTIONS */
#define PALM_OPTION_STRIDE_32 "0.5:1:1.0:1.0:0.5:0.5:32"
/* The same anchors moved to the corner of their cells */
#define PALM_OPTION_STRIDE_32_CORNER "0.5:1:1.0:1.0:0.0:0.0:32"
/* One layer of stride 64: 3x3 cells of 2 anchors, fewer than PALM_DETECTIONS */
#define PALM_OPTION_STRIDE_64 "0.5:1:1.0:1.0:0.5:0.5:64"
/* The first anchor of cell (2, 2), which only the stride-32 anchors reach */
#define PALM_DETECTION_INDEX (28U)

/**
 * @brief mp-palm-detection tensors with one detection, as many as PALM_DETECTIONS.
 */
class PalmDetectionTensors
{
  public:
  float boxes[PALM_INFO_SIZE * PALM_DETECTIONS]; /**< box tensor */
  float scores[PALM_DETECTIONS]; /**< score tensor */
  GstTensorMemory input[2]; /**< the two tensors */
  GstTensorsConfig config; /**< their config */

  /**
   * @brief Put a 48x48 box of the model input on the anchor PALM_DETECTION_INDEX.
   */
  PalmDetectionTensors ()
  {
    const gchar *const dims[] = { "18:72:1", "1:72:1" };
    guint i;

    memset (boxes, 0, sizeof (boxes));
    for (i = 0; i < PALM_DETECTIONS; i++)
      scores[i] = -10.0f;

    boxes[PALM_DETECTION_INDEX * PALM_INFO_SIZE + 2] = 48.0f;
    boxes[PALM_DETECTION_INDEX * PALM_INFO_SIZE + 3] = 48.0f;
    scores[PALM_DETECTION_INDEX] = 10.0f;

    input[0].data = boxes;
    input[0].size = sizeof (boxes);
    input[1].data = scores;
    input[1].size = sizeof (scores);
    setFloatConfig (&config, 2, dims);
  }

  /**
   * @brief Release the config.
   */
  ~PalmDetectionTensors ()
  {
    gst_tensors_config_free (&config);
  }
};

/**
 * @brief Start a mp-palm-detection decoder with the output and model sizes set.
 */
static gboolean
initPalmDecoder (const GstTensorDecoderDef *decoder, void **pdata)
{
  return decoder != NULL && decoder->init (pdata)
         && decoder->setOption (pdata, 0, "mp-palm-detection")
         && decoder->setOption (pdata, 3, "64:48")
         && decoder->setOption (pdata, 4, "192:192");
}

/**
 * @brief A new option3 replaces the anchors instead of adding to them.
 * @details The anchors used to be appended on every option3, so a stream kept
 *          being decoded with the first anchors ever generated in the process.
 */
TEST (tensorDecoderBoundingBox, palmOptionReplacesAnchors)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  PalmDetectionTensors t;
  uint32_t first[BOX_OUT_PIXELS] = { 0U };
  uint32_t corner[BOX_OUT_PIXELS] = { 0U };
  uint32_t again[BOX_OUT_PIXELS] = { 0U };
  void *pdata = NULL;

  ASSERT_TRUE (initPalmDecoder (decoder, &pdata));

  EXPECT_TRUE (decoder->setOption (&pdata, 2, PALM_OPTION_STRIDE_32));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &t.config));
  EXPECT_TRUE (decodeFrame (decoder, &pdata, &t.config, t.input, first));
  EXPECT_GT (countDrawnPixels (first, BOX_OUT_PIXELS), 0U);

  EXPECT_TRUE (decoder->setOption (&pdata, 2, PALM_OPTION_STRIDE_32_CORNER));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &t.config));
  EXPECT_TRUE (decodeFrame (decoder, &pdata, &t.config, t.input, corner));
  EXPECT_GT (countDrawnPixels (corner, BOX_OUT_PIXELS), 0U);
  EXPECT_NE (memcmp (first, corner, sizeof (first)), 0);

  EXPECT_TRUE (decoder->setOption (&pdata, 2, PALM_OPTION_STRIDE_32));
  EXPECT_TRUE (decodeFrame (decoder, &pdata, &t.config, t.input, again));
  EXPECT_EQ (memcmp (first, again, sizeof (first)), 0);

  decoder->exit (&pdata);
}

/**
 * @brief A stream with more detections than anchors is refused.
 */
TEST (tensorDecoderBoundingBox, palmFewerAnchorsThanDetections_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  PalmDetectionTensors t;
  void *pdata = NULL;

  ASSERT_TRUE (initPalmDecoder (decoder, &pdata));

  EXPECT_TRUE (decoder->setOption (&pdata, 2, PALM_OPTION_STRIDE_64));
  EXPECT_FALSE (acceptsConfig (decoder, &pdata, &t.config));

  decoder->exit (&pdata);
}

/**
 * @brief Anchors reduced while the stream runs bound the detections decoded.
 * @details option3 is writable in PLAYING without a renegotiation, so decode ()
 *          cannot rely on the anchor count checked with the caps.
 */
TEST (tensorDecoderBoundingBox, palmAnchorsReducedWhileDecoding_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  PalmDetectionTensors t;
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };
  void *pdata = NULL;

  ASSERT_TRUE (initPalmDecoder (decoder, &pdata));

  EXPECT_TRUE (decoder->setOption (&pdata, 2, PALM_OPTION_STRIDE_32));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &t.config));

  EXPECT_TRUE (decoder->setOption (&pdata, 2, PALM_OPTION_STRIDE_64));
  EXPECT_TRUE (decodeFrame (decoder, &pdata, &t.config, t.input, frame));
  EXPECT_EQ (countDrawnPixels (frame, BOX_OUT_PIXELS), 0U);

  decoder->exit (&pdata);
}

/**
 * @brief An option3 whose layers cannot be generated is refused and changes nothing.
 * @details The number of layers indexes a stride table of 13 entries of which the
 *          option can set 7, and a stride divides the feature map size.
 */
TEST (tensorDecoderBoundingBox, palmInvalidLayers_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  PalmDetectionTensors t;
  uint32_t expected[BOX_OUT_PIXELS] = { 0U };
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };
  void *pdata = NULL;

  ASSERT_TRUE (initPalmDecoder (decoder, &pdata));

  EXPECT_TRUE (decoder->setOption (&pdata, 2, PALM_OPTION_STRIDE_32));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &t.config));
  EXPECT_TRUE (decodeFrame (decoder, &pdata, &t.config, t.input, expected));

  EXPECT_FALSE (decoder->setOption (&pdata, 2, "0.5:0"));
  EXPECT_FALSE (decoder->setOption (&pdata, 2, "0.5:-1"));
  EXPECT_FALSE (decoder->setOption (&pdata, 2, "0.5:8:1.0:1.0:0.5:0.5:32:32:32:32:32:32:32"));
  EXPECT_FALSE (decoder->setOption (&pdata, 2, "0.5:100"));
  EXPECT_FALSE (decoder->setOption (&pdata, 2, "0.5:2:1.0:1.0:0.5:0.5:32:0"));
  EXPECT_FALSE (decoder->setOption (&pdata, 2, "0.5:2:1.0:1.0:0.5:0.5:-8:32"));
  EXPECT_FALSE (decoder->setOption (&pdata, 2, "0.5:7:1.0:1.0:0.5:0.5:32:32:32:32"));

  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &t.config));
  EXPECT_TRUE (decodeFrame (decoder, &pdata, &t.config, t.input, frame));
  EXPECT_EQ (memcmp (frame, expected, sizeof (frame)), 0);

  decoder->exit (&pdata);
}

/**
 * @brief Layers without a stride in option3 take the strides already set.
 */
TEST (tensorDecoderBoundingBox, palmLayersKeepSetStrides)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  PalmDetectionTensors t;
  uint32_t expected[BOX_OUT_PIXELS] = { 0U };
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };
  void *pdata = NULL;

  ASSERT_TRUE (initPalmDecoder (decoder, &pdata));

  EXPECT_TRUE (decoder->setOption (&pdata, 2, PALM_OPTION_STRIDE_32));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &t.config));
  EXPECT_TRUE (decodeFrame (decoder, &pdata, &t.config, t.input, expected));

  EXPECT_TRUE (decoder->setOption (&pdata, 2, "0.5:1"));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &t.config));
  EXPECT_TRUE (decodeFrame (decoder, &pdata, &t.config, t.input, frame));
  EXPECT_EQ (memcmp (frame, expected, sizeof (frame)), 0);

  decoder->exit (&pdata);
}

/**
 * @brief A mp-palm-detection stream without option3 is refused.
 * @details Without option3 no anchor is generated, and decode () used to index
 *          the empty anchor array for every detection. Another decoder of this
 *          process has generated anchors, which this one must not see.
 */
TEST (tensorDecoderBoundingBox, palmWithoutOption_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  PalmDetectionTensors t;
  void *other = NULL;
  void *pdata = NULL;

  ASSERT_TRUE (initPalmDecoder (decoder, &other));
  EXPECT_TRUE (decoder->setOption (&other, 2, PALM_OPTION_STRIDE_32));
  EXPECT_TRUE (acceptsConfig (decoder, &other, &t.config));

  ASSERT_TRUE (initPalmDecoder (decoder, &pdata));
  EXPECT_FALSE (acceptsConfig (decoder, &pdata, &t.config));

  decoder->exit (&pdata);
  decoder->exit (&other);
}

#define SSD_DETECTIONS (4U)
#define SSD_LABELS (2U)

/**
 * @brief Write a box prior file of 4 rows, each of @a columns priors of @a value.
 * @return the file name, to be released with removeTempFile ()
 */
static gchar *
writeBoxPriors (guint columns, const gchar *value)
{
  GString *text = g_string_new (NULL);
  gchar *name = getTempFilename ();
  guint row, col;

  for (row = 0; row < 4; row++) {
    for (col = 0; col < columns; col++)
      g_string_append_printf (text, "%s%s", col ? " " : "", value);
    g_string_append (text, "\n");
  }

  if (name != NULL && !g_file_set_contents (name, text->str, -1, NULL))
    removeTempFile (&name);

  g_string_free (text, TRUE);
  return name;
}

/**
 * @brief mobilenet-ssd tensors with one detection of class 1 at @a index.
 */
class SsdTensors
{
  public:
  float boxes[4 * SSD_DETECTIONS]; /**< box tensor */
  float detections[SSD_LABELS * SSD_DETECTIONS]; /**< class score tensor */
  GstTensorMemory input[2]; /**< the two tensors */
  GstTensorsConfig config; /**< their config */

  /**
   * @brief Build the tensors.
   */
  SsdTensors (guint index)
  {
    const gchar *const dims[] = { "4:1:4", "2:4" };
    guint i;

    memset (boxes, 0, sizeof (boxes));
    for (i = 0; i < SSD_LABELS * SSD_DETECTIONS; i++)
      detections[i] = -5.0f;
    detections[index * SSD_LABELS + 1] = 5.0f;

    input[0].data = boxes;
    input[0].size = sizeof (boxes);
    input[1].data = detections;
    input[1].size = sizeof (detections);
    setFloatConfig (&config, 2, dims);
  }

  /**
   * @brief Release the config.
   */
  ~SsdTensors ()
  {
    gst_tensors_config_free (&config);
  }
};

/**
 * @brief Start a mobilenet-ssd decoder with labels, output and model sizes set.
 */
static gboolean
initSsdDecoder (const GstTensorDecoderDef *decoder, void **pdata, const gchar *label_file)
{
  return decoder != NULL && label_file != NULL && decoder->init (pdata)
         && decoder->setOption (pdata, 0, "mobilenet-ssd")
         && decoder->setOption (pdata, 1, label_file)
         && decoder->setOption (pdata, 3, "64:48")
         && decoder->setOption (pdata, 4, "300:300");
}

/**
 * @brief Write the label file of SSD_LABELS labels.
 */
static gchar *
writeSsdLabels (void)
{
  gchar *name = getTempFilename ();

  if (name != NULL && !g_file_set_contents (name, "background\nobject\n", -1, NULL))
    removeTempFile (&name);

  return name;
}

/**
 * @brief A box prior file with a prior for every detection is accepted and decoded.
 */
TEST (tensorDecoderBoundingBox, ssdPriorsForEveryDetection)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  gchar *labels = writeSsdLabels ();
  gchar *priors = writeBoxPriors (SSD_DETECTIONS, "0.5");
  SsdTensors t (SSD_DETECTIONS - 1);
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };
  void *pdata = NULL;

  ASSERT_TRUE (priors != NULL);
  ASSERT_TRUE (initSsdDecoder (decoder, &pdata, labels));

  EXPECT_TRUE (decoder->setOption (&pdata, 2, priors));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &t.config));
  EXPECT_TRUE (decodeFrame (decoder, &pdata, &t.config, t.input, frame));
  EXPECT_GT (countDrawnPixels (frame, BOX_OUT_PIXELS), 0U);

  decoder->exit (&pdata);
  removeTempFile (&priors);
  removeTempFile (&labels);
}

/**
 * @brief A stream with more detections than box priors is refused.
 * @details The priors of the missing columns used to be read uninitialised, or
 *          left over from a previous box prior file.
 */
TEST (tensorDecoderBoundingBox, ssdFewerPriorsThanDetections_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  gchar *labels = writeSsdLabels ();
  gchar *priors = writeBoxPriors (SSD_DETECTIONS - 1, "0.5");
  SsdTensors t (0);
  void *pdata = NULL;

  ASSERT_TRUE (priors != NULL);
  ASSERT_TRUE (initSsdDecoder (decoder, &pdata, labels));

  EXPECT_TRUE (decoder->setOption (&pdata, 2, priors));
  EXPECT_FALSE (acceptsConfig (decoder, &pdata, &t.config));

  decoder->exit (&pdata);
  removeTempFile (&priors);
  removeTempFile (&labels);
}

/**
 * @brief Box priors reduced while the stream runs bound the detections decoded.
 * @details The detection is at the column the second file does not have, which
 *          still holds the prior of the first file.
 */
TEST (tensorDecoderBoundingBox, ssdPriorsReducedWhileDecoding_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  gchar *labels = writeSsdLabels ();
  gchar *priors = writeBoxPriors (SSD_DETECTIONS, "0.5");
  gchar *fewer = writeBoxPriors (SSD_DETECTIONS - 1, "0.5");
  SsdTensors t (SSD_DETECTIONS - 1);
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };
  void *pdata = NULL;

  ASSERT_TRUE (priors != NULL);
  ASSERT_TRUE (fewer != NULL);
  ASSERT_TRUE (initSsdDecoder (decoder, &pdata, labels));

  EXPECT_TRUE (decoder->setOption (&pdata, 2, priors));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &t.config));

  EXPECT_TRUE (decoder->setOption (&pdata, 2, fewer));
  EXPECT_TRUE (decodeFrame (decoder, &pdata, &t.config, t.input, frame));
  EXPECT_EQ (countDrawnPixels (frame, BOX_OUT_PIXELS), 0U);

  decoder->exit (&pdata);
  removeTempFile (&fewer);
  removeTempFile (&priors);
  removeTempFile (&labels);
}

/**
 * @brief A box prior file that fails to load leaves no priors to decode with.
 * @details The rows are parsed into the prior table in place, so the priors
 *          of the previous file are no longer intact.
 */
TEST (tensorDecoderBoundingBox, ssdFailedPriorLoad_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  gchar *labels = writeSsdLabels ();
  gchar *priors = writeBoxPriors (SSD_DETECTIONS, "0.5");
  gchar *inconsistent = getTempFilename ();
  SsdTensors t (0);
  void *pdata = NULL;

  ASSERT_TRUE (priors != NULL);
  ASSERT_TRUE (inconsistent != NULL);
  ASSERT_TRUE (g_file_set_contents (
      inconsistent, "1 1 1 1\n1 1 1 1\n1 1\n1 1 1 1\n", -1, NULL));
  ASSERT_TRUE (initSsdDecoder (decoder, &pdata, labels));

  EXPECT_TRUE (decoder->setOption (&pdata, 2, priors));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &t.config));
  EXPECT_FALSE (decoder->setOption (&pdata, 2, inconsistent));
  EXPECT_FALSE (acceptsConfig (decoder, &pdata, &t.config));

  EXPECT_TRUE (decoder->setOption (&pdata, 2, priors));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &t.config));
  removeTempFile (&inconsistent);
  EXPECT_FALSE (decoder->setOption (&pdata, 2, "/no/such/box_priors.txt"));
  EXPECT_FALSE (acceptsConfig (decoder, &pdata, &t.config));

  decoder->exit (&pdata);
  removeTempFile (&priors);
  removeTempFile (&labels);
}

/* yolov8 has a box for every cell of its three strides, 32, 16 and 8 */
#define YOLO_SMALL_MODEL "32:32"
#define YOLO_SMALL_BOXES (21U) /* 1 + 4 + 16 */
#define YOLO_LARGE_MODEL "64:64"
#define YOLO_LARGE_BOXES (84U) /* 4 + 16 + 64 */
#define YOLO_BOX_INFO (5U) /* cx, cy, w, h and the score of the single label */

/**
 * @brief Start a yolov8 decoder of one label for a model of the input size @a model.
 */
static gboolean
initYoloV8Decoder (const GstTensorDecoderDef *decoder, void **pdata,
    const gchar *label_file, const gchar *model)
{
  return decoder != NULL && label_file != NULL && decoder->init (pdata)
         && decoder->setOption (pdata, 0, "yolov8")
         && decoder->setOption (pdata, 1, label_file)
         && decoder->setOption (pdata, 3, "64:48")
         && decoder->setOption (pdata, 4, model);
}

/**
 * @brief Build the tensors config of a yolov8 model of one label and @a boxes boxes.
 */
static void
setYoloV8Config (GstTensorsConfig *config, guint boxes)
{
  gchar *dim = g_strdup_printf ("%u:%u:1", YOLO_BOX_INFO, boxes);
  const gchar *const dims[] = { dim };

  setFloatConfig (config, 1, dims);
  g_free (dim);
}

/**
 * @brief Two decoders of the same mode for models of different sizes decode their own streams.
 * @details The box properties of a mode used to be one object per process, so
 *          the second decoder's option5 became the first one's too, and the
 *          first decoder read the boxes of the larger model past its own tensor.
 *          The tensor of the first decoder ends in front of an unreadable page,
 *          so that over-read faults at once.
 */
TEST (tensorDecoderBoundingBox, yoloV8DecodersOfDifferentModelSizes)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  gchar *labels = getTempFilename ();
  float small[YOLO_BOX_INFO * YOLO_SMALL_BOXES] = { 0.0f };
  float large[YOLO_BOX_INFO * YOLO_LARGE_BOXES] = { 0.0f };
  const guint last = (YOLO_SMALL_BOXES - 1) * YOLO_BOX_INFO;
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };
  uint32_t large_frame[BOX_OUT_PIXELS] = { 0U };
  GstTensorsConfig small_config, large_config;
  GstTensorMemory small_input, large_input;
  GuardedFloats g = { NULL, 0 };
  void *small_pdata = NULL;
  void *large_pdata = NULL;

  ASSERT_TRUE (labels != NULL);
  ASSERT_TRUE (g_file_set_contents (labels, "object\n", -1, NULL));

  small[last] = 0.5f;
  small[last + 1] = 0.5f;
  small[last + 2] = 0.5f;
  small[last + 3] = 0.5f;
  small[last + 4] = 0.9f;
  small_input.data = newGuardedFloats (&g, small, YOLO_BOX_INFO * YOLO_SMALL_BOXES);
  small_input.size = sizeof (small);
  ASSERT_TRUE (small_input.data != NULL);
  large_input.data = large;
  large_input.size = sizeof (large);

  setYoloV8Config (&small_config, YOLO_SMALL_BOXES);
  setYoloV8Config (&large_config, YOLO_LARGE_BOXES);

  ASSERT_TRUE (initYoloV8Decoder (decoder, &small_pdata, labels, YOLO_SMALL_MODEL));
  EXPECT_TRUE (acceptsConfig (decoder, &small_pdata, &small_config));

  ASSERT_TRUE (initYoloV8Decoder (decoder, &large_pdata, labels, YOLO_LARGE_MODEL));
  EXPECT_TRUE (acceptsConfig (decoder, &large_pdata, &large_config));
  EXPECT_FALSE (acceptsConfig (decoder, &large_pdata, &small_config));

  EXPECT_TRUE (decodeFrame (decoder, &small_pdata, &small_config, &small_input, frame));
  EXPECT_GT (countDrawnPixels (frame, BOX_OUT_PIXELS), 0U);
  EXPECT_TRUE (acceptsConfig (decoder, &small_pdata, &small_config));

  EXPECT_TRUE (decodeFrame (decoder, &large_pdata, &large_config, &large_input, large_frame));
  EXPECT_EQ (countDrawnPixels (large_frame, BOX_OUT_PIXELS), 0U);

  decoder->exit (&large_pdata);
  decoder->exit (&small_pdata);
  gst_tensors_config_free (&large_config);
  gst_tensors_config_free (&small_config);
  freeGuardedFloats (&g);
  removeTempFile (&labels);
}

/**
 * @brief Two mobilenet-ssd-postprocess decoders accept their own detection counts.
 * @details The mode keeps the first detection count it accepts. That used to
 *          hold for the whole process, so a model of another count was refused
 *          by every other decoder.
 */
TEST (tensorDecoderBoundingBox, ssdPpDecodersOfDifferentDetectionCounts)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  const gchar *const one_dims[] = { "1", "1:1", "1:1", "4:1" };
  const gchar *const two_dims[] = { "1", "2:1", "2:1", "4:2" };
  GstTensorsConfig one, two;
  void *one_pdata = NULL;
  void *two_pdata = NULL;

  ASSERT_TRUE (decoder != NULL);
  setFloatConfig (&one, 4, one_dims);
  setFloatConfig (&two, 4, two_dims);

  ASSERT_TRUE (decoder->init (&one_pdata));
  EXPECT_TRUE (decoder->setOption (&one_pdata, 0, "mobilenet-ssd-postprocess"));
  EXPECT_TRUE (decoder->setOption (&one_pdata, 3, "64:48"));
  EXPECT_TRUE (decoder->setOption (&one_pdata, 4, "640:480"));
  EXPECT_TRUE (acceptsConfig (decoder, &one_pdata, &one));

  ASSERT_TRUE (decoder->init (&two_pdata));
  EXPECT_TRUE (decoder->setOption (&two_pdata, 0, "mobilenet-ssd-postprocess"));
  EXPECT_TRUE (decoder->setOption (&two_pdata, 3, "64:48"));
  EXPECT_TRUE (decoder->setOption (&two_pdata, 4, "640:480"));
  EXPECT_TRUE (acceptsConfig (decoder, &two_pdata, &two));

  EXPECT_TRUE (acceptsConfig (decoder, &one_pdata, &one));
  EXPECT_FALSE (acceptsConfig (decoder, &one_pdata, &two));

  decoder->exit (&two_pdata);
  decoder->exit (&one_pdata);
  gst_tensors_config_free (&two);
  gst_tensors_config_free (&one);
}

/**
 * @brief A decoder does not take the options another decoder gave the same mode.
 * @details The second decoder is never given option5, so it has no model input
 *          size to scale boxes by, whatever the first decoder was given.
 */
TEST (tensorDecoderBoundingBox, optionsOfAnotherDecoder_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  GstTensorsConfig config;
  GstTensorMemory input;
  GstBuffer *outbuf;
  void *configured = NULL;
  void *pdata = NULL;

  ASSERT_TRUE (decoder != NULL);
  setOvDetectionConfig (&config);

  ASSERT_TRUE (decoder->init (&configured));
  EXPECT_TRUE (decoder->setOption (&configured, 0, "ov-person-detection"));
  EXPECT_TRUE (decoder->setOption (&configured, 3, "64:48"));
  EXPECT_TRUE (decoder->setOption (&configured, 4, "640:480"));
  EXPECT_TRUE (acceptsConfig (decoder, &configured, &config));

  ASSERT_TRUE (decoder->init (&pdata));
  EXPECT_TRUE (decoder->setOption (&pdata, 0, "ov-person-detection"));
  EXPECT_TRUE (decoder->setOption (&pdata, 3, "64:48"));
  EXPECT_FALSE (acceptsConfig (decoder, &pdata, &config));

  memset (&input, 0, sizeof (input));
  outbuf = gst_buffer_new ();
  EXPECT_EQ (decoder->decode (&pdata, &config, &input, outbuf), GST_FLOW_ERROR);
  gst_buffer_unref (outbuf);

  decoder->exit (&pdata);
  EXPECT_TRUE (acceptsConfig (decoder, &configured, &config));
  decoder->exit (&configured);
  gst_tensors_config_free (&config);
}

/**
 * @brief Giving option1 the mode it already has keeps the options of the mode.
 * @details tf-ssd is the deprecated name of mobilenet-ssd-postprocess, so it
 *          selects the same mode as well.
 */
TEST (tensorDecoderBoundingBox, sameModeKeepsOptions)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  const gchar *const dims[] = { "1", "1:1", "1:1", "4:1" };
  GstTensorsConfig config;
  void *pdata = NULL;

  ASSERT_TRUE (decoder != NULL);
  setFloatConfig (&config, 4, dims);

  ASSERT_TRUE (decoder->init (&pdata));
  EXPECT_TRUE (decoder->setOption (&pdata, 0, "mobilenet-ssd-postprocess"));
  EXPECT_TRUE (decoder->setOption (&pdata, 3, "64:48"));
  EXPECT_TRUE (decoder->setOption (&pdata, 4, "640:480"));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &config));

  EXPECT_TRUE (decoder->setOption (&pdata, 0, "mobilenet-ssd-postprocess"));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &config));
  EXPECT_TRUE (decoder->setOption (&pdata, 0, "tf-ssd"));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &config));

  decoder->exit (&pdata);
  gst_tensors_config_free (&config);
}

/**
 * @brief Switching option1 to another mode and back restores the options of the mode.
 * @details option1 is writable while the stream runs, so the box properties a
 *          decoder switches away from are kept until the decoder exits: decode ()
 *          may still be using them. The other mode starts from its defaults.
 */
TEST (tensorDecoderBoundingBox, switchModeBackKeepsOptions)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  GstTensorsConfig config;
  void *pdata = NULL;

  ASSERT_TRUE (decoder != NULL);
  setOvDetectionConfig (&config);

  ASSERT_TRUE (decoder->init (&pdata));
  EXPECT_TRUE (decoder->setOption (&pdata, 0, "ov-person-detection"));
  EXPECT_TRUE (decoder->setOption (&pdata, 3, "64:48"));
  EXPECT_TRUE (decoder->setOption (&pdata, 4, "640:480"));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &config));

  EXPECT_TRUE (decoder->setOption (&pdata, 0, "yolov8"));
  EXPECT_FALSE (acceptsConfig (decoder, &pdata, &config));
  EXPECT_TRUE (decoder->setOption (&pdata, 0, "ov-person-detection"));
  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &config));

  decoder->exit (&pdata);
  gst_tensors_config_free (&config);
}

/**
 * @brief A mode selected after another one does not take the options of the other.
 */
TEST (tensorDecoderBoundingBox, switchModeTakesNoOptions_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  const gchar *const dims[] = { "1", "1:1", "1:1", "4:1" };
  GstTensorsConfig config;
  void *pdata = NULL;

  ASSERT_TRUE (decoder != NULL);
  setFloatConfig (&config, 4, dims);

  ASSERT_TRUE (decoder->init (&pdata));
  EXPECT_TRUE (decoder->setOption (&pdata, 0, "ov-person-detection"));
  EXPECT_TRUE (decoder->setOption (&pdata, 3, "64:48"));
  EXPECT_TRUE (decoder->setOption (&pdata, 4, "640:480"));

  EXPECT_TRUE (decoder->setOption (&pdata, 0, "mobilenet-ssd-postprocess"));
  EXPECT_FALSE (acceptsConfig (decoder, &pdata, &config));

  decoder->exit (&pdata);
  gst_tensors_config_free (&config);
}

/**
 * @brief Every mode name option1 accepts gives a decoder box properties of its own.
 * @details A decoder that selects every mode, and every mode again, keeps one of each.
 */
TEST (tensorDecoderBoundingBox, createEveryMode)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  const gchar *const modes[] = { "mobilenet-ssd", "mobilenet-ssd-postprocess",
    "ov-person-detection", "tflite-ssd", "tf-ssd", "yolov5",
    "mp-palm-detection", "yolov8", "yolov8-obb", "yolov10" };
  void *pdata = NULL;
  guint i;

  ASSERT_TRUE (decoder != NULL);

  for (i = 0; i < G_N_ELEMENTS (modes); i++) {
    ASSERT_TRUE (decoder->init (&pdata));
    EXPECT_TRUE (decoder->setOption (&pdata, 0, modes[i])) << modes[i];
    EXPECT_TRUE (decoder->setOption (&pdata, 4, "320:320")) << modes[i];
    decoder->exit (&pdata);
  }

  ASSERT_TRUE (decoder->init (&pdata));
  for (i = 0; i < 2 * G_N_ELEMENTS (modes); i++)
    EXPECT_TRUE (decoder->setOption (&pdata, 0, modes[i % G_N_ELEMENTS (modes)]))
        << modes[i % G_N_ELEMENTS (modes)];
  decoder->exit (&pdata);
}

#define SSD_PP_MAX_BOXES (4U)

/**
 * @brief mobilenet-ssd-postprocess tensors with room for SSD_PP_MAX_BOXES boxes.
 * @details The count tensor says how many of them the decoder reads.
 */
class SsdPpBoxes
{
  public:
  float num; /**< count tensor */
  float classes[SSD_PP_MAX_BOXES]; /**< class tensor */
  float scores[SSD_PP_MAX_BOXES]; /**< score tensor */
  float boxes[4 * SSD_PP_MAX_BOXES]; /**< location tensor */
  GstTensorMemory input[4]; /**< the four tensors */
  GstTensorsConfig config; /**< their config */

  /**
   * @brief Start with no box.
   */
  SsdPpBoxes ()
  {
    const gchar *const dims[] = { "1", "4:1", "4:1", "4:4" };

    num = 0.0f;
    memset (classes, 0, sizeof (classes));
    memset (scores, 0, sizeof (scores));
    memset (boxes, 0, sizeof (boxes));

    input[0].data = &num;
    input[0].size = sizeof (num);
    input[1].data = classes;
    input[1].size = sizeof (classes);
    input[2].data = scores;
    input[2].size = sizeof (scores);
    input[3].data = boxes;
    input[3].size = sizeof (boxes);
    setFloatConfig (&config, 4, dims);
  }

  /**
   * @brief Release the config.
   */
  ~SsdPpBoxes ()
  {
    gst_tensors_config_free (&config);
  }

  /**
   * @brief Add a box of class @a class_id, in coordinates normalized to the model input.
   */
  void add (float class_id, float x_min, float y_min, float x_max, float y_max)
  {
    guint n = (guint) num;

    ASSERT_LT (n, SSD_PP_MAX_BOXES);
    classes[n] = class_id;
    scores[n] = 0.9f;
    boxes[4 * n] = y_min;
    boxes[4 * n + 1] = x_min;
    boxes[4 * n + 2] = y_max;
    boxes[4 * n + 3] = x_max;
    num += 1.0f;
  }

  /**
   * @brief Remove every box.
   */
  void clear ()
  {
    num = 0.0f;
  }
};

/**
 * @brief Write the labels to a new temp file.
 * @return the file name, to be released with removeTempFile ()
 */
static gchar *
writeLabelFile (const gchar *labels)
{
  gchar *name = getTempFilename ();

  if (name != NULL && !g_file_set_contents (name, labels, -1, NULL))
    removeTempFile (&name);

  return name;
}

/**
 * @brief Start a mobilenet-ssd-postprocess decoder that draws the labels in @a label_file.
 * @param[in] track option6, "1" to track the boxes
 * @param[in] log option7, "1" to log the boxes
 */
static gboolean
initSsdPpDecoder (const GstTensorDecoderDef *decoder, void **pdata,
    const gchar *label_file, const gchar *track, const gchar *log)
{
  return decoder != NULL && label_file != NULL && decoder->init (pdata)
         && decoder->setOption (pdata, 0, "mobilenet-ssd-postprocess")
         && decoder->setOption (pdata, 1, label_file)
         && decoder->setOption (pdata, 3, "64:48")
         && decoder->setOption (pdata, 4, "640:480")
         && decoder->setOption (pdata, 5, track) && decoder->setOption (pdata, 6, log);
}

/**
 * @brief Decode the boxes of @a t and copy out the frame drawn for them.
 */
static gboolean
decodeSsdPpBoxes (const GstTensorDecoderDef *decoder, void **pdata,
    SsdPpBoxes *t, uint32_t *frame)
{
  return acceptsConfig (decoder, pdata, &t->config)
         && decodeFrame (decoder, pdata, &t->config, t->input, frame);
}

/* Two boxes side by side, and a third one below them */
#define TRACK_LEFT 0.0f, 0.4f, 0.15f, 0.6f
#define TRACK_RIGHT 0.6f, 0.4f, 0.75f, 0.6f
#define TRACK_BELOW 0.3f, 0.8f, 0.45f, 0.95f

/**
 * @brief A tracked box keeps the id of the nearest box of the previous frame when there are fewer boxes.
 * @details The tracking id is drawn after the label, so a frame drawn with
 *          tracking has to match the one drawn without it from a label that
 *          spells the expected id. The distances of the second centroid used
 *          to be stored past the ones in use, so the right box took the id of
 *          the left one.
 */
TEST (tensorDecoderBoundingBox, trackFewerBoxesThanCentroids)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  gchar *labels = writeLabelFile ("X\n");
  gchar *expected_labels = writeLabelFile ("X-1\nX-2\n");
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };
  uint32_t expected[BOX_OUT_PIXELS] = { 0U };
  void *pdata = NULL;
  void *expected_pdata = NULL;
  SsdPpBoxes t;

  ASSERT_TRUE (initSsdPpDecoder (decoder, &pdata, labels, "1", "0"));
  ASSERT_TRUE (initSsdPpDecoder (decoder, &expected_pdata, expected_labels, "0", "0"));

  t.add (0.0f, TRACK_LEFT);
  t.add (0.0f, TRACK_RIGHT);
  EXPECT_TRUE (decodeSsdPpBoxes (decoder, &pdata, &t, frame));

  t.clear ();
  t.add (0.0f, TRACK_RIGHT);
  EXPECT_TRUE (decodeSsdPpBoxes (decoder, &pdata, &t, frame));

  t.clear ();
  t.add (1.0f, TRACK_RIGHT);
  EXPECT_TRUE (decodeSsdPpBoxes (decoder, &expected_pdata, &t, expected));

  EXPECT_GT (countDrawnPixels (expected, BOX_OUT_PIXELS), 0U);
  EXPECT_EQ (memcmp (frame, expected, sizeof (frame)), 0);

  decoder->exit (&expected_pdata);
  decoder->exit (&pdata);
  removeTempFile (&expected_labels);
  removeTempFile (&labels);
}

/**
 * @brief Tracked boxes keep the ids of the nearest boxes of the previous frame when there are more boxes.
 * @details The distance of the first centroid to the last box used to be
 *          overwritten by the second centroid's, so the left box lost its id
 *          to the new box below.
 */
TEST (tensorDecoderBoundingBox, trackMoreBoxesThanCentroids)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  gchar *labels = writeLabelFile ("X\n");
  gchar *expected_labels = writeLabelFile ("X-1\nX-2\nX-3\n");
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };
  uint32_t expected[BOX_OUT_PIXELS] = { 0U };
  void *pdata = NULL;
  void *expected_pdata = NULL;
  SsdPpBoxes t;

  ASSERT_TRUE (initSsdPpDecoder (decoder, &pdata, labels, "1", "0"));
  ASSERT_TRUE (initSsdPpDecoder (decoder, &expected_pdata, expected_labels, "0", "0"));

  t.add (0.0f, TRACK_LEFT);
  t.add (0.0f, TRACK_RIGHT);
  EXPECT_TRUE (decodeSsdPpBoxes (decoder, &pdata, &t, frame));

  t.clear ();
  t.add (0.0f, TRACK_RIGHT);
  t.add (0.0f, TRACK_BELOW);
  t.add (0.0f, TRACK_LEFT);
  EXPECT_TRUE (decodeSsdPpBoxes (decoder, &pdata, &t, frame));

  t.clear ();
  t.add (1.0f, TRACK_RIGHT);
  t.add (2.0f, TRACK_BELOW);
  t.add (0.0f, TRACK_LEFT);
  EXPECT_TRUE (decodeSsdPpBoxes (decoder, &expected_pdata, &t, expected));

  EXPECT_GT (countDrawnPixels (expected, BOX_OUT_PIXELS), 0U);
  EXPECT_EQ (memcmp (frame, expected, sizeof (frame)), 0);

  decoder->exit (&expected_pdata);
  decoder->exit (&pdata);
  removeTempFile (&expected_labels);
  removeTempFile (&labels);
}

/**
 * @brief A tracked mp-palm-detection box keeps its id in the next frame.
 * @details The box used to start with whatever tracking id the stack held, and
 *          a non-zero one kept it out of the matching.
 */
TEST (tensorDecoderBoundingBox, trackPalmDetection)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  gchar *labels = writeLabelFile ("X\n");
  gchar *expected_labels = writeLabelFile ("X-1\n");
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };
  uint32_t expected[BOX_OUT_PIXELS] = { 0U };
  void *pdata = NULL;
  void *expected_pdata = NULL;
  PalmDetectionTensors t;

  ASSERT_TRUE (labels != NULL);
  ASSERT_TRUE (expected_labels != NULL);
  ASSERT_TRUE (initPalmDecoder (decoder, &pdata));
  EXPECT_TRUE (decoder->setOption (&pdata, 1, labels));
  EXPECT_TRUE (decoder->setOption (&pdata, 2, PALM_OPTION_STRIDE_32));
  EXPECT_TRUE (decoder->setOption (&pdata, 5, "1"));
  ASSERT_TRUE (initPalmDecoder (decoder, &expected_pdata));
  EXPECT_TRUE (decoder->setOption (&expected_pdata, 1, expected_labels));
  EXPECT_TRUE (decoder->setOption (&expected_pdata, 2, PALM_OPTION_STRIDE_32));

  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &t.config));
  EXPECT_TRUE (decodeFrame (decoder, &pdata, &t.config, t.input, frame));
  EXPECT_TRUE (decodeFrame (decoder, &pdata, &t.config, t.input, frame));

  EXPECT_TRUE (acceptsConfig (decoder, &expected_pdata, &t.config));
  EXPECT_TRUE (decodeFrame (decoder, &expected_pdata, &t.config, t.input, expected));

  EXPECT_GT (countDrawnPixels (expected, BOX_OUT_PIXELS), 0U);
  EXPECT_EQ (memcmp (frame, expected, sizeof (frame)), 0);

  decoder->exit (&expected_pdata);
  decoder->exit (&pdata);
  removeTempFile (&expected_labels);
  removeTempFile (&labels);
}

/**
 * @brief Decode frames with a box that is gone for @a empty_frames frames and comes back.
 * @details The box coming back has to be drawn with @a expected_label, a
 *          label that spells the tracking id it is expected to get.
 */
static void
checkTrackAfterEmptyFrames (guint empty_frames, const gchar *expected_label)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  gchar *labels = writeLabelFile ("X\n");
  gchar *expected_labels = writeLabelFile (expected_label);
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };
  uint32_t expected[BOX_OUT_PIXELS] = { 0U };
  void *pdata = NULL;
  void *expected_pdata = NULL;
  guint i, decoded = 0;
  SsdPpBoxes t;

  ASSERT_TRUE (initSsdPpDecoder (decoder, &pdata, labels, "1", "0"));
  ASSERT_TRUE (initSsdPpDecoder (decoder, &expected_pdata, expected_labels, "0", "0"));

  t.add (0.0f, TRACK_RIGHT);
  EXPECT_TRUE (decodeSsdPpBoxes (decoder, &pdata, &t, frame));

  t.clear ();
  for (i = 0; i < empty_frames; i++) {
    if (decodeSsdPpBoxes (decoder, &pdata, &t, frame))
      decoded++;
  }
  EXPECT_EQ (decoded, empty_frames);
  EXPECT_EQ (countDrawnPixels (frame, BOX_OUT_PIXELS), 0U);

  t.add (0.0f, TRACK_RIGHT);
  EXPECT_TRUE (decodeSsdPpBoxes (decoder, &pdata, &t, frame));
  EXPECT_TRUE (decodeSsdPpBoxes (decoder, &expected_pdata, &t, expected));

  EXPECT_GT (countDrawnPixels (expected, BOX_OUT_PIXELS), 0U);
  EXPECT_EQ (memcmp (frame, expected, sizeof (frame)), 0);

  decoder->exit (&expected_pdata);
  decoder->exit (&pdata);
  removeTempFile (&expected_labels);
  removeTempFile (&labels);
}

/**
 * @brief A tracked box that is gone for fewer frames than the threshold keeps its id.
 */
TEST (tensorDecoderBoundingBox, trackBoxBackBeforeThreshold)
{
  checkTrackAfterEmptyFrames (99U, "X-1\n");
}

/**
 * @brief A tracked box that is gone for as many frames as the threshold gets a new id.
 */
TEST (tensorDecoderBoundingBox, trackBoxBackAtThreshold)
{
  checkTrackAfterEmptyFrames (100U, "X-2\n");
}

/**
 * @brief The element types the box properties decode.
 */
static const tensor_type decoded_types[] = { _NNS_INT8, _NNS_UINT8, _NNS_INT16, _NNS_UINT16,
  _NNS_INT32, _NNS_UINT32, _NNS_INT64, _NNS_UINT64, _NNS_FLOAT32, _NNS_FLOAT64 };

/**
 * @brief Convert the floats to a new array of @a type.
 * @details The floats have to be integers that every type can hold.
 * @return the array, to be released with g_free ()
 */
static gpointer
newTypedData (tensor_type type, const float *src, gsize elements)
{
  gpointer data = g_malloc0 (elements * gst_tensor_get_element_size (type));
  gsize i;

  for (i = 0; i < elements; i++) {
    switch (type) {
      case _NNS_INT8:
        ((int8_t *) data)[i] = (int8_t) src[i];
        break;
      case _NNS_UINT8:
        ((uint8_t *) data)[i] = (uint8_t) src[i];
        break;
      case _NNS_INT16:
        ((int16_t *) data)[i] = (int16_t) src[i];
        break;
      case _NNS_UINT16:
        ((uint16_t *) data)[i] = (uint16_t) src[i];
        break;
      case _NNS_INT32:
        ((int32_t *) data)[i] = (int32_t) src[i];
        break;
      case _NNS_UINT32:
        ((uint32_t *) data)[i] = (uint32_t) src[i];
        break;
      case _NNS_INT64:
        ((int64_t *) data)[i] = (int64_t) src[i];
        break;
      case _NNS_UINT64:
        ((uint64_t *) data)[i] = (uint64_t) src[i];
        break;
      case _NNS_FLOAT32:
        ((float *) data)[i] = src[i];
        break;
      case _NNS_FLOAT64:
        ((double *) data)[i] = (double) src[i];
        break;
      default:
        break;
    }
  }

  return data;
}

#define TYPED_TENSORS_MAX (4U)

/**
 * @brief Tensors of one element type, converted from floats.
 */
class TypedTensors
{
  public:
  GstTensorMemory input[TYPED_TENSORS_MAX]; /**< the tensors */
  GstTensorsConfig config; /**< their config */

  /**
   * @brief Convert @a num float tensors of the dimensions @a dims to @a type.
   */
  TypedTensors (tensor_type type, guint num, const gchar *const *dims, const float *const *src)
  {
    guint i;

    setFloatConfig (&config, num, dims);
    memset (input, 0, sizeof (input));
    for (i = 0; i < num; i++) {
      gsize elements = gst_tensor_get_element_count (config.info.info[i].dimension);

      config.info.info[i].type = type;
      input[i].data = newTypedData (type, src[i], elements);
      input[i].size = elements * gst_tensor_get_element_size (type);
    }
  }

  /**
   * @brief Release the tensors and the config.
   */
  ~TypedTensors ()
  {
    guint i;

    for (i = 0; i < TYPED_TENSORS_MAX; i++)
      g_free (input[i].data);
    gst_tensors_config_free (&config);
  }
};

/**
 * @brief Decode the tensors twice and copy out the second frame.
 * @param[in] mode option1
 * @param[in] option3 option3, or NULL to leave it unset
 * @param[in] model option5
 * @param[in] label_text the content of the label file
 * @param[in] track option6
 */
static gboolean
decodeTwice (const gchar *mode, const gchar *option3, const gchar *model,
    const gchar *label_text, const gchar *track, TypedTensors *t, uint32_t *frame)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  gchar *labels = writeLabelFile (label_text);
  void *pdata = NULL;
  gboolean ret = FALSE;

  if (decoder != NULL && labels != NULL && decoder->init (&pdata)) {
    ret = decoder->setOption (&pdata, 0, mode) && decoder->setOption (&pdata, 1, labels)
          && (option3 == NULL || decoder->setOption (&pdata, 2, option3))
          && decoder->setOption (&pdata, 3, "64:48")
          && decoder->setOption (&pdata, 4, model)
          && decoder->setOption (&pdata, 5, track)
          && acceptsConfig (decoder, &pdata, &t->config)
          && decodeFrame (decoder, &pdata, &t->config, t->input, frame)
          && decodeFrame (decoder, &pdata, &t->config, t->input, frame);
    decoder->exit (&pdata);
  }

  removeTempFile (&labels);
  return ret;
}

/**
 * @brief A tracked mobilenet-ssd-postprocess box of every element type keeps its id.
 * @details The box covers the whole frame, so that its corners are integers. The
 *          frame drawn with tracking has to match the one drawn without it from
 *          a label that spells the id.
 */
TEST (tensorDecoderBoundingBox, ssdPpTrackEveryType)
{
  const gchar *const dims[] = { "1", "4:1", "4:1", "4:4" };
  const float num[] = { 1.0f };
  const float classes[4] = { 0.0f };
  const float scores[4] = { 1.0f };
  const float boxes[16] = { 0.0f, 0.0f, 1.0f, 1.0f };
  const float *const src[] = { num, classes, scores, boxes };
  uint32_t expected[BOX_OUT_PIXELS] = { 0U };
  guint i;

  {
    TypedTensors t (_NNS_FLOAT32, 4, dims, src);

    ASSERT_TRUE (decodeTwice ("mobilenet-ssd-postprocess", NULL, "640:480",
        "X-1\n", "0", &t, expected));
  }
  EXPECT_GT (countDrawnPixels (expected, BOX_OUT_PIXELS), 0U);

  for (i = 0; i < G_N_ELEMENTS (decoded_types); i++) {
    TypedTensors t (decoded_types[i], 4, dims, src);
    uint32_t frame[BOX_OUT_PIXELS] = { 0U };

    EXPECT_TRUE (decodeTwice (
        "mobilenet-ssd-postprocess", NULL, "640:480", "X\n", "1", &t, frame))
        << gst_tensor_get_type_string (decoded_types[i]);
    EXPECT_EQ (memcmp (frame, expected, sizeof (frame)), 0)
        << gst_tensor_get_type_string (decoded_types[i]);
  }
}

/* PALM_OPTION_STRIDE_32 with a score threshold that a score of 0 does not reach */
#define PALM_OPTION_STRIDE_32_ABOVE_HALF "0.6:1:1.0:1.0:0.5:0.5:32"

/**
 * @brief A tracked mp-palm-detection box of every element type keeps its id.
 * @details Unsigned types cannot hold the negative scores of PalmDetectionTensors,
 *          so the other anchors score 0, which the threshold of 0.6 drops.
 */
TEST (tensorDecoderBoundingBox, palmTrackEveryType)
{
  const gchar *const dims[] = { "18:72:1", "1:72:1" };
  float boxes[PALM_INFO_SIZE * PALM_DETECTIONS] = { 0.0f };
  float scores[PALM_DETECTIONS] = { 0.0f };
  const float *const src[] = { boxes, scores };
  uint32_t expected[BOX_OUT_PIXELS] = { 0U };
  guint i;

  boxes[PALM_DETECTION_INDEX * PALM_INFO_SIZE + 2] = 48.0f;
  boxes[PALM_DETECTION_INDEX * PALM_INFO_SIZE + 3] = 48.0f;
  scores[PALM_DETECTION_INDEX] = 10.0f;

  {
    TypedTensors t (_NNS_FLOAT32, 2, dims, src);

    ASSERT_TRUE (decodeTwice ("mp-palm-detection",
        PALM_OPTION_STRIDE_32_ABOVE_HALF, "192:192", "X-1\n", "0", &t, expected));
  }
  EXPECT_GT (countDrawnPixels (expected, BOX_OUT_PIXELS), 0U);

  for (i = 0; i < G_N_ELEMENTS (decoded_types); i++) {
    TypedTensors t (decoded_types[i], 2, dims, src);
    uint32_t frame[BOX_OUT_PIXELS] = { 0U };

    EXPECT_TRUE (decodeTwice ("mp-palm-detection",
        PALM_OPTION_STRIDE_32_ABOVE_HALF, "192:192", "X\n", "1", &t, frame))
        << gst_tensor_get_type_string (decoded_types[i]);
    EXPECT_EQ (memcmp (frame, expected, sizeof (frame)), 0)
        << gst_tensor_get_type_string (decoded_types[i]);
  }
}

/**
 * @brief Collect every message logged to the default GLib log handler.
 */
static void
collectLog (const gchar *log_domain, GLogLevelFlags log_level,
    const gchar *message, gpointer user_data)
{
  UNUSED (log_domain);
  UNUSED (log_level);

  g_ptr_array_add ((GPtrArray *) user_data, g_strdup (message));
}

/**
 * @brief Count the collected messages that start with @a prefix.
 */
static guint
countLogLines (GPtrArray *log, const gchar *prefix)
{
  guint i, count = 0;

  for (i = 0; i < log->len; i++) {
    if (g_str_has_prefix ((const gchar *) g_ptr_array_index (log, i), prefix))
      count++;
  }

  return count;
}

/**
 * @brief Decode the boxes of @a t with option7 set and collect what the decoder logs.
 */
static gboolean
logSsdPpBoxes (const gchar *label_file, SsdPpBoxes *t, GPtrArray *log)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };
  GLogFunc old_handler;
  void *pdata = NULL;
  gboolean ret = FALSE;

  if (!initSsdPpDecoder (decoder, &pdata, label_file, "0", "1"))
    return FALSE;

  if (acceptsConfig (decoder, &pdata, &t->config)) {
    old_handler = g_log_set_default_handler (collectLog, log);
    ret = decodeFrame (decoder, &pdata, &t->config, t->input, frame);
    g_log_set_default_handler (old_handler, NULL);
  }

  decoder->exit (&pdata);
  return ret;
}

/**
 * @brief A logged box of a known class is logged with its label.
 */
TEST (tensorDecoderBoundingBox, logLabelOfBox)
{
  gchar *labels = writeLabelFile ("X\nY\n");
  GPtrArray *log = g_ptr_array_new_with_free_func (g_free);
  SsdPpBoxes t;

  t.add (1.0f, TRACK_LEFT);
  EXPECT_TRUE (logSsdPpBoxes (labels, &t, log));
#ifndef __TIZEN__
  EXPECT_EQ (countLogLines (log, "[Y] x:"), 1U);
#endif

  g_ptr_array_unref (log);
  removeTempFile (&labels);
}

/**
 * @brief A logged box of a class past the labels is logged without a label.
 * @details The label used to be read from past the end of the label array.
 */
TEST (tensorDecoderBoundingBox, logBoxOfClassPastLabels_n)
{
  gchar *labels = writeLabelFile ("X\nY\n");
  GPtrArray *log = g_ptr_array_new_with_free_func (g_free);
  SsdPpBoxes t;

  t.add (2.0f, TRACK_LEFT);
  t.add (100000000.0f, TRACK_RIGHT);
  EXPECT_TRUE (logSsdPpBoxes (labels, &t, log));
  EXPECT_EQ (countLogLines (log, "["), 0U);
#ifndef __TIZEN__
  EXPECT_EQ (countLogLines (log, "x:"), 2U);
#endif

  g_ptr_array_unref (log);
  removeTempFile (&labels);
}

/**
 * @brief A logged ov-person-detection box, which has no class, is logged without a label.
 * @details The mode gives its boxes the class -1, and the label used to be read
 *          from in front of the label array.
 */
TEST (tensorDecoderBoundingBox, logBoxOfNegativeClass_n)
{
  const GstTensorDecoderDef *decoder = nnstreamer_decoder_find ("bounding_boxes");
  gchar *labels = writeLabelFile ("X\n");
  GPtrArray *log = g_ptr_array_new_with_free_func (g_free);
  float tensor[OV_TENSOR_ELEMENTS] = { 0.0f };
  uint32_t frame[BOX_OUT_PIXELS] = { 0U };
  GstTensorMemory input;
  GstTensorsConfig config;
  GLogFunc old_handler;
  void *pdata = NULL;

  ASSERT_TRUE (decoder != NULL);
  ASSERT_TRUE (labels != NULL);
  ASSERT_TRUE (decoder->init (&pdata));
  EXPECT_TRUE (decoder->setOption (&pdata, 0, "ov-person-detection"));
  EXPECT_TRUE (decoder->setOption (&pdata, 1, labels));
  EXPECT_TRUE (decoder->setOption (&pdata, 3, "64:48"));
  EXPECT_TRUE (decoder->setOption (&pdata, 4, "640:480"));
  EXPECT_TRUE (decoder->setOption (&pdata, 6, "1"));

  setOvDetectionConfig (&config);
  setDetection (tensor, 0.25f, 0.25f, 0.75f, 0.75f);
  input.data = tensor;
  input.size = sizeof (tensor);

  EXPECT_TRUE (acceptsConfig (decoder, &pdata, &config));
  old_handler = g_log_set_default_handler (collectLog, log);
  EXPECT_TRUE (decodeFrame (decoder, &pdata, &config, &input, frame));
  g_log_set_default_handler (old_handler, NULL);

  EXPECT_EQ (countLogLines (log, "["), 0U);
#ifndef __TIZEN__
  EXPECT_EQ (countLogLines (log, "x:"), 1U);
#endif

  gst_tensors_config_free (&config);
  decoder->exit (&pdata);
  g_ptr_array_unref (log);
  removeTempFile (&labels);
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
