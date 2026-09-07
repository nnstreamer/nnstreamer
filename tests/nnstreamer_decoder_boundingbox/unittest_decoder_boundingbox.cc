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
#include <gst/gst.h>
#include <string.h>
#include <unittest_util.h>

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
 * @param[out] frame LABEL_PIXELS RGBA pixels drawn by the decoder
 */
static gboolean
decodeLabelledBox (uint32_t *frame)
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
      && g_file_set_contents (label_file, LABEL_TEXT, -1, NULL)) {
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

  ASSERT_TRUE (decodeLabelledBox (frame));

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
