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
 * @brief Decode the given tensor with the bounding_boxes decoder.
 * @param[in] tensor OV_TENSOR_ELEMENTS floats of ov-person-detection output
 * @param[out] frame OUT_PIXELS RGBA pixels drawn by the decoder
 */
static gboolean
decodeBoundingBoxes (const float *tensor, uint32_t *frame)
{
  gchar *in_file = getTempFilename ();
  gchar *out_file = getTempFilename ();
  gchar *pipeline_str;
  gchar *content = NULL;
  gsize len = 0;
  GstElement *pipeline;
  gboolean ret = FALSE;

  if (in_file == NULL || out_file == NULL)
    return FALSE;

  if (g_file_set_contents (in_file, (const gchar *) tensor, OV_TENSOR_SIZE, NULL)) {
    pipeline_str = g_strdup_printf (
        "filesrc location=%s blocksize=%u ! application/octet-stream ! "
        "tensor_converter input-dim=%u:%u:1:1 input-type=float32 ! "
        "tensor_decoder mode=bounding_boxes option1=ov-person-detection "
        "option4=%u:%u option5=%u:%u ! "
        "filesink location=%s buffer-mode=unbuffered sync=false async=false",
        in_file, (guint) OV_TENSOR_SIZE, OV_DESC_SIZE, OV_DETECTION_MAX,
        OUT_WIDTH, OUT_HEIGHT, MODEL_WIDTH, MODEL_HEIGHT, out_file);

    pipeline = gst_parse_launch (pipeline_str, NULL);
    g_free (pipeline_str);

    if (pipeline != NULL) {
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
    }
  }

  if (ret) {
    ret = g_file_get_contents (out_file, &content, &len, NULL)
          && len == OUT_PIXELS * sizeof (uint32_t);
    if (ret)
      memcpy (frame, content, len);
    g_free (content);
  }

  removeTempFile (&in_file);
  removeTempFile (&out_file);

  return ret;
}

/**
 * @brief Count the pixels the decoder has drawn.
 */
static guint
countDrawnPixels (const uint32_t *frame)
{
  guint i, count = 0;

  for (i = 0; i < OUT_PIXELS; i++) {
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

  EXPECT_EQ (countDrawnPixels (frame), 0U);
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

  EXPECT_EQ (countDrawnPixels (frame), 0U);
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

  EXPECT_EQ (countDrawnPixels (frame), 0U);
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
