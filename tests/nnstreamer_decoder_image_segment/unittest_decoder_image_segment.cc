/**
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * @file        unittest_decoder_image_segment.cc
 * @date        15 Sep 2026
 * @brief       Unit test for the image_segment mode of tensor_decoder
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <string.h>
#include <unittest_util.h>

#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api_util.h>

/* The number of labels image_segment assumes without option2, background included */
#define DEFAULT_COLORS (21U)

#define ALPHA_MASK (0xFF000000U)

/* Bytes after the output memory the decoder must not write */
#define OUTPUT_GUARD (64U)
#define OUTPUT_GUARD_VALUE (0xA5)

/* Floats after the input tensor the decoder must not read */
#define INPUT_GUARD (4U)

/**
 * @brief Test fixture holding an instance of the image_segment decoder.
 */
class tensorDecoderImageSegment : public ::testing::Test
{
  protected:
  const GstTensorDecoderDef *decoder;
  void *pdata;

  /**
   * @brief Initialize the decoder instance.
   */
  void SetUp () override
  {
    pdata = NULL;
    decoder = nnstreamer_decoder_find ("image_segment");
    ASSERT_TRUE (decoder != NULL);
    ASSERT_TRUE (decoder->init (&pdata));
  }

  /**
   * @brief Release the decoder instance.
   */
  void TearDown () override
  {
    if (pdata != NULL)
      decoder->exit (&pdata);
  }

  /**
   * @brief Set an option of the decoder instance.
   */
  gboolean setOption (int op_num, const gchar *param)
  {
    return decoder->setOption (&pdata, op_num, param);
  }

  /**
   * @brief Get the frame size the decoder negotiates for the given config.
   */
  gboolean getOutSize (const GstTensorsConfig *config, gint *width, gint *height)
  {
    GstCaps *caps = decoder->getOutCaps (&pdata, config);
    GstStructure *s;
    gboolean ret;

    if (caps == NULL)
      return FALSE;

    s = gst_caps_get_structure (caps, 0);
    ret = gst_structure_get_int (s, "width", width)
          && gst_structure_get_int (s, "height", height);
    gst_caps_unref (caps);
    return ret;
  }

  /**
   * @brief Decode one frame into an output memory followed by guard bytes.
   * @param[in] config The tensors config of the input
   * @param[in] in The input tensor, sized as config describes
   * @param[in] input_guard The value of the INPUT_GUARD floats after the input
   * @param[in] out_size The size of the output memory, 0 to let the decoder allocate it
   * @param[out] out The output pixels, out_size bytes (may be NULL if out_size is 0)
   * @param[out] result_size The size of the output buffer after decoding
   * @param[out] guard_ok Whether the bytes after the output memory are untouched
   * @param[in] negotiate Whether to run getOutCaps () on the config first, as
   *            tensor_decoder does when the caps of the input are set
   */
  GstFlowReturn decode (const GstTensorsConfig *config, const float *in,
      float input_guard, gsize out_size, uint32_t *out, gsize *result_size,
      gboolean *guard_ok, gboolean negotiate = TRUE)
  {
    GstTensorMemory input;
    GstBuffer *outbuf = gst_buffer_new ();
    guint8 *block = NULL;
    float *in_block;
    GstFlowReturn ret;
    gsize i;

    if (negotiate) {
      GstCaps *caps = decoder->getOutCaps (&pdata, config);
      if (caps != NULL)
        gst_caps_unref (caps);
    }

    input.size = gst_tensor_info_get_size (&config->info.info[0]);
    in_block = (float *) g_malloc0 (input.size + INPUT_GUARD * sizeof (float));
    memcpy (in_block, in, input.size);
    for (i = 0; i < INPUT_GUARD; i++)
      in_block[input.size / sizeof (float) + i] = input_guard;
    input.data = in_block;

    if (out_size > 0) {
      block = (guint8 *) g_malloc0 (out_size + OUTPUT_GUARD);
      memset (block + out_size, OUTPUT_GUARD_VALUE, OUTPUT_GUARD);
      gst_buffer_append_memory (
          outbuf, gst_memory_new_wrapped ((GstMemoryFlags) 0, block,
                      out_size + OUTPUT_GUARD, 0, out_size, NULL, NULL));
    }

    ret = decoder->decode (&pdata, config, &input, outbuf);
    *result_size = gst_buffer_get_size (outbuf);
    *guard_ok = TRUE;

    if (block != NULL) {
      for (i = 0; i < OUTPUT_GUARD; i++) {
        if (block[out_size + i] != OUTPUT_GUARD_VALUE)
          *guard_ok = FALSE;
      }
      memcpy (out, block, out_size);
    }

    gst_buffer_unref (outbuf);
    g_free (block);
    g_free (in_block);
    return ret;
  }
};

/**
 * @brief Describe one float32 tensor of the given dimensions.
 */
static void
setConfig (GstTensorsConfig *config, guint d0, guint d1, guint d2, tensor_type type = _NNS_FLOAT32)
{
  GstTensorInfo *info;

  gst_tensors_config_init (config);
  config->rate_n = 0;
  config->rate_d = 1;
  config->info.num_tensors = 1;

  info = gst_tensors_info_get_nth_info (&config->info, 0);
  info->type = type;
  info->dimension[0] = d0;
  info->dimension[1] = d1;
  info->dimension[2] = d2;
  info->dimension[3] = 1;
}

/**
 * @brief Build a tflite-deeplab probability map giving each pixel its label.
 * @return colors x pixels floats, to be released with g_free ()
 */
static float *
makeProbabilities (const guint *labels, guint pixels, guint colors)
{
  float *prob = (float *) g_malloc0 (sizeof (float) * pixels * colors);
  guint i;

  for (i = 0; i < pixels; i++)
    prob[i * colors + labels[i]] = 1.0f;

  return prob;
}

/**
 * @brief Check the pixels of a label map: background is transparent, every
 *        other label is opaque and a label keeps one color within the frame.
 */
static void
expectLabelColors (const guint *labels, const uint32_t *pixels, guint num)
{
  guint i, j;

  for (i = 0; i < num; i++) {
    if (labels[i] == 0) {
      EXPECT_EQ (pixels[i], 0U) << "pixel " << i;
      continue;
    }

    EXPECT_EQ (pixels[i] & ALPHA_MASK, ALPHA_MASK) << "pixel " << i;
    for (j = 0; j < i; j++) {
      if (labels[j] == labels[i]) {
        EXPECT_EQ (pixels[i], pixels[j]) << "pixels " << j << " and " << i;
      }
    }
  }
}

/**
 * @brief snpe-depth over frames whose pixel count is not a multiple of four
 *        must neither read past the input nor write past the output.
 */
TEST_F (tensorDecoderImageSegment, snpeDepthOddPixels)
{
  const guint sizes[][2] = { { 3, 3 }, { 5, 3 }, { 3, 1 }, { 1, 1 }, { 4, 2 } };
  GstTensorsConfig config;
  guint s, i;

  ASSERT_TRUE (setOption (0, "snpe-depth"));

  for (s = 0; s < G_N_ELEMENTS (sizes); s++) {
    const guint num = sizes[s][0] * sizes[s][1];
    float *in = g_new0 (float, num);
    uint32_t *out = g_new0 (uint32_t, num);
    const float max = 8.0f;
    gsize result_size;
    gboolean guard_ok;

    /* The maximum is a power of two, so the expected gray levels are exact */
    for (i = 0; i < num; i++)
      in[i] = (float) ((i * 5) % 9);
    in[0] = max;

    setConfig (&config, 1, sizes[s][0], sizes[s][1]);
    /* A read past the input would raise the maximum and darken every pixel */
    EXPECT_EQ (decode (&config, in, 1000000.0f, num * 4, out, &result_size, &guard_ok),
        GST_FLOW_OK);
    EXPECT_EQ (result_size, num * 4);
    EXPECT_TRUE (guard_ok) << sizes[s][0] << "x" << sizes[s][1];

    for (i = 0; i < num; i++) {
      guint gray = (guint) ((in[i] / max) * 255);
      EXPECT_EQ (out[i], gray * 0x010101U | ALPHA_MASK)
          << sizes[s][0] << "x" << sizes[s][1] << " pixel " << i;
    }

    g_free (in);
    g_free (out);
  }
}

/**
 * @brief snpe-deeplab over frames whose pixel count is not a multiple of four
 *        must not write past the output.
 */
TEST_F (tensorDecoderImageSegment, snpeDeeplabOddPixels)
{
  const guint sizes[][2] = { { 3, 3 }, { 5, 3 }, { 7, 1 }, { 2, 1 }, { 4, 2 } };
  GstTensorsConfig config;
  guint s, i;

  ASSERT_TRUE (setOption (0, "snpe-deeplab"));

  for (s = 0; s < G_N_ELEMENTS (sizes); s++) {
    const guint num = sizes[s][0] * sizes[s][1];
    guint *labels = g_new0 (guint, num);
    float *in = g_new0 (float, num);
    uint32_t *out = g_new0 (uint32_t, num);
    gsize result_size;
    gboolean guard_ok;

    for (i = 0; i < num; i++) {
      labels[i] = (i + 1) % 3;
      in[i] = (float) labels[i];
    }

    /* snpe-deeplab takes width from the first dimension */
    setConfig (&config, sizes[s][0], sizes[s][1], 1);
    EXPECT_EQ (decode (&config, in, 1.0f, num * 4, out, &result_size, &guard_ok), GST_FLOW_OK);
    EXPECT_EQ (result_size, num * 4);
    EXPECT_TRUE (guard_ok) << sizes[s][0] << "x" << sizes[s][1];
    expectLabelColors (labels, out, num);

    g_free (labels);
    g_free (in);
    g_free (out);
  }
}

/**
 * @brief tflite-deeplab over a frame whose pixel count is not a multiple of four.
 */
TEST_F (tensorDecoderImageSegment, tfliteDeeplabOddPixels)
{
  const guint labels[] = { 1, 0, 2, 20, 0, 5, 1, 7, 20 };
  const guint num = G_N_ELEMENTS (labels);
  GstTensorsConfig config;
  uint32_t out[G_N_ELEMENTS (labels)];
  float *prob = makeProbabilities (labels, num, DEFAULT_COLORS);
  gsize result_size;
  gboolean guard_ok;

  ASSERT_TRUE (setOption (0, "tflite-deeplab"));

  setConfig (&config, DEFAULT_COLORS, 3, 3);
  EXPECT_EQ (decode (&config, prob, 0.0f, num * 4, out, &result_size, &guard_ok), GST_FLOW_OK);
  EXPECT_EQ (result_size, num * 4);
  EXPECT_TRUE (guard_ok);
  expectLabelColors (labels, out, num);

  g_free (prob);
}

/**
 * @brief tflite-deeplab must follow the frame size of each buffer, growing
 *        and shrinking, instead of the size it decoded first.
 */
TEST_F (tensorDecoderImageSegment, tfliteDeeplabFrameSizeChanges)
{
  const guint sizes[][2] = { { 2, 2 }, { 512, 384 }, { 3, 3 }, { 128, 97 } };
  GstTensorsConfig config;
  guint s, i;

  ASSERT_TRUE (setOption (0, "tflite-deeplab"));

  for (s = 0; s < G_N_ELEMENTS (sizes); s++) {
    const guint num = sizes[s][0] * sizes[s][1];
    guint *labels = g_new0 (guint, num);
    uint32_t *out = g_new0 (uint32_t, num);
    float *prob;
    gsize result_size;
    gboolean guard_ok;

    for (i = 0; i < num; i += 5)
      labels[i] = i % DEFAULT_COLORS;
    labels[num - 1] = 7;
    prob = makeProbabilities (labels, num, DEFAULT_COLORS);

    setConfig (&config, DEFAULT_COLORS, sizes[s][0], sizes[s][1]);
    EXPECT_EQ (decode (&config, prob, 0.0f, num * 4, out, &result_size, &guard_ok), GST_FLOW_OK);
    EXPECT_EQ (result_size, num * 4);
    EXPECT_TRUE (guard_ok) << sizes[s][0] << "x" << sizes[s][1];
    EXPECT_EQ (out[num - 1] & ALPHA_MASK, ALPHA_MASK);
    expectLabelColors (labels, out, MIN (num, 64U));

    g_free (prob);
    g_free (labels);
    g_free (out);
  }
}

/**
 * @brief The frame size getOutCaps () reports follows the mode's dimension order.
 */
TEST_F (tensorDecoderImageSegment, getOutCapsFrameSize)
{
  GstTensorsConfig config;
  gint width = 0, height = 0;

  ASSERT_TRUE (setOption (0, "tflite-deeplab"));
  setConfig (&config, DEFAULT_COLORS, 5, 7);
  EXPECT_TRUE (getOutSize (&config, &width, &height));
  EXPECT_EQ (width, 5);
  EXPECT_EQ (height, 7);

  ASSERT_TRUE (setOption (0, "snpe-depth"));
  setConfig (&config, 1, 6, 8);
  EXPECT_TRUE (getOutSize (&config, &width, &height));
  EXPECT_EQ (width, 6);
  EXPECT_EQ (height, 8);

  ASSERT_TRUE (setOption (0, "snpe-deeplab"));
  setConfig (&config, 9, 4, 1);
  EXPECT_TRUE (getOutSize (&config, &width, &height));
  EXPECT_EQ (width, 9);
  EXPECT_EQ (height, 4);
}

/**
 * @brief A caps query for another frame size (e.g., a renegotiation that has
 *        not completed) must not change the size of the frame being decoded.
 */
TEST_F (tensorDecoderImageSegment, getOutCapsDoesNotResizeDecode)
{
  const guint labels[] = { 0, 3, 20, 1 };
  GstTensorsConfig config, query;
  uint32_t out[G_N_ELEMENTS (labels)];
  float *prob = makeProbabilities (labels, 4, DEFAULT_COLORS);
  gint width = 0, height = 0;
  gsize result_size;
  gboolean guard_ok;

  ASSERT_TRUE (setOption (0, "tflite-deeplab"));

  setConfig (&query, DEFAULT_COLORS, 256, 256);
  EXPECT_TRUE (getOutSize (&query, &width, &height));
  EXPECT_EQ (width, 256);

  setConfig (&config, DEFAULT_COLORS, 2, 2);
  EXPECT_EQ (decode (&config, prob, 0.0f, 16, out, &result_size, &guard_ok, FALSE), GST_FLOW_OK);
  EXPECT_EQ (result_size, 16U);
  EXPECT_TRUE (guard_ok);
  expectLabelColors (labels, out, 4);

  /* The same with an output buffer the decoder allocates */
  EXPECT_TRUE (getOutSize (&query, &width, &height));
  EXPECT_EQ (decode (&config, prob, 0.0f, 0, NULL, &result_size, &guard_ok, FALSE), GST_FLOW_OK);
  EXPECT_EQ (result_size, 16U);

  g_free (prob);
}

/**
 * @brief Changing option2 after decoding must resize the color map.
 */
TEST_F (tensorDecoderImageSegment, labelsChangeAfterDecode)
{
  GstTensorsConfig config;
  guint labels[9] = { 20, 1, 2, 3, 4, 5, 6, 7, 20 };
  float in[9];
  uint32_t out[9];
  gsize result_size;
  gboolean guard_ok;
  float *prob;
  guint i;

  ASSERT_TRUE (setOption (0, "snpe-deeplab"));
  setConfig (&config, 3, 3, 1);

  for (i = 0; i < 9; i++)
    in[i] = (float) labels[i];
  EXPECT_EQ (decode (&config, in, 0.0f, 36, out, &result_size, &guard_ok), GST_FLOW_OK);
  EXPECT_TRUE (guard_ok);
  expectLabelColors (labels, out, 9);

  /* More labels: the new ones get their own opaque colors */
  ASSERT_TRUE (setOption (1, "400"));
  labels[0] = 400;
  labels[4] = 399;
  /**
   * Pixel 8 is the last of a 3x3 frame, so the scalar loop draws it on
   * AArch64 too, and a label beyond the color map is left transparent
   * there. A NEON lane would still give such a label a color.
   */
  labels[8] = 401;
  for (i = 0; i < 9; i++)
    in[i] = (float) labels[i];
  EXPECT_EQ (decode (&config, in, 0.0f, 36, out, &result_size, &guard_ok), GST_FLOW_OK);
  EXPECT_TRUE (guard_ok);
  EXPECT_EQ (out[8], 0U);
  labels[8] = 0;
  expectLabelColors (labels, out, 9);

  /* Fewer labels: tflite-deeplab now expects 11 of them */
  ASSERT_TRUE (setOption (0, "tflite-deeplab"));
  ASSERT_TRUE (setOption (1, "10"));
  for (i = 0; i < 9; i++)
    labels[i] = i % 11;
  prob = makeProbabilities (labels, 9, DEFAULT_COLORS);
  setConfig (&config, DEFAULT_COLORS, 3, 3);
  EXPECT_EQ (decode (&config, prob, 0.0f, 36, out, &result_size, &guard_ok), GST_FLOW_ERROR);
  g_free (prob);

  labels[8] = 10;
  prob = makeProbabilities (labels, 9, 11);
  setConfig (&config, 11, 3, 3);
  EXPECT_EQ (decode (&config, prob, 0.0f, 36, out, &result_size, &guard_ok), GST_FLOW_OK);
  EXPECT_TRUE (guard_ok);
  expectLabelColors (labels, out, 9);
  g_free (prob);
}

/**
 * @brief option2 at its limits is accepted and decodes its last label.
 */
TEST_F (tensorDecoderImageSegment, labelsLimits)
{
  const guint max_labels[] = { 16777214, 0, 1, 16777214 };
  const guint labels[] = { 1, 0, 0, 1 };
  GstTensorsConfig config;
  uint32_t out[4];
  float in[4];
  float *prob;
  gsize result_size;
  gboolean guard_ok;
  guint i;

  /* The most labels; 16777214 is exact in float32 */
  ASSERT_TRUE (setOption (0, "snpe-deeplab"));
  EXPECT_TRUE (setOption (1, "16777214"));
  for (i = 0; i < 4; i++)
    in[i] = (float) max_labels[i];
  setConfig (&config, 2, 2, 1);
  EXPECT_EQ (decode (&config, in, 0.0f, 16, out, &result_size, &guard_ok), GST_FLOW_OK);
  EXPECT_TRUE (guard_ok);
  expectLabelColors (max_labels, out, 4);

  /* The fewest labels */
  ASSERT_TRUE (setOption (0, "tflite-deeplab"));
  EXPECT_TRUE (setOption (1, "1"));

  prob = makeProbabilities (labels, 4, 2);
  setConfig (&config, 2, 2, 2);
  EXPECT_EQ (decode (&config, prob, 0.0f, 16, out, &result_size, &guard_ok), GST_FLOW_OK);
  EXPECT_TRUE (guard_ok);
  expectLabelColors (labels, out, 4);
  g_free (prob);
}

/**
 * @brief An option2 that is not a positive number of labels within the limit
 *        is refused and the previous number of labels is kept.
 */
TEST_F (tensorDecoderImageSegment, labelsInvalid_n)
{
  const gchar *invalid[] = { "0", "-1", "abc", "16777215", "4294967295",
    "4294967296", "18446744073709551616", "", "20abc", "2.5", "0x14" };
  const guint labels[] = { 20, 0, 3, 20 };
  GstTensorsConfig config;
  uint32_t out[4];
  float *prob;
  gsize result_size;
  gboolean guard_ok;
  guint i;

  ASSERT_TRUE (setOption (0, "tflite-deeplab"));

  for (i = 0; i < G_N_ELEMENTS (invalid); i++)
    EXPECT_FALSE (setOption (1, invalid[i])) << invalid[i];

  prob = makeProbabilities (labels, 4, DEFAULT_COLORS);
  setConfig (&config, DEFAULT_COLORS, 2, 2);
  EXPECT_EQ (decode (&config, prob, 0.0f, 16, out, &result_size, &guard_ok), GST_FLOW_OK);
  EXPECT_TRUE (guard_ok);
  expectLabelColors (labels, out, 4);
  g_free (prob);
}

/**
 * @brief Decoding is refused without a mode, an output buffer, a config or
 *        a matching input tensor.
 */
TEST_F (tensorDecoderImageSegment, decodeInvalid_n)
{
  GstTensorsConfig config;
  GstTensorMemory input;
  GstBuffer *outbuf;
  uint32_t out[4];
  float prob[5 * 4] = { 0.0f };
  gsize result_size;
  gboolean guard_ok;

  setConfig (&config, DEFAULT_COLORS, 2, 2);
  input.size = sizeof (prob);
  input.data = prob;

  outbuf = gst_buffer_new ();
  EXPECT_EQ (decoder->decode (&pdata, &config, &input, outbuf), GST_FLOW_ERROR);

  ASSERT_TRUE (setOption (0, "tflite-deeplab"));
  EXPECT_EQ (decoder->decode (&pdata, &config, &input, NULL), GST_FLOW_ERROR);
  EXPECT_EQ (decoder->decode (&pdata, NULL, &input, outbuf), GST_FLOW_ERROR);
  gst_buffer_unref (outbuf);

  /* The number of labels does not match option2 */
  setConfig (&config, 5, 2, 2);
  EXPECT_EQ (decode (&config, prob, 0.0f, 16, out, &result_size, &guard_ok), GST_FLOW_ERROR);
  EXPECT_TRUE (guard_ok);

  /* Not float32 */
  ASSERT_TRUE (setOption (0, "snpe-depth"));
  setConfig (&config, 1, 2, 2, _NNS_INT32);
  EXPECT_EQ (decode (&config, prob, 0.0f, 16, out, &result_size, &guard_ok), GST_FLOW_ERROR);
  EXPECT_TRUE (guard_ok);
}

/**
 * @brief A frame whose pixel count does not fit in guint is refused before
 *        anything is allocated or read, in every mode.
 */
TEST_F (tensorDecoderImageSegment, decodeTooLargeFrame_n)
{
  const struct {
    const gchar *mode;
    guint dim[3];
  } cases[] = {
    { "snpe-depth", { 1, 65536, 65536 } },
    { "snpe-deeplab", { 65536, 65536, 1 } },
    { "tflite-deeplab", { DEFAULT_COLORS, 65536, 65536 } },
    { "snpe-depth", { 1, 4294967295U, 2 } },
  };
  GstTensorsConfig config;
  GstTensorMemory input;
  float in[16] = { 0.0f };
  guint i;

  /* The input is much smaller than the config says: it must not be read */
  input.size = sizeof (in);
  input.data = in;

  for (i = 0; i < G_N_ELEMENTS (cases); i++) {
    GstBuffer *outbuf = gst_buffer_new_allocate (NULL, 16, NULL);
    GstCaps *caps;

    ASSERT_TRUE (setOption (0, cases[i].mode));
    setConfig (&config, cases[i].dim[0], cases[i].dim[1], cases[i].dim[2]);
    caps = decoder->getOutCaps (&pdata, &config);
    if (caps != NULL)
      gst_caps_unref (caps);

    EXPECT_EQ (decoder->decode (&pdata, &config, &input, outbuf), GST_FLOW_ERROR)
        << cases[i].mode << " " << cases[i].dim[1];
    EXPECT_EQ (gst_buffer_get_size (outbuf), 16U);
    gst_buffer_unref (outbuf);
  }
}

/**
 * @brief Record the size of every buffer reaching the fakesink.
 */
static void
recordBufferSize (GstElement *sink, GstBuffer *buffer, GstPad *pad, gpointer user_data)
{
  GArray *sizes = (GArray *) user_data;
  gsize size = gst_buffer_get_size (buffer);

  (void) sink;
  (void) pad;
  g_array_append_val (sizes, size);
}

/**
 * @brief Push one zeroed tflite-deeplab frame of the given size into appsrc.
 */
static void
pushFrame (GstElement *src, guint colors, guint width, guint height)
{
  gchar *caps_str;
  GstCaps *caps;
  GstBuffer *buf;
  GstFlowReturn ret;

  caps_str = g_strdup_printf ("other/tensors,format=static,num_tensors=1,"
                              "types=float32,dimensions=%u:%u:%u:1,framerate=0/1",
      colors, width, height);
  caps = gst_caps_from_string (caps_str);
  g_object_set (src, "caps", caps, NULL);
  gst_caps_unref (caps);
  g_free (caps_str);

  buf = gst_buffer_new_allocate (
      NULL, (gsize) colors * width * height * sizeof (float), NULL);
  gst_buffer_memset (buf, 0, 0, gst_buffer_get_size (buf));

  /* The push-buffer action signal does not take the buffer */
  g_signal_emit_by_name (src, "push-buffer", buf, &ret);
  gst_buffer_unref (buf);
  EXPECT_EQ (ret, GST_FLOW_OK);
}

/**
 * @brief Run tensor_decoder over tflite-deeplab frames of the given sizes.
 * @param[in] options The tensor_decoder properties
 * @param[in] sizes width and height of each frame
 * @param[in] num The number of frames
 * @param[out] out_sizes The size of each decoded buffer
 * @return GST_MESSAGE_EOS or GST_MESSAGE_ERROR, GST_MESSAGE_ANY if the
 *         pipeline could not be built and GST_MESSAGE_UNKNOWN on a timeout.
 */
static GstMessageType
runFrames (const gchar *options, const guint sizes[][2], guint num, GArray *out_sizes)
{
  gchar *pipeline_str;
  GstElement *pipeline, *src, *sink;
  GstMessageType type = GST_MESSAGE_UNKNOWN;
  GstFlowReturn ret;
  GstBus *bus;
  GstMessage *msg;
  guint i;

  pipeline_str = g_strdup_printf ("appsrc name=src format=time ! "
                                  "tensor_decoder mode=image_segment %s ! "
                                  "fakesink name=sink signal-handoffs=true sync=false",
      options);
  pipeline = gst_parse_launch (pipeline_str, NULL);
  g_free (pipeline_str);
  if (pipeline == NULL)
    return GST_MESSAGE_ANY;

  src = gst_bin_get_by_name (GST_BIN (pipeline), "src");
  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");
  g_signal_connect (sink, "handoff", G_CALLBACK (recordBufferSize), out_sizes);

  /* appsrc prerolls only once a frame is pushed, so do not wait for PLAYING */
  EXPECT_NE (gst_element_set_state (pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);

  for (i = 0; i < num; i++)
    pushFrame (src, DEFAULT_COLORS, sizes[i][0], sizes[i][1]);
  g_signal_emit_by_name (src, "end-of-stream", &ret);
  EXPECT_EQ (ret, GST_FLOW_OK);

  bus = gst_element_get_bus (pipeline);
  msg = gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  if (msg != NULL) {
    type = GST_MESSAGE_TYPE (msg);
    gst_message_unref (msg);
  }
  gst_object_unref (bus);

  setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT);
  gst_object_unref (src);
  gst_object_unref (sink);
  gst_object_unref (pipeline);

  return type;
}

/**
 * @brief tensor_decoder renegotiates to a larger and a smaller frame mid-stream.
 */
TEST (tensorDecoderImageSegmentPipeline, renegotiateFrameSize)
{
  const guint sizes[][2] = { { 2, 2 }, { 512, 384 }, { 3, 3 } };
  GArray *out_sizes = g_array_new (FALSE, FALSE, sizeof (gsize));
  guint i;

  EXPECT_EQ (runFrames ("option1=tflite-deeplab", sizes, G_N_ELEMENTS (sizes), out_sizes),
      GST_MESSAGE_EOS);

  ASSERT_EQ (out_sizes->len, G_N_ELEMENTS (sizes));
  for (i = 0; i < out_sizes->len; i++)
    EXPECT_EQ (g_array_index (out_sizes, gsize, i), (gsize) sizes[i][0] * sizes[i][1] * 4);

  g_array_free (out_sizes, TRUE);
}

/**
 * @brief An option2 whose number of colors does not fit is refused by the
 *        element and the stream is decoded with the default labels.
 */
TEST (tensorDecoderImageSegmentPipeline, invalidLabels_n)
{
  const guint sizes[][2] = { { 3, 3 } };
  GArray *out_sizes = g_array_new (FALSE, FALSE, sizeof (gsize));

  EXPECT_EQ (runFrames ("option1=tflite-deeplab option2=4294967295", sizes, 1, out_sizes),
      GST_MESSAGE_EOS);

  ASSERT_EQ (out_sizes->len, 1U);
  EXPECT_EQ (g_array_index (out_sizes, gsize, 0), (gsize) 36);

  g_array_free (out_sizes, TRUE);
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
