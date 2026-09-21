/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_decoder_direct_video.cc
 * @date        20 Sep 2026
 * @brief       Unit test for the direct_video tensor_decoder subplugin with
 *              a tensor that is not a single video frame.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <cstring>
#include <glib.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api_util.h>
#include <tensor_typedef.h>

/**
 * @brief Bytes kept behind the video frame to see a write running past it.
 */
#define GUARD_SIZE (64U)

/**
 * @brief The value filling the output block before direct_video writes to it.
 */
#define GUARD_BYTE (0xA5U)

/**
 * @brief The direct_video subplugin with its private data, driven as tensor_decoder does.
 */
class DirectVideo
{
  public:
  /**
   * @brief Find and initialize the subplugin, then give option1 and the config.
   * @param[in] dim_str the dimension of the only tensor
   * @param[in] type the type of the only tensor
   * @param[in] format option1 of direct_video, NULL to leave it unset
   */
  DirectVideo (const gchar *dim_str, tensor_type type = _NNS_UINT8, const gchar *format = NULL)
      : dec (nnstreamer_decoder_find ("direct_video")), pdata (NULL), option_set (TRUE)
  {
    gst_tensors_config_init (&config);
    config.rate_n = 30;
    config.rate_d = 1;
    config.info.num_tensors = 1;
    config.info.info[0].type = type;
    gst_tensor_parse_dimension (dim_str, config.info.info[0].dimension);

    if (dec && dec->init (&pdata) && format)
      option_set = dec->setOption (&pdata, 0, format);
  }

  /**
   * @brief Release the private data of the subplugin.
   */
  ~DirectVideo ()
  {
    if (dec && pdata)
      dec->exit (&pdata);
    gst_tensors_config_free (&config);
  }

  /**
   * @brief Ask the video caps of the config, which tensor_decoder does before decode ().
   * @return the caps the caller unrefs, NULL if the subplugin refuses the config
   */
  GstCaps *getOutCaps ()
  {
    return dec->getOutCaps (&pdata, &config);
  }

  /**
   * @brief Ask the size of the video frame that tensor_decoder allocates for decode ().
   * @param[in] direction the pad of which the size is given, the sink pad for tensor_decoder
   */
  gsize getTransformSize (GstPadDirection direction = GST_PAD_SINK)
  {
    return dec->getTransformSize (&pdata, &config, NULL, 0, NULL, direction);
  }

  /**
   * @brief Decode @a declared_size bytes of @a block into @a outbuf.
   */
  GstFlowReturn decode (gpointer block, gsize declared_size, GstBuffer *outbuf)
  {
    GstTensorMemory input;
    GstCaps *caps = getOutCaps ();

    if (caps)
      gst_caps_unref (caps);

    input.data = block;
    input.size = declared_size;
    return dec->decode (&pdata, &config, &input, outbuf);
  }

  const GstTensorDecoderDef *dec; /**< the subplugin */
  void *pdata; /**< the private data of the subplugin */
  gboolean option_set; /**< FALSE if the subplugin has refused option1 */
  GstTensorsConfig config; /**< the config of the incoming tensor */
};

/**
 * @brief Get a block of @a size bytes filled with an ascending pattern.
 */
static guint8 *
new_pattern (gsize size)
{
  guint8 *block = (guint8 *) g_malloc0 (size);

  for (gsize i = 0; i < size; i++)
    block[i] = (guint8) (i + 1);
  return block;
}

/**
 * @brief Get a buffer of @a size bytes whose memory block keeps GUARD_SIZE more bytes behind it.
 * @param[in] size the size of the buffer
 * @param[out] block the whole block, GUARD_BYTE-filled, which the caller frees after the buffer
 */
static GstBuffer *
new_guarded_buffer (gsize size, guint8 **block)
{
  GstBuffer *buf = gst_buffer_new ();
  GstMemory *mem;

  *block = (guint8 *) g_malloc (size + GUARD_SIZE);
  memset (*block, GUARD_BYTE, size + GUARD_SIZE);
  mem = gst_memory_new_wrapped (
      (GstMemoryFlags) 0, *block, size + GUARD_SIZE, 0, size, NULL, NULL);
  gst_buffer_append_memory (buf, mem);
  return buf;
}

/**
 * @brief Count the bytes of the guard area that are not GUARD_BYTE any longer.
 */
static guint
count_broken_guard (const guint8 *block, gsize size)
{
  guint broken = 0;

  for (gsize i = size; i < size + GUARD_SIZE; i++)
    if (block[i] != GUARD_BYTE)
      broken++;
  return broken;
}

/**
 * @brief Get a harness of tensor_decoder in direct_video mode fed with a static uint8 tensor.
 */
static GstHarness *
new_direct_video_harness (const gchar *dim_str)
{
  GstElement *dec = gst_element_factory_make ("tensor_decoder", NULL);
  GstTensorsConfig config;
  GstHarness *h;

  if (!dec)
    return NULL;
  gst_object_ref_sink (dec);
  g_object_set (dec, "mode", "direct_video", NULL);

  h = gst_harness_new_with_element (dec, "sink", "src");
  gst_object_unref (dec);
  if (!h)
    return NULL;

  gst_tensors_config_init (&config);
  config.rate_n = 0;
  config.rate_d = 1;
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension (dim_str, config.info.info[0].dimension);
  gst_harness_set_src_caps (h, gst_tensors_caps_from_config (&config));
  gst_tensors_config_free (&config);

  return h;
}

/**
 * @brief Get a zero-filled buffer of @a size bytes with an ascending pattern.
 */
static GstBuffer *
new_pattern_buffer (gsize size)
{
  GstBuffer *buf = gst_buffer_new_allocate (NULL, size, NULL);
  guint8 *pattern = new_pattern (size);

  gst_buffer_memset (buf, 0, 0, size);
  gst_buffer_fill (buf, 0, pattern, size);
  g_free (pattern);
  return buf;
}

/**
 * @brief A frame whose rows need no padding is copied as it is.
 */
TEST (testDecoderDirectVideo, decodeNoPadding)
{
  DirectVideo dv ("4:4:2");
  guint8 *in = new_pattern (32);
  GstBuffer *outbuf = gst_buffer_new ();
  GstMapInfo map;

  ASSERT_TRUE (dv.dec != NULL);
  ASSERT_EQ (GST_FLOW_OK, dv.decode (in, 32, outbuf));

  ASSERT_TRUE (gst_buffer_map (outbuf, &map, GST_MAP_READ));
  ASSERT_EQ (map.size, 32U);
  EXPECT_EQ (0, memcmp (map.data, in, 32));
  gst_buffer_unmap (outbuf, &map);

  gst_buffer_unref (outbuf);
  g_free (in);
}

/**
 * @brief Each row of a frame is padded to a multiple of 4 bytes.
 */
TEST (testDecoderDirectVideo, decodePadding)
{
  DirectVideo dv ("3:3:2");
  guint8 *in = new_pattern (18);
  GstBuffer *outbuf = gst_buffer_new ();
  GstMapInfo map;

  ASSERT_TRUE (dv.dec != NULL);
  ASSERT_EQ (GST_FLOW_OK, dv.decode (in, 18, outbuf));

  ASSERT_TRUE (gst_buffer_map (outbuf, &map, GST_MAP_READ));
  ASSERT_EQ (map.size, 24U);
  EXPECT_EQ (0, memcmp (map.data, in, 9));
  EXPECT_EQ (0, memcmp (map.data + 12, in + 9, 9));
  gst_buffer_unmap (outbuf, &map);

  gst_buffer_unref (outbuf);
  g_free (in);
}

/**
 * @brief A buffer that already has the memory, as tensor_decoder gives, is filled in place.
 */
TEST (testDecoderDirectVideo, decodeIntoGivenBuffer)
{
  DirectVideo dv ("4:4:2");
  guint8 *in = new_pattern (32);
  guint8 *block = NULL;
  GstBuffer *outbuf = new_guarded_buffer (32, &block);

  ASSERT_TRUE (dv.dec != NULL);
  ASSERT_EQ (GST_FLOW_OK, dv.decode (in, 32, outbuf));

  EXPECT_EQ (gst_buffer_get_size (outbuf), 32U);
  EXPECT_EQ (0, memcmp (block, in, 32));
  EXPECT_EQ (count_broken_guard (block, 32), 0U);

  gst_buffer_unref (outbuf);
  g_free (block);
  g_free (in);
}

/**
 * @brief A given buffer larger than the frame is shrunk to the frame.
 */
TEST (testDecoderDirectVideo, decodeIntoLargerBuffer)
{
  DirectVideo dv ("4:4:2");
  guint8 *in = new_pattern (32);
  guint8 *block = NULL;
  GstBuffer *outbuf = new_guarded_buffer (48, &block);

  ASSERT_TRUE (dv.dec != NULL);
  ASSERT_EQ (GST_FLOW_OK, dv.decode (in, 32, outbuf));

  EXPECT_EQ (gst_buffer_get_size (outbuf), 32U);
  EXPECT_EQ (0, memcmp (block, in, 32));
  EXPECT_EQ (block[32], GUARD_BYTE);

  gst_buffer_unref (outbuf);
  g_free (block);
  g_free (in);
}

/**
 * @brief A 16-bit gray frame takes two bytes for each pixel.
 */
TEST (testDecoderDirectVideo, decodeGray16)
{
  DirectVideo dv ("1:4:2", _NNS_UINT16, "GRAY16_LE");
  guint8 *in = new_pattern (16);
  GstBuffer *outbuf = gst_buffer_new ();
  GstCaps *caps;
  GstMapInfo map;

  ASSERT_TRUE (dv.dec != NULL);
  EXPECT_TRUE (dv.option_set);

  caps = dv.getOutCaps ();
  ASSERT_TRUE (caps != NULL);
  EXPECT_STREQ ("GRAY16_LE",
      gst_structure_get_string (gst_caps_get_structure (caps, 0), "format"));
  gst_caps_unref (caps);

  ASSERT_EQ (GST_FLOW_OK, dv.decode (in, 16, outbuf));
  ASSERT_TRUE (gst_buffer_map (outbuf, &map, GST_MAP_READ));
  ASSERT_EQ (map.size, 16U);
  EXPECT_EQ (0, memcmp (map.data, in, 16));
  gst_buffer_unmap (outbuf, &map);

  gst_buffer_unref (outbuf);
  g_free (in);
}

/**
 * @brief The size of the video frame counts the padding of each row.
 */
TEST (testDecoderDirectVideo, transformSize)
{
  DirectVideo unpadded ("4:4:2");
  DirectVideo padded ("3:3:2");
  DirectVideo gray16 ("1:4:2", _NNS_UINT16, "GRAY16_LE");

  ASSERT_TRUE (unpadded.dec != NULL);
  EXPECT_EQ (unpadded.getTransformSize (), 32U);
  EXPECT_EQ (padded.getTransformSize (), 24U);
  EXPECT_EQ (gray16.getTransformSize (), 16U);
  EXPECT_EQ (padded.getTransformSize (GST_PAD_SRC), 0U);
}

/**
 * @brief The size of a video frame is 0 only if the frame cannot be addressed.
 * @details 2^30 + 8 rows of 1 byte are 2^32 + 32 bytes with the padding, which
 *          gsize of 64 bits holds and gsize of 32 bits cannot.
 */
TEST (testDecoderDirectVideo, transformSizeUnaddressable_n)
{
  DirectVideo no_height ("4:4");
  DirectVideo row_overflow ("4:1073741824:1");
  DirectVideo frame_overflow ("4:315916329:1824726041", _NNS_FLOAT64);
  DirectVideo padding_overflow ("1:1:1073741832");
  guint64 expected = (sizeof (gsize) > 4) ? G_GUINT64_CONSTANT (4294967328) : 0;

  ASSERT_TRUE (no_height.dec != NULL);
  EXPECT_EQ (no_height.getTransformSize (), 0U);
  EXPECT_EQ (row_overflow.getTransformSize (), 0U);
  EXPECT_EQ (frame_overflow.getTransformSize (), 0U);
  EXPECT_EQ ((guint64) padding_overflow.getTransformSize (), expected);
}

/**
 * @brief A tensor larger than the frame is refused and nothing is written behind the frame.
 */
TEST (testDecoderDirectVideo, decodeInputTooLarge_n)
{
  DirectVideo dv ("4:4:2");
  guint8 *in = new_pattern (48);
  guint8 *block = NULL;
  GstBuffer *outbuf = new_guarded_buffer (32, &block);

  ASSERT_TRUE (dv.dec != NULL);
  EXPECT_EQ (GST_FLOW_ERROR, dv.decode (in, 48, outbuf));
  EXPECT_EQ (count_broken_guard (block, 32), 0U);

  gst_buffer_unref (outbuf);
  g_free (block);
  g_free (in);
}

/**
 * @brief A tensor smaller than the frame is refused where the rows need no padding.
 */
TEST (testDecoderDirectVideo, decodeInputTooSmall_n)
{
  DirectVideo dv ("4:4:2");
  guint8 *in = new_pattern (32);
  GstBuffer *outbuf = gst_buffer_new ();

  ASSERT_TRUE (dv.dec != NULL);
  EXPECT_EQ (GST_FLOW_ERROR, dv.decode (in, 16, outbuf));
  EXPECT_EQ (gst_buffer_get_size (outbuf), 0U);

  gst_buffer_unref (outbuf);
  g_free (in);
}

/**
 * @brief A tensor smaller than the frame is refused where the rows are padded.
 * @details The block holds a whole frame and only the declared size is short,
 *          so reading each row of the frame stays inside the block.
 */
TEST (testDecoderDirectVideo, decodeInputTooSmallPadding_n)
{
  DirectVideo dv ("3:3:2");
  guint8 *in = new_pattern (18);
  GstBuffer *outbuf = gst_buffer_new ();

  ASSERT_TRUE (dv.dec != NULL);
  EXPECT_EQ (GST_FLOW_ERROR, dv.decode (in, 9, outbuf));
  EXPECT_EQ (gst_buffer_get_size (outbuf), 0U);

  gst_buffer_unref (outbuf);
  g_free (in);
}

/**
 * @brief A tensor of multiple frames is refused and nothing is written behind the frame.
 */
TEST (testDecoderDirectVideo, decodeMultipleFrames_n)
{
  DirectVideo dv ("4:4:2:2");
  guint8 *in = new_pattern (64);
  guint8 *block = NULL;
  GstBuffer *outbuf = new_guarded_buffer (32, &block);

  ASSERT_TRUE (dv.dec != NULL);
  ASSERT_EQ (gst_tensors_info_get_size (&dv.config.info, 0), 64U);
  EXPECT_EQ (GST_FLOW_ERROR, dv.decode (in, 64, outbuf));
  EXPECT_EQ (count_broken_guard (block, 32), 0U);

  gst_buffer_unref (outbuf);
  g_free (block);
  g_free (in);
}

/**
 * @brief A tensor without the height, whose frame is 0 byte, is refused.
 */
TEST (testDecoderDirectVideo, decodeNoHeight_n)
{
  DirectVideo dv ("4:1");
  guint8 *in = new_pattern (4);
  GstBuffer *outbuf = gst_buffer_new ();

  ASSERT_TRUE (dv.dec != NULL);
  ASSERT_EQ (gst_tensors_info_get_size (&dv.config.info, 0), 4U);
  EXPECT_EQ (GST_FLOW_ERROR, dv.decode (in, 4, outbuf));
  EXPECT_EQ (gst_buffer_get_size (outbuf), 0U);

  gst_buffer_unref (outbuf);
  g_free (in);
}

/**
 * @brief A tensor without the width, whose frame is 0 byte as the tensor given here is, is refused.
 */
TEST (testDecoderDirectVideo, decodeNoWidth_n)
{
  DirectVideo dv ("4:4:5");
  guint8 *in = new_pattern (4);
  guint8 *block = NULL;
  GstBuffer *outbuf = new_guarded_buffer (32, &block);

  ASSERT_TRUE (dv.dec != NULL);
  dv.config.info.info[0].dimension[1] = 0;
  EXPECT_EQ (GST_FLOW_ERROR, dv.decode (in, 0, outbuf));
  EXPECT_EQ (gst_buffer_get_size (outbuf), 32U);

  gst_buffer_unref (outbuf);
  g_free (block);
  g_free (in);
}

/**
 * @brief A dimension whose frame size does not fit in gsize is refused.
 * @details The row, 4 * 315916329, is a valid one without padding, and the
 *          frame, 4 * 315916329 * 1824726041 * 8 bytes, is 2^64 + 32. It wraps
 *          64 bits to 32, which is the size of the tensor given here.
 */
TEST (testDecoderDirectVideo, decodeFrameSizeOverflow_n)
{
  DirectVideo dv ("4:315916329:1824726041", _NNS_FLOAT64);
  guint8 *in = new_pattern (32);
  guint8 *block = NULL;
  GstBuffer *outbuf = new_guarded_buffer (32, &block);

  ASSERT_TRUE (dv.dec != NULL);
  ASSERT_EQ (dv.config.info.info[0].dimension[1], 315916329U);
  ASSERT_EQ (dv.config.info.info[0].dimension[2], 1824726041U);
  EXPECT_EQ (GST_FLOW_ERROR, dv.decode (in, 32, outbuf));
  EXPECT_EQ (block[0], GUARD_BYTE);

  gst_buffer_unref (outbuf);
  g_free (block);
  g_free (in);
}

/**
 * @brief tensor_decoder hands a frame to direct_video and pushes the padded video frame.
 */
TEST (testDecoderDirectVideo, elementDecode)
{
  GstHarness *h = new_direct_video_harness ("3:3:2");
  guint8 *pattern = new_pattern (18);
  GstBuffer *out;
  GstMapInfo map;

  ASSERT_TRUE (h != NULL);
  ASSERT_EQ (GST_FLOW_OK, gst_harness_push (h, new_pattern_buffer (18)));

  out = gst_harness_pull (h);
  ASSERT_TRUE (out != NULL);
  ASSERT_TRUE (gst_buffer_map (out, &map, GST_MAP_READ));
  ASSERT_EQ (map.size, 24U);
  EXPECT_EQ (0, memcmp (map.data, pattern, 9));
  EXPECT_EQ (0, memcmp (map.data + 12, pattern + 9, 9));
  gst_buffer_unmap (out, &map);

  gst_buffer_unref (out);
  g_free (pattern);
  gst_harness_teardown (h);
}

/**
 * @brief tensor_decoder accepts a tensor of multiple frames by its caps, and direct_video refuses it.
 * @details The tensor is 8 bytes and the frame is 4 bytes; the 7 alignment
 *          bytes of the 4-byte system memory keep the copy that is not
 *          refused inside the block.
 */
TEST (testDecoderDirectVideo, elementMultipleFrames_n)
{
  GstHarness *h = new_direct_video_harness ("4:1:1:2");

  ASSERT_TRUE (h != NULL);
  EXPECT_EQ (GST_FLOW_ERROR, gst_harness_push (h, new_pattern_buffer (8)));

  gst_harness_teardown (h);
}

/**
 * @brief tensor_decoder accepts a tensor without the height by its caps, and direct_video refuses it.
 * @details The tensor is 4 bytes and the frame is 0 byte; the 7 alignment
 *          bytes of the 0-byte system memory keep the copy that is not
 *          refused inside the block.
 */
TEST (testDecoderDirectVideo, elementNoHeight_n)
{
  GstHarness *h = new_direct_video_harness ("4:1");

  ASSERT_TRUE (h != NULL);
  EXPECT_EQ (GST_FLOW_ERROR, gst_harness_push (h, new_pattern_buffer (4)));

  gst_harness_teardown (h);
}

/**
 * @brief A frame that fits in gsize of 32 bits while its padded video frame does not is refused.
 * @details The tensor is 2^30 + 8 rows of 1 byte and the video frame is
 *          2^32 + 32 bytes. It is a valid frame where gsize is 64 bits, which
 *          this case cannot give for real, so it runs only where gsize is 32
 *          bits. The declared size is never read there, for it is refused.
 */
TEST (testDecoderDirectVideo, decodeOutputSizeOverflow_n)
{
  DirectVideo dv ("1:1:1073741832");
  guint8 *in;
  GstBuffer *outbuf;

  if (sizeof (gsize) > 4)
    GTEST_SKIP ();

  in = new_pattern (4);
  outbuf = gst_buffer_new ();

  ASSERT_TRUE (dv.dec != NULL);
  EXPECT_EQ (GST_FLOW_ERROR, dv.decode (in, 1073741832U, outbuf));
  EXPECT_EQ (gst_buffer_get_size (outbuf), 0U);

  gst_buffer_unref (outbuf);
  g_free (in);
}

/**
 * @brief A row that does not fit in 32 bits is refused even if the tensor has the size of the frame.
 * @details 4 * 2^30 is 0 in the 32-bit arithmetic that sizes the output, while the
 *          frame is 4 GiB. The declared size is never read, for it is refused.
 *          Where gsize is 32 bits, the row wraps gsize as well and the declared
 *          size is 0; it is the checked multiplication that refuses it there.
 *          This and the case above are the last ones, for the copy that is not
 *          refused ends the process.
 */
TEST (testDecoderDirectVideo, decodeRowSizeOverflow_n)
{
  DirectVideo dv ("4:1073741824:1");
  guint8 *in = new_pattern (4);
  guint8 *block = NULL;
  GstBuffer *outbuf = new_guarded_buffer (32, &block);
  gsize declared = (gsize) G_MAXUINT32 + 1U;

  ASSERT_TRUE (dv.dec != NULL);
  ASSERT_EQ (dv.config.info.info[0].dimension[1], 1073741824U);
  EXPECT_EQ (GST_FLOW_ERROR, dv.decode (in, declared, outbuf));
  EXPECT_EQ (gst_buffer_get_size (outbuf), 32U);

  gst_buffer_unref (outbuf);
  g_free (block);
  g_free (in);
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
