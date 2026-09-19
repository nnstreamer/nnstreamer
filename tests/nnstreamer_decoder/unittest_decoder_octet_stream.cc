/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_decoder_octet_stream.cc
 * @date        18 Sep 2026
 * @brief       Unit test for the octet_stream tensor_decoder subplugin with
 *              peer-controlled (possibly malformed) input.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <cstring>
#include <glib.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api_util.h>
#include <tensor_typedef.h>
#include <vector>

/**
 * @brief The fixed header size (in bytes) that a flexible-format tensor carries in front of its data.
 */
static gsize
meta_header_size (void)
{
  GstTensorMetaInfo meta;

  gst_tensor_meta_info_init (&meta);
  return gst_tensor_meta_info_get_header_size (&meta);
}

/**
 * @brief Build a meta header describing @a dim_str of @a type, followed by @a payload_size bytes.
 * @note The declared data size follows @a dim_str, which is what makes a short
 *       @a payload_size a tensor that describes more data than it carries.
 */
static std::vector<guint8>
build_flex_memory (tensor_type type, const gchar *dim_str, gsize payload_size)
{
  GstTensorInfo info;
  GstTensorMetaInfo meta;
  gsize hsize;
  std::vector<guint8> out;

  gst_tensor_info_init (&info);
  info.type = type;
  gst_tensor_parse_dimension (dim_str, info.dimension);
  gst_tensor_info_convert_to_meta (&info, &meta);

  hsize = gst_tensor_meta_info_get_header_size (&meta);
  out.resize (hsize + payload_size, 0);
  gst_tensor_meta_info_update_header (&meta, out.data ());

  for (gsize i = 0; i < payload_size; i++)
    out[hsize + i] = (guint8) (i + 1);

  gst_tensor_info_free (&info);
  return out;
}

/**
 * @brief Run octet_stream's decode () over a single tensor built from @a bytes.
 * @param[in] dec the octet_stream subplugin
 * @param[in] format the tensor format of the config handed to the subplugin
 * @param[in] bytes the memory the subplugin reads
 * @param[in] declared the config dimension, used by the static format only
 * @param[out] outbuf the buffer the subplugin fills, which the caller unrefs
 */
static GstFlowReturn
decode_one (const GstTensorDecoderDef *dec, tensor_format format,
    const std::vector<guint8> &bytes, const gchar *declared, GstBuffer **outbuf)
{
  GstTensorsConfig config;
  GstTensorMemory input;
  GstFlowReturn ret;

  gst_tensors_config_init (&config);
  config.rate_n = 0;
  config.rate_d = 1;
  config.info.format = format;
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension (declared, config.info.info[0].dimension);

  input.size = bytes.size ();
  input.data = g_malloc (input.size ? input.size : 1);
  if (input.size)
    memcpy (input.data, bytes.data (), input.size);

  *outbuf = gst_buffer_new ();
  ret = dec->decode (NULL, &config, &input, *outbuf);

  g_free (input.data);
  gst_tensors_config_free (&config);
  return ret;
}

/**
 * @brief Positive: a static tensor is copied whole.
 */
TEST (testDecoderOctetStream, decodeStatic)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("octet_stream");
  std::vector<guint8> bytes = { 1, 2, 3, 4 };
  GstBuffer *outbuf = NULL;
  GstMapInfo map;

  ASSERT_TRUE (dec != NULL);

  ASSERT_EQ (GST_FLOW_OK,
      decode_one (dec, _NNS_TENSOR_FORMAT_STATIC, bytes, "4:1:1:1", &outbuf));
  ASSERT_TRUE (outbuf != NULL);
  ASSERT_EQ (gst_buffer_n_memory (outbuf), 1U);
  ASSERT_TRUE (gst_buffer_map (outbuf, &map, GST_MAP_READ));
  EXPECT_EQ (map.size, bytes.size ());
  EXPECT_EQ (0, memcmp (map.data, bytes.data (), map.size));
  gst_buffer_unmap (outbuf, &map);

  gst_buffer_unref (outbuf);
}

/**
 * @brief Positive: a flexible tensor loses its meta header and keeps its payload.
 */
TEST (testDecoderOctetStream, decodeFlexible)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("octet_stream");
  std::vector<guint8> bytes = build_flex_memory (_NNS_UINT8, "4:1:1:1", 4U);
  GstBuffer *outbuf = NULL;
  GstMapInfo map;

  ASSERT_TRUE (dec != NULL);

  ASSERT_EQ (GST_FLOW_OK,
      decode_one (dec, _NNS_TENSOR_FORMAT_FLEXIBLE, bytes, "1:1:1:1", &outbuf));
  ASSERT_TRUE (outbuf != NULL);
  ASSERT_EQ (gst_buffer_n_memory (outbuf), 1U);
  ASSERT_TRUE (gst_buffer_map (outbuf, &map, GST_MAP_READ));
  ASSERT_EQ (map.size, 4U);
  EXPECT_EQ (0, memcmp (map.data, bytes.data () + meta_header_size (), map.size));
  gst_buffer_unmap (outbuf, &map);

  gst_buffer_unref (outbuf);
}

/**
 * @brief Positive: more than one tensor is decoded in one call.
 */
TEST (testDecoderOctetStream, decodeMultipleTensors)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("octet_stream");
  GstTensorsConfig config;
  GstTensorMemory input[2];
  GstBuffer *outbuf;
  guint i;

  ASSERT_TRUE (dec != NULL);

  gst_tensors_config_init (&config);
  config.rate_n = 0;
  config.rate_d = 1;
  config.info.format = _NNS_TENSOR_FORMAT_STATIC;
  config.info.num_tensors = 2;

  for (i = 0; i < 2; i++) {
    config.info.info[i].type = _NNS_UINT8;
    gst_tensor_parse_dimension ("2:1:1:1", config.info.info[i].dimension);
    input[i].size = 2;
    input[i].data = g_malloc (2);
    ((guint8 *) input[i].data)[0] = (guint8) (i + 1);
    ((guint8 *) input[i].data)[1] = (guint8) (i + 1);
  }

  outbuf = gst_buffer_new ();
  EXPECT_EQ (GST_FLOW_OK, dec->decode (NULL, &config, input, outbuf));
  EXPECT_EQ (gst_buffer_n_memory (outbuf), 2U);
  EXPECT_EQ (gst_buffer_get_size (outbuf), 4U);

  for (i = 0; i < 2; i++)
    g_free (input[i].data);
  gst_buffer_unref (outbuf);
  gst_tensors_config_free (&config);
}

/**
 * @brief Positive: the subplugin describes itself and answers the caps query.
 */
TEST (testDecoderOctetStream, getOutCaps)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("octet_stream");
  GstTensorsConfig config;
  GstCaps *caps;

  ASSERT_TRUE (dec != NULL);
  EXPECT_TRUE (dec->setOption != NULL);
  EXPECT_TRUE (dec->setOption (NULL, 0, "unused"));

  gst_tensors_config_init (&config);
  config.rate_n = 5;
  config.rate_d = 1;
  config.info.format = _NNS_TENSOR_FORMAT_STATIC;
  config.info.num_tensors = 1;

  caps = dec->getOutCaps (NULL, &config);
  ASSERT_TRUE (caps != NULL);
  ASSERT_EQ (gst_caps_get_size (caps), 1U);
  EXPECT_STREQ ("application/octet-stream",
      gst_structure_get_name (gst_caps_get_structure (caps, 0)));
  gst_caps_unref (caps);

  gst_tensors_config_free (&config);
}

/**
 * @brief Negative: a NULL parameter is refused instead of dereferenced.
 */
TEST (testDecoderOctetStream, nullParam_n)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("octet_stream");
  GstTensorsConfig config;
  GstTensorMemory input;
  GstBuffer *outbuf;
  guint8 data[4] = { 0 };

  ASSERT_TRUE (dec != NULL);

  gst_tensors_config_init (&config);
  config.info.format = _NNS_TENSOR_FORMAT_STATIC;
  config.info.num_tensors = 1;
  config.info.info[0].type = _NNS_UINT8;
  gst_tensor_parse_dimension ("4:1:1:1", config.info.info[0].dimension);
  input.data = data;
  input.size = sizeof (data);
  outbuf = gst_buffer_new ();

  EXPECT_EQ (GST_FLOW_ERROR, dec->decode (NULL, NULL, &input, outbuf));
  EXPECT_EQ (GST_FLOW_ERROR, dec->decode (NULL, &config, NULL, outbuf));
  EXPECT_EQ (GST_FLOW_ERROR, dec->decode (NULL, &config, &input, NULL));

  gst_buffer_unref (outbuf);
  gst_tensors_config_free (&config);
}

/**
 * @brief Negative: a flexible tensor shorter than the meta header is refused.
 */
TEST (testDecoderOctetStream, flexibleHeaderTooShort_n)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("octet_stream");
  std::vector<guint8> bytes = build_flex_memory (_NNS_UINT8, "4:1:1:1", 4U);
  GstBuffer *outbuf = NULL;

  bytes.resize (meta_header_size () - 1U);

  ASSERT_TRUE (dec != NULL);

  EXPECT_EQ (GST_FLOW_ERROR,
      decode_one (dec, _NNS_TENSOR_FORMAT_FLEXIBLE, bytes, "1:1:1:1", &outbuf));
  EXPECT_EQ (gst_buffer_n_memory (outbuf), 0U);

  gst_buffer_unref (outbuf);
}

/**
 * @brief Negative: a flexible tensor without a valid meta header is refused.
 */
TEST (testDecoderOctetStream, flexibleBrokenHeader_n)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("octet_stream");
  std::vector<guint8> bytes (meta_header_size () + 4U, 0);
  GstBuffer *outbuf = NULL;

  ASSERT_TRUE (dec != NULL);

  EXPECT_EQ (GST_FLOW_ERROR,
      decode_one (dec, _NNS_TENSOR_FORMAT_FLEXIBLE, bytes, "1:1:1:1", &outbuf));
  EXPECT_EQ (gst_buffer_n_memory (outbuf), 0U);

  gst_buffer_unref (outbuf);
}

/**
 * @brief Negative: a flexible tensor describing more data than it carries is refused.
 * @details Without the bound the copy reads far past the end of the memory.
 */
TEST (testDecoderOctetStream, flexibleDataBeyondMemory_n)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("octet_stream");
  std::vector<guint8> bytes = build_flex_memory (_NNS_UINT8, "67108864:1:1:1", 4U);
  GstBuffer *outbuf = NULL;

  ASSERT_TRUE (dec != NULL);

  EXPECT_EQ (GST_FLOW_ERROR,
      decode_one (dec, _NNS_TENSOR_FORMAT_FLEXIBLE, bytes, "1:1:1:1", &outbuf));
  EXPECT_EQ (gst_buffer_n_memory (outbuf), 0U);

  gst_buffer_unref (outbuf);
}

/**
 * @brief Negative: a flexible tensor one byte short of its declared data is refused.
 */
TEST (testDecoderOctetStream, flexibleDataOneByteShort_n)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("octet_stream");
  std::vector<guint8> bytes = build_flex_memory (_NNS_UINT8, "4:1:1:1", 3U);
  GstBuffer *outbuf = NULL;

  ASSERT_TRUE (dec != NULL);

  EXPECT_EQ (GST_FLOW_ERROR,
      decode_one (dec, _NNS_TENSOR_FORMAT_FLEXIBLE, bytes, "1:1:1:1", &outbuf));
  EXPECT_EQ (gst_buffer_n_memory (outbuf), 0U);

  gst_buffer_unref (outbuf);
}

/**
 * @brief Negative: a flexible tensor whose header describes no data at all is refused.
 * @details A sparse header with no non-zero element sizes its data at zero
 *          bytes, which would otherwise wrap a NULL into the outgoing buffer.
 */
TEST (testDecoderOctetStream, flexibleZeroData_n)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("octet_stream");
  std::vector<guint8> bytes = build_flex_memory (_NNS_UINT8, "4:1:1:1", 4U);
  GstTensorMetaInfo meta;
  GstBuffer *outbuf = NULL;

  ASSERT_TRUE (gst_tensor_meta_info_parse_header (&meta, bytes.data ()));
  meta.format = _NNS_TENSOR_FORMAT_SPARSE;
  meta.sparse_info.nnz = 0;
  gst_tensor_meta_info_update_header (&meta, bytes.data ());

  ASSERT_TRUE (dec != NULL);

  EXPECT_EQ (GST_FLOW_ERROR,
      decode_one (dec, _NNS_TENSOR_FORMAT_FLEXIBLE, bytes, "1:1:1:1", &outbuf));
  EXPECT_EQ (gst_buffer_n_memory (outbuf), 0U);

  gst_buffer_unref (outbuf);
}

/**
 * @brief Negative: a static tensor smaller than its negotiated size is refused.
 * @details Without the bound the copy reads far past the end of the memory.
 */
TEST (testDecoderOctetStream, staticDataBeyondMemory_n)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("octet_stream");
  std::vector<guint8> bytes = { 1, 2, 3, 4 };
  GstBuffer *outbuf = NULL;

  ASSERT_TRUE (dec != NULL);

  EXPECT_EQ (GST_FLOW_ERROR,
      decode_one (dec, _NNS_TENSOR_FORMAT_STATIC, bytes, "67108864:1:1:1", &outbuf));
  EXPECT_EQ (gst_buffer_n_memory (outbuf), 0U);

  gst_buffer_unref (outbuf);
}

/**
 * @brief Negative: a static tensor of an unusable config is refused.
 */
TEST (testDecoderOctetStream, staticZeroSize_n)
{
  const GstTensorDecoderDef *dec = nnstreamer_decoder_find ("octet_stream");
  std::vector<guint8> bytes = { 1, 2, 3, 4 };
  GstBuffer *outbuf = NULL;

  ASSERT_TRUE (dec != NULL);

  EXPECT_EQ (GST_FLOW_ERROR,
      decode_one (dec, _NNS_TENSOR_FORMAT_STATIC, bytes, "0:0:0:0", &outbuf));
  EXPECT_EQ (gst_buffer_n_memory (outbuf), 0U);

  gst_buffer_unref (outbuf);
}

/**
 * @brief Main gtest entry point.
 */
int
main (int argc, char **argv)
{
  int result = -1;

  testing::InitGoogleTest (&argc, argv);

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  return result;
}
