/**
 * @file        unittest_converter.cc
 * @date        18 Mar 2021
 * @brief       Unit test for tensor_converter
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <tensor_common.h>
#include <unittest_util.h>
#include <tensor_converter_custom.h>
#include <flatbuffers/flexbuffers.h>

#define TEST_TIMEOUT_MS (1000U)

static int data_received;
/**
 * @brief custom callback function
 */
GstBuffer * tensor_converter_custom_cb (GstBuffer *in_buf,
    GstTensorsConfig *config) {
  GstMemory *in_mem, *out_mem;
  GstBuffer *out_buf = NULL;
  GstMapInfo in_info;
  guint mem_size;
  gpointer mem_data;

  if (!in_buf || !config)
    return NULL;

  data_received++;
  in_mem = gst_buffer_peek_memory (in_buf, 0);
  if (gst_memory_map (in_mem, &in_info, GST_MAP_READ) == FALSE) {
    ml_loge ("Cannot map input memory / tensor_converter::flexbuf.\n");
    return NULL;
  }
  flexbuffers::Map tensors = flexbuffers::GetRoot (in_info.data, in_info.size).AsMap ();
  config->info.num_tensors = tensors["num_tensors"].AsUInt32 ();

  if (config->info.num_tensors > NNS_TENSOR_SIZE_LIMIT) {
    nns_loge ("The number of tensors is limited to %d", NNS_TENSOR_SIZE_LIMIT);
    goto done;
  }
  config->rate_n = tensors["rate_n"].AsInt32 ();
  config->rate_d = tensors["rate_d"].AsInt32 ();

  out_buf = gst_buffer_new ();
  for (guint i = 0; i < config->info.num_tensors; i++) {
    gchar * tensor_key = g_strdup_printf ("tensor_%d", i);
    flexbuffers::Vector tensor = tensors[tensor_key].AsVector ();
    config->info.info[i].name = g_strdup (tensor[0].AsString ().c_str ());
    config->info.info[i].type = (tensor_type) tensor[1].AsInt32 ();

    flexbuffers::TypedVector dim = tensor[2].AsTypedVector ();
    for (guint j = 0; j < NNS_TENSOR_RANK_LIMIT; j++) {
      config->info.info[i].dimension[j] = dim[j].AsInt32 ();
    }
    flexbuffers::Blob tensor_data = tensor[3].AsBlob ();
    mem_size = tensor_data.size ();
    mem_data = g_memdup (tensor_data.data (), mem_size);

    out_mem = gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY, mem_data,
        mem_size, 0, mem_size, mem_data, g_free);

    gst_buffer_append_memory (out_buf, out_mem);
    g_free (tensor_key);
  }

  /** copy timestamps */
  gst_buffer_copy_into (
      out_buf, in_buf, (GstBufferCopyFlags)GST_BUFFER_COPY_METADATA, 0, -1);
done:
  gst_memory_unmap (in_mem, &in_info);

  return out_buf;
}

/**
 * @brief Test behavior: custom callback
 */
TEST (tensorConverterCustom, normal0)
{
  gchar *content1 = NULL;
  gchar *content2 = NULL;
  gsize len1, len2;
  char *tmp_tensor_raw = getTempFilename ();
  char *tmp_flex_raw = getTempFilename ();
  char *tmp_flex_to_tensor = getTempFilename ();

  EXPECT_NE (tmp_tensor_raw, nullptr);
  EXPECT_NE (tmp_flex_raw, nullptr);
  EXPECT_NE (tmp_flex_to_tensor, nullptr);

  gchar *str_pipeline = g_strdup_printf (
      "videotestsrc num-buffers=1 pattern=12 ! videoconvert ! videoscale ! "
      "video/x-raw,format=RGB,width=320,height=240 ! tensor_converter ! tee name=t "
      "t. ! queue ! filesink location=%s buffer-mode=unbuffered sync=false async=false "
      "t. ! queue ! tensor_decoder mode=flexbuf ! "
      "filesink location=%s buffer-mode=unbuffered sync=false async=false ",
      tmp_tensor_raw, tmp_flex_raw);

  GstElement *pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  data_received = 0;
  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  gst_object_unref (pipeline);
  g_free (str_pipeline);

  str_pipeline = g_strdup_printf (
      "filesrc location=%s blocksize=-1 ! application/octet-stream ! tensor_converter mode=custom:tconv ! "
      "filesink location=%s buffer-mode=unbuffered sync=false async=false ",
      tmp_flex_raw, tmp_flex_to_tensor);

  EXPECT_EQ (0, nnstreamer_converter_custom_register ("tconv", tensor_converter_custom_cb, NULL));

  pipeline = gst_parse_launch (str_pipeline, NULL);
  EXPECT_NE (pipeline, nullptr);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (1000000);

  EXPECT_EQ (1, data_received);
  _wait_pipeline_save_files (tmp_tensor_raw, content1, len1, 230400, TEST_TIMEOUT_MS);
  _wait_pipeline_save_files (tmp_flex_to_tensor, content2, len2, 230400, TEST_TIMEOUT_MS);
  EXPECT_EQ (len1, len2);
  EXPECT_EQ (memcmp (content1, content2, 230400), 0);
  g_free (content1);
  g_free (content2);

  EXPECT_EQ (setPipelineStateSync (pipeline, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);
  g_usleep (100000);

  EXPECT_EQ (0, nnstreamer_converter_custom_unregister ("tconv"));

  gst_object_unref (pipeline);
  g_free (str_pipeline);
  g_free (tmp_tensor_raw);
  g_free (tmp_flex_raw);
  g_free (tmp_flex_to_tensor);
}

/**
 * @brief Register custom callback with NULL parameter
 */
TEST (tensorConverterCustom, invalidParam0_n)
{
  EXPECT_NE (0, nnstreamer_converter_custom_register (NULL, tensor_converter_custom_cb, NULL));
  EXPECT_NE (0, nnstreamer_converter_custom_register ("tconv", NULL, NULL));
}

/**
 * @brief Register custom callback twice with same name
 */
TEST (tensorConverterCustom, invalidParam1_n)
{
  EXPECT_EQ (0, nnstreamer_converter_custom_register ("tconv", tensor_converter_custom_cb, NULL));
  EXPECT_NE (0, nnstreamer_converter_custom_register ("tconv", tensor_converter_custom_cb, NULL));
  EXPECT_EQ (0, nnstreamer_converter_custom_unregister ("tconv"));
}

/**
 * @brief Unregister custom callback with NULL parameter
 */
TEST (tensorConverterCustom, invalidParam2_n)
{
  EXPECT_NE (0, nnstreamer_converter_custom_unregister (NULL));
}

/**
 * @brief Unregister custom callback which is not registered
 */
TEST (tensorConverterCustom, invalidParam3_n)
{
  EXPECT_NE (0, nnstreamer_converter_custom_unregister ("tconv"));
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
