/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file	gstdatareposrc.c
 * @date	31 January 2023
 * @brief	GStreamer plugin to read file in MLOps Data repository into buffers
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyunil Park <hyunil46.park@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 *
 * ## Example launch line
 * |[
 * gst-launch-1.0 datareposrc location=mnist.data json=mnist.json start-sample-index=3 stop-sample-index=202 epochs=5 ! \
 * ! tensor_sink
 * gst-launch-1.0 datareposrc location=image_%02ld.png json=image.json start-sample-index=3 stop-sample-index=9 epochs=2 ! fakesink
 * gst-launch-1.0 datareposrc location=audiofile json=audio.json ! fakesink
 * gst-launch-1.0 datareposrc location=videofile json=video.json ! fakesink
 * |]
 * |[ Unknown sample file(has not JSON) need to set caps and blocksize or set caps to tensors type without blocksize
 * gst-launch-1.0 datareposrc blocksize=3176 location=unknown.data start-sample-index=3 stop-sample-index=202 epochs=5 \
 * caps ="application/octet-stream" ! tensor_converter input-dim=1:1:784:1,1:1:10:1 input-type=float32,float32 ! fakesink
 * |]
 * or
 * |[
 * gst-launch-1.0 datareposrc location=unknown.data start-sample-index=3 stop-sample-index=202 epochs=5 \
 * caps ="other/tensors, format=(string)static, framerate=(fraction)0/1, num_tensors=(int)2, dimensions=(string)1:1:784:1.1:1:10:1, types=(string)float32.float32" \
 * ! fakesink
 * ]|
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include <gst/gst.h>
#include <gst/video/video-info.h>
#include <gst/audio/audio-info.h>
#include <glib/gstdio.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_util.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <inttypes.h>
#include "gstdatareposrc.h"

#define struct_stat struct stat
#ifndef S_ISREG
/* regular file */
#define S_ISREG(mode) ((mode)&_S_IFREG)
#endif
#ifndef S_ISDIR
#define S_ISDIR(mode) ((mode)&_S_IFDIR)
#endif
/* socket */
#ifndef S_ISSOCK
#define S_ISSOCK(x) (0)
#endif
#ifndef O_BINARY
#define O_BINARY (0)
#endif

static GstStaticPadTemplate srctemplate = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

GST_DEBUG_CATEGORY_STATIC (gst_data_repo_src_debug);
#define GST_CAT_DEFAULT gst_data_repo_src_debug

/* datareposrc signals and args */
enum
{
  PROP_0,
  PROP_LOCATION,
  PROP_JSON,
  PROP_START_SAMPLE_INDEX,
  PROP_STOP_SAMPLE_INDEX,
  PROP_EPOCHS,
  PROP_IS_SHUFFLE,
  PROP_TENSORS_SEQUENCE,
  PROP_CAPS,                    /* for setting caps of sample data directly */
};

#define DEFAULT_INDEX 0
#define DEFAULT_EPOCHS 1
#define DEFAULT_IS_SHUFFLE TRUE

static void gst_data_repo_src_finalize (GObject * object);
static GstStateChangeReturn gst_data_repo_src_change_state (GstElement *
    element, GstStateChange transition);
static void gst_data_repo_src_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_data_repo_src_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static gboolean gst_data_repo_src_stop (GstBaseSrc * basesrc);
static GstCaps *gst_data_repo_src_get_caps (GstBaseSrc * basesrc,
    GstCaps * filter);
static gboolean gst_data_repo_src_set_caps (GstBaseSrc * basesrc,
    GstCaps * caps);
static GstFlowReturn gst_data_repo_src_create (GstPushSrc * pushsrc,
    GstBuffer ** buffer);
#define _do_init \
  GST_DEBUG_CATEGORY_INIT (gst_data_repo_src_debug, "datareposrc", 0, "datareposrc element");

#define gst_data_repo_src_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstDataRepoSrc, gst_data_repo_src, GST_TYPE_PUSH_SRC,
    _do_init);

/**
 * @brief initialize the datareposrc's class
 */
static void
gst_data_repo_src_class_init (GstDataRepoSrcClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS (klass);
  GstPushSrcClass *gstpushsrc_class = GST_PUSH_SRC_CLASS (klass);

  gobject_class->set_property =
      GST_DEBUG_FUNCPTR (gst_data_repo_src_set_property);
  gobject_class->get_property =
      GST_DEBUG_FUNCPTR (gst_data_repo_src_get_property);

  g_object_class_install_property (gobject_class, PROP_LOCATION,
      g_param_spec_string ("location", "File Location",
          "Location of the file to read that is stored in MLOps Data Repository, "
          "if the files are images, write the index of filename name "
          "like %04ld or %04lld (e.g., filename%04ld.png)",
          NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_JSON,
      g_param_spec_string ("json", "Json file path",
          "Json file path containing the meta information of the file "
          "specified as location", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_START_SAMPLE_INDEX,
      g_param_spec_uint ("start-sample-index", "Start index of samples",
          "Start index of sample to read, in case of image, "
          "the starting index of the numbered files. start at 0."
          "Set start index of range of samples or files to read",
          0, G_MAXINT, DEFAULT_INDEX,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_STOP_SAMPLE_INDEX,
      g_param_spec_uint ("stop-sample-index", "Stop index of samples",
          "Stop index of sample to read, in case of image, "
          "the stopping index of the numbered files. start at 0."
          "Set stop index of range of samples or files to read",
          0, G_MAXINT, DEFAULT_INDEX,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_EPOCHS,
      g_param_spec_uint ("epochs", "Epochs",
          "Repetition of range of files or samples to read, set number of repetitions",
          0, G_MAXINT, DEFAULT_EPOCHS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_TENSORS_SEQUENCE,
      g_param_spec_string ("tensors-sequence", "Tensors sequence",
          "Tensors in a sample are read into gstBuffer according to tensors-sequence."
          "Only read the set tensors among all tensors in a sample"
          "e.g, if a sample has '1:1:1:1','1:1:10:1','1:1:784:1' and each index is '0,1,2', "
          "'tensors-sequence=2,1' means that only '1:1:784:1' then '1:1:10:1' are read. "
          "Use for other/tensors and the default value is NULL"
          "(all tensors are read in the order stored in a sample).",
          NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_IS_SHUFFLE,
      g_param_spec_boolean ("is-shuffle", "Is shuffle",
          "If the value is true, samples are shuffled",
          DEFAULT_IS_SHUFFLE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  g_object_class_install_property (gobject_class, PROP_CAPS,
      g_param_spec_boxed ("caps", "Caps",
          "Optional property, Caps describing the format of the sample data.",
          GST_TYPE_CAPS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  gobject_class->finalize = gst_data_repo_src_finalize;
  gstelement_class->change_state = gst_data_repo_src_change_state;

  gst_element_class_set_static_metadata (gstelement_class,
      "NNStreamer MLOps Data Repository Source",
      "Source/File",
      "Read files in MLOps Data Repository into buffers",
      "Samsung Electronics Co., Ltd.");
  gst_element_class_add_static_pad_template (gstelement_class, &srctemplate);

  gstbasesrc_class->stop = GST_DEBUG_FUNCPTR (gst_data_repo_src_stop);
  gstbasesrc_class->get_caps = GST_DEBUG_FUNCPTR (gst_data_repo_src_get_caps);
  gstbasesrc_class->set_caps = GST_DEBUG_FUNCPTR (gst_data_repo_src_set_caps);
  gstpushsrc_class->create = GST_DEBUG_FUNCPTR (gst_data_repo_src_create);

  if (sizeof (off_t) < 8) {
    GST_LOG ("No large file support, sizeof (off_t) = %" G_GSIZE_FORMAT "!",
        sizeof (off_t));
  }
}

/**
 * @brief Initialize datareposrc
 */
static void
gst_data_repo_src_init (GstDataRepoSrc * src)
{
  src->filename = NULL;
  src->json_filename = NULL;
  src->tensors_seq_str = NULL;
  src->file_size = 0;
  src->fd = 0;
  src->data_type = GST_DATA_REPO_DATA_UNKNOWN;
  src->fd_offset = 0;
  src->start_offset = 0;
  src->last_offset = 0;
  src->successful_read = FALSE;
  src->is_start = FALSE;
  src->current_sample_index = 0;
  src->start_sample_index = 0;
  src->stop_sample_index = 0;
  src->epochs = DEFAULT_EPOCHS;
  src->shuffled_index_array = g_array_new (FALSE, FALSE, sizeof (guint));
  src->array_index = 0;
  src->sample_offset_array = NULL;
  src->sample_offset_array_len = 0;
  src->tensor_size_array = NULL;
  src->tensor_size_array_len = 0;
  src->tensor_count_array = NULL;
  src->tensor_count_array_len = 0;
  src->first_epoch_is_done = FALSE;
  src->is_shuffle = DEFAULT_IS_SHUFFLE;
  src->num_samples = 0;
  src->total_samples = 0;
  src->tensors_seq_cnt = 0;
  src->caps = NULL;
  src->sample_size = 0;
  src->need_changed_caps = FALSE;
  src->n_frame = 0;
  src->running_time = 0;
  src->parser = NULL;
  gst_tensors_config_init (&src->config);

  /* Filling the buffer should be pending until set_caps() */
  gst_base_src_set_format (GST_BASE_SRC (src), GST_FORMAT_TIME);
  gst_base_src_set_live (GST_BASE_SRC (src), TRUE);
}

/**
 * @brief Safely close file descriptor with error handling
 */
static void
gst_data_repo_src_safe_close_fd (GstDataRepoSrc * src)
{
  if (src->fd) {
    if (!g_close (src->fd, NULL)) {
      GST_WARNING_OBJECT (src, "Failed to close file descriptor %d: %s",
          src->fd, g_strerror (errno));
      return;
    }
    src->fd = 0;
  }
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_data_repo_src_finalize (GObject * object)
{
  GstDataRepoSrc *src = GST_DATA_REPO_SRC (object);

  g_free (src->filename);
  g_free (src->json_filename);
  g_free (src->tensors_seq_str);

  /* close the file */
  gst_data_repo_src_safe_close_fd (src);

  if (src->parser)
    g_object_unref (src->parser);

  if (src->shuffled_index_array)
    g_array_free (src->shuffled_index_array, TRUE);

  if (src->caps)
    gst_caps_replace (&src->caps, NULL);

  gst_tensors_config_free (&src->config);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Parse gst-caps to get media type and data size.
 * @note This function always takes the ownership of gst-caps.
 */
static gboolean
gst_data_repo_src_parse_caps (GstDataRepoSrc * src, GstCaps * caps)
{
  GstStructure *s;

  g_return_val_if_fail (src != NULL, FALSE);
  g_return_val_if_fail (caps != NULL, FALSE);

  src->data_type = gst_data_repo_get_data_type_from_caps (caps);
  s = gst_caps_get_structure (caps, 0);

  switch (src->data_type) {
    case GST_DATA_REPO_DATA_VIDEO:
    {
      GstVideoInfo video_info;

      gst_video_info_init (&video_info);
      gst_video_info_from_caps (&video_info, caps);

      /* https://gstreamer.freedesktop.org/documentation/additional/design/mediatype-video-raw.html?gi-language=c */
      src->sample_size = (guint) GST_VIDEO_INFO_SIZE (&video_info);

      GST_DEBUG_OBJECT (src, "format(%s), width(%d), height(%d): %zd byte/frame",
          gst_video_format_to_string (GST_VIDEO_INFO_FORMAT (&video_info)),
          GST_VIDEO_INFO_WIDTH (&video_info),
          GST_VIDEO_INFO_HEIGHT (&video_info), src->sample_size);
      break;
    }
    case GST_DATA_REPO_DATA_AUDIO:
    {
      GstAudioInfo audio_info;
      gint rate, channel, depth;

      gst_audio_info_init (&audio_info);
      gst_audio_info_from_caps (&audio_info, caps);

      rate = GST_AUDIO_INFO_RATE (&audio_info);
      channel = GST_AUDIO_INFO_CHANNELS (&audio_info);
      depth = GST_AUDIO_INFO_DEPTH (&audio_info);

      src->sample_size = (gsize) channel * (depth / 8) * rate;

      GST_DEBUG_OBJECT (src,
          "format(%s), depth(%d), rate(%d), channel(%d): %zd bps",
          gst_audio_format_to_string (GST_AUDIO_INFO_FORMAT (&audio_info)),
          depth, rate, channel, src->sample_size);
      break;
    }
    case GST_DATA_REPO_DATA_TENSOR:
    {
      GstTensorInfo *_info;
      guint i;

      /* Set current tensors information if given data type is tensor. */
      gst_tensors_config_free (&src->config);
      gst_tensors_config_from_caps (&src->config, caps, TRUE);
      src->sample_size = 0;

      for (i = 0; i < src->config.info.num_tensors; i++) {
        src->tensors_offset[i] = src->sample_size;
        _info = gst_tensors_info_get_nth_info (&src->config.info, i);
        src->tensors_size[i] = gst_tensor_info_get_size (_info);
        GST_DEBUG_OBJECT (src, "offset[%u]: %zd", i, src->tensors_offset[i]);
        GST_DEBUG_OBJECT (src, "size[%u]: %zd", i, src->tensors_size[i]);
        src->sample_size += src->tensors_size[i];
      }
      break;
    }
    default:
      break;
  }

  src->rate_n = 0;
  src->rate_d = 1;
  if (gst_structure_has_field (s, "framerate")) {
    gst_structure_get_fraction (s, "framerate", &src->rate_n, &src->rate_d);
  }

  GST_LOG_OBJECT (src, "framerate %d/%d", src->rate_n, src->rate_d);
  GST_LOG_OBJECT (src, "data type: %d", src->data_type);

  /* This function always takes caps. */
  if (src->data_type != GST_DATA_REPO_DATA_UNKNOWN)
    gst_caps_take (&src->caps, caps);
  else
    gst_caps_unref (caps);

  return (src->data_type != GST_DATA_REPO_DATA_UNKNOWN);
}

/**
 * @brief Function to set file path.
 */
static gboolean
gst_data_repo_src_set_file_path (GstDataRepoSrc * src, const int prop,
    const gchar * file_path, GError ** err)
{
  gchar *filename = NULL;
  GstState state;

  g_return_val_if_fail (prop == PROP_LOCATION || prop == PROP_JSON, FALSE);

  /* the element must be stopped in order to do this */
  GST_OBJECT_LOCK (src);
  state = GST_STATE (src);
  if (state != GST_STATE_READY && state != GST_STATE_NULL)
    goto wrong_state;
  GST_OBJECT_UNLOCK (src);

  /* clear the filename if we get a NULL */
  if (file_path == NULL) {
    filename = NULL;
  } else {
    /* should be UTF8 */
    filename = g_strdup (file_path);
    GST_INFO_OBJECT (src, "%sname : %s",
        (prop == PROP_LOCATION) ? "file" : "json_file", filename);
  }

  if (prop == PROP_LOCATION) {
    g_free (src->filename);
    src->filename = filename;
  } else {                      /* PROP_JSON */
    g_free (src->json_filename);
    src->json_filename = filename;
  }

  return TRUE;

  /* ERROR */
wrong_state:
  {
    g_warning
        ("Changing the `location or json' property on datareposrc when a file is "
        "open is not supported.");
    if (err)
      g_set_error (err, GST_URI_ERROR, GST_URI_ERROR_BAD_STATE,
          "Changing the `location or json' property on datareposrc when a file is "
          "open is not supported.");
    GST_OBJECT_UNLOCK (src);
    return FALSE;
  }
}

/**
 * @brief Function to set tensors sequence
 */
static gboolean
gst_data_repo_src_set_tensors_sequence (GstDataRepoSrc * src)
{
  g_auto (GStrv) strv = NULL;
  guint length;
  guint i = 0;

  g_return_val_if_fail (src != NULL, FALSE);
  g_return_val_if_fail (src->tensors_seq_str != NULL, FALSE);

  GST_INFO_OBJECT (src, "tensors sequence = %s", src->tensors_seq_str);

  /* not use NNS_TENSOR_SIZE_LIMIT */
  strv = g_strsplit (src->tensors_seq_str, ",", -1);

  length = g_strv_length (strv);
  if (length > NNS_TENSOR_SIZE_LIMIT) {
    GST_ERROR_OBJECT (src, "The total number of indices exceeded %d.",
        NNS_TENSOR_SIZE_LIMIT);
    goto error;
  }

  while (strv[i] != NULL && strlen (strv[i]) > 0) {
    src->tensors_seq[i] = (guint) g_ascii_strtoull (strv[i], NULL, 10);
    if (src->tensors_seq[i] > src->config.info.num_tensors - 1) {
      GST_ERROR_OBJECT (src, "Invalid index %d, max is %d", src->tensors_seq[i],
          src->config.info.num_tensors - 1);
      goto error;
    }
    GST_INFO_OBJECT (src, "%d", src->tensors_seq[i]);
    i++;
  }
  src->tensors_seq_cnt = i;
  GST_INFO_OBJECT (src, "The number of selected tensors is %d",
      src->tensors_seq_cnt);

  /* num_tensors was calculated from JSON file */
  if (src->config.info.num_tensors < src->tensors_seq_cnt) {
    GST_ERROR_OBJECT (src,
        "The number of tensors selected(%d) "
        "is greater than the total number of tensors(%d) in a sample.",
        src->tensors_seq_cnt, src->config.info.num_tensors);
    goto error;
  }
  return TRUE;

error:
  src->tensors_seq_cnt = 0;
  return FALSE;
}

/**
 * @brief Function to get file offset with sample index
 */
static guint64
gst_data_repo_src_get_file_offset (GstDataRepoSrc * src, guint sample_index)
{
  guint64 offset;

  g_return_val_if_fail (src != NULL, 0);
  g_return_val_if_fail (src->data_type != GST_DATA_REPO_DATA_IMAGE, 0);
  g_return_val_if_fail (src->fd != 0, 0);

  offset = src->sample_size * sample_index;

  return offset;
}

/**
 * @brief Function to shuffle samples index
 */
static void
gst_data_repo_src_shuffle_samples_index (GstDataRepoSrc * src)
{
  guint i, j;
  guint value_i, value_j;
  g_return_if_fail (src != NULL);
  g_return_if_fail (src->shuffled_index_array != NULL);

  GST_LOG_OBJECT (src, "samples index are shuffled");
  /* Fisher-Yates algorithm */
  /* The last index is the number of samples - 1. */
  for (i = src->num_samples - 1; i > 0; i--) {
    j = g_random_int_range (0, src->num_samples);
    value_i = g_array_index (src->shuffled_index_array, guint, i);
    value_j = g_array_index (src->shuffled_index_array, guint, j);

    /* shuffled_index_array->data type is gchar * */
    *(guint *) (src->shuffled_index_array->data + (sizeof (guint) * i)) =
        value_j;
    *(guint *) (src->shuffled_index_array->data + (sizeof (guint) * j)) =
        value_i;
  }

  for (i = 0; i < src->shuffled_index_array->len; i++) {
    GST_DEBUG_OBJECT (src, "%u -> %u", i,
        g_array_index (src->shuffled_index_array, guint, i));
  }
}

/**
 * @brief Function to check epoch and EOS
 */
static gboolean
gst_data_repo_src_epoch_is_done (GstDataRepoSrc * src)
{
  g_return_val_if_fail (src != NULL, FALSE);
  if (src->num_samples != src->array_index)
    return FALSE;

  src->first_epoch_is_done = TRUE;
  src->array_index = 0;
  src->epochs--;

  return TRUE;
}

/**
 * @brief Function to read tensors
 */
static GstFlowReturn
gst_data_repo_src_read_tensors (GstDataRepoSrc * src, GstBuffer ** buffer)
{
  GstFlowReturn ret = GST_FLOW_OK;
  guint i = 0, seq_idx = 0;
  GstBuffer *buf = NULL;
  gsize to_read, byte_read;
  gssize read_size;
  guint8 *data;
  GstMemory *mem = NULL;
  GstMapInfo info;
  guint shuffled_index = 0;
  guint64 sample_offset = 0;
  guint64 offset = 0;           /* offset from 0 */

  g_return_val_if_fail (src->fd != 0, GST_FLOW_ERROR);
  g_return_val_if_fail (src->shuffled_index_array != NULL, GST_FLOW_ERROR);

  if (gst_data_repo_src_epoch_is_done (src)) {
    if (src->epochs == 0) {
      GST_LOG_OBJECT (src, "send EOS");
      return GST_FLOW_EOS;
    }
    if (src->is_shuffle)
      gst_data_repo_src_shuffle_samples_index (src);
  }

  /* only do for first epoch */
  if (!src->first_epoch_is_done) {
    /* append samples index to array */
    g_array_append_val (src->shuffled_index_array, src->current_sample_index);
    src->current_sample_index++;
  }
  shuffled_index =
      g_array_index (src->shuffled_index_array, guint, src->array_index++);
  GST_LOG_OBJECT (src, "shuffled_index [%d] -> %d", src->array_index - 1,
      shuffled_index);

  /* sample offset from 0 */
  sample_offset = gst_data_repo_src_get_file_offset (src, shuffled_index);
  GST_LOG_OBJECT (src, "sample offset 0x%" G_GINT64_MODIFIER "x (%d size)",
      sample_offset, (guint) sample_offset);

  buf = gst_buffer_new ();

  for (i = 0; i < src->tensors_seq_cnt; i++) {
    seq_idx = src->tensors_seq[i];
    mem = gst_allocator_alloc (NULL, src->tensors_size[seq_idx], NULL);

    if (!gst_memory_map (mem, &info, GST_MAP_WRITE)) {
      GST_ERROR_OBJECT (src, "Could not map GstMemory[%d]", i);
      ret = GST_FLOW_ERROR;
      goto error;
    }

    GST_INFO_OBJECT (src, "sequence index: %d", seq_idx);
    GST_INFO_OBJECT (src, "tensor_size[%d]: %zd", seq_idx,
        src->tensors_size[seq_idx]);
    GST_INFO_OBJECT (src, "tensors_offset[%d]: %zd", seq_idx,
        src->tensors_offset[seq_idx]);

    /** offset and sample_offset(byte size) are from 0.
      if the size of one sample is 6352 and num_tensors is 4,
      dimensions are  '1:1:784:1' , '1:1:10:1',  '1:1:784:1' and '1:1:10:1' with float32.
      the offset of the second sample is as follows.
        -------------------------------------------------
          sample_offset: 6352
          tensors index: [ 0    | 1    | 2    | 3     ]
           tensors_size: [ 3136 | 40   | 3136 | 40    ]
         tensors_offset: [ 0    | 3136 | 3176 | 6312  ]
              fd offset: [ 6352 | 9488 | 9528 | 12664 ]
        -------------------------------------------------
      if user sets "tensor-sequence=2,1", datareposrc read offset 9528 then 9488.
    */

    data = info.data;

    byte_read = 0;
    to_read = src->tensors_size[seq_idx];
    offset = sample_offset + src->tensors_offset[seq_idx];
    src->fd_offset = lseek (src->fd, offset, SEEK_SET);

    while (to_read > 0) {
      GST_LOG_OBJECT (src,
          "Reading %zd bytes at offset 0x%" G_GINT64_MODIFIER "x (%zd size)",
          to_read, src->fd_offset + byte_read,
          (guint) src->fd_offset + byte_read);
      errno = 0;
      read_size = read (src->fd, data + byte_read, to_read);
      GST_LOG_OBJECT (src, "Read: %zd", read_size);
      if (read_size < 0) {
        if (errno == EAGAIN || errno == EINTR)
          continue;
        GST_ELEMENT_ERROR (src, RESOURCE, READ, (NULL), GST_ERROR_SYSTEM);
        ret = GST_FLOW_ERROR;
        goto error;
      }
      /* files should eos if they read 0 and more was requested */
      if (read_size == 0) {
        /* .. but first we should return any remaining data */
        if (byte_read > 0)
          break;
        GST_DEBUG_OBJECT (src, "EOS");
        ret = GST_FLOW_EOS;
        goto error;
      }
      to_read -= read_size;
      byte_read += read_size;

      src->read_position += read_size;
      src->fd_offset += read_size;
    }

    gst_memory_unmap (mem, &info);

    gst_tensor_buffer_append_memory (buf, mem,
        gst_tensors_info_get_nth_info (&src->config.info, i));
    mem = NULL;
  }

  *buffer = buf;

  return GST_FLOW_OK;

error:
  if (mem) {
    gst_memory_unmap (mem, &info);
    gst_memory_unref (mem);
  }
  if (buf)
    gst_buffer_unref (buf);

  return ret;
}

/**
 * @brief Function to get num_tensors from tensor_count_array
 */
static guint
gst_data_repo_src_get_num_tensors (GstDataRepoSrc * src, guint shuffled_index)
{
  guint num_tensors = 0;
  guint cur_idx_tensor_cnt = 0;
  guint next_idx_tensor_cnt = 0;
  gint64 val;

  g_return_val_if_fail (src != NULL, 0);

  val = json_array_get_int_element (src->tensor_count_array, shuffled_index);
  g_return_val_if_fail (val >= 0, 0);

  cur_idx_tensor_cnt = (guint) val;
  GST_DEBUG_OBJECT (src, "cur_idx_tensor_cnt:%u", cur_idx_tensor_cnt);

  if (shuffled_index + 1 == src->tensor_count_array_len) {
    next_idx_tensor_cnt = src->tensor_size_array_len;
  } else {
    val = json_array_get_int_element (src->tensor_count_array,
        shuffled_index + 1);
    g_return_val_if_fail (val >= 0, 0);

    next_idx_tensor_cnt = (guint) val;
  }
  GST_DEBUG_OBJECT (src, "next_idx_tensor_cnt:%u", next_idx_tensor_cnt);

  num_tensors = next_idx_tensor_cnt - cur_idx_tensor_cnt;
  GST_DEBUG_OBJECT (src, "num_tensors:%u", num_tensors);

  return num_tensors;
}

/**
 * @brief Function to read flexible or sparse tensors
 */
static GstFlowReturn
gst_data_repo_src_read_flexible_or_sparse_tensors (GstDataRepoSrc * src,
    GstBuffer ** buffer)
{
  GstFlowReturn ret = GST_FLOW_OK;
  guint i;
  guint shuffled_index = 0;
  gint64 val;
  guint num_tensors = 0;
  GstBuffer *buf = NULL;
  GstMemory *mem = NULL;
  GstMapInfo info;
  GstTensorMetaInfo meta;
  GstTensorInfo tinfo;
  gsize to_read, byte_read;
  gssize read_size;
  guint8 *data;
  guint tensor_count;
  gsize tensor_size;

  g_return_val_if_fail (src->fd != 0, GST_FLOW_ERROR);
  g_return_val_if_fail (src->shuffled_index_array != NULL, GST_FLOW_ERROR);

  if (gst_data_repo_src_epoch_is_done (src)) {
    if (src->epochs == 0) {
      GST_LOG_OBJECT (src, "send EOS");
      return GST_FLOW_EOS;
    }
    if (src->is_shuffle)
      gst_data_repo_src_shuffle_samples_index (src);
  }

  /* only do for first epoch */
  if (!src->first_epoch_is_done) {
    /* append samples index to array */
    g_array_append_val (src->shuffled_index_array, src->current_sample_index);
    src->current_sample_index++;
  }

  shuffled_index =
      g_array_index (src->shuffled_index_array, guint, src->array_index++);
  GST_LOG_OBJECT (src, "shuffled_index [%d] -> %d", src->array_index - 1,
      shuffled_index);

  /* sample offset from 0 */
  val = json_array_get_int_element (src->sample_offset_array, shuffled_index);
  if (val < 0) {
    GST_ERROR_OBJECT (src, "Could not get the sample offset from json.");
    return GST_FLOW_ERROR;
  }

  GST_LOG_OBJECT (src, "sample offset 0x%" G_GINT64_MODIFIER "x (%d size)",
      val, (guint) val);

  src->fd_offset = lseek (src->fd, val, SEEK_SET);

  val = json_array_get_int_element (src->tensor_count_array, shuffled_index);
  if (val < 0) {
    GST_ERROR_OBJECT (src, "Could not get the tensor count from json.");
    return GST_FLOW_ERROR;
  }
  tensor_count = (guint) val;

  num_tensors = gst_data_repo_src_get_num_tensors (src, shuffled_index);

  buf = gst_buffer_new ();

  for (i = 0; i < num_tensors; i++) {
    val = json_array_get_int_element (src->tensor_size_array, tensor_count + i);
    if (val < 0) {
      GST_ERROR_OBJECT (src, "Could not get the size of tensor from json.");
      ret = GST_FLOW_ERROR;
      goto error;
    }

    tensor_size = (gsize) val;
    mem = gst_allocator_alloc (NULL, tensor_size, NULL);

    if (!gst_memory_map (mem, &info, GST_MAP_WRITE)) {
      GST_ERROR_OBJECT (src, "Could not map GstMemory[%d]", i);
      ret = GST_FLOW_ERROR;
      goto error;
    }

    data = info.data;
    byte_read = 0;
    to_read = tensor_size;
    while (to_read > 0) {
      GST_LOG_OBJECT (src,
          "Reading %zd bytes at offset 0x%" G_GINT64_MODIFIER "x (%lld size)",
          to_read, src->fd_offset + byte_read, (long long) src->fd_offset);
      errno = 0;
      read_size = read (src->fd, data + byte_read, to_read);
      GST_LOG_OBJECT (src, "Read: %zd", read_size);
      if (read_size < 0) {
        if (errno == EAGAIN || errno == EINTR)
          continue;
        GST_ELEMENT_ERROR (src, RESOURCE, READ, (NULL), GST_ERROR_SYSTEM);
        ret = GST_FLOW_ERROR;
        goto error;
      }
      /* files should eos if they read 0 and more was requested */
      if (read_size == 0) {
        /* .. but first we should return any remaining data */
        if (byte_read > 0)
          break;
        GST_DEBUG_OBJECT (src, "EOS");
        ret = GST_FLOW_EOS;
        goto error;
      }
      to_read -= read_size;
      byte_read += read_size;

      src->read_position += read_size;
      src->fd_offset += read_size;
    }

    /* check invalid flexible tensor */
    if (!gst_tensor_meta_info_parse_header (&meta, info.data)) {
      GST_ERROR_OBJECT (src, "Invalid flexible tensors");
      ret = GST_FLOW_ERROR;
      goto error;
    }

    gst_memory_unmap (mem, &info);

    gst_tensor_meta_info_convert (&meta, &tinfo);
    gst_tensor_buffer_append_memory (buf, mem, &tinfo);
    gst_tensor_info_free (&tinfo);
    mem = NULL;
  }

  *buffer = buf;

  return GST_FLOW_OK;

error:
  if (mem) {
    gst_memory_unmap (mem, &info);
    gst_memory_unref (mem);
  }
  if (buf)
    gst_buffer_unref (buf);

  return ret;
}

/**
 * @brief Get image filename
 */
static gchar *
gst_data_repo_src_get_image_filename (GstDataRepoSrc * src)
{
  gchar *filename = NULL;
  guint shuffled_index = 0;
  g_return_val_if_fail (src != NULL, NULL);
  g_return_val_if_fail (src->data_type == GST_DATA_REPO_DATA_IMAGE, NULL);
  g_return_val_if_fail (src->filename != NULL, NULL);
  g_return_val_if_fail (src->shuffled_index_array != NULL, NULL);

  /* GST_DATA_REPO_DATA_IMAGE must have %d in src->filename */
  if (src->shuffled_index_array->len > 0)
    shuffled_index =
        g_array_index (src->shuffled_index_array, guint, src->array_index);
  else
    shuffled_index = 0;         /* Used for initial file open verification */

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
  /* let's set value by property */
  filename = g_strdup_printf (src->filename, shuffled_index);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

  return filename;
}

/**
 * @brief Function to read multi image files
 */
static GstFlowReturn
gst_data_repo_src_read_multi_images (GstDataRepoSrc * src, GstBuffer ** buffer)
{
  g_autofree gchar *filename = NULL;
  gsize size;
  gchar *data;
  GstBuffer *buf;
  gboolean read_size;
  GError *error = NULL;

  g_return_val_if_fail (src->shuffled_index_array != NULL, GST_FLOW_ERROR);

  if (gst_data_repo_src_epoch_is_done (src)) {
    if (src->epochs == 0) {
      GST_LOG_OBJECT (src, "send EOS");
      return GST_FLOW_EOS;
    }
    if (src->is_shuffle)
      gst_data_repo_src_shuffle_samples_index (src);
  }

  /* only do for first epoch */
  if (!src->first_epoch_is_done) {
    /* append samples index to array */
    g_array_append_val (src->shuffled_index_array, src->current_sample_index);
    src->current_sample_index++;
  }

  filename = gst_data_repo_src_get_image_filename (src);
  GST_DEBUG_OBJECT (src, "Reading from file \"%s\".", filename);
  src->array_index++;

  /* Try to read one image */
  read_size = g_file_get_contents (filename, &data, &size, &error);
  if (!read_size) {
    if (src->successful_read) {
      /* If we've read at least one buffer successfully, not finding the next file is EOS. */
      if (error != NULL)
        g_error_free (error);
      return GST_FLOW_EOS;
    }
    goto handle_error;
  }

  /* Success reading on image */
  src->successful_read = TRUE;
  GST_DEBUG_OBJECT (src, "file size is %zd", size);

  buf = gst_buffer_new ();
  gst_buffer_append_memory (buf,
      gst_memory_new_wrapped (0, data, size, 0, size, data, g_free));

  *buffer = buf;

  return GST_FLOW_OK;

handle_error:
  {
    if (error != NULL) {
      GST_ELEMENT_ERROR (src, RESOURCE, READ,
          ("Error while reading from file \"%s\".", filename),
          ("%s", error->message));
      g_error_free (error);
    } else {
      GST_ELEMENT_ERROR (src, RESOURCE, READ,
          ("Error while reading from file \"%s\".", filename),
          ("%s", g_strerror (errno)));
    }
    return GST_FLOW_ERROR;
  }
}

/**
 * @brief Function to read others media type (video, audio, octet and text)
 */
static GstFlowReturn
gst_data_repo_src_read_others (GstDataRepoSrc * src, GstBuffer ** buffer)
{
  GstFlowReturn ret = GST_FLOW_OK;
  GstBuffer *buf;
  gsize to_read, byte_read;
  gssize read_size;
  guint8 *data;
  GstMemory *mem;
  GstMapInfo info;
  guint shuffled_index = 0;
  guint64 offset = 0;

  g_return_val_if_fail (src->fd != 0, GST_FLOW_ERROR);
  g_return_val_if_fail (src->shuffled_index_array != NULL, GST_FLOW_ERROR);

  if (gst_data_repo_src_epoch_is_done (src)) {
    if (src->epochs == 0) {
      GST_LOG_OBJECT (src, "send EOS");
      return GST_FLOW_EOS;
    }
    if (src->is_shuffle)
      gst_data_repo_src_shuffle_samples_index (src);
  }

  /* only do for first epoch */
  if (!src->first_epoch_is_done) {
    /* append samples index to array */
    g_array_append_val (src->shuffled_index_array, src->current_sample_index);
    src->current_sample_index++;
  }

  shuffled_index =
      g_array_index (src->shuffled_index_array, guint, src->array_index++);
  GST_LOG_OBJECT (src, "shuffled_index [%d] -> %d", src->array_index - 1,
      shuffled_index);
  offset = gst_data_repo_src_get_file_offset (src, shuffled_index);
  src->fd_offset = lseek (src->fd, offset, SEEK_SET);

  mem = gst_allocator_alloc (NULL, src->sample_size, NULL);

  if (!gst_memory_map (mem, &info, GST_MAP_WRITE)) {
    GST_ERROR_OBJECT (src, "Could not map GstMemory");
    gst_memory_unref (mem);
    return GST_FLOW_ERROR;
  }

  data = info.data;

  byte_read = 0;
  to_read = src->sample_size;
  while (to_read > 0) {
    GST_LOG_OBJECT (src, "Reading %zd bytes at offset 0x%" G_GINT64_MODIFIER "x",
        to_read, src->fd_offset + byte_read);
    errno = 0;
    read_size = read (src->fd, data + byte_read, to_read);
    GST_LOG_OBJECT (src, "Read: %zd", read_size);
    if (read_size < 0) {
      if (errno == EAGAIN || errno == EINTR)
        continue;
      GST_ELEMENT_ERROR (src, RESOURCE, READ, (NULL), GST_ERROR_SYSTEM);
      ret = GST_FLOW_ERROR;
      goto error;
    }
    /* files should eos if they read 0 and more was requested */
    if (read_size == 0) {
      /* .. but first we should return any remaining data */
      if (byte_read > 0)
        break;
      GST_DEBUG_OBJECT (src, "EOS");
      ret = GST_FLOW_EOS;
      goto error;
    }
    to_read -= read_size;
    byte_read += read_size;

    src->read_position += read_size;
    src->fd_offset += read_size;
  }

  gst_memory_unmap (mem, &info);

  buf = gst_buffer_new ();
  gst_buffer_append_memory (buf, mem);

  *buffer = buf;
  return GST_FLOW_OK;

error:
  if (mem) {
    gst_memory_unmap (mem, &info);
    gst_memory_unref (mem);
  }

  return ret;
}

/**
 * @brief Start datareposrc, open the file
 */
static gboolean
gst_data_repo_src_start (GstDataRepoSrc * src)
{
  g_autofree gchar *filename = NULL;
  struct_stat stat_results;
  int flags = O_RDONLY | O_BINARY;

  g_return_val_if_fail (src != NULL, FALSE);

  if (src->filename == NULL || src->filename[0] == '\0')
    goto no_filename;

  src->current_sample_index = src->start_sample_index;
  src->num_samples = src->stop_sample_index - src->start_sample_index + 1;
  GST_INFO_OBJECT (src,
      "The number of samples to be used out of the total samples in the file is %d, [%d] ~ [%d]",
      src->num_samples, src->start_sample_index, src->stop_sample_index);
  GST_INFO_OBJECT (src, "data type: %d", src->data_type);
  if (src->data_type == GST_DATA_REPO_DATA_IMAGE) {
    filename = gst_data_repo_src_get_image_filename (src);
  } else {
    filename = g_strdup (src->filename);
  }

  GST_INFO_OBJECT (src, "opening file %s", filename);

  /* open the file */
  src->fd = g_open (filename, flags, 0);

  if (src->fd < 0)
    goto open_failed;

  /* check if it is a regular file, otherwise bail out */
  if (fstat (src->fd, &stat_results) < 0)
    goto no_stat;

  src->file_size = stat_results.st_size;
  if (src->file_size == 0)
    goto error_close;

  if (S_ISDIR (stat_results.st_mode))
    goto was_directory;

  if (S_ISSOCK (stat_results.st_mode))
    goto was_socket;

  /* record if it's a regular (hence seekable and lengthable) file */
  if (!S_ISREG (stat_results.st_mode))
    goto error_close;

  src->read_position = 0;

  if (src->data_type == GST_DATA_REPO_DATA_IMAGE) {
    /* no longer used */
    gst_data_repo_src_safe_close_fd (src);
  } else {
    /* set start offset and last offset */
    src->start_offset =
        gst_data_repo_src_get_file_offset (src, src->start_sample_index);

    /* If the user does not set stop_sample_index, datareposrc need to calculate the last offset */
    src->last_offset =
        gst_data_repo_src_get_file_offset (src, src->stop_sample_index);

    src->fd_offset = lseek (src->fd, src->start_offset, SEEK_SET);
    GST_LOG_OBJECT (src, "Start file offset 0x%" G_GINT64_MODIFIER "x",
        src->fd_offset);
  }

  return TRUE;

  /* ERROR */
no_filename:
  {
    GST_ELEMENT_ERROR (src, RESOURCE, NOT_FOUND,
        ("No file name specified for reading."), (NULL));
    goto error_exit;
  }
open_failed:
  {
    switch (errno) {
      case ENOENT:
        GST_ELEMENT_ERROR (src, RESOURCE, NOT_FOUND, (NULL),
            ("No such file \"%s\"", src->filename));
        break;
      default:
        GST_ELEMENT_ERROR (src, RESOURCE, OPEN_READ,
            (("Could not open file \"%s\" for reading."), src->filename),
            GST_ERROR_SYSTEM);
        break;
    }
    goto error_exit;
  }
no_stat:
  {
    GST_ELEMENT_ERROR (src, RESOURCE, OPEN_READ,
        (("Could not get info on \"%s\"."), src->filename), (NULL));
    goto error_close;
  }
was_directory:
  {
    GST_ELEMENT_ERROR (src, RESOURCE, OPEN_READ,
        (("\"%s\" is a directory."), src->filename), (NULL));
    goto error_close;
  }
was_socket:
  {
    GST_ELEMENT_ERROR (src, RESOURCE, OPEN_READ,
        (("File \"%s\" is a socket."), src->filename), (NULL));
    goto error_close;
  }

error_close:
  gst_data_repo_src_safe_close_fd (src);
error_exit:
  return FALSE;
}

/**
 * @brief Set timestamp
 */
static void
gst_data_repo_src_set_timestamp (GstDataRepoSrc * src, GstBuffer * buffer)
{

  GstClockTime next_time;
  GstClockTime duration = GST_CLOCK_TIME_NONE;

  g_return_if_fail (src != NULL);
  g_return_if_fail (buffer != NULL);

  /* Unlike video, audio should control one sample by framerate. */
  if (src->data_type == GST_DATA_REPO_DATA_AUDIO) {
    GST_WARNING_OBJECT (src,
        "Use audiorate element for the framerate of audio");
    return;
  }

  duration = gst_util_uint64_scale_int (GST_SECOND, src->rate_d, src->rate_n);
  GST_BUFFER_DURATION (buffer) = duration;
  GST_BUFFER_TIMESTAMP (buffer) = src->running_time;
  gst_object_sync_values (GST_OBJECT (src), GST_BUFFER_TIMESTAMP (buffer));

  next_time =
      gst_util_uint64_scale (src->n_frame++, src->rate_d * GST_SECOND,
      src->rate_n);
  src->running_time = next_time;

  GST_LOG_OBJECT (src, "next_time %" GST_TIME_FORMAT "",
      GST_TIME_ARGS (next_time));
  GST_LOG_OBJECT (src,
      "timestamp [%" GST_TIME_FORMAT " dur %" GST_TIME_FORMAT "]",
      GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (buffer)),
      GST_TIME_ARGS (GST_BUFFER_DURATION (buffer)));
}

/**
 * @brief Function to create a buffer
 */
static GstFlowReturn
gst_data_repo_src_create (GstPushSrc * pushsrc, GstBuffer ** buffer)
{
  GstFlowReturn ret = GST_FLOW_OK;
  GstDataRepoSrc *src;
  src = GST_DATA_REPO_SRC (pushsrc);

  /** set_caps is completed after PAUSED_TO_PLAYING, so we cannot use change_state.
      datareposrc can get type and size after set_caps() */
  if (!src->is_start) {
    if (!gst_data_repo_src_start (src)) {
      return GST_FLOW_ERROR;
    }
    src->is_start = TRUE;
  }

  switch (src->data_type) {
    case GST_DATA_REPO_DATA_VIDEO:
    case GST_DATA_REPO_DATA_AUDIO:
    case GST_DATA_REPO_DATA_TEXT:
    case GST_DATA_REPO_DATA_OCTET:
      ret = gst_data_repo_src_read_others (src, buffer);
      break;
    case GST_DATA_REPO_DATA_TENSOR:
      if (gst_tensors_config_is_static (&src->config))
        ret = gst_data_repo_src_read_tensors (src, buffer);
      else
        ret = gst_data_repo_src_read_flexible_or_sparse_tensors (src, buffer);
      break;
    case GST_DATA_REPO_DATA_IMAGE:
      ret = gst_data_repo_src_read_multi_images (src, buffer);
      break;
    default:
      return GST_FLOW_ERROR;
  }

  if (ret != GST_FLOW_OK)
    return ret;

  if (src->rate_n)
    gst_data_repo_src_set_timestamp (src, *buffer);

  return GST_FLOW_OK;
}

/**
 * @brief Stop datareposrc, unmap and close the file
 */
static gboolean
gst_data_repo_src_stop (GstBaseSrc * basesrc)
{
  GstDataRepoSrc *src = GST_DATA_REPO_SRC (basesrc);

  /* close the file */
  gst_data_repo_src_safe_close_fd (src);

  return TRUE;
}

/**
 * @brief Get caps with tensors_sequence applied
 */
static gboolean
gst_data_repo_get_caps_by_tensors_sequence (GstDataRepoSrc * src)
{
  GstTensorsConfig src_config, dst_config;
  GstTensorInfo *_src_info, *_dst_info;
  guint i;
  guint seq_num = 0;
  GstCaps *new_caps;

  g_return_val_if_fail (src != NULL, FALSE);
  g_return_val_if_fail (src->caps != NULL, FALSE);

  if (!gst_tensors_config_from_caps (&src_config, src->caps, TRUE))
    return FALSE;

  gst_tensors_config_init (&dst_config);

  /* Copy selected tensors in sequence */
  for (i = 0; i < src->tensors_seq_cnt; i++) {
    seq_num = src->tensors_seq[i];
    _src_info = gst_tensors_info_get_nth_info (&src_config.info, seq_num);
    _dst_info = gst_tensors_info_get_nth_info (&dst_config.info, i);
    gst_tensor_info_copy (_dst_info, _src_info);
  }
  dst_config.rate_n = src_config.rate_n;
  dst_config.rate_d = src_config.rate_d;
  dst_config.info.format = src_config.info.format;
  dst_config.info.num_tensors = src->tensors_seq_cnt;

  new_caps = gst_tensors_caps_from_config (&dst_config);

  GST_DEBUG_OBJECT (src,
      "datareposrc caps by tensors_sequence %" GST_PTR_FORMAT, new_caps);

  gst_tensors_config_free (&dst_config);
  gst_tensors_config_free (&src_config);

  return gst_data_repo_src_parse_caps (src, new_caps);
}

/**
 * @brief Get caps for caps negotiation
 */
static GstCaps *
gst_data_repo_src_get_caps (GstBaseSrc * basesrc, GstCaps * filter)
{
  GstDataRepoSrc *src = GST_DATA_REPO_SRC (basesrc);

  if (src->data_type == GST_DATA_REPO_DATA_TENSOR && src->need_changed_caps) {
    gst_data_repo_get_caps_by_tensors_sequence (src);
    src->need_changed_caps = FALSE;
  }

  GST_DEBUG_OBJECT (src, "Current datareposrc caps %" GST_PTR_FORMAT,
      src->caps);

  if (src->caps) {
    if (filter)
      return gst_caps_intersect_full (filter, src->caps,
          GST_CAPS_INTERSECT_FIRST);
    else
      return gst_caps_ref (src->caps);
  } else {
    if (filter)
      return gst_caps_ref (filter);
    else
      return gst_caps_new_any ();
  }
}

/**
 * @brief caps after caps negotiation
 */
static gboolean
gst_data_repo_src_set_caps (GstBaseSrc * basesrc, GstCaps * caps)
{
  GstDataRepoSrc *src = GST_DATA_REPO_SRC (basesrc);

  GST_INFO_OBJECT (src, "set caps: %" GST_PTR_FORMAT, caps);

  return TRUE;
}

/**
 * @brief Read JSON file
 */
static gboolean
gst_data_repo_src_read_json_file (GstDataRepoSrc * src)
{
  g_autofree gchar *contents = NULL;
  GError *error = NULL;
  GFile *file;
  JsonNode *root;
  JsonObject *object;
  const gchar *caps_str = NULL;
  GstCaps *new_caps;
  gint64 val;

  g_return_val_if_fail (src != NULL, FALSE);
  g_return_val_if_fail (src->json_filename != NULL, FALSE);

  if ((file = g_file_new_for_path (src->json_filename)) == NULL) {
    GST_ERROR_OBJECT (src, "Failed to get file object of %s.",
        src->json_filename);
    return FALSE;
  }

  if (!g_file_load_contents (file, NULL, &contents, NULL, NULL, &error)) {
    GST_ERROR_OBJECT (src, "Failed to open %s: %s", src->json_filename,
        error ? error->message : "Unknown error");
    g_clear_error (&error);
    g_object_unref (file);
    return FALSE;
  }

  if (src->parser)
    g_object_unref (src->parser);
  src->parser = json_parser_new ();

  if (!json_parser_load_from_data (src->parser, contents, -1, NULL)) {
    GST_ERROR_OBJECT (src, "Failed to load data from %s", src->json_filename);
    goto error;
  }

  root = json_parser_get_root (src->parser);
  if (!JSON_NODE_HOLDS_OBJECT (root)) {
    GST_ERROR_OBJECT (src, "it does not contain a JsonObject: %s", contents);
    goto error;
  }

  object = json_node_get_object (root);

  GST_INFO_OBJECT (src, ">>>>>>> Start parsing JSON file(%s)",
      src->json_filename);

  if (!json_object_has_member (object, "gst_caps")) {
    GST_ERROR_OBJECT (src, "There is no gst_caps field: %s", contents);
    goto error;
  }

  caps_str = json_object_get_string_member (object, "gst_caps");
  GST_INFO_OBJECT (src, "caps_str : %s", caps_str);

  new_caps = gst_caps_from_string (caps_str);

  /* calculate media size from gst caps */
  if (!gst_data_repo_src_parse_caps (src, new_caps))
    goto error;

  GST_INFO_OBJECT (src, "gst_caps : %" GST_PTR_FORMAT, src->caps);

  /* In the case of below media type, get sample_size from JSON */
  if (src->data_type == GST_DATA_REPO_DATA_TEXT
      || src->data_type == GST_DATA_REPO_DATA_OCTET
      || src->data_type == GST_DATA_REPO_DATA_IMAGE) {
    if (!json_object_has_member (object, "sample_size")) {
      GST_ERROR_OBJECT (src, "There is not sample_size field: %s", contents);
      goto error;
    }

    val = json_object_get_int_member (object, "sample_size");
    if (val < 0) {
      GST_ERROR_OBJECT (src, "Invalid sample_size: %" G_GINT64_FORMAT, val);
      goto error;
    }

    src->sample_size = (gsize) val;
    GST_INFO_OBJECT (src, "sample_size: %zd", src->sample_size);
  }

  if (src->data_type == GST_DATA_REPO_DATA_TENSOR &&
      !gst_tensors_config_is_static (&src->config)) {
    if (!json_object_has_member (object, "sample_offset")) {
      GST_ERROR_OBJECT (src, "There is no sample_offset field: %s", contents);
      goto error;
    }
    src->sample_offset_array =
        json_object_get_array_member (object, "sample_offset");
    src->sample_offset_array_len =
        json_array_get_length (src->sample_offset_array);

    if (!json_object_has_member (object, "tensor_size")) {
      GST_ERROR_OBJECT (src, "There is no tensor_size field: %s", contents);
      goto error;
    }
    src->tensor_size_array =
        json_object_get_array_member (object, "tensor_size");
    src->tensor_size_array_len = json_array_get_length (src->tensor_size_array);
    GST_INFO_OBJECT (src, "tensor_size_array_len:%u",
        src->tensor_size_array_len);

    if (!json_object_has_member (object, "tensor_count")) {
      GST_ERROR_OBJECT (src, "There is no tensor_count field: %s", contents);
      goto error;
    }
    src->tensor_count_array =
        json_object_get_array_member (object, "tensor_count");
    src->tensor_count_array_len =
        json_array_get_length (src->tensor_count_array);
    GST_INFO_OBJECT (src, "tensor_count_array_len:%u",
        src->tensor_count_array_len);
  } else {
    if (src->sample_size == 0)
      goto error;
  }

  if (!json_object_has_member (object, "total_samples")) {
    GST_ERROR_OBJECT (src, "There is no total_samples field: %s", contents);
    goto error;
  }

  val = json_object_get_int_member (object, "total_samples");
  if (val <= 0) {
    GST_ERROR_OBJECT (src, "Invalid total_samples: %" G_GINT64_FORMAT, val);
    goto error;
  }

  src->total_samples = (guint) val;
  GST_INFO_OBJECT (src, "total_samples: %u", src->total_samples);

  g_object_unref (file);

  return TRUE;
error:
  src->data_type = GST_DATA_REPO_DATA_UNKNOWN;
  GST_ERROR_OBJECT (src, "Failed to parse %s", src->json_filename);
  g_object_unref (file);

  return FALSE;
}

/**
 * @brief Setter for datareposrc properties.
 */
static void
gst_data_repo_src_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDataRepoSrc *src;
  const GstCaps *caps;

  g_return_if_fail (GST_IS_DATA_REPO_SRC (object));

  src = GST_DATA_REPO_SRC (object);

  switch (prop_id) {
    case PROP_LOCATION:
      gst_data_repo_src_set_file_path (src, PROP_LOCATION,
          g_value_get_string (value), NULL);
      break;
    case PROP_JSON:
      gst_data_repo_src_set_file_path (src, PROP_JSON,
          g_value_get_string (value), NULL);
      /** To get caps, read JSON before Caps negotiation,
          to get information on sample data */
      if (!gst_data_repo_src_read_json_file (src)) {
        GST_ERROR_OBJECT (src, "Failed to get data format");
      }
      break;
    case PROP_START_SAMPLE_INDEX:
      src->start_sample_index = g_value_get_uint (value);
      break;
    case PROP_STOP_SAMPLE_INDEX:
      src->stop_sample_index = g_value_get_uint (value);
      break;
    case PROP_EPOCHS:
      src->epochs = g_value_get_uint (value);
      break;
    case PROP_IS_SHUFFLE:
      src->is_shuffle = g_value_get_boolean (value);
      break;
    case PROP_TENSORS_SEQUENCE:
      g_free (src->tensors_seq_str);
      src->tensors_seq_str = g_value_dup_string (value);
      if (!gst_data_repo_src_set_tensors_sequence (src)) {
        GST_ERROR_OBJECT (src, "Failed to set tensors sequence");
      } else {
        src->need_changed_caps = TRUE;
      }
      break;
    case PROP_CAPS:
      caps = gst_value_get_caps (value);
      if (caps) {
        gst_data_repo_src_parse_caps (src, gst_caps_copy (caps));
      }
      /** let's retry set tensors-sequence.
          if caps property is set later than tensors-sequence property,
          setting tensors-sequence fails because caps information is unknown.*/
      if (src->tensors_seq_str) {
        if (gst_data_repo_src_set_tensors_sequence (src))
          src->need_changed_caps = TRUE;
      }
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter datareposrc properties
 */
static void
gst_data_repo_src_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstDataRepoSrc *src;

  g_return_if_fail (GST_IS_DATA_REPO_SRC (object));

  src = GST_DATA_REPO_SRC (object);

  switch (prop_id) {
    case PROP_LOCATION:
      g_value_set_string (value, src->filename);
      break;
    case PROP_JSON:
      g_value_set_string (value, src->json_filename);
      break;
    case PROP_START_SAMPLE_INDEX:
      g_value_set_uint (value, src->start_sample_index);
      break;
    case PROP_STOP_SAMPLE_INDEX:
      g_value_set_uint (value, src->stop_sample_index);
      break;
    case PROP_EPOCHS:
      g_value_set_uint (value, src->epochs);
      break;
    case PROP_IS_SHUFFLE:
      g_value_set_boolean (value, src->is_shuffle);
      break;
    case PROP_TENSORS_SEQUENCE:
      g_value_set_string (value, src->tensors_seq_str);
      break;
    case PROP_CAPS:
      gst_value_set_caps (value, src->caps);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Change state of datareposrc.
 */
static GstStateChangeReturn
gst_data_repo_src_change_state (GstElement * element, GstStateChange transition)
{
  guint i;
  GstDataRepoSrc *src = GST_DATA_REPO_SRC (element);
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;
  GstBaseSrc *basesrc = NULL;
  guint blocksize;

  switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
      GST_INFO_OBJECT (src, "NULL_TO_READY");

      if (src->data_type == GST_DATA_REPO_DATA_UNKNOWN)
        goto state_change_failed;

      /** if data_type is not GST_DATA_REPO_DATA_UNKNOWN and sample_size is 0 then
          'caps' is set by property and sample size needs to be set by blocksize
          (in the case of otect and text) */
      if (src->sample_size == 0 && (src->data_type == GST_DATA_REPO_DATA_OCTET
              || src->data_type == GST_DATA_REPO_DATA_TEXT)) {
        basesrc = GST_BASE_SRC (src);
        g_object_get (G_OBJECT (basesrc), "blocksize", &blocksize, NULL);
        GST_DEBUG_OBJECT (src, "blocksize = %u", blocksize);
        if (blocksize == 0) {
          GST_ERROR_OBJECT (src, "Please set the 'blocksize' property "
              "when using the 'caps' property to set the sample format without JSON.");
          goto state_change_failed;
        }
        src->sample_size = blocksize;
      }

      /** A case of importing a sample format using 'caps' property without JSON. */
      if (src->total_samples == 0 && src->stop_sample_index == 0) {
        GST_ERROR_OBJECT (src, "Please set the 'stop-sample-index' property "
            "when using the 'caps' property to set the sample format without JSON.");
        goto state_change_failed;
      }

      /* total_samples -1 is the default value of 'stop-sample-index' property */
      if (src->stop_sample_index == 0)
        src->stop_sample_index = src->total_samples - 1;

      /* Check invalid property value */
      if (src->start_sample_index > (src->total_samples - 1)
          || src->stop_sample_index > (src->total_samples - 1)
          || src->epochs == 0) {
        GST_ERROR_OBJECT (src, "Check for invalid range values");

        goto state_change_failed;
      }

      /* If tensors-sequence properties is set */
      if (src->tensors_seq_str != NULL) {
        if (src->data_type != GST_DATA_REPO_DATA_TENSOR) {
          GST_ERROR_OBJECT (src,
              "tensors-sequence properties is only for tensor/others type(%d), current type(%d)",
              GST_DATA_REPO_DATA_TENSOR, src->data_type);
          goto state_change_failed;
        }
        /* After gst_data_repo_src_set_tensors_sequence() */
        if (src->tensors_seq_cnt == 0)
          goto state_change_failed;
      } else {
        for (i = 0; i < src->config.info.num_tensors; i++)
          src->tensors_seq[i] = i;
        src->tensors_seq_cnt = i;
      }

      break;

    case GST_STATE_CHANGE_READY_TO_PAUSED:
      GST_INFO_OBJECT (src, "READY_TO_PAUSED");
      break;

    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      GST_INFO_OBJECT (src, "PAUSED_TO_PLAYING");
      break;

    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      GST_INFO_OBJECT (src, "PLAYING_TO_PAUSED");
      break;

    case GST_STATE_CHANGE_PAUSED_TO_READY:
      GST_INFO_OBJECT (src, "PAUSED_TO_READY");
      break;

    case GST_STATE_CHANGE_READY_TO_NULL:
      GST_INFO_OBJECT (src, "READY_TO_NULL");
      break;

    default:
      break;
  }

  return ret;

state_change_failed:
  GST_ERROR_OBJECT (src, "state change failed");

  return GST_STATE_CHANGE_FAILURE;
}
