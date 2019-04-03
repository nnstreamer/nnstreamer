/**
 * GStreamer Tensor_Source_IIO
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2019 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	tensor_src_iio.c
 * @date	27 Feb 2019
 * @brief	GStreamer plugin to capture sensor data as tensor(s)
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @todo  create a sample example and unit tests
 * @todo  support specific channels as input
 * @todo  support buffer timestamps based on IIO channel timestamp
 * @todo  support one-shot mode
 *
 *
 * This is the plugin to capture data from sensors
 * and convert them to tensor format.
 * Current implementation will support accelerators, light and gyro sensors.
 *
 */

/**
 * SECTION:element-tensor_src_iio
 *
 * Source element to handle sensors as input.
 * The output are always in the format of other/tensor or other/tensors.
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gstinfo.h>
#include <gst/gst.h>
#include <glib.h>
#include <string.h>
#include <endian.h>
#include <unistd.h>
#include <errno.h>

#include "tensor_src_iio.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

/**
 * @brief Macro for debug message.
 */
#define silent_debug(...) do { \
    if (DBG) { \
      GST_DEBUG_OBJECT (self, __VA_ARGS__); \
    } \
  } while (0)

GST_DEBUG_CATEGORY_STATIC (gst_tensor_src_iio_debug);
#define GST_CAT_DEFAULT gst_tensor_src_iio_debug

/**
 * @brief Macro to generate data processing functions for various types
 */
#define PROCESS_SCANNED_DATA(DTYPE_UNSIGNED, DTYPE_SIGNED) \
/**
 * @brief process scanned data to float based on type info from channel
 * @param[in] prop Proprty of the channel whose data is processed
 * @param[in] value Raw value scanned from the channel
 * @returns processed value in float
 */ \
static gfloat \
gst_tensor_src_iio_process_scanned_data_from_##DTYPE_UNSIGNED ( \
    GstTensorSrcIIOChannelProperties *prop, DTYPE_UNSIGNED value_unsigned) { \
  gfloat value_float; \
  \
  g_assert (sizeof (DTYPE_UNSIGNED) == sizeof (DTYPE_SIGNED)); \
  \
  value_unsigned >>= prop->shift; \
  value_unsigned &= prop->mask; \
  if (prop->is_signed) { \
    DTYPE_SIGNED value_signed; \
    guint shift_value; \
    \
    shift_value = (sizeof (DTYPE_UNSIGNED) * 8) - prop->used_bits; \
    value_signed = ((DTYPE_SIGNED) (value_unsigned << shift_value)) >> \
        shift_value; \
    value_float = ((gfloat) value_signed + prop->offset) * prop->scale; \
  } else { \
    value_float = ((gfloat) value_unsigned + prop->offset) * prop->scale; \
  } \
  return value_float; \
}

/**
 * @brief tensor_src_iio properties.
 */
enum
{
  PROP_0,
  PROP_MODE,
  PROP_SILENT,
  PROP_DEVICE,
  PROP_DEVICE_NUM,
  PROP_TRIGGER,
  PROP_TRIGGER_NUM,
  PROP_CHANNELS,
  PROP_BUFFER_CAPACITY,
  PROP_FREQUENCY,
  PROP_MERGE_CHANNELS
};

/**
 * @brief iio device channel enabled mode
 */
#define CHANNELS_ENABLED_AUTO_CHAR "auto"
#define CHANNELS_ENABLED_ALL_CHAR "all"
#define DEFAULT_OPERATING_CHANNELS_ENABLED CHANNELS_ENABLED_AUTO_CHAR

/**
 * @brief tensor_src_iio device modes
 */
#define MODE_ONE_SHOT "one-shot"
#define MODE_CONTINUOUS "continuous"
#define DEFAULT_OPERATING_MODE MODE_CONTINUOUS

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_PROP_SILENT TRUE

/**
 * @brief Flag for general default value of string
 */
#define DEFAULT_PROP_STRING NULL

/**
 * @brief Minimum and maximum buffer length for iio
 */
#define MIN_BUFFER_CAPACITY 1
#define MAX_BUFFER_CAPACITY G_MAXUINT
#define DEFAULT_BUFFER_CAPACITY 1

/**
 * @brief Minimum and maximum operating frequency for the device
 * Frequency 0 chooses the first available frequency supported by device
 */
#define MIN_FREQUENCY 0
#define MAX_FREQUENCY G_MAXULONG
#define DEFAULT_FREQUENCY 0

/**
 * @brief Default behavior on merging channels
 */
#define DEFAULT_MERGE_CHANNELS TRUE

/**
 * @brief default trigger and device numbers
 */
#define DEFAULT_PROP_DEVICE_NUM -1
#define DEFAULT_PROP_TRIGGER_NUM -1

/**
 * blocksize for buffer
 */
#define BLOCKSIZE 1

/**
 * @brief IIO devices/triggers
 */
#define DEVICE "device"
#define BUFFER "buffer"
#define TRIGGER "trigger"
#define CHANNELS "scan_elements"
#define IIO "iio:"
#define TIMESTAMP "timestamp"
#define DEVICE_PREFIX IIO DEVICE
#define TRIGGER_PREFIX IIO TRIGGER
#define CURRENT_TRIGGER "current_trigger"

/**
 * @brief IIO device channels
 */
#define EN_SUFFIX "_en"
#define INDEX_SUFFIX "_index"
#define TYPE_SUFFIX "_type"
#define SCALE_SUFFIX "_scale"
#define OFFSET_SUFFIX "_offset"

/**
 * @brief filenames for IIO devices/triggers characteristics
 */
#define NAME_FILE "name"
#define AVAIL_FREQUENCY_FILE "sampling_frequency_available"
#define SAMPLING_FREQUENCY "sampling_frequency"

/**
 * @brief Template for src pad.
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT));

/** Define data processing functions for various types */
PROCESS_SCANNED_DATA (guint8, gint8);
PROCESS_SCANNED_DATA (guint16, gint16);
PROCESS_SCANNED_DATA (guint32, gint32);
PROCESS_SCANNED_DATA (guint64, gint64);

/** GObject method implementation */
static void gst_tensor_src_iio_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_src_iio_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_src_iio_finalize (GObject * object);

/** GstBaseSrc method implementation */
static gboolean gst_tensor_src_iio_start (GstBaseSrc * src);
static gboolean gst_tensor_src_iio_stop (GstBaseSrc * src);
static gboolean gst_tensor_src_iio_event (GstBaseSrc * src, GstEvent * event);
static gboolean gst_tensor_src_iio_query (GstBaseSrc * src, GstQuery * query);
static gboolean gst_tensor_src_iio_set_caps (GstBaseSrc * src, GstCaps * caps);
static GstCaps *gst_tensor_src_iio_get_caps (GstBaseSrc * src,
    GstCaps * filter);
static GstCaps *gst_tensor_src_iio_fixate (GstBaseSrc * src, GstCaps * caps);
static gboolean gst_tensor_src_iio_is_seekable (GstBaseSrc * src);
static GstFlowReturn gst_tensor_src_iio_create (GstBaseSrc * src,
    guint64 offset, guint size, GstBuffer ** buf);
static GstFlowReturn gst_tensor_src_iio_fill (GstBaseSrc * src, guint64 offset,
    guint size, GstBuffer * buf);
static void gst_tensor_src_iio_get_times (GstBaseSrc * basesrc,
    GstBuffer * buffer, GstClockTime * start, GstClockTime * end);

/** internal functions */

#define gst_tensor_src_iio_parent_class parent_class
G_DEFINE_TYPE (GstTensorSrcIIO, gst_tensor_src_iio, GST_TYPE_BASE_SRC);

/**
 * @brief initialize the tensor_src_iio class.
 */
static void
gst_tensor_src_iio_class_init (GstTensorSrcIIOClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSrcClass *bsrc_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);
  bsrc_class = GST_BASE_SRC_CLASS (klass);

  /** GObject methods */
  gobject_class->set_property = gst_tensor_src_iio_set_property;
  gobject_class->get_property = gst_tensor_src_iio_get_property;
  gobject_class->finalize = gst_tensor_src_iio_finalize;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent",
          "Produce verbose output", DEFAULT_PROP_SILENT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODE,
      g_param_spec_string ("mode", "Operating mode",
          "Mode for the device to run in - one-shot or continuous",
          DEFAULT_OPERATING_MODE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_DEVICE,
      g_param_spec_string ("device", "Device Name",
          "Name of the device to be opened", DEFAULT_PROP_STRING,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_DEVICE_NUM,
      g_param_spec_int ("device-number", "Device Number",
          "Number (numeric id) of the device to be opened",
          -1, G_MAXINT, DEFAULT_PROP_DEVICE_NUM, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_TRIGGER,
      g_param_spec_string ("trigger", "Trigger Name",
          "Name of the trigger to be used", DEFAULT_PROP_STRING,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_TRIGGER_NUM,
      g_param_spec_int ("trigger-number", "Trigger Number",
          "Number (numeric id) of the trigger to be opened",
          -1, G_MAXINT, DEFAULT_PROP_TRIGGER_NUM, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_CHANNELS,
      g_param_spec_string ("channels", "Channels to be enabled",
          "Enable channels -"
          "auto: enable all channels when no channels are enabled automatically"
          "all: enable all channels",
          DEFAULT_OPERATING_CHANNELS_ENABLED, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_BUFFER_CAPACITY,
      g_param_spec_uint ("buffer-capacity", "Buffer Capacity",
          "Capacity of the data buffer", MIN_BUFFER_CAPACITY,
          MAX_BUFFER_CAPACITY, DEFAULT_BUFFER_CAPACITY,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_FREQUENCY,
      g_param_spec_ulong ("frequency", "Frequency",
          "Operating frequency of the device", MIN_FREQUENCY, MAX_FREQUENCY,
          DEFAULT_FREQUENCY, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MERGE_CHANNELS,
      g_param_spec_boolean ("merge-channels-data", "Merge Channels Data",
          "Merge the data of channels into single tensor",
          DEFAULT_MERGE_CHANNELS, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata (gstelement_class,
      "TensorSrcIIO",
      "SrcIIO/Tensor",
      "Src element to support linux IIO",
      "Parichay Kapoor <pk.kapoor@samsung.com>");

  /** pad template */
  gst_element_class_add_static_pad_template (gstelement_class, &src_factory);

  /** GstBaseSrcIIO methods */
  bsrc_class->start = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_start);
  bsrc_class->stop = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_stop);
  bsrc_class->event = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_event);
  bsrc_class->query = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_query);
  bsrc_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_set_caps);
  bsrc_class->get_caps = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_get_caps);
  bsrc_class->fixate = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_fixate);
  bsrc_class->is_seekable = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_is_seekable);
  bsrc_class->create = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_create);
  bsrc_class->fill = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_fill);
  bsrc_class->get_times = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_get_times);
}

/**
 * @brief delete GstTensorSrcIIODeviceProperties structure
 * @param[in] data Data pointer to be freed
 */
static void
gst_tensor_src_iio_channel_properties_free (gpointer data)
{
  GstTensorSrcIIOChannelProperties *prop =
      (GstTensorSrcIIOChannelProperties *) data;
  g_free (prop->name);
  g_free (prop->generic_name);
  g_free (prop->base_dir);
  g_free (prop->base_file);
  g_free (prop);
}

/**
 * @brief initialize GstTensorSrcIIODeviceProperties structure
 * @param[in] data Device properties pointer to be initialized
 */
static void
gst_tensor_src_iio_device_properties_init (GstTensorSrcIIODeviceProperties *
    prop)
{
  prop->name = NULL;
  prop->base_dir = NULL;
  prop->id = -1;
}

/**
 * @brief initialize tensor_src_iio element.
 */
static void
gst_tensor_src_iio_init (GstTensorSrcIIO * self)
{
  /** init properties */
  self->configured = FALSE;
  self->channels = DEFAULT_PROP_STRING;
  self->mode = g_strdup (DEFAULT_OPERATING_MODE);
  self->channels_enabled = CHANNELS_ENABLED_AUTO;
  gst_tensor_src_iio_device_properties_init (&self->trigger);
  gst_tensor_src_iio_device_properties_init (&self->device);
  self->silent = DEFAULT_PROP_SILENT;
  self->buffer_capacity = DEFAULT_BUFFER_CAPACITY;
  self->sampling_frequency = DEFAULT_FREQUENCY;
  self->merge_channels_data = DEFAULT_MERGE_CHANNELS;
  self->is_tensor = FALSE;
  self->tensors_config = NULL;
  self->default_sampling_frequency = 0;
  self->default_buffer_capacity = 0;
  self->default_trigger = NULL;

  /**
   * format of the source since IIO device as a source is live and operates
   * at a fixed frequency, GST_FORMAT_TIME is used
   */
  gst_base_src_set_format (GST_BASE_SRC (self), GST_FORMAT_TIME);
  /** set the source to be live */
  gst_base_src_set_live (GST_BASE_SRC (self), TRUE);
  /** set the timestamps on each buffer */
  gst_base_src_set_do_timestamp (GST_BASE_SRC (self), TRUE);
}

/**
 * @brief merge multiple other/tensor
 * @note they should have matching type and shape to form 1 other/tensors
 * @note extra dimension should be available for other/tensors
 * @note order of merge is stable
 * @note merging into 1 tensor only supported using innermost dimension.
 * @param[in/out] info Tensor info to be merged
 * @param[in] size Info array size
 * @param[in] dir Innermost/outermost/innermost-outer (0/1/2) available dimension
 * @returns >=0 number of valid entries in the info after merge
 *         -1 failed due to missing extra dimension/mismatch shape/type
 */
static gint
gst_tensor_src_merge_tensor_by_type (GstTensorInfo * info, guint size,
    guint dir)
{
  gint info_idx, base_idx, dim_idx;
  gboolean mismatch = FALSE, dim_avail = FALSE;
  gint merge_dim = -1;

  /** base error control check */
  g_return_val_if_fail (size > 0, 0);
  base_idx = 0;

  /** verify extra dimension (innermost to outermost) */
  for (dim_idx = 0; dim_idx < NNS_TENSOR_RANK_LIMIT; dim_idx++) {
    if (info[base_idx].dimension[dim_idx] == 1) {
      dim_avail = TRUE;
    }
  }

  /** verify that all the types and shapes match */
  for (info_idx = 0; info_idx < size; info_idx++) {
    if (!gst_tensor_info_is_equal (info + base_idx, info + info_idx)) {
      mismatch = TRUE;
      break;
    }
  }

  /** return original if cant be merged and size within limits */
  if (mismatch || !dim_avail) {
    if (size > NNS_TENSOR_SIZE_LIMIT) {
      return -1;
    } else {
      return size;
    }
  }

  /**
   * If there are multiple available dimensions to merge along, we use dir
   * to choose which the dimension to merge. If there is just 1 dimension,
   * dir variable has no effect
   */
  if (dir == 0) {
    for (dim_idx = 0; dim_idx < NNS_TENSOR_RANK_LIMIT; dim_idx++) {
      if (info[base_idx].dimension[dim_idx] == 1) {
        merge_dim = dim_idx;
        break;
      }
    }
  } else if (dir == 1) {
    for (dim_idx = NNS_TENSOR_RANK_LIMIT - 1; dim_idx >= 0; dim_idx--) {
      if (info[base_idx].dimension[dim_idx] == 1) {
        merge_dim = dim_idx;
        break;
      }
    }
  } else if (dir == 2) {
    for (dim_idx = NNS_TENSOR_RANK_LIMIT - 1; dim_idx >= 0; dim_idx--) {
      if (info[base_idx].dimension[dim_idx] != 1) {
        merge_dim = dim_idx + 1;
        break;
      }
    }
  } else {
    return -1;
  }

  /** No outer dimension available to merge */
  if (merge_dim >= NNS_TENSOR_RANK_LIMIT || merge_dim < 0) {
    return size;
  }

  /** Now merge into 1 tensor using the selected dimension*/
  info[0].dimension[merge_dim] = size;
  return 1;
}

/**
 * @brief check if device/trigger with the given name exists
 * @param[in] dir_name Directory containing all the devices
 * @param[in] name Name of the device to be found
 * @param[in] prefix Prefix to match with the filename of the device
 * @return >=0 if OK, represents device/trigger number
 *         -1  if returned with error
 */
static gint
gst_tensor_src_iio_get_id_by_name (const gchar * dir_name, const gchar * name,
    const gchar * prefix)
{
  DIR *dptr = NULL;
  GError *error = NULL;
  struct dirent *dir_entry;
  gchar *filename = NULL;
  gint id = -1;
  gchar *file_contents = NULL;
  gint ret = -1;

  if (!g_file_test (dir_name, G_FILE_TEST_IS_DIR)) {
    GST_ERROR ("No channels available.");
    return ret;
  }
  dptr = opendir (dir_name);
  if (G_UNLIKELY (NULL == dptr)) {
    GST_ERROR ("Error in opening directory %s.\n", dir_name);
    return ret;
  }

  while ((dir_entry = readdir (dptr)) != NULL) {
    /** check for prefix and the next digit should be a number */
    if (g_str_has_prefix (dir_entry->d_name, prefix) &&
        g_ascii_isdigit (dir_entry->d_name[strlen (prefix)])) {

      id = g_ascii_strtoll (dir_entry->d_name + strlen (prefix), NULL, 10);
      filename =
          g_build_filename (dir_name, dir_entry->d_name, NAME_FILE, NULL);

      if (!g_file_get_contents (filename, &file_contents, NULL, &error)) {
        GST_ERROR ("Unable to read %s, error: %s.\n", filename, error->message);
        g_error_free (error);
        goto error_free_filename;
      }
      g_free (filename);

      if (!g_strcmp0 (file_contents, name)) {
        ret = id;
        g_free (file_contents);
        break;
      }
      g_free (file_contents);
    }
  }

  closedir (dptr);
  return ret;

error_free_filename:
  g_free (filename);
  closedir (dptr);
  return ret;
}

/**
 * @brief check if device/trigger with the given id exists
 * @param[in] dir_name Directory containing all the devices
 * @param[in] id ID of the device to be found
 * @param[in] prefix Prefix to match with the filename of the device
 * @return name on success (owned by caller), else NULL
 */
static gchar *
gst_tensor_src_iio_get_name_by_id (const gchar * dir_name, const gint id,
    const gchar * prefix)
{
  GError *error = NULL;
  gchar *filename = NULL;
  gchar *dev_dirname = NULL;
  gchar *file_contents = NULL;

  dev_dirname = g_strdup_printf ("%s%d", prefix, id);
  filename = g_build_filename (dir_name, dev_dirname, NAME_FILE, NULL);
  g_free (dev_dirname);

  if (!g_file_test (filename, G_FILE_TEST_IS_REGULAR)) {
    GST_ERROR ("No device available with id %d.", id);
    goto exit_free_filename;
  }

  if (!g_file_get_contents (filename, &file_contents, NULL, &error)) {
    GST_ERROR ("Unable to read %s, error: %s.\n", filename, error->message);
    g_error_free (error);
    goto exit_free_filename;
  }

exit_free_filename:
  g_free (filename);
  return file_contents;
}

/**
 * @brief parse float value from the file
 * @param[in] dirname Directory containing the file
 * @param[in] name Filename of the file
 * @param[in] suffix Suffix to be attached to the filename
 * @param[in/out] value Output value returned via value
 * @return FALSE on errors, else TRUE
 */
static gboolean
gst_tensor_src_iio_get_float_from_file (const gchar * dirname,
    const gchar * name, const gchar * suffix, gfloat * value)
{
  gchar *filename, *filepath, *file_contents = NULL;

  filename = g_strdup_printf ("%s%s", name, suffix);
  filepath = g_build_filename (dirname, filename, NULL);

  if (!g_file_get_contents (filepath, &file_contents, NULL, NULL)) {
    GST_WARNING ("Unable to retrieve data from file %s.", filename);
  } else {
    if (!sscanf (file_contents, "%f", value)) {
      GST_ERROR ("Error in parsing float.");
      goto failure;
    }
    g_free (file_contents);
  }
  g_free (filename);
  g_free (filepath);

  return TRUE;

failure:
  g_free (file_contents);
  g_free (filename);
  g_free (filepath);
  return FALSE;
}

/**
 * @brief get type info about the channel from the string
 * @param[in/out] prop Channel properties where type info will be set
 * @param[in] contents Contains type unparsed information to be set
 * @return True if info was successfully set, false is info is not be parsed
 *         correctly
 */
static gboolean
gst_tensor_src_iio_set_channel_type (GstTensorSrcIIOChannelProperties * prop,
    const gchar * contents)
{
  gchar endianchar = '\0', signchar = '\0';
  gint arguments_filled;
  gboolean ret = TRUE;
  arguments_filled =
      sscanf (contents, "%ce:%c%u/%u>>%u", &endianchar, &signchar,
      &prop->used_bits, &prop->storage_bits, &prop->shift);
  if (prop->storage_bits > 0) {
    prop->storage_bytes = ((prop->storage_bits - 1) >> 3) + 1;
  } else {
    GST_WARNING ("Storage bits are 0 for channel %s.", prop->name);
    prop->storage_bytes = 0;
  }

  if (prop->used_bits == 64) {
    prop->mask = G_MAXUINT64;
  } else {
    prop->mask = (G_GUINT64_CONSTANT (1) << prop->used_bits) - 1;
  }

  /** checks to verify device channel type settings */
  g_return_val_if_fail (arguments_filled == 5, FALSE);
  g_return_val_if_fail (prop->storage_bytes <= 8, FALSE);
  g_return_val_if_fail (prop->storage_bits >= prop->used_bits, FALSE);
  g_return_val_if_fail (prop->storage_bits > prop->shift, FALSE);

  if (endianchar == 'b') {
    prop->big_endian = TRUE;
  } else if (endianchar == 'l') {
    prop->big_endian = FALSE;
  } else {
    ret = FALSE;
  }
  if (signchar == 's') {
    prop->is_signed = TRUE;
  } else if (signchar == 'u') {
    prop->is_signed = FALSE;
  } else {
    ret = FALSE;
  }

  return ret;
}

/**
 * @brief get generic name for channel from the string
 * @param[in] channel_name Name of the channel with its id embedded in it
 * @return Ptr to the generic name of the channel, caller should free the
 *         returned string
 */
static gchar *
gst_tensor_src_iio_get_generic_name (const gchar * channel_name)
{
  guint digit_len = 1;
  gchar *generic_name;
  guint channel_name_len = strlen (channel_name);

  while (g_ascii_isdigit (channel_name[channel_name_len - digit_len])) {
    digit_len++;
  }
  generic_name = g_strndup (channel_name, channel_name_len - digit_len + 1);

  return generic_name;
}

/**
 * @brief compare channels for sort based on their indices
 * @param[in] a First param to be compared
 * @param[in] b Second param to be compared
 * @return negative if a<b
 *         zero if a==b
 *         positive if a>b
 */
static gint
gst_tensor_channel_list_sort_cmp (gconstpointer a, gconstpointer b)
{
  const GstTensorSrcIIOChannelProperties *a_ch = a;
  const GstTensorSrcIIOChannelProperties *b_ch = b;
  gint compare_result = a_ch->index - b_ch->index;
  return compare_result;
}

/**
 * @brief compare channels for filtering if enabled
 * @param[in] data Pointer of the data of the element
 * @param[in/out] user_data Pointer to the address of the list to be filtered
 */
static void
gst_tensor_channel_list_filter_enabled (gpointer data, gpointer user_data)
{
  GstTensorSrcIIOChannelProperties *channel;
  GList **list_addr;
  GList *list;

  channel = (GstTensorSrcIIOChannelProperties *) data;
  list_addr = (GList **) user_data;
  list = *list_addr;

  if (!channel->enabled) {
    *list_addr = g_list_remove (list, data);
    gst_tensor_src_iio_channel_properties_free (channel);
  }
}

/**
 * @brief get info about all the channels in the device
 * @param[in/out] self Tensor src IIO object
 * @param[in] dir_name Directory name with all the scan elements for device
 * @return >=0 number of enabled channels
 *         -1  if any error when scanning channels
 */
static gint
gst_tensor_src_iio_get_all_channel_info (GstTensorSrcIIO * self,
    const gchar * dir_name)
{
  DIR *dptr = NULL;
  GError *error = NULL;
  const struct dirent *dir_entry;
  gchar *filename = NULL;
  gchar *file_contents = NULL;
  gint ret = -1;
  guint value;
  guint num_channels_enabled = 0;
  gboolean generic_val, specific_val;
  gchar *generic_type_filename;
  GstTensorSrcIIOChannelProperties *channel_prop;

  if (!g_file_test (dir_name, G_FILE_TEST_IS_DIR)) {
    GST_ERROR_OBJECT (self, "No channels available.");
    return ret;
  }
  dptr = opendir (dir_name);
  if (G_UNLIKELY (NULL == dptr)) {
    GST_ERROR_OBJECT (self, "Error in opening directory %s.\n", dir_name);
    return ret;
  }

  while ((dir_entry = readdir (dptr)) != NULL) {
    /** check for enable */
    if (g_str_has_suffix (dir_entry->d_name, EN_SUFFIX)) {
      /** not enabling and handling buffer timestamps for now */
      if (g_str_has_prefix (dir_entry->d_name, TIMESTAMP)) {
        continue;
      }

      channel_prop = g_new (GstTensorSrcIIOChannelProperties, 1);
      self->channels = g_list_prepend (self->channels, channel_prop);

      /** set the name and base_dir */
      channel_prop->name = g_strndup (dir_entry->d_name,
          strlen (dir_entry->d_name) - strlen (EN_SUFFIX));
      channel_prop->base_dir = g_strdup (dir_name);
      channel_prop->base_file =
          g_build_filename (dir_name, channel_prop->name, NULL);
      channel_prop->generic_name =
          gst_tensor_src_iio_get_generic_name (channel_prop->name);
      silent_debug ("Generic name = %s", channel_prop->generic_name);

      /** find and set the current state */
      filename = g_strdup_printf ("%s%s", channel_prop->base_file, EN_SUFFIX);
      if (!g_file_get_contents (filename, &file_contents, NULL, &error)) {
        GST_ERROR_OBJECT (self, "Unable to read %s, error: %s.\n", filename,
            error->message);
        goto error_free_filename;
      }
      g_free (filename);

      value = g_ascii_strtoull (file_contents, NULL, 10);
      g_free (file_contents);
      if (value == 1) {
        channel_prop->enabled = TRUE;
        channel_prop->pre_enabled = TRUE;
        num_channels_enabled += 1;
      } else if (value == 0) {
        channel_prop->enabled = FALSE;
        channel_prop->pre_enabled = FALSE;
      } else {
        GST_ERROR_OBJECT
            (self,
            "Enable bit %u (out of range) in current state of channel %s.\n",
            value, channel_prop->name);
        goto error_cleanup_list;
      }

      /** find and set the index */
      filename =
          g_strdup_printf ("%s%s", channel_prop->base_file, INDEX_SUFFIX);
      if (!g_file_get_contents (filename, &file_contents, NULL, &error)) {
        GST_ERROR_OBJECT (self, "Unable to read %s, error: %s.\n", filename,
            error->message);
        goto error_free_filename;
      }
      g_free (filename);

      value = g_ascii_strtoull (file_contents, NULL, 10);
      g_free (file_contents);
      channel_prop->index = value;

      /** find and set the type information */
      filename = g_strdup_printf ("%s%s", channel_prop->base_file, TYPE_SUFFIX);
      if (!g_file_test (filename, G_FILE_TEST_IS_REGULAR)) {
        /** if specific type info unavailable, use generic type info */
        g_free (filename);
        generic_type_filename =
            g_strdup_printf ("%s%s", channel_prop->generic_name, TYPE_SUFFIX);
        filename =
            g_build_filename (channel_prop->base_dir, generic_type_filename,
            NULL);
        g_free (generic_type_filename);
      }
      if (!g_file_get_contents (filename, &file_contents, NULL, &error)) {
        GST_ERROR_OBJECT (self, "Unable to read %s, error: %s.\n", filename,
            error->message);
        goto error_free_filename;
      }
      g_free (filename);

      if (!gst_tensor_src_iio_set_channel_type (channel_prop, file_contents)) {
        GST_ERROR_OBJECT (self,
            "Error while setting up channel type for channel %s.\n",
            channel_prop->name);
        g_free (file_contents);
        goto error_cleanup_list;
      }
      g_free (file_contents);

      /** find and setup offset info */
      channel_prop->scale = 1.0;

      specific_val =
          gst_tensor_src_iio_get_float_from_file (self->device.base_dir,
          channel_prop->name, SCALE_SUFFIX, &channel_prop->scale);
      generic_val =
          gst_tensor_src_iio_get_float_from_file (self->device.base_dir,
          channel_prop->generic_name, SCALE_SUFFIX, &channel_prop->scale);
      if (!specific_val || !generic_val) {
        goto error_cleanup_list;
      }

      /** find and setup scale info */
      channel_prop->offset = 0.0;

      specific_val =
          gst_tensor_src_iio_get_float_from_file (self->device.base_dir,
          channel_prop->name, OFFSET_SUFFIX, &channel_prop->offset);
      generic_val =
          gst_tensor_src_iio_get_float_from_file (self->device.base_dir,
          channel_prop->generic_name, OFFSET_SUFFIX, &channel_prop->offset);
      if (!specific_val || !generic_val) {
        goto error_cleanup_list;
      }
    }
  }

  /** sort the list with the order of the indices */
  self->channels =
      g_list_sort (self->channels, gst_tensor_channel_list_sort_cmp);
  ret = num_channels_enabled;

  closedir (dptr);
  return ret;

error_free_filename:
  g_error_free (error);
  g_free (filename);

error_cleanup_list:
  g_list_free_full (self->channels, gst_tensor_src_iio_channel_properties_free);
  self->channels = NULL;

  closedir (dptr);
  return ret;
}

/**
 * @brief check if device/trigger with the given name exists
 * @param[in] base_dir Device base directory (containing sampling freq file)
 * @param[in] frequency Frequency specified by user (else 0)
 * @return >0 if OK, represents sampling frequency to be set
 *         0  if sampling frequency file does not exist, dont change anything
 *         -1 if any error occurs
 */
static gint64
gst_tensor_src_iio_set_frequency (const gchar * base_dir,
    const guint64 frequency)
{
  GError *error = NULL;
  gchar *filename = NULL;
  gchar *file_contents = NULL;
  gint i = 0;
  guint64 val = 0;
  gint64 ret = 0;

  /** get frequency list supported by the device */
  filename = g_build_filename (base_dir, AVAIL_FREQUENCY_FILE, NULL);
  if (!g_file_test (filename, G_FILE_TEST_IS_REGULAR)) {
    GST_WARNING ("Sampling frequency file does not exist for the file %s.\n",
        base_dir);
    goto del_filename;
  }
  if (!g_file_get_contents (filename, &file_contents, NULL, &error)) {
    GST_ERROR ("Unable to read sampling frequency for device %s.\n", base_dir);
    g_error_free (error);
    ret = -1;
    goto del_filename;
  }

  gchar **freq_list = g_strsplit (file_contents, " ", -1);
  gint num = g_strv_length (freq_list);
  if (num == 0) {
    GST_ERROR ("No sampling frequencies for device %s.\n", base_dir);
    ret = -1;
    goto del_freq_list;
  }
  /**
   * if the frequency is set 0, set the first available frequency
   * else verify the frequency received from user is supported by the device
   */
  if (frequency == 0) {
    ret = g_ascii_strtoull (freq_list[0], NULL, 10);
  } else {
    for (i = 0; i < num; i++) {
      val = g_ascii_strtoull (freq_list[i], NULL, 10);
      if (frequency == val) {
        ret = frequency;
        break;
      }
    }
  }

del_freq_list:
  g_strfreev (freq_list);

del_filename:
  g_free (filename);
  return ret;
}

/**
 * @brief set tensor_src_iio properties
 */
static void
gst_tensor_src_iio_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorSrcIIO *self;
  self = GST_TENSOR_SRC_IIO (object);

  GstStateChangeReturn status;
  GstState state;

  /**
   * No support for setting properties in PAUSED/PLAYING state as it needs to
   * reset the device. To change the properties, user should stop the pipeline
   * and set element state to READY/NULL and then change the properties
   */
  status = gst_element_get_state (GST_ELEMENT (self), &state, NULL,
      GST_CLOCK_TIME_NONE);
  if (status == GST_STATE_CHANGE_FAILURE || status == GST_STATE_CHANGE_ASYNC
      || state == GST_STATE_PLAYING || state == GST_STATE_PAUSED) {
    GST_ERROR_OBJECT (self, "Can only set property in NULL or READY state.");
    return;
  }

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      break;

    case PROP_MODE:
    {
      if (self->mode != NULL) {
        g_free (self->mode);
      }
      self->mode = g_value_dup_string (value);
      break;
    }

    case PROP_DEVICE:
    {
      if (self->device.name != NULL) {
        g_free (self->device.name);
      }
      self->device.name = g_value_dup_string (value);
      break;
    }

    case PROP_DEVICE_NUM:
      self->device.id = g_value_get_int (value);
      break;

    case PROP_TRIGGER:
    {
      if (self->trigger.name != NULL) {
        g_free (self->trigger.name);
      }
      self->trigger.name = g_value_dup_string (value);
      break;
    }

    case PROP_TRIGGER_NUM:
      self->trigger.id = g_value_get_int (value);
      break;

    case PROP_CHANNELS:
    {
      const gchar *param = g_value_get_string (value);
      if (!g_strcmp0 (param, CHANNELS_ENABLED_ALL_CHAR)) {
        self->channels_enabled = CHANNELS_ENABLED_ALL;
      } else if (!g_strcmp0 (param, CHANNELS_ENABLED_AUTO_CHAR)) {
        self->channels_enabled = CHANNELS_ENABLED_AUTO;
      }
      break;
    }

    case PROP_BUFFER_CAPACITY:
      self->buffer_capacity = g_value_get_uint (value);
      break;

    case PROP_FREQUENCY:
      self->sampling_frequency = (guint64) g_value_get_ulong (value);
      break;

    case PROP_MERGE_CHANNELS:
      self->merge_channels_data = g_value_get_boolean (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief get tensor_src_iio properties
 */
static void
gst_tensor_src_iio_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorSrcIIO *self;
  self = GST_TENSOR_SRC_IIO (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;

    case PROP_MODE:
      g_value_set_string (value, self->mode);
      break;

    case PROP_DEVICE:
      g_value_set_string (value, self->device.name);
      break;

    case PROP_DEVICE_NUM:
      g_value_set_int (value, self->device.id);
      break;

    case PROP_TRIGGER:
      g_value_set_string (value, self->trigger.name);
      break;

    case PROP_TRIGGER_NUM:
      g_value_set_int (value, self->trigger.id);
      break;

    case PROP_CHANNELS:
    {
      if (self->channels_enabled == CHANNELS_ENABLED_ALL) {
        g_value_set_string (value, CHANNELS_ENABLED_ALL_CHAR);
      } else if (self->channels_enabled == CHANNELS_ENABLED_AUTO) {
        g_value_set_string (value, CHANNELS_ENABLED_AUTO_CHAR);
      }
      break;
    }

    case PROP_BUFFER_CAPACITY:
      g_value_set_uint (value, self->buffer_capacity);
      break;

    case PROP_FREQUENCY:
      /** interface of frequency is kept long for outside but uint64 inside */
      g_value_set_ulong (value, (gulong) self->sampling_frequency);
      break;

    case PROP_MERGE_CHANNELS:
      g_value_set_boolean (value, self->merge_channels_data);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief finalize the instance
 */
static void
gst_tensor_src_iio_finalize (GObject * object)
{
  GstTensorSrcIIO *self;
  self = GST_TENSOR_SRC_IIO (object);

  g_free (self->mode);
  g_free (self->device.name);
  g_free (self->trigger.name);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief write the string in to the file
 * @param[in] self Tensor src IIO object
 * @param[in] file Destination file for the data
 * @param[in] base_dir Directory containing the file
 * @param[in] contents Data to be written to the file
 * @return True if write was successful, false on failure
 */
static gboolean
gst_tensor_write_sysfs_string (GstTensorSrcIIO * self, const gchar * file,
    const gchar * base_dir, const gchar * contents)
{
  gchar *filename = NULL;
  gboolean ret = FALSE;
  gint bytes_printed = 0;
  FILE *fd = NULL;
  GError *error = NULL;

  filename = g_build_filename (base_dir, file, NULL);
  fd = fopen (filename, "w");
  if (fd == NULL) {
    GST_ERROR_OBJECT (self, "Unable to open file to write %s.\n", filename);
    goto error_free_filename;
  }

  bytes_printed = fprintf (fd, "%s", contents);
  if (bytes_printed != strlen (contents)) {
    GST_ERROR_OBJECT (self, "Unable to write to file %s.\n", filename);
    goto error_close_file;
  }
  if (fclose (fd)) {
    GST_ERROR_OBJECT (self, "Unable to close file %s after write.\n", filename);
    goto error_free_filename;
  }
  ret = TRUE;

  if (DBG) {
    gchar *file_contents = NULL;
    ret = FALSE;
    if (!g_file_get_contents (filename, &file_contents, NULL, &error)) {
      GST_ERROR_OBJECT (self, "Unable to read file %s with error %s.\n",
          filename, error->message);
      g_error_free (error);
      goto error_free_filename;
    } else {
      if (!g_strcmp0 (contents, file_contents)) {
        ret = TRUE;
      }
      g_free (file_contents);
    }
  }

  g_free (filename);
  return ret;

error_close_file:
  fclose (fd);

error_free_filename:
  g_free (filename);
  return ret;
}

/**
 * @brief write the int in to the file
 * @param[in] self Tensor src IIO object
 * @param[in] file Destination file for the data
 * @param[in] base_dir Directory containing the file
 * @param[in] contents Data to be written to the file
 * @return True if write was successful, false on failure
 */
static gboolean
gst_tensor_write_sysfs_int (GstTensorSrcIIO * self, const gchar * file,
    const gchar * base_dir, const gint contents)
{
  gchar *contents_char = NULL;
  gboolean ret;

  contents_char = g_strdup_printf ("%d", contents);
  ret = gst_tensor_write_sysfs_string (self, file, base_dir, contents_char);

  g_free (contents_char);
  return ret;
}

/**
 * @brief set value to all the channels
 * @param[in] self Tensor src IIO object
 * @param[in] contents Data to be written to the file
 * @return True if write was successful, false if failure on any channel
 */
static gboolean
gst_tensor_set_all_channels (GstTensorSrcIIO * self, const gint contents)
{
  GList *ch_list;
  gchar *filename = NULL;
  GstTensorSrcIIOChannelProperties *channel_prop = NULL;
  gboolean ret = TRUE;

  for (ch_list = self->channels; ch_list != NULL; ch_list = ch_list->next) {
    channel_prop = (GstTensorSrcIIOChannelProperties *) ch_list->data;
    filename = g_strdup_printf ("%s%s", channel_prop->name, EN_SUFFIX);
    ret &=
        gst_tensor_write_sysfs_int (self, filename, channel_prop->base_dir,
        contents);
    g_free (filename);
  }

  return ret;
}

/**
 * @brief get the size of the combined data from channels
 * @param[in] channels List of all the channels
 * @return Size of one scan of data combined from all the channels
 *
 * Also evaluates the location of each channel in the buffer
 */
static guint
gst_tensor_get_size_from_channels (GList * channels)
{
  guint size_bytes = 0;
  GList *list;
  GstTensorSrcIIOChannelProperties *channel_prop = NULL;

  for (list = channels; list != NULL; list = list->next) {
    channel_prop = (GstTensorSrcIIOChannelProperties *) list->data;
    guint remain = size_bytes % channel_prop->storage_bytes;
    if (remain == 0) {
      channel_prop->location = size_bytes;
    } else {
      channel_prop->location =
          size_bytes - remain + channel_prop->storage_bytes;
    }
    size_bytes = channel_prop->location + channel_prop->storage_bytes;
  }

  return size_bytes;
}

/**
 * @brief create the structure for the caps to update the src pad caps
 * @param[in/out] structure Caps structure which will filled
 * @returns True if structure is created and filled, False for any error
 */
static gboolean
gst_tensor_src_iio_create_config (GstTensorSrcIIO * tensor_src_iio)
{
  GList *list;
  GstTensorSrcIIOChannelProperties *channel_prop;
  gint tensor_info_merged_size;
  gint info_idx = 0, dim_idx = 0;
  GstTensorInfo *info;

  /**
   * create a bigger array, insert info in it and
   * then merge tensors with same type+size
   */
  info = g_new (GstTensorInfo, tensor_src_iio->num_channels_enabled);

  /** compile tensor info data */
  for (list = tensor_src_iio->channels; list != NULL; list = list->next) {
    channel_prop = (GstTensorSrcIIOChannelProperties *) list->data;
    if (!channel_prop->enabled)
      continue;
    info[info_idx].name = channel_prop->name;
    info[info_idx].type = _NNS_FLOAT32;
    for (dim_idx = 0; dim_idx < NNS_TENSOR_RANK_LIMIT; dim_idx++) {
      info[info_idx].dimension[dim_idx] = 1;
    }
    info[info_idx].dimension[1] = tensor_src_iio->buffer_capacity;
    info_idx += 1;
  }
  g_assert_cmpint (info_idx, ==, tensor_src_iio->num_channels_enabled);

  /** merge info about the tensors with same type */
  tensor_info_merged_size = tensor_src_iio->num_channels_enabled;
  if (tensor_src_iio->merge_channels_data) {
    tensor_info_merged_size =
        gst_tensor_src_merge_tensor_by_type (info,
        tensor_src_iio->num_channels_enabled, 0);
  }

  /** verify the merging of the array */
  if (tensor_info_merged_size < 0) {
    GST_ERROR_OBJECT (tensor_src_iio, "Mismatch while merging tensor");
    goto error_ret;
  } else if (tensor_info_merged_size == 0) {
    GST_ERROR_OBJECT (tensor_src_iio, "No info to be merged");
    goto error_ret;
  } else if (tensor_info_merged_size > NNS_TENSOR_SIZE_LIMIT) {
    GST_ERROR_OBJECT (tensor_src_iio,
        "Number of tensors required %u for data exceed the max limit",
        tensor_info_merged_size);
    goto error_ret;
  }

  /** tensors config data */
  tensor_src_iio->is_tensor = (tensor_info_merged_size == 1);
  GstTensorsConfig *config = g_new (GstTensorsConfig, 1);
  if (config == NULL) {
    goto error_ret;
  }
  gst_tensors_config_init (config);
  for (info_idx = 0; info_idx < tensor_info_merged_size; info_idx++) {
    gst_tensor_info_copy (&config->info.info[info_idx], &info[info_idx]);
  }

  /**
   * buffer_capacity number of data samples are captured at once, packed
   * together and sent downstream
   */
  config->rate_n = tensor_src_iio->sampling_frequency;
  config->rate_d = tensor_src_iio->buffer_capacity;
  config->info.num_tensors = tensor_info_merged_size;

  tensor_src_iio->tensors_config = config;

  g_free (info);
  return TRUE;

error_ret:
  g_free (info);
  return FALSE;
}

/**
 * @brief start function, called when state changed null to ready.
 * load the device and init the device resources
 */
static gboolean
gst_tensor_src_iio_start (GstBaseSrc * src)
{
  /** load and init resources */
  GstTensorSrcIIO *self;
  self = GST_TENSOR_SRC_IIO_CAST (src);
  gint64 sampling_frequency;
  gchar *dirname = NULL;
  gchar *filename = NULL;
  gint num_channels_enabled;
  gchar *file_contents = NULL;
  gsize length = 0;

  /** no support one shot mode for now */
  if (!g_strcmp0 (self->mode, MODE_ONE_SHOT)) {
    GST_ERROR_OBJECT (self, "One-shot mode not yet supported.");
    goto error_return;
  }

  /** Find the device */
  if (self->device.name != NULL) {
    self->device.id =
        gst_tensor_src_iio_get_id_by_name (IIO_BASE_DIR, self->device.name,
        DEVICE_PREFIX);
  } else if (self->device.id >= 0) {
    self->device.name = gst_tensor_src_iio_get_name_by_id (IIO_BASE_DIR,
        self->device.id, DEVICE_PREFIX);
  } else {
    GST_ERROR_OBJECT (self, "IIO device information not provided.");
    goto error_return;
  }
  if (G_UNLIKELY (self->device.name == NULL || self->device.id < 0)) {
    GST_ERROR_OBJECT (self, "Cannot find the specified IIO device.");
    goto error_return;
  }
  dirname = g_strdup_printf ("%s%d", DEVICE_PREFIX, self->device.id);
  self->device.base_dir = g_build_filename (IIO_BASE_DIR, dirname, NULL);
  g_free (dirname);

  /** register the trigger */
  if (self->trigger.name != NULL || self->trigger.id >= 0) {
    /** verify if trigger is supported by our device */
    gchar *trigger_device_dir =
        g_build_filename (self->device.base_dir, TRIGGER, NULL);
    if (!g_file_test (trigger_device_dir, G_FILE_TEST_IS_DIR)) {
      GST_ERROR_OBJECT (self, "IIO device %s does not supports trigger.\n",
          self->device.name);
      g_free (trigger_device_dir);
      goto error_device_free;
    }
    g_free (trigger_device_dir);

    /** find if the provided trigger exists */
    if (self->trigger.name != NULL) {
      self->trigger.id =
          gst_tensor_src_iio_get_id_by_name (IIO_BASE_DIR, self->trigger.name,
          TRIGGER_PREFIX);
    } else {
      self->trigger.name =
          gst_tensor_src_iio_get_name_by_id (IIO_BASE_DIR, self->trigger.id,
          TRIGGER_PREFIX);
    }
    if (G_UNLIKELY (self->trigger.name == NULL || self->trigger.id < 0)) {
      GST_ERROR_OBJECT (self, "Cannot find the specified IIO trigger.");
      goto error_device_free;
    }
    dirname = g_strdup_printf ("%s%d", TRIGGER_PREFIX, self->trigger.id);
    self->trigger.base_dir = g_build_filename (IIO_BASE_DIR, dirname, NULL);
    g_free (dirname);

    /** get the default trigger, if any */
    filename =
        g_build_filename (self->device.base_dir, TRIGGER, CURRENT_TRIGGER,
        NULL);
    if (!g_file_get_contents (filename, &self->default_trigger, NULL, NULL)) {
      GST_WARNING_OBJECT (self, "Unable to read default set trigger.");
    }
    g_free (filename);
    /** set the trigger */
    filename = g_build_filename (TRIGGER, CURRENT_TRIGGER, NULL);
    if (G_UNLIKELY (!gst_tensor_write_sysfs_string (self, filename,
                self->device.base_dir, self->trigger.name))) {
      GST_ERROR_OBJECT (self,
          "Cannot set the IIO device trigger: %s for device: %s.\n",
          self->trigger.name, self->device.name);
      g_free (filename);
      goto error_trigger_free;
    }
    g_free (filename);
  }

  /** setup the frequency (only verifying the frequency now) */
  /** @todo verify setting up frequency */
  sampling_frequency =
      gst_tensor_src_iio_set_frequency (self->device.base_dir,
      self->sampling_frequency);
  if (-1 == sampling_frequency) {
    GST_ERROR_OBJECT (self, "Error in setting frequency for device %s.\n",
        self->device.name);
    goto error_trigger_free;
  } else if (0 == sampling_frequency) {
    /** if sampling frequency file does not exist, no error */
    GST_WARNING_OBJECT (self, "Cannot verify against sampling frequency list.");
  } else {
    filename =
        g_build_filename (self->device.base_dir, SAMPLING_FREQUENCY, NULL);
    /** check if sampling frequency can be set */
    if (!g_file_test (filename, G_FILE_TEST_IS_REGULAR)) {
      GST_WARNING_OBJECT (self, "Cannot set sampling frequency.");
      g_free (filename);
    } else {
      /** store the default frequency */
      if (!g_file_get_contents (filename, &file_contents, &length, NULL)) {
        GST_WARNING_OBJECT (self, "Unable to read default sampling frequency.");
      } else if (file_contents != NULL) {
        self->default_sampling_frequency =
            g_ascii_strtoull (file_contents, NULL, 10);
      }
      g_free (file_contents);
      g_free (filename);

      self->sampling_frequency = sampling_frequency;
      /** interface of frequency is kept long for outside but uint64 inside */
      gchar *sampling_frequency_char =
          g_strdup_printf ("%lu", (gulong) self->sampling_frequency);
      if (G_UNLIKELY (!gst_tensor_write_sysfs_string (self, SAMPLING_FREQUENCY,
                  self->device.base_dir, sampling_frequency_char))) {
        GST_ERROR_OBJECT (self,
            "Cannot set the sampling frequency for device: %s.\n",
            self->device.name);
        g_free (sampling_frequency_char);
        goto error_trigger_free;
      }
      g_free (sampling_frequency_char);
    }
  }

  /** once all these are set, set the buffer related thingies */
  dirname = g_build_filename (self->device.base_dir, BUFFER, NULL);
  filename = g_build_filename (dirname, "length", NULL);
  if (!g_file_get_contents (filename, &file_contents, &length, NULL)) {
    GST_WARNING_OBJECT (self, "Unable to read default buffer capacity.");
  } else if (file_contents != NULL && length > 0) {
    self->default_buffer_capacity = g_ascii_strtoull (file_contents, NULL, 10);
  }
  g_free (file_contents);
  g_free (filename);

  if (G_UNLIKELY (!gst_tensor_write_sysfs_int (self, "length", dirname,
              self->buffer_capacity))) {
    GST_ERROR_OBJECT (self,
        "Cannot set the IIO device buffer capacity for device: %s.\n",
        self->device.name);
    g_free (dirname);
    goto error_trigger_free;
  }
  g_free (dirname);

  /** get all the channels that exist and then set enable on them */
  dirname = g_build_filename (self->device.base_dir, CHANNELS, NULL);
  num_channels_enabled =
      gst_tensor_src_iio_get_all_channel_info (self, dirname);
  g_free (dirname);
  if (G_UNLIKELY (num_channels_enabled == -1)) {
    GST_ERROR_OBJECT (self, "Error while scanning channels for device: %s.\n",
        self->device.name);
    goto error_trigger_free;
  }

  if ((num_channels_enabled != g_list_length (self->channels)) &&
      (num_channels_enabled == 0
          || self->channels_enabled == CHANNELS_ENABLED_ALL)) {
    if (!gst_tensor_set_all_channels (self, 1)) {
      /** if enabling all channels failed, disable all channels */
      GST_ERROR_OBJECT (self, "Enabling all channels failed for device: %s,"
          "disabling all the channels.\n", self->device.name);
      gst_tensor_set_all_channels (self, 0);
      goto error_channels_free;
    }
  }

  /** filter out disabled channels */
  g_list_foreach (self->channels, gst_tensor_channel_list_filter_enabled,
      &self->channels);
  self->scan_size = gst_tensor_get_size_from_channels (self->channels);
  self->num_channels_enabled = g_list_length (self->channels);

  /** set fixed caps for the src pad */
  gst_pad_use_fixed_caps (src->srcpad);

  /** create tensor_config */
  if (!gst_tensor_src_iio_create_config (self)) {
    GST_ERROR_OBJECT (self, "Error creating config.\n");
    goto error_channels_free;
  }

  /** enable the buffer for the data to be captured */
  dirname = g_build_filename (self->device.base_dir, BUFFER, NULL);
  if (G_UNLIKELY (!gst_tensor_write_sysfs_int (self, "enable", dirname, 1))) {
    GST_ERROR_OBJECT (self,
        "Cannot enable the IIO device buffer for device: %s.\n",
        self->device.name);
    g_free (dirname);
    goto error_config_free;
  }
  g_free (dirname);

  /** open the buffer to read and ready the file descriptor */
  gchar *device_name = g_strdup_printf ("%s%d", DEVICE_PREFIX, self->device.id);
  filename = g_build_filename (IIO_DEV_DIR, device_name, NULL);
  g_free (device_name);

  self->buffer_data_fp = g_new (struct pollfd, 1);
  self->buffer_data_file = fopen (filename, "r");
  if (self->buffer_data_file == NULL) {
    GST_ERROR_OBJECT (self, "Failed to open buffer %s for device %s.\n",
        filename, self->device.name);
    g_free (filename);
    goto error_pollfd_free;
  }
  self->buffer_data_fp->fd = fileno (self->buffer_data_file);
  self->buffer_data_fp->events = POLLIN;
  g_free (filename);

  self->configured = TRUE;
  /** bytes every buffer will be fixed */
  gst_base_src_set_dynamic_size (src, FALSE);
  /** complete the start of the base src */
  gst_base_src_start_complete (src, GST_FLOW_OK);
  return TRUE;

error_pollfd_free:
  g_free (self->buffer_data_fp);

error_config_free:
  gst_tensors_info_free (&self->tensors_config->info);
  g_free (self->tensors_config);

error_channels_free:
  g_list_free_full (self->channels, gst_tensor_src_iio_channel_properties_free);
  self->channels = NULL;

error_trigger_free:
  g_free (self->trigger.base_dir);
  g_free (self->default_trigger);
  self->trigger.base_dir = NULL;
  self->default_trigger = NULL;

error_device_free:
  g_free (self->device.base_dir);
  self->device.base_dir = NULL;

error_return:
  /** complete the start of the base src */
  gst_base_src_start_complete (src, GST_FLOW_ERROR);
  return FALSE;
}

/**
 * @brief restore the iio device to its original device.
 */
static void
gst_tensor_src_restore_iio_device (GstTensorSrcIIO * self)
{
  GList *ch_list;
  gchar *filename = NULL, *dirname = NULL, *file_contents = NULL;
  GstTensorSrcIIOChannelProperties *channel_prop = NULL;

  /** disable the buffer */
  dirname = g_build_filename (self->device.base_dir, BUFFER, NULL);
  if (G_UNLIKELY (!gst_tensor_write_sysfs_int (self, "enable", dirname, 0))) {
    GST_ERROR_OBJECT (self,
        "Error in disabling the IIO device buffer for device: %s.\n",
        self->device.name);
  }
  g_free (dirname);

  /** reset enabled channels */
  for (ch_list = self->channels; ch_list != NULL; ch_list = ch_list->next) {
    channel_prop = (GstTensorSrcIIOChannelProperties *) ch_list->data;
    filename = g_strdup_printf ("%s%s", channel_prop->name, EN_SUFFIX);
    gst_tensor_write_sysfs_int (self, filename, channel_prop->base_dir,
        (int) channel_prop->pre_enabled);
    g_free (filename);
  }

  /** reset sampling_frequency */
  if (self->default_sampling_frequency > 0) {
    /** converting to long as setting interface to device */
    file_contents =
        g_strdup_printf ("%lu", (gulong) self->default_sampling_frequency);
    gst_tensor_write_sysfs_string (self, "sampling_frequency",
        self->device.base_dir, file_contents);
    g_free (file_contents);
  }

  /** reset buffer_capacity */
  dirname = g_build_filename (self->device.base_dir, BUFFER, NULL);
  if (self->default_buffer_capacity > 0) {
    gst_tensor_write_sysfs_int (self, "length", dirname,
        self->default_buffer_capacity);
  } else {
    gst_tensor_write_sysfs_string (self, "length", dirname, "");
  }
  g_free (dirname);

  /** reset current trigger */
  if (self->default_trigger != NULL) {
    filename = g_build_filename (TRIGGER, CURRENT_TRIGGER, NULL);
    gst_tensor_write_sysfs_string (self, filename, self->device.base_dir,
        self->default_trigger);
    g_free (filename);
  }
}

/**
 * @brief stop function, called when state changed ready to null.
 */
static gboolean
gst_tensor_src_iio_stop (GstBaseSrc * src)
{
  /** free resources related to the device */
  GstTensorSrcIIO *self;
  self = GST_TENSOR_SRC_IIO_CAST (src);

  self->configured = FALSE;

  /** restore the iio device */
  gst_tensor_src_restore_iio_device (self);

  g_free (self->buffer_data_fp);
  fclose (self->buffer_data_file);

  gst_tensors_info_free (&self->tensors_config->info);
  g_free (self->tensors_config);

  g_list_free_full (self->channels, gst_tensor_src_iio_channel_properties_free);
  self->channels = NULL;

  g_free (self->trigger.base_dir);
  g_free (self->default_trigger);
  self->trigger.base_dir = NULL;
  self->default_trigger = NULL;

  g_free (self->device.base_dir);
  self->device.base_dir = NULL;

  return TRUE;
}

/**
 * @brief handle events
 */
static gboolean
gst_tensor_src_iio_event (GstBaseSrc * src, GstEvent * event)
{
  /** No events to be handled yet */
  return GST_BASE_SRC_CLASS (parent_class)->event (src, event);
}

/**
 * @brief handle queries
 */
static gboolean
gst_tensor_src_iio_query (GstBaseSrc * src, GstQuery * query)
{
  gboolean res = FALSE;

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_SCHEDULING:
    {
      /** Only support sequential data access */
      gst_query_set_scheduling (query, GST_SCHEDULING_FLAG_SEQUENTIAL, 1, -1,
          0);
      /** Only support push mode for now */
      gst_query_add_scheduling_mode (query, GST_PAD_MODE_PUSH);

      res = TRUE;
      break;
    }
    default:
      res = GST_BASE_SRC_CLASS (parent_class)->query (src, query);
      break;
  }

  return res;
}

/**
 * @brief set new caps
 */
static gboolean
gst_tensor_src_iio_set_caps (GstBaseSrc * src, GstCaps * caps)
{
  GstTensorSrcIIO *self;
  GstPad *pad;

  self = GST_TENSOR_SRC_IIO (src);
  pad = src->srcpad;

  if (DBG) {
    GstCaps *cur_caps = gst_pad_get_current_caps (pad);
    if (cur_caps != NULL && !gst_caps_is_subset (caps, cur_caps)) {
      silent_debug ("Setting caps in source mismatch");
      silent_debug ("caps = %" GST_PTR_FORMAT, caps);
      silent_debug ("cur_caps = %" GST_PTR_FORMAT, cur_caps);
      gst_caps_unref (cur_caps);
    }
  }

  if (!gst_pad_set_caps (pad, caps)) {
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief get caps of subclass
 * @note basesrc _get_caps returns the caps from the pad_template
 * however, we set the caps manually and needs to returned here
 */
static GstCaps *
gst_tensor_src_iio_get_caps (GstBaseSrc * src, GstCaps * filter)
{
  GstCaps *caps;
  GstPad *pad;

  pad = src->srcpad;
  caps = gst_pad_get_current_caps (pad);
  if (caps == NULL) {
    caps = gst_pad_get_pad_template_caps (pad);
  }

  if (filter) {
    GstCaps *intersection;
    intersection =
        gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (caps);
    caps = intersection;
  }

  return caps;
}

/**
 * @brief fixate the caps when needed during negotiation
 */
static GstCaps *
gst_tensor_src_iio_fixate (GstBaseSrc * src, GstCaps * caps)
{
  /**
   * Caps are fixated based on the device source in _start().
   */
  GstTensorSrcIIO *self;
  self = GST_TENSOR_SRC_IIO_CAST (src);
  GstCaps *updated_caps, *fixated_caps;

  if (self->is_tensor) {
    GstTensorConfig tensor_config;
    gst_tensor_info_copy (&tensor_config.info,
        &(self->tensors_config->info.info[0]));
    tensor_config.rate_n = self->tensors_config->rate_n;
    tensor_config.rate_d = self->tensors_config->rate_d;
    fixated_caps = gst_tensor_caps_from_config (&tensor_config);
    gst_tensor_info_free (&tensor_config.info);
  } else {
    fixated_caps = gst_tensors_caps_from_config (self->tensors_config);
  }

  if (fixated_caps == NULL) {
    GST_ERROR_OBJECT (self, "Error creating fixated caps from config.");
    return NULL;
  }
  silent_debug ("Fixated caps from device = %" GST_PTR_FORMAT, fixated_caps);

  if (gst_caps_can_intersect (caps, fixated_caps)) {
    updated_caps = gst_caps_intersect (caps, fixated_caps);
  } else {
    GST_ERROR_OBJECT (self,
        "No intersection while fixating caps of the element.");
    gst_caps_unref (caps);
    gst_caps_unref (fixated_caps);
    return NULL;
  }

  gst_caps_unref (caps);
  gst_caps_unref (fixated_caps);
  return gst_caps_fixate (updated_caps);
}

/**
 * @brief check if source supports seeking
 */
static gboolean
gst_tensor_src_iio_is_seekable (GstBaseSrc * src)
{
  /** iio sensors are live source without any support for seeking */
  return FALSE;
}

/**
 * @brief returns the time for the buffers
 */
static void
gst_tensor_src_iio_get_times (GstBaseSrc * basesrc, GstBuffer * buffer,
    GstClockTime * start, GstClockTime * end)
{
  GstClockTime timestamp;
  GstClockTime duration;

  timestamp = GST_BUFFER_DTS (buffer);
  duration = GST_BUFFER_DURATION (buffer);

  /** can't sync using DTS, use PTS */
  if (!GST_CLOCK_TIME_IS_VALID (timestamp))
    timestamp = GST_BUFFER_PTS (buffer);

  if (GST_CLOCK_TIME_IS_VALID (timestamp)) {
    *start = timestamp;
    if (GST_CLOCK_TIME_IS_VALID (duration)) {
      *end = timestamp + duration;
    }
  }
}

/**
 * @brief create a buffer with requested size and offset
 * @note offset, size ignored as the tensor src iio does not support pull mode
 */
static GstFlowReturn
gst_tensor_src_iio_create (GstBaseSrc * src, guint64 offset,
    guint size, GstBuffer ** buffer)
{
  GstTensorSrcIIO *self;
  GstBuffer *buf;
  GstMemory *mem;
  guint buffer_size;

  self = GST_TENSOR_SRC_IIO_CAST (src);
  buffer_size = gst_tensor_info_get_size (&self->tensors_config->info.info[0]);

  mem = gst_allocator_alloc (NULL, buffer_size, NULL);
  if (mem == NULL) {
    GST_ERROR_OBJECT (self, "Error allocating memory for buffer.");
    return GST_FLOW_ERROR;
  }

  buf = gst_buffer_new ();
  gst_buffer_append_memory (buf, mem);

  if (gst_tensor_src_iio_fill (src, offset, buffer_size, buf) != GST_FLOW_OK) {
    goto error_buffer_unref;
  }

  *buffer = buf;
  return GST_FLOW_OK;

error_buffer_unref:
  gst_buffer_unref (buf);
  return GST_FLOW_ERROR;
}

/**
 * @brief process the scanned data from IIO device
 * @param[in] channels List of the all enabled channels
 * @param[in] data Data read from the IIO device
 * @param[in/out] buffer_map Gst buffer map to write data to
 * @returns FALSE if fail, else TRUE
 *
 * assumes each data starting point is byte aligned
 */
static gboolean
gst_tensor_src_iio_process_scanned_data (GList * channels, gchar * data,
    gfloat * buffer_map)
{
  GList *list;
  GstTensorSrcIIOChannelProperties *prop;
  guint64 storage_mask;
  guint channel_idx;

  for (list = channels, channel_idx = 0; list != NULL;
      list = list->next, channel_idx += 1) {
    prop = list->data;
    switch (prop->storage_bytes) {
      case 1:
      {
        guint8 value = *(guint8 *) (data + prop->location);
        /** right shift the extra storage bits */
        value >>= (8 - prop->storage_bits);
        buffer_map[channel_idx] =
            gst_tensor_src_iio_process_scanned_data_from_guint8 (prop, value);
        break;
      }
      case 2:
      {
        guint16 value = *(guint16 *) (data + prop->location);
        if (prop->big_endian) {
          value = GUINT16_FROM_BE (value);
          /** right shift the extra storage bits for big endian */
          value >>= (16 - prop->storage_bits);
        } else {
          value = GUINT16_FROM_LE (value);
          /** mask out the extra storage bits for little endian */
          storage_mask = (G_GUINT64_CONSTANT (1) << prop->storage_bits) - 1;
          value &= storage_mask;
        }
        buffer_map[channel_idx] =
            gst_tensor_src_iio_process_scanned_data_from_guint16 (prop, value);
        break;
      }
      case 3:
      /** follow through */
      case 4:
      {
        guint32 value = *(guint32 *) (data + prop->location);
        if (prop->big_endian) {
          value = GUINT32_FROM_BE (value);
          /** right shift the extra storage bits for big endian */
          value >>= (32 - prop->storage_bits);
        } else {
          value = GUINT32_FROM_LE (value);
          /** mask out the extra storage bits for little endian */
          storage_mask = (G_GUINT64_CONSTANT (1) << prop->storage_bits) - 1;
          value &= storage_mask;
        }
        buffer_map[channel_idx] =
            gst_tensor_src_iio_process_scanned_data_from_guint32 (prop, value);
        break;
      }
      case 5:
      /** follow through */
      case 6:
      /** follow through */
      case 7:
      /** follow through */
      case 8:
      {
        guint64 value = *(guint64 *) (data + prop->location);
        if (prop->big_endian) {
          value = GUINT64_FROM_BE (value);
          /** right shift the extra storage bits for big endian */
          value >>= (64 - prop->storage_bits);
        } else {
          value = GUINT64_FROM_LE (value);
          /** mask out the extra storage bits for little endian */
          storage_mask = (G_GUINT64_CONSTANT (1) << prop->storage_bits) - 1;
          value &= storage_mask;
        }
        buffer_map[channel_idx] =
            gst_tensor_src_iio_process_scanned_data_from_guint64 (prop, value);
        break;
      }
      default:
        GST_ERROR ("Storage bytes for channel %s out of bounds", prop->name);
        return FALSE;
    }
  }
  return TRUE;
}

/**
 * @brief fill the buffer with data
 * @note ignore offset as there is no header
 * @note buffer timestamp is already handled by gstreamer with gst clock
 *
 * @todo handle timestamp received from device
 * @todo sync buffer timestamp with IIO timestamps
 */
static GstFlowReturn
gst_tensor_src_iio_fill (GstBaseSrc * src, guint64 offset,
    guint size, GstBuffer * buffer)
{
  GstTensorSrcIIO *self;
  gint status, bytes_to_read;
  guint idx;
  gchar *raw_data_base, *raw_data;
  gfloat *map_data_float;
  GstMemory *mem;
  GstMapInfo map;

  self = GST_TENSOR_SRC_IIO (src);

  /** Only supporting tensors made of 1 tensor for now */
  g_assert (self->tensors_config->info.num_tensors == 1);
  g_assert (size ==
      gst_tensor_info_get_size (&self->tensors_config->info.info[0]));
  g_assert (gst_buffer_n_memory (buffer) ==
      self->tensors_config->info.num_tensors);

  /** get writable buffer */
  mem = gst_buffer_peek_memory (buffer, 0);
  g_assert (gst_memory_map (mem, &map, GST_MAP_WRITE));

  /** wait for the data to arrive */
  if (self->trigger.name != NULL) {
    status = poll (self->buffer_data_fp, 1, -1);
    if (status < 0) {
      GST_ERROR_OBJECT (self, "Error %d while polling the buffer.", status);
      goto error_mem_unmap;
    } else if (status == 0) {
      /** this case happens for timeout, which is not set yet */
      GST_ERROR_OBJECT (self, "Timeout while polling the buffer.");
      goto error_mem_unmap;
    }
  } else {
    /** sleep for a device tick */
    g_usleep (MAX (1,
            (guint64) 1000000 / (self->sampling_frequency /
                self->buffer_capacity)));
  }

  /** read the data from file */
  bytes_to_read = self->scan_size * self->buffer_capacity;
  raw_data_base = g_malloc (bytes_to_read);
  /** using read for non-blocking access */
  status = read (self->buffer_data_fp->fd, raw_data_base, bytes_to_read);
  if (status < bytes_to_read) {
    GST_ERROR_OBJECT (self,
        "Error: read %d/%d bytes while reading from the buffer. Error no: %d",
        status, bytes_to_read, errno);
    goto error_data_free;
  }

  /** parse the read data */
  map_data_float = (gfloat *) map.data;
  raw_data = raw_data_base;

  /**
   * current assumption is that the all data is float and merged to form
   * a 1 dimension data. 2nd dimension comes from buffer capacity.
   */
  for (idx = 0; idx < self->buffer_capacity; idx++) {
    map_data_float = ((gfloat *) map.data) + idx * self->num_channels_enabled;
    if (!gst_tensor_src_iio_process_scanned_data (self->channels, raw_data,
            map_data_float)) {
      GST_ERROR_OBJECT (self, "Error while processing scanned data.");
      goto error_data_free;
    }
    raw_data += self->scan_size;
  }

  /** wrap up the buffer */
  g_free (raw_data_base);
  gst_memory_unmap (mem, &map);

  return GST_FLOW_OK;

error_data_free:
  g_free (raw_data_base);

error_mem_unmap:
  gst_memory_unmap (mem, &map);

  return GST_FLOW_ERROR;
}

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
NNSTREAMER_PLUGIN_INIT (tensor_src_iio)
{
  /**
   * debug category for filtering log messages
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_src_iio_debug, "tensor_src_iio",
      0, "tensor_src_iio element");

  return gst_element_register (plugin, "tensor_src_iio", GST_RANK_NONE,
      GST_TYPE_TENSOR_SRC_IIO);
}
