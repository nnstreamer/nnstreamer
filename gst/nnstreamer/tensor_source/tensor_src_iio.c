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
 * @todo  fill in empty functions
 * @todo  create a sample example and unit tests
 * @todo  set limit on buffer capacity, frequency
 * @todo  support device/trigger number as input
 * @todo  support for trigger frequency
 * @todo  support specific channels as input
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
#include <stdio.h>

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
 * @brief tensor_src_iio properties.
 */
enum
{
  PROP_0,
  PROP_MODE,
  PROP_SILENT,
  PROP_DEVICE,
  PROP_TRIGGER,
  PROP_CHANNELS,
  PROP_BUFFER_CAPACITY,
  PROP_FREQUENCY
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
#define MAX_BUFFER_CAPACITY 100
#define DEFAULT_BUFFER_CAPACITY 1

/**
 * @brief Minimum and maximum operating frequency for the device
 * Default frequency chooses the first available frequency supported by device
 */
#define MIN_FREQUENCY 1
#define MAX_FREQUENCY 999999999
#define DEFAULT_FREQUENCY 0

/**
 * @brief IIO devices/triggers
 */
#define DEVICE "device"
#define BUFFER "buffer"
#define TRIGGER "trigger"
#define CHANNELS "scan_elements"
#define IIO "iio:"
#define DEVICE_PREFIX IIO DEVICE
#define TRIGGER_PREFIX IIO TRIGGER
#define CURRENT_TRIGGER "current_trigger"

/**
 * @brief IIO device channels
 */
#define EN_SUFFIX "_en"
#define INDEX_SUFFIX "_index"
#define TYPE_SUFFIX "_type"

/**
 * @brief filenames for IIO devices/triggers characteristics
 */
#define NAME_FILE "name"
#define AVAIL_FREQUENCY_FILE "sampling_frequency_available"

/**
 * @brief IIO system paths
 */
#define IIO_BASE_DIR "/sys/bus/iio/devices/"

/**
 * @brief Template for src pad.
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT));

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

  g_object_class_install_property (gobject_class, PROP_TRIGGER,
      g_param_spec_string ("trigger", "Trigger Name",
          "Name of the trigger to be used", DEFAULT_PROP_STRING,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_CHANNELS,
      g_param_spec_string ("channels", "Channels to be enabled",
          "Enable channels -"
          "auto: enable all channels when no channels are enabled automatically"
          "all: enable all channels",
          DEFAULT_OPERATING_CHANNELS_ENABLED, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_BUFFER_CAPACITY,
      g_param_spec_uint ("buffer_capacity", "Buffer Capacity",
          "Capacity of the data buffer", MIN_BUFFER_CAPACITY,
          MAX_BUFFER_CAPACITY, DEFAULT_BUFFER_CAPACITY,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_FREQUENCY,
      g_param_spec_uint64 ("frequency", "Frequency",
          "Operating frequency of the device", MIN_FREQUENCY, MAX_FREQUENCY,
          DEFAULT_FREQUENCY, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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
  // TODO: verify where locking is needed
  g_mutex_init (&self->mutex);

  /** init properties */
  self->configured = FALSE;
  self->channels = DEFAULT_PROP_STRING;
  self->channels_enabled = CHANNELS_ENABLED_AUTO;
  gst_tensor_src_iio_device_properties_init (&self->trigger);
  gst_tensor_src_iio_device_properties_init (&self->device);
  self->silent = DEFAULT_PROP_SILENT;
  self->buffer_capacity = DEFAULT_BUFFER_CAPACITY;
  self->sampling_frequency = DEFAULT_FREQUENCY;

  return;
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
  // @todo update cppcheck to work with G_UNLIKELY here
  if (NULL == dptr) {
    GST_ERROR ("Error in opening directory %s.\n", dir_name);
    return ret;
  }

  while ((dir_entry = readdir (dptr)) != NULL) {
    // check for prefix and the next digit should be a number
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
      &prop->mask_bits, &prop->storage_bits, &prop->shift);
  if (arguments_filled < 5) {
    ret = FALSE;
    return ret;
  }

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
 * @return negative is a<b
 *         0 is a==b
 *         positive if a>b
 */
static gboolean
gst_tensor_channel_list_sort_cmp (gconstpointer a, gconstpointer b)
{
  const GstTensorSrcIIOChannelProperties *a_ch = a;
  const GstTensorSrcIIOChannelProperties *b_ch = b;
  gint compare_result = a_ch->index - b_ch->index;
  return compare_result;
}

/**
 * @brief get info about all the channels in the device
 * @param[in/out] self Tensor src IIO object
 * @param[in] dir_name Directory name with all the scan elements for device
 * @return >=0 number of enabled channels
 *         -1  if any error when scanning channels
 * @todo: verify scale and offset can exist in continuous mode
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

  if (!g_file_test (dir_name, G_FILE_TEST_IS_DIR)) {
    GST_ERROR ("No channels available.");
    return ret;
  }
  dptr = opendir (dir_name);
  // @todo update cppcheck to work with G_UNLIKELY here
  if (NULL == dptr) {
    GST_ERROR ("Error in opening directory %s.\n", dir_name);
    return ret;
  }

  while ((dir_entry = readdir (dptr)) != NULL) {
    // check for enable
    if (g_str_has_suffix (dir_entry->d_name, EN_SUFFIX)) {
      GstTensorSrcIIOChannelProperties channel_prop;
      self->channels = g_list_prepend (self->channels, &channel_prop);

      // set the name and base_dir
      channel_prop.name = g_strndup (dir_entry->d_name,
          strlen (dir_entry->d_name) - strlen (EN_SUFFIX));
      channel_prop.base_dir = g_strdup (dir_name);
      channel_prop.base_file =
          g_build_filename (dir_name, channel_prop.name, NULL);

      // find and set the current state
      filename = g_strdup_printf ("%s%s", channel_prop.base_file, EN_SUFFIX);
      if (!g_file_get_contents (filename, &file_contents, NULL, &error)) {
        GST_ERROR ("Unable to read %s, error: %s.\n", filename, error->message);
        goto error_free_filename;
      }
      g_free (filename);

      value = g_ascii_strtoull (file_contents, NULL, 10);
      g_free (file_contents);
      if (value == 1) {
        channel_prop.enabled = TRUE;
        num_channels_enabled += 1;
      } else if (value == 0) {
        channel_prop.enabled = FALSE;
      } else {
        GST_ERROR
            ("Enable bit %u (out of range) in current state of channel %s.\n",
            value, channel_prop.name);
        goto error_cleanup_list;
      }

      // find and set the index
      filename = g_strdup_printf ("%s%s", channel_prop.base_file, INDEX_SUFFIX);
      if (!g_file_get_contents (filename, &file_contents, NULL, &error)) {
        GST_ERROR ("Unable to read %s, error: %s.\n", filename, error->message);
        goto error_free_filename;
      }
      g_free (filename);

      value = g_ascii_strtoull (file_contents, NULL, 10);
      g_free (file_contents);
      channel_prop.index = value;

      // find and set the type information
      filename = g_strdup_printf ("%s%s", channel_prop.base_file, TYPE_SUFFIX);
      if (!g_file_test (filename, G_FILE_TEST_IS_REGULAR)) {
        channel_prop.generic_name =
            gst_tensor_src_iio_get_generic_name (channel_prop.name);
        silent_debug ("Generic name = %s", channel_prop.generic_name);
        g_free (filename);
        filename =
            g_strdup_printf ("%s%s", channel_prop.generic_name, TYPE_SUFFIX);
      }
      if (!g_file_get_contents (filename, &file_contents, NULL, &error)) {
        GST_ERROR ("Unable to read %s, error: %s.\n", filename, error->message);
        goto error_free_filename;
      }
      g_free (filename);

      if (!gst_tensor_src_iio_set_channel_type (&channel_prop, file_contents)) {
        GST_ERROR ("Error while setting up channel type for channel %s.\n",
            channel_prop.name);
        g_free (file_contents);
        goto error_cleanup_list;
      }
      g_free (file_contents);
    }
  }

  // sort the list with the order of the indices
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
 *         0  if any cannot find the matching frequency
 */
static guint64
gst_tensor_src_iio_set_frequency (const gchar * base_dir,
    const guint64 frequency)
{
  GError *error = NULL;
  gchar *filename = NULL;
  gchar *file_contents = NULL;
  gint i = 0;
  guint64 ret = 0, val = 0;

  // get frequency list supported by the device
  filename = g_build_filename (base_dir, AVAIL_FREQUENCY_FILE, NULL);
  if (!g_file_get_contents (filename, &file_contents, NULL, &error)) {
    GST_ERROR ("Unable to read sampling frequency for device %s.\n", base_dir);
    g_error_free (error);
    goto del_filename;
  }

  gchar **freq_list = g_strsplit (file_contents, " ", -1);
  gint num = g_strv_length (freq_list);
  if (num == 0) {
    GST_ERROR ("No sampling frequencies for device %s.\n", base_dir);
    goto del_freq_list;
  }
  // if the frequency is set 0, set the first available frequency
  // else verify the the frequency recceived from user is supported by the device
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

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      break;

    case PROP_MODE:
      self->mode = g_value_dup_string (value);
      break;

    case PROP_DEVICE:
      self->device.name = g_value_dup_string (value);
      break;

    case PROP_TRIGGER:
      self->trigger.name = g_value_dup_string (value);
      break;

    case PROP_CHANNELS:
    {
      const gchar *param = g_value_get_string (value);
      if (g_strcmp0 (param, CHANNELS_ENABLED_ALL_CHAR)) {
        self->channels_enabled = CHANNELS_ENABLED_ALL;
      } else if (g_strcmp0 (param, CHANNELS_ENABLED_AUTO_CHAR)) {
        self->channels_enabled = CHANNELS_ENABLED_AUTO;
      }
      break;
    }

    case PROP_BUFFER_CAPACITY:
      self->buffer_capacity = g_value_get_uint (value);
      break;

    case PROP_FREQUENCY:
      self->sampling_frequency = g_value_get_uint64 (value);
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

    case PROP_TRIGGER:
      g_value_set_string (value, self->trigger.name);
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
      // interface of frequency is kept long for outside but uint64 inside
      g_value_set_ulong (value, self->sampling_frequency);
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
  //FIXME: fill this function
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
    GST_ERROR ("Unable to open file to write %s.\n", filename);
    goto error_free_filename;
  }

  bytes_printed = fprintf (fd, "%s", contents);
  if (bytes_printed != strlen (contents)) {
    GST_ERROR ("Unable to write to file %s.\n", filename);
    goto error_close_file;
  }
  if (!fclose (fd)) {
    GST_ERROR ("Unable to close file %s after write.\n", filename);
    goto error_free_filename;
  }
  ret = TRUE;

  if (DBG) {
    gchar *file_contents = NULL;
    ret = FALSE;
    if (!g_file_get_contents (filename, &file_contents, NULL, &error)) {
      GST_ERROR ("Unable to read file %s with error %s.\n", filename,
          error->message);
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
  g_error_free (error);
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
 * @brief start function, called when state changed null to ready.
 * load the device and init the device resources
 */
static gboolean
gst_tensor_src_iio_start (GstBaseSrc * src)
{
  /** load and init resources */
  GstTensorSrcIIO *self;
  self = GST_TENSOR_SRC_IIO_CAST (src);
  gint id;
  guint64 sampling_frequency;
  gchar *dirname = NULL;
  gchar *filename = NULL;

  // Find the device
  id = gst_tensor_src_iio_get_id_by_name (IIO_BASE_DIR, self->device.name,
      DEVICE_PREFIX);
  if (G_UNLIKELY (id < 0)) {
    GST_ERROR_OBJECT (self, "Cannot find the IIO device with name: %s.\n",
        self->device.name);
    goto error_return;
  }
  self->device.id = id;
  dirname = g_strdup_printf ("%s%d", DEVICE_PREFIX, self->device.id);
  self->device.base_dir = g_build_filename (IIO_BASE_DIR, dirname, NULL);
  g_free (dirname);

  // @todo: support scale/offset in one-shot mode for shared/non-shared channels
  // no more configuration for one shot mode
  if (!g_strcmp0 (self->mode, MODE_ONE_SHOT)) {
    goto safe_return;
  }
  // register the trigger
  if (self->trigger.name != NULL) {
    // verify if trigger is supported by our device
    gchar *trigger_device_dir =
        g_build_filename (self->device.base_dir, TRIGGER, NULL);
    if (!g_file_test (trigger_device_dir, G_FILE_TEST_IS_DIR)) {
      GST_ERROR_OBJECT (self, "IIO device %s does not supports trigger.\n",
          self->device.name);
      g_free (trigger_device_dir);
      goto error_device_free;
    }
    g_free (trigger_device_dir);

    // find if the provided trigger exists
    id = gst_tensor_src_iio_get_id_by_name (IIO_BASE_DIR, self->trigger.name,
        TRIGGER_PREFIX);
    if (G_UNLIKELY (id < 0)) {
      GST_ERROR_OBJECT (self, "Cannot find the IIO trigger: %s.\n",
          self->trigger.name);
      goto error_device_free;
    }
    self->trigger.id = id;
    dirname = g_strdup_printf ("%s%d", TRIGGER_PREFIX, self->trigger.id);
    self->trigger.base_dir = g_build_filename (IIO_BASE_DIR, dirname, NULL);
    g_free (dirname);

    // set the trigger
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
  // setup the frequency (only verifying the frequency now)
  // @todo verify setting up frequency
  sampling_frequency =
      gst_tensor_src_iio_set_frequency (self->device.base_dir,
      self->sampling_frequency);
  if (0 == sampling_frequency) {
    GST_ERROR_OBJECT (self, "IIO device does not support %lu frequency.\n",
        self->sampling_frequency);
    goto error_trigger_free;
  } else {
    self->sampling_frequency = sampling_frequency;
    // interface of frequency is kept long for outside but uint64 inside
    gulong sampling_frequency_long = (long) self->sampling_frequency;
    gchar *sampling_frequency_char =
        g_strdup_printf ("%lu", sampling_frequency_long);
    if (G_UNLIKELY (!gst_tensor_write_sysfs_string (self, "sampling_frequency",
                self->device.base_dir, sampling_frequency_char))) {
      GST_ERROR_OBJECT (self,
          "Cannot set the sampling frequency for device: %s.\n",
          self->device.name);
      g_free (sampling_frequency_char);
      goto error_trigger_free;
    }
    g_free (sampling_frequency_char);
  }

  // once all these are set, set the buffer related thingies
  dirname = g_build_filename (self->device.base_dir, BUFFER, NULL);
  if (G_UNLIKELY (!gst_tensor_write_sysfs_int (self, "length", dirname,
              self->buffer_capacity))) {
    GST_ERROR_OBJECT (self,
        "Cannot set the IIO device buffer capacity for device: %s.\n",
        self->device.name);
    g_free (dirname);
    goto error_trigger_free;
  }
  g_free (dirname);

  // get all the channels that exist and then set enable on them
  dirname = g_build_filename (self->device.base_dir, CHANNELS, NULL);
  guint num_channels_enabled =
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
      // if enabling all channels failed, disable all channels
      GST_ERROR_OBJECT (self, "Enabling all channels failed for device: %s,"
          "disabling all the channels.\n", self->device.name);
      gst_tensor_set_all_channels (self, 0);
      goto error_channels_free;
    }
  }


safe_return:
  self->configured = TRUE;
  // set the source as live
  gst_base_src_set_live (src, TRUE);
  // complete the start of the base src
  gst_base_src_start_complete (src, GST_FLOW_OK);
  return TRUE;

error_channels_free:
  g_list_free_full (self->channels, gst_tensor_src_iio_channel_properties_free);
  self->channels = NULL;

error_trigger_free:
  g_free (self->trigger.base_dir);
  self->trigger.base_dir = NULL;

error_device_free:
  g_free (self->device.base_dir);
  self->device.base_dir = NULL;

error_return:
  // complete the start of the base src
  gst_base_src_start_complete (src, GST_FLOW_ERROR);
  return FALSE;
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

  g_list_free_full (self->channels, gst_tensor_src_iio_channel_properties_free);
  self->channels = NULL;

  g_free (self->trigger.base_dir);
  self->trigger.base_dir = NULL;

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
  /* No events to be handled yet */
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
      /* Only support sequential data access */
      gst_query_set_scheduling (query, GST_SCHEDULING_FLAG_SEQUENTIAL, 1, -1,
          0);
      /* Only support push mode for now */
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
  //FIXME: fill this function
  return TRUE;
}

/**
 * @brief get caps of subclass
 */
static GstCaps *
gst_tensor_src_iio_get_caps (GstBaseSrc * src, GstCaps * filter)
{
  //FIXME: fill this function
  GstCaps *caps = NULL;
  return caps;
}

/**
 * @brief fixate the caps when needed during negotiation
 */
static GstCaps *
gst_tensor_src_iio_fixate (GstBaseSrc * src, GstCaps * caps)
{
  //FIXME: fill this function
  GstCaps *ret_caps = NULL;
  return ret_caps;
}

/**
 * @brief check if source supports seeking
 */
static gboolean
gst_tensor_src_iio_is_seekable (GstBaseSrc * src)
{
  /* iio sensors are live source without any support for seeking */
  return FALSE;
}

/**
 * @brief create a buffer with requested size and offset
 */
static GstFlowReturn
gst_tensor_src_iio_create (GstBaseSrc * src, guint64 offset,
    guint size, GstBuffer ** buf)
{
  //FIXME: fill this function
  return GST_FLOW_ERROR;
}

/**
 * @brief fill the buffer with data
 */
static GstFlowReturn
gst_tensor_src_iio_fill (GstBaseSrc * src, guint64 offset,
    guint size, GstBuffer * buf)
{
  //FIXME: fill this function
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
