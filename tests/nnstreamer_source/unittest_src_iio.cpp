/**
 * @file	unittest_src_iio.cpp
 * @date	22 March 2019
 * @brief	Unit test for tensor_src_iio
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs.
 */
#include <gtest/gtest.h>
#include <gst/gst.h>
#include <gst/check/gstharness.h>
#include <fcntl.h>
#include <unistd.h>

/**
 * @brief element name to be tested
 */
#define ELEMENT_NAME "tensor_src_iio"

/**
 * @brief original location of iio to be modified for testing
 */
extern gchar *IIO_BASE_DIR;
extern gchar *IIO_DEV_DIR;

/**
 * @brief to store the original location for iio
 */
gchar *PREV_IIO_BASE_DIR;
gchar *PREV_IIO_DEV_DIR;

/**
 * @brief iio default and test values
 */
#define DEFAULT_BUFFER_CAPACITY 1
#define DEFAULT_FREQUENCY 0
#define DEFAULT_SILENT TRUE
#define DEFAULT_MERGE_CHANNELS TRUE
#define DEVICE_NAME "test-device-1"
#define TRIGGER_NAME "test-trigger-1"
#define SCALE  10.1
#define OFFSET 1.1

/**
 * @brief iio states and values
 */

const gchar *mode[] = { "one-shot", "continuous" };
const gchar *channels[] = { "auto", "all" };
const gchar *samp_freq_avail[] = { "1000", "2000", "3000", NULL };

/**
 * @brief structure for iio device file/folder names
 */
typedef struct _iio_dev_dir_struct
{
  gchar *base_dir;
  gchar *sys_dir;
  gchar *bus_dir;
  gchar *iio_dir;
  gchar *iio_base_dir_sim;

  gchar *device;
  gchar *name;

  gchar *trigger_dev;
  gchar *trigger_name;

  gchar *trigger;
  gchar *cur_trig;

  gchar *samp_freq;
  gchar *samp_freq_avail;
  gchar *buffer;
  gchar *buf_en;
  gchar *buf_length;
  gchar *scale;
  gchar *offset;

  static const int num_scan_elements = 8;
  gchar *scan_el;
  gchar *scan_el_en[num_scan_elements];
  gchar *scan_el_index[num_scan_elements];
  gchar *scan_el_type[num_scan_elements];
  gchar *scan_el_raw[num_scan_elements];
  gchar *scan_el_time_en;
  gchar *scan_el_time_index;
  gchar *scan_el_time_type;

  gchar *dev_dir;
  gchar *dev_device_dir;
} iio_dev_dir_struct;


static gint safe_remove (const char *filename);
static gint safe_rmdir (const char *dirname);

/**
 * @brief make structure for iio device with all file names
 * @param[in] num number assigned to the device
 * @returns made structure (owned by the caller)
 */
static iio_dev_dir_struct *
make_iio_dev_structure (int num)
{
  gchar *device_folder_name;
  gchar *trigger_folder_name;
  gchar *scan_element_name;
  const gchar *_tmp_dir = g_get_tmp_dir ();
  const gchar *_dirname = "nnst-src-XXXXXX";

  iio_dev_dir_struct *iio_dev = g_new (iio_dev_dir_struct, 1);
  iio_dev->base_dir = g_build_filename (_tmp_dir, _dirname, NULL);
  iio_dev->base_dir = g_mkdtemp_full (iio_dev->base_dir, 0777);
  EXPECT_EQ (safe_rmdir (iio_dev->base_dir), 0);

  iio_dev->sys_dir = g_build_filename (iio_dev->base_dir, "sys", NULL);
  iio_dev->bus_dir = g_build_filename (iio_dev->sys_dir, "bus", NULL);
  iio_dev->iio_dir = g_build_filename (iio_dev->bus_dir, "iio", NULL);
  iio_dev->iio_base_dir_sim =
      g_build_filename (iio_dev->iio_dir, "devices", NULL);

  PREV_IIO_BASE_DIR = IIO_BASE_DIR;
  IIO_BASE_DIR = g_strdup (iio_dev->iio_base_dir_sim);

  device_folder_name = g_strdup_printf ("%s%d", "iio:device", num);
  iio_dev->device = g_build_filename (IIO_BASE_DIR, device_folder_name, NULL);
  iio_dev->name = g_build_filename (iio_dev->device, "name", NULL);

  trigger_folder_name = g_strdup_printf ("%s%d", "iio:trigger", num);
  iio_dev->trigger_dev =
      g_build_filename (IIO_BASE_DIR, trigger_folder_name, NULL);
  iio_dev->trigger_name = g_build_filename (iio_dev->trigger_dev, "name", NULL);

  iio_dev->trigger = g_build_filename (iio_dev->device, "trigger", NULL);
  iio_dev->cur_trig =
      g_build_filename (iio_dev->trigger, "current_trigger", NULL);

  iio_dev->samp_freq =
      g_build_filename (iio_dev->device, "sampling_frequency", NULL);
  iio_dev->samp_freq_avail =
      g_build_filename (iio_dev->device, "sampling_frequency_available", NULL);

  iio_dev->buffer = g_build_filename (iio_dev->device, "buffer", NULL);
  iio_dev->buf_en = g_build_filename (iio_dev->buffer, "enable", NULL);
  iio_dev->buf_length = g_build_filename (iio_dev->buffer, "length", NULL);

  iio_dev->scale = g_build_filename (iio_dev->device, "in_voltage_scale", NULL);
  iio_dev->offset =
      g_build_filename (iio_dev->device, "in_voltage_offset", NULL);

  iio_dev->scan_el = g_build_filename (iio_dev->device, "scan_elements", NULL);
  for (int idx = 0; idx < iio_dev->num_scan_elements; idx++) {
    scan_element_name = g_strdup_printf ("%s%d%s", "in_voltage", idx, "_en");
    iio_dev->scan_el_en[idx] =
        g_build_filename (iio_dev->scan_el, scan_element_name, NULL);
    g_free (scan_element_name);
    scan_element_name = g_strdup_printf ("%s%d%s", "in_voltage", idx, "_index");
    iio_dev->scan_el_index[idx] =
        g_build_filename (iio_dev->scan_el, scan_element_name, NULL);
    g_free (scan_element_name);
    scan_element_name = g_strdup_printf ("%s%d%s", "in_voltage", idx, "_type");
    iio_dev->scan_el_type[idx] =
        g_build_filename (iio_dev->scan_el, scan_element_name, NULL);
    g_free (scan_element_name);
    scan_element_name = g_strdup_printf ("%s%d%s", "in_voltage", idx, "_raw");
    iio_dev->scan_el_raw[idx] =
        g_build_filename (iio_dev->scan_el, scan_element_name, NULL);
    g_free (scan_element_name);
  }
  iio_dev->scan_el_time_en =
      g_build_filename (iio_dev->scan_el, "timestamp_en", NULL);
  iio_dev->scan_el_time_index =
      g_build_filename (iio_dev->scan_el, "timestamp_index", NULL);
  iio_dev->scan_el_time_type =
      g_build_filename (iio_dev->scan_el, "timestamp_type", NULL);

  iio_dev->dev_dir = g_build_filename (iio_dev->base_dir, "dev", NULL);
  iio_dev->dev_device_dir =
      g_build_filename (iio_dev->dev_dir, device_folder_name, NULL);

  PREV_IIO_DEV_DIR = IIO_DEV_DIR;
  IIO_DEV_DIR = g_strdup (iio_dev->dev_dir);

  g_free (device_folder_name);
  g_free (trigger_folder_name);

  return iio_dev;
}

/**
 * @brief write string in to the file
 * @param[in] filename Destination file for the data
 * @param[in] contents Data to be written to the file
 * @returns 0 on success, non-zero on failure
 */
static gint
write_file_string (const gchar * filename, const gchar * contents)
{
  if (!g_file_set_contents (filename, contents, -1, NULL)) {
    return -1;
  }
  return 0;
}

/**
 * @brief write int in to the file
 * @param[in] filename Destination file for the data
 * @param[in] contents Data to be written to the file
 * @returns 0 on success, non-zero on failure
 */
static gint
write_file_int (const gchar * filename, const gint contents)
{
  g_autofree gchar *contents_char = NULL;
  contents_char = g_strdup_printf ("%d", contents);

  return write_file_string (filename, contents_char);
}

/**
 * @brief write float in to the file
 * @param[in] filename Destination file for the data
 * @param[in] contents Data to be written to the file
 * @returns 0 on success, non-zero on failure
 */
static gint
write_file_float (const gchar * filename, const gfloat contents)
{
  g_autofree gchar *contents_char = NULL;
  contents_char = g_strdup_printf ("%f", contents);

  return write_file_string (filename, contents_char);
}

/**
 * @brief create a file
 * @param[in] filename name of the file to be created
 * @returns 0 on success, non-zero on failure
 */
static gint
touch_file (const gchar * filename)
{
  return write_file_string (filename, "");
}

/**
 * @brief build base dir for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_base (const iio_dev_dir_struct * iio_dev)
{
  gint status = 0;

  status += g_mkdir_with_parents (iio_dev->device, 0777);
  status += write_file_string (iio_dev->name, DEVICE_NAME);

  return status;
}

/**
 * @brief build dev data dir and fills it for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_dev_data (const iio_dev_dir_struct * iio_dev)
{
  gint status = 0;

  status += g_mkdir_with_parents (iio_dev->dev_dir, 0777);
  /** @todo fill in proper data */
  status += write_file_string (iio_dev->dev_device_dir, "FIXME");

  return status;
}

/**
 * @brief build buffer dir for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_buffer (const iio_dev_dir_struct * iio_dev)
{
  gint status = 0;

  status += g_mkdir_with_parents (iio_dev->buffer, 0777);
  status += touch_file (iio_dev->buf_en);
  status += touch_file (iio_dev->buf_length);

  return status;
}

/**
 * @brief build trigger dir for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_trigger (const iio_dev_dir_struct * iio_dev)
{
  gint status = 0;

  status += g_mkdir_with_parents (iio_dev->trigger, 0777);
  status += touch_file (iio_dev->cur_trig);

  return status;
}

/**
 * @brief build trigger device dir for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_trigger_dev (const iio_dev_dir_struct * iio_dev)
{
  gint status = 0;

  status += g_mkdir_with_parents (iio_dev->trigger_dev, 0777);
  status += write_file_string (iio_dev->trigger_name, TRIGGER_NAME);

  return status;
}

/**
 * @brief build sampling frequency for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_samp_freq (const iio_dev_dir_struct * iio_dev)
{
  gint status = 0;

  status += touch_file (iio_dev->samp_freq);

  return status;
}

/**
 * @brief build list of sampling frequency for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_samp_freq_avail (const iio_dev_dir_struct * iio_dev)
{
  gint status = 0;
  gchar *samp_freq_avail_string = g_strjoinv (",", (char **) samp_freq_avail);

  status +=
      write_file_string (iio_dev->samp_freq_avail, samp_freq_avail_string);

  g_free (samp_freq_avail_string);
  return status;
}

/**
 * @brief build scale and offset for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_scale_offset (const iio_dev_dir_struct * iio_dev)
{
  gint status = 0;

  status += write_file_float (iio_dev->scale, SCALE);
  status += write_file_float (iio_dev->offset, OFFSET);

  return status;
}

/**
 * @brief build raw data elements for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @param[in] skip skip some of the elements
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_raw_elements (const iio_dev_dir_struct * iio_dev, gboolean skip =
    FALSE)
{
  gint status = 0;

  for (int idx = 0; idx < iio_dev->num_scan_elements; idx++) {
    if (idx % 2 && skip) {
      continue;
    }
    /** todo: verify what is the dtype of this data */
    status += write_file_int (iio_dev->scan_el_raw[idx], 1000);
  }

  return status;
}

/**
 * @brief build scan data elements dir for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @param[in] num_bits num of bits for storage of data
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_scan_elements (const iio_dev_dir_struct * iio_dev,
    const guint num_bits)
{
  gint status = 0;
  gchar *type_data;
  gchar endianchar, signchar;
  gint storage, used, shift;

  if (!g_file_test (iio_dev->scan_el, G_FILE_TEST_IS_DIR)) {
    status += g_mkdir_with_parents (iio_dev->scan_el, 0777);
  }
  /** total 8 possible cases */
  for (int idx = 0; idx < iio_dev->num_scan_elements; idx++) {
    status += write_file_int (iio_dev->scan_el_en[idx], idx % 2);
    status += write_file_int (iio_dev->scan_el_index[idx], idx);
    /** form 4 possible combinations */
    endianchar = 'b';
    signchar = 's';
    switch (idx % (iio_dev->num_scan_elements / 2)) {
      case 0:
        break;
      case 1:
        endianchar = 'l';
      case 2:
        signchar = 'u';
        break;
      case 3:
        endianchar = 'l';
        break;
    }
    storage = num_bits;
    used = storage - 2 * (idx / (iio_dev->num_scan_elements / 2));
    if (used <= 0) {
      used = 1;
    }
    /** shift will be 0 or non-zero (2 in this case) */
    shift = storage - used;
    type_data =
        g_strdup_printf ("%ce:%c%u/%u>>%u", endianchar, signchar, used, storage,
        shift);
    status += write_file_string (iio_dev->scan_el_type[idx], type_data);
    g_free (type_data);
  }

  return status;
}

/**
 * @brief build timestamp in scan data elements dir for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_timestamp (const iio_dev_dir_struct * iio_dev)
{
  gint status = 0;
  gchar *type_data;
  gchar endianchar, signchar;
  gint storage, used, shift;

  if (!g_file_test (iio_dev->scan_el, G_FILE_TEST_IS_DIR)) {
    status += g_mkdir_with_parents (iio_dev->scan_el, 0777);
  }
  /** total 8 possible cases */
  status += write_file_int (iio_dev->scan_el_time_en, 1);
  status +=
      write_file_int (iio_dev->scan_el_time_index, iio_dev->num_scan_elements);
  endianchar = 'b';
  signchar = 's';
  storage = 64;
  used = 64;
  /** shift will be 0 or non-zero (2 in this case) */
  shift = storage - used;
  type_data =
      g_strdup_printf ("%ce:%c%u/%u>>%u", endianchar, signchar, used, storage,
      shift);
  status += write_file_string (iio_dev->scan_el_time_type, type_data);
  g_free (type_data);

  return status;
}

/**
 * @brief cleans memory of iio device structure
 * @param[in] iio_dev struct of iio device
 */
static void
clean_iio_dev_structure (iio_dev_dir_struct * iio_dev)
{
  g_free (IIO_BASE_DIR);
  g_free (IIO_DEV_DIR);

  IIO_BASE_DIR = PREV_IIO_BASE_DIR;
  IIO_DEV_DIR = PREV_IIO_DEV_DIR;

  g_free (iio_dev->base_dir);
  g_free (iio_dev->sys_dir);
  g_free (iio_dev->bus_dir);
  g_free (iio_dev->iio_dir);
  g_free (iio_dev->iio_base_dir_sim);
  g_free (iio_dev->device);
  g_free (iio_dev->name);
  g_free (iio_dev->trigger_dev);
  g_free (iio_dev->trigger_name);
  g_free (iio_dev->trigger);
  g_free (iio_dev->cur_trig);
  g_free (iio_dev->samp_freq);
  g_free (iio_dev->samp_freq_avail);
  g_free (iio_dev->buffer);
  g_free (iio_dev->buf_en);
  g_free (iio_dev->buf_length);
  g_free (iio_dev->scale);
  g_free (iio_dev->offset);
  g_free (iio_dev->scan_el);
  g_free (iio_dev->scan_el_time_en);
  g_free (iio_dev->scan_el_time_index);
  g_free (iio_dev->scan_el_time_type);

  for (int idx = 0; idx < iio_dev->num_scan_elements; idx++) {
    g_free (iio_dev->scan_el_en[idx]);
    g_free (iio_dev->scan_el_index[idx]);
    g_free (iio_dev->scan_el_type[idx]);
    g_free (iio_dev->scan_el_raw[idx]);
  }

  g_free (iio_dev->dev_dir);
  g_free (iio_dev->dev_device_dir);

  g_free (iio_dev);
  return;
}

/**
 * @brief removes file if exists
 * @param[in] filename Name of the file to be removed
 * @returns 0 on success, -1 on failure
 * @note returns success if the file does not exist
 */
static gint
safe_remove (const char *filename)
{
  if (g_file_test (filename, G_FILE_TEST_IS_REGULAR)) {
    return remove (filename);
  }
  return 0;
}

/**
 * @brief removes directory if exists
 * @param[in] dirname Name of the directory to be removed
 * @returns 0 on success, -1 on failure
 * @note returns success if the directory does not exist
 * @note callers responsibility to empty the directory before calling
 */
static gint
safe_rmdir (const char *dirname)
{
  if (g_file_test (dirname, G_FILE_TEST_IS_DIR)) {
    return rmdir (dirname);
  }
  return 0;
}

/**
 * @brief destroy dir for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, -1 on failure
 */
static gint
destroy_dev_dir (const iio_dev_dir_struct * iio_dev)
{
  gint status = 0;

  for (int idx = 0; idx < iio_dev->num_scan_elements; idx++) {
    status += safe_remove (iio_dev->scan_el_en[idx]);
    status += safe_remove (iio_dev->scan_el_index[idx]);
    status += safe_remove (iio_dev->scan_el_type[idx]);
    status += safe_remove (iio_dev->scan_el_raw[idx]);
  }
  status += safe_remove (iio_dev->scan_el_time_en);
  status += safe_remove (iio_dev->scan_el_time_index);
  status += safe_remove (iio_dev->scan_el_time_type);
  status += safe_rmdir (iio_dev->scan_el);

  status += safe_remove (iio_dev->buf_en);
  status += safe_remove (iio_dev->buf_length);
  status += safe_rmdir (iio_dev->buffer);

  status += safe_remove (iio_dev->cur_trig);
  status += safe_rmdir (iio_dev->trigger);

  status += safe_remove (iio_dev->scale);
  status += safe_remove (iio_dev->offset);
  status += safe_remove (iio_dev->samp_freq);
  status += safe_remove (iio_dev->samp_freq_avail);
  status += safe_remove (iio_dev->name);

  status += safe_remove (iio_dev->trigger_name);

  status += safe_rmdir (iio_dev->trigger_dev);
  status += safe_rmdir (iio_dev->device);
  status += safe_rmdir (iio_dev->iio_base_dir_sim);
  status += safe_rmdir (iio_dev->iio_dir);
  status += safe_rmdir (iio_dev->bus_dir);
  status += safe_rmdir (iio_dev->sys_dir);

  status += safe_remove (iio_dev->dev_device_dir);
  status += safe_rmdir (iio_dev->dev_dir);
  status += safe_rmdir (iio_dev->base_dir);

  return status;
}

/**
 * @brief tests properties of tensor source IIO
 */
TEST (test_tensor_src_iio, properties)
{
  const gchar default_name[] = "tensorsrciio0";

  GstHarness *hrnss = NULL;
  GstElement *src_iio = NULL;
  gchar *name;
  gboolean silent;
  guint buffer_capacity;
  gulong frequency;
  gboolean merge_channels;
  gint number;

  gboolean ret_silent;
  gchar *ret_mode;
  gchar *ret_device;
  gchar *ret_trigger;
  gchar *ret_channels;
  guint ret_buffer_capacity;
  gulong ret_frequency;
  gboolean ret_merge_channels;
  gint ret_number;

  /** setup */
  hrnss = gst_harness_new_empty ();
  ASSERT_TRUE (hrnss != NULL);
  gst_harness_add_parse (hrnss, ELEMENT_NAME);
  src_iio = gst_harness_find_element (hrnss, ELEMENT_NAME);
  ASSERT_TRUE (src_iio != NULL);

  /** check the default name */
  name = gst_element_get_name (src_iio);
  ASSERT_TRUE (name != NULL);
  EXPECT_STREQ (default_name, name);
  g_free (name);

  /** silent mode test */
  g_object_get (src_iio, "silent", &ret_silent, NULL);
  EXPECT_EQ (ret_silent, DEFAULT_SILENT);
  silent = FALSE;
  g_object_set (src_iio, "silent", silent, NULL);
  g_object_get (src_iio, "silent", &ret_silent, NULL);
  EXPECT_EQ (ret_silent, silent);

  /** operating mode test */
  g_object_get (src_iio, "mode", &ret_mode, NULL);
  EXPECT_STREQ (ret_mode, mode[1]);
  g_object_set (src_iio, "mode", mode[0], NULL);
  g_object_get (src_iio, "mode", &ret_mode, NULL);
  EXPECT_STREQ (ret_mode, mode[0]);
  g_object_set (src_iio, "mode", mode[1], NULL);
  g_object_get (src_iio, "mode", &ret_mode, NULL);
  EXPECT_STREQ (ret_mode, mode[1]);

  /** setting device test */
  g_object_set (src_iio, "device", DEVICE_NAME, NULL);
  g_object_get (src_iio, "device", &ret_device, NULL);
  EXPECT_STREQ (ret_device, DEVICE_NAME);

  /** setting device num test */
  number = 5;
  g_object_set (src_iio, "device-number", number, NULL);
  g_object_get (src_iio, "device-number", &ret_number, NULL);
  EXPECT_EQ (ret_number, number);

  /** setting trigger test */
  g_object_set (src_iio, "trigger", TRIGGER_NAME, NULL);
  g_object_get (src_iio, "trigger", &ret_trigger, NULL);
  EXPECT_STREQ (ret_trigger, TRIGGER_NAME);

  /** setting trigger num test */
  number = 5;
  g_object_set (src_iio, "trigger-number", number, NULL);
  g_object_get (src_iio, "trigger-number", &ret_number, NULL);
  EXPECT_EQ (ret_number, number);

  /** setting channels test */
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, channels[0]);
  g_object_set (src_iio, "channels", channels[1], NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, channels[1]);
  g_object_set (src_iio, "channels", channels[0], NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, channels[0]);

  /** buffer_capacity test */
  g_object_get (src_iio, "buffer-capacity", &ret_buffer_capacity, NULL);
  EXPECT_EQ (ret_buffer_capacity, DEFAULT_BUFFER_CAPACITY);
  buffer_capacity = 100;
  g_object_set (src_iio, "buffer-capacity", buffer_capacity, NULL);
  g_object_get (src_iio, "buffer-capacity", &ret_buffer_capacity, NULL);
  EXPECT_EQ (ret_buffer_capacity, buffer_capacity);

  /** frequency test */
  g_object_get (src_iio, "frequency", &ret_frequency, NULL);
  EXPECT_EQ (ret_frequency, DEFAULT_FREQUENCY);
  frequency = 100;
  g_object_set (src_iio, "frequency", frequency, NULL);
  g_object_get (src_iio, "frequency", &ret_frequency, NULL);
  EXPECT_EQ (ret_frequency, frequency);

  /** merge_channels mode test */
  g_object_get (src_iio, "merge-channels-data", &ret_merge_channels, NULL);
  EXPECT_EQ (ret_merge_channels, DEFAULT_MERGE_CHANNELS);
  merge_channels = TRUE;
  g_object_set (src_iio, "merge-channels-data", merge_channels, NULL);
  g_object_get (src_iio, "merge-channels-data", &ret_merge_channels, NULL);
  EXPECT_EQ (ret_merge_channels, merge_channels);

  /** teardown */
  gst_harness_teardown (hrnss);
}

/**
 * @brief tests state change of tensor source IIO
 */
TEST (test_tensor_src_iio, start_stop)
{
  iio_dev_dir_struct *dev0;
  GstHarness *hrnss = NULL;
  GstElement *src_iio = NULL;
  GstStateChangeReturn status;
  GstState state;

  /** build iio dummy device */
  dev0 = make_iio_dev_structure (0);
  ASSERT_EQ (build_dev_dir_base (dev0), 0);

  /** build iio dummy trigger */
  ASSERT_EQ (build_dev_dir_trigger_dev (dev0), 0);

  /** add trigger support in device */
  ASSERT_EQ (build_dev_dir_trigger (dev0), 0);

  /** dir for continuous mode */
  ASSERT_EQ (build_dev_dir_buffer (dev0), 0);
  ASSERT_EQ (build_dev_dir_scan_elements (dev0, 32), 0);
  ASSERT_EQ (build_dev_dir_timestamp (dev0), 0);
  ASSERT_EQ (build_dev_dir_dev_data (dev0), 0);

  /** dir for single-shot mode */
  ASSERT_EQ (build_dev_dir_raw_elements (dev0), 0);

  /** other attributes */
  ASSERT_EQ (build_dev_dir_samp_freq (dev0), 0);
  ASSERT_EQ (build_dev_dir_samp_freq_avail (dev0), 0);
  ASSERT_EQ (build_dev_dir_scale_offset (dev0), 0);

  /** setup */
  hrnss = gst_harness_new_empty ();
  ASSERT_TRUE (hrnss != NULL);
  gst_harness_add_parse (hrnss, ELEMENT_NAME);
  src_iio = gst_harness_find_element (hrnss, ELEMENT_NAME);
  ASSERT_TRUE (src_iio != NULL);

  /** setup properties */
  g_object_set (src_iio, "device", DEVICE_NAME, NULL);
  g_object_set (src_iio, "silent", FALSE, NULL);
  g_object_set (src_iio, "trigger", TRIGGER_NAME, NULL);

  /** silent mode test */
  status = gst_element_set_state (src_iio, GST_STATE_NULL);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  status = gst_element_get_state (src_iio, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_NULL);

  status = gst_element_set_state (src_iio, GST_STATE_READY);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  status = gst_element_get_state (src_iio, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_READY);

  status = gst_element_set_state (src_iio, GST_STATE_PAUSED);
  EXPECT_EQ (status, GST_STATE_CHANGE_NO_PREROLL);
  status = gst_element_get_state (src_iio, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_NO_PREROLL);
  EXPECT_EQ (state, GST_STATE_PAUSED);

  status = gst_element_set_state (src_iio, GST_STATE_PLAYING);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  status = gst_element_get_state (src_iio, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_PLAYING);

  /** let a frames load */
  g_usleep (50000);

  /** this will be resolved once correct data has been fed */
  status = gst_element_get_state (src_iio, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_PLAYING);

  status = gst_element_set_state (src_iio, GST_STATE_NULL);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  status = gst_element_get_state (src_iio, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_NULL);

  /** teardown */
  gst_harness_teardown (hrnss);

  /** delete device structure */
  ASSERT_EQ (destroy_dev_dir (dev0), 0);
  clean_iio_dev_structure (dev0);
}


/**
 * @brief Main function for unit test.
 */
int
main (int argc, char **argv)
{
  testing::InitGoogleTest (&argc, argv);

  gst_init (&argc, &argv);

  return RUN_ALL_TESTS ();
}
