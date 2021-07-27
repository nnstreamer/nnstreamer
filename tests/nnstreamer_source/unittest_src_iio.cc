/**
 * @file	unittest_src_iio.cc
 * @date	22 March 2019
 * @brief	Unit test for tensor_src_iio
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs.
 */
#include <gtest/gtest.h>
#include <fcntl.h>
#include <glib/gstdio.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <pthread.h>
#include <tensor_common.h>
#include <unistd.h>
#include <unittest_util.h>

/**
 * @brief element name to be tested
 */
#define ELEMENT_NAME "tensor_src_iio"

/**
 * @brief iio default and test values
 */
#define DEFAULT_BUFFER_CAPACITY (1U)
#define DEFAULT_FREQUENCY (0U)
#define DEFAULT_SILENT TRUE
#define DEFAULT_MERGE_CHANNELS TRUE
#define DEFAULT_POLL_TIMEOUT 10000
#define DEVICE_NAME "test-device-1"
#define TRIGGER_NAME "test-trigger-1"
#define BUF_LENGTH 1
#define SCALE 10.1
#define OFFSET 1.1
#define DATA 98
#define NUM_FRAMES 5

/**
 * @brief iio states and values
 */

const gchar *mode[] = { "one-shot", "continuous" };
const gchar *channels[] = { "auto", "all" };
const gchar *samp_freq_avail[] = { "1000", "2000", "3000", NULL };

#define CHANGE_ENDIANNESS(NUMBITS)                                  \
  {                                                                 \
    if (signchar == 's') {                                          \
      sdata##NUMBITS = (gint##NUMBITS *)(scan_el_data + location);  \
      if (endianchar == 'l') {                                      \
        *sdata##NUMBITS = GINT##NUMBITS##_TO_LE (sdata << shift);   \
      } else {                                                      \
        *sdata##NUMBITS = GINT##NUMBITS##_TO_BE (sdata << shift);   \
      }                                                             \
    } else {                                                        \
      udata##NUMBITS = (guint##NUMBITS *)(scan_el_data + location); \
      if (endianchar == 'l') {                                      \
        *udata##NUMBITS = GUINT##NUMBITS##_TO_LE (udata << shift);  \
      } else {                                                      \
        *udata##NUMBITS = GUINT##NUMBITS##_TO_BE (udata << shift);  \
      }                                                             \
    }                                                               \
    location += num_bytes;                                          \
    break;                                                          \
  }


/**
 * @brief structure for iio device file/folder names
 */
typedef struct _iio_dev_dir_struct {
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
  gchar *scan_el_type_generic;

  gchar *dev_dir;
  gchar *dev_device_dir;
  gchar *log_file;
  gint dev_device_dir_fd;
  gint dev_device_dir_fd_read;
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

  iio_dev_dir_struct *iio_dev = g_new0 (iio_dev_dir_struct, 1);
  g_assert (iio_dev != NULL);
  iio_dev->base_dir = g_build_filename (_tmp_dir, _dirname, NULL);
  iio_dev->base_dir = g_mkdtemp_full (iio_dev->base_dir, 0777);
  EXPECT_EQ (safe_rmdir (iio_dev->base_dir), 0);

  iio_dev->sys_dir = g_build_filename (iio_dev->base_dir, "sys", NULL);
  iio_dev->bus_dir = g_build_filename (iio_dev->sys_dir, "bus", NULL);
  iio_dev->iio_dir = g_build_filename (iio_dev->bus_dir, "iio", NULL);
  iio_dev->iio_base_dir_sim = g_build_filename (iio_dev->iio_dir, "devices", NULL);

  device_folder_name = g_strdup_printf ("%s%d", "iio:device", num);
  iio_dev->device = g_build_filename (iio_dev->iio_base_dir_sim, device_folder_name, NULL);
  iio_dev->name = g_build_filename (iio_dev->device, "name", NULL);

  trigger_folder_name = g_strdup_printf ("%s%d", "iio:trigger", num);
  iio_dev->trigger_dev
      = g_build_filename (iio_dev->iio_base_dir_sim, trigger_folder_name, NULL);
  iio_dev->trigger_name = g_build_filename (iio_dev->trigger_dev, "name", NULL);

  iio_dev->trigger = g_build_filename (iio_dev->device, "trigger", NULL);
  iio_dev->cur_trig = g_build_filename (iio_dev->trigger, "current_trigger", NULL);

  iio_dev->samp_freq = g_build_filename (iio_dev->device, "sampling_frequency", NULL);
  iio_dev->samp_freq_avail
      = g_build_filename (iio_dev->device, "sampling_frequency_available", NULL);

  iio_dev->buffer = g_build_filename (iio_dev->device, "buffer", NULL);
  iio_dev->buf_en = g_build_filename (iio_dev->buffer, "enable", NULL);
  iio_dev->buf_length = g_build_filename (iio_dev->buffer, "length", NULL);

  iio_dev->scale = g_build_filename (iio_dev->device, "in_voltage_scale", NULL);
  iio_dev->offset = g_build_filename (iio_dev->device, "in_voltage_offset", NULL);

  iio_dev->scan_el = g_build_filename (iio_dev->device, "scan_elements", NULL);
  for (int idx = 0; idx < iio_dev->num_scan_elements; idx++) {
    scan_element_name = g_strdup_printf ("%s%d%s", "in_voltage", idx, "_en");
    iio_dev->scan_el_en[idx]
        = g_build_filename (iio_dev->scan_el, scan_element_name, NULL);
    g_free (scan_element_name);
    scan_element_name = g_strdup_printf ("%s%d%s", "in_voltage", idx, "_index");
    iio_dev->scan_el_index[idx]
        = g_build_filename (iio_dev->scan_el, scan_element_name, NULL);
    g_free (scan_element_name);
    scan_element_name = g_strdup_printf ("%s%d%s", "in_voltage", idx, "_type");
    iio_dev->scan_el_type[idx]
        = g_build_filename (iio_dev->scan_el, scan_element_name, NULL);
    g_free (scan_element_name);
    scan_element_name = g_strdup_printf ("%s%d%s", "in_voltage", idx, "_raw");
    iio_dev->scan_el_raw[idx]
        = g_build_filename (iio_dev->device, scan_element_name, NULL);
    g_free (scan_element_name);
  }
  iio_dev->scan_el_time_en = g_build_filename (iio_dev->scan_el, "timestamp_en", NULL);
  iio_dev->scan_el_time_index
      = g_build_filename (iio_dev->scan_el, "timestamp_index", NULL);
  iio_dev->scan_el_time_type = g_build_filename (iio_dev->scan_el, "timestamp_type", NULL);
  iio_dev->scan_el_type_generic
      = g_build_filename (iio_dev->scan_el, "in_voltage_type", NULL);

  iio_dev->dev_dir = g_build_filename (iio_dev->base_dir, "dev", NULL);
  iio_dev->dev_device_dir = g_build_filename (iio_dev->dev_dir, device_folder_name, NULL);

  iio_dev->log_file = NULL;
  iio_dev->dev_device_dir_fd = -1;
  iio_dev->dev_device_dir_fd_read = -1;

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
write_file_string (const gchar *filename, const gchar *contents)
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
write_file_int (const gchar *filename, const gint contents)
{
  g_autofree gchar *contents_char = NULL;
  contents_char = g_strdup_printf ("%d", contents);

  return write_file_string (filename, contents_char);
}

/**
 * @brief append string in to the file
 * @param[in] filename Destination file for the data
 * @param[in] contents Data to be written to the file
 * @param[in] size Size of the contents
 * @returns 0 on success, non-zero on failure
 */
static gint
write_file_string_single_trigger (const gint fd, const gchar *contents, const gint size)
{
  gint write_size = 0;

  write_size = write (fd, contents, size);
  if (write_size != size) {
    return -1;
  }

  return 0;
}

/**
 * @brief write float in to the file
 * @param[in] filename Destination file for the data
 * @param[in] contents Data to be written to the file
 * @returns 0 on success, non-zero on failure
 */
static gint
write_file_float (const gchar *filename, const gfloat contents)
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
touch_file (const gchar *filename)
{
  return write_file_string (filename, "");
}

/**
 * @brief build base dir for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_base (const iio_dev_dir_struct *iio_dev)
{
  gint status = 0;

  status += g_mkdir_with_parents (iio_dev->device, 0777);
  status += write_file_string (iio_dev->name, DEVICE_NAME);

  return status;
}

/**
 * @brief open a read on fifo to allow write to open in parallel
 * @param[in] vargp struct of iio device
 * @returns NULL
 */
static void *
open_fifo_read (void *vargp)
{
  iio_dev_dir_struct *iio_dev = (iio_dev_dir_struct *)vargp;
  iio_dev->dev_device_dir_fd_read = open (iio_dev->dev_device_dir, O_RDONLY);
  return NULL;
}

/**
 * @brief build dev data dir and fills it for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_dev_data (iio_dev_dir_struct *iio_dev)
{
  gint status = 0;

  status += g_mkdir_with_parents (iio_dev->dev_dir, 0777);
  status += mkfifo (iio_dev->dev_device_dir, 0777);
  pthread_t thread_read;
  /**
   * create a new thread to open pipe in read mode, main thread will open
   * pipe in write mode. Without an extra thread, attempt to open pipe in write
   * will block indefinitely till the pipe is opened in read mode as well
   */
  gint val = pthread_create (&thread_read, NULL, open_fifo_read, (void *)iio_dev);
  if (val != 0) {
    status = -1;
  } else {
    iio_dev->dev_device_dir_fd = open (iio_dev->dev_device_dir, O_WRONLY);
    if (iio_dev->dev_device_dir_fd < 0) {
      status = -1;
    }
  }

  return status;
}

/**
 * @brief build buffer dir for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_buffer (const iio_dev_dir_struct *iio_dev)
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
build_dev_dir_trigger (const iio_dev_dir_struct *iio_dev)
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
build_dev_dir_trigger_dev (const iio_dev_dir_struct *iio_dev)
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
build_dev_dir_samp_freq (const iio_dev_dir_struct *iio_dev)
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
build_dev_dir_samp_freq_avail (const iio_dev_dir_struct *iio_dev)
{
  gint status = 0;
  gchar *samp_freq_avail_string = g_strjoinv (" ", (char **)samp_freq_avail);

  status += write_file_string (iio_dev->samp_freq_avail, samp_freq_avail_string);

  g_free (samp_freq_avail_string);
  return status;
}

/**
 * @brief build scale and offset for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_scale_offset (const iio_dev_dir_struct *iio_dev)
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
build_dev_dir_raw_elements (
    const iio_dev_dir_struct *iio_dev, const gint data, gboolean skip = FALSE)
{
  gint status = 0;

  for (int idx = 0; idx < iio_dev->num_scan_elements; idx++) {
    if (idx % 2 && skip) {
      continue;
    }
    status += write_file_int (iio_dev->scan_el_raw[idx], data);
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
build_dev_dir_scan_elements (iio_dev_dir_struct *iio_dev, const guint num_bits,
    const guint64 udata, const gint64 sdata, const gint skip = 1)
{
  gint status = 0;
  gchar *type_data;
  gchar endianchar, signchar;
  gint storage, used, shift;
  guint num_bytes;
  guint8 *udata8;
  gint8 *sdata8;
  guint16 *udata16;
  gint16 *sdata16;
  guint32 *udata32;
  gint32 *sdata32;
  guint64 *udata64;
  gint64 *sdata64;
  gint data_size;
  gchar *scan_el_data;
  gint location = 0;
  gint enabled = 0;

  if (!g_file_test (iio_dev->scan_el, G_FILE_TEST_IS_DIR)) {
    status += g_mkdir_with_parents (iio_dev->scan_el, 0777);
  }

  /** ensures num_bytes should round up to 1,2,4,8 */
  num_bytes = ((num_bits - 1) >> 3) + 1;
  num_bytes--;
  num_bytes |= num_bytes >> 1;
  num_bytes |= num_bytes >> 2;
  num_bytes |= num_bytes >> 4;
  num_bytes++;

  data_size = num_bytes * iio_dev->num_scan_elements / skip;
  scan_el_data = (char *)g_malloc0 (data_size);
  if (scan_el_data == NULL) {
    return -1;
  }
  /** total 8 possible cases */
  for (int idx = 0; idx < iio_dev->num_scan_elements; idx++) {
    enabled = (idx % skip == 0);
    status += write_file_int (iio_dev->scan_el_en[idx], enabled);
    status += write_file_int (iio_dev->scan_el_index[idx], idx);
    /** form 4 possible combinations */
    endianchar = 'b';
    signchar = 's';
    switch (idx % (iio_dev->num_scan_elements / 2)) {
    case 0:
      /** big endian and signed */
      break;
    case 1:
      /** little endian and unsigned (missing break is intended) */
      endianchar = 'l';
    /* fallthrough */
    case 2:
      /** big endian and unsigned */
      signchar = 'u';
      break;
    case 3:
      /** little endian and signed */
      endianchar = 'l';
      break;
    }
    storage = num_bytes * 8;
    used = num_bits - 2 * (idx / (iio_dev->num_scan_elements / 2));
    if (used <= 0) {
      used = 1;
    }

    /** shift will be 0 or non-zero (2 in this case) */
    shift = storage - used;
    type_data = g_strdup_printf (
        "%ce:%c%u/%u>>%u", endianchar, signchar, used, storage, shift);
    status += write_file_string (iio_dev->scan_el_type[idx], type_data);
    g_free (type_data);

    if (enabled) {
      switch (num_bytes) {
      case 1: {
        /** endian-ness does not matter for 1 byte */
        if (signchar == 's') {
          sdata8 = (gint8 *)(scan_el_data + location);
          *sdata8 = sdata << shift;
        } else {
          udata8 = (guint8 *)(scan_el_data + location);
          *udata8 = udata << shift;
        }
        location += num_bytes;
        break;
      }
      case 2:
        CHANGE_ENDIANNESS (16);
      case 4:
        CHANGE_ENDIANNESS (32);
      case 8:
        CHANGE_ENDIANNESS (64);
      default: {
        g_free (scan_el_data);
        return -1;
      }
      };
    }
  }

  gchar *copied_scan_el_data = (gchar *)g_malloc (data_size * BUF_LENGTH);
  if (copied_scan_el_data == NULL) {
    g_free (scan_el_data);
    return -1;
  }

  for (int idx = 0; idx < BUF_LENGTH; idx++) {
    memcpy (copied_scan_el_data + data_size * idx, scan_el_data, data_size);
  }
  EXPECT_TRUE (0 == memcmp (copied_scan_el_data, scan_el_data, data_size * BUF_LENGTH));
  status += write_file_string_single_trigger (
      iio_dev->dev_device_dir_fd, copied_scan_el_data, data_size * BUF_LENGTH);
  g_free (copied_scan_el_data);
  g_free (scan_el_data);

  return status;
}

/**
 * @brief build timestamp in scan data elements dir for iio device simulation
 * @param[in] iio_dev struct of iio device
 * @returns 0 on success, non-zero on failure
 */
static gint
build_dev_dir_timestamp (const iio_dev_dir_struct *iio_dev)
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
  status += write_file_int (iio_dev->scan_el_time_index, iio_dev->num_scan_elements);
  endianchar = 'b';
  signchar = 's';
  storage = 64;
  used = 64;
  /** shift will be 0 or non-zero (2 in this case) */
  shift = storage - used;
  type_data = g_strdup_printf ("%ce:%c%u/%u>>%u", endianchar, signchar, used, storage, shift);
  status += write_file_string (iio_dev->scan_el_time_type, type_data);
  g_free (type_data);

  return status;
}

/**
 * @brief cleans memory of iio device structure
 * @param[in] iio_dev struct of iio device
 */
static void
clean_iio_dev_structure (iio_dev_dir_struct *iio_dev)
{
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
  g_free (iio_dev->scan_el_type_generic);

  for (int idx = 0; idx < iio_dev->num_scan_elements; idx++) {
    g_free (iio_dev->scan_el_en[idx]);
    g_free (iio_dev->scan_el_index[idx]);
    g_free (iio_dev->scan_el_type[idx]);
    g_free (iio_dev->scan_el_raw[idx]);
  }

  g_free (iio_dev->dev_dir);
  g_free (iio_dev->dev_device_dir);
  g_free (iio_dev->log_file);

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
  /** cover for both regular file as well as pipes */
  if (filename && g_file_test (filename, G_FILE_TEST_EXISTS)
      && !g_file_test (filename, G_FILE_TEST_IS_DIR)) {
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
  if (dirname && g_file_test (dirname, G_FILE_TEST_IS_DIR)) {
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
destroy_dev_dir (const iio_dev_dir_struct *iio_dev)
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
  status += safe_remove (iio_dev->scan_el_type_generic);
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

  if (iio_dev->dev_device_dir_fd >= 0)
    close (iio_dev->dev_device_dir_fd);
  if (iio_dev->dev_device_dir_fd_read >= 0)
    close (iio_dev->dev_device_dir_fd_read);
  status += safe_remove (iio_dev->dev_device_dir);
  status += safe_rmdir (iio_dev->dev_dir);
  if (iio_dev->log_file != NULL) {
    status += safe_remove (iio_dev->log_file);
  }
  status += safe_rmdir (iio_dev->base_dir);

  return status;
}

/**
 * @brief tests properties of tensor source IIO
 */
TEST (testTensorSrcIio, properties)
{
  const gchar default_name[] = "tensorsrciio0";

  GstHarness *hrnss = NULL;
  GstElement *src_iio = NULL;
  gchar *name;
  gboolean silent;
  guint buffer_capacity;
  gulong frequency;
  gboolean merge_channels;
  gint poll_timeout;
  gint number;

  gboolean ret_silent;
  gchar *ret_mode;
  gchar *ret_device;
  gchar *ret_trigger;
  gchar *ret_channels;
  guint ret_buffer_capacity;
  gulong ret_frequency;
  gboolean ret_merge_channels;
  gint ret_poll_timeout;
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
  g_free (ret_mode);
  g_object_set (src_iio, "mode", mode[0], NULL);
  g_object_get (src_iio, "mode", &ret_mode, NULL);
  EXPECT_STREQ (ret_mode, mode[0]);
  g_free (ret_mode);
  g_object_set (src_iio, "mode", mode[1], NULL);
  g_object_get (src_iio, "mode", &ret_mode, NULL);
  EXPECT_STREQ (ret_mode, mode[1]);
  g_free (ret_mode);

  /** setting device test */
  g_object_set (src_iio, "device", DEVICE_NAME, NULL);
  g_object_get (src_iio, "device", &ret_device, NULL);
  EXPECT_STREQ (ret_device, DEVICE_NAME);
  g_free (ret_device);

  /** setting device num test */
  number = 5;
  g_object_set (src_iio, "device-number", number, NULL);
  g_object_get (src_iio, "device-number", &ret_number, NULL);
  EXPECT_EQ (ret_number, number);

  /** setting trigger test */
  g_object_set (src_iio, "trigger", TRIGGER_NAME, NULL);
  g_object_get (src_iio, "trigger", &ret_trigger, NULL);
  EXPECT_STREQ (ret_trigger, TRIGGER_NAME);
  g_free (ret_trigger);

  /** setting trigger num test */
  number = 5;
  g_object_set (src_iio, "trigger-number", number, NULL);
  g_object_get (src_iio, "trigger-number", &ret_number, NULL);
  EXPECT_EQ (ret_number, number);

  /** setting channels test */
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, channels[0]);
  g_free (ret_channels);
  g_object_set (src_iio, "channels", channels[1], NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, channels[1]);
  g_free (ret_channels);
  g_object_set (src_iio, "channels", channels[0], NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, channels[0]);
  g_free (ret_channels);

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

  /** poll timeout test */
  g_object_get (src_iio, "poll-timeout", &ret_poll_timeout, NULL);
  EXPECT_EQ (ret_poll_timeout, DEFAULT_POLL_TIMEOUT);
  poll_timeout = 100;
  g_object_set (src_iio, "poll-timeout", poll_timeout, NULL);
  g_object_get (src_iio, "poll-timeout", &ret_poll_timeout, NULL);
  EXPECT_EQ (ret_poll_timeout, poll_timeout);

  /** teardown */
  gst_object_unref (src_iio);
  gst_harness_teardown (hrnss);
}

/**
 * @brief makes full device structure
 * @param[in] data_value value of the data for making
 * @param[in] data_bits number of bits for the data
 * @param[in] trigger if the device should support trigger
 * @returns allocated device structure on success, NULL on error
 */
static iio_dev_dir_struct *
make_full_device (const guint64 data_value, const gint data_bits,
    const gboolean trigger = TRUE, const gint skip = 1)
{
  iio_dev_dir_struct *dev0;
  gint status = 0;

  /** build iio dummy device */
  dev0 = make_iio_dev_structure (0);
  status += build_dev_dir_base (dev0);

  /** build iio dummy trigger */
  status += build_dev_dir_trigger_dev (dev0);

  /** add trigger support in device */
  if (trigger) {
    status += build_dev_dir_trigger (dev0);
  }

  /** dir for continuous mode */
  status += build_dev_dir_buffer (dev0);
  status += build_dev_dir_dev_data (dev0);
  status += build_dev_dir_scan_elements (dev0, data_bits, data_value, data_value, skip);
  status += build_dev_dir_timestamp (dev0);

  /** dir for single-shot mode */
  status += build_dev_dir_raw_elements (dev0, data_value);

  /** other attributes */
  status += build_dev_dir_samp_freq (dev0);
  status += build_dev_dir_samp_freq_avail (dev0);
  status += build_dev_dir_scale_offset (dev0);

  if (status != 0) {
    /** delete device structure */
    destroy_dev_dir (dev0);
    clean_iio_dev_structure (dev0);
    return NULL;
  }

  return dev0;
}

/**
 * @brief tests state change of tensor source IIO
 */
TEST (testTensorSrcIio, startStop)
{
  iio_dev_dir_struct *dev0;
  GstHarness *hrnss = NULL;
  GstElement *src_iio = NULL;
  GstStateChangeReturn status;
  GstState state;

  /** Make device */
  dev0 = make_full_device (DATA, 16);
  ASSERT_NE (dev0, nullptr);

  /** setup */
  hrnss = gst_harness_new_empty ();
  ASSERT_TRUE (hrnss != NULL);
  gst_harness_add_parse (hrnss, ELEMENT_NAME);
  src_iio = gst_harness_find_element (hrnss, ELEMENT_NAME);
  ASSERT_TRUE (src_iio != NULL);

  /** setup properties */
  g_object_set (src_iio, "iio-base-dir", dev0->iio_base_dir_sim, NULL);
  g_object_set (src_iio, "dev-dir", dev0->dev_dir, NULL);
  g_object_set (src_iio, "device", DEVICE_NAME, NULL);
  g_object_set (src_iio, "silent", FALSE, NULL);

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

  EXPECT_EQ (setPipelineStateSync (src_iio, GST_STATE_PLAYING, DEFAULT_POLL_TIMEOUT), 0);
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
  gst_object_unref (src_iio);
  gst_harness_teardown (hrnss);

  /** delete device structure */
  ASSERT_EQ (destroy_dev_dir (dev0), 0);
  clean_iio_dev_structure (dev0);
}

/**
 * @brief generate tests for tensor source IIO data with trigger
 */
#define GENERATE_TESTS_TO_VERIFY_DATA_WO_TRIGGER(DATA_VALUE, DATA_BITS, SKIP)                    \
  /**                                                                                            \
   * @brief tests tensor source IIO data without trigger                                         \
   */                                                                                            \
  TEST (testTensorSrcIio, dataVerifyNoTriggerBits##DATA_BITS##alternate##SKIP)                   \
  {                                                                                              \
    static const int MAX_NUM_TRY = 100;                                                          \
    int num_try;                                                                                 \
    iio_dev_dir_struct *dev0;                                                                    \
    GstElement *src_iio_pipeline;                                                                \
    GstStateChangeReturn status;                                                                 \
    GstState state;                                                                              \
    gchar *parse_launch;                                                                         \
    gint data_value;                                                                             \
    gint samp_freq;                                                                              \
    guint data_bits;                                                                             \
    gint fd, ret;                                                                                \
    size_t bytes_read, bytes_to_read;                                                            \
    gchar *data_buffer;                                                                          \
    gfloat expect_val, actual_val;                                                               \
    guint64 expect_val_mask;                                                                     \
    gchar *expect_val_char, *actual_val_char;                                                    \
    struct stat stat_buf;                                                                        \
    data_value = DATA_VALUE;                                                                     \
    data_bits = DATA_BITS;                                                                       \
    /** Make device */                                                                           \
    dev0 = make_full_device (data_value, data_bits, FALSE, SKIP);                                \
    ASSERT_NE (dev0, nullptr);                                                                   \
    /** setup */                                                                                 \
    samp_freq = (gint)g_ascii_strtoll (samp_freq_avail[0], NULL, 10);                            \
    dev0->log_file = g_build_filename (dev0->base_dir, "temp.log", NULL);                        \
    parse_launch = g_strdup_printf (                                                             \
        "%s iio-base-dir=%s dev-dir=%s device=%s silent=FALSE ! multifilesink location=%s",      \
        ELEMENT_NAME, dev0->iio_base_dir_sim, dev0->dev_dir, DEVICE_NAME, dev0->log_file);       \
    src_iio_pipeline = gst_parse_launch (parse_launch, NULL);                                    \
    g_free (parse_launch);                                                                       \
    /** state transition test upwards */                                                         \
    EXPECT_EQ (setPipelineStateSync (src_iio_pipeline, GST_STATE_PLAYING, DEFAULT_POLL_TIMEOUT), \
        0);                                                                                      \
    status = gst_element_get_state (src_iio_pipeline, &state, NULL, GST_CLOCK_TIME_NONE);        \
    EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);                                                \
    EXPECT_EQ (state, GST_STATE_PLAYING);                                                        \
    /**                                                                                          \
     * while loop is to make sure the element if fed with data with device                       \
     * freq                                                                                      \
     * till multifilesink makes the needed file                                                  \
     */                                                                                          \
    num_try = 0;                                                                                 \
    while ((fd = open (dev0->log_file, O_RDONLY)) < 0) {                                         \
      if (num_try >= MAX_NUM_TRY) {                                                              \
        FAIL () << "Failed to open " << dev0->log_file;                                          \
      }                                                                                          \
      g_usleep (MAX (10, 1000000 / samp_freq));                                                  \
      num_try++;                                                                                 \
    }                                                                                            \
    for (num_try = 0; num_try < MAX_NUM_TRY; ++num_try) {                                        \
      g_usleep (MAX (10, 1000000 / samp_freq));                                                  \
      if (build_dev_dir_scan_elements (dev0, data_bits, data_value, data_value, SKIP) != 0) {    \
        close (fd);                                                                              \
        FAIL () << "Failed to build and fill scan elements directory";                           \
      }                                                                                          \
      ret = stat (dev0->log_file, &stat_buf);                                                    \
      if (ret == 0 && stat_buf.st_size != 0) {                                                   \
        /** verify playing state has been maintained */                                          \
        status = gst_element_get_state (src_iio_pipeline, &state, NULL, GST_CLOCK_TIME_NONE);    \
        EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);                                            \
        EXPECT_EQ (state, GST_STATE_PLAYING);                                                    \
        /** state transition test downwards */                                                   \
        status = gst_element_set_state (src_iio_pipeline, GST_STATE_NULL);                       \
        EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);                                            \
        status = gst_element_get_state (src_iio_pipeline, &state, NULL, GST_CLOCK_TIME_NONE);    \
        EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);                                            \
        EXPECT_EQ (state, GST_STATE_NULL);                                                       \
        break;                                                                                   \
      }                                                                                          \
    }                                                                                            \
    /** verify correctness of data */                                                            \
    bytes_to_read = sizeof (float) * BUF_LENGTH * dev0->num_scan_elements / SKIP;                \
    data_buffer = (gchar *)malloc (bytes_to_read);                                               \
    if (data_buffer == NULL) {                                                                   \
      close (fd);                                                                                \
      FAIL () << "Failed to malloc for data_buffer";                                             \
    }                                                                                            \
    ret = read (fd, data_buffer, bytes_to_read);                                                 \
    if (ret < 0) {                                                                               \
      close (fd);                                                                                \
      free (data_buffer);                                                                        \
      FAIL () << "Failed to read file descriptor, " << fd;                                       \
    }                                                                                            \
    bytes_read = static_cast<size_t> (ret);                                                      \
    EXPECT_EQ (bytes_read, bytes_to_read);                                                       \
    expect_val_mask = G_MAXUINT64 >> (64 - data_bits);                                           \
    expect_val = ((data_value & expect_val_mask) + OFFSET) * SCALE;                              \
    expect_val_char = g_strdup_printf ("%.2f", expect_val);                                      \
    for (size_t idx = 0; idx < bytes_to_read; idx += sizeof (float)) {                           \
      actual_val = *((gfloat *)data_buffer);                                                     \
      actual_val_char = g_strdup_printf ("%.2f", actual_val);                                    \
      EXPECT_STREQ (expect_val_char, actual_val_char);                                           \
      g_free (actual_val_char);                                                                  \
    }                                                                                            \
    g_free (expect_val_char);                                                                    \
    close (fd);                                                                                  \
    free (data_buffer);                                                                          \
                                                                                                 \
    /** delete device structure */                                                               \
    ASSERT_EQ (safe_remove (dev0->log_file), 0);                                                 \
    gst_object_unref (src_iio_pipeline);                                                         \
    ASSERT_EQ (destroy_dev_dir (dev0), 0);                                                       \
    clean_iio_dev_structure (dev0);                                                              \
  }

/**
 * @brief generate various tests to test data correctness without trigger
 * @note verifies data correctness for various number of data size (byte aligned
 * and un-aligned as well)
 * @note verifies data correctness when some of half the channels are disabled
 */
GENERATE_TESTS_TO_VERIFY_DATA_WO_TRIGGER (DATA, 4, 1);
GENERATE_TESTS_TO_VERIFY_DATA_WO_TRIGGER (DATA, 8, 2);
GENERATE_TESTS_TO_VERIFY_DATA_WO_TRIGGER (DATA, 12, 1);
GENERATE_TESTS_TO_VERIFY_DATA_WO_TRIGGER (DATA, 16, 2);
GENERATE_TESTS_TO_VERIFY_DATA_WO_TRIGGER (DATA, 24, 1);
GENERATE_TESTS_TO_VERIFY_DATA_WO_TRIGGER (DATA, 32, 2);
GENERATE_TESTS_TO_VERIFY_DATA_WO_TRIGGER (DATA, 40, 1);
GENERATE_TESTS_TO_VERIFY_DATA_WO_TRIGGER (DATA, 48, 2);
GENERATE_TESTS_TO_VERIFY_DATA_WO_TRIGGER (DATA, 56, 1);
GENERATE_TESTS_TO_VERIFY_DATA_WO_TRIGGER (DATA, 64, 2);

/**
 * @brief Run pipeline for a few frames with data verification
 */
static gboolean
test_tensor_src_iio_data_verify_util (
    iio_dev_dir_struct *dev0, gint data_value, guint data_bits, gint samp_freq)
{
  static const gint MAX_NUM_TRY = 100;
  gint num_try;
  gint fd;
  size_t bytes_to_read, bytes_read;
  gchar *data_buffer;
  guint64 expect_val_mask;
  gfloat expect_val, actual_val;
  gchar *expect_val_char, *actual_val_char;
  gint ret;
  struct stat stat_buf;

  for (int idx = 0; idx < NUM_FRAMES; idx++) {
    ret = 0;
    memset (&stat_buf, 0, sizeof (stat_buf));
    num_try = 0;

    /** wait for filter to process the frame and multifilesink to write it */
    while ((fd = open (dev0->log_file, O_RDONLY)) < 0) {
      if (num_try >= MAX_NUM_TRY) {
        return FALSE;
      }
      g_usleep (MAX (10, 1000000 / samp_freq));
      num_try++;
    }

    /** wait for filter to process the frame and multifilesink to write it */
    for (num_try = 0; num_try < MAX_NUM_TRY; ++num_try) {
      ret = stat (dev0->log_file, &stat_buf);
      if (ret == 0 && stat_buf.st_size != 0) {
        break;
      }
      g_usleep (MAX (1, 1000000 / samp_freq));
    }

    /** verify correctness of data */
    bytes_to_read = sizeof (float) * BUF_LENGTH * dev0->num_scan_elements;
    data_buffer = (gchar *)malloc (bytes_to_read);
    if (data_buffer == NULL) {
      close (fd);
      return FALSE;
    }
    ret = read (fd, data_buffer, bytes_to_read);
    if (ret < 0) {
      close (fd);
      free (data_buffer);
      return FALSE;
    }
    bytes_read = static_cast<size_t> (ret);
    EXPECT_EQ (bytes_read, bytes_to_read);
    expect_val_mask = G_MAXUINT64 >> (64 - data_bits);
    expect_val = ((data_value & expect_val_mask) + OFFSET) * SCALE;
    expect_val_char = g_strdup_printf ("%.2f", expect_val);
    for (size_t idx = 0; idx < bytes_to_read; idx += sizeof (float)) {
      actual_val = *((gfloat *)data_buffer);
      actual_val_char = g_strdup_printf ("%.2f", actual_val);
      EXPECT_STREQ (expect_val_char, actual_val_char);
      g_free (actual_val_char);
    }
    g_free (expect_val_char);
    close (fd);
    free (data_buffer);
    if (safe_remove (dev0->log_file) != 0)
      return FALSE;
    /** update data value to check data updates */
    data_value += 1;
    ret = build_dev_dir_scan_elements (dev0, data_bits, data_value, data_value);
    if (ret != 0)
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief tests tensor source IIO data with trigger
 * @note verifies that no frames have been lost as well
 * @note verifies caps for the src pad as well
 */
TEST (testTensorSrcIio, dataVerifyTrigger)
{
  iio_dev_dir_struct *dev0;
  GstElement *src_iio_pipeline;
  GstElement *src_iio;
  GstStateChangeReturn status;
  GstState state;
  gchar *parse_launch;
  gint samp_freq;
  gint data_value;
  guint data_bits;
  data_value = DATA;
  data_bits = 16;
  GstCaps *caps;
  GstPad *src_pad;
  GstStructure *structure;
  GstTensorsConfig config;
  gint num_scan_elements;
  /** Make device */
  dev0 = make_full_device (data_value, data_bits);
  ASSERT_NE (dev0, nullptr);
  /** setup */
  num_scan_elements = dev0->num_scan_elements;
  samp_freq = (gint)g_ascii_strtoll (samp_freq_avail[0], NULL, 10);
  dev0->log_file = g_build_filename (dev0->base_dir, "temp.log", NULL);
  parse_launch = g_strdup_printf ("%s iio-base-dir=%s dev-dir=%s device-number=%d trigger=%s silent=FALSE "
                                  "name=my-src-iio ! multifilesink location=%s",
      ELEMENT_NAME, dev0->iio_base_dir_sim, dev0->dev_dir, 0, TRIGGER_NAME, dev0->log_file);
  src_iio_pipeline = gst_parse_launch (parse_launch, NULL);
  g_free (parse_launch);
  /** state transition test upwards */
  EXPECT_EQ (setPipelineStateSync (src_iio_pipeline, GST_STATE_PLAYING, DEFAULT_POLL_TIMEOUT), 0);
  status = gst_element_get_state (src_iio_pipeline, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_PLAYING);

  /** get and verify the caps */
  src_iio = gst_bin_get_by_name (GST_BIN (src_iio_pipeline), "my-src-iio");
  ASSERT_NE (src_iio, nullptr);
  src_pad = gst_element_get_static_pad (src_iio, "src");
  ASSERT_NE (src_pad, nullptr);
  caps = gst_pad_get_current_caps (src_pad);
  ASSERT_NE (caps, nullptr);
  structure = gst_caps_get_structure (caps, 0);
  ASSERT_NE (structure, nullptr);

  /** Default has merge channels enabled */
  EXPECT_STREQ (gst_structure_get_name (structure), "other/tensor");
  EXPECT_EQ (gst_tensors_config_from_structure (&config, structure), TRUE);
  EXPECT_EQ (config.rate_n, samp_freq);
  EXPECT_EQ (config.rate_d, 1);
  EXPECT_EQ (config.info.num_tensors, 1U);
  EXPECT_EQ (config.info.info[0].type, _NNS_FLOAT32);
  EXPECT_EQ (config.info.info[0].dimension[0], (guint)num_scan_elements);
  EXPECT_EQ (config.info.info[0].dimension[1], 1U);
  EXPECT_EQ (config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (config.info.info[0].dimension[3], 1U);

  gst_object_unref (src_iio);
  gst_object_unref (src_pad);
  gst_caps_unref (caps);

  EXPECT_EQ (test_tensor_src_iio_data_verify_util (dev0, data_value, data_bits, samp_freq), TRUE);

  /** verify playing state has been maintained */
  status = gst_element_get_state (src_iio_pipeline, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_PLAYING);
  /** state transition test downwards */
  status = gst_element_set_state (src_iio_pipeline, GST_STATE_NULL);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  status = gst_element_get_state (src_iio_pipeline, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_NULL);

  /** delete device structure */
  gst_object_unref (src_iio_pipeline);
  ASSERT_EQ (destroy_dev_dir (dev0), 0);
  clean_iio_dev_structure (dev0);
}

/**
 * @brief tests tensor source IIO caps with custom channels
 * @note data verification with/without all channels is verified in another test
 */
TEST (testTensorSrcIio, dataVerifyCustomChannels)
{
  iio_dev_dir_struct *dev0;
  GstElement *src_iio_pipeline;
  GstElement *src_iio;
  GstStateChangeReturn status;
  GstState state;
  gchar *parse_launch;
  gint samp_freq;
  gint data_value;
  guint data_bits;
  GstCaps *caps;
  GstPad *src_pad;
  GstStructure *structure;
  GstTensorsConfig config;
  data_value = DATA;
  data_bits = 16;
  /** Make device */
  dev0 = make_full_device (data_value, data_bits);
  ASSERT_NE (dev0, nullptr);
  /** setup */
  samp_freq = (gint)g_ascii_strtoll (samp_freq_avail[0], NULL, 10);
  dev0->log_file = g_build_filename (dev0->base_dir, "temp.log", NULL);
  parse_launch = g_strdup_printf ("%s iio-base-dir=%s dev-dir=%s device-number=%d trigger=%s silent=FALSE channels=1,3 "
                                  "name=my-src-iio ! multifilesink location=%s",
      ELEMENT_NAME, dev0->iio_base_dir_sim, dev0->dev_dir, 0, TRIGGER_NAME, dev0->log_file);
  src_iio_pipeline = gst_parse_launch (parse_launch, NULL);
  g_free (parse_launch);
  /** state transition test upwards */
  EXPECT_EQ (setPipelineStateSync (src_iio_pipeline, GST_STATE_PLAYING, DEFAULT_POLL_TIMEOUT), 0);
  status = gst_element_get_state (src_iio_pipeline, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_PLAYING);

  /** get and verify the caps */
  src_iio = gst_bin_get_by_name (GST_BIN (src_iio_pipeline), "my-src-iio");
  ASSERT_NE (src_iio, nullptr);
  src_pad = gst_element_get_static_pad (src_iio, "src");
  ASSERT_NE (src_pad, nullptr);
  caps = gst_pad_get_current_caps (src_pad);
  ASSERT_NE (caps, nullptr);
  structure = gst_caps_get_structure (caps, 0);
  ASSERT_NE (structure, nullptr);

  /** Default has merge channels enabled */
  EXPECT_STREQ (gst_structure_get_name (structure), "other/tensor");
  EXPECT_EQ (gst_tensors_config_from_structure (&config, structure), TRUE);
  EXPECT_EQ (config.rate_n, samp_freq);
  EXPECT_EQ (config.rate_d, 1);
  EXPECT_EQ (config.info.num_tensors, 1U);
  EXPECT_EQ (config.info.info[0].type, _NNS_FLOAT32);
  EXPECT_EQ (config.info.info[0].dimension[0], 2U);
  EXPECT_EQ (config.info.info[0].dimension[1], 1U);
  EXPECT_EQ (config.info.info[0].dimension[2], 1U);
  EXPECT_EQ (config.info.info[0].dimension[3], 1U);

  gst_object_unref (src_iio);
  gst_object_unref (src_pad);
  gst_caps_unref (caps);

  /** verify paused state has been maintained */
  status = gst_element_get_state (src_iio_pipeline, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_PLAYING);
  /** state transition test downwards */
  status = gst_element_set_state (src_iio_pipeline, GST_STATE_NULL);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  status = gst_element_get_state (src_iio_pipeline, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_NULL);

  /** delete device structure */
  gst_object_unref (src_iio_pipeline);
  ASSERT_EQ (destroy_dev_dir (dev0), 0);
  clean_iio_dev_structure (dev0);
}

/**
 * @brief tests tensor source IIO data with set frequency
 * @note verifies restoration of default values
 * @note verifies setting trigger using trigger number
 * @note verifies setting frequency manually
 * @note verifies enabling of channels automatically
 * @note verifies using generic type information for channels
 */
TEST (testTensorSrcIio, dataVerifyFreqGenericType)
{
  iio_dev_dir_struct *dev0;
  GstElement *src_iio_pipeline;
  GstElement *src_iio;
  GstStateChangeReturn status;
  GstState state;
  gchar *parse_launch;
  gint samp_freq;
  gint data_value;
  guint data_bits;
  GstCaps *caps;
  GstPad *src_pad;
  GstStructure *structure;
  GstTensorsConfig config;
  gint num_scan_elements;

  data_value = DATA;
  data_bits = 16;
  gint samp_freq_idx = 1;
  gchar *ret_string = NULL;
  const gchar *buffer_length_char = "3";
  /** Make device */
  dev0 = make_full_device (data_value, data_bits);
  ASSERT_NE (dev0, nullptr);
  /** setup */
  num_scan_elements = dev0->num_scan_elements;
  samp_freq = (gint)g_ascii_strtoll (samp_freq_avail[samp_freq_idx], NULL, 10);
  dev0->log_file = g_build_filename (dev0->base_dir, "temp.log", NULL);
  parse_launch = g_strdup_printf (
      "%s iio-base-dir=%s dev-dir=%s device-number=%d trigger-number=%d silent=FALSE frequency=%d "
      "merge-channels-data=False name=my-src-iio ! multifilesink location=%s",
      ELEMENT_NAME, dev0->iio_base_dir_sim, dev0->dev_dir, 0, 0, samp_freq, dev0->log_file);
  src_iio_pipeline = gst_parse_launch (parse_launch, NULL);
  g_free (parse_launch);

  /** move channel specific type for channel 1 to generic */
  ASSERT_EQ (g_rename (dev0->scan_el_type[1], dev0->scan_el_type_generic), 0);
  /** disable all/some channels */
  for (int idx = 0; idx < num_scan_elements; idx++) {
    write_file_int (dev0->scan_el_en[idx], 0);
  }
  /** set default sampling frequency and verify reset after closure */
  write_file_string (dev0->samp_freq, samp_freq_avail[samp_freq_idx + 1]);
  /** set default sampling frequency and verify reset after closure */
  write_file_string (dev0->buf_length, buffer_length_char);

  /** state transition test upwards */
  EXPECT_EQ (setPipelineStateSync (src_iio_pipeline, GST_STATE_PLAYING, DEFAULT_POLL_TIMEOUT), 0);
  status = gst_element_get_state (src_iio_pipeline, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_PLAYING);

  /** get and verify the caps */
  src_iio = gst_bin_get_by_name (GST_BIN (src_iio_pipeline), "my-src-iio");
  ASSERT_NE (src_iio, nullptr);
  src_pad = gst_element_get_static_pad (src_iio, "src");
  ASSERT_NE (src_pad, nullptr);
  caps = gst_pad_get_current_caps (src_pad);
  ASSERT_NE (caps, nullptr);
  structure = gst_caps_get_structure (caps, 0);
  ASSERT_NE (structure, nullptr);

  /** Default has merge channels enabled */
  EXPECT_STREQ (gst_structure_get_name (structure), "other/tensors");
  EXPECT_EQ (gst_tensors_config_from_structure (&config, structure), TRUE);
  EXPECT_EQ (config.rate_n, samp_freq);
  EXPECT_EQ (config.rate_d, 1);
  EXPECT_EQ (config.info.num_tensors, (guint)num_scan_elements);
  for (int idx = 0; idx < num_scan_elements; idx++) {
    EXPECT_EQ (config.info.info[idx].type, _NNS_FLOAT32);
    EXPECT_EQ (config.info.info[idx].dimension[0], 1U);
    EXPECT_EQ (config.info.info[idx].dimension[1], 1U);
    EXPECT_EQ (config.info.info[idx].dimension[2], 1U);
    EXPECT_EQ (config.info.info[idx].dimension[3], 1U);
  }
  for (int idx = num_scan_elements; idx < NNS_TENSOR_SIZE_LIMIT; idx++) {
    EXPECT_EQ (config.info.info[idx].dimension[0], 0U);
    EXPECT_EQ (config.info.info[idx].dimension[1], 0U);
    EXPECT_EQ (config.info.info[idx].dimension[2], 0U);
    EXPECT_EQ (config.info.info[idx].dimension[3], 0U);
  }

  gst_object_unref (src_iio);
  gst_object_unref (src_pad);
  gst_caps_unref (caps);

  EXPECT_EQ (test_tensor_src_iio_data_verify_util (dev0, data_value, data_bits, samp_freq), TRUE);

  /** verify playing state has been maintained */
  status = gst_element_get_state (src_iio_pipeline, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_PLAYING);
  /** state transition test downwards */
  status = gst_element_set_state (src_iio_pipeline, GST_STATE_NULL);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  status = gst_element_get_state (src_iio_pipeline, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_NULL);

  /** Verify default values */
  EXPECT_EQ (g_file_get_contents (dev0->samp_freq, &ret_string, NULL, NULL), TRUE);
  EXPECT_STREQ (ret_string, samp_freq_avail[samp_freq_idx + 1]);
  g_free (ret_string);
  EXPECT_EQ (g_file_get_contents (dev0->buf_length, &ret_string, NULL, NULL), TRUE);
  EXPECT_STREQ (ret_string, buffer_length_char);
  g_free (ret_string);
  for (int idx = 0; idx < num_scan_elements; idx++) {
    EXPECT_EQ (g_file_get_contents (dev0->scan_el_en[idx], &ret_string, NULL, NULL), TRUE);
    EXPECT_STREQ (ret_string, "0");
    g_free (ret_string);
  }

  /** delete device structure */
  gst_object_unref (src_iio_pipeline);
  ASSERT_EQ (destroy_dev_dir (dev0), 0);
  clean_iio_dev_structure (dev0);
}

/** @todo Ignore src iio unusual_cases on ubuntu.
 *  Src iio test fails occasionally when getting state of the pipeline.
 *  Related issue : Unit TC Potential Bug / Ubuntu / IIO (https://github.com/nnstreamer/nnstreamer/issues/2434)
 *  Ignore this test on Ubuntu until an accurate solution is found.
 */
#ifdef __TIZEN__
/**
 * @brief test the unusual/exceptional cases.
 */
TEST (testTensorSrcIio, unusualCases)
#else
/**
 * @brief test the unusual/exceptional cases.
 */
TEST (testTensorSrcIio, DISABLED_unusualCases)
#endif
{
  iio_dev_dir_struct *dev0;
  GstElement *src_iio_pipeline;
  GstElement *src_iio;
  GstStateChangeReturn status;
  GstState state;
  gchar *parse_launch;
  gint samp_freq;
  gint data_value;
  guint data_bits;
  gchar *ret_device;
  gchar *ret_trigger;
  gchar *iio_base_dir;
  gchar *dev_dir;
  gchar *ret_channels;

  data_value = DATA;
  data_bits = 16;
  gint samp_freq_idx = 1;
  /** Make device */
  dev0 = make_full_device (data_value, data_bits);
  ASSERT_NE (dev0, nullptr);
  /** setup */
  samp_freq = (gint)g_ascii_strtoll (samp_freq_avail[samp_freq_idx], NULL, 10);
  dev0->log_file = g_build_filename (dev0->base_dir, "temp.log", NULL);
  parse_launch = g_strdup_printf (
      "%s iio-base-dir=%s dev-dir=%s device-number=%d trigger-number=%d silent=FALSE frequency=%d "
      "merge-channels-data=False name=my-src-iio channels=1,3,5 ! multifilesink location=%s",
      ELEMENT_NAME, dev0->iio_base_dir_sim, dev0->dev_dir, 0, 0, samp_freq, dev0->log_file);
  src_iio_pipeline = gst_parse_launch (parse_launch, NULL);
  g_free (parse_launch);

  /** get and verify the caps */
  src_iio = gst_bin_get_by_name (GST_BIN (src_iio_pipeline), "my-src-iio");
  ASSERT_NE (src_iio, nullptr);

  /** state transition test upwards */
  EXPECT_EQ (setPipelineStateSync (src_iio_pipeline, GST_STATE_PLAYING, DEFAULT_POLL_TIMEOUT), 0);
  status = gst_element_get_state (src_iio_pipeline, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_PLAYING);

  /** set property at PLAYING state */
  g_object_set (src_iio, "device", "playing-not-allow", NULL);
  g_object_get (src_iio, "device", &ret_device, NULL);
  EXPECT_STREQ (ret_device, DEVICE_NAME);
  g_free (ret_device);

  status = gst_element_set_state (src_iio, GST_STATE_READY);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  status = gst_element_get_state (src_iio, &state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (state, GST_STATE_READY);

  /** set new device name */
  g_object_set (src_iio, "device", "new-device-name", NULL);
  g_object_get (src_iio, "device", &ret_device, NULL);
  EXPECT_STREQ (ret_device, "new-device-name");
  g_free (ret_device);

  /** set new trigger name */
  g_object_set (src_iio, "trigger", "new-trigger-name", NULL);
  g_object_get (src_iio, "trigger", &ret_trigger, NULL);
  EXPECT_STREQ (ret_trigger, "new-trigger-name");
  g_free (ret_trigger);

  /** set relative path, not allowed */
  g_object_set (src_iio, "iio-base-dir", "relative/base/path/", NULL);
  g_object_get (src_iio, "iio-base-dir", &iio_base_dir, NULL);
  EXPECT_STREQ (iio_base_dir, dev0->iio_base_dir_sim);
  g_free (iio_base_dir);

  g_object_set (src_iio, "dev-dir", "relative/dev/path/", NULL);
  g_object_get (src_iio, "dev-dir", &dev_dir, NULL);
  EXPECT_STREQ (dev_dir, dev0->dev_dir);
  g_free (dev_dir);

  /** get custom channel */
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, "1,3,5");
  g_free (ret_channels);

  status = gst_element_set_state (src_iio_pipeline, GST_STATE_NULL);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);

  /** delete device structure */
  gst_object_unref (src_iio);
  gst_object_unref (src_iio_pipeline);
  ASSERT_EQ (destroy_dev_dir (dev0), 0);
  clean_iio_dev_structure (dev0);
}

/**
 * @brief test the logic with invalid frequency value
 */
TEST (testTensorSrcIio, setFrequency_n)
{
  iio_dev_dir_struct *dev0;
  GstElement *src_iio_pipeline;
  GstElement *src_iio;
  GstStateChangeReturn status;
  gchar *parse_launch;
  gint data_value;
  guint data_bits;
  gint ret_freq;

  data_value = DATA;
  data_bits = 16;
  /** Make device */
  dev0 = make_full_device (data_value, data_bits);
  ASSERT_NE (dev0, nullptr);
  /** setup */
  dev0->log_file = g_build_filename (dev0->base_dir, "temp.log", NULL);
  parse_launch = g_strdup_printf ("%s iio-base-dir=%s device-number=%d name=my-src-iio ! fakesink",
      ELEMENT_NAME, dev0->iio_base_dir_sim, 0);
  src_iio_pipeline = gst_parse_launch (parse_launch, NULL);
  g_free (parse_launch);
  ASSERT_NE (src_iio_pipeline, nullptr);

  /** get and verify the caps */
  src_iio = gst_bin_get_by_name (GST_BIN (src_iio_pipeline), "my-src-iio");
  ASSERT_NE (src_iio, nullptr);

  /** set the invalid frequency */
  g_object_set (src_iio, "frequency", -1, NULL);
  g_object_get (src_iio, "frequency", &ret_freq, NULL);
  EXPECT_EQ (ret_freq, -1);

  /** fail with invalid frequency */
  status = gst_element_set_state (src_iio_pipeline, GST_STATE_PLAYING);
  EXPECT_NE (status, GST_STATE_CHANGE_ASYNC);

  status = gst_element_set_state (src_iio_pipeline, GST_STATE_NULL);
  EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);

  /** delete device structure */
  gst_object_unref (src_iio);
  gst_object_unref (src_iio_pipeline);
}

/**
 * @brief test the logic with invalid base dir
 */
TEST (testTensorSrcIio, setBaseDir_n)
{
  iio_dev_dir_struct *dev0;
  GstHarness *hrnss = NULL;
  GstElement *src_iio = NULL;
  GstStateChangeReturn status;

  /** Make device */
  dev0 = make_full_device (DATA, 16);
  ASSERT_NE (dev0, nullptr);

  /** setup */
  hrnss = gst_harness_new_empty ();
  ASSERT_TRUE (hrnss != NULL);
  gst_harness_add_parse (hrnss, ELEMENT_NAME);
  src_iio = gst_harness_find_element (hrnss, ELEMENT_NAME);
  ASSERT_TRUE (src_iio != NULL);

  /** setup properties */
  g_object_set (src_iio, "iio-base-dir", "/not/existed/path", NULL);
  g_object_set (src_iio, "dev-dir", dev0->dev_dir, NULL);
  g_object_set (src_iio, "device", DEVICE_NAME, NULL);

  status = gst_element_set_state (src_iio, GST_STATE_PAUSED);
  EXPECT_NE (status, GST_STATE_CHANGE_NO_PREROLL);

  /** teardown */
  gst_object_unref (src_iio);
  gst_harness_teardown (hrnss);
}

/**
 * @brief Main function for unit test.
 */
int
main (int argc, char **argv)
{
  int ret = -1;
  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  try {
    ret = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return ret;
}
