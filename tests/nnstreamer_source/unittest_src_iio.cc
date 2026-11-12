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

#define CHANGE_ENDIANNESS(NUMBITS)                                   \
  {                                                                  \
    if (signchar == 's') {                                           \
      sdata##NUMBITS = (gint##NUMBITS *) (scan_el_data + location);  \
      if (endianchar == 'l') {                                       \
        *sdata##NUMBITS = GINT##NUMBITS##_TO_LE (sdata << shift);    \
      } else {                                                       \
        *sdata##NUMBITS = GINT##NUMBITS##_TO_BE (sdata << shift);    \
      }                                                              \
    } else {                                                         \
      udata##NUMBITS = (guint##NUMBITS *) (scan_el_data + location); \
      if (endianchar == 'l') {                                       \
        *udata##NUMBITS = GUINT##NUMBITS##_TO_LE (udata << shift);   \
      } else {                                                       \
        *udata##NUMBITS = GUINT##NUMBITS##_TO_BE (udata << shift);   \
      }                                                              \
    }                                                                \
    location += num_bytes;                                           \
    break;                                                           \
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
  iio_dev_dir_struct *iio_dev = (iio_dev_dir_struct *) vargp;
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
  gint val = pthread_create (&thread_read, NULL, open_fifo_read, (void *) iio_dev);
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
  gchar *samp_freq_avail_string = g_strjoinv (" ", (char **) samp_freq_avail);

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
  gsize data_size;
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
  scan_el_data = (char *) g_malloc0 (data_size);
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
        case 1:
          {
            /** endian-ness does not matter for 1 byte */
            if (signchar == 's') {
              sdata8 = (gint8 *) (scan_el_data + location);
              *sdata8 = sdata << shift;
            } else {
              udata8 = (guint8 *) (scan_el_data + location);
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
        default:
          {
            g_free (scan_el_data);
            return -1;
          }
      };
    }
  }

  gchar *copied_scan_el_data = (gchar *) g_malloc (data_size * BUF_LENGTH);
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
 * @brief build scan data elements dir with an exact storage size
 * @param[in] iio_dev struct of iio device
 * @param[in] num_bits num of bits used and stored for the data, multiple of 8
 * @param[in] value data stored in every channel
 * @returns 0 on success, non-zero on failure
 * @note the IIO ABI also allows a storage size of 3, 5, 6 and 7 bytes, which
 * build_dev_dir_scan_elements() cannot describe as it rounds the size up
 */
static gint
build_dev_dir_scan_elements_exact (
    iio_dev_dir_struct *iio_dev, const guint num_bits, const guint64 value)
{
  gint status = 0;
  gchar *type_data;
  gchar endianchar, signchar;
  guint num_bytes, byte_idx;
  guint8 data_byte;
  gsize data_size;
  gchar *scan_el_data;
  gint location = 0;

  if (!g_file_test (iio_dev->scan_el, G_FILE_TEST_IS_DIR)) {
    status += g_mkdir_with_parents (iio_dev->scan_el, 0777);
  }

  num_bytes = ((num_bits - 1) >> 3) + 1;
  data_size = num_bytes * iio_dev->num_scan_elements;
  scan_el_data = (gchar *) g_malloc0 (data_size);
  if (scan_el_data == NULL) {
    return -1;
  }

  for (int idx = 0; idx < iio_dev->num_scan_elements; idx++) {
    endianchar = (idx % 2) ? 'l' : 'b';
    signchar = (idx % 4 < 2) ? 's' : 'u';

    status += write_file_int (iio_dev->scan_el_en[idx], 1);
    status += write_file_int (iio_dev->scan_el_index[idx], idx);
    type_data = g_strdup_printf (
        "%ce:%c%u/%u>>0", endianchar, signchar, num_bits, num_bytes * 8);
    status += write_file_string (iio_dev->scan_el_type[idx], type_data);
    g_free (type_data);

    for (byte_idx = 0; byte_idx < num_bytes; byte_idx++) {
      data_byte = (guint8) ((value >> (8 * byte_idx)) & 0xFF);
      if (endianchar == 'l') {
        scan_el_data[location + byte_idx] = data_byte;
      } else {
        scan_el_data[location + num_bytes - 1 - byte_idx] = data_byte;
      }
    }
    location += num_bytes;
  }

  status += write_file_string_single_trigger (
      iio_dev->dev_device_dir_fd, scan_el_data, data_size);
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
  /** disabled as by default; the element skips an enabled one in a scan */
  status += write_file_int (iio_dev->scan_el_time_en, 0);
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
 * @param[in] skip skip some of the channels
 * @param[in] exact_storage keep the storage size of the channels unrounded
 * @returns allocated device structure on success, NULL on error
 */
static iio_dev_dir_struct *
make_full_device (const guint64 data_value, const gint data_bits,
    const gboolean trigger = TRUE, const gint skip = 1, const gboolean exact_storage = FALSE)
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
  if (exact_storage) {
    status += build_dev_dir_scan_elements_exact (dev0, data_bits, data_value);
  } else {
    status += build_dev_dir_scan_elements (dev0, data_bits, data_value, data_value, skip);
  }
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
    samp_freq = (gint) g_ascii_strtoll (samp_freq_avail[0], NULL, 10);                           \
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
    data_buffer = (gchar *) malloc (bytes_to_read);                                              \
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
      actual_val = *((gfloat *) data_buffer);                                                    \
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
    data_buffer = (gchar *) malloc (bytes_to_read);
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
      actual_val = *((gfloat *) data_buffer);
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
  samp_freq = (gint) g_ascii_strtoll (samp_freq_avail[0], NULL, 10);
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
  EXPECT_EQ (config.info.info[0].dimension[0], (guint) num_scan_elements);
  EXPECT_EQ (config.info.info[0].dimension[1], 1U);
  EXPECT_EQ (config.info.info[0].dimension[2], 0U);

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
  samp_freq = (gint) g_ascii_strtoll (samp_freq_avail[0], NULL, 10);
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
  EXPECT_EQ (config.info.info[0].dimension[2], 0U);

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
  samp_freq = (gint) g_ascii_strtoll (samp_freq_avail[samp_freq_idx], NULL, 10);
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
  EXPECT_EQ (config.info.num_tensors, (guint) num_scan_elements);
  for (int idx = 0; idx < num_scan_elements; idx++) {
    EXPECT_EQ (config.info.info[idx].type, _NNS_FLOAT32);
    EXPECT_EQ (config.info.info[idx].dimension[0], 1U);
    EXPECT_EQ (config.info.info[idx].dimension[1], 1U);
    EXPECT_EQ (config.info.info[idx].dimension[2], 0U);
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
  samp_freq = (gint) g_ascii_strtoll (samp_freq_avail[samp_freq_idx], NULL, 10);
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
 * @brief run a pipeline and verify the data of channels with an exact storage
 * @param[in] num_bits num of bits used and stored for the data, multiple of 8
 */
static void
run_data_verify_exact_storage (const guint num_bits)
{
  static const gint MAX_NUM_TRY = 100;
  gint num_try;
  iio_dev_dir_struct *dev0;
  GstElement *src_iio_pipeline;
  GstStateChangeReturn status;
  GstState state;
  gchar *parse_launch;
  gint samp_freq;
  gint fd, ret;
  size_t idx, bytes_read, bytes_to_read;
  gchar *data_buffer;
  gfloat expect_val, actual_val;
  guint64 expect_val_mask;
  gchar *expect_val_char, *actual_val_char;
  struct stat stat_buf;

  /** Make device */
  dev0 = make_full_device (DATA, num_bits, FALSE, 1, TRUE);
  ASSERT_NE (dev0, nullptr);

  /** setup */
  samp_freq = (gint) g_ascii_strtoll (samp_freq_avail[0], NULL, 10);
  dev0->log_file = g_build_filename (dev0->base_dir, "temp.log", NULL);
  parse_launch = g_strdup_printf (
      "%s iio-base-dir=%s dev-dir=%s device=%s silent=FALSE ! multifilesink location=%s",
      ELEMENT_NAME, dev0->iio_base_dir_sim, dev0->dev_dir, DEVICE_NAME, dev0->log_file);
  src_iio_pipeline = gst_parse_launch (parse_launch, NULL);
  g_free (parse_launch);
  ASSERT_NE (src_iio_pipeline, nullptr);

  EXPECT_EQ (setPipelineStateSync (src_iio_pipeline, GST_STATE_PLAYING, DEFAULT_POLL_TIMEOUT), 0);

  /** feed the device till multifilesink makes the needed file */
  num_try = 0;
  while ((fd = open (dev0->log_file, O_RDONLY)) < 0) {
    if (num_try >= MAX_NUM_TRY) {
      FAIL () << "Failed to open " << dev0->log_file;
    }
    g_usleep (MAX (10, 1000000 / samp_freq));
    num_try++;
  }
  for (num_try = 0; num_try < MAX_NUM_TRY; ++num_try) {
    g_usleep (MAX (10, 1000000 / samp_freq));
    if (build_dev_dir_scan_elements_exact (dev0, num_bits, DATA) != 0) {
      close (fd);
      FAIL () << "Failed to build and fill scan elements directory";
    }
    ret = stat (dev0->log_file, &stat_buf);
    if (ret == 0 && stat_buf.st_size != 0) {
      status = gst_element_set_state (src_iio_pipeline, GST_STATE_NULL);
      EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
      status = gst_element_get_state (src_iio_pipeline, &state, NULL, GST_CLOCK_TIME_NONE);
      EXPECT_EQ (status, GST_STATE_CHANGE_SUCCESS);
      EXPECT_EQ (state, GST_STATE_NULL);
      break;
    }
  }

  /** verify correctness of data of every channel */
  bytes_to_read = sizeof (float) * BUF_LENGTH * dev0->num_scan_elements;
  data_buffer = (gchar *) g_malloc (bytes_to_read);
  ret = read (fd, data_buffer, bytes_to_read);
  close (fd);
  ASSERT_GE (ret, 0);
  bytes_read = static_cast<size_t> (ret);
  EXPECT_EQ (bytes_read, bytes_to_read);

  expect_val_mask = G_MAXUINT64 >> (64 - num_bits);
  expect_val = ((DATA & expect_val_mask) + OFFSET) * SCALE;
  expect_val_char = g_strdup_printf ("%.2f", expect_val);
  for (idx = 0; idx < bytes_read / sizeof (float); idx++) {
    actual_val = ((gfloat *) data_buffer)[idx];
    actual_val_char = g_strdup_printf ("%.2f", actual_val);
    EXPECT_STREQ (expect_val_char, actual_val_char);
    g_free (actual_val_char);
  }
  g_free (expect_val_char);
  g_free (data_buffer);

  /** delete device structure */
  ASSERT_EQ (safe_remove (dev0->log_file), 0);
  gst_object_unref (src_iio_pipeline);
  ASSERT_EQ (destroy_dev_dir (dev0), 0);
  clean_iio_dev_structure (dev0);
}

/**
 * @brief tests data of channels stored in 3 bytes
 */
TEST (testTensorSrcIio, dataVerifyExactStorage24)
{
  run_data_verify_exact_storage (24);
}

/**
 * @brief tests data of channels stored in 7 bytes
 */
TEST (testTensorSrcIio, dataVerifyExactStorage56)
{
  run_data_verify_exact_storage (56);
}

/**
 * @brief add scan elements of a disabled channel and run the device
 * @param[in] names files of the extra channel, contents in the same order
 * @param[in] contents contents of the files
 * @param[in] num_files number of the extra files
 * @param[in] expected state change expected when the device is configured
 * @note the generic name of a channel is its name without the trailing digits,
 * which is empty for the names used here
 */
static void
run_extra_channel_case (const gchar *const *names, const gchar *const *contents,
    const guint num_files, const GstStateChangeReturn expected)
{
  iio_dev_dir_struct *dev0;
  GstHarness *hrnss = NULL;
  GstElement *src_iio = NULL;
  GstStateChangeReturn status;
  gchar **extra_files;
  guint idx;

  /** Make device */
  dev0 = make_full_device (DATA, 16);
  ASSERT_NE (dev0, nullptr);

  extra_files = g_new0 (gchar *, num_files);
  for (idx = 0; idx < num_files; idx++) {
    extra_files[idx] = g_build_filename (dev0->scan_el, names[idx], NULL);
    ASSERT_EQ (write_file_string (extra_files[idx], contents[idx]), 0);
  }

  /** setup */
  hrnss = gst_harness_new_empty ();
  ASSERT_TRUE (hrnss != NULL);
  gst_harness_add_parse (hrnss, ELEMENT_NAME);
  src_iio = gst_harness_find_element (hrnss, ELEMENT_NAME);
  ASSERT_TRUE (src_iio != NULL);

  g_object_set (src_iio, "iio-base-dir", dev0->iio_base_dir_sim, NULL);
  g_object_set (src_iio, "dev-dir", dev0->dev_dir, NULL);
  g_object_set (src_iio, "device", DEVICE_NAME, NULL);

  status = gst_element_set_state (src_iio, GST_STATE_PAUSED);
  EXPECT_EQ (status, expected);
  EXPECT_EQ (gst_element_set_state (src_iio, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);

  /** teardown */
  gst_object_unref (src_iio);
  gst_harness_teardown (hrnss);

  for (idx = 0; idx < num_files; idx++) {
    EXPECT_EQ (safe_remove (extra_files[idx]), 0);
    g_free (extra_files[idx]);
  }
  g_free (extra_files);
  ASSERT_EQ (destroy_dev_dir (dev0), 0);
  clean_iio_dev_structure (dev0);
}

/**
 * @brief tests a channel whose name holds nothing but its index
 */
TEST (testTensorSrcIio, channelNameDigitsOnly)
{
  const gchar *names[] = { "123_en", "123_index", "123_type" };
  const gchar *contents[] = { "0", "8", "le:s16/16>>0" };

  /** the extra channel is disabled, so the device is configured as usual */
  run_extra_channel_case (names, contents, G_N_ELEMENTS (names), GST_STATE_CHANGE_NO_PREROLL);
}

/**
 * @brief tests a channel whose name is empty
 */
TEST (testTensorSrcIio, channelNameEmpty_n)
{
  const gchar *names[] = { "_en", "_index", "_type" };
  const gchar *contents[] = { "0", "8", "le:s16/16>>0" };

  /** such a channel has no file of its own to read the state from */
  run_extra_channel_case (names, contents, G_N_ELEMENTS (names), GST_STATE_CHANGE_FAILURE);
}

/**
 * @brief tests the bounds of the used bits of a channel type
 */
TEST (testTensorSrcIio, channelTypeUsedBits)
{
  const gchar *names[] = { "123_en", "123_index", "123_type" };
  const gchar *types[] = { "le:u1/8>>7", "be:s1/8>>0", "le:s64/64>>0", "be:u64/64>>0" };
  const gchar *contents[3] = { "0", "8", NULL };

  for (guint idx = 0; idx < G_N_ELEMENTS (types); idx++) {
    contents[2] = types[idx];
    run_extra_channel_case (names, contents, G_N_ELEMENTS (names), GST_STATE_CHANGE_NO_PREROLL);
  }
}

/**
 * @brief tests that a channel type with no used bits or more than 64 is refused
 */
TEST (testTensorSrcIio, channelTypeUsedBits_n)
{
  const gchar *names[] = { "123_en", "123_index", "123_type" };
  const gchar *types[] = { "le:s0/16>>0", "be:u0/64>>0", "le:u/16>>0", "le:u1/0>>0",
    "le:s65/64>>0", "le:u18446744073709551616/16>>0", "le:u4294967296/16>>0" };
  const gchar *contents[3] = { "0", "8", NULL };

  for (guint idx = 0; idx < G_N_ELEMENTS (types); idx++) {
    contents[2] = types[idx];
    run_extra_channel_case (names, contents, G_N_ELEMENTS (names), GST_STATE_CHANGE_FAILURE);
  }
}

/**
 * @brief tests that a buffer capacity larger than the readable size is refused
 */
TEST (testTensorSrcIio, bufferCapacityOverflow_n)
{
  iio_dev_dir_struct *dev0;
  GstHarness *hrnss = NULL;
  GstElement *src_iio = NULL;
  GstStateChangeReturn status;

  /** Make device, 8 channels of 2 bytes make a scan of 16 bytes */
  dev0 = make_full_device (DATA, 16);
  ASSERT_NE (dev0, nullptr);

  /** setup */
  hrnss = gst_harness_new_empty ();
  ASSERT_TRUE (hrnss != NULL);
  gst_harness_add_parse (hrnss, ELEMENT_NAME);
  src_iio = gst_harness_find_element (hrnss, ELEMENT_NAME);
  ASSERT_TRUE (src_iio != NULL);

  g_object_set (src_iio, "iio-base-dir", dev0->iio_base_dir_sim, NULL);
  g_object_set (src_iio, "dev-dir", dev0->dev_dir, NULL);
  g_object_set (src_iio, "device", DEVICE_NAME, NULL);
  /** 16 bytes per scan of this capacity do not fit in a signed int */
  g_object_set (src_iio, "buffer-capacity", 268435457U, NULL);

  status = gst_element_set_state (src_iio, GST_STATE_PAUSED);
  EXPECT_EQ (status, GST_STATE_CHANGE_FAILURE);
  EXPECT_EQ (gst_element_set_state (src_iio, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);

  /** teardown */
  gst_object_unref (src_iio);
  gst_harness_teardown (hrnss);

  ASSERT_EQ (destroy_dev_dir (dev0), 0);
  clean_iio_dev_structure (dev0);
}

/**
 * @brief tests that the buffer capacity of a configured device does not change
 * @note the size of a read is checked against the buffer capacity when the
 * device is configured, which holds only while the property stays where the
 * check left it
 */
TEST (testTensorSrcIio, bufferCapacityWhileConfigured_n)
{
  iio_dev_dir_struct *dev0;
  GstHarness *hrnss = NULL;
  GstElement *src_iio = NULL;
  GstStateChangeReturn status;
  guint ret_buffer_capacity;

  /** Make device */
  dev0 = make_full_device (DATA, 16);
  ASSERT_NE (dev0, nullptr);

  /** setup */
  hrnss = gst_harness_new_empty ();
  ASSERT_TRUE (hrnss != NULL);
  gst_harness_add_parse (hrnss, ELEMENT_NAME);
  src_iio = gst_harness_find_element (hrnss, ELEMENT_NAME);
  ASSERT_TRUE (src_iio != NULL);

  g_object_set (src_iio, "iio-base-dir", dev0->iio_base_dir_sim, NULL);
  g_object_set (src_iio, "dev-dir", dev0->dev_dir, NULL);
  g_object_set (src_iio, "device", DEVICE_NAME, NULL);
  g_object_set (src_iio, "buffer-capacity", 2U, NULL);

  status = gst_element_set_state (src_iio, GST_STATE_PAUSED);
  EXPECT_EQ (status, GST_STATE_CHANGE_NO_PREROLL);

  /** the buffer of the device and the caps are sized by the old value */
  g_object_set (src_iio, "buffer-capacity", 1000U, NULL);
  g_object_get (src_iio, "buffer-capacity", &ret_buffer_capacity, NULL);
  EXPECT_EQ (ret_buffer_capacity, 2U);

  EXPECT_EQ (gst_element_set_state (src_iio, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);

  /** the value is writable again once the device is released */
  g_object_set (src_iio, "buffer-capacity", 1000U, NULL);
  g_object_get (src_iio, "buffer-capacity", &ret_buffer_capacity, NULL);
  EXPECT_EQ (ret_buffer_capacity, 1000U);

  /** teardown */
  gst_object_unref (src_iio);
  gst_harness_teardown (hrnss);

  ASSERT_EQ (destroy_dev_dir (dev0), 0);
  clean_iio_dev_structure (dev0);
}

/**
 * @brief counts the critical logs of glib
 */
static void
count_glib_critical (const gchar *, GLogLevelFlags, const gchar *, gpointer user_data)
{
  guint *count = (guint *) user_data;

  *count = *count + 1;
}

/**
 * @brief tests the custom channels property
 */
TEST (testTensorSrcIio, channelsCustom)
{
  GstHarness *hrnss = NULL;
  GstElement *src_iio = NULL;
  gchar *ret_channels;
  gchar **strv;

  /** setup */
  hrnss = gst_harness_new_empty ();
  ASSERT_TRUE (hrnss != NULL);
  gst_harness_add_parse (hrnss, ELEMENT_NAME);
  src_iio = gst_harness_find_element (hrnss, ELEMENT_NAME);
  ASSERT_TRUE (src_iio != NULL);

  g_object_set (src_iio, "channels", "1", NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, "1");
  g_free (ret_channels);

  /** a new value replaces the previous one */
  g_object_set (src_iio, "channels", "2", NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, "2");
  g_free (ret_channels);

  g_object_set (src_iio, "channels", "3,4;5", NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  strv = g_strsplit (ret_channels, ",", -1);
  EXPECT_EQ (g_strv_length (strv), 3U);
  EXPECT_TRUE (g_strv_contains ((const gchar *const *) strv, "3"));
  EXPECT_TRUE (g_strv_contains ((const gchar *const *) strv, "4"));
  EXPECT_TRUE (g_strv_contains ((const gchar *const *) strv, "5"));
  g_strfreev (strv);
  g_free (ret_channels);

  /** a duplicated index is registered once */
  g_object_set (src_iio, "channels", "6,6", NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, "6");
  g_free (ret_channels);

  /** blanks around an index are ignored */
  g_object_set (src_iio, "channels", " 7 , 8 ", NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  strv = g_strsplit (ret_channels, ",", -1);
  EXPECT_EQ (g_strv_length (strv), 2U);
  EXPECT_TRUE (g_strv_contains ((const gchar *const *) strv, "7"));
  EXPECT_TRUE (g_strv_contains ((const gchar *const *) strv, "8"));
  g_strfreev (strv);
  g_free (ret_channels);

  /** the largest index the table can hold */
  g_object_set (src_iio, "channels", "2147483647", NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, "2147483647");
  g_free (ret_channels);

  /** the enum values are still accepted */
  g_object_set (src_iio, "channels", channels[1], NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, channels[1]);
  g_free (ret_channels);

  /** in any letter case */
  g_object_set (src_iio, "channels", "AUTO", NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, channels[0]);
  g_free (ret_channels);

  /** and with blanks around, as the indices are */
  g_object_set (src_iio, "channels", " All ", NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, channels[1]);
  g_free (ret_channels);

  /** teardown */
  gst_object_unref (src_iio);
  gst_harness_teardown (hrnss);
}

/**
 * @brief tests that an unparsable custom channels value keeps the old one
 */
TEST (testTensorSrcIio, channelsCustomInvalid_n)
{
  GstHarness *hrnss = NULL;
  GstElement *src_iio = NULL;
  gchar *ret_channels;
  guint critical_count = 0;
  guint log_id;
  GLogLevelFlags fatal_mask;
  const gchar *invalid[] = { "abc", "-1", "2,abc", "99999999999999999999",
    "4294967297", "1x", "2,", ",2", "2;;3", "+1", "0x1", "1 2", "2147483648",
    "allx", "all,1", "automatic", "auto1", "" };
  guint idx;

  /** setup */
  hrnss = gst_harness_new_empty ();
  ASSERT_TRUE (hrnss != NULL);
  gst_harness_add_parse (hrnss, ELEMENT_NAME);
  src_iio = gst_harness_find_element (hrnss, ELEMENT_NAME);
  ASSERT_TRUE (src_iio != NULL);

  log_id = g_log_set_handler ("GLib",
      (GLogLevelFlags) (G_LOG_LEVEL_CRITICAL | G_LOG_FLAG_FATAL | G_LOG_FLAG_RECURSION),
      count_glib_critical, &critical_count);
  fatal_mask = g_log_set_always_fatal ((GLogLevelFlags) G_LOG_FATAL_MASK);
  g_log ("GLib", G_LOG_LEVEL_CRITICAL, "[probe] tensor_src_iio test GLib handler self-check");
  g_log_set_always_fatal (fatal_mask);
  EXPECT_EQ (critical_count, 1U);
  critical_count = 0;

  g_object_set (src_iio, "channels", "7", NULL);

  for (idx = 0; idx < G_N_ELEMENTS (invalid); idx++) {
    g_object_set (src_iio, "channels", invalid[idx], NULL);
    g_object_get (src_iio, "channels", &ret_channels, NULL);
    EXPECT_STREQ (ret_channels, "7") << "with the value " << invalid[idx];
    g_free (ret_channels);
  }

  g_object_set (src_iio, "channels", NULL, NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, "7") << "with the null value";
  g_free (ret_channels);

  /** a refused value does not block the next one */
  g_object_set (src_iio, "channels", "8", NULL);
  g_object_get (src_iio, "channels", &ret_channels, NULL);
  EXPECT_STREQ (ret_channels, "8");
  g_free (ret_channels);

  EXPECT_EQ (critical_count, 0U);
  g_log_remove_handler ("GLib", log_id);

  /** teardown */
  gst_object_unref (src_iio);
  gst_harness_teardown (hrnss);
}

/**
 * @brief number of scans in a buffer of the mock devices below
 */
#define MOCK_CAPACITY 4U

/**
 * @brief a scan element of a mock device, named as the kernel names it
 */
typedef struct {
  const gchar *name; /**< channel name without the attribute suffix */
  gint index; /**< scan index */
  gboolean is_signed; /**< sign of the data */
  guint used_bits; /**< bits of the data */
  guint storage_bytes; /**< bytes the channel takes in a scan */
  guint shift; /**< shift of the data in its storage */
  gboolean timestamp; /**< the timestamp, which is never put in the tensor */
  gint64 samples[MOCK_CAPACITY]; /**< data of each scan */
} mock_channel;

/**
 * @brief an accelerometer: three 12-bit axes in the upper bits of 16 bits
 * ("le:s12/16>>4") and the 64-bit timestamp ("le:s64/64>>0")
 */
static const mock_channel accel_channels[] = {
  { "in_accel_x", 0, TRUE, 12, 2, 4, FALSE, { 100, -2048, -1, 1000 } },
  { "in_accel_y", 1, TRUE, 12, 2, 4, FALSE, { -200, 0, 512, -1000 } },
  { "in_accel_z", 2, TRUE, 12, 2, 4, FALSE, { 2047, 1, -512, 7 } },
  { "in_timestamp", 3, TRUE, 64, 8, 0, TRUE,
      { G_GINT64_CONSTANT (1700000000000000001), G_GINT64_CONSTANT (1700000000000000002),
          G_GINT64_CONSTANT (-1), G_GINT64_CONSTANT (1700000000000000004) } },
};

/**
 * @brief path of an attribute of a mock scan element
 */
static gchar *
mock_attr_path (const iio_dev_dir_struct *dev0, const gchar *name, const gchar *suffix)
{
  g_autofree gchar *file = g_strdup_printf ("%s%s", name, suffix);

  return g_build_filename (dev0->scan_el, file, NULL);
}

/**
 * @brief read the enable attribute of a mock scan element
 * @returns the value read, -1 on failure
 */
static gint
read_mock_enable (const iio_dev_dir_struct *dev0, const gchar *name)
{
  g_autofree gchar *path = mock_attr_path (dev0, name, "_en");
  g_autofree gchar *contents = NULL;

  if (!g_file_get_contents (path, &contents, NULL, NULL))
    return -1;

  return (gint) g_ascii_strtoll (contents, NULL, 10);
}

/**
 * @brief make a device with the given scan elements
 * @param[in] chs scan elements of the device
 * @param[in] num number of the scan elements
 * @param[in] pre_enabled whether each scan element is enabled
 * @returns allocated device structure on success, NULL on error
 */
static iio_dev_dir_struct *
make_mock_device (const mock_channel *chs, const guint num, const gboolean *pre_enabled)
{
  iio_dev_dir_struct *dev0;
  gint status = 0;
  gchar *path, *type;
  guint idx;

  dev0 = make_iio_dev_structure (0);
  status += build_dev_dir_base (dev0);
  status += build_dev_dir_buffer (dev0);
  status += build_dev_dir_dev_data (dev0);
  status += build_dev_dir_samp_freq (dev0);
  status += build_dev_dir_samp_freq_avail (dev0);
  status += g_mkdir_with_parents (dev0->scan_el, 0777);

  for (idx = 0; idx < num; idx++) {
    path = mock_attr_path (dev0, chs[idx].name, "_en");
    status += write_file_int (path, pre_enabled[idx] ? 1 : 0);
    g_free (path);

    path = mock_attr_path (dev0, chs[idx].name, "_index");
    status += write_file_int (path, chs[idx].index);
    g_free (path);

    path = mock_attr_path (dev0, chs[idx].name, "_type");
    type = g_strdup_printf ("le:%c%u/%u>>%u", chs[idx].is_signed ? 's' : 'u',
        chs[idx].used_bits, chs[idx].storage_bytes * 8, chs[idx].shift);
    status += write_file_string (path, type);
    g_free (type);
    g_free (path);
  }

  if (status != 0) {
    destroy_dev_dir (dev0);
    clean_iio_dev_structure (dev0);
    return NULL;
  }

  return dev0;
}

/**
 * @brief remove a device made by make_mock_device()
 */
static void
destroy_mock_device (iio_dev_dir_struct *dev0, const mock_channel *chs, const guint num)
{
  const gchar *suffixes[] = { "_en", "_index", "_type" };
  gchar *path;

  for (guint idx = 0; idx < num; idx++) {
    for (guint s = 0; s < G_N_ELEMENTS (suffixes); s++) {
      path = mock_attr_path (dev0, chs[idx].name, suffixes[s]);
      EXPECT_EQ (safe_remove (path), 0);
      g_free (path);
    }
  }

  EXPECT_EQ (destroy_dev_dir (dev0), 0);
  clean_iio_dev_structure (dev0);
}

/**
 * @brief round up to a multiple of the given size
 */
#define MOCK_ALIGN(x, a) ((((x) + (a) - 1) / (a)) * (a))

/**
 * @brief write MOCK_CAPACITY scans of the enabled channels as the kernel
 * lays them out: the channels in the order of their index, each aligned to
 * its storage size, then the timestamp, and the scan padded to a multiple of
 * its largest channel
 * @note chs has to be sorted by the index; the bits under the data are not
 * zero, as a device may leave anything there
 * @returns 0 on success, non-zero on failure
 */
static gint
feed_mock_scans (iio_dev_dir_struct *dev0, const mock_channel *chs, const guint num)
{
  g_autofree guint *location = g_new (guint, num);
  g_autofree guint8 *data = NULL;
  guint scan_bytes = 0, largest = 0;
  guint idx, pass, scan, b;
  guint64 raw;
  gint enabled;

  for (pass = 0; pass < 2; pass++) {
    for (idx = 0; idx < num; idx++) {
      if (chs[idx].timestamp != (pass == 1))
        continue;

      enabled = read_mock_enable (dev0, chs[idx].name);
      if (enabled < 0)
        return -1;

      location[idx] = G_MAXUINT;
      if (enabled == 0)
        continue;

      scan_bytes = MOCK_ALIGN (scan_bytes, chs[idx].storage_bytes);
      location[idx] = scan_bytes;
      scan_bytes += chs[idx].storage_bytes;
      largest = MAX (largest, chs[idx].storage_bytes);
    }
  }

  if (largest == 0)
    return -1;
  scan_bytes = MOCK_ALIGN (scan_bytes, largest);

  data = (guint8 *) g_malloc0 (scan_bytes * MOCK_CAPACITY);
  for (scan = 0; scan < MOCK_CAPACITY; scan++) {
    for (idx = 0; idx < num; idx++) {
      if (location[idx] == G_MAXUINT)
        continue;

      raw = ((guint64) chs[idx].samples[scan]) & (G_MAXUINT64 >> (64 - chs[idx].used_bits));
      raw = (raw << chs[idx].shift)
            | (0x5 & ((G_GUINT64_CONSTANT (1) << chs[idx].shift) - 1));
      for (b = 0; b < chs[idx].storage_bytes; b++)
        data[scan * scan_bytes + location[idx] + b] = (guint8) (raw >> (8 * b));
    }
  }

  return write_file_string_single_trigger (
      dev0->dev_device_dir_fd, (const gchar *) data, scan_bytes * MOCK_CAPACITY);
}

/**
 * @brief data and caps a fakesink received
 */
typedef struct {
  GByteArray *data; /**< contents of the buffers */
  GstCaps *caps; /**< caps of the first buffer */
  guint num_buffers; /**< number of buffers */
} mock_sink_data;

/**
 * @brief keep what the fakesink receives
 */
static void
mock_sink_handoff (GstElement *, GstBuffer *buffer, GstPad *pad, gpointer user_data)
{
  mock_sink_data *sink_data = (mock_sink_data *) user_data;
  GstMapInfo map;

  if (gst_buffer_map (buffer, &map, GST_MAP_READ)) {
    g_byte_array_append (sink_data->data, map.data, map.size);
    gst_buffer_unmap (buffer, &map);
  }
  if (sink_data->caps == NULL)
    sink_data->caps = gst_pad_get_current_caps (pad);
  sink_data->num_buffers++;
}

/**
 * @brief run a mock device and check the buffer the element hands downstream
 * @param[in] chs scan elements of the device, sorted by the index
 * @param[in] num number of the scan elements
 * @param[in] pre_enabled whether each scan element is enabled before the element starts
 * @param[in] channels value of the channels property, NULL to keep the default
 * @param[in] refused values set after channels, each of which has to be refused
 * @param[in] num_refused number of the refused values
 * @param[in] expected whether each scan element is expected in the tensor
 */
static void
run_mock_device (const mock_channel *chs, const guint num,
    const gboolean *pre_enabled, const gchar *channels,
    const gchar *const *refused, const guint num_refused, const gboolean *expected)
{
  iio_dev_dir_struct *dev0;
  GstElement *pipeline, *src_iio, *sink;
  GstBus *bus;
  GstMessage *msg;
  GstTensorsConfig config;
  mock_sink_data sink_data = { g_byte_array_new (), NULL, 0 };
  gchar *parse_launch, *before, *after;
  const gfloat *values;
  guint idx, scan, col, num_expected = 0;

  for (idx = 0; idx < num; idx++) {
    if (!chs[idx].timestamp && expected[idx])
      num_expected++;
  }

  dev0 = make_mock_device (chs, num, pre_enabled);
  ASSERT_NE (dev0, nullptr);

  parse_launch = g_strdup_printf ("%s name=src iio-base-dir=%s dev-dir=%s device=%s "
                                  "buffer-capacity=%u num-buffers=1 silent=FALSE ! "
                                  "fakesink name=sink signal-handoffs=true sync=false",
      ELEMENT_NAME, dev0->iio_base_dir_sim, dev0->dev_dir, DEVICE_NAME, MOCK_CAPACITY);
  pipeline = gst_parse_launch (parse_launch, NULL);
  g_free (parse_launch);
  ASSERT_NE (pipeline, nullptr);
  src_iio = gst_bin_get_by_name (GST_BIN (pipeline), "src");
  ASSERT_NE (src_iio, nullptr);
  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");
  ASSERT_NE (sink, nullptr);
  g_signal_connect (sink, "handoff", G_CALLBACK (mock_sink_handoff), &sink_data);

  if (channels != NULL) {
    g_object_set (src_iio, "channels", channels, NULL);
    g_object_get (src_iio, "channels", &before, NULL);
    for (idx = 0; idx < num_refused; idx++) {
      g_object_set (src_iio, "channels", refused[idx], NULL);
      g_object_get (src_iio, "channels", &after, NULL);
      EXPECT_STREQ (after, before) << "with the value " << refused[idx];
      g_free (after);
    }
    g_free (before);
  }

  /** the element sets the scan elements up when it starts */
  EXPECT_EQ (gst_element_set_state (pipeline, GST_STATE_PAUSED), GST_STATE_CHANGE_NO_PREROLL);
  for (idx = 0; idx < num; idx++) {
    if (chs[idx].timestamp)
      EXPECT_EQ (read_mock_enable (dev0, chs[idx].name), pre_enabled[idx] ? 1 : 0)
          << chs[idx].name << " is not for the element to switch";
    else
      EXPECT_EQ (read_mock_enable (dev0, chs[idx].name), expected[idx] ? 1 : 0)
          << chs[idx].name;
  }
  EXPECT_EQ (feed_mock_scans (dev0, chs, num), 0);

  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  bus = gst_element_get_bus (pipeline);
  msg = gst_bus_timed_pop_filtered (bus, 10 * GST_SECOND,
      (GstMessageType) (GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  ASSERT_NE (msg, nullptr);
  EXPECT_EQ (GST_MESSAGE_TYPE (msg), GST_MESSAGE_EOS);
  gst_message_unref (msg);
  gst_object_unref (bus);
  EXPECT_EQ (gst_element_set_state (pipeline, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);

  /** the enabled state of the device is restored */
  for (idx = 0; idx < num; idx++) {
    EXPECT_EQ (read_mock_enable (dev0, chs[idx].name), pre_enabled[idx] ? 1 : 0)
        << chs[idx].name;
  }

  /** a buffer of MOCK_CAPACITY scans of the expected channels */
  EXPECT_EQ (sink_data.num_buffers, 1U);
  ASSERT_NE (sink_data.caps, nullptr);
  EXPECT_TRUE (gst_tensors_config_from_structure (
      &config, gst_caps_get_structure (sink_data.caps, 0)));
  EXPECT_EQ (config.info.num_tensors, 1U);
  EXPECT_EQ (config.info.info[0].type, _NNS_FLOAT32);
  EXPECT_EQ (config.info.info[0].dimension[0], num_expected);
  EXPECT_EQ (config.info.info[0].dimension[1], MOCK_CAPACITY);
  gst_tensors_config_free (&config);
  ASSERT_EQ (sink_data.data->len, sizeof (gfloat) * num_expected * MOCK_CAPACITY);

  values = (const gfloat *) sink_data.data->data;
  for (scan = 0; scan < MOCK_CAPACITY; scan++) {
    col = 0;
    for (idx = 0; idx < num; idx++) {
      if (chs[idx].timestamp || !expected[idx])
        continue;
      EXPECT_FLOAT_EQ (values[scan * num_expected + col], (gfloat) chs[idx].samples[scan])
          << chs[idx].name << " of scan " << scan;
      col++;
    }
  }

  gst_caps_unref (sink_data.caps);
  g_byte_array_unref (sink_data.data);
  gst_object_unref (sink);
  gst_object_unref (src_iio);
  gst_object_unref (pipeline);
  destroy_mock_device (dev0, chs, num);
}

/**
 * @brief the accelerometer with nothing enabled, as the kernel leaves it
 */
static const gboolean accel_none[] = { FALSE, FALSE, FALSE, FALSE };

/**
 * @brief the accelerometer with nothing but the timestamp enabled
 */
static const gboolean accel_ts[] = { FALSE, FALSE, FALSE, TRUE };

/**
 * @brief the x and z axes of the accelerometer
 */
static const gboolean accel_xz[] = { TRUE, FALSE, TRUE, FALSE };

/**
 * @brief the three axes of the accelerometer
 */
static const gboolean accel_xyz[] = { TRUE, TRUE, TRUE, FALSE };

/**
 * @brief tests a subset of the channels of an accelerometer
 */
TEST (testTensorSrcIio, accelChannelsCustom)
{
  const guint num = G_N_ELEMENTS (accel_channels);

  run_mock_device (accel_channels, num, accel_none, "0,2", NULL, 0, accel_xz);
  run_mock_device (accel_channels, num, accel_ts, "0,2", NULL, 0, accel_xz);
}

/**
 * @brief tests all the channels of an accelerometer
 * @note the timestamp, enabled or not, is neither switched nor put in the tensor
 */
TEST (testTensorSrcIio, accelChannelsAll)
{
  const guint num = G_N_ELEMENTS (accel_channels);

  run_mock_device (accel_channels, num, accel_none, " all ", NULL, 0, accel_xyz);
  run_mock_device (accel_channels, num, accel_ts, " all ", NULL, 0, accel_xyz);
}

/**
 * @brief tests a subset listing the index of the timestamp, which is ignored
 */
TEST (testTensorSrcIio, accelChannelsCustomTimestamp)
{
  const guint num = G_N_ELEMENTS (accel_channels);
  const gboolean x[] = { TRUE, FALSE, FALSE, FALSE };

  run_mock_device (accel_channels, num, accel_none, "0,3", NULL, 0, x);
  run_mock_device (accel_channels, num, accel_ts, "0,3", NULL, 0, x);
}

/**
 * @brief tests the automatic selection of the channels of an accelerometer
 */
TEST (testTensorSrcIio, accelChannelsAuto)
{
  const guint num = G_N_ELEMENTS (accel_channels);
  const gboolean y_ts[] = { FALSE, TRUE, FALSE, TRUE };
  const gboolean y[] = { FALSE, TRUE, FALSE, FALSE };

  /** an enabled timestamp does not count as an enabled channel */
  run_mock_device (accel_channels, num, accel_ts, NULL, NULL, 0, accel_xyz);
  /** the channels enabled already are kept as they are */
  run_mock_device (accel_channels, num, y_ts, NULL, NULL, 0, y);
}

/**
 * @brief tests the timestamp of an old kernel, named without the direction
 */
TEST (testTensorSrcIio, accelLegacyTimestamp)
{
  mock_channel chs[G_N_ELEMENTS (accel_channels)];

  memcpy (chs, accel_channels, sizeof (accel_channels));
  chs[3].name = "timestamp";
  run_mock_device (chs, G_N_ELEMENTS (chs), accel_ts, " all ", NULL, 0, accel_xyz);
}

/**
 * @brief tests that the refused channels values keep the previous selection,
 * which the device then runs with
 */
TEST (testTensorSrcIio, accelChannelsRefused_n)
{
  const gchar *refused[] = { "all,1", "0x1" };

  run_mock_device (accel_channels, G_N_ELEMENTS (accel_channels), accel_ts,
      "0,2", refused, G_N_ELEMENTS (refused), accel_xz);
}

/**
 * @brief tests a scan padded after its last channel to the largest channel
 */
TEST (testTensorSrcIio, scanPadding)
{
  const mock_channel chs[] = {
    { "in_voltage0", 0, TRUE, 16, 2, 0, FALSE, { -32768, 32767, -1, 1234 } },
    { "in_voltage1", 1, FALSE, 8, 1, 0, FALSE, { 255, 0, 128, 7 } },
  };
  const gboolean none[] = { FALSE, FALSE };
  const gboolean both[] = { TRUE, TRUE };

  run_mock_device (chs, G_N_ELEMENTS (chs), none, "all", NULL, 0, both);
}

/**
 * @brief start the accelerometer with the given timestamp type
 * @param[in] type contents of the type attribute, NULL to remove it
 * @param[in] ts_enabled whether the timestamp is enabled
 * @param[in] expected state change expected
 */
static void
run_accel_timestamp_type (const gchar *type, const gboolean ts_enabled,
    const GstStateChangeReturn expected)
{
  iio_dev_dir_struct *dev0;
  GstElement *src_iio;
  gchar *path;

  dev0 = make_mock_device (accel_channels, G_N_ELEMENTS (accel_channels),
      ts_enabled ? accel_ts : accel_none);
  ASSERT_NE (dev0, nullptr);

  path = mock_attr_path (dev0, "in_timestamp", "_type");
  if (type)
    EXPECT_EQ (write_file_string (path, type), 0);
  else
    EXPECT_EQ (safe_remove (path), 0);
  g_free (path);

  src_iio = gst_element_factory_make (ELEMENT_NAME, NULL);
  ASSERT_NE (src_iio, nullptr);
  g_object_set (src_iio, "iio-base-dir", dev0->iio_base_dir_sim, "dev-dir",
      dev0->dev_dir, "device", DEVICE_NAME, NULL);

  EXPECT_EQ (gst_element_set_state (src_iio, GST_STATE_PAUSED), expected);
  EXPECT_EQ (gst_element_set_state (src_iio, GST_STATE_NULL), GST_STATE_CHANGE_SUCCESS);
  gst_object_unref (src_iio);

  destroy_mock_device (dev0, accel_channels, G_N_ELEMENTS (accel_channels));
}

/**
 * @brief tests that the type of a disabled timestamp is not needed
 */
TEST (testTensorSrcIio, accelTimestampTypeDisabled)
{
  run_accel_timestamp_type ("le:s0/64>>0", FALSE, GST_STATE_CHANGE_NO_PREROLL);
  run_accel_timestamp_type (NULL, FALSE, GST_STATE_CHANGE_NO_PREROLL);
}

/**
 * @brief tests that an enabled timestamp whose size is unknown is refused
 */
TEST (testTensorSrcIio, accelTimestampType_n)
{
  run_accel_timestamp_type ("le:s0/64>>0", TRUE, GST_STATE_CHANGE_FAILURE);
  run_accel_timestamp_type (NULL, TRUE, GST_STATE_CHANGE_FAILURE);
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
