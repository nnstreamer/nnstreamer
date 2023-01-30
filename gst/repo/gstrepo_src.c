/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file	gstrepo_src.c
 * @date	31 January 2023
 * @brief	GStreamer plugin to read data using NN Frameworks
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyunil Park <hyunil46.park@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * ## Example launch line
 * |[
 * gst-launch-1.0 repo_src location=mnist_trainingSet.dat ! \
 * other/tensors, format=static, num_tensors=2, framerate=0/1, \
 * dimensions=1:1:784:1.1:1:10:1, types=float32.float32 ! tensor_sink
 * ]|
 * 
 * |[
 * gst-launch-1.0 repo_src location=mnist_trainingSet.dat ! \
 * application/octet-stream ! \
 * tensor_converter input-dim=1:1:784:1,1:1:10:1 input-type=float32,float32 ! \
 * tensor_sink
 * ]|
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif
#include <gst/gst.h>
#include <glib/gstdio.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <inttypes.h>
#include "nnstreamer_util.h"
#include "gstrepo_src.h"

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

GST_DEBUG_CATEGORY_STATIC (gst_repo_src_debug);
#define GST_CAT_DEFAULT gst_repo_src_debug

/* RepoSrc signals and args */
enum
{
  PROP_0,
  PROP_LOCATION
};

static void gst_repo_src_finalize (GObject * object);
static void gst_repo_src_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_repo_src_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static gboolean gst_repo_src_start (GstBaseSrc * basesrc);
static gboolean gst_repo_src_stop (GstBaseSrc * basesrc);
static GstFlowReturn gst_repo_src_create (GstPushSrc * pushsrc,
    GstBuffer ** buffer);

#define _do_init \
  GST_DEBUG_CATEGORY_INIT (gst_repo_src_debug, "repo_src", 0, "repo_src element");

#define gst_repo_src_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstRepoSrc, gst_repo_src, GST_TYPE_PUSH_SRC, _do_init);

/**
 * @brief initialize the repo_src's class
 */
static void
gst_repo_src_class_init (GstRepoSrcClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS (klass);
  GstPushSrcClass *gstpushsrc_class = GST_PUSH_SRC_CLASS (klass);

  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_repo_src_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_repo_src_get_property);

  g_object_class_install_property (gobject_class, PROP_LOCATION,
      g_param_spec_string ("location", "File Location",
          "Location of the file to read", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY));

  gobject_class->finalize = gst_repo_src_finalize;

  gst_element_class_set_static_metadata (gstelement_class,
      "Repo Source",
      "Source/File",
      "Read data using NN Frameworks", "Samsung Electronics Co., Ltd.");
  gst_element_class_add_static_pad_template (gstelement_class, &srctemplate);

  gstbasesrc_class->start = GST_DEBUG_FUNCPTR (gst_repo_src_start);
  gstbasesrc_class->stop = GST_DEBUG_FUNCPTR (gst_repo_src_stop);
  gstpushsrc_class->create = GST_DEBUG_FUNCPTR (gst_repo_src_create);

  if (sizeof (off_t) < 8) {
    GST_LOG ("No large file support, sizeof (off_t) = %" G_GSIZE_FORMAT "!",
        sizeof (off_t));
  }
}

/**
 * @brief Initialize repo_src
 */
static void
gst_repo_src_init (GstRepoSrc * src)
{
  src->filename = NULL;
  src->fd = 0;
  src->offset = 0;
  src->read_position = 0;

  /* for test */
  src->length = 3176;           /* Calculation is required using property, 3176 is MNIST size */
  //src->item_size[0] = 3136; /* Calculation is required using property */
  //src->item_size[1] = 40; /* Calculation is required using property */
  src->item_size[0] = 3176;
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_repo_src_finalize (GObject * object)
{
  GstRepoSrc *src = GST_REPO_SRC (object);

  g_free (src->filename);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Function to set file path.
 */
static gboolean
gst_repo_src_set_location (GstRepoSrc * src, const gchar * location,
    GError ** err)
{
  GstState state;

  /* the element must be stopped in order to do this */
  GST_OBJECT_LOCK (src);
  state = GST_STATE (src);
  if (state != GST_STATE_READY && state != GST_STATE_NULL)
    goto wrong_state;
  GST_OBJECT_UNLOCK (src);

  g_free (src->filename);

  /* clear the filename if we get a NULL */
  if (location == NULL) {
    src->filename = NULL;
  } else {
    /* we store the filename as received by the application. On Windows this
     * should be UTF8 */
    src->filename = g_strdup (location);
    GST_INFO ("filename : %s", src->filename);
  }
  g_object_notify (G_OBJECT (src), "location");

  return TRUE;

  /* ERROR */
wrong_state:
  {
    g_warning ("Changing the `location' property on repo_src when a file is "
        "open is not supported.");
    if (err)
      g_set_error (err, GST_URI_ERROR, GST_URI_ERROR_BAD_STATE,
          "Changing the `location' property on repo_src when a file is "
          "open is not supported.");
    GST_OBJECT_UNLOCK (src);
    return FALSE;
  }
}

/**
 * @brief Setter for repo_src properties.
 */
static void
gst_repo_src_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstRepoSrc *src;

  g_return_if_fail (GST_IS_REPO_SRC (object));

  src = GST_REPO_SRC (object);

  switch (prop_id) {
    case PROP_LOCATION:
      gst_repo_src_set_location (src, g_value_get_string (value), NULL);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter repo_src properties
 */
static void
gst_repo_src_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstRepoSrc *src;

  g_return_if_fail (GST_IS_REPO_SRC (object));

  src = GST_REPO_SRC (object);

  switch (prop_id) {
    case PROP_LOCATION:
      g_value_set_string (value, src->filename);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

#if 0
/**
 * @brief Function to read octet_stream
 */
static GstFlowReturn
gst_repo_src_read_octet_stream (GstRepoSrc * src, GstBuffer ** buffer)
{
  int i = 0;
  GstBuffer *buf;
  guint to_read, byte_read;
  int ret;
  guint8 *data;
  GstMemory *mem[MAX_ITEM] = { 0, };
  GstMapInfo info[MAX_ITEM];
  //guint length; //need to get property
  //guint64 offset;

  /* for test */
  mem[0] = gst_allocator_alloc (NULL, src->item_size[0], NULL);

  if (!gst_memory_map (mem[0], &info[0], GST_MAP_WRITE)) {
    GST_ERROR_OBJECT (src, "Could not map in_mem[%d] GstMemory", i);
    goto error;
  }

  data = info[0].data;

  byte_read = 0;
  to_read = src->length;
  while (to_read > 0) {
    GST_LOG_OBJECT (src, "Reading %d bytes at offset 0x%" G_GINT64_MODIFIER "x",
        to_read, src->offset + byte_read);
    errno = 0;
    ret = read (src->fd, data + byte_read, to_read);
    GST_LOG_OBJECT (src, "Read: %d", ret);
    if (ret < 0) {
      if (errno == EAGAIN || errno == EINTR)
        continue;
      goto could_not_read;
    }
    /* files should eos if they read 0 and more was requested */
    if (ret == 0) {
      /* .. but first we should return any remaining data */
      if (byte_read > 0)
        break;
      goto eos;
    }
    to_read -= ret;
    byte_read += ret;

    src->read_position += ret;
    src->offset += ret;
  }

  if (mem[0])
    gst_memory_unmap (mem[0], &info[0]);

  /* todo */
  /*if (bytes_read != length) */
  /* in case of media,if blocksize is smaller then frame size, need to check byte_read != length */
  /* alloc memory using byte_read, memocpy data to new memory, and append */

  buf = gst_buffer_new ();
  gst_buffer_append_memory (buf, mem[0]);

  *buffer = buf;
  return GST_FLOW_OK;

could_not_read:
  {
    GST_ELEMENT_ERROR (src, RESOURCE, READ, (NULL), GST_ERROR_SYSTEM);
    gst_memory_unmap (mem[0], &info[0]);
    return GST_FLOW_ERROR;
  }
eos:
  {
    GST_DEBUG ("EOS");
    gst_memory_unmap (mem[0], &info[0]);
    return GST_FLOW_EOS;
  }
error:
  return GST_FLOW_ERROR;
}
#endif
/**
 * @brief Function to read tensors
 */
static GstFlowReturn
gst_repo_src_read_tensors (GstRepoSrc * src, GstBuffer ** buffer)
{
  int i = 0;
  GstBuffer *buf;
  guint to_read, byte_read;
  int ret;
  guint8 *data;
  GstMemory *mem[MAX_ITEM] = { 0, };
  GstMapInfo info[MAX_ITEM];

  /* for MNIST test */
  src->item_size[0] = 3136;
  src->item_size[1] = 40;

  buf = gst_buffer_new ();

  for (i = 0; i < 2; i++) {
    mem[i] = gst_allocator_alloc (NULL, src->item_size[i], NULL);

    if (!gst_memory_map (mem[i], &info[i], GST_MAP_WRITE)) {
      GST_ERROR_OBJECT (src, "Could not map in_mem[%d] GstMemory", i);
      goto error;
    }

    data = info[i].data;

    byte_read = 0;
    to_read = src->item_size[i];
    while (to_read > 0) {
      GST_LOG_OBJECT (src,
          "Reading %d bytes at offset 0x%" G_GINT64_MODIFIER "x", to_read,
          src->offset + byte_read);
      errno = 0;
      ret = read (src->fd, data + byte_read, to_read);
      GST_LOG_OBJECT (src, "Read: %d", ret);
      if (ret < 0) {
        if (errno == EAGAIN || errno == EINTR)
          continue;
        goto could_not_read;
      }
      /* files should eos if they read 0 and more was requested */
      if (ret == 0) {
        /* .. but first we should return any remaining data */
        if (byte_read > 0)
          break;
        goto eos;
      }
      to_read -= ret;
      byte_read += ret;

      src->read_position += ret;
      src->offset += ret;
    }

    if (mem[i])
      gst_memory_unmap (mem[i], &info[i]);

    /* TODO */
    /*if (bytes_read != length) */
    /* in case of media,if blocksize is smaller then frame size, need to check byte_read != length */
    /* alloc memory using byte_read, memocpy data to new memory, and append */

    gst_buffer_append_memory (buf, mem[i]);
  }
  *buffer = buf;

  return GST_FLOW_OK;

could_not_read:
  {
    GST_ELEMENT_ERROR (src, RESOURCE, READ, (NULL), GST_ERROR_SYSTEM);
    gst_memory_unmap (mem[0], &info[0]);
    gst_buffer_unref (buf);
    return GST_FLOW_ERROR;
  }
eos:
  {
    GST_DEBUG ("EOS");
    gst_memory_unmap (mem[0], &info[0]);
    gst_buffer_unref (buf);
    return GST_FLOW_EOS;
  }
error:
  gst_buffer_unref (buf);
  return GST_FLOW_ERROR;
}

/**
 * @brief Function to create a buffer
 */
static GstFlowReturn
gst_repo_src_create (GstPushSrc * pushsrc, GstBuffer ** buffer)
{
  GstFlowReturn ret;
  GstRepoSrc *src;
  src = GST_REPO_SRC (pushsrc);

  //let's read data by property?
#if 0
  /*case application/octet-stream */
  ret = gst_repo_src_read_octet_stream (src, buffer);
#else
  ret = gst_repo_src_read_tensors (src, buffer);
#endif

  return ret;
}

/**
 * @brief Start repo_src, open the file
 */
static gboolean
gst_repo_src_start (GstBaseSrc * basesrc)
{
  struct_stat stat_results;
  GstRepoSrc *src = GST_REPO_SRC (basesrc);
  int flags = O_RDONLY | O_BINARY;

  if (src->filename == NULL || src->filename[0] == '\0')
    goto no_filename;

  GST_INFO_OBJECT (src, "opening file %s", src->filename);

  /* open the file */
  src->fd = g_open (src->filename, flags, 0);

  if (src->fd < 0)
    goto open_failed;

  /* check if it is a regular file, otherwise bail out */
  if (fstat (src->fd, &stat_results) < 0)
    goto no_stat;

  if (S_ISDIR (stat_results.st_mode))
    goto was_directory;

  if (S_ISSOCK (stat_results.st_mode))
    goto was_socket;

  /* record if it's a regular (hence seekable and lengthable) file */
  if (!S_ISREG (stat_results.st_mode))
    goto error_close;;

  src->read_position = 0;

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
  close (src->fd);
error_exit:
  return FALSE;
}

/**
 * @brief Stop repo_src, unmap and close the file
 */
static gboolean
gst_repo_src_stop (GstBaseSrc * basesrc)
{
  GstRepoSrc *src = GST_REPO_SRC (basesrc);

  /* close the file */
  g_close (src->fd, NULL);
  src->fd = 0;

  return TRUE;
}

/**
 * @brief register repo_src
 */
static gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, "repo_src", GST_RANK_NONE,
      GST_TYPE_REPO_SRC);
}

#ifndef PACKAGE
#define PACKAGE "repo_src"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    repo_src,
    "nnstreamer plugin library",
    plugin_init, VERSION, "LGPL", PACKAGE,
    "https://github.com/nnstreamer/nnstreamer")
