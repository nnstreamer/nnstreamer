/**
 * NNStreamer Common Header's Contents
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	tensor_common.c
 * @date	29 May 2018
 * @brief	Common data for NNStreamer, the GStreamer plugin for neural networks
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <tensor_common.h>
#include <string.h>
#include <glib.h>

/**
 * @brief String representations for each tensor element type.
 */
const gchar *tensor_element_typename[] = {
  [_NNS_INT32] = "int32",
  [_NNS_UINT32] = "uint32",
  [_NNS_INT16] = "int16",
  [_NNS_UINT16] = "uint16",
  [_NNS_INT8] = "int8",
  [_NNS_UINT8] = "uint8",
  [_NNS_FLOAT64] = "float64",
  [_NNS_FLOAT32] = "float32",
  [_NNS_END] = NULL,
};

/**
 * @brief Get media type from caps
 * @param caps caps to be interpreted
 * @return corresponding media type (returns _NNS_MEDIA_END for unsupported type)
 */
media_type
get_media_type_from_caps (const GstCaps * caps)
{
  GstStructure *structure;
  const gchar *name;

  structure = gst_caps_get_structure (caps, 0);
  name = gst_structure_get_name (structure);

  g_return_val_if_fail (name != NULL, _NNS_MEDIA_END);

  /** @todo Support other types */
  if (g_str_has_prefix (name, "video/")) {
    return _NNS_VIDEO;
  } else if (g_str_has_prefix (name, "audio/")) {
    return _NNS_AUDIO;
  }

  /** unknown, or not-supported type */
  return _NNS_MEDIA_END;
}

/**
 * @brief Get tensor_type from string tensor_type input
 * @return Corresponding tensor_type. _NNS_END if unrecognized value is there.
 * @param typestr The string type name, supposed to be one of tensor_element_typename[]
 */
tensor_type
get_tensor_type (const gchar * typestr)
{
  int len;

  if (!typestr)
    return _NNS_END;
  len = strlen (typestr);

  if (typestr[0] == 'u' || typestr[0] == 'U') {
    /**
     * Let's believe the developer and the following three letters are "int"
     * (case insensitive)
     */
    if (len == 6) {             /* uint16, uint32 */
      if (typestr[4] == '1' && typestr[5] == '6')
        return _NNS_UINT16;
      else if (typestr[4] == '3' && typestr[5] == '2')
        return _NNS_UINT32;
    } else if (len == 5) {      /* uint8 */
      if (typestr[4] == '8')
        return _NNS_UINT8;
    }
  } else if (typestr[0] == 'i' || typestr[0] == 'I') {
    /**
     * Let's believe the developer and the following two letters are "nt"
     * (case insensitive)
     */
    if (len == 5) {             /* int16, int32 */
      if (typestr[3] == '1' && typestr[4] == '6')
        return _NNS_INT16;
      else if (typestr[3] == '3' && typestr[4] == '2')
        return _NNS_INT32;
    } else if (len == 4) {      /* int8 */
      if (typestr[3] == '8')
        return _NNS_INT8;
    }
    return _NNS_END;
  } else if (typestr[0] == 'f' || typestr[0] == 'F') {
    /* Let's assume that the following 4 letters are "loat" */
    if (len == 7) {
      if (typestr[5] == '6' && typestr[6] == '4')
        return _NNS_FLOAT64;
      else if (typestr[5] == '3' && typestr[6] == '2')
        return _NNS_FLOAT32;
    }
  }

  return _NNS_END;
}

/**
 * @brief Find the index value of the given key string array
 * @return Corresponding index. Returns -1 if not found.
 * @param strv Null terminated array of gchar *
 * @param key The key string value
 */
int
find_key_strv (const gchar ** strv, const gchar * key)
{
  int cursor = 0;

  g_assert (strv != NULL);
  while (strv[cursor]) {
    if (!g_ascii_strcasecmp (strv[cursor], key))
      return cursor;
    cursor++;
  }

  return -1;                    /* Not Found */
}

/**
 * @brief Parse tensor dimension parameter string
 * @return The Rank. 0 if error.
 * @param param The parameter string in the format of d1:d2:d3:d4, d1:d2:d3, d1:d2, or d1, where dN is a positive integer and d1 is the innermost dimension; i.e., dim[d4][d3][d2][d1];
 */
int
get_tensor_dimension (const gchar * param, uint32_t dim[NNS_TENSOR_RANK_LIMIT])
{
  gchar **strv = g_strsplit (param, ":", NNS_TENSOR_RANK_LIMIT);
  int i, retval = 0;
  guint64 val;

  g_assert (strv != NULL);

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    if (strv[i] == NULL)
      break;
    val = g_ascii_strtoull (strv[i], NULL, 10);
    dim[i] = val;
    retval = i + 1;
  }
  for (; i < NNS_TENSOR_RANK_LIMIT; i++)
    dim[i] = 1;

  g_strfreev (strv);
  return retval;
}

/**
 * @brief Count the number of elemnts of a tensor
 * @return The number of elements. 0 if error.
 * @param dim The tensor dimension
 */
size_t
get_tensor_element_count (const uint32_t dim[NNS_TENSOR_RANK_LIMIT])
{
  size_t count = 1;
  int i;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    count *= dim[i];
  }

  return count;
}

/**
 * @brief Extract other/tensor dim/type from GstStructure
 */
GstTensor_Filter_CheckStatus
get_tensor_from_structure (const GstStructure * str, tensor_dim dim,
    tensor_type * type, int *framerate_num, int *framerate_denum)
{
  GstTensor_Filter_CheckStatus ret = _TFC_INIT;
  const gchar *strval;
  int rank;
  int j;
  gint fn, fd;

  if (!gst_structure_has_name (str, "other/tensor"))
    return ret;

  if (gst_structure_get_int (str, "dim1", (int *) &dim[0]) &&
      gst_structure_get_int (str, "dim2", (int *) &dim[1]) &&
      gst_structure_get_int (str, "dim3", (int *) &dim[2]) &&
      gst_structure_get_int (str, "dim4", (int *) &dim[3])) {
    ret |= _TFC_DIMENSION;
    if (gst_structure_get_int (str, "rank", &rank)) {
      for (j = rank; j < NNS_TENSOR_RANK_LIMIT; j++)
        g_assert (dim[j] == 1);
    }
  }
  strval = gst_structure_get_string (str, "type");
  if (strval) {
    *type = get_tensor_type (strval);
    g_assert (*type != _NNS_END);
    ret |= _TFC_TYPE;
  }
  if (gst_structure_get_fraction (str, "framerate", &fn, &fd)) {
    if (framerate_num)
      *framerate_num = fn;
    if (framerate_denum)
      *framerate_denum = fd;
    ret |= _TFC_FRAMERATE;
  }
  return ret;
}

/**
 * @brief internal static function to trim the front.
 */
static const gchar *
ftrim (const gchar * str)
{
  if (!str)
    return str;
  while (*str && (*str == ' ' || *str == '\t')) {
    str++;
  }
  return str;
}

/**
 * @brief Extract other/tensors dim/type from GstStructure
 */
int
get_tensors_from_structure (const GstStructure * str,
    GstTensor_TensorsMeta * meta, int *framerate_num, int *framerate_denom)
{
  int num = 0;
  int rank = 0;
  const gchar *strval;
  gint fn = 0, fd = 0;
  gchar **strv;
  int counter = 0;

  if (!gst_structure_has_name (str, "other/tensors"))
    return 0;

  if (gst_structure_get_int (str, "num_tensors", (int *) &num)) {
    if (num > 16 || num < 1)
      num = 0;
  }
  if (0 == num)
    return 0;

  meta->num_tensors = num;
  meta->dims = g_new (tensor_dim, num);
  meta->types = g_new (tensor_type, num);
  meta->ranks = (unsigned int *) g_malloc (sizeof (gint) * num);

  if (gst_structure_get_int (str, "rank", (int *) &rank)) {
    if (rank != NNS_TENSOR_RANK_LIMIT) {
      err_print ("rank value of other/tensors incorrect.\n");
      rank = 0;
    }
  }
  if (0 == rank)
    goto err_alloc;

  if (gst_structure_get_fraction (str, "framerate", &fn, &fd)) {
    if (framerate_num)
      *framerate_num = fn;
    if (framerate_denom)
      *framerate_denom = fd;
  }

  strval = gst_structure_get_string (str, "dimensions");
  strv = g_strsplit (strval, ",", -1);
  counter = 0;
  while (strv[counter]) {
    int ret;

    if (counter >= num) {
      err_print
          ("The number of dimensions does not match the number of tensors.\n");
      goto err_alloc;
    }
    ret = get_tensor_dimension (ftrim (strv[counter]), meta->dims[counter]);
    if (ret > NNS_TENSOR_RANK_LIMIT || ret < 1)
      goto err_alloc;
    counter++;
  }
  if (counter != num) {
    err_print
        ("The number of dimensions does not match the number of tensors.\n");
    goto err_alloc;
  }
  g_strfreev (strv);

  strval = gst_structure_get_string (str, "types");
  strv = g_strsplit (strval, ",", -1);
  counter = 0;
  while (strv[counter]) {
    if (counter >= num) {
      err_print ("The number of types does not match the number of tensors.\n");
      goto err_alloc;
    }
    meta->types[counter] = get_tensor_type (ftrim (strv[counter]));
    if (meta->types[counter] >= _NNS_END)
      goto err_alloc;
    counter++;
  }
  if (counter != num) {
    err_print ("The number of types does not match the number of tensors.\n");
    goto err_alloc;
  }
  g_strfreev (strv);
  return num;

err_alloc:
  meta->num_tensors = 0;
  g_free (meta->dims);
  meta->dims = NULL;
  g_free (meta->types);
  meta->types = NULL;
  g_free (meta->ranks);
  meta->ranks = NULL;
  return 0;
}

/**
 * @brief Get tensor dimension/type from GstCaps
 */
GstTensor_Filter_CheckStatus
get_tensor_from_padcap (const GstCaps * caps, tensor_dim dim,
    tensor_type * type, int *framerate_num, int *framerate_denum)
{
  GstTensor_Filter_CheckStatus ret = _TFC_INIT;
  unsigned int i, capsize;
  const GstStructure *str;
  gint fn = 0, fd = 0;

  g_assert (NNS_TENSOR_RANK_LIMIT == 4);        /* This code assumes rank limit is 4 */
  if (!caps)
    return ret;

  capsize = gst_caps_get_size (caps);
  for (i = 0; i < capsize; i++) {
    str = gst_caps_get_structure (caps, i);

    tensor_dim _dim;
    tensor_type _type = _NNS_END;
    int _fn, _fd;

    GstTensor_Filter_CheckStatus tmpret = get_tensor_from_structure (str,
        _dim, &_type, &_fn, &_fd);

    /**
     * Already cofnigured and more cap info is coming.
     * I'm not sure how this happens, but let's be ready for this.
     */
    if (tmpret & _TFC_DIMENSION) {
      if (ret & _TFC_DIMENSION) {
        g_assert (0 == memcmp (_dim, dim, sizeof (tensor_dim)));
      } else {
        memcpy (dim, _dim, sizeof (tensor_dim));
        ret |= _TFC_DIMENSION;
      }
    }

    if (tmpret & _TFC_TYPE) {
      if (ret & _TFC_TYPE) {
        g_assert (*type == _type);
      } else {
        *type = _type;
        ret |= _TFC_TYPE;
      }
    }

    if (tmpret & _TFC_FRAMERATE) {
      if (ret & _TFC_FRAMERATE) {
        g_assert (fn == _fn && fd == _fd);
      } else {
        fn = _fn;
        fd = _fd;
        if (framerate_num)
          *framerate_num = fn;
        if (framerate_denum)
          *framerate_denum = fd;
        ret |= _TFC_FRAMERATE;
      }
    }
  }
  return ret;
}

/**
 * @brief A callback for typefind, trying to find whether a file is other/tensors or not.
 * For the concrete definition of headers, please look at the wiki page of nnstreamer:
 * https://github.com/nnsuite/nnstreamer/wiki/Design-External-Save-Format-for-other-tensor-and-other-tensors-Stream-for-TypeFind
 */
void
gst_tensors_typefind_function (GstTypeFind * tf, gpointer pdata)
{
  const guint8 *data = gst_type_find_peek (tf, 0, 40);  /* The first 40 bytes are header-0 in v.1 protocol */
  const char formatstr[] = "TENSORST";
  const unsigned int *supported_version = (const unsigned int *) (&data[8]);
  const unsigned int *num_tensors = (const unsigned int *) (&data[12]);
  if (data &&
      memcmp (data, formatstr, 8) == 0 &&
      *supported_version == 1 && *num_tensors <= 16 && *num_tensors >= 1) {
    gst_type_find_suggest (tf, GST_TYPE_FIND_MAXIMUM,
        gst_caps_new_simple ("other/tensorsave", NULL, NULL));
  }
}
