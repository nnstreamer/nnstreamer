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
 * @brief Get tensor dimension/type from GstCaps
 */
GstTensor_Filter_CheckStatus
get_tensor_from_padcap (const GstCaps * caps, tensor_dim dim,
    tensor_type * type, int *framerate_num, int *framerate_denum)
{
  GstTensor_Filter_CheckStatus ret = _TFC_INIT;
  tensor_dim backup_dim;
  tensor_type backup_type = _NNS_END;
  unsigned int i, capsize;
  const GstStructure *str;
  int rank;
  const gchar *strval;
  gint fn, fd;

  if (!caps)
    return ret;

  capsize = gst_caps_get_size (caps);
  for (i = 0; i < capsize; i++) {
    str = gst_caps_get_structure (caps, i);
    g_assert (NNS_TENSOR_RANK_LIMIT == 4);      /* This code assumes rank limit is 4 */

    /**
     * Already cofnigured and more cap info is coming.
     * I'm not sure how this happens, but let's be ready for this.
     */
    if ((i > 1) && (ret & _TFC_DIMENSION)) {
      memcpy (backup_dim, dim, sizeof (uint32_t) * NNS_TENSOR_RANK_LIMIT);
    }
    if ((i > 1) && (ret & _TFC_TYPE)) {
      backup_type = *type;
    }

    if (gst_structure_get_int (str, "dim1", (int *) &dim[0]) &&
        gst_structure_get_int (str, "dim2", (int *) &dim[1]) &&
        gst_structure_get_int (str, "dim3", (int *) &dim[2]) &&
        gst_structure_get_int (str, "dim4", (int *) &dim[3])) {
      int j;

      if (ret & _TFC_DIMENSION) {
        /* Already configured by previous "cap"? */
        for (j = 0; j < NNS_TENSOR_RANK_LIMIT; j++)
          g_assert (dim[j] == backup_dim[j]);
      }
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

      if (ret & _TFC_TYPE) {
        /* Already configured by previous "cap"? */
        g_assert (*type == backup_type);
      }
      ret |= _TFC_TYPE;
    }
    if (gst_structure_get_fraction (str, "framerate", &fn, &fd)) {
      if ((ret & _TFC_FRAMERATE) && framerate_num)
        g_assert (fn == *framerate_num);
      if ((ret & _TFC_FRAMERATE) && framerate_denum)
        g_assert (fd == *framerate_denum);

      if (framerate_num)
        *framerate_num = fn;
      if (framerate_denum)
        *framerate_denum = fd;
      ret |= _TFC_FRAMERATE;
    }
  }
  return ret;
}
