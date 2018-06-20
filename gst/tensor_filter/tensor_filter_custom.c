/**
 * GStreamer Tensor_Filter, Tensorflow-Lite Module
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
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
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * @file	tensor_filter_custom.c
 * @date	01 Jun 2018
 * @brief	Custom tensor post-processing interface for NNStreamer suite between NN developer-plugins and NNstreamer.
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This is the per-NN-framework plugin (custom) for tensor_filter.
 * Fill in "GstTensor_Filter_Framework" for tensor_filter.h/c
 *
 */

#include "tensor_filter.h"
#include "tensor_filter_custom.h"
#include <glib.h>
#include <dlfcn.h>

struct _internal_data
{
  GstTensor_Filter *parent;

  void *handle;
  NNStreamer_custom_class *methods;

  void *customFW_private_data;
};
typedef struct _internal_data internal_data;

/**
 * @brief Load the custom library. Will skip loading if it's already loaded.
 * @return 0 if successfully loaded. 1 if skipped (already loaded). -1 if error
 */
static int
custom_loadlib (const GstTensor_Filter * filter, void **private_data)
{
  internal_data *ptr;
  char *dlsym_error;

  if (filter->privateData != NULL) {
    /* @TODO : Check the integrity of filter->data and filter->modelFilename, nnfw */
    return 1;
  }

  ptr = g_new0 (internal_data, 1);      /* Fill Zero! */
  *private_data = ptr;
  g_assert (filter->privateData == ptr);
  ptr->parent = GstTensor_Filter_of_privateData (private_data);

  /* Load .so if this is the first time for this instance. */
  ptr->handle = dlopen (filter->prop.modelFilename, RTLD_NOW);
  if (!ptr->handle) {
    g_free (ptr);
    *private_data = NULL;
    return -1;
  }

  dlerror ();
  ptr->methods =
      *((NNStreamer_custom_class **) dlsym (ptr->handle, "NNStreamer_custom"));
  dlsym_error = dlerror ();
  if (dlsym_error) {
    err_print ("tensor_filter_custom:loadlib error: %s\n", dlsym_error);
    dlclose (ptr->handle);
    g_free (ptr);
    *private_data = NULL;
    return -1;
  }

  g_assert (ptr->methods->initfunc);
  ptr->customFW_private_data = ptr->methods->initfunc (&(filter->prop));
  return 0;
}

/**
 * @brief The mandatory callback for GstTensor_Filter_Framework
 * @param filter The parent object
 * @param[in] inptr The input tensor
 * @param[out] outptr The output tensor
 */
static int
custom_invoke (const GstTensor_Filter * filter, void **private_data,
    const uint8_t * inptr, uint8_t * outptr)
{
  int retval = custom_loadlib (filter, private_data);
  internal_data *ptr;

  /* Actually, tensor_filter must have called getInput/OotputDim first. */
  g_assert (retval != 0);

  if (retval < 0)
    return retval;

  g_assert (filter->privateData && *private_data == filter->privateData);
  ptr = *private_data;

  return ptr->methods->invoke (ptr->customFW_private_data, &(filter->prop),
      inptr, outptr);
}

/**
 * @brief The optional callback for GstTensor_Filter_Framework
 */
static int
custom_getInputDim (const GstTensor_Filter * filter, void **private_data,
    tensor_dim inputDimension, tensor_type * type)
{
  int retval = custom_loadlib (filter, private_data);
  internal_data *ptr;

  if (retval < 0)
    return retval;

  g_assert (filter->privateData && *private_data == filter->privateData);
  ptr = *private_data;

  return ptr->methods->getInputDim (ptr->customFW_private_data, &(filter->prop),
      inputDimension, type);
}

/**
 * @brief The optional callback for GstTensor_Filter_Framework
 */
static int
custom_getOutputDim (const GstTensor_Filter * filter, void **private_data,
    tensor_dim outputDimension, tensor_type * type)
{
  int retval = custom_loadlib (filter, private_data);
  internal_data *ptr;

  if (retval < 0)
    return retval;

  g_assert (filter->privateData && *private_data == filter->privateData);
  ptr = *private_data;

  return ptr->methods->getOutputDim (ptr->customFW_private_data,
      &(filter->prop), outputDimension, type);
}

/**
 * @brief Free privateData and move on.
 */
static void
custom_close (const GstTensor_Filter * filter, void **private_data)
{
  internal_data *ptr = *private_data;

  ptr->methods->exitfunc (ptr->customFW_private_data, &(filter->prop));
  g_free (ptr);
  *private_data = NULL;
  g_assert (filter->privateData == NULL);
}

GstTensor_Filter_Framework NNS_support_custom = {
  .name = "custom",
  .allow_in_place = FALSE,      /* custom cannot support in-place (outptr == inptr). */
  .invoke_NN = custom_invoke,
  .getInputDimension = custom_getInputDim,
  .getOutputDimension = custom_getOutputDim,
  .close = custom_close,
};
