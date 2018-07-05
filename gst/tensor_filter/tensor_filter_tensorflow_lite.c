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
 * @file	tensor_filter_tensorflow_lite.c
 * @date	24 May 2018
 * @brief	Tensorflow-lite module for tensor_filter gstreamer plugin
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This is the per-NN-framework plugin (tensorflow-lite) for tensor_filter.
 * Fill in "GstTensor_Filter_Framework" for tensor_filter.h/c
 *
 */

#include "tensor_filter.h"
#include <glib.h>

/**
 * @brief Load tensorflow lite modelfile
 * @return 0 if successfully loaded. 1 if skipped (already loaded). -1 if error
 */
static int
tflite_loadModelFile (const GstTensor_Filter * filter, void **private_data)
{
  /* @TODO need to decide wheter make internal data structure or not
   * need to add skip logic when model file already loaded
   * need to load tensorflow lite model file by FlatBufferModel::BuildFromFile
   * after configuration of c->cpp api of tflite works done
   */

  /* @TODO call tflite core api "tflite_new"  */
  return 0;
}

/**
 * @brief The open callback for GstTensor_Filter_Framework. Called before anything else
 */
static void
tflite_open (const GstTensor_Filter * filter, void **private_data)
{
  int retval = tflite_loadModelFile (filter, private_data);

  g_assert (retval == 0);       /* This must be called only once */
}

/**
 * @brief The mandatory callback for GstTensor_Filter_Framework
 */
static uint8_t *
tflite_invoke (const GstTensor_Filter * filter, void **private_data,
    const uint8_t * inptr, uint8_t * outptr)
{
  /* @TODO fill in *outputDimension (uint32_t[MAX_RANK]), *type */
  /* @TODO call tflite core apis */

  return outptr;                /* NYI */
}

/**
 * @brief The optional callback for GstTensor_Filter_Framework
 */
static int
tflite_getInputDim (const GstTensor_Filter * filter, void **private_data,
    tensor_dim inputDimension, tensor_type * type)
{
  /* @TODO fill in *inputDimension (uint32_t[MAX_RANK]), *type */
  /* @TODO call tflite core api "tflite_getInputDim" */

  return 0;                     /* NYI */
}

/**
 * @brief The optional callback for GstTensor_Filter_Framework
 */
static int
tflite_getOutputDim (const GstTensor_Filter * filter, void **private_data,
    tensor_dim outputDimension, tensor_type * type)
{
  /* @TODO fill in *outputDimension (uint32_t[MAX_RANK]), *type */
  /* @TODO call tflite core api "tflite_getOutputDim" */

  return 0;                     /* NYI */
}

/**
 * @brief The set-input-dim callback for GstTensor_Filter_Framework
 */
static int
tflite_setInputDim (const GstTensor_Filter * filter, void **private_data,
    const tensor_dim iDimension, const tensor_type iType,
    tensor_dim oDimension, tensor_type * oType)
{
  /* @TODO call tflite core apis */
  return 0;                     /* NYI */
}

/**
 * @brief Free privateData and move on.
 */
static void
tflite_close (const GstTensor_Filter * filter, void **private_data)
{

  /* @TODO call tflite core api "tflite_delete" */
}

GstTensor_Filter_Framework NNS_support_tensorflow_lite = {
  .name = "tensorflow-lite",
  .allow_in_place = FALSE,      /* Let's not do this yet. @TODO: support this to optimize performance later. */
  .allocate_in_invoke = FALSE,  /* TFLite may need to use TRUE in the future. However, it is not supported, yet */
  .invoke_NN = tflite_invoke,
  .getInputDimension = tflite_getInputDim,
  .getOutputDimension = tflite_getOutputDim,
  .setInputDimension = tflite_setInputDim,
  .open = tflite_open,
  .close = tflite_close,
};
