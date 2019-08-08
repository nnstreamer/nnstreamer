/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All rights reserved.
 * Copyright (C) 2019 Dongju Chae <dongju.chae@samsung.com>
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
 * @file   tensor_filter_python_core.h
 * @author Dongju Chae <dongju.chae@samsung.com>
 * @date   03/29/2019
 * @brief  connection with python libraries.
 *
 * @bug     No known bugs.
 */
#ifndef TENSOR_FILTER_PYTHON_CORE_H
#define TENSOR_FILTER_PYTHON_CORE_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <glib.h>
#include <gst/gst.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>
#include <structmember.h>

#include "nnstreamer_plugin_api_filter.h"

#define Py_ERRMSG(...) do {PyErr_Print(); g_critical (__VA_ARGS__);} while (0);

/** @brief Callback type for custom filter */
typedef enum _cb_type
{
  CB_SETDIM = 0,
  CB_GETDIM,

  CB_END,
} cb_type;

#ifdef __cplusplus
#include <vector>
#include <map>

/**
 * @brief	Python embedding core structure
 */
class PYCore
{
public:
  /**
   * member functions.
   */
  PYCore (const char* _script_path, const char* _custom);
  ~PYCore ();

  int init (const GstTensorFilterProperties * prop);
  int loadScript ();
  const char* getScriptPath();
  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int setInputTensorDim (const GstTensorsInfo * in_info, GstTensorsInfo * out_info);
  int run (const GstTensorMemory * input, GstTensorMemory * output);

  int parseOutputTensors(PyObject* result, GstTensorsInfo * info);

  /** @brief Return callback type */
  cb_type getCbType () { return callback_type; }
  /** @brief Lock python-related actions */
  void Py_LOCK() { pthread_mutex_lock(&py_mutex); }
  /** @brief Unlock python-related actions */
  void Py_UNLOCK() { pthread_mutex_unlock(&py_mutex); }

  PyObject* PyTensorShape_New (const GstTensorInfo *info);

  int checkTensorType (GstTensorMemory *output, PyArrayObject *array);
  int checkTensorSize (GstTensorMemory *output, PyArrayObject *array);

  tensor_type getTensorType (NPY_TYPES npyType);
  NPY_TYPES getNumpyType (tensor_type tType);

  static std::map <void*, PyArrayObject*> outputArrayMap;
private:

  const std::string script_path;  /**< from model_path property */
  const std::string module_args;  /**< from custom property */

  std::string module_name;

  cb_type callback_type;

  PyObject* core_obj;
  PyObject* shape_cls;
  pthread_mutex_t py_mutex;

  GstTensorsInfo inputTensorMeta;  /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta;  /**< The tensor info of output tensors */

  bool configured; /**< True if the script is successfully loaded */
  void *handle; /**< returned handle by dlopen() */
};

/**
 * @brief	the definition of functions to be used at C files.
 */
extern "C"
{
#endif
  void *py_core_new (const char *_script_path, const char *_custom_path);
  void py_core_delete (void * py);
  int py_core_init (void * py, const GstTensorFilterProperties * prop);
  const char *py_core_getScriptPath (void * py);
  int py_core_getInputDim (void * py, GstTensorsInfo * info);
  int py_core_getOutputDim (void * py, GstTensorsInfo * info);
  int py_core_setInputDim (void * py, const GstTensorsInfo * in_info, GstTensorsInfo * out_info);
  int py_core_run (void * py, const GstTensorMemory * input,
      GstTensorMemory * output);
  void py_core_destroyNotify (void * data);
  cb_type py_core_getCbType (void * py);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_FILTER_PYTHON_CORE_H */
