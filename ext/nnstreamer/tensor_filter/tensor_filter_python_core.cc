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
 * @file   tensor_filter_python_core.cc
 * @author Dongju Chae <dongju.chae@samsung.com>
 * @date   03/29/2019
 * @brief  connection with python libraries.
 * @bug     No known bugs.
 */

#include "tensor_filter_python_core.h"
#include <nnstreamer_plugin_api.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

std::map <void*, PyArrayObject*> PYCore::outputArrayMap;

/**
 * @brief	PYCore creator
 * @param	_script_path	: the logical path to '{script_name}.py' file
 * @note	the script of _script_path will be loaded simultaneously
 * @return	Nothing
 */
PYCore::PYCore (const char* _script_path, const char* _custom)
  : script_path(_script_path), module_args(_custom)
{
  /**
   * To fix import error of python extension modules
   * (e.g., multiarray.x86_64-linux-gnu.so: undefined symbol: PyExc_SystemError)
   */
  gchar libname[32];

  g_snprintf (libname, sizeof(libname), 
#if PY_VERSION_HEX >= 0x03000000 
              "libpython%d.%dm.so.1.0",
#else
              "libpython%d.%d.so.1.0",
#endif
              PY_MAJOR_VERSION, PY_MINOR_VERSION);

  handle = dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);
  g_assert(handle);

  Py_Initialize();
  g_assert(Py_IsInitialized());

  _import_array();  /** for numpy */

  /**
   * Parse script path to get module name
   * The module name should drop its extension (i.e., .py)
   */
  module_name = script_path;

  const size_t last_idx = module_name.find_last_of("/\\");
  if (last_idx != std::string::npos)
    module_name.erase(0, last_idx + 1);

  const size_t ext_idx = module_name.rfind('.');
  if (ext_idx != std::string::npos)
    module_name.erase(ext_idx);

  /** Add current/directory path to sys.path */
  PyObject *sys_module = PyImport_ImportModule("sys");
  PyObject *sys_path = PyObject_GetAttrString(sys_module, "path");

  PyList_Append(sys_path, PyUnicode_FromString("."));
  PyList_Append(sys_path, PyUnicode_FromString(script_path.substr(0, last_idx).c_str()));

  Py_XDECREF(sys_path);
  Py_XDECREF(sys_module);

  /** Find nnstreamer_api module */
  PyObject *api_module = PyImport_ImportModule("nnstreamer_python");
  g_assert(api_module);
  shape_cls = PyObject_GetAttrString(api_module, "TensorShape"); 
  g_assert(shape_cls);
  Py_XDECREF(api_module);

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);

  callback_type = CB_END;
  core_obj = NULL;
  configured = false;

  /** to prevent concurrent Python C-API calls */
  pthread_mutex_init(&py_mutex, NULL);
}

/**
 * @brief	PYCore Destructor
 * @return	Nothing
 */
PYCore::~PYCore ()
{
  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);

  if (core_obj)
    Py_XDECREF(core_obj);
  if (shape_cls)
    Py_XDECREF(shape_cls);

  PyErr_Clear();
  Py_Finalize();

  dlclose(handle);
}

/**
 * @brief	initialize the object with python script
 * @return 0 if OK. non-zero if error.
 */
int
PYCore::init (const GstTensorFilterProperties * prop)
{
  gst_tensors_info_copy (&inputTensorMeta, &prop->input_meta);
  gst_tensors_info_copy (&outputTensorMeta, &prop->output_meta);

  return loadScript ();
}

/**
 * @brief	get the script path
 * @return the script path.
 */
const char *
PYCore::getScriptPath ()
{
  return script_path.c_str();
}

/**
 * @brief	load the py script
 * @note	the script will be loaded
 * @return 0 if OK. non-zero if error.
 *        -1 if the py file is not loaded.
 *        -2 if the target class is not found.
 *        -3 if the class is not created.
 */
int
PYCore::loadScript ()
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  PyObject *module = PyImport_ImportModule(module_name.c_str());
  if (module) {
    PyObject *cls = PyObject_GetAttrString(module, "CustomFilter");
    if (cls) {
      PyObject *py_args;
      if (!module_args.empty()) {
        gchar **g_args = g_strsplit(module_args.c_str(), " ", 0);
        char **args = g_args;
        int argc = 0;
        while (*(args++) != NULL) argc++;

        g_assert(argc > 0);
          
        py_args = PyTuple_New(argc);

        for (int i = 0; i < argc; i++)
          PyTuple_SetItem(py_args, i, PyUnicode_FromString(g_args[i]));

        core_obj = PyObject_CallObject(cls, py_args);

        Py_XDECREF(py_args);
        g_strfreev(g_args);
      } else 
        core_obj = PyObject_CallObject(cls, NULL);

      if (core_obj) {
        /** check whther either setInputDim or getInputDim/getOutputDim are defined */
        if (PyObject_HasAttrString(core_obj, (char*) "setInputDim"))
          callback_type = CB_SETDIM;
        else if (PyObject_HasAttrString(core_obj, (char*) "getInputDim") && 
                 PyObject_HasAttrString(core_obj, (char*) "getOutputDim"))
          callback_type = CB_GETDIM;
        else
          callback_type = CB_END;
      } else {
        Py_ERRMSG ("Fail to create an instance 'CustomFilter'\n");
        return -3;
      }

      Py_XDECREF(cls);
    } else {
      Py_ERRMSG ("Cannot find 'CustomFilter' class in the script\n");
      return -2;
    }

    Py_XDECREF(module);
  } else {
    Py_ERRMSG ("the script is not properly loaded\n");
    return -1;
  }

  configured = true;

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Script is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  
  return 0;
}

/**
 * @brief	check the data type of tensors in array
 * @param output : tensor memory for output tensors
 * @param array  : python array
 * @return a boolean value for whether the types are matched
 */
int
PYCore::checkTensorType (GstTensorMemory *output, PyArrayObject *array)
{
  g_assert (output);
  g_assert (array);

  int nns_type = output->type;
  int np_type = PyArray_TYPE(array);

  switch (nns_type) {
    case _NNS_INT64: return np_type == NPY_INT64;
    case _NNS_UINT64: return np_type == NPY_UINT64;
    case _NNS_INT32: return np_type == NPY_INT32;
    case _NNS_UINT32: return np_type == NPY_UINT32;
    case _NNS_INT16: return np_type == NPY_INT16;
    case _NNS_UINT16: return np_type == NPY_UINT16;
    case _NNS_INT8: return np_type == NPY_INT8;
    case _NNS_UINT8: return np_type == NPY_UINT8;
    case _NNS_FLOAT64: return np_type == NPY_FLOAT64;
    case _NNS_FLOAT32: return np_type == NPY_FLOAT32;
  }

  return 0;
}

/**
 * @brief	check the data size of tensors in array
 * @param output : tensor memory for output tensors
 * @param array  : python array
 * @return a boolean value for whether the sizes are matched
 */
int
PYCore::checkTensorSize (GstTensorMemory *output, PyArrayObject *array)
{
  g_assert (output);
  g_assert (array);

  size_t total_size = PyArray_ITEMSIZE(array);

  for (int i = 0; i < PyArray_NDIM(array); i++) 
    total_size *= PyArray_DIM(array, i);

  return (output->size == total_size);
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
PYCore::getInputTensorDim (GstTensorsInfo * info)
{
  int res = 0;

  g_assert (info);

  Py_LOCK();

  PyObject *result = PyObject_CallMethod(core_obj, (char*) "getInputDim", NULL);
  if (result) {
    parseOutputTensors(result, info);
    Py_XDECREF(result);
  } else { 
    Py_ERRMSG("Fail to call 'getInputDim'");
    res = -1;
  }

  Py_UNLOCK();

  return res;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
PYCore::getOutputTensorDim (GstTensorsInfo * info)
{
  int res = 0;
  
  g_assert (info);

  Py_LOCK();

  PyObject *result = PyObject_CallMethod(core_obj, (char*) "getOutputDim", NULL);
  if (result) {
    parseOutputTensors(result, info);
    Py_XDECREF(result);
  } else { 
    Py_ERRMSG("Fail to call 'getOutputDim'");
    res = -1;
  }

  Py_UNLOCK();

  return res;
}

/**
 * @brief	set the Dimension of Input Tensor and return the Dimension of Input Tensor.
 * @param[in] info Structure for input tensor info.
 * @param[out] info Structure for output tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
PYCore::setInputTensorDim (const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  int res = 0;

  g_assert (in_info);
  g_assert (out_info);

  Py_LOCK();

  /** to Python list object */
  PyObject *param = PyList_New(0);
  g_assert (param);

  inputTensorMeta.num_tensors = in_info->num_tensors;
  for (int i = 0; i < in_info->num_tensors; i++) {
    PyObject *shape = PyTensorShape_New (&in_info->info[i]);
    assert (shape);
    
    PyList_Append(param, shape);
  }

  PyObject *result = PyObject_CallMethod(
      core_obj, (char*) "setInputDim", (char*) "(O)", param);

  /** dereference input param */
  for (int i = 0; i < in_info->num_tensors; i++) {
    PyObject *shape = PyList_GetItem(param, (Py_ssize_t) i);
    Py_XDECREF (shape);
  }
  Py_XDECREF (param);

  if (result) {
    parseOutputTensors(result, out_info);
    outputTensorMeta.num_tensors = out_info->num_tensors;
    Py_XDECREF(result);
  } else { 
    Py_ERRMSG("Fail to call 'setInputDim'");
    res = -1;
  }

  Py_UNLOCK();

  return res;
}

/**
 * @brief	allocate TensorShape object
 * @param info : the tensor info
 * @return created object
 */
PyObject* 
PYCore::PyTensorShape_New (const GstTensorInfo* info)
{
  PyObject *args = PyTuple_New(2);
  PyObject *dims = PyList_New(0);
  PyObject *type = (PyObject*) PyArray_DescrFromType(getNumpyType(info->type));

  g_assert(args);
  g_assert(dims);
  g_assert(type);

  for (int i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) 
    PyList_Append(dims, PyLong_FromLong((uint64_t) info->dimension[i]));

  PyTuple_SetItem(args, 0, dims);
  PyTuple_SetItem(args, 1, type);
 
  PyObject *obj = PyObject_CallObject(shape_cls, args);
  g_assert(obj);

  return obj;
}

/**
 * @brief parse the invoke result to feed output tensors
 * @param[result] Python object retunred by invoke
 * @param[info] info Structure for output tensor info
 * @return None
 */
void
PYCore::parseOutputTensors(PyObject* result, GstTensorsInfo * info)
{
  info->num_tensors = PyList_Size(result);
  for (int i = 0; i < info->num_tensors; i++) {
    /** don't own the reference */
    PyObject *tensor_shape = PyList_GetItem(result, (Py_ssize_t) i); 
    g_assert(tensor_shape);

    PyObject *shape_dims = PyObject_CallMethod(tensor_shape, (char*) "getDims", NULL);
    g_assert(shape_dims);

    PyObject *shape_type = PyObject_CallMethod(tensor_shape, (char*) "getType", NULL);
    g_assert(shape_type);

    /** convert numpy type to tensor type */
    info->info[i].type = getTensorType((NPY_TYPES)(((PyArray_Descr*) shape_type)->type_num));
    for (int j = 0; j < PyList_Size(shape_dims); j++)
      info->info[i].dimension[j] = 
        (uint32_t) PyLong_AsLong(PyList_GetItem(shape_dims, (Py_ssize_t) j));

    Py_XDECREF (shape_dims);
    Py_XDECREF (shape_type);
  }
}

/**
 * @brief	run the script with the input.
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 *        -1 if the script does not work properly.
 *        -2 if the output properties are different with script.
 */
int
PYCore::run (const GstTensorMemory * input, GstTensorMemory * output)
{
  int res = 0;

#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  g_assert(input);
  g_assert(output);

  Py_LOCK();

  PyObject *param = PyList_New(0);
  for (int i = 0; i < inputTensorMeta.num_tensors; i++) {
    /** create a Numpy array wrapper (1-D) for NNS tensor data */
    npy_intp input_dims[] = {(npy_intp) input[i].size};
    PyObject *input_array = PyArray_SimpleNewFromData(
        1, input_dims, getNumpyType(input[i].type), input[i].data);
    PyList_Append(param, input_array);
  }

  PyObject *result = PyObject_CallMethod(core_obj, (char*) "invoke", (char*) "(O)", param);
  if (result) {
    g_assert(PyList_Size(result) == outputTensorMeta.num_tensors);

    for (int i = 0; i < outputTensorMeta.num_tensors; i++) {
      PyArrayObject* output_array = (PyArrayObject*) PyList_GetItem(result, (Py_ssize_t) i);
      /** type/size checking */
      if (checkTensorType(&output[i], output_array) && 
          checkTensorSize(&output[i], output_array)) { 
        /** obtain the pointer to the buffer for the output array */
        output[i].data = PyArray_DATA(output_array);
        Py_XINCREF(output_array);
        outputArrayMap.insert (std::make_pair (output[i].data, output_array));
      } else {
        g_critical ("Output tensor type/size is not matched\n");
        res = -2;
        break;
      }
    }

    Py_XDECREF(result);
  } else {
    Py_ERRMSG("Fail to call 'invoke'");
    res = -1;
  }

  /** dereference input param */
  for (int i = 0; i < inputTensorMeta.num_tensors; i++) {
    PyObject *input_array = PyList_GetItem(param, (Py_ssize_t) i);
    Py_XDECREF (input_array);
  }
  Py_XDECREF (param);

  Py_UNLOCK();

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Invoke() is finished: %" G_GINT64_FORMAT,
      (stop_time - start_time));
#endif

  return res;
}

/**
 * @brief	return the data type of the tensor
 * @param npyType	: the defined type of Python numpy
 * @return the enum of defined _NNS_TYPE
 */
tensor_type
PYCore::getTensorType (NPY_TYPES npyType)
{
  switch (npyType) {
    case NPY_INT32:
      return _NNS_INT32;
    case NPY_UINT32:
      return _NNS_UINT32;
    case NPY_INT16:
      return _NNS_INT16;
    case NPY_UINT16:
      return _NNS_UINT16;
    case NPY_INT8:
      return _NNS_INT8;
    case NPY_UINT8:
      return _NNS_UINT8;
    case NPY_INT64:
      return _NNS_INT64;
    case NPY_UINT64:
      return _NNS_UINT64;
    case NPY_FLOAT32:
      return _NNS_FLOAT32;
    case NPY_FLOAT64:
      return _NNS_FLOAT64;
    default:
      /** @todo Support other types */
      break;
  }

  return _NNS_END;
}

/**
 * @brief	return the data type of the tensor for Python numpy
 * @param tType	: the defined type of NNStreamer
 * @return the enum of defined numpy datatypes
 */
NPY_TYPES
PYCore::getNumpyType (tensor_type tType)
{
  switch (tType) {
    case _NNS_INT32:
      return NPY_INT32;
    case _NNS_UINT32:
      return NPY_UINT32;
    case _NNS_INT16:
      return NPY_INT16;
    case _NNS_UINT16:
      return NPY_UINT16;
    case _NNS_INT8:
      return NPY_INT8;
    case _NNS_UINT8:
      return NPY_UINT8;
    case _NNS_INT64:
      return NPY_INT64;
    case _NNS_UINT64:
      return NPY_UINT64;
    case _NNS_FLOAT32:
      return NPY_FLOAT32;
    case _NNS_FLOAT64:
      return NPY_FLOAT64;
    default:
      /** @todo Support other types */
      break;
  }
  return NPY_NOTYPE;
}

/**
 * @brief allocate new PYCore class.
 * @param script_path : python script path
 * @return  Nothing
 */
void *
py_core_new (const char * _script_path, const char * _custom)
{
  return new PYCore (_script_path, _custom != NULL ? _custom : "");
}

/**
 * @brief	delete the PYCore class.
 * @param	py	: the class object
 * @return	Nothing
 */
void
py_core_delete (void * py)
{
  PYCore *c = (PYCore *) py;
  delete c;
}

/**
 * @brief	initialize the object with py script
 * @param	py	: the class object
 * @return 0 if OK. non-zero if error.
 */
int
py_core_init (void * py, const GstTensorFilterProperties * prop)
{
  PYCore *c = (PYCore *) py;
  return c->init (prop);
}

/**
 * @brief	get script path
 * @param	py	: the class object
 * @return	script path
 */
const char *
py_core_getScriptPath (void * py)
{
  PYCore *c = (PYCore *) py;
  return c->getScriptPath ();
}

/**
 * @brief	get the Dimension of Input Tensor of script
 * @param	py	the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
py_core_getInputDim (void * py, GstTensorsInfo * info)
{
  PYCore *c = (PYCore *) py;
  return c->getInputTensorDim (info);
}

/**
 * @brief	get the Dimension of Output Tensor of script
 * @param	py	the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
py_core_getOutputDim (void * py, GstTensorsInfo * info)
{
  PYCore *c = (PYCore *) py;
  return c->getOutputTensorDim (info);
}

/**
 * @brief	set the Dimension of Input Tensor and return the Dimension of Output Tensor
 * @param	py	the class object
 * @param[in] info Structure for input tensor info.
 * @param[out] info Structure for output tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
py_core_setInputDim (void * py, const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  PYCore *c = (PYCore *) py;
  return c->setInputTensorDim (in_info, out_info);
}

/**
 * @brief	run the script
 * @param	py	: the class object
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
py_core_run (void * py, const GstTensorMemory * input, GstTensorMemory * output)
{
  PYCore *c = (PYCore *) py;
  return c->run (input, output);
}

/**
 * @brief	the destroy notify method for Python. it will free the output array buffer
 * @param[in] data : the data element destroyed at the pipeline
 */
void
py_core_destroyNotify (void * data)
{
  std::map <void*, PyArrayObject*>::iterator it;
  it = PYCore::outputArrayMap.find(data);
  if (it != PYCore::outputArrayMap.end()){
    Py_XDECREF(it->second);
    PYCore::outputArrayMap.erase (it);
  } else
    g_critical("Cannot find output data: 0x%lx", (unsigned long) data);
}

/**
 * @brief return this module's callback type
 * @param None
 * @return callback type
 */
cb_type
py_core_getCbType (void * py)
{
  PYCore *c = (PYCore *) py;
  return c->getCbType ();
}
