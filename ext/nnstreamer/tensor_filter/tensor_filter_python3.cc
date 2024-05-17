/**
 * GStreamer Tensor_Filter, Python Module
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All rights reserved.
 * Copyright (C) 2019 Dongju Chae <dongju.chae@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file    tensor_filter_python3.cc
 * @date    10 Apr 2019
 * @brief   Python3 module for tensor_filter gstreamer plugin
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

/**
 * SECTION:element-tensor_filter_python3
 *
 * A filter that loads and executes a python3 script implementing a custom
 * filter.
 * The python3 script should be provided.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 videotestsrc ! video/x-raw,format=RGB,width=640,height=480 !
 *    tensor_converter ! tensor_filter framework="python3"
 *    model="${PATH_TO_SCRIPT}" ! tensor_sink
 * ]|
 * </refsect2>
 */

#if defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6))
#pragma GCC diagnostic ignored "-Wformat"
#endif

/* nnstreamer plugin api headers */
#include <map>
#include <nnstreamer_conf.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include "nnstreamer_python3_helper.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_filter_py (void) __attribute__ ((constructor));
void fini_filter_py (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

namespace nnstreamer
{
namespace tensorfilter_python3
{
/** @brief Callback type for custom filter */
enum class cb_type {
  CB_SETDIM = 0,
  CB_GETDIM,

  CB_END,
};

/**
 * @brief	Python embedding core structure
 */
class PYCore
{
  public:
  /**
   * member functions.
   */
  PYCore (const char *_script_path, const char *_custom);
  ~PYCore ();

  int init (const GstTensorFilterProperties *prop);
  const char *getScriptPath ();
  int getInputTensorDim (GstTensorsInfo *info);
  int getOutputTensorDim (GstTensorsInfo *info);
  int setInputTensorDim (const GstTensorsInfo *in_info, GstTensorsInfo *out_info);
  int run (const GstTensorMemory *input, GstTensorMemory *output);

  void freeOutputTensors (void *data);

  /** @brief Return callback type */
  cb_type getCbType ()
  {
    return callback_type;
  }

  private:
  int loadScript ();
  const std::string script_path; /**< from model_path property */
  const std::string module_args; /**< from custom property */

  std::string module_name;
  std::map<void *, PyArrayObject *> outputArrayMap;

  cb_type callback_type;

  PyObject *core_obj;
  PyObject *shape_cls;

  GstTensorsInfo inputTensorMeta; /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta; /**< The tensor info of output tensors */

  bool configured; /**< True if the script is successfully loaded */
  void *handle; /**< returned handle by dlopen() */
  GMutex py_mutex;

  int checkTensorSize (GstTensorMemory *output, PyArrayObject *array);
  int checkTensorType (int nns_type, int np_type);

  /** @brief Lock python-related actions */
  void Py_LOCK ()
  {
    g_mutex_lock (&py_mutex);
  }
  /** @brief Unlock python-related actions */
  void Py_UNLOCK ()
  {
    g_mutex_unlock (&py_mutex);
  }
};

/**
 * @brief	PYCore creator
 * @param	_script_path	: the logical path to '{script_name}.py' file
 * @note	the script of _script_path will be loaded simultaneously
 * @return	Nothing
 */
PYCore::PYCore (const char *_script_path, const char *_custom)
    : script_path (_script_path), module_args (_custom != NULL ? _custom : "")
{
  if (openPythonLib (&handle))
    throw std::runtime_error (dlerror ());

  _import_array (); /** for numpy */

  /**
   * Parse script path to get module name
   * The module name should drop its extension (i.e., .py)
   */
  module_name = script_path;

  const size_t last_idx = module_name.find_last_of ("/\\");
  if (last_idx != std::string::npos)
    module_name.erase (0, last_idx + 1);

  const size_t ext_idx = module_name.rfind ('.');
  if (ext_idx != std::string::npos)
    module_name.erase (ext_idx);

  addToSysPath (script_path.substr (0, last_idx).c_str ());

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);

  callback_type = cb_type::CB_END;
  core_obj = NULL;
  configured = false;
  shape_cls = NULL;
  g_mutex_init (&py_mutex);
}

/**
 * @brief	PYCore Destructor
 * @return	Nothing
 */
PYCore::~PYCore ()
{
  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);

  Py_LOCK ();
  Py_SAFEDECREF (core_obj);
  Py_SAFEDECREF (shape_cls);

  PyErr_Clear ();
  Py_UNLOCK ();

  dlclose (handle);
}

/**
 * @brief	initialize the object with python script
 * @return 0 if OK. non-zero if error.
 */
int
PYCore::init (const GstTensorFilterProperties *prop)
{
  int ret = -EINVAL;
  /** Find nnstreamer_api module */
  Py_LOCK ();
  PyObject *api_module = PyImport_ImportModule ("nnstreamer_python");
  if (api_module == NULL) {
    Py_ERRMSG ("Cannt find `nnstreamer_python` module");
    goto exit;
  }

  shape_cls = PyObject_GetAttrString (api_module, "TensorShape");
  Py_SAFEDECREF (api_module);

  if (shape_cls == NULL) {
    Py_ERRMSG ("Failed to get `TensorShape` from `nnstreamer_python` module");
    goto exit;
  }

  gst_tensors_info_copy (&inputTensorMeta, &prop->input_meta);
  gst_tensors_info_copy (&outputTensorMeta, &prop->output_meta);

  ret = loadScript ();
exit:
  Py_UNLOCK ();
  return ret;
}

/**
 * @brief	get the script path
 * @return the script path.
 */
const char *
PYCore::getScriptPath ()
{
  return script_path.c_str ();
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
  /** This is a private method that needs the lock kept locked */
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  int ret = -EINVAL;

  PyObject *module = PyImport_ImportModule (module_name.c_str ());
  if (module) {
    PyObject *cls = PyObject_GetAttrString (module, "CustomFilter");
    if (cls) {
      PyObject *py_args;
      if (!module_args.empty ()) {
        gchar **g_args = g_strsplit (module_args.c_str (), " ", 0);
        char **args = g_args;
        int argc = 0;
        while (*(args++) != NULL)
          argc++;

        if (argc < 1) {
          g_strfreev (g_args);
          ml_loge ("Cannot load python script for python-custom-filter.\n");
          goto exit;
        }

        py_args = PyTuple_New (argc);

        for (int i = 0; i < argc; i++)
          PyTuple_SetItem (py_args, i, PyUnicode_FromString (g_args[i]));

        core_obj = PyObject_CallObject (cls, py_args);

        Py_SAFEDECREF (py_args);
        g_strfreev (g_args);
      } else
        core_obj = PyObject_CallObject (cls, NULL);

      if (core_obj) {
        /** check whther either setInputDim or getInputDim/getOutputDim are
         * defined */
        if (PyObject_HasAttrString (core_obj, (char *) "setInputDim"))
          callback_type = cb_type::CB_SETDIM;
        else if (PyObject_HasAttrString (core_obj, (char *) "getInputDim")
                 && PyObject_HasAttrString (core_obj, (char *) "getOutputDim"))
          callback_type = cb_type::CB_GETDIM;
        else
          callback_type = cb_type::CB_END;
      } else {
        Py_ERRMSG ("Fail to create an instance 'CustomFilter'\n");
        ret = -3;
        goto exit;
      }

      Py_SAFEDECREF (cls);
    } else {
      Py_ERRMSG ("Cannot find 'CustomFilter' class in the script\n");
      ret = -2;
      goto exit;
    }

    Py_SAFEDECREF (module);
  } else {
    Py_ERRMSG ("the script is not properly loaded\n");
    ret = -1;
    goto exit;
  }

  configured = true;

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Script is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif

  ret = 0;
exit:
  return ret;
}

/**
 * @brief	check the data type of tensors in array
 * @param nns_type : tensor type for output tensor
 * @param np_type  : python array type for output tensor
 * @return a boolean value for whether the types are matched
 */
int
PYCore::checkTensorType (int nns_type, int np_type)
{
  switch (nns_type) {
    case _NNS_INT64:
      return np_type == NPY_INT64;
    case _NNS_UINT64:
      return np_type == NPY_UINT64;
    case _NNS_INT32:
      return np_type == NPY_INT32;
    case _NNS_UINT32:
      return np_type == NPY_UINT32;
    case _NNS_INT16:
      return np_type == NPY_INT16;
    case _NNS_UINT16:
      return np_type == NPY_UINT16;
    case _NNS_INT8:
      return np_type == NPY_INT8;
    case _NNS_UINT8:
      return np_type == NPY_UINT8;
    case _NNS_FLOAT64:
      return np_type == NPY_FLOAT64;
    case _NNS_FLOAT32:
      return np_type == NPY_FLOAT32;
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
  /** This is a private method that needs the lock kept locked */
  if (nullptr == output || nullptr == array)
    throw std::invalid_argument ("Null pointers are given to PYCore::checkTensorSize().\n");

  size_t total_size = PyArray_ITEMSIZE (array);

  for (int i = 0; i < PyArray_NDIM (array); i++)
    total_size *= PyArray_DIM (array, i);

  return (output->size == total_size);
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
PYCore::getInputTensorDim (GstTensorsInfo *info)
{
  int res = 0;

  if (nullptr == info)
    throw std::invalid_argument ("A null pointer is given to PYCore::getInputTensorDim().\n");

  Py_LOCK ();

  PyObject *result = PyObject_CallMethod (core_obj, (char *) "getInputDim", NULL);
  if (result) {
    res = parseTensorsInfo (result, info);
    Py_SAFEDECREF (result);
  } else {
    Py_ERRMSG ("Fail to call 'getInputDim'");
    res = -1;
  }

  Py_UNLOCK ();

  return res;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
PYCore::getOutputTensorDim (GstTensorsInfo *info)
{
  int res = 0;

  if (nullptr == info)
    throw std::invalid_argument ("A null pointer is given to PYCore::getOutputTensorDim().\n");

  Py_LOCK ();

  PyObject *result = PyObject_CallMethod (core_obj, (char *) "getOutputDim", NULL);
  if (result) {
    res = parseTensorsInfo (result, info);
    Py_SAFEDECREF (result);
  } else {
    Py_ERRMSG ("Fail to call 'getOutputDim'");
    res = -1;
  }

  Py_UNLOCK ();

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
PYCore::setInputTensorDim (const GstTensorsInfo *in_info, GstTensorsInfo *out_info)
{
  GstTensorInfo *_info;
  int res = 0;

  if (nullptr == in_info || nullptr == out_info)
    throw std::invalid_argument ("Null pointers are given to PYCore::setInputTensorDim().\n");

  Py_LOCK ();

  /** to Python list object */
  PyObject *param = PyList_New (in_info->num_tensors);
  if (nullptr == param) {
    Py_UNLOCK ();
    throw std::runtime_error ("PyList_New(); has failed.");
  }

  for (unsigned int i = 0; i < in_info->num_tensors; i++) {
    PyObject *shape;

    _info = gst_tensors_info_get_nth_info ((GstTensorsInfo *) in_info, i);
    shape = PyTensorShape_New (shape_cls, _info);
    if (nullptr == shape) {
      Py_UNLOCK ();
      throw std::runtime_error ("PyTensorShape_New(); has failed.");
    }

    PyList_SetItem (param, i, shape);
  }

  PyObject *result
      = PyObject_CallMethod (core_obj, (char *) "setInputDim", (char *) "(O)", param);

  Py_SAFEDECREF (param);

  if (result) {
    gst_tensors_info_copy (&inputTensorMeta, in_info);
    res = parseTensorsInfo (result, out_info);
    if (res == 0)
      gst_tensors_info_copy (&outputTensorMeta, out_info);
    Py_SAFEDECREF (result);
  } else {
    Py_ERRMSG ("Fail to call 'setInputDim'");
    res = -1;
  }

  Py_UNLOCK ();

  return res;
}

/**
 * @brief free output tensor corresponding to the given data
 * @param[data] The data element
 */
void
PYCore::freeOutputTensors (void *data)
{
  std::map<void *, PyArrayObject *>::iterator it;

  it = outputArrayMap.find (data);
  if (it != outputArrayMap.end ()) {
    Py_SAFEDECREF (it->second);
    outputArrayMap.erase (it);
  } else {
    ml_loge ("Cannot find output data: 0x%lx", (unsigned long) data);
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
PYCore::run (const GstTensorMemory *input, GstTensorMemory *output)
{
  GstTensorInfo *_info;
  int res = 0;
  PyObject *result;

#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  if (nullptr == output || nullptr == input)
    throw std::invalid_argument ("Null pointers are given to PYCore::run().\n");

  Py_LOCK ();

  PyObject *param = PyList_New (inputTensorMeta.num_tensors);
  for (unsigned int i = 0; i < inputTensorMeta.num_tensors; i++) {
    _info = gst_tensors_info_get_nth_info (&inputTensorMeta, i);

    /** create a Numpy array wrapper (1-D) for NNS tensor data */
    tensor_type nns_type = _info->type;
    npy_intp input_dims[]
        = { (npy_intp) (input[i].size / gst_tensor_get_element_size (nns_type)) };
    PyObject *input_array = PyArray_SimpleNewFromData (
        1, input_dims, getNumpyType (nns_type), input[i].data);
    PyList_SetItem (param, i, input_array);
  }

  result = PyObject_CallMethod (core_obj, (char *) "invoke", (char *) "(O)", param);

  if (result) {
    if ((unsigned int) PyList_Size (result) != outputTensorMeta.num_tensors) {
      res = -EINVAL;
      ml_logf ("The Python allocated size mismatched. Cannot proceed.\n");
      Py_SAFEDECREF (result);
      goto exit_decref;
    }

    for (unsigned int i = 0; i < outputTensorMeta.num_tensors; i++) {
      PyArrayObject *output_array
          = (PyArrayObject *) PyList_GetItem (result, (Py_ssize_t) i);

      _info = gst_tensors_info_get_nth_info (&outputTensorMeta, i);

      /** type/size checking */
      if (checkTensorType (_info->type, PyArray_TYPE (output_array))
          && checkTensorSize (&output[i], output_array)) {
        /** obtain the pointer to the buffer for the output array */
        output[i].data = PyArray_DATA (output_array);
        Py_XINCREF (output_array);
        outputArrayMap.insert (std::make_pair (output[i].data, output_array));
      } else {
        ml_loge ("Output tensor type/size is not matched\n");
        res = -2;
        break;
      }
    }

    Py_SAFEDECREF (result);
  } else {
    Py_ERRMSG ("Fail to call 'invoke'");
    res = -1;
  }

exit_decref:
  Py_SAFEDECREF (param);
  Py_UNLOCK ();

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Invoke() is finished: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif

  return res;
}

/**
 * @brief Class for Python3 subplugin
 */
class TensorFilterPython : public tensor_filter_subplugin
{
  public:
  TensorFilterPython ();
  ~TensorFilterPython ();

  /* mandatory methods */
  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);

  /* static methods */
  static void init_filter_py ();
  static void fini_filter_py ();

  private:
  PYCore *core;

  static TensorFilterPython *registered;
  static const char *name;
  static const accl_hw hw_list[];
  static const int num_hw;
};

void init_filter_py (void) __attribute__ ((constructor));
void fini_filter_py (void) __attribute__ ((destructor));

TensorFilterPython *TensorFilterPython::registered = nullptr;
const char *TensorFilterPython::name = "python3";
const accl_hw TensorFilterPython::hw_list[] = {};
const int TensorFilterPython::num_hw = 0;

/**
 * @brief Construct a new Python subplugin instance
 */
TensorFilterPython::TensorFilterPython () : core (nullptr)
{
  if (!Py_IsInitialized ())
    throw std::runtime_error ("Python is not initialize.");
}

/**
 * @brief Destructor of TensorFilterPython
 */
TensorFilterPython::~TensorFilterPython ()
{
  if (core != nullptr) {
    PyGILState_STATE gstate = PyGILState_Ensure ();
    delete core;
    PyGILState_Release (gstate);
  }
}

/**
 * @brief Method to get an empty object
 */
tensor_filter_subplugin &
TensorFilterPython::getEmptyInstance ()
{
  return *(new TensorFilterPython ());
}

/**
 * @brief Configure Python instance
 */
void
TensorFilterPython::configure_instance (const GstTensorFilterProperties *prop)
{
  const gchar *script_path;

  if (prop->num_models != 1)
    return;

  /**
   * prop->model_files[0] contains the path of a python script
   * prop->custom contains its arguments seperated by ' '
   */
  script_path = prop->model_files[0];

  if (core != nullptr) {
    if (g_strcmp0 (script_path, core->getScriptPath ()) == 0) {
      return; /* skipped */
    }

    PyGILState_STATE gstate = PyGILState_Ensure ();
    delete core;
    PyGILState_Release (gstate);
  }

  PyGILState_STATE gstate = PyGILState_Ensure ();
  core = new PYCore (script_path, prop->custom_properties);
  if (core == nullptr) {
    g_printerr ("Failed to allocate memory for filter subplugin: Python\n");
    goto done;
  }

  if (core->init (prop) != 0) {
    delete core;
    g_printerr ("failed to initailize the object: Python\n");
    PyGILState_Release (gstate);
    throw std::runtime_error ("Python is not initialize");
  }

  /** check methods in python script */
  if (core->getCbType () != cb_type::CB_SETDIM && core->getCbType () != cb_type::CB_GETDIM) {
    delete core;
    g_printerr ("Wrong callback type\n");
    goto done;
  }

done:
  PyGILState_Release (gstate);
}

/**
 * @brief Invoke Python using input tensors
 */
void
TensorFilterPython::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  PyGILState_STATE gstate = PyGILState_Ensure ();
  core->run (input, output);
  PyGILState_Release (gstate);
}

/**
 * @brief Get Python framework info.
 */
void
TensorFilterPython::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = FALSE;
  /** @todo: support this to optimize performance later. */
  info.allocate_in_invoke = TRUE;
  info.run_without_model = FALSE;
  info.verify_model_path = TRUE;
  info.hw_list = hw_list;
  info.num_hw = num_hw;
  info.statistics = nullptr;
}

/**
 * @brief Get Python model info.
 */
int
TensorFilterPython::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  UNUSED (ops);
  int ret = 0;
  if (core->getCbType () == cb_type::CB_END) {
    ml_loge ("cb type wrong");
    return -ENOENT;
  }

  PyGILState_STATE gstate = PyGILState_Ensure ();

  if (core->getCbType () == cb_type::CB_GETDIM) {
    ret = core->getInputTensorDim (&in_info);
    if (!ret)
      ret = core->getOutputTensorDim (&out_info);
  } else if (core->getCbType () == cb_type::CB_SETDIM) {
    ret = core->setInputTensorDim (&in_info, &out_info);
  }

  PyGILState_Release (gstate);
  return ret;
}

/**
 * @brief Method to handle the event
 */
int
TensorFilterPython::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  if (ops == DESTROY_NOTIFY) {
    if (core) {
      PyGILState_STATE gstate = PyGILState_Ensure ();
      core->freeOutputTensors (data.data);
      PyGILState_Release (gstate);
    }
  }
  return 0;
}

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
TensorFilterPython::init_filter_py ()
{
  /** Python should be initialized and finalized only once */
  nnstreamer_python_init_refcnt ();
  registered = tensor_filter_subplugin::register_subplugin<TensorFilterPython> ();
}

/** @brief Destruct the subplugin */
void
TensorFilterPython::fini_filter_py ()
{
  nnstreamer_python_status_check ();
  nnstreamer_python_fini_refcnt ();

  /* internal logic error */
  assert (registered != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registered);

/**
 * @todo Remove below lines after this issue is addressed.
 * Tizen issues: After python version has been upgraded from 3.9.1 to 3.9.10,
 * python converter is stopped at Py_Finalize. Since Py_Initialize is not called
 * twice from this object, Py_Finalize is temporarily removed.
 * We do not know if it's safe to call this at this point.
 * We can finalize when ALL python subplugins share the same ref counter.
 */
#if 0
  /** Python should be initialized and finalized only once */
  if (Py_IsInitialized ())
    Py_Finalize ();
#endif
}

/**
 * @brief Subplugin initializer
 */
void
init_filter_py ()
{
  TensorFilterPython::init_filter_py ();
}

/**
 * @brief Subplugin finalizer
 */
void
fini_filter_py ()
{
  TensorFilterPython::fini_filter_py ();
}
} /* namespace tensorfilter_python3 */
} /* namespace nnstreamer */
