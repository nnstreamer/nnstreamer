/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer tensor_converter subplugin, "python3"
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file	tensor_converter_python3.c
 * @date	May 03 2021
 * @brief	NNStreamer tensor-converter subplugin, "python3",
*         which converts to tensors using python.
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Gichan Jang <gichan2.jang@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <stdexcept>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <dlfcn.h>
#include <numpy/arrayobject.h>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_converter.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

#define SO_EXT "so.1.0"

#define Py_ERRMSG(...)     \
  do {                     \
    PyErr_Print ();        \
    ml_loge (__VA_ARGS__); \
  } while (0);

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_converter_py (void) __attribute__((constructor));
void fini_converter_py (void) __attribute__((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/**
 * @brief	Python embedding core structure
 */
class PYConverterCore
{
  public:
  /**
   * member functions.
   */
  PYConverterCore (const char *_script_path);
  ~PYConverterCore ();

  int init ();
  int loadScript ();
  const char *getScriptPath ();
  GstBuffer *convert (GstBuffer*in_buf, GstTensorsConfig *config);
  int parseTensorsInfo (PyObject *result, GstTensorsInfo *info);
  tensor_type getTensorType (NPY_TYPES npyType);

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

  private:
  std::string module_name;
  const std::string script_path;
  PyObject *shape_cls;
  PyObject *core_obj;
  void *handle; /**< returned handle by dlopen() */
  GMutex py_mutex;
};

/**
 * @brief	PYConverterCore creator
 * @param	_script_path	: the logical path to '{script_name}.py' file
 * @note	the script of _script_path will be loaded simultaneously
 * @return	Nothing
 */
PYConverterCore::PYConverterCore (const char *_script_path)
    : script_path (_script_path)
{
  /**
   * To fix import error of python extension modules
   * (e.g., multiarray.x86_64-linux-gnu.so: undefined symbol: PyExc_SystemError)
   */
  gchar libname[32] = { 0, };

  g_snprintf (libname, sizeof (libname), "libpython%d.%d.%s",
      PY_MAJOR_VERSION, PY_MINOR_VERSION, SO_EXT);
  handle = dlopen (libname, RTLD_LAZY | RTLD_GLOBAL);
  if (nullptr == handle) {
    /* check the python was compiled with '--with-pymalloc' */
    g_snprintf (libname, sizeof (libname), "libpython%d.%dm.%s",
        PY_MAJOR_VERSION, PY_MINOR_VERSION, SO_EXT);

    handle = dlopen (libname, RTLD_LAZY | RTLD_GLOBAL);
    if (nullptr == handle)
      throw std::runtime_error (dlerror ());
  }

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

  /** Add current/directory path to sys.path */
  PyObject *sys_module = PyImport_ImportModule ("sys");
  if (nullptr == sys_module)
    throw std::runtime_error ("Cannot import python module 'sys'.");

  PyObject *sys_path = PyObject_GetAttrString (sys_module, "path");
  if (nullptr == sys_path)
    throw std::runtime_error ("Cannot import python module 'path'.");

  PyList_Append (sys_path, PyUnicode_FromString ("."));
  PyList_Append (sys_path,
  PyUnicode_FromString (script_path.substr (0, last_idx).c_str ()));

  Py_XDECREF (sys_path);
  Py_XDECREF (sys_module);

  core_obj = NULL;
  shape_cls = NULL;

  g_mutex_init (&py_mutex);
}

/**
 * @brief	PYConverterCore Destructor
 * @return	Nothing
 */
PYConverterCore::~PYConverterCore ()
{
  if (core_obj)
    Py_XDECREF (core_obj);
  if (shape_cls)
    Py_XDECREF (shape_cls);
  PyErr_Clear ();

  dlclose (handle);
  g_mutex_clear (&py_mutex);
}

/**
 * @brief parse the converting result to feed output tensors
 * @param[result] Python object retunred by convert
 * @param[info] info Structure for output tensors info
 * @return 0 if no error, otherwise negative errno
 */
int
PYConverterCore::parseTensorsInfo (PyObject *result, GstTensorsInfo *info)
{
  if (PyList_Size (result) < 0)
    return -1;

  info->num_tensors = PyList_Size (result);
  for (unsigned int i = 0; i < info->num_tensors; i++) {
    /** don't own the reference */
    PyObject *tensor_shape = PyList_GetItem (result, (Py_ssize_t)i);
    if (nullptr == tensor_shape)
      throw std::runtime_error ("parseTensorsInfo() has failed (1).");

    PyObject *shape_dims = PyObject_CallMethod (tensor_shape, (char *)"getDims", NULL);
    if (nullptr == shape_dims)
      throw std::runtime_error ("parseTensorsInfo() has failed (2).");


    PyObject *shape_type = PyObject_CallMethod (tensor_shape, (char *)"getType", NULL);
    if (nullptr == shape_type)
      throw std::runtime_error ("parseOutputTensors() has failed (3).");

    /** convert numpy type to tensor type */
    info->info[i].type
        = getTensorType ((NPY_TYPES) (((PyArray_Descr *)shape_type)->type_num));

    for (int j = 0; j < PyList_Size (shape_dims); j++)
      info->info[i].dimension[j]
          = (uint32_t)PyLong_AsLong (PyList_GetItem (shape_dims, (Py_ssize_t)j));

    info->info[i].name = g_strdup("");
    Py_XDECREF (shape_dims);
  }

  return 0;
}

/**
 * @brief	convert any media stream to tensor
 */
GstBuffer *
PYConverterCore::convert (GstBuffer *in_buf, GstTensorsConfig *config)
{
  GstMemory *in_mem, *out_mem;
  GstMapInfo in_info;
  GstBuffer *out_buf = NULL;
  PyObject *tensors_info = NULL, *output = NULL, *pyValue = NULL;
  gint rate_n, rate_d;
  guint mem_size;
  gpointer mem_data;
  if (nullptr == in_buf)
    throw std::invalid_argument ("Null pointers are given to PYConverterCore::convert().\n");

  in_mem = gst_buffer_peek_memory (in_buf, 0);

  if (!gst_memory_map (in_mem, &in_info, GST_MAP_READ)) {
    Py_ERRMSG ("Cannot map input memory / tensor_converter::custom-script");
    return NULL;
  }

  npy_intp input_dims[] = { (npy_intp) (gst_buffer_get_size (in_buf)) };

  PyObject *param = PyList_New (0);
  PyObject *input_array = PyArray_SimpleNewFromData (
      1, input_dims, NPY_UINT8, in_info.data);
  PyList_Append (param, input_array);

  Py_LOCK ();
  if (!PyObject_HasAttrString (core_obj, (char *)"convert")) {
    Py_ERRMSG ("Cannot find 'convert'");
    goto done;
  }

  pyValue = PyObject_CallMethod (core_obj, "convert", "(O)", param);

  if (!PyArg_ParseTuple (pyValue, "OOii", &tensors_info, &output,
      &rate_n, &rate_d)) {
    Py_ERRMSG ("Failed to parse converting result");
    goto done;
  }

  if (parseTensorsInfo (tensors_info, &config->info) != 0) {
    Py_ERRMSG ("Failed to parse tensors info");
    goto done;
  }
  config->rate_n = rate_n;
  config->rate_d = rate_d;
  Py_XDECREF (tensors_info);

  if (output) {
    unsigned int num_tensors = PyList_Size (output);

    out_buf = gst_buffer_new ();
    for (unsigned int i = 0; i < num_tensors; i++) {
      PyArrayObject *output_array
          = (PyArrayObject *)PyList_GetItem (output, (Py_ssize_t)i);

      mem_size = PyArray_SIZE (output_array);
      mem_data = g_memdup ((guint8 *) PyArray_DATA (output_array), mem_size);

      out_mem = gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY, mem_data,
          mem_size, 0, mem_size, mem_data, g_free);
      gst_buffer_append_memory (out_buf, out_mem);

    }
    Py_XDECREF (output);
  } else {
    Py_ERRMSG ("Fail to get output from 'convert'");
  }

done:
  Py_UNLOCK ();
  gst_memory_unmap (in_mem, &in_info);
  return out_buf;
}

/**
 * @brief	load the python script
 */
int
PYConverterCore::loadScript ()
{
  PyObject *module = PyImport_ImportModule (module_name.c_str ());

  if (module) {
    PyObject *cls = PyObject_GetAttrString (module, "CustomConverter");
    if (cls) {
      core_obj = PyObject_CallObject (cls, NULL);
      Py_XDECREF (cls);
    } else {
      Py_ERRMSG ("Cannot find 'CustomConverter' class in the script\n");
      return -2;
    }
    Py_XDECREF (module);
  } else {
    Py_ERRMSG ("the script is not properly loaded\n");
    return -1;
  }

  return 0;
}

/**
 * @brief	return the data type of the tensor
 * @param npyType	: the defined type of Python numpy
 * @return the enum of defined _NNS_TYPE
 */
tensor_type
PYConverterCore::getTensorType (NPY_TYPES npyType)
{
  npyType = NPY_INT;
  if (npyType == NPY_INT32)
    g_critical ("NPY_INT32");
  else if (npyType == NPY_INT)
    g_critical ("NPY_INT");
  else
   g_critical ("what~?");
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
 * @brief	initialize the object with python script
 * @return 0 if OK. non-zero if error.
 */
int
PYConverterCore::init ()
{
  /** Find nnstreamer_api module */
  PyObject *api_module = PyImport_ImportModule ("nnstreamer_python");
  if (api_module == NULL) {
    return -EINVAL;
  }

  shape_cls = PyObject_GetAttrString (api_module, "TensorShape");
  Py_XDECREF (api_module);

  if (shape_cls == NULL)
    return -EINVAL;

  return loadScript ();
}

/**
 * @brief	get the script path
 * @return the script path.
 */
const char *
PYConverterCore::getScriptPath ()
{
  return script_path.c_str ();
}

/**
 * @brief Free privateData and move on.
 */
static void
py_close (void **private_data)
{
  PYConverterCore *core = static_cast<PYConverterCore *> (*private_data);

  g_return_if_fail (core != NULL);
  delete core;

  *private_data = NULL;
}

/**
 * @brief The open callback for GstTensorConverterFramework. Called before anything else
 * @param path: python script path
 * @param private_data: python plugin's private data
 */
static int
py_open (const gchar *path, void **priv_data)
{
  PYConverterCore *core;

  if (!Py_IsInitialized ())
    throw std::runtime_error ("Python is not initialize.");

  /** Load python script file */
  core = static_cast<PYConverterCore *> (*priv_data);

  if (core != NULL) {
    if (g_strcmp0 (path, core->getScriptPath ()) == 0)
      return 1; /* skipped */

    py_close (priv_data);
  }

  /* init null */
  *priv_data = NULL;

  core = new PYConverterCore (path);
  if (core == NULL) {
    Py_ERRMSG ("Failed to allocate memory for converter subplugin: Python\n");
    return -1;
  }

  if (core->init () != 0) {
    delete core;
    Py_ERRMSG ("failed to initailize the object: Python\n");
    return -2;
  }
  *priv_data = core;

  return 0;
}

/** @brief tensor converter plugin's NNStreamerExternalConverter callback */
static GstCaps *
python_query_caps (const GstTensorsConfig *config)
{
  return gst_caps_from_string ("application/octet-stream");
}

/**
 * @brief tensor converter plugin's NNStreamerExternalConverter callback
 */
static gboolean
python_get_out_config (const GstCaps *in_cap, GstTensorsConfig *config)
{
  GstStructure *structure;
  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensors_config_init (config);
  g_return_val_if_fail (in_cap != NULL, FALSE);

  structure = gst_caps_get_structure (in_cap, 0);
  g_return_val_if_fail (structure != NULL, FALSE);

  /* All tensor info should be updated later in chain function. */
  config->info.info[0].type = _NNS_UINT8;
  config->info.num_tensors = 1;
  if (gst_tensor_parse_dimension ("1:1:1:1", config->info.info[0].dimension) == 0) {
    Py_ERRMSG ("Failed to set initial dimension for subplugin");
    return FALSE;
  }

  if (gst_structure_has_field (structure, "framerate")) {
    gst_structure_get_fraction (structure, "framerate", &config->rate_n, &config->rate_d);
  } else {
    /* cannot get the framerate */
    config->rate_n = 0;
    config->rate_d = 1;
  }
  return TRUE;
}

/**
 * @brief tensor converter plugin's NNStreamerExternalConverter callback
 */
static GstBuffer *
python_convert (GstBuffer *in_buf, GstTensorsConfig *config, void *priv_data)
{
  PYConverterCore *core = static_cast<PYConverterCore *> (priv_data);
  g_return_val_if_fail (in_buf, NULL);
  g_return_val_if_fail (config, NULL);
  return core->convert (in_buf, config);
}

static const gchar converter_subplugin_python[] = "python3";

/** @brief flatbuffer tensor converter sub-plugin NNStreamerExternalConverter instance */
static NNStreamerExternalConverter Python = {
  .name = converter_subplugin_python,
  .convert = python_convert,
  .get_out_config = python_get_out_config,
  .query_caps = python_query_caps,
  .open = py_open,
  .close = py_close,
};

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/** @brief Initialize this object for tensor converter sub-plugin */
void
init_converter_py (void)
{
  registerExternalConverter (&Python);
  /** Python should be initialized and finalized only once */
  Py_Initialize ();
}

/** @brief Destruct this object for tensor converter sub-plugin */
void
fini_converter_py (void)
{
  /** Python should be initialized and finalized only once */
  Py_Finalize ();
  unregisterExternalConverter (Python.name);
}
#ifdef __cplusplus
}
#endif /* __cplusplus */
