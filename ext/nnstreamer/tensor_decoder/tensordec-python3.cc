/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "python3"
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file        tensordec-python3.cc
 * @date        May 06 2021
 * @brief       NNStreamer tensor-decoder subplugin, "python3",
 *              which decodes tensor or tensors to any media type.
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs except for NYI items
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdexcept>
#include <dlfcn.h>
#include <numpy/arrayobject.h>
#include <glib.h>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_decoder.h>
#include "tensordecutil.h"

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
void init_decoder_py (void) __attribute__ ((constructor));
void fini_decoder_py (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */

/**
 * @brief	Python embedding core structure
 */
class PYDecoderCore
{
  public:
  /**
   * member functions.
   */
  PYDecoderCore (const char *_script_path);
  ~PYDecoderCore ();

  int init ();
  int loadScript ();
  const char *getScriptPath ();
  GstFlowReturn decode (const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf);
  GstCaps *getOutCaps (const GstTensorsConfig *config);

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
  PyObject *PyTensorShape_New (const GstTensorInfo *info);
  NPY_TYPES getNumpyType (tensor_type tType);

  private:
  std::string module_name;
  const std::string script_path;
  PyObject *shape_cls;
  PyObject *core_obj;
  void *handle; /**< returned handle by dlopen() */
  GMutex py_mutex;
};

/**
 * @brief	PYDecoderCore creator
 * @param	_script_path	: the logical path to '{script_name}.py' file
 * @note	the script of _script_path will be loaded simultaneously
 * @return	Nothing
 */
PYDecoderCore::PYDecoderCore (const char *_script_path)
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
 * @brief	PYDecoderCore Destructor
 * @return	Nothing
 */
PYDecoderCore::~PYDecoderCore ()
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
 * @brief	decode tensor(s) to any media stream
 */
GstFlowReturn
PYDecoderCore::decode (const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  GstMapInfo out_info;
  GstMemory *out_mem;
  gboolean need_alloc;
  size_t mem_size;
  int rate_n = 0, rate_d = 1;
  PyObject *output = NULL;
  PyObject *raw_data = PyList_New (0);
  PyObject *in_info = PyList_New (0);
  GstFlowReturn ret = GST_FLOW_OK;

  rate_n = config->rate_n;
  rate_d = config->rate_d;
  for (unsigned int i = 0; i < config->info.num_tensors; i++) {
    tensor_type nns_type = config->info.info[i].type;
    npy_intp input_dims[] = { (npy_intp) (input[i].size / gst_tensor_get_element_size (nns_type)) };
    PyObject *input_array = PyArray_SimpleNewFromData (
        1, input_dims, getNumpyType (nns_type), input[i].data);
    PyList_Append (raw_data, input_array);

    PyObject *shape = PyTensorShape_New (&config->info.info[i]);
    PyList_Append (in_info, shape);
  }

  Py_LOCK ();
  if (!PyObject_HasAttrString (core_obj, (char *)"decode")) {
    Py_ERRMSG ("Cannot find 'decode'");
    ret = GST_FLOW_ERROR;
    goto done;
  }

  output = PyObject_CallMethod (core_obj, "decode",  "OOii", raw_data, in_info, rate_n, rate_d);

  if (output) {
    need_alloc = (gst_buffer_get_size (outbuf) == 0);
    mem_size = PyBytes_Size (output);

    if (need_alloc) {
      out_mem = gst_allocator_alloc (NULL, mem_size, NULL);
    } else {
      if (gst_buffer_get_size (outbuf) < mem_size) {
        gst_buffer_set_size (outbuf, mem_size);
      }
      out_mem = gst_buffer_get_all_memory (outbuf);
    }

    if (!gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
      gst_memory_unref (out_mem);
      nns_loge ("Cannot map gst memory (tensor decoder flexbuf)\n");
      ret = GST_FLOW_ERROR;
      goto done;
    }

    memcpy (out_info.data, PyBytes_AsString (output), mem_size);

    gst_memory_unmap (out_mem, &out_info);

    if (need_alloc)
      gst_buffer_append_memory (outbuf, out_mem);
    else
      gst_memory_unref (out_mem);

    Py_XDECREF (output);
  } else {
    Py_ERRMSG ("Fail to get output from 'convert'");
    ret = GST_FLOW_ERROR;
  }

done:
  Py_UNLOCK ();
  return ret;
}

/**
 * @brief	get output caps
 */
GstCaps *
PYDecoderCore::getOutCaps (const GstTensorsConfig *config)
{
  PyObject *result = NULL;
  GstCaps *caps = NULL;

  Py_LOCK ();
  if (!PyObject_HasAttrString (core_obj, (char *)"getOutCaps")) {
    ml_loge ("Cannot find 'getOutCaps'");
    ml_loge ("defualt caps is `application/octet-stream`");
    caps = gst_caps_from_string ("application/octet-stream");
    goto done;
  }

  result = PyObject_CallMethod (core_obj, (char *)"getOutCaps", NULL);
  if (result) {
    gchar * caps_str = PyBytes_AsString (result);
    caps = gst_caps_from_string (caps_str);
    Py_XDECREF (result);
  } else {
    caps = gst_caps_from_string ("application/octet-stream");
  }

done:
  Py_UNLOCK ();
  return caps;
}

/**
 * @brief	load the python script
 */
int
PYDecoderCore::loadScript ()
{
  PyObject *module = PyImport_ImportModule (module_name.c_str ());

  if (module) {
    PyObject *cls = PyObject_GetAttrString (module, "CustomDecoder");
    if (cls) {
      core_obj = PyObject_CallObject (cls, NULL);
      Py_XDECREF (cls);
    } else {
      Py_ERRMSG ("Cannot find 'CustomDecoder' class in the script\n");
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
 * @brief	initialize the object with python script
 */
int
PYDecoderCore::init ()
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
PYDecoderCore::getScriptPath ()
{
  return script_path.c_str ();
}

/**
 * @brief	allocate TensorShape object
 * @param info : the tensor info
 * @return created object
 */
PyObject *
PYDecoderCore::PyTensorShape_New (const GstTensorInfo *info)
{
  PyObject *args = PyTuple_New (2);
  PyObject *dims = PyList_New (0);
  PyObject *type = (PyObject *)PyArray_DescrFromType (getNumpyType (info->type));

  if (nullptr == args || nullptr == dims || nullptr == type)
    throw std::runtime_error ("PYCore::PyTensorShape_New() has failed (1).");

  for (int i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    PyList_Append (dims, PyLong_FromLong ((uint64_t)info->dimension[i]));

  PyTuple_SetItem (args, 0, dims);
  PyTuple_SetItem (args, 1, type);

  return PyObject_CallObject (shape_cls, args);
  /* Its value is checked by setInputTensorDim */
}

/**
 * @brief	return the data type of the tensor for Python numpy
 * @param tType	: the defined type of NNStreamer
 * @return the enum of defined numpy datatypes
 */
NPY_TYPES
PYDecoderCore::getNumpyType (tensor_type tType)
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

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
decoder_py_init (void **pdata)
{
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
decoder_py_exit (void **pdata)
{
  PYDecoderCore *core = static_cast<PYDecoderCore *> (*pdata);

  g_return_if_fail (core != NULL);
  delete core;

  *pdata = NULL;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
decoder_py_setOption (void **pdata, int opNum, const char *param)
{
  gchar *path = (gchar *) param;

  /* opNum 1 = python script path */
  if (opNum == 0) {
    PYDecoderCore *core;

    if (!Py_IsInitialized ())
      throw std::runtime_error ("Python is not initialize.");

    /** Load python script file */
    core = static_cast<PYDecoderCore *> (*pdata);

    if (core != NULL) {
      if (g_strcmp0 (path, core->getScriptPath ()) == 0)
        return TRUE; /* skipped */

      decoder_py_exit (pdata);
    }

    /* init null */
    *pdata = NULL;

    try {
      core = new PYDecoderCore (path);
    } catch (std::bad_alloc & exception) {
      ml_loge ("Failed to allocate memory for decoder subplugin: python3\n");
      ml_loge ("%s", exception.what());
      return FALSE;
    }

    if (core->init () != 0) {
      delete core;
      ml_loge ("failed to initailize the object: Python3\n");
      return FALSE;
    }
    *pdata = core;

    return TRUE;
  }

  GST_INFO ("Property mode-option-%d is ignored", opNum + 1);
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstCaps *
decoder_py_getOutCaps (void **pdata, const GstTensorsConfig *config)
{
  GstCaps *caps;
  PYDecoderCore *core = static_cast<PYDecoderCore *> (*pdata);

  caps = core->getOutCaps (config);
  setFramerateFromConfig (caps, config);
  return caps;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
decoder_py_decode (void **pdata, const GstTensorsConfig *config,
    const GstTensorMemory *input, GstBuffer *outbuf)
{
  PYDecoderCore *core = static_cast<PYDecoderCore *> (*pdata);
  g_return_val_if_fail (config, GST_FLOW_ERROR);
  g_return_val_if_fail (input, GST_FLOW_ERROR);
  g_return_val_if_fail (outbuf, GST_FLOW_ERROR);
  return core->decode (config, input, outbuf);
}

static gchar decoder_subplugin_python3[] = "python3";

/** @brief python3fer tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef Python = { .modename = decoder_subplugin_python3,
  .init = decoder_py_init,
  .exit = decoder_py_exit,
  .setOption = decoder_py_setOption,
  .getOutCaps = decoder_py_getOutCaps,
  .decode = decoder_py_decode };

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** @brief Initialize this object for tensordec-plugin */
void
init_decoder_py (void)
{
  nnstreamer_decoder_probe (&Python);
  /** Python should be initialized and finalized only once */
  Py_Initialize ();
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_decoder_py (void)
{
  /** Python should be initialized and finalized only once */
  Py_Finalize ();
  nnstreamer_decoder_exit (Python.modename);
}
#ifdef __cplusplus
}
#endif /* __cplusplus */
