/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file	nnstreamer_python3_helper.cc
 * @date	10 Apr 2019
 * @brief	python helper structure for nnstreamer tensor_filter
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	Dongju Chae <dongju.chae@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * A python module that provides a wrapper for internal structures in tensor_filter_python.
 * Users can import this module to access such a functionality
 *
 * -- Example python script
 *  import numpy as np
 *  import nnstreamer_python as nns
 *  dim = nns.TensorShape([1,2,3], np.uint8)
 */

#if defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6))
#pragma GCC diagnostic ignored "-Wformat"
#endif

#include <nnstreamer_util.h>
#include "nnstreamer_python3_helper.h"

/** @brief object structure for custom Python type: TensorShape */
typedef struct
{
  PyObject_HEAD PyObject * dims;
  PyArray_Descr *type;
} TensorShapeObject;

/** @brief define a prototype for this python module */
PyMODINIT_FUNC PyInit_nnstreamer_python (void);

/**
 * @brief method impl. for setDims
 * @param self : Python type object
 * @param args : arguments for the method
 */
static PyObject *
TensorShape_setDims (TensorShapeObject * self, PyObject * args)
{
  PyObject *dims;
  PyObject *new_dims;
  unsigned int i, len;

  /** PyArg_ParseTuple() returns borrowed references */
  if (!PyArg_ParseTuple (args, "O", &dims))
    Py_RETURN_NONE;

  len = PyList_Size (dims);
  if (len < NNS_TENSOR_RANK_LIMIT) {
    for (i = 0; i < NNS_TENSOR_RANK_LIMIT - len; i++)
      /** fill '1's in remaining slots */
      PyList_Append (dims, PyLong_FromLong (1));
    new_dims = dims;
    Py_XINCREF (new_dims);
  } else {
    /** PyList_GetSlice() returns new reference */
    new_dims = PyList_GetSlice (dims, 0, NNS_TENSOR_RANK_LIMIT);
  }

  /** swap 'self->dims' */
  Py_SAFEDECREF (self->dims);
  self->dims = new_dims;

  Py_RETURN_NONE;
}

/**
 * @brief method impl. for getDims
 * @param self : Python type object
 * @param args : arguments for the method
 */
static PyObject *
TensorShape_getDims (TensorShapeObject * self, PyObject * args)
{
  UNUSED (args);
  return Py_BuildValue ("O", self->dims);
}

/**
 * @brief method impl. for getType
 * @param self : Python type object
 * @param args : arguments for the method
 */
static PyObject *
TensorShape_getType (TensorShapeObject * self, PyObject * args)
{
  UNUSED (args);
  return Py_BuildValue ("O", self->type);
}

/**
 * @brief new callback for custom type object
 * @param self : Python type object
 * @param args : arguments for the method
 * @param kw : keywords for the arguments
 */
static PyObject *
TensorShape_new (PyTypeObject * type, PyObject * args, PyObject * kw)
{
  TensorShapeObject *self = (TensorShapeObject *) type->tp_alloc (type, 0);
  UNUSED (args);
  UNUSED (kw);

  g_assert (self);

  /** Assign default values */
  self->dims = PyList_New (0);
  self->type = PyArray_DescrFromType (NPY_UINT8);
  Py_XINCREF (self->type);

  return (PyObject *) self;
}

/**
 * @brief init callback for custom type object
 * @param self : Python type object
 * @param args : arguments for the method
 * @param kw : keywords for the arguments
 */
static int
TensorShape_init (TensorShapeObject * self, PyObject * args, PyObject * kw)
{
  char *keywords[] = { (char *) "dims", (char *) "type", NULL };
  PyObject *dims = NULL;
  PyObject *type = NULL;

  if (!PyArg_ParseTupleAndKeywords (args, kw, "|OO", keywords, &dims, &type))
    return -1;

  if (dims) {
    PyObject *none =
        PyObject_CallMethod ((PyObject *) self, (char *) "setDims",
        (char *) "O", dims);
    Py_SAFEDECREF (none);
  }

  if (type) {
    PyArray_Descr *dtype;
    if (PyArray_DescrConverter (type, &dtype) != NPY_FAIL) {
      /** swap 'self->type' */
      Py_SAFEDECREF (self->type);
      self->type = dtype;
      Py_XINCREF (self->type);
    } else
      ml_loge ("Wrong data type");
  }

  return 0;
}

/**
 * @brief dealloc callback for custom type object
 * @param self : Python type object
 */
static void
TensorShape_dealloc (TensorShapeObject * self)
{
  Py_SAFEDECREF (self->dims);
  Py_SAFEDECREF (self->type);
  Py_TYPE (self)->tp_free ((PyObject *) self);
}

/** @brief members for custom type object */
static PyMemberDef TensorShape_members[] = {
  {(char *) "dims", T_OBJECT_EX, offsetof (TensorShapeObject, dims), 0, NULL},
  {(char *) "type", T_OBJECT_EX, offsetof (TensorShapeObject, type), 0, NULL}
};

/** @brief methods for custom type object */
static PyMethodDef TensorShape_methods[] = {
  {(char *) "setDims", (PyCFunction) TensorShape_setDims,
        METH_VARARGS | METH_KEYWORDS, NULL},
  {(char *) "getDims", (PyCFunction) TensorShape_getDims,
        METH_VARARGS | METH_KEYWORDS, NULL},
  {(char *) "getType", (PyCFunction) TensorShape_getType,
        METH_VARARGS | METH_KEYWORDS, NULL},
  {NULL, NULL, 0, NULL}
};

/** @brief Structure for custom type object */
static PyTypeObject TensorShapeType = []{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
  PyTypeObject ret = {
    PyVarObject_HEAD_INIT (NULL, 0)
  };
#pragma GCC diagnostic pop
  ret.tp_name = "nnstreamer_python.TensorShape";
  ret.tp_basicsize = sizeof (TensorShapeObject);
  ret.tp_itemsize = 0;
  ret.tp_dealloc = (destructor) TensorShape_dealloc;
  ret.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  ret.tp_doc = "TensorShape type";
  ret.tp_methods = TensorShape_methods;
  ret.tp_members = TensorShape_members;
  ret.tp_init = (initproc) TensorShape_init;
  ret.tp_new = TensorShape_new;
  return ret;
}();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
static PyModuleDef nnstreamer_python_module = {
  PyModuleDef_HEAD_INIT, "nnstreamer_python", NULL, -1, NULL
};
#pragma GCC diagnostic pop

/** @brief module initialization (python 3.x) */
PyMODINIT_FUNC
PyInit_nnstreamer_python (void)
{
  PyObject *type_object = (PyObject *) & TensorShapeType;
  PyObject *module;

  /** Check TensorShape type */
  if (PyType_Ready (&TensorShapeType) < 0)
    return NULL;

  module = PyModule_Create (&nnstreamer_python_module);
  if (module == NULL)
    return NULL;

  /** For numpy array init. */
  import_array ();

  Py_INCREF (type_object);
  PyModule_AddObject (module, "TensorShape", type_object);

  return module;
}

/**
 * @brief	return the data type of the tensor
 * @param npyType	: the defined type of Python numpy
 * @return the enum of defined _NNS_TYPE
 */
tensor_type
getTensorType (NPY_TYPES npyType)
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
getNumpyType (tensor_type tType)
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
 * @brief	load the python script
 */
int
loadScript (PyObject **core_obj, const gchar *module_name, const gchar *class_name)
{
  PyObject *module = PyImport_ImportModule (module_name);

  if (module) {
    PyObject *cls = PyObject_GetAttrString (module, class_name);
    Py_SAFEDECREF (module);

    if (cls) {
      *core_obj = PyObject_CallObject (cls, NULL);
      Py_SAFEDECREF (cls);
    } else {
      Py_ERRMSG ("Cannot find '%s' class in the script\n", class_name);
      return -2;
    }
  } else {
    Py_ERRMSG ("the script is not properly loaded\n");
    return -1;
  }

  return 0;
}

/**
 * @brief	loads the dynamic shared object of the python
 */
int openPythonLib (void **handle)
{
    /**
   * To fix import error of python extension modules
   * (e.g., multiarray.x86_64-linux-gnu.so: undefined symbol: PyExc_SystemError)
   */
  gchar libname[32] = { 0, };

  g_snprintf (libname, sizeof (libname), "libpython%d.%d.%s",
      PY_MAJOR_VERSION, PY_MINOR_VERSION, SO_EXT);
  *handle = dlopen (libname, RTLD_LAZY | RTLD_GLOBAL);
  if (NULL == *handle) {
    /* check the python was compiled with '--with-pymalloc' */
    g_snprintf (libname, sizeof (libname), "libpython%d.%dm.%s",
        PY_MAJOR_VERSION, PY_MINOR_VERSION, SO_EXT);

    *handle = dlopen (libname, RTLD_LAZY | RTLD_GLOBAL);
    if (NULL == *handle)
      return -1;
  }

  return 0;
}

/**
 * @brief	Add custom python module to system path
 */
int addToSysPath (const gchar *path)
{
  /** Add current/directory path to sys.path */
  PyObject *sys_module = PyImport_ImportModule ("sys");
  if (nullptr == sys_module) {
    Py_ERRMSG ("Cannot import python module 'sys'.");
    return -1;
  }

  PyObject *sys_path = PyObject_GetAttrString (sys_module, "path");
  if (nullptr == sys_path) {
    Py_ERRMSG ("Cannot import python module 'path'.");
    Py_SAFEDECREF (sys_module);
    return -1;
  }

  PyList_Append (sys_path, PyUnicode_FromString ("."));
  PyList_Append (sys_path, PyUnicode_FromString (path));

  Py_SAFEDECREF (sys_path);
  Py_SAFEDECREF (sys_module);

  return 0;
}

/**
 * @brief parse the converting result to feed output tensors
 * @param[result] Python object retunred by convert
 * @param[info] info Structure for output tensors info
 * @return 0 if no error, otherwise negative errno
 */
int
parseTensorsInfo (PyObject *result, GstTensorsInfo *info)
{
  if (PyList_Size (result) < 0)
    return -1;

  info->num_tensors = PyList_Size (result);
  for (guint i = 0; i < info->num_tensors; i++) {
    /** don't own the reference */
    PyObject *tensor_shape = PyList_GetItem (result, (Py_ssize_t)i);
    if (nullptr == tensor_shape) {
      Py_ERRMSG ("parseTensorsInfo() has failed (1).");
      return -1;
    }

    PyObject *shape_dims = PyObject_CallMethod (tensor_shape, (char *)"getDims", NULL);
    if (nullptr == shape_dims) {
      Py_ERRMSG ("parseTensorsInfo() has failed (2).");
      return -1;
    }

    PyObject *shape_type = PyObject_CallMethod (tensor_shape, (char *)"getType", NULL);
    if (nullptr == shape_type) {
      Py_ERRMSG ("parseTensorsInfo() has failed (3).");
      Py_SAFEDECREF (shape_dims);
      return -1;
    }

    /** convert numpy type to tensor type */
    info->info[i].type
        = getTensorType ((NPY_TYPES) (((PyArray_Descr *)shape_type)->type_num));

    for (gint j = 0; j < PyList_Size (shape_dims); j++)
      info->info[i].dimension[j]
          = (guint)PyLong_AsLong (PyList_GetItem (shape_dims, (Py_ssize_t)j));

    info->info[i].name = g_strdup("");
    Py_SAFEDECREF (shape_dims);
    Py_SAFEDECREF (shape_type);
  }

  return 0;
}

/**
 * @brief	allocate TensorShape object
 * @param info : the tensor info
 * @return created object
 */
PyObject *
PyTensorShape_New (PyObject * shape_cls, const GstTensorInfo *info)
{
  _import_array (); /** for numpy */

  PyObject *args = PyTuple_New (2);
  PyObject *dims = PyList_New (NNS_TENSOR_RANK_LIMIT);
  PyObject *type = (PyObject *)PyArray_DescrFromType (getNumpyType (info->type));

  if (nullptr == args || nullptr == dims || nullptr == type) {
    Py_ERRMSG ("PYCore::PyTensorShape_New() has failed (1).");
    PyErr_Clear ();
    Py_SAFEDECREF (args);
    Py_SAFEDECREF (dims);
    Py_SAFEDECREF (type);
    return nullptr;
  }

  for (int i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    PyList_SetItem (dims, i, PyLong_FromLong ((uint64_t)info->dimension[i]));

  PyTuple_SetItem (args, 0, dims);
  PyTuple_SetItem (args, 1, type);

  return PyObject_CallObject (shape_cls, args);
  /* Its value is checked by setInputTensorDim */
}
