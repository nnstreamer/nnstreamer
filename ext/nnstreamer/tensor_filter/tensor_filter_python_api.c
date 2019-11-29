/**
 * GStreamer Tensor_Filter, Python Module
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
 * @file	tensor_filter_python_api.c
 * @date	10 Apr 2019
 * @brief	python helper structure for nnstreamer tensor_filter
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	Dongju Chae <dongju.chae@samsung.com>
 * @bug		No known bugs except for NYI items
 */

/**
 * SECTION:element-tensor_filter_python_api
 *
 * A python module that provides a wrapper for internal structures in tensor_filter_python.
 * Users can import this module to access such a functionality
 *
 * <refsect2>
 * <title>Example python script</title>
 * |[
 * import numpy as np
 * import nnstreamer_python as nns
 * dim = nns.TensorShape([1,2,3], np.uint8)
 * ]|
 * </refsect2>
 */

#include "tensor_filter_python_core.h"

/** @brief object structure for custom Python type: TensorShape */
typedef struct {
  PyObject_HEAD
  PyObject *dims;
  PyArray_Descr* type;
} TensorShapeObject;

/** @brief define a prototype for this python module */
#if PY_VERSION_HEX >= 0x03000000
PyMODINIT_FUNC PyInit_nnstreamer_python(void);
#else
PyMODINIT_FUNC initnnstreamer_python(void);
#endif

/**
 * @brief method impl. for setDims
 * @param self : Python type object
 * @param args : arguments for the method
 */
static PyObject *
TensorShape_setDims(TensorShapeObject *self, PyObject *args) {
  PyObject *dims = args;
  PyObject *new_dims;

  /** PyArg_ParseTuple() returns borrowed references */
  if (!PyArg_ParseTuple(args, "O", &dims))
    Py_RETURN_NONE;

  if (PyList_Size(dims) < NNS_TENSOR_RANK_LIMIT) {
    int i;
    for (i = 0; i < NNS_TENSOR_RANK_LIMIT - PyList_Size(dims); i++)
      /** fill '1's in remaining slots */
      PyList_Append(dims, PyLong_FromLong(1));
    new_dims = dims;
    Py_XINCREF(new_dims);
  } else {
    /** PyList_GetSlice() returns new reference */
    new_dims = PyList_GetSlice(dims, 0, NNS_TENSOR_RANK_LIMIT);
  }

  /** swap 'self->dims' */
  Py_XDECREF(self->dims);
  self->dims = new_dims;

  Py_RETURN_NONE;
}

/**
 * @brief method impl. for getDims
 * @param self : Python type object
 * @param args : arguments for the method
 */
static PyObject *
TensorShape_getDims(TensorShapeObject *self, PyObject *args) {
  return Py_BuildValue("O", self->dims);
}

/**
 * @brief method impl. for getType
 * @param self : Python type object
 * @param args : arguments for the method
 */
static PyObject *
TensorShape_getType(TensorShapeObject *self, PyObject *args) {
  return Py_BuildValue("O", self->type);
}

/**
 * @brief new callback for custom type object
 * @param self : Python type object
 * @param args : arguments for the method
 * @param kw : keywords for the arguments
 */
static PyObject *
TensorShape_new(PyTypeObject *type, PyObject *args, PyObject *kw) {
  TensorShapeObject *self = (TensorShapeObject *) type->tp_alloc(type, 0);

  g_assert(self);

  /** Assign default values */
  self->dims = PyList_New(0);
  self->type = PyArray_DescrFromType(NPY_UINT8);
  Py_XINCREF(self->type);

  return (PyObject *) self;
}

/**
 * @brief init callback for custom type object
 * @param self : Python type object
 * @param args : arguments for the method
 * @param kw : keywords for the arguments
 */
static int
TensorShape_init(TensorShapeObject *self, PyObject *args, PyObject *kw) {
  char *keywords[] = {(char*) "dims", (char*) "type", NULL};
  PyObject *dims = NULL;
  PyObject *type = NULL;

  if (!PyArg_ParseTupleAndKeywords(args, kw, "|OO", keywords, &dims, &type))
    return -1;

  if (dims) {
    PyObject *none = PyObject_CallMethod((PyObject*) self, (char*) "setDims", (char*) "O", dims);
    Py_XDECREF(none);
  }

  if (type) {
    PyArray_Descr* dtype;
    if (PyArray_DescrConverter(type, &dtype) != NPY_FAIL) {
      /** swap 'self->type' */
      Py_XDECREF(self->type);
      self->type = dtype;
      Py_XINCREF(dtype);
    } else
      g_critical("Wrong data type");
  }

  return 0;
}

/**
 * @brief dealloc callback for custom type object
 * @param self : Python type object
 */
static void
TensorShape_dealloc(TensorShapeObject *self) {
  Py_XDECREF(self->dims);
  Py_XDECREF(self->type);
  Py_TYPE(self)->tp_free((PyObject *) self);
}

/** @brief members for custom type object */
static PyMemberDef TensorShape_members[] = {
  {(char*) "dims", T_OBJECT_EX, offsetof(TensorShapeObject, dims), 0, NULL},
  {(char*) "type", T_OBJECT_EX, offsetof(TensorShapeObject, type), 0, NULL}
};

/** @brief methods for custom type object */
static PyMethodDef TensorShape_methods[] = {
  {(char*) "setDims", (PyCFunction)TensorShape_setDims, METH_VARARGS | METH_KEYWORDS, NULL},
  {(char*) "getDims", (PyCFunction)TensorShape_getDims, METH_VARARGS | METH_KEYWORDS, NULL},
  {(char*) "getType", (PyCFunction)TensorShape_getType, METH_VARARGS | METH_KEYWORDS, NULL},
  {NULL, NULL, 0, NULL}
};

/** @brief Structure for custom type object */
static PyTypeObject TensorShapeType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "nnstreamer_python.TensorShape",
  .tp_doc = "TensorShape type",
  .tp_basicsize = sizeof(TensorShapeObject),
  .tp_itemsize = 0,
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_new = TensorShape_new,
  .tp_init = (initproc) TensorShape_init,
  .tp_dealloc = (destructor) TensorShape_dealloc,
  .tp_members = TensorShape_members,
  .tp_methods = TensorShape_methods
};

#if PY_VERSION_HEX >= 0x03000000
#define RETVAL(x) x
static PyModuleDef nnstreamer_python_module = {
  PyModuleDef_HEAD_INIT, "nnstreamer_python", NULL, -1, NULL
};
/** @brief module initialization (python 3.x) */
PyMODINIT_FUNC
PyInit_nnstreamer_python(void) {
#else
#define RETVAL(x)
static PyMethodDef nnstreamer_python_methods[] = {
  {NULL, NULL}
};
/** @brief module initialization (python 2.x) */
PyMODINIT_FUNC
initnnstreamer_python(void) {
#endif
  PyObject *type_object = (PyObject*) &TensorShapeType;
  PyObject *module;
#if PY_VERSION_HEX >= 0x03000000
  module = PyModule_Create(&nnstreamer_python_module);
#else
  module = Py_InitModule("nnstreamer_python", nnstreamer_python_methods);
#endif
  if (module == NULL)
    return RETVAL(NULL);

  /** For numpy array init. */
  import_array();

  /** Check TensorShape type */
  g_assert(!(PyType_Ready(&TensorShapeType) < 0));

  Py_INCREF(type_object);
  PyModule_AddObject(module, "TensorShape", type_object);

  return RETVAL(module);
}
