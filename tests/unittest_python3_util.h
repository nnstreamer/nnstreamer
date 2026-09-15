/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    unittest_python3_util.h
 * @date    15 Sep 2026
 * @brief   Helpers for the unit tests of the python3 sub-plugins to observe the embedded interpreter
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * A reference leak in a python sub-plugin keeps its objects alive without making
 * them unreachable to a memory checker, so these helpers count what the
 * interpreter holds instead. Include this only in a test binary built with python3.
 */
#ifndef __NNS_UNITTEST_PYTHON3_UTIL_H__
#define __NNS_UNITTEST_PYTHON3_UTIL_H__

#include <Python.h>
#include <glib.h>

/** @brief Tracked objects a run of calls may leave behind (caches, lazy imports); a leaked container adds at least one per call */
#define PY_TEST_GC_SLACK (20)

/**
 * @brief Number of objects the python garbage collector tracks after a full collection.
 * @return the number of objects, or -1 if the interpreter cannot answer.
 * @note Only containers such as lists, tuples and dicts are tracked. A numpy array, bytes or str is not,
 *       so a leak of those does not show up here; pin such an object with py_test_attr_refcount().
 */
static inline Py_ssize_t
py_test_gc_object_count (void)
{
  Py_ssize_t count = -1;
  PyGILState_STATE gstate = PyGILState_Ensure ();
  PyObject *gc = PyImport_ImportModule ("gc");

  if (gc) {
    PyObject *collected = PyObject_CallMethod (gc, "collect", NULL);
    PyObject *objects = PyObject_CallMethod (gc, "get_objects", NULL);

    if (objects)
      count = PyList_Size (objects);
    Py_XDECREF (objects);
    Py_XDECREF (collected);
    Py_DECREF (gc);
  }
  PyErr_Clear ();
  PyGILState_Release (gstate);

  return count;
}

/**
 * @brief Number of entries in sys.path.
 * @return the number of entries, or -1 if the interpreter cannot answer.
 */
static inline Py_ssize_t
py_test_sys_path_length (void)
{
  Py_ssize_t length = -1;
  PyGILState_STATE gstate = PyGILState_Ensure ();
  PyObject *path = PySys_GetObject ("path");

  if (path && PyList_Check (path))
    length = PyList_Size (path);
  PyGILState_Release (gstate);

  return length;
}

/**
 * @brief Reference count of an attribute of an imported module.
 * @param module The module name, which has to be imported already.
 * @param attr The attribute name.
 * @return the reference count without the reference this function takes, or -1 if not found.
 */
static inline Py_ssize_t
py_test_attr_refcount (const char *module, const char *attr)
{
  Py_ssize_t count = -1;
  PyGILState_STATE gstate = PyGILState_Ensure ();
  PyObject *modules = PyImport_GetModuleDict ();
  PyObject *mod = modules ? PyDict_GetItemString (modules, module) : NULL;
  PyObject *obj = mod ? PyObject_GetAttrString (mod, attr) : NULL;

  if (obj) {
    count = Py_REFCNT (obj) - 1;
    Py_DECREF (obj);
  }
  PyErr_Clear ();
  PyGILState_Release (gstate);

  return count;
}

/**
 * @brief Set an environment variable as the embedded interpreter sees it.
 * @param name The variable name.
 * @param value The value.
 * @return TRUE if os.environ was updated.
 * @note os.environ is a copy taken when the interpreter imports os, so g_setenv() does not reach a script.
 */
static inline gboolean
py_test_setenv (const char *name, const char *value)
{
  gboolean done = FALSE;
  PyGILState_STATE gstate = PyGILState_Ensure ();
  PyObject *os = PyImport_ImportModule ("os");
  PyObject *environ = os ? PyObject_GetAttrString (os, "environ") : NULL;
  PyObject *key = PyUnicode_FromString (name);
  PyObject *val = PyUnicode_FromString (value);

  if (environ && key && val)
    done = (PyObject_SetItem (environ, key, val) == 0);
  Py_XDECREF (val);
  Py_XDECREF (key);
  Py_XDECREF (environ);
  Py_XDECREF (os);
  PyErr_Clear ();
  PyGILState_Release (gstate);

  return done;
}

#endif /* __NNS_UNITTEST_PYTHON3_UTIL_H__ */
