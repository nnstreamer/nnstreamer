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
#include <glib/gstdio.h>

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

/**
 * @brief Make nnstreamer_python unimportable, as in a tree installed by meson without the module on sys.path.
 * @param script The module name of the script to open next, dropped from sys.modules so that its own
 *        import of nnstreamer_python runs again; NULL if the script does not import it.
 * @return TRUE if importing it fails afterwards; FALSE also if no python sub-plugin has started the interpreter.
 * @note sys.modules['nnstreamer_python'] = None makes import raise ModuleNotFoundError; a sub-plugin
 *       replaces that entry with the module it creates. py_test_restore_api_module() drops the entry.
 */
static inline gboolean
py_test_hide_api_module (const char *script)
{
  gboolean hidden = FALSE;
  PyGILState_STATE gstate;
  gchar *code;

  if (!Py_IsInitialized ())
    return FALSE;

  code = g_strdup_printf ("import sys\nsys.modules.pop ('%s', None)\nsys.modules['nnstreamer_python'] = None\n",
      script ? script : "nnstreamer_python");
  gstate = PyGILState_Ensure ();
  if (PyRun_SimpleString (code) == 0) {
    PyObject *mod = PyImport_ImportModule ("nnstreamer_python");

    hidden = (mod == NULL);
    Py_XDECREF (mod);
  }
  PyErr_Clear ();
  PyGILState_Release (gstate);
  g_free (code);

  return hidden;
}

/**
 * @brief Drop nnstreamer_python from sys.modules, so that the next import loads it from sys.path again.
 * @return TRUE if sys.modules was updated.
 */
static inline gboolean
py_test_restore_api_module (void)
{
  gboolean done;
  PyGILState_STATE gstate;

  if (!Py_IsInitialized ())
    return FALSE;

  gstate = PyGILState_Ensure ();
  done = (PyRun_SimpleString ("import sys\nsys.modules.pop ('nnstreamer_python', None)\n") == 0);
  PyErr_Clear ();
  PyGILState_Release (gstate);

  return done;
}

/**
 * @brief Take what py_test_put_broken_api_module() set, or the part of it that was set, out of the interpreter.
 * @return TRUE if the restore ran.
 */
static inline gboolean
_py_test_undo_broken_api_module (void)
{
  gboolean done;
  PyGILState_STATE gstate = PyGILState_Ensure ();

  done = (PyRun_SimpleString ("import importlib, sys\n"
                              "if getattr (sys, '_nns_test_dir', None) in sys.path:\n"
                              "  sys.path.remove (sys._nns_test_dir)\n"
                              "if hasattr (sys, '_nns_test_dwb'):\n"
                              "  sys.dont_write_bytecode = sys._nns_test_dwb\n"
                              "sys.__dict__.pop ('_nns_test_dir', None)\n"
                              "sys.__dict__.pop ('_nns_test_dwb', None)\n"
                              "sys.modules.pop ('nnstreamer_python', None)\n"
                              "importlib.invalidate_caches ()\n")
          == 0);
  PyErr_Clear ();
  PyGILState_Release (gstate);

  return done;
}

/**
 * @brief Put a nnstreamer_python.py first on sys.path, as a module file that fails to load.
 * @param body The python source of that file; NULL for one that raises ImportError.
 * @return The temporary directory holding it, to pass to py_test_remove_broken_api_module(); NULL on failure.
 * @note No bytecode is written while it is there, so that the directory can be removed.
 */
static inline gchar *
py_test_put_broken_api_module (const char *body)
{
  gchar *dir;
  gchar *file;
  gboolean done = FALSE;

  if (!Py_IsInitialized () || (dir = g_dir_make_tmp ("nns-py-XXXXXX", NULL)) == NULL)
    return NULL;

  file = g_build_filename (dir, "nnstreamer_python.py", NULL);
  if (g_file_set_contents (file, body ? body : "raise ImportError ('broken')\n", -1, NULL)) {
    PyGILState_STATE gstate = PyGILState_Ensure ();
    PyObject *entry = PyUnicode_FromString (dir);

    done = entry && PySys_SetObject ("_nns_test_dir", entry) == 0
           && PyRun_SimpleString ("import importlib, sys\nsys._nns_test_dwb = sys.dont_write_bytecode\n"
                                  "sys.dont_write_bytecode = True\nsys.path.insert (0, sys._nns_test_dir)\n"
                                  "sys.modules.pop ('nnstreamer_python', None)\nimportlib.invalidate_caches ()\n")
                  == 0;
    Py_XDECREF (entry);
    PyErr_Clear ();
    PyGILState_Release (gstate);
    if (!done)
      _py_test_undo_broken_api_module ();
  }

  if (!done) {
    g_remove (file);
    g_rmdir (dir);
    g_clear_pointer (&dir, g_free);
  }
  g_free (file);

  return dir;
}

/**
 * @brief Undo py_test_put_broken_api_module() and free the directory name.
 * @param dir The directory it returned.
 * @return TRUE if sys.path and sys.dont_write_bytecode are restored and the directory is removed.
 */
static inline gboolean
py_test_remove_broken_api_module (gchar *dir)
{
  gchar *file = g_build_filename (dir, "nnstreamer_python.py", NULL);
  gboolean done = _py_test_undo_broken_api_module ();

  done = (g_remove (file) == 0) && done;
  done = (g_rmdir (dir) == 0) && done;
  g_free (file);
  g_free (dir);

  return done;
}

/**
 * @brief Whether a nnstreamer_python module file can be found on sys.path, whatever sys.modules holds.
 * @return TRUE if the path finder finds it.
 */
static inline gboolean
py_test_api_module_on_sys_path (void)
{
  gboolean found = FALSE;
  PyGILState_STATE gstate = PyGILState_Ensure ();
  PyObject *machinery = PyImport_ImportModule ("importlib.machinery");
  PyObject *finder = machinery ? PyObject_GetAttrString (machinery, "PathFinder") : NULL;
  PyObject *spec = finder ? PyObject_CallMethod (finder, "find_spec", "s", "nnstreamer_python") : NULL;

  found = (spec != NULL && spec != Py_None);
  Py_XDECREF (spec);
  Py_XDECREF (finder);
  Py_XDECREF (machinery);
  PyErr_Clear ();
  PyGILState_Release (gstate);

  return found;
}

/**
 * @brief Whether nnstreamer_python in sys.modules is the one created in-process, not loaded from a file.
 * @return TRUE if the module is in sys.modules and has no __file__.
 */
static inline gboolean
py_test_api_module_in_process (void)
{
  gboolean in_process = FALSE;
  PyGILState_STATE gstate = PyGILState_Ensure ();
  PyObject *modules = PyImport_GetModuleDict ();
  PyObject *mod = modules ? PyDict_GetItemString (modules, "nnstreamer_python") : NULL;

  if (mod && mod != Py_None)
    in_process = !PyObject_HasAttrString (mod, "__file__");
  PyErr_Clear ();
  PyGILState_Release (gstate);

  return in_process;
}

#endif /* __NNS_UNITTEST_PYTHON3_UTIL_H__ */
