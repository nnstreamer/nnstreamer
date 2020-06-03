/**
 * GStreamer Tensor_filter, C++ Subplugin Support. (this is not a subplugin)
 * Copyright (C) 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	tensor_filter_support_cc.h
 * @date	15 Jan 2020
 * @brief	Base class for tensor_filter subplugins of C++ classes.
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * @details
 *    This is not a subplugin, but a helper for C++ subplugins.
 *    If you want to write a wrapper for neural network frameworks
 *    or a hardware adaptor in C++, this is what you want.
 *    If you want to attach a single C++ class/object as a filter
 *    (a.k.a. custom filter) to a pipeline, use tensor_filter_cpp
 *
 * To Packagers:
 *
 * This is to be exposed with "nnstreamer-c++-dev"
 */
#ifndef __NNS_TENSOR_FILTER_CPP_SUBPLUGIN_SUPPORT_H__
#define __NNS_TENSOR_FILTER_CPP_SUBPLUGIN_SUPPORT_H__
#ifdef __cplusplus

#include <stdexcept>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <nnstreamer_plugin_api_filter.h>
#if __cplusplus < 201103L
#warn C++11 is required for safe execution
#else
#include <type_traits>
#endif


namespace nnstreamer {
/**
 * @brief The base class for C++ subplugins. Derive this to write one.
 *
 * @detail The subplugin writers (derived class writer) are supposed to
 *        write virtual functions for their subplugins. They may
 *        add their own data structures as well in the derived class.
 *         Unlike C-version, constructing an object will automatically
 *        register(probe) the subplugin for nnstreamer.
 *         Optional virtual functions (non pure virtual functions) may
 *        be kept un-overriden if you don't support such.
 *         For getInput/Output and setInput, return -EINVAL if you don't
 *        support it.
 *
 *         We support tensor_filter_subplugin V1 only. (NNStreamer 1.5.1 or higher)
 **/
class tensor_filter_subplugin {
  private: /** Derived classes should NEVER access these */
    const uint64_t sanity; /**< Checks if dlopened obejct is really tensor_filter_subplugin */

    static const GstTensorFilterFramework fwdesc_template; /**< Template for fwdesc. Each subclass or object may have different subplugin_data while the C callbacks are identical. Thus, we copy C callbacks from this template to fwdesc and let each subplugin start customizing based on fwdesc, not on fwdesc_template */

    /** Helper function */
    static inline tensor_filter_subplugin * get_tfsp_with_checks (void * ptr);
    /** tensor_filter/C wrapper functions */
    static int cpp_open (const GstTensorFilterProperties * prop, void **private_data); /**< C wrapper func, open */
    static void cpp_close (const GstTensorFilterProperties * prop, void **private_data); /**< C wrapper func, close */
    static int cpp_invoke (const GstTensorFilterFramework *tf, const GstTensorFilterProperties *prop, void *private_data, const GstTensorMemory *input, GstTensorMemory *output); /**< C V1 wrapper func, invoke */
    static int cpp_getFrameworkInfo (const GstTensorFilterFramework *tf, const GstTensorFilterProperties * prop, void *private_data, GstTensorFilterFrameworkInfo *fw_info); /**< C V1 wrapper func, getFrameworkInfo */
    static int cpp_getModelInfo (const GstTensorFilterFramework *tf, const GstTensorFilterProperties * prop, void *private_data, model_info_ops ops, GstTensorsInfo *in_info, GstTensorsInfo *out_info); /**< C V1 wrapper func, getModelInfo */
    static int cpp_eventHandler (const GstTensorFilterFramework *tf, const GstTensorFilterProperties * prop, void *private_data, event_ops ops, GstTensorFilterFrameworkEventData *data); /**< C V1 wrapper func, eventHandler */

    GstTensorFilterFramework fwdesc; /**< Represents C/V1 wrapper for the derived class and its objects. Derived should not access this anyway; the base class will handle this with the C wrapper functions, base static-functions, and base constructors/destructors. */

  protected: /** Derived classes should call these at init/exit */
/**
 * @brief Register the subplugin "derived" class.
 * @detail A derived class MUST register itself with this function in order
 *         to be available for nnstreamer pipelines, i.e., at its init().
 *         The derived class type should be the template typename.
 * @retval Returns an "emptyInstnace" of the derived class. It is recommended
 *         to keep the object and feed to the unregister function.
 */
    template<typename T>
    static T * register_subplugin () {
#if __cplusplus < 201103L
#warn C++11 is required for safe execution
#else
      /** The given class T should be derived from tensor_filter_subplugin */
      assert ((std::is_base_of<tensor_filter_subplugin, T>::value));
#endif
      T *emptyInstance = new T();

      assert (emptyInstance);
      memcpy (&emptyInstance->fwdesc, &fwdesc_template,
          sizeof (fwdesc_template));
      emptyInstance->fwdesc.subplugin_data = emptyInstance;
      nnstreamer_filter_probe (&emptyInstance->fwdesc);

      return emptyInstance;
    }

/**
 * @brief Unregister the registered "derived" class.
 * @detail The registered derived class may unregister itself if it can
 *         guarantee that the class won't be used anymore; i.e., at its exit().
 *         The derived class type should be the template typename.
 * @param [in] emptyInstance An emptyInstance that mey be "delete"d by this
 *             function. It may be created by getEmptyInstance() or the one
 *             created by register_subplugin(); It is recommended to keep
 *             the object created by register_subplugin() and feed it to
 *             unregister_subplugin().
 */
    template<typename T>
    static void unregister_subplugin (T * emptyInstance) {
      GstTensorFilterFrameworkInfo info;
#if __cplusplus < 201103L
#warn C++11 is required for safe execution
#else
      /** The given class T should be derived from tensor_filter_subplugin */
      assert ((std::is_base_of<tensor_filter_subplugin, T>::value));
#endif
      assert (emptyInstance);

      emptyInstance->getFrameworkInfo (info);
      nnstreamer_filter_exit (info.name);

      delete emptyInstance;
    }

  public:
    /*************************************************************
     ** These should be filled/implemented by subplugin authors **
     *************************************************************/
    tensor_filter_subplugin ();
        /**< Creates a non-functional "empty" object
             Subplugin (derived class) should make a constructor with same role and input arguments!
          */

    virtual tensor_filter_subplugin & getEmptyInstance () = 0;
        /**< Return a newly created "empty" object of the derived class. */

    virtual void configure_instance (const GstTensorFilterProperties *prop) = 0;
        /**< Configure a non-functional "empty" object as a
             functional "filled" object ready to be invoked.

             Possible exceptions: ... @todo describe them!
         */

    virtual ~tensor_filter_subplugin (); /**< called by Close */

    virtual void invoke (const GstTensorMemory *input, GstTensorMemory *output) = 0;
        /**< Mandatory virtual method.  Invoke NN.

             Possible exceptions: ... @todo describe them!
         */

    virtual void getFrameworkInfo (GstTensorFilterFrameworkInfo &info) = 0;
        /**< Mandatory virtual method.
          * An empty instance created by derived() should support this with
          * its default info value.
          * "name" property of info should NEVER be changed.
          * info.name is deleted and reallocated if info.name is not null.
          *
          * Possible exceptions: ... @todo describe them!
          */

    virtual int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info) = 0;
        /**< Mandatory virtual method.
          *  For a given opened model (an instance of the derived class),
          * provide input/output dimensions.
          *  At least one of the two possible ops should be available.
          *  Return -ENOENT if the ops is not available by this object
          *  Return -EINVAL if it is an invalid request.
          */

    virtual int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
        /**< Optional. If not implemented, no event is handled
          *  Return -ENOENT if the ops is not to be handled by this object
          *                 (e.g., let the framework do "free")
          *  Return -EINVAL if it is an invalid request.
          */
};

} /* namespace nnstreamer */

#endif /* __cplusplus */
#endif /* __NNS_TENSOR_FILTER_CPP_SUBPLUGIN_SUPPORT_H__ */
