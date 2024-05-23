/* SPDX-License-Identifier: LGPL-2.1-only */

/**
 * @file    tensor_filter_executorch.cc
 * @date    26 Apr 2024
 * @brief   NNStreamer tensor-filter sub-plugin for ExecuTorch
 * @author  
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs.
 *
 * This is the executorch plugin for tensor_filter.
 *
 * @note Currently only skeleton
 * 
 **/


#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

#include <iostream>
#include <memory>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

// static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

// DEFINE_string(
//     model_path,
//     "model.pte",
//     "Model serialized in flatbuffer format.");

using namespace torch::executor;
using torch::executor::util::FileDataLoader;

namespace nnstreamer
{
namespace tensorfilter_executorch
{

G_BEGIN_DECLS

void init_filter_executorch (void) __attribute__ ((constructor));
void fini_filter_executorch (void) __attribute__ ((destructor));

G_END_DECLS

/**
 * @brief Class for executorch subplugin (skeleton)
 */

class executorch_subplugin final: public tensor_filter_subplugin
{
    public:
    static void init_filter_executorch (); /**< Dynamic library contstructor helper */
    static void fini_filter_executorch (); /**< Dynamic library desctructor helper */

    executorch_subplugin ();
    ~executorch_subplugin ();

    /**< Implementations of ncnn tensor_filter_subplugin */
    tensor_filter_subplugin &getEmptyInstance ();
    void configure_instance (const GstTensorFilterProperties *prop);
    void invoke (const GstTensorMemory *input, GstTensorMemory *output);
    void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
    int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
    int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);

    private:
    bool empty_model; /**< Empty (not initialized) model flag */
};

}
}