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
#include <vector>

#include <executorch/cmake-out/kernels/portable/portable_ops_lib/RegisterCodegenUnboxedKernelsEverything.cpp>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/kernels/prim_ops/register_prim_ops.cpp>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

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
 * @brief tensor-filter-subplugin concrete class for ExecuTorch
 */

class executorch_subplugin final : public tensor_filter_subplugin
{
  private:
  static executorch_subplugin *registeredRepresentation;
  static const GstTensorFilterFrameworkInfo framework_info;

  bool configured;
  char *model_path; /**< The model *.pte file */
  void cleanup (); /**< cleanup function */
  GstTensorsInfo inputInfo; /**< Input tensors metadata */
  GstTensorsInfo outputInfo; /**< Output tensors metadata */

  /** executorch */
  std::unique_ptr<Result<FileDataLoader>> loader;
  std::unique_ptr<Result<Program>> program;
  std::unique_ptr<Result<Method>> method;
  const char *method_name;
  MemoryManager *memory_manager;
  MemoryAllocator *method_allocator;
  HierarchicalAllocator *planned_memory;
  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
  std::vector<Span<uint8_t>> planned_spans;
  Error fill_data (Tensor tensor, void *data);

  public:
  static void init_filter_executorch ();
  static void fini_filter_executorch ();

  executorch_subplugin ();
  ~executorch_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

/**
 * @brief Describe framework information.
 */
const GstTensorFilterFrameworkInfo executorch_subplugin::framework_info = { .name = "executorch",
  .allow_in_place = FALSE,
  .allocate_in_invoke = FALSE,
  .run_without_model = FALSE,
  .verify_model_path = TRUE,
  .hw_list = (const accl_hw[]){ ACCL_CPU },
  .num_hw = 1,
  .accl_auto = ACCL_CPU,
  .accl_default = ACCL_CPU,
  .statistics = nullptr };

/**
 * @brief Constructor for executorch subplugin.
 */
executorch_subplugin::executorch_subplugin () : tensor_filter_subplugin ()
{
  gst_tensors_info_init (std::addressof (inputInfo));
  gst_tensors_info_init (std::addressof (outputInfo));
}

/**
 * @brief Destructor for executorch subplugin.
 */
executorch_subplugin::~executorch_subplugin ()
{
  cleanup ();
}

/**
 * @brief Method to get empty object.
 */
tensor_filter_subplugin &
executorch_subplugin::getEmptyInstance ()
{
  return *(new executorch_subplugin ());
}

/**
 * @brief Method to cleanup executorch subplugin.
 */
void
executorch_subplugin::cleanup ()
{
  g_free (model_path);
  model_path = nullptr;

  if (!configured)
    return;

  gst_tensors_info_free (std::addressof (inputInfo));
  gst_tensors_info_free (std::addressof (outputInfo));

  method_name = nullptr;
  configured = false;
  delete method_allocator;
  delete planned_memory;
  delete memory_manager;
}

/**
 * @brief Method to prepare/configure ExecuTorch instance.
 */
void
executorch_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  /* get input / output info from properties */
  gst_tensors_info_copy (std::addressof (inputInfo), std::addressof (prop->input_meta));
  gst_tensors_info_copy (std::addressof (outputInfo), std::addressof (prop->output_meta));

  try {
    /* Load network (.pte file) */
    if (!g_file_test (prop->model_files[0], G_FILE_TEST_IS_REGULAR)) {
      const std::string err_msg
          = "Given file " + (std::string) prop->model_files[0] + " is not valid";
      throw std::invalid_argument (err_msg);
    }

    model_path = g_strdup (prop->model_files[0]);

    // Create a loader to get the data of the program file.
    loader = std::make_unique<Result<FileDataLoader>> (FileDataLoader::from (model_path));
    ET_CHECK_MSG (loader->ok (), "FileDataLoader::from() failed: 0x%" PRIx32,
        (uint32_t) loader->error ());

    // Parse the program file.
    program = std::make_unique<Result<Program>> (Program::load (&loader->get ()));
    if (!program->ok ()) {
      ET_LOG (Error, "Failed to parse model file %s", model_path);
      return;
    }
    ET_LOG (Info, "Model file %s is loaded.", model_path);

    // Use the first method in the program.
    const auto method_name_result = program->get ().get_method_name (0);
    ET_CHECK_MSG (method_name_result.ok (), "Program has no methods");
    method_name = *method_name_result;
    ET_LOG (Info, "Using method %s", method_name);

    // MethodMeta describes the memory requirements of the method.
    Result<MethodMeta> method_meta = program->get ().method_meta (method_name);
    ET_CHECK_MSG (method_meta.ok (), "Failed to get method_meta for %s: 0x%" PRIx32,
        method_name, (uint32_t) method_meta.error ());

    method_allocator
        = new MemoryAllocator (sizeof (method_allocator_pool), method_allocator_pool);


    size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers ();
    for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
      // .get() will always succeed because id < num_memory_planned_buffers.
      size_t buffer_size = static_cast<size_t> (
          method_meta->memory_planned_buffer_size (id).get ());
      ET_LOG (Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);
      planned_buffers.push_back (std::make_unique<uint8_t[]> (buffer_size));
      planned_spans.push_back ({ planned_buffers.back ().get (), buffer_size });
    }
    planned_memory = new HierarchicalAllocator (
        { planned_spans.data (), planned_spans.size () });

    // Assemble all of the allocators into the MemoryManager that the Executor
    // will use.
    memory_manager = new MemoryManager (method_allocator, planned_memory);

    //
    // Load the method from the program, using the provided allocators. Running
    // the method can mutate the memory-planned buffers, so the method should
    // only be used by a single thread at at time, but it can be reused.
    //

    method = std::make_unique<Result<Method>> (
        program->get ().load_method (method_name, memory_manager));
    ET_CHECK_MSG (method->ok (), "Loading of method %s failed with status 0x%" PRIx32,
        method_name, (uint32_t) method->error ());
    ET_LOG (Info, "Method loaded.");

    configured = true;
  } catch (const std::exception &e) {
    cleanup ();
    /* throw exception upward */
    throw;
  }
}

/**
 * @brief Method to execute the model.
 */
void
executorch_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  if (!input)
    throw std::runtime_error ("Invalid input buffer, it is NULL.");
  if (!output)
    throw std::runtime_error ("Invalid output buffer, it is NULL.");

  if (!method->ok ()) {
    throw std::runtime_error ("Method is not properly initialized.");
  }

  Method &method_ref = method->get ();

  // Set inputs
  MethodMeta method_meta = method_ref.method_meta ();
  size_t num_allocated = 0;
  void **inputs = (void **) malloc (sizeof (void *));

  Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta (0);

  void *data_ptr = malloc (tensor_meta->nbytes ());
  inputs[0] = data_ptr;

  TensorImpl impl = TensorImpl (tensor_meta->scalar_type (),
      tensor_meta->sizes ().size (),
      const_cast<TensorImpl::SizesType *> (tensor_meta->sizes ().data ()), data_ptr,
      const_cast<TensorImpl::DimOrderType *> (tensor_meta->dim_order ().data ()));
  Tensor t (&impl);

  Error err = fill_data (t, input->data);
  if (err == Error::InvalidArgument) {
    throw std::runtime_error ("Unsupported data type");
  }

  Error set_input_error = method_ref.set_input (t, 0);
  assert (set_input_error == Error::Ok);

  Result<util::BufferCleanup> inputs_prepared
      = util::BufferCleanup ({ inputs, num_allocated });
  ET_CHECK_MSG (inputs_prepared.ok (), "Could not prepare inputs: 0x%" PRIx32,
      (uint32_t) inputs_prepared.error ());
  ET_LOG (Info, "Inputs prepared.");

  // Execute the method
  Error status = method_ref.execute ();
  ET_CHECK_MSG (status == Error::Ok, "Execution of method %s failed with status 0x%" PRIx32,
      method_name, (uint32_t) status);
  ET_LOG (Info, "Model executed successfully.");

  // Get outputs
  std::vector<EValue> outputs (method_ref.outputs_size ());
  status = method_ref.get_outputs (outputs.data (), outputs.size ());
  ET_CHECK (status == Error::Ok);
  const auto &output_tensor = outputs[0].toTensor ();
  size_t output_tensor_size = output_tensor.nbytes ();
  std::memcpy (output->data, output_tensor.const_data_ptr (), output_tensor_size);
}

Error
executorch_subplugin::fill_data (Tensor tensor, void *data)
{
  switch (tensor.scalar_type ()) {
#define FILL_CASE(T, n)                                                      \
  case (torch::executor::ScalarType::n):                                     \
    {                                                                        \
      T *ptr = static_cast<T *> (data);                                      \
      std::copy (ptr, ptr + tensor.numel (), tensor.mutable_data_ptr<T> ()); \
      break;                                                                 \
    }
    ET_FORALL_REAL_TYPES_AND (Bool, FILL_CASE)
    default:
      ET_LOG (Error, "Unsupported scalar type %d", (int) tensor.scalar_type ());
      return Error::InvalidArgument;
  }
#undef FILL_CASE
  return Error::Ok;
}

/**
 * @brief Method to get the information of ExecuTorch subplugin.
 */
void
executorch_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info = executorch_subplugin::framework_info;
}

/**
 * @brief Method to get the model information.
 */
int
executorch_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputInfo));
    gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputInfo));
    return 0;
  }

  return -ENOENT;
}

/**
 * @brief Method to handle events.
 */
int
executorch_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);

  return -ENOENT;
}

executorch_subplugin *executorch_subplugin::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
executorch_subplugin::init_filter_executorch (void)
{
  runtime_init ();
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<executorch_subplugin> ();
}

/** @brief Destruct the subplugin */
void
executorch_subplugin::fini_filter_executorch (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/**
 * @brief Register the sub-plugin for ExecuTorch.
 */
void
init_filter_executorch ()
{
  executorch_subplugin::init_filter_executorch ();
}

/**
 * @brief Destruct the sub-plugin for ExecuTorch.
 */
void
fini_filter_executorch ()
{
  executorch_subplugin::fini_filter_executorch ();
}


} // namespace tensorfilter_executorch
} // namespace nnstreamer