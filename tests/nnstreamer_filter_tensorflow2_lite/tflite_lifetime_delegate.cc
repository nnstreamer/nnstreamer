/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    tflite_lifetime_delegate.cc
 * @date    17 Sep 2026
 * @brief   TensorFlow Lite external delegate for tests, checking object lifetimes
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * TensorFlow Lite requires the delegate and the FlatBufferModel to outlive the
 * interpreter using them. This delegate takes over one node that reads a
 * constant (mmap-ed) tensor of the model. When the interpreter frees that
 * node, the kernel records a violation if its delegate has already been
 * destroyed or if the model memory has already been unmapped. Both checks
 * only look at bookkeeping and mincore(), so a violation is counted instead
 * of crashing the test.
 */

#include <cstdint>
#include <errno.h>
#include <set>
#include <stddef.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#ifdef USE_TENSORFLOW2_HEADER_PATH
#include <tensorflow2/lite/builtin_ops.h>
#include <tensorflow2/lite/c/common.h>
#else
#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/common.h>
#endif

extern "C" {
TfLiteDelegate *tflite_plugin_create_delegate (const char *const *options_keys,
    const char *const *options_values, size_t num_options,
    void (*report_error) (const char *));
void tflite_plugin_destroy_delegate (TfLiteDelegate *delegate);
int nns_tflite_lifetime_delegate_get_count (const char *name);
void nns_tflite_lifetime_delegate_reset (void);
}

/**
 * @brief The delegate object with the id used by its kernels.
 */
typedef struct {
  TfLiteDelegate base; /**< must be the first member */
  int id; /**< delegate id, never reused */
} lifetime_delegate_s;

/**
 * @brief The state of a delegate kernel.
 */
typedef struct {
  int delegate_id; /**< id of the delegate that created this kernel */
  void *model_page; /**< page of a constant tensor in the model mapping */
} lifetime_kernel_s;

static std::set<int> live_delegates;
static int last_delegate_id = 0;
static int kernels_prepared = 0;
static int kernels_freed = 0;
static int model_checks = 0;
static int violations = 0;
static int options_given = 0;
static bool refuse_prepare = false;

/**
 * @brief Find an input of the node that points into the mmap-ed model.
 */
static const TfLiteTensor *
find_mmap_input (TfLiteContext *context, const TfLiteIntArray *inputs)
{
  for (int i = 0; i < inputs->size; i++) {
    int idx = inputs->data[i];

    if (idx < 0)
      continue;
    const TfLiteTensor *t = &context->tensors[idx];
    if (t->allocation_type == kTfLiteMmapRo && t->data.raw && t->bytes > 0)
      return t;
  }
  return nullptr;
}

/**
 * @brief Kernel init: remember the delegate and a page of the model.
 */
static void *
kernel_init (TfLiteContext *context, const char *buffer, size_t length)
{
  const TfLiteDelegateParams *params = (const TfLiteDelegateParams *) buffer;
  lifetime_kernel_s *kernel = new lifetime_kernel_s ();
  const TfLiteTensor *t = find_mmap_input (context, params->input_tensors);
  long page_size = sysconf (_SC_PAGESIZE);
  (void) length;

  kernel->delegate_id = last_delegate_id;
  kernel->model_page = nullptr;
  if (t && page_size > 0)
    kernel->model_page
        = (void *) ((uintptr_t) t->data.raw & ~((uintptr_t) page_size - 1));

  kernels_prepared++;
  return kernel;
}

/**
 * @brief Kernel free: called while the interpreter is destroyed.
 */
static void
kernel_free (TfLiteContext *context, void *buffer)
{
  lifetime_kernel_s *kernel = (lifetime_kernel_s *) buffer;
  (void) context;

  if (live_delegates.count (kernel->delegate_id) == 0)
    violations++;

  if (kernel->model_page) {
    unsigned char vec;

    model_checks++;
    if (mincore (kernel->model_page, 1, &vec) != 0 && errno == ENOMEM)
      violations++;
  }

  kernels_freed++;
  delete kernel;
}

/**
 * @brief Kernel prepare: the output shapes come from the model.
 */
static TfLiteStatus
kernel_prepare (TfLiteContext *context, TfLiteNode *node)
{
  (void) context;
  (void) node;
  return kTfLiteOk;
}

/**
 * @brief Kernel invoke: the tests do not run this delegate.
 */
static TfLiteStatus
kernel_invoke (TfLiteContext *context, TfLiteNode *node)
{
  (void) context;
  (void) node;
  return kTfLiteError;
}

/**
 * @brief Delegate prepare: take over the first node reading a model constant.
 * @note "mode#fail" given in ExtDelegateKeyVal makes this refuse, so a test can
 *       reach the sub-plugin's delegate-application error path.
 */
static TfLiteStatus
delegate_prepare (TfLiteContext *context, TfLiteDelegate *delegate)
{
  TfLiteIntArray *plan;
  TfLiteNode *node;
  TfLiteRegistration *reg;
  TfLiteRegistration kernel = {};

  if (refuse_prepare)
    return kTfLiteError;

  if (context->GetExecutionPlan (context, &plan) != kTfLiteOk)
    return kTfLiteError;

  kernel.init = kernel_init;
  kernel.free = kernel_free;
  kernel.prepare = kernel_prepare;
  kernel.invoke = kernel_invoke;
  kernel.builtin_code = kTfLiteBuiltinDelegate;
  kernel.custom_name = "NNStreamerLifetimeDelegate";
  kernel.version = 1;

  for (int i = 0; i < plan->size; i++) {
    int node_index = plan->data[i];

    if (context->GetNodeAndRegistration (context, node_index, &node, &reg) != kTfLiteOk)
      return kTfLiteError;
    if (!find_mmap_input (context, node->inputs))
      continue;

    TfLiteIntArray *nodes = TfLiteIntArrayCreate (1);
    TfLiteStatus status;

    nodes->data[0] = node_index;
    status = context->ReplaceNodeSubsetsWithDelegateKernels (context, kernel, nodes, delegate);
    TfLiteIntArrayFree (nodes);
    return status;
  }

  return kTfLiteOk;
}

/**
 * @brief Create the delegate (external delegate interface).
 */
TfLiteDelegate *
tflite_plugin_create_delegate (const char *const *options_keys, const char *const *options_values,
    size_t num_options, void (*report_error) (const char *))
{
  lifetime_delegate_s *delegate = new lifetime_delegate_s ();
  (void) report_error;

  options_given = (int) num_options;
  refuse_prepare = false;
  for (size_t i = 0; i < num_options; i++) {
    if (!options_keys[i] || !options_values[i])
      continue;
    if (std::string (options_keys[i]) == "mode" && std::string (options_values[i]) == "fail")
      refuse_prepare = true;
  }

  delegate->base.data_ = delegate;
  delegate->base.Prepare = delegate_prepare;
  delegate->base.flags = kTfLiteDelegateFlagsNone;
  delegate->id = ++last_delegate_id;
  live_delegates.insert (delegate->id);

  return &delegate->base;
}

/**
 * @brief Destroy the delegate (external delegate interface).
 */
void
tflite_plugin_destroy_delegate (TfLiteDelegate *delegate)
{
  lifetime_delegate_s *d = (lifetime_delegate_s *) delegate;

  if (!d)
    return;

  live_delegates.erase (d->id);
  delete d;
}

/**
 * @brief Get a counter: "prepared", "freed", "model_checks", "violations",
 *        "options" (the key values the last create call was given) or "live"
 *        (delegates created and not destroyed yet).
 * @return the counter value, -1 for an unknown name.
 */
int
nns_tflite_lifetime_delegate_get_count (const char *name)
{
  const std::string key (name ? name : "");

  if (key == "live")
    return (int) live_delegates.size ();
  if (key == "prepared")
    return kernels_prepared;
  if (key == "freed")
    return kernels_freed;
  if (key == "model_checks")
    return model_checks;
  if (key == "violations")
    return violations;
  if (key == "options")
    return options_given;
  return -1;
}

/**
 * @brief Reset the counters.
 */
void
nns_tflite_lifetime_delegate_reset (void)
{
  kernels_prepared = 0;
  kernels_freed = 0;
  model_checks = 0;
  violations = 0;
  options_given = 0;
  refuse_prepare = false;
}
