# Asynchronous tensor-filter output

Only a V1 sub-plugin that explicitly supports asynchronous output may be used
with `invoke-async=true`. After opening the model, the core sends
`CHECK_ASYNC_SUPPORT` through `eventHandler`. The handler must return 0 and set
`data.async_support = TRUE`; an unhandled query, an error, or a zero support
flag causes element and single-shot start to fail before invoking the model.
The error identifies the framework. The opened model is then closed. Synchronous
V1 sub-plugins need no changes, and V0 sub-plugins cannot enable asynchronous
invoke. A registered output callback alone is not a capability declaration.

The query does not invoke the model, allocate results, or dispatch callbacks.
For example, a C++ sub-plugin that implements the contract below can answer:

```cpp
if (ops == CHECK_ASYNC_SUPPORT) {
  data.async_support = TRUE;
  return 0;
}
```

This uses a new event and a member of the existing event-data union, leaving
framework-info and callback structure sizes unchanged. Existing asynchronous
out-of-tree sub-plugins must implement the query; a legacy handler returning
success for every unknown event does not opt in unless it sets the support flag.
The in-tree llama.cpp and ExecuTorch-Llama sub-plugins implement it.

Set `invoke-async` before starting the filter. Changing it while the model is
open is refused and leaves the current value unchanged. Stop the element by
setting it to NULL, or stop the single-shot filter, before selecting another
mode. This also preserves callback ownership for work already in flight.

With `invoke-async=true`, invoking a filter queues an input. An accepted input
does not supply a synchronous output buffer: output arrives only through the
registered callback. Input processing therefore does not inspect or allocate
from `prop->output_meta`. This also applies to the single-shot invocation API;
its synchronous output argument is not a callback result.

Workers producing variable-sized results should use
`nnstreamer_filter_dispatch_output_async_with_info(prop, output, info)`.
Construct a `GstTensorsInfo` for the result without modifying
`prop->output_meta`. The element uses this description for tensor counts,
flexible headers, output combination and cleanup. The single-shot callback
also releases the count described by this result, rather than the configured
model count. The llama.cpp streaming worker uses a local description for each
token; its synchronous path still updates the synchronous model output info.

Initialize the description with `gst_tensors_info_init()`, fill one valid
description per allocated output, dispatch it, and then release the description
with `gst_tensors_info_free()`. Keep the description and its tensor names alive
until dispatch returns. Metadata is borrowed synchronously; data pointers are
owned by the consumer after dispatch. The caller retains its memory-wrapper
array. If no callback exists, dispatch releases data with `g_free()`. Null
arguments or counts outside 1..NNS_TENSOR_SIZE_LIMIT are rejected before data
ownership transfers. A callback must treat the supplied description as read-only.

The existing `nnstreamer_filter_dispatch_output_async()` remains available for
source and binary compatibility. It dispatches using `prop->output_meta`.
Prefer the explicit-info API for workers to avoid sharing mutable output
descriptions with property and caps handling. Neither API serializes worker
dispatches or extends the filter's lifetime: serialize outputs for each
instance, stop and join workers before teardown, and do not change properties
or renegotiate caps concurrently with a legacy worker's metadata writes.

`unittest_filter_common` contains a model-free asynchronous C++ sub-plugin.
Its tests cover deferred output, multiple results per input, input bursts,
EOS, errors, suspension, independent result counts and dimensions, missing
memory and invalid descriptions. `concurrentLegacyMetadata` overlaps 256
additional inputs with 128 worker results of changing dimensions. It can be
run with a ThreadSanitizer build to detect input-side reads of worker metadata.
System GLib/GStreamer libraries must either be instrumented with compatible
sanitizers or accounted for separately when interpreting their diagnostics.
These model-free tests do not replace framework-specific model execution.

The invoke mode stays fixed for the entire active element or single-shot lifecycle,
even if the idle watchdog unloads its model. After an explicit stop, changing the
mode closes any model retained by a successful SUSPEND event. The next start or
single-shot invoke then opens and validates a model in the new mode. Setting the
same mode does not discard a retained model.
