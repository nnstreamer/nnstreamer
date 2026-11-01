# Asynchronous tensor-filter output

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
