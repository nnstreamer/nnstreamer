# A mock of the SNPE SDK for the tensor_filter sub-plugin tests

`ext/nnstreamer/tensor_filter/tensor_filter_snpe.cc` and
`tensor_filter_snpe_v1.cc` are built only when the Qualcomm Neural Processing
SDK is installed, which no continuous integration job of this repository has.
This directory stands in for that SDK so that both sources are compiled and
unit tested on every pull request.

## Provenance and licensing

The SDK headers are a proprietary Qualcomm distribution and **none of them is
copied into this repository**. The declarations here were written from two
inputs:

- the call sites in the two sub-plugin sources, which fix the names, the
  argument counts and the argument types; and
- the public API reference for the Qualcomm Neural Processing SDK
  (<https://docs.qualcomm.com/>), which fixes the handle ownership rules and
  the meaning of each call.

Two API generations are covered, one per directory: `v1/` stands in for the
C++ API of SNPE 1.x, which `tensor_filter_snpe_v1.cc` uses, and `v2/` for the
C API of SNPE 2.x, which `tensor_filter_snpe.cc` uses. No point release of
either is pinned, and none needs to be: what has to exist here is decided by
the calls the two sub-plugin sources make, and the reference only settles what
each of those calls means. A newer SDK that keeps them needs no change here;
one that drops or renames one breaks the sub-plugin first, and the mock after.

Each header says at its top what it covers and what it leaves out. Enumerator
values are illustrative and the layout of the C++ classes is our own: binary
compatibility with the real SDK is not a goal, because a test binary uses these
headers together with the mock library as one pair, and never loads
`libSNPE.so`.

## What is emulated

One model with a single input and a single output, named `input` and `output`,
computing `output = input + 2`. A container file is never parsed; its name
decides the properties of the model, so the in-tree `add2_float.*.dlc` and
`add2_uint8.*.dlc` fixtures work unchanged, and a test that needs a shape they
do not have creates an empty file whose name carries a keyword. See
`snpe_mock_model_load()` in `snpe_mock_common.cc`.

`tests/meson.build` builds each sub-plugin source into a test binary of its
own, `unittest_filter_snpe_mock_v2` and `unittest_filter_snpe_mock_v1`. Both
also compile `../unittest_filter_snpe.cc`, so the cases written for a device
with the real SDK run here too; that file is therefore built into three
binaries, and a change to it has to keep working on all of them. Two binaries
rather than one because both sub-plugins register a sub-plugin named `snpe`.

`snpe_mock.h` adds what a test needs beyond the API itself: a live-instance
counter per object kind, a fault to inject, and a ledger of the GLib string
allocations the sub-plugin makes. The ledger interposes the allocators through
the linker's `--wrap` option and is therefore built only where that option
works. Ask `snpe_mock_ledger_available()` before asserting on it: an
interposer only catches a call that reaches the symbol it wraps, and GLib
inlines `g_strdup()` of a string the compiler knows, which an optimised build
then never calls, so that predicate measures one duplication rather than
assuming. For the same reason a test must duplicate a string it built at run
time, the way both sub-plugins do, and not a literal.

## A release the mock tolerates on purpose

The SNPE 2.x sub-plugin releases the buffer attributes of a quantized tensor
and then calls `Snpe_UserBufferEncodingTfN_Delete()` on the handle that
`Snpe_IBufferAttributes_GetEncoding_Ref()` returned, which belongs to the
attributes it has just released. Against the real SDK that is a use after free
followed by a release of a handle the caller does not own. It is tracked in
issue #5032 and is not fixed here, so the mock hands out one process-wide
encoding for that accessor and ignores a release of it. A test therefore
cannot see it, and `snpe_mock_over_release_count()` stays at zero. The fix of
that issue removes this section, together with the `@todo` at the call, and
makes the mock own that handle so a case can pin it.

## What is not emulated

The real runtime, in any of its forms. Nothing here reads a `.dlc` container,
runs a network, or touches a DSP, a GPU or an AIP core, and the mock reports
only the CPU runtime as available. A behaviour that depends on the real SDK,
such as how a genuine model quantizes or resizes, is outside what these tests
can say anything about. They cover the sub-plugin's own logic: option parsing,
tensor metadata, buffer setup, error handling and resource release.
