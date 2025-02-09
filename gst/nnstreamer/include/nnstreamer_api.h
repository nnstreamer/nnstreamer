#ifndef __NNS_API_H__
#define __NNS_API_H__

#if (defined(_WIN32) || defined(__CYGWIN__)) && !defined(NNS_STATIC_COMPILATION)
#  define _NNS_EXPORT __declspec(dllexport)
#  define _NNS_IMPORT __declspec(dllimport)
// The below seems like correct behavior, but leaving out for now in order to maintain the behavior of
// the existing build
// #elif __GNUC__ >= 4
// #  define _NNS_EXPORT __attribute__((visibility("default")))
// #  define _NNS_IMPORT
#else
#  define _NNS_EXPORT
#  define _NNS_IMPORT
#endif

// TODO: setting NNS_API_IMPORTS is backwards and should be inverted (set NNS_API_EXPORTS instead)
// Did this to avoid having to track down all the libraries that produce headers with NNS_API
#ifdef NNS_API_IMPORTS
#define NNS_API _NNS_IMPORT
#else
#define NNS_API _NNS_EXPORT
#endif

#endif
