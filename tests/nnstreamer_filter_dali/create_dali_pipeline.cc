/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Tool to create a dali pipeline for testing purposes.
 * Copyright (C) 2024 Bram Veldhoen
 */
/**
 * @file        create_dali_pipeline.cc
 * @date        Jun 2024
 * @brief       Tool to create a dali pipeline for testing purposes.
 * @see         http://github.com/nnstreamer/nnstreamer
 * @see         https://github.com/NVIDIA/DALI
 * @author      Bram Veldhoen
 * @bug         No known bugs except for NYI items
 *
 * Tool to create a dali pipeline for testing purposes.
 *
 */

#define _GLIBCXX_USE_CXX11_ABI 0

#include <dali/pipeline/pipeline.h>
#include <dali/pipeline/workspace/workspace.h>
#include "dali/operators.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace dali;

void
usage ()
{
  std::cout << "Usage: create_dali_pipeline <output_folder> [<H> [<W> [<C>]]]" << std::endl;
  std::cout << "  output_folder: Folder in which to store the created pipeline binary file."
            << std::endl;
  std::cout << "  H: Output height of the preprocessed image." << std::endl;
  std::cout << "  W: Output width of the preprocessed image." << std::endl;
  std::cout << "  C: Number of channels of the preprocessed image." << std::endl;
}

int
main (int argc, char *argv[])
{
  if ((argc < 2) || (argc > 5)) {
    usage ();
    return EXIT_FAILURE;
  }

  std::filesystem::path output_folder{ argv[1] };
  if (!std::filesystem::exists (output_folder)) {
    std::cerr << "Specified output path does not exist: " << argv[1] << std::endl
              << std::endl;
    usage ();
    return EXIT_FAILURE;
  }

  int H_out{ 320 };
  int W_out{ 320 };
  int C{ 3 };
  if (argc >= 3) {
    H_out = atoi (argv[2]);
    if (H_out == 0) {
      std::cerr << "Specified H_out invalid: " << argv[2] << std::endl
                << std::endl;
      usage ();
      return EXIT_FAILURE;
    }
  }
  if (argc >= 4) {
    W_out = atoi (argv[3]);
    if (W_out == 0) {
      std::cerr << "Specified W_out invalid: " << argv[3] << std::endl
                << std::endl;
      usage ();
      return EXIT_FAILURE;
    }
  }
  if (argc >= 5) {
    C = atoi (argv[4]);
    if (C == 0) {
      std::cerr << "Specified C invalid: " << argv[4] << std::endl << std::endl;
      usage ();
      return EXIT_FAILURE;
    }
  }

  InitOperatorsLib ();

  int max_batch_size{ 1 };
  int num_threads{ 1 };
  int device_id{ 0 };
  int seed = { -1 };
  bool pipelined_execution{ true };
  int prefetch_queue_depth{ 1 };
  bool async_execution{ true };

  std::string serialized_pipe_filename ("dali_pipeline_N_" + std::to_string (C)
                                        + "_" + std::to_string (H_out) + "_"
                                        + std::to_string (W_out) + ".bin");
  auto output_path = output_folder / serialized_pipe_filename;

  // Create the pipeline
  Pipeline pipe (max_batch_size, num_threads, device_id, seed,
      pipelined_execution, prefetch_queue_depth, async_execution);

  // Add pipeline operators
  // Input: (N, H, W, C), dtype=uint8, Output: (N, H, W, C), dtype=uint8
  pipe.AddOperator (OpSpec ("ExternalSource")
                        .AddArg ("batch", false)
                        .AddArg ("device", "cpu")
                        .AddArg ("dtype", DALIDataType::DALI_UINT8)
                        .AddArg ("name", "input0")
                        .AddArg ("ndim", 4)
                        .AddOutput ("input0", "cpu"),
      "input0");
  // Input: (N, H, W, C), dtype=uint8, Output: (N, H, W, C), dtype=uint8
  pipe.AddOperator (OpSpec ("MakeContiguous")
                        .AddArg ("device", "mixed")
                        .AddInput ("input0", "cpu")
                        .AddOutput ("input0_gpu", "gpu"));
  // Input: (N, H, W, C), dtype=uint8, Output: (N, W_out, H_out, C), dtype=uint8
  pipe.AddOperator (OpSpec ("Resize")
                        .AddArg ("antialias", false)
                        .AddArg ("device", "gpu")
                        .AddArg ("dtype", DALIDataType::DALI_UINT8)
                        .AddArg ("interp_type", DALIInterpType::DALI_INTERP_LINEAR)
                        .AddArg ("resize_x", static_cast<float> (W_out))
                        .AddArg ("resize_y", static_cast<float> (H_out))
                        .AddInput ("input0_gpu", "gpu")
                        .AddOutput ("resized", "gpu"));
  // Input: (N, W_out, H_out, C), dtype=uint8, Output: (N, W_out, H_out, C), dtype=float32
  pipe.AddOperator (OpSpec ("ArithmeticGenericOp")
                        .AddArg ("device", "gpu")
                        .AddArg ("expression_desc", "fdiv(&0 $0:int32)")
                        .AddArg ("integer_constants", std::vector<int>{ 255 })
                        .AddInput ("resized", "gpu")
                        .AddOutput ("normalized", "gpu"));
  // Input: (N, W_out, H_out, C), dtype=float32, Output: (N, C, W_out, H_out), dtype=float32
  std::vector<int> perm{ 0, 3, 1, 2 };
  pipe.AddOperator (OpSpec ("Transpose")
                        .AddArg ("device", "gpu")
                        .AddArg ("output_layout", "FCHW")
                        .AddArg ("perm", perm)
                        .AddInput ("normalized", "gpu")
                        .AddOutput ("output0", "gpu"));

  // Build the pipeline
  std::vector<std::pair<std::string, std::string>> outputs = { { "output0", "gpu" } };
  pipe.Build (outputs);

  // Serialize the pipeline
  auto serialized_pipe_str = pipe.SerializeToProtobuf ();

  // Save to file
  std::ofstream serialized_pipe_out (output_path);
  serialized_pipe_out << serialized_pipe_str;
  serialized_pipe_out.close ();

  // Deserialize the pipeline
  Pipeline pipe2 (serialized_pipe_str);
  pipe2.Build ();

  return 0;
}
