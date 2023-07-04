configure-tflite:
	mkdir -p build_tflite 
	cmake tensorflow/tensorflow/lite -B build_tflite/  \
	  find build-DSYSTEM_FARMHASH=OFF \
	  -DTFLITE_ENABLE_GPU=ON \
	  -DCMAKE_BUILD_TYPE=Debug \
	  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON 

build-tflite:
	cmake --build build_tflite/ -j 4 

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

package-tflite:
	mkdir -p package_tflite/lib/
	mkdir -p package_tflite/include/
	install build_tflite/libtensorflow-lite.a package_tflite/lib/
	install build_tflite/_deps/cpuinfo-build/libcpuinfo.a package_tflite/lib/
	install build_tflite/_deps/xnnpack-build/libXNNPACK.a package_tflite/lib/
	install build_tflite/pthreadpool/libpthreadpool.a package_tflite/lib/
	install build_tflite/_deps/farmhash-build/libfarmhash.a package_tflite/lib/
	install build_tflite/_deps/fft2d-build/libfft2d_fftsg.a package_tflite/lib/
	install build_tflite/_deps/fft2d-build/libfft2d_fftsg2d.a package_tflite/lib/
	install build_tflite/_deps/flatbuffers-build/libflatbuffers.a package_tflite/lib/
	install build_tflite/_deps/abseil-cpp-build/absl/*/libabsl_*.a package_tflite/lib/
	install build_tflite/_deps/ruy-build/ruy/libruy*.a package_tflite/lib/
	(cd  tensorflow/ && find tensorflow/lite -type f -name "*.h" -exec install -Dm 755 "{}" "../package_tflite/include/{}" \;)
	(cd  build_tflite/flatbuffers/include/ && find flatbuffers -type f -name "*.h" -exec install -Dm 755 "{}" "../package_tflite/include/{}" \;)
	mkdir -p package_tflite/lib/pkgconfig/
	echo "Name: tensorflow2-lite\n\
Description: tensorflow lite static library\n\
Version: 2.13.0-RC2\n\
Libs: -L$(mkfile_dir)/package_tflite/lib/ -ltensorflow-lite -lflatbuffers \
-labsl_hash -labsl_status -labsl_strings -labsl_strings_internal -labsl_raw_hash_set -labsl_time \
-labsl_raw_logging_internal -labsl_int128 -labsl_cord -labsl_cord_internal -labsl_cordz_info -labsl_crc_cord_state \
-labsl_crc32c -labsl_cordz_handle -labsl_crc_internal -labsl_synchronization -labsl_base -labsl_str_format_internal \
-labsl_throw_delegate -labsl_examine_stack -labsl_malloc_internal -labsl_city -labsl_low_level_hash \
-labsl_cordz_functions -labsl_strerror -labsl_graphcycles_internal -labsl_spinlock_wait -labsl_time_zone \
-labsl_symbolize -labsl_exponential_biased -labsl_stacktrace -labsl_debugging_internal -labsl_symbolize \
-labsl_demangle_internal \
-lruy_kernel_avx -lruy_kernel_avx2_fma -lruy_pack_avx -lruy_pack_avx512 -lruy_pack_avx2_fma \
-lruy_kernel_avx512 -lruy_have_built_path_for_avx -lruy_have_built_path_for_avx512 -lruy_have_built_path_for_avx2_fma \
-lruy_kernel_arm -lruy_pack_arm -lruy_trmul -lruy_thread_pool -lruy_denormal -lruy_frontend \
-lruy_context -lruy_ctx -lruy_apply_multiplier -lruy_allocator -lpthreadpool -lruy_block_map -lruy_trmul \
-lruy_blocking_counter -lruy_cpuinfo -lruy_context_get_ctx -lruy_prepacked_cache -lruy_tune -lruy_wait \
-lruy_system_aligned_alloc -lruy_prepare_packed_matrices \
-lcpuinfo -lXNNPACK -lfft2d_fftsg -lfft2d_fftsg2d -lfarmhash\n\
Cflags: -I$(mkfile_dir)/package_tflite/include/  " \
> package_tflite/lib/pkgconfig/tensorflow2-lite.pc
