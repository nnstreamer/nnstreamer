CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# check whether TENSORFLOW_LITE is available.
# DISABLE_TENSORFLOW_LITE is defined at /debian/rules according to the build environment
IF(DISABLE_TENSORFLOW_LITE) # NOT AVAILABE
	ADD_DEFINITIONS(-DDISABLE_TENSORFLOW_LITE)
	ADD_LIBRARY(tensor_filter SHARED
		tensor_filter.c
		tensor_filter_custom.c
		)
	TARGET_LINK_LIBRARIES(tensor_filter dl ${pkgs_LIBRARIES})
	TARGET_INCLUDE_DIRECTORIES(tensor_filter PUBLIC ${pkgs_INCLUDE_DIRS})
	TARGET_COMPILE_OPTIONS(tensor_filter PUBLIC ${pkgs_CFLAGS_OTHER})

	INSTALL(TARGETS tensor_filter
		RUNTIME DESTINATION ${EXEC_PREFIX}
		LIBRARY DESTINATION ${GST_INSTALL_DIR}
		ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
		)
ELSE(DISABLE_TENSORFLOW_LITE) # AVAILABE
	ADD_LIBRARY(tensor_filter SHARED
		tensor_filter.c
		tensor_filter_tensorflow_lite.c
		tensor_filter_custom.c
		)
	TARGET_LINK_LIBRARIES(tensor_filter tensor_filter_tflitecore dl ${pkgs_LIBRARIES})
	TARGET_INCLUDE_DIRECTORIES(tensor_filter PUBLIC ${pkgs_INCLUDE_DIRS})
	TARGET_COMPILE_OPTIONS(tensor_filter PUBLIC ${pkgs_CFLAGS_OTHER})

	ADD_LIBRARY(tensor_filter_tflitecore SHARED
		tensor_filter_tensorflow_lite_core.cc
		)

	TARGET_LINK_LIBRARIES(tensor_filter_tflitecore ${pkgs_LIBRARIES} tensorflow-lite)
	TARGET_INCLUDE_DIRECTORIES(tensor_filter_tflitecore PUBLIC ${pkgs_INCLUDE_DIRS})
	TARGET_COMPILE_OPTIONS(tensor_filter_tflitecore PUBLIC ${pkgs_CFLAGS_OTHER})

	INSTALL(TARGETS tensor_filter tensor_filter_tflitecore
		RUNTIME DESTINATION ${EXEC_PREFIX}
		LIBRARY DESTINATION ${GST_INSTALL_DIR}
		ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
		)
ENDIF(DISABLE_TENSORFLOW_LITE)
