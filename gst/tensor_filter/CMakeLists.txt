# check whether TENSORFLOW_LITE is available.
# DISABLE_TENSORFLOW_LITE is defined at /debian/rules according to the build environment

set(FILTER_SOURCE tensor_filter.c tensor_filter_custom.c)
set(FILTER_TARGET)

IF(DISABLE_TENSORFLOW_LITE) # NOT AVAILABLE
  ADD_DEFINITIONS(-DDISABLE_TENSORFLOW_LITE)
ENDIF(DISABLE_TENSORFLOW_LITE)

IF(DISABLE_TENSORFLOW) # NOT AVAILABLE
  ADD_DEFINITIONS(-DDISABLE_TENSORFLOW)
ENDIF(DISABLE_TENSORFLOW)

IF(NOT DISABLE_TENSORFLOW_LITE)
  list(APPEND FILTER_SOURCE tensor_filter_tensorflow_lite.c)
  list(APPEND FILTER_TARGET tensor_filter_tflitecore)
  list(APPEND FILTER_TARGET tensor_filter_tflitecoreStatic)
  PKG_CHECK_MODULES(TENSORFLOW_LITE REQUIRED tensorflow-lite)
  LINK_DIRECTORIES(${TENSORFLOW_LITE_LIBRARY_DIRS})

  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_CFLAGS} -Wno-sign-compare")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXXFLAGS} -Wno-sign-compare")

  ADD_LIBRARY(tensor_filter_tflitecore SHARED tensor_filter_tensorflow_lite_core.cc)
  TARGET_LINK_LIBRARIES(tensor_filter_tflitecore ${pkgs_LIBRARIES} tensorflow-lite)
  TARGET_INCLUDE_DIRECTORIES(tensor_filter_tflitecore PUBLIC ${pkgs_INCLUDE_DIRS} ${TENSORFLOW_LITE_INCLUDE_DIRS})
  TARGET_COMPILE_OPTIONS(tensor_filter_tflitecore PUBLIC ${pkgs_CFLAGS_OTHER})

  ADD_LIBRARY(tensor_filter_tflitecoreStatic STATIC tensor_filter_tensorflow_lite_core.cc)
  TARGET_LINK_LIBRARIES(tensor_filter_tflitecoreStatic ${pkgs_LIBRARIES} tensorflow-lite)
  TARGET_INCLUDE_DIRECTORIES(tensor_filter_tflitecoreStatic PUBLIC ${pkgs_INCLUDE_DIRS} ${TENSORFLOW_LITE_INCLUDE_DIRS})
  TARGET_COMPILE_OPTIONS(tensor_filter_tflitecoreStatic PUBLIC ${pkgs_CFLAGS_OTHER})
ENDIF(NOT DISABLE_TENSORFLOW_LITE)

IF(NOT DISABLE_TENSORFLOW)
  list(APPEND FILTER_SOURCE tensor_filter_tensorflow.c)
  list(APPEND FILTER_TARGET tensor_filter_tfcore)
  list(APPEND FILTER_TARGET tensor_filter_tfcoreStatic)
  PKG_CHECK_MODULES(TENSORFLOW REQUIRED tensorflow)
  LINK_DIRECTORIES(${TENSORFLOW_LIBRARY_DIRS})

  ADD_LIBRARY(tensor_filter_tfcore SHARED tensor_filter_tensorflow_core.cc)
  TARGET_LINK_LIBRARIES(tensor_filter_tfcore ${pkgs_LIBRARIES} ${TENSORFLOW_LIBRARIES})
  TARGET_INCLUDE_DIRECTORIES(tensor_filter_tfcore PUBLIC ${pkgs_INCLUDE_DIRS} ${TENSORFLOW_INCLUDE_DIRS})
  TARGET_COMPILE_OPTIONS(tensor_filter_tfcore PUBLIC ${pkgs_CFLAGS_OTHER})

  ADD_LIBRARY(tensor_filter_tfcoreStatic STATIC tensor_filter_tensorflow_core.cc) 
  TARGET_LINK_LIBRARIES(tensor_filter_tfcoreStatic ${pkgs_LIBRARIES} ${TENSORFLOW_LIBRARIES})
  TARGET_INCLUDE_DIRECTORIES(tensor_filter_tfcoreStatic PUBLIC ${pkgs_INCLUDE_DIRS} ${TENSORFLOW_INCLUDE_DIRS})
  TARGET_COMPILE_OPTIONS(tensor_filter_tfcoreStatic PUBLIC ${pkgs_CFLAGS_OTHER})
ENDIF(NOT DISABLE_TENSORFLOW)

MESSAGE(filterOBJ ${FILTER_SOURCE})
ADD_LIBRARY(tensor_filterOBJ OBJECT ${FILTER_SOURCE})

INSTALL(TARGETS ${FILTER_TARGET}
	RUNTIME DESTINATION ${EXEC_PREFIX}
	LIBRARY DESTINATION ${GST_INSTALL_DIR}
	ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
)
