# check whether TENSORFLOW_LITE is available.
# ENABLE_TENSORFLOW_LITE is defined at /debian/rules according to the build environment

set(FILTER_SOURCE tensor_filter.c tensor_filter_custom.c)
set(FILTER_TARGET)

IF(ENABLE_TENSORFLOW_LITE) # AVAILABLE
  ADD_DEFINITIONS(-DENABLE_TENSORFLOW_LITE)
ENDIF(ENABLE_TENSORFLOW_LITE)

IF(ENABLE_TENSORFLOW) # AVAILABLE
  ADD_DEFINITIONS(-DENABLE_TENSORFLOW)
ENDIF(ENABLE_TENSORFLOW)

IF(ENABLE_TENSORFLOW_LITE)
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
  SET_TARGET_PROPERTIES(tensor_filter_tflitecoreStatic PROPERTIES OUTPUT_NAME tensor_filter_tflitecore)
  TARGET_LINK_LIBRARIES(tensor_filter_tflitecoreStatic ${pkgs_LIBRARIES} tensorflow-lite)
  TARGET_INCLUDE_DIRECTORIES(tensor_filter_tflitecoreStatic PUBLIC ${pkgs_INCLUDE_DIRS} ${TENSORFLOW_LITE_INCLUDE_DIRS})
  TARGET_COMPILE_OPTIONS(tensor_filter_tflitecoreStatic PUBLIC ${pkgs_CFLAGS_OTHER})
ENDIF(ENABLE_TENSORFLOW_LITE)

IF(ENABLE_TENSORFLOW)
  list(APPEND FILTER_SOURCE tensor_filter_tensorflow.c)
  list(APPEND FILTER_TARGET tensor_filter_tfcore)
  list(APPEND FILTER_TARGET tensor_filter_tfcoreStatic)
  PKG_CHECK_MODULES(protobuf_pkg REQUIRED protobuf)
  PKG_CHECK_MODULES(TENSORFLOW REQUIRED tensorflow)
  LINK_DIRECTORIES(${TENSORFLOW_LIBRARY_DIRS})

  ADD_LIBRARY(tensor_filter_tfcore SHARED tensor_filter_tensorflow_core.cc)
  TARGET_LINK_LIBRARIES(tensor_filter_tfcore ${pkgs_LIBRARIES} ${TENSORFLOW_LIBRARIES} ${protobuf_pkg_LIBRARIES})
  TARGET_INCLUDE_DIRECTORIES(tensor_filter_tfcore PUBLIC ${pkgs_INCLUDE_DIRS} ${TENSORFLOW_INCLUDE_DIRS})
  TARGET_COMPILE_OPTIONS(tensor_filter_tfcore PUBLIC ${pkgs_CFLAGS_OTHER})

  ADD_LIBRARY(tensor_filter_tfcoreStatic STATIC tensor_filter_tensorflow_core.cc) 
  SET_TARGET_PROPERTIES(tensor_filter_tfcoreStatic PROPERTIES OUTPUT_NAME tensor_filter_tfcore)
  TARGET_LINK_LIBRARIES(tensor_filter_tfcoreStatic ${pkgs_LIBRARIES} ${TENSORFLOW_LIBRARIES})
  TARGET_INCLUDE_DIRECTORIES(tensor_filter_tfcoreStatic PUBLIC ${pkgs_INCLUDE_DIRS} ${TENSORFLOW_INCLUDE_DIRS})
  TARGET_COMPILE_OPTIONS(tensor_filter_tfcoreStatic PUBLIC ${pkgs_CFLAGS_OTHER})
ENDIF(ENABLE_TENSORFLOW)

ADD_LIBRARY(tensor_filterOBJ OBJECT ${FILTER_SOURCE})

INSTALL(TARGETS ${FILTER_TARGET}
	RUNTIME DESTINATION ${EXEC_PREFIX}
	LIBRARY DESTINATION ${LIB_INSTALL_DIR}
	ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
)
