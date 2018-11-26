CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(nnstreamer C CXX)

OPTION(ENABLE_TENSORFLOW_LITE "Enable tensorflow-lite support" ON)
OPTION(ENABLE_TENSORFLOW "Enable tensorflow support" OFF)
OPTION(ENABLE_MODEL_DOWNLOAD "Enable download the model files" ON)
OPTION(INSTALL_EXAMPLE_APP "Install example applications" OFF)
OPTION(ENABLE_TEST "Enable tests" ON)

SET(VERSION "0.0.3")
ADD_DEFINITIONS(-DVERSION="${VERSION}")

SET(PREFIX ${CMAKE_INSTALL_PREFIX})
SET(EXEC_PREFIX "${PREFIX}/bin")
SET(INCLUDE_INSTALL_DIR "${PREFIX}/include")

IF(LIB_INSTALL_DIR)
ELSE(LIB_INSTALL_DIR)
    SET(LIB_INSTALL_DIR "${PREFIX}/${CMAKE_INSTALL_LIBDIR}")
ENDIF(LIB_INSTALL_DIR)

IF(GST_INSTALL_DIR)
ELSE(GST_INSTALL_DIR)
    SET(GST_INSTALL_DIR "${LIB_INSTALL_DIR}/gstreamer-1.0")
ENDIF(GST_INSTALL_DIR)

FIND_PACKAGE(PkgConfig REQUIRED)
SET(PKG_MODULES
	gstreamer-1.0
	gstreamer-base-1.0
	gstreamer-controller-1.0
	gstreamer-video-1.0
	gstreamer-audio-1.0
	glib-2.0
)

pkg_check_modules(pkgs REQUIRED ${PKG_MODULES})

INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}/gst/nnstreamer
)
INCLUDE_DIRECTORIES(
	SYSTEM
	${pkgs_INCLUDE_DIRS}
)
LINK_DIRECTORIES(${pkgs_LIBRARY_DIRS})
FOREACH(flag ${pkgs_CFALGS})
	SET(EXTRA_CFLAGS "${EXTRA_CFLAGS} ${flag}")
ENDFOREACH(flag)

# Gstreamer uses C89
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_CFLAGS} -Wall -Werror -fPIC -g -std=c89")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXXFLAGS} -Wall -Werror -fPIC -g -std=c++11")

# Supported projects for the single binary
SET(PROJECTS
	tensor_aggregator
	tensor_converter
	tensor_decoder
	tensor_demux
	tensor_merge
	tensor_mux
	tensor_saveload
	tensor_sink
	tensor_split
	tensor_transform
	tensor_filter
)

ADD_SUBDIRECTORY(nnstreamer_example)

IF(ENABLE_TEST)
	enable_testing()
	ADD_SUBDIRECTORY(tests)
ENDIF(ENABLE_TEST)

SET(OBJECTS)
FOREACH(PRJ IN LISTS PROJECTS)
	ADD_SUBDIRECTORY(gst/${PRJ})
	LIST(APPEND OBJECTS $<TARGET_OBJECTS:${PRJ}OBJ>)
ENDFOREACH(PRJ)

SET(FILTER_LIB dl)
SET(FILTER_DEP)
IF(ENABLE_TENSORFLOW_LITE)
	LIST(APPEND FILTER_LIB tensor_filter_tflitecore)
	LIST(APPEND FILTER_DEP tensor_filter_tflitecore)
ENDIF(ENABLE_TENSORFLOW_LITE)
IF(ENABLE_TENSORFLOW)
	LIST(APPEND FILTER_LIB tensor_filter_tfcore)
	LIST(APPEND FILTER_DEP tensor_filter_tfcore)
ENDIF(ENABLE_TENSORFLOW)
LIST(LENGTH FILTER_DEP FILTER_DEP_LEN)

ADD_LIBRARY(nnstreamerStatic STATIC gst/nnstreamer/nnstreamer.c gst/nnstreamer/tensor_common.c ${OBJECTS})
SET_TARGET_PROPERTIES(nnstreamerStatic PROPERTIES OUTPUT_NAME nnstreamer)
TARGET_INCLUDE_DIRECTORIES(nnstreamerStatic PUBLIC ${pkgs_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamerStatic PUBLIC ${pkgs_CFLAGS_OTHER})

ADD_LIBRARY(nnstreamer SHARED gst/nnstreamer/nnstreamer.c gst/nnstreamer/tensor_common.c ${OBJECTS})
TARGET_INCLUDE_DIRECTORIES(nnstreamer PUBLIC ${pkgs_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamer PUBLIC ${pkgs_CFLAGS_OTHER})
TARGET_LINK_LIBRARIES(nnstreamer ${pkgs_LIBRARIES} ${FILTER_LIB})

IF (FILTER_DEP_LEN GREATER 0)
	ADD_DEPENDENCIES(nnstreamer ${FILTER_DEP})
	ADD_DEPENDENCIES(nnstreamerStatic ${FILTER_DEP})
ENDIF (FILTER_DEP_LEN GREATER 0)

INSTALL(TARGETS nnstreamerStatic nnstreamer
	RUNTIME DESTINATION ${EXEC_PREFIX}
	LIBRARY DESTINATION ${GST_INSTALL_DIR}
	ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
)

CONFIGURE_FILE(nnstreamer.pc.in nnstreamer.pc @ONLY)

# For nnstreamer users (gst app developers using via gstreamer API)
INSTALL(FILES gst/nnstreamer/tensor_typedef.h
	DESTINATION ${INCLUDE_INSTALL_DIR}/nnstreamer
	)

# For nnstreamer custom filter developers
INSTALL(FILES gst/nnstreamer/tensor_common.h gst/nnstreamer/tensor_filter_custom.h
	DESTINATION ${INCLUDE_INSTALL_DIR}/nnstreamer
	)

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/nnstreamer.pc
	DESTINATION ${LIB_INSTALL_DIR}/pkgconfig
	)
