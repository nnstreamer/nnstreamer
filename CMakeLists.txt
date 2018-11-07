CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(nnstreamer C CXX)

OPTION(TIZEN "Enable Tizen build mode" OFF)
OPTION(ENABLE_TENSORFLOW_LITE "Enable tensorflow-lite support" ON)
OPTION(ENABLE_TENSORFLOW "Enable tensorflow support" OFF)
OPTION(ENABLE_MODEL_DOWNLOAD "Enable download the model files" ON)
OPTION(INSTALL_EXAMPLE_APP "Install example applications" OFF)
OPTION(SINGLE_BINARY "Generate a single binary for all nnstreamer elements" ON)
OPTION(ENABLE_TEST "Enable tests" ON)

IF(SINGLE_BINARY)
	ADD_DEFINITIONS(-DSINGLE_BINARY)
	MESSAGE(STATUS "The nnstreamer elements will be created as a single plugin binary.")
ELSE(SINGLE_BINARY)
	MESSAGE(STATUS "The nnstreamer elements will be created as multple plugin binaries, one binary for one element. This is no more the standard method to release since nnstreamer 0.0.3.")
ENDIF(SINGLE_BINARY)

SET(VERSION "0.0.3")
ADD_DEFINITIONS(-DVERSION="${VERSION}")

SET(PREFIX ${CMAKE_INSTALL_PREFIX})
SET(EXEC_PREFIX "${PREFIX}/bin")
SET(INCLUDE_INSTALL_DIR "${PREFIX}/include")

IF(LIB_INSTALL_DIR)
ELSE(LIB_INSTALL_DIR)
    SET(LIB_INSTALL_DIR "${PREFIX}/lib")
ENDIF(LIB_INSTALL_DIR)

IF(GST_INSTALL_DIR)
ELSE(GST_INSTALL_DIR)
    SET(GST_INSTALL_DIR "${PREFIX}/lib/gstreamer-1.0")
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

IF(TIZEN)
	ADD_DEFINITIONS(-DTIZEN=1)
	# Enable dlog of Tizen.
	SET(PKG_MODULES ${PKG_MODULES} dlog)
ENDIF(TIZEN)

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

# Provide common data
ADD_LIBRARY(common STATIC gst/nnstreamer/tensor_common.c)
TARGET_LINK_LIBRARIES(common ${pkgs_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(common PUBLIC ${pkgs_INCLUDE_DIRS})
SET(pkgs_LIBRARIES ${pkgs_LIBRARIES} common)

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

IF (SINGLE_BINARY)
	ADD_LIBRARY(nnstreamerStatic STATIC gst/nnstreamer/nnstreamer.c ${OBJECTS} gst/nnstreamer/tensor_common.c)
	SET_TARGET_PROPERTIES(nnstreamerStatic PROPERTIES OUTPUT_NAME nnstreamer)
	TARGET_INCLUDE_DIRECTORIES(nnstreamerStatic PUBLIC ${pkgs_INCLUDE_DIRS})
	TARGET_COMPILE_OPTIONS(nnstreamerStatic PUBLIC ${pkgs_CFLAGS_OTHER})

	ADD_LIBRARY(nnstreamer SHARED gst/nnstreamer/nnstreamer.c ${OBJECTS})
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
ELSE (SINGLE_BINARY)
	FOREACH(PRJ IN LISTS PROJECTS)
		ADD_LIBRARY(${PRJ}Static STATIC $<TARGET_OBJECTS:${PRJ}OBJ>)
		SET_TARGET_PROPERTIES(${PRJ}Static PROPERTIES OUTPUT_NAME ${PRJ})
		TARGET_INCLUDE_DIRECTORIES(${PRJ}Static PUBLIC ${pkgs_INCLUDE_DIRS})
		TARGET_COMPILE_OPTIONS(${PRJ}Static PUBLIC ${pkgs_CFLAGS_OTHER})

		ADD_LIBRARY(${PRJ} SHARED $<TARGET_OBJECTS:${PRJ}OBJ>)
		TARGET_INCLUDE_DIRECTORIES(${PRJ} PUBLIC ${pkgs_INCLUDE_DIRS} ${NNSTREAMER_ADDITIONAL_INCDIR})
		TARGET_COMPILE_OPTIONS(${PRJ} PUBLIC ${pkgs_CFLAGS_OTHER} ${NNSTREAMER_ADDITIONAL_FLAGS})
		IF(PRJ STREQUAL "tensor_filter")
			TARGET_LINK_LIBRARIES(${PRJ} ${pkgs_LIBRARIES} ${FILTER_LIB})
			IF (FILTER_DEP_LEN GREATER 0)
				ADD_DEPENDENCIES(${PRJ} ${FILTER_DEP})
			ENDIF (FILTER_DEP_LEN GREATER 0)
		ELSE(PRJ STREQUAL "tensor_filter")
			TARGET_LINK_LIBRARIES(${PRJ} ${pkgs_LIBRARIES})
		ENDIF(PRJ STREQUAL "tensor_filter")

		INSTALL(TARGETS ${PRJ}Static ${PRJ}
			RUNTIME DESTINATION ${EXEC_PREFIX}
			LIBRARY DESTINATION ${GST_INSTALL_DIR}
			ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
		)
	ENDFOREACH(PRJ)
ENDIF (SINGLE_BINARY)

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

# Custom plugins need to use common archive (static lib)
INSTALL(TARGETS common
	RUNTIME DESTINATION ${EXEC_PREFIX}
	LIBRARY DESTINATION ${GST_INSTALL_DIR}
	ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
	)
