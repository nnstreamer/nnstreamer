CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(nnstreamer C CXX)

OPTION(TIZEN "Enable Tizen build mode" OFF)

ADD_DEFINITIONS(-DVERSION="0.0.0")

SET(PREFIX ${CMAKE_INSTALL_PREFIX})
SET(EXEC_PREFIX "${PREFIX}/bin")
SET(INCLUDEDIR "${PREFIX}/include/${PROJECT_NAME}")

IF(LIB_INSTALL_DIR)
ELSE(LIB_INSTALL_DIR)
    SET(LIB_INSTALL_DIR "${PREFIX}/lib")
ENDIF(LIB_INSTALL_DIR)

FIND_PACKAGE(PkgConfig REQUIRED)
SET(PKG_MODULES
	gstreamer-1.0
	gstreamer-base-1.0
	gstreamer-controller-1.0
	gstreamer-video-1.0
	gstreamer-audio-1.0
	glib-2.0
)

FIND_LIBRARY(GTEST_LIB gtest)
IF(TIZEN OR GTEST_LIB)
	# Enable dlog of Tizen.
	ADD_DEFINITIONS(-DTIZEN=1)
	SET(PKG_MODULES ${PKG_MODULES} dlog)
	# Use GTEST shared lib
	SET(gtestLink gtest pthread)
	SET(gtestSrc "")
	SET(gtestInc "")
ELSE(TIZEN OR GTEST_LIB)
	IF (NOT EXISTS /usr/src/gtest/src/gtest-all.cc)
		MESSAGE(FATAL FATAL_ERROR "You need to install libgtest-dev or libgtest.so.")
	ENDIF ()
	SET(gtestLink pthread)
	SET(gtestSrc /usr/src/gtest/src/gtest-all.cc)
	SET(gtestInc /usr/src/gtest)
ENDIF(TIZEN OR GTEST_LIB)

IF (NOT INCLUDE_INSTALL_DIR)
	# We need to define includedir path
	SET (INCLUDE_INSTALL_DIR /usr/include)
	MESSAGE ("Warning: INCLUDE_INSTALL_DIR not defined. Using /usr/include")
ENDIF (NOT INCLUDE_INSTALL_DIR)

pkg_check_modules(pkgs REQUIRED ${PKG_MODULES})

INCLUDE_DIRECTORIES(
	${pkgs_INCLUDE_DIRS}
	${CMAKE_SOURCE_DIR}/include
)
LINK_DIRECTORIES(${pkgs_LIBRARY_DIRS})
FOREACH(flag ${pkgs_CFALGS})
	SET(EXTRA_CFLAGS "${EXTRA_CFLAGS} ${flag}")
ENDFOREACH(flag)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_CFLAGS} -Wall -Werror -fPIC -g")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXXFLAGS} -Wall -Werror -fPIC -g")

# Provide common data
ADD_LIBRARY(common STATIC common/tensor_common.c common/tensor_meta.c)
TARGET_LINK_LIBRARIES(common ${pkgs_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(common PUBLIC ${pkgs_INCLUDE_DIRS})
SET(pkgs_LIBRARIES ${pkgs_LIBRARIES} common)

# Unit test for common
ADD_EXECUTABLE(unittest_common common/test/unittest_common.cpp ${gtestSrc})
TARGET_INCLUDE_DIRECTORIES(unittest_common PUBLIC ${pkgs_INCLUDE_DIRS} ${gtestInc})
TARGET_LINK_LIBRARIES(unittest_common ${pkgs_LIBRARIES} ${gtestLink})

ADD_SUBDIRECTORY(gst/tensor_converter)
ADD_SUBDIRECTORY(gst/tensor_filter)
ADD_SUBDIRECTORY(gst/tensor_decoder)
ADD_SUBDIRECTORY(gst/tensor_sink)
ADD_SUBDIRECTORY(nnstreamer_example)
ADD_SUBDIRECTORY(tests/nnstreamer_tensors/tensors_test)
enable_testing()
ADD_SUBDIRECTORY(tests)

CONFIGURE_FILE(nnstreamer.pc.in nnstreamer.pc @ONLY)

# For nnstreamer users (gst app developers using via gstreamer API)
INSTALL(FILES include/tensor_typedef.h
	DESTINATION ${INCLUDE_INSTALL_DIR}/nnstreamer
	)

# For nnstreamer custom filter developers
INSTALL(FILES include/tensor_common.h include/tensor_filter_custom.h include/tensor_meta.h
	DESTINATION ${INCLUDE_INSTALL_DIR}/nnstreamer
	)

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/nnstreamer.pc
	DESTINATION ${LIB_INSTALL_DIR}/pkgconfig
	)

# Custom plugins need to use common archive (static lib)
INSTALL(TARGETS common
	RUNTIME DESTINATION ${EXEC_PREFIX}
	LIBRARY DESTINATION ${LIB_INSTALL_DIR}
	ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
	)
