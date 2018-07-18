CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(nnstreamer_testapp C)

OPTION(INSTALL_EXAMPLE_APP "Install example applications" ON)

SET(PREFIX ${CMAKE_INSTALL_PREFIX})
SET(EXEC_PREFIX "${PREFIX}/bin")
SET(INCLUDEDIR "${PREFIX}/include/${PROJECT_NAME}")
IF(LIB_INSTALL_DIR)
ELSE(LIB_INSTALL_DIR)
    SET(LIB_INSTALL_DIR "${PREFIX}/lib")
ENDIF(LIB_INSTALL_DIR)

FIND_PACKAGE(PkgConfig REQUIRED)
SET(PKGAPP_MODULES
	nnstreamer
	gstreamer-1.0
	gstreamer-base-1.0
	gstreamer-controller-1.0
	gstreamer-video-1.0
	gstreamer-app-1.0
)
pkg_check_modules(apppkgs REQUIRED ${PKGAPP_MODULES})

ADD_SUBDIRECTORY(../example_sink ${CMAKE_CURRENT_BINARY_DIR}/example_Sink)
ADD_SUBDIRECTORY(../example_cam ${CMAKE_CURRENT_BINARY_DIR}/example_cam)
ADD_SUBDIRECTORY(../example_filter ${CMAKE_CURRENT_BINARY_DIR}/example_filter)
