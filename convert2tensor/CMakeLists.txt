cmake_minimum_required(VERSION 2.8)

project(convert2tensor C)

add_definitions(-DVERSION="0.0.0")

set(PREFIX ${CMAKE_INSTALL_PREFIX})
set(EXEC_PREFIX "${PREFIX}/bin")
if(LIB_INSTALL_DIR)
else(LIB_INSTALL_DIR)
    set(LIB_INSTALL_DIR "${PREFIX}/lib")
endif(LIB_INSTALL_DIR)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-base-1.0 gstreamer-controller-1.0 gstreamer-video-1.0 gstreamer-audio-1.0 glib-2.0)
INCLUDE_DIRECTORIES(${GST_INCLUDE_DIRS})
LINK_DIRECTORIES(${GST_LIBRARY_DIRS})

set(PACKAGE convert2tensor)

add_library(convert2tensor SHARED convert2tensor.c)
add_library(convert2tensorStatic STATIC convert2tensor.c)
target_link_libraries(convert2tensor ${GST_LIBRARIES})
target_include_directories(convert2tensor PUBLIC ${GST_INCLUDE_DIRS})
target_compile_options(convert2tensor PUBLIC ${GST_CFLAGS_OTHER})
target_link_libraries(convert2tensorStatic ${GST_LIBRARIES})
target_include_directories(convert2tensorStatic PUBLIC ${GST_INCLUDE_DIRS})
target_compile_options(convert2tensorStatic PUBLIC ${GST_CFLAGS_OTHER})

install(TARGETS convert2tensor convert2tensorStatic
	RUNTIME DESTINATION ${EXEC_PREFIX}
	LIBRARY DESTINATION ${LIB_INSTALL_DIR}
	ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
	)
