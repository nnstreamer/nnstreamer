cmake_minimum_required(VERSION 2.8)

project(gst_example)

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
LINK_LIBRARIES(${GST_LIBRARIES})


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GST_CFLAGS_OTHER}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GST_CFLAGS_OTHER}")

add_executable(basic-tutorial-1 tutorial1/basic-tutorial-1.c)

add_executable(basic-tutorial-2 tutorial2/basic-tutorial-2.c)

add_executable(basic-tutorial-2-add-filter tutorial2/basic-tutorial-2-add-filter.c)

add_executable(basic-tutorial-3 tutorial3/basic-tutorial-3.c)

add_executable(basic-tutorial-3-exercise tutorial3/basic-tutorial-3-exercise.c)

add_executable(basic-tutorial-6 tutorial6/basic-tutorial-6.c)

add_executable(basic-tutorial-7 tutorial7/basic-tutorial-7.c)

add_executable(basic-tutorial-8 tutorial8/basic-tutorial-8.c)

install(TARGETS basic-tutorial-1 basic-tutorial-2 basic-tutorial-2-add-filter basic-tutorial-3 basic-tutorial-3-exercise basic-tutorial-6 basic-tutorial-7 basic-tutorial-8
	RUNTIME DESTINATION ${EXEC_PREFIX}
	LIBRARY DESTINATION ${LIB_INSTALL_DIR}
	ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
	)
