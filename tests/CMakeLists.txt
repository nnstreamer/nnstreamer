# For unit test

SET(PKGTEST_MODULES
	gstreamer-app-1.0
	gstreamer-check-1.0
)

FIND_LIBRARY(GTEST_LIB gtest)

IF(GTEST_LIB)
	# Use GTEST shared lib
	SET(gtestLink gtest pthread)
	SET(gtestSrc "")
	SET(gtestInc "")
ELSE(GTEST_LIB)
	IF (NOT EXISTS /usr/src/gtest/src/gtest-all.cc)
		MESSAGE(FATAL FATAL_ERROR "You need to install libgtest-dev or (supplied with GTEST_LIB) libgtest.so.")
	ENDIF ()
	SET(gtestLink pthread)
	SET(gtestSrc /usr/src/gtest/src/gtest-all.cc)
	SET(gtestInc /usr/src/gtest)
ENDIF(GTEST_LIB)

pkg_check_modules(testpkgs REQUIRED ${PKGTEST_MODULES})

# sub-directory for test plugins
ADD_SUBDIRECTORY(nnstreamer_tensors/tensors_test)

# Unit test for common
ADD_EXECUTABLE(unittest_common common/unittest_common.cpp ${gtestSrc})
TARGET_INCLUDE_DIRECTORIES(unittest_common PUBLIC ${pkgs_INCLUDE_DIRS} ${testpkgs_INCLUDE_DIRS} ${gtestInc})
TARGET_LINK_LIBRARIES(unittest_common ${pkgs_LIBRARIES} ${testpkgs_LIBRARIES} ${gtestLink})

# Unit test using tensor_sink
ADD_EXECUTABLE(unittest_sink nnstreamer_sink/unittest_sink.cpp ${gtestSrc})
TARGET_INCLUDE_DIRECTORIES(unittest_sink PUBLIC ${pkgs_INCLUDE_DIRS} ${testpkgs_INCLUDE_DIRS} ${gtestInc})
TARGET_LINK_LIBRARIES(unittest_sink ${pkgs_LIBRARIES} ${testpkgs_LIBRARIES} ${gtestLink})

# Unit test uses this custom filter
ADD_LIBRARY(nnscustom_framecounter SHARED nnstreamer_sink/nnscustom_framecounter.c)
TARGET_LINK_LIBRARIES(nnscustom_framecounter ${CUSTOMFILTERDEP} glib-2.0)

# Unit test for nnstreamer plugins
ADD_EXECUTABLE(unittest_plugins nnstreamer_plugins/unittest_plugins.cpp ${gtestSrc})
TARGET_INCLUDE_DIRECTORIES(unittest_plugins PUBLIC ${pkgs_INCLUDE_DIRS} ${testpkgs_INCLUDE_DIRS} ${gtestInc})
TARGET_LINK_LIBRARIES(unittest_plugins ${pkgs_LIBRARIES} ${testpkgs_LIBRARIES} ${gtestLink})

ADD_TEST(NAME unit_test
        COMMAND sh -c "GST_PLUGIN_PATH=${PROJECT_SOURCE_DIR}/build ssat ${CMAKE_CURRENT_SOURCE_DIR}")

SET_TESTS_PROPERTIES(unit_test PROPERTIES TIMEOUT 120)

pkg_check_modules(pnglib REQUIRED libpng)
ADD_EXECUTABLE(bmp2png bmp2png.c)
TARGET_INCLUDE_DIRECTORIES(bmp2png PUBLIC ${pnglib_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(bmp2png ${pnglib_LIBRARIES})

# Copy bmp2png to /tests so that unittest scripts can use it.
ADD_CUSTOM_COMMAND(
	TARGET bmp2png POST_BUILD
	COMMAND cp
	${CMAKE_CURRENT_BINARY_DIR}/bmp2png
	${CMAKE_CURRENT_SOURCE_DIR}/
	)
