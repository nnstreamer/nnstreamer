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
