# demo app
pkg_check_modules(cairo_pkg REQUIRED cairo)

ADD_EXECUTABLE(nnstreamer_example_object_detection nnstreamer_example_object_detection.cc)

TARGET_LINK_LIBRARIES(nnstreamer_example_object_detection ${apppkgs_LIBRARIES} ${cairo_pkg_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(nnstreamer_example_object_detection PUBLIC ${apppkgs_INCLUDE_DIRS} ${cairo_pkg_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamer_example_object_detection PUBLIC ${apppkgs_CFLAGS_OTHER})

IF (INSTALL_EXAMPLE_APP)
	INSTALL(TARGETS nnstreamer_example_object_detection RUNTIME DESTINATION ${EXAMPLE_EXEC_PREFIX})
	INSTALL(FILES get_model.sh gst-launch-object-detection.sh DESTINATION ${EXAMPLE_EXEC_PREFIX})
ENDIF (INSTALL_EXAMPLE_APP)
