pkg_check_modules(app_ex_obj_pkgs gstreamer-1.0 gstreamer-video-1.0 glib-2.0 cairo REQUIRED)

ADD_EXECUTABLE(nnstreamer_example_object_detection_tensorflow nnstreamer_example_object_detection_tensorflow.cc)

TARGET_LINK_LIBRARIES(nnstreamer_example_object_detection_tensorflow ${app_ex_obj_pkgs_LIBRARIES} m)
TARGET_INCLUDE_DIRECTORIES(nnstreamer_example_object_detection_tensorflow PUBLIC ${app_ex_obj_pkgs_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamer_example_object_detection_tensorflow PUBLIC ${app_ex_obj_pkgs_CFLAGS_OTHER})

IF (INSTALL_EXAMPLE_APP)
	INSTALL(TARGETS nnstreamer_example_object_detection_tensorflow RUNTIME DESTINATION ${EXAMPLE_EXEC_PREFIX})
	INSTALL(FILES get_model.sh DESTINATION ${EXAMPLE_EXEC_PREFIX}) # shell script example will be updated after decoder for tf-ssd model.
ENDIF (INSTALL_EXAMPLE_APP)
