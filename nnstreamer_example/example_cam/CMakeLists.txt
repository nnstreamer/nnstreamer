pkg_check_modules(app_ex_cam_pkgs gstreamer-1.0 glib-2.0 REQUIRED)

ADD_EXECUTABLE(nnstreamer_example_cam nnstreamer_example_cam.c)

TARGET_LINK_LIBRARIES(nnstreamer_example_cam ${app_ex_cam_pkgs_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(nnstreamer_example_cam PUBLIC ${app_ex_cam_pkgs_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamer_example_cam PUBLIC ${app_ex_cam_pkgs_CFLAGS_OTHER})

IF (INSTALL_EXAMPLE_APP)
	INSTALL(TARGETS nnstreamer_example_cam RUNTIME DESTINATION ${EXAMPLE_EXEC_PREFIX})
ENDIF (INSTALL_EXAMPLE_APP)
