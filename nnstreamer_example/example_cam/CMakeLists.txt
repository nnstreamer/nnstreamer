CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# example app for webcam
ADD_EXECUTABLE(nnstreamer_example_cam nnstreamer_example_cam.c)

TARGET_LINK_LIBRARIES(nnstreamer_example_cam ${apppkgs_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(nnstreamer_example_cam PUBLIC ${apppkgs_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamer_example_cam PUBLIC ${apppkgs_CFLAGS_OTHER})

IF (INSTALL_EXAMPLE_APP)
	INSTALL(TARGETS nnstreamer_example_cam RUNTIME DESTINATION ${EXEC_PREFIX})
ENDIF (INSTALL_EXAMPLE_APP)
