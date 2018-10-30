# demo app
pkg_check_modules(cairo_pkg REQUIRED cairo)

if(ENABLE_MODEL_DOWNLOAD)
	if(NOT EXISTS "${CMAKE_BINARY_DIR}/tflite_model_ssd/ssd_mobilenet_v2_coco.tflite")
		file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tflite_model_ssd)

		message("-- [START TO DOWNLOAD REQUIRED RESOURCES: Model]")
		file(DOWNLOAD
			"https://github.com/nnsuite/testcases/raw/master/DeepLearningModels/tensorflow-lite/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.tflite"
			"${CMAKE_BINARY_DIR}/tflite_model_ssd/ssd_mobilenet_v2_coco.tflite"
			SHOW_PROGRESS
		)
	endif()

	if(NOT EXISTS "${CMAKE_BINARY_DIR}/tflite_model_ssd/coco_labels_list.txt")
		message("-- [START TO DOWNLOAD REQUIRED RESOURCES: Label]")
		file(DOWNLOAD
			"https://github.com/nnsuite/testcases/raw/master/DeepLearningModels/tensorflow-lite/ssd_mobilenet_v2_coco/coco_labels_list.txt"
			"${CMAKE_BINARY_DIR}/tflite_model_ssd/coco_labels_list.txt"
			SHOW_PROGRESS
		)
	endif()


	if(NOT EXISTS "${CMAKE_BINARY_DIR}/tflite_model_ssd/box_priors.txt")
		message("-- [START TO DOWNLOAD REQUIRED RESOURCES: Box_prior]")
		file(DOWNLOAD
			"https://github.com/nnsuite/testcases/raw/master/DeepLearningModels/tensorflow-lite/ssd_mobilenet_v2_coco/box_priors.txt"
			"${CMAKE_BINARY_DIR}/tflite_model_ssd/box_priors.txt"
			SHOW_PROGRESS
		)
	endif()
else()
	message("-- [MODEL DOWNLOAD IS BLOCKED]")
endif(ENABLE_MODEL_DOWNLOAD)

ADD_EXECUTABLE(nnstreamer_example_object_detection nnstreamer_example_object_detection.cc)

TARGET_LINK_LIBRARIES(nnstreamer_example_object_detection ${apppkgs_LIBRARIES} ${cairo_pkg_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(nnstreamer_example_object_detection PUBLIC ${apppkgs_INCLUDE_DIRS} ${cairo_pkg_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamer_example_object_detection PUBLIC ${apppkgs_CFLAGS_OTHER})

IF (INSTALL_EXAMPLE_APP)
	INSTALL(TARGETS nnstreamer_example_object_detection RUNTIME DESTINATION ${EXEC_PREFIX})
ENDIF (INSTALL_EXAMPLE_APP)
