# Example App build requires a few additional attention if tizen_app_build_example is not used.

SET(PKGAPP_MODULES
	gstreamer-1.0
	gstreamer-base-1.0
	gstreamer-controller-1.0
	gstreamer-video-1.0
	gstreamer-app-1.0
)
pkg_check_modules(apppkgs REQUIRED ${PKGAPP_MODULES})

ADD_SUBDIRECTORY(custom_example_passthrough)
ADD_SUBDIRECTORY(custom_example_scaler)
ADD_SUBDIRECTORY(custom_example_average)
ADD_SUBDIRECTORY(example_cam)
ADD_SUBDIRECTORY(example_sink)
ADD_SUBDIRECTORY(example_filter)
ADD_SUBDIRECTORY(example_filter_performance_profile)
