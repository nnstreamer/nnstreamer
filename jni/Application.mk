# Android NDK r12b supports API level as following:
# - From 8 (Fryoyo, 2.2) to 24 (Nougat, 7.0)
# - https://developer.android.com/ndk/guides/stable_apis#a24

LIBCXX_USE_GABIXX := true
APP_ABI           := arm64-v8a
APP_STL           := c++_shared
APP_PLATFORM      := android-24
