# @note About Android NDK
# Android NDK r12b supports API level as following:
# - From 8 (Fryoyo, 2.2) to 24 (Nougat, 7.0)
# - https://developer.android.com/ndk/guides/stable_apis#a24
#
# @note About Application ABI
# If you want to generate a binary file for all architectures, please append additional architech name
# such as "arm64-v8a armeabi-v7a x86 x86_64" as following:
# APP_ABI = armeabi armeabi-v7a arm64-v8a x86 x86_64

APP_ABI           := arm64-v8a
LIBCXX_USE_GABIXX := true
APP_STL           := c++_shared
APP_PLATFORM      := android-24
