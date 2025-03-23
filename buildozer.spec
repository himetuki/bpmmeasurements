[app]
title = BPM Detector
package.name = bpmdetector
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1

# 添加所需的所有依赖
requirements = python3,kivy,numpy,scipy,sounddevice,audiostream,plyer

# 应用程序配置
orientation = portrait
fullscreen = 0

# Python 配置
osx.python_version = 3
osx.kivy_version = 2.1.0

# Android 特定配置
android.permissions = RECORD_AUDIO,INTERNET,WRITE_EXTERNAL_STORAGE
android.api = 31
android.minapi = 21
android.ndk = 25b
android.sdk = 31
android.accept_sdk_license = True
android.arch = arm64-v8a armeabi-v7a

# p4a 配置
p4a.branch = develop
p4a.bootstrap = sdl2

# 添加图标和启动画面
android.presplash_color = #FFFFFF
android.icon.filename = %(source.dir)s/icon.png
android.presplash.filename = %(source.dir)s/presplash.png

# 构建配置
android.release_artifact = apk
android.debug = True
android.logcat_filters = *:S python:D

[buildozer]
log_level = 2
warn_on_root = 1
