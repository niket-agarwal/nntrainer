#!/usr/bin/env bash

set -e

# Function to prepare protobuf for Android build
prepare_protobuf() {
  local build_dir_prefix="$1"
  local ndk_path="$2"
  
  echo "Preparing protobuf for Android build"
  
  if [ ! -d "$ndk_path" ]; then
    echo "Warning: Android NDK not found. Please set ANDROID_NDK or ANDROID_NDK_HOME environment variable."
    echo "Using existing protobuf libraries from subprojects (may cause compatibility issues)"
    # Use existing protobuf from subprojects
    if [ ! -d "${build_dir_prefix}builddir/protobuf-25.2" ]; then
      echo "Copying protobuf from subprojects to build directory"
      cp -r subprojects/protobuf-25.2 ${build_dir_prefix}builddir/
    fi
    # Copy protobuf libraries and headers to jni directory for Android NDK
    echo "Copying protobuf libraries and headers to jni directory"
    mkdir -p ${build_dir_prefix}builddir/jni/protobuf-25.2/lib
    # Check if libraries exist in the build directory
    local lib_path_prefix=""
    if [ "$build_dir_prefix" = "../" ]; then
      lib_path_prefix="../../"
    elif [ "$build_dir_prefix" = "" ]; then
      lib_path_prefix="build/"
    fi
    
    if [ -f "${lib_path_prefix}subprojects/protobuf-25.2/lib/arm64-v8a/libprotobuf.a" ] && [ -f "${lib_path_prefix}subprojects/protobuf-25.2/lib/arm64-v8a/libprotobuf-lite.a" ]; then
      echo "Using protobuf libraries from build directory"
      cp ${lib_path_prefix}subprojects/protobuf-25.2/lib/arm64-v8a/libprotobuf.a ${build_dir_prefix}builddir/jni/protobuf-25.2/lib/
      cp ${lib_path_prefix}subprojects/protobuf-25.2/lib/arm64-v8a/libprotobuf-lite.a ${build_dir_prefix}builddir/jni/protobuf-25.2/lib/
    else
      echo "Using existing protobuf libraries from subprojects"
      # Copy the libraries from the subproject (this is a temporary solution)
      # In a real scenario, these should be built for Android
      find subprojects/protobuf-25.2 -name "libprotobuf.a" -exec cp {} ${build_dir_prefix}builddir/jni/protobuf-25.2/lib/ \;
      find subprojects/protobuf-25.2 -name "libprotobuf-lite.a" -exec cp {} ${build_dir_prefix}builddir/jni/protobuf-25.2/lib/ \;
    fi
    # Copy protobuf headers
    if [ "$build_dir_prefix" = "../" ]; then
      cp -r ${build_dir_prefix}builddir/protobuf-25.2/src ${build_dir_prefix}builddir/jni/protobuf-25.2/
    else
      cp -r subprojects/protobuf-25.2/src ${build_dir_prefix}builddir/jni/protobuf-25.2/
    fi
    # Copy abseil headers for protobuf - using exact path
    mkdir -p ${build_dir_prefix}builddir/jni/protobuf-25.2/third_party
    if [ -d "subprojects/abseil-cpp-20250814.1" ]; then
      echo "Copying abseil-cpp from subprojects/abseil-cpp-20250814.1"
      cp -r subprojects/abseil-cpp-20250814.1 ${build_dir_prefix}builddir/jni/protobuf-25.2/third_party/abseil-cpp
    else
      echo "Warning: abseil-cpp-20250814.1 not found at expected location, skipping copy"
    fi
  else
    echo "Using Android NDK at $ndk_path"
    # Use our prepare_protobuf.sh script to build protobuf for Android
    # Ensure prepare_protobuf.sh has execute permissions
    if [ "$build_dir_prefix" = "../" ]; then
      chmod +x ${TARGET}/jni/prepare_protobuf.sh 2>/dev/null || echo "Warning: Could not set execute permissions on prepare_protobuf.sh"
      ${TARGET}/jni/prepare_protobuf.sh 25.2 . "$ndk_path"
    else
      chmod +x ./jni/prepare_protobuf.sh 2>/dev/null || echo "Warning: Could not set execute permissions on prepare_protobuf.sh"
      ./jni/prepare_protobuf.sh 25.2 builddir "$ndk_path"
    fi
    # Copy protobuf libraries
    mkdir -p ${build_dir_prefix}builddir/jni/protobuf-25.2/lib
    if [ -d "${build_dir_prefix}builddir/protobuf-25.2/lib" ]; then
      cp -r ${build_dir_prefix}builddir/protobuf-25.2/lib/* ${build_dir_prefix}builddir/jni/protobuf-25.2/lib/
    fi
    # Copy protobuf headers
    mkdir -p ${build_dir_prefix}builddir/jni/protobuf-25.2
    if [ -d "${build_dir_prefix}builddir/protobuf-25.2/src" ]; then
      cp -r ${build_dir_prefix}builddir/protobuf-25.2/src ${build_dir_prefix}builddir/jni/protobuf-25.2/
    elif [ -d "subprojects/protobuf-25.2/src" ]; then
      cp -r subprojects/protobuf-25.2/src ${build_dir_prefix}builddir/jni/protobuf-25.2/
    fi
    # Copy abseil headers for protobuf - using exact path
    mkdir -p ${build_dir_prefix}builddir/jni/protobuf-25.2/third_party
    if [ -d "subprojects/abseil-cpp-20250814.1" ]; then
      echo "Copying abseil-cpp from subprojects/abseil-cpp-20250814.1"
      cp -r subprojects/abseil-cpp-20250814.1 ${build_dir_prefix}builddir/jni/protobuf-25.2/third_party/abseil-cpp
    else
      echo "Warning: abseil-cpp-20250814.1 not found at expected location, skipping copy"
    fi
  fi
}

# Ensure all subprojects are downloaded
echo "Downloading subprojects..."
# Download subprojects that have proper meson support
meson subprojects download abseil-cpp
meson subprojects download benchmark
meson subprojects download clblast
meson subprojects download googletest
meson subprojects download protobuf
meson subprojects download ruy
# Download iniparser for Android build (will be handled by prepare script and CMake integration)
meson subprojects download iniparser

TARGET=$1
[ -z $1 ] && TARGET=$(pwd)
echo $TARGET

if [ ! -d $TARGET ]; then
    if [[ $1 == -D* ]]; then
	TARGET=$(pwd)
	echo $TARGET
    else
	echo $TARGET is not a directory. please put project root of nntrainer
	exit 1
    fi
fi

pushd $TARGET

filtered_args=()

for arg in "$@"; do
    if [[ $arg == -D* ]]; then
	filtered_args+=("$arg")
    fi
done

# Set up common meson arguments
MESON_ARGS="-Dplatform=android -Dopenblas-num-threads=1 -Denable-tflite-interpreter=false -Denable-tflite-backbone=false -Denable-fp16=true -Domp-num-threads=1 -Dhgemm-experimental-kernel=false -Denable-onnx-interpreter=true ${filtered_args[@]}"

if [ ! -d builddir ]; then
    #default value of openblas num threads is 1 for android
    #enable-tflite-interpreter=false is just temporary until ci system is stable
    #enable-opencl=true will compile OpenCL related changes or remove this option to exclude OpenCL compilations.
  meson builddir $MESON_ARGS

  # Prepare protobuf for Android build
  ANDROID_NDK_PATH=${ANDROID_NDK:-/opt/android-ndk}
  if [ ! -d "$ANDROID_NDK_PATH" ]; then
    ANDROID_NDK_PATH=${ANDROID_NDK_HOME:-/usr/local/android-ndk}
  fi
  prepare_protobuf "" "$ANDROID_NDK_PATH"
else
  echo "warning: $TARGET/builddir has already been taken, this script tries to reconfigure and try building"
  pushd builddir
    #default value of openblas num threads is 1 for android
    #enable-tflite-interpreter=false is just temporary until ci system is stable  
    #enable-opencl=true will compile OpenCL related changes or remove this option to exclude OpenCL compilations.
    meson configure $MESON_ARGS
    meson --wipe
    
  # Prepare protobuf for Android build
  ANDROID_NDK_PATH=${ANDROID_NDK:-/opt/android-ndk}
  if [ ! -d "$ANDROID_NDK_PATH" ]; then
    ANDROID_NDK_PATH=${ANDROID_NDK_HOME:-/usr/local/android-ndk}
  fi
  prepare_protobuf "../" "$ANDROID_NDK_PATH"
  popd
fi

pushd builddir
# Compile ONNX protobuf files before install
echo "Compiling ONNX protobuf files"
# Try to compile the ONNX protobuf files
if ! meson compile nntrainer/schema/onnx_proto; then
  echo "Meson compile failed, trying ninja directly..."
  ninja
fi

# Copy generated protobuf files to the source directory where Android build expects them
echo "Copying ONNX protobuf files to source directory"
cp nntrainer/schema/onnx.pb.cc ../nntrainer/schema/
cp nntrainer/schema/onnx.pb.h ../nntrainer/schema/

# Continue with the rest of the build process
ninja install

tar -czvf $TARGET/nntrainer_for_android.tar.gz --directory=android_build_result .

popd
popd
