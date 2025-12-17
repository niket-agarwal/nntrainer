#!/bin/bash
set -e
TARGET=$1
SOURCE_ROOT=$2
INIPARSER_DIR=iniparser

echo "PREPARING INIPARSER at ${TARGET}"

[ ! -d ${TARGET} ] && mkdir -p ${TARGET}

# Check if iniparser directory already exists and has content
if [ -d "${TARGET}/${INIPARSER_DIR}" ] && [ "$(ls -A ${TARGET}/${INIPARSER_DIR})" ]; then
  echo "[INIPARSER] ${INIPARSER_DIR} exists and has content, skip downloading"
  exit 0
fi

# If directory exists but is empty, remove it
if [ -d "${TARGET}/${INIPARSER_DIR}" ]; then
  rm -rf ${TARGET}/${INIPARSER_DIR}
fi

# Download iniparser using meson subprojects download from source root
echo "[INIPARSER] downloading iniparser using meson"
cd ${SOURCE_ROOT}
meson subprojects download iniparser

# Copy the downloaded iniparser to target directory
if [ -d "subprojects/${INIPARSER_DIR}" ]; then
  echo "[INIPARSER] copying iniparser to target directory"
  cp -r subprojects/${INIPARSER_DIR} ${TARGET}/
else
  echo "[INIPARSER] Download failed, iniparser directory not found"
  exit 1
fi

echo "[INIPARSER] Finish preparing iniparser"
