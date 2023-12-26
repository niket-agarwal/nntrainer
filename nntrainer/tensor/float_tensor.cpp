// SPDX-License-Identifier: Apache-2.0
/**
 * @file	float_tensor.cpp
 * @date	01 December 2023
 * @brief	This is FloatTensor class for 32-bit floating point calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <iomanip>
#include <iostream>

#include <blas_interface.h>
#include <float_tensor.h>
#include <util_func.h>

namespace nntrainer {

FloatTensor::FloatTensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm, Tdatatype::FP32) {}

/// @todo support allocation by src_tensor
void FloatTensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    allocateSrcTensor();
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data
    MemoryData *mem_data;

    mem_data = new MemoryData((void *)(new float[dim.getDataLen()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<float>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

void FloatTensor::deallocate() {
  data = nullptr;
  offset = 0;
}

void *FloatTensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<float>() + offset;
}

void *FloatTensor::getData(size_t idx) const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<float>() + offset + idx;
}

void *FloatTensor::getAddress(unsigned int i) {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((float *)getData())[i];
}

const void *FloatTensor::getAddress(unsigned int i) const {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((float *)getData())[i];
}

void FloatTensor::setValue(float value) {
  float *data = (float *)getData();
  std::fill(data, data + size(), value);
}

void FloatTensor::setValue(unsigned int batch, unsigned int c, unsigned int h,
                           unsigned int w, float value) {
  ((float *)getData())[getIndex(batch, c, h, w)] = value;
}

void FloatTensor::setZero() {
  if (contiguous) {
    sscal(size(), 0, (float *)getData(), 1);
  } else {
    /// @todo implement apply_i
    // apply_i<float>([](float val) -> float { return 0; });
    setValue(0);
  }
}

/// @todo support additional initializer
void FloatTensor::initialize() {
  if (empty() || !isAllocated())
    return;

  switch (initializer) {
  case Initializer::ZEROS:
    setZero();
    break;
  case Initializer::ONES:
    setValue(1.0f);
    break;
  default:
    break;
  }

  putData();
}

void FloatTensor::initialize(Initializer init) {
  initializer = init;
  initialize();
}

void FloatTensor::print(std::ostream &out) const {
  printInstance(out, this);
  const float *data = (float *)getData();
  unsigned int len = size();
  out << "data addr: " << data << '\n';
  out << dim;

  if (len > 100) {
    out << '[' << data[0] << ' ' << data[1] << ' ' << data[2] << " ... "
        << data[len - 3] << ' ' << data[len - 2] << ' ' << data[len - 1] << ']'
        << std::endl;
    return;
  }

  std::ios init(NULL);
  init.copyfmt(out);

  if (getFormat() == Tformat::NCHW) {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int l = 0; l < channel(); l++) {
        for (unsigned int i = 0; i < height(); i++) {
          for (unsigned int j = 0; j < width(); j++) {
            out << std::setw(10) << std::setprecision(10)
                << data[getIndex(k, l, i, j)] << " ";
          }
          out << std::endl;
        }
        out << std::endl;
      }
      out << "-------" << std::endl;
    }
  } else {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int i = 0; i < height(); i++) {
        for (unsigned int j = 0; j < width(); j++) {
          for (unsigned int l = 0; l < channel(); l++) {
            out << std::setw(10) << std::setprecision(10)
                << data[getIndex(k, l, i, j)] << " ";
          }
          out << std::endl;
        }
        out << std::endl;
      }
      out << "-------" << std::endl;
    }
  }
  out.copyfmt(init);
}

} // namespace nntrainer
