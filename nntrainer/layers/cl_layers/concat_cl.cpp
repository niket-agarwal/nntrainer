// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   concat_layer.cpp
 * @date   27 Oct 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Concat Layer Class for Neural Network
 *
 * @todo merge concat and split layer to a common implementation
 */

#include <cstring>
#include <vector>

#include <concat_cl.h>
#include <iostream>
#include <layer_context.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor_dim.h>
#include <util_func.h>

std::string concat_cl_kernel_ =
  R"(__kernel void concat_cl(__global const float* in1, 
                                           __global const float* in2, 
                                           __global float* out,
                                           const int batch_size, 
                                           const int channels, 
                                           const int height, 
                                           const int width1, 
                                           const int width2) {
    int global_id = get_global_id(0);

    printf("global_id : %d  \n", global_id);
    
    int total_width = width1 + width2;
    
    int width = total_width;
    
    // int total_elements_per_tensor = batch_size * channels * height * total_width;
    // if (global_id >= total_elements_per_tensor) {
    //     return;
    // }
    
    // Calculate the coordinates in the 4D space
    int w = global_id % total_width;
    int h = (global_id / total_width) % height;
    int c = (global_id / (total_width * height)) % channels;
    int b = global_id / (total_width * height * channels);

    printf("w, h, c, b : %d, %d, %d, %d  \n ", w,h,c,b);

    int output_index = ((b * channels + c) * height + h) * total_width + w;

    printf("output_index : %d  \n", output_index);

    
    // Determine if the index is in input1 or input2
    if (w < width1) {
        // Calculate the input1 index
        int input1_index = ((b * channels + c) * height + h) * width1 + w;
        printf("input1_index : %d  \n", input1_index);
        out[output_index] = in1[input1_index];
  
    } else {
        // Calculate the input2 index
        int input2_index = ((b * channels + c) * height + h) * width2 + (w - width1);
        printf("input2_index : %d  \n", input2_index);
        out[output_index] = in2[input2_index];
    }
})";

namespace nntrainer {
ConcatLayerCl::ConcatLayerCl() : Layer(), leading_helper_dim(1) {}

static constexpr size_t SINGLE_INOUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0;
static constexpr size_t INPUT_IDX_2 = 1;

void ConcatLayerCl::finalize(InitLayerContext &context) {
  auto &concat_dimension_prop = std::get<props::ConcatDimension>(concat_props);
  /** for backward compatibility, default concat dimension will be channel */
  /// @todo this is hacky way to force concat dimension to width if channel
  /// dimension is taken, this is because recurrent realizer, return sequence
  /// exploits concat layer but have no control over where to stack/axis
  unsigned int concat_dimension =
    context.getInputDimensions().front().channel() > 1 ? 3 : 1;
  if (!concat_dimension_prop.empty())
    concat_dimension = concat_dimension_prop.get();

  /**
   * The concat is only done along the axis dimension.
   * For example, consider 2 inputs a, b with dimensions [b,c,h,w] each
   * 1. concat_dimension = 1, output_dim = [b,c_a+c_b,h,w]
   * 2. concat_dimension = 2, output_dim = [b,c,h_a+h_b,w]
   * 3. concat_dimension = 3, output_dim = [b,c,h,w_a+w_b]
   */
  auto const &input_dims = context.getInputDimensions();
  const TensorDim &input_dim_0 = input_dims[SINGLE_INOUT_IDX];
  unsigned int concat_dim_val = input_dim_0.getTensorDim(concat_dimension);

  for (unsigned int idx = 1; idx < input_dims.size(); ++idx) {
    const TensorDim &dim = input_dims[idx];

    for (unsigned int i = 0; i < ml::train::TensorDim::getNumDim(); ++i) {
      if (i == concat_dimension)
        continue;
      NNTR_THROW_IF(input_dim_0[i] != dim[i], std::runtime_error)
        << "Error: concat layer requires same shape from all input layers "
           "along non-concat dimension";
    }
    concat_dim_val += dim[concat_dimension];
  }

  TensorDim output_dim = input_dim_0;
  output_dim.setTensorDim(concat_dimension, concat_dim_val);

  context.setOutputDimensions({output_dim});

  /**
   * Setup output_reshape_helper to which output will be reshaped in forwarding
   * to facilitate easier processing.
   *
   * The helper shape consolidates all the dimensions before the axis
   * together and all the dimensions after the axis to facilitate
   * easier splitting of the data.
   */
  leading_helper_dim = 1;
  output_reshape_helper.channel(1);
  output_reshape_helper.height(1);
  output_reshape_helper.width(1);
  for (unsigned int idx = 1; idx < concat_dimension; ++idx) {
    leading_helper_dim *= output_dim.getTensorDim(idx);
  }

  output_reshape_helper.height(output_dim.getTensorDim(concat_dimension));

  for (unsigned int idx = concat_dimension + 1;
       idx < ml::train::TensorDim::getNumDim(); ++idx) {
    output_reshape_helper.width(output_reshape_helper.width() *
                                output_dim.getTensorDim(idx));
  }

  /**
   * Setup input_reshape_helper to which inputs will be reshaped in forwarding
   * to facilitate easier processing.
   */
  input_reshape_helper.resize(input_dims.size());
  for (unsigned int idx = 0; idx < input_reshape_helper.size(); idx++) {
    input_reshape_helper[idx] = output_reshape_helper;
    input_reshape_helper[idx].height(
      input_dims[idx].getTensorDim(concat_dimension));
  }

  setBatch(input_dims[SINGLE_INOUT_IDX].batch());
}

void ConcatLayerCl::forwarding(RunLayerContext &context, bool training) {
  /**
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  const Tensor &in1 = context.getInput(INPUT_IDX_1);
  const Tensor &in2 = context.getInput(INPUT_IDX_2);
  ConcatProcess(in1, in2, out, context);

  //   const TensorDim out_dim = output.getDim();
  //   output.reshape(output_reshape_helper);
  //   unsigned int output_height_offset = 0;
  //   unsigned int data_copy_size = output_reshape_helper.width();
  //   TensorDim::TensorType tensor_type = out_dim.getTensorType();

  //   for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
  //     Tensor &input = context.getInput(idx);
  //     const TensorDim in_dim = input.getDim();
  //     auto const &irh = input_reshape_helper[idx];
  //     input.reshape(irh);

  //     if (in_dim.getDataType() == TensorDim::DataType::FP32) {
  //       /** loop over the dimensions before the concat dimension */
  //       for (unsigned int batch = 0; batch < output.batch(); batch++) {
  //         /** loop over the concat dimension itself */
  //         for (unsigned int count = 0; count < irh.height(); count++) {
  //           Tensor dest_tensor = Tensor::Map<float>(
  //             output.getAddress<float>(batch, 0, output_height_offset +
  //             count, 0), data_copy_size * sizeof(float), {1, 1, 1,
  //             data_copy_size, tensor_type});
  //           const Tensor source_tensor =
  //             Tensor::Map<float>(input.getAddress<float>(batch, 0, count, 0),
  //                                data_copy_size * sizeof(float),
  //                                {1, 1, 1, data_copy_size, tensor_type});
  //           dest_tensor.copy(source_tensor);
  //         }
  //       }
  //     } else if (in_dim.getDataType() == TensorDim::DataType::FP16) {
  // #ifdef ENABLE_FP16
  //       /** loop over the dimensions before the concat dimension */
  //       for (unsigned int batch = 0; batch < output.batch(); batch++) {
  //         /** loop over the concat dimension itself */
  //         for (unsigned int count = 0; count < irh.height(); count++) {
  //           Tensor dest_tensor = Tensor::Map<_FP16>(
  //             output.getAddress<_FP16>(batch, 0, output_height_offset +
  //             count, 0), data_copy_size * sizeof(_FP16), {1, 1, 1,
  //             data_copy_size, tensor_type});
  //           const Tensor source_tensor =
  //             Tensor::Map<_FP16>(input.getAddress<_FP16>(batch, 0, count, 0),
  //                                data_copy_size * sizeof(_FP16),
  //                                {1, 1, 1, data_copy_size, tensor_type});
  //           dest_tensor.copy(source_tensor);
  //         }
  //       }
  // #else
  //       throw std::invalid_argument("Error: enable-fp16 is not enabled");
  // #endif
  //     }

  //     input.reshape(in_dim);
  //     output_height_offset += irh.height();
  //   }

  //   output.reshape(out_dim);
}

void ConcatLayerCl::incremental_forwarding(RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  // /**
  //  * @todo avoid copy by creating input here as a shared_tensor of the output
  //  * here and then this layer can be in_place as well
  //  */
  // Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  // const TensorDim out_dim = output.getDim();
  // output.reshape(output_reshape_helper);
  // unsigned int output_height_offset = 0;
  // unsigned int data_copy_size = output_reshape_helper.width();

  // // @todo: this implementation is only works when axis is 3(width). Consider
  // // for other axes
  // unsigned int batch_channel = out_dim.batch() * out_dim.channel();

  // for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
  //   Tensor &input = context.getInput(idx);
  //   const TensorDim in_dim = input.getDim();
  //   auto const &irh = input_reshape_helper[idx];
  //   input.reshape(irh);

  //   /** loop over the dimensions before the concat dimension */
  //   for (unsigned int batch = batch_channel * from; batch < batch_channel *
  //   to;
  //        batch++) {
  //     /** loop over the concat dimension itself */
  //     for (unsigned int count = 0; count < irh.height(); count++) {
  //       Tensor dest_tensor = Tensor::Map(
  //         output.getAddress(batch, 0, output_height_offset + count, 0),
  //         data_copy_size * sizeof(float), {1, 1, 1, data_copy_size});
  //       const Tensor source_tensor = Tensor::Map(
  //         input.getAddress(batch, 0, count, 0), data_copy_size *
  //         sizeof(float), {1, 1, 1, data_copy_size});
  //       dest_tensor.copy(source_tensor);
  //     }
  //   }

  //   input.reshape(in_dim);
  //   output_height_offset += irh.height();
  // }

  // output.reshape(out_dim);
}

opencl::Kernel ConcatLayerCl::kernel_concat;

void ConcatLayerCl::ConcatProcess(Tensor const &in1, Tensor const &in2,
                                  Tensor &result, RunLayerContext &context) {

  unsigned int input_batch_size, input_height, in1_width, input_channels,
    in2_width;
  auto dim = in1.getDim();
  input_batch_size = dim.batch();
  input_height = dim.height();
  in1_width = dim.width();
  input_channels = dim.channel();
  in2_width = dim.width();

    printf("   input_batch_size  : %d  \n", input_batch_size);
    printf("      input_channels  : %d  \n", input_channels);
    printf("      input_height  : %d  \n", input_height);
    printf("    in1_width  : %d  \n", in1_width);
    printf("   in2_width  : %d  \n", in2_width);

  if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data1 = in1.getData();
    const float *data2 = in2.getData();
    float *rdata = result.getData();
    concat_cl(data1, data2, rdata, input_batch_size, input_channels,
               input_height, in1_width, in2_width, context);
  }
  // else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
  // #ifdef ENABLE_FP16
  //     const _FP16 *data = input.getData<_FP16>();
  //     _FP16 *rdata = output.getData<_FP16>();
  //     reshape_cl_fp16(data, rdata, input_batch_size, input_channels,
  //     input_height,
  //                     input_width, context);
  // #else
  //     throw std::invalid_argument("Error: enable-fp16 is not enabled");
  // #endif
  //   }
}

void ConcatLayerCl::concat_cl(const float *matAdata, const float *vecXdata,
                              float *vecYdata, unsigned int input_batch_size,
                              unsigned int input_channels,
                              unsigned int input_height, unsigned int in1_width,
                              unsigned int in2_width,
                              RunLayerContext &context) {

  bool result = false;

  do {
    result =
      context.clCreateKernel(concat_cl_kernel_, context.LayerKernel::CONCAT,
                             ConcatLayerCl::kernel_concat);
    if (!result) {
      break;
    }

    int dim = int(input_batch_size * input_channels * input_height *
                  (in1_width + in2_width));

    printf("  dim  : %d  \n", dim);
    // int dim = int(dim1 * dim2);
    opencl::Buffer inputA(context.context_inst_,
                          sizeof(float) * input_batch_size * input_channels *
                            input_height * in1_width,
                          true, nullptr);

    opencl::Buffer inputX(context.context_inst_,
                          sizeof(float) * input_batch_size * input_channels *
                            input_height * in2_width,
                          true, nullptr);

    opencl::Buffer inOutY(context.context_inst_,
                          sizeof(float) * input_batch_size * input_channels *
                            input_height * (in1_width + in2_width),
                          true, nullptr);

    result = inputA.WriteData(context.command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(context.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = inOutY.WriteData(context.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

    result = ConcatLayerCl::kernel_concat.SetKernelArguments(0, &inputA,
                                                             sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = ConcatLayerCl::kernel_concat.SetKernelArguments(1, &inputX,
                                                             sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = ConcatLayerCl::kernel_concat.SetKernelArguments(2, &inOutY,
                                                             sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = ConcatLayerCl::kernel_concat.SetKernelArguments(
      3, &input_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result = ConcatLayerCl::kernel_concat.SetKernelArguments(4, &input_channels,
                                                             sizeof(int));
    if (!result) {
      break;
    }

    result = ConcatLayerCl::kernel_concat.SetKernelArguments(5, &input_height,
                                                             sizeof(int));
    if (!result) {
      break;
    }

    result = ConcatLayerCl::kernel_concat.SetKernelArguments(6, &in1_width,
                                                             sizeof(int));
    if (!result) {
      break;
    }

    result = ConcatLayerCl::kernel_concat.SetKernelArguments(7, &in2_width,
                                                             sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {dim, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      ConcatLayerCl::kernel_concat, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(context.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

void ConcatLayerCl::calcDerivative(RunLayerContext &context) {
  //   /**
  //    * @todo avoid copy by creating input here as a shared_tensor of the
  //    output
  //    * here and then this layer can be in_place as well
  //    */
  //   Tensor output = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  //   output.reshape(output_reshape_helper);
  //   unsigned int output_height_offset = 0;
  //   unsigned int data_copy_size = output_reshape_helper.width();
  //   TensorDim::TensorType tensor_type = output.getTensorType();

  //  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
  //     Tensor &input = context.getOutgoingDerivative(idx);
  //     const TensorDim in_dim = input.getDim();
  //     auto const &irh = input_reshape_helper[idx];
  //     input.reshape(irh);

  //     if (in_dim.getDataType() == TensorDim::DataType::FP32) {
  //       /** loop over the dimensions before the concat dimension */
  //       for (unsigned int batch = 0; batch < output.batch(); batch++) {
  //         /** loop over the concat dimension itself */
  //         for (unsigned int count = 0; count < irh.height(); count++) {
  //           const Tensor source_tensor = Tensor::Map<float>(
  //             output.getAddress<float>(batch, 0, output_height_offset +
  //             count, 0), data_copy_size * sizeof(float), {1, 1, 1,
  //             data_copy_size, tensor_type});
  //           Tensor dest_tensor =
  //             Tensor::Map<float>(input.getAddress<float>(batch, 0, count, 0),
  //                                data_copy_size * sizeof(float),
  //                                {1, 1, 1, data_copy_size, tensor_type});
  //           dest_tensor.copy(source_tensor);
  //         }
  //       }
  //     } else if (in_dim.getDataType() == TensorDim::DataType::FP16) {
  // #ifdef ENABLE_FP16
  //       /** loop over the dimensions before the concat dimension */
  //       for (unsigned int batch = 0; batch < output.batch(); batch++) {
  //         /** loop over the concat dimension itself */
  //         for (unsigned int count = 0; count < irh.height(); count++) {
  //           const Tensor source_tensor = Tensor::Map<_FP16>(
  //             output.getAddress<_FP16>(batch, 0, output_height_offset +
  //             count, 0), data_copy_size * sizeof(_FP16), {1, 1, 1,
  //             data_copy_size, tensor_type});
  //           Tensor dest_tensor =
  //             Tensor::Map<_FP16>(input.getAddress<_FP16>(batch, 0, count, 0),
  //                                data_copy_size * sizeof(_FP16),
  //                                {1, 1, 1, data_copy_size, tensor_type});
  //           dest_tensor.copy(source_tensor);
  //         }
  //       }
  // #else
  //       throw std::invalid_argument("Error: enable-fp16 is not enabled");
  // #endif
  //     }

  //     input.reshape(in_dim);
  //     output_height_offset += irh.height();
  //   }
}

void ConcatLayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, concat_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[ConcatLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void ConcatLayerCl::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  Layer::exportTo(exporter, method);
  exporter.saveResult(concat_props, method, this);
}

} /* namespace nntrainer */
