// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   swiglu_cl.cpp
 * @date   
 * @brief  Implementation of SwiGLU activation function
 * @see    https://github.com/nnstreamer/nntrainer
 * @author 
 * @bug    
 *
 */

#include <iostream>

#include "swiglu_cl.h"



std::string swiglu_cl_kernel_ =
  R"(__kernel void swiglu_cl(__global const float *in1, __global const float *in2, __global float *out) {
    int i = get_global_id(0);
    float swish = in1[i] * exp(in1[i]) / (1 + exp(in1[i]));
    out[i] = swish * in2[i];
    printf("going in the kernel");
})";


namespace nntrainer {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0;
static constexpr size_t INPUT_IDX_2 = 1;

// namespace ActivationOp {
/**
 * @brief activation function swish
 * @param x input
 * @retval swish(x)
 */
// float swish(float x) { return x / (1 + nntrainer::exp_util(-x)); }
// // namespace ActivationOp
// } // namespace ActivationOp

// float swish(float x) { return x / (1 + nntrainer::exp_util(-x)); }

void SwiGLULayerCl::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

// SwigluCl::SwigluCl() :
//   LayerImpl(), fc_props(props::Unit()) {
//   weight_idx.fill(std::numeric_limits<unsigned>::max());
// }

void SwiGLULayerCl::forwarding(RunLayerContext &context,
                             bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
      swigluProcess(in1, in2, out, context);
    } 
}


// void SwiGLULayerCl::forwarding(nntrainer::RunLayerContext &context,
//                              bool training) {
//   nntrainer::Tensor &in1 = context.getInput(INPUT_IDX_1);
//   nntrainer::Tensor &in2 = context.getInput(INPUT_IDX_2);
//   nntrainer::Tensor &out = context.getOutput(OUT_IDX);

//   if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
//     for (int b = 0; b < (int)in1.batch(); b++) {
//       for (int c = 0; c < (int)in1.channel(); c++) {
//         for (int h = 0; h < (int)in1.height(); h++) {
//           for (int w = 0; w < (int)in1.width(); w++) {
//             out.setValue(b, c, h, w,
//                          swish(in1.getValue<float>(b, c, h, w)) *
//                            in2.getValue<float>(b, c, h, w));
//           }
//         }
//       }
//     }
// //   } else if (in1.getDataType() == ml::train::TensorDim::DataType::FP16) {
// // #ifdef ENABLE_FP16
// //     for (int b = 0; b < (int)in1.batch(); b++) {
// //       for (int c = 0; c < (int)in1.channel(); c++) {
// //         for (int h = 0; h < (int)in1.height(); h++) {
// //           for (int w = 0; w < (int)in1.width(); w++) {
// //             out.setValue(
// //               b, c, h, w,
// //               static_cast<float>(
// //                 ActivationOp::swish(
// //                   static_cast<float>(in1.getValue<_FP16>(b, c, h, w))) *
// //                 static_cast<float>(in2.getValue<_FP16>(b, c, h, w))));
// //           }
// //         }
// //       }
// //     }
// // #else
// //     NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
// // #endif
//    }
// }

void SwiGLULayerCl::incremental_forwarding(RunLayerContext &context, unsigned int from, unsigned int to,
                             bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

   if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
      swigluProcess(in1, in2, out, context);
    } 
}

opencl::Kernel SwiGLULayerCl::kernel_swiglu;

void SwiGLULayerCl::swigluProcess(Tensor const &in1, Tensor const &in2,
                                 Tensor &result, 
                                 RunLayerContext &context)  {

  // unsigned int in1dim1, in1dim2, in2dim1, in2dim2;
  // if (in1.getFormat() == Tformat::NHWC) {
  //   in1dim1 = in1.batch() * in1.height() * in1.width();
  //   in1dim2 = in1.channel();
  //   in2dim1 = in2.batch() * in2.height() * in2.width();
  //   in2dim2 = in2.channel();
  // } else {
  //   in1dim1 = in1.batch() * in1.channel() * in1.height();
  //   in1dim2 = in1.width();
  //   in2dim1 = in2.batch() * in2.channel() * in2.height();
  //   in2dim2 = in2.width();
  // }
  unsigned int dim1,dim2;
  dim1 = in1.batch() * in1.channel() * in1.height();
  dim2 = in1.width();



    //unsigned int size = context.getNumInputs();
    if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const float *data1 = in1.getData();
      const float *data2 = in2.getData();
      float *rdata = result.getData();

      swiglu_cl(data1,  data2, rdata, dim1, dim2, context);
    } else
    throw std::invalid_argument("Error: OpenCL fp16 is not supported yet.");
}

void SwiGLULayerCl::swiglu_cl(const float *matAdata,
                                        const float *vecXdata, float *vecYdata,
                                        unsigned int dim1, unsigned int dim2,
                                        RunLayerContext &context) {

  bool result = false;
  

  do {
    result =
      context.clCreateKernel(swiglu_cl_kernel_, context.LayerKernel::SWIGLU,
                             SwiGLULayerCl::kernel_swiglu);
    if (!result) {
      break;
    }

    size_t dim1_size = sizeof(float) * dim1;
    size_t dim2_size = sizeof(float) * dim2;
    int dim = int (dim1 * dim2);
    opencl::Buffer inputA(context.context_inst_, dim1_size * dim2_size, true,
                          nullptr);

    opencl::Buffer inputX(context.context_inst_, dim1_size * dim2_size, true, nullptr);

    opencl::Buffer inOutY(context.context_inst_, dim1_size * dim2_size, true, nullptr);

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

    result = SwiGLULayerCl::kernel_swiglu.SetKernelArguments(
      0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = SwiGLULayerCl::kernel_swiglu.SetKernelArguments(
      1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = SwiGLULayerCl::kernel_swiglu.SetKernelArguments(
      2, &inOutY, sizeof(cl_mem));
    if (!result) {
      break;
    }

    // result = SwiGLULayerCl::kernel_swiglu.SetKernelArguments(
    //   3, &dim, sizeof(int));
    // if (!result) {
    //   break;
    // }

    const int work_groups_count[3] = {dim, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      SwiGLULayerCl::kernel_swiglu, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(context.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

    ml_logi("BBBBBBBBBBBBBBBBBBBBBBBBBBBBB");

  } while (false);
}



// void SwiGLULayer::incremental_forwarding(nntrainer::RunLayerContext &context,
//                                          unsigned int from, unsigned int to,
//                                          bool training) {
//   nntrainer::Tensor &in1 = context.getInput(INPUT_IDX_1);
//   nntrainer::Tensor &in2 = context.getInput(INPUT_IDX_2);
//   nntrainer::Tensor &out = context.getOutput(OUT_IDX);

//   if (from) {
//     NNTR_THROW_IF(to - from != 1, std::invalid_argument)
//       << "incremental step size is not 1";
//     from = 0;
//     to = 1;
//   }

//   if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
//     for (unsigned int b = 0; b < in1.batch(); b++) {
//       for (unsigned int c = 0; c < in1.channel(); c++) {
//         for (unsigned int h = from; h < to; h++) {
//           for (unsigned int w = 0; w < in1.width(); w++) {
//            out.setValue(
//               b, c, h, w,
//               static_cast<_FP16>(
//                 ActivationOp::swish(
//                   static_cast<float>(in1.getValue<_FP16>(b, c, h, w))) *
//                 static_cast<float>(in2.getValue<_FP16>(b, c, h, w))));
//           }
//         }
//       }
//     }        out.setValue(b, c, h, w,
//                          ActivationOp::swish(in1.getValue<float>(b, c, h, w)) *
//                            in2.getValue<float>(
//                             ;
//           }
//         }
//       }
//     }
//   } else if (in1.getDataType() == ml::train::TensorDim::DataType::FP16) {
// #ifdef ENABLE_FP16
//     for (unsigned int b = 0; b < in1.batch(); b++) {
//       for (unsigned int c = 0; c < in1.channel(); c++) {
//         for (unsigned int h = from; h < to; h++) {
//           for (unsigned int w = 0; w < in1.width(); w++) {
     
// #else
//     NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
// #endif
//   }
// }

void SwiGLULayerCl::calcDerivative(nntrainer::RunLayerContext &context) {
  // std::throw_with_nested(std::runtime_error("Training is not supported
  // yet."));
}

#ifdef PLUGGABLE

Layer *create_swiglu_layer_cl() {
  auto layer = new SwiGLULayerCl();
  std::cout << "swiglu created\n";
  return layer;
}

void destroy_swiglu_layer_cl(Layer *layer) {
  std::cout << "swiglu deleted\n";
  delete layer;
}

extern "C" {
LayerPluggable ml_train_layer_pluggable{create_swiglu_layer_cl,
                                                   destroy_swiglu_layer_cl};
}

#endif
} // namespace custom
