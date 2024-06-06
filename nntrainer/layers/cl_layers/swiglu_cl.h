// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   swiglu.h
 * @date   14 July 2023
 * @brief  Implementation of SwiGLU activation function
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __SWIGLU_LAYER_CL_H__
#define __SWIGLU_LAYER_CL_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

#include <common_properties.h>
#include <utility>
#include <layer_impl.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>

namespace nntrainer {

/**
 * @brief A SwiGLU layer for llama.
 *
 */
class SwiGLULayerCl final : public Layer {
public:
  /**
   * @brief Construct a new SwiGLU layer object
   *
   */
  SwiGLULayerCl() : Layer(), swiglu_props(props::Print()) {}

  /**
   * @brief Destroy the SwiGLU layer object
   *
   */
  ~SwiGLULayerCl() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return SwiGLULayerCl::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override{};

  inline static const std::string type = "swiglu";

  static opencl::Kernel kernel_swiglu;
  static opencl::Kernel kernel_swiglu_fp16;
  
  std::tuple<props::Print>
    swiglu_props; /**< fc layer properties : unit - number of output neurons */

  void swigluProcess( Tensor const &in1,  Tensor const &in2,
                                  Tensor &result, 
                                 RunLayerContext &context) ;
  
  void swiglu_cl(const float *matAdata, const float *vecXdata,
                   float *vecYdata, unsigned int dim1, unsigned int dim2,
                   RunLayerContext &context);
  void swiglu_cl_fp16(const __fp16 *matAdata,
                                        const __fp16 *vecXdata, __fp16 *vecYdata,
                                        unsigned int dim1, unsigned int dim2,
                                        RunLayerContext &context);

};

} // namespace nntrainer

#endif /* __SWIGLU_LAYER_H__ */
