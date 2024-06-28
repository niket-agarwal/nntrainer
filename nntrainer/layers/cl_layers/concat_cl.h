// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   concat_layer.h
 * @date   27 Oct 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Concat Layer Class for Neural Network
 *
 */

#ifndef __CONCAT_LAYER_CL_H__
#define __CONCAT_LAYER_CL_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <layer_context.h>
#include <layer_impl.h>
#include <tensor_dim.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>
#include <utility>

namespace nntrainer {

/**
 * @class   Concat Layer
 * @brief   Concat Layer
 */
class ConcatLayerCl : public Layer {
public:
  /**
   * @brief     Constructor of Concat Layer
   */
  ConcatLayerCl();

  /**
   * @brief     Destructor of Concat Layer
   */
  ~ConcatLayerCl() = default;

  /**
   *  @brief  Move constructor of ConcatLayer.
   *  @param[in] ConcatLayer &&
   */
  ConcatLayerCl(ConcatLayerCl &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs ConcatLayer to be moved.
   */
  ConcatLayerCl &operator=(ConcatLayerCl &&rhs) = default;

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
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return ConcatLayerCl::type; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override {
    setBatch(batch);
  }

  inline static const std::string type = "concat";

  static opencl::Kernel kernel_concat;

  void ConcatProcess(Tensor const &in1, Tensor const &in2, Tensor &result,
                     RunLayerContext &context);

  void concat_cl(const float *matAdata, const float *vecXdata,
                              float *vecYdata, unsigned int input_batch_size,
                              unsigned int input_channels,
                              unsigned int input_height, unsigned int in1_width,
                              unsigned int in2_width,
                              RunLayerContext &context);

private:
  unsigned int leading_helper_dim; /**< batch dimension of helper dimension not
                                containing the actual batch */
  std::vector<TensorDim>
    input_reshape_helper;          /** helper dimension to reshape inputs */
  TensorDim output_reshape_helper; /** helper dimension to reshape outputs */
  std::tuple<props::ConcatDimension> concat_props;

  /**
   * @brief set batch for the internal variables
   *
   * @param batch update batch size
   */
  void setBatch(unsigned int batch) {
    for (auto &irh : input_reshape_helper)
      irh.batch(batch * leading_helper_dim);
    output_reshape_helper.batch(batch * leading_helper_dim);
  }
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONCAT_LAYER_CL_H__ */
