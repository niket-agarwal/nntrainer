// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file unittest_layers_fully_connected_cl.cpp
 * @date 7 June 2024
 * @brief Fully Connected Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <swiglu_cl.h>
#include <layers_common_tests.h>

auto semantic_swiglu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SwiGLULayerCl>,
  nntrainer::SwiGLULayerCl::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(SwigluGPU, LayerSemantics,
                     ::testing::Values(semantic_swiglu));



auto swiglu_basic_plain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::SwiGLULayerCl>, {},
  "2:3:3:3,2:3:3:3", "swiglu.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD | 
    LayerGoldenTestParamOptions::USE_INC_FORWARD,
  "nchw", "fp32", "fp32");

// auto swiglu_basic_plain_nhwc = LayerGoldenTestParamType(
//   nntrainer::createLayer<nntrainer::SwiGLULayerCl>, {},
//   "2:3:3:3", "swiglu.nnlayergolden",
//   LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
//     LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
//     LayerGoldenTestParamOptions::USE_INC_FORWARD,
//   "nhwc", "fp32", "fp32");

GTEST_PARAMETER_TEST(SwigluGPU, LayerGoldenTest,
                     ::testing::Values(swiglu_basic_plain));
