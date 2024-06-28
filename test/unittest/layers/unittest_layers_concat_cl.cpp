// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_concat.cpp
 * @date 7 July 2021
 * @brief Concat Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <concat_cl.h>
#include <layers_common_tests.h>

auto semantic_concat_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ConcatLayerCl>, nntrainer::ConcatLayerCl::type,
  {}, LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(ConcatGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_concat_gpu));

auto concat_dim3 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConcatLayerCl>, {"axis=3"},
  "2:3:3:2,2:3:3:3", "concat_dim3.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(ConcatGPU, LayerGoldenTest,
                     ::testing::Values(concat_dim3));
