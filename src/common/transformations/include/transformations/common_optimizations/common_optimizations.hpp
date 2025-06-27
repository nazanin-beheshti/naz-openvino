// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"
#include "../../../../../inference/include/openvino/runtime/properties.hpp"


namespace ov {
namespace pass {

class TRANSFORMATIONS_API CommonOptimizations;

}  // namespace pass
}  // namespace ov

class ov::pass::CommonOptimizations : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("CommonOptimizations");
    explicit CommonOptimizations(ov::hint::Graph_compiler_level graph_compiler_optimization_level = ov::hint::Graph_compiler_level::ADVANCED)
          :m_graph_compiler_optimization_level(graph_compiler_optimization_level){}

    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;

private:
    ov::hint::Graph_compiler_level m_graph_compiler_optimization_level;
};
