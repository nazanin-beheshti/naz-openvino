// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"
#include "../../../../../inference/include/openvino/runtime/properties.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MOCTransformations;

}  // namespace pass
}  // namespace ov

/**
 * @brief This transformation is an entry point for OpenVINO transformations that will be
 * applied inside MOC. And in future this transformations container will be filled
 * with transformations pipeline but now it remains empty.
 */

class ov::pass::MOCTransformations : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MOCTransformations");

    /**
     * use_shapes = True enables transformations which are depends on shapes and also it
     * enables ConstantFolding for all ShapeOf operations.
     *
     * low_precision_enabled = True enables preserving mechanisms that helps to keep
     * low_precision sub-graphs as is.
     */
    explicit MOCTransformations(bool use_shapes, bool low_precision_enabled = true, ov::hint::Graph_optimization_level graph_compiler_optimization_level = ov::hint::Graph_optimization_level::FULL)
        : m_use_shapes(use_shapes),
          m_low_precision_enabled(low_precision_enabled),
          m_graph_compiler_optimization_level(graph_compiler_optimization_level){}

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    bool m_use_shapes;
    bool m_low_precision_enabled;
    ov::hint::Graph_optimization_level m_graph_compiler_optimization_level;
};
