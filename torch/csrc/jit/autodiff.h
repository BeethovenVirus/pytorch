#pragma once

#include "torch/csrc/jit/ir.h"

#include <ATen/ATen.h>
#include <vector>

namespace torch { namespace jit {

struct Capture {
  enum class Kind {Input, Output};
  Capture(Kind kind, std::size_t offset)
    : kind(kind), offset(offset) {}
  Kind kind;
  std::size_t offset;
};

using value_list = std::vector<Value*>;
struct Gradient {
  std::shared_ptr<Graph> f;
  std::shared_ptr<Graph> df;

  // Describes how to construct outputs of f from what its graph will return.
  // This is necessary because some trailing outputs are intermediates produced
  // only to be saved for df (and should be ignored).
  std::size_t f_real_outputs;

  // df inputs are split into two sections: captures are vjps (aka grad_outputs).
  // Captures are values the need to be saved when f is run. We handle inputs
  // specially, because this allows us to avoid adding extra vjps as df inputs.
  // VJPs are "seeds" for the gradient computation given for each input capture
  // of an Output kind.
  std::vector<Capture> df_input_captures;
  std::vector<std::size_t> df_input_vjps; // Offsets into f's outputs.

  // df will produce vjps for a subset of inputs of f that required grad.
  // df_output_vjps[idx] == inp_idx means that idx-th output of df produces a vjp
  // for inp_idx-th input of f.
  std::vector<std::size_t> df_output_vjps; // Offsets into f's inputs.

  // How to use gradient to implement a differentiable autograd function:
  // When running f:
  //   - Unwrap input Variables
  //   - Run f's graph
  //   - Create grad_fn
  //   - Wrap outputs in Variables (assume we have a tensor_outputs array):
  //       outputs = map(Variable, tensor_output)
  //       for i, offset in enumerate(df_input_vjps):
  //         outputs[offset].set_grad_fn(grad_fn, output_nr=i)
  //   - Use df_output_vjps to connect next_functions of grad_fn:
  //       for idx in df_output_vjps:
  //         grad_fn.next_functions.push_back(inputs[idx].grad_fn(), inputs[idx].output_nr)
  //   - Save captures for df (care needs to be taken to use SavedVariables for inputs and
  //                           outputs that we will actually return)
  //   - Return outputs
  //
  // When running df:
  //   - Concatenate captured Variables with received vjps
  //   - Interpret df
  //   - Wrap outputs of df into Variables (that don't require grad)
};
Gradient differentiate(std::shared_ptr<Graph>& graph);

// can we take a derivative of this node symbolically?
bool isDifferentiable(Node * n);

}}
