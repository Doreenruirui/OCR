from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import tensor_array_ops
import lib_tensor.nest as nest
import tensorflow as tf
#from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs



def _state_size_with_prefix(state_size, prefix=None):
    """Helper function that enables int or TensorShape shape specification.
    This function takes a size specification, which can be an integer or a
    TensorShape, and converts it into a list of integers. One may specify any
    additional dimensions that precede the final state size specification.
    Args:
      state_size: TensorShape or int that specifies the size of a tensor.
      prefix: optional additional list of dimensions to prepend.
    Returns:
      result_state_size: list of dimensions the resulting tensor size.
    """
    result_state_size = tensor_shape.as_shape(state_size).as_list()
    if prefix is not None:
        if not isinstance(prefix, list):
            raise TypeError("prefix of _state_size_with_prefix should be a list.")
        result_state_size = prefix + result_state_size
    return result_state_size


def _infer_state_dtype(explicit_dtype, state):
    """Infer the dtype of an RNN state.
    Args:
      explicit_dtype: explicitly declared dtype or None.
      state: RNN's hidden state. Must be a Tensor or a nested iterable containing
        Tensors.
    Returns:
      dtype: inferred dtype of hidden state.
    Raises:
      ValueError: if `state` has heterogeneous dtypes or is empty.
    """
    if explicit_dtype is not None:
        return explicit_dtype
    elif nest.is_sequence(state):
        inferred_dtypes = [element.dtype for element in nest.flatten(state)]
        if not inferred_dtypes:
            raise ValueError("Unable to infer dtype from empty state.")
        all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
        if not all_same:
            raise ValueError(
                "State has tensors of different inferred_dtypes. Unable to infer a "
                "single representative dtype.")
        return inferred_dtypes[0]
    else:
        return state.dtype


def _create_zero_output(output_size, batch_size, dtype):
    # convert int to TensorShape if necessary
    size = _state_size_with_prefix(output_size, prefix=[batch_size])
    output = array_ops.zeros(
        array_ops.stack(size), dtype)
    shape = _state_size_with_prefix(
        output_size, prefix=[batch_size])
    output.set_shape(tensor_shape.TensorShape(shape))
    return output


def dynamic_rnn_loop(cell,
                     inputs,
                     initial_state,
                     state_size,
                     output_size,
                     scope_name,
                     parallel_iterations,
                     swap_memory,
                     sequence_length=None,
                     dtype=None):
    state = initial_state
    assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

    flat_input = nest.flatten(inputs)
    flat_output_size = nest.flatten(output_size)

    # Construct an initial output
    input_shape = array_ops.shape(flat_input[0])
    time_steps = input_shape[0]
    batch_size = input_shape[1]

    inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                             for input_ in flat_input)

    const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

    for shape in inputs_got_shape:
        if not shape[2:].is_fully_defined():
            raise ValueError(
                "Input size (depth of inputs) must be accessible via shape inference,"
                " but saw value None.")
        got_time_steps = shape[0].value
        got_batch_size = shape[1].value
        if const_time_steps != got_time_steps:
            raise ValueError(
                "Time steps is not the same for all the elements in the input in a "
                "batch.")
        if const_batch_size != got_batch_size:
            raise ValueError(
                "Batch_size is not the same for all the elements in the input.")

    # Prepare dynamic conditional copying of state & output
    def _create_zero_arrays(size):
        size = _state_size_with_prefix(size, prefix=[batch_size])
        return array_ops.zeros(
            array_ops.stack(size), _infer_state_dtype(dtype, state))

    flat_zero_output = tuple(_create_zero_arrays(output)
                             for output in flat_output_size)
    zero_output = nest.pack_sequence_as(structure=cell.output_size,
                                        flat_sequence=flat_zero_output)

    if sequence_length is not None:
        min_sequence_length = math_ops.reduce_min(sequence_length)
        max_sequence_length = math_ops.reduce_max(sequence_length)

    time = array_ops.constant(0, dtype=dtypes.int32, name="time")

    with ops.name_scope(scope_name) as scope:
        base_name = scope

    def _create_ta(name, dtype):
        return tensor_array_ops.TensorArray(dtype=dtype,
                                            size=time_steps,
                                            tensor_array_name=base_name + name)

    output_ta = tuple(_create_ta("output_%d" % i,
                                 _infer_state_dtype(dtype, state))
                      for i in range(len(flat_output_size)))
    input_ta = tuple(_create_ta("input_%d" % i, flat_input[0].dtype)
                     for i in range(len(flat_input)))

    input_ta = tuple(ta.unstack(input_)
                     for ta, input_ in zip(input_ta, flat_input))

    def _time_step(time, output_ta_t, state):
        """Take a time step of the dynamic RNN.
        Args:
          time: int32 scalar Tensor.
          output_ta_t: List of `TensorArray`s that represent the output.
          state: nested tuple of vector tensors that represent the state.
        Returns:
          The tuple (time + 1, output_ta_t with updated flow, new_state).
        """

        input_t = tuple(ta.read(time) for ta in input_ta)
        # Restore some shape information
        for input_, shape in zip(input_t, inputs_got_shape):
            input_.set_shape(shape[1:])

        input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
        call_cell = lambda: cell(input_t, state)

        if sequence_length is not None:
            (output, new_state) = _rnn_step(
                time=time,
                sequence_length=sequence_length,
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                zero_output=zero_output,
                state=state,
                call_cell=call_cell,
                state_size=state_size,
                skip_conditionals=True)
        else:
            (output, new_state) = call_cell()

        # Pack state if using state tuples
        output = nest.flatten(output)

        output_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(output_ta_t, output))

        return (time + 1, output_ta_t, new_state)

    _, output_final_ta, final_state = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_time_step,
        loop_vars=(time, output_ta, state),
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    # Unpack final output if not using output tuples.
    final_outputs = tuple(ta.stack() for ta in output_final_ta)

    # Restore some shape information
    for output, output_size in zip(final_outputs, flat_output_size):
        shape = _state_size_with_prefix(
            output_size, prefix=[const_time_steps, const_batch_size])
        output.set_shape(shape)

    final_outputs = nest.pack_sequence_as(
        structure=cell.output_size, flat_sequence=final_outputs)

    return (final_outputs, final_state)


def _on_device(fn, device):
    """Build the subgraph defined by lambda `fn` on `device` if it's not None."""
    if device:
        with ops.device(device):
            return fn()
    else:
        return fn()


def _rnn_step(
        time, sequence_length, min_sequence_length, max_sequence_length,
        zero_output, state, call_cell, state_size, skip_conditionals=False):

    # Convert state to a list for ease of use
    flat_state = nest.flatten(state)
    flat_zero_output = nest.flatten(zero_output)

    def _copy_one_through(output, new_output):
        copy_cond = (time >= sequence_length)
        return _on_device(
            lambda: array_ops.where(copy_cond, output, new_output),
            device=new_output.op.device)

    def _copy_some_through(flat_new_output, flat_new_state):
        # Use broadcasting select to determine which values should get
        # the previous state & zero output, and which values should get
        # a calculated state & output.
        flat_new_output = [
            _copy_one_through(zero_output, new_output)
            for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
        flat_new_state = [
            _copy_one_through(state, new_state)
            for state, new_state in zip(flat_state, flat_new_state)]
        return flat_new_output + flat_new_state

    def _maybe_copy_some_through():
        """Run RNN step.  Pass through either no or some past state."""
        new_output, new_state = call_cell()

        nest.assert_same_structure(state, new_state)

        flat_new_state = nest.flatten(new_state)
        flat_new_output = nest.flatten(new_output)
        return control_flow_ops.cond(
            # if t < min_seq_len: calculate and return everything
            time < min_sequence_length, lambda: flat_new_output + flat_new_state,
            # else copy some of it through
            lambda: _copy_some_through(flat_new_output, flat_new_state))

    # TODO(ebrevdo): skipping these conditionals may cause a slowdown,
    # but benefits from removing cond() and its gradient.  We should
    # profile with and without this switch here.
    if skip_conditionals:
        # Instead of using conditionals, perform the selective copy at all time
        # steps.  This is faster when max_seq_len is equal to the number of unrolls
        # (which is typical for dynamic_rnn).
        new_output, new_state = call_cell()
        nest.assert_same_structure(state, new_state)
        new_state = nest.flatten(new_state)
        new_output = nest.flatten(new_output)
        final_output_and_state = _copy_some_through(new_output, new_state)
    else:
        empty_update = lambda: flat_zero_output + flat_state
        final_output_and_state = control_flow_ops.cond(
            # if t >= max_seq_len: copy all state through, output zeros
            time >= max_sequence_length, empty_update,
            # otherwise calculation is required: copy some or all of it through
            _maybe_copy_some_through)

    if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
        raise ValueError("Internal error: state and output were not concatenated "
                         "correctly.")
    final_output = final_output_and_state[:len(flat_zero_output)]
    final_state = final_output_and_state[len(flat_zero_output):]

    for output, flat_output in zip(final_output, flat_zero_output):
        output.set_shape(flat_output.get_shape())
    for substate, flat_substate in zip(final_state, flat_state):
        substate.set_shape(flat_substate.get_shape())

    final_output = nest.pack_sequence_as(
        structure=zero_output, flat_sequence=final_output)
    final_state = nest.pack_sequence_as(
        structure=state, flat_sequence=final_state)

    return final_output, final_state
