r"""
The torch.onnx module contains functions to export models into the ONNX
IR format.  These models can be loaded with the ONNX library and then
converted to models which run on other deep learning frameworks.
"""

import torch
import torch.jit
import torch.autograd
import torch.serialization
import re
from torch._six import container_abcs
import contextlib
import numbers
import warnings
import functools
import types
from torch._six import string_classes
from torch.autograd import Function, function
from torch.jit import _unique_state_dict
from torch.onnx import ONNX_ARCHIVE_MODEL_PROTO_NAME, ExportTypes, OperatorExportTypes
from torch._C import ListType


@contextlib.contextmanager
def set_training(model, mode):
    r"""
    A context manager to temporarily set the training mode of 'model'
    to 'mode', resetting it when we exit the with-block.  A no-op if
    mode is None.
    """
    if mode is None:
        yield
        return
    old_mode = model.training
    if old_mode != mode:
        model.train(mode)
    try:
        yield
    finally:
        if old_mode != mode:
            model.train(old_mode)


def export(model, args, f, export_params=True, verbose=False, training=False,
           input_names=None, output_names=None, aten=False, export_raw_ir=False,
           operator_export_type=None, opset_version=None):
    r"""
    Export a model into ONNX format.  This exporter runs your model
    once in order to get a trace of its execution to be exported;
    at the moment, it supports a limited set of dynamic models (e.g., RNNs.)

    See also: :ref:`onnx-export`

    Arguments:
        model (torch.nn.Module): the model to be exported.
        args (tuple of arguments): the inputs to
            the model, e.g., such that ``model(*args)`` is a valid
            invocation of the model.  Any non-Tensor arguments will
            be hard-coded into the exported model; any Tensor arguments
            will become inputs of the exported model, in the order they
            occur in args.  If args is a Tensor, this is equivalent
            to having called it with a 1-ary tuple of that Tensor.
            (Note: passing keyword arguments to the model is not currently
            supported.  Give us a shout if you need it.)
        f: a file-like object (has to implement fileno that returns a file descriptor)
            or a string containing a file name.  A binary Protobuf will be written
            to this file.
        export_params (bool, default True): if specified, all parameters will
            be exported.  Set this to False if you want to export an untrained model.
            In this case, the exported model will first take all of its parameters
            as arguments, the ordering as specified by ``model.state_dict().values()``
        verbose (bool, default False): if specified, we will print out a debug
            description of the trace being exported.
        training (bool, default False): export the model in training mode.  At
            the moment, ONNX is oriented towards exporting models for inference
            only, so you will generally not need to set this to True.
        input_names(list of strings, default empty list): names to assign to the
            input nodes of the graph, in order
        output_names(list of strings, default empty list): names to assign to the
            output nodes of the graph, in order
        aten (bool, default False): [DEPRECATED. use operator_export_type] export the
            model in aten mode. If using aten mode, all the ops original exported
            by the functions in symbolic.py are exported as ATen ops.
        export_raw_ir (bool, default False): [DEPRECATED. use operator_export_type]
            export the internal IR directly instead of converting it to ONNX ops.
        operator_export_type (enum, default OperatorExportTypes.ONNX):
            OperatorExportTypes.ONNX: all ops are exported as regular ONNX ops.
            OperatorExportTypes.ONNX_ATEN: all ops are exported as ATen ops.
            OperatorExportTypes.ONNX_ATEN_FALLBACK: if symbolic is missing,
                                                    fall back on ATen op.
            OperatorExportTypes.RAW: export raw ir.
        opset_version (int, default is 9): by default we export the model to the
            opset version of the onnx submodule. Since ONNX's latest opset may
            evolve before next stable release, we may want to export to some stable
            opset version. Right now, supported stable opset version is 9.
    """
    if aten or export_raw_ir:
        assert operator_export_type is None
        assert aten ^ export_raw_ir
        operator_export_type = OperatorExportTypes.ATEN if aten else OperatorExportTypes.RAW
    elif operator_export_type is None:
        if torch.onnx.PYTORCH_ONNX_CAFFE2_BUNDLE:
            operator_export_type = OperatorExportTypes.ONNX_ATEN_FALLBACK
        else:
            operator_export_type = OperatorExportTypes.ONNX
    _export(model, args, f, export_params, verbose, training, input_names, output_names,
            operator_export_type=operator_export_type, opset_version=opset_version)


# ONNX can't handle constants that are lists of tensors, which can
# get generated in constant prop. So we split them back into prim::ListConstructs
def _split_tensor_list_constants(g, block):
    for node in block.nodes():
        for subblock in node.blocks():
            _split_tensor_list_constants(g, subblock)
        if node.kind() == "prim::Constant":
            output_type = node.output().type()
            if output_type.isSubtypeOf(ListType.ofTensors()):
                inputs = [g.create("prim::Constant").t_('value', t)
                           .insertBefore(node).output()
                          for t in node['value']]
                lc = (g.create("prim::ListConstruct", inputs)
                      .insertBefore(node)
                      .output()
                      .setType(ListType.ofTensors()))
                node.output().replaceAllUsesWith(lc)


def _optimize_graph(graph, operator_export_type):
    # Remove fork/wait nodes
    torch._C._jit_pass_inline_fork_wait(graph)
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_remove_inplace_ops(graph)
    # we record now record some ops like ones/zeros
    # into a trace where we previously recorded constants
    # use constant prop to maintain our current level of onnx support
    # without implementing symbolics for all of them
    torch._C._jit_pass_constant_propagation(graph)
    _split_tensor_list_constants(graph, graph)
    # run dce to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_canonicalize_ops(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_peephole(graph, True)
    torch._C._jit_pass_lint(graph)

    # onnx only supports tensors, but 1 / 2 = 0.5 and tensor(1) / tensor(2) = 0
    torch._C._jit_pass_prepare_division_for_onnx(graph)
    # onnx only supports tensors, so we turn all out number types into tensors
    torch._C._jit_pass_erase_number_types(graph)
    # onnx does not support tuples, so try to remove them
    torch._C._jit_pass_lower_all_tuples(graph)
    torch._C._jit_pass_peephole(graph, True)
    torch._C._jit_pass_lint(graph)

    if operator_export_type != OperatorExportTypes.RAW:
        graph = torch._C._jit_pass_onnx(graph, operator_export_type)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_onnx_peephole(graph)
        torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_fixup_onnx_loops(graph)
    torch._C._jit_pass_lint(graph)
    graph = torch._C._jit_pass_canonicalize(graph)
    torch._C._jit_pass_lint(graph)
    return graph


def _trace(func, args, operator_export_type, return_outs=False):
    # Special case for common case of passing a single Tensor
    if isinstance(args, torch.Tensor):
        args = (args, )

    trace, torch_out = torch.jit.get_trace_graph(func, args, _force_outplace=True)
    trace.set_graph(_optimize_graph(trace.graph(), operator_export_type))
    if return_outs:
        return trace, torch_out
    return trace


def _trace_and_get_graph_from_model(model, args, training):

    # A basic sanity check: make sure the state_dict keys are the same
    # before and after running the model.  Fail fast!
    orig_state_dict_keys = _unique_state_dict(model).keys()

    # By default, training=False, which is good because running a model in
    # training mode could result in internal buffers getting updated, dropout
    # getting applied, etc.  If you really know what you're doing, you
    # can turn training=True (or None, to preserve whatever the original
    # training mode was.)
    with set_training(model, training):
        trace, torch_out = torch.jit.get_trace_graph(model, args, _force_outplace=True)

    if orig_state_dict_keys != _unique_state_dict(model).keys():
        raise RuntimeError("state_dict changed after running the tracer; "
                           "something weird is happening in your model!")

    return trace.graph(), torch_out


def _get_source_node(block):
    # NOTE: This method returns the first prim::Param node it encounters,
    # which is fine assuming that there is always one and only one prim::Param
    # node in a PyTorch IR graph.
    for node in block.nodes():
        for val in node.inputs():
            if val.node().kind() == 'prim::Param':
                return val.node()

    raise RuntimeError("Malformed IR graph - did not find any prim::Param node in the block.")


def _run_torch_backend_for_onnx(node, input_tensor_values):    
    if node.kind() == 'onnx::Slice':
        assert len(input_tensor_values) == 1
        attrs = {k: node[k] for k in node.attributeNames()}
        if not ('axes' in attrs and 'starts' in attrs and 'ends' in attrs):
            raise RuntimeError('\'onnx::Slice\' node must have attributes \'axes\', \
                \'starts\', and \'ends\'.')
        if (len(attrs['axes']) != len(attrs['starts'])) or \
            (len(attrs['axes']) != len(attrs['ends'])):
            raise RuntimeError('\'onnx::Slice\' node attributes \'axes\', \
                \'starts\', and \'ends\' must have same length.')
        updated_val = input_tensor_values[0]
        for dim, start, end in zip(attrs['axes'], attrs['starts'], attrs['ends']):
            updated_val = torch.narrow(updated_val, dim, start, end - start)

    elif node.kind() == 'onnx::Concat':
        attrs = {k: node[k] for k in node.attributeNames()}
        updated_val = torch.cat(input_tensor_values, dim=attrs['axis'])

    elif node.kind() == 'onnx::Unsqueeze':
        assert len(input_tensor_values) == 1
        attrs = {k: node[k] for k in node.attributeNames()}
        if 'axes' not in attrs:
            raise RuntimeError("\'onnx::Unsqueeze\' node must have attribute \'axes\'.")
        updated_val = input_tensor_values[0]
        for dim in attrs['axes']:
            updated_val = torch.unsqueeze(updated_val, dim)

    elif node.kind() == 'onnx::Transpose':
        assert len(input_tensor_values) == 1
        attrs = {k: node[k] for k in node.attributeNames()}
        if 'perm' not in attrs:
            raise RuntimeError("\'onnx::Transpose\' node must have attribute \'perm\'.")
        updated_val = input_tensor_values[0].permute(attrs['perm'])

    else:
        updated_val = None

    return updated_val


def _erase_unused_outputs(node):
    for k in reversed(range(node.outputsSize())):
        if not node.outputsAt(k).hasUses():
            node.eraseOutput(k)


def _optimize_graph_constant_folding(block, params_dict):
    # # This method updates the graph in-place to replace
    # # all the one-time constant-based computations into 
    # # a constant or initializer node.
    class LeafNodes:
        # Currently only prim::Param and prim::Constant are supported.
        # More can be added if needed.
        PRIM_PARAM = 0
        ONNX_CONSTANT = 1

    # TODO: Add a while loop to encapsulates this, so that we can 
    # do two more levels deep constant folding.
    source_node = _get_source_node(block)
    for node in block.nodes():
        for nested_block in node.blocks():
            _optimize_graph_constant_folding(nested_block, params_dict)

        input_vals = list(node.inputs())
        input_tensor_values = []
        kind_of_leaf_node = [] # Only two states needed for now, hence boolean       
        for val in input_vals:
            # print(val)
            input_node = val.node()
            # if node.kind() == 'onnx::Unsqueeze' and "Constant" in input_node.kind():
            # if node.kind() == 'onnx::Unsqueeze' and input_node.kind() == 'onnx::Constant':
            #     print("Node with input coming from Constant node.")
            # The second condition in the statement below is needed because actual
            # inputs (not params) are also outputs of the prim::Param node.
            is_param = input_node.kind() == 'prim::Param' and val.uniqueName() in params_dict
            if is_param:
                assert input_node is source_node # Always one and only one prim::Param source node in a PTIR graph.

            is_constant = input_node.kind() == "onnx::Constant" and \
                not input_node.mustBeNone() and \
                (input_node.kindOf("value") == "t" or input_node.kindOf("value") == "is")
            if is_param:
                input_tensor_values.append(params_dict[val.uniqueName()])
                kind_of_leaf_node.append(LeafNodes.PRIM_PARAM)
            elif is_constant:
                input_tensor_values.append(input_node["value"])
                kind_of_leaf_node.append(LeafNodes.ONNX_CONSTANT)

        if input_tensor_values and len(input_tensor_values) == len(input_vals):
            # Do folding for this node and delete the node
            # print(input_tensor_values)
            updated_val = _run_torch_backend_for_onnx(node, input_tensor_values)
            if updated_val is None:
                # Skip this node
                continue            
            new_source_node_output = source_node.addOutput()
            params_dict[new_source_node_output.uniqueName()] = updated_val # Assumes 'node' is single output
            # new_source_node_output.copyMetadata(node.outputsAt(0))
            new_source_node_output.inferTypeFrom(updated_val)
            node.outputsAt(0).replaceAllUsesWith(new_source_node_output) # Assumes 'node' is single output
            # TODO: Shall we copy metadata of the output value above using Value::copyMetadata?

            source_output_names = [source_output.uniqueName() for source_output in source_node.outputs()]
            idxs_matching_source_output = []
            for idx, val in enumerate(input_vals):
                if kind_of_leaf_node[idx] == LeafNodes.PRIM_PARAM:
                    # Find the output of the source prim::Param node that corresponds
                    # to this input, check to see if that output is feeding into any
                    # other node, and if it is not (given that we replaced it above)
                    # then delete this output of the prim::Param node, and the corresponding
                    # entry in params_dict.

                    idx_matching_source_output = [i for i in range(len(source_output_names)) \
                        if source_output_names[i] == val.uniqueName()]
                    assert len(idx_matching_source_output) == 1 # Only one source output name should match the name of this input (val)
                    idxs_matching_source_output.append(idx_matching_source_output[0])

            node.removeAllInputs()
            # TODO: This for loop below needs to be done only for PRIM_PARAM case.
            # Can this be brought before removeAllInputs() above and into the if PRIM_PARAM
            # code block above? I think it can be.
            for node_idx_in_source_output in idxs_matching_source_output:
                if len(list(source_node.outputsAt(node_idx_in_source_output).uses())) == 0:
                        # # Delete the particular output of the source prim::Param node
                        # source_node.eraseOutput(idx_matching_source_output[0])

                        # Delete the corresponding entry in params_dict
                        del params_dict[source_output_names[node_idx_in_source_output]]

    _erase_unused_outputs(source_node)
    # for k in reversed(range(source_node.outputsSize())):
    #     # if len(list(source_node.outputsAt(k).uses())) == 0:
    #     if not source_node.outputsAt(k).hasUses():
    #         # print(source_node.outputsAt(k).uniqueName())
    #         source_node.eraseOutput(k)

    # source_node_outputs = list(source_node.outputs())
    # output_idxs_to_remove = [j for j in range(len(source_node_outputs)) \
    #                 if len(list(source_node_outputs[j].uses())) == 0]
    # for idx_to_remove in output_idxs_to_remove:
    #     print(source_node_outputs[idx_to_remove].uniqueName())
    #     source_node.eraseOutput(idx_to_remove)
    print('Constant Folding Done!')
            # for idx, val in enumerate(input_vals):                
            #     if kind_of_leaf_node[idx] == LeafNodes.PRIM_PARAM:
            #         # Find the output of the source prim::Param node that corresponds
            #         # to this input, check to see if that output is feeding into any
            #         # other node, and if it is not (given that we replaced it above)
            #         # then delete this output of the prim::Param node, and the corresponding
            #         # entry in params_dict.
            #         source_output_names = [source_output.uniqueName() for source_output in source_node.outputs()]
            #         idx_matching_source_output = [i for i in range(len(source_output_names)) \
            #             if source_output_names[i] == val.uniqueName()]
            #         assert len(idx_matching_source_output) == 1 # Only one source output name should match the name of this input (val)
            #         node.removeInput(idx) # Needs to be done here so that in the next line when we check uses(), we get a 0.
            #         if len(list(source_node.outputsAt(idx_matching_source_output[0]).uses())) == 0:
            #             # Delete the particular output of the source prim::Param node
            #             # source_node.eraseOutput(idx_matching_source_output[0])
            #             # Delete the corresponding entry in params_dict
            #             del params_dict[source_output_names[idx_matching_source_output[0]]]
            #     elif kind_of_leaf_node[idx] == LeafNodes.PRIM_CONSTANT:
            #         print('Encountered prim::Constant node. Nothing to do here.')
            #         # node.removeAllInputs()
            #     else:
            #         raise RuntimeError("Unsupported LeafNodes category encountered during constant folding.")


            # node.removeAllInputs()
    # source_node_outputs = list(source_node.outputs())
    # output_idxs_to_remove = [j for j in range(len(source_node_outputs)) \
    #                 if len(list(source_node_outputs[j].uses())) == 0]
    # for idx_to_remove in output_idxs_to_remove:
    #     print(source_node_outputs[idx_to_remove].uniqueName())
    #     source_node.eraseOutput(idx_to_remove)


def _model_to_graph(model, args, f, verbose=False, training=False,
                    input_names=None, output_names=None,
                    operator_export_type=OperatorExportTypes.ONNX,
                    example_outputs=None, propagate=False, do_constant_folding=False):
    # Special case for common case of passing a single Tensor
    if isinstance(args, torch.Tensor):
        args = (args, )

    if isinstance(model, torch.jit.ScriptModule):
        torch_out = None
        assert example_outputs is not None, "example_outputs must be provided when exporting a ScriptModule"
        if isinstance(example_outputs, torch.Tensor):
            example_outputs = [example_outputs]
        try:
            method = model.__getattr__('forward')
            graph = method.propagate_and_assign_input_and_output_shapes(
                args, example_outputs, False, propagate)
            # Erase number types to bring the graph to a pre-NumberType state
            params = method.initial_ivalues()
        except AttributeError:
            # TODO: just trace it
            raise RuntimeError('\'forward\' method must be a script method')
    else:
        graph, torch_out = _trace_and_get_graph_from_model(model, args, training)
        params = list(_unique_state_dict(model).values())

    input_and_param_names = [val.uniqueName() for val in graph.inputs()]
    param_names = input_and_param_names[len(input_and_param_names) - len(params):]
    params_dict = dict(zip(param_names, params))

    graph = _optimize_graph(graph, operator_export_type)

    # NB: ONNX requires complete information about output types, which might be
    # erased by some optimizations, so we need to set it explicitly again.
    if torch_out is not None:
        output_tensors, _ = torch._C._jit_flatten(torch_out)
        for output, tensor in zip(graph.outputs(), output_tensors):
            output.inferTypeFrom(tensor)

    _set_input_and_output_names(graph, input_names, output_names)

    if do_constant_folding:
        # Works on ONNX graphs, therefore must come after _optimize_graph() call.
        _optimize_graph_constant_folding(graph.block(), params_dict)

    if verbose:
        print(graph)

    return graph, params_dict, torch_out


def export_to_pretty_string(model, args, f, export_params=True, verbose=False, training=False,
                            input_names=None, output_names=None, aten=False, export_raw_ir=False,
                            operator_export_type=None, export_type=ExportTypes.PROTOBUF_FILE,
                            example_outputs=None, propagate=False, google_printer=False,
                            opset_version=None):
    if aten or export_raw_ir:
        assert operator_export_type is None
        assert aten ^ export_raw_ir
        operator_export_type = OperatorExportTypes.ATEN if aten else OperatorExportTypes.RAW
    elif operator_export_type is None:
        operator_export_type = OperatorExportTypes.ONNX
    return _export_to_pretty_string(model, args, f, export_params, verbose, training,
                                    input_names, output_names, operator_export_type,
                                    export_type, example_outputs, propagate, google_printer,
                                    opset_version)


def _export_to_pretty_string(model, args, f, export_params=True, verbose=False, training=False,
                             input_names=None, output_names=None, operator_export_type=OperatorExportTypes.ONNX,
                             export_type=ExportTypes.PROTOBUF_FILE, example_outputs=None, propagate=False,
                             google_printer=False, opset_version=None, do_constant_folding=True):
    from torch.onnx.symbolic import _default_onnx_opset_version, _set_opset_version
    if opset_version is None:
        opset_version = _default_onnx_opset_version
    _set_opset_version(opset_version)
    graph, params, torch_out = _model_to_graph(model, args, f, verbose,
                                               training, input_names,
                                               output_names, operator_export_type,
                                               example_outputs, propagate, do_constant_folding)

    return graph._pretty_print_onnx(params, opset_version, False, operator_export_type, google_printer)


# NOTE: the output `torch_out` will contain the output tensors resulting from
# the trace of a Module. In the case that a torch.nn.ScriptModule is passed in,
# this output will be None, since we are not doing any tracing but rather
# directly extracting the graph.
def _export(model, args, f, export_params=True, verbose=False, training=False,
            input_names=None, output_names=None, operator_export_type=OperatorExportTypes.ONNX,
            export_type=ExportTypes.PROTOBUF_FILE, example_outputs=None, propagate=False,
            opset_version=None, do_constant_folding=False):
    from torch.onnx.symbolic import _default_onnx_opset_version, _set_opset_version
    if opset_version is None:
        opset_version = _default_onnx_opset_version
    _set_opset_version(opset_version)
    graph, params_dict, torch_out = _model_to_graph(model, args, f, verbose,
                                                    training, input_names,
                                                    output_names, operator_export_type,
                                                    example_outputs, propagate, do_constant_folding)

    # TODO: Don't allocate a in-memory string for the protobuf
    defer_weight_export = export_type is not ExportTypes.PROTOBUF_FILE
    if export_params:
        proto, export_map = graph._export_onnx(params_dict, opset_version, defer_weight_export, operator_export_type)
    else:
        proto, export_map = graph._export_onnx({}, opset_version, False, operator_export_type)

    if export_type == ExportTypes.PROTOBUF_FILE:
        assert(len(export_map) == 0)
        torch.serialization._with_file_like(f, "wb", lambda f: f.write(proto))
    elif export_type in [ExportTypes.ZIP_ARCHIVE, ExportTypes.COMPRESSED_ZIP_ARCHIVE]:
        import zipfile
        compression = zipfile.ZIP_DEFLATED \
            if export_type == ExportTypes.COMPRESSED_ZIP_ARCHIVE \
            else zipfile.ZIP_STORED
        with zipfile.ZipFile(f, 'w', compression=compression) as z:
            z.writestr(ONNX_ARCHIVE_MODEL_PROTO_NAME, proto)
            for k, v in export_map.items():
                z.writestr(k, v)
    elif export_type == ExportTypes.DIRECTORY:
        import os
        if os.path.exists(f):
            assert(os.path.isdir(f))
        else:
            os.makedirs(f)

        model_proto_file = os.path.join(f, ONNX_ARCHIVE_MODEL_PROTO_NAME)
        torch.serialization._with_file_like(
            model_proto_file, "wb", lambda f: f.write(proto))

        for k, v in export_map.items():
            weight_proto_file = os.path.join(f, k)
            torch.serialization._with_file_like(
                weight_proto_file, "wb", lambda f: f.write(v))
    else:
        raise RuntimeError('Unknown export type')
    return torch_out


def _set_input_and_output_names(graph, input_names, output_names):
    def set_names(node_list, name_list, descriptor):
        if name_list is None:
            return
        if len(name_list) > len(node_list):
            raise RuntimeError(
                "number of %s names provided (%d) exceeded number of %ss (%d)"
                % (descriptor, len(name_list), descriptor, len(node_list)))
        for name, node in zip(name_list, node_list):
            if node.uniqueName() != name:
                node.setUniqueName(name)
    set_names(list(graph.inputs()), input_names, 'input')
    set_names(list(graph.outputs()), output_names, 'output')

attr_pattern = re.compile("^(.+)_([ifstgz])$")


def _run_symbolic_method(op_name, symbolic_fn, args):
    r"""
    This trampoline function gets invoked for every symbolic method
    call from C++.
    """
    try:
        return symbolic_fn(*args)
    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch
        # to symbolic_fn.  Otherwise, the backtrace will have the clues
        # you need.
        e.args = ("{} (occurred when translating {})".format(e.args[0], op_name), )
        raise


def _is_onnx_list(value):
    if not isinstance(value, string_classes) and \
            not isinstance(value, torch.Tensor) and \
            isinstance(value, container_abcs.Iterable):
        return True
    return False


def _add_attribute(node, key, value, aten):
    r""" initializes the right attribute based on type of value """
    m = attr_pattern.match(key)
    if m is None:
        raise IndexError((
            "Invalid attribute specifier '{}' names " +
            " must be suffixed with type, e.g. 'dim_i' or 'dims_i'").format(key))
    name, kind = m.group(1), m.group(2)
    if _is_onnx_list(value):
        kind += "s"
    if aten:
        if isinstance(value, torch.Tensor):
            # Caffe2 proto does not support tensor attribute.
            if value.numel() > 1:
                raise ValueError("Should not pass tensor attribute")
            value = _scalar(value)
            if isinstance(value, float):
                kind = "f"
            else:
                kind = "i"
    return getattr(node, kind + "_")(name, value)


def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x[0]


def _newNode(g, opname, outputs, *args, **kwargs):
    if "::" in opname:
        aten = False
        ns_opname = opname
    else:
        aten = kwargs.pop("aten", False)
        ns = "aten" if aten else "onnx"
        ns_opname = ns + "::" + opname
    n = g.create(ns_opname, args, outputs)
    for k, v in sorted(kwargs.items()):
        # TODO: enable inplace in aten exporting mode.
        if k == "inplace":
            continue
        _add_attribute(n, k, v, aten=aten)
    return n


def _graph_op(g, opname, *raw_args, **kwargs):
    r"""
    Create an ONNX operator 'opname', taking 'args' as inputs and attributes
    'kwargs'; returning the node representing the single output of this operator
    (see the `outputs` keyword argument for multi-return nodes).

    The set of operators and the inputs/attributes they take
    is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

    This function is monkey-patched onto Graph.

    Arguments:
        opname (string): The ONNX operator name, e.g., `Abs` or `Add`.
        args (Node...): The inputs to the operator; usually provided
            as arguments to the `symbolic` definition.
        kwargs: The attributes of the ONNX operator, with keys named
            according to the following convention: `alpha_f` indicates
            the `alpha` attribute with type `f`.  The valid type specifiers are
            `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
            specified with type float accepts either a single float, or a
            list of floats (e.g., you would say `dims_i` for a `dims` attribute
            that takes a list of integers).
        outputs (int, optional):  The number of outputs this operator returns;
            by default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Node`, representing each output of the ONNX operator
            in positional.
    """
    outputs = kwargs.pop('outputs', 1)

    # Filter out None attributes, this can be convenient client side because
    # now they can pass through None attributes, and have them not show up
    kwargs = dict((k, v) for k, v in kwargs.items() if v is not None)

    def const_if_tensor(arg):
        if arg is None:
            return arg
        elif isinstance(arg, torch._C.Value):
            return arg
        else:
            return g.op("Constant", value_z=arg)

    args = list(const_if_tensor(arg) for arg in raw_args)
    n = g.insertNode(_newNode(g, opname, outputs, *args, **kwargs))
    if outputs == 1:
        return n.output()
    return tuple(o for o in n.outputs())


# Note [Export inplace]
# ~~~~~~~~~~~~~~~~~~~~~
# In abstract, it would be better for us to export inplace annotations,
# than to not export them, since it is useful information that can
# help the target of an ONNX export export more efficiently.  However,
# ONNX doesn't currently formalize inplace.  Fortunately, it's sound to drop
# inplace annotations, but we are losing information this way.


def _run_symbolic_function(g, n, inputs, env, operator_export_type=OperatorExportTypes.ONNX):
    # NB: Returning None means the node gets cloned as is into
    # the new graph
    try:
        import torch.onnx.symbolic

        # See Note [Export inplace]
        # TODO: I think this is not necessary anymore
        if n.kind().endswith('_'):
            ns_op_name = n.kind()[:-1]
        else:
            ns_op_name = n.kind()
        ns, op_name = ns_op_name.split("::")

        if ns == "onnx":
            # Use the original node directly
            return None

        elif ns == "aten":
            is_exportable_aten_op = hasattr(torch.onnx.symbolic, op_name)
            is_onnx_aten_export = operator_export_type == OperatorExportTypes.ONNX_ATEN
            is_aten_fallback_export = operator_export_type == OperatorExportTypes.ONNX_ATEN_FALLBACK
            if is_onnx_aten_export or (not is_exportable_aten_op and is_aten_fallback_export):
                # Direct ATen export requested
                attrs = {k + "_" + n.kindOf(k)[0]: n[k] for k in n.attributeNames()}
                outputs = n.outputsSize()
                attrs["outputs"] = outputs
                return _graph_at(g, op_name, *inputs, aten=True, **attrs)

            else:
                # Export it regularly
                attrs = {k: n[k] for k in n.attributeNames()}
                if not is_exportable_aten_op:
                    warnings.warn("ONNX export failed on ATen operator {} because torch.onnx.symbolic.{} does not exist"
                                  .format(op_name, op_name))
                    return None
                fn = getattr(torch.onnx.symbolic, op_name)
                return fn(g, *inputs, **attrs)

        elif ns == "prim":
            if op_name == "Constant" and not n.mustBeNone():
                if n.kindOf("value") == "t":
                    return g.op("Constant", value_t=n["value"])
                elif n.kindOf("value") == "is":
                    value = torch.stack([torch.tensor(v) for v in n["value"]]) if n["value"] else []
                    return g.op("Constant", value_t=value)
                elif n.output().type().kind() == "DeviceObjType":
                    return None
                else:
                    raise RuntimeError("Unsupported prim::Constant kind: `{}`. Send a bug report.".format(
                        n.kindOf("value")))
            elif n.mustBeNone() or op_name == "ListConstruct" or op_name == "ListUnpack":
                # None is not an ONNX operator; keep it as None
                # let the exporter handle finally eliminating these

                # For ListConstruct/ListUnpack, it will be erased in the ONNX peephole pass
                return None
            elif op_name == 'Loop' or op_name == 'If':
                new_op_outputs = g.op(op_name, *inputs, outputs=n.outputsSize())
                new_node = new_op_outputs[0].node() if n.outputsSize() > 1 else new_op_outputs.node()
                for b in n.blocks():
                    new_block = new_node.addBlock()
                    torch._C._jit_pass_onnx_block(b, new_block, operator_export_type, env)
                return new_op_outputs
            else:
                symbolic_name = 'prim_' + op_name
                symbolic_fn = getattr(torch.onnx.symbolic, symbolic_name, None)
                if symbolic_fn is None:
                    warnings.warn("ONNX export failed on primitive operator {}; please report a bug".format(op_name))
                    return None
                attrs = {k: n[k] for k in n.attributeNames()}
                return symbolic_fn(g, *inputs, **attrs)

        else:
            warnings.warn("ONNX export failed on an operator with unrecognized namespace {}::{}; "
                          "please report a bug".format(ns, op_name))
            return None

    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch.
        # Otherwise, the backtrace will have the clues you need.
        e.args = ("{} (occurred when translating {})".format(e.args[0], op_name), )
        raise


# Generate an ONNX ATen op node.
def _graph_at(g, opname, *args, **kwargs):
    return g.op("ATen", *args, operator_s=opname, **kwargs)


# This helper function can create either constant tensor or constant scalar.
# If dims is None or 0 or [0], generate a 0-d tensor (scalar).
#
# TODO: We might not need this anymore, since most scalars now show up
# as tensors
def _graph_constant(g, value, dims, type, *args, **kwargs):
    assert isinstance(value, numbers.Number)
    assert type is not None
    isscalar = False
    if dims is None or dims == 0 or set(dims) == set([0]):
        dims = [1]
        isscalar = True
    type = type.lower()
    if type == "char":
        tensor = torch.CharTensor(*dims)
    elif type == "short":
        tensor = torch.ShortTensor(*dims)
    elif type == "int":
        tensor = torch.IntTensor(*dims)
    elif type == "long":
        tensor = torch.LongTensor(*dims)
    elif type == "half":
        tensor = torch.HalfTensor(*dims)
    elif type == "float":
        tensor = torch.FloatTensor(*dims)
    elif type == "double":
        tensor = torch.DoubleTensor(*dims)
    else:
        raise ValueError("Unknown type, type should be one of the following strings: "
                         "char, short, int, long, half, float, double")
    tensor.fill_(value)
    if isscalar:
        return g.op("Constant", *args, value_z=tensor, **kwargs)
    return g.op("Constant", *args, value_t=tensor, **kwargs)


def _node_getitem(self, k):
    r"""
    Accessor for attributes of a node which is polymorphic over
    return type.

    NB: This is monkey-patched onto Node.
    """
    sel = self.kindOf(k)
    return getattr(self, sel)(k)


torch._C.Graph.op = _graph_op
torch._C.Graph.at = _graph_at
torch._C.Graph.constant = _graph_constant
torch._C.Node.__getitem__ = _node_getitem
