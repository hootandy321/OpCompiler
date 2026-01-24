import functools
import inspect
import itertools
import ast

import ninetoothed.jit
import ninetoothed.naming as naming
from ninetoothed.generation import cache_source
from ninetoothed.jit import import_from_path
from ninetoothed.make import make
from ninetoothed.symbol import Symbol
from ninetoothed.tensor import Tensor


class Node:
    def __init__(self, kernel, args=None, kwargs=None):
        if args is None:
            args = ()

        if kwargs is None:
            kwargs = {}

        self.kernel = kernel

        self.args = args

        self.kwargs = kwargs


class ProducerConsumerInfo:
    """Information about producer-consumer relationships between kernels."""

    def __init__(self):
        # Maps intermediate tensor position in input_kernel.args to output position
        self.intermediate_tensors = {}
        # Position of producer's output tensor in its args
        self.producer_output_positions = []
        # Position of consumer's input tensor in its args that receives producer output
        self.consumer_input_positions = []

    def has_intermediates(self):
        return len(self.intermediate_tensors) > 0


def _detect_producer_consumer(input_node, other_node):
    """Detect producer-consumer relationships between two kernel nodes.

    Returns ProducerConsumerInfo identifying which output tensors from
    input_node are consumed as inputs by other_node.
    """
    info = ProducerConsumerInfo()

    input_kernel = input_node.kernel
    other_kernel = other_node.kernel

    # Get the number of input and output tensors for each kernel
    # Convention: last tensor in args is typically the output
    input_num_params = len(inspect.signature(input_kernel.application).parameters)
    other_num_params = len(inspect.signature(other_kernel.application).parameters)

    # Assume last parameter is output for simple elementwise operations
    input_output_pos = input_num_params - 1
    other_input_positions = list(range(other_num_params - 1))

    # Check if input_kernel's output tensor is used as input to other_kernel
    if len(input_node.args) > input_output_pos:
        input_output_tensor = input_node.args[input_output_pos]

        for other_pos in other_input_positions:
            if other_pos < len(other_node.args):
                other_input_tensor = other_node.args[other_pos]
                if input_output_tensor is other_input_tensor:
                    info.intermediate_tensors[input_output_pos] = other_pos
                    info.producer_output_positions.append(input_output_pos)
                    info.consumer_input_positions.append(other_pos)

    return info


def _fuse_nodes(nodes):
        if len(nodes) == 1:
            return (Node(nodes[0].kernel, args=nodes[0].args, kwargs=nodes[0].kwargs),)

        fused = functools.reduce(_fuse_node_pair, nodes)

        if fused is None:
            return nodes

        return (fused,)

def fuser(graph_module, _example_inputs):
    graph = graph_module.graph

    ninetoothed_nodes = []
    past_args = set()

    def _is_hoistable(node):
        if hasattr(node.target, "__name__") and node.target.__name__ in (
            "empty",
            "empty_like",
        ):
            return True

        for arg in _iterate_recursively(
            itertools.chain(node.args, node.kwargs.values())
        ):
            if arg in past_args:
                return False

        return True 

    for node in graph.nodes:
        if isinstance(node.target, ninetoothed.jit.__globals__["_Handle"]):
            ninetoothed_node = Node(node.target, args=node.args, kwargs=node.kwargs)
            ninetoothed_nodes.append(ninetoothed_node)
            past_args.update(
                _iterate_recursively(itertools.chain(node.args, node.kwargs.values()))
            )
            graph.erase_node(node)

            continue

        if not _is_hoistable(node):
            if ninetoothed_nodes:
                with graph.inserting_before(node):
                    for ninetoothed_node in _fuse_nodes(ninetoothed_nodes):
                        graph.call_function(
                            ninetoothed_node.kernel,
                            args=ninetoothed_node.args,
                            kwargs=ninetoothed_node.kwargs,
                        )

                ninetoothed_nodes = []
                past_args = set()

            continue

    return graph_module.forward


class _FusionInfo:
    def __init__(self, input_prefix, input_suffix, other_prefix, other_suffix,
                 producer_consumer_info=None):
        self.input_prefix = input_prefix

        self.input_suffix = input_suffix

        self.other_prefix = other_prefix

        self.other_suffix = other_suffix

        self.producer_consumer_info = producer_consumer_info or ProducerConsumerInfo()


def _fuse_node_pair(input_node, other_node):
    if input_node.kwargs or other_node.kwargs:
        return None

    input_kernel = input_node.kernel
    other_kernel = other_node.kernel

    # Detect producer-consumer relationships
    pc_info = _detect_producer_consumer(input_node, other_node)

    mapping = {}

    for other_position, arg in enumerate(other_node.args):
        if arg not in input_node.args:
            continue

        mapping[other_position] = input_node.args.index(arg)

    fused_kernel = _fuse_kernel_pair(input_kernel, other_kernel, mapping, pc_info)

    if fused_kernel is None:
        return None

    # Keep all tensors in fused_args for compatibility with arrangement
    # The application fusion will optimize the actual computation
    fused_args = input_node.args + other_node.args
    fused_kwargs = input_node.kwargs | other_node.kwargs

    fused_node = Node(fused_kernel, args=fused_args, kwargs=fused_kwargs)

    return fused_node


def _fuse_kernel_pair(input_kernel, other_kernel, mapping, pc_info=None):
    if pc_info is None:
        pc_info = ProducerConsumerInfo()

    arrangement, tensors, fusion_info = _fuse_arrangement_pair(
        input_kernel, other_kernel, mapping, pc_info
    )

    if arrangement is None:
        return None

    application = _fuse_application_pair(input_kernel, other_kernel, fusion_info)

    if application is None:
        return None

    input_num_warps = (
        input_kernel.num_warps
        if not isinstance(input_kernel.num_warps, int)
        else (input_kernel.num_warps,)
    )
    other_num_warps = (
        other_kernel.num_warps
        if not isinstance(other_kernel.num_warps, int)
        else (other_kernel.num_warps,)
    )

    num_warps = tuple(set(input_num_warps) | set(other_num_warps))

    input_num_stages = (
        input_kernel.num_stages
        if not isinstance(input_kernel.num_stages, int)
        else (input_kernel.num_stages,)
    )
    other_num_stages = (
        other_kernel.num_stages
        if not isinstance(other_kernel.num_stages, int)
        else (other_kernel.num_stages,)
    )

    num_stages = tuple(set(input_num_stages) | set(other_num_stages))

    if input_kernel.max_num_configs is None or other_kernel.max_num_configs is None:
        max_num_configs = None
    else:
        max_num_configs = max(
            input_kernel.max_num_configs, other_kernel.max_num_configs
        )

    return make(
        arrangement,
        application,
        tensors,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )


def _fuse_arrangement_pair(input_kernel, other_kernel, mapping, pc_info=None):
    if pc_info is None:
        pc_info = ProducerConsumerInfo()

    input_arrangement = input_kernel.arrangement
    other_arrangement = other_kernel.arrangement

    def rename_tensor(tensor, name):
        return Tensor(
            tensor.ndim,
            other=tensor.source.other,
            shape_options=tensor.shape_options,
            name=name,
        )

    input_tensors = tuple(
        rename_tensor(tensor, f"{tensor.name}_0") for tensor in input_kernel.tensors
    )
    other_tensors = tuple(
        rename_tensor(tensor, f"{tensor.name}_1") for tensor in other_kernel.tensors
    )

    input_tensors_arranged = input_arrangement(*input_tensors)
    other_tensors_arranged = other_arrangement(*other_tensors)

    input_tensor_positions = tuple(mapping.values())
    other_tensor_positions = tuple(mapping.keys())

    mapping = {}

    for old_tensor, new_tensor in itertools.chain(
        zip(input_kernel.tensors, input_tensors),
        zip(other_kernel.tensors, other_tensors),
    ):
        old_names = sorted(old_tensor.names(), key=str)
        new_names = sorted(new_tensor.names(), key=str)

        for old_name, new_name in zip(old_names, new_names):
            mapping[old_name] = new_name

    for input_tensor_position, other_tensor_position in zip(
        input_tensor_positions, other_tensor_positions
    ):
        for input_block_size, other_block_size in zip(
            input_tensors_arranged[input_tensor_position].innermost().shape,
            other_tensors_arranged[other_tensor_position].innermost().shape,
        ):
            for block_size in (input_block_size, other_block_size):
                if not (
                    Symbol.is_name(block_size) and naming.is_meta(block_size.node.id)
                ):
                    return None, None

            new_lower_bound = max(
                input_block_size.lower_bound, other_block_size.lower_bound
            )
            new_upper_bound = min(
                input_block_size.upper_bound, other_block_size.upper_bound
            )

            new_block_size = ninetoothed.block_size(
                lower_bound=new_lower_bound, upper_bound=new_upper_bound
            )

            mapping[input_block_size] = new_block_size
            mapping[other_block_size] = new_block_size

    for tensor in itertools.chain(input_tensors_arranged, other_tensors_arranged):
        _replace_history(tensor, mapping)

    fusion_info = _get_fusion_info(
        input_tensors_arranged[input_tensor_positions[0]],
        other_tensors_arranged[other_tensor_positions[0]],
        pc_info,
    )

    if fusion_info is None:
        return None, None

    input_prefix = fusion_info.input_prefix
    input_suffix = fusion_info.input_suffix
    other_prefix = fusion_info.other_prefix
    other_suffix = fusion_info.other_suffix

    for input_tensor_position, other_tensor_position in zip(
        input_tensor_positions[1:], other_tensor_positions[1:]
    ):
        (input_prefix_, input_suffix_), (other_prefix_, other_suffix_) = _get_fusion_prefix_and_suffix(
            input_tensors_arranged[input_tensor_position],
            other_tensors_arranged[other_tensor_position],
        )

        if (
            input_prefix_ != input_prefix
            or input_suffix_ != input_suffix
            or other_prefix_ != other_prefix
            or other_suffix_ != other_suffix
        ):
            return None, None

    records_on_tensors = []
    tensors = []

    def _get_records_on_tensor(tensor):
        records = []

        curr = tensor

        while isinstance(curr, type(tensor)):
            records.append(list(curr._history))

            curr = curr.dtype

        return records

    for input_tensor_position, (input_tensor, input_tensor_arranged) in enumerate(
        zip(input_tensors, input_tensors_arranged)
    ):
        records_on_tensor = _get_records_on_tensor(input_tensor_arranged)

        records_on_tensor[0] = (
            type(records_on_tensor[0])(input_prefix)
            + records_on_tensor[0]
            + type(records_on_tensor[0])(input_suffix)
        )

        for func, _, _ in itertools.chain(input_prefix, input_suffix):
            if func is not Tensor.tile:
                continue

            records_on_tensor.insert(1, ())

        records_on_tensors.append(records_on_tensor)
        tensors.append(input_tensor)

    for other_tensor_position, (other_tensor, other_tensor_arranged) in enumerate(
        zip(other_tensors, other_tensors_arranged)
    ):
        records_on_tensor = _get_records_on_tensor(other_tensor_arranged)

        records_on_tensor[0] = (
            type(records_on_tensor[0])(other_prefix)
            + records_on_tensor[0]
            + type(records_on_tensor[0])(other_suffix)
        )

        for func, _, _ in itertools.chain(other_prefix, other_suffix):
            if func is not Tensor.tile:
                continue

            records_on_tensor.insert(1, ())

        records_on_tensors.append(records_on_tensor)
        tensors.append(other_tensor)

    def arrangement(*tensors):
        tensors_arranged = []

        for records_on_tensor, tensor in zip(records_on_tensors, tensors):
            records_on_level_iter = iter(records_on_tensor)

            prev = None
            curr = tensor

            while isinstance(curr, type(tensor)):
                records_on_level = next(records_on_level_iter)

                for func, args, kwargs in records_on_level:
                    curr = func(curr, *args, **kwargs)

                if prev is not None:
                    prev.dtype = curr
                else:
                    tensors_arranged.append(curr)

                prev = curr
                curr = curr.dtype

        return tuple(tensors_arranged)

    return arrangement, tuple(tensors), fusion_info


def _fuse_application_pair(input_kernel, other_kernel, fusion_info):
    """Fuse two application functions into one.

    If producer-consumer relationship exists (output of first kernel is input to second),
    generates truly fused code that eliminates intermediate memory access.
    """
    pc_info = fusion_info.producer_consumer_info

    # Check if we can do true vertical fusion (eliminate intermediate)
    if pc_info.has_intermediates():
        fused_app = _generate_vertically_fused_application(
            input_kernel, other_kernel, fusion_info, pc_info
        )
        if fused_app is not None:
            return fused_app

    # Fall back to horizontal fusion (sequential invocation)
    return _generate_horizontally_fused_application(
        input_kernel, other_kernel, fusion_info
    )


def _generate_vertically_fused_application(input_kernel, other_kernel, fusion_info, pc_info):
    """Generate a truly fused application that eliminates intermediate memory access.

    This analyzes the source of both application functions and combines them
    so that the producer's output expression directly substitutes into the
    consumer's input, keeping intermediate values in registers.

    The function maintains the same parameter signature as horizontal fusion
    for compatibility with the arrangement/tensor system.
    """
    try:
        input_app = input_kernel.application
        other_app = other_kernel.application

        # Get source code and parse AST for both applications
        input_source = inspect.getsource(input_app)
        other_source = inspect.getsource(other_app)

        input_tree = ast.parse(input_source)
        other_tree = ast.parse(other_source)

        # Extract the assignment expressions from each application
        input_func_def = input_tree.body[0]
        other_func_def = other_tree.body[0]

        input_params = [arg.arg for arg in input_func_def.args.args]
        other_params = [arg.arg for arg in other_func_def.args.args]

        # Find the output assignment in input kernel (typically last param = expr)
        input_output_expr = None
        input_output_param = input_params[-1] if input_params else None

        for stmt in input_func_def.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == input_output_param:
                        input_output_expr = stmt.value
                        break

        if input_output_expr is None:
            return None

        # Find which consumer input corresponds to producer output
        consumer_input_pos = pc_info.consumer_input_positions[0] if pc_info.consumer_input_positions else None
        if consumer_input_pos is None or consumer_input_pos >= len(other_params):
            return None

        consumer_intermediate_param = other_params[consumer_input_pos]

        # Generate parameter names for the fused function
        # Keep ALL parameters for compatibility with arrangement
        count = [0]

        def make_param():
            param = naming.auto_generate(f"parameter_{count[0]}")
            count[0] += 1
            return param

        # Build parameter mapping for input kernel (ALL params including output)
        input_param_mapping = {}
        fused_params = []
        for param in input_params:
            new_param = make_param()
            input_param_mapping[param] = new_param
            fused_params.append(new_param)

        # Build parameter mapping for other kernel (ALL params)
        other_param_mapping = {}
        for param in other_params:
            new_param = make_param()
            other_param_mapping[param] = new_param
            fused_params.append(new_param)

        # Rename variables in the input expression (using input params except output)
        class VarRenamer(ast.NodeTransformer):
            def __init__(self, mapping):
                self.mapping = mapping

            def visit_Name(self, node):
                if node.id in self.mapping:
                    return ast.Name(id=self.mapping[node.id], ctx=node.ctx)
                return node

        # Create a copy for substitution (don't include output param mapping)
        input_expr_mapping = {k: v for k, v in input_param_mapping.items() if k != input_output_param}
        renamed_input_expr = VarRenamer(input_expr_mapping).visit(
            ast.parse(ast.unparse(input_output_expr), mode='eval').body
        )

        # Find the output assignment in other kernel
        other_output_param = other_params[-1] if other_params else None
        other_output_expr = None

        for stmt in other_func_def.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == other_output_param:
                        other_output_expr = stmt.value
                        break

        if other_output_expr is None:
            return None

        # Substitute the intermediate parameter with the input expression
        class IntermediateSubstituter(ast.NodeTransformer):
            def __init__(self, intermediate_name, replacement_expr, var_mapping):
                self.intermediate_name = intermediate_name
                self.replacement_expr = replacement_expr
                self.var_mapping = var_mapping

            def visit_Name(self, node):
                if node.id == self.intermediate_name:
                    # Return a copy of the replacement expression
                    import copy
                    return copy.deepcopy(self.replacement_expr)
                if node.id in self.var_mapping:
                    return ast.Name(id=self.var_mapping[node.id], ctx=node.ctx)
                return node

        # Exclude intermediate param from mapping since it will be substituted
        other_expr_mapping = {k: v for k, v in other_param_mapping.items() 
                             if k != consumer_intermediate_param}
        
        fused_output_expr = IntermediateSubstituter(
            consumer_intermediate_param,
            renamed_input_expr,
            other_expr_mapping
        ).visit(ast.parse(ast.unparse(other_output_expr), mode='eval').body)

        # Get the output parameter name for the fused function
        fused_output_param = other_param_mapping[other_output_param]

        # Generate the fused application source
        fused_expr_str = ast.unparse(fused_output_expr)
        param_names = ", ".join(fused_params)

        _APPLICATION_NAME = "application"

        # Import any modules needed
        input_module = inspect.getmodule(input_app)
        other_module = inspect.getmodule(other_app)

        imports = set()
        if input_module and input_module.__name__ != "__main__":
            imports.add(f"import {input_module.__name__}")
        if other_module and other_module.__name__ != "__main__":
            imports.add(f"import {other_module.__name__}")

        # Check if ninetoothed.language is used
        if "ntl." in fused_expr_str or "ninetoothed.language" in fused_expr_str:
            imports.add("import ninetoothed.language as ntl")

        imports_str = "\n".join(sorted(imports)) if imports else ""

        # Generate the fused application
        # Note: We keep all parameters for compatibility, but only use what's needed
        application_source = f"""{imports_str}

def {_APPLICATION_NAME}({param_names}):
    # Vertically fused: intermediate value computed inline
    {fused_output_param} = {fused_expr_str}  # noqa: F841
"""

        source_file = cache_source(application_source)
        module = import_from_path(source_file.stem, source_file)
        module_vars = vars(module)

        return module_vars[_APPLICATION_NAME]

    except Exception:
        # If vertical fusion fails, return None to fall back to horizontal
        return None


def _generate_horizontally_fused_application(input_kernel, other_kernel, fusion_info):
    """Generate horizontally fused application (sequential invocation).

    This is the fallback when vertical fusion is not possible.
    """
    count = 0

    def _generate_invocation_info(application, prefix, suffix, indent=4):
        nonlocal count
        indentation = " " * indent

        def _make_param():
            nonlocal count

            param = naming.auto_generate(f"parameter_{count}")
            count += 1

            return param

        params = inspect.signature(application).parameters

        arg_names = [_make_param() for _ in params]

        param_names = ", ".join(arg_names)

        tile_func_count = 0

        for func, args, kwargs in itertools.chain(prefix, suffix):
            if func is not Tensor.tile:
                continue

            if len(args) != 1 or kwargs:
                return None, None, None

            tile_shape = args[0]

            if (
                any(tile_size not in (1, -1) for tile_size in tile_shape)
                or sum(1 if tile_size == -1 else 0 for tile_size in tile_shape) != 1
            ):
                return None, None, None

            tile_func_count += 1

            if tile_func_count > 1:
                return None, None, None

        def _generate_for_loop_source():
            reduction_dim = tile_shape.index(-1)

            index_name = naming.auto_generate(f"index_{count}")
            range_stop = f"{arg_names[0]}.shape[{reduction_dim}]"

            tensor_indexing = ", ".join(
                index_name if dim == reduction_dim else "0"
                for dim in range(len(tile_shape))
            )

            for i, arg_name in enumerate(arg_names):
                arg_names[i] = f"{arg_name}[{tensor_indexing}]"

            return f"for {index_name} in range({range_stop})"

        module = inspect.getmodule(application)

        invocation_source = ""

        if tile_func_count != 0:
            invocation_source += f"{_generate_for_loop_source()}:\n{indentation * 2}"

        invocation_source += (
            f"{module.__name__}.{application.__name__}({', '.join(arg_names)})"
        )

        return param_names, module, invocation_source

    input_param_names, input_module, input_invocation_source = (
        _generate_invocation_info(
            input_kernel.application, fusion_info.input_prefix, fusion_info.input_suffix
        )
    )

    if input_param_names is None:
        return None

    other_param_names, other_module, other_invocation_source = (
        _generate_invocation_info(
            other_kernel.application, fusion_info.other_prefix, fusion_info.other_suffix
        )
    )

    if other_param_names is None:
        return None

    param_names = f"{input_param_names}, {other_param_names}"

    _APPLICATION_NAME = "application"

    application_source = f"""import {input_module.__name__}
import {other_module.__name__}


def {_APPLICATION_NAME}({param_names}):
    {input_invocation_source}
    {other_invocation_source}
"""

    source_file = cache_source(application_source)

    module = import_from_path(source_file.stem, source_file)
    module_vars = vars(module)

    application = module_vars[_APPLICATION_NAME]

    return application



def _get_fusion_info(input, other, pc_info=None):
    (input_prefix, input_suffix), (other_prefix, other_suffix) = (
        _get_fusion_prefix_and_suffix(input, other)
    )

    if input_prefix is None:
        return None

    return _FusionInfo(input_prefix, input_suffix, other_prefix, other_suffix, pc_info)


def _get_fusion_prefix_and_suffix(input, other):
    if (fusion_position := _get_fusion_position(input, other)) is not None:
        prefix = tuple(input._history[:-fusion_position])
        suffix = tuple(other._history[fusion_position:])

        return ((), suffix), (prefix, ())

    if (fusion_position := _get_fusion_position(other, input)) is not None:
        prefix = tuple(other._history[:-fusion_position])
        suffix = tuple(input._history[fusion_position:])

        return (prefix, ()), ((), suffix)

    return (None, None), (None, None)


def _get_fusion_position(input, other):
    input_history = tuple(input._history)
    other_history = tuple(other._history)

    for k in range(min(len(input_history), len(other_history)), 0, -1):
        if input_history[-k:] == other_history[:k]:
            return k

    return None


def _replace_history(tensor, mapping):
    curr = tensor

    while isinstance(curr, type(tensor)):
        history = []

        for record in curr._history:
            record = _replace_record(record, mapping)

            history.append(record)

        curr._history = tuple(history)

        curr = curr.dtype


def _replace_record(record, mapping):
    return (record[0], _replace(record[1], mapping), _replace(record[2], mapping))


def _replace(object, mapping):
    if isinstance(object, (list, tuple, set)):
        return type(object)(_replace(item, mapping) for item in object)

    if isinstance(object, dict):
        return {
            _replace(key, mapping): _replace(value, mapping)
            for key, value in object.items()
        }

    if object in mapping:
        return mapping[object]

    if isinstance(object, Symbol):
        for old, new in mapping.items():
            object = object.find_and_replace(old, new)

        return object

    return object


def _iterate_recursively(object):
    if isinstance(object, (str, bytes)):
        yield object

        return

    if isinstance(object, dict):
        for value in object.values():
            yield from _iterate_recursively(value)

        return

    try:
        for item in object:
            yield from _iterate_recursively(item)

        return
    except TypeError:
        yield object
