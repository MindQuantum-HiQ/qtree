"""
Test programs to demonstrate various use cases of the
Qtree quantum circuit simulator. Functions in this file
can be used as main functions in the final simulator program
"""
import sys
import os
if os.path.abspath('../') not in sys.path:
    sys.path.append(os.path.abspath('../'))

import numpy as np
import cirq

import qtree.operators as ops
import qtree.optimizer as opt
import qtree.graph_model as gm
import qtree.np_framework as npfr

try:
    import tensorflow as tf
    import qtree.tf_framework as tffr
except ImportError:
    print('Tensorflow can not be imported. Interface disabled')
import qtree.utils as utils

from mpi4py import MPI


def get_amplitudes_from_cirq(filename, initial_state=0):
    """
    Calculates amplitudes for a circuit in file filename using Cirq
    """
    n_qubits, circuit = ops.read_circuit_file(filename)

    cirq_circuit = cirq.Circuit()

    for layer in circuit:
        cirq_circuit.append(op.to_cirq_1d_circ_op() for op in layer)

    print("Circuit:")
    print(cirq_circuit)
    simulator = cirq.Simulator()

    result = simulator.simulate(cirq_circuit, initial_state=initial_state)
    print("Simulation completed\n")

    # Cirq for some reason computes all amplitudes with phase -1j
    return result.final_state


def get_optimal_graphical_model(
        filename):
    """
    Builds a graphical model to contract a circuit in ``filename``
    and finds its tree decomposition
    """
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, data_dict, bra_vars, ket_vars = opt.circ2buckets(
        n_qubits, circuit)
    graph = gm.buckets2graph(buckets, ignore_variables=bra_vars+ket_vars)
    peo, tw = gm.get_peo(graph)
    graph_optimal, label_dict = gm.relabel_graph_nodes(
        graph, dict(zip(peo, range(1, len(peo) + 1)))
    )
    return graph_optimal


def eval_with_np(filename, initial_state=0):
    """
    Loads circuit from file and evaluates all amplitudes
    using the bucket elimination algorithm (with Numpy tensors).
    Same amplitudes are evaluated with Cirq for comparison.
    """
    # Prepare graphical model
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, data_dict, bra_vars, ket_vars = opt.circ2buckets(
        n_qubits, circuit)

    graph = gm.buckets2graph(
        buckets,
        ignore_variables=bra_vars+ket_vars)

    # Run quickbb
    peo, treewidth = gm.get_peo(graph)
    # place bra and ket variables to beginning, so these variables
    # will be contracted first
    peo = ket_vars + bra_vars + peo
    perm_buckets, perm_dict = opt.reorder_buckets(buckets, peo)

    # extract bra and ket variables from variable list and sort according
    # to qubit order
    ket_vars = sorted([perm_dict[idx] for idx in ket_vars], key=str)
    bra_vars = sorted([perm_dict[idx] for idx in bra_vars], key=str)

    # Take the subtensor corresponding to the initial state
    slice_dict = utils.slice_from_bits(initial_state, ket_vars)

    amplitudes = []
    for target_state in range(2**n_qubits):
        # Take appropriate subtensors for different target bitstrings
        slice_dict.update(
            utils.slice_from_bits(target_state, bra_vars)
        )
        sliced_buckets = npfr.get_sliced_np_buckets(
            perm_buckets, data_dict, slice_dict)
        result = opt.bucket_elimination(
            sliced_buckets, npfr.process_bucket_np)
        amplitudes.append(result.data)

    # Cirq returns the amplitudes in big endian (largest bit first)

    amplitudes_reference = get_amplitudes_from_cirq(
        filename, initial_state)
    print('Result:')
    print(np.round(np.array(amplitudes), 3))
    print('Reference:')
    print(np.round(np.array(amplitudes_reference), 3))
    print('Max difference:')
    print(np.max(np.abs(
        np.array(amplitudes) - np.array(amplitudes_reference))))


def prepare_parallel_evaluation_np(filename, n_var_parallel):
    """
    Prepares for parallel evaluation of the quantum circuit.
    Some of the variables in the circuit are parallelized over.
    Unsliced Numpy buckets in the optimal order of elimination
    are returned
    """
    # import pdb
    # pdb.set_trace()
    # Prepare graphical model
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, data_dict, bra_vars, ket_vars = opt.circ2buckets(
        n_qubits, circuit)

    graph = gm.buckets2graph(
        buckets,
        ignore_variables=bra_vars+ket_vars)

    # find a reduced graph
    vars_parallel, graph_reduced = gm.split_graph_by_metric_greedy(
        graph, n_var_parallel,
        metric_fn=gm.splitters.get_node_by_mem_reduction)

    # run quickbb once again to get peo and treewidth
    peo, treewidth = gm.get_peo(graph_reduced)

    # place bra and ket variables to beginning, so these variables
    # will be contracted first
    peo = ket_vars + bra_vars + vars_parallel + peo
    perm_buckets, perm_dict = opt.reorder_buckets(buckets, peo)

    # extract bra and ket variables from variable list and sort according
    # to qubit order
    ket_vars = sorted([perm_dict[idx] for idx in ket_vars], key=str)
    bra_vars = sorted([perm_dict[idx] for idx in bra_vars], key=str)
    vars_parallel = sorted([perm_dict[idx] for idx in vars_parallel],
                           key=str)

    environment = dict(
        bra_vars=bra_vars,
        ket_vars=ket_vars,
        vars_parallel=vars_parallel,
        buckets=perm_buckets,
        data_dict=data_dict,
    )

    return environment


def eval_with_np_parallel_mpi(filename, initial_state=0):
    """
    Evaluate quantum circuit using MPI to parallelize
    over some of the variables.
    """
    comm = MPI.COMM_WORLD
    comm_size = comm.size
    rank = comm.rank

    # number of variables to split by parallelization
    # this should be adjusted by the algorithm from memory/cpu
    # requirements
    n_var_parallel = 2
    if rank == 0:
        env = prepare_parallel_evaluation_np(filename, n_var_parallel)
    else:
        env = None

    env = comm.bcast(env, root=0)

    # restore buckets
    buckets = env['buckets']

    # restore other parts of the environment
    bra_vars = env['bra_vars']
    ket_vars = env['ket_vars']
    vars_parallel = env['vars_parallel']

    # restore data dictionary
    data_dict = env['data_dict']

    # Construct slice dictionary for initial state
    slice_dict = utils.slice_from_bits(initial_state, ket_vars)

    # Loop over all amplitudes
    amplitudes = []
    for target_state in range(2**len(bra_vars)):
        # Construct slice dictionary for the target state
        slice_dict.update(
            utils.slice_from_bits(target_state, bra_vars))

        # main computation loop. Populate respective slices
        # and do contraction

        amplitude = 0
        for parallel_slice_dict in utils.slice_values_generator(
                vars_parallel, rank, comm_size):
            slice_dict.update(parallel_slice_dict)
            sliced_buckets = npfr.get_sliced_np_buckets(
                buckets, data_dict, slice_dict)
            result = opt.bucket_elimination(
                sliced_buckets, npfr.process_bucket_np)

            amplitude += result.data

        amplitude = comm.reduce(amplitude, op=MPI.SUM, root=0)
        amplitudes.append(amplitude)

    if rank == 0:
        amplitudes_reference = get_amplitudes_from_cirq(filename)
        print('Result:')
        print(np.round(np.array(amplitudes), 3))
        print('Reference:')
        print(np.round(amplitudes_reference, 3))
        print('Max difference:')
        print(np.max(np.array(amplitudes)
                     - np.array(amplitudes_reference)))


def eval_with_tf(filename, initial_state=0):
    """
    Loads circuit from file and evaluates all amplitudes
    using the bucket elimination algorithm (with tensorflow tensors).
    Same amplitudes are evaluated with Cirq for comparison.
    """
    # Prepare graphical model
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, data_dict, bra_vars, ket_vars = opt.circ2buckets(
        n_qubits, circuit)

    graph = gm.buckets2graph(
        buckets,
        ignore_variables=bra_vars+ket_vars)

    # Run quickbb
    peo, treewidth = gm.get_peo(graph)
    # place bra and ket variables to beginning, so these variables
    # will be contracted first
    peo = ket_vars + bra_vars + peo
    perm_buckets, perm_dict = opt.reorder_buckets(buckets, peo)

    # extract bra and ket variables from variable list and sort according
    # to qubit order
    ket_vars = sorted([perm_dict[idx] for idx in ket_vars], key=str)
    bra_vars = sorted([perm_dict[idx] for idx in bra_vars], key=str)

    # Populate slice dict. Only shapes of slices are needed at this stage
    slice_dict = utils.slice_from_bits(initial_state, ket_vars)
    slice_dict.update(utils.slice_from_bits(
        initial_state, bra_vars))

    # create placeholders with proper shapes
    tf_buckets, placeholders_dict = tffr.get_sliced_tf_buckets(
        perm_buckets, slice_dict)

    # build the Tensorflow operation graph
    result = opt.bucket_elimination(
        tf_buckets, tffr.process_bucket_tf)
    comput_graph = result.data

    # prepare static part of the feed_dict
    feed_dict = tffr.assign_tensor_placeholders(
        placeholders_dict, data_dict)

    amplitudes = []
    for target_state in range(2**n_qubits):
        # Now the bounds of slices are needed
        slice_dict.update(
            utils.slice_from_bits(target_state, bra_vars))
        # populate feed dict with slice variables
        feed_dict.update(tffr.assign_variable_placeholders(
            placeholders_dict, slice_dict))

        amplitude = tffr.run_tf_session(comput_graph, feed_dict)
        amplitudes.append(amplitude)

    amplitudes_reference = get_amplitudes_from_cirq(filename,
                                                    initial_state)
    print('Result:')
    print(np.round(np.array(amplitudes), 3))
    print('Reference:')
    print(np.round(amplitudes_reference, 3))
    print('Max difference:')
    print(np.max(np.abs(amplitudes - np.array(amplitudes_reference))))


def prepare_parallel_evaluation_tf(filename, n_var_parallel):
    """
    Prepares for parallel evaluation of the quantum circuit.
    Some of the variables in the circuit are parallelized over.
    Symbolic bucket elimination is performed with tensorflow and
    the resulting computation graph (as GraphDef) and other
    supporting information is returned
    """
    # Prepare graphical model
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, data_dict, bra_vars, ket_vars = opt.circ2buckets(
        n_qubits, circuit)

    graph = gm.buckets2graph(
        buckets,
        ignore_variables=bra_vars+ket_vars)

    # find a reduced graph
    vars_parallel, graph_reduced = gm.split_graph_by_metric(
        graph, n_var_parallel,
        metric_fn=gm.splitters.get_node_by_mem_reduction)

    # run quickbb once again to get peo and treewidth
    peo, treewidth = gm.get_peo(graph_reduced)

    # place bra and ket variables to beginning, so these variables
    # will be contracted first
    peo = ket_vars + bra_vars + vars_parallel + peo
    perm_buckets, perm_dict = opt.reorder_buckets(buckets, peo)

    # extract bra and ket variables from variable list and sort according
    # to qubit order
    ket_vars = sorted([perm_dict[idx] for idx in ket_vars], key=str)
    bra_vars = sorted([perm_dict[idx] for idx in bra_vars], key=str)
    vars_parallel = sorted([perm_dict[idx] for idx in vars_parallel],
                           key=str)

    # Populate slice dict. Only shapes of slices are needed at this stage
    slice_dict = utils.slice_from_bits(
        0, bra_vars + ket_vars + vars_parallel)

    # create placeholders with proper shapes
    tf.reset_default_graph()
    tf_buckets, placeholders_dict = tffr.get_sliced_tf_buckets(
        perm_buckets, slice_dict)
    # save only placeholder's names as they are not picklable
    picklable_placeholders = {key.name: val for key, val in
                              placeholders_dict.items()}

    # Do symbolic computation of the result
    result = opt.bucket_elimination(
        tf_buckets, tffr.process_bucket_tf)
    comput_graph = tf.identity(result.data, name='result')

    environment = dict(
        bra_vars=bra_vars,
        ket_vars=ket_vars,
        vars_parallel=vars_parallel,
        tf_graph_def=tf.get_default_graph().as_graph_def(),
        data_dict=data_dict,
        picklable_placeholders=picklable_placeholders
    )

    return environment


def eval_with_tf_parallel_mpi(filename, initial_state=0):
    """
    Evaluate quantum circuit using MPI to parallelize
    over some of the variables.
    """
    comm = MPI.COMM_WORLD
    comm_size = comm.size
    rank = comm.rank

    # number of variables to split by parallelization
    # this should be adjusted by the algorithm from memory/cpu
    # requirements
    n_var_parallel = 2
    if rank == 0:
        env = prepare_parallel_evaluation_tf(filename, n_var_parallel)
    else:
        env = None

    env = comm.bcast(env, root=0)

    # restore tensorflow graph, extract inputs and outputs
    tf.reset_default_graph()
    tf.import_graph_def(env['tf_graph_def'], name='')
    tgraph = tf.get_default_graph()
    result = tgraph.get_tensor_by_name('result:0')

    # restore placeholder and data dictionaries
    picklable_placeholders = env['picklable_placeholders']
    placeholder_dict = {tgraph.get_tensor_by_name(key): val
                        for key, val in picklable_placeholders.items()}

    data_dict = env['data_dict']

    # restore other parts of the environment
    bra_vars = env['bra_vars']
    ket_vars = env['ket_vars']
    vars_parallel = env['vars_parallel']

    # Construct slice dictionary for initial state
    slice_dict = utils.slice_from_bits(initial_state, ket_vars)

    # Construct part of the feed dictionary
    feed_dict = tffr.assign_tensor_placeholders(
        placeholder_dict, data_dict)
    feed_dict.update(tffr.assign_variable_placeholders(
        placeholder_dict, slice_dict
    ))

    # Loop over all amplitudes
    amplitudes = []
    for target_state in range(2**len(bra_vars)):
        # Construct slice dictionary for the target state and
        # populate feed dictionary with proper values
        slice_dict = utils.slice_from_bits(target_state, bra_vars)
        feed_dict.update(tffr.assign_variable_placeholders(
            placeholder_dict, slice_dict))

        amplitude = 0
        for parallel_slice_dict in utils.slice_values_generator(
                vars_parallel, rank, comm_size):
            # Update feed dict with proper slices for parallelized
            # variables
            feed_dict.update(tffr.assign_variable_placeholders(
                placeholder_dict, parallel_slice_dict))

            amplitude += tffr.run_tf_session(result, feed_dict)

        amplitude = comm.reduce(amplitude, op=MPI.SUM, root=0)
        amplitudes.append(amplitude)

    if rank == 0:
        amplitudes_reference = get_amplitudes_from_cirq(filename)
        print('Result:')
        print(np.round(np.array(amplitudes), 3))
        print('Reference:')
        print(np.round(amplitudes_reference, 3))
        print('Max difference:')
        print(np.max(np.array(amplitudes)
                     - np.array(amplitudes_reference)))


def eval_contraction_cost(filename):
    """
    Loads circuit from file, evaluates contraction cost
    with and without optimization
    """
    # Prepare graphical model
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, data_dict, bra_vars, ket_vars = opt.circ2buckets(
        n_qubits, circuit)

    graph_raw = gm.buckets2graph(
        buckets,
        ignore_variables=bra_vars+ket_vars)

    # estimate cost
    mem_raw, flop_raw = gm.get_contraction_costs(graph_raw)
    mem_raw_tot = sum(mem_raw)

    # optimize node order
    peo, treewidth = gm.get_peo(graph_raw)

    # get cost for reordered graph
    graph, label_dict = gm.relabel_graph_nodes(
        graph_raw, dict(zip(peo, sorted(graph_raw.nodes(), key=int)))
    )
    mem_opt, flop_opt = gm.get_contraction_costs(graph)
    mem_opt_tot = sum(mem_opt)

    # split graph and relabel in optimized way
    n_var_parallel = 3
    _, reduced_graph = gm.split_graph_by_metric(
        graph_raw, n_var_parallel)
    # peo, treewidth = gm.get_peo(reduced_graph)
    peo, treewidth = gm.get_peo(reduced_graph)

    graph_parallel, label_dict = gm.relabel_graph_nodes(
        reduced_graph, dict(zip(
            peo, sorted(reduced_graph.nodes(), key=int)))
    )

    mem_par, flop_par = gm.get_contraction_costs(graph_parallel)
    mem_par_tot = sum(mem_par)

    print('Memory (in doubles):\n raw: {} optimized: {}'.format(
        mem_raw_tot, mem_opt_tot))
    print(' parallel:\n  node: {} total: {} n_tasks: {}'.format(
        mem_par_tot, mem_par_tot*2**(n_var_parallel),
        2**(n_var_parallel)
    ))


def test_circ2graph(filename='inst_2x2_7_0.txt'):
    """
    This function tests direct reading of circuits to graphs.
    It should be noted that graphs can not to be used in place
    of buckets yet, since the information about transpositions
    of tensors (denoted by edges) is not kept during node
    relabelling
    """
    import networkx as nx

    nq, circuit = ops.read_circuit_file(filename)
    graph = gm.circ2graph(nq, circuit)

    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets_original, _, bra_vars, ket_vars = opt.circ2buckets(
        n_qubits, circuit)
    graph_original = gm.buckets2graph(
        buckets_original, ignore_variables=bra_vars+ket_vars)

    from networkx.algorithms import isomorphism
    GM = isomorphism.GraphMatcher(graph, graph_original)

    print('Isomorphic? : {}'.format(GM.is_isomorphic()))
    graph = nx.relabel_nodes(graph, GM.mapping, copy=True)

    if not GM.is_isomorphic():
        gm.draw_graph(graph, 'new_graph.png')
        gm.draw_graph(graph_original, 'orig_graph.png')

    return GM.is_isomorphic()


#@utils.sequential_profile_decorator(filename='buckets_transform_profile')
def test_bucket_operation_speed():
    """
    This tests the speed of forming, permuting and transforming
    buckets.
    """
    import time
    tim1 = time.time()

    filename = 'test_circuits/inst/cz_v2/10x10/inst_10x10_60_1.txt'

    # Prepare graphical model
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, data_dict, bra_vars, ket_vars = opt.circ2buckets(
        n_qubits, circuit)

    graph = gm.buckets2graph(
        buckets,
        ignore_variables=bra_vars+ket_vars)

    # Get peo
    peo = [opt.Var(node, name=data['name'], size=data['size']) for
           node, data in graph.nodes(data=True)]
    peo = list(np.random.permutation(peo))

    # place bra and ket variables to beginning, so these variables
    # will be contracted first
    peo = ket_vars + bra_vars + peo
    perm_buckets, perm_dict = opt.reorder_buckets(buckets, peo)

    # extract bra and ket variables from variable list and sort according
    # to qubit order
    ket_vars = sorted([perm_dict[idx] for idx in ket_vars], key=str)
    bra_vars = sorted([perm_dict[idx] for idx in bra_vars], key=str)

    # Take the subtensor corresponding to the initial state
    initial_state = 0
    slice_dict = utils.slice_from_bits(initial_state, ket_vars)

    # Take appropriate subtensors for target bitstring
    target_state = 0
    slice_dict.update(
        utils.slice_from_bits(target_state, bra_vars)
    )
    # Form final buckets
    sliced_buckets = npfr.get_sliced_np_buckets(
        perm_buckets, data_dict, slice_dict)

    tim2 = time.time()
    print(tim2 - tim1)


def eval_with_multiamp_np(filename, initial_state=0):
    """
    Loads circuit from file and evaluates
    multiple amplitudes at once using np framework
    """
    # Values of the fixed bra qubits.
    # this can be changed to your taste
    target_state = 0

    # Prepare graphical model
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, data_dict, bra_vars, ket_vars = opt.circ2buckets(
        n_qubits, circuit)

    # Collect free qubit variables
    free_final_qubits = [1, 3]
    free_bra_vars = []
    for ii in free_final_qubits:
        try:
            free_bra_vars.append(bra_vars[ii])
        except IndexError:
            pass
    bra_vars = [var for var in bra_vars if var not in free_bra_vars]

    if len(free_bra_vars) > 0:
        print('Evaluate all amplitudes over final qubits:')
        print(free_final_qubits)
        print('Free variables in the resulting expression:')
        print(free_bra_vars)

    graph_initial = gm.buckets2graph(
        buckets,
        ignore_variables=bra_vars+ket_vars)

    graph = gm.make_clique_on(graph_initial, free_bra_vars)

    # Run quickbb
    peo_initial, treewidth = gm.get_peo(graph)

    # transform peo so free_bra_vars are at the end
    peo = gm.get_equivalent_peo(graph, peo_initial, free_bra_vars)

    # place bra and ket variables to beginning, so these variables
    # will be contracted first
    perm_buckets, perm_dict = opt.reorder_buckets(
        buckets, bra_vars + ket_vars + peo)
    perm_graph, _ = gm.relabel_graph_nodes(
        graph, perm_dict)

    # extract bra and ket variables from variable list and sort according
    # to qubit order
    ket_vars = sorted([perm_dict[idx] for idx in ket_vars], key=str)
    bra_vars = sorted([perm_dict[idx] for idx in bra_vars], key=str)

    # make proper slice dictionaries. We choose ket = |0>,
    # bra = |0> on fixed entries
    slice_dict = utils.slice_from_bits(initial_state, ket_vars)
    slice_dict.update(utils.slice_from_bits(target_state, bra_vars))
    slice_dict.update({var: slice(None) for var in free_bra_vars})

    # Finally make numpy buckets and calculate
    sliced_buckets = npfr.get_sliced_np_buckets(
        perm_buckets, data_dict, slice_dict)
    result = opt.bucket_elimination(
        sliced_buckets, npfr.process_bucket_np,
        n_var_nosum=len(free_bra_vars))
    amplitudes = result.data.flatten()

    # Now calculate the reference
    amplitudes_reference = get_amplitudes_from_cirq(filename)

    # Get a slice as we do not need full amplitude
    bra_slices = {var: slice_dict[var] for var in slice_dict
                  if var.name.startswith('o')}

    # sort slice in the big endian order for Cirq
    computed_subtensor = [slice_dict[var]
                          for var in sorted(bra_slices, key=str)]

    slice_of_amplitudes = amplitudes_reference.reshape(
        [2]*n_qubits)[tuple(computed_subtensor)]
    slice_of_amplitudes = slice_of_amplitudes.flatten()

    print('Result:')
    print(np.round(amplitudes, 3))
    print('Reference:')
    print(np.round(slice_of_amplitudes, 3))
    print('Max difference:')
    print(np.max(np.abs(amplitudes - slice_of_amplitudes)))


if __name__ == "__main__":
    eval_with_tf('inst_2x2_7_0.txt')
    eval_with_np('inst_2x2_7_0.txt')
    eval_with_tf_parallel_mpi('inst_2x2_7_0.txt')
    eval_with_np_parallel_mpi('inst_2x2_7_0.txt')
    eval_contraction_cost('inst_2x2_7_0.txt')
    eval_with_multiamp_np('inst_2x2_7_0.txt')
