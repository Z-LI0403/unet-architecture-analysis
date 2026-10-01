import copy
import numpy as np


# ---------------------------------------------------------------------------
# Affinity matrix
# ---------------------------------------------------------------------------

def dag2affinity(dag):
    """Convert a DAG (list-of-lists) to a square affinity matrix.

    Returns an (N x N) numpy array where N = len(dag) + 1 (number of nodes).
    Entry Aff[i, j]:
        -1  no connection from i to j
         0  skip connection (edge type 1)
         1  parameterized conv (edge type 2)
    Diagonal entries are 0.
    """
    num_nodes = len(dag) + 1
    Aff = np.ones((num_nodes, num_nodes)) * -1
    np.fill_diagonal(Aff, 0)
    for to_offset, edges in enumerate(dag):
        to_node = to_offset + 1
        for from_node, edge_type in enumerate(edges):
            Aff[from_node, to_node] = edge_type - 1  # 0→-1, 1→0, 2→1
    return Aff


# ---------------------------------------------------------------------------
# Path enumeration & effective depth / width
# ---------------------------------------------------------------------------

def find_all_paths(Aff, all_paths=None, all_paths_idx=None,
                   curr_path=None, curr_path_idx=None,
                   curr_pos=0, end_pos=None):
    """Enumerate all directed paths from *curr_pos* to *end_pos* in an
    affinity matrix.  Each path is a list of edge weights (0 or 1).
    """
    if all_paths is None:
        all_paths = []
    if all_paths_idx is None:
        all_paths_idx = []
    if curr_path is None:
        curr_path = []
    if curr_path_idx is None:
        curr_path_idx = []
    if end_pos is None:
        end_pos = len(Aff) - 1

    if curr_pos == end_pos:
        all_paths.append(list(curr_path))
        all_paths_idx.append(list(curr_path_idx))
        return all_paths, all_paths_idx

    next_nodes = np.where(Aff[curr_pos, (curr_pos + 1):] >= 0)[0] + curr_pos + 1
    for node in next_nodes:
        curr_path.append(Aff[curr_pos, node])
        curr_path_idx.append([curr_pos, node])
        find_all_paths(Aff, all_paths, all_paths_idx,
                       curr_path, curr_path_idx, node, end_pos)
        curr_path.pop(-1)
        curr_path_idx.pop(-1)
    return all_paths, all_paths_idx


def effective_depth_width(Aff):
    """Compute path-based structural metrics for a DAG affinity matrix.

    Returns
    -------
    depth : float
        Average number of parameterised (type-2, affinity=1) edges per
        directed path from input to output. All end-to-end paths are included,
        including paths made entirely of skip edges.
    parameterized_end_to_end_paths : int
        Number of end-to-end paths that contain at least one parameterised
        edge.
    width : float
        Effective width, defined as
        ``parameterized_end_to_end_paths / depth``.
    all_end_to_end_paths : int
        Total number of end-to-end paths in the DAG, including paths with no
        parameterised edge.
    depth_total : int
        Number of unique parameterised edges across all paths.
    """
    paths, paths_idx = find_all_paths(Aff, end_pos=len(Aff) - 1)
    if not paths:
        return 0, 0, 0, 0, 0
    depth = 0
    parameterized_end_to_end_paths = 0
    param_edges = []
    for path, path_idx in zip(paths, paths_idx):
        path_depth = int(np.sum(path))
        depth += path_depth
        parameterized_end_to_end_paths += int(path_depth > 0)
        for edge_val, idx_pair in zip(path, path_idx):
            if edge_val == 1:  # parameterised edge
                param_edges.append("-".join(str(i) for i in idx_pair))
    if depth == 0:
        return 0, parameterized_end_to_end_paths, 0, len(paths), len(set(param_edges))
    depth = depth / len(paths)
    width = parameterized_end_to_end_paths / depth
    return depth, parameterized_end_to_end_paths, width, len(paths), len(set(param_edges))


# ---------------------------------------------------------------------------
# DAG enumeration (for exhaustive search)
# ---------------------------------------------------------------------------

def find_all_dags(all_dags=None, curr_dag=None, max_num_nodes=4,
                  candidate_ops=None):
    """Recursively enumerate **all** DAGs with up to *max_num_nodes* nodes.

    Each DAG follows the list-of-lists encoding used throughout this project.
    """
    if all_dags is None:
        all_dags = []
    if curr_dag is None:
        curr_dag = []
    if candidate_ops is None:
        candidate_ops = [0, 1, 2]

    # Termination: DAG is complete
    if (len(curr_dag) == max_num_nodes - 1
            and len(curr_dag[-1]) == max_num_nodes - 1):
        all_dags.append(copy.deepcopy(curr_dag))
        return all_dags

    # Start a new node if needed
    if len(curr_dag) == 0 or len(curr_dag[-1]) == len(curr_dag):
        curr_dag.append([])

    for op in candidate_ops:
        curr_dag[-1].append(op)
        find_all_dags(all_dags, curr_dag, max_num_nodes, candidate_ops)
        curr_dag[-1].pop(-1)

    if len(curr_dag[-1]) == 0:
        curr_dag.pop(-1)
    return all_dags


# ---------------------------------------------------------------------------
# DAG string codec
# ---------------------------------------------------------------------------

def dag_to_string(dag):
    """Encode a DAG (list-of-lists) as a compact string.

    Example: [[2], [0, 2], [0, 0, 2]] → "2_02_002"
    """
    return "_".join("".join(str(e) for e in node) for node in dag)


def string_to_dag(s):
    """Decode a DAG string back to list-of-lists.

    Example: "2_02_002" → [[2], [0, 2], [0, 0, 2]]
    """
    return [[int(ch) for ch in node] for node in s.split("_")]


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def dag_summary(dag, node_scales=None):
    """Return a dict of structural metrics for a DAG."""
    Aff = dag2affinity(dag)
    depth, param_end_to_end_paths, width, end_to_end_paths, depth_total = effective_depth_width(Aff)
    num_nodes = len(dag) + 1
    num_edges = sum(1 for row in dag for e in row if e != 0)
    num_param_edges = sum(1 for row in dag for e in row if e == 2)
    num_skip_edges = sum(1 for row in dag for e in row if e == 1)

    info = {
        "dag_string": dag_to_string(dag),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_param_edges": num_param_edges,
        "num_skip_edges": num_skip_edges,
        "effective_depth": depth,
        "parameterized_end_to_end_paths": param_end_to_end_paths,
        "end_to_end_paths": end_to_end_paths,
        "effective_width": width,
        "unique_param_edges": depth_total,
    }
    if node_scales is not None:
        info["node_scales"] = node_scales
    return info
