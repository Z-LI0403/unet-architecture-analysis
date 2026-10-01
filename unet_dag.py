"""
DAG-based UNet-family model (used for structural/theoretical analysis only).

Models UNet, UNet++, and UNet3+ as Directed Acyclic Graphs following the
methodology of ``no_free_lunch_architectures-main/models``.

Each architecture is specified by:
    - ``dag``: list-of-lists of edge types (0=zero, 1=skip, 2=conv)
    - ``node_scales``: spatial scale level of each node
      (0 = full resolution, 1 = 1/2, 2 = 1/4, …)

The model is built automatically from these two specifications.

Edge operations:
    0 – Zero: no connection.
    1 – Skip: spatial resize only (no learned parameters).
              Traditional UNet skip connection — just concatenation.
    2 – Conv: spatial resize + DoubleConv/SingleConv (full parameterised block).

At each node the outputs of all active incoming edges are **concatenated**
and processed by a DoubleConv block, matching the feature-aggregation
strategy used in real UNet architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_components import DoubleConv, initialize_weights

from typing import cast


# ======================================================================
# Edge operations
# ======================================================================

class Zero2d(nn.Module):
    """Edge type 0: produces nothing (placeholder)."""

    def forward(self, x, target_size=None):
        return None


class SkipEdge(nn.Module):
    """Edge type 1: spatial resize only (traditional UNet skip connection).

    No learned parameters — just resizes spatially so the features can be
    concatenated at the target node.  Channel width is preserved as-is.
    """

    def __init__(self, in_ch, out_ch, source_scale, target_scale, bn=False):
        super().__init__()
        self._scale_diff = source_scale - target_scale
        if self._scale_diff > 0:
            self.resize = nn.Upsample(scale_factor=2 ** self._scale_diff,
                                      mode='bilinear', align_corners=False)
        elif self._scale_diff < 0:
            self.resize = nn.MaxPool2d(kernel_size=2 ** (-self._scale_diff),
                                       ceil_mode=True)
        else:
            self.resize = nn.Identity()

    def forward(self, x, target_size=None):
        x = self.resize(x)
        if target_size is not None and (x.shape[2], x.shape[3]) != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear',
                              align_corners=False)
        return x


class ConvEdge(nn.Module):
    """Edge type 2: spatial resize + DoubleConv → ``out_ch`` channels."""

    def __init__(self, in_ch, out_ch, source_scale, target_scale, bn=False):
        super().__init__()
        self._scale_diff = source_scale - target_scale
        if self._scale_diff > 0:
            self.resize = nn.Upsample(scale_factor=2 ** self._scale_diff,
                                      mode='bilinear', align_corners=False)
        elif self._scale_diff < 0:
            self.resize = nn.MaxPool2d(kernel_size=2 ** (-self._scale_diff),
                                       ceil_mode=True)
        else:
            self.resize = nn.Identity()
        self.conv = DoubleConv(in_ch, out_ch, bn=bn)

    def forward(self, x, target_size=None):
        x = self.resize(x)
        if target_size is not None and (x.shape[2], x.shape[3]) != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear',
                              align_corners=False)
        return self.conv(x)


# ======================================================================
# Generic DAG-driven UNet
# ======================================================================

class UNetDAG(nn.Module):
    """A UNet-family model fully defined by its DAG topology.

    Parameters
    ----------
    dag : list[list[int]]
        Edge types for every node (see module docstring).
    node_scales : list[int]
        Spatial scale level for each node.
    n_channels : int
        Number of input image channels.
    n_classes : int
        Number of output segmentation classes.
    base_channels : int
        Channel width at scale 0.  Width at scale *s* is
        ``base_channels * 2**s``.
    cat_channels : int
        Channel width produced by each incoming edge before
        concatenation.  Defaults to *base_channels*.
    """

    def __init__(self, dag, node_scales, n_channels=3, n_classes=21,
                 base_channels=32, cat_channels=None, bn=False):
        super().__init__()
        if isinstance(dag, str):
            from dag_utils import string_to_dag
            dag = string_to_dag(dag)

        self.n_channels = n_channels
        self.n_classes = n_classes
        self._dag = dag
        self._node_scales = list(node_scales)
        self._base_channels = base_channels
        self._cat_channels = cat_channels if cat_channels is not None else base_channels
        self._bn = bn

        num_nodes = len(node_scales)
        assert len(dag) == num_nodes - 1, (
            f"dag has {len(dag)} entries but there are {num_nodes} nodes "
            f"(need {num_nodes - 1} entries; node 0 is the stem)"
        )

        # Channel width per node
        self._node_ch = [base_channels * (2 ** s) for s in node_scales]

        # Stem: raw image → node-0 features
        self.stem = DoubleConv(n_channels, self._node_ch[0], bn=bn)

        # Build edge operators and per-node fusion convolutions
        self.edge_ops = nn.ModuleList()
        self.node_convs = nn.ModuleList()

        for to_offset, edges in enumerate(dag):
            to_idx = to_offset + 1
            to_scale = node_scales[to_idx]
            to_ch = self._node_ch[to_idx]

            node_edge_ops = nn.ModuleList()
            num_active = 0
            fusion_in_ch = 0

            for from_idx, etype in enumerate(edges):
                from_ch = self._node_ch[from_idx]
                from_scale = node_scales[from_idx]

                if etype == 0:
                    node_edge_ops.append(Zero2d())
                elif etype == 1:
                    node_edge_ops.append(
                        SkipEdge(from_ch, self._cat_channels,
                                 from_scale, to_scale, bn=bn))
                    num_active += 1
                    fusion_in_ch += from_ch  # skip preserves source channels
                elif etype == 2:
                    node_edge_ops.append(
                        ConvEdge(from_ch, self._cat_channels,
                                 from_scale, to_scale, bn=bn))
                    num_active += 1
                    fusion_in_ch += self._cat_channels  # conv projects to cat_channels
                else:
                    raise ValueError(f"Unknown edge type {etype}")

            self.edge_ops.append(node_edge_ops)

            # Fusion: concat(active edges) → node output channels
            if num_active > 0:
                self.node_convs.append(
                    DoubleConv(fusion_in_ch, to_ch, bn=bn))
            else:
                # Dead node – should not happen in valid architectures
                self.node_convs.append(nn.Identity())

        # Readout
        self.readout = nn.Conv2d(self._node_ch[-1], n_classes, kernel_size=1)

        initialize_weights(self)

    # ------------------------------------------------------------------
    def forward(self, x, return_all=False):
        nodes = [self.stem(x)]

        for to_offset, edges in enumerate(self._dag):
            to_idx = to_offset + 1
            to_scale = self._node_scales[to_idx]

            # Expected spatial size at this scale
            target_h = max(1, x.shape[2] // (2 ** to_scale))
            target_w = max(1, x.shape[3] // (2 ** to_scale))
            target_size = (target_h, target_w)

            active = []
            edge_ops_for_node = cast(nn.ModuleList, self.edge_ops[to_offset])
            for from_idx, etype in enumerate(edges):
                if etype != 0:
                    feat = edge_ops_for_node[from_idx](
                        nodes[from_idx], target_size)
                    active.append(feat)

            if active:
                out = self.node_convs[to_offset](torch.cat(active, dim=1))
            else:
                out = torch.zeros(
                    x.shape[0], self._node_ch[to_idx],
                    target_size[0], target_size[1],
                    device=x.device, dtype=x.dtype)
            nodes.append(out)

        logits = self.readout(nodes[-1])
        if return_all:
            return [nodes[-1]], logits
        return logits

    # ------------------------------------------------------------------
    # re-initialise weights (for complexity measurements)
    def _init(self):
        initialize_weights(self)


# ======================================================================
# Pre-defined architecture DAG specifications
# ======================================================================

def get_unet_dag_spec(depth=5):
    """Return (dag, node_scales) for standard UNet with *depth* encoder levels.

    Node layout (2*depth - 1 nodes total, topological order):
        enc0 (s0) … enc_{depth-1}/bottleneck (s_{depth-1})
        dec_{depth-2} (s_{depth-2}) … dec0 (s0)

    Edges mirror the classic U-shape: each encoder feeds the next via
    a down+conv (type 2), and each decoder receives an up+conv from the
    node below (type 2) plus a parameter-free skip from the corresponding
    encoder (type 1).

    Examples
    --------
    depth=5 → 9 nodes, 8 DAG entries (matches the original hardcoded spec)
    depth=4 → 7 nodes, 6 DAG entries
    depth=3 → 5 nodes, 4 DAG entries
    depth=2 → 3 nodes, 2 DAG entries
    """
    assert depth >= 2, f"depth must be >= 2, got {depth}"
    node_scales = list(range(depth)) + list(range(depth - 2, -1, -1))
    dag = []
    # Encoder edges: enc_i ← enc_{i-1} via conv (type 2)
    for i in range(depth - 1):
        row = [0] * i + [2]
        dag.append(row)
    # Decoder edges: decoder k (0 = closest to bottleneck)
    # receives a skip from enc[depth-2-k] (type 1) and
    # an up+conv from the previous node (type 2).
    for k in range(depth - 1):
        row = [0] * (depth + k)          # filled with zeros
        row[depth - 2 - k] = 1           # skip from same-level encoder
        prev_node = depth - 1 if k == 0 else depth + k - 1
        row[prev_node] = 2               # up+conv from bottleneck / previous decoder
        dag.append(row)
    return dag, node_scales


def get_unet_plus_plus_dag_spec(depth=5):
    r"""Return (dag, node_scales) for UNet++ with *depth* encoder levels.

    Nodes are ordered by anti-diagonal bands (i+j = const) then by j:
        band 0: x_{0,0}
        band 1: x_{1,0}, x_{0,1}
        band 2: x_{2,0}, x_{1,1}, x_{0,2}
        …
        band depth-1: x_{depth-1,0}, …, x_{0,depth-1}

    Node index for x_{i,j}: (i+j)*(i+j+1)//2 + j
    Scale for x_{i,j}: i

    Total nodes: depth*(depth+1)//2

    Each intermediate node x_{i,j} (j≥1) aggregates skip connections
    (type 1) from all same-level predecessors x_{i,0..j-1} plus an
    up+conv (type 2) from x_{i+1,j-1}.

    Examples
    --------
    depth=5 → 15 nodes (matches original hardcoded spec)
    depth=4 → 10 nodes
    depth=3 → 6 nodes
    depth=2 → 3 nodes (degenerates to standard UNet)
    """
    assert depth >= 2, f"depth must be >= 2, got {depth}"

    def node_idx(i, j):
        b = i + j
        return b * (b + 1) // 2 + j

    num_nodes = depth * (depth + 1) // 2
    node_scales = [0] * num_nodes
    for b in range(depth):
        for j in range(b + 1):
            node_scales[node_idx(b - j, j)] = b - j  # scale = i = b - j

    dag = []
    for b in range(1, depth):          # bands 1 … depth-1
        for j in range(b + 1):
            i = b - j
            to_idx = node_idx(i, j)
            row = [0] * to_idx
            if j == 0:
                # Encoder node: conv from (i-1, 0)
                row[node_idx(i - 1, 0)] = 2
            else:
                # Nested node: skips from (i, 0..j-1) + up+conv from (i+1, j-1)
                for k in range(j):
                    row[node_idx(i, k)] = 1
                row[node_idx(i + 1, j - 1)] = 2
            dag.append(row)

    return dag, node_scales


def get_unet_3plus_dag_spec(depth=5):
    """Return (dag, node_scales) for UNet 3+ with *depth* encoder levels.

    Node layout (same 2*depth-1 nodes as UNet):
        enc0 (s0) … bottleneck (s_{depth-1})
        dec_{depth-2} (s_{depth-2}) … dec0 (s0)

    Full-scale skip connections: every decoder node (indexed k, where
    k=0 is closest to the bottleneck) receives exactly *depth* type-2
    conv edges:
      - enc0 … enc_{depth-2-k}  (finer encoders, downsampled)
      - bottleneck enc_{depth-1} (upsampled)
      - dec_{depth-2} … dec_{depth-1-k+1}  (previously computed decoders,
        upsampled)

    This keeps the number of incoming edges constant at *depth* per
    decoder node, matching the parameter budget of this project.

    Examples
    --------
    depth=5 → 9 nodes (matches original hardcoded spec)
    depth=4 → 7 nodes
    depth=3 → 5 nodes
    depth=2 → 3 nodes
    """
    assert depth >= 2, f"depth must be >= 2, got {depth}"
    node_scales = list(range(depth)) + list(range(depth - 2, -1, -1))
    dag = []
    # Encoder edges (same as UNet)
    for i in range(depth - 1):
        row = [0] * i + [2]
        dag.append(row)
    # Decoder edges: decoder k (0 = closest to bottleneck)
    for k in range(depth - 1):
        row = [0] * (depth + k)
        # Fine encoders enc[0..depth-2-k]
        for enc_i in range(depth - 1 - k):
            row[enc_i] = 2
        # Bottleneck enc[depth-1]
        row[depth - 1] = 2
        # Previously computed decoders (nodes depth .. depth+k-1)
        for dec_node in range(depth, depth + k):
            row[dec_node] = 2
        dag.append(row)
    return dag, node_scales


# ======================================================================
# Factory helpers
# ======================================================================

# Registry mapping architecture names to DAG-spec functions
_DAG_SPEC_REGISTRY = {
    "UNet": get_unet_dag_spec,
    "UNetPlusPlus": get_unet_plus_plus_dag_spec,
    "UNet3Plus": get_unet_3plus_dag_spec,
}


def build_unet_dag(arch_name, n_channels=3, n_classes=21,
                    base_channels=32, cat_channels=None, bn=False, depth=5):
    """Instantiate a UNetDAG model for a named architecture.

    Parameters
    ----------
    arch_name : str
        One of ``"UNet"``, ``"UNetPlusPlus"``, ``"UNet3Plus"``.
    depth : int
        Number of encoder resolution levels (default 5).
    """
    if arch_name not in _DAG_SPEC_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch_name}'. "
            f"Available: {list(_DAG_SPEC_REGISTRY.keys())}"
        )
    dag, node_scales = _DAG_SPEC_REGISTRY[arch_name](depth=depth)
    return UNetDAG(dag, node_scales,
                   n_channels=n_channels,
                   n_classes=n_classes,
                   base_channels=base_channels,
                   cat_channels=cat_channels,
                   bn=bn)


def get_all_dag_specs(depth=5):
    """Return a dict of ``{name: (dag, node_scales)}`` for every registered
    architecture at the given *depth*.

    Parameters
    ----------
    depth : int
        Number of encoder resolution levels.  All three specs are generated
        for this depth so that structural metrics correspond to the actual
        trained models.
    """
    return {name: fn(depth=depth) for name, fn in _DAG_SPEC_REGISTRY.items()}
