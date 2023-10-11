# Graph Neural Networks and Applied Linear Algebra
# 
# Copyright 2023 National Technology and Engineering Solutions of
# Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with
# NTESS, the U.S. Government retains certain rights in this software. 
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
# 
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer. 
# 
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution. 
# 
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
# 
# 
# 
# For questions, comments or contributions contact 
# Chris Siefert, csiefer@sandia.gov 
import torch
import torch_scatter
from torch.nn import Linear, Sequential, ReLU, LeakyReLU
from torch_geometric.nn import MetaLayer

class LearnDiffusionGNN(torch.nn.Module):
    def __init__(self, n_layers_external, n_layers_internal, n_hidden=32, encoder=False, decoder=False, init_func = None):
        super().__init__()
        self.layers = []

        n_input_edge = 3
        n_hidden_edge = n_hidden
        n_output_edge = n_hidden

        n_input_vertex = 1
        n_hidden_vertex = n_hidden
        n_output_vertex = 2

        n_input_global = 1
        n_hidden_global = n_hidden
        n_output_global = n_hidden

        if decoder is not False:
            n_output_decoder = n_output_vertex
            n_output_vertex = n_hidden

        if encoder is not False:
            self.layers.append(self.buildEncoder(n_hidden, encoder, n_input_edge, n_input_vertex, n_input_global, init_func))
            n_input_edge = n_hidden
            n_input_vertex = n_hidden
            n_input_global = n_hidden

        if n_layers_external == 1:
            self.layers.append(self.buildOnlyGNN(n_layers_internal, n_input_edge, n_hidden_edge, n_output_edge, n_input_vertex, n_hidden_vertex, n_output_vertex, n_input_global, n_hidden_global, n_output_global, init_func))
        else:
            # Layer 1
            self.layers.append(self.buildFirstLayer(n_layers_internal, n_input_edge, n_hidden_edge, n_input_vertex, n_hidden_vertex, n_input_global, n_hidden_global, init_func))
            
            # Layers 2 through n_external-1
            self.layers.extend(self.buildIntermediateLayers(n_layers_external, n_layers_internal, n_hidden_edge, n_hidden_vertex, n_hidden_global, init_func))

            # Last layer
            self.layers.append(self.buildLastLayer(n_layers_internal, n_hidden_edge, n_output_edge, n_hidden_vertex, n_output_vertex, n_hidden_global, n_output_global, init_func))

        if decoder is not False:
            self.layers.append(MetaLayer(None, EncoderVertex(decoder[0], n_output_vertex, decoder[1], n_output_decoder, init_func), None))

        self.layers = torch.nn.ModuleList(self.layers)

    def buildLastLayer(self, n_layers_internal, n_hidden_edge, n_output_edge, n_hidden_vertex, n_output_vertex, n_hidden_global, n_output_global, init_func = None):
        n_edge, n_vert, n_global = get_n_features(n_hidden_edge, n_hidden_edge, n_output_edge,
                                                    n_hidden_vertex, n_hidden_vertex, n_output_vertex,
                                                    n_hidden_global, n_hidden_global, n_output_global,
                                                    4, 4, 4)
        EdgeUp = EdgeUpdate(n_layers_internal, *n_edge, init_func)
        VertexUp = VertexUpdate(n_layers_internal, edge_to_vertex_aggregation, *n_vert, init_func)
        GlobalUp = GlobalUpdate(n_layers_internal, 
                                    edge_to_global_aggregation, 
                                    vertex_to_global_aggregation, 
                                    *n_global,
                                    init_func)
        return MetaLayer(EdgeUp, VertexUp, GlobalUp)

    def buildIntermediateLayers(self, n_layers_external, n_layers_internal, n_hidden_edge, n_hidden_vertex, n_hidden_global, init_func = None):
        layers = []
        n_edge, n_vert, n_global = get_n_features(n_hidden_edge, n_hidden_edge, n_hidden_edge,
                                                    n_hidden_vertex, n_hidden_vertex, n_hidden_vertex,
                                                    n_hidden_global, n_hidden_global, n_hidden_global,
                                                    4, 4, 4)
        for i in range(n_layers_external-2):
            EdgeUp = EdgeUpdate(n_layers_internal, *n_edge, init_func)
            VertexUp = VertexUpdate(n_layers_internal, edge_to_vertex_aggregation, *n_vert, init_func)
            GlobalUp = GlobalUpdate(n_layers_internal, 
                                        edge_to_global_aggregation, 
                                        vertex_to_global_aggregation, 
                                        *n_global,
                                        init_func)
            layers.append(MetaLayer(EdgeUp, VertexUp, GlobalUp))

        return layers

    def buildFirstLayer(self, n_layers_internal, n_input_edge, n_hidden_edge, n_input_vertex, n_hidden_vertex, n_input_global, n_hidden_global, init_func = None):
        n_edge, n_vert, n_global = get_n_features(n_input_edge, n_hidden_edge, n_hidden_edge,
                                                    n_input_vertex, n_hidden_vertex, n_hidden_vertex,
                                                    n_input_global, n_hidden_global, n_hidden_global,
                                                    4, 4, 4)
        EdgeUp = EdgeUpdate(n_layers_internal, *n_edge, init_func)
        VertexUp = VertexUpdate(n_layers_internal, edge_to_vertex_aggregation, *n_vert, init_func)
        GlobalUp = GlobalUpdate(n_layers_internal, 
                                    edge_to_global_aggregation, 
                                    vertex_to_global_aggregation, 
                                    *n_global,
                                    init_func)
            
        return MetaLayer(EdgeUp, VertexUp, GlobalUp)

    def buildOnlyGNN(self, n_layers_internal, n_input_edge, n_hidden_edge, n_output_edge, n_input_vertex, n_hidden_vertex, n_output_vertex, n_input_global, n_hidden_global, n_output_global, init_func = None):
        n_edge, n_vert, n_global = get_n_features(n_input_edge, n_hidden_edge, n_output_edge,
                                                    n_input_vertex, n_hidden_vertex, n_output_vertex,
                                                    n_input_global, n_hidden_global, n_output_global,
                                                    4, 4, 4)

        EdgeUp = EdgeUpdate(n_layers_internal, *n_edge, init_func)
        VertexUp = VertexUpdate(n_layers_internal, edge_to_vertex_aggregation, *n_vert, init_func)
        #GlobalUp = GlobalUpdate(n_layers_internal, 
        #                            edge_to_global_aggregation, 
        #                            vertex_to_global_aggregation, 
        #                            *n_global,
        #                            init_func)
        return MetaLayer(EdgeUp, VertexUp, None)

    def buildEncoder(self, n_hidden, encoder, n_input_edge, n_input_vertex, n_input_global, init_func = None):
        Encoder_edge = EncoderEdge(encoder[0], n_input_edge, encoder[1], n_hidden, init_func)
        Encoder_vertex = EncoderVertex(encoder[0], n_input_vertex, encoder[1], n_hidden, init_func)
        Encoder_global = EncoderGlobal(encoder[0], n_input_global, encoder[1], n_hidden, init_func)
        return MetaLayer(Encoder_edge, Encoder_vertex, Encoder_global)
        
    def forward(self, v_attr, edgeij_pair, e_attr, g, batch):
        for l in self.layers:
            v_attr, e_attr, g = l(v_attr, edgeij_pair, e_attr, g, batch)

        return LeakyReLU()(v_attr)


class EncoderEdge(torch.nn.Module):
    def __init__(self, n_layers, n_inputs, n_hidden, n_outputs, init_func = None):
        super().__init__()

        if n_layers == 1:
            self.nn = getInitializedLinear(n_inputs, n_outputs, init_func)
        else:
            layers = []
            layers.append(getInitializedLinear(n_inputs, n_hidden, init_func))
            layers.append(ReLU())
            for i in range(n_layers-2):
                layers.append(getInitializedLinear(n_hidden, n_hidden, init_func))
                layers.append(ReLU())
            layers.append(getInitializedLinear(n_hidden, n_outputs, init_func))
            self.nn = Sequential(*layers)
    def forward(self, v_attr_i, vattr_rj, e_attr, g, batch):
        return self.nn(e_attr)

class EdgeUpdate(torch.nn.Module):
    def __init__(self, n_layers, n_inputs, n_hidden, n_outputs, init_func = None):
        super().__init__()

        if n_layers == 1:
            self.nn = getInitializedLinear(n_inputs, n_outputs, init_func)
        else:
            layers = []
            layers.append(getInitializedLinear(n_inputs, n_hidden, init_func))
            layers.append(ReLU())
            for i in range(n_layers-2):
                layers.append(getInitializedLinear(n_hidden, n_hidden, init_func))
                layers.append(ReLU())
            layers.append(getInitializedLinear(n_hidden, n_outputs, init_func))
            self.nn = Sequential(*layers)

    def forward(self, vattr_i, vattr_j, e_attr, g, batch):
        out = torch.cat([vattr_i, vattr_j, e_attr, g[batch]], 1)
        out = self.nn(out)
        return out

class EncoderVertex(torch.nn.Module):
    def __init__(self, n_layers, n_inputs, n_hidden, n_outputs, init_func = None):
        super().__init__()
        if n_layers == 1:
            self.nn = getInitializedLinear(n_inputs, n_outputs, init_func)
        else:
            layers = []
            layers.append(getInitializedLinear(n_inputs, n_hidden, init_func))
            layers.append(ReLU())
            for i in range(n_layers-2):
                layers.append(getInitializedLinear(n_hidden, n_hidden, init_func))
                layers.append(ReLU())
            layers.append(getInitializedLinear(n_hidden, n_outputs, init_func))
            self.nn = Sequential(*layers)

    def forward(self, v_attr, edgeij_pair, e_attr, g, batch):
        return self.nn(v_attr)

class VertexUpdate(torch.nn.Module):
    def __init__(self, n_layers, edge_to_vertex_aggregation, n_inputs, n_hidden, n_outputs, init_func = None):
        super().__init__()
        self.edge_to_vertex_aggregation = edge_to_vertex_aggregation

        if n_layers == 1:
            self.nn = getInitializedLinear(n_inputs, n_outputs, init_func)
        else:
            layers = []
            layers.append(getInitializedLinear(n_inputs, n_hidden, init_func))
            layers.append(ReLU())
            for i in range(n_layers-2):
                layers.append(getInitializedLinear(n_hidden, n_hidden, init_func))
                layers.append(ReLU())
            layers.append(getInitializedLinear(n_hidden, n_outputs, init_func))
            self.nn = Sequential(*layers)

    def forward(self, v_attr, edgeij_pair, e_attr, g, batch):
        out = self.edge_to_vertex_aggregation(edgeij_pair, e_attr, v_attr.shape[0])
        out = torch.cat([v_attr, out, g[batch]], 1)
        out = self.nn(out)
        return out

class EncoderGlobal(torch.nn.Module):
    def __init__(self, n_layers, n_inputs, n_hidden, n_outputs, init_func = None):
        super().__init__()
        if n_layers == 1:
            self.nn = getInitializedLinear(n_inputs, n_outputs, init_func)
        else:
            layers = []
            layers.append(getInitializedLinear(n_inputs, n_hidden, init_func))
            layers.append(ReLU())
            for i in range(n_layers-2):
                layers.append(getInitializedLinear(n_hidden, n_hidden, init_func))
                layers.append(ReLU())
            layers.append(getInitializedLinear(n_hidden, n_outputs, init_func))
            self.nn = Sequential(*layers)

    def forward(self, v_attr, edgeij_pair, e_attr, g, batch):
        return self.nn(g)

class GlobalUpdate(torch.nn.Module):
    def __init__(self,n_layers, edge_to_global_aggregation, vertex_to_global_aggregation, n_inputs, n_hidden, n_outputs, init_func = None):
        super().__init__()
        self.edge_to_global_aggregation = edge_to_global_aggregation
        self.vertex_to_global_aggregation = vertex_to_global_aggregation

        if n_layers == 1:
            self.nn = getInitializedLinear(n_inputs, n_outputs, init_func)

        else:
            layers = []
            layers.append(getInitializedLinear(n_inputs, n_hidden, init_func))
            layers.append(ReLU())

            for i in range(n_layers-2):
                layers.append(getInitializedLinear(n_hidden, n_hidden, init_func))
                layers.append(ReLU())

            layers.append(getInitializedLinear(n_hidden, n_outputs, init_func))

            self.nn = Sequential(*layers)

    def forward(self, v_attr, edgeij_pair, e_attr, g, batch):
        e_to_g_agg = self.edge_to_global_aggregation(e_attr, batch[edgeij_pair[0]])
        v_to_g_agg = self.vertex_to_global_aggregation(v_attr, batch)
        out = torch.cat([g, e_to_g_agg, v_to_g_agg], 1)
        out = self.nn(out)
        return out


def edge_to_vertex_aggregation(edgeij_pair, e_attr, n_vertices):
    """
    Aggregate edge information for use in the vertex update.

    In this case, we return the min, mean, sum, and max of all e_attr with the same
    sending vertex.
    """
    # edge_index: [2, # edges] with max entry (# nodes - 1)
    # e_attr: [# edges, # edge attrib]
    # num_nodes: total number of nodes - needed for allocating memory
    # output should be [# nodes, # aggregated attrib]

    agg_min  = torch_scatter.scatter(e_attr, edgeij_pair[0], dim=0, dim_size=n_vertices, reduce="min")
    agg_mean = torch_scatter.scatter(e_attr, edgeij_pair[0], dim=0, dim_size=n_vertices, reduce="mean")
    agg_sum  = torch_scatter.scatter(e_attr, edgeij_pair[0], dim=0, dim_size=n_vertices, reduce="sum")
    agg_max  = torch_scatter.scatter(e_attr, edgeij_pair[0], dim=0, dim_size=n_vertices, reduce="max")

    return torch.cat((agg_min, agg_mean, agg_sum, agg_max), 1)

def edge_to_global_aggregation(e_attr, batch):
    """
    Aggregate edge information for use in the global update.

    In this case, we return the min, mean, sum, and max of all e_attr
    """
    # e_attr: [# edges, # edge features]
    # batch: [# edges]
    # output should be [# batches, # aggregated attrib]

    agg_min  = torch_scatter.scatter(e_attr, batch, dim=0, reduce="min")
    agg_mean = torch_scatter.scatter(e_attr, batch, dim=0, reduce="mean")
    agg_sum  = torch_scatter.scatter(e_attr, batch, dim=0, reduce="sum")
    agg_max  = torch_scatter.scatter(e_attr, batch, dim=0, reduce="max")

    return torch.cat((agg_min, agg_mean, agg_sum, agg_max), 1)

def vertex_to_global_aggregation(v_attr, batch):
    """
    Aggregate vertex information for use in the global update.

    In this case, we return the min, mean, sum, and max of all v_attr
    """
    # v_attr: [# node, # node features]
    # batch: [# nodes]
    # output should be [# batches, # node features]

    agg_min  = torch_scatter.scatter(v_attr, batch, dim=0, reduce="min")
    agg_mean = torch_scatter.scatter(v_attr, batch, dim=0, reduce="mean")
    agg_sum  = torch_scatter.scatter(v_attr, batch, dim=0, reduce="sum")
    agg_max  = torch_scatter.scatter(v_attr, batch, dim=0, reduce="max")

    return torch.cat((agg_min, agg_mean, agg_sum, agg_max), 1)

def get_n_features(n_input_edge, n_hidden_edge, n_output_edge,
                   n_input_vertex, n_hidden_vertex, n_output_vertex,
                   n_input_global, n_hidden_global, n_output_global,
                   n_agg_e_to_v, n_agg_e_to_g, n_agg_v_to_g):
                   
    edge_features   = [n_input_edge+2*n_input_vertex+n_input_global, n_hidden_edge, n_output_edge]
    vert_features   = [n_input_vertex + n_agg_e_to_v*n_output_edge + n_input_global, n_hidden_vertex, n_output_vertex]
    global_features = [n_input_global+n_agg_e_to_g*n_output_edge+n_agg_v_to_g*n_output_vertex, n_hidden_global, n_output_global]

    return edge_features, vert_features, global_features

def getInitializedLinear(n_inputs, n_outputs, init_func):

    # If init_func is None, use the default
    if init_func is None:
        return Linear(n_inputs, n_outputs)

    layer = Linear(n_inputs, n_outputs)
    init_func(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    return layer
