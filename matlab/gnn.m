% GNN - Matlab interface to a general Graph Network Block, following the ordering Battaglia et al., 2018.
%
% Input arguments (Graph Specification):
%  Nvertices     -  # vertices in the graph
%  II            - "row" vertex in the graph
%  JJ            - "col" vertex in the graph
% NOTE: If you have a square matrix, A, you can get these input parameters
% via the following calls:
%   Nvertices = size(A,1);
%   [II,JJ,~]=find(A);
%
% Input arguments (Data):
%  x_fixed       - Cell array containing data storage for fixed
%                  features {vertices,edges,globals}
%  x_mutable     - Cell array containing data storage for mutable
%                  features {vertices,edges,globals}
% NOTE: These are dense matrices of data for the entire graph,
% e.g. if there are 2 vertex features and 10 vertices in the graph,
% x_vertices is 10x2.
%
% NOTE: x_mutable is assumed to be initialized before calling
%
% Input arguments (Updates):
%  vertex_update - Vertex update function, \phi^v.
%                  Functional form: e_ij = edge_update(e_ij,v_i,v_j,g,model_params)
%  edge_update   - Edge update function, \phi^e.
%                  Functional form: v_i = vertex_update(v_i,zbar_i,g,model_params)
%  global_update - Global update function, \phi^g
%                  Functional form: g = global_update(g,vg,eg,model_params)
% NOTE: Pass in [], if one of these functions isn't used.
%
% Input arguments (Aggregation):
%  Ne2v                 - Number of outputs of the agg_edge_to_vertex function
%  agg_edge_to_vertex   - Edge-to-vertex aggregation function
%                         Functional form: zbar_i = agg_edge_to_vertex(e_i*)
%  agg_edge_to_global   - Edge-to-global aggregation function
%                         Functional form: eg = agg_edge_to_global(edge_data)
%  agg_vertex_to_global - Vertex-to-global aggregation fucntion
%                         Functional form: vg = agg_edge_to_global(vertex_data)
% NOTE: Pass in [], if one of these functions isn't used.
%
% Input arguments (Other):
%  model_params         - Generic structure for passing arbitrary
%                         user data to the functions.
%
%
% GNN functions are executed in the following order:
%
% e_ij = edge_update(e_ij,v_i,v_j,g,model_params)
% zbar_i = agg_edge_to_vertex(e_i*)
% v_i = vertex_update(v_i,zbar_i,g,model_params)
% eg = agg_edge_to_global(edge_data)
% vg = agg_vertex_to_global(vertex_data)
% g  = global_update(g,vg,eg,model_params)


% Graph Neural Networks and Applied Linear Algebra
% 
% Copyright 2023 National Technology and Engineering Solutions of
% Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with
% NTESS, the U.S. Government retains certain rights in this software. 
% 
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met: 
% 
% 1. Redistributions of source code must retain the above copyright
% notice, this list of conditions and the following disclaimer. 
% 
% 2. Redistributions in binary form must reproduce the above copyright
% notice, this list of conditions and the following disclaimer in the
% documentation and/or other materials provided with the distribution. 
% 
% 3. Neither the name of the copyright holder nor the names of its
% contributors may be used to endorse or promote products derived from
% this software without specific prior written permission. 
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
% “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
% A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
% HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
% SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
% LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
% DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
% THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
% 
% 
% 
% For questions, comments or contributions contact 
% Chris Siefert, csiefer@sandia.gov 


function x=gnn(Nvertices,II,JJ,x_fixed,x_mutable,vertex_update,edge_update,global_update,Ne2v,agg_edge_to_vertex,agg_edge_to_global,agg_vertex_to_global,model_parameters)

Nedges=length(II);

v_attr_fixed = x_fixed{1};
e_attr_fixed = x_fixed{2};
g_attr_fixed = x_fixed{3};

v_attr_mutable = x_mutable{1};
e_attr_mutable = x_mutable{2};
g_attr_mutable = x_mutable{3};

global_data=[g_attr_fixed,g_attr_mutable];

% In case we're missing stuff
if(isempty(e_attr_fixed)), e_attr_fixed = zeros(Nvertices,0);end
if(isempty(e_attr_mutable)), e_attr_mutable = zeros(Nedges,0);end
if(isempty(v_attr_fixed)), v_attr_fixed = zeros(Nvertices,0);end
if(isempty(v_attr_mutable)), v_attr_mutable = zeros(Nvertices,0);end


% Edge update
if(isa(edge_update,'function_handle')),
  for K=1:Nedges,
    I=II(K); J = JJ(K);
    e_attr_mutable(K,:) = edge_update([e_attr_fixed(K,:),e_attr_mutable(K,:)],[v_attr_fixed(I,:),v_attr_mutable(I,:)],[v_attr_fixed(J,:),v_attr_mutable(J,:)],global_data,model_parameters);
  end
end


% Agg edge->vertex
zbar = zeros(Nvertices, Ne2v);
if(isa(agg_edge_to_vertex,'function_handle')),
  for K=1:Nvertices,
    IDX=find(II==K);
    zbar(K,:) = agg_edge_to_vertex([e_attr_fixed(IDX,:),e_attr_mutable(IDX,:)]);
  end
end

% Vertex update
if(isa(vertex_update,'function_handle')),
  for K=1:Nvertices,
    v_attr_mutable(K,:) = vertex_update([v_attr_fixed(K,:),v_attr_mutable(K,:)],zbar(K,:),global_data,model_parameters);
  end
end

% Agg edge->global
if(isa(agg_edge_to_global,'function_handle')),
  aggE = agg_edge_to_global([e_attr_fixed,e_attr_mutable]);
else
  aggE = [];
end

% Agg vertex->global
if(isa(agg_vertex_to_global,'function_handle')),
  aggV = agg_vertex_to_global([v_attr_fixed,v_attr_mutable]);
else
  aggV = [];
end

% Global updated
if(isa(global_update,'function_handle')),
  g_attr_mutable = global_update(global_data,aggV,aggE,model_parameters);
end

% Set output
x={};
x{1} = v_attr_mutable;
x{2} = e_attr_mutable;
x{3} = g_attr_mutable;
