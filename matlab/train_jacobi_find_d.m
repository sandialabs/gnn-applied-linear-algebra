% train_jacobi_find_d example

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


function train_jacobi_find_d

%% Training, Test and Validation Data %%
% 1. Training data is used in the optimization problem
% 2. Validation data is used to "check" the optimization as it
% progresses, but does not directly feed into the optimization.
% 3. Test data is only checked after the optimization concludes.

% A small collection of stretched matrices
%train_list={heateqnfem2dfun([5,5],[1,1],[2,2]),heateqnfem2dfun([5,5],[1,2],[2,2]),heateqnfem2dfun([5,5],[1,4],[2,2])};
%validation_list={heateqnfem2dfun([5,5],[1,1.5],[2,2])};
%test_list={heateqnfem2dfun([5,5],[1,2.25],[2,2]),heateqnfem2dfun([5,5],[1,1.75],[2,2])};


% A larger collection of stretched meshes
% Training data
N_train=10; max_stretch = 5;
train_list={};
for I=1:N_train,
  stretch = 1+ (I-1)/(N_train-1) * (max_stretch-1);
  train_list{I} = heateqnfem2dfun([5,5],[1,stretch],[2,2]);
end


% Validation data
N_validation=7; max_stretch = 5;
validation_list={};
for I=1:N_train,
  stretch = 1+ (I-1)/(N_validation-1) * (max_stretch-1);
  validation_list{I} = heateqnfem2dfun([5,5],[1,stretch],[2,2]);
end


% Test data
N_test=9; max_stretch = 2;
test_list={};
for I=1:N_test,
  stretch = 1+ (I-1)/(N_test-1) * (max_stretch-1);
  test_list{I} = heateqnfem2dfun([5,5],[1,stretch],[2,2]);
end



%% Initialization and Training Options %%
% Seed the RNG
rng(8675309,'twister');

% Get the parameters
model_parameters = set_model_parameters();

% Training Options for Adam
num_epochs = 10;
learnRate = 0.01;
validationFrequency = 2;
trailingAvg = [];
trailingAvgSq = [];

% Plot
myfig=1;
figure(myfig);close(myfig);
figure(myfig);
C = colororder;
lineLossTrain = animatedline(Color=C(2,:),LineWidth=3);
lineLossValidation = animatedline( ...
    LineStyle="--", ...
    Marker="o", ...
    LineWidth=3,...
    MarkerFaceColor="black");
ylim([0 inf])
set(gca,'FontSize',15);
xlabel('Epoch')
ylabel('Loss')
grid on


%% Training Loop %%
start = tic;
for epoch=1:num_epochs,
  fprintf('Starting epoch %d\n',epoch);
  [loss,gradients]=dlfeval(@loss_function_all,train_list,model_parameters);

  % Update the network parameters using the Adam optimizer.
  [model_parameters,trailingAvg,trailingAvgSq] = adamupdate(model_parameters,gradients, ...
                                                    trailingAvg,trailingAvgSq,epoch,learnRate);

  % Update the training progress plot.
  D = duration(0,0,toc(start),Format="hh:mm:ss");
  title("Epoch: " + epoch + ", Elapsed: " + string(D))
  loss = extractdata(loss);
  addpoints(lineLossTrain,epoch,loss);
  drawnow

  % Display the validation metrics.
  if epoch == 1 || mod(epoch,validationFrequency) == 0
    lossValidation = loss_function_all_no_gradient(validation_list,model_parameters);
    lossValidation = extractdata(lossValidation);
    addpoints(lineLossValidation,epoch,lossValidation);
    drawnow

    if(epoch == 1),
      legend('Train','Validate','Location','SouthWest');
    end

  end
  set(gca,'YLim',[0,1])
end

%% Post-Training Diagnostics %%
% Show final model
print_model_parameters(model_parameters);


% Final test
fprintf('Machine Learning (power method) ');
loss_function_all_no_gradient(test_list,model_parameters);
fprintf('Machine Learning (exact)        ');
loss_function_all_no_gradient(test_list,model_parameters,1);
fprintf('Fixed 2/3 (exact)               ');
loss_function_normal_jacobi(test_list,2/3);
fprintf('Optimal Jacobi (exact)          ');
loss_function_optimal_jacobi(test_list);


% If you want to see how the trained diagonal compares to the
% actual diagonal, you can print that here
if(1==0),
fprintf('* Comparative diagonals *\n')
for I=1:length(test_list),
  A=test_list{I};
  N=size(A,1);
  D=spdiags(diag(A),0,N,N);

  [~,ai_diagonal]=evaluate_single_matrix(A,model_parameters);

  fprintf('[%d] Normal Diag: ',I);fprintf('%6.4e ',full(diag(A)));fprintf('\n');
  fprintf('[%d] AI/ML Diag : ',I);fprintf('%6.4e ',ai_diagonal);fprintf('\n');

  fprintf('[%d] PM AI/Trad/Optim DF = %6.4e / %6.4e / %6.4e\n',I,...
          damping_factor(A,2/3,ai_diagonal,0),...
          damping_factor(A,2/3,full(diag(A)),0),...
          damping_factor(A,optimal_omega(A),full(diag(A)),0));

  fprintf('[%d] EX AI/Trad/Optim DF = %6.4e / %6.4e / %6.4e\n',I,...
          damping_factor(A,2/3,ai_diagonal,1),...
          damping_factor(A,2/3,full(diag(A)),1),...
          damping_factor(A,optimal_omega(A),full(diag(A)),1));
end
end

print('-dpng','sample_training.png');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function print_model_parameters(model_parameters)
fprintf('*** Model Parameters ***\n');
fprintf('* Vertex Model * \n');
for I=1:length(model_parameters.vertex),
  myweights=model_parameters.vertex{I}{1};
  mybias=model_parameters.vertex{I}{2};
  fprintf('Level %d weights | bias:\n',I);
  for I=1:size(myweights,1),
    fprintf('- ');
    for J=1:size(myweights,2),
      fprintf('%6.4e ',myweights(I,J));
    end
    fprintf(' | %6.4e\n',mybias(I));
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model_parameters = set_model_parameters()
model_parameters = struct;

% A bit of a hack here
v_attr_fixed=[1];
v_attr_mutable=[1];

v_agg_edge_features = 4;

output_size_per_level=[3,3,1];
v_weights = {};

in_size=size(v_attr_fixed,2) + size(v_attr_mutable,2) + v_agg_edge_features;
for I=1:length(output_size_per_level),
  out_size = output_size_per_level(I);
  v_weights{I} = {dlarray(rand(out_size,in_size)), dlarray(rand(out_size,1))};
  in_size = out_size;
end
model_parameters.vertex = v_weights;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loss,gradients] = loss_function_all(Alist,model_parameters)
loss = dlarray(0);
gradients = dlarray(0);
%fprintf('Computing loss function: ');
% The loss function here is just an average loss across all of the
% matrices in Alist.
for I=1:length(Alist),
  df = evaluate_single_matrix(Alist{I},model_parameters);

  loss = loss+df;
end
loss = loss/length(Alist);

gradients = dlgradient(loss,model_parameters);
%fprintf('\n');
%fprintf('- Total loss = %6.4e\n',loss);
%gradients = dlgradient(loss, model_parameters);


function loss = loss_function_all_no_gradient(Alist,model_parameters,exact)
if(~exist('exact')), exact=0;end
loss = 0;
fprintf('Loss function: ');
for I=1:length(Alist),
  df = evaluate_single_matrix(Alist{I},model_parameters,exact);
  fprintf('%6.4e ',df);
  loss = loss + df;
end
loss = loss / length(Alist);
fprintf('\n');
fprintf('- Total loss = %6.4e\n',loss);



function loss = loss_function_normal_jacobi(Alist,omega)
loss = 0;
fprintf('Loss function: ');
for I=1:length(Alist),
  D=full(diag(Alist{I}));
  df=damping_factor(Alist{I},omega,D,1);
  fprintf('%6.4e ',df);
  loss = loss + df;
end
loss = loss / length(Alist);
fprintf('\n');
fprintf('- Total loss = %6.4e\n',loss);


function loss = loss_function_optimal_jacobi(Alist)
loss = 0;
fprintf('Loss function: ');
for I=1:length(Alist),
  D=full(diag(Alist{I}));
  omega = optimal_omega(Alist{I});
  df=damping_factor(Alist{I},omega,D,1);
  fprintf('%6.4e ',df);
  loss = loss + df;
end
loss = loss / length(Alist);
fprintf('\n');
fprintf('- Total loss = %6.4e\n',loss);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [df,diagonal]=evaluate_single_matrix(A,model_parameters,exact)
if (~exist('exact')), exact=0;end
omega = 2/3; % For giggles
[II,JJ,VV]=find(A);
N=size(A,1);
v_agg_edge_features = 4; % HAQ

%% Edge features %%
% 1 immutable: A_ij
e_attr_fixed = VV;
% 0 mutable
e_attr_mutable = [];


%% Vertex features
% 1 immutable: A_i
v_attr_fixed = dlarray(full(diag(A)));
% 1 mutable: 0 on input, diagonal value on output
v_attr_mutable = dlarray(zeros(N,1));

%% Global features
% 1 immutable: omega
g_attr_fixed = omega;
% 0 mutable
g_attr_mutable = [];


%% Define the GNN model
x_fixed={v_attr_fixed,e_attr_fixed,g_attr_fixed};
x_mutable={v_attr_mutable,e_attr_mutable,g_attr_mutable};

output = gnn(N,II,JJ,x_fixed,x_mutable,...
             @update_vertex,[],[],...
             v_agg_edge_features,@agg_edge_to_vertex,[],[],model_parameters);



%% Evaluate the loss function
df=damping_factor(A,omega,dlarray(output{1}),exact);

diagonal=output{1};


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% e_ij = update_edge(e_ij,v_i,v_j,g,model_params)
% ehat_i = agg_edge_to_vertex(e_i*)
% n_i = update_vertex(n_i,ehat_i,g,model_params)
% eg = agg_edge_to_global(edges)
% vg = agg_vertex_to_global(edges)
% g  = update_global(g,vg,eg,model_params)
function ehat_i = agg_edge_to_vertex(e_istar)
ehat_i = [min(e_istar(:,1)),mean(e_istar(:,1)),sum(e_istar(:,1)),max(e_istar(:,1))];


% 1 immutable: A_ii
% 1 mutable: D_i on output
% NN params
function out_n_i= update_vertex(n_i,ehat_i,g,model_parameters)
% Just use a sequential NN to update the mutable variable
out_n_i = sequential_nn([n_i,ehat_i],model_parameters.vertex);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function omega=optimal_omega(A)
N=size(A,1);
D=spdiags(diag(A),0,N,N);

opts = struct;
opts.tol=1e-6;

DinvA=D\A;


lmax = eigs(DinvA,1,'largestabs');
lmin = eigs(DinvA,1,'smallestabs');

omega = 2/(lmax+lmin);
