% Experiment Setup:
% - Hardware: Intel Core i9-10940X CPU @ 3.30GHz, 96GB RAM
% - OS: Windows 11 Pro (64-bit)
% - Software: MATLAB R2019b
% - Note: Results may vary slightly across different hardware/software
% environments due to numerical precision or parallelization differences.
% - Dataset: ORL face database
% - Missing Rate: 90% 
% - Clustering Method: Unified Grassmann Manifold-Based Completion and Alignment for Incomplete Multi-View Clustering
% Performance Metrics:
%ACC = 0.9425;  Clustering Accuracy (ACC)
%NMI = 0.9640;  Normalized Mutual Information (NMI)
%ARI = 0.9097;  Adjusted Rand Index (ARI)

clear memory

Dataname = 'ORL_mtv';
MODE = 'Per';
for percentDel = [9]
    Datafold = [Dataname,'_',MODE,'_',num2str(percentDel),'.mat'];
    load(Datafold);

                
ind_folds = view_ind;

truthF = truelabel{1};

numClust = length(unique(truthF));
X = data;  
num_view = length(X);
for iv = 1:num_view
    X1 = X{iv}';
    X1 = NormalizeFea(X1,1);
    ind_0 = find(ind_folds(:,iv) == 0);
    X1(ind_0,:) = [];       
    Y{iv} = X1';            
    W1 = eye(size(ind_folds,1));
    W1(ind_0,:) = [];
    G{iv} = W1;                                               
end

X = Y;
     
for iv = 1:num_view
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 5;
    options.WeightMode = 'Binary';
    Z1 = constructW(X{iv}',options);
    Z_ini{iv} = full(Z1);
    clear Z1;
end

for iv = 1:num_view
    invXX{iv} = inv(X{iv}'*X{iv}+eye(size(X{iv},2)));
end
F_ini = solveF(Z_ini,G,numClust);
max_iter = 60;
miu = 1e-2;
rho = 1.1;

lambda1list = [1000];
lambda2list = [0.01];
lambda3list = [0.0001];
for i = 1:length(lambda1list)
    for j = 1:length(lambda2list)
        for k = 1:length(lambda3list)
                    lambda1 = lambda1list(i);
                    lambda2 = lambda2list(j);
                    lambda3 = lambda3list(k);

[U,H,Z,W,E] = UGMCA(X,G,Z_ini,F_ini,invXX,lambda1,lambda2,lambda3,miu,rho,max_iter);

for iii = 1:num_view
new_F = U{iii};
norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
for ii = 1:size(norm_mat,1)
    if (norm_mat(ii,1)==0)
        norm_mat(ii,:) = 1;
    end
end
new_F = new_F./norm_mat; 
U{iii} = new_F;
end
runtimes = 1;
AC = zeros(runtimes);
NMI = zeros(runtimes);
ARI = zeros(runtimes);
    for iter_c = 1:runtimes
    [pre_labels, ~, ~] = AWP(U);
    result_cluster = ClusteringMeasure(truthF, pre_labels);
    AC(iter_c)    = result_cluster(1)*100
    NMI(iter_c) = result_cluster(2)*100
    ARI(iter_c)    = result_cluster(4)*100
    end

mean_ACC = mean(AC);
std_ACC = std(AC);
mean_NMI = mean(NMI);
std_NMI = std(NMI);
mean_ARI = mean(ARI);
std_ARI = std(ARI);

text_name = [Dataname,'_',MODE,'_10940X_',num2str(percentDel),'.txt'];
dlmwrite(text_name, [lambda1,lambda2,lambda3,mean_ACC, std_ACC, mean_NMI, std_NMI, mean_ARI, std_ARI],'-append','delimiter','\t','newline','pc');

end
        end
    end
end
