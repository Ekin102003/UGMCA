function [F,H,Z,W,E] = UGMCA(X,G,Z_ini,F_ini,invXX,lambda1,lambda2,lambda3,miu,rho,max_iter)

num_view = length(X);
Z = Z_ini;
W = Z;
F = F_ini;
H = F;
opts.record = 0;
opts.mxitr  = 100;
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;

for i = 1:num_view
    E{i}  = zeros(size(X{i}));
    C1{i} = zeros(size(X{i}));
    C2{i} = zeros(size(F{i}));
    C3{i} = zeros(size(Z{i}));
end
for iter = 1:max_iter
    
    Z_pre = Z;
    E_pre = E;
    W_pre = W;
    F_pre = F;
    H_pre = H;

    for i = 1:num_view
        % --------------1 Z{i} ------------ % 
        G1 = X{i}-E{i}+C1{i}/miu;
        %G2 = S{i} - C2{i}/miu;
        G3 = W{i} - C3{i}/miu;
        Z{i} = invXX{i}*(X{i}'*G1+G3);
        clear G1 G3
        % ------------2 W{i} -------------- %
        P = G{i}*F{i};
        Q = L2_distance_1(P',P');
        M = Z{i}+C3{i}/miu;
        linshi_W = M-0.5*lambda1*Q/miu;
        linshi_W = linshi_W-diag(diag(linshi_W));
        for ic = 1:size(Z{i},2)
            ind = 1:size(Z{i},2);
            ind(ic) = [];
            W{i}(ic,ind) = EProjSimplex_new(linshi_W(ic,ind));
        end
        clear linshi_W P Q M ind ic
        % -------------- 4 F{i} --------- %
        L = zeros(size(F{i}*F{i}'));
        for k = 1:num_view
            if k == i
                continue;
            end
            L = L + F{k}*F{k}';  
        end
        Lt{i} = L;
        WW = (abs(W{i})+abs(W{i}'))*0.5;
        LL{i} = diag(sum(WW))-WW;
        L = lambda1*G{i}'*LL{i}*G{i} - lambda3*L;
        L(isnan(L))=0;
        L(isinf(L))=1e5;
        [F{i},~]= solveFv(F{i},@fun1,opts,0.5*miu,H{i},C2{i}/miu,L);
        clear L WW 
        % -------------5 E{i} -------- %
        temp1 = X{i}-X{i}*Z{i}+C1{i}/miu;
        temp2 = lambda2/miu;
        E{i} = max(0,temp1-temp2)+min(0,temp1+temp2);
        clear temp1 temp2
    end
    % ----------------H ---------------- %
     F_tensor  = cat(3, F{:,:});
     C2_tensor = cat(3, C2{:,:});
     [H_tensor,~,~] = prox_tnn(F_tensor + C2_tensor/miu,1/miu);
    
    for i = 1:num_view
        % ----------- C1 C2 C3 --------- %
        H{i} = H_tensor(:,:,i);
        leq1 = X{i}-X{i}*Z{i}-E{i};
        leq2 = F{i}-H{i};
        leq3 = Z{i}-W{i};
        C1{i} = C1{i} + miu*leq1;
        C2{i} = C2{i} + miu*leq2;
        C3{i} = C3{i} + miu*leq3;
    end

    miu = min(rho*miu,1e8);

end

end

function [F,G]=fun1(P,alpha,H,C2,L)
    G=2*L*P+2*alpha*(P-H+C2);
    F=trace(P'*L*P)+alpha*(norm(P-H+C2,'fro'))^2;
end