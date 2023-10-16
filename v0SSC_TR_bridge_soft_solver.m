function [A,Z,F] = v0SSC_TR_bridge_soft_solver(X, W_SC, L_cannotlink,W_mustlink, para,display) 

alpha1 = para.alpha1;
alpha2 = para.alpha2;
lambda = para.lambda;
lambda_Z = para.lambda_Z;
lambda_M = para.lambda_M;
k = para.k;
maxiter = para.maxiter;
tol = para.tol;

n = size(X,2);
XtX = X'*X;
I = eye(n);
invXtXI = I/(XtX+lambda*I);

alpha1overlambda = alpha1/lambda;
Z = zeros(n,n);
F = zeros(n,k);
A = zeros(n,n);
iter = 0;
while iter < maxiter
    iter = iter + 1;   
    
    %update F (transpose of H)
    CKSym = 0.5 * BuildAdjacency(Z); 
    W =  alpha1* CKSym +  alpha2 * (W_SC + lambda_M *  W_mustlink);
    W = 0.5 * (W + W');
    
    D_inv = diag(1./sqrt(sum(W,2)+eps));
    L = I - D_inv * W * D_inv;      

    [F,~] = trace_ratio_optim(L_cannotlink,L,k,20); 
    
    Fk = F;
    F = normr(real(F));   
    
    %udpate A
    Ak = A;
    A = invXtXI * (XtX  + lambda *Z);
    
    % update Z
    Zk = Z;
    temp = Fk' * L_cannotlink * Fk; 
    tr = 2 * trace(temp);
    threshold = alpha1overlambda * 0.5 * squareform(pdist(F,'squaredeuclidean'))./tr + lambda_Z/lambda * ones(n,n);
    
    Z = sign(A) .* max(0, abs(A) - threshold);
    Z = Z-diag(diag(Z));
    
    

    diffZ = max(max(abs(Z-Zk)));
    diffA = max(max(abs(A-Ak))); 
  
    stopC = max([diffZ,diffA]);
    if display && (iter==1 || mod(iter,10)==0 || stopC<tol)
        disp(['iter ' num2str(iter) ',lambda=' num2str(lambda,'%2.1e') ...
                ',nnzZ=' num2str(nnz(Z))   ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC < tol 
        fprintf('convergence after iteration %d\n',iter);
        break;   
    end
end


