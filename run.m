rng('default')
totalrun= 20;

Knearest = 5;
Knearest_sigma = 3;


load ./dataset/Umist.mat
%load ./dataset/Yale_32x32.mat
%X = fea;
%Y = gnd;
k = length(unique(Y));
n = size(X,1);
mode = 2;
num_mustlinks = [40:20:120];
num_cannotlinks = 3 * num_mustlinks;
%fs = [3,4] * k;
para.k = k;
display = true;
para.tol = 1e-2; 
para.maxiter = 50;  
para.lambda = 100; 
para.alpha2overalpha1 = 0.02; 
para.alpha1s = [0.5]; %[0.05,0.1,0.25,0.5,1.0,2.5,5.0,10]; 
para.lambda_Zs =[2.5]; % [0.1,0.25,0.5,1.0,2.5,5.0];  
para.lambda_M = 100; 
rho = 1.0; 

nCluster = k;
tic
X = X';
gnd = Y;


% simple scalar for ORL Umist YaleB18 and Yale
X = X./255.0;

W_SC = weight_calv2(X',Knearest,Knearest_sigma); 
W_SC = (W_SC + W_SC') * 0.5;


for outer_iter = 1: length(num_mustlinks)
    %f = fs(outer_iter);
    num_mustlink = num_mustlinks(outer_iter);
    num_cannotlink = num_cannotlinks(outer_iter);
    for middle_iter  = 1 : length(para.lambda_Zs)
        para.lambda_Z = para.lambda_Zs(middle_iter);
        for iter = 1: length(para.alpha1s)
            para.alpha1 = para.alpha1s(iter);
            para.alpha2 = para.alpha2overalpha1 * para.alpha1;
            for runid = 1: totalrun
                rng(1000 * runid)
                switch(mode)
                    case 1
                        disp('construct pairwise constraints by partial labels')
                        idx_fidelity = zeros(f,1);
                        h = f/k;
                        for t = 1:k
                            index = find(gnd==t);
                            random_sampler = randperm(length(index));
                            idx_fidelity(((t-1)*h+1):(t*h),:) = index(random_sampler(1:h));
                        end
                        fidelity = [idx_fidelity, gnd(idx_fidelity)];
                        [W_cannotlink,W_mustlink] = construct_link(W_SC,fidelity,k,1);
                    case 2
                        disp('use the random pairwise constrains')
                        [W_cannotlink,W_mustlink] = construct_pairwise_link(Y, k, num_mustlink, num_cannotlink,1);
                    otherwise
                        error('wrong mode number')
                end
                
                temp = W_SC + para.lambda_M * W_mustlink;
                temp = 0.5 * (temp + temp');
                D_inv = diag(1./sqrt(sum(temp,2)+eps));
                L = eye(n) - D_inv * temp * D_inv;
                
                W_cannotlink = 0.5 * (W_cannotlink + W_cannotlink');
                L_cannotlink = diag(sum(W_cannotlink,2)) - W_cannotlink;
                L_cannotlink = 1/nnz(W_cannotlink) * L_cannotlink; 
                
                [Vs,~] = trace_ratio_optim(L_cannotlink,L,k,20);
                
                Vs = normr(Vs);
                grps = kmeans(Vs,nCluster,'maxiter',1000,'replicates',20,'EmptyAction','singleton');
                ACCTR(runid) = munkres_acc_cal(grps,Y,nCluster);
                NMITR(runid) = nmi(grps, Y, k,k);
                
                [A,Z,F] = v0SSC_TR_bridge_soft_solver(X,W_SC, L_cannotlink,W_mustlink, para, display); 
                
                 % use Z
                CKSym = BuildAdjacency(Z);
                grps = SpectralClustering(CKSym,nCluster);
                ACCZ(runid) = munkres_acc_cal(grps,Y,nCluster);
                NMIZ(runid) = nmi(grps, Y, k,k);
                % use A
                CKSym = BuildAdjacency(A);
                grps = SpectralClustering(CKSym,nCluster);
                ACCA(runid) = munkres_acc_cal(grps,Y,nCluster);
                NMIA(runid) = nmi(grps, Y, k,k);
                
                % use F
                F = normr(F);
                [grps,~] = kmeans(F,nCluster,'maxiter',1000,'replicates',20,'EmptyAction','singleton');
                ACCF(runid) = munkres_acc_cal(grps,Y,nCluster);
                NMIF(runid) = nmi(grps, Y, k,k);
                fprintf('\t runid: %.1f  used Z: %.3f, use A: %.3f   use F: %.3f  use TR: %.3f   \n',runid, ACCZ(runid),ACCA(runid), ACCF(runid), ACCTR(runid));
            end
            fprintf('\t avgF: %.4f, medianF: %.4f  stdF: %.4f \n',mean(ACCF),median(ACCF), std(ACCF));
            toc
            
            if mode == 1
                output1 = [ 'mode=' num2str(mode,'%.1f'),',f=' num2str(f,'%.1f')];
            elseif mode == 2 
                output1 = [ 'mode=' num2str(mode,'%.1f'),',num_mustlink=' num2str(num_mustlink,'%.1f'), ',num_cannotlink=' num2str(num_cannotlink,'%.1f')];
            end
            output2 = ['totalrun=' num2str(totalrun),  ',nCluster=' num2str(nCluster),',lambda=' num2str(para.lambda),',lambda_Z=' num2str(para.lambda_Z,'%.4f'), ...
                      ',alpha1=' num2str(para.alpha1,'%.4f'),  ',alpha2=' num2str(para.alpha2,'%.4f'), ',alpha2overalpha1=' num2str(para.alpha2overalpha1,'%.4f'),...
                      ',lambda_M=' num2str(para.lambda_M,'%.4f'), ',Knearest_neighbor=(' num2str(Knearest) ',' num2str(Knearest_sigma) ')' ...           
                      ',tol=' num2str(para.tol,'%.4f'), ',maxiter=' num2str(para.maxiter,'%.1f')
                      ];

            output3 = ['Acc:',...
                     ',avgrateF=' num2str(mean(ACCF),'%.4f'),',medianF=' num2str(median(ACCF),'%.4f'),',stdF=' num2str(std(ACCF),'%.4f'),'\n'
                     ];
            output4 = ['NMI:',...
                     ',avgrateF=' num2str(mean(NMIF),'%.4f'),',medianF=' num2str(median(NMIF),'%.4f'),',stdF=' num2str(std(NMIF),'%.4f'),'\n'
                      ];        
            fid = fopen('./results/umist_results.txt','a');
            fprintf(fid, '%s\n', output1);
            fprintf(fid, '%s\n', output2);
            fprintf(fid, '%s\n', output3);
            fprintf(fid, '%s\n', output4);
            fclose(fid);
        end
    end
end
