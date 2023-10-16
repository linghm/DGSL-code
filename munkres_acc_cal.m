function munkres_acc = munkres_acc_cal(idx, gnd,k)
% gnd : n * 1
% idx: n * 1
n = length(idx);
uf = zeros(n,1);
%assign label to the cluster 
costMat = zeros(k,k);
for j = 1:k
    for t=1:k
        costMat(j,t) = sum(gnd(idx==j)~=t);
    end
end
[assignment,~] = munkres(costMat);
[assignedrows,~]=find(assignment');
for j = 1:k
    uf(idx==j) = assignedrows(j);
end
munkres_acc = length(find(gnd==uf)) / n;
end