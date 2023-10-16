function [W_cannotlink,W_mustlink] = construct_pairwise_link(gnd,k, num_mustlink, num_cannotlink,lambda)

n = length(gnd);

% choose the pairwise constraints randomly
W = ones(n,n);
for t = 1:k
    index = find(gnd==t);
    W(index,index) = 0;
end

[row,col] = find(W > 1e-10);
index = randperm(length(row));
W_cannotlink = sparse(row(index(1:num_cannotlink)),col(index(1:num_cannotlink)),ones(1,num_cannotlink), n,n);
W_cannotlink = W_cannotlink + W_cannotlink';
W_cannotlink = W_cannotlink > 0.5;

[row,col] = find(W < 1e-10);
index = randperm(length(row));
W_mustlink = sparse(row(index(1:num_mustlink)),col(index(1:num_mustlink)),  ones(1,num_mustlink), n,n);
W_mustlink = W_mustlink + W_mustlink';
W_mustlink = W_mustlink > 0.5;
W_mustlink = W_mustlink * lambda;
end