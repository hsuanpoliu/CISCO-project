function X_ss = secretshare_dataset(X,w,T,num_parties,sigma_n,t)

N_X = zeros(size(X,1),size(X,2),num_parties,T);
for m = 1:num_parties
    for p = 1:T
        N_X(:,:,m,p) = noise_gen_t(sigma_n/sqrt(T),size(X,1),size(X,2),t);
    end
end
X_ss = zeros(size(X,1),size(X,2),num_parties,num_parties);

for m = 1:num_parties % the j-th party; share X_1 to all parties, then, share X_2 to all parties; until X_n
    for k = 1:num_parties % secret share to the i-th party [Xm]k from m-th to k-th
        Noise_sum = zeros(size(X,1),size(X,2));
        for p = 1:T
            Noise_sum = Noise_sum + (w(k))^(p)*N_X(:,:,m,p);
        end
        X_ss(:,:,m,k) = X(:,:,m) + Noise_sum; % X_ss(:,:, m-th party, secret share from "m-th to k-th" party)
    end
end