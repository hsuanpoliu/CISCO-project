function w_initial_ss = secretshare_parameter(w_initial,w,T,num_parties,sigma_n,t)
% weight_initial(784,1)
N_w = zeros(size(w_initial,1),size(w_initial,2),T);
% N_w(784,1,3,2)
for p = 1:T
    N_w(:,:,p) = noise_gen_t(sigma_n/sqrt(T),size(w_initial,1),size(w_initial,2),t);
end
w_initial_ss = zeros(size(w_initial,1),size(w_initial,2),num_parties);
% w_initial_ss(784,1,3)
%%
for k=1:num_parties
    Noise_sum = zeros(size(w_initial));
    for p = 1:T
        Noise_sum = Noise_sum + (w(k))^(p)*N_w(:,:,p);
    end
    w_initial_ss(:,:,k) = w_initial + Noise_sum; % w_initial_ss(:,:, k-th party), secret share to "k-th" party)
end