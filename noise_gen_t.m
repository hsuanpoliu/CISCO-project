function N = noise_gen_t(sigma_n,m,n,t)
N = normrnd(0,sigma_n,m,n);

while sum(abs(N)>t)>0
    N = normrnd(0,sigma_n,m,n);
end
