function UV_ss = AMT_general(U,V,sigma_n,T,w,num_parties,t)

A = normrnd(0,sigma_n/sqrt(T),size(U,1),size(U,2));
B = normrnd(0,sigma_n/sqrt(T),size(V,1),size(V,2));
C = A*B;

%% Step 1
% U_ss = zeros(size(U,1),size(U,2),num_parties);
% V_ss = zeros(size(V,1),size(V,2),num_parties);
% A_ss = zeros(size(A,1),size(A,2),num_parties);
% B_ss = zeros(size(B,1),size(B,2),num_parties);
% C_ss = zeros(size(C,1),size(C,2),num_parties);

U_ss = secretshare_parameter(U,w,T,num_parties,sigma_n,t);
V_ss = secretshare_parameter(V,w,T,num_parties,sigma_n,t);
A_ss = secretshare_parameter(A,w,T,num_parties,sigma_n,t);
B_ss = secretshare_parameter(B,w,T,num_parties,sigma_n,t);
C_ss = secretshare_parameter(C,w,T,num_parties,sigma_n,t);


%% Step 2
D_ss = zeros(size(U));
E_ss = zeros(size(V));

for m = 1:num_parties
    D_ss(:,:,m) = U_ss(:,:,m) - A_ss(:,:,m);
    E_ss(:,:,m) = V_ss(:,:,m) - B_ss(:,:,m);
end

%% Step 3: Reconstruct D and E
D = reconstruct_parameter(D_ss,T,w);
E = reconstruct_parameter(E_ss,T,w);

%% Step 4
UV_ss = zeros(size(C));
for m = 1:num_parties
    UV_ss(:,:,m) = D*B_ss(:,:,m) + A_ss(:,:,m)*E + D*E + C_ss(:,:,m);
end