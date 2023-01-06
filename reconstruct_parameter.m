function D_reconstruct = reconstruct_parameter(D_ss,T,w)

V = vander_generate(D_ss,T,w);
V_inv = V^-1;
v_tilde = V_inv(1:size(D_ss),:);

m = size(D_ss,1);
D_de_con = zeros(m*(T+1),size(D_ss,2));
for k = 1:T+1
    D_de_con(((k-1)*m+1):(m*k),:) = D_ss(:,:,k);
end
D_reconstruct = v_tilde*D_de_con;