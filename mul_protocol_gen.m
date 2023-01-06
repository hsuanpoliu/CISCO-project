function Xw_ss = mul_protocol_gen(X_concatenate_ss,w_t_ss,w,T,num_parties,sigma_n,t)

Xw_ss = zeros(size(X_concatenate_ss,1),size(w_t_ss,2),num_parties);
Xw_temp_ss = zeros(size(X_concatenate_ss,1),size(w_t_ss,2),num_parties,num_parties,num_parties);

%% 
for m = 1:num_parties
    for n = 1:num_parties
        if m == n
            A_ss = secretshare_parameter(X_concatenate_ss(:,:,m)*w_t_ss(:,:,m),w,T,num_parties,sigma_n,t);
            for p = 1:num_parties
                Xw_temp_ss(:,:,m,m,p) = A_ss(:,:,p);
            end
        else
            B_ss = AMT_general(X_concatenate_ss(:,:,m),w_t_ss(:,:,n),sigma_n,T,w,num_parties,t);
            for p = 1:num_parties
                Xw_temp_ss(:,:,m,n,p) = B_ss(:,:,p);
            end
        end
    end
end

%% Combine
for p = 1:num_parties % computation at client p
    for m = 1:num_parties
        for n = 1:num_parties
            Xw_ss(:,:,p) = Xw_ss(:,:,p) + Xw_temp_ss(:,:,m,n,p)/(num_parties^2);
        end
    end
end

