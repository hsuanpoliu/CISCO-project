function X_concatenate = concatenate(X_ind_ss,num_parties)

X_concatenate = zeros(size(X_ind_ss,1)*num_parties,size(X_ind_ss,2),num_parties); % third term is for the k-th party
for k = 1:num_parties
    for m = 1:num_parties
        X_concatenate((size(X_ind_ss,1)*(m-1)+1):(size(X_ind_ss,1)*(m)),:,k) = X_ind_ss(:,:,m,k); % X_ind_ss(:,:, m-th party, secret share from "m-th to k-th" party)
    end 
end