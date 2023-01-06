clc
clear
close all
%% load dataset
X_train = readmatrix("MNIST_X_train.csv")/255;
Y_train_original = readmatrix("MNIST_Y_train.csv");
Y_train = ((Y_train_original > 3));
n_col = length(X_train(1,:));

%% spliting dataset to the parties and settings
sigma_n = [5*10^4,1.8*10^5,3*10^5];
J = 21;
Trial_cen = 20;
Trial = 30;

num_parties = 2;
lr_cen = 0.01;
lr_dist = [0.0005,10^(-9),10^(-12)*ones(1,2),10^(-13)*ones(1,5),10^(-15)*ones(1,12)]; 
T = num_parties-1;
t = 10^6;
B = 100;
len_train = length(X_train(:,1))/num_parties;

w_temp = exp(2*1i*pi/num_parties);
w = (w_temp.^(0:(num_parties-1))');

X_ind = zeros(len_train,length(X_train(1,:)),num_parties);
Y_ind = zeros(len_train,1,num_parties);
for m = 1:num_parties
    X_ind(:,:,m) = X_train((len_train*(m-1)+1):((len_train)*m),:);
    Y_ind(:,:,m) = Y_train((len_train*(m-1)+1):((len_train)*m),:);
end
acc_train_plot_all = zeros(length(sigma_n),J);
acc_valid_plot_all = zeros(length(sigma_n),J);
acc_train = zeros(1,J);
acc_test = zeros(1,J);

%% Centralized
acc_train_total_cen = zeros(Trial_cen,J);
for trial = 1:Trial_cen
    w_t_cen = randn(n_col,1)*100;

    for iter = 1:J
        Xw = X_train*w_t_cen;
        g_hat =  1./(1+exp(-Xw));
        XTgXwy = X_train.'*(g_hat-Y_train);

        w_t_cen_temp = w_t_cen - lr_cen/(num_parties*n_col)*XTgXwy;

        pred_vals_train = X_train*w_t_cen_temp;
        g= @(x)(1./(1+exp(-x)));
        pred_prob_train=g(pred_vals_train);
        pred_train = pred_prob_train>0.5;
        acc_train(1,iter) = 1-sum(abs(pred_train-Y_train))/length(Y_train);

        w_t_cen = normalize(w_t_cen_temp,'norm');
        fprintf("Sigma order: 1; Trail: %d; iter = %d; acc: %.2f;\n",trial,iter,acc_train(1,iter))
        
    end
    acc_train_total_cen(trial,:) = acc_train;
end
acc_train_plot_all(1,:) = movmean(sum(acc_train_total_cen/Trial_cen,1),1);


%% Decentralized
for p = 1:length(sigma_n)
    %% Training
    acc_train_total = zeros(Trial,J);

    for trial = 1:Trial
        X_ind = zeros(len_train,size(X_train,2),num_parties);
        Y_ind = zeros(len_train,1,num_parties);
        for m = 1:num_parties
            X_ind(:,:,m) = X_train((len_train*(m-1)+1):((len_train)*m),:);
            Y_ind(:,:,m) = Y_train((len_train*(m-1)+1):((len_train)*m),:);
        end

        weight_initial = randn(n_col,1);
        w_t_ss = secretshare_parameter(weight_initial,w,T,num_parties,sigma_n(p),t);

        

        for iter = 1:J
            X_ind_B = zeros(B,size(X_ind,2),num_parties);
            Y_ind_B = zeros(B,size(Y_ind,2),num_parties);
            for m = 1:num_parties
                idx = randperm(size(X_ind,1),B);
                X_ind_B(:,:,m) = X_ind(idx,:,m);
                Y_ind_B(:,:,m) = Y_ind(idx,:,m);
            end
            
            %% Secret Sharing for the dataset X
            X_ind_ss = secretshare_dataset(X_ind_B,w,T,num_parties,sigma_n(p),t);
        
            %% Secret Sharing for the labels y
            y_ind_ss = secretshare_dataset(Y_ind_B,w,T,num_parties,sigma_n(p),t);
            
            %% Concatenate the dataset at each client
            X_concatenate_ss = concatenate(X_ind_ss,num_parties);
            y_concatenate_ss = concatenate(y_ind_ss,num_parties);
            %% ROUND 1
            Xw_ss = mul_protocol_gen(X_concatenate_ss,w_t_ss,w,T,num_parties,sigma_n(p),t);

            g_hat_ss = zeros(size(Xw_ss));
            g_hat_Xw_y_ss = zeros(size(Xw_ss));
            for m = 1:num_parties
                g_hat_ss(:,:,m) = ones(size(Xw_ss(:,:,m),1),1)/2 + Xw_ss(:,:,m)/4;
                g_hat_Xw_y_ss(:,:,m) = g_hat_ss(:,:,m) - y_concatenate_ss(:,:,m);
            end
           
            %% ROUND 2
            X_concatenate_Tran_ss = zeros(size(X_concatenate_ss,2),size(X_concatenate_ss,1),num_parties);
            for m = 1:num_parties
                X_concatenate_Tran_ss(:,:,m) = X_concatenate_ss(:,:,m).';
            end

            XTgXwy_ss = mul_protocol_gen(X_concatenate_Tran_ss,g_hat_Xw_y_ss,w,T,num_parties,sigma_n(p),t);

            %% Update
            w_t_temp_ss = zeros(size(w_t_ss));
            for m = 1:num_parties
                w_t_temp_ss(:,:,m) = w_t_ss(:,:,m) - lr_dist(iter)/(num_parties*n_col)*XTgXwy_ss(:,:,m);
            end

            w_t_temp = reconstruct_parameter(w_t_temp_ss,T,w);
            for m = 1:num_parties
                w_t_ss(:,:,m) = normalize(w_t_temp_ss(:,:,m),'norm');
            end

            %% Accuracy Calculation
            pred_vals_train = X_train*w_t_temp;
            g = @(x)(1./(1+exp(-x)));
            pred_prob_train = g(pred_vals_train);
            pred_train = pred_prob_train>0.5;
            acc_train(1,iter) = 1-sum(abs(pred_train-Y_train))/length(Y_train);

            fprintf("Sigma order: %d; Trail: %d; iter = %d; acc: %.2f;\n",p,trial,iter,acc_train(1,iter))
        end
        acc_train_total(trial,:) = acc_train;
    end
    acc_train_plot_all(p+1,:) = movmean(sum(acc_train_total,1)/Trial,1);
end

%% Plot
sp = 2;

legendInfo{1} = ['Centralized'];
legendInfo{2} = ['$\sigma=5\times 10^4$'];
legendInfo{3} = ['$\sigma=1.8\times 10^5$'];
legendInfo{4} = ['$\sigma=3\times 10^5$'];

figure(1)
plot(0:sp:J-1,acc_train_plot_all(1,1:sp:end),'s--','color',[0.6350 0.0780 0.1840],'linewidth',2,'markersize',12)
hold on
plot(0:sp:J-1,acc_train_plot_all(2,1:sp:end),'kx--','linewidth',2,'markersize',12)
hold on
plot(0:sp:J-1,acc_train_plot_all(3,1:sp:end),'bo--','linewidth',2,'markersize',12)
hold on
plot(0:sp:J-1,acc_train_plot_all(4,1:sp:end),'diamond--r','linewidth',2,'markersize',12)
hold on

grid on
axis([0,iter-1,0,1])
legend(legendInfo,'Location','southeast','Interpreter','latex','FontSize',15)
xlabel('Iteration','fontsize',12)
ylabel('Accuarcy','fontsize',12)
set(gcf,'units','centimeters','position',[11 3 16 8])