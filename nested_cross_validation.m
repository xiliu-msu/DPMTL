clear;


rng(1);
%% make settings
dirname = '../data/';
dataname = 'ncep_ghcn_XY_des_prcp_atleastsize_30_smalldata_10000_lat_49.35_24.74_lon_-66.95_-124.79';

hpara.method = 'zubin';
hpara.stp_cr_gd = 0.1;
hpara.valid_size_s = 30;
hpara.KDE_htype = 'given';
hpara.h_given = 0.5;
hpara.hhat_given = 0.5;
hpara.rbf_gamma = 100;
% hpara.A_metric = '(1/S)*ones(S,S)';
hpara.A_metric = '1./exp(haversine_pairwise(Y_lat_lon))';
% hpara.A_metric = 'correlation';

hpara.minPoints = 270;
kfold = 9;
vfold = 2;

reg_dp_list = [0,0.1,0.5,1,2,5];
reg_graph_list = [0];
reg_U_list = [0];
reg_V_list = [0];
k_list = [0];


crossvalid_idx = load([dirname,'nest_crossvalididx_N_540_kfold_',num2str(kfold),'_vkfold_',num2str(vfold),'.mat']);



%% Loading the Data ...
matfn = [dirname,dataname,'.mat']; 
data = load(matfn);   
Y_lat_lon = data.Y_lat_lon;
X_lat_lon = data.X_lat_lon;
Y_var = data.Y_var;
X_var = data.X_var;
compressed_X = data.compressed_X;
X_adj = data.X_adj + 1;
Y_load = data.Y;
Y_time = data.Y_time;
X_load = compressed_X(X_adj(:,2),:,:);
[S,~,~] = size(X_load);


% use only big enough stations
numPoints = zeros(S,1);
for s = 1:1:S
    numPoints(s) = length(find(~isnan(Y_load(s,:))));
end
sstations = find(numPoints>hpara.minPoints);
X_load = X_load(sstations,:,:);
Y_load = Y_load(sstations,:);
X_adj = X_adj(sstations,:);
Y_lat_lon = Y_lat_lon(sstations,:);

% use only 197001~201412
T = 540;
compressed_X = compressed_X(:,1:T,:);
X_load = X_load(:,1:T,:);
Y_load = Y_load(:,1:T);
Y_time = Y_time(1:T,:);
[S,T,D] = size(X_load);
fprintf('S = %d,T = %d,D = %d\n',S,T,D);


%% Data Preprocessing
% remove missing values ...
nonmiss_compressed_X = ~isnan(compressed_X(:,:,1));
for idx_var = 2:1:D
    nonmiss_compressed_X = nonmiss_compressed_X & (~isnan(compressed_X(:,:,idx_var)));
end
nonmiss_X = nonmiss_compressed_X(X_adj(:,2),:);
nonmiss_Y = ~isnan(Y_load(:,:));
nonmiss = nonmiss_X & nonmiss_Y;


% add bias
X = ones(size(X_load,1),size(X_load,2),size(X_load,3)+1);
X(:,:,1:end-1) = X_load;
Y = Y_load;
D = D+1;


%% Split Training and Validation Set
% split training and testing
nest_train_index = crossvalid_idx.nest_train_index;
nest_valid_index = crossvalid_idx.nest_valid_index;

X_train = cell(S,kfold,vfold);
Y_train  = cell(S,kfold,vfold);
nonmiss_train = cell(S,kfold,vfold);
for s = 1:1:S
    for ff_outer = 1:1:kfold
        for ff_inner = 1:1:vfold
            X_train{s,ff_outer,ff_inner} = squeeze(X(s,nest_train_index{ff_outer,ff_inner},:));
            if (size(X_train{s,ff_outer,ff_inner},2) == 1)
                X_train{s,ff_outer,ff_inner} = X_train{s,ff_outer,ff_inner}';
            end
            Y_train{s,ff_outer,ff_inner} = Y(s,nest_train_index{ff_outer,ff_inner})';
            nonmiss_train{s,ff_outer,ff_inner} = nonmiss(s,nest_train_index{ff_outer,ff_inner})';
            X_train{s,ff_outer,ff_inner} = X_train{s,ff_outer,ff_inner}(nonmiss_train{s,ff_outer,ff_inner},:);
            Y_train{s,ff_outer,ff_inner} = Y_train{s,ff_outer,ff_inner}(nonmiss_train{s,ff_outer,ff_inner});
        end
    end
end
X_valid = cell(S,kfold,vfold);
Y_valid = cell(S,kfold,vfold);
nonmiss_valid = cell(S,kfold,vfold);
for s = 1:1:S
    for ff_outer = 1:1:kfold
        for ff_inner = 1:1:vfold
            X_valid{s,ff_outer,ff_inner} = squeeze(X(s,nest_valid_index{ff_outer,ff_inner},:));
            if (size(X_valid{s,ff_outer,ff_inner},2) == 1)
                X_valid{s,ff_outer,ff_inner} = X_valid{s,ff_outer,ff_inner}';
            end
            Y_valid{s,ff_outer,ff_inner} = Y(s,nest_valid_index{ff_outer,ff_inner})';
            nonmiss_valid{s,ff_outer,ff_inner} = nonmiss(s,nest_valid_index{ff_outer,ff_inner})';
            X_valid{s,ff_outer,ff_inner} = X_valid{s,ff_outer,ff_inner}(nonmiss_valid{s,ff_outer,ff_inner},:);
            Y_valid{s,ff_outer,ff_inner} = Y_valid{s,ff_outer,ff_inner}(nonmiss_valid{s,ff_outer,ff_inner});
        end
    end
end





%% Parameter Settings
if strcmp(hpara.A_metric,'(1/S)*ones(S,S)')
    hpara.A = (1/S)*ones(S,S);
elseif strcmp(hpara.A_metric,'1./exp(haversine_pairwise(Y_lat_lon))')
    hpara.A = 1./exp(haversine_pairwise(Y_lat_lon)/hpara.rbf_gamma);
end


% parameter list
para_list = cell(length(reg_dp_list)*length(reg_graph_list)*length(reg_U_list)*length(reg_V_list)*length(k_list),1);
para_cnt = 0;
for reg_dp_iter = reg_dp_list
    for reg_graph_iter = reg_graph_list
        for reg_U_iter = reg_U_list
            for reg_V_iter = reg_V_list
                for k_iter = k_list
                    para_cnt = para_cnt + 1;
                    para_list{para_cnt} = [reg_dp_iter,reg_graph_iter,reg_U_iter,reg_V_iter,k_iter];
                end
            end
        end
    end
end


% build the parallel parameters
pp = cell(length(para_list)*kfold,1);
cnt = 0;
para_cnt = 0;
for para_iter = 1:1:length(para_list)
    para_cnt = para_cnt + 1;
    for ff_outer = 1:1:kfold
        for ff_inner = 1:1:vfold
            cnt = cnt + 1;
            pp{cnt}.reg_dp = para_list{para_cnt}(1);
            pp{cnt}.reg_graph = para_list{para_cnt}(2);
            pp{cnt}.reg_U = para_list{para_cnt}(3);
            pp{cnt}.reg_V = para_list{para_cnt}(4);
            pp{cnt}.k = para_list{para_cnt}(5);
            pp{cnt}.para_cnt = para_cnt;
            pp{cnt}.ff_outer = ff_outer;
            pp{cnt}.ff_inner = ff_inner;
        end
    end
end




%% Training and Prediction
micro_rms_train = nan(length(pp),1);
micro_rmscdf_train = nan(length(pp),1);
micro_rms_valid = nan(length(pp),1);
micro_rmscdf_valid = nan(length(pp),1);
micro_l2kde_train = nan(length(pp),1);
micro_l2kde_valid = nan(length(pp),1);

macro_rms_train = nan(length(pp),1);
macro_rmscdf_train = nan(length(pp),1);
macro_rms_valid = nan(length(pp),1);
macro_rmscdf_valid = nan(length(pp),1);
macro_l2kde_train = nan(length(pp),1);
macro_l2kde_valid = nan(length(pp),1);

W = cell(length(pp),1);
% obj = parpool(9);
for iter = 1:1:length(pp)
%     reg_sort = pp{iter}.reg_sort;
%     reg_graph = pp{iter}.reg_graph;
%     reg_U = pp{iter}.reg_U;
%     reg_V = pp{iter}.reg_V;
%     k = pp{iter}.k;
    ff_outer = pp{iter}.ff_outer;
    ff_inner = pp{iter}.ff_inner;
    
   
    % Training 
    if strcmp(hpara.method,'global_lasso')
        W{iter} = global_lasso(X_train(:,ff_outer,ff_inner), Y_train(:,ff_outer,ff_inner),pp{iter}.reg_V,hpara);
    elseif strcmp(hpara.method,'local_lasso')
        W{iter} = local_lasso(X_train(:,ff_outer,ff_inner), Y_train(:,ff_outer,ff_inner),pp{iter}.reg_V,hpara);
    elseif strcmp(hpara.method,'latent_graph_L2KDE')  && ~strcmp(hpara.A,'correlation') 
        [U,V,convg] = minbatchall_latent_graph_L2KDE(X_train(:,ff_outer,ff_inner), Y_train(:,ff_outer,ff_inner), pp{iter}, hpara);
        W{iter} = U*V;
    elseif strcmp(hpara.method,'zubin') 
        [W{iter},convg_sort] = zubin_MCR(X_train(:,ff_outer,ff_inner), Y_train(:,ff_outer,ff_inner),pp{iter}, hpara);
    else
        error('No method has been chosen!!!');
    end
    
    % Prediction
    Yhat_train = cell(S,1);
    Yhat_valid = cell(S,1);
    for s = 1:1:S
        Yhat_train{s} = X_train{s,ff_outer,ff_inner}*W{iter}(:,s);
        Yhat_valid{s} = X_valid{s,ff_outer,ff_inner}*W{iter}(:,s);
    end
    
    % Evaluation
    performance = evaluation( Y_train(:,ff_outer,ff_inner),Yhat_train,hpara );
    macro_rms_train(iter) = performance.macro_rms;
    macro_rmscdf_train(iter) = performance.macro_rmscdf;
    macro_l2kde_train(iter) = performance.macro_l2kde;
    micro_rms_train(iter) = performance.micro_rms;
    micro_rmscdf_train(iter) = performance.micro_rmscdf;
    micro_l2kde_train(iter) = performance.micro_l2kde;
    
    performance = evaluation( Y_valid(:,ff_outer,ff_inner),Yhat_valid,hpara );
    macro_rms_valid(iter) = performance.macro_rms;
    macro_rmscdf_valid(iter) = performance.macro_rmscdf;
    macro_l2kde_valid(iter) = performance.macro_l2kde;
    micro_rms_valid(iter) = performance.micro_rms;
    micro_rmscdf_valid(iter) = performance.micro_rmscdf;
    micro_l2kde_valid(iter) = performance.micro_l2kde;

    
    % Print
    fprintf('Train: f = %d-%d, reg_dp = %.2f, reg_graph = %.2f, reg_U = %.2f, reg_V = %.2f, k = %d, macro_rms = %.4f, macro_rms_cdf = %.4f, macro_l2kde = %.4f\n',ff_outer,ff_inner,pp{iter}.reg_dp,pp{iter}.reg_graph,pp{iter}.reg_U,pp{iter}.reg_V,pp{iter}.k,macro_rms_train(iter),macro_rmscdf_train(iter),macro_l2kde_train(iter));
    fprintf('Valid: f = %d-%d, reg_dp = %.2f, reg_graph = %.2f, reg_U = %.2f, reg_V = %.2f, k = %d, macro_rms = %.4f, macro_rms_cdf = %.4f, macro_l2kde = %.4f\n',ff_outer,ff_inner,pp{iter}.reg_dp,pp{iter}.reg_graph,pp{iter}.reg_U,pp{iter}.reg_V,pp{iter}.k,macro_rms_valid(iter),macro_rmscdf_valid(iter),macro_l2kde_valid(iter));

end
% delete(obj);


%% K-Fold results
% print the results
num_para = length(para_list);
[ kfold_vfold_mean_micro_rms_train,kfold_vfold_std_micro_rms_train ] = kfold_vfold_result( micro_rms_train,pp,num_para,kfold,vfold );
[ kfold_vfold_mean_micro_rmscdf_train,kfold_vfold_std_micro_rmscdf_train ] = kfold_vfold_result( micro_rmscdf_train,pp,num_para,kfold,vfold );
[ kfold_vfold_mean_micro_l2kde_train,kfold_vfold_std_micro_l2kde_train ] = kfold_vfold_result( micro_l2kde_train,pp,num_para,kfold,vfold );

[ kfold_vfold_mean_macro_rms_train,kfold_vfold_std_macro_rms_train ] = kfold_vfold_result( macro_rms_train,pp,num_para,kfold,vfold );
[ kfold_vfold_mean_macro_rmscdf_train,kfold_vfold_std_macro_rmscdf_train ] = kfold_vfold_result( macro_rmscdf_train,pp,num_para,kfold,vfold );
[ kfold_vfold_mean_macro_l2kde_train,kfold_vfold_std_macro_l2kde_train ] = kfold_vfold_result( macro_l2kde_train,pp,num_para,kfold,vfold );


[ kfold_vfold_mean_micro_rms_valid,kfold_vfold_std_micro_rms_valid ] = kfold_vfold_result( micro_rms_valid,pp,num_para,kfold,vfold );
[ kfold_vfold_mean_micro_rmscdf_valid,kfold_vfold_std_micro_rmscdf_valid ] = kfold_vfold_result( micro_rmscdf_valid,pp,num_para,kfold,vfold );
[ kfold_vfold_mean_micro_l2kde_valid,kfold_vfold_std_micro_l2kde_valid ] = kfold_vfold_result( micro_l2kde_valid,pp,num_para,kfold,vfold );


[ kfold_vfold_mean_macro_rms_valid,kfold_vfold_std_macro_rms_valid ] = kfold_vfold_result( macro_rms_valid,pp,num_para,kfold,vfold );
[ kfold_vfold_mean_macro_rmscdf_valid,kfold_vfold_std_macro_rmscdf_valid ] = kfold_vfold_result( macro_rmscdf_valid,pp,num_para,kfold,vfold );
[ kfold_vfold_mean_macro_l2kde_valid,kfold_vfold_std_macro_l2kde_valid ] = kfold_vfold_result( macro_l2kde_valid,pp,num_para,kfold,vfold );


%% Save Kfold Results
save (['./lastrun/nest_MTLMCR_',datestr(datetime,30),'.mat'],'dataname','para_list','crossvalid_idx','hpara',...
'kfold_vfold_mean_micro_rms_train',...
'kfold_vfold_std_micro_rms_train',...
'kfold_vfold_mean_micro_rmscdf_train',...
'kfold_vfold_std_micro_rmscdf_train',...
'kfold_vfold_mean_micro_l2kde_train',...
'kfold_vfold_std_micro_l2kde_train',...
'kfold_vfold_mean_macro_rms_train',...
'kfold_vfold_std_macro_rms_train',...
'kfold_vfold_mean_macro_rmscdf_train',...
'kfold_vfold_std_macro_rmscdf_train',...
'kfold_vfold_mean_macro_l2kde_train',...
'kfold_vfold_std_macro_l2kde_train',...
'kfold_vfold_mean_micro_rms_valid',...
'kfold_vfold_std_micro_rms_valid',...
'kfold_vfold_mean_micro_rmscdf_valid',...
'kfold_vfold_std_micro_rmscdf_valid',...
'kfold_vfold_mean_micro_l2kde_valid',...
'kfold_vfold_std_micro_l2kde_valid',...
'kfold_vfold_mean_macro_rms_valid',...
'kfold_vfold_std_macro_rms_valid',...
'kfold_vfold_mean_macro_rmscdf_valid',...
'kfold_vfold_std_macro_rmscdf_valid',...
'kfold_vfold_mean_macro_l2kde_valid',...
'kfold_vfold_std_macro_l2kde_valid');
