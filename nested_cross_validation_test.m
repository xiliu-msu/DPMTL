clear;

%addpath(genpath('/Users/xiliu/Documents/MATLAB/MALSAR1/'));
%% make settings ...
dirname = '../data/';
setting_name = '';

rng(1);


% para_matrix = [[100,0.100000000000000,0,0,10;
% 100,0.100000000000000,0,0,10;
% 100,0.100000000000000,0,0,10;
% 100,0.100000000000000,0,0,10;
% 100,0.100000000000000,0,0,10;
% 100,0.100000000000000,0,0,10;
% 100,0.100000000000000,0,0,10;
% 100,0.100000000000000,0,0,10;
% 100,0.100000000000000,0,0,10]];   % rms

% para_matrix = [[100,0.100000000000000,0,0,10;
% 500,0.100000000000000,0,0,10;
% 500,0.100000000000000,0,0,10;
% 100,0.100000000000000,0,0,10;
% 100,0.100000000000000,0,0,10;
% 100,0.100000000000000,0,0,10;
% 100,0.100000000000000,0,0,10;
% 100,0.100000000000000,0,0,10;
% 500,0.100000000000000,0,0,10]];   % 0.75*rms + 0.25*rms_cdf

para_matrix = [1000,0.100000000000000,0,0,10;
    2000,0.0100000000000000,0,0,10;
    1000,0.100000000000000,0,0,10;
    1000,0.100000000000000,0,0,10;
    1000,0.100000000000000,0,0,10;
    1000,0.100000000000000,0,0,10;
    1000,0.100000000000000,0,0,10;
    2000,0.100000000000000,0,0,10;
    1000,0.100000000000000,0,0,10];   % 0.5*rms + 0.5*rms_cdf

% para_matrix = [[2000,0.100000000000000,0,0,10;
% 2000,0.0100000000000000,0,0,10;
% 2000,0.100000000000000,0,0,10;
% 2000,0.100000000000000,0,0,10;
% 2000,0,0,0,10;2000,0,0,0,10;
% 4000,0.100000000000000,0,0,10;
% 4000,0.100000000000000,0,0,10;
% 4000,0.100000000000000,0,0,10]];   % 0.25*rms + 0.75*rms_cdf

% para_matrix = [[4000,0.100000000000000,0,0,10;
% 4000,0.100000000000000,0,0,10;
% 4000,0.100000000000000,0,0,10;
% 2000,0.100000000000000,0,0,10;
% 4000,0.100000000000000,0,0,10;
% 4000,0.100000000000000,0,0,10;
% 4000,0.100000000000000,0,0,10;
% 10000,0.100000000000000,0,0,10;
% 4000,0.100000000000000,0,0,10]];   % rms_cdf


hpara.minPoints = 270;
kfold = 9;
vfold = 2;




%% Loading settings ... 
if strcmp(setting_name, '')
    dataname = 'ncep_ghcn_XY_des_prcp_atleastsize_30_smalldata_10000_lat_49.35_24.74_lon_-66.95_-124.79';
    hpara.stp_cr_gd = 0.1;
    hpara.valid_size_s = 30;
    hpara.KDE_htype = 'given';
    hpara.h_given = 0.5;
    hpara.hhat_given = 0.5;
    hpara.rbf_gamma = 100;
%     hpara.A_metric = '(1/S)*ones(S,S)';
    hpara.A_metric = '1./exp(haversine_pairwise(Y_lat_lon))';
    crossvalid_idx = load([dirname,'nest_crossvalididx_N_540_kfold_9_vkfold_2.mat']);
else
    load_setting = load(setting_name);
    dataname = load_setting.data;
    hpara = load_setting.hpara;
    crossvalid_idx = load_setting.crossvalid_idx;
end



%% Loading data ...
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


%% Split Training and Testing
% split training and testing
train_index = crossvalid_idx.train_index;
test_index = crossvalid_idx.test_index;

X_train = cell(S,kfold);
Y_train  = cell(S,kfold);
nonmiss_train = cell(S,kfold);
for s = 1:1:S
    for f = 1:1:kfold
        X_train{s,f} = squeeze(X(s,train_index{f},:));
        if (size(X_train{s,f},2) == 1)
            X_train{s,f} = X_train{s,f}';
        end
        Y_train{s,f} = Y(s,train_index{f})';
        nonmiss_train{s,f} = nonmiss(s,train_index{f})';
        X_train{s,f} = X_train{s,f}(nonmiss_train{s,f},:);
        Y_train{s,f} = Y_train{s,f}(nonmiss_train{s,f});
    end
end

X_test = cell(S,kfold);
Y_test  = cell(S,kfold);
nonmiss_test = cell(S,kfold);
for s = 1:1:S
    for f = 1:1:kfold
        X_test{s,f} = squeeze(X(s,test_index{f},:));
        if (size(X_test{s,f},2) == 1)
            X_test{s,f} = X_test{s,f}';
        end
        Y_test{s,f} = Y(s,test_index{f})';
        nonmiss_test{s,f} = nonmiss(s,test_index{f})';
        X_test{s,f} = X_test{s,f}(nonmiss_test{s,f},:);
        Y_test{s,f} = Y_test{s,f}(nonmiss_test{s,f});
    end
end




%% Parameter Settings
if strcmp(hpara.A_metric,'(1/S)*ones(S,S)')
    hpara.A = (1/S)*ones(S,S);
elseif strcmp(hpara.A_metric,'1./exp(haversine_pairwise(Y_lat_lon))')
    hpara.A = 1./exp(haversine_pairwise(Y_lat_lon)/hpara.rbf_gamma);
end


% build the parallel parameterstop -u
pp = cell(kfold,1);
for f = 1:1:kfold
    pp{f}.reg_dp = para_matrix(f,1);
    pp{f}.reg_graph = para_matrix(f,2);
    pp{f}.reg_U = para_matrix(f,3);
    pp{f}.reg_V = para_matrix(f,4);
    pp{f}.k = para_matrix(f,5);
    pp{f}.para_cnt = 1;
    pp{f}.f = f;
end




%% Training and Prediction
micro_rms_train = nan(length(pp),1);
micro_rmscdf_train = nan(length(pp),1);
micro_rms_test = nan(length(pp),1);
micro_rmscdf_test = nan(length(pp),1);
micro_l2kde_train = nan(length(pp),1);
micro_l2kde_test = nan(length(pp),1);


macro_rms_train = nan(length(pp),1);
macro_rmscdf_train = nan(length(pp),1);
macro_rms_test = nan(length(pp),1);
macro_rmscdf_test = nan(length(pp),1);
macro_l2kde_train = nan(length(pp),1);
macro_l2kde_test = nan(length(pp),1);

W = cell(length(pp),1);

% obj = parpool(9);
for iter = 1:1:length(pp)
%     reg_dp = pp{iter}.reg_dp;
%     reg_graph = pp{iter}.reg_graph;
%     reg_U = pp{iter}.reg_U;
%     reg_V = pp{iter}.reg_V;
%     k = pp{iter}.k;
    f = pp{iter}.f;

    
    
    % Training 

%     elseif strcmp(hpara.method,'latent_graph_L2KDE') && ~strcmp(hpara.A_metric,'correlation') 
    [U,V,convg] = minbatchall_latent_graph_L2KDE(X_train(:,f), Y_train(:,f), pp{iter}, hpara);
    W{iter} = U*V;

    
   
     
    % Prediction

    Yhat_train_temp = cell(S,1);
    Yhat_test_temp = cell(S,1);
    for s = 1:1:S
        Yhat_train_temp{s} = X_train{s,f}*W{iter}(:,s);
        Yhat_test_temp{s} = X_test{s,f}*W{iter}(:,s);
    end
    
    % Evaluation
    performance = evaluation( Y_train(:,f),Yhat_train_temp,hpara );
    macro_rms_train(iter) = performance.macro_rms;
    macro_rmscdf_train(iter) = performance.macro_rmscdf;
    macro_l2kde_train(iter) = performance.macro_l2kde;
    micro_rms_train(iter) = performance.micro_rms;
    micro_rmscdf_train(iter) = performance.micro_rmscdf;
    micro_l2kde_train(iter) = performance.micro_l2kde;
    
    performance = evaluation( Y_test(:,f),Yhat_test_temp,hpara );
    macro_rms_test(iter) = performance.macro_rms;
    macro_rmscdf_test(iter) = performance.macro_rmscdf;
    macro_l2kde_test(iter) = performance.macro_l2kde;
    micro_rms_test(iter) = performance.micro_rms;
    micro_rmscdf_test(iter) = performance.micro_rmscdf;
    micro_l2kde_test(iter) = performance.micro_l2kde;

    
    % Print
    fprintf('Train: f = %d, reg_dp = %.2f, reg_graph = %.2f, reg_U = %.2f, reg_V = %.2f, k = %d, macro_rms = %.4f, macro_rms_cdf = %.4f, macro_l2kde = %.4f\n',f,pp{iter}.reg_dp,pp{iter}.reg_graph,pp{iter}.reg_U,pp{iter}.reg_V,pp{iter}.k,macro_rms_train(iter),macro_rmscdf_train(iter),macro_l2kde_train(iter));
    fprintf('Test: f = %d, reg_dp = %.2f, reg_graph = %.2f, reg_U = %.2f, reg_V = %.2f, k = %d, macro_rms = %.4f, macro_rms_cdf = %.4f, macro_l2kde = %.4f\n',f,pp{iter}.reg_dp,pp{iter}.reg_graph,pp{iter}.reg_U,pp{iter}.reg_V,pp{iter}.k,macro_rms_test(iter),macro_rmscdf_test(iter),macro_l2kde_test(iter));

end
% delete(obj);


%% K-Fold predictions
Yhat_train = cell(S,kfold);
Yhat_test = cell(S,kfold);
for iter = 1:1:length(pp)
    f = pp{iter}.f;
    if (f ~= iter)
        error('The indexing is wrong!!!');
    end
    for s = 1:1:S
        Yhat_train{s,f} = X_train{s,f}*W{f}(:,s);
        Yhat_test{s,f} = X_test{s,f}*W{f}(:,s);
    end
end


%% K-Fold results
% print the results
[ kfold_mean_micro_rms_train,kfold_std_micro_rms_train ] = kfold_result( micro_rms_train,pp,1,kfold );
[ kfold_mean_micro_rmscdf_train,kfold_std_micro_rmscdf_train ] = kfold_result( micro_rmscdf_train,pp,1,kfold );
[ kfold_mean_micro_l2kde_train,kfold_std_micro_l2kde_train ] = kfold_result( micro_l2kde_train,pp,1,kfold );


[ kfold_mean_macro_rms_train,kfold_std_macro_rms_train ] = kfold_result( macro_rms_train,pp,1,kfold );
[ kfold_mean_macro_rmscdf_train,kfold_std_macro_rmscdf_train ] = kfold_result( macro_rmscdf_train,pp,1,kfold );
[ kfold_mean_macro_l2kde_train,kfold_std_macro_l2kde_train ] = kfold_result( macro_l2kde_train,pp,1,kfold );


[ kfold_mean_micro_rms_test,kfold_std_micro_rms_test ] = kfold_result( micro_rms_test,pp,1,kfold );
[ kfold_mean_micro_rmscdf_test,kfold_std_micro_rmscdf_test ] = kfold_result( micro_rmscdf_test,pp,1,kfold );
[ kfold_mean_micro_l2kde_test,kfold_std_micro_l2kde_test ] = kfold_result( micro_l2kde_test,pp,1,kfold );

[ kfold_mean_macro_rms_test,kfold_std_macro_rms_test ] = kfold_result( macro_rms_test,pp,1,kfold );
[ kfold_mean_macro_rmscdf_test,kfold_std_macro_rmscdf_test ] = kfold_result( macro_rmscdf_test,pp,1,kfold );
[ kfold_mean_macro_l2kde_test,kfold_std_macro_l2kde_test ] = kfold_result( macro_l2kde_test,pp,1,kfold );

% fprintf('\nreg_sort = %f, reg_graph = %f, reg_U = %f, reg_V = %f, k=%d\n',para_list{para_cnt});

fprintf('micro_rms_train = %.4f +- %.4f\n',kfold_mean_micro_rms_train,kfold_std_micro_rms_train);
fprintf('micro_rmscdf_train = %.4f +- %.4f\n',kfold_mean_micro_rmscdf_train,kfold_std_micro_rmscdf_train);
fprintf('micro_l2kde_train = %.4f +- %.4f\n',kfold_mean_micro_l2kde_train,kfold_std_micro_l2kde_train);
fprintf('macro_rms_train = %.4f +- %.4f\n',kfold_mean_macro_rms_train,kfold_std_macro_rms_train);
fprintf('macro_rmscdf_train = %.4f +- %.4f\n',kfold_mean_macro_rmscdf_train,kfold_std_macro_rmscdf_train);
fprintf('macro_l2kde_train = %.4f +- %.4f\n',kfold_mean_macro_l2kde_train,kfold_std_macro_l2kde_train);

fprintf('micro_rms_test = %.4f +- %.4f\n',kfold_mean_micro_rms_test,kfold_std_micro_rms_test);
fprintf('micro_rmscdf_test = %.4f +- %.4f\n',kfold_mean_micro_rmscdf_test,kfold_std_micro_rmscdf_test);
fprintf('micro_l2kde_test = %.4f +- %.4f\n',kfold_mean_micro_l2kde_test,kfold_std_micro_l2kde_test);
fprintf('macro_rms_test = %.4f +- %.4f\n',kfold_mean_macro_rms_test,kfold_std_macro_rms_test);
fprintf('macro_rmscdf_test = %.4f +- %.4f\n',kfold_mean_macro_rmscdf_test,kfold_std_macro_rmscdf_test);
fprintf('macro_l2kde_test = %.4f +- %.4f\n',kfold_mean_macro_l2kde_test,kfold_std_macro_l2kde_test);  


% %% Save Kfold Results
% save (['./lastrun/MTLMCR_',datestr(datetime,30),'.mat'],'dataname','para_matrix','crossvalid_idx','hpara',...
% 'Yhat_train',...
% 'Yhat_test',...
% 'Y_train',...
% 'Y_test',...
% 'kfold_mean_micro_rms_train',...
% 'kfold_std_micro_rms_train',...
% 'kfold_mean_micro_rmscdf_train',...
% 'kfold_std_micro_rmscdf_train',...
% 'kfold_mean_micro_l2kde_train',...
% 'kfold_std_micro_l2kde_train',...
% 'kfold_mean_macro_rms_train',...
% 'kfold_std_macro_rms_train',...
% 'kfold_mean_macro_rmscdf_train',...
% 'kfold_std_macro_rmscdf_train',...
% 'kfold_mean_macro_l2kde_train',...
% 'kfold_std_macro_l2kde_train',...
% 'kfold_mean_micro_rms_test',...
% 'kfold_std_micro_rms_test',...
% 'kfold_mean_micro_rmscdf_test',...
% 'kfold_std_micro_rmscdf_test',...
% 'kfold_mean_micro_l2kde_test',...
% 'kfold_std_micro_l2kde_test',...
% 'kfold_mean_macro_rms_test',...
% 'kfold_std_macro_rms_test',...
% 'kfold_mean_macro_rmscdf_test',...
% 'kfold_std_macro_rmscdf_test',...
% 'kfold_mean_macro_l2kde_test',...
% 'kfold_std_macro_l2kde_test');
