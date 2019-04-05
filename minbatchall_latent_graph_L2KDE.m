function [U,V,convg] = minbatchall_latent_graph_L2KDE(X, y, para, hpara)
    % X is S x {T x d} cell
    % y is S x {T x 1} cell
    % W is D x S matrix
    % A is S x S symmetric matrix
    
    % minbatch_ratio = 0.3;
    minbatch_size = 64;
    % dimensions
    S = length(X);
    D = 14;
    
    % check errors
    if (size(hpara.A,1) ~= S || size(hpara.A,2) ~= S)
        error('erros about A matrix');
    end
    
    % laplacian
    DDiag = zeros(size(hpara.A));
    DDiag(1:size(hpara.A,1)+1:end) = sum(hpara.A,2);
    lap = DDiag - hpara.A;

    
    % initilize U and V , W
    ini = 'rand';
    if strcmp(ini,'rand')
        rng(1);
        U = rand(D,para.k);
        V = rand(para.k,S);
    else
        X_all = zeros(S*1000,D);
        y_all = zeros(S*1000,1);
        cnt_start = 1;
        for s = 1:1:S
            cnt_end = cnt_start + size(X{s},1)-1;
            X_all(cnt_start:cnt_end,:) = X{s};
            y_all(cnt_start:cnt_end) = y{s};
            cnt_start = cnt_end + 1;
        end
        cnt_end = cnt_start + size(X{s},1)-1;
        X_all(cnt_end+1:end,:) = [];
        y_all(cnt_end+1:end) = [];

        global_W = (X_all'*X_all)\(X_all'*y_all); 
        W = repmat(global_W,[1,S]); 
        for s = 1:1:S
            if size(X{s},1)>0
                W(:,s) = pinv(X{s}'*X{s})*(X{s}'*y{s});
            end
        end
        rng(1);
        V = rand(k,S);
        for iter = 1:1:100000
            U = (W*V')/(V*V');
            V = (U'*U)\(U'*W);
        end
    end
    
    % pre-calculated variables (do not change during iterations)
    pw_gauss_y_s_to_y_s = cell(S,1);
    for s = 1:1:S
        h = hpara.h_given;
        pw_gauss_y_s_to_y_s{s} = pairwise_gaussian(y{s},y{s},h^2);
    end
    

    
    



    % batch gradiant descent
    gd_maxiter = 1000000;
    convg = nan;
    stpsz_U = 10^(-8);
    stpsz_V = 10^(-7);
    U_mmt = U;
    V_mmt = V;    
    
    min_fx = 10^10;
    min_para.U = U;
    min_para.V = V;
    for rr = 2:1:gd_maxiter 
        
        % mini batch
        batch_idx = cell(S,1);
        for s = 1:1:S
            size_s = length(y{s});
            if (size_s<hpara.valid_size_s)
                batch_idx{s} = 1:1:size_s;
            else          
                %batch_size_s = round(max(minbatch_ratio*size_s,valid_size_s));
                batch_size_s = min(size_s,minbatch_size);
                batch_idx{s} = randsample(size_s,batch_size_s);
            end
        end
   
        % update U
        U_old = U;
        gd_U = minbatchall_latent_graph_L2KDE_gd(batch_idx,X,y,U_mmt,V,lap,para,hpara,'U');   
        U_new = wthresh(U_mmt - stpsz_U*gd_U, 's', para.reg_U*stpsz_U);
        U = U_new;
        U_mmt = U + (rr-1)/(rr+2)*(U-U_old); 
        

  
        % update V
        V_old = V;
        gd_V = minbatchall_latent_graph_L2KDE_gd(batch_idx,X,y,U,V_mmt,lap,para,hpara,'V');   
        V_new = wthresh(V_mmt - stpsz_V*gd_V, 's', para.reg_V*stpsz_V);
        V = V_new;
        V_mmt = V + (rr-1)/(rr+2)*(V-V_old); 
        
        

        
        % display the progress
        check_period = 5;
        check_point_gd = 10;
        if (mod(rr,check_point_gd) == 1)
            [f_x,l2_dist_x] = latent_graph_L2KDE_loss_comp( X,y,pw_gauss_y_s_to_y_s,U,V,lap,para,hpara );
            if f_x<min_fx(end)
                min_fx = [min_fx;f_x];
                min_para.U = U;
                min_para.V = V;
            else
                min_fx = [min_fx;min_fx(end)];
                stpsz_U = 0.9*stpsz_U;
                stpsz_V = 0.9*stpsz_V;
            end     
            fprintf('gditer = %d,f_old = %.4f,l2_dist_old = %.4f,stpsz_U = %e,stpsz_V = %e, min_fx = %.1f\n',rr,f_x,l2_dist_x,stpsz_U,stpsz_V,min_fx(end));
            % stop criteria            
            if  length(min_fx)>check_period && abs(min_fx(end) - min_fx(end-check_period))/check_period/check_point_gd<=hpara.stp_cr_gd
                break;
            end
            if isnan(f_x) 
               fprintf('fx = nan !!!\n');
               break;
            end
        end

    end
    
    U = min_para.U;
    V = min_para.V;
        
       
      
end
