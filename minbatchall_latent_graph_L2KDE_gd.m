function gd_f = minbatchall_latent_graph_L2KDE_gd(batch_idx,X,y,U,V,lap,para,hpara,update_para)   
    % get the dimensions
    S = length(X);

    

    W = U*V;
    yhat = cell(S,1);
    for s = 1:1:S
        yhat{s} = X{s}*W(:,s);
    end

    y_minbatch = cell(S,1);
    X_minbatch = cell(S,1);
    yhat_minbatch = cell(S,1);
    for s = 1:1:S
        X_minbatch{s} = X{s}(batch_idx{s},:);
        y_minbatch{s} = y{s}(batch_idx{s},:);
        yhat_minbatch{s} = yhat{s}(batch_idx{s},:);
    end
    
    %% get mu_hat,sigma_hat, and pairwise gaussian pdf    
    if (para.reg_dp ~= 0)

        
        h = zeros(S,1);
        hhat = zeros(S,1);
        pw_gauss_yhat_to_y = cell(S,1);
        pw_gauss_yhat_to_yhat = cell(S,1);
        pw_dif_yhat_to_y = cell(S,1);
        pw_dif_yhat_to_yhat = cell(S,1);
        for s = 1:1:S
            size_s = length(y_minbatch{s});
            h(s) = hpara.h_given;
            hhat(s) = hpara.hhat_given;
            pw_gauss_yhat_to_y{s} = pairwise_gaussian(yhat_minbatch{s},y_minbatch{s},h(s)^2+hhat(s)^2);
            pw_gauss_yhat_to_yhat{s} = pairwise_gaussian(yhat_minbatch{s},yhat_minbatch{s},2*hhat(s)^2);
            temp1 = repmat(y_minbatch{s}',size_s,1);
            temp2 = repmat(yhat_minbatch{s},1,size_s);
            pw_dif_yhat_to_y{s} = temp2 - temp1;
            pw_dif_yhat_to_yhat{s} = temp2 - temp2';
        end
                       
        h_square = h.^2;   % S x 1
        hhat_square = hhat.^2;   % S x 1
        
        
        
        

        %% get the gradient to yhat
        gd_yhat = cell(S,1);
        for s = 1:1:S
            size_s = length(y_minbatch{s});
            temp1 = sum(pw_gauss_yhat_to_y{s}.*pw_dif_yhat_to_y{s},2);
            temp2 = sum(pw_gauss_yhat_to_yhat{s}.*pw_dif_yhat_to_yhat{s},2);
            gd_yhat{s} = para.reg_dp * (((2/(h_square(s)+hhat_square(s)))*temp1 - (1/hhat_square(s))*temp2)/(size_s^2));  % n x 1
        end
    end
    
    
    

    %% get gradients on U, V
    if (strcmp(update_para,'U'))
        gd_f = 2*para.reg_graph*U*V*lap*V';  % d x k
        for s = 1:1:S
            gd_f = gd_f + 2*X_minbatch{s}'*X_minbatch{s}*U*V(:,s)*V(:,s)' - 2*X_minbatch{s}'*y_minbatch{s}*V(:,s)';   % d x k
            if (para.reg_dp ~= 0)
                size_s = size(X_minbatch{s},1);
                if (size_s < hpara.valid_size_s)
                    continue;
                end
                gd_f = gd_f + X_minbatch{s}'*gd_yhat{s}*V(:,s)';
            end
        end
    elseif (strcmp(update_para,'V'))
        gd_f = 2*para.reg_graph*(U'*U)*V*lap;    % k x s
        for s = 1:1:S
            gd_f(:,s) = gd_f(:,s) + 2*U'*X_minbatch{s}'*X_minbatch{s}*U*V(:,s) - 2*U'*X_minbatch{s}'*y_minbatch{s};   % k x 1
            if (para.reg_dp ~= 0)
                size_s = size(X_minbatch{s},1);
                if (size_s < hpara.valid_size_s)
                    continue;
                end
                gd_yhat_s_to_vs = U'*X_minbatch{s}';    % k x n
                gd_f(:,s) = gd_f(:,s) + gd_yhat_s_to_vs*gd_yhat{s};
            end
        end
    end
end
