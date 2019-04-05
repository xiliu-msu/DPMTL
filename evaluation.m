function performance = evaluation( y,yhat,hpara )

    S = length(y);
    
    
    y_sort = cell(S,1);
    yhat_sort = cell(S,1);
    y_all = zeros(S*1000,1);
    yhat_all = zeros(S*1000,1);
    y_sort_all = zeros(S*1000,1);
    yhat_sort_all = zeros(S*1000,1);
    cnt = 0;    
    for s = 1:1:S
        size_s = length(y{s});
        if (size_s == 0) || (any(isnan(yhat{s})))
            continue;
        end
        y_sort{s} = sort(y{s});
        yhat_sort{s} = sort(yhat{s});
        y_all(cnt+1:cnt+size_s) = y{s};
        yhat_all(cnt+1:cnt+size_s) = yhat{s};
        y_sort_all(cnt+1:cnt+size_s) = y_sort{s};
        yhat_sort_all(cnt+1:cnt+size_s) = yhat_sort{s};
        cnt = cnt + size_s;  
    end    
    y_all = y_all(1:cnt);
    yhat_all = yhat_all(1:cnt);
    y_sort_all = y_sort_all(1:cnt);
    yhat_sort_all = yhat_sort_all(1:cnt);
    
    
    % micro evaluations
    if (S<=10)
        performance.micro_rms = sqrt(mean((y_all - yhat_all).^2));
        performance.micro_rmscdf = sqrt(mean((y_sort_all - yhat_sort_all).^2));
        performance.micro_l2kde = L2KDE( y_all,yhat_all,hpara );
    else
        performance.micro_rms = sqrt(mean((y_all - yhat_all).^2));
        performance.micro_rmscdf = sqrt(mean((y_sort_all - yhat_sort_all).^2));
        performance.micro_l2kde = nan;
    end
    
    
    % macro evaluations
    macro_rms_temp = nan(S,1);
    macro_rms_cdf_temp = nan(S,1);
    macro_l2kde_temp = nan(S,1);
    % macro evaluations
    for s = 1:1:S
        size_s = length(y{s});
        if (size_s == 0) || (any(isnan(yhat{s})))
            continue;
        end
        % rms
        macro_rms_temp(s) = sqrt(mean((y{s} - yhat{s}).^2));
        % rms_cdf
        macro_rms_cdf_temp(s) = sqrt(mean((y_sort{s} - yhat_sort{s}).^2));      
        % L2-KDE
        macro_l2kde_temp(s) = L2KDE( y{s},yhat{s},hpara );
        
       
    end

  
    
    
    
    performance.macro_rms = nanmean(macro_rms_temp);
    performance.macro_rmscdf = nanmean(macro_rms_cdf_temp);
    performance.macro_l2kde = nanmean(macro_l2kde_temp);
    
    



end

