function [ kfold_vfold_mean,kfold_vfold_std  ] = kfold_vfold_result( pp_result,pp,num_para,kfold,vfold )
    num_result = length(pp_result);
    if num_para*kfold*vfold ~= num_result || length(pp)~=num_result
        error('The results are wrong!!!');
    end
    
    % kfold table
    kfold_vfold_result = nan(num_para,kfold,vfold);
    for iter = 1:1:num_result
        para_cnt = pp{iter}.para_cnt;
        ff_outer = pp{iter}.ff_outer;
        ff_inner = pp{iter}.ff_inner;      
        kfold_vfold_result(para_cnt,ff_outer,ff_inner) = pp_result(iter);
    end


    % kfold mean and std
    kfold_vfold_mean = nan(num_para,kfold);
    kfold_vfold_std = nan(num_para,kfold);
    for para_cnt = 1:1:num_para
        for ff_outer = 1:1:kfold
            kfold_vfold_mean(para_cnt,ff_outer) = nanmean(kfold_vfold_result(para_cnt,ff_outer,:));
            kfold_vfold_std(para_cnt,ff_outer) = nanstd(kfold_vfold_result(para_cnt,ff_outer,:));  
        end
    end


end

