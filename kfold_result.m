function [ kfold_mean,kfold_std ] = kfold_result( pp_result,pp,num_para,kfold )

    num_result = length(pp_result);
    if num_para*kfold ~= num_result || length(pp)~=num_result
        error('The results are wrong!!!');
    end
    
    % kfold table
    kfold_result = nan(num_para,kfold);
    for iter = 1:1:num_result
        para_cnt = pp{iter}.para_cnt;
        f = pp{iter}.f;
        kfold_result(para_cnt,f) = pp_result(iter);
    end


    % kfold mean and std
    kfold_mean = nan(num_para,1);
    kfold_std = nan(num_para,1);
    for para_cnt = 1:1:num_para
        kfold_mean(para_cnt) = nanmean(kfold_result(para_cnt,:));
        kfold_std(para_cnt) = nanstd(kfold_result(para_cnt,:));   
    end
end

