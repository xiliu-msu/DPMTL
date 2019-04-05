function A = pairwise_spatial_correlation( y,nonmiss,T,S)  
    y_complete = nan(T,S);
    for s = 1:1:S
        subset = find(nonmiss{s} == 1);
        y_complete(subset,s) = y{s};
    end


    A = nan(S,S);
    for s = 1:1:S
        if (mod(s,10) == 0)
            fprintf('%d\n',s);
        end
        for r = s:1:S
            temp = nonmiss{s}.*nonmiss{r};
            subset = find(temp == 1);
            if length(subset)<30
                continue;
            end
            corr_temp = corr(y_complete(subset,s),y_complete(subset,r));
            A(s,r) = corr_temp;
            A(r,s) = corr_temp;
        end
    end
    
 
end

