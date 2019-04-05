function l2_dist = L2KDE( y,yhat,hpara )
    n = length(y);
    if (n < hpara.valid_size_s)
        l2_dist = nan;
    else
        h = hpara.h_given;
        hhat = hpara.hhat_given;
        pw_gauss_y_to_y = pairwise_gaussian(y,y,h^2);
        pw_gauss_yhat_to_yhat = pairwise_gaussian(yhat,yhat,hhat^2);
        pw_gauss_yhat_to_y = pairwise_gaussian( yhat,y,h^2+hhat^2);
        l2_dist = pw_gauss_y_to_y + pw_gauss_yhat_to_yhat - 2*pw_gauss_yhat_to_y;
        l2_dist = sum(sum(l2_dist))/(n^2);
    end
end

