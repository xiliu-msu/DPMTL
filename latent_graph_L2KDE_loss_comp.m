function [loss,l2_dist] = latent_graph_L2KDE_loss_comp( X,y,pw_gauss_y_s_to_y_s,U,V,lap,para,hpara )
    S = length(y);
    W = U*V;
    yhat = cell(S,1);
    for s = 1:1:S
        yhat{s} = X{s}*W(:,s);
    end
    
    
    loss = para.reg_graph*trace(W*lap*W') + para.reg_U*sum(sum(abs(U))) + para.reg_V*sum(sum(abs(V)));
    
    l2_dist = 0;
    for s = 1:1:S
        size_s = length(y{s});
        loss = loss + norm(y{s}-yhat{s},'fro')^2;
        if (size_s < hpara.valid_size_s)
            continue;
        end
        if (para.reg_dp ~= 0)
            h = hpara.h_given;
            hhat = hpara.hhat_given;
            l2_dist_s = pw_gauss_y_s_to_y_s{s} + pairwise_gaussian( yhat{s},yhat{s},hhat^2) - 2*pairwise_gaussian( yhat{s},y{s},h^2+hhat^2);
            l2_dist_s = sum(sum(l2_dist_s))/((size_s)^2);
            loss = loss + para.reg_dp*l2_dist_s;
            l2_dist = l2_dist + l2_dist_s;
        end
    end
end

