function G = pairwise_gaussian( y1,y2,sigma )
        G = exp(-pdist2(y1,y2,'squaredeuclidean')/(2*sigma))/sqrt(2*pi*sigma);
end

