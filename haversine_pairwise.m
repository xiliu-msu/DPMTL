function [km, nmi, mi] = haversine_pairwise(locs)

    n = size(locs,1);
    % Convert all decimal degrees to radians
    locs = arrayfun(@(x) x .* pi./180, locs);


    %% Begin calculation

    R = 6371;                                   % Earth's radius in km
    
%     delta_lat = locs{2}(1) - locs{1}(1);        % difference in latitude
%     delta_lon = locs{2}(2) - locs{1}(2);        % difference in longitude
    delta_lat = pdist2(locs(:,1),locs(:,1));
    delta_lon = pdist2(locs(:,2),locs(:,2));
   
    temp = cos(locs(:,1));
    a = sin(delta_lat/2).^2 + repmat(temp,1,n) .* repmat(temp',n,1) .* (sin(delta_lon/2).^2);
    c = 2 * atan2(sqrt(a), sqrt(1-a));
    km = R * c;                                 % distance in km


    %% Convert result to nautical miles and miles

    nmi = km * 0.539956803;                     % nautical miles
    mi = km * 0.621371192;                      % miles
end