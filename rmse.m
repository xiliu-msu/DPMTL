function [macroR,microR] = rmse(pred, actual)
%

[numStations, T] = size(actual);

I = find(~isnan(actual));
microR = sqrt(sum((pred(I) - actual(I)).^2)/length(I));

macroR = zeros(numStations,1);
for i=1:numStations
    I = find(~isnan(actual(i,:)));
    if length(I) == 0
        warning('Error: length zero found');
        return;
    end
    macroR(i) = sqrt(sum((pred(i,I)-actual(i,I)).^2)/length(I));
end

macroR = mean(macroR);