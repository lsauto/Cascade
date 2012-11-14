% This problem simulate the function of calcNormFactor(fetures.cpp) in OpenCV
% Author : ls
% Date   : 14, November, 2012
% Revise : 14, November, 2012

function normfactor = calcNormFactor(innsum, innsqsum)

    % check 
    if size(innsum, 1) <3 || size(innsqsum, 2) < 3 || size(innsum, 1) ~= size(innsqsum, 1) ...
            || size(innsum, 2) ~= size(innsqsum, 2),
        error('The size of sum and sqsum must be same');
    end
    
    area = size(innsum, 1) * size(innsum, 2);
    valSum = innsum(end-2, end-2) - innsum(2, end-2) - innsum(end-2, 2) + innsum(2, 2);
    
    valSqSum = innsqsum(end-2, end-2) - innsqsum(2, end-2) - innsqsum(end-2, 2) + innsqsum(2, 2);
    
    normfactor = sqrt(area*valSqSum - valSum*valSum);
end