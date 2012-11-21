% This problem simulate the function of int CvCascadeClassifier::fillPassedSamples in OpenCV
% Author : ls
% Date   : 14, November, 2012
% Revise : 16, November, 2012

function [data, consumed, getcount] = FillPassedSamples(data, first, count, isPositive, img_tol)
    addpath('D:\Program Files\kyamagu-mexopencv-55c2e80');
       
    consumed = 0;
    getcount = 0;
    for ii = first:first+count-1,
        while(consumed < size(img_tol, 1))
            consumed = consumed +1;
            
            img = reshpae(temp.img_tolPos(consumed, :), [cascadeParams.sampleHight, cascadeParams.sampleWidth]);
            [innSum, innSqSum, innTilted] = cv.integral(img);
            if predict() ~= 1
                continue;
            end
            
            getcount = getcount + 1;
            data.innSum{ii} = innSum;
            data.innTilted{ii} = innTilted;
            data.normfactor = [data.normfactor; calcNormFactor(data.innSum{ii}, innSqSum)];
        end    
    end
    
    if (isPositive)
        data.cslabel = 1;
    else
        data.cslabel = -1;
    end
    
end