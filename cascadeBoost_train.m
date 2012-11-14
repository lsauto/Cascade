% This problem simulate the function of CvCascadeBoost::train(boost.cpp) in OpenCV
% Author : ls
% Date   : 14, November, 2012
% Revise : 14, November, 2012

function stage = cascadeBoost_trian(data, params)

    w = updateWeights;
    
    disp('+----+---------+---------+');
    disp('|  N |    HR   |    FA   |');
    disp('+----+---------+---------+');
    
    while weak.total < params.weak_count,
        tree = cascadeBoostTree_train(data);     
    end
    

end