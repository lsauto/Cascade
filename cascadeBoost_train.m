% This problem simulate the function of CvCascadeBoost::train(boost.cpp) in OpenCV
% Author : ls
% Date   : 14, November, 2012
% Revise : 19, November, 2012

function stage = CascadeBoost_trian(data, params)
       
%    global G_haarfeature; % this is once time compute (length(trainData):length(haarEvaluator))
    global G_response;   % assignment in ComputeFeature.m
    
    disp('+----+---------+---------+');
    disp('|  N |    HR   |    FA   |');
    disp('+----+---------+---------+');
    
    stage.stumps = {};
    stage.alphas = [];
    w = [ones(1, data.curPos)/data.curPos, ones(1, data.curNeg)/data.curNeg]; % initial the weights
    w = w / sum(w);
    
    isErrDes = false;
    while ~isErrDes && length(stage) < params.maxWeakCount,  
%         

        %The Adaboost 
%         [learners, alphas, w] = Adaboost(weakLrn, [data.indexPos, data.indexNeg], alphas, learners, w);
        [stump, tags] = WeakLearner([data.indexPos, data.indexNeg], w);
%         [stump, tags] = LibbLearnerDSTUMP(G_haarfeature([data.indexPos, data.indexNeg], :), G_response([data.indexPos, data.indexNeg]), w);
        
        err = sum(w(tags ~= G_response([data.indexPos, data.indexNeg])'));
        if err > 0.5,
            disp('The error of weak classer is greater than 0.5');
            break;
        end
        
        alp = log((1 - err) / (err + eps));
        
        w(tags ~= G_response([data.indexPos, data.indexNeg])') = w(tags ~= G_response([data.indexPos, data.indexNeg])') * exp(alp);
        w = w / sum(w);
        
        stage.stumps{end+1} = stump;
        stage.alphas(end+1) = alp;
        
        [isErrDes, threshold] = IsErrDesired(stage, params, data.indexPos, data.indexNeg);
        stage.threshold = single(threshold-eps);
    end

    
end

%% Reference the bool CvCascadeBoost::isErrDesired()
function [bool, threshold] = IsErrDesired(classifier, params, indexPos, indexNeg)
    
    global G_haarfeature;
    if isempty(classifier.stumps),
        bool = false;
        threshold = inf;
        return;
    end
    
    % classifier predict
    value_pos = zeros(1, length(indexPos));
    for i = 1:length(classifier.stumps),
        value_pos = value_pos + classifier.alphas(i) * sign(classifier.stumps{i}(1:end-1) * G_haarfeature(indexPos, :)' + classifier.stumps{i}(end) * ones(size(value_pos)));
    end
    
    value_pos = sort(value_pos, 2); % ascending
    
    thresholdIdx = ceil((1 - params.minHitRate) * length(indexPos));
    threshold = value_pos(thresholdIdx);
    
    numPosTrue = length(indexPos) - thresholdIdx;
    numPosTrue = numPosTrue + sum(value_pos(1:thresholdIdx) == threshold);
    hitRate = numPosTrue / length(indexPos);
    
    
    % classifier negtive
    value_neg = zeros(1, length(indexNeg));
    for i = 1:length(classifier.stumps),
        value_neg = value_neg + classifier.alphas(i) * sign(classifier.stumps{i}(1:end-1) * G_haarfeature(indexNeg, :)' + classifier.stumps{i}(end) * ones(size(value_neg)));
    end
    
    numFalse = sum(value_neg >= single(threshold));
    falseAlarm = numFalse / length(indexNeg);
    
    fprintf('| weak total number : %d\n', length(classifier.stumps));
    fprintf('| hitRate = %d\n', hitRate);
    fprintf('| falseAlarm = %d\n', falseAlarm);
    
    if falseAlarm < params.maxFalseAlarm,
        bool = true;
    else
        bool = false;
    end
end

%% Reference the LibbLearnerDSTUPMP.m
function [stump, tags] = WeakLearner(data_index, data_w)
    
    global G_haarfeature; % this is once time compute (length(trainData):length(haarEvaluator))
    global G_response;   % assignment in ComputeFeature.m

    
    num_data = length(G_response(data_index));
%     if sum(size(data) == num_data) ~= 1,
%         error('Data and label can not fit with each other!');
%     end
%     if size(data, 2) == num_data,
%         data = data';
%     end
    
    dim = size(G_haarfeature, 2);
    
   
    err = 10^5;
    idx = 1;
    w_element = 0;
    B = 0;
    for i = 1:dim,
        [stump_temp, werr] = LibbLearnerDSTUMP_LS(G_haarfeature(data_index, i), G_response(data_index), data_w);
        
        if werr < err,
            idx = i;
            w_element = stump_temp(1);
            B = stump_temp(2);
            err = werr;
        end   
    end
    W = zeros(1, dim);
    W(idx) = w_element;
    stump = [W, B];  
    tags = sign(W * G_haarfeature(data_index, :)' + B * ones(1, num_data));
%     errtemp = sum(data_w(tags ~= G_response(data_index)')); 
end
%% Have little difference with the original 
function [stump, werr] = LibbLearnerDSTUMP_LS(data, label, data_w)

%     if sum(size(data) == num_data) ~= 1
%         error('Data and label can not fit with each other!');
%     end
%     if size(data, 2) == num_data
%         data = data';
%     end
    dim = size(data, 2);

    [sorted_data, sorted_index] = sort(data, 1, 'descend'); 
    sorted_data = double(sorted_data);
    sorted_index = int32(sorted_index);

    W = zeros(1, dim);
    B = zeros(1, 1);

    sum_w1 = sum(label .* data_w(:)); % sum_w1 =(s^+) - (s^-)%disp('There may be same error, because werr can be negative');
    t_plus = sum(data_w(label == 1));
    t_minus = sum(data_w(label == -1));
    [w_elemt, b, nz_idx, werr] = WeakLearnerSTUMP(sorted_data, (label .* data_w(:)), sorted_index, sum_w1, t_plus, t_minus);

    W(nz_idx) = w_elemt;
    B = b;

    stump = [W B];

end