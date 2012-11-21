% This problem simulate the function of CvCascadeClassifier::updateTrainingSet in OpenCV
% Author : ls
% Date   : 16, November, 2012
% Revise : 16, November, 2012

function [boolUpdate, data, acceptanceRatio] = UpdateTrainSet(cascadeClassifier, pathPos, pathNeg, data)
    
    if isempty(cascadeClassifier),
        boolUpdate = true;
        acceptanceRatio = 1;
        return;
    end
    % check 
    if ~isfield(data, 'indexPos') || ~isfield(data, 'indexNeg') || ~isfield(data, 'curNeg') || ~isfield(data, 'curPos'),
        error('the number of field in data is wrong');
    end
    
    if length(data.indexPos) ~= data.curPos || length(data.indexNeg) ~= data.curNeg,
        error('The lenght must be match');
    end
    
    %--------The pos ----------------- 
    delteIndex = [];
%     temp = load(pathPos);
    for i = 1:data.curPos,
        if 1 == Predict(cascadeClassifier, data.indexPos(i)),
            continue;
        end
        delteIndex = [delteIndex; i];
    end
    
    posCount = data.curPos - length(delteIndex);
    if 0 == posCount,
        boolUpdate = false;
        acceptanceRatio = 0;
        return;
    end
    fprintf('POS count : consumed [%d : %d]\n', posCount, data.curPos);
    
    %Update 
    data.indexPos(delteIndex) = [];
    data.curPos = length(data.indexPos);
    
     %--------The neg ----------------- 
    delteIndex = [];
%     temp = load(pathNeg);
    for i = 1:data.curNeg,
        if 1 == Predict(cascadeClassifier, data.indexNeg(i)),
            continue;
        end
        delteIndex = [delteIndex; i];
    end
    
    negCount = data.curNeg - length(delteIndex);
    if 0 == negCount,
        boolUpdate = false;
        acceptanceRatio = 0;
        return;
    end
    acceptanceRatio = negCount / (data.curNeg + eps);
    fprintf('NEG count : acceptanceRatio [%d : %d]\n', negCount, acceptanceRatio);
    
    %UPdate
    data.indexNeg(delteIndex) = [];
    data.curNeg = length(data.indexNeg);
    
    % return
    boolUpdate = true;
end

%% temp
function cat = Predict(classifier, idx)
    
    global G_haarfeature;
    
    % Only predict using the last classifier
    pre = 0;
    for j = 1:length(classifier{end}.stumps),
        pre = pre + classifier{end}.alphas(j) * sign(classifier{end}.stumps{j}(1:end-1) * G_haarfeature(idx, :)' + classifier{end}.stumps{j}(end) * ones(size(pre)));
    end

    if pre >= classifier{end}.threshold,
        cat = 1;
    else
        cat = 0;
    end
        
end