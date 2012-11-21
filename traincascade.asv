% This problem simulate the function of traincascade in OpenCV
% Author : ls
% Date   : 13, November, 2012
% Revise : 16, November, 2012

function traincascade()

    clearvars -global;
    clear
    clc
    
    global G_haarfeature; % this is once time compute (length(trainData):length(haarEvaluator))
    global G_response;   % assignment in ComputeFeature.m
                        % 

    pathPos = 'E:\cv\Google\merge_pos.mat';
    pathNeg = 'E:\cv\Google\merge_neg.mat';
    % Parameter
    numStages = 20;
    
    boolFormatSave = 0; 
    tempLeafFARate = inf;
    requiredLeafFARate = power(10, 10);
    % CascadeParams
    cascadeParams.numPos = 30;
    cascadeParams.numNeg = 100; %numPos * 5
    cascadeParams.stageType = 'BOOST';
    cascadeParams.featureType = 'HAAR';
    cascadeParams.sampleWidth = 22;
    cascadeParams.sampleHeight = 42;
    % stageParams
    stageParams.boostType = 'GAB';
    stageParams.minHitRate = 0.7;
    stageParams.maxFalseAlarm = 0.5;
    stageParams.maxWeakCount = 5;
    % featureParams
    featureParams.mode = 'BASIC';
    featureParams.winSize.width =  cascadeParams.sampleWidth;
    featureParams.winSize.height =  cascadeParams.sampleHeight;
    
    % Compute the Global haar feature
    [totalPos, totalNeg] = ComputeFeature(pathPos, pathNeg, featureParams);
    
    % data initial 
    data.curPos = cascadeParams.numPos;
    data.curNeg = cascadeParams.numNeg;
    data.indexPos = 1:cascadeParams.numPos;
    data.indexNeg = totalPos + (1:cascadeParams.numNeg); disp('There need to be change');% temp this place need to change
    
    requiredLeafFARate = power(stageParams.maxFalseAlarm, numStages);
    cascadeClassifier = [];
    % Train
    for i = 1:numStages,
        fprintf('\n ==== TRAINING %d-stages ====\n', i);
        fprintf('<BEGIN\n');
        
        % UpdateTrainingSet
        [boolUpdate, data, tempLeafFARate] = UpdateTrainSet(cascadeClassifier, pathPos, pathNeg, data); % data(indexPos, indexNeg, curNeg, curPos)
        if ~boolUpdate,
            disp('Train dataset for temp stage can not be filled.');
            disp( 'Branch training terminated');
            break;
        end
        
        if tempLeafFARate < requiredLeafFARate,
            fprintf('Required leaf false alarm rate achieved.\n Branch training terminated\n');
            break;
        end
        
        tempStage = CascadeBoost_train(data, stageParams);
        cascadeClassifier{end+1} = tempStage;
        fprintf('END>\n');     
    end
    disp('-------Train is over---------')
    %------------------------------------------------
    % Detection
    % detect the pos picture
%     data.indexPos = 1:cascadeParams.numPos;
%     data.indexNeg = totalPos + (1:cascadeParams.numNeg);
%     detect.posIdx = 1:totalPos;
%     detect.negIdx = totalPos + (1:totalNeg);
    detect.posIdx = 1:cascadeParams.numPos;
    detect.negIdx = totalPos + (1:cascadeParams.numNeg);
    num = 0;
    for i = detect.posIdx,
        if (1 == classifierDetect(cascadeClassifier, i))
            num = num + 1;
        end
    end
    fprintf('Pos : %d / %d [%2.2f%%]\n', num, length(detect.posIdx), 100*num /length(detect.posIdx));
     % detect the pos picture
     num = 0;
    for i = detect.negIdx,
        if (-1 == classifierDetect(cascadeClassifier, i))
            num = num + 1;
        end
    end
    fprintf('Neg : %d / %d [%2.2f%%]\n', num, length(detect.negIdx), 100*num /length(detect.negIdx));

end

%%
function cat = classifierDetect(classifier, idx)

    global G_haarfeature;
    for i = 1:length(classifier),    
        pre = 0;
        for j = 1:length(classifier{i}.stumps),
            pre = pre + classifier{i}.alphas(j) * sign(classifier{i}.stumps{j}(1:end-1) * G_haarfeature(idx, :)' + classifier{i}.stumps{j}(end) * ones(size(pre)));
        end
        if pre >= classifier{i}.threshold,
            cat = 1;           
        else
            cat = -1;
            break;
        end
            
    end
end