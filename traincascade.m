% This problem simulate the function of traincascade in OpenCV
% Author : ls
% Date   : 13, November, 2012
% Revise : 14, November, 2012

function traincascade()

    pathPos = 'E:\cv\Google\00001_139_pos';
    pathNeg = 'E:\cv\Google\00001_139_neg';
    % Parameter
    numStages = 20;
    
    boolFormatSave = 0; 
    tempLeafFARate = INF;
    requiredLeafFARate = power(10, 10);
    % CascadeParams
    cascadeParams.numPos = 12;
    cascadeParams.numNeg = 60; %numPos * 5
    cascadeParams.stageType = 'BOOST';
    cascadeParams.featureType = 'HAAR';
    cascadeParams.sampleWidth = 22;
    cascadeParams.sampleHight = 42;
    % stageParams
    stageParams.boostType = 'GAB';
    stageParams.minHitRate = 0.995;
    stageParams.maxFalseAlarm = 0.5;
    stageParams.maxWeakCount = 100;
    % featureParams
    featureParams.mode = 'BASIC';
    
    requiredLeafFARate = pow(stageParams.maxFalseAlarm, numStages);
    % Train
    for i = 1:numStages,
        fprintf('\n ==== TRAINING %d-stages ====\n', i);
        fprintf('<BEGIN');
        
        % UpdateTrainingSet
        [boolUpdate, data, tempLeafFARate] = updateTrainSet(pathPos, pathNeg, cascadeParams);
        if ~boolUpdate,
            error('Train dataset for temp stage can not be filled. \n Branch training terminated\n');
            break;
        end
        
        if tempLeafFARate < requiredLeafFARate,
            fprintf('Required leaf false alarm rate achieved.\n Branch training terminated\n');
            break;
        end
        
        tempStage = cascadeBoost_train(data, stageParams);
        
        fprintf('END>\n');     
    end
    
    

end