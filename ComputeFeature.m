% This problem simulate the function of traincascade in OpenCV
% The output is a global variable 
% Author : ls
% Date   : 16, November, 2012
% Revise : 16, November, 2012

function [numPos, numNeg] = ComputeFeature(pathPos, pathNeg, para)

    addpath('D:\Program Files\kyamagu-mexopencv-55c2e80');

    % Compute the pos images
    tempPos = load(pathPos);
    numPos = size(tempPos.img_tolPos, 3);
    
    col_pos.innSum = cell(1, numPos);
    col_pos.norFactor = zeros(1, numPos);
    for i = 1:numPos,
        [innSum, innSqSum, ~] = cv.integral(tempPos.img_tolPos(:, :, i));
        col_pos.innSum{i} = innSum;
        col_pos.norFactor(i) = CalcNormFactor(innSum, innSqSum);
    end
    clear tempPos
    
    % Compute the neg images
    tempNeg = load(pathNeg);
    numNeg = size(tempNeg.img_tolNeg, 3);
    
    col_Neg.innSum = cell(1, numNeg);
    col_Neg.norFactor = zeros(1, numNeg);
    for i = 1:numNeg,
        [innSum, innSqSum, ~] = cv.integral(tempNeg.img_tolNeg(:, :, i)); % the size of innSum is the size(img) + 1
        col_Neg.innSum{i} = innSum;
        col_Neg.norFactor(i) = CalcNormFactor(innSum, innSqSum);
    end
    clear tempNeg
    
    global G_haarfeature; % this is once time compute (length(trainData):length(haarEvaluator))
    global G_response;   % assignment in ComputeFeature.m
    G_response = [ones(numPos, 1); -ones(numNeg, 1)]; 
    % ------check exist -------------------
    if exist('G_haarfeature.mat', 'file'),
        load('G_haarfeature.mat');
        return;
    end
    % this only for speed up
    col_innSum = [col_pos.innSum, col_Neg.innSum];
    col_norFactor = [col_pos.norFactor, col_Neg.norFactor];
    clear col_pos col_Neg
    
    haarEvaluator = GenerateHaar(para);
    fprintf('Start compute feature\n');
    tic
    G_haarfeature = zeros(numPos+numNeg, length(haarEvaluator), 'single');
    for i = 1:length(col_innSum),
        for j = 1:length(haarEvaluator),
            G_haarfeature(i, j) = HaarFeature(col_innSum{i}, haarEvaluator{j}, col_norFactor(i));
        end
        fprintf('The [%d] : %d\n', i, length(col_innSum));
    end
    fprintf('End compute feature (time: %d second)\n', toc);
    
    save('G_haarfeature.mat', 'G_haarfeature');
end

%% generate the haar feature
function haar = HaarFeature(innSum, rect, normfactor)
    % The reason of add 1 is that the size of innSum is the size(img) + 1
    haar = rect(1, 1).wt*(innSum(rect(1, 1).y + rect(1, 1).h+1, (rect(1, 1).x + rect(1, 1).w+1)) - innSum(rect(1, 1).y+1, (rect(1, 1).x + rect(1, 1).w+1))...
                         -innSum(rect(1, 1).y + rect(1, 1).h+1, (rect(1, 1).x +1)) + innSum(rect(1, 1).y+1, rect(1, 1).x+1));
    haar = haar + rect(1, 2).wt*(innSum(rect(1, 2).y + rect(1, 2).h+1, (rect(1, 2).x + rect(1, 2).w+1)) - innSum(rect(1, 2).y+1, (rect(1, 2).x + rect(1, 2).w+1))...
                                 -innSum(rect(1, 2).y + rect(1, 2).h+1, (rect(1, 2).x +1)) + innSum(rect(1, 2).y+1, rect(1, 2).x+1));
    if length(rect) == 3,
        haar = haar + rect(1, 3).wt*(innSum(rect(1, 3).y + rect(1, 3).h+1, (rect(1, 3).x + rect(1, 3).w+1)) - innSum(rect(1, 3).y+1, (rect(1, 3).x + rect(1, 3).w+1))...
                                     -innSum(rect(1, 3).y + rect(1, 3).h+1, (rect(1, 3).x +1)) + innSum(rect(1, 3).y+1, rect(1, 3).x+1));
    end
    haar = single(single(haar) / normfactor);
end