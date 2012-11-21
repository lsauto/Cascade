% This problem simulate the function of CvHaarEvaluator::generateFeatures(haarfeature) in OpenCV
% Author : ls
% Date   : 16, November, 2012
% Revise : 16, November, 2012

function evaluator = GenerateHaar(para)
    
    % Check
    if ~isfield(para, 'winSize'),
        error('The winSize should be appoint');
    end
    % 
%     if exist('haarEvaluator.mat', 'file')
%         load('haarEvaluator.mat');
%         return;
%     end
    tic
%     mmode = para.mode;
    evaluator = cell(1, para.winSize.width*para.winSize.width*para.winSize.height*para.winSize.height);
    num = 1;
    for i = 1:3:para.winSize.width,
        for j = 1:3:para.winSize.height,
            for di = 2:2:para.winSize.width-1,% there little difference in the c++ source code , <= para.winSize.width+1, <=  para.winSize.height+1
                for dj = 2:2:para.winSize.height-1,% there little difference in the c++ source code , <= para.winSize.width+1, <=  para.winSize.height+1
                    % haar_x2
                    if (i+di*2 <= para.winSize.width && j + dj <= para.winSize.height), % there little difference in the c++ source code , <= para.winSize.width+1, <=  para.winSize.height+1
%                         evaluator = {evaluator, Feature(false, i, j, di*2, dj, -1, i+di, j, di, dj, 2)};
                         evaluator{num} = Feature(false, i, j, di*2, dj, -1, i+di, j, di, dj, 2);
                         num = num +1;
                    end
                    % haar_y2
                    if (i+di <= para.winSize.width && j + dj*2 <= para.winSize.height), % there little difference in the c++ source code , <= para.winSize.width+1, <=  para.winSize.height+1
%                         evaluator = {evaluator, Feature(false, i, j, di, dj*2, -1, i, j+dj, di, dj, 2)};
                        evaluator{num} = Feature(false, i, j, di, dj*2, -1, i, j+dj, di, dj, 2);
                        num = num +1;
                    end
                    % haar_x3
                    if (i+di*3 <= para.winSize.width && j + dj <= para.winSize.height), % there little difference in the c++ source code , <= para.winSize.width+1, <=  para.winSize.height+1
%                         evaluator = {evaluator, Feature(false, i, j, di*3, dj, -1, i+di, j, di, dj, 3)};
                        evaluator{num} = Feature(false, i, j, di*3, dj, -1, i+di, j, di, dj, 3);
                        num = num +1;
                    end
                    % haar_y3
                    if (i+di <= para.winSize.width && j + dj*3 <= para.winSize.height), % there little difference in the c++ source code , <= para.winSize.width+1, <=  para.winSize.height+1
%                         evaluator = {evaluator, Feature(false, i, j, di, dj*3, -1, i, j+dj, di, dj, 3)};
                        evaluator{num} = Feature(false, i, j, di, dj*3, -1, i, j+dj, di, dj, 3);
                        num = num +1;
                    end
                    % if mode ~= basic
                     % x2_y2
                    if (i+di*2 <= para.winSize.width && j + dj*2 <= para.winSize.height), % there little difference in the c++ source code , <= para.winSize.width+1, <=  para.winSize.height+1
%                         evaluator = {evaluator, Feature(false, i, j, di*2, dj*2, -1, i, j, di, dj, 2, i+di, j+dj, di, dj, 2)};
                        evaluator{num} = Feature(false, i, j, di*2, dj*2, -1, i, j, di, dj, 2, i+di, j+dj, di, dj, 2);
                        num = num +1;
                    end
                end
            end
        end
    end
    toc
    evaluator = evaluator(1:num-1);
%     save('haarEvaluator.mat', 'evaluator');
end


%% CvHaarEvaluator::Feature::Feature
function rect = Feature(tilted, x1, y1, w1, h1, wt1,...
                               x2, y2, w2, h2, wt2,...
                               x3, y3, w3, h3, wt3)
    if (mod(nargin-1, 5) ~= 0|| nargin < 11)
        error('there are some value deficiency');
    end
    
    rect(1).x = x1;
    rect(1).y = y1;
    rect(1).w = w1;
    rect(1).h = h1;
    rect(1).wt = wt1;
    
    rect(2).x = x2;
    rect(2).y = y2;
    rect(2).w = w2;
    rect(2).h = h2;
    rect(2).wt = wt2;
    
    if nargin < 12,
        return;
    end
    
    rect(3).x = x3;
    rect(3).y = y3;
    rect(3).w = w3;
    rect(3).h = h3;
    rect(3).wt = wt3;
    
end