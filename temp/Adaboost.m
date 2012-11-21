% This program is realize one step adaboost in (Paul Viola's cascade paper)
% Author : ls
% Date   : 14, November, 2012
% Revise : 19, November, 2012
% Reference : GentleAdaBoost in GML Toolbox

function [learners, alphas, w] = Adaboost(weakLrn, data_index, oldA, oldLrn, w)

    global G_haarfeature; % this is once time compute (length(trainData):length(haarEvaluator))
    global G_response;   % assignment in ComputeFeature.m
    
    % check the number of variable in the input
    narginchk(5, 5);
    
    if isempty(w),
        error('The weights of data must be have');
    end
    w = w / sum(w(:)); % norm the weights

    learners = oldLrn;
    alphas = oldA;
    
    %Only run once time
    %chose best learner

    nodes = train(weakLrn, G_haarfeature(data_index, :)', G_response(data_index)', w); % the input data must be DxN matrix, where D is the
                                             % dimensionality of data, and N is the number of training samples.

    for i = 1:length(nodes)
        curr_tr = nodes{i};

        step_out = calc_output(curr_tr, G_haarfeature(data_index, :)');  %Output the [0 1]

        err = sum(w(step_out ~= G_response(data_index)'));
        
        alp = log((1 - err) / (err + eps));

        alphas(end+1) = alp;

        learners{end+1} = curr_tr;
    end
    
    % Only update the weights in the error classifier
    w(step_out ~= G_response(data_index)) = w(step_out ~= G_response(data_index)) * exp(alp);
end

