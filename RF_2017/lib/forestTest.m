function [Yhard, Ysoft] = forestTest(model, X, opts)
    % X is NxD, where rows are data points
    % model comes from forestTrain()
    % Yhard are hard assignments to X's, Ysoft is NxK array of
    % probabilities, where there are K classes.
    
    if nargin<3, opts= struct; end
    
    %----Wuzheng adjusted this code----
    numTrees= length(model);
    u= model{1}.classes; % Assume we have at least one tree!
    Ysoft= zeros(size(X,1), length(u));
    for i=1:numTrees
        [~, ysoft] = treeTest(model{i}, X, opts);
        Ysoft= Ysoft + ysoft;
    end
    %----------------------------------

    Ysoft = Ysoft/numTrees;
    [~, ix]= max(Ysoft, [], 2);
    Yhard = u(ix);
end
