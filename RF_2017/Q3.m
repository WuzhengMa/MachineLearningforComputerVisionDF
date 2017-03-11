clear;

init;

% Select dataset
% we do bag-of-words technique to convert images to vectors (histogram of codewords)
% Set 'showImg' in getData.m to 0 to stop displaying training and testing images and their feature vectors
[data_train, data_test] = getData('Caltech');
close all;

save('TR_TE_data.mat','data_train','data_test');
       
%load('TR_TE_data.mat');

testLabels = cell(3);
depthTrial = [6, 8, 10];
numTreeTrial = [10, 50, 100];
randomness = [10, 50, 100];
for k = 1:3
    for j = 1:3
        opts = struct;
        opts.depth = 50; 
        opts.numTrees= numTreeTrial(k); 
        opts.numSplits= randomness(j);  %Number of splits to try
        opts.classifierID= 1; % which split function to be used
        
        %Bagging
        Bagsize=95;
        bagged_data_train = cell(1,opts.numTrees);
        for i=1:opts.numTrees
            bagged_data_train{i} = datasample(data_train,Bagsize);
        end
        
        %Train several trees, use different bag for each tree
        treeModels = cell(1, opts.numTrees);
        for i = 1:opts.numTrees
            %!!!Modify here for different Vocab size
            treeModels{i} = treeTrain(bagged_data_train{i}(:,1:1024), bagged_data_train{i}(:,1025), opts); 
        end
        
        %Test phase
        %!!!Modify here for different Vocab size
        [testLabel, testProb] = forestTest(treeModels, data_test(:,1:1024), opts);
        testLabels{k,j} = testLabel;
    end
end