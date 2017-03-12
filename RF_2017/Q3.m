clear;

init;

% Select dataset
% we do bag-of-words technique to convert images to vectors (histogram of codewords)
% Set 'showImg' in getData.m to 0 to stop displaying training and testing images and their feature vectors

%[data_train, data_test] = getData('Caltech');
close all;
%{
save('TR_TE_data_RFCB.mat','data_train','data_test');
  %}   
%load('TR_TE_data.mat'); % 1024 kmeans dataset
load('TR_TE_data_RFCB.mat'); % random forest codebook dataset

testLabels = cell(3);
accuracy=zeros(3,3);
depthTrial = [6, 8, 10];
numTreeTrial = [10, 50, 100];
randomness = [10, 50, 100];
for k = 1:1
    for j = 1:1
        opts = struct;
        opts.depth = 10; 
        opts.numTrees= 200; 
        opts.numSplits= 50;  %Number of splits to try
        opts.classifierID= 1; % which split function to be used
        
        %Bagging
        Bagsize=120;
        bagged_data_train = cell(1,opts.numTrees);
        for i=1:opts.numTrees
            bagged_data_train{i} = datasample(data_train,Bagsize);
        end
        
        %Train several trees, use different bag for each tree
        %treeModels = cell(1, opts.numTrees);
        %for i = 1:opts.numTrees
            %!!!Modify here for different Vocab size. kmeans: 1024
            % RFCB:2560
         %   treeModels{i} = treeTrain(bagged_data_train{i}(:,1:2560), bagged_data_train{i}(:,2561), opts); 
        %end
        tic
        treeModels = forestTrain(bagged_data_train{i}(:,1:2560), bagged_data_train{i}(:,2561), opts); 
        toc 
        
        %Test phase
        %!!!Modify here for different Vocab size kmeans: 1024
        % RFCB:2560
        [testLabel, testProb] = forestTest(treeModels.treeModels, data_test(:,1:2560), opts);
        testLabels{k,j} = testLabel;
        
        % Evaluations
        %!!!Modify here for different Vocab size kmeans: 1024
        % RFCB:2560
        %opts.depth 
        %opts.numSplits
        confusion = testLabel==data_test(:,2561);
        accuracy(k,j)= sum(confusion)/150
        
        %
        % Plot confusion matrix
        confusionM=confusionmat(testLabel,data_test(:,2561));
        imagesc(confusionM);
        colormap cool;
        title('Confusion Matrix');
        xlabel('Output Class');
        ylabel('Target Class');
        %
    end
end