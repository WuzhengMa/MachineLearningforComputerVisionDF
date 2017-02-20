% Simple Random Forest Toolbox for Matlab
% written by Mang Shao and Tae-Kyun Kim, June 20, 2014.
% updated by Tae-Kyun Kim, Feb 09, 2017

% The codes are made for educational purposes only.
% Some parts are inspired by Karpathy's RF Toolbox

% Student Wuzheng Ma and Kezhen Liu uses this code for completing
% coursework at 02/15/2017
% We also used Karpathy's RF Toolbox at https://github.com/karpathy/Random-Forest-Matlab

% Under BSD Licence

%% Q1 Initialisation and bagging data
% Initialisation
init;

% Select dataset
[data_train, data_test] = getData('Toy_Spiral'); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}

%Bagging
k=95;
bagged_data_train = cell(1,4);
for i=1:4
    bagged_data_train{i} = datasample(data_train,k);
end;

%%Train the decision forest
opts = struct;
opts.depth = 5; 
opts.numTrees= 4; 

%Comparing IG among all split functions
IG1 = [];
IG2 = [];
IG3 = [];
IG4 = [];   
for degreeOfFreedom = 1:120
    opts.numSplits= degreeOfFreedom;  %Number of splits to try
    opts.classifierID= 1; % which split function to be used
    model1 = weakTrain(bagged_data_train{1}(:,[1,2]),bagged_data_train{1}(:,3), opts);
    %IG1 = [IG1, model1.bestGain];
    
    opts.classifierID= 2; % which split function to be used
    model2 = weakTrain(bagged_data_train{1}(:,[1,2]),bagged_data_train{1}(:,3), opts);
    %IG2 = [IG2, model2.bestGain];

    opts.classifierID= 3; % which split function to be used
    model3 = weakTrain(bagged_data_train{1}(:,[1,2]),bagged_data_train{1}(:,3), opts);
    %IG3 = [IG3, model3.bestGain];

    opts.classifierID= 4; % which split function to be used
    model4 = weakTrain(bagged_data_train{1}(:,[1,2]),bagged_data_train{1}(:,3), opts);
    %IG4 = [IG4, model4.bestGain];
end

% figure;
% hold on;
% plot(IG1);
% title('Information Gain varies against different degree of freedom');
% xlabel('Degree of Freedom');
% ylabel('Information Gain');
% 
% plot(IG2);
% title('Information Gain varies against different degree of freedom');
% xlabel('Degree of Freedom');
% ylabel('Information Gain');
% 
% plot(IG3);
% title('Information Gain varies against different degree of freedom');
% xlabel('Degree of Freedom');
% ylabel('Information Gain');
% 
% plot(IG4);
% title('Information Gain varies against different degree of freedom');
% xlabel('Degree of Freedom');
% ylabel('Information Gain');
% legend('decision stump','2D linear', 'Conic section learner', 'Distance learner')
% hold off;

%Train A Decision Forest
opts.depth = 5; 
opts.numTrees= 4; 
opts.numSplits= 15;  %Number of splits to try
opts.classifierID= 1; % which split function to be used

%Train 4 trees, use different bag for each tree
treeModels = cell(1, 4);
for i = 1:4
    treeModels{i} = treeTrain(bagged_data_train{i}(:,[1,2]), bagged_data_train{i}(:,3), opts);
end

%Visualise parent node split
% figure;
% subplot(2,2,1);
% %visualize axis-aligned
% subplot(2,2,2);
% hist(bagged_data_train{1}(:,3), 1:3);
% title('Histogram of Training Set');
% subplot(2,2,3);
% hist(treeModels{1}.weakModels{1}.leftHistogram, 1:3);
% title('Histogram of Left Child Node');
% subplot(2,2,4);
% hist(treeModels{1}.weakModels{1}.rightHistogram, 1:3);
% title('Histogram of Right Child Node');

%Visualize some leaf nodes
% figure;
% subplot(2,2,1);
% bar(treeModels{1}.leafdist(13,:));
% title('Visualization on Leaf Node (13th)');
% subplot(2,2,2);
% bar(treeModels{1}.leafdist(14,:));
% title('Visualization on Leaf Node (14th)');
% subplot(2,2,3);
% bar(treeModels{1}.leafdist(15,:));
% title('Visualization on Leaf Node (15th)');
% subplot(2,2,4);
% bar(treeModels{1}.leafdist(16,:));
% title('Visualization on Leaf Node (16th)');


%% Q2

%Classification on Novel Test Points 
test_point = [-.5 -.7; .4 .3; -.7 .4; .5 -.5];
%[testLabel, testProb] = forestTest(treeModels, test_point, opts);
% hold on;
% scatterTestData([test_point, testLabel], 'Novel');
% plot_toydata(data_train);
% hold off;
% title('Novel Testing Points Classification Result');

%Classification on Dense Test Points
[testLabel, testProb] = forestTest(treeModels, data_test(:,1:2), opts);
figure;
hold on;
scatterTestData([data_test, testLabel], 'Dense');
plot_toydata(data_train);
hold off;
