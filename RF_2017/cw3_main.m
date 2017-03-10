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
% figure;
% title('Plots of baggings');
% subplot(2,2,1);
% plot_toydata(bagged_data_train{1});
% subplot(2,2,2);
% plot_toydata(bagged_data_train{2});
% subplot(2,2,3);
% plot_toydata(bagged_data_train{3});
% subplot(2,2,4);
% plot_toydata(bagged_data_train{4});

%%Train the decision forest
opts = struct;
opts.depth = 5; 
opts.numTrees= 4; 

%Comparing IG among all split functions

IG1_100 = zeros(1,120);
IG2_100 = zeros(1,120);
IG3_100 = zeros(1,120);
IG4_100 = zeros(1,120); 
IG5_100 = zeros(1,120); 
for times = 1:100 %run 100 times and compute average
    IG1 = [];
    IG2 = [];
    IG3 = [];
    IG4 = [];
    IG5 = []; 
    for degreeOfFreedom = 1:120
        opts.numSplits= degreeOfFreedom;  %Number of splits to try
        opts.classifierID= 1; % which split function to be used
        model1 = weakTrain(bagged_data_train{1}(:,[1,2]),bagged_data_train{1}(:,3), opts);
        IG1 = [IG1, model1.bestGain];

        opts.classifierID= 2; % which split function to be used
        model2 = weakTrain(bagged_data_train{1}(:,[1,2]),bagged_data_train{1}(:,3), opts);
        IG2 = [IG2, model2.bestGain];

        opts.classifierID= 3; % which split function to be used
        model3 = weakTrain(bagged_data_train{1}(:,[1,2]),bagged_data_train{1}(:,3), opts);
        IG3 = [IG3, model3.bestGain];

        opts.classifierID= 4; % which split function to be used
        model4 = weakTrain(bagged_data_train{1}(:,[1,2]),bagged_data_train{1}(:,3), opts);
        IG4 = [IG4, model4.bestGain];
        
        opts.classifierID= 5; % which split function to be used
        model5 = weakTrain(bagged_data_train{1}(:,[1,2]),bagged_data_train{1}(:,3), opts);
        IG5 = [IG5, model5.bestGain];
    end
    IG1_100 = IG1_100 + IG1;
    IG2_100 = IG2_100 + IG2;
    IG3_100 = IG3_100 + IG3;
    IG4_100 = IG4_100 + IG4;
    IG5_100 = IG5_100 + IG5;
end

figure;
hold on;
plot(IG1_100/100);
title('Average Information Gain varies against different degree of freedom');
xlabel('Degree of Freedom');
ylabel('Information Gain');

plot(IG2_100/100);
xlabel('Degree of Freedom');
ylabel('Information Gain');

plot(IG3_100/100);
xlabel('Degree of Freedom');
ylabel('Information Gain');

plot(IG4_100/100);
xlabel('Degree of Freedom');
ylabel('Information Gain');

plot(IG5_100/100);
xlabel('Degree of Freedom');
ylabel('Information Gain');
legend('decision stump','2D linear', 'Conic section learner', 'Distance learner', '2-pixel learner')
hold off;

%Train A Decision Forest
testLabels = cell(3);
depthTrial = [6, 8, 10];
numTreeTrial = [10, 50, 100];
randomness = [10, 50, 100];
for k = 1:3
    for j = 1:3
        opts = struct;
        opts.depth = 5; 
        opts.numTrees= numTreeTrial(k); 
        opts.numSplits= randomness(j);  %Number of splits to try
        opts.classifierID= 1; % which split function to be used
        
        %Bagging
        Bagsize=95;
        bagged_data_train = cell(1,opts.numTrees);
        for i=1:opts.numTrees
            bagged_data_train{i} = datasample(data_train,Bagsize);
        end
        
        %Train 4 trees, use different bag for each tree
        treeModels = cell(1, opts.numTrees);
        for i = 1:opts.numTrees
            treeModels{i} = treeTrain(bagged_data_train{i}(:,[1,2]), bagged_data_train{i}(:,3), opts);
        end

        %Visualise parent node split
%         range = [-1.5, 1.5];
%         figure;
%         subplot(2,2,1);
        % %visualise axis-aligned learner
%         plot_toydata(bagged_data_train{1});
%         threshold = treeModels{1}.weakModels{1}.t;
%         if treeModels{1}.weakModels{1}.r == 1
%             plot([threshold threshold],[range(1),range(2)],'black');
%         else
%             plot([range(1),range(2)], [threshold threshold],'black');
%         end
%         title('Axis-aligned learner')
%         subplot(2,2,2);
%         hist(bagged_data_train{1}(:,3), 1:3);
%         title('Histogram of Training Set');
%         subplot(2,2,3);
%         hist(treeModels{1}.weakModels{1}.leftHistogram, 1:3);
%         title('Histogram of Left Child Node');
%         subplot(2,2,4);
%         hist(treeModels{1}.weakModels{1}.rightHistogram, 1:3);
%         title('Histogram of Right Child Node');

        %Visualize some leaf nodes
        % figure;
        % subplot(2,2,1);
        % bar(treeModels{1}.leafdist(13,:));
        % title('Visualization on Leaf Node (28th)');
        % subplot(2,2,2);
        % bar(treeModels{1}.leafdist(14,:));
        % title('Visualization on Leaf Node (29th)');
        % subplot(2,2,3);
        % bar(treeModels{1}.leafdist(15,:));
        % title('Visualization on Leaf Node (30th)');
        % subplot(2,2,4);
        % bar(treeModels{1}.leafdist(16,:));
        % title('Visualization on Leaf Node (31th)');


        %% Q2

        %Classification on Novel Test Points 
        test_point = [-.5 -.7; .4 .3; -.7 .4; .5 -.5];
        [testLabel, testProb] = forestTest(treeModels, test_point, opts);
        % figure;
        % hold on;
        % scatterTestData([test_point, testLabel], 'Novel');
        % plot_toydata(data_train);
        % hold off;
        % title('Novel Testing Points Classification Result');
        % figure;
        % subplot(2,2,1);
        % bar(testProb(1,:));
        % title('Class distribution of the 1st test data')
        % subplot(2,2,2);
        % bar(testProb(2,:));
        % title('Class distribution of the 2nd test data')
        % subplot(2,2,3);
        % bar(testProb(3,:));
        % title('Class distribution of the 3rd test data')
        % subplot(2,2,4);
        % bar(testProb(4,:));
        % title('Class distribution of the 4th test data')
        % figure;
        % bar(mean(testProb,1));
        % title('Average class distribution of the novel test data')


        %Classification on Dense Test Points
        [testLabel, testProb] = forestTest(treeModels, data_test(:,1:2), opts);
        testLabels{k,j} = testLabel;
%         figure;
%         hold on;
%         scatterTestData([data_test, testLabel], 'Dense');
%         plot_toydata(data_train);
%         title('Classification Result of Dense Data')
%         hold off;
%     end
% end

%Plots Changing paramters for decision forests
%VisualNumOfTreeVSDepth(testLabels, data_test, data_train);
%VisualRandomnessVSDepth(testLabels, data_test, data_train);
%VisualNumOfTreeVSRandomness(testLabels, data_test, data_train);