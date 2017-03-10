function VisualNumOfTreeVSDepth(testLabels, data_test)
    figure;
    subplot(3,3,1);
    hold on;
    scatterTestData([data_test, testLabels{1,1}], 'Dense');
    plot_toydata(data_train);
    title('NumOfTree=10, Depth=6')
    hold off;
    subplot(3,3,2);
    hold on;
    scatterTestData([data_test, testLabels{1,2}], 'Dense');
    plot_toydata(data_train);
    title('NumOfTree=50, Depth=6')
    hold off;
    subplot(3,3,3);
    hold on;
    scatterTestData([data_test, testLabels{1,3}], 'Dense');
    plot_toydata(data_train);
    title('NumOfTree=100, Depth=6')
    hold off;
    subplot(3,3,4);
    hold on;
    scatterTestData([data_test, testLabels{2,1}], 'Dense');
    plot_toydata(data_train);
    title('NumOfTree=10, Depth=8')
    hold off;
    subplot(3,3,5);
    hold on;
    scatterTestData([data_test, testLabels{2,2}], 'Dense');
    plot_toydata(data_train);
    title('NumOfTree=50, Depth=8')
    hold off;
    subplot(3,3,6);
    hold on;
    scatterTestData([data_test, testLabels{2,3}], 'Dense');
    plot_toydata(data_train);
    title('NumOfTree=100, Depth=8')
    hold off;
    subplot(3,3,7);
    hold on;
    scatterTestData([data_test, testLabels{3,1}], 'Dense');
    plot_toydata(data_train);
    title('NumOfTree=10, Depth=10')
    hold off;
    subplot(3,3,8);
    hold on;
    scatterTestData([data_test, testLabels{3,2}], 'Dense');
    plot_toydata(data_train);
    title('NumOfTree=50, Depth=10')
    hold off;
    subplot(3,3,9);
    hold on;
    scatterTestData([data_test, testLabels{3,3}], 'Dense');
    plot_toydata(data_train);
    title('NumOfTree=100, Depth=10')
    hold off
end