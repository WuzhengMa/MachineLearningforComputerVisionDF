function  scatterTestData(testData, dataType)
if dataType == 'Novel'
    plot(testData(testData(:,end)==1,1), testData(testData(:,end)==1,2), 'd', 'MarkerFaceColor', [.9 .5 .5], 'MarkerEdgeColor','k','MarkerSize', 15);
    plot(testData(testData(:,end)==2,1), testData(testData(:,end)==2,2), 'd', 'MarkerFaceColor', [.5 .9 .5], 'MarkerEdgeColor','k','MarkerSize', 15);
    plot(testData(testData(:,end)==3,1), testData(testData(:,end)==3,2), 'd', 'MarkerFaceColor', [.5 .5 .9], 'MarkerEdgeColor','k','MarkerSize', 15);
else
    plot(testData(testData(:,end)==1,1), testData(testData(:,end)==1,2), 'o', 'MarkerFaceColor', [.7 .1 .1], 'MarkerEdgeColor','k');
    plot(testData(testData(:,end)==2,1), testData(testData(:,end)==2,2), 'o', 'MarkerFaceColor', [.1 .7 .1], 'MarkerEdgeColor','k');
    plot(testData(testData(:,end)==3,1), testData(testData(:,end)==3,2), 'o', 'MarkerFaceColor', [.1 .1 .7], 'MarkerEdgeColor','k');
end
end