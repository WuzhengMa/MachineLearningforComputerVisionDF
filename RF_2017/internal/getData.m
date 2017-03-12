function [ data_train, data_query ] = getData( MODE )
% Generate training and testing data

% Data Options:
%   1. Toy_Gaussian
%   2. Toy_Spiral
%   3. Toy_Circle
%   4. Caltech 101

showImg = 0;%1; % Show training & testing images and their image feature vector (histogram representation)
useKmeansCB=0; % Use kmeans codebook for 1 or RF codebook for 0

if ~useKmeansCB
    opts = struct;
    opts.depth = 9; 
    opts.numTrees= 10; 
    opts.numSplits= 5;  %Number of splits to try
    opts.classifierID= 1; % which split function to be used
end

PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}

switch MODE
    case 'Toy_Gaussian' % Gaussian distributed 2D points
        %rand('state', 0);
        %randn('state', 0);
        N= 150;
        D= 2;
        
        cov1 = randi(4);
        cov2 = randi(4);
        cov3 = randi(4);
        
        X1 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov1 0;0 cov1]);
        X2 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov2 0;0 cov2]);
        X3 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov3 0;0 cov3]);
        
        X= real([X1; X2; X3]);
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Spiral' % Spiral (from Karpathy's matlab toolbox)
        
        N= 50;
        t = linspace(0.5, 2*pi, N);
        x = t.*cos(t);
        y = t.*sin(t);
        
        t = linspace(0.5, 2*pi, N);
        x2 = t.*cos(t+2);
        y2 = t.*sin(t+2);
        
        t = linspace(0.5, 2*pi, N);
        x3 = t.*cos(t+4);
        y3 = t.*sin(t+4);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Circle' % Circle
        
        N= 50;
        t = linspace(0, 2*pi, N);
        r = 0.4
        x = r*cos(t);
        y = r*sin(t);
        
        r = 0.8
        t = linspace(0, 2*pi, N);
        x2 = r*cos(t);
        y2 = r*sin(t);
        
        r = 1.2;
        t = linspace(0, 2*pi, N);
        x3 = r*cos(t);
        y3 = r*sin(t);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Caltech' % Caltech dataset
        close all;
        imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        classList = {classList(3:end).name} % 10 classes
        
        disp('Loading training images...')
        % Load Images -> Description (Dense SIFT)
        cnt = 1;
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Training image samples');
        end
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx{c} = randperm(length(imgList));
            imgIdx_tr = imgIdx{c}(1:imgSel(1));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_tr)
                I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I); % PHOW work on gray scale image
                end
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [~, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
            
                %{
                % visualize one of the pic
                if c==9 && i==1
                    subplot(1,2,2);
                    imshow(I);
                    title('.. is misclassified into..');
                end
                %}
            end
        end
        
        %----------------------------------------------------------------------------
        % Kmeans codebook
        if useKmeansCB
            
            disp('Building visual codebook...')
            % Build visual vocabulary (codebook) for 'Bag-of-Words method'
            desc_sel = single(vl_colsubset(cat(2,desc_tr{:}), 10e4)); % Randomly select 100k SIFT descriptors for clustering
        
            % K-means clustering
            numBins = 1024; % for instance,


            % write your own codes here
            % ...
            %[idx,C] = kmeans(desc_sel',numBins,'Distance','sqeuclidean','MaxIter',200, 'Replicates',20);

            %save('KmeansClus2048.mat','idx','C');

            load('KmeansClus1024.mat');
        %----------------------------------------------------------------------------
        % RF codebook
        else
            disp('Building visual codebook...')
            % Build visual vocabulary (codebook) for 'Bag-of-Words method'
            %[desc_sel, desc_idx] = vl_colsubset(cat(2,desc_tr{:}), 10e4); % Randomly select 100k SIFT descriptors for clustering
            
            %Train several trees, use different bag for each tree
            RFCB_treeModels = cell(1, opts.numTrees);
            for i = 1:opts.numTrees
                desc_sel=[];
                desc_label=[];
                for cls = 1:length(classList)
                    for idx = 1:length(imgIdx_tr)
                        desc_sel = cat(2,desc_sel,single(vl_colsubset(cat(2,desc_tr{:}), 1000))); % Randomly select 100k SIFT descriptors for clustering
                        desc_label = cat(2, desc_label, ((cls-1)*length(imgIdx_tr)+idx)*ones(1, 1000));
                    end
                end
                RFCB_treeModels{i} = treeTrain(desc_sel', desc_label', opts);
            end
                        
        end
        %----------------------------------------------------------------------------
        disp('Encoding Images...')
        % Vector Quantisation
        
        % write your own codes here
        % ...
        %----------------------------------------------------------------------------
        % Kmeans codebook
        if useKmeansCB
            bagOfW=zeros(150, numBins);
            labelsDummy=ones(150,1);
            for imgClass=1:10
                for imgIndex=1:15

                      %Find the nearest neighbour to of the query image
                      nnCluster=knnsearch(C,single(desc_tr{imgClass,imgIndex}')); 

                      %Using BoW to represent the image
                      bagOfW((imgClass-1)*15+imgIndex,:)=hist(nnCluster,(1:numBins));

                      labelsDummy((imgClass-1)*15+imgIndex,:)=imgClass;
                end
            end
            %{
            % visualize one of the hist
            subplot(1,2,2);
            histogram(bagOfW(1,:));
            title('Bag of Word representation');
            %}
            data_train = [bagOfW labelsDummy];
        else
        %----------------------------------------------------------------------------
        % RFCB codebook
            bagOfW=zeros(150, 2^(opts.depth-1)*opts.numTrees);
            labelsDummy=ones(150,1);
            for imgClass=1:10
                for imgIndex=1:15

                  %Find the nearest neighbour to of the query image
                    for i = 1:opts.numTrees
                        [~, ~, leafIdx] = treeTest_rIdx(RFCB_treeModels{i}, single(desc_tr{imgClass,imgIndex}'), opts);
                        %Using BoW to represent the image
                        bagOfW((imgClass-1)*15+imgIndex,(i-1)*2^(opts.depth-1)+1:i*2^(opts.depth-1))=hist(leafIdx,(1:2^(opts.depth-1)));
                    end

                      labelsDummy((imgClass-1)*15+imgIndex,:)=imgClass;
                end
            end

            data_train = [bagOfW labelsDummy];
            
        end
        
        
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel
end

switch MODE
    case 'Caltech'
        if showImg
        figure('Units','normalized','Position',[.05 .1 .4 .9]);
        suptitle('Testing image samples');
        end
        disp('Processing testing images...');
        cnt = 1;
        % Load Images -> Description (Dense SIFT)
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_te)
                I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                %{
                % visualize one of the hist
                if c==3 && i==1
                    subplot(1,2,1);
                    imshow(I);
                    title('The Test Image');
                end
                %}
                
                if size(I,3) == 3
                    I = rgb2gray(I);
                end
                [~, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
            
            end
        end
        
                if showImg
                    suptitle('Testing image samples');
            figure('Units','normalized','Position',[.5 .1 .4 .9]);
        suptitle('Testing image representations: 256-D histograms');
        end

        % Quantisation
        
        % write your own codes here
        % ...
        %----------------------------------------------------------------------------
        % Kmeans codebook
        if useKmeansCB
            bagOfW_te=zeros(150, numBins);
            for imgClass=1:10
                for imgIndex=1:15

                      %Find the nearest neighbour to of the query image
                      nnCluster_te=knnsearch(C,single(desc_te{imgClass,imgIndex}')); 

                      %Using BoW to represent the image
                      bagOfW_te((imgClass-1)*15+imgIndex,:)=hist(nnCluster_te,(1:numBins));

                      %{
                      %Visualise training and testing images for selected class
                      %and index
                      if imgIdx<2 && (imgClass==3 || imgClass==4 )
                          figure('Position', [100, 100, 800, 300]);
                          subplot(1,2,1);
                          subFolderName = fullfile(folderName,classList{imgClass});
                          imgList = dir(fullfile(subFolderName,'*.jpg'));
                          I = imread(fullfile(subFolderName,imgList(imgIdx_tr(imgIdx)).name));
                          imshow(I);
                          title('Sample training image ');
                          subplot(1,2,2);
                          hist(nnCluster,[1:kclass]);
                          axis([0,kclass,0,inf]);
                          title('Training image histogram ');
                      end
                        %}
                end
            end
        else
        %----------------------------------------------------------------------------
        % RFCB codebook
            bagOfW_te=zeros(150, 2^(opts.depth-1)*opts.numTrees);
            for imgClass=1:10
                for imgIndex=1:15

                    for i = 1:opts.numTrees
                        [~, ~, leafIdx_te] = treeTest_rIdx(RFCB_treeModels{i}, single(desc_te{imgClass,imgIndex}'), opts);
                        %Using BoW to represent the image
                        bagOfW_te((imgClass-1)*15+imgIndex,(i-1)*2^(opts.depth-1)+1:i*2^(opts.depth-1))=hist(leafIdx_te,(1:2^(opts.depth-1)));
                    end
                end
            end
        end
        
        data_query = [bagOfW_te labelsDummy];
        
        
    otherwise % Dense point for 2D toy data
        xrange = [-1.5 1.5];
        yrange = [-1.5 1.5];
        inc = 0.02;
        [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        data_query = [x(:) y(:) zeros(length(x)^2,1)];
end
end

