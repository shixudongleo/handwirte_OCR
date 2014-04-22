%---------------------------------------------------
% author    : SHI XUDONG
% title     : OCR Main Code
% date      : 2014.02.06
%---------------------------------------------------

cd ../data;
addpath(pwd);
cd ../src;


% prepare data for training and testing
% parameters:
% train_x   :   (n by d)  :  pixel feature samples for training
% train_y   :   (n by 1)  :  corresponding label for train_x
% train_m_x :   (n by d)  :  total 1000 pixel feature samples for training
% train_m_y :   (n by 1)  :  corresponding lablel for train_m_x
% test_x    :   (n by d)  :  pixel feature vector for testing
% test_y    :   (n by 1)  :  corresponding ground truth label for test_x

fprintf('load raw pixel feature.\n');
[train_x train_y train_m_x train_m_y test_x test_y] = load_data();

fprintf('load gradient feature.\n');
[g_train_x g_train_y] = feature_extraction(train_x, train_y, 'train_fv.mat');
[g_train_m_x g_train_m_y] = feature_extraction(train_m_x, train_m_y, 'train_fv_m.mat');
[g_test_x g_test_y] = feature_extraction(test_x, test_y, 'test_fv.mat');


%%
% (1) Design a Bayes classifier using the training set, 
%     and evaluate its performance on the test set.
% using PCA to avoid singularity problem 
max_PCs = 200;
[T] = PCA(train_x, max_PCs);
PCA_train_x = train_x*T;
PCA_test_x = test_x*T;

% training
fprintf('\nUsing Bayes classifier.\n');
[Mu Sigma] = bayes_mv_train(PCA_train_x(:, 1:58), train_y);

% testing
fprintf('\nPredicate using Bayes classifier.\n');
y_bayes = bayes_mv_predicate(PCA_test_x(:, 1:58), Mu, Sigma);

% show statistics 
fprintf('\nUsing Bayes classifier. The Result:\n');
[conf_mtx all_error] = make_statistics(test_y, y_bayes);


% accuracy = zeros(1, max_PCs);
% for ii = 1:max_PCs
% % training
% fprintf('\nUsing Bayes classifier.\n');
% [Mu Sigma] = bayes_mv_train(PCA_train_x(:, 1:ii), train_y);
% 
% % testing
% fprintf('\nPredicate using Bayes classifier.\n');
% y_bayes = bayes_mv_predicate(PCA_test_x(:, 1:ii), Mu, Sigma);
% 
% % show statistics 
% fprintf('\nUsing Bayes classifier. The Result:\n');
% [conf_mtx all_error] = make_statistics(test_y, y_bayes);
% accuracy(ii) = all_error;
% end



%%
% (2) Design a k-nearest neighbor classifier, 
%     and evaluate its performance on the test set.
% training
fprintf('\nUsing KNN classifier.\n');
knn_model = ClassificationKNN.fit(train_x, train_y, 'NumNeighbors', 5);

% testing
fprintf('\nPredicate using KNN classifier.\n');
y_knn = predict(knn_model, test_x);

% show result
fprintf('\nUsing KNN classifier. The Result:\n');
[conf_mtx all_error] = make_statistics(test_y, y_knn);

% K = 50;
% accuracy = zeros(1, K);
% for ii = 1:K
% % training
% fprintf('\nUsing KNN classifier.\n');
% knn_model = ClassificationKNN.fit(train_x, train_y, 'NumNeighbors', ii);
% 
% % testing
% fprintf('\nPredicate using KNN classifier.\n');
% y_knn = predict(knn_model, test_x);
% 
% % show result
% fprintf('\nUsing KNN classifier. The Result:\n');
% [conf_mtx all_error] = make_statistics(test_y, y_knn);
% accuracy(ii) = all_error;
% end

%%
% (3) above two method using more training data
max_PCs = 58;
[T] = PCA(train_m_x, max_PCs);
PCA_train_m_x = train_m_x*T;
PCA_test_x = test_x*T;

% training
fprintf('\nUsing Bayes classifier.\n');
[Mu Sigma] = bayes_mv_train(PCA_train_m_x, train_m_y);

% testing
fprintf('\nPredicate using Bayes classifier.\n');
y_bayes = bayes_mv_predicate(PCA_test_x, Mu, Sigma);

% show statistics 
fprintf('\nUsing Bayes classifier. The Result:\n');
[conf_mtx all_error] = make_statistics(test_y, y_bayes);

fprintf('\nUsing KNN classifier.\n');
knn_model = ClassificationKNN.fit(train_m_x, train_m_y, 'NumNeighbors', 5);

% testing
fprintf('\nPredicate using KNN classifier.\n');
y_knn = predict(knn_model, test_x);

% show result
fprintf('\nUsing KNN classifier. The Result:\n');
[conf_mtx all_error] = make_statistics(test_y, y_knn);


%%
% (4) using 1000 data set, improved classifier using gradient feature
% guassian, KNN, PCA,  

% PCA feature extraction 
max_PCs = 62;
[T] = PCA(g_train_m_x, max_PCs);
PCA_train_m_x = g_train_m_x*T;
PCA_test_x = g_test_x*T;

% training
fprintf('\nUsing Bayes classifier.\n');
[Mu Sigma] = bayes_mv_train(PCA_train_m_x, train_m_y);

% testing
fprintf('\nPredicate using Bayes classifier.\n');
y_bayes = bayes_mv_predicate(PCA_test_x, Mu, Sigma);

% show statistics 
fprintf('\nUsing Bayes classifier. The Result:\n');
[conf_mtx all_error] = make_statistics(test_y, y_bayes);


max_PCs = 30;
[T] = PCA(g_train_m_x, max_PCs);
PCA_train_m_x = g_train_m_x*T;
PCA_test_x = g_test_x*T;

fprintf('\nUsing KNN classifier.\n');
knn_model = ClassificationKNN.fit(PCA_train_m_x, g_train_m_y, 'NumNeighbors', 3);

% testing
fprintf('\nPredicate using KNN classifier.\n');
y_knn = predict(knn_model, PCA_test_x);

% show result
fprintf('\nUsing KNN classifier. The Result:\n');
[conf_mtx all_error] = make_statistics(test_y, y_knn);

