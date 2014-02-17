%---------------------------------------------------
% author    : SHI XUDONG
% title     : OCR Main Code
% date      : 2014.02.06
%---------------------------------------------------

cd ../data;
addpath(pwd);
cd ../src;


% (1) 
% load data and construct training and testing data
% parameters:
% train_x   :   (n by d)  :  pixel feature samples for training
% train_y   :   (n by 1)  :  corresponding label for train_x
% train_m_x :   (n by d)  :  total 1000 pixel feature samples for training
% train_m_y :   (n by 1)  :  corresponding lablel for train_m_x
% test_x    :   (n by d)  :  pixel feature vector for testing
% test_y    :   (n by 1)  :  corresponding ground truth label for test_x

RAW_PIXEL = 'pixel';
IMAGE_FEATURE = 'image feature';
feature = IMAGE_FEATURE;
switch feature 
    case RAW_PIXEL
        fprintf('\nUsing pixel raw feature.\n');
        [train_x train_y...
         train_m_x train_m_y...
         test_x test_y] = load_data();
     
    case IMAGE_FEATURE
        fprintf('\nUsing image related feature.\n');
        [train_x train_y...
         train_m_x train_m_y...
         test_x test_y] = load_data();
     
        [train_x train_y] = feature_extraction(train_x, train_y);
        [train_m_x train_m_y] = feature_extraction(train_m_x, train_m_y);
        [test_x test_y] = feature_extraction(test_x, test_y);
        
    otherwise
        fprintf('\nUnknown feature\n');
end


%(2)
% training model and get model parameters
% input     :   training data
% output    :   model paremeter

BAYES = 'bayes';
KNN = 'KNN';
PARZEN_WINDOW = 'parzen window';
classifier = KNN;

switch classifier
    case BAYES
        fprintf('\nUsing Bayes classifier.\n');
        [Mu Sigma] = bayes_mv_train(train_x, train_y);
        [Mu_m Sigma_m] = bayes_mv_train(train_m_x, train_m_y);
        
    case KNN
        fprintf('\nUsing KNN classifier.\n');
        knn_model = ClassificationKNN.fit(train_x, train_y);
        knn_m_model = ClassificationKNN.fit(train_m_x, train_m_y);
        
    case PARZEN_WINDOW
        fprintf('\nUsing parzen window density estimation classifier.\n');
        
    otherwise
        fprintf('\nUnknown classifier\n');
end




%(3)
% predicate using trained classifier
% input     :   test data
% ouput     :   predicate label for test data

switch classifier
    case BAYES
        fprintf('\nPredicate using Bayes classifier.\n');
        y_bayes = bayes_mv_predicate(test_x, Mu, Sigma);
        y_bayes_m = bayes_mv_predicate(test_x, Mu_m, Sigma_m);
        
    case KNN
        fprintf('\nPredicate using KNN classifier.\n');
        y_knn = predict(knn_model, test_x);
        y_knn_m = predict(knn_m_model, test_x);
        
    otherwise
        fprintf('\nUnknown classifier\n');
end



%(4)
% generate statistics for report
% conf_mtx  : confusion matrix
% all_error : overall error rate
switch classifier
    case BAYES
        fprintf('\nUsing Bayes classifier. The Result:\n');
        [conf_mtx all_error] = make_statistics(test_y, y_bayes);
        [conf_mtx_m all_error_m] = make_statistics(test_y, y_bayes_m);
        fprintf('\nImproved in error rate: %f\n', all_error - all_error_m);
        
    case KNN
        fprintf('\nUsing KNN classifier. The Result:\n');
        [conf_mtx all_error] = make_statistics(test_y, y_knn);
        [conf_mtx_m all_error_m] = make_statistics(test_y, y_knn_m);
        fprintf('\nImproved in error rate: %f\n', all_error - all_error_m);
        
    otherwise
        fprintf('\nUnknown classifier\n');
end







