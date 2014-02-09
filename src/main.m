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

[train_x train_y...
 train_m_x train_m_y...
 test_x test_y] = load_data();



%(2)
% training model and get model parameters
% input     :   training data
% output    :   model paremeter

BAYES = 'bayes';
KNN = 'KNN';
classifier = BAYES;

switch classifier
    case BAYES
        fprintf('\nUsing Bayes classifier.\n');
        [Mu Sigma] = bayes_mv_train(train_x, train_y);
        [Mu_m Sigma_m] = bayes_mv_train(train_m_x, train_m_y);
        
    case KNN
        fprintf('\nUsing KNN classifier.\n');
        
    otherwise
        fprintf('\nUnknown classifier\n');s
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
        
    otherwise
        fprintf('\nUnknown classifier\n');s
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
        fprintf('\nImproved in error rate: %f', all_error - all_error_m);
        
    case KNN
        fprintf('\nUsing KNN classifier. The Result:\n');
        
    otherwise
        fprintf('\nUnknown classifier\n');s
end







