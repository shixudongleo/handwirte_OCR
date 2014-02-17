function [ Mu Sigma ] = bayes_mv_train(x, y)
% bayes_mv function is model classifier using multivarite Gaussian
% input:
% x     :   training feature
% y     :   traing label
%
% ouput:
% Mu    :   mean for gaussian 
% Sigma :   Sigma for gaussian

% PCA of training data
[coff pca] = princomp(x);

label = 0:9;
dim = 300;

Mu = zeros(length(label), dim);
Sigma = ones(length(label), dim);


for ii = label
    index = find(y == ii);
    train_x = pca(index, 1:dim);
    n = size(train_x, 1);
    gaussian = gmdistribution.fit(train_x, 1,'CovType', 'diagonal',...
                                  'Regularize', 0.001);
                              
    Mu(ii + 1, :) = gaussian.mu(1, :);
    Sigma(ii + 1, :, :) = gaussian.Sigma(:, :, 1);   
end


end

