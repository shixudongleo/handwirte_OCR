function [ label_y ] = bayes_mv_predicate( test_x, Mu, Sigma )
% predicate label using model paremeter
% input:
% test_x    :   testing data
% Mu        :   mean for gaussian model
% Sigma     :   Sigma for gaussian model
%
% ouput:
% label_y   :   predicate result

label = 0:9;
n = size(test_x);
prediction = zeros(n, length(label));

for ii = label
    pre_tmp = mvnpdf(test_x,...
                     Mu(ii + 1, :), diag(Sigma(ii + 1, :)));
    prediction(:, ii + 1) = pre_tmp;
end

[val index] = max(prediction, [], 2);

label_y = index - 1;

end

