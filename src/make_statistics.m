function [ conf_mtx all_error ] = make_statistics( gt, predicated )
% using ground truth and model predication result to report accuracy. 
% input:
% gt        :   ground truth label
% predicated:   model predicate result
% 
% output:
% conf_mtx  : confusion matrix
% all_error : overall error rate



% http://www.mathworks.com/help/stats/confusionmat.html
fprintf('\nConfusion Matrix:\n');
conf_mtx = confusionmat(gt, predicated);
disp(conf_mtx);

n = length(gt);
true_pre = sum(diag(conf_mtx));
all_error = 1 - true_pre/n;
fprintf('\nThe overall error:%f\n', all_error);

end

