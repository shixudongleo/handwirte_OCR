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
disp(conf_mtx);


fprintf('\nThe overall error:%f\n', all_error);


end

