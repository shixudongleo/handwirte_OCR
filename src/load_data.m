function [train_x train_y...
          train_m_x train_m_y...
          test_x test_y] = load_data()
% parameters:
% train_x   :   (n by d)  :  pixel feature samples for training
% train_y   :   (n by 1)  :  corresponding label for train_x
% train_m_x :   (n by d)  :  total 1000 pixel feature samples for training
% train_m_y :   (n by 1)  :  corresponding lablel for train_m_x
% test_x    :   (n by d)  :  pixel feature vector for testing
% test_y    :   (n by 1)  :  corresponding label for test_x

train_x = zeros(800*10, 28*28);
train_y = zeros(800*10, 1);

train_m_x = zeros(1000*10, 28*28);
train_m_y = zeros(1000*10, 1);

test_x = zeros(200*10, 28*28);
test_y = zeros(200*10, 1);


mnist_sub = load('mnist_sub.mat');
mnist_sub_more = load('mnist_sub_more.mat');

for ii = 0:9
    tmp = mnist_sub.(['train' num2str(ii)]);
    label_range = (1 + 800*ii):(800*(ii + 1));
    train_x(label_range, :) = tmp;
    train_y(label_range, :) = ii*ones(800, 1);
end

for ii = 0:9
    tmp = mnist_sub_more.(['train' num2str(ii)]);
    label_range = (1 + 1000*ii):(1000*(ii + 1));
    train_m_x(label_range, :) = tmp;
    train_m_y(label_range, :) = ii*ones(1000, 1);
end

for ii = 0:9
    tmp = mnist_sub_more.(['test', num2str(ii)]);
    label_range = (1 + 200*ii):(200*(ii + 1));
    test_x(label_range, :) = tmp;
    test_y(label_range, :) = ii*ones(200, 1);
end

clear mnist_sub;
clear mnist_sub_more;

end

