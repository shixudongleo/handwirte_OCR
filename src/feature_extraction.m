function [ fv_vectors labels ] = feature_extraction( x, y, file_name )
% extract all needing features from row data
% input:
% x         :   data from raw file
% y         :   classification label
%
% ouput:
% fv_vectors    :   make feature into n * d matrix
% labels        :   classification label
if exist(file_name)
    load(file_name);
    labels = y;
else
    labels = y;
    n = size(x, 1);
    dim = size(x, 2);
    %fv_vectors = zeros(n, 5);
    for imnum = 1:n
       image = reshape(x(imnum, :), 28, 28);
       image = im2bw(image, 0.5);

       % region feature
       region_stats = get_region_fv(image);
       % gradient feature
       gradient = get_gradient_fv(image);

       fv = [region_stats gradient];
       fv_vectors(imnum, :) = fv;
    end

    % normalization
    %fv_vectors = normalize(fv_vectors);
    save(file_name, 'fv_vectors');
end



end

function [region_fv] = get_region_fv(image)
% return 1 by d region feature in image
area = bwarea(image);
perim = bwperim(image, 8);
perim = sum(sum(perim));
[x y] = find(image);
centroid = [mean(x) mean(y)];

region_fv = [area perim centroid];
end

function [gradient_fv] = get_gradient_fv(image)
% return 1 by d gradient feature in image
[rows cols] = size(image);
dim = rows*cols;

% sobel operator
h_x = [1  0  -1;
       2  0  -2;
       1  0  -1];
   
h_y = [1  2  1; 
       0  0  0;
      -1 -2 -1];
  
g_x = imfilter(double(image), h_x, 'replicate');
g_y = imfilter(double(image), h_y, 'replicate');

g_x = reshape(g_x, 1, dim);
g_y = reshape(g_y, 1, dim);
gradient_fv = sqrt(g_x.^2 + g_y.^2);

% can also return g_x g_y directly for more dimensional data
% gradient_fv = [gx_ g_y];

end


function [hist_fv] = get_hist_fv(image)
% return 1 by d histogram based feature in image

end


function [mat_out] = normalize(mat_in)
% have difficulty in costant values
% scale to [0 1]
% (x - min)/(max - min)
rows = size(mat_in, 1);
min_vals = min(mat_in);
max_vals = max(mat_in);

diff = max_vals - min_vals;

min_vals = repmat(min_vals, rows, 1);
diff = repmat(diff, rows, 1);

mat_out = (mat_in - min_vals)./diff;

% zero-mean unit variance the std may be zero
% rows = size(mat_in, 1);
% avg = mean(mat_in, 1);
% avg = repmat(avg, rows, 1);
% 
% dev = std(mat_in, 0, 1);
% dev = repmat(dev, rows, 1);
% 
% mat_out = (mat_in - avg)./ dev;
end