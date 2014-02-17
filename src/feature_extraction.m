function [ fv_vectors labels ] = feature_extraction( x, y )
% extract all needing features from row data
% input:
% x         :   data from raw file
% y         :   classification label
%
% ouput:
% fv_vectors    :   make feature into n * d matrix
% labels        :   classification label

labels = y;
n = size(x, 1);
dim = size(x, 2);
%fv_vectors = zeros(n, 5);
for imnum = 1:n
   image = reshape(x(imnum, :), 28, 28);
   image = im2bw(image, 0.5);
   
   % region based features
   stats = regionprops(image, 'Area', 'Centroid', 'Orientation',...
                       'Perimeter');
  
   if length(stats) > 1
       areas = extractfield(stats, 'Area');
        [val index] = max(areas);
        stats = stats(index);       
   end
   
   fv = [stats.Area stats.Centroid stats.Orientation stats.Perimeter];
   %disp(fv);
   
   % gradient feature
   h_x = [1 0 -1;
          2 0 -2;
          1 0 -1];
   h_y = [1 2 1; 
          0 0 0;
          -1 -2 -1];
      
   g_x = imfilter(double(image), h_x, 'replicate');
   g_y = imfilter(double(image), h_y, 'replicate');
   
   gradient = sqrt(g_x.^2 + g_y.^2);
   gradient = reshape(gradient, 1, dim);
   
   tangent = atan(g_y ./ g_x);
   tangent = reshape(tangent, 1, dim);
   
   fv = [fv gradient tangent];
   fv_vectors(imnum, :) = fv;
end
   % normalization
   %fv_vectors = normc(fv_vectors);
%    avg = mean(fv_vectors, 1);
%    dev = std(fv_vectors, 0, 1);
%    cols = size(fv_vectors, 2);
%    for ii = 1:cols
%       fv_vectors(:, ii) = (fv_vectors(:, ii) - avg(ii))/dev(ii);
%    end
   
end

