% Starter code prepared by James Hays
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path_pos'. Each face should be converted into a
% HoG template according to 'feature_params'. For improved performance, try
% mirroring or warping the positive training examples.

function features_pos = get_positive_features(train_path_pos, feature_params)
% 'train_path_pos' is a string. This directory contains 36x36 images of
%   faces
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.


% 'features_pos' is N by D matrix where N is the number of faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files); % N
dimension = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31; % D

%initialize with a proper size.
features_pos = zeros(num_images,dimension);

for i = 1:num_images
    %getting image from image file.
    image = imread([train_path_pos,'/',image_files(i).name]);
    %vl_hog takes a single precision image. We transform image to a single
    %precision one with im2single as in the http://www.vlfeat.org/matlab/vl_hog.html
    hog = vl_hog(im2single(image),feature_params.hog_cell_size);
    %vl_hog returns hog_cell_sizexhog_cell_sizex31 matrix by default.
    %In order to be able put this into features_pos, we need to reshape it
    %to 1xD vector.
    features_pos(i,:) = reshape(hog,1,dimension);
    
end