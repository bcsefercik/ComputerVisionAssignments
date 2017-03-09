function [bboxes, confidences, image_ids] = ....
    run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   cs (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i.
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

%Created simple variables for convenience.
cs = feature_params.hog_cell_size;
num_temp_cells = feature_params.template_size / cs;

%Calculate dimension as in the previous parts for vl_hog.
dimension = num_temp_cells^2*31;

%Created a ratio array for multiple scale detections. 
%It creates logaritmically spaced array from 0.316x to 1.031x
ratios = logspace(-1.5,0.013,13);

confidence_threshold = 0.65;

for i = 1:length(test_scenes)
    
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    %Necessary initializations.
    cur_bboxes = zeros(0,4);
    cur_confidences = zeros(0,1);
    cur_image_ids = cell(0,1);
    
    
    for j = 1:length(ratios)
        
        %Resize image corresponding to current ratio.
        resized_img = imresize(img,ratios(j));
        
        hog = vl_hog(resized_img,cs);
        
        %Computing nuber of windows to examine.
        num_x = size(hog,2) - num_temp_cells + 1;
        num_y = size(hog,1) - num_temp_cells + 1;
        
        %Necessary initializations.
        traversed = zeros(num_x*num_y,dimension);
        traversed_x = zeros(num_x*num_y,1);
        traversed_y = zeros(num_x*num_y,1);
        for x = 1:num_x
            for y = 1:num_y
                t_index = num_y*(x-1)+y; 
                %Getting HoG of the windows according to its coordinates.
                %Sinc we calculated HoG for the whole image, we do not need
                %to calculate here.
                traversed(t_index,:) =  reshape(hog(y:(y + num_temp_cells - 1),x:(x + num_temp_cells - 1),:),1,dimension) ;
                traversed_x(t_index,1) = x;
                traversed_y(t_index,1) = y;
            end
        end
        
        %Evaluating the traversed windows with given weights.
        eval = (traversed * w) + b;
        %Findig the window indices whose evalution results are higher than
        %the chosen threshold.
        confident_indices = find(eval > confidence_threshold);
        
        if isempty(confident_indices),continue, end
        
        %Getting HoGs and coordinates of the confident windows.
        confident_hog = eval(confident_indices);
        confident_x = traversed_x(confident_indices);
        confident_y = traversed_y(confident_indices);
        
        %Finding the top left and bottom right corners of the windows.
        %Then, we are deviding each coordinate value with current ratio in
        %order to get the corresponding window coodinates 
        %on the original image.
        cur_x_min = (cs * (confident_x)) ./ratios(j);
        cur_y_min = (cs * (confident_y)) ./ratios(j);
        cur_x_max = (cs * (confident_x + num_temp_cells)) ./ratios(j);
        cur_y_max = (cs * (confident_y + num_temp_cells)) ./ratios(j);
        
        
        %For every confident, window we put image name to current image id
        %array.
        for a = 1:size(confident_indices,1)
            cur_image_ids = [cur_image_ids; {test_scenes(i).name}];
        end
        
        %Here, windows and confidences are added to related matrices. For
        %a little more accurracy chance I expand windows by 4 at each axis.
        cur_bboxes      = [cur_bboxes; [cur_x_min-2,cur_y_min-2,cur_x_max+2,cur_y_max+2]];
        cur_confidences = [cur_confidences; confident_hog];
    end
    
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    
    
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));
    
    
    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
    
    add_num = size(cur_confidences,1);
    bboxes(end+1:end+add_num,:)      = cur_bboxes;
    confidences(end+1:end+add_num,:) = cur_confidences;
    image_ids(end+1:end+add_num,:)   = cur_image_ids;
end
