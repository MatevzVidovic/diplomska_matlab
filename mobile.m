%% Config
% Source directory for images
src_dir = '/home/matej/Downloads/dataset_example/Calvin/Xiaomi/Cropped/';

% Matching algorithm
%alg_dir = 'Miura';
alg_dir = 'B-COSFIRE';
old_dir = cd(alg_dir);

% Directory to save results to
save_dir = strcat(src_dir, '/', alg_dir, '/');
overwrite = true;

% Evaluate a single algorithm
evaluate_alg(@mobile_pred, src_dir, save_dir, overwrite);

cd(old_dir);

%% Evaluate a single algorithm
function evaluate_alg(alg, src_dir, save, overwrite)
	if ~exist(save, 'dir')
		mkdir(save);
	end
	
	set = imageSet(src_dir);
	k = 0;
	% Reduce number of workers below if too much memory is being used
	parfor (id_image = 1:set.Count, 4)
		% If it's a mask image, skip it
		[path, basename, ~] = fileparts(set.ImageLocation{id_image});
		if regexp(basename, '\d+[LR]_[lrsu]_\d+_')
			continue
		end

		% Evaluate a single image
		score = evaluate(path, basename, alg, save, overwrite);

		% If sclera mask was not found, skip
		if score == -1
			continue
		end

		% Otherwise save the result
		k = k + 1;
		%results(k) = score;
	end
	disp(['Found ' num2str(k) ' images.']);
	%results = results(1:k);
end

%% Load an image and its corresponding sclera prediction and vessels GT mask and evaluate the algorithm on it
function score = evaluate(path, basename, alg, save, overwrite)
	% Load base image
	image = im2double(imread(strcat(path, '/', basename, '.jpg')));
	s = size(image);
		
	if overwrite || ~isfile(strcat(save, basename, '.png'))
		% Run the matching algorithm
		prediction = alg(image);
		% Save prediction
		imwrite(prediction, strcat(save, basename, '.png'));
	end
	
	% TODO: Calculate and return error rate
	score = 0;
end
