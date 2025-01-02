function main(dir, save_dir, alg, overwrite)
	%% Config
	% Source directory for images
	if nargin < 1 || strcmp(dir, '')
		%dir = '/hdd/EyeZ/Segmentation/SIP/Datasets/MOBIUS';
		%dir = '/hdd/EyeZ/Segmentation/Vessels/Datasets/MOBIUS (param_search)';
		dir = '/hdd/EyeZ/Segmentation/Vessels/Datasets/SBVPI';
	end
	global src_dir;
	global mask_dir;
	src_dir = strcat(dir, '/', 'Images');
	mask_dir = strcat(dir, '/', 'Masks');

	% Segmentation algorithm
	if nargin < 3 || strcmp(alg, '')
		alg = 'B-COSFIRE';
	end
	old_dir = cd(alg);
	cleaner = onCleanup(@() cd(old_dir));

	% Directory to save results to
	if nargin < 2 || strcmp(save_dir, '')
		save_dir = strcat('/hdd/EyeZ/Segmentation/Vessels/Results/MOBIUS/', alg);
	end

	if nargin < 4
		overwrite = true;
	else
		overwrite = get_bool(overwrite);
	end

	%% Run
	if strcmpi(alg, 'miura')
		% Evaluate Miura algorithms
		for alg_name = {'MC', 'RLTGS'}
			evaluate_alg_pair(alg_name{1}, save_dir);
		end
	else
		% Evaluate a single algorithm
		evaluate_alg(@my_pred, strcat(save_dir, '/'), overwrite);

		% Param search
		%param_search(@my_pred, @random_bcosfire, @write_bcosfire, strcat(save_dir, '/'), 500);
	end
end

%% Translate passed argument into boolean value
function bool = get_bool(arg)
	if isstring(arg) || ischar(arg)
		bool = strcmpi(arg, 'true');
		arg = str2double(arg);
		if ~isnan(arg)
			bool = arg ~= 0;
		end
	elseif isnumeric(arg)
		bool = arg ~= 0;
	else
		bool = arg;
	end
end

%% Repeatedly randomly sample hyperparameters and run the algorithm
function param_search(alg, randomizer, writer, save_dir, steps)
	if nargin < 5
		steps = 100;
	end

	for i = 1:steps
		save = strcat(save_dir, num2str(i), '/');
		if ~exist(save, 'dir')
			mkdir(save);
		end

		cfg = randomizer();
		writer(cfg, strcat(save, 'params.txt'));

		evaluate_alg(alg, save, true, cfg.binthresh);
	end
end

%% Evaluate binarised and normalised algorithm version
function evaluate_alg_pair(alg_name, save_dir)
	save = strcat(save_dir, '_', alg_name, '/');
	alg = @(image, sclera) my_pred(image, sclera, alg_name);
	evaluate_alg(alg, save);

	save = strcat(save_dir, '_', alg_name, '_norm/');
	alg = @(image, sclera) my_pred(image, sclera, alg_name, 'norm');
	evaluate_alg(alg, save);
end

%% Evaluate a single algorithm
function evaluate_alg(alg, save, overwrite, binthresh)
	global src_dir;
	global mask_dir;
	m_dir = mask_dir;

	if ~exist(save, 'dir')
		mkdir(save);
	end

	if nargin < 4
		binthresh = 0.0;
	end

	set = imageSet(src_dir);
	k = 0;
	% Reduce number of workers below if too much memory is being used
	parfor (id_image = 1:set.Count, 1)
		% If it's a mask image, skip it
		[path, basename, ~] = fileparts(set.ImageLocation{id_image});
		if regexp(basename, '\d+[LR]_[lrsu]_\d+_')
			continue
		end

		% Evaluate a single image
		score = evaluate(path, m_dir, basename, alg, save, overwrite, binthresh);

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
function score = evaluate(path, mask_dir, basename, alg, save, overwrite, binthresh)
	sclera_file = strcat(mask_dir, '/', basename, '_sclera.png');
	vessels_file = strcat(mask_dir, '/', basename, '_vessels.png');

	disp(basename);
	if ~isfile(sclera_file)
		fprintf('404: Sclera not found: %s\n', sclera_file);
		score = -1;
		return
	end
	vessels_exist = isfile(vessels_file);

	% Load base image
	image = im2double(imread(strcat(path, '/', basename, '.jpg')));
	s = size(image);

	% Load masks
	sclera = imread(sclera_file);
	if ~ismatrix(sclera)
		sclera = rgb2gray(sclera);
	end
	sclera = im2double(sclera);
	if vessels_exist
		vessels = im2double(rgb2gray(imread(vessels_file)));
	end

	if overwrite || ~isfile(strcat(save, basename, '.png'))
		% Run the matching algorithm
		prediction = alg(image, sclera);
		% Save prediction
		imwrite(prediction, strcat(save, basename, '.png'));
		if nargin >= 7
			% Save binarised prediction
			imwrite(prediction >= binthresh, strcat(save, basename, '_bin.png'));
		end
	end
	if vessels_exist
		% Save GT mask
		imwrite(vessels, strcat(save, basename, '_gt.png'));
	end

	% TODO: Calculate and return error rate
	score = 0;
end

%% Segment the part of the RGB image where grayscale mask has a value > threshold
function image = segment(image, mask, threshold)
	if nargin < 3
		threshold = 0.5;
	end
	mask = mask > threshold;
	mask = cat(3, mask, mask, mask);
	image = image .* mask;
end

function r = rnd(min, max)
	r = rand() * (max - min) + min;
end

function cfg = random_bcosfire()
	cfg.normalise = rand() >= 0.5;
	cfg.preprocessthresh = rnd(0, 0.5);
	cfg.binthresh = rnd(0, 0.5);

	cfg.symmfilter = struct();
	cfg.symmfilter.sigma = rnd(1.5, 6);
	cfg.symmfilter.len = 2 * randi([4 24]);
	cfg.symmfilter.sigma0 = rnd(1, 4);
	cfg.symmfilter.alpha = rnd(0, 1);

	cfg.asymmfilter = struct();
	cfg.asymmfilter.sigma = rnd(1.5, 6);
	cfg.asymmfilter.len = 2 * randi([4 24]);
	cfg.asymmfilter.sigma0 = rnd(1, 3);
	cfg.asymmfilter.alpha = rnd(0, 1);
end

function write_bcosfire(cfg, f)
	dlmwrite(f, cfg.normalise);
	dlmwrite(f, cfg.preprocessthresh, '-append');
	dlmwrite(f, cfg.binthresh, '-append');
	writetable(struct2table(cfg.symmfilter), insertAfter(f, 'params', '_sym'));
	writetable(struct2table(cfg.asymmfilter), insertAfter(f, 'params', '_asym'));
end
