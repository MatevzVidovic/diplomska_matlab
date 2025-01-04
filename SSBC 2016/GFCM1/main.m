overwrite = false;

rootdir = '/hdd/EyeZ/Segmentation/Sclera/Results/2020 SSBC/Group Evaluation/Images';
filelist = dir(fullfile(rootdir, '**/*.jpg'));

currdir = strsplit(mfilename('fullpath'), filesep);
alg = currdir{end-1};
savedir = fullfile('/hdd/EyeZ/Segmentation/Sclera/Results/2020 SSBC/Group Evaluation/Models', alg);
traindirs = ["All" "MASD+SBVPI" "MASD+SMD" "SBVPI" "SMD"];
typedirs = ["Binarised" "Predictions"];
[x, y] = meshgrid(traindirs, typedirs);
outdirs = [x(:) y(:)];

len = numel(filelist);
showTimeToCompletion;
startTime = tic;
dq = parallel.pool.DataQueue;
wb = waitbar(0, 'Segmenting images', 'Name', alg);
set(findall(wb, 'type', 'text'), 'Interpreter', 'none');
wb.UserData = [0 len];
afterEach(dq, @(fname) progress(wb, fname, startTime));

%for i = 1:len
parfor (i = 1:len, 14)
	file = filelist(i);

	[~, outname, ~] = fileparts(file.name);
	relpath = erase(file.folder, sprintf('%s/', rootdir));
	if contains(relpath, filesep)
		relpath = regexp(relpath, filesep, 'split', 'once');
		outfiles = compose(convertCharsToStrings(fullfile(savedir, '%s', relpath{1}, '%s', relpath{2}, '%s.png')), outdirs, outname)';
	else
		outfiles = compose(convertCharsToStrings(fullfile(savedir, '%s', relpath, '%s', '%s.png')), outdirs, outname)';
	end

	if ~overwrite
		allexist = true;
		for outfile = outfiles
			if ~isfile(outfile)
				allexist = false;
				break
			end
		end
		if allexist
			send(dq, file.name);
			continue
		end
	end

	img = imread(fullfile(file.folder, file.name));
	img = rgb2gray(img);

	k = 7;
	[~, mask] = fcmSeg(img, k);
	output = maskmin(mask);
	
	for outfile = outfiles
		currdir = fileparts(outfile);
		if ~isfolder(currdir)
			mkdir(currdir);
		end
		imwrite(output, outfile);
	end
	send(dq, file.name);
end

close(wb);

function progress(wb, fname, startTime)
	ud = wb.UserData;
	ud(1) = ud(1) + 1;
	completed = ud(1) / ud(2);
	waitbar(completed, wb, sprintf('Segmented %s', fname));
	showTimeToCompletion(completed, fname, [], [], startTime);
	wb.UserData = ud;
end