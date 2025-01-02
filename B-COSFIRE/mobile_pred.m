function prediction = mobile_pred(image)

	% Example with an image from DRIVE data set
	%image = double(imread('./data/Retina_example/test/images/01_test.tif')) ./ 255;
	
	%% Symmetric filter params
	symmfilter = struct();
	%symmfilter.sigma     = 2.4;
	symmfilter.sigma = 2.7;
	%symmfilter.len       = 8;
	symmfilter.len = 12;
	symmfilter.sigma0    = 1;
	%symmfilter.alpha     = 0.7;
	symmfilter.alpha = 0.6;

	%% Asymmetric filter params
	asymmfilter = struct();
	%asymmfilter.sigma     = 1.8;
	asymmfilter.sigma = 2.4;
	%asymmfilter.len       = 22;
	asymmfilter.len = 22;
	asymmfilter.sigma0    = 1;
	asymmfilter.alpha     = 0.1;

	%% Filters responses
	% Tresholds values
	% DRIVE -> preprocessthresh = 0.5, thresh = 37
	% STARE -> preprocessthresh = 0.5, thresh = 40
	% CHASE_DB1 -> preprocessthresh = 0.1, thresh = 38
	prediction = BCOSFIRE_media15(image, symmfilter, asymmfilter, 0.5);
	prediction = (prediction - min(prediction(:))) / (max(prediction(:)) - min(prediction(:)));
	
	%output.segmented = (output.respimage > 37);

end