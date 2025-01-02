function prediction = my_pred(image, sclera, cfg)
	image = segment(image, sclera);

	if nargin > 2
		if cfg.normalise
			image = (image - min(image(:))) / (max(image(:)) - min(image(:)));
		end
		preprocessthresh = cfg.preprocessthresh;
		symmfilter = cfg.symmfilter;
		asymmfilter = cfg.asymmfilter;
	else
		preprocessthresh = 0.5;
		
		symmfilter = struct();
		symmfilter.sigma = 2.4;		% original
		symmfilter.sigma = 2.7;		% SBVPI
		%symmfilter.sigma = 4.8;	% MOBIUS
		symmfilter.len = 8;			% original
		symmfilter.len = 12;		% SBVPI
		%symmfilter.len = 18;		% MOBIUS
		symmfilter.sigma0 = 1;		% original/SBVPI
		%symmfilter.sigma0 = 3;		% MOBIUS
		symmfilter.alpha = 0.7;		% original
		symmfilter.alpha = 0.6;		% SBVPI
		%symmfilter.alpha = 0.2;	% MOBIUS

		asymmfilter = struct();
		asymmfilter.sigma = 1.8;	% original
		asymmfilter.sigma = 2.1;	% SBVPI
		%asymmfilter.sigma = 4.3;	% MOBIUS
		asymmfilter.len = 24;		% original/SBVPI
		%asymmfilter.len = 34;		% MOBIUS
		asymmfilter.sigma0 = 1;		% original/SBVPI/MOBIUS
		asymmfilter.alpha = 0.1;	% original/SBVPI/MOBIUS
	end

	%% Filters responses
	% Tresholds values
	% DRIVE -> preprocessthresh = 0.5, thresh = 37
	% STARE -> preprocessthresh = 0.5, thresh = 40
	% CHASE_DB1 -> preprocessthresh = 0.1, thresh = 38
	prediction = BCOSFIRE_media15(image, symmfilter, asymmfilter, preprocessthresh);
	prediction = (prediction - min(prediction(:))) / (max(prediction(:)) - min(prediction(:)));

	%output.segmented = (output.respimage > 37);

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
