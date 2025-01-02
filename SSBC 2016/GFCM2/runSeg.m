function Ir = runSeg (rgbImg)
%function runSeg
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


%E_1 (1)

%filename = 'E_1_ (4)';

%rgbImg = imread(filename,'jpg');


g=rgb2gray(rgbImg);

k=7;
[mu,mask]=fcmSeg(g,k);

Ir = maskmin( mask, k );
%Ir = maskmin( mask);


%imwrite( Ir, strcat('out_',filename,'.jpg') );










end

