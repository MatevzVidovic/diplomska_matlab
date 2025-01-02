% Sclera Segmentation  of an eye image.
% Based on Robust Spatial Kenrel Fuzzy c-means Clustering
% Author: 
% S V Aruna Kumar and B S Harish
% arunkumarsv55@gmail.com
% Department of Information Science and Engineering
% SJCE, Mysuru,Karnataka,India
% Jan 2016

%output file path %
output_filepath='/hdd/EyeZ/Segmentation/Sclera/Results/2020 SSBC/Group Evaluation/Models/RSKFCM/';
ch='M';

%Input file path%
d = dir('/hdd/EyeZ/Segmentation/Sclera/Datasets/MOBIUS/Images/');
isub = [d(:).isdir];
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];
subfolder_length = size(nameFolds);
directory = '/hdd/EyeZ/Segmentation/Sclera/Datasets/MOBIUS/Images/' ;

for leng=1:subfolder_length(1,1)
   nam = nameFolds{leng,1};
   folder1 = strcat(directory,nam);
   Path2= strcat(folder1,'\');
   Path3 = strcat(Path2,'*.jpg');
   srcFiles = dir(Path3); 
    for i = 1 : length(srcFiles)
         filename =srcFiles(i).name; 
         [len,col]=size(filename);
         output_filename=strcat(ch,filename(2:col));
         out_filename=strcat(output_filepath,output_filename);
         filename1=strcat(Path2,filename);
         Input_image=imread(filename1);
         
         %Robust Spatial Kernel FCM%
         %Input is image file output is segmented image%
         [output]=RSKFCM(Input_image);
         
         %Writing output image%
         imwrite(output,out_filename);
    end
end