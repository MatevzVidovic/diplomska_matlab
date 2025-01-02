% Sclera Segmentation  of an eye image.
% Based on Modified Intutionistic Fuzzy Clustering method
% Author: 
% S V Aruna Kumar and B S Harish
% arunkumarsv55@gmail.com
% Department of Information Science and Engineering
% SJCE, Mysuru,Karnataka,India
% April 2017

%output file path %
output_filepath='H:\SSRBC2017_seg\IFS\';
ch='M';
in=1;
%Input file path%
directory = 'E:\DATA SET\Biometric\ISSBR2017\test sserbc 2017\Seg\' ;
d = dir(directory);
isub = [d(:).isdir];
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];
subfolder_length = size(nameFolds);


for leng=1:subfolder_length(1,1)
   nam = nameFolds{leng,1};
   folder1 = strcat(directory,nam);
   Path2= strcat(folder1,'\');
   Path3 = strcat(Path2,'*.jpg');
   srcFiles = dir(Path3); 
    for i = 4 : length(srcFiles)
         filename =srcFiles(i).name; 
         [len,col]=size(filename);
         output_filename=strcat(ch,filename(2:col));
         out_filename=strcat(output_filepath,output_filename);
         filename1=strcat(Path2,filename);
         Input_image=imread(filename1);
         
         %Intutionistic Fuzzy Clustering method%
         %Input is image file output is segmented image%
         [output]=IFS(Input_image);
         
         %Writing output image%
         imwrite(output,out_filename);
         in=in+1;
         if(in==18)
            disp(in);
         end    
    end
end

