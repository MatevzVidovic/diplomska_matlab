spath='C:\Users\129073\Desktop\sserbc\reco\sclera';
%spath1='C:\Users\129073\Desktop\MASD';
dpath='C:\Users\129073\Desktop\sserbc\reco\sclera1';
Name=dir(spath);
for i=3:length(Name)
    name=dir(strcat(spath,'\',Name(i).name));
    %name1=dir(strcat(spath1,'\',Name(i).name));%%
    for j=3:length(name)
        mkdir(strcat(dpath,'\',Name(i).name));
        I=imread(strcat(spath,'\',Name(i).name,'\',name(j).name));
        %I=imcomplement(im2bw(I,graythresh(I)-0.005));
        %I1=imread(strcat(spath1,'\',Name(i).name,'\',name1(j).name),'.JPG');
%         I1=imdilate(I1,strel('line',30,30));
%         I=(I & I1);
%        for k=1:size(I,1)
%            for l=1:size(I,2)
%            if(I(k,l)==0)
%                I1(k,l,1:3)=0;
%            end
%            end
%        end
        %I=imcomplement(im2bw(I(:,:,2)*255));
        imwrite(I,strcat(dpath,'\',Name(i).name,'\',name(j).name,'.jpg'))
        %system(strcat('java -jar segmentSclera.jar C:\Users\129073\Desktop\MASD\',Name(i).name,'C:\Users\129073\Desktop\sserbc\seg\sumata\output\',Name(i).name));
    end
end
