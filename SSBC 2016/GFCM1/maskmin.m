function Ir = maskmin( mask)
%function Ir = maskmin( mask, k )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%mask is fcm out using mat2gray

[m,n]=size(mask);

ms=zeros(m,n);

%kk = ((k-2)/(k-1));

for i=1:m
   for j=1:n
       
       if mask(i,j)==1 %|| mask(i,j)== kk
           ms(i,j)=1;
       else
           ms(i,j)=0;
       end
          
    
    end

end

mss=mat2gray(ms);
%imshow(mss);


BW=im2bw(mss);

%figure, imshow(Ir);

BWW=BW;
CC1 = bwconncomp(BWW, 8);
%Compute the area of each component:
S = regionprops(CC1, 'Area');
%Remove small objects:
L = labelmatrix(CC1);
P= m*n/20 ;
BW2 = ismember(L, find([S.Area] >= P));

if (BW2==0)
    level = graythresh(mask);
    Ir=im2bw(mask, level);
    BW=Ir;
end



BW1 = BW;
CC = bwconncomp(BW);


numPixels = cellfun(@numel,CC.PixelIdxList);

[biggest,idx] = max(numPixels);

BW(CC.PixelIdxList{idx}) = 0;

%figure, imshow(BW);

%figure, imshow(BW1);

Ir = imsubtract(BW1,BW);

Ir = imfill(Ir,'holes');

end
