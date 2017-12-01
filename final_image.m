function [ new ] = final_image(image)

im=imgaussfilt(image,2);
for i=0:10:180
    se=strel('line',7,i);
    im=imerode(im,se);
end
im=imgaussfilt(im,3);
for i=0:10:180
    se=strel('line',5,i);
    im=imdilate(im,se);
end
im=imgaussfilt(im,2);
%kernel creation
for i=1:11
   for j=1:11
      mat(i,j)=20*(50-(i-6)^21-(j-6)^2);
   end
end

[rows,cols]=size(im);
new=zeros(size(im));
mxm=max(max(im));
% apply kernel
for i=6:rows-6
   for j=6:cols-6
        new(i-5:i+5,j-5:j+5)=new(i-5:i+5,j-5:j+5)+(mat.*double(im(i-5:i+5,j-5:j+5)));
   end
end
figure;
maxm=max(max(new));
new=255*new/maxm;
%threshold percentage th
th=40;
new(find(abs(new)<(th*255/100))) = 0; 
new=imgaussfilt(new,2);
imshow(uint8(new));
end

