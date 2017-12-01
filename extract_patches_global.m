files=dir('images\train\');
files=files(3:end);
fix=dir('fixations\train\');
fix=fix(3:end);
map=dir('maps\train\');
map=map(3:end);
sal_vals=[];
t=1;
rr=0.463241614681625;
gg=0.430633775453174;
bb=0.389898652047589;

for i=1:length(files)
    name=files(i).name;
    name=name(1:length(name)-4);
    im=im2double(imread([files(i).folder '\' name '.jpg']));
        [a b c]=size(im);
        if(c~=3)
        im=cat(3,im,im,im);
        end
     A=im;
        [L N]=(superpixels((A),200));
outputImage = zeros(size(A),'like',A);
idx = label2idx(L);
numRows = size(A,1);
numCols = size(A,2);
for labelVal = 1:N
redIdx = idx{labelVal};
greenIdx = idx{labelVal}+numRows*numCols;
blueIdx = idx{labelVal}+2*numRows*numCols;
outputImage(redIdx) = mean(A(redIdx));
outputImage(greenIdx) = mean(A(greenIdx));
outputImage(blueIdx) = mean(A(blueIdx));
end
im=outputImage;
%    im=rgb2hsv(im);
    map_im=im2double(imread([map(i).folder '\' name '.png']));
    nosp=length(find(map_im>0.1));
    fix1=(load(['fixations\train\' name '.mat']));
    salient=fix1.gaze.fixations;
    if(a~=480 && b~=640)
        continue;
        disp('size not compatible');
    end
        
    for j=1:size(salient,1)
%         r=mean(mean(im(:,:,1)));
%         g=mean(mean(im(:,:,2)));
%         b=mean(mean(im(:,:,3)));
        final_img=double(zeros(2*size(im,1)+1,2*size(im,2)+1,3));
        cx=ceil(size(final_img,1)/2);cy=ceil(size(final_img,2)/2);
        final_img(:,:,1)=rr;
        final_img(:,:,2)=gg;
        final_img(:,:,3)=bb;
        final_img(cx-salient(j,2)+1:cx+(-salient(j,2)+size(im,1)),cy-salient(j,1)+1:cy+(-salient(j,1)+size(im,2)),:)=im;
        if(-25+salient(j,2)<1 && salient(j,2)+25<=a)
        %patch=im(1:salient(j,2)+25,:,:);
        map_patch=map_im(1:salient(j,2)+25,:);
        elseif(-25+salient(j,2)>=1 && salient(j,2)+25>a)
        %patch=im(salient(j,2)-25:end,:,:);
                map_patch=map_im(salient(j,2)-25:end,:,:);
        else
        %     patch=im(salient(j,2)-25:salient(j,2)+25,:,:);
                map_patch=map_im(salient(j,2)-25:salient(j,2)+25,:);
        end
        if(-25+salient(j,1)<1 && salient(j,1)+25<=b)
        %patch=patch(:,1:salient(j,1)+25,:);
        map_patch=map_patch(:,1:salient(j,1)+25);
        elseif(-25+salient(j,1)>=1 && salient(j,1)+25>b)
        %patch=patch(:,salient(j,1)-25:end,:);
        map_patch=map_patch(:,salient(j,1)-25:end);
        else
        %     patch=patch(:,salient(j,1)-25:salient(j,1)+25,:);
             map_patch=map_patch(:,salient(j,1)-25:salient(j,1)+25,:);
        end
        saliency_val=map_im(salient(j,2),salient(j,1));
        
        if(saliency_val<=0.1 && length(find(map_patch<=0.1))>=0.7*size(map_patch,1)*size(map_patch,2))
             final_img=imresize(final_img,[227 227]);
        sal_vals=[sal_vals;saliency_val];
%                 imshow(map_patch);
     
        imwrite(final_img,['F:\Project\Dataset\global\salient_train\' int2str(t) '.jpg']);
        t=t+1;
            continue;
        elseif(saliency_val>0.1 && length(find(map_patch>0.1))>=0.7*size(map_patch,1)*size(map_patch,2))
             final_img=imresize(final_img,[227 227]);
        sal_vals=[sal_vals;saliency_val];
        
        imwrite(final_img,['F:\Project\Dataset\global\salient_train\' int2str(t) '.jpg']);
        t=t+1;
        end
    end
end
sal_vals1=[];
for i=1:length(files)
    name=files(i).name;
    name=name(1:length(name)-4);
    im=im2double(imread([files(i).folder '\' name '.jpg']));
        [a b c]=size(im);
        if(c~=3)
        im=cat(3,im,im,im);
        end
     A=im;
        [L N]=(superpixels((A),200));
outputImage = zeros(size(A),'like',A);
idx = label2idx(L);
numRows = size(A,1);
numCols = size(A,2);
for labelVal = 1:N
redIdx = idx{labelVal};
greenIdx = idx{labelVal}+numRows*numCols;
blueIdx = idx{labelVal}+2*numRows*numCols;
outputImage(redIdx) = mean(A(redIdx));
outputImage(greenIdx) = mean(A(greenIdx));
outputImage(blueIdx) = mean(A(blueIdx));
end
im=outputImage;

    map_im=im2double(imread([map(i).folder '\' name '.png']));
    mp2=map_im(81:400,101:540);
    [B I]=sort(find(mp2<=0.1));
    for j=1:10
       [y x]=ind2sub(size(mp2),B(j));
       y=y+80;x=x+100;
       patch=im(y-25:y+25,:,:);
       map_patch=map_im(y-25:y+25,:);
       patch=patch(:,x-25:x+25,:);
       map_patch=map_patch(:,x-25:x+25,:);
        saliency_val=map_im(y,x);
        final_img=double(zeros(2*size(im,1)+1,2*size(im,2)+1,3));
        cx=ceil(size(final_img,1)/2);cy=ceil(size(final_img,2)/2);
        final_img(:,:,1)=rr;
        final_img(:,:,2)=gg;
        final_img(:,:,3)=bb;
        final_img(cx-y+1:cx+(-y+size(im,1)),cy-x+1:cy+(-x+size(im,2)),:)=im;    
        patch=imresize(final_img,[227 227]);
        if(saliency_val<=0.1 && length(find(map_patch<=0.1))>=0.7*size(map_patch,1)*size(map_patch,2))
        sal_vals1=[sal_vals1;saliency_val];
        imwrite(patch,['F:\Project\Dataset\global\non_salient_extras\' int2str(t) '.jpg']);
        t=t+1;
      
        end
    end
end

