files=dir('images\train\');
files=files(3:end);
fix=dir('fixations\train\');
fix=fix(3:end);
map=dir('maps\train\');
map=map(3:end);
sal_vals=[];
t=1;
for i=1:length(files)
    name=files(i).name;
    name=name(1:length(name)-4);
    im=im2double(imread([files(i).folder '\' name '.jpg']));
        [a b c]=size(im);
        if(c~=3)
        im=cat(3,im,im,im);
        end
    im=rgb2hsv(im);
    map_im=im2double(imread([map(i).folder '\' name '.png']));
    nosp=length(find(map_im>0.1));
    fix1=(load(['fixations\train\' name '.mat']));
    salient=fix1.gaze.fixations;
    if(a~=480 && b~=640)
        continue;
        disp('size not compatible');
    end
        
    for j=1:size(salient,1)
        if(-25+salient(j,2)<1 && salient(j,2)+25<=a)
        patch=im(1:salient(j,2)+25,:,:);
        map_patch=map_im(1:salient(j,2)+25,:);
        elseif(-25+salient(j,2)>=1 && salient(j,2)+25>a)
        patch=im(salient(j,2)-25:end,:,:);
                map_patch=map_im(salient(j,2)-25:end,:,:);
        else
             patch=im(salient(j,2)-25:salient(j,2)+25,:,:);
                map_patch=map_im(salient(j,2)-25:salient(j,2)+25,:);
        end
        if(-25+salient(j,1)<1 && salient(j,1)+25<=b)
        patch=patch(:,1:salient(j,1)+25,:);
        map_patch=map_patch(:,1:salient(j,1)+25);
        elseif(-25+salient(j,1)>=1 && salient(j,1)+25>b)
        patch=patch(:,salient(j,1)-25:end,:);
        map_patch=map_patch(:,salient(j,1)-25:end);
        else
             patch=patch(:,salient(j,1)-25:salient(j,1)+25,:);
             map_patch=map_patch(:,salient(j,1)-25:salient(j,1)+25,:);
        end
        saliency_val=map_im(salient(j,2),salient(j,1));
        
        if(saliency_val<=0.1 && length(find(map_patch<=0.1))>=0.7*size(map_patch,1)*size(map_patch,2))
             patch=imresize(patch,[51 51]);
        sal_vals=[sal_vals;saliency_val];
     
        imwrite(patch,['F:\Project\Dataset\local\salient_train\' int2str(t) '.jpg']);
        t=t+1;
            continue;
        elseif(saliency_val>0.1 && length(find(map_patch>0.1))>=0.7*size(map_patch,1)*size(map_patch,2))
             patch=imresize(patch,[51 51]);
        sal_vals=[sal_vals;saliency_val];
        
        imwrite(patch,['F:\Project\Dataset\local\salient_train\' int2str(t) '.jpg']);
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
        if(saliency_val<=0.1 && length(find(map_patch<=0.1))>=0.7*size(map_patch,1)*size(map_patch,2))
        sal_vals1=[sal_vals1;saliency_val];
        imwrite(patch,['F:\Project\Dataset\non_salient_extras\' int2str(t) '.jpg']);
        t=t+1;
        end
    end
end

 

