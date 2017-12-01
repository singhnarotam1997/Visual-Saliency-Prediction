files=dir('maps\train');
files=files(3:end);
files1=dir('predicted1');
files1=files1(3:end);
t=1;
beta=0.3;
p_ar=zeros(length(0.05:0.05:0.9),1);
r_ar=zeros(length(0.05:0.05:0.9),1);
for th=0.05:0.05:0.9
    tp=0;fp=0;fn=0;
    for i=1:length(files1)
    name=files1(i).name;
    name=name(1:length(name)-4);
    im=im2double(imread([files(i).folder '\' name '.png']));
    im1=im2double(imread([files1(i).folder '\' name '.jpg']));
    im1(find(im1<=th))=0;
    im1(find(im1>th))=1;
    im(find(im<=th))=0;
    im(find(im>th))=1;
    temp1=im1(find(im==0));
    fn=fn+length(find(temp1==0));
    fp=fp+length(find(temp1==1));
    temp1=im1(find(im==1));
    tp=tp+length(find(temp1==1));
    end
    p_ar(t)=tp/(tp+fp);
    r_ar(t)=tp/(tp+fn);
    t=t+1;
end
f_ar=(1+beta*beta).*p_ar.*r_ar;
f_ar=f_ar./(beta*beta.*p_ar+r_ar);
