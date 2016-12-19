%-----------------------------------------------------
% Use RGB average value of each hyperplane as vector
% Test different classifier performance
%-----------------------------------------------------
function [rgb_vector, area, centriod, bbox, perimeter] = avgRGBb(x)
a1=imresize(imread(x),[300 300]);
refcolor_a1=squeeze(a1(120,160, :));
[a1rows, a1cols, a1planes]=size(a1);
distances=zeros(a1rows,a1cols);

for i=1:a1rows
    for j=1:a1cols
        tmp=squeeze(a1(i,j,:));
        distances(i,j)=norm(double(tmp) - double(refcolor_a1));
    end
end
figure(2),imagesc(distances),colormap(gray);
figure(3),imagesc(distances<100),colormap(gray);
bw=logical(distances>100);
figure(4),imagesc(imopen(bw,ones(4,4))),colormap(gray);
L=bwlabel(imopen(bw,ones(4,4)));
bw1=L==1;
bw=logical(bw1==0)
a1r=a1(:,:,1);
a1g=a1(:,:,2);
a1b=a1(:,:,3);
bw=uint8(bw);
n=0;
rr=zeros(a1rows,a1cols,'uint8');
gg=zeros(a1rows,a1cols,'uint8');
bb=zeros(a1rows,a1cols,'uint8');
for i=1:a1rows
	for j=1:a1cols
		rr(i,j)=bw(i,j)*a1r(i,j);
		gg(i,j)=bw(i,j)*a1g(i,j);
		bb(i,j)=bw(i,j)*a1b(i,j);
	
	end
end
n=sum(sum(bw),2);
ravg=sum(sum(rr),2)/n;
gavg=sum(sum(gg),2)/n;
bavg=sum(sum(bb),2)/n;
rgb_vector = [ravg gavg bavg]';

H=vision.BlobAnalysis('PerimeterOutputPort',true);
[area, centriod, bbox, perimeter]=step(H,bw1);

end




















