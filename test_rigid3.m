% test rigid3 with shifts of 2
clear all
close all

im1 = abs(phantom3d([128 128 9]));
im1 = cat(2,zeros(size(im1,1),2,size(im1,3)),im1,zeros(size(im1,1),2,size(im1,3)));

im2 = abs(circshift(im1,[2 0 0])+randn(size(im1))/1000);
[im3 delta] = rigid3(im1,im2);
figure(1);ims(abs(im1-im3));title(num2str(delta,'%.1f '));drawnow

im2 = abs(circshift(im1,[0 2 0])+randn(size(im1))/1000);
[im3 delta] = rigid3(im1,im2);
figure(2);ims(abs(im1-im3));title(num2str(delta,'%.1f '));drawnow

im2 = abs(circshift(im1,[0 0 2])+randn(size(im1))/1000);
[im3 delta] = rigid3(im1,im2);
figure(3);ims(abs(im1-im3));title(num2str(delta,'%.1f '));drawnow

im2 = abs(imrotate3(im1,2,[1 0 0],'cubic','crop')+randn(size(im1))/1000);
[im3 delta] = rigid3(im1,im2);
figure(4);ims(abs(im1-im3));title(num2str(delta,'%.1f '));drawnow

im2 = abs(imrotate3(im1,2,[0 1 0],'cubic','crop')+randn(size(im1))/1000);
[im3 delta] = rigid3(im1,im2);
figure(5);ims(abs(im1-im3));title(num2str(delta,'%.1f '));drawnow

im2 = abs(imrotate3(im1,2,[0 0 1],'cubic','crop')+randn(size(im1))/1000);
[im3 delta] = rigid3(im1,im2);
figure(6);ims(abs(im1-im3));title(num2str(delta,'%.1f '));drawnow

im2 = circshift(im1,[2 2 2])+randn(size(im1))/1000;
im2 = imrotate3(im2,2,[1 0 0],'cubic','crop');
im2 = imrotate3(im2,2,[0 1 0],'cubic','crop');
im2 = imrotate3(im2,2,[0 0 1],'cubic','crop');
im2 = abs(im2);
[im3 delta] = rigid3(im1,im2);
figure(7);ims(abs(im1-im3));title(num2str(delta,'%.1f '));drawnow
