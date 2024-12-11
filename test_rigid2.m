% test rigid2 with shifts of 2
clear all
close all

im1 = abs(phantom(128));
im1 = cat(2,zeros(size(im1,1),2),im1,zeros(size(im1,1),2));

im2 = abs(circshift(im1,[2 0])+randn(size(im1))/1000);
[im3 delta] = rigid2(im1,im2);
figure(1);imagesc(abs(im1-im3));colorbar;title(num2str(delta,'%.2f '));drawnow

im2 = abs(circshift(im1,[0 2])+randn(size(im1))/1000);
[im3 delta] = rigid2(im1,im2);
figure(2);imagesc(abs(im1-im3));colorbar;title(num2str(delta,'%.2f '));drawnow

im2 = abs(imrotate(im1,2,'bicubic','crop')+randn(size(im1))/1000);
[im3 delta] = rigid2(im1,im2);
figure(3);imagesc(abs(im1-im3));colorbar;title(num2str(delta,'%.2f '));drawnow

im2 = circshift(im1,[2 2]);
im2 = abs(imrotate(im2,2,'bicubic','crop')+randn(size(im1))/1000);
[im3 delta] = rigid2(im1,im2);
figure(4);imagesc(abs(im1-im3));colorbar;title(num2str(delta,'%.2f '));drawnow
