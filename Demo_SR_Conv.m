close all;
clear all;

run matconvnet/matlab/vl_setupnn;
addpath('utils')

load('Model/model.mat');%%20(32)+10(32)

use_gpu = 1;
up_scale = 8;
shave = 1;

index = size(model.weight,2);
if use_gpu
    for i = 1:index
        model.weight{i} = gpuArray(model.weight{i});
        model.bias{i} = gpuArray(model.bias{i});
    end
end

folder = './Data/infrared20';
filepaths = dir(fullfile(folder,'*.png'));
%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%
% for num = 1 : length(filepaths)
for num = 10:10
fprintf('%d / %d ',num,length(filepaths));
im_gt = imread(fullfile(folder,filepaths(num).name));



im_gt = modcrop(im_gt,up_scale);
im_gt_ycbcr = double(rgb2ycbcr(im_gt));
im_gt_y = im_gt_ycbcr(:,:,1);

im_gt = double(im_gt);
im_l = imresize(im_gt / 255.,1/up_scale,'bicubic');

im_l_ycbcr = rgb2ycbcr(im_l);



im_l_y = im_l_ycbcr(:,:,1);

if use_gpu
    im_l_y = gpuArray(im_l_y);
end

%%%%%%%%%%%%%%%
tic;
[im_h_y, convfea] = cdn_mrf_Matconvnet(im_l_y, model,up_scale);
toc;
% figure,imshow(convfea,[]);
%%%%%%%%%%%%%%%%%


if use_gpu
    im_h_y = gather(im_h_y);
end

im_h_y = im_h_y * 255.0;
im_h_ycbcr = imresize(im_l_ycbcr,up_scale,'bicubic');
im_b_y = im_h_ycbcr(:,:,1)*255.0;
im_b = ycbcr2rgb(im_h_ycbcr) * 255.0;
im_h_ycbcr(:,:,1) = im_h_y / 255.0;
im_h  = ycbcr2rgb(im_h_ycbcr) * 255.0;

figure,imshow([uint8(im_b) uint8(im_h) uint8(im_gt)]);


if shave == 1;
    shave_border = round(up_scale);
else
    shave_border = 0;
end

sr_psnr(num) = compute_psnr(im_h,im_gt,shave_border);
bi_psnr(num) = compute_psnr(im_b,im_gt,shave_border);
fprintf('sr_psnr: %f dB\n',sr_psnr(num));
fprintf('bi_psnr: %f dB\n',bi_psnr(num));

[mssim, ~] = ssim_index(im_h_y,im_gt_y);
SR_SSIM(num) = mssim;
[mssim, ssim_map] = ssim_index(im_b_y,im_gt_y);
BI_SSIM(num) = mssim;
fprintf('SR_SSIM: %f \n',SR_SSIM(num));
fprintf('BI_SSIM: %f \n',BI_SSIM(num));
end