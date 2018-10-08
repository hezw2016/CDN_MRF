function [im_h_y, convfea] = cdn_mrf_Matconvnet(im_l_y,model,scale)

model_scale = scale;
weight = model.weight;
bias = model.bias;
layer_num = size(weight,2);

    im_y = single(imresize(im_l_y,model_scale,'bicubic'));
    convfea = vl_nnconv(im_y,weight{1},bias{1},'Pad',1);
%     figure,imshow(convfea(:,:,1));
    for i = 2:20
        convfea = vl_nnrelu(convfea);
        convfea = vl_nnconv(convfea,weight{i},bias{i},'Pad',1);
%         figure,imshow(convfea(:,:,1));
    end
    im_h_y = convfea + im_y;
%     figure,imshow(convfea,[]);
    
    convfea = vl_nnconv(im_h_y,weight{21},bias{21},'Pad',1);
    for i = 22:layer_num
        convfea = vl_nnrelu(convfea);
        convfea = vl_nnconv(convfea,weight{i},bias{i},'Pad',1);
%         figure,imshow(convfea(:,:,45),[]);
    end
%     figure,imshow(convfea,[]);
    im_h_y = convfea + im_h_y;
    
    

% if size(im_h_y,1) > lh * scale
%    im_h_y = gather(im_h_y);
%    im_h_y = imresize(im_h_y,[lh * scale,lw * scale],'bicubic');
% end
end