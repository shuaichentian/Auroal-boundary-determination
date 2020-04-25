clear
close all;
fileFolder=fullfile('./test/2');
dirOutput=dir(fullfile(fileFolder,'*.bmp'));
filename={dirOutput.name};
fileFolder1='./logits_img';
dirOutput=dir(fullfile(fileFolder1,'*.bmp'));
filename1={dirOutput.name};

% imageDate = '010_033213';
% imageDate = '31-Jul-1997_09_59_53';
%%%%%%%% loading image %%%%%%%
a=length(filename);
for i=1:a
    imageDate=strrep(filename{1,i},'.bmp','')
    %~strcmp([imageDate ],filename1)
    %if ~strcmp([imageDate ],filename1)
     %   continue
    %end

    disp('loading image');
    % load_image = 'images/test.bmp';
    load_image = ['test/2/' imageDate '.bmp'];
    mask_image = 'mask.bmp'; % fov mask
    image = imread(load_image);
    image_o=image;
    [m, n] = size(image); 
    %image = histeq(image);
    mask = imread(mask_image);
    mask = double(mask);
    flag1 = imread(['ini/' imageDate '_inner.bmp']);
    flag1 = im2bw(flag1);
    flag1 = 1 - flag1 * 2;
    flag2 = imread(['ini/' imageDate '_outer.bmp']);
    flag2 = im2bw(flag2);
    flag2 = flag2 * 2 - 1;
    [m, n] = size(image);
    fMaxDist = m * n;
    phi1 = -sdf_mex(single(flag1),single(fMaxDist),int32(1),int32(1));
    phi2 = sdf_mex(single(flag2),single(fMaxDist),int32(1),int32(1));
    %[m, n] = size(image); 
    %image = histeq(image);
    %mask = imread(mask_image);
    %mask = double(mask);
    loadimg=['logits_img/' imageDate '.bmp'];
    img_log=imread(loadimg);
    img_log1=img_log
    img_log = double(img_log);
    img_log=img_log/255.0-0.5;
    %flag=load('flag.mat');
    %flag=struct2cell(flag);
    %flag=cell2mat(flag);
    %flag1=flag;
    %flag2=flag;
    %flag1(flag>23)=0;
    %flag1(flag<23 | flag==23)=1;
    %flag2(flag>98)=0;   %120 before changed
    %flag2(flag<98| flag==98)=1;
    %phi1 = SDF(flag1);
    %phi2 = SDF(flag2);
    %flag1 = 1 - flag1 * 2;
    %flag2 = imread(['images/010_012021_outer.bmp']);
    %flag2 = im2bw(flag2);
    fMaxDist = m * n;
    h = fspecial('gaussian', 3, 1);
    image = imfilter(image, h);
    figure;imshow(image);
    Img = mat2gray(image);
% figure;imshow(uint8(Img));


%%%%%%%% Parameters setting %%%%%%%
    rad = 13; % the radius of local information
    alpha = 2; % controls the smoothness of segmenting shapes
    beta = 0.1; % weights the significance of the shape prior knowledge
    gama = 1; % weights the significance of the local information   '''1 before changed'''
    lamda = 1.5; %  is a coefficient of the global region energy which helps speed up the curve evolution
    sigma = 10; % the variance of shape distance  %20 before cahnged
    deltaT = 0.5; % time step
    epsilon = 10; % narrowband width
    lamda2=0;  % the weights of the logits term.  2 before changed



%%%%%%%% curve initialization %%%%%%%
% flag  = roipoly(mat2gray(Img));
% flag1 = flag * 2 - 1;
%phi1 = -SDF(flag1);
% flag  = roipoly(mat2gray(Img));
% flag2 = flag * 2 - 1;
%phi2 = SDF(flag2);

    tic;
    for i = 1 : 560
    
    %显示水平集的演化状况
    if mod(i-1, 10) == 0      
            figure(3);
            imagesc(image_o,[0, 255]); axis off; axis equal; colormap(gray);
            title(num2str(i));
            hold on;
            contour(phi1, [0,0], 'c', 'LineWidth', 1.5);
            contour(phi2, [0,0], 'r', 'LineWidth', 1.5);
            %contour(phi1, [0,0], 'b', 'LineWidth', 3);
            hold off;
            pause(0.001);
    end
    

    [phi1, phi2] = evolution2(i,phi1, phi2, Img,img_log, mask, rad, alpha, beta, gama, lamda, lamda2, sigma, deltaT, epsilon);

    %显示轮廓曲线
%     figure(4);mesh(phi1);
%     figure(5);mesh(phi2);


    end
    toc;
    
    '''找到零等值线的位置'''
    [c1,h1]=contour(phi1, [0,0]);
    [c2,h2]=contour(phi2, [0,0]);
    number1=c1(2,1);
    number2=c2(2,1);
    pos1= c1(:,2:1+number1) ;
    pos2= c2(:,2:1+number2) ;
    
    save(['dmsp/', imageDate, '_manual.mat'], 'pos1', 'pos2')
    
    Iauto = zeros(size(image));
    Iauto(phi1 <= 0 & phi2 >=0) = 1;

    imwrite(Iauto, ['dmsp/' imageDate '_our.bmp']);
 end



