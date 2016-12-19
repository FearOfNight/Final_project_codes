%-----------------------------------------------------------------
% Function: Extract fruits on plate
% Author:   Lin, Bor-Tyng
%-----------------------------------------------------------------

% Train/Test data of apple/banana/orange/lemon/cherry/strawberry
% 5 for training, 5 for testing total 10 images with size 300x300
%-----------------------------------------------------------------
%                           APPLE
%-----------------------------------------------------------------
[a1, area1, centriod1, bbox1, perimeter1] = avgRGBb('apple1.jpg');
[a2, area1, centriod1, bbox1, perimeter1] = avgRGBb('apple2.jpg');
[a3, area1, centriod1, bbox1, perimeter1] = avgRGBb('apple3.jpg');
[a4, area1, centriod1, bbox1, perimeter1] = avgRGBb('apple4.jpg');
[a5, area1, centriod1, bbox1, perimeter1] = avgRGBb('apple5.jpg');
[a6, area1, centriod1, bbox1, perimeter1] = avgRGBb('apple6.jpg');
[a7, area1, centriod1, bbox1, perimeter1] = avgRGBb('apple7.jpg');
[a8, area1, centriod1, bbox1, perimeter1] = avgRGBb('apple8.jpg');
[a9, area1, centriod1, bbox1, perimeter1] = avgRGBb('apple9.jpg');
[a10, area1, centriod1, bbox1, perimeter1] = avgRGBb('apple10.jpg');

%test_a = [a6'; a7'; a8'; a9';a10'];
%-----------------------------------------------------------------
%                           BANANA
%-----------------------------------------------------------------
[b1, area1, centriod1, bbox1, perimeter1] = avgRGBw('banana1.jpg');
[b2, area1, centriod1, bbox1, perimeter1] = avgRGBw('banana2.jpg');
[b3, area1, centriod1, bbox1, perimeter1] = avgRGBw('banana3.jpg');
[b4, area1, centriod1, bbox1, perimeter1] = avgRGBw('banana4.jpg');
[b5, area1, centriod1, bbox1, perimeter1] = avgRGBw('banana5.jpg');
[b6, area1, centriod1, bbox1, perimeter1] = avgRGBw('banana6.jpg');
[b7, area1, centriod1, bbox1, perimeter1] = avgRGBw('banana7.jpg');
[b8, area1, centriod1, bbox1, perimeter1] = avgRGBw('banana8.jpg');
[b9, area1, centriod1, bbox1, perimeter1] = avgRGBw('banana9.jpg');
[b10, area1, centriod1, bbox1, perimeter1] = avgRGBw('banana10.jpg');

%test_b = [b6'; b7'; b8'; b9'; b10'];

%-----------------------------------------------------------------
%                           ORANGE
%-----------------------------------------------------------------
[o1, area1, centriod1, bbox1, perimeter1] = avgRGBb('orange1.jpg');
[o2, area1, centriod1, bbox1, perimeter1] = avgRGBb('orange2.jpg');
[o3, area1, centriod1, bbox1, perimeter1] = avgRGBb('orange3.jpg');
[o4, area1, centriod1, bbox1, perimeter1] = avgRGBb('orange4.jpg');
[o5, area1, centriod1, bbox1, perimeter1] = avgRGBb('orange5.jpg');
[o6, area1, centriod1, bbox1, perimeter1] = avgRGBb('orange6.jpg');
[o7, area1, centriod1, bbox1, perimeter1] = avgRGBb('orange7.jpg');
[o8, area1, centriod1, bbox1, perimeter1] = avgRGBb('orange8.jpg');
[o9, area1, centriod1, bbox1, perimeter1] = avgRGBb('orange9.jpg');
[o10, area1, centriod1, bbox1, perimeter1] = avgRGBb('orange10.jpg');

%test_o = [o6'; o7'; o8'; o9'; o10'];


%-----------------------------------------------------------------
%                           LEMON
%-----------------------------------------------------------------
[l1, area1, centriod1, bbox1, perimeter1] = avgRGBb('lemon1.jpg');
[l2, area1, centriod1, bbox1, perimeter1] = avgRGBb('lemon2.jpg');
[l3, area1, centriod1, bbox1, perimeter1] = avgRGBb('lemon3.jpg');
[l4, area1, centriod1, bbox1, perimeter1] = avgRGBb('lemon4.jpg');
[l5, area1, centriod1, bbox1, perimeter1] = avgRGBb('lemon5.jpg');
[l6, area1, centriod1, bbox1, perimeter1] = avgRGBb('lemon6.jpg');
[l7, area1, centriod1, bbox1, perimeter1] = avgRGBb('lemon7.jpg');
[l8, area1, centriod1, bbox1, perimeter1] = avgRGBb('lemon8.jpg');
[l9, area1, centriod1, bbox1, perimeter1] = avgRGBb('lemon9.jpg');
[l10, area1, centriod1, bbox1, perimeter1] = avgRGBb('lemon10.jpg');

%test_l = [l6'; l7'; l8'; l9'; l10'];

%-----------------------------------------------------------------
%                           CHERRY
%-----------------------------------------------------------------
[c1, area1, centriod1, bbox1, perimeter1] = avgRGBw('cherry1.jpg');
[c2, area1, centriod1, bbox1, perimeter1] = avgRGBw('cherry2.jpg');
[c3, area1, centriod1, bbox1, perimeter1] = avgRGBw('cherry3.jpg');
[c4, area1, centriod1, bbox1, perimeter1] = avgRGBw('cherry4.jpg');
[c5, area1, centriod1, bbox1, perimeter1] = avgRGBw('cherry5.jpg');
[c6, area1, centriod1, bbox1, perimeter1] = avgRGBw('cherry6.jpg');
[c7, area1, centriod1, bbox1, perimeter1] = avgRGBw('cherry7.jpg');
[c8, area1, centriod1, bbox1, perimeter1] = avgRGBw('cherry8.jpg');
[c9, area1, centriod1, bbox1, perimeter1] = avgRGBw('cherry9.jpg');
[c10, area1, centriod1, bbox1, perimeter1] = avgRGBw('cherry10.jpg');

%test_c = [c6'; c7'; c8'; c9'; c10'];

%-----------------------------------------------------------------
%                           STRAWBERRY
%-----------------------------------------------------------------
[s1, area1, centriod1, bbox1, perimeter1] = avgRGBb('strawberry1.jpg');
[s2, area1, centriod1, bbox1, perimeter1] = avgRGBb('strawberry2.jpg');
[s3, area1, centriod1, bbox1, perimeter1] = avgRGBb('strawberry3.jpg');
[s4, area1, centriod1, bbox1, perimeter1] = avgRGBb('strawberry4.jpg');
[s5, area1, centriod1, bbox1, perimeter1] = avgRGBb('strawberry5.jpg');
[s6, area1, centriod1, bbox1, perimeter1] = avgRGBb('strawberry6.jpg');
[s7, area1, centriod1, bbox1, perimeter1] = avgRGBb('strawberry7.jpg');
[s8, area1, centriod1, bbox1, perimeter1] = avgRGBb('strawberry8.jpg');
[s9, area1, centriod1, bbox1, perimeter1] = avgRGBb('strawberry9.jpg');
[s10, area1, centriod1, bbox1, perimeter1] = avgRGBb('strawberry10.jpg');

train_a = [a1'; a2'; a3'; a4';a5'; a6'; a7'; a8'; a9';a10'];
train_b = [b1'; b2'; b3'; b4'; b5'; b6'; b7'; b8'; b9'; b10'];
train_o = [o1'; o2'; o3'; o4'; o5'; o6'; o7'; o8'; o9'; o10'];
train_l = [l1'; l2'; l3'; l4'; l5'; l6'; l7'; l8'; l9'; l10'];
train_c = [c1'; c2'; c3'; c4'; c5'; c6'; c7'; c8'; c9'; c10'];
train_s = [s1'; s2'; s3'; s4'; s5'; s6'; s7'; s8'; s9'; s10'];
%test_s = [s6'; s7'; s8'; s9'; s10'];
group=[];
group=[0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 3; 3; 3; 3; 3; 3; 3; 3; 3; 3; 4; 4; 4; 4; 4; 4; 4; 4; 4; 4; 5; 5; 5; 5; 5; 5; 5; 5; 5; 5];
train_data=double([train_a; train_b; train_o; train_l; train_c; train_s]);

%-----------------------------------------------------------------
% single target fruit a picture
%-----------------------------------------------------------------
%test_data=double([test_a; test_b; test_o; test_l; test_c; test_s]);
test1=imread('apple6.jpg');
test2=imread('banana10.jpg');
test3=imread('orange8.jpg');
test4=imread('lemon9.jpg');
test5=imread('strawberry10.jpg');
test1=imresize(test1,[300 300]);
test2=imresize(test2,[300 300]);
test3=imresize(test3,[300 300]);
test4=imresize(test4,[300 300]);
test5=imresize(test5,[300 300]);

tt1=reshape(test1,[],size(test1,3),1);
tt1=double(tt1);
tt2=reshape(test2,[],size(test2,3),1);
tt2=double(tt2);
tt3=reshape(test3,[],size(test3,3),1);
tt3=double(tt3);
tt4=reshape(test4,[],size(test4,3),1);
tt4=double(tt4);
tt5=reshape(test5,[],size(test5,3),1);
tt5=double(tt5);

%linear classifier
[class_linear,err,P,logp,coeff] = classify(tt1,train_data,group,'linear');
class_linear = reshape(class_linear,300,300);
figure(1), imagesc(class_linear);
[class_linear,err,P,logp,coeff] = classify(tt2,train_data,group,'linear');
class_linear = reshape(class_linear,300,300);
figure(2), imagesc(class_linear);
[class_linear,err,P,logp,coeff] = classify(tt3,train_data,group,'linear');
class_linear = reshape(class_linear,300,300);
figure(3), imagesc(class_linear);
[class_linear,err,P,logp,coeff] = classify(tt4,train_data,group,'linear');
class_linear = reshape(class_linear,300,300);
figure(4), imagesc(class_linear);
[class_linear,err,P,logp,coeff] = classify(tt5,train_data,group,'linear');
class_linear = reshape(class_linear,300,300);
figure(5), imagesc(class_linear);

%quadratic classifier
[class_quad,err,P,logp,coeff] = classify(tt1,train_data,group,'quadratic');
class_quad = reshape(class_quad,300,300);
figure(6), imagesc(class_quad);
[class_quad,err,P,logp,coeff] = classify(tt2,train_data,group,'quadratic');
class_quad = reshape(class_quad,300,300);
figure(7), imagesc(class_quad);
[class_quad,err,P,logp,coeff] = classify(tt3,train_data,group,'quadratic');
class_quad = reshape(class_quad,300,300);
figure(8), imagesc(class_quad);
[class_quad,err,P,logp,coeff] = classify(tt4,train_data,group,'quadratic');
class_quad = reshape(class_quad,300,300);
figure(9), imagesc(class_quad);
[class_quad,err,P,logp,coeff] = classify(tt5,train_data,group,'quadratic');
class_quad = reshape(class_quad,300,300);
figure(10), imagesc(class_quad);

% svm classifier (only allow 2 group training data)
% redefine group and train_data size
group=[];
group=[0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1];
train_data=double([train_a; train_b]);
svm_struct =svmtrain(train_data,group);
svm_class = svmclassify(svm_struct,tt1);
svm_class = reshape(svm_class,300,300);
figure(11), imagesc(svm_class);
svm_class = svmclassify(svm_struct,tt2);
svm_class = reshape(svm_class,300,300);
figure(12), imagesc(svm_class);
svm_class = svmclassify(svm_struct,tt3);
svm_class = reshape(svm_class,300,300);
figure(13), imagesc(svm_class);
svm_class = svmclassify(svm_struct,tt4);
svm_class = reshape(svm_class,300,300);
figure(14), imagesc(svm_class);
svm_class = svmclassify(svm_struct,tt5);
svm_class = reshape(svm_class,300,300);
figure(15), imagesc(svm_class);

% svm classifier (radial-base-function, sigma=1)
svm_struct =svmtrain(train_data, group, 'kernel_function', 'rbf', 'rbf_sigma', 1);
svmrbf_class = svmclassify(svm_struct,tt1);
svmrbf_class = reshape(svmrbf_class,300,300);
figure(16), imagesc(svmrbf_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'rbf', 'rbf_sigma', 1);
svmrbf_class = svmclassify(svm_struct,tt2);
svmrbf_class = reshape(svmrbf_class,300,300);
figure(17), imagesc(svmrbf_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'rbf', 'rbf_sigma', 1);
svmrbf_class = svmclassify(svm_struct,tt3);
svmrbf_class = reshape(svmrbf_class,300,300);
figure(18), imagesc(svmrbf_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'rbf', 'rbf_sigma', 1);
svmrbf_class = svmclassify(svm_struct,tt4);
svmrbf_class = reshape(svmrbf_class,300,300);
figure(19), imagesc(svmrbf_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'rbf', 'rbf_sigma', 1);
svmrbf_class = svmclassify(svm_struct,tt5);
svmrbf_class = reshape(svmrbf_class,300,300);
figure(20), imagesc(svmrbf_class);  

% svm classifier (Multilayer Perceptron, default: [P1 P2]=[1 -1])
svm_struct =svmtrain(train_data, group, 'kernel_function', 'mlp');
svmmlp_class = svmclassify(svm_struct,tt1);
svmmlp_class = reshape(svmmlp_class,300,300);
figure(21), imagesc(svmmlp_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'mlp');
svmmlp_class = svmclassify(svm_struct,tt2);
svmmlp_class = reshape(svmmlp_class,300,300);
figure(22), imagesc(svmmlp_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'mlp');
svmmlp_class = svmclassify(svm_struct,tt3);
svmmlp_class = reshape(svmmlp_class,300,300);
figure(23), imagesc(svmmlp_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'mlp');
svmmlp_class = svmclassify(svm_struct,tt4);
svmmlp_class = reshape(svmmlp_class,300,300);
figure(24), imagesc(svmmlp_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'mlp');
svmmlp_class = svmclassify(svm_struct,tt5);
svmmlp_class = reshape(svmmlp_class,300,300);
figure(25), imagesc(svmmlp_class);

% svm classifier (linear)
svm_struct =svmtrain(train_data, group, 'kernel_function', 'linear');
svmlinear_class = svmclassify(svm_struct,tt1);
svmlinear_class = reshape(svmlinear_class,300,300);
figure(26), imagesc(svmlinear_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'linear');
svmlinear_class = svmclassify(svm_struct,tt2);
svmlinear_class = reshape(svmlinear_class,300,300);
figure(27), imagesc(svmlinear_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'linear');
svmlinear_class = svmclassify(svm_struct,tt3);
svmlinear_class = reshape(svmlinear_class,300,300);
figure(28), imagesc(svmlinear_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'linear');
svmlinear_class = svmclassify(svm_struct,tt4);
svmlinear_class = reshape(svmlinear_class,300,300);
figure(29), imagesc(svmlinear_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'linear');
svmlinear_class = svmclassify(svm_struct,tt5);
svmlinear_class = reshape(svmlinear_class,300,300);
figure(30), imagesc(svmlinear_class);

% grnn classifier
group=[];
group=[0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 3; 3; 3; 3; 3; 3; 3; 3; 3; 3; 4; 4; 4; 4; 4; 4; 4; 4; 4; 4; 5; 5; 5; 5; 5; 5; 5; 5; 5; 5];
train_data=double([train_a; train_b; train_o; train_l; train_c; train_s]);
net=newgrnn(tt1,train_data);


% newpnn classifier

%-----------------------------------------------------------------
% multiple fruits in a picture
%-----------------------------------------------------------------
fp1 = imread('fruits1.jpg');
fp2 = imread('fruits2.jpg');
fp3 = imread('fruits3.jpg');
fp4 = imread('fruits4.jpg');
fp5 = imread('fruits5.jpg');
fp1=imresize(fp1,[300 300]);
fp2=imresize(fp2,[300 300]);
fp3=imresize(fp3,[300 300]);
fp4=imresize(fp4,[300 300]);
fp5=imresize(fp5,[300 300]);

testfp1=reshape(fp1,[],size(fp1,3),1);
testfp1=double(testfp1);
testfp2=reshape(fp2,[],size(fp2,3),1);
testfp2=double(testfp2);
testfp3=reshape(fp3,[],size(fp3,3),1);
testfp3=double(testfp3);
testfp4=reshape(fp4,[],size(fp4,3),1);
testfp4=double(testfp4);
testfp5=reshape(fp5,[],size(fp5,3),1);
testfp5=double(testfp5);

[class_linear,err,P,logp,coeff] = classify(testfp1,train_data,group,'linear');
class_linear = reshape(class_linear,300,300);
figure(101), imagesc(class_linear);
[class_linear,err,P,logp,coeff] = classify(testfp2,train_data,group,'linear');
class_linear = reshape(class_linear,300,300);
figure(102), imagesc(class_linear);
[class_linear,err,P,logp,coeff] = classify(testfp3,train_data,group,'linear');
class_linear = reshape(class_linear,300,300);
figure(103), imagesc(class_linear);
[class_linear,err,P,logp,coeff] = classify(testfp4,train_data,group,'linear');
class_linear = reshape(class_linear,300,300);
figure(104), imagesc(class_linear);
[class_linear,err,P,logp,coeff] = classify(testfp5,train_data,group,'linear');
class_linear = reshape(class_linear,300,300);
figure(105), imagesc(class_linear);

[class_linear,err,P,logp,coeff] = classify(testfp1,train_data,group,'quadratic');
class_linear = reshape(class_linear,300,300);
figure(106), imagesc(class_linear);
[class_linear,err,P,logp,coeff] = classify(testfp2,train_data,group,'quadratic');
class_linear = reshape(class_linear,300,300);
figure(107), imagesc(class_linear);
[class_linear,err,P,logp,coeff] = classify(testfp3,train_data,group,'quadratic');
class_linear = reshape(class_linear,300,300);
figure(108), imagesc(class_linear);
[class_linear,err,P,logp,coeff] = classify(testfp4,train_data,group,'quadratic');
class_linear = reshape(class_linear,300,300);
figure(109), imagesc(class_linear);
[class_linear,err,P,logp,coeff] = classify(testfp5,train_data,group,'quadratic');
class_linear = reshape(class_linear,300,300);
figure(110), imagesc(class_linear);

group=[];
group=[0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1];
train_data=double([train_a; train_b]);
svm_struct =svmtrain(train_data,group);
svm_class = svmclassify(svm_struct,testfp1);
svm_class = reshape(svm_class,300,300);
figure(111), imagesc(svm_class);
svm_class = svmclassify(svm_struct,testfp2);
svm_class = reshape(svm_class,300,300);
figure(112), imagesc(svm_class);
svm_class = svmclassify(svm_struct,testfp3);
svm_class = reshape(svm_class,300,300);
figure(113), imagesc(svm_class);
svm_class = svmclassify(svm_struct,testfp4);
svm_class = reshape(svm_class,300,300);
figure(114), imagesc(svm_class);
svm_class = svmclassify(svm_struct,testfp5);
svm_class = reshape(svm_class,300,300);
figure(115), imagesc(svm_class);

svm_struct =svmtrain(train_data, group, 'kernel_function', 'rbf', 'rbf_sigma', 1);
svmrbf_class = svmclassify(svm_struct,testfp1);
svmrbf_class = reshape(svmrbf_class,300,300);
figure(116), imagesc(svmrbf_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'rbf', 'rbf_sigma', 1);
svmrbf_class = svmclassify(svm_struct,testfp2);
svmrbf_class = reshape(svmrbf_class,300,300);
figure(117), imagesc(svmrbf_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'rbf', 'rbf_sigma', 1);
svmrbf_class = svmclassify(svm_struct,testfp3);
svmrbf_class = reshape(svmrbf_class,300,300);
figure(118), imagesc(svmrbf_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'rbf', 'rbf_sigma', 1);
svmrbf_class = svmclassify(svm_struct,testfp4);
svmrbf_class = reshape(svmrbf_class,300,300);
figure(119), imagesc(svmrbf_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'rbf', 'rbf_sigma', 1);
svmrbf_class = svmclassify(svm_struct,testfp5);
svmrbf_class = reshape(svmrbf_class,300,300);
figure(120), imagesc(svmrbf_class);

svm_struct =svmtrain(train_data, group, 'kernel_function', 'mlp');
svmmlp_class = svmclassify(svm_struct,testfp1);
svmmlp_class = reshape(svmmlp_class,300,300);
figure(121), imagesc(svmmlp_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'mlp');
svmmlp_class = svmclassify(svm_struct,testfp2);
svmmlp_class = reshape(svmmlp_class,300,300);
figure(122), imagesc(svmmlp_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'mlp');
svmmlp_class = svmclassify(svm_struct,testfp3);
svmmlp_class = reshape(svmmlp_class,300,300);
figure(123), imagesc(svmmlp_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'mlp');
svmmlp_class = svmclassify(svm_struct,testfp4);
svmmlp_class = reshape(svmmlp_class,300,300);
figure(124), imagesc(svmmlp_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'mlp');
svmmlp_class = svmclassify(svm_struct,testfp5);
svmmlp_class = reshape(svmmlp_class,300,300);
figure(125), imagesc(svmmlp_class);

svm_struct =svmtrain(train_data, group, 'kernel_function', 'linear');
svmlinear_class = svmclassify(svm_struct,testfp1);
svmlinear_class = reshape(svmlinear_class,300,300);
figure(126), imagesc(svmlinear_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'linear');
svmlinear_class = svmclassify(svm_struct,testfp2);
svmlinear_class = reshape(svmlinear_class,300,300);
figure(127), imagesc(svmlinear_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'linear');
svmlinear_class = svmclassify(svm_struct,testfp3);
svmlinear_class = reshape(svmlinear_class,300,300);
figure(128), imagesc(svmlinear_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'linear');
svmlinear_class = svmclassify(svm_struct,testfp4);
svmlinear_class = reshape(svmlinear_class,300,300);
figure(129), imagesc(svmlinear_class);
svm_struct =svmtrain(train_data, group, 'kernel_function', 'linear');
svmlinear_class = svmclassify(svm_struct,testfp5);
svmlinear_class = reshape(svmlinear_class,300,300);
figure(130), imagesc(svmlinear_class);