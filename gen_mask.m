
addpath generate_mask;
addpath utils;
%%

%%
% 
% ImgSize = [561 716];
% R = 6;
% for acs = 66
%     % acs = 25;
%     [mask, AccSamRate] = Generate_Uniform_Mask2(ImgSize, R, acs);
%     % mask = double(mask);
%     [calibSize, ACS_edge] = getCalibSize_1D_Edt(mask);
%     samRate = ceil(AccSamRate * 100);
%     path1 = sprintf('./mask/%dx%d/uniform/', ImgSize(1), ImgSize(2));
%     if ~exist(path1,'dir')
%         mkdir(path1)
%     end
%     imwrite(imrotate(mask,90), [path1, 'mask_uniform_', num2str(ImgSize(1)),'_', ...
%         num2str(ImgSize(2)), '_R_' num2str(R) '_SR_', num2str(samRate), '_AC_', num2str(calibSize(1)),'.png']);
%     fprintf([path1, 'mask_uniform_', num2str(ImgSize(1)),'_', ...
%         num2str(ImgSize(2)), '_R_' num2str(R) '_SR_', num2str(samRate), '_AC_', num2str(calibSize(1)),'.png\n']);
% end



%% generate a random mask,
%
% note: because we need to rotate the mask when we use it, so size(mask) = [size(ksfull,2), size(ksfull,1)], not [size(ksfull,1), size(ksfull,2)].
% ImgSize = [1432 1122];
% for ratei = 19
%     rate = ratei/100;
%     [mask, AccSamRate] = GenCartesianSampling(ImgSize, rate);
%     temp = imrotate(mask, 90);
%     [calibSize, ACS_edge] = getCalibSize_1D_Edt(temp);
%     samRate = ceil(AccSamRate * 100);
%     path2 = sprintf('./mask/%dx%d/random/', ImgSize(2), ImgSize(1));
%     if ~exist(path2,'dir')
%         mkdir(path2)
%     end
%     mask = logical(mask);
%     if ~exist([path2, 'mask_random_', num2str(ImgSize(2)),'_', ...
%         num2str(ImgSize(1)), '_SR_', num2str(samRate), '_AC_', num2str(calibSize(1)),'.png\n'])
% 
%         imwrite(mask, [path2 'mask_random_', num2str(ImgSize(2)),'_', ...
%             num2str(ImgSize(1)), '_SR_', num2str(samRate), '_AC_', num2str(calibSize(1)),'.png']);
% 
%         fprintf([path2, 'mask_random_', num2str(ImgSize(2)),'_', ...
%             num2str(ImgSize(1)), '_SR_', num2str(samRate), '_AC_', num2str(calibSize(1)),'.png\n']);
%     else
%         fprintf('the mask has existed, can not be covered\n');
%     end
%     % imwrite(mask, [path2 'mask_random_', num2str(ImgSize(2)),'_', ...
%     %     num2str(ImgSize(1)), '_SR_', num2str(ceil(rate*100)), '_AC_', num2str(calibSize(1)),'.png']);
% end



%% generate a variable acceleration factor mask

% ImgSize = [768 616];
ImgSize = [372 640];
dataCoverRates = [0.2,0.4,1];
% dataCoverRates = [0.1,0.2,0.3,1];
% dataCoverRates = [0.2,0.4,0.6,0.8,1];

for acs = 30: 40
    R = 3;
    [mask, AccSamRate] = Generate_Variable_R_Mask(ImgSize, acs, dataCoverRates, R);
    % mask = double(mask);
    [calibSize, ACS_edge] = getCalibSize_1D_Edt(mask);
    samRate = ceil(AccSamRate * 100);
    path1 = sprintf('./mask/%dx%d/variableR/', ImgSize(1), ImgSize(2));
    if ~exist(path1,'dir')
      mkdir(path1);
    end
    dataCoverInfo = '';

    for i = 1:length(dataCoverRates)
        dataCoverInfo = strcat(dataCoverInfo,'_R',num2str(R));
        dataCoverInfo = strcat(dataCoverInfo,'_',num2str(dataCoverRates(i)));
        R = R+1;
    end

    imwrite(imrotate(mask,90), [path1, 'mask_variableR_', num2str(ImgSize(1)),'_', ...
        num2str(ImgSize(2)), '_SR_', num2str(samRate), '_AC_', num2str(calibSize(1)),dataCoverInfo,'.png']);
    fprintf([path1, 'mask_variableR_', num2str(ImgSize(1)),'_', ...
        num2str(ImgSize(2)), '_SR_', num2str(samRate), '_AC_', num2str(calibSize(1)),dataCoverInfo,'.png','\n']);
end
