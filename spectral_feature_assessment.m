%% spectral_featureextraction.m
% extracts various features from spectrogram
% Analyzes them
% Input: data matrix as M samples x N songs
% Output: related features we want
[M,N]  = size(audioMatrix);

% Different features to populate
spec_mean_w = [];
spec_var_w = [];
pow_vectors = [];
max_w = 0;
for i = 1:N
    win_size = 500;
    pctOverlap = 0.20;
    sig = audioMatrix(:,i);
    [spec,w,t] = spectrogram(sig, hanning(win_size),round(pctOverlap*win_size)); % spec is w x t

    %% Feature extraction
    pow_spec = abs(spec); % w x t
    spec_mean_w = [spec_mean_w, mean(pow_spec,2)];  % spectral frequency  w frequencies x N songs
    spec_var_w = [spec_var_w, var(pow_spec,0,2)]; % spectral frequency w frequencies x N songs

    pow_vectors = [pow_vectors, reshape(pow_spec, [length(w)*length(t),1])]; % b bins x N songs, b = w*t

end

%% Assessing separability of new features
%% spectral mean and frequency
spec_data = [spec_mean_w', spec_var_w']; % row vector, N songs x w freq , N songs x w freq = N songs x 2w freq
[Y,loss] = tsne(spec_data);
% Using tsne, plot
% Plot Figure
figure
start_ind = 1;
end_ind = 100;
for i = 1:10
    c = '.';
    if i > 7
        c = 'x';
    end
    plot(Y(start_ind:end_ind,1),Y(start_ind:end_ind,2), c)
    hold on;
    start_ind = end_ind + 1;
    end_ind = end_ind + 100;
    if end_ind > N
        end_ind = N;
    end
end
title("TSNE for Spectrogram features")
legend('blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock','jazz')
hold off;
% PCA of new features
numcomp = 2;
[coeff,score,~,~,explained,mu] = pca(spec_data, 'NumComponents',numcomp);
% Plot Figure
figure
start_ind = 1;
end_ind = 100;
for i = 1:10
    c = '.';
    if i > 7
        c = 'x';
    end
    plot(score(start_ind:end_ind,1),score(start_ind:end_ind,2), c)
    hold on;
    start_ind = end_ind + 1;
    end_ind = end_ind + 100;
    if end_ind > N
        end_ind = N;
    end
end
title("PCA for Spectrogram Features")
legend('blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock','jazz')
hold off;
pow_vectors = pow_vectors';
%% spectral as an image
clearvars -except audioMatrix pow_vectors N
%spec_data = pow_vectors'; % row vector, N songs x b bins
[Y,loss] = tsne(pow_vectors);
% Using tsne, plot
% Plot Figure
figure
start_ind = 1;
end_ind = 100;
for i = 1:10
    c = '.';
    if i > 7
        c = 'x';
    end
    plot(Y(start_ind:end_ind,1),Y(start_ind:end_ind,2), c)
    hold on;
    start_ind = end_ind + 1;
    end_ind = end_ind + 100;
    if end_ind > N
        end_ind = N;
    end
end
title("TSNE for Spectrogram as image")
legend('blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock','jazz')
hold off;
% Matlab's PCA
numcomp = 100;
[coeff,score,~,~,explained,mu]= pca(pow_vectors , 'NumComponents',numcomp); % Data as row vectors
% Plot Figure
figure
start_ind = 1;
end_ind = 100;
for i = 1:10
    c = '.';
    if i > 7
        c = 'x';
    end
    plot(score(start_ind:end_ind,1),score(start_ind:end_ind,2), c)
    hold on;
    start_ind = end_ind + 1;
    end_ind = end_ind + 100;
    if end_ind > N
        end_ind = N;
    end
end
title("PCA for Spectrogram as Image")
legend('blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock','jazz')
hold off;
% Using tsne, plot for pca
[Y,loss] = tsne(score);
% Plot Figure
figure
start_ind = 1;
end_ind = 100;
for i = 1:10
    c = '.';
    if i > 7
        c = 'x';
    end
    plot(Y(start_ind:end_ind,1),Y(start_ind:end_ind,2), c)
    hold on;
    start_ind = end_ind + 1;
    end_ind = end_ind + 100;
    if end_ind > N
        end_ind = N;
    end
end
title("TSNE for Spectrogram as pca image")
legend('blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock','jazz')
hold off;
% Plot subplotFigure
figure
sgtitle("Subplots for TSNE for Spectrogram as PCA")
start_ind = 1;
end_ind = 100;
leg = ["blues", "classical", "country", "disco","hiphop", "metal", "pop", "reggae", "rock","jazz"]
for i = 1:10
    c = '.';
    if i > 7
        c = 'x';
    end
    subplot(2,5,i)
    plot(Y(start_ind:end_ind,1),Y(start_ind:end_ind,2), c)
    hold on;
    start_ind = end_ind + 1;
    end_ind = end_ind + 100;
    if end_ind > N
        end_ind = N;
    end
    legend(leg(i))
    xlim([-30,30])
    ylim([-30,30])
end
hold off;
pow_vectors = pow_vectors';
%% NMF
opt = statset('MaxIter', 500, 'Display', 'final');
[B, W] = nnmf(pow_vectors, 40, 'options', opt, 'algorithm', 'mult'); % data as column vectors
% Using tsne, plot for pca
[Y,loss] = tsne(W');
% Plot Figure
figure
start_ind = 1;
end_ind = 100;
for i = 1:10
    c = '.';
    if i > 7
        c = 'x';
    end
    plot(Y(start_ind:end_ind,1),Y(start_ind:end_ind,2), c)
    hold on;
    start_ind = end_ind + 1;
    end_ind = end_ind + 100;
    if end_ind > N
        end_ind = N;
    end
end
title("TSNE for Spectrogram as NMF")
legend('blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock','jazz')
hold off;
% Plot subplotFigure
figure
sgtitle("Subplots for TSNE for Spectrogram as NMF")
start_ind = 1;
end_ind = 100;
leg = ["blues", "classical", "country", "disco","hiphop", "metal", "pop", "reggae", "rock","jazz"]
for i = 1:10
    c = '.';
    if i > 7
        c = 'x';
    end
    subplot(2,5,i)
    plot(Y(start_ind:end_ind,1),Y(start_ind:end_ind,2), c)
    hold on;
    start_ind = end_ind + 1;
    end_ind = end_ind + 100;
    if end_ind > N
        end_ind = N;
    end
    legend(leg(i))
    xlim([-30,30])
    ylim([-30,30])
end
hold off;