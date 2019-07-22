clc;
clear all;
close all;
 
 
[y , Fs] = audioread('American_Crow.wav');           %Input read
 
Mfcc =mfcc1(y,Fs);
ctrs=cluster(Mfcc);
figure;
plot(Mfcc(:,17));
[y1 , Fs] = audioread('Dark-Eyed_Junco1.wav');           %Input read
 
Mfcc1 =mfcc1(y1,Fs);
ctrs1=cluster(Mfcc1);
figure;
plot(Mfcc1(:,17));
d=eucld(ctrs1,ctrs);
d1=eucld(ctrs,ctrs);


function ctrs=cluster(Mfcc)
    c=1;
      [N,n]=size(Mfcc);
    index=randperm(N);
    ctrs = Mfcc(index(1:c),:);

    while size(unique(ctrs, 'rows'), 1) ~= c
        index=randperm(N);
        ctrs = X(index(1:c),:);
    end

    old_label = zeros(1,N);
    label = ones(1,N);

    iter = 0;
    while ~isequal(old_label, label)
        old_label = label;
        label = assign_labels(Mfcc, ctrs);

        for i = 1:c
            ctrs(i,:) = mean(Mfcc(label == i,:));
            if sum(isnan(ctrs(i,:))) ~= 0
                ctrs(i,:) = zeros(1,n);
            end
        end
        iter = iter + 1;
    end

    result = ctrs;
    function label = assign_labels(X, ctrs)
    [N,~]=size(X);
    [c,~]=size(ctrs);
    dist = zeros(N,c);
    for i = 1:c
        dist(:,i) = sum(bsxfun(@minus, X, ctrs(i,:)).^2, 2);
    end

    [~,label] = min(dist,[],2);
    end
  end

function d=eucld(ctrs1,ctrs)
    d=sqrt((ctrs1-ctrs).^2);
end
 
function x_filter=mfcc1(y,Fs)
    hz2mel = @( hz )( 2595*log(1+hz/700) );
    mel2hz = @( mel )( 700*exp(mel/2595)-700 );

    NFFT = 256;
    no = 35;
    Framesize = 160;

    f = Fs/2*linspace(0,1,NFFT);                %Freq axis

    if size(y,2)==2
        y = y(:,1);   
    end
    time = (1:numel(y))/Fs;


    a = 0.95;
    y1 = filter([1, -a], 1, y);                %Pre-emphasis
    F = buffer(y1,Framesize,Framesize/2);       %Framing




    F_n=F;
    en = (sum(power(F,2),1));                     %Energy per Frame
    en = en./max(en);                             %Normalized energy



    H = hamming(Framesize);                     %Generate Hamming Window
    W = gmultiply(F_n,H) ;                       %Windowing


    lfreq = 0;                                  %Low frequency
    hfreq = Fs/2;                               %Maximum frequency
    lmel = hz2mel(lfreq);
    hmel = hz2mel(hfreq);

    ft = 40;
    spacingMel=(hmel-lmel)/(NFFT*(ft+2));
    t1 = floor((hmel-lmel)/(ft+2));


    melScale=lmel:spacingMel:hmel;
    temp1=1:ft+2:size(melScale,2)-1;
    melaxis=melScale(:,1:ft+2:size(melScale,2)-1);
    freqaxis = mel2hz(melaxis);
    FilterPtMel=melScale(:,1:NFFT:size(melScale,2)-1);
    freqaxisFilt = mel2hz(FilterPtMel);

    lfM=FilterPtMel(1:ft);
    cfM=FilterPtMel(2:ft+1);
    ufM=FilterPtMel(3:ft+2);


    FilterWeights_mel=zeros(ft,NFFT);
    FilterArea=zeros(1,ft);

    for C = 1:ft
        FilterWeights_mel(C,:) = ((melaxis>lfM(C)&melaxis<=cfM(C)).*(melaxis-lfM(C))/(cfM(C)-lfM(C)))+...
            ((melaxis>cfM(C)&melaxis<ufM(C)).*(ufM(C)-melaxis)/(ufM(C)-cfM(C)));
        FilterArea(C)=0.5*(ufM(C)-lfM(C));
        trihtf(C)=FilterArea(C)*2./(mel2hz(ufM(C))-mel2hz(lfM(C)));
        FilterWeights_freq(C,:)=FilterWeights_mel(C,:).*trihtf(C);
    end

    % FilterWeights_freq=FilterWeights_mel.*

    for i = 1:ft
        [r,c] = min(abs(f-freqaxis(i)));
        f(c) = freqaxis(i);                 
    end


    figure(1);
    subplot 211
    plot(melaxis,FilterWeights_mel);
    %subplot 212
    %plot(freqaxis,FilterWeights_freq);



    % Filter bank design
    lf = freqaxis(1:ft);
    cf = freqaxis(2:ft+1);
    uf = freqaxis(3:ft+2);
    triHgt = 2./(uf-lf);
    % triHgt=ones(1,ft);
    FilterWeights = zeros(0,NFFT);



    for C = 1:ft
        FilterWeights(C,:) = ((f>lf(C)&f<=cf(C)).*triHgt(C).*(f-lf(C))/(cf(C)-lf(C)))+...
            ((f>cf(C)&f<uf(C)).*triHgt(C).*(uf(C)-f)/(uf(C)-cf(C)));
    end



    % Power Spectrum of each frame

    Y = abs(fft(W,NFFT));               % Power Spectrum of each frame
    fa1 = Fs/2*linspace(0,1,NFFT/2);

    % stem(abs(Y));

    Melspectrum = FilterWeights_freq*Y;      %filtered weighted spectral components
    logMag = log10(Melspectrum);        
    x = dct(logMag);                    % DCT of the log of Mel spectrum(cesptrum)
    % x = idct(logMag);
    x_filter = x(2:13,:);                      % liftering the cepstrals
 
end
 


