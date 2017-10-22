function snip()

%C=cell(200,1);
counter=1;
y_t=[];
path='/usr2/elizalde/NELS/users/rohan/fingerprinting/youtube_all/youtube_batch_1/';
dirData12 = dir('/usr2/elizalde/NELS/users/rohan/fingerprinting/youtube_all/youtube_batch_1/**/*.wav');
%disp(dirData12)
cqw=[]
for ji=1:length(dirData12)
%disp(dirData12(i).folder)
a=strsplit(dirData12(ji).folder,'/');
b=a(10);
disp(b{1})
dir1=strcat(path,b{1},'/*.wav');
if strcmp(b{1},cqw)==0
cqw=b{1};
dirData=dir(dir1);


i = randi(length(dirData));
 disp(i) 
 %id = IDs{i};
  fname = [dirData(i).folder,'/',dirData(i).name];
  [pth,nm,ext] = fileparts(fname);
  if strcmp(ext,'.wav') == 1
    [d,sr] = audioread(fname);
    time=length(d)/sr ;
    %disp(time);
  end
  if size(d,2) == 2
    % convert to mono if stereo
    d = mean(d,2);
  end
  % choose random excerpt
  ld = length(d);
  snippets=[2];
  snr=[0,5,10];
   j=1;
       qlen = round(snippets(j)* sr);
 %disp(i);
    if(ld-qlen)>=0
      %disp(j)
    sp = round((ld - qlen)*rand(1));
    Npts = length(d([1:qlen])); % Number of input time samples
    Noise = randn(1,Npts); % Generate initial noise; mean zero, variance one
    
        for k=1:1
          
            Noise_Power=sum(abs(Noise).*abs(Noise));
            Signal_Power = sum(abs(d([1:qlen])).*abs(d([1:qlen])));
            K = (Signal_Power/Noise_Power)*10^(-snr(k)/10);
            New_Noise = sqrt(K)*Noise;
            Noise_Power=sum(abs(New_Noise).*abs(New_Noise));
            Initial_SNR = 10*(log10(Signal_Power./Noise_Power));
            if k==1
            d1=d([1:qlen]);
           
            targetSR=16000;
            fft_ms = 64;
            fft_hop = 32;
            nfft = round(targetSR/1000*fft_ms);
            S1=logfsgram(d1,nfft,targetSR,nfft,nfft-round(targetSR/1000*fft_hop));
            
            S(counter,:)=S1(:);
            
            y_t(counter,1)=ji+i-1;
            
            counter=counter+1;
            
            else
            d2=d(sp + [1:qlen]) + New_Noise';
            QW{1,k,:}=d2;
            targetSR=16000;
            fft_ms = 64;
            fft_hop = 32;
            nfft = round(targetSR/1000*fft_ms);
            S1= logfsgram(d2,nfft,targetSR,nfft,nfft-round(targetSR/1000*fft_hop));
            size(S1)
            S(counter,:)=S1(:);
            y_t(counter,1)=messi;
            disp(messi)
            counter=counter+1;   
            end              
	    %Z= d(sp + [1:qlen]) + New_Noise';
            %awgn(d(sp + [1:qlen]),snr(k));
end
 
%messi=messi+1;
else
disp('missed')
disp(fname)
end

%a_p=strcat('filesmat/snippets_from_intermediate_outputs','_',b{1},'.mat');

%save(a_p,'S')
%clearvars S
end
end

save('76_queries.mat','S','y_t')

end
