function snip()

%C=cell(200,1);
dirData = dir('/usr2/elizalde/NELS/users/rohan/fingerprinting/youtube_batch_1/**/*.wav');
disp(dirData);
%dirname1 = '';
 % find all the MP3 files
%dlist1 = dir(fullfile(dirname1, '*.wav'));

 % for  i=str2double('1'):str2double('1')+199
 %      C{i-str2double('1')+1}=dlist1(i).name;
 %   end

%IDs=C;
%nIDs=length(IDs);
%postpend='';
counter=1;
messi=1;
y_t=[];
for i = 1:length(dirData)
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
      disp(j)
    %sp = round((ld - qlen)*rand(1));
    Npts = length(d([1:qlen])); % Number of input time samples
    Noise = randn(1,Npts); % Generate initial noise; mean zero, variance one
    %disp(j);
    %Q{i,j} = d(sp + [1:qlen]);
        for k=1:1
            %k=1;
          %  Npts = length(d(sp + [1:qlen])); % Number of input time samples
           % Noise = randn(1,Npts); % Generate initial noise; mean zero, variance one
            Noise_Power=sum(abs(Noise).*abs(Noise));
            Signal_Power = sum(abs(d([1:qlen])).*abs(d([1:qlen])));
            K = (Signal_Power/Noise_Power)*10^(-snr(k)/10);
            New_Noise = sqrt(K)*Noise;
            Noise_Power=sum(abs(New_Noise).*abs(New_Noise));
            Initial_SNR = 10*(log10(Signal_Power./Noise_Power));
            %disp(Initial_SNR)
            if k==1
            d1=d([1:qlen]);
            %d2=round((ld-qlen)*rand(1));
            %d3=round((ld-qlen)*rand(1));
%            QW{1,k,:}=d1;
            targetSR=16000;
            fft_ms = 64;
            fft_hop = 32;
            nfft = round(targetSR/1000*fft_ms);
            S1=logfsgram(d1,nfft,targetSR,nfft,nfft-round(targetSR/1000*fft_hop));
            %S2=abs(specgram(d(d2+[1:qlen]),nfft,targetSR,nfft,nfft-round(targetSR/1000*fft_hop)));
            %S3=abs(specgram(d(d3+[1:qlen]),nfft,targetSR,nfft,nfft-round(targetSR/1000*fft_hop)));
            size(S1)
            S(counter,:)=S1(:);
            %S(counter+1,:)=S2(:);
            %S(counter+2,:)=S3(:);
            y_t(counter,1)=messi;
            %y_t(counter+1,1)=messi;
            %y_t(counter+2,1)=messi;
            disp(messi)
            counter=counter+1;
            
            else
            d2=d( [1:qlen]);% + New_Noise';
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
 
messi=messi+1;
end
%  else
%for p=1:3
%   Q{i,j,p}=[];
%end
    end
 % e%nd


%save('Q12.mat','QW')
%fileID1 = fopen('magic1.bin','w');
%fileID2 = fopen('magic2.bin','w');
%fwrite(fileID1,S,'float');
%fwrite(fileID2,y_t,'float');
%fclose(fileID1);
%fclose(fileID2);

save('snippets_from_query_cat_sound.mat','S','y_t')

