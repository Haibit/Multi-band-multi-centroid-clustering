function [W] = multi_centroid_EM(W,X,nsou,nfreq,Nb,Nc)
% Reference:
% [Wang L. Multi-band multi-centroid clustering based permutation alignment 
%  for frequency-domain blind speech separation[J].Digital Signal Processing, 
%  2014, 31: 79-92.]

% kelsey Leng  20190110

% Parameters:
% nsou£º  source number
% nfreq£º number of frequencies
% W£º     unmixing matrix£¬nsou ¡Á nsou ¡Á nfreq
% X£º     mixture signal, nsou ¡Á number of time frame ¡Á nfreq
% Nb£º    number of frequency bands
% Nc£º    number of centroid in step2


pe = perms(1:nsou);  % all combinations
pe = pe(end:-1:1,:);
numpe = size(pe,1);
irow = zeros(1,nsou);
irowck = zeros(1,Nc);
row = zeros(1,numpe);
N = size(X,2);


%% Stage 1  Full-band permutaiton alignment with one centroid clustering
oldrecordPe = zeros(1,nfreq);
recordPe = zeros(1,nfreq);
ite = 0;
while 1>0
    ite = ite+1;
    v = power_ratio(W,X,nsou,nfreq,N); % Power ratio(Time activity sequency)
    c = centroid(v);  % Calculate centroid
    for k = 1:nfreq
        for i = 1:numpe
            for insou = 1:nsou
                irow(insou) = correlation(transpose(v(pe(i,insou),:,k)),c(:,pe(1,insou)));
            end
            row(i) = sum(irow);
        end
        [~,maxindex] = max(row);
        W(:,:,k) = W(pe(maxindex,:),:,k);
        recordPe(k) = maxindex;
    end
    if isequal(recordPe, oldrecordPe)
        break;
    else
        oldrecordPe = recordPe;
    end
end


%% Stage 2  Permutation alignment with M-centroid clustering inside one subband
[indexBlock,Len] = auxiliary.divide_Equal(nfreq,Nb);
% %{
for iband = 1:Nb
    % one centoid clustering
    clear oldrecordPe recordPe
    oldrecordPe = zeros(1,Len(iband));
    recordPe = zeros(1,Len(iband));
    ite1 = 0;
    while 1>0
        ite1 = ite1+1;
        v = power_ratio(W(:,:,indexBlock(iband,1):indexBlock(iband,2)),X(:,:,indexBlock(iband,1):indexBlock(iband,2)),nsou,Len(iband),N);
        c = centroid(v);  % Calculate centroid
        for k = indexBlock(iband,1):indexBlock(iband,2)
            for i = 1:numpe
                for insou = 1:nsou
                    irow(insou) = correlation(v(pe(i,insou),:,k-indexBlock(iband,1)+1),c(:,pe(1,insou)));
                end
                row(i) = sum(irow);
            end
            [~,maxindex] = max(row);
            W(:,:,k) = W(pe(maxindex,:),:,k);
            recordPe(k-indexBlock(iband,1)+1) = maxindex;
        end
        if isequal(recordPe, oldrecordPe)
            break;
        else
            oldrecordPe = recordPe;
        end
    end
    
    
    %  multi-centroid clustering
    clear oldrecordPe recordPe
    oldrecordPe = zeros(1,Len(iband));
    recordPe = zeros(1,Len(iband));
    ite2 = 0;
    while 1>0
        ite2 = ite2+1;
        v = power_ratio(W(:,:,indexBlock(iband,1):indexBlock(iband,2)),X(:,:,indexBlock(iband,1):indexBlock(iband,2)),nsou,Len(iband),N);
        for insou = 1:nsou
            [idck{insou},Ck(insou,:,:)] = kmeans(transpose(squeeze(v(insou,:,:))),Nc);
        end
        
        for k = indexBlock(iband,1):indexBlock(iband,2)
            for i = 1:numpe
                for insou = 1:nsou
                    for ick = 1:Nc
                        irowck(ick) = correlation(transpose(v(pe(i,insou),:,k-indexBlock(iband,1)+1)),Ck(pe(1,insou),ick,:));
                    end
                    irow(insou) = max(irowck);
                end
                row(i) = sum(irow);
            end
            [~,maxindex] = max(row);
            W(:,:,k) = W(pe(maxindex,:),:,k);
            recordPe(k-indexBlock(iband,1)+1) = maxindex;
        end
        if isequal(recordPe, oldrecordPe)
            break;
        else
            oldrecordPe = recordPe;
        end
    end
    
end


%}

%% Stage 3  Permutation alignment between subbands
% %{
for iband = 1:Nb-1
    pband = iband+1; % The band to be permuted
    v = power_ratio(W(:,:,indexBlock(iband,1):indexBlock(pband,2)),X(:,:,indexBlock(iband,1):indexBlock(pband,2)),nsou,Len(iband)+Len(pband),N);
    cband(:,:,1) = centroid(v(:,:,1:Len(iband)));
    cband(:,:,2) = centroid(v(:,:,Len(iband)+1:end));
    for i = 1:numpe
        for insou = 1:nsou
            irow(insou) = correlation(cband(:,pe(1,insou),1),cband(:,pe(i,insou),2));
        end
        row(i) = sum(irow);
    end
    [~,maxindex] = max(row);
    W(:,:,indexBlock(pband,1):indexBlock(pband,2)) = W(pe(maxindex,:),:,indexBlock(pband,1):indexBlock(pband,2));
end

%}
end


%% *************************************************************************
function [row] = correlation(vi,vj)
% Calculate correlation coefficient between two time activity sequences
coe = corrcoef(vi,vj);
row = coe(1,2);
end


function [v] = power_ratio(W,X,nsou,nfreq,N)
S = zeros(nsou,N,nfreq);
v1 = zeros(nsou,N,nfreq);
v = zeros(nsou,N,nfreq);
for k=1:nfreq
    S(:,:,k) = W(:,:,k)*X(:,:,k);
end
% Power ratio(Time activity sequency)
for m = 1:N
    for k = 1:nfreq
        A = inv(W(:,:,k));
        for i = 1:nsou
            v1(i,m,k) = norm(A(:,i)*S(i,m,k),2);
        end
        v2 = sum(v1(:,m,k));
        v(:,m,k) = v1(:,m,k)./v2;
    end
end
end


function [c] = centroid(v)
[nsou,N,Lenv] = size(v);
c = zeros(N,nsou);
for i = 1:nsou
    c(:,i) = sum(v(i,:,:),3)/Lenv;
end
end