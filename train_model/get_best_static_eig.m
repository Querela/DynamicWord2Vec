clc
clear

%%

load initial_cooccur_freq.mat

freq = double(freq);
cooccur = cooccur + diag(freq);
cooccur = cooccur*sum(freq) ./ (freq*freq');

pmi = log(max(cooccur,0)) ;
pmi(isinf(pmi)) = 0;
asdf
%%

opts.issym = true;
opts.isreal = true;
[X, D] = eigs(pmi,50,'la', opts);

'done'
save -v7.3 eigs_static X D
%%
D = diag(D);
D = max(D,0);
emb = X*diag(sqrt(D));

save emb_static emb