% LibTLDA: installation script

% Download packages
urlwrite('http://www.cs.ubc.ca/~schmidtm/Software/minFunc_2012.zip', 'minFunc_2012.zip')
urlwrite('https://github.com/cjlin1/libsvm/archive/master.zip', 'libSVM.zip')

% Unzip packages
unzip('minFunc_2012');
unzip('libSVM');

% Compile for libsvm
cd libsvm-master/matlab
make
cd ../../

% Add all packages to path
addpath(genpath('./minFunc_2012'));
addpath(genpath('./libSVM-3.22'));
addpath(genpath('../matlab'));
savepath
