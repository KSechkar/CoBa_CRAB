%% dream_fit.m
% fit parameters to experimental data using DREAM

% NOTE:
% due to the nature of the package, the simulator and simulation parameters
% are defined within the function file dream_model.m and not here

%% CLEAR parameters, add paths of all files

clear
close all

%% true parameters
load("constrep_data.mat")

U_true=[par('q_switch')./(par('q_r')+par('q_o'));
    par('q_ofp')./(par('q_r')+par('q_o'));
    par('n_switch');
    par('n_ofp');
    par('mu_ofp');
    par('baseline_switch');
    par('K_switch');
    par('eta_switch');
    ].';

%% DEFINE starting parameter values

% record all values in a row vector
U0=1.05*U_true;

% initial sampling distirbution covariances (parameters assumed independent)
% equal to (1/4 mean)^2, where the mean is the 'initial value'
covar0=zeros(size(U0,1),size(U0,1));
for i=1:size(U0,1)
    covar0(i,i)=(0.25.*U0(i)).^2;
end

%% SET UP the DREAM sampler

% User-defined problem settings
DREAMPar.d = size(U0,2); % dimension of the problem
DREAMPar.N = 10; % number of Markov chains sampled in parallel
DREAMPar.T = 1000; % number of generations
DREAMPar.lik = 2; % 'model' function outputs numerical steady-state predictions
DREAMPar.outlier='iqr'; % handling outliers - 

% Backing up the progress every 10 steps
DREAMPar.save='yes';
DREAMPar.steps=10;  

% Initial sampling distribution
Par_info.prior = 'normal'; % Sample initial state of chains from prior distribution
Par_info.mu=U0; % means of prior distribution
Par_info.cov=covar0; % covariances of prior distribution (parameters distributed independently)

% Boundaries of parameter value domains
Par_info.min = U_true/50;
Par_info.max = U_true*50;
Par_info.boundhandling = 'fold'; % handle samples out of bounds by imagaing the domain as a torus, upholding MCMC detailed balance

% do not output diagnostic
DREAMPar.diagnostics='no';

% Parallel computing to speed the simulation up
DREAMPar.parallel = 'yes'; 
DREAMPar.CPU=10;

DREAMPar.restart='no';

Meas_info.Y=[];

%% RUN the sampler
[chain,output,fx]=DREAM('selfact_constrep_sos', ...  % name of the file with the function that takes parameter vector and spits out the log-likelihood of measurements ofr the underlying model with these parameters
    DREAMPar,Par_info,Meas_info);