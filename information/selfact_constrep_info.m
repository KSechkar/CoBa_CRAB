%% selfact_constrep_info.m - evaluating the informativeness of observing a self-activating genetic switch with constitut#ive reporters

%% CLEAR parameters, add paths of all files

addpath(genpath('sens_analysis_1.2'))

clear
close all

%% DEFINE SYSTEM PARAMETERS

% initialise parameter storage
par = containers.Map('KeyType', 'char', ...
    'ValueType', 'double');

% host cell parameters
par('M') = 1.19e9;  % mass of protein in the cell (aa)
par('e') = 66077.664;  % translation elongation rate (aa/h)
par('q_r') = 13005.314453125;  % resource demand of ribosomal genes
par('n_r') = 7459.0;  % protein length (aa) of ribosomes
par('q_o') = 61169.44140625;  % resource demand of other native genes
par('n_o') = 300.0;  % protein length (aa) of other native proteins

% self-activating switch parameters
par('q_switch') = 125.0;  % resource competition factor for the switch gene
par('q_ofp') = 100*par('q_switch'); % RC factor for the switch's fluorescent output gene
par('n_switch') = 300.0; % switch protein length
par('n_ofp') = 300.0; % switch OFP length
par('baseline_switch') = 0.05;  % baseline expression of switch gene
par('K_switch') = 250.0;    % half-saturation constant for the switch protein's self-regulation
par('I_switch') = 0.1;  % share of switch proteins bound by an inducer molecule

% default constitutive reporter parameters
par('q_ofp2') = 3000.0;  % resource competition factor for the const. reporter output fluorescent protein
par('n_ofp2') = 300.0; % reporter OFP length

% bake system parameters into the ODE function
sys_ode=@(t,X,U) selfact_constrep_ode(t,X,U,par);

%% SET SIMULATION PARAMETERS

tspan=[0.0,24.0];
Y0=[par('M')/(2*par('n_r'));
    par('M')/(2*par('n_o'));
    0; 0; 0; 0; 0;
    ];

%% RUN THE SIMULATION

% set the U vector entries to appropriate parameters
U=[par('q_switch'); par('q_ofp'); par('q_ofp2')]./(par('q_r')+par('q_o'));

% simulate
[T, Y, DYDY]=sens_sys(sys_ode,tspan,Y0,[],U)

