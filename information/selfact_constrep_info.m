%% selfact_constrep_info.m - evaluating the informativeness of observing a self-activating genetic switch with constitut#ive reporters

%% CLEAR parameters, add paths of all files

addpath(genpath('sens_analysis_1.2'))

clear
% close all

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
par('mu_ofp') = 1 / (13.6 / 60); % switch OFP maturation rate
par('baseline_switch') = 0.05;  % baseline expression of switch gene
par('K_switch') = 250.0;    % half-saturation constant for the switch protein's self-regulation
par('I_switch') = 0.1;  % share of switch proteins bound by an inducer molecule
par('eta_switch') = 2; % cooperativity coefficicient of switch protein-DNA binding

% default constitutive reporter parameters
par('q_ofp2') = 3000.0;  % resource competition factor for the const. reporter output fluorescent protein
par('n_ofp2') = 300.0; % reporter OFP length
par('mu_ofp2') = 1 / (13.6 / 60); % reporter OFP maturation rate

%% SAVE system parameters
save('par_data.mat', 'par')

%% DEFINE the fitted parameter vector

U=[par('q_switch')./(par('q_r')+par('q_o'));
    par('q_ofp')./(par('q_r')+par('q_o'));
    par('n_switch');
    par('baseline_switch');
    par('K_switch');
    par('eta_switch');
    ];

%% LOAD the simulation data
load('constrep_data.mat')
Q_constrep_ss=Qdash_moi_ss;

%% GET the actual Q'_moi values for Q_moi values from data

Qdash_moi_ss_actual=Qdash_moi_from_Q_moi(Q_moi_ss, par, 1.05*U);

%% GET THE Q'_moi values according to the parameter fit

load('constrep_u_map.mat')
Qdash_moi_ss_map=Qdash_moi_from_Q_moi(Q_moi_ss, par, U_selfact_constrep_map);
disp(selfact_constrep_sos(U_selfact_constrep_map))
disp(selfact_constrep_sos(U))

%% PLOT the steady-state readings

F = figure('Position',[0 0 275 204]);
set(F, 'defaultAxesFontSize', 8)
set(F, 'defaultLineLineWidth', 1.5)
hold on

plot(Q_constrep_ss, Q_moi_ss,'.')
plot(Qdash_moi_ss_actual, Q_moi_ss, '-')
plot(Qdash_moi_ss_map, Q_moi_ss, '-')

xlabel('Q_{constrep}, reporter demand','FontName','Arial')
ylabel('Q_{moi}, MOI demand','FontName','Arial')
