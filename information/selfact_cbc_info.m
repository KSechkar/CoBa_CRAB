%% selfact_constrep_info.m - evaluating the informativeness of observing a self-activating genetic switch with CBC

%% CLEAR parameters, add paths of all files

addpath(genpath('sens_analysis_1.2'))

clear
% close all

%% DEFINE SYSTEM PARAMETERS

% initialise parameter storage
par = containers.Map('KeyType', 'char', ...
    'ValueType', 'double');

% host cell parameters
par('M') = 1.19e9;              % mass of protein in the cell (aa)
par('e') = 66077.664;           % translation elongation rate (aa/h)
par('q_r') = 13005.314453125;   % resource demand of ribosomal genes
par('n_r') = 7459.0;            % protein length (aa) of ribosomes
par('q_o') = 61169.44140625;    % resource demand of other native genes
par('n_o') = 300.0;             % protein length (aa) of other native proteins

% self-activating switch parameters
par('q_switch') = 125.0;            % resource competition factor for the switch gene
par('q_ofp') = 100*par('q_switch'); % RC factor for the switch's fluorescent output gene
par('n_switch') = 300.0;            % switch protein length
par('n_ofp') = 300.0;               % switch OFP length
par('mu_ofp') = 1 / (13.6 / 60);    % switch OFP maturation rate
par('baseline_switch') = 0.05;      % baseline expression of switch gene
par('K_switch') = 250.0;            % half-saturation constant for the switch protein's self-regulation
par('I_switch') = 0.1;              % share of switch proteins bound by an inducer molecule
par('eta_switch') = 2;              % cooperativity coefficicient of switch protein-DNA binding

% CBC probe parameters
par('q_ta') = 45.0;             % resource competition factor for the probe's transcription activation factor
par('q_b') = 6e4;               % resource competition factor for the probe's burdensome OFP
par('n_ta') = 300.0;            % probe's transcription activator protein length
par('n_b') = 300.0;             % probe's burdensome OFP length
par('mu_b') = 1 / (13.6 / 60);  % probe's burdensome OFP maturation rate
par('baseline_tai_dna') = 0.01; % baseline expression of the burdnesome gene
par('K_ta_i') = 100.0;          % half-saturation constant for the binding between the inducer and the activator protein
par('K_tai_dna') = 100.0;       % half-saturation constant for the binding between the protein-inducer complex and the promoter DNA
par('eta_tai_dna') =  2.0;      % cooperativity coefficicient of complex-DNA binding

% controller parameters - assuming no delay as the ultimate steady state is not affected by it
par('Kp') = -0.0005;        % proprotional feedback gain
par('max_u') = 1000.0;      % maximum inducer concentration which can be supplied

%% DEFINE the fitted parameter vector

U=[par('q_switch')./(par('q_r')+par('q_o'));
    par('q_ofp')./(par('q_r')+par('q_o'));
    par('n_switch');
    par('baseline_switch');
    par('K_switch');
    par('eta_switch');
    ];

%% LOAD the simulation data
load('cbc_data.mat')
Q_probe_ss=Qdash_moi_ss;

%% GET the actual Q'_moi values for Q_moi values from data

Qdash_moi_ss_actual=Qdash_moi_from_Q_moi(Q_moi_ss, par, U);

%% PLOT the steady-state readings

F = figure('Position',[0 0 275 204]);
set(F, 'defaultAxesFontSize', 8)
set(F, 'defaultLineLineWidth', 1.5)
hold on

plot(Q_probe_ss, Q_moi_ss,'.')
plot(Qdash_moi_ss_actual, Q_moi_ss, '-')

xlabel('Q_{constrep}, reporter demand','FontName','Arial')
ylabel('Q_{moi}, MOI demand','FontName','Arial')

