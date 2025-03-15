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

%% DEFINE THE REPORTER RC FACTORS CONSIDERED
constrep_qs = linspace(45.0,4e4,30);
constrep_mus = par('mu_ofp2').*ones(size(constrep_qs));

%% SET SIMULATION PARAMETERS

% times at which Y and DY/DU are recorded
% (zero, one hour before the end of simulation, end of simulation - the
% last two allow to check if the steady state has been reached)
tspan=[0.0, 72.0];  % time span of the entire simulation
teval=[tspan(1), tspan(end)-1.0, tspan(end)];
ss_check_tol=1e-1;

% initial condition for the dynamics variables
Y0=[par('M')/(2*par('n_r'));
    par('M')/(2*par('n_o'));
    0; 0; 0; 0; 0;
    ];

%% RUN the simulations

% initialise storage
Q_constrep_ss=zeros(size(constrep_qs)); % NORMALISED RC factors to be recorded
% outputs
ofp_mature_ss=zeros(size(constrep_qs)); % switch OFP
ofp2_mature_ss=zeros(size(constrep_qs));    % reporter OFP
% sensitivities
dofp_dQswitch=zeros(size(constrep_qs)); % sensitivity of switch OFP to switch gene RC factor
dofp_dQofp=zeros(size(constrep_qs)); % sensitivity of switch OFP to OFP RC factor
dofp2_dQswitch=zeros(size(constrep_qs)); % sensitivity of reporter OFP to switch gene RC factor
dofp2_dQofp=zeros(size(constrep_qs)); % sensitivity of reporter OFP to OFP RC factor

for i=1:size(constrep_qs,2)
    % set the new parameters
    par('q_ofp2')=constrep_qs(i);
    par('mu_ofp2')=constrep_mus(i);

    % set the U vector entries to appropriate parameters
    U=[par('q_switch'); par('q_ofp'); par('q_ofp2')]./(par('q_r')+par('q_o'));
    
    % simulate
    [T, Y, DYDU]=sens_sys('selfact_constrep_ode', ...
        teval, ...
        append_par(Y0, par), ...
        [],U);
    
    % check if the steady state has been reached, warn if not
    if(~ss_check(Y(end-1,1:7),Y(end,1:7),ss_check_tol))
        disp(['Steady state not reached for q_constrep = ', ...
            num2str(constrep_qs(i))])
    end
    
    % reocrd normalised RC factor
    Q_constrep_ss(i) = constrep_qs(i)/(par('q_r')+par('q_o'));
    % record outputs
    ofp_mature_ss(i) = Y(end,6);  % switch OFP
    ofp2_mature_ss(i) = Y(end,7); % reporter OFP
    % record sensitivities
%     dofp_dQswitch(i) = DYDU(end,6,1);   % sensitivity of switch OFP to switch gene RC factor
%     dofp_dQofp(i) = DYDU(end,6,2);         % sensitivity of switch OFP to OFP RC factor
    dofp2_dQswitch(i) = DYDU(end,7,1);     % sensitivity of reporter OFP to switch gene RC factor
    dofp2_dQofp(i) = DYDU(end,7,2);        % sensitivity of reporter OFP to OFP RC factor
end

%% PLOT the steady-state readings

F = figure('Position',[0 0 275 204]);
set(F, 'defaultAxesFontSize', 8)
set(F, 'defaultLineLineWidth', 1.5)
hold on

plot(Q_constrep_ss, ofp_mature_ss,'.')

xlabel('Q_{constrep}, reporter RC factor','FontName','Arial')
ylabel('ofp_{mature}, switch output','FontName','Arial')

%% CALCULATE the Fisher Information Matrix for q_switch and q_ofp

% specify the measurement noise's stdev for both OFPs
sigma=50;

% find entries element-wise
FIM_Qswitch_Qswitch = 1./(sigma.^2) .*sum( ...
    dofp_dQswitch.*dofp_dQswitch + ...
    dofp2_dQswitch.*dofp2_dQswitch);

FIM_Qswitch_Qofp = 1./(sigma.^2) .*sum( ...
            dofp_dQswitch.*dofp_dQofp + ...
            dofp2_dQswitch.*dofp2_dQofp);

FIM_Qofp_Qofp = 1./(sigma.^2) .*sum( ...
            dofp_dQofp.*dofp_dQofp + ...
            dofp2_dQofp.*dofp2_dQofp);

% gather the entires into a single matrix
FIM=[FIM_Qswitch_Qswitch, FIM_Qswitch_Qofp;
    FIM_Qswitch_Qofp, FIM_Qofp_Qofp];

disp(FIM)

%% CALCULATE the FIM's determinant

disp(det(FIM))

%% FUNCTION: append parameters to the initial condition
function Y0_with_par = append_par(Y0, par)
Y0_with_par=[
    Y0;
    % host cell parameters
    par('M');
    par('e');
    par('q_r');
    par('n_r');
    par('q_o');
    par('n_o');
    % self-activating switch parameters - save for RC factors
    par('n_switch');
    par('n_ofp');
    par('mu_ofp');
    par('baseline_switch');
    par('K_switch');
    par('I_switch');
    par('eta_switch');
    % constitutive reporter parameters - save for the RC factor
    par('n_ofp2');
    par('mu_ofp2');
    ];
end

%% FUNCTION: CHECK IF THE STEADY STATE HAS BEEN REACHED
function is_steady = ss_check(x_penult, x_last, ss_tol)
    is_steady = sum(abs(x_last-x_penult)) < ss_tol;
end