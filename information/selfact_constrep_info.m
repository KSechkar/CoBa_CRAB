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
teval=[tspan(1), 1.0, tspan(end)-1.0, tspan(end)];
ss_check_tol=1e-1;

% ODE integration tolerances
tol_settings=odeset('RelTol', 1e-13, 'AbsTol', 1e-13);

% initial condition for the dynamics variables
Y0=[par('M')/(2*par('n_r'));
    par('M')/(2*par('n_o'));
    0; 0; 0; 0; 0;
    ];

% set the switch parameters for which we calculate sensitivities
U=[par('q_switch')./(par('q_r')+par('q_o'));
    par('q_ofp')./(par('q_r')+par('q_o'));
    par('n_switch');
    par('n_ofp');
    par('mu_ofp');
    par('baseline_switch');
    par('K_switch');
    par('eta_switch');
    ];

%% RUN the simulations

% initialise storage
Q_constrep_ss=zeros(size(constrep_qs)); % NORMALISED RC factors to be recorded
% outputs
ofp_mature_ss=zeros(size(constrep_qs)); % switch OFP
ofp2_mature_ss=zeros(size(constrep_qs));    % reporter OFP
% relevant (i.e. output) sensitivites - element indexing is (experiment, output, parameter)
DYDU_relevant=zeros(size(constrep_qs,2),2,size(U,1));

for i=1:size(constrep_qs,2)
    % set the new parameters
    par('q_ofp2')=constrep_qs(i);
    par('mu_ofp2')=constrep_mus(i);
    
    % simulate
    [T, Y, DYDU]=sens_sys('selfact_constrep_ode', ...
        teval, ...
        selfact_constrep_append_par(Y0, par), ...
        tol_settings,U);
    
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
    % record relevant sensitivities
    DYDU_relevant(i,:,:)=DYDU(end,6:7,:);
end

%% RUN the simulations with fits

load('constrep_u_map.mat')
U_fit=U_selfact_constrep_map;

% initialise storage
Q_constrep_ss_fit=zeros(size(constrep_qs)); % NORMALISED RC factors to be recorded
% outputs
ofp_mature_ss_fit=zeros(size(constrep_qs)); % switch OFP
ofp2_mature_ss_fit=zeros(size(constrep_qs));    % reporter OFP

for i=1:size(constrep_qs,2)
    % set the new parameters
    par('q_ofp2')=constrep_qs(i);
    par('mu_ofp2')=constrep_mus(i);
    
    % simulate
    [T, Y, DYDU]=ode15s('selfact_constrep_ode', ...
        tspan, ...
        selfact_constrep_append_par(Y0, par), ...
        tol_settings,U_fit);
    
    % check if the steady state has been reached, warn if not
    if(~ss_check(Y(end-1,1:7),Y(end,1:7),ss_check_tol))
        disp(['Steady state not reached for q_constrep = ', ...
            num2str(constrep_qs(i))])
    end
    
    % reocrd normalised RC factor
    Q_constrep_ss_fit(i) = constrep_qs(i)/(par('q_r')+par('q_o'));
    % record outputs
    ofp_mature_ss_fit(i) = Y(end,6);  % switch OFP
    ofp2_mature_ss_fit(i) = Y(end,7); % reporter OFP
end

%% PLOT the steady-state readings

F = figure('Position',[0 0 275 204]);
set(F, 'defaultAxesFontSize', 8)
set(F, 'defaultLineLineWidth', 1.5)
hold on

plot(Q_constrep_ss, ofp_mature_ss,'.')
plot(Q_constrep_ss_fit, ofp_mature_ss_fit,'.')

xlabel('Q_{constrep}, reporter RC factor','FontName','Arial')
ylabel('ofp_{mature}, switch output','FontName','Arial')

%% SAVE the steady-state readings

ofp_mature_data=ofp_mature_ss;
ofp2_mature_data = ofp2_mature_ss;

save('constrep_data.mat', ...
    'ofp_mature_data', 'ofp2_mature_data', 'par', 'constrep_mus', 'constrep_qs')

%% FUNCTION: CHECK IF THE STEADY STATE HAS BEEN REACHED
function is_steady = ss_check(x_penult, x_last, ss_tol)
    is_steady = sum(abs(x_last-x_penult)) < ss_tol;
end

%%
function sos=selfact_constrep_cost(constrep_qs, ...
    ofp_mature_data, ofp2_mature_data, ...
    constrep_mus, ...
    par,U)
    tspan=[0.0, 72.0];  % time span of the entire simulation
    ss_check_tol=1e-1;
    
    % ODE integration tolerances
    tol_settings=odeset('RelTol', 1e-13, 'AbsTol', 1e-13);
    
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
    
    for i=1:size(constrep_qs,2)
        % set the new parameters
        par('q_ofp2')=constrep_qs(i);
        par('mu_ofp2')=constrep_mus(i);
        
        % simulate
        [~, Y]=ode15s('selfact_constrep_ode', ...
            tspan, ...
            append_par(Y0, par), ...
            tol_settings,U);
        
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
    end

    sos=sum((ofp_mature_ss-ofp_mature_data).^2+(ofp2_mature_ss-ofp2_mature_data).^2);
    
end