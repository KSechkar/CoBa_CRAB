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

%% DEFINE THE CONTROL REFERENCES CONSIDERED

refs = linspace(0.0,378528.87513669,30);

%% SET SIMULATION PARAMETERS

% times at which Y and DY/DU are recorded
% (zero, one hour before the end of simulation, end of simulation - the
% last two allow to check if the steady state has been reached)
tspan=[0.0, 72.0];  % time span of the entire simulation
teval=[tspan(1), tspan(end)-1.0, tspan(end)];
ss_check_tol=1e-1;

% ODE integration tolerances
tol_settings=odeset('RelTol', 1e-13, 'AbsTol', 1e-13);

% initial condition for the dynamics variables
Y0=[par('M')/(2*par('n_r'));
    par('M')/(2*par('n_o'));
    0; 0; 0; 0; 0; 0;
    ];

% set the switch parameters for which we calculate sensitivities
U=[par('q_switch')./(par('q_r')+par('q_o'));
    par('q_ofp')./(par('q_r')+par('q_o'));
    par('n_switch');
    par('n_ofp');
    par('mu_ofp');
    par('baseline_switch');
    par('K_switch');
    par('I_switch');
    par('eta_switch');
    ];

%% RUN the simulations

% initialise storage
Q_probe_ss=zeros(size(refs)); % NORMALISED RC factors to be recorded
u_ss=zeros(size(refs)); % control inputs
% outputs
ofp_mature_ss=zeros(size(refs)); % switch OFP
b_mature_ss=zeros(size(refs));    % probe's burdensome OFP
% relevant (i.e. output) sensitivites - element indexing is (experiment, output, parameter)
DYDU_relevant=zeros(size(refs,2),2,size(U,1));

for i=1:size(refs,2)
    % set the new parameters
    par('ref')=refs(i);
    
    % simulate
    [T, Y, DYDU]=sens_sys('selfact_cbc_ode', ...
        teval, ...
        append_par(Y0, par), ...
        [],U);
    
    % check if the steady state has been reached, warn if not
    if(~ss_check(Y(end-1,1:7),Y(end,1:7),ss_check_tol))
        disp(['Steady state not reached for q_constrep = ', ...
            num2str(refs(i))])
    end
    
    % record normalised RC factor
    u_unclipped = par('Kp')*(par('ref')-Y(end,7));  % get the proportional feedback signal as calculated
    u=max(min(u_unclipped,par('max_u')),0.0);    % clip to feasible range
    F_b = F_b_calc(Y(end,5), u,...
        par('baseline_tai_dna'), par('K_ta_i'), par('K_tai_dna'), par('eta_tai_dna'));
    Q_probe_ss(i) = (par('q_ta')+F_b*par('q_b'))/(par('q_r')+par('q_o'));
    u_ss(i)=u;

    % record outputs
    ofp_mature_ss(i) = Y(end,7);  % switch OFP
    b_mature_ss(i) = Y(end,8); % probe's burdensome OFP
    % record relevant sensitivities
    DYDU_relevant(i,:,:)=DYDU(end,7:8,:);
end

%% PLOT the steady-state readings

F = figure('Position',[0 0 275 204]);
set(F, 'defaultAxesFontSize', 8)
set(F, 'defaultLineLineWidth', 1.5)
hold on

plot(Q_probe_ss, ofp_mature_ss,'.')


xlabel('Q_{probe}, RC from the probe','FontName','Arial')
ylabel('ofp_{mature}, switch output','FontName','Arial')
% set(gca, 'XScale', 'log')
% xlim([0.01 11])

%% CALCULATE the Fisher Information Matrix for q_switch and q_ofp

% specify the measurement noise's stdev for both OFPs
sigma=50;

FIM = zeros(size(U,1),size(U,1));
for param1_cntr=1:size(U,1)
    for param2_cntr=1:size(U,1)
        sum_output_sensitivities=0;
        for exp_cntr=1:size(refs,2)
            for output_cntr=1:2
                sum_output_sensitivities = sum_output_sensitivities + 1./(sigma.^2) .*(...
                    DYDU_relevant(exp_cntr,output_cntr,param1_cntr).* ...
                    DYDU_relevant(exp_cntr,output_cntr,param2_cntr) ...
                    );
            end
        end
        FIM(param1_cntr,param2_cntr)=sum_output_sensitivities;
    end
end

disp(FIM(2:end,2:end))

%% CALCULATE the FIM's determinant

cond(FIM(1:2,1:2))

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
    % CBC probe parameters
    par('q_ta')/(par('q_r')+par('q_o'));
    par('q_b')/(par('q_r')+par('q_o'));
    par('n_ta');
    par('n_b');
    par('mu_b');
    par('baseline_tai_dna');
    par('K_ta_i');
    par('K_tai_dna');
    par('eta_tai_dna');
    % controller parameters
    par('Kp');
    par('max_u');
    par('ref');
    ];
end

%% FUNCTION: CHECK IF THE STEADY STATE HAS BEEN REACHED
function is_steady = ss_check(x_penult, x_last, ss_tol)
    is_steady = sum(abs(x_last-x_penult)) < ss_tol;
end

%% FUNCTION: CHECK IF 