%% selfact_constrep_sos.m

%%
function sos=selfact_constrep_sos(U)
    %% LOAD experimental data
    persistent ofp_mature_data ofp2_mature_data par constrep_mus constrep_qs
    
    if(isempty(ofp_mature_data))
        load('constrep_data.mat')
    end
    
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
            selfact_constrep_append_par(Y0, par), ...
            tol_settings,U);
        
        % reocrd normalised RC factor
        Q_constrep_ss(i) = constrep_qs(i)/(par('q_r')+par('q_o'));
        % record outputs
        ofp_mature_ss(i) = Y(end,6);  % switch OFP
        ofp2_mature_ss(i) = Y(end,7); % reporter OFP
    end

    sos=sum((ofp_mature_ss-ofp_mature_data).^2+(ofp2_mature_ss-ofp2_mature_data).^2);
end