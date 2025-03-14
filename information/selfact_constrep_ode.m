% selfact_constrep_ode.m - ODE for simulating measurements with
% constitutive reporters

function dxdt=selfact_constrep_ode( ...
    t, ...      % time
    X, ...      % state vector
    U, ...      % parameters to find sensitivities for
    par ...     % other parameters
    )

    % unpack parameters
    Q_switch=U(1);      % normalised switch gene resource competition factor
    Q_ofp=U(2);         % normalised the switch's output fluorescent protein gene RC factor
    Q_ofp2=U(3);        % normalised constitutive reporter RC factor

    % unpack the state vector (all entries are specie concentrations in nM)
    R=X(1);             % ribosomes
    p_o=X(2);           % other native proteins
    p_switch=X(3);      % self-activating switch protein
    p_ofp=X(4);         % the switch's output fluorescent protein
    p_ofp2=X(5);        % constitutive reporter output fluorescent protein
    ofp_mature=X(6);    % mature switch OFP
    ofp2_mature=X(7);   % mature reporter OFP
    
    % calculate the switch (and switch ofp) genes' regulatory function
    F_switch = F_switch_calc(p_switch, par);

    % find the total resource demand of all genes
    Q_het = F_switch.*Q_switch + F_switch.*Q_ofp + Q_ofp2; % normalised demand of all synthetic genes
    q_het = Q_het*(par('q_r')+par('q_o'));  % actual demand of all synthetic genes
    D = 1 + par('q_r') + par('q_o') + q_het;    % total resource demand in the cell

    % find the cell growth rate
    B = R*(1-1./D); % concentration of actively tranlsating ribosomes
    l = par('e').*B./par('M'); % cell growth rate

    % find dx/dt
    dxdt=[
        % ribosomes
        (par('e')./par('n_r')) .* (par('q_r')./D) .* R - l .* R;
        % other native proteins
        (par('e')./par('n_o')) .* (par('q_o')./D) .* R - l .* p_o;
        % self-activating switch protein
        (par('e')./par('n_switch')) .* (par('q_switch')./D) .* R - l .* p_switch;
        % the switch's output fluorescent protein
        (par('e')./par('n_ofp')) .* (par('q_ofp')./D) .* R - l .* p_ofp - par('mu_ofp') .* p_ofp;
        % constitutive reporter output fluorescent protein
        (par('e')./par('n_ofp2')) .* (par('q_ofp2')./D) .* R - l .* p_ofp - par('mu_ofp2') .* p_ofp;
        % mature switch OFP
        par('mu_ofp') .* p_ofp - l .* ofp_mature;
        % mature reporter OFP
        par('mu_ofp2') .* p_ofp2 - l .* ofp2_mature;
        ];    
end