% selfact_constrep_ode.m - ODE for simulating measurements with
% constitutive reporters

function dXdt = selfact_constrep_ode( ...
    t, ...      % time
    X, ...      % state vector with dynamic variables and most parameters
    flag, ...   % flag, needed for sens_sys functioning
    U ...       % parameters to find sensitivities for
    )

    % unpack the switch parameters for which we calculate sensitivities
    Q_switch=U(1);          % normalised switch gene resource competition factor
    Q_ofp=U(2);             % normalised switch's output fluorescent protein gene RC factor
    n_switch = U(3);        % switch protein length
    n_ofp = U(4);           % switch OFP length
    mu_ofp = U(5);          % switch OFP maturation rate
    baseline_switch = U(6); % baseline expression of switch gene
    K_switch = U(7);        % half-saturation constant for the switch protein's self-regulation
    eta_switch = U(8);      % cooperativity coefficicient of switch protein-DNA binding

    % unpack the dynamic variables from the state vector (all entries are specie concentrations in nM)
    R=X(1);             % ribosomes
    p_o=X(2);           % other native proteins
    p_switch=X(3);      % self-activating switch protein
    p_ofp=X(4);         % the switch's output fluorescent protein
    p_ofp2=X(5);        % constitutive reporter output fluorescent protein
    ofp_mature=X(6);    % mature switch OFP
    ofp2_mature=X(7);   % mature reporter OFP

    % unpack the parameters from the state vector
    % host cell parameters
    M = X(8);       % mass of protein in the cell (aa)
    e = X(9);       % translation elongation rate (aa/h)
    q_r = X(10);    % resource demand of ribosomal genes
    n_r = X(11);    % protein length (aa) of ribosomes
    q_o = X(12);    % resource demand of other native genes
    n_o = X(13);    % protein length (aa) of other native proteins
    % constitutive reporter parameters
    Q_ofp2 = X(14);     % reporter RC factor
    n_ofp2 = X(15);     % reporter OFP length
    mu_ofp2 = X(16);    % reporter OFP maturation rate
    % switch parameters
    I_switch = X(17);
    
    % calculate the switch (and switch ofp) genes' regulatory function
    F_switch = F_switch_calc(p_switch, ...
         baseline_switch, K_switch, I_switch, eta_switch);

    % find the synthetic genes' resource demands - taking regulation into account
    q_switch = F_switch.*Q_switch.*(q_r+q_o);   % switch gene auto-regulated
    q_ofp = F_switch.*Q_ofp.*(q_r+q_o);   % switch OFP co-expressed with the switch protein => same regulation
    q_ofp2 = Q_ofp2.*(q_r+q_o); % reporter OFP constitutive

    % find the total resource demand of all genes
    q_het = q_switch + q_ofp + q_ofp2;  % actual demand of all synthetic genes
    D = 1 + q_r + q_o + q_het;    % total resource demand in the cell

    % find the cell growth rate
    B = R*(1-1./D); % concentration of actively tranlsating ribosomes
    l = e.*B./M; % cell growth rate

    % find dX/dt for dynamic variables
    dXdt_dynvars=[
        % ribosomes
        (e./n_r) .* (q_r./D) .* R - l .* R;
        % other native proteins
        (e./n_o) .* (q_o./D) .* R - l .* p_o;
        % self-activating switch protein
        (e./n_switch) .* (q_switch./D) .* R - l .* p_switch;
        % the switch's output fluorescent protein
        (e./n_ofp) .* (q_ofp./D) .* R - l .* p_ofp - mu_ofp .* p_ofp;
        % constitutive reporter output fluorescent protein
        (e./n_ofp2) .* (q_ofp2./D) .* R - l .* p_ofp2 - mu_ofp2 .* p_ofp;
        % mature switch OFP
        mu_ofp .* p_ofp - l .* ofp_mature;
        % mature reporter OFP
        mu_ofp2 .* p_ofp2 - l .* ofp2_mature;
        ];    

    % append zeros for parameters (static state vector entries)
    dXdt=[dXdt_dynvars; 
        zeros(size(X,1)-size(dXdt_dynvars,1),1)];
end