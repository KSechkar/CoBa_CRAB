% selfact_constrep_ode.m - ODE for simulating measurements with
% constitutive reporters

function dXdt = selfact_constrep_ode( ...
    t, ...      % time
    X, ...      % state vector with dynamic variables and most parameters
    flag, ...   % flag, needed for sens_sys functioning
    U ...       % parameters to find sensitivities for
    )

    % unpack the parameters for which we calculate sensitivities
    Q_switch=U(1);  % normalised switch gene resource competition factor
    Q_ofp=U(2);     % normalised the switch's output fluorescent protein gene RC factor
    Q_ta=U(3);      % normalised probe transcription activator's RC factor
    Q_b=U(4);       % normalised probe burdensome OFP's RC factor
    ref=U(5);       % control reference

    % unpack the dynamic variables from the state vector (all entries are specie concentrations in nM)
    R=X(1);             % ribosomes
    p_o=X(2);           % other native proteins
    p_switch=X(3);      % self-activating switch protein
    p_ofp=X(4);         % the switch's output fluorescent protein
    p_ta=X(5);          % the probe's transcription activator protein
    p_b=X(6);           % the probe's output fluorescent protein
    ofp_mature=X(7);    % mature switch OFP
    b_mature=X(8);      % mature probe OFP

    % unpack the parameters from the state vector
    % host cell parameters
    M = X(9);       % mass of protein in the cell (aa)
    e = X(10);       % translation elongation rate (aa/h)
    q_r = X(11);    % resource demand of ribosomal genes
    n_r = X(12);    % protein length (aa) of ribosomes
    q_o = X(13);    % resource demand of other native genes
    n_o = X(14);    % protein length (aa) of other native proteins
    % self-activating switch parameters - save for RC factors
    n_switch = X(15);           % switch protein length
    n_ofp = X(16);              % switch OFP length
    mu_ofp = X(17);             % switch OFP maturation rate
    baseline_switch = X(18);        % baseline expression of switch gene
    K_switch = X(19);               % half-saturation constant for the switch protein's self-regulation
    I_switch = X(20);               % share of switch proteins bound by an inducer molecule
    eta_switch = X(21);             % cooperativity coefficicient of switch protein-DNA binding
    % CBC probe parameters -save for the RC factor
    n_ta = X(22);               % probe's transcription activator protein length
    n_b = X(23);                % probe's burdensome OFP length
    mu_b = X(24);               % probe's burdensome OFP maturation rate
    baseline_tai_dna = X(25);   % baseline expression of the burdnesome gene
    K_ta_i = X(26);             % half-saturation constant for the binding between the inducer and the activator protein
    K_tai_dna = X(27);          % half-saturation constant for the binding between the protein-inducer complex and the promoter DNA
    eta_tai_dna = X(28);        % cooperativity coefficicient of complex-DNA binding
    % controller parameters
    Kp = X(29);     % proportional feedback gain
    max_u = X(30);  % maximum inducer concentration which can be supplied

    % calculate the external input (assume no delay)
    u_unclipped = Kp*(ref-ofp_mature);  % get the proportional feedback signal as calculated
    u=max(min(u_unclipped,max_u),0.0);    % clip to feasible range

    % calculate the switch (and switch ofp) genes' regulatory function
    F_switch = F_switch_calc(p_switch, ...
         baseline_switch, K_switch, I_switch, eta_switch);
    
    % calculate the probe OFP gene' regulatory function
    F_b = F_b_calc(p_ta, u,...
        baseline_tai_dna, K_ta_i, K_tai_dna, eta_tai_dna);

    % find the synthetic genes' resource demands - taking regulation into account
    q_switch = F_switch.*Q_switch.*(q_r+q_o);   % switch gene auto-regulated
    q_ofp = F_switch.*Q_ofp.*(q_r+q_o);   % switch OFP co-expressed with the switch protein => same regulation
    q_ta = Q_ta.*(q_r+q_o);     % transcription activator gene constitutive
    q_b = F_b.*Q_b.*(q_r+q_o);   % the probe's burdensome OFP regulated by the transcription activator

    % find the total resource demand of all genes
    q_het = q_switch + q_ofp + q_ta + q_b;  % actual demand of all synthetic genes
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
        % probe transcription activator
        (e./n_ta) .* (q_ta./D) .* R - l .* p_ta;
        % the probe's burdensome output fluorescent protein
        (e./n_b) .* (q_b./D) .* R - l .* p_b - mu_b .* p_b;
        % mature switch OFP
        mu_ofp .* p_ofp - l .* ofp_mature;
        % mature probe's burdensome OFP
        mu_b .* p_b - l .* b_mature;
        ];    

    % append zeros for parameters (static state vector entries)
    dXdt=[dXdt_dynvars; 
        zeros(size(X,1)-size(dXdt_dynvars,1),1)];
end