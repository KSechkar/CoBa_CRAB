% F_b_calc.m - burdensome probe OFP gene's regulatory function

function F_b=F_b_calc( ...
    ... % dynamic variables
    p_ta, ...   % switch protein level
    u,...   % external input (inducer concentration)
    ... % system parameters
    baseline_tai_dna, ...    % baseline expression of the burdnesome gene
    K_ta_i, ...           % half-saturation constant for the binding between the inducer and the activator protein
    K_tai_dna, ...           % half-saturation constant for the binding between the protein-inducer complex and the promoter DNA
    eta_tai_dna...           % cooperativity coefficicient of complex-DNA binding
    )

    % transcription activator protein-dependent term:
    % the concentration of active (inducer-bound) switch proteins divided by half-saturation constant
    p_ta_term = (p_ta .* u./(u+K_ta_i))./K_tai_dna;

    % switch protein regulation function
    F_b = baseline_tai_dna + (1-baseline_tai_dna).*( ...
        (p_ta_term.^eta_tai_dna) ./ (p_ta_term.^eta_tai_dna + 1));
    
end