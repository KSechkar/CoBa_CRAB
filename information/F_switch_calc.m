% F_switch_calc.m - self-activating switch gene's regulatory function

function F_switch=F_switch_calc( ...
    ... % dynamic variables
    p_switch, ...   % switch protein level
    ... % system parameters
    baseline_switch, ...    % baseline expression of switch gene
    K_switch, ...           % half-saturation constant for the switch protein's self-regulation
    I_switch, ...           % share of switch proteins bound by an inducer molecule
    eta_switch...           % cooperativity coefficicient of switch protein-DNA binding
    )
    % switch protein-dependent term:
    % the concentration of active (inducer-bound) switch proteins divided by half-saturation constant
    p_switch_term = (p_switch.*I_switch)./K_switch;

    % switch protein regulation function
    F_switch = baseline_switch + (1-baseline_switch).*( ...
        (p_switch_term.^eta_switch) ./ (p_switch_term.^eta_switch + 1));
    
end