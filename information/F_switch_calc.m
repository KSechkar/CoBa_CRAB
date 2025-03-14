% F_switch.m - self-activating switch gene's regulatory
% function

function F_switch=F_switch_calc( ...
    p_switch, ...   % switch protein level
    par ...         % system parameters
    )
    % switch protein-dependent term:
    % the concentration of active (inducer-bound) switch proteins divided by half-saturation constant
    p_switch_term = (p_switch.*par('I_switch'))./par('K_switch');

    % switch protein regulation function
    F_switch = par('baseline_switch') + (1-par('baseline_switch')).*( ...
        (p_switch_term.^par('eta_switch')) ./ (p_switch_term.^par('eta_switch') + 1));
    
end