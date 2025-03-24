function Y0_with_par = selfact_constrep_append_par(Y0, par)
Y0_with_par=[
    Y0;
    % host cell parameters
    par('M');
    par('e');
    par('q_r');
    par('n_r');
    par('q_o');
    par('n_o');    
    % constitutive reporter parameters
    par('q_ofp2')./(par('q_r')+par('q_o'));
    par('n_ofp2');
    par('mu_ofp2');
    % switch parameters
    par('I_switch');
    ];
end