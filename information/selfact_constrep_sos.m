%% selfact_constrep_sos.m

%%
function sos=selfact_constrep_sos(U)
    %% LOAD experimental data
    persistent par Q_moi_ss Qdash_moi_ss
    
    if(isempty(Q_moi_ss))
        load('constrep_data.mat')
    end
    if(isempty(par))
        load('par_data.mat')
    end
    
    %% GET Q'_MOI PREDICTIONS FOR U
    Qdash_moi_predicted = Qdash_moi_from_Q_moi(Q_moi_ss, par, U);

    sos=sum((Qdash_moi_ss-Qdash_moi_predicted).^2);
    disp(sos)
end