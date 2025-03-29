## cell_model_parameters.md

Descriptions, values, and motivations for the parameters of the coarse-grained
mechanistic resource-aware cell model used by the simulation scripts in the _cell_model_case_ folder.
The unit $aa$ here stands for 'amino acid residues', and all the genes have the same function
and labels as decribed for the basic cell model in Section II of the paper.

### Parameter decsriptions and values
Host cell parameters taken from [Sechkar et al. 2024](https://www.nature.com/articles/s41467-024-46410-9), where the model was originally published.
The cell culture medium is assumed to have nutrient quality $\sigma=0.5$ (rich culture media such as RDM with glucose 
as the carbon source).

Constitutive reporter modules (a library of three reporters with linearly spaced resource 
demands is used for probe characterisation):

| Parameter                                 | Description                                     | Value             | Units            | 
|:------------------------------------------|:------------------------------------------------|:------------------|:-----------------| 
| $c_{c_1}$                                 | First reporter gene's DNA concentration         | $1$               | $nM$             |
| $c_{c_2}$                                 | Second reporter gene's DNA concentration        | $100$             | $nM$             |
| $c_{c_3}$                                 | Third reporter gene's DNA concentration         | $100$             | $nM$             |
| $\alpha_{c_1}$                            | First reporter gene's promoter strength         | $4.00 \cdot 10^2$ | None             |
| $\alpha_{c_2}$                            | Second reporter gene's promoter strength        | $1.25 \cdot 10^3$ | None             |
| $\alpha_{c_3}$                            | Third reporter gene's promoter strength         | $2.50 \cdot 10^3$ | None             |
| $\beta_{c_1} = \beta_{c_2} = \beta_{c_3}$ | mRNA degradation rates                          | $6$               | $1/h$            |
| $k^+_{c_1} = k^+_{c_2} = k^+_{c_3}$       | mRNA-ribosome biding rates                      | $60$              | $1/(nM \cdot h)$ |
| $k^-_{c_1} = k^-_{c_2} = k^-_{c_3}$       | mRNA-ribosome dissociation rates                | $60$              | $1/h$            |
| $n_{c_1}=n_{c_2}=n_{c_3}$                 | Reporter proteins' masses                       | $300$             | $aa$             |
| $\mu_{c_1}=\mu_{c_2}=\mu_{c_3}$           | Fluorescent reporter proteins' maturation rates | $3.06$            | $1/h$            |

Probe module:

| Parameter                | Description                                                                                                               | Value             | Units            | 
|:-------------------------|:--------------------------------------------------------------------------------------------------------------------------|:------------------|:-----------------| 
| $c_{ta}$                 | Transcription activator gene's DNA concentration                                                                          | $10$              | $nM$             |
| $c_{b}$                  | Output fluorescent protein gene's DNA concentration                                                                       | $100$             | $nM$             |
| $\alpha_{ta}$            | Transcription activator gene's promoter strength                                                                          | $4.00 \cdot 10^2$ | None             |
| $\alpha_{b}$             | Output fluorescent protein gene's promoter strength                                                                       | $1.25 \cdot 10^3$ | None             |
| $\beta_{ta} = \beta_{b}$ | mRNA degradation rates                                                                                                    | $6$               | $1/h$            |
| $k^+_{ta} = k^+_b$       | mRNA-ribosome binding rates                                                                                               | $0.6$             | $1/(nM \cdot h)$ |
| $k^-_{ta} = k^-_b$       | mRNA-ribosome dissociation rates                                                                                          | $60$              | $1/h$            |
| $n_{ta}=n_{b}$           | Protein masses                                                                                                            | $300$             | $aa$             |
| $\mu_{b}$                | Output fluorescent protein's maturation rate                                                                              | $3.06$            | $1/h$            |
| $K_{ta,i}$               | Half-saturation constant of the binding between <br> inducer molecules and transcription activator proteins               | $100$             | $nM$             |
| $K_{tai,b}$              | Half-saturation constant of the binding between <br> TA protein-inducer complexes and the <br> output gene's promoter DNA | $100$             | $nM$             |
| $\eta_{tai,b}$           | Cooperativity of the binding between <br> TA protein-inducer complexes and the <br> output gene's promoter DNA            | $2$               | None             |
| $F_{b,0}$                | Baseline output gene expression <br> in absence of transcription activators                                               | $0.01$            | None             |

Module of interest (self-activaing genetic switch):

| Parameter               | Description                                                                                                         | Value             | Units            | 
|:------------------------|:--------------------------------------------------------------------------------------------------------------------|:------------------|:-----------------| 
| $c_s=c_{ofp}$           | Gene DNA concentrations                                                                                             | $100$             | $nM$             |
| $\alpha_s=\alpha_{ofp}$ | Gene promoter strengths                                                                                             | $3.00 \cdot 10^3$ | None             |
| $\beta_s = \beta_{ofp}$ | mRNA degradation rates                                                                                              | $6$               | None             |
| $k^+_s$                 | Switch gene's mRNA-ribosome binding rate                                                                            | $0.6$             | $1/(nM \cdot h)$ |
| $k^+_{ofp}$             | Output gene's mRNA-ribosome binding rate                                                                            | $60$              | $1/(nM \cdot h)$ |
| $k^-_{s} = k^-_{ofp}$   | mRNA-ribosome dissociation rates                                                                                    | $6$               | $1/h$            |
| $n_{s}=n_{ofp}$         | Protein masses                                                                                                      | $300$             | $aa$             |
| $\mu_{ofp}$             | Output fluorescent protein's maturation rate                                                                        | $3.06$            | $1/h$            |
| $I$                     | Share of active (i.e. bound to an inducer molecule) <br> switch proteins                                            | $0.1^*$           | None             |
| $K_{s,s}$               | Half-saturation constant of the binding between <br> switch-inducer complexes and the <br> regulated promoters' DNA | $250$             | $nM$             |
| $\eta_{s,s}$            | Cooperativity of the binding between <br> switch-inducer complexes and the <br> regulated promoters' DNA            | $2$               | None             |
| $F_{s,0}$               | Baseline switch and output gene expression <br> in absence of transcription activators                              | $0.05$            | None             |

$^*$Everywhere except the second genetic switch in
_basic_model_case/basic_selfact_otherind.py_ (Fig.5d in the paper),
for which $I=1/9$.

Cybergenetic feedback controller (integral feedback implemented in the scripts 
but currently not used):

| Parameter | Description                | Value             | Units | 
|:----------|:---------------------------|:------------------|:------| 
| $K_p$     | Proportional feedback gain | $5 \cdot 10^{-4}$ | None  |


### Motivation for parameter values
Wherever a parameter was found both in this cell model and the basic gene expression model
from _basic_model_case_, the same value as for the basic model was taken. The values of
promoter strengths $\{\alpha_i\}$ were taken from the feasible ranges identified for the
model in [Sechkar et al. 2024](https://www.nature.com/articles/s41467-024-46410-9). Synthetic gene DNA concentrations $\{c_i\}$ were 
assumed to range between $1\ nM$ (single copy per cell, e.g. for genomic integration) 
and $\approx 100\ nM$ high copy number plasmid). For mRNA degradation rates $\{\beta_i\}$ 
and mRNA-ribosome dissociation rates 
$\{k^-_i\}$, generic values from [Sechkar et al. 2024](https://www.nature.com/articles/s41467-024-46410-9) were taken. For all genes 
but $s$, the mRNA-ribosome association rates were assumed to correspond to a very strong
ribosome binding sequence (RBS) as per [Sechkar et al. 2024](https://www.nature.com/articles/s41467-024-46410-9), 
whilst for the switch gene the RBS was assumed to be
100 times weaker - this is within the 250-fold range of RBS strengths
suggested for this cell model in
[Sechkar et al. 2025](https://royalsocietypublishing.org/doi/10.1098/rsif.2024.0602).


