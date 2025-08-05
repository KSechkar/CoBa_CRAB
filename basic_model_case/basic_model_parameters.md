## basic_model_parameters.md

Descriptions, values, and motivations for the parameters of the basic resource-aware 
gene expression model used by the simulation scripts in the _basic_model_case_ folder.
The unit $aa$ here stands for 'amino acid residues'.

### Parameter decsriptions and values
Host cell:

| Parameter  | Description                         | Value             | Units   | 
|:-----------|:------------------------------------|:------------------|:--------| 
| $M$        | Total cellular protein mass density | $1.19 \cdot 10^9$ | $aa/fL$ |
| $\epsilon$ | Translation elongation rate         | $6.61 \cdot 10^4$ | $aa/h$  |
| $q_r$      | Ribosomal genes' resource demand    | $1.30 \cdot 10^4$ | None    |
| $q_o$      | Other native genes' resource demand | $6.12 \cdot 10^4$ | None    |
| $n_r$      | Mass of a ribosome                  | $7459$            | $aa$    |
| $n_r$      | Other native proteins' masses       | $300$             | $aa$    |

Constitutive reporter modules (a library of three reporters with linearly spaced resource 
demands is used for probe characterisation):

| Parameter                       | Description                                     | Value             | Units  | 
|:--------------------------------|:------------------------------------------------|:------------------|:-------| 
| $q_{c_1}$                       | First reporter's resource demand                | $4.50 \cdot 10^1$ | None |
| $q_{c_2}$                       | Second reporter's resource demand               | $1.50 \cdot 10^4$ | None |
| $q_{c_3}$                       | Third reporter's resource demand                | $3.00 \cdot 10^4$ | None |
| $n_{c_1}=n_{c_2}=n_{c_3}$       | Reporter proteins' masses                       | $300$             | $aa$   |
| $\mu_{c_1}=\mu_{c_2}=\mu_{c_3}$ | Fluorescent reporter proteins' maturation rates | $3.06$            | $1/h$  |

Probe module:

| Parameter      | Description                                                                                                               | Value             | Units | 
|:---------------|:--------------------------------------------------------------------------------------------------------------------------|:------------------|:------| 
| $q_{ta}$       | Transcription activator gene's resource demand                                                                            | $4.50 \cdot 10^1$ | None  |
| $q_{b}$        | Output fluorescent protein gene's resource demand                                                                         | $6.00 \cdot 10^4$ | None  |
| $n_{ta}=n_{b}$ | Protein masses                                                                                                            | $300$             | $aa$  |
| $\mu_{b}$      | Output fluorescent protein's maturation rate                                                                              | $3.06$            | $1/h$ |
| $K_{ta,i}$     | Half-saturation constant of the binding between <br> inducer molecules and transcription activator proteins               | $100$             | $nM$  |
| $K_{tai,b}$    | Half-saturation constant of the binding between <br> TA protein-inducer complexes and the <br> output gene's promoter DNA | $100$             | $nM$  |
| $\eta_{tai,b}$ | Cooperativity of the binding between <br> TA protein-inducer complexes and the <br> output gene's promoter DNA            | $2$               | None  |
| $F_{b,0}$      | Baseline output gene expression <br> in absence of transcription activators                                               | $0.01$            | None  |

Module of interest (self-activating genetic switch):

| Parameter       | Description                                                                                                         | Value             | Units | 
|:----------------|:--------------------------------------------------------------------------------------------------------------------|:------------------|:------| 
| $q_{s}$         | Switch gene's resource demand                                                                                       | $1.20 \cdot 10^2$ | None  |
| $q_{ofp}$       | Output gene's resource demand                                                                                       | $1.20 \cdot 10^4$ | None  |
| $n_{s}=n_{ofp}$ | Protein masses                                                                                                      | $300$             | $aa$  |
| $\mu_{ofp}$     | Output fluorescent protein's maturation rate                                                                        | $3.06$            | $1/h$ |
| $I$             | Share of active (i.e. bound to an inducer molecule) <br> switch proteins                                            | $0.1^*$           | None  |
| $K_{s,s}$       | Half-saturation constant of the binding between <br> switch-inducer complexes and the <br> regulated promoters' DNA | $250$             | $nM$  |
| $\eta_{s,s}$    | Cooperativity of the binding between <br> switch-inducer complexes and the <br> regulated promoters' DNA            | $2$               | None  |
| $F_{s,0}$       | Baseline switch and output gene expression <br> in absence of transcription activators                              | $0.05$            | None  |

$^*$ Everywhere except the second genetic switch in
_basic_model_case/basic_selfact_otherind.py_ (Fig.5d in the manuscript),
for which $I=1/9$.

Cybergenetic feedback controller (integral feedback implemented in the scripts 
but currently not used):

| Parameter | Description                | Value                                                                                       | Units | 
|:----------|:---------------------------|:--------------------------------------------------------------------------------------------|:------| 
| $K_p$     | Proportional feedback gain | $5 \cdot 10^{-4}$                                                                           | None  |
| $\tau$    | Control delay              | $0.01$ (on the scale of [de Cesare et al. 2022](https://doi.org/10.1021/acssynbio.1c00632)) | $h$   |


### Motivation for host cell parameter values
Besides this basic model, the study accompanied by this repository includes simulations
of a coarse-grained resource-aware mechanistic cell model originally published 
in [Sechkar et al. 2024](https://www.nature.com/articles/s41467-024-46410-9). Therefore, for the sake of consistency, parameters
describing the host cell were either taken from that publication for the protein mass density
$M$ and protein masses $n_r$ and $n_o$ (whilst more non-ribosomal genes that a single
'other' category are considered in the mechanistic model, they all are assumed to have
the same length).

When there was no one-to-one match between the two models, the basic model's parameters 
were selected so as to produce predictions similar to simulating the mechanistic cell model
for the culture medium nutrient quality $\sigma=0.5$ (rich culture media such as RDM with glucose 
as the carbon source). For instance, for the translation elongation rate $\epsilon$
we took the simulated steady-state value of the mechanistic cell model's translation rate,
where it is a dynamic variable. Similarly, $q_r$ and $q_o$ were chosen to give rise to the 
steady-state ribosomal and other protein mass fractions in the cell that would be similar 
to the mass fractions predicted by the mechanistic model. The code used to obtained 
these values is provided in the script _common/cell_to_basic.py_.

## Motivation for synthetic gene parameter values
[Sechkar et al. 2024](https://www.nature.com/articles/s41467-024-46410-9) also provides feasible ranges for the synthetic gene parameters
within the mechanistic cell modelling framework. The same ranges were used for our basic
model's extents of baseline gene expression $F_{b,0}$ and $F_{s,0}$, 
transcription factor-inducer binding half-saturation constant $K_{ta,i}$, 
as well as the cooperativities ($\eta_{tai,b}$ and $\eta_{s,s}$) and 
half-saturation constants of binding between transcription factors and promoter DNA
($K_{tai,b}$ and $K_{s,s}$). Generic protein lengths $\{n_i=300\}$ were taken for
all genes for simplicity.

As for synthetic gene expression parameters, there was once again no one-to-one match
between the mechanistic model parameters and the resource demands $\{q_i\}$ of the basic 
model. We thus assumed that synthetic gene resource demands could vary from $\approx 0$ 
(negligible non-native gene expression) to $\approx 7 \cdot 10^4$, i.e. up to the value
for the non-ribosomal native genes in the case of very high and burdensome synthetic 
gene expression.

The rates of fluorescent protein maturation, which was not considered in the mechanistic
cell model, were instead all assumed to be equal to that of sfGFP. The characteristic half-time
of this maturation process, according to
[Balleza et al. 2018](https://www.nature.com/articles/nmeth.4509), is $13.6\ min$.


