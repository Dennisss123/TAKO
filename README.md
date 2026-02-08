# TAKO (Topology-Adjusted KnockOut)

Core implementation of WT-only single-gene KO ranking on gene graphs.

Pipeline:  
`X (WT cells×genes) -> A -> P -> P^KO_g -> (s_wt, s_ko) -> delta_raw/pos/abs`

## Environment

- Python >= 3.10
- See `requirements.txt` for pinned dependencies.

## Installation

```bash
git clone https://github.com/Dennisss123/WANG-HAO.git
cd WANG-HAO
pip install -r requirements.txt
