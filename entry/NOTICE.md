# Final ARC-AGI-3 entry notebook

`duck-franzen-m2-loadguard-lb.ipynb` is the notebook behind our two selected final submissions in ARC Prize 2026 ARC-AGI-3 (Kaggle notebook `hivemindadmin/duck-franzen-m2-loadguard-lb`, version 1; selected draws with public scores 33.60 and 31.89).

## Licences

- Almost all of the code is Daniel Franzen's public Milestone-2 solution (Kaggle `dfranzen/arc-agi-3-milestone-2-solution`, github.com/da-fr/arc-agi-3-solution), licensed **Apache-2.0**. It is unchanged here and remains under Apache-2.0. Credit goes to its author and to Tufa Labs for the Duck harness it builds on.
- Our additions are a mount fix and the "loadguard" cell. The loadguard cell holds games until the model server is healthy and sends a keepalive RESET every 7 minutes. These additions are released under this repository's MIT-0 licence.
- Model weights and third-party runtimes that the notebook loads stay under their own licences.

## Reproducing

Open the notebook on Kaggle with the competition's RTX PRO 6000 GPU and the inputs listed in its metadata. Outside the competition environment it will not run as is.
