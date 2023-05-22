## CODE ACCOMPANIED THE PAPER "EXISTENCE OF OPTIMAL SPARSE RELU NEURAL NETWORKS"
### Code organisation 
```
├── quantifiersElimination      # Code for Section 3.3
│   ├── LU2x2.py                # Z3Prover code for 2 x 2 LU architecture
│   ├── LU3x3.py                # Z3Prover code for 3 x 3 LU architecture
│   ├── fullsupport.py          # Z3Prover code for 2 x 2 plain support
├── regularizationImpact        # Code for Section 3.2
│   ├── training.py             # Training in Example 3.1
│   ├── data_generate.py        
│   ├── models.py
│   ├── plot_err.py             # Code to plot the training curves 
├── requirements.txt
├── README.md
```

### How to run the code
To run the code in Example 3.1.
```
python training.py
python plot_err.py
```

To run the code in Section 3.3.
```
python LU2x2.py
```
```
python LU3x3.py
```
```
python fullsupport.py
```