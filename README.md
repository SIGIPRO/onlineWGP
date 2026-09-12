Repository for the paper:
**Online Gradient Computation for Warping Gaussian Process Transformations**

## LIBRARIES
> [numpy](https://numpy.org/) \
> [matplotlib](https://matplotlib.org/)

### VIRTUAL ENVIRONMENT

If you use virtual environments you can install the necessary libraries from the `requirements.txt` file. \
For example:
```
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## (SOME) CONTEXT
The `storage` folder contain a trained GP (`experiment10`) and warped GP (`experiment20`) for the regression task in Sec. 5. \
It also contains a `.json` file with the experimental setup.

If you want to rerun the experiment for a different setup change the `conf` within experiments `XX` and type:
```
python main.py -id XX
```
for `XX` = `10` or `20`.

To visualize the results, run:
```
python main.py -id YY
```
with `YY` = `11` or `21`.

## PAPER PLOTS
To reproduce Fig. 1 run:
```
python rasmussen.py
```

To reproduce Fig. 2 run: 
```
python experiment.py -id 40
```
and 
```
python experiment.py -id 41
```