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

## RESULTS
The empirical comparison between a warped GP and a GP presented in the paper is done in `experiment50()`. 
To see the results run:
```
python main.py -id 50
```

> NOTE: the heuristic observation model noise variance for the GP is chosen a follows:
> At the zero crossings $y\sim \mathcal{N}(0,\sigma^2)$. 
> We can factor out the standard deviation as: $y = \sigma \cdot u$, where $u\sim\mathcal{N}(0,1)$.
> Then, $z = y^{\frac{1}{3}} = \sigma^{\frac{1}{3}} \cdot u^{\frac{1}{3}}$.
> Finally, $\text{std}(z)=\sigma^{\frac{1}{3}}\cdot\text{std}(u^{\frac{1}{3}}) \approx \frac{3}{4}\sigma^{\frac{1}{3}}$.

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