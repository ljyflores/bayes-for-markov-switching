# bayes-for-markov-switching
<b>TLDR: </b> We provide a Python implementation of the original R code for fitting a Markov Switching Model using Bayesian inference (Gibbs Sampling) by <a href="https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007839">Lim et al (2020)</a>. We explore such methods of estimation as Bayesian methods have been found to be more flexible and efficient that standard MLE approaches <a href="http://www.jstor.org/stable/223255">(Ghysels, 1998)</a>, <a href="https://www.cambridge.org/core/journals/astin-bulletin-journal-of-the-iaa/article/markov-chain-monte-carlo-estimation-of-regime-switching-vector-autoregressions/94B66142AFAAF7D2D21B1B5DC745F72F">(Harris, 2014)</a>.

This repository can be used to fit the following model using Bayesian inference:

$$ y_t = \mu_{S_t} + \sum_{i=1}^{k} \phi_{S_t}^i (y_{t-i}-\mu_{S_t-i}) + e_t, \quad e_t \sim N(0,\sigma_{S_t}^2) $$

$$ \mu_{S_t} = \sum_{i=1}^l \mu_i S_{it} \quad \text{(Mean of State $i$)}, \quad \sigma_{S_t}^2 = \sum_{i=1}^l \sigma_i^2 S_{it} \quad \text{(Variance of State $i$)} $$

$$ S_{it} = I[S_t=i] \quad \text{(Indicator Variable)}, \quad p_{i,j} = P[S_t=j|S_{t-1}=i] \quad \sum_{j=1}^l p_{i,j}=1 \quad \text{(Transition Probabilities)} $$

The `main.ipynb` notebook provides an example of applying this to FED return data. 

Check out the writeup explaining the algorithm <a href="https://drive.google.com/file/d/1nfoNjcJfUpudIiWJt5PEUaczfERx-7gk/view?usp=sharing">here</a>!

In case this repository was helpful, please consider citing the main authors:
```
@article{10.1371/journal.pcbi.1007839,
    doi = {10.1371/journal.pcbi.1007839},
    author = {Lim, Jue Tao AND Dickens, Borame Sue AND Haoyang, Sun AND Ching, Ng Lee AND Cook, Alex R.},
    journal = {PLOS Computational Biology},
    publisher = {Public Library of Science},
    title = {Inference on dengue epidemics with Bayesian regime switching models},
    year = {2020},
    month = {05},
    volume = {16},
    url = {https://doi.org/10.1371/journal.pcbi.1007839},
    pages = {1-15},
    number = {5},
}
```
