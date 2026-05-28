## Tuning axes

#### Key axes for varying
- Primary (Smoothing strength)
- Smoothing target (measures how useful clustering actually is, compared to random cluster comparison and smoothing towards global means) This essentially seperates smoothing helps from clustering helps.


#### If time likely tune and dont vary:

- Cluster size (controls how many users give useful information)
    How to tune:
    Compare multiple cluster sizes and validation performance

- Bin size (Offers its own smoothing effect)
    This is harder to tune as broader will be easier to predict on the validation set. Instead look at train information loss. This also has an interaction with itself and smoothing strengh with larger and therefore smoother bins likely requiring less smoothing.

- Alert sensitivity to smoothing.

No investigation:
- Clustering methodology 
- Count based online detection metholdogy
(Results will obviously be somewhat dependent on clustering methodology and online detection methodology but they are not key axes for the experiment and likely more influential parameters to tune exist.)

#### How to smooth

- Ideas 2:

1) Smooth by linearly smoothing the mean and variance parameters to the observed cluster means. 
2) Smooth in a baysian fashion by having a prior

First one allows a clean interpretation of the endpoints of complete smoothing and no smoothing and is computationally easier. The baysian doesnt allow a complete smoothing

Rank order cluster parameters then smooth on the log scale.
Log smoothing is best because count rates and dispersion are positive, multiplicative quantities: moving halfway from 1 to 100 should usually mean around 10, not 50.5. The search explores many different scales with log smoothing.