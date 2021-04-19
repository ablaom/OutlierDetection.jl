
<a id='Models'></a>

<a id='Models-1'></a>

# Models


All models, both supervised and unsupervised, define a [`Detector`](base.md#OutlierDetection.Detector), which is just a mutable collection of hyperparameters. Each detector implements a [`fit`](base.md#OutlierDetection.fit) and [`transform`](base.md#OutlierDetection.transform) method, where *fitting refers to learning a model from training data* and *transform refers to using a learned model to calculate outlier scores of test data*.


<a id='Proximity-Models'></a>

<a id='Proximity-Models-1'></a>

## Proximity Models


<a id='ABOD'></a>

<a id='ABOD-1'></a>

### ABOD

<a id='OutlierDetection.ABOD' href='#OutlierDetection.ABOD'>#</a>
**`OutlierDetection.ABOD`** &mdash; *Type*.



```julia
ABOD(k = 5,
     metric = Euclidean(),
     algorithm = :kdtree,
     leafsize = 10,
     reorder = true,
     parallel = false,
     enhanced = false)
```

Determine outliers based on the angles to its nearest neighbors. This implements the `FastABOD` variant described in the paper, that is, it uses the variance of angles to its nearest neighbors, not to the whole dataset, see [1]. 

*Notice:* The scores are inverted, to conform to our notion that higher scores describe higher outlierness.

**Parameters**

```
k::Integer
```

Number of neighbors (must be greater than 0).

```
metric::Metric
```

This is one of the Metric types defined in the Distances.jl package. It is possible to define your own metrics by creating new types that are subtypes of Metric.

```
algorithm::Symbol
```

One of `(:kdtree, :brutetree, :balltree)`. In a `kdtree`, points are recursively split into groups using hyper-planes. Therefore a KDTree only works with axis aligned metrics which are: Euclidean, Chebyshev, Minkowski and Cityblock. A *brutetree* linearly searches all points in a brute force fashion and works with any Metric. A *balltree* recursively splits points into groups bounded by hyper-spheres and works with any Metric.

```
leafsize::Int
```

Determines at what number of points to stop splitting the tree further. There is a trade-off between traversing the tree and having to evaluate the metric function for increasing number of points.

```
reorder::Bool
```

While building the tree this will put points close in distance close in memory since this helps with cache locality. In this case, a copy of the original data will be made so that the original data is left unmodified. This can have a significant impact on performance and is by default set to true.

```
parallel::Bool
```

Parallelize `transform` and `predict` using all threads available. The number of threads can be set with the `JULIA_NUM_THREADS` environment variable. Note: `fit` is not parallel.

```
enhanced::Bool
```

When `enhanced=true`, it uses the enhanced ABOD (EABOD) adaptation proposed by [2].

**Examples**

```julia
using OutlierDetection: ABOD, fit, transform
detector = ABOD()
X = rand(10, 100)
model, scores = fit(detector, X)
transform(detector, model, X)
```

**References**

[1] Kriegel, Hans-Peter; S hubert, Matthias; Zimek, Arthur (2008): Angle-based outlier detection in high-dimensional data.

[2] Li, Xiaojie; Lv, Jian Cheng; Cheng, Dongdong (2015): Angle-Based Outlier Detection Algorithm with More Stable Relationships.


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/models/proximity/abod.jl#LL5-L37' class='documenter-source'>source</a><br>


<a id='COF'></a>

<a id='COF-1'></a>

### COF

<a id='OutlierDetection.COF' href='#OutlierDetection.COF'>#</a>
**`OutlierDetection.COF`** &mdash; *Type*.



```julia
COF(k = 5
    metric = Euclidean()
    algorithm = :kdtree
    leafsize = 10
    reorder = true
    parallel = false)
```

Local outlier density based on chaining distance between graphs of neighbors, as described in [1].

**Parameters**

```
k::Integer
```

Number of neighbors (must be greater than 0).

```
metric::Metric
```

This is one of the Metric types defined in the Distances.jl package. It is possible to define your own metrics by creating new types that are subtypes of Metric.

```
algorithm::Symbol
```

One of `(:kdtree, :brutetree, :balltree)`. In a `kdtree`, points are recursively split into groups using hyper-planes. Therefore a KDTree only works with axis aligned metrics which are: Euclidean, Chebyshev, Minkowski and Cityblock. A *brutetree* linearly searches all points in a brute force fashion and works with any Metric. A *balltree* recursively splits points into groups bounded by hyper-spheres and works with any Metric.

```
leafsize::Int
```

Determines at what number of points to stop splitting the tree further. There is a trade-off between traversing the tree and having to evaluate the metric function for increasing number of points.

```
reorder::Bool
```

While building the tree this will put points close in distance close in memory since this helps with cache locality. In this case, a copy of the original data will be made so that the original data is left unmodified. This can have a significant impact on performance and is by default set to true.

```
parallel::Bool
```

Parallelize `transform` and `predict` using all threads available. The number of threads can be set with the `JULIA_NUM_THREADS` environment variable. Note: `fit` is not parallel.

**Examples**

```julia
using OutlierDetection: COF, fit, transform
detector = COF()
X = rand(10, 100)
model, scores = fit(detector, X)
transform(detector, model, X)
```

**References**

[1] Tang, Jian; Chen, Zhixiang; Fu, Ada Wai-Chee; Cheung, David Wai-Lok (2002): Enhancing Effectiveness of Outlier Detections for Low Density Patterns.


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/models/proximity/cof.jl#LL1-L23' class='documenter-source'>source</a><br>


<a id='DNN'></a>

<a id='DNN-1'></a>

### DNN

<a id='OutlierDetection.DNN' href='#OutlierDetection.DNN'>#</a>
**`OutlierDetection.DNN`** &mdash; *Type*.



```julia
DNN(metric = Euclidean()
algorithm = :kdtree
leafsize = 10
reorder = true
parallel = false)
```

Anomaly score based on the number of neighbors in a hypersphere of radius `d`. Knorr et al. [1] directly converted the resulting outlier scores to labels, thus this implementation does not fully reflect the approach from the paper.

**Parameters**

```
k::Integer
```

Number of neighbors (must be greater than 0).

```
metric::Metric
```

This is one of the Metric types defined in the Distances.jl package. It is possible to define your own metrics by creating new types that are subtypes of Metric.

```
algorithm::Symbol
```

One of `(:kdtree, :brutetree, :balltree)`. In a `kdtree`, points are recursively split into groups using hyper-planes. Therefore a KDTree only works with axis aligned metrics which are: Euclidean, Chebyshev, Minkowski and Cityblock. A *brutetree* linearly searches all points in a brute force fashion and works with any Metric. A *balltree* recursively splits points into groups bounded by hyper-spheres and works with any Metric.

```
leafsize::Int
```

Determines at what number of points to stop splitting the tree further. There is a trade-off between traversing the tree and having to evaluate the metric function for increasing number of points.

```
reorder::Bool
```

While building the tree this will put points close in distance close in memory since this helps with cache locality. In this case, a copy of the original data will be made so that the original data is left unmodified. This can have a significant impact on performance and is by default set to true.

```
parallel::Bool
```

Parallelize `transform` and `predict` using all threads available. The number of threads can be set with the `JULIA_NUM_THREADS` environment variable. Note: `fit` is not parallel.

```
d::Real
```

The hypersphere radius used to calculate the global density of an instance.

**Examples**

```julia
using OutlierDetection: DNN, fit, transform
detector = DNN()
X = rand(10, 100)
model, scores = fit(detector, X)
transform(detector, model, X)
```

**References**

[1] Knorr, Edwin M.; Ng, Raymond T. (1998): Algorithms for Mining Distance-Based Outliers in Large Datasets.


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/models/proximity/dnn.jl#LL1-L25' class='documenter-source'>source</a><br>


<a id='KNN'></a>

<a id='KNN-1'></a>

### KNN

<a id='OutlierDetection.KNN' href='#OutlierDetection.KNN'>#</a>
**`OutlierDetection.KNN`** &mdash; *Type*.



```julia
KNN(k=5,
    metric=Euclidean,
    algorithm=:kdtree,
    leafsize=10,
    reorder=true,
    reduction=:maximum)
```

Calculate the anomaly score of an instance based on the distance to its k-nearest neighbors.

**Parameters**

```
k::Integer
```

Number of neighbors (must be greater than 0).

```
metric::Metric
```

This is one of the Metric types defined in the Distances.jl package. It is possible to define your own metrics by creating new types that are subtypes of Metric.

```
algorithm::Symbol
```

One of `(:kdtree, :brutetree, :balltree)`. In a `kdtree`, points are recursively split into groups using hyper-planes. Therefore a KDTree only works with axis aligned metrics which are: Euclidean, Chebyshev, Minkowski and Cityblock. A *brutetree* linearly searches all points in a brute force fashion and works with any Metric. A *balltree* recursively splits points into groups bounded by hyper-spheres and works with any Metric.

```
leafsize::Int
```

Determines at what number of points to stop splitting the tree further. There is a trade-off between traversing the tree and having to evaluate the metric function for increasing number of points.

```
reorder::Bool
```

While building the tree this will put points close in distance close in memory since this helps with cache locality. In this case, a copy of the original data will be made so that the original data is left unmodified. This can have a significant impact on performance and is by default set to true.

```
parallel::Bool
```

Parallelize `transform` and `predict` using all threads available. The number of threads can be set with the `JULIA_NUM_THREADS` environment variable. Note: `fit` is not parallel.

```
reduction::Symbol
```

One of `(:maximum, :median, :mean)`. (`reduction=:maximum`) was proposed by [1]. Angiulli et al. [2] proposed sum to reduce the distances, but mean has been implemented for numerical stability.

**Examples**

```julia
using OutlierDetection: KNN, fit, transform
detector = KNN()
X = rand(10, 100)
model, scores = fit(detector, X)
transform(detector, model, X)
```

**References**

[1] Ramaswamy, Sridhar; Rastogi, Rajeev; Shim, Kyuseok (2000): Efficient Algorithms for Mining Outliers from Large Data Sets.

[2] Angiulli, Fabrizio; Pizzuti, Clara (2002): Fast Outlier Detection in High Dimensional Spaces.


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/models/proximity/knn.jl#LL1-L29' class='documenter-source'>source</a><br>


<a id='LOF'></a>

<a id='LOF-1'></a>

### LOF

<a id='OutlierDetection.LOF' href='#OutlierDetection.LOF'>#</a>
**`OutlierDetection.LOF`** &mdash; *Type*.



```julia
LOF(k = 5
    metric = Euclidean()
    algorithm = :kdtree
    leafsize = 10
    reorder = true
    parallel = false)
```

Calculate an anomaly score based on the density of an instance in comparison to its neighbors. This algorithm introduced the notion of local outliers and was developed by Breunig et al., see [1].

**Parameters**

```
k::Integer
```

Number of neighbors (must be greater than 0).

```
metric::Metric
```

This is one of the Metric types defined in the Distances.jl package. It is possible to define your own metrics by creating new types that are subtypes of Metric.

```
algorithm::Symbol
```

One of `(:kdtree, :brutetree, :balltree)`. In a `kdtree`, points are recursively split into groups using hyper-planes. Therefore a KDTree only works with axis aligned metrics which are: Euclidean, Chebyshev, Minkowski and Cityblock. A *brutetree* linearly searches all points in a brute force fashion and works with any Metric. A *balltree* recursively splits points into groups bounded by hyper-spheres and works with any Metric.

```
leafsize::Int
```

Determines at what number of points to stop splitting the tree further. There is a trade-off between traversing the tree and having to evaluate the metric function for increasing number of points.

```
reorder::Bool
```

While building the tree this will put points close in distance close in memory since this helps with cache locality. In this case, a copy of the original data will be made so that the original data is left unmodified. This can have a significant impact on performance and is by default set to true.

```
parallel::Bool
```

Parallelize `transform` and `predict` using all threads available. The number of threads can be set with the `JULIA_NUM_THREADS` environment variable. Note: `fit` is not parallel.

**Examples**

```julia
using OutlierDetection: LOF, fit, transform
detector = LOF()
X = rand(10, 100)
model, scores = fit(detector, X)
transform(detector, model, X)
```

**References**

[1] Breunig, Markus M.; Kriegel, Hans-Peter; Ng, Raymond T.; Sander, Jörg (2000): LOF: Identifying Density-Based Local Outliers.


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/models/proximity/lof.jl#LL3-L26' class='documenter-source'>source</a><br>


<a id='Neural-Models'></a>

<a id='Neural-Models-1'></a>

## Neural Models


<a id='AE'></a>

<a id='AE-1'></a>

### AE

<a id='OutlierDetection.AE' href='#OutlierDetection.AE'>#</a>
**`OutlierDetection.AE`** &mdash; *Type*.



```julia
AE(encoder= Chain(),
   decoder = Chain(),
   batchsize= 32,
   epochs = 1,
   shuffle = false,
   partial = true,
   opt = ADAM(),
   loss = mse)
```

Calculate the anomaly score of an instance based on the reconstruction loss of an autoencoder, see [1] for an explanation of auto encoders.

**Parameters**

```
encoder::Chain
```

Transforms the input data into a latent state with a fixed shape.

```
decoder::Chain
```

Transforms the latent state back into the shape of the input data.

```
batchsize::Integer
```

The number of samples to work through before updating the internal model parameters.

```
epochs::Integer
```

The number of passes of the entire training dataset the machine learning algorithm has completed. 

```
shuffle::Bool
```

If `shuffle=true`, shuffles the observations each time iterations are re-started, else no shuffling is performed.

```
partial::Bool
```

If `partial=false`, drops the last mini-batch if it is smaller than the batchsize.

```
opt::Any
```

Any Flux-compatibale optimizer, typically a `struct`  that holds all the optimiser parameters along with a definition of `apply!` that defines how to apply the update rule associated with the optimizer.

```
loss::Function
```

The loss function used to calculate the reconstruction error, see [https://fluxml.ai/Flux.jl/stable/models/losses/](https://fluxml.ai/Flux.jl/stable/models/losses/) for examples.

**Examples**

```julia
using OutlierDetection: AE, fit, transform
detector = AE()
X = rand(10, 100)
model, scores = fit(detector, X)
transform(detector, model, X)
```

**References**

[1] Aggarwal, Charu C. (2017): Outlier Analysis.


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/models/neural/ae.jl#LL8-L34' class='documenter-source'>source</a><br>


<a id='DeepSAD'></a>

<a id='DeepSAD-1'></a>

### DeepSAD

<a id='OutlierDetection.DeepSAD' href='#OutlierDetection.DeepSAD'>#</a>
**`OutlierDetection.DeepSAD`** &mdash; *Type*.



```julia
DeepSAD(encoder = Chain()
        decoder = Chain()
        batchsize = 32
        epochs = 1
        shuffle = true
        partial = false
        opt = ADAM()
        loss = mse
        eta = 1
        eps = 1e-6
        callback = _ -> () -> ())
```

Deep Semi-Supervised Anomaly detection technique based on the distance to a hypersphere center as described in [1].

**Parameters**

```
encoder::Chain
```

Transforms the input data into a latent state with a fixed shape.

```
decoder::Chain
```

Transforms the latent state back into the shape of the input data.

```
batchsize::Integer
```

The number of samples to work through before updating the internal model parameters.

```
epochs::Integer
```

The number of passes of the entire training dataset the machine learning algorithm has completed. 

```
shuffle::Bool
```

If `shuffle=true`, shuffles the observations each time iterations are re-started, else no shuffling is performed.

```
partial::Bool
```

If `partial=false`, drops the last mini-batch if it is smaller than the batchsize.

```
opt::Any
```

Any Flux-compatibale optimizer, typically a `struct`  that holds all the optimiser parameters along with a definition of `apply!` that defines how to apply the update rule associated with the optimizer.

```
loss::Function
```

The loss function used to calculate the reconstruction error, see [https://fluxml.ai/Flux.jl/stable/models/losses/](https://fluxml.ai/Flux.jl/stable/models/losses/) for examples.

```
eta::Real
```

Weighting parameter for the labeled data; i.e. higher values of eta assign higher weight to labeled data in the svdd loss function. For a sensitivity analysis of this parameter, see [1].

```
eps::Real
```

Because the inverse distance used in the svdd loss can lead to division by zero, the parameters `eps` is added for numerical stability.

```
callback::Function
```

*Experimental parameter that might change*. A function to be called after the model parameters have been updated that can call Flux's callback helpers, see [https://fluxml.ai/Flux.jl/stable/utilities/#Callback-Helpers-1](https://fluxml.ai/Flux.jl/stable/utilities/#Callback-Helpers-1).

**Notice:** The parameters `batchsize`, `epochs`, `shuffle`, `partial`, `opt` and `callback` can also be tuples of size 2, specifying the corresponding values for (1) pretraining and (2) training; otherwise the same values are used for pretraining and training.

**Examples**

```julia
using OutlierDetection: DeepSAD, fit, transform
detector = DeepSAD()
X = rand(10, 100)
y = rand([-1,1], 100)
model, scores = fit(detector, X, y)
transform(detector, model, X)
```

**References**

[1] Ruff, Lukas; Vandermeulen, Robert A.; Görnitz, Nico; Binder, Alexander; Müller, Emmanuel; Müller, Klaus-Robert; Kloft, Marius (2019): Deep Semi-Supervised Anomaly Detection.


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/models/neural/deepsad.jl#LL5-L50' class='documenter-source'>source</a><br>


<a id='ESAD'></a>

<a id='ESAD-1'></a>

### ESAD

<a id='OutlierDetection.ESAD' href='#OutlierDetection.ESAD'>#</a>
**`OutlierDetection.ESAD`** &mdash; *Type*.



```julia
ESAD(encoder = Chain()
     decoder = Chain()
     batchsize = 32
     epochs = 1
     shuffle = false
     partial = true
     opt = ADAM()
     λ1 = 1
     λ2 = 1
     noise = identity)
```

End-to-End semi-supervised anomaly detection algorithm similar to DeepSAD, but without the pretraining phase. The algorithm was published by Huang et al., see [1].

**Parameters**

```
encoder::Chain
```

Transforms the input data into a latent state with a fixed shape.

```
decoder::Chain
```

Transforms the latent state back into the shape of the input data.

```
batchsize::Integer
```

The number of samples to work through before updating the internal model parameters.

```
epochs::Integer
```

The number of passes of the entire training dataset the machine learning algorithm has completed. 

```
shuffle::Bool
```

If `shuffle=true`, shuffles the observations each time iterations are re-started, else no shuffling is performed.

```
partial::Bool
```

If `partial=false`, drops the last mini-batch if it is smaller than the batchsize.

```
opt::Any
```

Any Flux-compatibale optimizer, typically a `struct`  that holds all the optimiser parameters along with a definition of `apply!` that defines how to apply the update rule associated with the optimizer.

```
λ1::Real
```

Weighting parameter of the norm loss, which minimizes the empirical variance and thus minimizes entropy.

```
λ2::Real
```

Weighting parameter of the assistent loss function to define the consistency between the two encoders.

```
noise::Function (AbstractArray{T} -> AbstractArray{T})
```

A function to be applied to a batch of input data to add noise, see [1] for an explanation.

**Examples**

```julia
using OutlierDetection: ESAD, fit, transform
detector = ESAD()
X = rand(10, 100)
y = rand([-1,1], 100)
model, scores = fit(detector, X, y)
transform(detector, model, X)
```

**References**

[1] Huang, Chaoqin; Ye, Fei; Zhang, Ya; Wang, Yan-Feng; Tian, Qi (2020): ESAD: End-to-end Deep Semi-supervised Anomaly Detection.


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/models/neural/esad.jl#LL6-L42' class='documenter-source'>source</a><br>

