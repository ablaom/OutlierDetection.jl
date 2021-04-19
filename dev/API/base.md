
<a id='Base'></a>

<a id='Base-1'></a>

# Base


Here we define the abstract supertypes that all outlier detectors share as well as the necessary [`fit`](base.md#OutlierDetection.fit) and [`transform`](base.md#OutlierDetection.transform) methods that have to be implemented for each detector.


<a id='Types'></a>

<a id='Types-1'></a>

## Types


<a id='Detector'></a>

<a id='Detector-1'></a>

### Detector

<a id='OutlierDetection.Detector' href='#OutlierDetection.Detector'>#</a>
**`OutlierDetection.Detector`** &mdash; *Type*.



```julia
Detector::Union{<:SupervisedDetector, <:UnsupervisedDetector}
```

The union type of all implemented detectors, including supervised, semi-supervised and unsupervised detectors. *Note:* A semi-supervised detector can be seen as a supervised detector with a specific class representing unlabeled data.


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/base.jl#LL19-L24' class='documenter-source'>source</a><br>


<a id='SupervisedDetector'></a>

<a id='SupervisedDetector-1'></a>

### SupervisedDetector

<a id='OutlierDetection.SupervisedDetector' href='#OutlierDetection.SupervisedDetector'>#</a>
**`OutlierDetection.SupervisedDetector`** &mdash; *Type*.



```julia
SupervisedDetector
```

This abstract type forms the basis for all implemented supervised outlier detection algorithms. To implement a new `SupervisedDetector` yourself, you have to implement the `fit(detector, X, y)::DetectorModel` and `transform(detector, model, X)::Scores` methods. 


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/base.jl#LL10-L16' class='documenter-source'>source</a><br>


<a id='UnsupervisedDetector'></a>

<a id='UnsupervisedDetector-1'></a>

### UnsupervisedDetector

<a id='OutlierDetection.UnsupervisedDetector' href='#OutlierDetection.UnsupervisedDetector'>#</a>
**`OutlierDetection.UnsupervisedDetector`** &mdash; *Type*.



```julia
UnsupervisedDetector
```

This abstract type forms the basis for all implemented unsupervised outlier detection algorithms. To implement a new `UnsupervisedDetector` yourself, you have to implement the `fit(detector, X)::DetectorModel` and `transform(detector, model, X)::Scores` methods. 


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/base.jl#LL1-L7' class='documenter-source'>source</a><br>


<a id='DetectorModel'></a>

<a id='DetectorModel-1'></a>

### DetectorModel

<a id='OutlierDetection.DetectorModel' href='#OutlierDetection.DetectorModel'>#</a>
**`OutlierDetection.DetectorModel`** &mdash; *Type*.



```julia
DetectorModel
```

A `DetectorModel` represents the learned behaviour for specific detector. This might include parameters in parametric models or other repesentations of the learned data in nonparametric models. In essence, it includes everything required to transform an instance to an outlier score.


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/base.jl#LL27-L33' class='documenter-source'>source</a><br>


<a id='Scores'></a>

<a id='Scores-1'></a>

### Scores

<a id='OutlierDetection.Scores' href='#OutlierDetection.Scores'>#</a>
**`OutlierDetection.Scores`** &mdash; *Type*.



```julia
Scores::AbstractVector{<:Real}
```

Scores are continuous values, where the range depends on the specific detector yielding the scores. *Note:* All detectors return increasing scores and higher scores are associated with higher outlierness. Concretely, scores are defined as an `AbstractVector{<:Real}`.


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/base.jl#LL36-L42' class='documenter-source'>source</a><br>


<a id='Labels'></a>

<a id='Labels-1'></a>

### Labels

<a id='OutlierDetection.Labels' href='#OutlierDetection.Labels'>#</a>
**`OutlierDetection.Labels`** &mdash; *Type*.



```julia
Labels::AbstractArray{<:Integer}
```

Labels are used for supervision and evaluation and are defined as an `AbstractArray{<:Integer}`. The convention for labels is that `-1` indicates outliers, `1` indicates inliers and `0` indicates unlabeled data in semi-supervised tasks.


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/base.jl#LL53-L58' class='documenter-source'>source</a><br>


<a id='Data'></a>

<a id='Data-1'></a>

### Data

<a id='OutlierDetection.Data' href='#OutlierDetection.Data'>#</a>
**`OutlierDetection.Data`** &mdash; *Type*.



```julia
Data::AbstractArray{<:Real}
```

The raw input data for every detector is defined as`AbstractArray{<:Real}` and should be a column-major n-dimensional array. The input data used to [`fit`](base.md#OutlierDetection.fit) a [`Detector`](base.md#OutlierDetection.Detector) and [`transform`](base.md#OutlierDetection.transform) [`Data`](base.md#OutlierDetection.Data).


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/base.jl#LL45-L50' class='documenter-source'>source</a><br>


<a id='Functions'></a>

<a id='Functions-1'></a>

## Functions


<a id='fit'></a>

<a id='fit-1'></a>

### fit

<a id='OutlierDetection.fit' href='#OutlierDetection.fit'>#</a>
**`OutlierDetection.fit`** &mdash; *Function*.



```julia
fit(detector,
    X,
    y)
```

Fit a specified unsupervised, supervised or semi-supervised outlier detector. That is, learn a `DetectorModel` from input data `X` and, in the supervised and semi-supervised setting, labels `y`. In a supervised setting, the label `-1` represents outliers and `1` inliers. In a semi-supervised setting, the label `0` additionally represents unlabeled data. *Note:* Unsupervised detectors can be fitted without specifying `y`, otherwise `y` is simply ignore.

**Parameters**

```
detector::Detector
```

Any [`UnsupervisedDetector`](base.md#OutlierDetection.UnsupervisedDetector) or [`SupervisedDetector`](base.md#OutlierDetection.SupervisedDetector) implementation.

```
X::Union{AbstractMatrix, Tables.jl-compatible}
```

Either a column-major matrix or a row-major [Tables.jl-compatible](https://github.com/JuliaData/Tables.jl) source.

**Returns**

```
model::DetectorModel
```

The learned model of the given detector, which contains all the necessary information for later prediction.

```
scores::Scores
```

The achieved outlier scores of the given training data `X`.

**Examples**

```julia
using OutlierDetection: KNN, fit, transform
detector = KNN()
X = rand(10, 100)
model, scores = fit(detector, X)
transform(detector, model, X)
```


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/base.jl#LL89-L116' class='documenter-source'>source</a><br>


<a id='transform'></a>

<a id='transform-1'></a>

### transform

<a id='OutlierDetection.transform' href='#OutlierDetection.transform'>#</a>
**`OutlierDetection.transform`** &mdash; *Function*.



```julia
transform(detector,
          model,
          X)
```

Transform input data `X` to outlier scores using an [`UnsupervisedDetector`](base.md#OutlierDetection.UnsupervisedDetector) or [`SupervisedDetector`](base.md#OutlierDetection.SupervisedDetector) and a corresponding [`DetectorModel`](base.md#OutlierDetection.DetectorModel).

**Parameters**

```
detector::Detector
```

Any [`UnsupervisedDetector`](base.md#OutlierDetection.UnsupervisedDetector) or [`SupervisedDetector`](base.md#OutlierDetection.SupervisedDetector) implementation.

```
model::DetectorModel
```

The model learned from using [`fit`](base.md#OutlierDetection.fit) with a supervised or unsupervised [`Detector`](base.md#OutlierDetection.Detector)

```
X::Union{AbstractMatrix, Tables.jl-compatible}
```

Either a column-major matrix or a row-major [Tables.jl-compatible](https://github.com/JuliaData/Tables.jl) source.

**Returns**

```
scores::Scores
```

The achieved outlier scores of the given test data `X`.

**Examples**

```julia
using OutlierDetection: KNN, fit, transform
detector = KNN()
X = rand(10, 100)
model, scores = fit(detector, X)
transform(detector, model, X)
```


<a target='_blank' href='https://github.com/davnn/OutlierDetection.jl/blob/ea70ca208859a6499674e7a6dd3af02b0aeae978/src/base.jl#LL122-L147' class='documenter-source'>source</a><br>

