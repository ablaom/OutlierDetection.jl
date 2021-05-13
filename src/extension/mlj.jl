function MMI.fit(detector::UnsupervisedDetector, verbosity::Int, X, y=nothing)
    fitresult = fit(detector, X)
    fitresult, nothing, (scores=fitresult.scores,)
end

function MMI.fit(detector::SupervisedDetector, verbosity::Int, X, y)
    fitresult = fit(detector, X, y)
    fitresult, nothing, (scores=fitresult.scores,)
end

function MMI.transform(detector::Detector, fitresult::Fit, X)
    score(detector, fitresult, X)
end

function MMI.predict(detector::Detector, fitresult::Fit, X)
    scores = detector.normalize(fitresult.scores, score(detector, fitresult, X))[2]
    MMI.UnivariateFinite([CLASS_NORMAL, CLASS_OUTLIER], hcat(1 .- scores, scores), pool=missing)
end

function MMI.predict_mode(detector::Detector, fitresult::Fit, X)
    train, test = detector.normalize(fitresult.scores, score(detector, fitresult, X))
    ifelse.(test .> quantile(train, 1 - detector.outlier_fraction), CLASS_OUTLIER, CLASS_NORMAL)
end

# pretend that unlabeled calls are supervised
struct Unlabeled <: AbstractVector{Multiclass{2}}
    n_rows::Int
end
MLJBase.source(u::Unlabeled) = Source(u, typeof(u))
MLJBase.nrows(u::Unlabeled) = u.n_rows

# enable unsupervised machine call syntax
MMI.machine(detector::UnsupervisedDetector, X) = machine(detector, X, Unlabeled(nrows(X)))

# specify scitypes
MMI.input_scitype(::Type{<:Detector}) = Union{MMI.Table(MMI.Continuous), AbstractMatrix{MMI.Continuous}}
MMI.output_scitype(::Type{<:Detector}) = AbstractVector{<:MMI.Continuous}
MMI.target_scitype(::Type{<:Detector}) = AbstractVector{<:MMI.Finite}

# data front-end for fit (supervised):
MMI.reformat(::Detector, X, y) = (MMI.matrix(X, transpose=true), y)
MMI.reformat(::Detector, X, y, w) = (MMI.matrix(X, transpose=true), y, w) 
MMI.selectrows(::Detector, I, Xmatrix, y) = (view(Xmatrix, :, I), view(y, I))
MMI.selectrows(::Detector, I, Xmatrix, y, w) = (view(Xmatrix, :, I), view(y, I), view(w, I))

# data front-end for fit (unsupervised)/predict/transform
MMI.reformat(::Detector, X) = (MMI.matrix(X, transpose=true),)
MMI.selectrows(::Detector, I, Xmatrix) = (view(Xmatrix, :, I),)

MODELS = (ABOD, COF, DNN, KNN, LOF, AE, DeepSAD, ESAD)
MMI.metadata_pkg.(MODELS,
    package_name="OutlierDetection.jl",
    package_uuid="262411bb-c475-4342-ba9e-03b8c0183ca6",
    package_url="https://github.com/davnn/OutlierDetection.jl",
    is_pure_julia=true,
    package_license="MIT",
    is_wrapper=false)
