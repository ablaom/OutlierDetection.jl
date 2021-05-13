using MLJModelInterface

"""
    Template

Converts an expression of a template struct to data.
"""
struct Template
    expr::Expr
    params::Vector{Symbol}
    defaults::Dict{Symbol, Any}
    constraints::Dict{Symbol, Any}
    Template(expr::Expr) = begin
        ex, _, params, defaults, constraints = MLJModelInterface._process_model_def(@__MODULE__, expr)
        new(ex, params, defaults, constraints)
    end
end

"""
    @detector_model

Macro to help define models with a specified template.
"""
macro detector_model(template_symbol::Symbol, ex)
    # TODO: Is there a nicer way to do this without evaluating a symbol?
    template = eval(template_symbol)
    @assert isa(template, Template) "The provided variable '$template_symbol' did not evaluate to a Template"

    # Add the necessary normalization function to the struct parameters
    push!(ex.args[3].args, :(normalize::Function = $normalize))
    push!(ex.args[3].args, :(outlier_fraction::Real = 0.1))

    # process expression
    ex, modelname, params, defaults, constraints = MLJModelInterface._process_model_def(__module__, ex)

    # merge parsed data
    params = vcat(params, template.params)
    defaults = merge(defaults, template.defaults)
    constraints = merge(constraints, template.constraints)

    # Add the template struct parameters to the expression struct parameters
    push!(ex.args[3].args, template.expr.args[3])

    # keyword constructor
    const_ex = MLJModelInterface._model_constructor(modelname, params, defaults)

    # associate the constructor with the definition of the struct
    push!(ex.args[3].args, const_ex)
    # cleaner
    clean_ex = MLJModelInterface._model_cleaner(modelname, defaults, constraints)

    esc(
        quote
            Base.@__doc__ $ex
            export $modelname
            $clean_ex
        end
    )
end

# Copied definition without template logic, basically mirroring @mlj_model and adding the normalization field
macro detector_model(ex)
    push!(ex.args[3].args, :(normalize::Function = $normalize))
    ex, modelname, params, defaults, constraints = MLJModelInterface._process_model_def(__module__, ex)
    const_ex = MLJModelInterface._model_constructor(modelname, params, defaults)
    push!(ex.args[3].args, const_ex)
    clean_ex = MLJModelInterface._model_cleaner(modelname, defaults, constraints)
    esc(
        quote
            Base.@__doc__ $ex
            export $modelname
            $clean_ex
        end
    )
end
