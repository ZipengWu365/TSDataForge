from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

from .compiler import Compiler
from .interventions import InterventionLike
from .policies import Policy
from .series import GeneratedSeries
from .specs import SeriesSpec


@dataclass
class CounterfactualPair:
    factual: GeneratedSeries
    counterfactual: GeneratedSeries
    base_spec: SeriesSpec
    counterfactual_spec: SeriesSpec
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_spec": self.base_spec.to_dict(),
            "counterfactual_spec": self.counterfactual_spec.to_dict(),
            "metadata": dict(self.metadata),
        }



def with_intervention(spec: SeriesSpec, intervention: InterventionLike) -> SeriesSpec:
    latent = spec.latent
    if hasattr(latent, "interventions"):
        existing = tuple(getattr(latent, "interventions") or ())
        latent = replace(latent, interventions=existing + (intervention,))
    elif hasattr(latent, "intervention"):
        latent = replace(latent, intervention=intervention)
    else:
        raise ValueError(
            f"Latent component {type(spec.latent).__name__} does not expose `interventions` or `intervention`."
        )
    tags = tuple(dict.fromkeys((*spec.tags, "intervention")))
    return replace(spec, latent=latent, tags=tags)



def with_policy(
    spec: SeriesSpec,
    policy: Policy,
    *,
    counterfactual_policies: Sequence[Policy] | None = None,
) -> SeriesSpec:
    latent = spec.latent
    if not hasattr(latent, "policy"):
        raise ValueError(f"Latent component {type(spec.latent).__name__} does not expose `policy`.")
    kwargs: dict[str, Any] = {"policy": policy}
    if counterfactual_policies is not None and hasattr(latent, "counterfactual_policies"):
        kwargs["counterfactual_policies"] = tuple(counterfactual_policies)
    latent = replace(latent, **kwargs)
    tags = tuple(dict.fromkeys((*spec.tags, "policy_driven")))
    return replace(spec, latent=latent, tags=tags)



def generate_counterfactual_pair(
    *,
    spec: SeriesSpec,
    length: int,
    seed: int | None = None,
    intervention: InterventionLike | None = None,
    policy: Policy | None = None,
) -> CounterfactualPair:
    """Compile a factual / counterfactual pair with matched randomness.

    The same `seed` is used for both rollouts so the stochastic innovation terms
    remain comparable. This is especially useful for intervention and policy
    comparison experiments.
    """

    factual = Compiler(seed=seed).compile(spec, length=length)
    cf_spec = spec
    meta: dict[str, Any] = {"seed": seed}
    if intervention is not None:
        cf_spec = with_intervention(cf_spec, intervention)
        meta["intervention"] = intervention.to_dict() if hasattr(intervention, "to_dict") else dict(intervention)
    if policy is not None:
        cf_spec = with_policy(cf_spec, policy)
        meta["policy"] = policy.to_dict()
    if cf_spec is spec:
        raise ValueError("Provide at least one of `intervention` or `policy` to generate a counterfactual pair.")
    counterfactual = Compiler(seed=seed).compile(cf_spec, length=length)
    return CounterfactualPair(
        factual=factual,
        counterfactual=counterfactual,
        base_spec=spec,
        counterfactual_spec=cf_spec,
        metadata=meta,
    )
