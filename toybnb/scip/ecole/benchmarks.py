from ecole import instance as ei
from ecole.core import RandomGenerator
from numpy.random import SeedSequence
from typing import Iterable


def get_generators(config: dict, entropy: int = None) -> dict[str, dict[str, Iterable]]:
    out = {}

    # fork the seed sequence from the given entropy
    ss = SeedSequence(entropy).spawn(len(config))

    # create instance generators
    for seed, (gen, cfg) in zip(ss, config.items()):
        # generate seeds for ecole's prng
        rngs = map(RandomGenerator, seed.generate_state(len(cfg)))

        name = gen.__name__.removesuffix("Generator")
        out[name] = {n: gen(**arg, rng=next(rngs)) for n, arg in cfg.items()}

    return out


def gasse2019(entropy: int = None) -> dict[str, dict[str, Iterable]]:
    """Initialize Ecole's instance generators to the setup in (Gasse et al. 2019)"""
    # see [(Gasse et al. 2019; p. 6 sec. 5.1)](
    #     https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
    # )
    erdos_renyi = ei.IndependentSetGenerator.erdos_renyi
    return get_generators(
        {
            ei.SetCoverGenerator: {
                "train": dict(n_rows=500, n_cols=1000),
                "test_easy": dict(n_rows=500, n_cols=1000),
                "test_medium": dict(n_rows=1000, n_cols=1000),
                "test_hard": dict(n_rows=2000, n_cols=1000),
            },
            ei.CombinatorialAuctionGenerator: {
                "train": dict(n_items=100, n_bids=500),
                "test_easy": dict(n_items=100, n_bids=500),
                "test_medium": dict(n_items=200, n_bids=1000),
                "test_hard": dict(n_items=300, n_bids=1500),
            },
            ei.CapacitatedFacilityLocationGenerator: {
                "train": dict(n_customers=100, n_facilities=100),
                "test_easy": dict(n_customers=100, n_facilities=100),
                "test_medium": dict(n_customers=200, n_facilities=100),
                "test_hard": dict(n_customers=400, n_facilities=100),
            },
            ei.IndependentSetGenerator: {
                "train": dict(n_nodes=500, graph_type=erdos_renyi, affinity=4),
                "test_easy": dict(n_nodes=500, graph_type=erdos_renyi, affinity=4),
                "test_medium": dict(n_nodes=1000, graph_type=erdos_renyi, affinity=4),
                "test_hard": dict(n_nodes=1500, graph_type=erdos_renyi, affinity=4),
            },
        },
        entropy,
    )
