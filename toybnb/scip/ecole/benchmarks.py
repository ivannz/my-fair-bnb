from typing import Iterable

from ecole import instance as ei
from ecole.core import RandomGenerator
from numpy.random import SeedSequence


def get_generators(config: dict, entropy: int = None) -> dict[str, dict[str, Iterable]]:
    out = {}

    # fork the seed sequence from the given entropy
    seed = entropy if isinstance(entropy, SeedSequence) else SeedSequence(entropy)
    ss = seed.spawn(len(config))

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
            # in sec 5.1 p.6 of (Gasse et al. 2019) the Erdos-Renyi graph is generated
            # with affinity 4, which trnasltes to edge probability of 1/4
            ei.IndependentSetGenerator: {
                "train": dict(
                    n_nodes=500, graph_type=erdos_renyi, edge_probability=0.25
                ),
                "test_easy": dict(
                    n_nodes=500, graph_type=erdos_renyi, edge_probability=0.25
                ),
                "test_medium": dict(
                    n_nodes=1000, graph_type=erdos_renyi, edge_probability=0.25
                ),
                "test_hard": dict(
                    n_nodes=1500, graph_type=erdos_renyi, edge_probability=0.25
                ),
            },
        },
        entropy,
    )


def scavuzzo2022(entropy: int = None) -> dict[str, dict[str, Iterable]]:
    """Initialize Ecole's instance generators to the setup in (Scavuzzo et al. 2022)"""
    # see [(Scavuzzo et al. 2022)](https://openreview.net/forum?id=M4OllVd70mJ)

    # settings from 01_generate_instances
    # XXX specified only if deviate from https://github.com/ds4dm/ecole
    #  as of commit SHA `172ea1b19c643438faa7696ce885d1ac4de3aa24`
    barabasi_albert = ei.IndependentSetGenerator.barabasi_albert
    # {
    #     "train": 10000, "valid": 2000, "test": 100, "transfer": 100,
    # }
    return get_generators(
        {
            # cauctions
            # XXX `add_item_prob` defaults to 0.9, overridden to 0.7 in __main__
            ei.CombinatorialAuctionGenerator: {
                "train": dict(n_items=100, n_bids=500, add_item_prob=0.7),
                "valid": dict(n_items=100, n_bids=500, add_item_prob=0.7),
                "test": dict(n_items=100, n_bids=500, add_item_prob=0.7),
                "transfer": dict(n_items=200, n_bids=1000, add_item_prob=0.7),
            },
            # setcover
            ei.SetCoverGenerator: {
                "train": dict(n_rows=400, n_cols=750),
                "valid": dict(n_rows=400, n_cols=750),
                "test": dict(n_rows=400, n_cols=750),
                "transfer": dict(n_rows=500, n_cols=1000),
            },
            # indset
            # XXX the default graph generator for MIS is Barabasi-Albert with
            #  affinity=4, unlike Gasse et al. 2019, who use Erdos-Renyi with
            #  edge_probability=0.25
            ei.IndependentSetGenerator: {
                "train": dict(n_nodes=500, graph_type=barabasi_albert, affinity=4),
                "valid": dict(n_nodes=500, graph_type=barabasi_albert, affinity=4),
                "test": dict(n_nodes=500, graph_type=barabasi_albert, affinity=4),
                "transfer": dict(n_nodes=1000, graph_type=barabasi_albert, affinity=4),
            },
            # the defaults coincide with the hardcoded demand, capacity, fixed-cost
            ei.CapacitatedFacilityLocationGenerator: {
                "train": dict(n_customers=35, n_facilities=35, ratio=5),
                "valid": dict(n_customers=35, n_facilities=35, ratio=5),
                "test": dict(n_customers=35, n_facilities=35, ratio=5),
                "transfer": dict(n_customers=60, n_facilities=35, ratio=5),
            },
        },
        entropy,
    )
