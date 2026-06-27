import math

from src.analysis.representation_collapse import CollapseAnalysisConfig, collapse_index, pair_category


def test_collapse_index_definition():
    assert collapse_index(0.2, 2.0) == 0.1
    assert math.isnan(collapse_index(0.2, 0.0))


def test_pair_categories_for_figure1():
    cfg = CollapseAnalysisConfig()
    assert pair_category(0.9, 1.0, cfg) == "cliff"
    assert pair_category(0.9, 0.2, cfg) == "similar_non_cliff"
    assert pair_category(0.2, 1.0, cfg) == "similar_intermediate"
