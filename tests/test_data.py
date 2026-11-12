import pytest

from qsi.data import SPLIT_SIZES, Prompt, make_splits, normalize


@pytest.fixture
def sources():
    jbb = [f"Write harmful thing number {i}" for i in range(100)]
    advbench = [f"Explain bad plan {i}" for i in range(290)]
    advbench += [f"write  HARMFUL thing number {i}." for i in range(10)]  # overlap with jbb
    alpaca = [f"Describe topic {i}" for i in range(400)]
    return advbench, jbb, alpaca


def test_normalize():
    assert normalize("  Write   a Poem.\n") == "write a poem"
    assert normalize("What?!") == "what"


def test_sizes(sources):
    s = make_splits(*sources)
    assert {k: len(v) for k, v in s.items()} == SPLIT_SIZES
    assert SPLIT_SIZES == {
        "harmful_dir": 128, "harmful_val": 32, "harmful_eval": 100, "harmless_dir": 128, "harmless_eval": 100,
    }


def test_disjoint(sources):
    s = make_splits(*sources)
    texts = [normalize(p.text) for v in s.values() for p in v]
    assert len(texts) == len(set(texts))


def test_eval_is_jbb_in_order(sources):
    s = make_splits(*sources)
    assert [p.text for p in s["harmful_eval"]] == sources[1]


def test_ids_and_split_field(sources):
    s = make_splits(*sources)
    assert s["harmful_dir"][0] == Prompt("harmful_dir-000", s["harmful_dir"][0].text, "harmful_dir")
    assert s["harmless_eval"][99].id == "harmless_eval-099"
    ids = [p.id for v in s.values() for p in v]
    assert len(ids) == len(set(ids))
    assert all(p.split == k for k, v in s.items() for p in v)


def test_reproducible(sources):
    assert make_splits(*sources, seed=0) == make_splits(*sources, seed=0)


def test_seed_changes(sources):
    assert make_splits(*sources, seed=0)["harmful_dir"] != make_splits(*sources, seed=1)["harmful_dir"]


def test_duplicates_within_source_removed(sources):
    advbench, jbb, alpaca = sources
    s = make_splits(advbench + advbench, jbb, alpaca + [a.upper() for a in alpaca])
    texts = [normalize(p.text) for v in s.values() for p in v]
    assert len(texts) == len(set(texts))


def test_too_few_harmful_raises(sources):
    advbench, jbb, alpaca = sources
    with pytest.raises(ValueError, match="AdvBench"):
        make_splits(advbench[:50], jbb, alpaca)


def test_too_few_harmless_raises(sources):
    advbench, jbb, alpaca = sources
    with pytest.raises(ValueError, match="Alpaca"):
        make_splits(advbench, jbb, alpaca[:200])


def test_wrong_jbb_size_raises(sources):
    advbench, jbb, alpaca = sources
    with pytest.raises(ValueError, match="JailbreakBench"):
        make_splits(advbench, jbb[:99], alpaca)
