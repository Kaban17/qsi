import pytest

from qsi.models import CONFIG_NAMES, MODEL_ID, ConfigSpec, get_blocks, parse_config


def test_config_names_match_spec():
    assert CONFIG_NAMES == ["fp16", "bnb-8bit", "bnb-nf4", "rtn-8", "rtn-6", "rtn-4", "rtn-3", "rtn-2"]
    assert MODEL_ID == "google/gemma-2-2b-it"


@pytest.mark.parametrize(
    "name, spec",
    [
        ("fp16", ConfigSpec("fp16", None)),
        ("bnb-8bit", ConfigSpec("bnb", 8)),
        ("bnb-nf4", ConfigSpec("bnb", 4)),
        ("rtn-8", ConfigSpec("rtn", 8)),
        ("rtn-6", ConfigSpec("rtn", 6)),
        ("rtn-4", ConfigSpec("rtn", 4)),
        ("rtn-3", ConfigSpec("rtn", 3)),
        ("rtn-2", ConfigSpec("rtn", 2)),
    ],
)
def test_parse_config(name, spec):
    assert parse_config(name) == spec


def test_every_config_name_parses():
    assert all(parse_config(n) for n in CONFIG_NAMES)


@pytest.mark.parametrize("name", ["rtn-x", "gguf", "rtn-", "rtn-1", "bnb-4bit", "", "FP16", "rtn-4 "])
def test_parse_config_rejects(name):
    with pytest.raises(ValueError):
        parse_config(name)


def test_get_blocks(tiny_model):
    assert len(get_blocks(tiny_model)) == 3
