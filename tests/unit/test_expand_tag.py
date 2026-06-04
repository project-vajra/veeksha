import pytest

from vidhi import create_class_from_dict, load_yaml_config
from vidhi.utils import expand_dict

from veeksha.config.generator.length import FixedLengthGeneratorConfig


def test_expand_tag_cartesian_product(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text("a: !expand [1, 2]\nb: !expand [3, 4]\n", encoding="utf-8")

    cfg = load_yaml_config(str(cfg_path))
    exploded = expand_dict(cfg)

    assert len(exploded) == 4
    assert {(d["a"], d["b"]) for d in exploded} == {(1, 3), (1, 4), (2, 3), (2, 4)}


def test_list_without_expand_tag_kept_as_is(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text("a: [1, 2]\n", encoding="utf-8")

    cfg = load_yaml_config(str(cfg_path))
    assert expand_dict(cfg) == [{"a": [1, 2]}]


def test_expand_tag_inside_nested_dict(tmp_path):
    """!expand inside a nested dict (non-list) is expanded."""
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        "a: !expand [1, 2]\nopts:\n  x: !expand [3, 4]\n", encoding="utf-8"
    )

    cfg = load_yaml_config(str(cfg_path))
    exploded = expand_dict(cfg)

    assert len(exploded) == 4
    assert {(d["a"], d["opts"]["x"]) for d in exploded} == {
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
    }
