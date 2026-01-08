from typing import List

import pytest

from veeksha.config.core.flat_dataclass import explode_dict as explode_dict_legacy
from veeksha.config.utils import create_class_from_dict, load_yaml_config
from veeksha.config.generator.length import FixedLengthGeneratorConfig
from veeksha.config.core.flat_dataclass import explode_dict as explode_dict_new


class _Dummy:
    __annotations__ = {"a": int, "b": int}
    base_poly_children_types = {}


class _DummyList:
    __annotations__ = {"a": List[int]}
    base_poly_children_types = {}


class _DummyNestedList:
    __annotations__ = {"a": int, "items": List[dict]}
    base_poly_children_types = {}


@pytest.mark.parametrize("explode_dict", [explode_dict_legacy, explode_dict_new])
def test_expand_tag_cartesian_product(tmp_path, explode_dict):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text("a: !expand [1, 2]\nb: !expand [3, 4]\n", encoding="utf-8")

    cfg = load_yaml_config(str(cfg_path))
    exploded = explode_dict(_Dummy, cfg)

    assert len(exploded) == 4
    assert {(d["a"], d["b"]) for d in exploded} == {(1, 3), (1, 4), (2, 3), (2, 4)}


@pytest.mark.parametrize("explode_dict", [explode_dict_legacy, explode_dict_new])
def test_list_without_expand_tag_errors_for_non_list_fields(tmp_path, explode_dict):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text("a: [1, 2]\n", encoding="utf-8")

    cfg = load_yaml_config(str(cfg_path))
    with pytest.raises(ValueError, match=r"!expand"):
        explode_dict(_Dummy, cfg)


@pytest.mark.parametrize("explode_dict", [explode_dict_legacy, explode_dict_new])
def test_list_fields_remain_literal_without_expand_tag(tmp_path, explode_dict):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text("a: [1, 2]\n", encoding="utf-8")

    cfg = load_yaml_config(str(cfg_path))
    assert explode_dict(_DummyList, cfg) == [{"a": [1, 2]}]


@pytest.mark.parametrize("explode_dict", [explode_dict_legacy, explode_dict_new])
def test_expand_tag_inside_list_fields(tmp_path, explode_dict):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        "a: !expand [1, 2]\nitems:\n  - x: !expand [3, 4]\n", encoding="utf-8"
    )

    cfg = load_yaml_config(str(cfg_path))
    exploded = explode_dict(_DummyNestedList, cfg)

    assert len(exploded) == 4
    assert {(d["a"], d["items"][0]["x"]) for d in exploded} == {
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
    }


def test_list_without_expand_tag_errors_for_nested_dataclass_fields():
    with pytest.raises(ValueError, match=r"!expand"):
        create_class_from_dict(FixedLengthGeneratorConfig, {"value": [512, 1024]})


