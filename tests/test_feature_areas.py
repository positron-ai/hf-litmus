"""Tests for feature area mapping."""

from __future__ import annotations

from hf_litmus.feature_areas import OP_FEATURE_MAP, classify_op_area


class TestOpFeatureMap:
    def test_scatter_is_moe(self):
        assert OP_FEATURE_MAP["aten.scatter"] == "MoE"

    def test_embedding_area(self):
        assert OP_FEATURE_MAP["aten.embedding"] == "embedding"

    def test_where_is_conditional(self):
        assert OP_FEATURE_MAP["aten.where"] == "conditional_ops"


class TestClassifyOpArea:
    def test_known_op(self):
        assert classify_op_area("aten.scatter_add") == "MoE"

    def test_dict_op(self):
        assert classify_op_area({"op": "aten.embedding"}) == "embedding"

    def test_unknown_op(self):
        assert classify_op_area("aten.totally_unknown") == "other"

    def test_empty_string(self):
        assert classify_op_area("") == "other"

    def test_none(self):
        assert classify_op_area(None) == "other"
