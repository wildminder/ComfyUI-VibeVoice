"""Tests verifying no V1 artifacts remain after migration."""

import os
import pytest


class TestNoV1Artifacts:
    """Ensure V1 patterns are fully removed."""

    def test_no_node_class_mappings_in_init(self):
        """__init__.py should not export NODE_CLASS_MAPPINGS."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        init_path = os.path.join(root, "__init__.py")
        with open(init_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "NODE_CLASS_MAPPINGS" not in content

    def test_no_input_types_in_nodes(self):
        """Node files should not use INPUT_TYPES (V1 pattern)."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        nodes_dir = os.path.join(root, "nodes")
        for fname in os.listdir(nodes_dir):
            if fname.endswith(".py"):
                fpath = os.path.join(nodes_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                assert "INPUT_TYPES" not in content, f"INPUT_TYPES found in {fname}"

    def test_no_return_types_in_nodes(self):
        """Node files should not use RETURN_TYPES (V1 pattern)."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        nodes_dir = os.path.join(root, "nodes")
        for fname in os.listdir(nodes_dir):
            if fname.endswith(".py"):
                fpath = os.path.join(nodes_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                assert "RETURN_TYPES" not in content, f"RETURN_TYPES found in {fname}"

    def test_no_function_attribute_in_nodes(self):
        """Node files should not use FUNCTION = '...' (V1 pattern)."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        nodes_dir = os.path.join(root, "nodes")
        for fname in os.listdir(nodes_dir):
            if fname.endswith(".py"):
                fpath = os.path.join(nodes_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                # V1 uses "FUNCTION = '...'" as a class attribute
                assert "FUNCTION =" not in content, f"FUNCTION = found in {fname}"

    def test_uses_comfy_node_base(self):
        """Node class should inherit from io.ComfyNode."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tts_path = os.path.join(root, "nodes", "tts_node.py")
        with open(tts_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "io.ComfyNode" in content

    def test_uses_define_schema(self):
        """Node should use define_schema (V3 pattern)."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tts_path = os.path.join(root, "nodes", "tts_node.py")
        with open(tts_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "define_schema" in content

    def test_uses_comfy_entrypoint(self):
        """vibevoice_nodes.py should use comfy_entrypoint (V3 pattern)."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        nodes_path = os.path.join(root, "vibevoice_nodes.py")
        with open(nodes_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "comfy_entrypoint" in content
        assert "ComfyExtension" in content
