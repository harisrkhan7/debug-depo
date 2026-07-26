import sys
import types

from debug_depo.package_model import package_model


def test_package_preserves_interrupted_output_and_promotes_complete_model(
    tmp_path, monkeypatch
):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "model"
    output.mkdir()
    (output / "partial.safetensors").write_text("partial", encoding="utf-8")

    class Merged:
        def __init__(self):
            self.config = types.SimpleNamespace(use_cache=False)

        def save_pretrained(self, directory, **_kwargs):
            (directory / "config.json").write_text("{}", encoding="utf-8")
            (directory / "model.safetensors").write_text("complete", encoding="utf-8")

    class Peft:
        def merge_and_unload(self, **_kwargs):
            return Merged()

    class AutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    class PeftModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return Peft()

    class Tokenizer:
        def save_pretrained(self, directory):
            (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return Tokenizer()

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=PeftModel))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForCausalLM=AutoModel,
            AutoTokenizer=AutoTokenizer,
        ),
    )

    manifest = package_model("base", adapter, output)

    assert manifest["format"] == "standalone_huggingface_model"
    assert (output / "package_manifest.json").is_file()
    assert (output / "model.safetensors").read_text(encoding="utf-8") == "complete"
    preserved = list(tmp_path.glob(".model.incomplete-*"))
    assert len(preserved) == 1
    assert (preserved[0] / "partial.safetensors").read_text(encoding="utf-8") == "partial"
