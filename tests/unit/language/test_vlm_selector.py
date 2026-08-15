"""VLM selector unit tests."""



from core.config.vlm_config import VlmModelIds, VlmSelectionConfig

from language.vlm.selector import VlmSelector





def _config(**kwargs: object) -> VlmSelectionConfig:

    ids = VlmModelIds(

        gemma_vision="gemma3:4b",

        florence_base="florence-base-ft-id",

        florence_plain="florence-base-id",

        florence_large="florence-large-id",

        moondream="moondream-id",

        blip2="blip2-id",

        blip="blip-id",

        qwen="qwen-id",

        internvl="internvl-id",

    )

    base = dict(

        auto_select=True,

        preferred_adapter="",

        model_ids=ids,

        min_vram_gemma_vision_gb=0.0,

        min_vram_florence_base_gb=1.5,

        min_vram_florence_plain_gb=1.5,

        min_vram_moondream_gb=1.7,

        min_vram_blip2_gb=3.5,

        min_vram_florence_large_gb=3.5,

        min_vram_qwen_gb=6.0,

        min_vram_internvl_gb=12.0,

    )

    base.update(kwargs)

    return VlmSelectionConfig(**base)  # type: ignore[arg-type]





def test_selector_uses_preferred_adapter() -> None:

    choice = VlmSelector(_config(preferred_adapter="blip")).select()

    assert choice.adapter_name == "blip"

    assert choice.model_id == "blip-id"





def test_selector_cpu_prefers_florence_when_ollama_absent(monkeypatch) -> None:

    monkeypatch.setattr(VlmSelector, "_detect_vram_gb", staticmethod(lambda: 0.0))

    monkeypatch.setattr(VlmSelector, "_ollama_model_available", staticmethod(lambda _model: False))

    choice = VlmSelector(_config()).select()

    assert choice.adapter_name == "florence_base"





def test_selector_auto_prefers_florence_not_gemma_vision(monkeypatch) -> None:
    """Perception uses Florence; Gemma 3 4B stays on the text reasoning path."""
    monkeypatch.setattr(VlmSelector, "_detect_vram_gb", staticmethod(lambda: 2.0))
    monkeypatch.setattr(VlmSelector, "_ollama_model_available", staticmethod(lambda _model: True))
    choice = VlmSelector(_config()).select()
    assert choice.adapter_name == "florence_base"
    assert choice.model_id == "florence-base-ft-id"





def test_selector_2gb_prefers_florence_base_ft(monkeypatch) -> None:

    monkeypatch.setattr(VlmSelector, "_detect_vram_gb", staticmethod(lambda: 2.0))

    monkeypatch.setattr(VlmSelector, "_ollama_model_available", staticmethod(lambda _model: False))

    choice = VlmSelector(_config()).select()

    assert choice.adapter_name == "florence_base"

    assert choice.model_id == "florence-base-ft-id"





def test_selector_fallback_chain_quality_order() -> None:

    chain = VlmSelector(_config()).fallback_chain("florence_base")

    names = [item.adapter_name for item in chain]

    assert names == ["florence_plain", "gemma_vision", "blip"]
    assert "qwen" not in names
    assert "internvl" not in names
    assert "moondream" not in names
    assert "blip2" not in names


