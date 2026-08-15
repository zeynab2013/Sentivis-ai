"""Florence engine falls back to BLIP when Florence-2 cannot load."""

from core.config.model_config import BlipModelConfig, FlorenceModelConfig
from language.florence.florence_engine import FlorenceEngine


def test_florence_engine_falls_back_to_blip_when_florence_unavailable(monkeypatch) -> None:
    florence = FlorenceModelConfig(
        model_id="microsoft/Florence-2-base-ft",
        preferred_device="cpu",
        max_new_tokens=32,
        fallback_to_blip=True,
    )
    blip = BlipModelConfig(
        model_id="Salesforce/blip-image-captioning-base",
        preferred_device="cpu",
        max_length=32,
    )
    engine = FlorenceEngine(florence, blip)

    def _fail_florence() -> bool:
        return False

    loaded = {"value": False}

    class _FakeBlip:
        def set_device(self, device: str) -> None:
            return None

        def load(self) -> None:
            loaded["value"] = True

        def release(self) -> None:
            return None

    monkeypatch.setattr(engine, "_try_load_florence", _fail_florence)
    monkeypatch.setattr("language.florence.florence_engine.BlipEngine", lambda config: _FakeBlip())
    engine.load()
    assert engine.backend == "blip_fallback"
    assert loaded["value"] is True
