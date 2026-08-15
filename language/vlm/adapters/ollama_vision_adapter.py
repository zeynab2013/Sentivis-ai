"""Ollama multimodal vision adapter (Gemma 3 Vision and compatible tags)."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

from core.constants.pipeline_stages import PipelineStage
from core.contracts.image import PreprocessedImage
from core.contracts.language import RawCaption
from core.contracts.reasoning import SceneUnderstanding
from core.exceptions.language import InferenceError, ModelLoadError
from language.vlm.adapters.base import BaseVisionAdapter


class OllamaVisionAdapter(BaseVisionAdapter):
    """Runs vision understanding through a local Ollama multimodal model."""

    def __init__(
        self,
        model_id: str,
        preferred_device: str = "cpu",
        *,
        base_url: str = "http://127.0.0.1:11434",
    ) -> None:
        super().__init__(model_id, "gemma_vision", preferred_device)
        self._base_url = base_url.rstrip("/")

    def load(self) -> None:
        if self._loaded:
            return
        try:
            request = urllib.request.Request(f"{self._base_url}/api/tags", method="GET")
            with urllib.request.urlopen(request, timeout=3.0) as response:
                _ = response.read()
            self._loaded = True
        except Exception as exc:
            raise ModelLoadError(
                "Ollama vision model could not be prepared.",
                f"Ollama vision adapter load failed: {exc}",
                stage=PipelineStage.BLIP_UNDERSTANDING,
            ) from exc

    def describe(self, image: PreprocessedImage) -> RawCaption:
        prompt = (
            "Describe this photograph as a careful human observer in one natural paragraph. "
            "Include the overall scene, main subjects, actions, clothing and colors when clear, "
            "important objects, background activity, spatial layout, and environment. "
            "For a detailed scene write about 80–140 words; keep simple scenes shorter. "
            "Do not invent identities, emotions, or unseen objects."
        )
        return self._run(image, prompt)

    def narrate(self, image: PreprocessedImage, understanding: SceneUnderstanding) -> RawCaption:
        brief = (understanding.evidence_brief or "")[:1600]
        prompt = (
            "Look at the image carefully and write one coherent, information-rich observational "
            "paragraph. Use the verified evidence brief for grounding; never invent facts. "
            "Connect subjects, actions, relationships, colors, objects, background, and place "
            "into flowing prose — not a checklist. Aim for roughly 80–140 words on complex "
            "scenes; stay shorter on simple ones. Avoid openings like 'The image shows'.\n\n"
            f"Evidence brief:\n{brief}\n\n"
            "Paragraph:"
        )
        return self._run(image, prompt)

    def _run(self, image: PreprocessedImage, prompt: str) -> RawCaption:
        if not self._loaded:
            raise InferenceError(
                "Vision model is not ready.",
                "Ollama vision before load",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            )
        try:
            pil = self._pil(image)
            buffer = io.BytesIO()
            pil.save(buffer, format="JPEG", quality=92)
            image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            payload = {
                "model": self._model_id,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0.15, "num_predict": 420},
            }
            body = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                f"{self._base_url}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=300.0) as response:
                data = json.loads(response.read().decode("utf-8"))
            text = str(data.get("response", "")).strip()
            if not text:
                raise InferenceError(
                    "Ollama vision returned empty text.",
                    "empty multimodal response",
                    stage=PipelineStage.BLIP_UNDERSTANDING,
                    recoverable=True,
                )
            return RawCaption(text=text, source="gemma_vision", confidence=0.9)
        except InferenceError:
            raise
        except urllib.error.URLError as exc:
            raise InferenceError(
                "Ollama vision inference failed.",
                str(exc),
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            ) from exc
        except Exception as exc:
            raise InferenceError(
                "Ollama vision inference failed.",
                str(exc),
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            ) from exc
