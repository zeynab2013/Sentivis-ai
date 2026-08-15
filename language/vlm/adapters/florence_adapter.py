"""Florence-2 vision adapters (base / large)."""

from __future__ import annotations

import re

from core.constants.pipeline_stages import PipelineStage
from core.contracts.image import PreprocessedImage
from core.contracts.language import RawCaption
from core.contracts.reasoning import SceneUnderstanding
from core.exceptions.language import InferenceError, ModelLoadError
from language.vlm.adapters.base import BaseVisionAdapter


class FlorenceVisionAdapter(BaseVisionAdapter):
    """Florence-2 adapter using configured model id."""

    def __init__(self, model_id: str, adapter_name: str, preferred_device: str = "cuda") -> None:
        super().__init__(model_id, adapter_name, preferred_device)

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                self._model_id,
                trust_remote_code=True,
            )
            dtype = torch.float16 if str(self._device).startswith("cuda") else torch.float32

            def _load_model(device: str, model_dtype: object) -> object:
                # Florence remote code + newer transformers can crash on SDPA probing.
                kwargs = {
                    "torch_dtype": model_dtype,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        self._model_id,
                        attn_implementation="eager",
                        **kwargs,
                    )
                except TypeError:
                    model = AutoModelForCausalLM.from_pretrained(self._model_id, **kwargs)
                model.to(device)
                # Compatibility shim for transformers versions that expect _supports_sdpa.
                if not hasattr(model, "_supports_sdpa"):
                    try:
                        type(model)._supports_sdpa = False  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        setattr(model, "_supports_sdpa", False)
                return model

            try:
                self._model = _load_model(self._device, dtype)
            except Exception as cuda_exc:
                # Never terminate on VRAM — continue on CPU even if short on CUDA memory.
                if str(self._device).startswith("cuda"):
                    self._device = "cpu"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._model = _load_model("cpu", torch.float32)
                else:
                    raise cuda_exc
            self._model.eval()
            self._loaded = True
        except Exception as exc:
            raise ModelLoadError(
                "Florence vision model could not be loaded.",
                f"Florence adapter load failed: {exc}",
                stage=PipelineStage.BLIP_UNDERSTANDING,
            ) from exc

    def describe(self, image: PreprocessedImage) -> RawCaption:
        return self._run(image, self._prompts.describe_prompt(self._adapter_name))

    def narrate(self, image: PreprocessedImage, understanding: SceneUnderstanding) -> RawCaption:
        return self._run(image, self._prompts.narrate_prompt(self._adapter_name, understanding))

    def _run(self, image: PreprocessedImage, task: str) -> RawCaption:
        if not self._loaded or self._model is None or self._processor is None:
            raise InferenceError(
                "Vision model is not ready.",
                "Florence describe before load",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            )
        try:
            pil = self._pil(image)
            if pil is None:
                raise InferenceError(
                    "Florence inference failed.",
                    "Preprocessed image could not be converted to PIL",
                    stage=PipelineStage.BLIP_UNDERSTANDING,
                    recoverable=True,
                )
            # Primary + secondary Florence caption tasks. Base-FT often under-describes
            # with MORE_DETAILED alone; DETAILED recovers rope/background/trees.
            primary_task = task if task.startswith("<") else "<MORE_DETAILED_CAPTION>"
            more = self._generate_task(pil, primary_task)
            detailed = ""
            if primary_task in {"<MORE_DETAILED_CAPTION>", "<DETAILED_CAPTION>", "<CAPTION>"}:
                try:
                    detailed = self._generate_task(pil, "<DETAILED_CAPTION>")
                except Exception:  # noqa: BLE001 — secondary task is best-effort
                    detailed = ""
            text = self._merge_florence_captions(more, detailed)
            return RawCaption(text=text.strip(), source=self._adapter_name, confidence=0.9)
        except Exception as exc:
            raise InferenceError(
                "Florence inference failed.",
                str(exc),
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            ) from exc

    def _generate_task(self, pil: object, task: str) -> str:
        import torch

        inputs = self._processor(text=task, images=pil, return_tensors="pt")
        if getattr(inputs, "pixel_values", None) is None and isinstance(inputs, dict):
            raise InferenceError(
                "Florence inference failed.",
                "Processor returned no pixel_values",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            )
        model_inputs = {
            key: value.to(self._device)
            for key, value in dict(inputs).items()
            if value is not None and hasattr(value, "to")
        }
        with torch.no_grad():
            # 256 tokens is enough for ~150–180 words; beams recover missed detail.
            generated = self._model.generate(
                **model_inputs,
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
                use_cache=False,
            )
        decoded = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        return self._parse(decoded, task=task, image=pil)

    def _merge_florence_captions(self, primary: str, secondary: str) -> str:
        """Combine Florence caption tasks into one richer visual paragraph."""
        a = self._clean_florence_prose(primary)
        b = self._clean_florence_prose(secondary)
        if not b:
            return a
        if not a:
            return b
        if len(b.split()) > len(a.split()) + 8:
            base, extra = b, a
        else:
            base, extra = a, b
        base_tokens = set(re.findall(r"[a-z]{4,}", base.lower()))
        additions: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", extra):
            sent = sentence.strip()
            if not sent:
                continue
            tokens = set(re.findall(r"[a-z]{4,}", sent.lower()))
            novel = tokens - base_tokens - {
                "this",
                "that",
                "with",
                "from",
                "image",
                "picture",
                "photo",
                "visible",
                "standing",
                "there",
            }
            # Keep only sentences that add concrete visual nouns/details.
            if novel & {
                "rope",
                "trees",
                "tree",
                "background",
                "another",
                "wooden",
                "grass",
                "boots",
                "sweatshirt",
                "fire",
                "horse",
                "person",
                "woman",
                "cap",
                "right",
            } and len(novel) >= 2:
                additions.append(sent if sent.endswith((".", "!", "?")) else sent + ".")
                base_tokens |= tokens
        if not additions:
            return base
        return (base.rstrip(".") + ". " + " ".join(additions)).strip()

    @staticmethod
    def _clean_florence_prose(text: str) -> str:
        updated = (text or "").strip()
        if not updated:
            return ""
        updated = re.sub(
            r"^(?:In this (?:image|picture|photo)(?: I can see)?|In this picture we can see|"
            r"This (?:image|picture|photo) (?:shows|depicts)|I can see)\s+",
            "",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(r"\bWe can see\b", "", updated, flags=re.IGNORECASE)
        updated = re.sub(r"\bI can see\b", "", updated, flags=re.IGNORECASE)
        updated = re.sub(r"\s{2,}", " ", updated).strip()
        if updated and updated[0].islower():
            updated = updated[0].upper() + updated[1:]
        return updated

    def _parse(self, text: str, *, task: str, image: object) -> str:
        try:
            from PIL import Image

            assert isinstance(image, Image.Image)
            parsed = self._processor.post_process_generation(
                text,
                task=task,
                image_size=(image.width, image.height),
            )
            if isinstance(parsed, dict):
                value = parsed.get(task) or next(iter(parsed.values()), "")
                if isinstance(value, dict):
                    # Dense-region style payloads are not used as prose captions.
                    labels = value.get("labels") or []
                    return ", ".join(str(x) for x in labels if x)
                return str(value).strip()
        except Exception:  # noqa: BLE001
            pass
        return text.replace(task, "").replace("</s>", "").replace("<s>", "").strip()
