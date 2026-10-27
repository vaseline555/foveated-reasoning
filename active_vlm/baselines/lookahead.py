"""FOVEA lookahead policy, isolated request adapters (upstream MIT)."""

from __future__ import annotations

import re

from .geometry import generate_candidates, rel_coords_to_abs


def make_policy(context):
    logger = context["logger"]
    AIMessage = context["AIMessage"]
    SystemMessage = context["SystemMessage"]
    HumanMessage = context["HumanMessage"]
    get_base64_url_for_qwen_vl = context["get_base64_url_for_qwen_vl"]
    BATCH_REQUEST_COMPLETE = context["BATCH_REQUEST_COMPLETE"]
    BATCH_REQUEST_ERROR = context["BATCH_REQUEST_ERROR"]
    INVALID_CONFIDENCE_ASSESS = context["INVALID_CONFIDENCE_ASSESS"]
    NODE_EVALUATED = context["NODE_EVALUATED"]
    BINDED_LLM_CALL_FAILED = context["BINDED_LLM_CALL_FAILED"]
    BINDED_LLM_NO_TOOL_CALL = context["BINDED_LLM_NO_TOOL_CALL"]
    INVALID_EXPANSION = context["INVALID_EXPANSION"]
    INVALID_IMAGE_REF = context["INVALID_IMAGE_REF"]
    LOOK_AHEAD_COMPLETE = context["LOOK_AHEAD_COMPLETE"]

    class LookAheadPolicy:
        def __init__(self, scaling_factors, n_samples):
            self.scaling_factors = scaling_factors
            self.n_samples = n_samples

        async def get_action(
            self,
            chat_model,
            question,
            original_image,
            image_to_crop,
            proposed_bbox,
            inner_crop_sys_inst,
            resolvability_sys_inst,
        ):
            if image_to_crop is None:
                i2c = original_image
            else:
                i2c = image_to_crop
            model_binded = chat_model.bind_tools([crop])
            coords_candidates = generate_candidates(proposed_bbox, self.scaling_factors)
            values = []
            for coords_candidate in coords_candidates:
                image_cropped = i2c.crop(rel_coords_to_abs(coords_candidate, i2c.size))
                expansion = await look_ahead_expansion(
                    model_binded,
                    question,
                    original_image,
                    image_to_crop,
                    image_cropped,
                    inner_crop_sys_inst,
                )
                if expansion is None:
                    logger.warning(INVALID_EXPANSION, fallback=0)
                    values.append(0.0)
                else:
                    image_ref = expansion["image"].lower()
                    if "original" in image_ref:
                        sub_i2c = original_image
                    elif "image_1" in image_ref:
                        sub_i2c = image_to_crop or original_image
                    elif "image_2" in image_ref:
                        sub_i2c = image_cropped
                    else:
                        logger.warning(INVALID_IMAGE_REF, image=image_ref, fallback="original")
                        sub_i2c = original_image
                    sub_candidates = generate_candidates(
                        expansion["coordinates"], self.scaling_factors
                    )
                    confidences = []
                    for sub_coords in sub_candidates:
                        sub_image = sub_i2c.crop(rel_coords_to_abs(sub_coords, sub_i2c.size))
                        evaluation = await evaluate_candidate_confidence(
                            chat_model,
                            question,
                            original_image,
                            image_cropped,
                            sub_image,
                            resolvability_sys_inst,
                            self.n_samples,
                        )
                        confidences.append(evaluation["score"])
                        logger.info(NODE_EVALUATED, bbox=f"{sub_coords}", **evaluation)
                    values.append(sum(confidences) / len(confidences))
            _, best_bbox = max(
                zip(values, coords_candidates, strict=True), key=lambda item: item[0]
            )
            logger.info(LOOK_AHEAD_COMPLETE, best_bbox=f"{best_bbox}", values=f"{values}")
            return best_bbox

    async def look_ahead_expansion(
        binded_model, question, original_image, image_to_crop, image_cropped, sys_inst
    ):
        prompt = construct_input_for_binded_model(
            question, original_image, image_to_crop, image_cropped, sys_inst
        )
        try:
            response = await binded_model.ainvoke(prompt)
        except Exception:
            logger.exception(BINDED_LLM_CALL_FAILED)
            return None
        assert isinstance(response, AIMessage)
        if not response.tool_calls:
            logger.warning(BINDED_LLM_NO_TOOL_CALL)
            return None
        tool_args = response.tool_calls[0]["args"]
        return {"image": tool_args["image"], "coordinates": tuple(tool_args["coordinates"])}

    def construct_input_for_confidence(
        question, original_image, image_to_crop, image_cropped, sys_inst
    ):
        labels, urls = ([], [])
        labels.append("original image (provided with the question)")
        urls.append(get_base64_url_for_qwen_vl(original_image))
        labels.append("image in tool args (agent wants to crop this)")
        urls.append(get_base64_url_for_qwen_vl(image_to_crop))
        labels.append("cropped candidate region")
        urls.append(get_base64_url_for_qwen_vl(image_cropped))
        content = [{"type": "text", "text": f"**Original Question:**\n{question}\n\n"}]
        for label, url in zip(labels, urls, strict=True):
            content.append({"type": "text", "text": f"Here is the {label}:"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages = [SystemMessage(content=sys_inst), HumanMessage(content=content)]
        return messages

    async def evaluate_candidate_confidence(
        chat_model, question, original_image, image_cropped, sub_image, sys_inst, n_samples
    ):
        prompt = construct_input_for_confidence(
            question, original_image, image_cropped, sub_image, sys_inst
        )
        batch_inputs = [prompt] * n_samples
        try:
            responses = await chat_model.abatch(batch_inputs)
        except Exception:
            logger.exception(BATCH_REQUEST_ERROR)
            return {"score": 0.0, "votes_yes": 0, "total_samples": 0}
        logger.info(BATCH_REQUEST_COMPLETE)
        votes_yes = 0
        total = 0
        for response in responses:
            if not isinstance(response.content, str):
                logger.warning(INVALID_CONFIDENCE_ASSESS, fallback="no")
            else:
                total += 1
                content_str = response.content.strip()
                match = re.search("<answer>(.*?)</answer>", content_str, re.DOTALL | re.IGNORECASE)
                if match:
                    answer = match.group(1).strip().lower()
                    if "yes" in answer:
                        votes_yes += 1
                elif "yes" in content_str.lower():
                    votes_yes += 1
        score = votes_yes / total
        return {"score": score, "votes_yes": votes_yes, "total_samples": total}

    def construct_input_for_binded_model(
        question, original_image, image_to_crop, candidate_image, sys_inst
    ):
        labels, urls = ([], [])
        labels.append(
            "original image (provided with the question, you can refer to it as `original`)"
        )
        urls.append(get_base64_url_for_qwen_vl(original_image))
        cnt = 1
        if image_to_crop is not None:
            labels.append(f"history image {cnt} (you can refer to it as `image_{cnt}`)")
            urls.append(get_base64_url_for_qwen_vl(image_to_crop))
            cnt += 1
        else:
            image_to_crop = original_image
        labels.append(f"history image {cnt} (you can refer to it as `image_{cnt}`)")
        urls.append(get_base64_url_for_qwen_vl(candidate_image))
        content = [{"type": "text", "text": f"**Original Question:**\n{question}\n\n"}]
        for label, url in zip(labels, urls, strict=True):
            content.append({"type": "text", "text": f"Here is the {label}:"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages = [SystemMessage(content=sys_inst), HumanMessage(content=content)]
        return messages

    def crop(image, coordinates):
        """Crop the given image based on the provided coordinates (x1, y1, x2, y2)."""
        del image, coordinates
        return {"result": "A placeholder"}

    return LookAheadPolicy
