"""FOVEA public probe and final-reader prompts (upstream MIT)."""

from __future__ import annotations

import re


def probe_messages(question, original, parent, candidate, descriptor, system):
    def SystemMessage(content):
        return dict(role="system", content=content)

    def HumanMessage(content):
        return dict(role="user", content=content)

    get_base64_url_for_qwen_vl = descriptor

    def construct_input_for_confidence(
        question, original_image, image_to_crop, image_cropped, sys_inst
    ):
        labels, urls = ([], [])
        labels.append("original image (provided with the question)")
        urls.append(get_base64_url_for_qwen_vl(original_image))
        if image_to_crop is not None:
            labels.append("image in tool args (agent wants to crop this)")
            urls.append(get_base64_url_for_qwen_vl(image_to_crop))
        else:
            image_to_crop = original_image
        labels.append("cropped candidate region")
        urls.append(get_base64_url_for_qwen_vl(image_cropped))
        content = [{"type": "text", "text": f"**Original Question:**\n{question}\n\n"}]
        for label, url in zip(labels, urls, strict=True):
            content.append({"type": "text", "text": f"Here is the {label}:"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages = [SystemMessage(content=sys_inst), HumanMessage(content=content)]
        return messages

    return construct_input_for_confidence(question, original, parent, candidate, system)


def yes_score(contents):
    yes, total = (0, 0)
    for content in contents:
        if isinstance(content, str):
            total += 1
            match = re.search("<answer>(.*?)</answer>", content.strip(), re.DOTALL | re.IGNORECASE)
            if "yes" in (match.group(1) if match else content).strip().lower():
                yes += 1
    return yes / total if total else 0.0


def aggregator_messages(question, original, history, direct, system):
    """Render the upstream turn report directly from transport message dicts."""
    original_content = [dict(type="text", text=f"Original User Question:\n{question}")]
    if original:
        original_content.append(dict(type="image_url", image_url=dict(url=original)))
    messages = [
        dict(role="system", content=system),
        dict(role="user", content=original_content),
        dict(role="user", content="--- Turn-by-Turn Report from ReAct Agent (with tools) ---"),
    ]
    if not history:
        return messages
    i = next((i for i, m in enumerate(history) if m["role"] == "assistant"), len(history))
    while i < len(history):
        message = history[i]
        if message["role"] != "assistant":
            i += 1
            continue
        if not message.get("tool_calls") and message.get("content"):
            messages.append(
                dict(
                    role="user",
                    content=f"Finally, the ReAct agent's analysis is:\n\n{message['content']}",
                )
            )
        elif (
            message.get("tool_calls") and i + 1 < len(history) and history[i + 1]["role"] == "tool"
        ):
            observation = history[i + 1]
            text = f"AGENT THOUGHT:\n{message.get('content')}\n\nTOOL OBSERVATION:\n{observation.get('content')}"
            if i + 2 < len(history) and history[i + 2]["role"] == "user":
                content = [dict(type="text", text=text)]
                images = history[i + 2].get("content")
                content.extend(
                    images if isinstance(images, list) else [dict(type="text", text=images)]
                )
                messages.append(dict(role="user", content=content))
                i += 3
                continue
            messages.append(dict(role="user", content=text))
            i += 2
            continue
        i += 1
    messages.extend(
        [
            dict(role="user", content=f"Analysis from Direct Reasoning (no tools):\n\n{direct}"),
            dict(
                role="user",
                content="--- Final Task ---\nYou have now seen the original request, the full ReAct agent's report, and the Direct Reasoning report. Please synthesize all of this information and provide your final, definitive answer.",
            ),
        ]
    )
    return messages
