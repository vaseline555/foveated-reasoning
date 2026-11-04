"""Offline scientific-contract tests for the official-function adapter."""

import base64
import hashlib
import json
import unittest
from io import BytesIO

from PIL import Image

import official_adapter as adapter


def image_from_uri(uri):
    return Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1])))


class OfficialAdapterTests(unittest.TestCase):
    def test_geometry_and_coordinate_rounding(self):
        self.assertEqual(adapter.generate_candidates([100, 200, 300, 400]),
                         [(50, 150, 350, 450), (100, 200, 300, 400), (120, 220, 280, 380)])
        self.assertEqual(adapter.generate_candidates([0, 0, 1000, 1000])[0],
                         (0, 0, 1000, 1000))
        self.assertEqual(adapter.absolute_coords([125, 125, 875, 875], (13, 13)),
                         [1, 1, 11, 11])
        self.assertEqual(adapter.absolute_coords([125, 125, 875, 875], (13, 13), probe=True),
                         [2, 2, 11, 11])
        self.assertEqual(adapter.validate_crop_coordinates([0, 0, 50, 50]), (0, 0, 50, 50))
        with self.assertRaises(ValueError):
            adapter.validate_crop_coordinates([0, 0, 49, 100])

    def test_original_pixels_preserved_but_probe_uses_qwen_resize(self):
        image = Image.new("RGBA", (53, 37), (0, 0, 0, 0))
        original = image_from_uri(adapter.jpeg_url(image))
        probe = image_from_uri(adapter.jpeg_url(image, probe=True))
        self.assertEqual(original.size, (53, 37))
        self.assertEqual(probe.size, (96, 64))
        self.assertEqual(original.mode, "RGB")
        self.assertGreater(min(original.getpixel((0, 0))), 250)

    def test_probe_has_original_parent_and_candidate_without_history(self):
        image = Image.new("RGB", (64, 64))
        messages = adapter.probe_messages("Which color?", image, None, image)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        blocks = messages[1]["content"]
        self.assertEqual(sum(b["type"] == "image_url" for b in blocks), 2)
        nested = adapter.probe_messages("Which color?", image, image, image)
        self.assertEqual(sum(b["type"] == "image_url" for b in nested[1]["content"]), 3)
        self.assertEqual(messages[0]["content"], adapter.template("resolvability"))

    def test_aggregator_preserves_tool_evidence_final_answer_and_direct(self):
        image = "data:image/jpeg;base64,abc"
        history = [
            {"role": "system", "content": "planner"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "inspect", "tool_calls": [{"id": "c"}]},
            {"role": "tool", "content": "crop succeeded", "tool_call_id": "c"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]},
            {"role": "assistant", "content": "<answer>B</answer>"},
        ]
        messages = adapter.aggregator_messages("Q?", image, history, "<answer>A</answer>")
        self.assertEqual(len(messages), 7)
        self.assertEqual(messages[3]["content"][0]["text"],
                         "AGENT THOUGHT:\ninspect\n\nTOOL OBSERVATION:\ncrop succeeded")
        self.assertEqual(messages[3]["content"][1]["image_url"]["url"], image)
        self.assertIn("<answer>B</answer>", messages[4]["content"])
        self.assertIn("<answer>A</answer>", messages[5]["content"])
        self.assertIn("Final Task", messages[6]["content"])
        self.assertEqual(history[0]["content"], "planner")

    def test_official_yes_parser_preserves_missing_tag_fallback(self):
        self.assertEqual(adapter.yes_score(["<answer>yes</answer>", "<answer>no</answer>",
                                            "Yes, sufficient", None]), 2 / 3)
        self.assertEqual(adapter.yes_score([None]), 0)
        # Deliberately retain the official substring behavior, not a stricter parser.
        self.assertEqual(adapter.yes_score(["yesterday"]), 1)

    def test_schema_templates_output_and_provenance(self):
        tool = adapter.crop_tool()["function"]
        self.assertEqual(tool["name"], "cropping")
        self.assertEqual(tool["parameters"]["required"], ["image_url", "coordinates"])
        self.assertEqual(tool["parameters"]["properties"]["x1"]["default"], 0)
        self.assertIn("<answer>", adapter.template("direct", "Which side?"))
        self.assertNotIn("{{ question }}", adapter.template("direct", "Which side?"))
        output = json.loads(adapter.process_output(dict(status="success", output_image="private",
                                                       boed_selected_coordinates=[1, 2, 3, 4])))
        self.assertNotIn("output_image", output)
        self.assertEqual(output["boed_selected_coordinates"], [1, 2, 3, 4])
        provenance = adapter.provenance()
        self.assertEqual(len(provenance["official_commit"]), 40)
        for relative, sha in provenance["source_hashes"].items():
            self.assertEqual(hashlib.sha256((adapter.OFFICIAL_ROOT / relative).read_bytes()).hexdigest(), sha)


if __name__ == "__main__":
    unittest.main()
