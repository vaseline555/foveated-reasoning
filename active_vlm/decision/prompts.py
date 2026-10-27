"""Compact question-conditioned experiments; Python owns coordinates and IDs."""

from ..baselines.prompts import array, obj

PLAN_SYSTEM = """Build a compact visual experiment DAG from the question and choices.
Every step.test is one concrete YES/NO visual measurement, preserving the exact target's
identifying clothing/ownership/order/reference frame. Dependencies are earlier zero-based
indices. A prerequisite such as locating the specified person may have equal likelihoods
for all answers: its downstream attribute still justifies it. Include at least one test
whose yes/no distinguishes the answer choices. Minimize redundant tasks, usually2-6 steps.
box is an ORIGINAL-image x1,y1,x2,y2 search proposal in0..1000, never a fact.
positive_if_answer[j] estimates P(positive measurement | choice j is true), in supplied
choice order, not confidence that choice j is correct. visibility estimates the chance
that this crop can actually resolve the measurement. These are fallible model estimates.
Do not make a crop complete just because it mentions a target. On revision use the saved
observations and failures; fix the wrong referent, region or dependency instead of repeating
an already tested crop. Do not invent visual facts. Return only the required tool call.
For an embedded question, first read its actual question and options into reading.
For ordinary text questions set reading=null and preserve the supplied question/choices."""

OBSERVE_SYSTEM = """Evaluate the fixed binary visual measurement in this CURRENT CROP.
Python has already bound the task and crop; do not generate IDs or coordinates.
The original image is context, not evidence that an object is visible in the crop.
The native FOVEA probe is a fallible relevance report, not a measurement or ground truth.
source=crop only for a concrete visible observation of the question's EXACT referent.
Describe that referent's actually visible identifying properties in target_description
and the measured fact in observation. Resolve clothing/ownership, ordinals and reference
frames before attributing a property to a nearby person/object. Read numbers with units.
Return positive/negative ONLY if the binary measurement is directly decidable in this
crop. If the target, identifying condition or required context is missing, use unobserved;
a relevance No, a question mention or 'can identify' statement does not prove absence.
A negative whole-scene claim needs visible coverage of that whole scene. Do not substitute
an option name for an observation. If the proposed target/task/region needs changing,
set needs_repair=true and give a concrete reason. confidence is measurement reliability,
not answer confidence. No long reasoning; only the required tool call."""


def tool(name, properties):
    return dict(
        type="function",
        function=dict(
            name=name,
            description="Return the requested bounded visual experiment data.",
            parameters=obj(properties),
        ),
    )


def plan_tool(embedded=True):
    step = obj(
        dict(
            test=dict(type="string", minLength=5, maxLength=1600),
            requires=array(dict(type="integer", minimum=0, maximum=10), 0, 11),
            box=array(dict(type="number", minimum=0, maximum=1000), 4, 4),
            positive_if_answer=array(dict(type="number", minimum=0, maximum=1), 2, 11),
            visibility=dict(type="number", minimum=0, maximum=1),
        )
    )
    reading = dict(
        anyOf=[
            dict(type="null"),
            obj(
                dict(
                    question=dict(type="string", minLength=5, maxLength=2400),
                    options=array(dict(type="string", minLength=1, maxLength=1200), 2, 11),
                )
            ),
        ]
    )
    return tool(
        "design_experiments",
        dict(reading=reading if embedded else dict(type="null"), steps=array(step, 1, 12)),
    )


def observation_tool():
    return tool(
        "observe_experiment",
        dict(
            outcome=dict(type="string", enum=["positive", "negative", "unobserved"]),
            confidence=dict(type="number", minimum=0, maximum=1),
            source=dict(type="string", enum=["crop", "original", "question", "inference"]),
            observation=dict(type="string", maxLength=1600),
            target_description=dict(type="string", maxLength=1600),
            needs_repair=dict(type="boolean"),
            reason=dict(type="string", maxLength=800),
        ),
    )
