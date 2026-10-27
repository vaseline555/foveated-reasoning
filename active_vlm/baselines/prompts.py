"""Original FOVEA templates and crop tools."""

from pathlib import Path

TEMPLATES = Path(__file__).parent / "templates"


def obj(p):
    return dict(type="object", properties=p, required=list(p), additionalProperties=False)


def array(item, minimum=0, maximum=12):
    return dict(type="array", items=item, minItems=minimum, maxItems=maximum)


def enum(values):
    values = list(values)
    kind = "integer" if type(values[0]) is int else "string"
    return dict(type=kind, enum=values)


TEXT = dict(type="string")

STRING = dict(type="string", minLength=1)

BOOL = dict(type="boolean")

BOX = dict(type="array", items=dict(type="number", minimum=0, maximum=1000), minItems=4, maxItems=4)

CROP_TOOL = dict(
    type="function",
    function=dict(
        name="cropping",
        description="Crop an available image using normalized 0..1000 coordinates to zoom into a region.",
        parameters=obj(dict(image_url=STRING, coordinates=BOX)),
    ),
)

INNER_CROP_TOOL = dict(
    type="function",
    function=dict(
        name="crop",
        description="Crop the given image based on coordinates (x1,y1,x2,y2).",
        parameters=obj(dict(image=STRING, coordinates=BOX)),
    ),
)


def template(name):
    return Path(TEMPLATES / name).read_text().strip()
