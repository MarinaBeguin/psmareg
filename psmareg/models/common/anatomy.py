"""Anatomical label groups shared by pretraining and fine-tuning."""


def macro_ct_label_groups(preset: str) -> list[tuple[str, tuple[int, ...]]]:
    """Return stable TotalSegmentator-style CT groups for supervision.

    Args:
        preset: Name of the anatomical grouping preset.

    Returns:
        Ordered pairs of group names and CT class identifiers.

    Raises:
        ValueError: If the preset is unknown.
    """
    skeleton = tuple(
        list(range(32, 57)) + list(range(69, 79)) + [92] + list(range(93, 118)))
    groups = {
        "body_foreground": tuple(range(1, 118)),
        "skeleton": skeleton,
        "lungs": tuple(range(9, 14)),
        "liver": (5,),
        "kidneys": (2, 3),
        "spleen": (1,),
        "heart_vessels": tuple(range(57, 69)),
        "muscles": tuple(range(80, 91)),
        "bladder": (29,),
    }
    names: tuple[str, ...]
    if preset == "skeleton_only":
        names = ("body_foreground", "skeleton")
    elif preset == "organs_only":
        names = ("body_foreground", "lungs", "liver", "kidneys", "spleen",
                 "heart_vessels")
    elif preset == "stable":
        names = ("body_foreground", "skeleton", "lungs", "liver", "kidneys",
                 "spleen", "heart_vessels", "muscles", "bladder")
    elif preset == "stable_no_bladder":
        names = ("body_foreground", "skeleton", "lungs", "liver", "kidneys",
                 "spleen", "heart_vessels", "muscles")
    else:
        raise ValueError(f"Unsupported macro CT label preset: {preset}")
    return [(name, groups[name]) for name in names]
