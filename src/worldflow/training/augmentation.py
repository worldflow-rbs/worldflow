# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Instruction cleaning and meaning-preserving Chinese paraphrases for Demo5."""

import re


_RELATIONS = (
    ("上", "上面", "上方", "上边"), ("下", "下面", "下方", "下边"),
    ("左边", "左面", "左侧"), ("右边", "右面", "右侧"),
    ("前面", "前边", "前方"), ("后面", "后边", "后方"),
    ("中", "里", "里面", "内"),
)
_INSIDE = _RELATIONS[-1]
_RELATION_GROUP = {word: group for group in _RELATIONS for word in group}
_SUFFIXES = tuple(sorted(_RELATION_GROUP, key=len, reverse=True))
_SURFACE_VERBS = ("放在", "摆在", "放到", "移到", "挪到", "搁在")
_CONTAINER_VERBS = ("放进", "放入", "装进", "放到")
_VERBS = ("放在", "放进", "放到", "放入", "摆在", "摆到", "移到", "挪到", "搁在", "装进")
_INSTRUCTION = re.compile(
    r"^(?:请)?(?:能不能|能否|能|可以|麻烦)?(?:帮我|替我)?(?:把|将)?"
    r"(?P<object>.+?)(?P<verb>" + "|".join(_VERBS) + r")(?P<target>.+?)"
    r"(?P<relation>" + "|".join(_SUFFIXES) + r")?(?:吗|呢)?[？?]?$"
)
_COLORS = ("米白色", "黑色", "白色", "红色", "绿色", "蓝色", "黄色", "粉色",
           "紫色", "青色", "灰色", "棕色", "橙色", "银色", "金色")
_QUESTIONS = ("能把{}吗", "能帮我把{}吗", "可以把{}吗", "帮我把{}", "麻烦把{}")


def clean_instruction(text):
    text = re.sub(r"^\s*\d+\s*[.、，,]\s*", "", text.strip())
    text = re.sub(r"^[\]\[）)》>〕]+", "", text)
    text = re.sub(r"(?<=[\u3400-\u9fff])\s+|\s+(?=[\u3400-\u9fff])", "", text)
    return re.sub(r"\s+", " ", text)


def _name_variants(noun):
    for color in _COLORS:
        suffix = noun[len(color):]
        if noun.startswith(color) and suffix and not suffix.startswith("的"):
            return (noun, color + "的" + suffix)
    return (noun,)


def _phrase(body, rng):
    if rng.random() < 0.06:
        return rng.choice(_QUESTIONS).format(body)
    return rng.choice(("把", "将", "请把")) + body


def augment_demo5(text, rng):
    """Draw a paraphrase from the stage-one recipe, retaining 20% of base captions."""
    text = clean_instruction(text)
    if not text or rng.random() < 0.2:
        return text
    match = _INSTRUCTION.match(text)
    if match is None:
        return text
    obj, verb, target, relation = match.groups()
    if target.endswith(_SUFFIXES):
        # Nested spatial descriptions retain the original nouns and relation.
        return _phrase(obj + verb + target + (relation or ""), rng)
    obj = rng.choice(_name_variants(obj))
    target = rng.choice(_name_variants(target))
    group = _RELATION_GROUP.get(relation)
    if group == _INSIDE or (relation is None and verb in _CONTAINER_VERBS):
        if relation is None:
            new_verb, end = rng.choice(("放进", "放入", "放到")), ""
        else:
            new_verb = rng.choice(_CONTAINER_VERBS)
            choices = _INSIDE + (("",) if new_verb in ("放进", "放入", "装进") else ())
            end = rng.choice(choices)
        return _phrase(obj + new_verb + target + end, rng)
    if group is None:
        return _phrase(obj + verb + target + (relation or ""), rng)
    add_particle = rng.choice((False, True))
    endings = tuple(word for word in group if len(word) >= 2) if add_particle else group
    end = ("的" if add_particle else "") + rng.choice(endings)
    new_verb = rng.choice(_SURFACE_VERBS)
    return _phrase(obj + new_verb + target + end, rng)
