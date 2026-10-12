"""CPU unit tests for ``slime.utils.data._build_messages`` with ``--multimodal-keys``.

Media placeholders such as ``<image>`` are located by splitting each message's
text on a pattern built from the media present in that row. A text-only row of a
multimodal dataset carries no media, which left the pattern as ``()``: an empty
group that ``re.split`` matches between every character, so the prompt reached
the chat template as one text part per character.
"""

from __future__ import annotations

import pytest

from slime.utils.data import _build_messages

NUM_GPUS = 0

MULTIMODAL_KEYS = {"image": "images"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "row",
    [
        {"text": "What is 2 + 3?", "images": None},
        {"text": "What is 2 + 3?"},
    ],
    ids=["images-null", "images-absent"],
)
def test_text_only_row_keeps_its_text_whole(row):
    messages = _build_messages(row, "text", True, MULTIMODAL_KEYS)

    assert messages == [{"role": "user", "content": [{"type": "text", "text": "What is 2 + 3?"}]}]


@pytest.mark.unit
def test_text_only_conversation_keeps_each_message_whole():
    row = {
        "text": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Name a prime."},
        ]
    }

    messages = _build_messages(row, "text", True, MULTIMODAL_KEYS)

    assert messages == [
        {"role": "system", "content": [{"type": "text", "text": "Answer briefly."}]},
        {"role": "user", "content": [{"type": "text", "text": "Name a prime."}]},
    ]


@pytest.mark.unit
def test_row_with_image_still_splits_on_the_placeholder():
    row = {"text": "<image> What is shown?", "images": ["a.png"]}

    messages = _build_messages(row, "text", True, MULTIMODAL_KEYS)

    assert messages == [
        {
            "role": "user",
            "content": [{"type": "image", "image": "a.png"}, {"type": "text", "text": " What is shown?"}],
        }
    ]


@pytest.mark.unit
@pytest.mark.parametrize("type_name", ["images", "Image", "img"])
def test_unknown_media_type_is_rejected(type_name):
    """A media type name that is not image/video/audio used to be skipped: the
    row's media was dropped and the literal "<image>" stayed in the prompt."""
    row = {"text": "<image> What is shown?", "images": ["a.png"]}

    with pytest.raises(ValueError, match=type_name):
        _build_messages(row, "text", True, {type_name: "images"})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
