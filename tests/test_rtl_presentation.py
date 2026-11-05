from app import _format_sources, _is_rtl


def test_arabic_answer_detection_is_based_on_rendered_text_not_question():
    assert _is_rtl("التقط صورة بانورامية باستخدام iPhone 4s (page 79).")
    assert not _is_rtl("Take a panorama on iPhone 4s (page 79).")


def test_sources_keep_deterministic_page_tokens_unchanged_for_arabic_display():
    sources = [{"page": 79, "section": "Camera", "chapter": "Photos"}]

    rendered = _format_sources(sources, rtl=True)

    assert "(page 79)" in rendered
    assert "\u2066" not in rendered
    assert "\u2069" not in rendered
    assert "**المصادر**" in rendered


def test_english_sources_remain_ltr_and_keep_the_same_page_token():
    rendered = _format_sources([{"page": 79}], rtl=False)

    assert rendered == "**Sources**\n- (page 79)"
