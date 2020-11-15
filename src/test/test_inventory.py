from src.data.dataset_utils import from_bn_mapping

if __name__ == "__main__":
    langs = ["en", "it", "es", "fr", "de", "zh"]
    multilingual_inventory = from_bn_mapping(langs, "bnoffsets")
    for lang in langs:
        lang_spec_inventory = from_bn_mapping([lang], "bnoffsets")
        assert lang_spec_inventory.get_inventory(lang) == multilingual_inventory.get_inventory(lang) or print("error", lang)
        for lexeme in lang_spec_inventory.get_inventory(lang).keys():
            m_synsets = multilingual_inventory.get(lexeme, lang)
            ls_synsets = lang_spec_inventory.get(lexeme, lang)
            assert m_synsets is not None
            assert ls_synsets is not None
            assert len(m_synsets) > 0
            assert len(ls_synsets) > 0
            assert m_synsets == ls_synsets
        print(lang, "OK")