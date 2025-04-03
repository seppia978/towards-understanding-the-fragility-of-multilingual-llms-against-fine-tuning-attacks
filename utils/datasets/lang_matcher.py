def aya_redteaming_matcher(lang):
    md = {
        'en': 'english',
        'ar': 'arabic',
        'hi': 'hindi',
        'fr': 'french',
        'es': 'spanish',
        'ru': 'russian',
        'ta': 'filipino'   
    }

    if lang not in md.keys():
        raise KeyError(f'{lang} not a key. Keys are: {md.keys()}')

    return md[lang]