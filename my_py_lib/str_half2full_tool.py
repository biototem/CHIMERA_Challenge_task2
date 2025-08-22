'''
字符串-全角半角转换工具

全角半角对照表，wiki
https://zh.wikipedia.org/wiki/%E5%85%A8%E5%BD%A2%E5%92%8C%E5%8D%8A%E5%BD%A2
'''

_jp_half2full_dict = {
    '｡': '。',
    '｢': '「',
    '｣': '」',
    '､': '、',
    '･': '・',
    'ｦ': 'ヲ',
    'ｧ': 'ァ',
    'ｨ': 'ィ',
    'ｩ': 'ゥ',
    'ｪ': 'ェ',
    'ｫ': 'ォ',
    'ｬ': 'ャ',
    'ｭ': 'ュ',
    'ｮ': 'ョ',
    'ｯ': 'ッ',
    'ｰ': 'ー',
    'ｱ': 'ア',
    'ｲ': 'イ',
    'ｳ': 'ウ',
    'ｴ': 'エ',
    'ｵ': 'オ',
    'ｶ': 'カ',
    'ｷ': 'キ',
    'ｸ': 'ク',
    'ｹ': 'ケ',
    'ｺ': 'コ',
    'ｻ': 'サ',
    'ｼ': 'シ',
    'ｽ': 'ス',
    'ｾ': 'セ',
    'ｿ': 'ソ',
    'ﾀ': 'タ',
    'ﾁ': 'チ',
    'ﾂ': 'ツ',
    'ﾃ': 'テ',
    'ﾄ': 'ト',
    'ﾅ': 'ナ',
    'ﾆ': 'ニ',
    'ﾇ': 'ヌ',
    'ﾈ': 'ネ',
    'ﾉ': 'ノ',
    'ﾊ': 'ハ',
    'ﾋ': 'ヒ',
    'ﾌ': 'フ',
    'ﾍ': 'ヘ',
    'ﾎ': 'ホ',
    'ﾏ': 'マ',
    'ﾐ': 'ミ',
    'ﾑ': 'ム',
    'ﾒ': 'メ',
    'ﾓ': 'モ',
    'ﾔ': 'ヤ',
    'ﾕ': 'ユ',
    'ﾖ': 'ヨ',
    'ﾗ': 'ラ',
    'ﾘ': 'リ',
    'ﾙ': 'ル',
    'ﾚ': 'レ',
    'ﾛ': 'ロ',
    'ﾜ': 'ワ',
    'ﾝ': 'ン',
    'ﾞ': '゛',
    'ﾟ': '゜',
}

_jp_full2half_dict = dict(zip(_jp_half2full_dict.values(), _jp_half2full_dict.keys()))


_kr_half2full_dict = {
    'ﾠ': 'ㅤ',
    'ﾡ': 'ㄱ',
    'ﾢ': 'ㄲ',
    'ﾣ': 'ㄳ',
    'ﾤ': 'ㄴ',
    'ﾥ': 'ㄵ',
    'ﾦ': 'ㄶ',
    'ﾧ': 'ㄷ',
    'ﾨ': 'ㄸ',
    'ﾩ': 'ㄹ',
    'ﾪ': 'ㄺ',
    'ﾫ': 'ㄻ',
    'ﾬ': 'ㄼ',
    'ﾭ': 'ㄽ',
    'ﾮ': 'ㄾ',
    'ﾯ': 'ㄿ',
    'ﾰ': 'ㅀ',
    'ﾱ': 'ㅁ',
    'ﾲ': 'ㅂ',
    'ﾳ': 'ㅃ',
    'ﾴ': 'ㅄ',
    'ﾵ': 'ㅅ',
    'ﾶ': 'ㅆ',
    'ﾷ': 'ㅇ',
    'ﾸ': 'ㅈ',
    'ﾹ': 'ㅉ',
    'ﾺ': 'ㅊ',
    'ﾻ': 'ㅋ',
    'ﾼ': 'ㅌ',
    'ﾽ': 'ㅍ',
    'ﾾ': 'ㅎ',
    'ￂ': 'ㅏ',
    'ￃ': 'ㅐ',
    'ￄ': 'ㅑ',
    'ￅ': 'ㅒ',
    'ￆ': 'ㅓ',
    'ￇ': 'ㅔ',
    'ￊ': 'ㅕ',
    'ￋ': 'ㅖ',
    'ￌ': 'ㅗ',
    'ￍ': 'ㅘ',
    'ￎ': 'ㅙ',
    'ￏ': 'ㅚ',
    'ￒ': 'ㅛ',
    'ￓ': 'ㅜ',
    'ￔ': 'ㅝ',
    'ￕ': 'ㅞ',
    'ￖ': 'ㅟ',
    'ￗ': 'ㅠ',
    'ￚ': 'ㅡ',
    'ￛ': 'ㅢ',
    'ￜ': 'ㅣ',
}

_kr_full2half_dict = dict(zip(_kr_half2full_dict.values(), _kr_half2full_dict.keys()))


_other_half2full_dict = {
    '⦅': '｟',
    '⦆': '｠',
    '¢': '￠',
    '£': '￡',
    '¬': '￢',
    '¯': '￣',
    '¦': '￤',
    '¥': '￥',
    '₩': '￦',
    '￨': '│',
    '￩': '←',
    '￪': '↑',
    '￫': '→',
    '￬': '↓',
    '￭': '■',
    '￮': '○',
}

_other_full2half_dict = dict(zip(_other_half2full_dict.values(), _other_half2full_dict.keys()))


def str_full2half(s: str, ignore_chars=None, ignore_ascii=False, ignore_jp=False, ignore_kr=False, ignore_other=False):
    ignore_chars = set() if ignore_chars is None else set(ignore_chars)

    ns = []
    for c in s:
        nc = None

        if c in ignore_chars:
            nc = c

        if nc is None and not ignore_ascii:
            if '\uff01' <= c <= '\uff5e':
                nc = chr(ord(c) - 0xfee0)
            elif c == '\u3000':
                nc = '\u0020'

        if nc is None and not ignore_jp:
            nc = _jp_full2half_dict.get(c, None)

        if nc is None and not ignore_kr:
            nc = _kr_full2half_dict.get(c, None)

        if nc is None and not ignore_other:
            nc = _other_full2half_dict.get(c, None)

        if nc is None:
            nc = c
        ns.append(nc)

    ns = ''.join(ns)
    return ns


def str_half2full(s: str, ignore_chars=None, ignore_ascii=False, ignore_jp=False, ignore_kr=False, ignore_other=False):
    ignore_chars = set() if ignore_chars is None else set(ignore_chars)

    ns = []
    for c in s:
        nc = None

        if c in ignore_chars:
            nc = c

        if nc is None and not ignore_ascii:
            if '\u0021' <= c <= '\u007e':
                nc = chr(ord(c) + 0xfee0)
            elif c == '\u0020':
                nc = '\u3000'

        if nc is None and not ignore_jp:
            nc = _jp_half2full_dict.get(c, None)

        if nc is None and not ignore_kr:
            nc = _kr_half2full_dict.get(c, None)

        if nc is None and not ignore_other:
            nc = _other_half2full_dict.get(c, None)

        if nc is None:
            nc = c
        ns.append(nc)

    ns = ''.join(ns)
    return ns
