from unittest import TestCase

from voicevox_engine.model import AccentPhrase, Mora
from voicevox_engine.tts_pipeline.text_analyzer import (
    AccentPhraseLabel,
    BreathGroupLabel,
    Label,
    MoraLabel,
    UtteranceLabel,
    mora_to_text,
    text_to_accent_phrases,
)


def contexts_to_feature(contexts: dict[str, str]) -> str:
    """ラベルの contexts を feature へ変換する"""
    return (
        "{p1}^{p2}-{p3}+{p4}={p5}"
        "/A:{a1}+{a2}+{a3}"
        "/B:{b1}-{b2}_{b3}"
        "/C:{c1}_{c2}+{c3}"
        "/D:{d1}+{d2}_{d3}"
        "/E:{e1}_{e2}!{e3}_{e4}-{e5}"
        "/F:{f1}_{f2}#{f3}_{f4}@{f5}_{f6}|{f7}_{f8}"
        "/G:{g1}_{g2}%{g3}_{g4}_{g5}"
        "/H:{h1}_{h2}"
        "/I:{i1}-{i2}@{i3}+{i4}&{i5}-{i6}|{i7}+{i8}"
        "/J:{j1}_{j2}"
        "/K:{k1}+{k2}-{k3}"
    ).format(**contexts)


# OpenJTalk コンテナクラス
OjtContainer = MoraLabel | AccentPhraseLabel | BreathGroupLabel | UtteranceLabel


def features(ojt_container: OjtContainer) -> list[str]:
    """コンテナインスタンスに直接的・間接的に含まれる全ての feature を返す"""
    return [contexts_to_feature(p.contexts) for p in ojt_container.labels]


class TestBaseLabels(TestCase):
    def setUp(self):
        super().setUp()
        # pyopenjtalk.extract_fullcontext("こんにちは、ヒホです。")の結果
        # 出来る限りテスト内で他のライブラリに依存しないため、
        # またテスト内容を透明化するために、テストケースを生成している
        self.test_case_hello_hiho = [
            # sil (無音)
            "xx^xx-sil+k=o/A:xx+xx+xx/B:xx-xx_xx/C:xx_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
            + "/F:xx_xx#xx_xx@xx_xx|xx_xx/G:5_5%0_xx_xx/H:xx_xx/I:xx-xx"
            + "@xx+xx&xx-xx|xx+xx/J:1_5/K:2+2-9",
            # k
            "xx^sil-k+o=N/A:-4+1+5/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
            + "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
            + "@1+2&1-2|1+9/J:1_4/K:2+2-9",
            # o
            "sil^k-o+N=n/A:-4+1+5/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
            + "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
            + "@1+2&1-2|1+9/J:1_4/K:2+2-9",
            # N (ん)
            "k^o-N+n=i/A:-3+2+4/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
            + "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
            + "@1+2&1-2|1+9/J:1_4/K:2+2-9",
            # n
            "o^N-n+i=ch/A:-2+3+3/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
            + "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
            + "@1+2&1-2|1+9/J:1_4/K:2+2-9",
            # i
            "N^n-i+ch=i/A:-2+3+3/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
            + "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
            + "@1+2&1-2|1+9/J:1_4/K:2+2-9",
            # ch
            "n^i-ch+i=w/A:-1+4+2/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
            + "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
            + "@1+2&1-2|1+9/J:1_4/K:2+2-9",
            # i
            "i^ch-i+w=a/A:-1+4+2/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
            + "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
            + "@1+2&1-2|1+9/J:1_4/K:2+2-9",
            # w
            "ch^i-w+a=pau/A:0+5+1/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
            + "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
            + "@1+2&1-2|1+9/J:1_4/K:2+2-9",
            # a
            "i^w-a+pau=h/A:0+5+1/B:xx-xx_xx/C:09_xx+xx/D:09+xx_xx/E:xx_xx!xx_xx-xx"
            + "/F:5_5#0_xx@1_1|1_5/G:4_1%0_xx_0/H:xx_xx/I:1-5"
            + "@1+2&1-2|1+9/J:1_4/K:2+2-9",
            # pau (読点)
            "w^a-pau+h=i/A:xx+xx+xx/B:09-xx_xx/C:xx_xx+xx/D:09+xx_xx/E:5_5!0_xx-xx"
            + "/F:xx_xx#xx_xx@xx_xx|xx_xx/G:4_1%0_xx_xx/H:1_5/I:xx-xx"
            + "@xx+xx&xx-xx|xx+xx/J:1_4/K:2+2-9",
            # h
            "a^pau-h+i=h/A:0+1+4/B:09-xx_xx/C:09_xx+xx/D:22+xx_xx/E:5_5!0_xx-0"
            + "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
            + "@2+1&2-1|6+4/J:xx_xx/K:2+2-9",
            # i
            "pau^h-i+h=o/A:0+1+4/B:09-xx_xx/C:09_xx+xx/D:22+xx_xx/E:5_5!0_xx-0"
            + "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
            + "@2+1&2-1|6+4/J:xx_xx/K:2+2-9",
            # h
            "h^i-h+o=d/A:1+2+3/B:09-xx_xx/C:22_xx+xx/D:10+7_2/E:5_5!0_xx-0"
            + "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
            + "@2+1&2-1|6+4/J:xx_xx/K:2+2-9",
            # o
            "i^h-o+d=e/A:1+2+3/B:09-xx_xx/C:22_xx+xx/D:10+7_2/E:5_5!0_xx-0"
            + "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
            + "@2+1&2-1|6+4/J:xx_xx/K:2+2-9",
            # d
            "h^o-d+e=s/A:2+3+2/B:22-xx_xx/C:10_7+2/D:xx+xx_xx/E:5_5!0_xx-0"
            + "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
            + "@2+1&2-1|6+4/J:xx_xx/K:2+2-9",
            # e
            "o^d-e+s=U/A:2+3+2/B:22-xx_xx/C:10_7+2/D:xx+xx_xx/E:5_5!0_xx-0"
            + "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
            + "@2+1&2-1|6+4/J:xx_xx/K:2+2-9",
            # s
            "d^e-s+U=sil/A:3+4+1/B:22-xx_xx/C:10_7+2/D:xx+xx_xx/E:5_5!0_xx-0"
            + "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
            + "@2+1&2-1|6+4/J:xx_xx/K:2+2-9",
            # U (無声母音)
            "e^s-U+sil=xx/A:3+4+1/B:22-xx_xx/C:10_7+2/D:xx+xx_xx/E:5_5!0_xx-0"
            + "/F:4_1#0_xx@1_1|1_4/G:xx_xx%xx_xx_xx/H:1_5/I:1-4"
            + "@2+1&2-1|6+4/J:xx_xx/K:2+2-9",
            # sil (無音)
            "s^U-sil+xx=xx/A:xx+xx+xx/B:10-7_2/C:xx_xx+xx/D:xx+xx_xx/E:4_1!0_xx-xx"
            + "/F:xx_xx#xx_xx@xx_xx|xx_xx/G:xx_xx%xx_xx_xx/H:1_4/I:xx-xx"
            + "@xx+xx&xx-xx|xx+xx/J:xx_xx/K:2+2-9",
        ]
        self.labels_hello_hiho = [
            Label.from_feature(feature) for feature in self.test_case_hello_hiho
        ]


def jointed_phonemes(ojt_container: OjtContainer) -> str:
    """コンテナインスタンスに直接的・間接的に含まれる全ラベルの音素文字を結合してを返す"""
    return "".join([label.phoneme for label in ojt_container.labels])


def space_jointed_phonemes(ojt_container: OjtContainer) -> str:
    """コンテナインスタンスに直接的・間接的に含まれる全ラベルの音素文字を ` ` 挟みながら結合してを返す"""
    return " ".join([label.phoneme for label in ojt_container.labels])


class TestLabel(TestBaseLabels):
    def test_phoneme(self):
        """Label に含まれる音素をテスト"""
        self.assertEqual(
            " ".join([label.phoneme for label in self.labels_hello_hiho]),
            "sil k o N n i ch i w a pau h i h o d e s U sil",
        )

    def test_is_pause(self):
        """Label のポーズ判定をテスト"""
        self.assertEqual(
            [label.is_pause() for label in self.labels_hello_hiho],
            [
                True,  # sil
                False,  # k
                False,  # o
                False,  # N
                False,  # n
                False,  # i
                False,  # ch
                False,  # i
                False,  # w
                False,  # a
                True,  # pau
                False,  # h
                False,  # i
                False,  # h
                False,  # o
                False,  # d
                False,  # e
                False,  # s
                False,  # u
                True,  # sil
            ],
        )

    def test_feature(self) -> None:
        """Label に含まれる features をテスト"""
        self.assertEqual(
            [contexts_to_feature(label.contexts) for label in self.labels_hello_hiho],
            self.test_case_hello_hiho,
        )


class TestMoraLabel(TestBaseLabels):
    def setUp(self) -> None:
        super().setUp()
        # contexts["a2"] == "1" ko
        self.mora_hello_1 = MoraLabel(
            consonant=self.labels_hello_hiho[1], vowel=self.labels_hello_hiho[2]
        )
        # contexts["a2"] == "2" N
        self.mora_hello_2 = MoraLabel(consonant=None, vowel=self.labels_hello_hiho[3])
        # contexts["a2"] == "3" ni
        self.mora_hello_3 = MoraLabel(
            consonant=self.labels_hello_hiho[4], vowel=self.labels_hello_hiho[5]
        )
        # contexts["a2"] == "4" chi
        self.mora_hello_4 = MoraLabel(
            consonant=self.labels_hello_hiho[6], vowel=self.labels_hello_hiho[7]
        )
        # contexts["a2"] == "5" wa
        self.mora_hello_5 = MoraLabel(
            consonant=self.labels_hello_hiho[8], vowel=self.labels_hello_hiho[9]
        )
        # contexts["a2"] == "1" hi
        self.mora_hiho_1 = MoraLabel(
            consonant=self.labels_hello_hiho[11], vowel=self.labels_hello_hiho[12]
        )
        # contexts["a2"] == "2" ho
        self.mora_hiho_2 = MoraLabel(
            consonant=self.labels_hello_hiho[13], vowel=self.labels_hello_hiho[14]
        )
        # contexts["a2"] == "3" de
        self.mora_hiho_3 = MoraLabel(
            consonant=self.labels_hello_hiho[15], vowel=self.labels_hello_hiho[16]
        )
        # contexts["a2"] == "1" sU
        self.mora_hiho_4 = MoraLabel(
            consonant=self.labels_hello_hiho[17], vowel=self.labels_hello_hiho[18]
        )

    def test_phonemes(self) -> None:
        """MoraLabel に含まれる音素系列をテスト"""
        self.assertEqual(jointed_phonemes(self.mora_hello_1), "ko")
        self.assertEqual(jointed_phonemes(self.mora_hello_2), "N")
        self.assertEqual(jointed_phonemes(self.mora_hello_3), "ni")
        self.assertEqual(jointed_phonemes(self.mora_hello_4), "chi")
        self.assertEqual(jointed_phonemes(self.mora_hello_5), "wa")
        self.assertEqual(jointed_phonemes(self.mora_hiho_1), "hi")
        self.assertEqual(jointed_phonemes(self.mora_hiho_2), "ho")
        self.assertEqual(jointed_phonemes(self.mora_hiho_3), "de")
        self.assertEqual(jointed_phonemes(self.mora_hiho_4), "sU")

    def test_features(self) -> None:
        """MoraLabel に含まれる features をテスト"""
        expects = self.test_case_hello_hiho
        self.assertEqual(features(self.mora_hello_1), expects[1:3])
        self.assertEqual(features(self.mora_hello_2), expects[3:4])
        self.assertEqual(features(self.mora_hello_3), expects[4:6])
        self.assertEqual(features(self.mora_hello_4), expects[6:8])
        self.assertEqual(features(self.mora_hello_5), expects[8:10])
        self.assertEqual(features(self.mora_hiho_1), expects[11:13])
        self.assertEqual(features(self.mora_hiho_2), expects[13:15])
        self.assertEqual(features(self.mora_hiho_3), expects[15:17])
        self.assertEqual(features(self.mora_hiho_4), expects[17:19])


class TestAccentPhraseLabel(TestBaseLabels):
    def setUp(self) -> None:
        super().setUp()
        # TODO: ValueErrorを吐く作為的ではない自然な例の模索
        # 存在しないなら放置でよい
        self.accent_phrase_hello = AccentPhraseLabel.from_labels(
            self.labels_hello_hiho[1:10]
        )
        self.accent_phrase_hiho = AccentPhraseLabel.from_labels(
            self.labels_hello_hiho[11:19]
        )

    def test_accent(self):
        """AccentPhraseLabel に含まれるアクセント位置をテスト"""
        self.assertEqual(self.accent_phrase_hello.accent, 5)
        self.assertEqual(self.accent_phrase_hiho.accent, 1)

    def test_phonemes(self):
        """AccentPhraseLabel に含まれる音素系列をテスト"""
        outputs_hello = space_jointed_phonemes(self.accent_phrase_hello)
        outputs_hiho = space_jointed_phonemes(self.accent_phrase_hiho)
        self.assertEqual(outputs_hello, "k o N n i ch i w a")
        self.assertEqual(outputs_hiho, "h i h o d e s U")

    def test_features(self):
        """AccentPhraseLabel に含まれる features をテスト"""
        expects = self.test_case_hello_hiho
        self.assertEqual(features(self.accent_phrase_hello), expects[1:10])
        self.assertEqual(features(self.accent_phrase_hiho), expects[11:19])


class TestBreathGroupLabel(TestBaseLabels):
    def setUp(self) -> None:
        super().setUp()
        self.breath_group_hello = BreathGroupLabel.from_labels(
            self.labels_hello_hiho[1:10]
        )
        self.breath_group_hiho = BreathGroupLabel.from_labels(
            self.labels_hello_hiho[11:19]
        )

    def test_phonemes(self):
        """BreathGroupLabel に含まれる音素系列をテスト"""
        outputs_hello = space_jointed_phonemes(self.breath_group_hello)
        outputs_hiho = space_jointed_phonemes(self.breath_group_hiho)
        self.assertEqual(outputs_hello, "k o N n i ch i w a")
        self.assertEqual(outputs_hiho, "h i h o d e s U")

    def test_features(self):
        """BreathGroupLabel に含まれる features をテスト"""
        expects = self.test_case_hello_hiho
        self.assertEqual(features(self.breath_group_hello), expects[1:10])
        self.assertEqual(features(self.breath_group_hiho), expects[11:19])


class TestUtteranceLabel(TestBaseLabels):
    def setUp(self) -> None:
        super().setUp()
        self.utterance_hello_hiho = UtteranceLabel.from_labels(self.labels_hello_hiho)

    def test_phonemes(self):
        """UtteranceLabel に含まれる音素系列をテスト"""
        outputs_hello_hiho = space_jointed_phonemes(self.utterance_hello_hiho)
        expects_hello_hiho = "sil k o N n i ch i w a pau h i h o d e s U sil"
        self.assertEqual(outputs_hello_hiho, expects_hello_hiho)

    def test_features(self):
        """UtteranceLabel に含まれる features をテスト"""
        self.assertEqual(features(self.utterance_hello_hiho), self.test_case_hello_hiho)


class TestMoraToText(TestCase):
    def test_voice(self):
        self.assertEqual(mora_to_text("a"), "ア")
        self.assertEqual(mora_to_text("i"), "イ")
        self.assertEqual(mora_to_text("ka"), "カ")
        self.assertEqual(mora_to_text("N"), "ン")
        self.assertEqual(mora_to_text("cl"), "ッ")
        self.assertEqual(mora_to_text("gye"), "ギェ")
        self.assertEqual(mora_to_text("ye"), "イェ")
        self.assertEqual(mora_to_text("wo"), "ウォ")

    def test_unvoice(self):
        self.assertEqual(mora_to_text("A"), "ア")
        self.assertEqual(mora_to_text("I"), "イ")
        self.assertEqual(mora_to_text("kA"), "カ")
        self.assertEqual(mora_to_text("gyE"), "ギェ")
        self.assertEqual(mora_to_text("yE"), "イェ")
        self.assertEqual(mora_to_text("wO"), "ウォ")

    def test_invalid_mora(self):
        """変なモーラが来ても例外を投げない"""
        self.assertEqual(mora_to_text("x"), "x")
        self.assertEqual(mora_to_text(""), "")


def _gen_mora(text: str, consonant: str | None, vowel: str) -> Mora:
    return Mora(
        text=text,
        consonant=consonant,
        consonant_length=0 if consonant else None,
        vowel=vowel,
        vowel_length=0,
        pitch=0,
    )


def test_text_to_accent_phrases_normal():
    """`text_to_accent_phrases` は正常な日本語文をパースする"""
    # Inputs
    text = "こんにちは、ヒホです。"
    # Expects
    true_accent_phrases = [
        AccentPhrase(
            moras=[
                _gen_mora("コ", "k", "o"),
                _gen_mora("ン", None, "N"),
                _gen_mora("ニ", "n", "i"),
                _gen_mora("チ", "ch", "i"),
                _gen_mora("ワ", "w", "a"),
            ],
            accent=5,
            pause_mora=_gen_mora("、", None, "pau"),
        ),
        AccentPhrase(
            moras=[
                _gen_mora("ヒ", "h", "i"),
                _gen_mora("ホ", "h", "o"),
                _gen_mora("デ", "d", "e"),
                _gen_mora("ス", "s", "U"),
            ],
            accent=1,
            pause_mora=None,
        ),
    ]
    # Outputs
    accent_phrases = text_to_accent_phrases(text)
    # Tests
    assert accent_phrases == true_accent_phrases


def stub_unknown_features_koxx(_: str) -> list[str]:
    """`sil-k-o-xx-sil` に相当する features を常に返す `text_to_features()` のStub"""
    return [
        ".^.-sil+.=./A:.+xx+./B:.-._./C:._.+./D:.+._./E:._.!._.-./F:xx_xx#xx_.@xx_.|._./G:._.%._._./H:._./I:.-.@xx+.&.-.|.+./J:._./K:.+.-.",
        ".^.-k+.=./A:.+1+./B:.-._./C:._.+./D:.+._./E:._.!._.-./F:2_1#0_.@1_.|._./G:._.%._._./H:._./I:.-.@1+.&.-.|.+./J:._./K:.+.-.",
        ".^.-o+.=./A:.+1+./B:.-._./C:._.+./D:.+._./E:._.!._.-./F:2_1#0_.@1_.|._./G:._.%._._./H:._./I:.-.@1+.&.-.|.+./J:._./K:.+.-.",
        ".^.-xx+.=./A:.+2+./B:.-._./C:._.+./D:.+._./E:._.!._.-./F:2_1#0_.@1_.|._./G:._.%._._./H:._./I:.-.@1+.&.-.|.+./J:._./K:.+.-.",
        ".^.-sil+.=./A:.+xx+./B:.-._./C:._.+./D:.+._./E:._.!._.-./F:xx_xx#xx_.@xx_.|._./G:._.%._._./H:._./I:.-.@xx+.&.-.|.+./J:._./K:.+.-.",
    ]


def test_text_to_accent_phrases_unknown():
    """`text_to_accent_phrases` は unknown 音素を含む features をパースする"""
    # Expects
    true_accent_phrases = [
        AccentPhrase(
            moras=[
                _gen_mora("コ", "k", "o"),
                _gen_mora("xx", None, "xx"),
            ],
            accent=1,
            pause_mora=None,
        ),
    ]
    # Outputs
    accent_phrases = text_to_accent_phrases(
        "dummy", text_to_features=stub_unknown_features_koxx
    )
    # Tests
    assert accent_phrases == true_accent_phrases
