import re

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.configs.base_config import is_half, device
from src.configs.tts_config import bert_path
from src.tts.text import cleaned_text_to_sequence
from src.tts.text.cleaner import clean_text

splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}

punctuation = {"!", "?", "…", ",", ".", "-", " "}

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def clean_text_inf(text, language):
    """
    把任意语言的原始文本转成音素序列和词级对齐
    :param text: 文本
    :param language: 语言
    :return: 音素，每个词的因素长度，归一化文本
    """
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text  # 音素，每个词的因素长度，归一化文本


def split_en_inf(text, language):
    """
    把一段混合语言文本（如 "你好hello世界world"）拆成交替的文本块 + 语言标签
    :param text: 文本
    :param language:语言
    :return: 文本块，语言标签
    """
    pattern = re.compile(r'[a-zA-Z. ]+')
    text_list, lang_list = [], []
    pos = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        if start > pos:
            text_list.append(text[pos:start])
            lang_list.append(language)
        text_list.append(text[start:end])
        lang_list.append("en")
        pos = end
    if pos < len(text):
        text_list.append(text[pos:])
        lang_list.append(language)

    return text_list, lang_list


def nonen_clean_text_inf(text, language):
    """
    把任意语言混合文本（中英/中日/中韩等）拆成纯音素序列 + 词级对齐 + 归一化文本
    :param text: 文本
    :param language: 语言
    :return: 音素，每个词的因素长度，归一化文本
    """
    text_list, lang_list = split_en_inf(text, language)
    phones_list, word2ph_list, norm_text_list = [], [], []
    for i in range(len(text_list)):
        lang = lang_list[i]
        phones, word2ph, norm_text = clean_text_inf(text_list[i], lang)
        phones_list.append(phones)
        if lang not in ("en", "ja"):
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)

    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)
    return phones, word2ph, norm_text


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1

    return todo_texts


def cut1(inp):
    """
    凑四句一切
    :param inp: 文本
    :return: 换行分隔的多行文本
    """
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    # 过滤掉只含标点的行
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    """
    按 50-char 滑窗切片，末行过短自动合并，最终返回干净的多行文本。
    :param inp: 文本
    :return: 换行分隔的多行文本
    """
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:  # 达到 50 立即切片
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    if len(opts) > 1 and len(opts[-1]) < 50:  # 末行不足 50 → 与前一行合并
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    # 过滤只含标点的行
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    """
    把文本按「中文句号」切成多行，并去掉只含标点的空行。
    :param inp:文本
    :return:换行分隔的多行文本
    """
    inp = inp.strip("\n")
    opts = [str(item) for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut4(inp):
    """
    按「非数字间的句号」切句，并去掉只含标点的空行。
    :param inp:文本
    :return:换行分隔的多行文本
    """
    inp = inp.strip("\n")
    opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut5(inp):
    """
    按任意中英文标点切句，但保留数字小数点，并自动合并成多行文本。
    :param inp:文本
    :return:换行分隔的多行文本
    """
    inp = inp.strip("\n")
    punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == "." and 0 < i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


def get_bert_feature(text, word2ph):
    """
    把「中文文本 + 词-音素对齐」扩展成「逐音素级 1024 维 BERT 特征」，用于下游 TTS 对齐。
    :param text:文本
    :param word2ph:词-音素对齐
    :return:逐音素级 1024 维 BERT 特征
    """
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in tokenizer(text, return_tensors="pt").items()}
        # 在返回结果里额外给出所有 Transformer 层的隐藏状态
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T


def get_bert_inf(phones, word2ph, norm_text, language):
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half else torch.float32,
        ).to(device)
    return bert


def nonen_get_bert_inf(text, language):
    text_list, lang_list = split_en_inf(text, language)
    bert_list = []
    for i in range(len(text_list)):
        text = text_list[i]
        lang = lang_list[i]
        phones, word2ph, norm_text = clean_text_inf(text, lang)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)
    return bert
