import re
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer


def clean_text(text):
    # 缩写替换
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    # 单独的数字替换为英文
    text = re.sub(r" 1 ", " one ", text)
    text = re.sub(r" 2 ", " two ", text)
    text = re.sub(r" 3 ", " three ", text)
    text = re.sub(r" 4 ", " four ", text)
    text = re.sub(r" 5 ", " five ", text)
    text = re.sub(r" 6 ", " six ", text)
    text = re.sub(r" 7 ", " seven ", text)
    text = re.sub(r" 8 ", " eight ", text)
    text = re.sub(r" 9 ", " nine ", text)
    # 替换不可见字符以及各分隔符
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\\+', ' ', text)
    text = re.sub(r'/+', ' ', text)
    text = re.sub(r'_+', ' ', text)
    text = re.sub(r'--+', ' ', text)
    text = re.sub(r'\.', ' ', text)
    text = re.sub(r' +', ' ', text)

    return text

# 分词
def tokenize(text):
    token_words = word_tokenize(text)
    token_words = pos_tag(token_words)
    return token_words

# 去掉词性
def stem(token_words):
    wordnet_lematizer = WordNetLemmatizer()
    words_lematizer = []
    for word, tag in token_words:
        if tag.startswith('NN'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='n')  # n代表名词
        elif tag.startswith('VB'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='v')  # v代表动词
        elif tag.startswith('JJ'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='a')  # a代表形容词
        elif tag.startswith('R'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='r')  # r代表代词
        else:
            word_lematizer = wordnet_lematizer.lemmatize(word)
        words_lematizer.append(word_lematizer)
    return words_lematizer

# 去掉停用词
def delete_stopwords(token_words):
    """ 去停用词"""
    sr = stopwords.words('english')
    cleaned_words = [word for word in token_words if word not in sr]
    return cleaned_words

# 去掉数字
def is_number(s):
    """ 判断字符串是否为数字"""
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

# 删除特殊字符
def delete_characters(token_words):
    """去除特殊字符、数字"""
    characters = ['\'', "''", '``', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&',
                  '!', '*', '@', '#', '$', '%', '-', '>', '<', '...', '^', '{', '}']
    words_list = [word for word in token_words if word not in characters and not is_number(word)]
    return words_list

# 全部转换为小写
def to_lower(token_words):
    words_lists = [x.lower() for x in token_words]
    return words_lists

# 文本预处理接口
def pre_process(text):
    text = clean_text(text)
    token_words = tokenize(text)
    token_words = stem(token_words)
    token_words = delete_stopwords(token_words)
    token_words = delete_characters(token_words)
    token_words = to_lower(token_words)
    return token_words
