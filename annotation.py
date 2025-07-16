import spacy
import json
from nltk.tokenize import sent_tokenize
from pathlib import Path
import random


def remove_punct(sentence):
    """
    Removes punctuation symbols from the sentence
    """
    clean_sentence = ''
    for symbol in sentence:
        if symbol.isalnum() or symbol.isspace():
            clean_sentence += symbol.lower()
    return clean_sentence


def context_extractor(corpus):
    """
    Extracts contexts where the token "fold" appears
    """
    sentences = sent_tokenize(corpus)
    contexts = []
    for sentence in sentences:
        tokens = sentence.split()
        for token in tokens:
            if token in ['fold', 'folds', 'folded', 'folding'] and sentence not in contexts:
                contexts.append(sentence)
    return contexts


def pos_tagger(sentences):
    """
    Annotates POS-tags in the lists of sentences
    """
    nlp = spacy.load("en_core_web_sm")
    tagged_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        tags = []
        for token in doc:
            if token.lemma_ == 'fold':
                tags.append(token.pos_)
        tagged_sentence = []
        tokens = sentence.split()
        count = 0
        for token in tokens:
            if 'fold' in token:
                if count < len(tags):
                    tagged_sentence.append(token + f'[{tags[count]}]')
                    count += 1
            else:
                tagged_sentence.append(token)
        tagged_sentences.append(' '.join(tagged_sentence))
    return tagged_sentences


def concordancer(sentence, window):
    """
    Extracts concordances from the sentence
    """
    tokens = sentence.split()
    concordances = []
    for token in tokens:
        if 'fold' in token and '[' in token:
            target_idx = tokens.index(token)
            aux_idx = tokens[target_idx].index('[')
            pos = token[token.index('[')+1:-1]
            if len(tokens[:target_idx]) < window:
                left_context = remove_punct(' '.join(tokens[:target_idx]))
            else:
                left_context = remove_punct(' '.join(tokens[target_idx-window:target_idx]))
            if len(tokens[target_idx+1:]) <= window:
                right_context = remove_punct(' '.join(tokens[target_idx+1:]))
            else:
                right_context = remove_punct(' '.join(tokens[target_idx+1:target_idx+window+1]))
            concordance = {'word': tokens[target_idx][:aux_idx], 'left': left_context,
                          'right': right_context, 'pos': pos}
            concordances.append(concordance)
    return concordances


def sem_analyzer(concordance):
    """
    Decides which meaning has the word in the concordance
    """
    nlp = spacy.load("en_core_web_sm")
    with open('semantic_tags.json', 'r', encoding='utf-8') as f:
        sem_tags = json.load(f)

    if concordance['pos'] == 'VERB':
        for token in nlp(concordance['right']):
            if token.lemma_ in sem_tags['verbs']['5'][1]:
                return 5
            if token.lemma_ in sem_tags['verbs']['3'][1]:
                return 3
        for token in nlp(concordance['left'] + ' ' + concordance['right']):
            if token.lemma_ in sem_tags['verbs']['4'][1]:
                return 4
            if token.lemma_ in sem_tags['verbs']['2'][1]:
                return 2
            if token.lemma_ in sem_tags['verbs']['1'][1]:
                return 1
        return 5
    if concordance['pos'] != 'VERB':
        word = concordance['word']
        for symbol in word:
            if symbol.isdigit():
                if '-' in word:
                    return 2
        for token in nlp(concordance['left']):
            if token.lemma_ in sem_tags['nouns']['3'][1]:
                return 3
            if token.lemma_ in sem_tags['nouns']['4'][1]:
                return 3
        return 1


def sem_tagger(sentence):
    """
    Conducts semantic annotation of the sentence
    """
    tags = []
    for concordance in concordancer(sentence, 5):
        tags.append(sem_analyzer(concordance))
    tagged_sentence = []
    count = 0
    for token in sentence.split():
        if 'fold' in token and '[' in token and ']' in token:
            if count < len(tags):
                tagged_sentence.append(f'{token[:-1]}{tags[count]}]')
                count += 1
        else:
            tagged_sentence.append(token)
    return ' '.join(tagged_sentence)


def create_test_sample(file_name, number):
    """
    Extracts from the corpus a test sample
    """
    directory = Path(file_name)
    corpus = ''
    for file in directory.iterdir():
        with open(file, 'r', encoding='utf-8') as f:
            corpus += f.read().strip() + '\n'

    population = context_extractor(corpus)

    for i in range(10):
        test_sample = random.sample(population, number)
        with open(f'test_sample/sample_{i+1}.txt', 'w', encoding='utf-8') as f:
            for element in test_sample:
                f.write(element)
                f.write('\n')


def annotate_test_sample(file_name):
    """
    Annotates the test sample
    """
    directory = Path(file_name)
    count = 1
    for file in directory.iterdir():
        with open(file, 'r', encoding='utf-8') as source:
            corpus = source.read().strip()
            corpus = pos_tagger(context_extractor(corpus))
            with open(f'annotated_sample/sample_{count}', 'w', encoding='utf-8') as result:
                for sentence in corpus:
                    result.write(sem_tagger(sentence))
                    result.write('\n')
            count += 1
