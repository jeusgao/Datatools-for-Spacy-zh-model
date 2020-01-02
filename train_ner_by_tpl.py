# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
import random
from pathlib import Path
import spacy
import time
import jieba
import pickle
import plac


@plac.annotations(
    model=("Model name. Defaults to blank 'zh_model' model.", "option", "m", str),
    output_dir=("Optional output directory. Defaults to 'zh_model' folder.", "option", "o", Path),
    train_data=("Train data pickle file-name with path", "option", "t", str),
    n_iter=("Number of training iterations", "option", "n", int)
)
def main(model='zh_model', output_dir='zh_model/', n_iter=50, train_data=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    jieba.initialize()

    print('loading User dict...')
    jieba.load_userdict('dict.txt')

    TRAIN_DATA = []
    with open(train_data, 'rb') as fr:
        TRAIN_DATA = pickle.load(fr)
    print(TRAIN_DATA)
    print(len(TRAIN_DATA))

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    print("Get annotations, add labels...")
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    print("Get names of other pipes to disable them during training...")
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for i, itn in enumerate(range(n_iter)):
            random.shuffle(TRAIN_DATA)
            losses = {}
            rd_st = time.time()
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            rd_ed = time.time()
            print('No. ', i, ': ', losses, ' Cost: ', '%.2f' % (rd_ed - rd_st), ' secs...')

    print("Test the trained model...")
    test_text = "查看明天的日程"
    doc = nlp(test_text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == '__main__':
    start = time.time()
    plac.call(main)
    end = time.time()
    print('Cost: ', end - start, ' secs.')
