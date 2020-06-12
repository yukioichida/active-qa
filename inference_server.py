from px.nmt import reformulator
from px.proto import reformulator_pb2
from flask import Flask, url_for
from markupsafe import escape


def generate_paraphrase(sentence):
    reformulator_instance = reformulator.Reformulator(
        hparams_path='px/nmt/example_configs/reformulator.json',
        source_prefix='<en> <2en> ',
        out_dir='reformulator/',
        environment_server_address='localhost:10000')

    questions = [
        sentence
    ]

    inference_mode = reformulator_pb2.ReformulatorRequest.BEAM_SEARCH
    responses = reformulator_instance.reformulate(
        questions=questions,
        inference_mode=inference_mode)

    if inference_mode == reformulator_pb2.ReformulatorRequest.GREEDY:
        # Since we are using greedy decoder, keep only the first rewrite.
        reformulations = [r[0].reformulation for r in responses]
    else:
        reformulations = [[t.reformulation for t in r] for r in responses]

    for sentence in reformulations:
        print(sentence)
    return reformulations


app = Flask(__name__)


@app.route('/paraphrase/<sentence>')
def profile(sentence):
    paraphrases = generate_paraphrase(sentence)
    return {
        "original_sentence": sentence,
        "paraphrases": paraphrases
    }
