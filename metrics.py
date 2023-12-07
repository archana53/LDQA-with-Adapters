import evaluate
import nltk
import numpy as np


class MetricComputer:
    """Class to compute metrics: BLEU, ROUGE, METEOR and return a dictionary.
    :param tokenizer: instance of AutoTokenizer
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # set up metrics using huggingface evaluate
        self.meteor = evaluate.load("meteor")
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")

    def __call__(self, eval_prediction):
        """
        :param eval_prediction: instance of EvalPrediction with predictions and labels
        :return: dictionary of metrics
        """
        preds, labels = eval_prediction.predictions, eval_prediction.label_ids

        # decode preds and labels
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        decoded_preds = [
            "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
        ]
        decoded_labels = [
            "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
        ]

        rouge_score = self.rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        meteor_score = self.meteor.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        # for bleu convert the references to a list of lists
        decoded_labels = [[label] for label in decoded_labels]

        # combine all metrics into one dictionary
        results = {
            k: v
            for score_dict in [rouge_score, meteor_score]
            for k, v in score_dict.items()
        }
        return results
