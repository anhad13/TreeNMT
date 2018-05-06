from collections import defaultdict, Counter
import math
import subprocess
import numpy as np

class EvalScore(object):
  def higher_is_better(self):
    raise NotImplementedError()
  def value(self):
    raise NotImplementedError()
  def metric_name(self):
    raise NotImplementedError()
  def score_str(self):
    raise NotImplementedError()
  def better_than(self, another_score):
    if another_score is None or another_score.value() is None: return True
    elif self.value() is None: return False
    assert type(self) == type(another_score)
    if self.higher_is_better():
      return self.value() > another_score.value()
    else:
      return self.value() < another_score.value()
  def __str__(self):
    desc = getattr(self, "desc", None)
    if desc:
      return "{self.metric_name()} ({desc}): {self.score_str()}"
    else:
      return "{self.metric_name()}: {self.score_str()}"



class Evaluator(object):
  """
  A class to evaluate the quality of output.
  """

  def evaluate(self, ref, hyp):
    """
  Calculate the quality of output given a references.

  Args:
    ref: list of reference sents ( a sent is a list of tokens )
    hyp: list of hypothesis sents ( a sent is a list of tokens )
  """
    raise NotImplementedError('evaluate must be implemented in Evaluator subclasses')

  def metric_name(self):
    """
  Return:
    str:
  """
    raise NotImplementedError('metric_name must be implemented in Evaluator subclasses')

  def evaluate_fast(self, ref, hyp):
    raise NotImplementedError('evaluate_fast is not implemented for:', self.__class__.__name__)

class BLEUScore(EvalScore):
  yaml_tag = "!BLEUScore"
  def __init__(self, bleu, frac_score_list=None, brevity_penalty_score=None, hyp_len=None, ref_len=None, ngram=1, desc=None):
    self.bleu = bleu
    self.frac_score_list = frac_score_list
    self.brevity_penalty_score = brevity_penalty_score
    self.hyp_len = hyp_len
    self.ref_len = ref_len
    self.ngram   = ngram
    self.desc = desc
    #self.serialize_params = {"bleu":bleu, "ngram":ngram}
    #self.serialize_params.update({k:getattr(self,k) for k in ["frac_score_list","brevity_penalty_score","hyp_len","ref_len","desc"] if getattr(self,k) is not None})

  def value(self): return self.bleu
  def metric_name(self): return "BLEU" + str(self.ngram)
  def higher_is_better(self): return True
  def score_str(self):
    if self.bleu is None:
      return "0"
    else:
      return "{self.bleu}, {'/'.join(self.frac_score_list)} (BP = {self.brevity_penalty_score:.6f}, ratio={self.hyp_len / self.ref_len:.2f}, hyp_len={self.hyp_len}, ref_len={self.ref_len})"




class BLEUEvaluator(Evaluator):
  # Class for computing BLEU Scores accroding to
  # K Papineni et al "BLEU: a method for automatic evaluation of machine translation"
  def __init__(self, ngram=1, smooth=0, desc=None):
    """
    Args:
      ngram: default value of 4 is generally used
    """
    self.ngram = ngram
    self.weights = (1 / ngram) * np.ones(ngram, dtype=np.float32)
    self.smooth = smooth
    self.reference_corpus = None
    self.candidate_corpus = None
    self.desc = desc

  def metric_name(self):
    return "BLEU%d score" % (self.ngram)

  def evaluate(self, ref, hyp):
    """
    Args:
      ref: list of reference sents ( a sent is a list of tokens )
      hyp: list of hypothesis sents ( a sent is a list of tokens )
    Return:
      Formatted string having BLEU Score with different intermediate results such as ngram ratio,
      sent length, brevity penalty
    """
    self.reference_corpus = ref
    self.candidate_corpus = hyp

    assert (len(self.reference_corpus) == len(self.candidate_corpus)), \
           "Length of Reference Corpus and Candidate Corpus should be same"

    # Modified Precision Score
    clipped_ngram_count = Counter()
    candidate_ngram_count = Counter()

    # Brevity Penalty variables
    word_counter = Counter()

    for ref_sent, can_sent in zip(self.reference_corpus, self.candidate_corpus):
      word_counter['reference'] += len(ref_sent)
      word_counter['candidate'] += len(can_sent)
      #print(ref_sent)
      #print("---")
      #print(can_sent)
      clip_count_dict, full_count_dict = self.modified_precision(ref_sent, can_sent)
      
      for ngram_type in full_count_dict:
        if ngram_type in clip_count_dict:
          clipped_ngram_count[ngram_type] += sum(clip_count_dict[ngram_type].values())
        else:
          clipped_ngram_count[ngram_type] += 0.  # This line may not be required

        candidate_ngram_count[ngram_type] += sum(full_count_dict[ngram_type].values())

    # Edge case
    # Return 0 if there are no matching n-grams
    # If there are no unigrams, return BLEU score of 0
    # No need to check for higher order n-grams
    if clipped_ngram_count[1] == 0:
      return BLEUScore(bleu=None, ngram=self.ngram, desc=self.desc)

    frac_score_list = list()
    log_precision_score = 0.
    # Precision Score Calculation
    for ngram_type in range(1, self.ngram + 1):
      frac_score = 0
      if clipped_ngram_count[ngram_type] == 0:
        log_precision_score += -1e10
      else:
        frac_score = float(clipped_ngram_count[ngram_type]) / candidate_ngram_count[ngram_type]
        #print(candidate_ngram_count[ngram_type])
        log_precision_score += self.weights[ngram_type - 1] * math.log(frac_score)
      frac_score_list.append("%.6f" % frac_score)

    precision_score = math.exp(log_precision_score)

    # Brevity Penalty Score
    brevity_penalty_score = self.brevity_penalty(word_counter['reference'], word_counter['candidate'])

    # BLEU Score
    bleu_score = brevity_penalty_score * precision_score
    return BLEUScore(bleu_score, frac_score_list, brevity_penalty_score, word_counter['candidate'], word_counter['reference'], ngram=self.ngram, desc=self.desc)

  # Doc to be added
  def brevity_penalty(self, r, c):
    """
    Args:
      r: number of words in reference corpus
      c: number of words in candidate corpus
    Return:
      brevity penalty score
    """

    penalty = 1.

    # If candidate sent length is 0 (empty), return 0.
    if c == 0:
      return 0.
    elif c <= r:
      penalty = np.exp(1. - (r / c))
    return penalty

  # Doc to be added
  def extract_ngrams(self, tokens):
    """
    Extracts ngram counts from the input string

    Args:
      tokens: tokens of string for which the ngram is to be computed
    Return:
      a Counter object containing ngram counts
    """

    ngram_count = defaultdict(Counter)
    num_words = len(tokens)

    for i, first_token in enumerate(tokens[0: num_words]):
      for j in range(0, self.ngram):
        outer_range = i + j + 1
        ngram_type = j + 1
        if outer_range <= num_words:
          ngram_tuple = tuple(tokens[i: outer_range])
          ngram_count[ngram_type][ngram_tuple] += 1

    return ngram_count

  def modified_precision(self, reference_sent, candidate_sent):
    """
    Computes counts useful in modified precision calculations

    Args:
      reference_sent: iterable of tokens
      candidate_sent: iterable of tokens
    Return: tuple of Counter objects
    """

    clipped_ngram_count = defaultdict(Counter)

    reference_ngram_count = self.extract_ngrams(reference_sent)
    candidate_ngram_count = self.extract_ngrams(candidate_sent)

    for ngram_type in candidate_ngram_count:
      clipped_ngram_count[ngram_type] = candidate_ngram_count[ngram_type] & reference_ngram_count[ngram_type]

    return clipped_ngram_count, candidate_ngram_count