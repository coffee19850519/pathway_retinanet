
from difflib import SequenceMatcher

def corrected_processing_by_dict(dictionary, ocr, thresh=0.9):
    """
    Use exHUGO dictionary to correct an ocr result by calculating similarity,
    thresh=0.7, the optimal threshold with recall,
    thresh=0.9, the optimal threshold with the mean of precision and recall,
    thresh=1.0, the optimal threshold with precision.

    :param dictionary: exHUGO dictionary, list
    :param ocr: one ocr results, string
    :param thresh: optimal threshold, ranges from 0 to 1, float
    :return:
    """
    seq_match_ratio = [SequenceMatcher(None, ocr.upper(), gene.upper()).ratio() for gene in dictionary]
    corrected_ocr = dictionary[seq_match_ratio.index(max(seq_match_ratio))] if round(max(seq_match_ratio),
                                                                                     3) >= thresh else '-'
    return corrected_ocr


