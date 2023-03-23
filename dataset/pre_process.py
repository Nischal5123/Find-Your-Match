import re

"""
Performs basic text cleansing on the unstructured field 
and adds additional column to the input dataframe
"""


class Preprocess:
    def __init__(self, stpwds_file_path):
        """
        Initializes regex patterns and load stopwords
        """
        self.PUNCTUATION_PATTERN = (
            "'’|!@$%^&*()_+<>?:.,;-"  ## all punctuation symbols to be removed
        )

    def remove_punctuations(self, text):
        """
        Removes punctuations from text field
        """
        return "".join([c for c in text if c not in self.PUNCTUATION_PATTERN])


    def perform_preprocessing(self, data, columns_mapping):
        ## normalizing text to lower case
        data["clean_sent1"] = data[columns_mapping["sent1"]].apply(
            lambda text: text.lower()
        )
        data["clean_sent2"] = data[columns_mapping["sent2"]].apply(
            lambda text: text.lower()
        )
        ## removing punctuations
        data["clean_sent1"] = data.clean_sent1.apply(self.remove_punctuations)
        data["clean_sent2"] = data.clean_sent2.apply(self.remove_punctuations)

        return data