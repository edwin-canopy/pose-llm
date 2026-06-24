



class Collator:
    """
    data format:
    words = [{}] with {word, start time, end time}
    pose tokens = Tensor (N, T')
    """

    def __init__(self, text_tokenizer):
        self.text_tokenizer = text_tokenizer
        pass


    def mark_word_starts(self):
        pass
    

    def shift_left(self):
        pass


    def assemble(self, sample):
        """
        We assume text has structure {word, start, end} and add a new field tokens
        """

        # 1. tokenize text
        # note: maybe faster to do this once in dataset creation

        text = sample["text"]
        assert isinstance(text[0], dict)
        text[0]["tokens"] = self.tokenizer.tokenize(text[0]["word"])
        if len(text) > 1:
            for i in range(1, len(text)):
                text[i]["tokens"] = self.tokenizer.tokenize(text[i]["word"])

        # 1b. initialise tensors with pad everywhere

        

        # 2. distribute tokens so that we have one word token per frame

        # 2b. shift text to give lookahead (where possible)

        # 2c. add new word tokens before subsequent new words

        # 3. stack pose tokens with text tokens

        # 4. add pose offset to pose tokens


    def __call__(self, samples):
        """
        Takes in a list of samples
        Collates samples into a single sample separated by special tokens

        note pose tokenizer is at 25 fps

        Returns tuple/dict of 
        - sequence_ids: Tensor (1, S, 3)
        - codes: Tensor (1, S, 2, depth)
        - separator_mask: Tensor (1, S)
        """
        assert isinstance(samples, list)

        # call assmble and concatenate with separators
        sequences = [self.assemble(sample) for sample in samples]

        sequence_ids = [s["ids"] for s in sequences]
        codes = [s["codes"] for s in sequences]

        pass


class SpeechCollator:

    def __init__(self):
        pass

    def __call__(self, sample):
        """
        sample = [text dict, pose tokens]
        """
