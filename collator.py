



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
        text = sample["text"]
        for dict_ in text:
            for dict in

        # 1. tokenize text

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

        pass


class SpeechCollator:

    def __init__(self):
        pass

    def __call__(self, sample):
        """
        sample = [text dict, pose tokens]
        """
