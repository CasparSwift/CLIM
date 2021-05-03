import nlpaug.augmenter.word as naw
from nltk.tokenize import sent_tokenize
import random


# def make_positive_pairs(text, tokenizer, max_seq_length, method):
#     aug_func = augment_factory[method]
#     augmented_text = aug_func(text)
#     aug_ids = [tokenizer.encode(t, max_length=max_seq_length, add_special_tokens=True, truncation=True)
#                for t in augmented_text]
#
#     # padding
#     max_len = max(len(ids) for ids in aug_ids)
#     input_ids = torch.tensor([ids + [0]*(max_len-len(ids)) for ids in aug_ids], dtype=torch.long)
#     masks = torch.tensor([[1]*len(ids) + [0]*(max_len-len(ids)) for ids in aug_ids], dtype=torch.long)
#     return augmented_text, input_ids, masks


class synonym_augmenter(object):
    def __init__(self, args):
        self.aug_p = args.aug_rate

    def transform(self, text):
        # print('aug')
        # print(text)
        aug = naw.SynonymAug(aug_src='wordnet', aug_p=self.aug_p, aug_max=None)
        augmented_text = aug.augment(text)
        del aug
        return augmented_text


class identity_augmenter(object):
    def __init__(self, args=None):
        pass

    def transform(self, text):
        return text


class context_augmenter(object):
    def __init__(self, args=None):
        pass

    def transform(self, text):
        sents = sent_tokenize(text)
        if len(sents) <= 1:
            return text, text
        id = random.choice(list(range(1, len(sents))))
        return ' '.join(sents[:id]), ' '.join(sents[id:])


class back_translation_augmenter(object):
    def __init__(self,args=None):
        # self.back_translation_aug = naw.BackTranslationAug(
        #     from_model_name='transformer.wmt19.en-de',
        #     to_model_name='transformer.wmt19.de-en'
        # )
        pass

    def transform(self, text):
        # return self.back_translation_aug.augment(text)
        return text


augment_factory = {
    'synonym_substitution': synonym_augmenter,
    'none': identity_augmenter,
    'context': context_augmenter,
    'back_translation': back_translation_augmenter
}



if __name__=='__main__':
    aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.7, aug_max=None)
    string = 'This book has good information for feeding and care of African Grey Parrots. \
        I wish there had been more on training, however the breeding sections are excellent.'
    print(string)
    print(aug.augment(string))
