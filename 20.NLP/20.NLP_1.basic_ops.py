import nltk

text = 'India, officially the Republic of India is a country in South Asia. ' \
       'It is the seventh-largest country by area; the most populous country ' \
       'and from the time of its independence in 1947, the world\'s' \
       'most populous democracy. It is physiographically bounded by the Indian Ocean ' \
       'on the south, the Arabian Sea on the southwest, the Bay of Bengal ' \
       'on the southeast, and High-mountain Asia on the northeast. ' \
       'It shares land borders with Pakistan to the northwest, China, Nepal, ' \
       'and Bhutan to the north; and Bangladesh and Myanmar to the east. ' \
       'In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives; ' \
       'its Andaman and Nicobar Islands share a maritime border with Thailand, Myanmar, ' \
       'and Indonesia.'

print('WT : ',nltk.word_tokenize(text))
print('ST : ',nltk.sent_tokenize(text))