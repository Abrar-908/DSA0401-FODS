spam = 0.25
offer_spam = 0.80
offer_notspam = 0.10

not_spam = 1 - spam

p_offer = (offer_spam * spam) + (offer_notspam * not_spam)

p_spam_offer = (offer_spam * spam) / p_offer

print("P(Spam | Offer) =", p_spam_offer)