import string


def preprocess(source_data, target_data):
    # TODO: Preprocess in one pass
    # Convert to lowercase characters
    source_data = source_data.apply(lambda x: x.str.lower())
    target_data = target_data.apply(lambda x: x.str.lower())

    # Add SOS and EOS tokens
    target_data = target_data.apply(lambda x: 'SOS ' + x + ' EOS')

    # Remove punctuation and digits
    source_data = source_data.apply(lambda x: x.str.replace('[^a-zA-Z\s]', ''))
    target_data = target_data.apply(lambda x: x.str.replace('[^a-zA-Z\s]', ''))
    return source_data.values.flatten(), target_data.values.flatten()
