# To be added to notebook later

# in: the training data
# out: neat matrices of the features followed by result
def extract(training_file):
    try:
        file = open(training_file, 'r')
    except FileNotFoundError:
        print("Bad file or path")

    '''
    Features (represented in this order)
    1. first longer than second? [done]
    2. do they have a middle name? [done]
    3. does their first name start and end w/ same letter [done?]
    4. does first name come alphabetically before last [done]
    5. is the second letter of their first name a letter [done]
    6. Is the number of letters in their last name even
    last: + / - (represented as binary 1 or 0)
    '''

    # feature generating methods
    # so many splits here boss, wonder if we can pass it as a list
    def first_longer_second(name):
        name = name.split(' ')

        # does not handle titles like general or sir
        # assume first name is always first and last always last
        return len(name[0]) > len(name[-1])

    def has_middle(name):
        # assumes that if the name has more than two parts, it has a middle name
        return len(name.split(' ')) > 2

    def same_letter(name):
        name = name.split(' ')

        return name[0][0].lower() == name[0][-1].lower()

    def alpha_firstlast(name):
        name = name.split(' ')
        return name[0].lower() < name[-1].lower()

    def second_letter(name):
        # TODO: what if their first name is one letter?
        name = name.split(' ' )
        if len(name[0]) < 2:
            return False
        else:
            return name[0][1].lower() in { 'a', 'e', 'i', 'o', 'u'}

    def even_last(name):
        name = name.split(' ')
        return len(name[-1]) % 2 == 0

    features = []
    for line in file:
        line = line[:-1] # ignore newline at the end
        label = line.split(' ')[0] # split into label
        name = ' '.join(line.split(' ')[1:]) # split into name

        feature = []
        feature.append(first_longer_second(name))
        feature.append(has_middle(name))
        feature.append(same_letter(name))
        feature.append(alpha_firstlast(name))
        feature.append(second_letter(name))
        feature.append(even_last(name))

        if (label == '+'):
            label = True
        else:
            label = False

        feature.append(label)

        features.append(feature)

    return features

if __name__ == '__main__':
    matrix = extract("data/training.data")
